//! Is this text natural language, or gibberish?
//!
//! `gibberish-or-not` tells natural-language text apart from everything else: cipher and
//! decoder output, encodings, hashes, random strings, mojibake. It reports which language
//! and gives a calibrated score. It is built for 13 languages, one Cargo feature each
//! (`docs/languages.md`); 6.0.0 ships English, and the others land in 6.x. It is built for programs that check thousands of
//! candidates per second, such as [Ciphey](https://github.com/bee-san/Ciphey): a check is a
//! few microseconds, the core is `no_std` + `alloc` with no OS, thread, file or network
//! use, and it builds for `wasm32-unknown-unknown` with default features.
//!
//! ```
//! use gibberish_or_not::{is_gibberish, Sensitivity};
//!
//! assert!(!is_gibberish("The quick brown fox jumps over the lazy dog", Sensitivity::Medium));
//! assert!(!is_gibberish("MEETMEATTHEUSUALPLACE", Sensitivity::Medium));
//! assert!(is_gibberish("Wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj", Sensitivity::Medium));
//! assert!(is_gibberish("5f4dcc3b5aa765d61d8327deb882cf99", Sensitivity::High));
//! ```
//!
//! The detector API gives the language and the score:
//!
//! ```
//! use gibberish_or_not::{Detector, Lang, Sensitivity};
//!
//! let detector = Detector::new().sensitivity(Sensitivity::Low);
//! let d = detector.detect("Hello, my name is Bee and I like to decode things");
//! assert!(d.is_language);
//! assert_eq!(d.language, Some(Lang::En));
//! assert!(d.score >= Sensitivity::Low.min_score());
//! ```
//!
//! # What counts as language
//!
//! Text written in a supported language, in any case, with or without spaces between the
//! words (`MEETMEATNOON`), short phrases included. Identifiers (numbers, URLs, email
//! addresses, hashes, keys, CTF flags) are *neutral*: they are left out, and the rest of
//! the text decides. On its own an identifier is not language: telling what it is, is
//! [pyWhat](https://github.com/bee-san/pyWhat)'s job (`docs/design.md`).
//!
//! # Scores and sensitivity
//!
//! The score is calibrated against the false-positive rate on junk, per length bucket: a
//! text is accepted at [`Sensitivity::High`] from 0.5, at [`Sensitivity::Medium`] from 0.75
//! and at [`Sensitivity::Low`] from 0.9, and each level's junk false-positive target per
//! length bucket is in `docs/benchmark.md` (stricter for short text).

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

// The README's examples are doctests too
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod lang;
mod model;
mod pack;
mod passwords;
mod registry;
mod text;
/// JavaScript bindings (feature `wasm`): `detect`, `score`, `isGibberish`, `isPassword`,
/// `languages`.
#[cfg(feature = "wasm")]
pub mod wasm;

pub use lang::{Lang, ISO_639_1};

/// This crate's version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub use pack::{Pack, PackError};

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use once_cell::race::OnceBox;

/// How many languages of one script are scored in full, at most (see [`Detector::detect`]).
const TOP_K: usize = 3;
/// For a text of 12 or more letters, the full n-gram pass is only run for the packs of its
/// script that the cheap [`model::pre_rank`] puts first: the best PRE_MIN, and the others
/// within PRE_WINDOW of the best, PRE_MAX at most (hashed packs always get it). With 9
/// packs of one script the full passes alone would cost more than the latency budget
/// (docs/design.md, "Many languages").
const PRE_MIN: usize = 2;
const PRE_MAX: usize = 5;
const PRE_WINDOW: f32 = 0.3;
/// Scoring stops at the first language that scores at least this (Low's threshold, or the
/// detector's `min_score` if higher): no other language can then change whether the text
/// is language at any sensitivity.
const STOP_SCORE: f32 = 0.9;
/// For a text of 12 or more letters, scoring also stops at a language that scores below
/// this (or is gated): the text is junk to a language that fits it better than the rest
/// (measured on en + es: no dev recall lost; docs/design.md).
const CONTINUE_MIN: f32 = 0.01;
/// Texts with fewer letters are scored in full by every language of their script.
const ALL_PACKS_BELOW_LETTERS: u32 = 12;
/// Of those, only languages whose n-gram rank is this close to the best are scored: the
/// others are far from the text's language (measured with the pilot packs, en and es: the
/// window cut the shared dev false positives at High by a quarter for 0.1 point of
/// Spanish recall).
const RANK_WINDOW: f32 = 0.3;

/// How readily text is accepted as language. `Low` is the strictest level and `High` the
/// most lenient (as in 5.x; see [PR #9](https://github.com/bee-san/gibberish-or-not/pull/9)).
///
/// Each level is a minimum calibrated score ([`Sensitivity::min_score`]) that targets a
/// junk false-positive rate, documented in `docs/benchmark.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Sensitivity {
    /// The strictest: the fewest false positives, for callers that check many candidates
    /// that look almost like language (Ciphey's classical ciphers check their best keys at
    /// Low).
    Low,
    /// Balanced: the default.
    #[default]
    Medium,
    /// The most lenient: the fewest missed texts, more junk accepted.
    High,
}

impl Sensitivity {
    /// The minimum score at which text is accepted at this level: 0.9, 0.75 or 0.5.
    pub const fn min_score(self) -> f32 {
        match self {
            Sensitivity::Low => 0.9,
            Sensitivity::Medium => 0.75,
            Sensitivity::High => 0.5,
        }
    }
}

/// What [`Detector::detect`] found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Whether the text is natural language at the detector's sensitivity.
    pub is_language: bool,
    /// The language, when `is_language` is true.
    pub language: Option<Lang>,
    /// The calibrated score in [0, 1] of the best-scoring language (0 when no language
    /// applies): at least [`Sensitivity::min_score`] means accepted at that level.
    pub score: f32,
}

/// A configured detector: which languages, and how strict.
///
/// Cheap to clone and to build (the packs are shared and parsed once); `Send` and `Sync`.
#[derive(Clone, Debug)]
pub struct Detector {
    packs: Vec<Arc<Pack>>,
    min_score: f32,
}

impl Default for Detector {
    fn default() -> Self {
        Detector::new()
    }
}

impl Detector {
    /// A detector for every language compiled in (the default features enable all of
    /// them), at [`Sensitivity::Medium`].
    pub fn new() -> Detector {
        Detector {
            packs: registry::builtin().to_vec(),
            min_score: Sensitivity::Medium.min_score(),
        }
    }

    /// A detector without any language, to add packs to with [`Detector::with_pack`].
    pub fn empty() -> Detector {
        Detector {
            packs: Vec::new(),
            min_score: Sensitivity::Medium.min_score(),
        }
    }

    /// Keeps only these languages. Languages that aren't compiled in or added are ignored:
    /// [`Detector::enabled`] says which remain.
    pub fn languages(mut self, langs: &[Lang]) -> Detector {
        self.packs.retain(|p| langs.contains(&p.lang()));
        self
    }

    /// Accepts text from this sensitivity's minimum score.
    pub fn sensitivity(mut self, sensitivity: Sensitivity) -> Detector {
        self.min_score = sensitivity.min_score();
        self
    }

    /// Accepts text from this score, instead of a sensitivity's.
    pub fn min_score(mut self, score: f32) -> Detector {
        self.min_score = score;
        self
    }

    /// Adds a pack, replacing the pack of the same language if there is one.
    pub fn with_pack(mut self, pack: Pack) -> Detector {
        self.packs.retain(|p| p.lang() != pack.lang());
        self.packs.push(Arc::new(pack));
        self
    }

    /// The languages this detector checks.
    pub fn enabled(&self) -> impl Iterator<Item = Lang> + '_ {
        self.packs.iter().map(|p| p.lang())
    }

    /// Whether `text` is natural language, in which language, and the score.
    pub fn detect(&self, text: &str) -> Detection {
        let (score, language) = self.best(text);
        let is_language = score >= self.min_score;
        Detection {
            is_language,
            language: if is_language { language } else { None },
            score,
        }
    }

    /// The language whose word list has `word` (one word, any case, apostrophes allowed)
    /// most frequently, if one does. For callers with word rules of their own (Ciphey's
    /// checks of split words and flag contents).
    pub fn word_language(&self, word: &str) -> Option<Lang> {
        self.packs
            .iter()
            .filter_map(|p| model::word_logp(p, word).map(|lp| (lp, p.lang())))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, l)| l)
    }

    /// The opposite of `detect(text).is_language`.
    pub fn is_gibberish(&self, text: &str) -> bool {
        !self.detect(text).is_language
    }

    /// The best score and its language, whatever the sensitivity.
    pub(crate) fn best(&self, text: &str) -> (f32, Option<Lang>) {
        let a = text::analyze(text);
        if model::vetoed(&a) {
            return (0.0, None);
        }
        let main = a.main_scripts();
        let long = a.letters >= ALL_PACKS_BELOW_LETTERS;
        // Packs compete within a script: their ranks compare only there. A text that is
        // half Latin and half Han has candidates of both, chosen and stopped apart, so that
        // one script's rejection can't stop the other's packs. Han and kana are one group
        // (Chinese and Japanese share Han).
        const CJK: u8 = lang::script::HAN | lang::script::KANA;
        let group_of = |p: &Pack| {
            if p.scripts() & CJK != 0 {
                p.scripts() | CJK
            } else {
                p.scripts()
            }
        };
        // The groups present, as a set of their values (scripts are 7 bits)
        let mut groups = 0u128;
        for p in self.packs.iter().filter(|p| p.scripts() & main != 0) {
            groups |= 1 << group_of(p);
        }
        let stop = STOP_SCORE.max(self.min_score);
        let mut best: Option<(f32, f32, Lang)> = None;
        while groups != 0 {
            let group = groups.trailing_zeros() as u8;
            groups &= groups - 1;
            let in_group = |p: &Pack| p.scripts() & main != 0 && group_of(p) == group;
            if let Some((score, margin, lang)) = self.best_in(&a, long, stop, in_group) {
                if best.is_none_or(|(s0, m0, _)| (score, margin) > (s0, m0)) {
                    best = Some((score, margin, lang));
                }
                if score >= stop {
                    break;
                }
            }
        }
        best.map_or((0.0, None), |(score, _, lang)| (score, Some(lang)))
    }

    /// The best (score, margin, language) among the packs `in_group` accepts, one script's
    /// packs: pre-rank and pre-gate (texts of 12+ letters), the full n-gram pass of the best,
    /// then candidates in rank order until one scores `stop`, or, of a long text, one scores
    /// under CONTINUE_MIN or is gated (docs/design.md, "Many languages").
    fn best_in(
        &self,
        a: &text::Analysis<'_>,
        long: bool,
        stop: f32,
        in_group: impl Fn(&Pack) -> bool,
    ) -> Option<(f32, f32, Lang)> {
        let mut packs: Vec<(&Pack, f32)> = self
            .packs
            .iter()
            .filter(|p| in_group(p))
            .map(|p| (&**p, f32::INFINITY))
            .collect();
        if long
            && (packs.len() > PRE_MIN
                || packs.iter().any(|(p, _)| p.calibration.pregate.is_finite()))
        {
            let lead = model::Lead::new(a);
            for (p, r) in &mut packs {
                // Hashed models have no pre-rank: they are kept (sorted first)
                *r = model::pre_rank(p, &lead).unwrap_or(f32::INFINITY);
            }
            packs.retain(|(p, r)| *r >= p.calibration.pregate);
            if packs.len() > PRE_MIN {
                // Stable: on a tie, the earlier pack (registry order) first
                packs.sort_by(|x, y| y.1.total_cmp(&x.1));
                let best = packs
                    .iter()
                    .map(|p| p.1)
                    .find(|r| r.is_finite())
                    .unwrap_or(f32::INFINITY);
                let mut dense = 0usize;
                packs.retain(|&(_, r)| {
                    if r == f32::INFINITY {
                        return true;
                    }
                    dense += 1;
                    dense <= PRE_MIN || (dense <= PRE_MAX && r >= best - PRE_WINDOW)
                });
            }
        }
        // The full n-gram pass ranks them. Of a long text, only the best TOP_K within
        // RANK_WINDOW of the best are looked at; short texts get all (their ranks are noisy,
        // and they are cheap).
        let mut candidates: Vec<(&Pack, model::NgramPass, f32)> = packs
            .iter()
            .filter_map(|&(p, _)| {
                let ng = model::ngram_pass(p, a)?;
                let rank = ng.rank(p);
                Some((p, ng, rank))
            })
            .collect();
        candidates.sort_by(|x, y| y.2.total_cmp(&x.2));
        if long && candidates.len() > 1 {
            candidates.truncate(TOP_K);
            let top = candidates[0].2;
            candidates.retain(|c| c.2 >= top - RANK_WINDOW);
        }
        let mut best: Option<(f32, f32, Lang)> = None;
        for (pack, ng, rank) in &candidates {
            // A pack whose gate the text's rank is below scores 0: of a long text it settles
            // this script (the packs after it fit the text worse); a short one goes on
            if *rank < pack.calibration.gate_for(a.content_chars) {
                if long {
                    break;
                }
                continue;
            }
            // Every candidate is scored as `features` (and so `gib-bench fit`) scores it:
            // a calibration only holds for features computed the same way
            let s = model::score_from(pack, a, ng, true);
            let better = match best {
                None => true,
                Some((score, margin, _)) => (s.score, s.margin) > (score, margin),
            };
            if better {
                best = Some((s.score, s.margin, pack.lang()));
            }
            if s.score >= stop || (long && s.score < CONTINUE_MIN) {
                break;
            }
        }
        best
    }

    /// The classifier features of `text` for each language that applies, its length without
    /// identifiers (what the calibration buckets by), the language's n-gram rank, and its
    /// pre-rank if the text has 12 or more letters and the pack a dense model (else +inf)
    /// (for `gib-bench fit`; not a stable API). Gates are not applied.
    #[doc(hidden)]
    pub fn features(&self, text: &str) -> Vec<(Lang, Vec<f32>, u32, f32, f32)> {
        let a = text::analyze(text);
        let mut out = Vec::new();
        if model::vetoed(&a) {
            return out;
        }
        let main = a.main_scripts();
        let lead = model::Lead::new(&a);
        for pack in &self.packs {
            if pack.scripts() & main == 0 {
                continue;
            }
            if let Some(ng) = model::ngram_pass(pack, &a) {
                let rank = ng.rank(pack);
                let pre = if a.letters >= ALL_PACKS_BELOW_LETTERS {
                    model::pre_rank(pack, &lead).unwrap_or(f32::INFINITY)
                } else {
                    f32::INFINITY
                };
                let (x, chars) = model::features_from(pack, &a, &ng, true);
                out.push((pack.lang(), x.to_vec(), chars, rank, pre));
            }
        }
        out
    }

    /// The best n-gram rank among the languages of the text's script (not a stable API).
    #[doc(hidden)]
    pub fn best_rank(&self, text: &str) -> Option<f32> {
        let a = text::analyze(text);
        if model::vetoed(&a) {
            return None;
        }
        let main = a.main_scripts();
        self.packs
            .iter()
            .filter(|p| p.scripts() & main != 0)
            .filter_map(|p| model::ngram_pass(p, &a).map(|ng| ng.rank(p)))
            .reduce(f32::max)
    }

    /// The names of the classifier features (not a stable API).
    #[doc(hidden)]
    pub fn feature_names() -> &'static [&'static str] {
        &pack::FEATURES
    }

    /// The packs (not a stable API).
    #[doc(hidden)]
    pub fn packs(&self) -> &[Arc<Pack>] {
        &self.packs
    }
}

pub(crate) fn default_detector() -> &'static Detector {
    static DEFAULT: OnceBox<Detector> = OnceBox::new();
    DEFAULT.get_or_init(|| Box::new(Detector::new()))
}

/// Whether `text` is natural language at the default detector's sensitivity (Medium), in
/// which language, and the score. Uses every language compiled in.
pub fn detect(text: &str) -> Detection {
    default_detector().detect(text)
}

/// Whether `text` is *not* natural language in any language compiled in, at `sensitivity`.
///
/// ```
/// use gibberish_or_not::{is_gibberish, Sensitivity};
/// assert!(!is_gibberish("call me", Sensitivity::Medium));
/// assert!(is_gibberish("q8Zr1Xk0pLm3Vb7Nc5Tw9Hd2Fg4Js6Ky1Ue8Ri0Oa3", Sensitivity::High));
/// ```
pub fn is_gibberish(text: &str, sensitivity: Sensitivity) -> bool {
    default_detector().best(text).0 < sensitivity.min_score()
}

/// Whether `word` is a word in the word list of a language compiled in (one word, any
/// case). Unlike [`is_gibberish`], it says nothing about how likely a lone word is to be
/// junk: rare words are words.
///
/// ```
/// use gibberish_or_not::is_word;
/// assert!(is_word("decode"));
/// assert!(is_word("Cryptography"));
/// assert!(!is_word("xqzvkj"));
/// ```
pub fn is_word(word: &str) -> bool {
    default_detector().word_language(word).is_some()
}

/// Whether `text` is exactly one of 10,000+ common passwords.
///
/// ```
/// use gibberish_or_not::is_password;
/// assert!(is_password("123456"));
/// assert!(!is_password("not-a-common-password"));
/// ```
pub fn is_password(text: &str) -> bool {
    passwords::PASSWORDS.contains(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensitivity_order() {
        assert!(Sensitivity::Low.min_score() > Sensitivity::Medium.min_score());
        assert!(Sensitivity::Medium.min_score() > Sensitivity::High.min_score());
        assert_eq!(Sensitivity::default(), Sensitivity::Medium);
    }

    #[test]
    fn passwords() {
        assert!(is_password("123456"));
        assert!(is_password("password"));
        assert!(is_password("iloveyou"));
        assert!(!is_password(""));
        assert!(!is_password("this is not a password"));
    }

    #[test]
    fn detector_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<Detector>();
        check::<Pack>();
    }

    #[test]
    fn empty_detector_accepts_nothing() {
        let d = Detector::empty();
        let r = d.detect("The quick brown fox jumps over the lazy dog");
        assert!(!r.is_language);
        assert_eq!(r.language, None);
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn a_dominant_latin_letter_in_cjk_is_not_a_veto() {
        // audit F24: laughter (`www`) or a grade (`AAAA`) in CJK text
        for text in [
            "今日は楽しかったですwwwwwwww",
            "本当に面白い映画でしたwwwwwwwwww",
            "这个评级是AAAAAAAAA级的",
        ] {
            assert!(!model::vetoed(&text::analyze(text)), "{text}");
        }
        // ... and still a veto where cased letters are the text
        assert!(model::vetoed(&text::analyze("AAAAAAAAAAAAAAAAAAAAAQ==")));
    }

    #[test]
    fn odd_case_latin_in_cjk_is_not_a_veto() {
        // One odd-cased name in a sentence of another script isn't decoder output
        let a = text::analyze("我们公司的所有开发人员都在使用macOS系统工作");
        assert!(!model::vetoed(&a));
        let a = text::analyze("이 앱은 제가 매일 쓰는 ContentDB 프로그램입니다");
        assert!(!model::vetoed(&a));
        // but mostly-Latin odd case still is
        let a = text::analyze("hHeREoUtHo 你");
        assert!(model::vetoed(&a));
    }

    #[test]
    fn near_repetitions_are_vetoed() {
        for junk in [
            "AAAAAAAAAAAAAAAAAAAAAQ==",
            "AQAAAAAAAAAAAAAAAAAAAA==",
            "aaaaaaaaaaaab",
            "IIIIIIIIIIIIIIIIIQ",
        ] {
            assert!(model::vetoed(&text::analyze(junk)), "{junk}");
        }
        for text in [
            "Mississippi",
            "aardvark",
            "The quick brown fox",
            "banana bread",
        ] {
            assert!(!model::vetoed(&text::analyze(text)), "{text}");
        }
    }

    #[test]
    fn vetoes() {
        let d = Detector::new();
        for junk in [
            "",
            "   ",
            "12345 67890",
            "hello\u{0}world",
            "hHeREoUtHo",
            "!@#$%^&*()",
        ] {
            assert_eq!(d.detect(junk).score, 0.0, "{junk:?}");
        }
    }
}
