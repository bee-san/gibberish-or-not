//! Scoring one text with one language pack: features, the classifier and its calibration.
//!
//! The features (see `docs/design.md` for why each is there):
//!
//! * character n-gram log-likelihood per letter, over the letters of the words with
//!   everything else removed, so the same model scores spaced and space-less text;
//! * word evidence: the share of letters inside known words and a word-model score per
//!   letter (known words by frequency, unknown letters at the pack's `lambda`), with
//!   Viterbi segmentation for long unknown tokens and space-less text;
//! * the text's shape: odd letter case, decoder symbols, other symbols, punctuation,
//!   letters outside the language's alphabet or script, identifiers, token length,
//!   unbalanced brackets.
//!
//! A logistic-regression logit over those features (weights per length bucket, from the
//! train split) is mapped piecewise-linearly to the calibrated score (anchors from the dev
//! split, see `pack::Bucket`).

use crate::lang::script;
use crate::pack::{fnv1a, mix, Ngrams, Pack, FNV_OFFSET, N_FEATURES};
use crate::text::Analysis;

/// Longest token (in characters) that is looked up or segmented; longer ones count as
/// unknown letters.
const MAX_TOKEN_CHARS: usize = 48;
/// At most this many letters per text are segmented (the rest count as unknown): it
/// bounds the time per check.
const SEGMENT_BUDGET: u32 = 64;
/// Tokens whose letters score below this mean log10 P per letter in the n-gram model are
/// not split into words: they are far from the language, and splitting costs a word lookup
/// per letter and word length. The value is for an alphabet of 26 symbols; packs with
/// another rank offset (`NgramPass::rank`) shift it by the difference, so that the gate
/// sits at the same rank in every language.
const SEGMENT_MIN_NGRAM: f32 = -1.45;

/// The rank offset `SEGMENT_MIN_NGRAM` was set for (26 symbols).
fn reference_rank_offset() -> f32 {
    log10_int(26)
}

/// A run of Han or kana (no spaces between words) longer than this many characters is
/// looked up and segmented in windows of this many characters.
const CJK_WINDOW_CHARS: usize = MAX_TOKEN_CHARS;
/// Han and kana characters segmented per text (their own budget: a run of them is a whole
/// sentence, and their words are short, so a lookup per character and word length costs
/// about what `SEGMENT_BUDGET` letters of an alphabetic script do).
const CJK_SEGMENT_BUDGET: u32 = 128;
/// At most this many unknown tokens per text are checked for being one letter off a word.
const NEAR_MISS_TOKENS: usize = 8;
/// ...and only when at least this share of the letters is in known words: the check is
/// for text that is almost right (a key with one wrong letter), not for junk.
const NEAR_MISS_MIN_COVERAGE: f32 = 0.25;
/// Known words at most this frequent (log10 P) are checked too: a wrong letter often makes
/// a rare word (`shem` for `them`).
const NEAR_MISS_RARE: f32 = -6.0;
/// A neighbour must be this much more frequent (log10) than the token to count.
const NEAR_MISS_GAIN: f32 = 2.0;
/// In spaced text, only tokens this long are split into words (run-together words such as
/// `passwordis`); shorter unknown tokens are typos, names or corrupted words, and a split
/// would give them credit they don't deserve (`sensence` -> `sense` + ...).
const SEGMENT_MIN_LETTERS_SPACED: u32 = 10;

/// A pack's verdict on a text.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Scored {
    /// Calibrated score in [0, 1].
    pub score: f32,
    /// The classifier's logit minus the bucket's Low anchor: for ranking languages.
    pub margin: f32,
}

/// Why a text gets score 0 before any pack looks at it.
pub(crate) fn vetoed(a: &Analysis<'_>) -> bool {
    a.controls > 0
        || a.letters == 0
        || (a.odd_case > 0 && (a.odd_case * 3 > a.cased || a.cased <= 3))
}

/// What the n-gram pass found: enough to rank the languages of a script, and the first
/// features.
pub(crate) struct NgramPass {
    /// Letters in the pack's scripts.
    pub letters: u32,
    /// Sum of log10 P over the scored letters, and how many were scored.
    pub ng_sum: f32,
    pub scored: u32,
    /// Letters outside the pack's alphabet or scripts.
    pub foreign: u32,
}

/// The pack's rank offset (see [`NgramPass::rank`]).
pub(crate) fn rank_offset(pack: &Pack) -> f32 {
    if pack.rank_offset.is_finite() {
        pack.rank_offset
    } else {
        match &pack.ngrams {
            Ngrams::Dense { alpha, .. } => log10_int(u32::from(*alpha)),
            Ngrams::Hashed { .. } => 0.0,
        }
    }
}

impl NgramPass {
    /// How well the text fits the pack's n-gram model, comparable across packs: mean
    /// log10 P per letter plus the pack's rank offset (by default log10 of the alphabet
    /// size, so that a uniform model scores 0), with each foreign letter at -3.
    pub fn rank(&self, pack: &Pack) -> f32 {
        let per = rank_offset(pack);
        let n = (self.scored + self.foreign).max(1) as f32;
        (self.ng_sum + per * self.scored as f32 - 3.0 * self.foreign as f32) / n
    }
}

/// log10 of a small integer, without libm (alphabet sizes are under 1000).
fn log10_int(n: u32) -> f32 {
    // log10(n) = log10(m * 2^e) with m in [1, 2): a short series is plenty here
    let (mut m, mut e) = (n.max(1) as f32, 0f32);
    while m >= 2.0 {
        m /= 2.0;
        e += 1.0;
    }
    // ln(m) for m in [1,2) by atanh series
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let ln_m = 2.0 * y * (1.0 + y2 / 3.0 + y2 * y2 / 5.0 + y2 * y2 * y2 / 7.0);
    (ln_m + e * core::f32::consts::LN_2) / core::f32::consts::LN_10
}

/// The n-gram pass of `pack` over `a`, or `None` if the text has no letters in the
/// pack's scripts.
pub(crate) fn ngram_pass(pack: &Pack, a: &Analysis<'_>) -> Option<NgramPass> {
    let scripts = pack.scripts();
    let in_script = |s: u8| s & scripts != 0;
    let letters: u32 = a
        .toks
        .iter()
        .filter(|t| in_script(t.script))
        .map(|t| u32::from(t.letters))
        .sum();
    if letters == 0 {
        return None;
    }
    let text = a.text;
    // Words in other scripts are another language's business: neutral, like identifiers
    // (`Paul (Russian: Собор Святых Апостолов) is the cathedral ...`). Only letters of the
    // pack's scripts that its alphabet lacks are foreign.
    let mut foreign = 0u32;
    let (mut ng_sum, mut scored) = (0f32, 0u32);
    match &pack.ngrams {
        Ngrams::Dense {
            order,
            alpha,
            scale,
            ..
        } => {
            let (order, alpha) = (*order as usize, *alpha as usize);
            let modulus = alpha.pow(order as u32 - 1);
            let (mut window, mut n) = (0usize, 0usize);
            let mut q_sum = 0u32;
            let mut prev_chunk: Option<u16> = None;
            for t in a.toks.iter().filter(|t| in_script(t.script)) {
                if prev_chunk.is_some_and(|p| t.chunk > p + 1) {
                    n = 0;
                    window = 0;
                }
                prev_chunk = Some(t.chunk);
                for c in text[t.start as usize..t.end as usize].chars() {
                    if !c.is_alphabetic() {
                        continue;
                    }
                    let Some(s) = pack.symbol(c) else {
                        foreign += 1;
                        n = 0;
                        window = 0;
                        continue;
                    };
                    let idx = window * alpha + s as usize;
                    q_sum += u32::from(pack.dense(n + 1, idx));
                    scored += 1;
                    if n + 1 < order {
                        window = idx;
                        n += 1;
                    } else {
                        window = idx % modulus;
                    }
                }
            }
            ng_sum = -(q_sum as f32) * scale;
        }
        Ngrams::Hashed { order, .. } => {
            let order = *order as usize;
            let mut units = [0u32; 6];
            let mut n = 0usize;
            let mut prev_chunk: Option<u16> = None;
            for t in a.toks.iter().filter(|t| in_script(t.script)) {
                if prev_chunk.is_some_and(|p| t.chunk > p + 1) {
                    n = 0;
                }
                prev_chunk = Some(t.chunk);
                for c in text[t.start as usize..t.end as usize].chars() {
                    if !c.is_alphabetic() {
                        continue;
                    }
                    let u = pack.lower(c) as u32;
                    if n + 1 > order {
                        units.copy_within(1..order, 0);
                        n = order - 1;
                    }
                    units[n] = u;
                    n += 1;
                    let (lp, hit) = pack.hashed(&units[..n]);
                    if hit == 0 {
                        foreign += 1;
                    }
                    ng_sum += lp;
                    scored += 1;
                }
            }
        }
    }
    Some(NgramPass {
        letters,
        ng_sum,
        scored,
        foreign,
    })
}

/// Mean log10 P per letter of one token on its own (for the segmentation gate).
fn token_ngram(pack: &Pack, word: &str) -> f32 {
    let (mut sum, mut scored) = (0f32, 0u32);
    match &pack.ngrams {
        Ngrams::Dense {
            order,
            alpha,
            scale,
            ..
        } => {
            let (order, alpha) = (*order as usize, *alpha as usize);
            let modulus = alpha.pow(order as u32 - 1);
            let (mut window, mut n, mut q) = (0usize, 0usize, 0u32);
            for c in word.chars().filter(|c| c.is_alphabetic()) {
                let Some(s) = pack.symbol(c) else {
                    return f32::NEG_INFINITY;
                };
                let idx = window * alpha + s as usize;
                q += u32::from(pack.dense(n + 1, idx));
                scored += 1;
                if n + 1 < order {
                    window = idx;
                    n += 1;
                } else {
                    window = idx % modulus;
                }
            }
            sum = -(q as f32) * scale;
        }
        Ngrams::Hashed { order, .. } => {
            let order = *order as usize;
            let (mut units, mut n) = ([0u32; 6], 0usize);
            for c in word.chars().filter(|c| c.is_alphabetic()) {
                if n + 1 > order {
                    units.copy_within(1..order, 0);
                    n = order - 1;
                }
                units[n] = pack.lower(c) as u32;
                n += 1;
                sum += pack.hashed(&units[..n]).0;
                scored += 1;
            }
        }
    }
    if scored == 0 {
        f32::NEG_INFINITY
    } else {
        sum / scored as f32
    }
}

/// The features, given the n-gram pass. `segment`: whether long unknown tokens may be split
/// into words (only the best-ranked languages of a text do it: it is the costly part).
pub(crate) fn features_from(
    pack: &Pack,
    a: &Analysis<'_>,
    ng: &NgramPass,
    segment: bool,
) -> ([f32; N_FEATURES], u32) {
    let scripts = pack.scripts();
    let in_script = |s: u8| s & scripts != 0;
    let text = a.text;
    let letters = ng.letters;
    let foreign = ng.foreign;
    let ngram = if ng.scored > 0 {
        ng.ng_sum / ng.scored as f32
    } else {
        -5.0
    };

    // Words
    let (mut known, mut wsum) = (0u32, 0f32);
    let mut chunks_with_words = 0u32;
    let mut last_chunk: Option<u16> = None;
    let mut ntoks = 0u32;
    let mut budget = if segment { SEGMENT_BUDGET } else { 0 };
    let mut cjk_budget = if segment { CJK_SEGMENT_BUDGET } else { 0 };
    let (mut unknown_tokens, mut unknown_lower) = (0u32, 0u32);
    // Tokens to check for being one letter off a word, with their own log10 P (or None)
    let mut suspects = [(0usize, f32::NEG_INFINITY); NEAR_MISS_TOKENS];
    let mut n_suspects = 0usize;
    for t in a.toks.iter().filter(|t| in_script(t.script)) {
        if last_chunk != Some(t.chunk) {
            chunks_with_words += 1;
            last_chunk = Some(t.chunk);
        }
    }
    let spaced_text = chunks_with_words >= 2;
    for (ti, t) in a
        .toks
        .iter()
        .enumerate()
        .filter(|(_, t)| in_script(t.script))
    {
        ntoks += 1;
        let letters_here = u32::from(t.letters);
        if !pack.has_words() {
            wsum += pack.lambda * letters_here as f32;
            continue;
        }
        let word = &text[t.start as usize..t.end as usize];
        let w = if t.script & (script::HAN | script::KANA) != 0
            && usize::from(t.letters) > CJK_WINDOW_CHARS
        {
            cjk_windows(pack, word, &mut cjk_budget)
        } else {
            word_evidence(pack, word, t.apostrophe, spaced_text, &mut budget)
        };
        wsum += w.logp;
        let k = w.known.min(letters_here);
        known += k;
        // A token mostly outside known words. In text with both cases, unknown words in
        // lower case are rare in writing (unknown words are mostly names, capitalised) and
        // common in partly decrypted ciphertext (`He lookec embarrarsed`).
        if k * 2 < letters_here {
            unknown_tokens += 1;
            if t.lower {
                unknown_lower += letters_here;
            }
        }
        let rare = w.whole.is_none_or(|lp| lp <= NEAR_MISS_RARE);
        if rare && (2..=12).contains(&letters_here) && n_suspects < NEAR_MISS_TOKENS {
            suspects[n_suspects] = (ti, w.whole.unwrap_or(f32::NEG_INFINITY));
            n_suspects += 1;
        }
    }
    let mut near_miss = 0u32;
    if pack.has_words() && (known as f32) >= NEAR_MISS_MIN_COVERAGE * letters as f32 {
        for &(ti, own) in &suspects[..n_suspects] {
            let t = &a.toks[ti];
            if is_near_miss(pack, &text[t.start as usize..t.end as usize], own) {
                near_miss += 1;
            }
        }
    }
    let lf = letters as f32;
    let coverage = known as f32 / lf;
    let wordscore = wsum / lf;
    let spaced = if chunks_with_words >= 2 { 1.0 } else { 0.0 };
    let nonspace = a.nonspace.max(1) as f32;
    let odd = if a.cased > 0 {
        a.odd_case as f32 / a.cased as f32
    } else {
        0.0
    };
    let toklen = (lf / ntoks.max(1) as f32).min(20.0) / 10.0;
    let x = [
        1.0,
        ngram,
        coverage,
        wordscore,
        odd,
        a.decoder_symbols as f32 / nonspace,
        a.symbols as f32 / nonspace,
        a.punct as f32 / nonspace,
        foreign as f32 / lf,
        spaced,
        toklen,
        a.neutral_chunks as f32 / a.chunks.max(1) as f32,
        if a.unbalanced { 1.0 } else { 0.0 },
        if spaced == 0.0 && known >= letters {
            1.0
        } else {
            0.0
        },
        ngram * (1.0 - spaced),
        coverage * (1.0 - spaced),
        wordscore * (1.0 - spaced),
        unknown_tokens as f32 / ntoks.max(1) as f32,
        if a.upper_letters > 0 && a.lower_letters > 0 {
            unknown_lower as f32 / lf
        } else {
            0.0
        },
        near_miss as f32 / ntoks.max(1) as f32,
    ];
    (x, a.chars)
}

/// The pack's verdict on `a`, given its n-gram pass.
pub(crate) fn score_from(pack: &Pack, a: &Analysis<'_>, ng: &NgramPass, segment: bool) -> Scored {
    let (x, chars) = features_from(pack, a, ng, segment);
    let bucket = pack.calibration.bucket(chars);
    let z: f32 = bucket.w.iter().zip(x.iter()).map(|(w, x)| w * x).sum();
    Scored {
        score: bucket.score(z),
        margin: z - bucket.anchors[3],
    }
}

/// Writes `word` lower-cased (the pack's way), apostrophes as `'`, without combining marks,
/// into `buf`; returns the length and the byte offset where each character starts (and
/// the end), or `None` if it doesn't fit.
fn normalise(
    pack: &Pack,
    word: &str,
    buf: &mut [u8; 4 * MAX_TOKEN_CHARS],
    bounds: &mut [u8; MAX_TOKEN_CHARS + 1],
) -> Option<(usize, usize)> {
    let mut len = 0usize;
    let mut chars = 0usize;
    for c in word.chars() {
        if matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
        {
            continue;
        }
        let c = if c == '\u{2019}' { '\'' } else { pack.lower(c) };
        if chars == MAX_TOKEN_CHARS || len + c.len_utf8() > buf.len() {
            return None;
        }
        bounds[chars] = len as u8;
        c.encode_utf8(&mut buf[len..]);
        len += c.len_utf8();
        chars += 1;
    }
    bounds[chars] = len as u8;
    Some((len, chars))
}

/// Whether a rare or unknown token (`own`: its log10 P, or -inf) becomes a much more
/// frequent word when one of its letters moves one place in the alphabet (a-z, or а-я):
/// the signature of a cipher key with one letter off by one (`romethinf` for `something`,
/// `shem` for `them`), which the rest of the text doesn't give away.
fn is_near_miss(pack: &Pack, word: &str, own: f32) -> bool {
    let mut buf = [0u8; 4 * MAX_TOKEN_CHARS];
    let mut bounds = [0u8; MAX_TOKEN_CHARS + 1];
    let Some((len, chars)) = normalise(pack, word, &mut buf, &mut bounds) else {
        return false;
    };
    let need = own + NEAR_MISS_GAIN;
    for i in 0..chars {
        let (s, e) = (bounds[i] as usize, bounds[i + 1] as usize);
        let Some(c) = core::str::from_utf8(&buf[s..e])
            .ok()
            .and_then(|x| x.chars().next())
        else {
            continue;
        };
        for delta in [1i32, -1] {
            let shifted = match c {
                'a'..='z' => char::from_u32(
                    ((c as u32 - 'a' as u32) as i32 + delta).rem_euclid(26) as u32 + 'a' as u32,
                ),
                '\u{430}'..='\u{44F}' => char::from_u32(
                    ((c as u32 - 0x430) as i32 + delta).rem_euclid(32) as u32 + 0x430,
                ),
                _ => None,
            };
            let Some(d) = shifted else { continue };
            let mut tmp = [0u8; 4];
            let enc = d.encode_utf8(&mut tmp).as_bytes();
            if enc.len() != e - s {
                continue;
            }
            let h = fnv1a(fnv1a(fnv1a(FNV_OFFSET, &buf[..s]), enc), &buf[e..len]);
            if pack.word_logp(mix(h)).is_some_and(|lp| lp >= need) {
                return true;
            }
        }
    }
    false
}

/// log10 P of `word` (one word, any case) in the pack's word list, if it is there.
pub(crate) fn word_logp(pack: &Pack, word: &str) -> Option<f32> {
    let mut buf = [0u8; 4 * MAX_TOKEN_CHARS];
    let mut bounds = [0u8; MAX_TOKEN_CHARS + 1];
    let (len, _) = normalise(pack, word.trim(), &mut buf, &mut bounds)?;
    if len == 0 {
        return None;
    }
    pack.word_logp(mix(fnv1a(FNV_OFFSET, &buf[..len])))
}

/// Word evidence of a long run of Han or kana, window by window (a window may cut a word:
/// at most one per window).
fn cjk_windows(pack: &Pack, word: &str, budget: &mut u32) -> WordEvidence {
    let mut out = WordEvidence {
        logp: 0.0,
        known: 0,
        whole: None,
    };
    let mut start = 0usize;
    let mut chars = 0usize;
    for (i, _) in word.char_indices() {
        if chars == CJK_WINDOW_CHARS {
            let w = word_evidence(pack, &word[start..i], false, false, budget);
            out.logp += w.logp;
            out.known += w.known;
            start = i;
            chars = 0;
        }
        chars += 1;
    }
    let w = word_evidence(pack, &word[start..], false, false, budget);
    out.logp += w.logp;
    out.known += w.known;
    out
}

/// What the word model says about one token.
struct WordEvidence {
    /// log10 score: the word's, or its best split's, or `lambda` per letter.
    logp: f32,
    /// Letters inside known words.
    known: u32,
    /// log10 P of the whole token, if it is a known word.
    whole: Option<f32>,
}

/// The word model's evidence for one token. In spaced text, only long tokens are split
/// into words, and only a split without unknown letters counts.
fn word_evidence(
    pack: &Pack,
    word: &str,
    apostrophe: bool,
    spaced: bool,
    budget: &mut u32,
) -> WordEvidence {
    let mut buf = [0u8; 4 * MAX_TOKEN_CHARS];
    let mut bounds = [0u8; MAX_TOKEN_CHARS + 1];
    let letters = word.chars().filter(|c| c.is_alphabetic()).count() as u32;
    let unknown = WordEvidence {
        logp: pack.lambda * letters as f32,
        known: 0,
        whole: None,
    };
    let Some((len, chars)) = normalise(pack, word, &mut buf, &mut bounds) else {
        return unknown;
    };
    let bytes = &buf[..len];
    if let Some(lp) = pack.word_logp(mix(fnv1a(FNV_OFFSET, bytes))) {
        return WordEvidence {
            logp: lp,
            known: letters,
            whole: Some(lp),
        };
    }
    if apostrophe {
        // `l'aube` -> `l` + `aube`, `rock'n'roll` -> `rock` + `n` + `roll`
        let (mut lp, mut k) = (0f32, 0u32);
        for part in bytes.split(|&b| b == b'\'') {
            let n = core::str::from_utf8(part)
                .map(|p| p.chars().count())
                .unwrap_or(0) as u32;
            if n == 0 {
                continue;
            }
            match pack.word_logp(mix(fnv1a(FNV_OFFSET, part))) {
                Some(p) => {
                    lp += p;
                    k += n;
                }
                None => lp += pack.lambda * n as f32,
            }
        }
        return WordEvidence {
            logp: lp,
            known: k,
            whole: None,
        };
    }
    let min = if spaced {
        SEGMENT_MIN_LETTERS_SPACED
    } else {
        u32::from(pack.seg_min_letters.max(2))
    };
    let gate = SEGMENT_MIN_NGRAM - (rank_offset(pack) - reference_rank_offset());
    if letters >= min && letters <= *budget && token_ngram(pack, word) >= gate {
        *budget -= letters;
        let (lp, k) = segment(pack, &buf, &bounds, chars);
        if !spaced || k >= letters {
            return WordEvidence {
                logp: lp,
                known: k,
                whole: None,
            };
        }
    }
    unknown
}

/// The best split of a normalised token into known words and unknown letters (each at the
/// pack's `lambda`): its log10 score and the letters inside known words.
fn segment(pack: &Pack, buf: &[u8], bounds: &[u8; MAX_TOKEN_CHARS + 1], n: usize) -> (f32, u32) {
    let mut best = [f32::NEG_INFINITY; MAX_TOKEN_CHARS + 1];
    let mut known = [0u16; MAX_TOKEN_CHARS + 1];
    best[0] = 0.0;
    let max_word = pack.max_word_chars as usize;
    for j in 0..n {
        let here = best[j];
        if here == f32::NEG_INFINITY {
            continue;
        }
        let unknown = here + pack.lambda;
        if unknown > best[j + 1] {
            best[j + 1] = unknown;
            known[j + 1] = known[j];
        }
        let mut h = FNV_OFFSET;
        for e in j + 1..=(j + max_word).min(n) {
            h = fnv1a(h, &buf[bounds[e - 1] as usize..bounds[e] as usize]);
            if let Some(lp) = pack.word_logp(mix(h)) {
                let cand = here + lp;
                if cand > best[e] {
                    best[e] = cand;
                    known[e] = known[j] + (e - j) as u16;
                }
            }
        }
    }
    (best[n], u32::from(known[n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log10_without_libm() {
        for n in [1u32, 2, 10, 26, 33, 100, 999] {
            let want = (n as f64).log10() as f32;
            assert!(
                (log10_int(n) - want).abs() < 1e-3,
                "{n}: {} vs {want}",
                log10_int(n)
            );
        }
    }
}
