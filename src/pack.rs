//! Language packs: the data the detector scores one language with.
//!
//! A pack has two parts (layout in `docs/languages.md`, writer in
//! `languages/common/packlib.py`):
//!
//! * `model.bin`: the character n-gram model and the word table, written by the language's
//!   build script from its training sources;
//! * `calibration.txt`: the classifier's weights and score anchors per length bucket,
//!   written by `gib-bench fit` from the benchmark corpus's train and dev splits.
//!
//! Built-in packs are embedded with `include_bytes!` (see `registry.rs`). Other packs are
//! loaded at run time with [`Pack::from_bytes`] (a container of both parts, made by
//! [`Pack::to_bytes`] or `packlib.py`) or [`Pack::from_parts`].

use alloc::{boxed::Box, string::String, vec::Vec};
use core::fmt;

use crate::lang::Lang;

/// Why a pack couldn't be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackError {
    /// The data ended early, or a section points outside it.
    Truncated(&'static str),
    /// Not a pack (wrong magic number).
    BadMagic,
    /// A format version this crate can't read.
    UnsupportedVersion(u16),
    /// A section's contents are inconsistent.
    Invalid(&'static str),
    /// The calibration text is malformed, on this line (1-based).
    Calibration(usize),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackError::Truncated(what) => write!(f, "pack truncated ({what})"),
            PackError::BadMagic => f.write_str("not a gibberish-or-not pack"),
            PackError::UnsupportedVersion(v) => write!(f, "unsupported pack version {v}"),
            PackError::Invalid(what) => write!(f, "invalid pack ({what})"),
            PackError::Calibration(line) => write!(f, "malformed calibration on line {line}"),
        }
    }
}

impl core::error::Error for PackError {}

// ------------------------------------------------------------------------- hashing --
// Shared with languages/common/packlib.py: change both or neither.

pub(crate) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
pub(crate) fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// SplitMix64's finaliser.
#[inline]
pub(crate) fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
fn reduce(x: u64, n: u32) -> usize {
    (((x & 0xFFFF_FFFF) * u64::from(n)) >> 32) as usize
}

// --------------------------------------------------------------------------- bytes --

enum Bytes {
    Static(&'static [u8]),
    Owned(Box<[u8]>),
}

impl Bytes {
    #[inline]
    fn get(&self) -> &[u8] {
        match self {
            Bytes::Static(b) => b,
            Bytes::Owned(b) => b,
        }
    }
}

#[inline]
fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(b, o))
}

// --------------------------------------------------------------------- calibration --

/// The classifier's features, in the order of a calibration file's weights.
pub(crate) const FEATURES: [&str; 20] = [
    "bias",
    "ngram",
    "coverage",
    "wordscore",
    "odd_case",
    "decoder_symbols",
    "symbols",
    "punctuation",
    "foreign",
    "spaced",
    "token_length",
    "identifiers",
    "unbalanced",
    "covered_nospace",
    "ngram_nospace",
    "coverage_nospace",
    "wordscore_nospace",
    "unknown_tokens",
    "unknown_lower",
    "near_miss",
];
pub(crate) const N_FEATURES: usize = FEATURES.len();

/// One length bucket of a calibration: texts of `lo..=hi` characters, the classifier's weights, and
/// the logits that map to scores 0, 0.5 (High), 0.75 (Medium), 0.9 (Low) and 1.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Bucket {
    pub lo: u32,
    pub hi: u32,
    pub w: [f32; N_FEATURES],
    pub anchors: [f32; 5],
}

/// Score anchors: what a bucket's five anchor logits map to.
pub(crate) const ANCHOR_SCORES: [f32; 5] = [0.0, 0.5, 0.75, 0.9, 1.0];

impl Bucket {
    /// The calibrated score of logit `z`: piecewise linear through the anchors.
    #[inline]
    pub fn score(&self, z: f32) -> f32 {
        let a = &self.anchors;
        // NaN and anything at or below the first anchor score 0
        if z.partial_cmp(&a[0]) != Some(core::cmp::Ordering::Greater) {
            return 0.0;
        }
        for i in 1..5 {
            if z <= a[i] {
                let span = a[i] - a[i - 1];
                let t = if span > 0.0 {
                    (z - a[i - 1]) / span
                } else {
                    1.0
                };
                return ANCHOR_SCORES[i - 1] + t * (ANCHOR_SCORES[i] - ANCHOR_SCORES[i - 1]);
            }
        }
        1.0
    }
}

/// A pack's calibration: buckets by length in characters, in order, and the n-gram gate:
/// texts whose n-gram rank (`model::NgramPass::rank`) for this pack is below it are
/// rejected by the pack without the word model (none of the dev split's positives are).
#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct Calibration {
    pub buckets: Vec<Bucket>,
    pub gate: f32,
}

impl Calibration {
    /// Parses `calibration.txt`. Lines: `gon-calibration 1`, `lang xx`, `features <names>`
    /// (must be [`FEATURES`]), then one `bucket <lo> <hi> <weights...> | <5 anchors>` per
    /// length bucket, from 1 character up. `#` starts a comment.
    pub fn parse(text: &str) -> Result<Calibration, PackError> {
        let mut buckets = Vec::new();
        let mut header = false;
        let mut gate = f32::NEG_INFINITY;
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let err = PackError::Calibration(n + 1);
            let mut it = line.split_ascii_whitespace();
            match it.next() {
                Some("gon-calibration") => {
                    if it.next() != Some("1") {
                        return Err(err);
                    }
                    header = true;
                }
                Some("lang") => {}
                Some("gate") => {
                    gate = it
                        .next()
                        .and_then(|v| v.parse::<f32>().ok())
                        .filter(|v| !v.is_nan())
                        .ok_or(err)?;
                }
                Some("features") => {
                    if !it.eq(FEATURES.iter().copied()) {
                        return Err(err);
                    }
                }
                Some("bucket") => {
                    let fields: Vec<&str> = it.collect();
                    let num = |s: &str| s.parse::<f32>().map_err(|_| err.clone());
                    if fields.len() != 2 + N_FEATURES + 1 + 5 || fields[2 + N_FEATURES] != "|" {
                        return Err(err);
                    }
                    let lo = num(fields[0])? as u32;
                    let hi = if fields[1] == "inf" {
                        u32::MAX
                    } else {
                        num(fields[1])? as u32
                    };
                    let mut w = [0f32; N_FEATURES];
                    for (x, s) in w.iter_mut().zip(&fields[2..2 + N_FEATURES]) {
                        *x = num(s)?;
                    }
                    let mut anchors = [0f32; 5];
                    for (x, s) in anchors.iter_mut().zip(&fields[3 + N_FEATURES..]) {
                        *x = num(s)?;
                    }
                    if w.iter().chain(anchors.iter()).any(|v| !v.is_finite())
                        || anchors.windows(2).any(|p| p[0] > p[1])
                    {
                        return Err(err);
                    }
                    buckets.push(Bucket { lo, hi, w, anchors });
                }
                _ => return Err(err),
            }
        }
        if !header || buckets.is_empty() {
            return Err(PackError::Calibration(0));
        }
        // Contiguous from 1 letter, open-ended
        let mut next = 1u32;
        for (i, b) in buckets.iter().enumerate() {
            if b.lo != next || b.hi < b.lo {
                return Err(PackError::Calibration(i + 1));
            }
            next = b.hi.saturating_add(1);
        }
        if buckets.last().map(|b| b.hi) != Some(u32::MAX) {
            return Err(PackError::Calibration(0));
        }
        Ok(Calibration { buckets, gate })
    }

    #[inline]
    pub fn bucket(&self, chars: u32) -> &Bucket {
        let i = self.buckets.partition_point(|b| b.hi < chars);
        &self.buckets[i.min(self.buckets.len() - 1)]
    }

    /// Writes the calibration back as text.
    pub fn to_text(&self, lang: Lang) -> String {
        use core::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "gon-calibration 1");
        let _ = writeln!(s, "lang {}", lang.code());
        let _ = write!(s, "features");
        for f in FEATURES {
            let _ = write!(s, " {f}");
        }
        let _ = writeln!(s);
        if self.gate.is_finite() {
            let _ = writeln!(s, "gate {}", self.gate);
        }
        for b in &self.buckets {
            let _ = write!(s, "bucket {} ", b.lo);
            if b.hi == u32::MAX {
                let _ = write!(s, "inf");
            } else {
                let _ = write!(s, "{}", b.hi);
            }
            for w in b.w {
                let _ = write!(s, " {w}");
            }
            let _ = write!(s, " |");
            for a in b.anchors {
                let _ = write!(s, " {a}");
            }
            let _ = writeln!(s);
        }
        s
    }
}

// ---------------------------------------------------------------------------- model --

const MODEL_MAGIC: &[u8; 4] = b"GONM";
const PACK_MAGIC: &[u8; 4] = b"GONP";
const NONE: u16 = u16::MAX;

/// N-gram tables over the alphabet (dense) or over hashed code points (hashed).
#[derive(Clone, Debug)]
pub(crate) enum Ngrams {
    /// `order` tables of `A^k` bytes: quantised log10 P(c_k | c_1 .. c_{k-1}).
    Dense {
        order: u8,
        alpha: u16,
        scale: f32,
        offsets: [usize; 6],
    },
    /// `order` hashed tables of `2^bits` u16 slots: fingerprint (high byte) and quantised
    /// log10 P (low byte); a fingerprint miss backs off one order, adding `backoff[k]`.
    Hashed {
        order: u8,
        scale: f32,
        floor: f32,
        offsets: [usize; 6],
        bits: [u8; 6],
        backoff: [f32; 6],
    },
}

/// A language pack.
pub struct Pack {
    lang: Lang,
    scripts: u8,
    bytes: Bytes,
    pub(crate) case_overrides: Vec<(char, char)>,
    /// Symbol of each char below U+0250 and in U+0400..U+0530 (Cyrillic), `NONE` if not in
    /// the alphabet; the rest of the alphabet, sorted by char, for a binary search.
    low: Box<[u16]>,
    cyrillic: Box<[u16]>,
    rest: Vec<(u32, u16)>,
    pub(crate) ngrams: Ngrams,
    /// Word table: offset of the slots, number of buckets, class log10 P.
    words: Option<(usize, u32, [f32; 16])>,
    pub(crate) lambda: f32,
    /// Added to the mean n-gram log10 P to rank packs against each other (default: log10
    /// of the alphabet size for dense models, 0 for hashed ones).
    pub(crate) rank_offset: f32,
    pub(crate) max_word_chars: u16,
    pub(crate) seg_min_letters: u16,
    pub(crate) calibration: Calibration,
}

impl fmt::Debug for Pack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pack")
            .field("lang", &self.lang)
            .field("bytes", &self.bytes.get().len())
            .finish_non_exhaustive()
    }
}

impl Pack {
    /// Loads a pack from a `model.bin` and a `calibration.txt` embedded in the binary.
    pub fn from_static(model: &'static [u8], calibration: &str) -> Result<Pack, PackError> {
        Pack::parse(Bytes::Static(model), calibration)
    }

    /// Loads a pack from a `model.bin` and a `calibration.txt`.
    pub fn from_parts(model: &[u8], calibration: &str) -> Result<Pack, PackError> {
        Pack::parse(Bytes::Owned(model.into()), calibration)
    }

    /// Loads a pack from the single-blob form made by [`Pack::to_bytes`] (or
    /// `languages/common/packlib.py`): `GONP`, the model's length (u32 LE), the model, and
    /// the calibration text.
    pub fn from_bytes(blob: &[u8]) -> Result<Pack, PackError> {
        if blob.len() < 8 {
            return Err(PackError::Truncated("container"));
        }
        if &blob[..4] != PACK_MAGIC {
            return Err(PackError::BadMagic);
        }
        let n = u32_at(blob, 4) as usize;
        let model = blob
            .get(8..8 + n)
            .ok_or(PackError::Truncated("container model"))?;
        let calibration =
            core::str::from_utf8(&blob[8 + n..]).map_err(|_| PackError::Calibration(0))?;
        Pack::from_parts(model, calibration)
    }

    /// The single-blob form of this pack, for [`Pack::from_bytes`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let model = self.bytes.get();
        let cal = self.calibration.to_text(self.lang);
        let mut out = Vec::with_capacity(8 + model.len() + cal.len());
        out.extend_from_slice(PACK_MAGIC);
        out.extend_from_slice(&(model.len() as u32).to_le_bytes());
        out.extend_from_slice(model);
        out.extend_from_slice(cal.as_bytes());
        out
    }

    /// The pack's language.
    pub fn lang(&self) -> Lang {
        self.lang
    }

    /// Size of the model data in bytes.
    pub fn model_size(&self) -> usize {
        self.bytes.get().len()
    }

    /// Number of words in the word table.
    pub fn word_count(&self) -> usize {
        match self.words {
            Some((o, _, _)) => u32_at(self.bytes.get(), o - 72 + 4) as usize,
            None => 0,
        }
    }

    pub(crate) fn scripts(&self) -> u8 {
        self.scripts
    }

    /// A pack with this calibration instead of its own (for `gib-bench fit`).
    #[doc(hidden)]
    pub fn with_calibration_text(mut self, calibration: &str) -> Result<Pack, PackError> {
        self.calibration = Calibration::parse(calibration)?;
        Ok(self)
    }

    /// The calibration as text.
    #[doc(hidden)]
    pub fn calibration_text(&self) -> String {
        self.calibration.to_text(self.lang)
    }

    fn parse(bytes: Bytes, calibration: &str) -> Result<Pack, PackError> {
        let calibration = Calibration::parse(calibration)?;
        let b = bytes.get();
        if b.len() < 16 {
            return Err(PackError::Truncated("header"));
        }
        if &b[..4] != MODEL_MAGIC {
            return Err(PackError::BadMagic);
        }
        let version = u16_at(b, 4);
        if version != 1 {
            return Err(PackError::UnsupportedVersion(version));
        }
        let code =
            core::str::from_utf8(&b[6..8]).map_err(|_| PackError::Invalid("language code"))?;
        let lang = Lang::from_code(code).ok_or(PackError::Invalid("language code"))?;
        let scripts = b[8];
        let n_sections = u16_at(b, 10) as usize;
        let table_end = 16 + 12 * n_sections;
        if b.len() < table_end {
            return Err(PackError::Truncated("section table"));
        }
        let section = |tag: &[u8; 4]| -> Result<Option<&[u8]>, PackError> {
            for i in 0..n_sections {
                let o = 16 + 12 * i;
                if &b[o..o + 4] == tag {
                    let off = u32_at(b, o + 4) as usize;
                    let len = u32_at(b, o + 8) as usize;
                    return b
                        .get(off..off + len)
                        .map(Some)
                        .ok_or(PackError::Truncated("section"));
                }
            }
            Ok(None)
        };
        let base = |s: &[u8]| s.as_ptr() as usize - b.as_ptr() as usize;

        // Parameters
        let parm = section(b"PARM")?.ok_or(PackError::Invalid("no PARM section"))?;
        if parm.len() < 8 {
            return Err(PackError::Truncated("PARM"));
        }
        let lambda = f32_at(parm, 0);
        let max_word_chars = u16_at(parm, 4).clamp(1, 32);
        let seg_min_letters = u16_at(parm, 6);
        let rank_offset = if parm.len() >= 12 {
            f32_at(parm, 8)
        } else {
            f32::NAN
        };

        // Lower-case overrides (Turkish I -> ı ...)
        let mut case_overrides = Vec::new();
        if let Some(s) = section(b"CASE")? {
            // chunks_exact rather than as_chunks (Rust 1.88), for older toolchains
            #[allow(clippy::chunks_exact_to_as_chunks)]
            for e in s.chunks_exact(8) {
                let from = char::from_u32(u32_at(e, 0)).ok_or(PackError::Invalid("CASE"))?;
                let to = char::from_u32(u32_at(e, 4)).ok_or(PackError::Invalid("CASE"))?;
                case_overrides.push((from, to));
            }
        }

        // Alphabet
        let mut entries: Vec<(u32, u16)> = Vec::new();
        let mut alpha = 0u16;
        if let Some(s) = section(b"ALPH")? {
            if s.len() < 4 {
                return Err(PackError::Truncated("ALPH"));
            }
            alpha = u16_at(s, 0);
            let n = u16_at(s, 2) as usize;
            if s.len() < 4 + 8 * n {
                return Err(PackError::Truncated("ALPH"));
            }
            for i in 0..n {
                let o = 4 + 8 * i;
                let sym = u16_at(s, o + 4);
                if sym >= alpha {
                    return Err(PackError::Invalid("ALPH symbol"));
                }
                entries.push((u32_at(s, o), sym));
            }
            entries.sort_unstable();
        }

        // N-grams
        let ng = section(b"NGRM")?.ok_or(PackError::Invalid("no NGRM section"))?;
        if ng.len() < 12 {
            return Err(PackError::Truncated("NGRM"));
        }
        let (kind, order) = (ng[0], ng[1]);
        if !(1..=5).contains(&order) {
            return Err(PackError::Invalid("n-gram order"));
        }
        let scale = f32_at(ng, 4);
        let floor = f32_at(ng, 8);
        let mut offsets = [0usize; 6];
        let mut pos = 12usize;
        let ngrams = match kind {
            0 => {
                if alpha == 0 {
                    return Err(PackError::Invalid("dense n-grams need an alphabet"));
                }
                let mut size = 1usize;
                for offset in offsets.iter_mut().skip(1).take(order as usize) {
                    size = size
                        .checked_mul(alpha as usize)
                        .ok_or(PackError::Invalid("n-gram size"))?;
                    *offset = base(ng) + pos;
                    pos += size;
                }
                if pos > ng.len() {
                    return Err(PackError::Truncated("dense n-grams"));
                }
                Ngrams::Dense {
                    order,
                    alpha,
                    scale,
                    offsets,
                }
            }
            1 => {
                let mut bits = [0u8; 6];
                let mut backoff = [0f32; 6];
                for k in 1..=order as usize {
                    if pos + 8 > ng.len() {
                        return Err(PackError::Truncated("hashed n-grams"));
                    }
                    bits[k] = ng[pos];
                    if bits[k] > 26 {
                        return Err(PackError::Invalid("hashed n-gram size"));
                    }
                    backoff[k] = f32_at(ng, pos + 4);
                    pos += 8;
                    offsets[k] = base(ng) + pos;
                    pos += 2 << bits[k];
                }
                if pos > ng.len() {
                    return Err(PackError::Truncated("hashed n-grams"));
                }
                Ngrams::Hashed {
                    order,
                    scale,
                    floor,
                    offsets,
                    bits,
                    backoff,
                }
            }
            _ => return Err(PackError::Invalid("n-gram kind")),
        };

        // Words
        let words = match section(b"WORD")? {
            None => None,
            Some(s) => {
                if s.len() < 72 {
                    return Err(PackError::Truncated("WORD"));
                }
                let n_buckets = u32_at(s, 0);
                let mut logp = [0f32; 16];
                for (i, x) in logp.iter_mut().enumerate() {
                    *x = f32_at(s, 8 + 4 * i);
                }
                if n_buckets == 0 || s.len() < 72 + 32 * n_buckets as usize {
                    return Err(PackError::Truncated("WORD slots"));
                }
                Some((base(s) + 72, n_buckets, logp))
            }
        };

        let mut pack = Pack {
            lang,
            scripts,
            bytes: Bytes::Static(&[]),
            case_overrides,
            low: Box::new([]),
            cyrillic: Box::new([]),
            rest: Vec::new(),
            ngrams,
            words,
            lambda,
            rank_offset,
            max_word_chars,
            seg_min_letters,
            calibration,
        };
        pack.rest = entries;
        // Fast tables for the blocks most packs use
        pack.low = (0u32..0x250).map(|u| pack.slow_symbol(u)).collect();
        pack.cyrillic = (0x400u32..0x530).map(|u| pack.slow_symbol(u)).collect();
        pack.bytes = bytes;
        Ok(pack)
    }

    /// The character after this pack's lower-casing (its overrides, then Unicode's).
    #[inline]
    pub(crate) fn lower(&self, c: char) -> char {
        if !self.case_overrides.is_empty() {
            for &(from, to) in &self.case_overrides {
                if from == c {
                    return to;
                }
            }
        }
        if c.is_ascii() {
            return c.to_ascii_lowercase();
        }
        c.to_lowercase().next().unwrap_or(c)
    }

    fn slow_symbol(&self, u: u32) -> u16 {
        let Some(c) = char::from_u32(u) else {
            return NONE;
        };
        let find = |c: char| {
            self.rest
                .binary_search_by_key(&(c as u32), |e| e.0)
                .map(|i| self.rest[i].1)
                .unwrap_or(NONE)
        };
        let s = find(c);
        if s != NONE {
            return s;
        }
        find(self.lower(c))
    }

    /// The alphabet symbol of `c`, if it is in the pack's alphabet.
    #[inline]
    pub(crate) fn symbol(&self, c: char) -> Option<u16> {
        let u = c as u32;
        let s = if u < 0x250 {
            self.low[u as usize]
        } else if (0x400..0x530).contains(&u) {
            self.cyrillic[(u - 0x400) as usize]
        } else if self.rest.is_empty() {
            NONE
        } else {
            self.slow_symbol(u)
        };
        (s != NONE).then_some(s)
    }

    /// Quantised log10 P from dense table `k` at `index`.
    #[inline]
    pub(crate) fn dense(&self, k: usize, index: usize) -> u8 {
        match &self.ngrams {
            Ngrams::Dense { offsets, .. } => self.bytes.get()[offsets[k] + index],
            Ngrams::Hashed { .. } => 0,
        }
    }

    /// log10 P of the last unit of `units` given the others, from the hashed tables,
    /// backing off to shorter contexts on a fingerprint miss, and the order that was found
    /// (0: not even the unit itself, so it gets the floor).
    pub(crate) fn hashed(&self, units: &[u32]) -> (f32, usize) {
        let Ngrams::Hashed {
            scale,
            floor,
            offsets,
            bits,
            backoff,
            ..
        } = &self.ngrams
        else {
            return (0.0, 0);
        };
        let b = self.bytes.get();
        let mut penalty = 0.0f32;
        let n = units.len();
        for k in (1..=n).rev() {
            let gram = &units[n - k..];
            let mut h = FNV_OFFSET;
            for u in gram {
                h = fnv1a(h, &u.to_le_bytes());
            }
            let h = mix(h.wrapping_add(k as u64));
            let slot = (h & ((1u64 << bits[k]) - 1)) as usize;
            let v = u16_at(b, offsets[k] + 2 * slot);
            if v != 0 && (v >> 8) as u8 == (h >> 56) as u8 {
                return (penalty - f32::from(v as u8) * scale, k);
            }
            penalty += backoff[k];
        }
        (*floor + penalty, 0)
    }

    /// The class log10 P of a normalised word (UTF-8 bytes, lower case), by its hash.
    #[inline]
    pub(crate) fn word_logp(&self, h: u64) -> Option<f32> {
        let (off, n, logp) = self.words.as_ref()?;
        let b = self.bytes.get();
        let fp = (mix(h ^ 0x5bd1_e995) >> 36) as u32;
        let b1 = reduce(h, *n);
        let mut b2 = reduce(h >> 32, *n);
        if b2 == b1 {
            b2 = (b1 + 1) % *n as usize;
        }
        for bucket in [b1, b2] {
            let o = off + 32 * bucket;
            for i in 0..8 {
                let slot = u32_at(b, o + 4 * i);
                if slot == 0 {
                    break;
                }
                if slot >> 4 == fp {
                    return Some(logp[(slot & 15) as usize]);
                }
            }
        }
        None
    }

    pub(crate) fn has_words(&self) -> bool {
        self.words.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_match_packlib() {
        // The same values are asserted in languages/common/packlib.py's self-test.
        assert_eq!(fnv1a(FNV_OFFSET, b"hello"), 0xa430_d846_80aa_bd0b);
        assert_eq!(mix(0x1234_5678_9abc_def0), 0x9629_f58e_8ec5_b906);
    }

    #[test]
    fn calibration_round_trip() {
        let mut text = String::from("gon-calibration 1\nlang en\nfeatures");
        for f in FEATURES {
            text.push(' ');
            text.push_str(f);
        }
        text.push('\n');
        let w: String = (0..N_FEATURES)
            .map(|i| alloc::format!(" {}", i as f32 * 0.5))
            .collect();
        text.push_str(&alloc::format!("bucket 1 9{w} | -1 0 1 2 3\n"));
        text.push_str(&alloc::format!("bucket 10 inf{w} | -1 0.5 1 2 4\n"));
        let c = Calibration::parse(&text).unwrap();
        assert_eq!(c.buckets.len(), 2);
        assert_eq!(c.bucket(1).hi, 9);
        assert_eq!(c.bucket(9).hi, 9);
        assert_eq!(c.bucket(10).lo, 10);
        assert_eq!(c.bucket(100_000).lo, 10);
        let again = Calibration::parse(&c.to_text(Lang::En)).unwrap();
        assert_eq!(again, c);
        let b = c.bucket(3);
        assert_eq!(b.score(-5.0), 0.0);
        assert_eq!(b.score(0.0), 0.5);
        assert_eq!(b.score(1.0), 0.75);
        assert_eq!(b.score(2.0), 0.9);
        assert!((b.score(2.5) - 0.95).abs() < 1e-6);
        assert_eq!(b.score(9.0), 1.0);
        assert_eq!(b.score(f32::NAN), 0.0);
        // Gaps, overlaps and unordered anchors are errors
        assert!(Calibration::parse(&text.replace("bucket 10", "bucket 11")).is_err());
        assert!(Calibration::parse(&text.replace("| -1 0 1 2 3", "| -1 2 1 2 3")).is_err());
        assert!(Calibration::parse("gon-calibration 1\n").is_err());
        assert!(Calibration::parse(&text.replace("features bias", "features nope")).is_err());
    }

    #[test]
    fn rejects_garbage() {
        let cal = "x";
        assert!(Pack::from_parts(b"", cal).is_err());
        assert!(Pack::from_bytes(b"GONP\xff\xff\xff\xff").is_err());
        assert!(matches!(
            Pack::from_bytes(b"NOPE0000"),
            Err(PackError::BadMagic)
        ));
    }
}
