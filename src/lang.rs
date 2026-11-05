//! Languages and scripts.

use core::fmt;

/// A language, by its ISO 639-1 code.
///
/// The 13 named variants are the languages this crate can ship packs for (see
/// `docs/languages.md`). A pack loaded with [`Pack::from_bytes`](crate::Pack::from_bytes)
/// for any other language is [`Lang::Other`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Lang {
    /// English (`en`)
    En,
    /// Spanish (`es`)
    Es,
    /// French (`fr`)
    Fr,
    /// German (`de`)
    De,
    /// Italian (`it`)
    It,
    /// Portuguese (`pt`)
    Pt,
    /// Dutch (`nl`)
    Nl,
    /// Polish (`pl`)
    Pl,
    /// Turkish (`tr`)
    Tr,
    /// Russian (`ru`)
    Ru,
    /// Chinese (`zh`)
    Zh,
    /// Japanese (`ja`)
    Ja,
    /// Korean (`ko`)
    Ko,
    /// Any other language, by its two-letter lower-case ISO 639-1 code (one of
    /// [`ISO_639_1`]).
    Other([u8; 2]),
}

/// Every ISO 639-1 code, sorted.
pub const ISO_639_1: [&str; 184] = [
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az", "ba", "be", "bg", "bh",
    "bi", "bm", "bn", "bo", "br", "bs", "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy", "da",
    "de", "dv", "dz", "ee", "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fj", "fo", "fr",
    "fy", "ga", "gd", "gl", "gn", "gu", "gv", "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu", "ja", "jv", "ka", "kg", "ki", "kj",
    "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky", "la", "lb", "lg", "li", "ln",
    "lo", "lt", "lu", "lv", "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "na", "nb",
    "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny", "oc", "oj", "om", "or", "os", "pa", "pi",
    "pl", "ps", "pt", "qu", "rm", "rn", "ro", "ru", "rw", "sa", "sc", "sd", "se", "sg", "si", "sk",
    "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw", "ta", "te", "tg", "th", "ti",
    "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty", "ug", "uk", "ur", "uz", "ve", "vi", "vo",
    "wa", "wo", "xh", "yi", "yo", "za", "zh", "zu",
];

impl Lang {
    /// The 13 languages this crate can ship packs for, in the order of `docs/languages.md`.
    pub const KNOWN: [Lang; 13] = [
        Lang::En,
        Lang::Es,
        Lang::Fr,
        Lang::De,
        Lang::It,
        Lang::Pt,
        Lang::Nl,
        Lang::Pl,
        Lang::Tr,
        Lang::Ru,
        Lang::Zh,
        Lang::Ja,
        Lang::Ko,
    ];

    /// The ISO 639-1 code, e.g. `"en"` (`"??"` for an [`Lang::Other`] that isn't one).
    pub fn code(&self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Es => "es",
            Lang::Fr => "fr",
            Lang::De => "de",
            Lang::It => "it",
            Lang::Pt => "pt",
            Lang::Nl => "nl",
            Lang::Pl => "pl",
            Lang::Tr => "tr",
            Lang::Ru => "ru",
            Lang::Zh => "zh",
            Lang::Ja => "ja",
            Lang::Ko => "ko",
            Lang::Other(code) => ISO_639_1
                .binary_search_by(|c| c.as_bytes().cmp(&code[..]))
                .map(|i| ISO_639_1[i])
                .unwrap_or("??"),
        }
    }

    /// The language with this ISO 639-1 code (any case), or `None` if `code` isn't one.
    pub fn from_code(code: &str) -> Option<Lang> {
        let b = code.as_bytes();
        if b.len() != 2 || !b.iter().all(u8::is_ascii_alphabetic) {
            return None;
        }
        let code = [b[0].to_ascii_lowercase(), b[1].to_ascii_lowercase()];
        ISO_639_1
            .binary_search_by(|c| c.as_bytes().cmp(&code[..]))
            .ok()?;
        Some(
            Lang::KNOWN
                .iter()
                .copied()
                .find(|l| l.code().as_bytes() == code)
                .unwrap_or(Lang::Other(code)),
        )
    }

    /// The language's English name, e.g. `"English"`; the code for [`Lang::Other`].
    pub fn name(&self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Es => "Spanish",
            Lang::Fr => "French",
            Lang::De => "German",
            Lang::It => "Italian",
            Lang::Pt => "Portuguese",
            Lang::Nl => "Dutch",
            Lang::Pl => "Polish",
            Lang::Tr => "Turkish",
            Lang::Ru => "Russian",
            Lang::Zh => "Chinese",
            Lang::Ja => "Japanese",
            Lang::Ko => "Korean",
            Lang::Other(_) => self.code(),
        }
    }
}

impl fmt::Display for Lang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// Writing systems, as bit flags: a pack says which scripts its language is written in,
/// and only packs whose scripts cover most of a text's letters score it.
pub(crate) mod script {
    pub const LATIN: u8 = 1;
    pub const CYRILLIC: u8 = 2;
    pub const HAN: u8 = 4;
    pub const KANA: u8 = 8;
    pub const HANGUL: u8 = 16;
    pub const GREEK: u8 = 32;
    pub const OTHER: u8 = 64;
    /// Index of each flag in per-script counters.
    pub const COUNT: usize = 7;

    /// The script of a letter.
    #[inline]
    pub fn of(c: char) -> u8 {
        let u = c as u32;
        if u < 0x80 {
            return LATIN;
        }
        match u {
            0x00C0..=0x024F | 0x1E00..=0x1EFF | 0x2C60..=0x2C7F | 0xA720..=0xA7FF => LATIN,
            0x0370..=0x03FF | 0x1F00..=0x1FFF => GREEK,
            0x0400..=0x052F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => CYRILLIC,
            0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => KANA,
            0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7AF => HANGUL,
            0x2E80..=0x2FDF
            | 0x3005..=0x3007
            | 0x3021..=0x3029
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x3134F => HAN,
            _ => OTHER,
        }
    }

    /// The counter index of a single flag.
    #[inline]
    pub fn index(flag: u8) -> usize {
        flag.trailing_zeros() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip() {
        for l in Lang::KNOWN {
            assert_eq!(Lang::from_code(l.code()), Some(l));
            assert_eq!(Lang::from_code(&l.code().to_ascii_uppercase()), Some(l));
        }
        assert_eq!(Lang::from_code("sv"), Some(Lang::Other(*b"sv")));
        assert_eq!(Lang::from_code("sv").unwrap().code(), "sv");
        assert_eq!(Lang::from_code("eng"), None);
        assert_eq!(Lang::from_code("e1"), None);
        assert_eq!(Lang::from_code("qq"), None);
        assert_eq!(Lang::Other(*b"qq").code(), "??");
    }

    #[test]
    fn scripts() {
        assert_eq!(script::of('a'), script::LATIN);
        assert_eq!(script::of('é'), script::LATIN);
        assert_eq!(script::of('ł'), script::LATIN);
        assert_eq!(script::of('ж'), script::CYRILLIC);
        assert_eq!(script::of('漢'), script::HAN);
        assert_eq!(script::of('か'), script::KANA);
        assert_eq!(script::of('ー'), script::KANA);
        assert_eq!(script::of('한'), script::HANGUL);
        assert_eq!(script::of('α'), script::GREEK);
        assert_eq!(script::of('ش'), script::OTHER);
    }
}
