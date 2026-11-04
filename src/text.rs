//! The language-independent pass over a text: where its words are, which tokens are
//! identifiers (neutral evidence), and the counts the classifier's shape features use.
//!
//! One pass over the characters, no allocation except the token list.

use alloc::vec::Vec;

use crate::lang::script;

/// Only the first this many bytes of a text are looked at (cut at a character boundary):
/// a longer prefix doesn't change the answer, and it bounds the time per check.
pub(crate) const MAX_BYTES: usize = 4096;

/// How the letters of a token are cased.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Case {
    /// A script without case (Han, kana, Hangul, ...).
    None,
    Lower,
    Upper,
    /// Capitals that each start a lower-case run (`Hello`, `GitHub`, `iPhone`, `McDonald`),
    /// an acronym's plural (`URLs`), or long words run together in alternating case
    /// (`hiTHEREwelcome`).
    Mixed,
    /// Cased like no written word: `hHeREoUtHo`, `baLL`, `ThE`. Ciphers that only change
    /// letters keep the random case of mixed-case input such as Base64.
    Odd,
}

/// A word token: a run of letters (with in-word apostrophes and combining marks) inside a
/// whitespace-separated chunk that isn't an identifier.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tok {
    /// Byte range in the text.
    pub start: u32,
    pub end: u32,
    /// Letters in it.
    pub letters: u16,
    /// Script of its letters (`script::*`); Han and kana runs are one token.
    pub script: u8,
    /// Index of its whitespace-separated chunk: tokens of one chunk are written together.
    pub chunk: u16,
    /// Whether it has an in-word apostrophe (`don't`, `l'aube`).
    pub apostrophe: bool,
    /// Whether it has no upper-case letter.
    pub lower: bool,
}

/// What the pass found.
pub(crate) struct Analysis<'a> {
    /// The text looked at (at most [`MAX_BYTES`]).
    pub text: &'a str,
    pub toks: Vec<Tok>,
    /// Characters, and those that aren't whitespace.
    pub chars: u32,
    pub nonspace: u32,
    /// Letters in word tokens.
    pub letters: u32,
    /// Whitespace-separated chunks: all, identifiers (URLs, emails, numbers, hashes, CTF
    /// flags ...) and those with words.
    pub chunks: u32,
    pub neutral_chunks: u32,
    pub word_chunks: u32,
    /// Outside identifiers: writing punctuation (`.,!?'"-()` ...), symbols decoders put in
    /// their output but writing hardly uses (`\^[]{}|~=<>%*+` and backtick), and any other
    /// symbol (`©`, `¤`, U+FFFD, emoji).
    pub punct: u32,
    pub decoder_symbols: u32,
    pub symbols: u32,
    /// Control characters other than tab, line feed, carriage return, vertical tab and form
    /// feed: bytes from a wrong decoding, never writing.
    pub controls: u32,
    /// Word tokens in a cased script, and how many of them are [`Case::Odd`].
    pub cased: u32,
    pub odd_case: u32,
    /// Square or curly brackets that don't balance.
    pub unbalanced: bool,
    /// Letters in upper and in lower case, in word tokens.
    pub upper_letters: u32,
    pub lower_letters: u32,
    /// Letters of word tokens per script (indexed by `script::index`).
    pub script_letters: [u32; script::COUNT],
}

impl<'a> Analysis<'a> {
    /// The word tokens' letters' scripts that make up at least half of them, as flags.
    pub fn main_scripts(&self) -> u8 {
        let mut flags = 0u8;
        for (i, &n) in self.script_letters.iter().enumerate() {
            if n > 0 && n * 2 >= self.letters {
                flags |= 1 << i;
            }
        }
        // Japanese mixes Han and kana: together they can be the majority.
        let han = self.script_letters[script::index(script::HAN)];
        let kana = self.script_letters[script::index(script::KANA)];
        if kana > 0 && (han + kana) * 2 >= self.letters {
            flags |= script::HAN | script::KANA;
        }
        flags
    }
}

#[inline]
fn is_combining_mark(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

#[inline]
fn is_apostrophe(c: char) -> bool {
    c == '\'' || c == '\u{2019}'
}

/// Writing punctuation: what separates and quotes words in prose, in any of the scripts.
#[inline]
fn is_writing_punct(c: char) -> bool {
    matches!(
        c,
        '.' | ','
            | '!'
            | '?'
            | ';'
            | ':'
            | '\''
            | '"'
            | '-'
            | '('
            | ')'
            | '/'
            | '&'
            | '_'
            | '#'
            | '@'
            | '\u{2010}'..='\u{2027}' // dashes, quotes, bullets, ellipsis
            | '\u{2039}' | '\u{203A}' // ‹ ›
            | '«' | '»' | '¿' | '¡' | '·' | '§'
            | '\u{3001}'..='\u{3003}' // 、。〃
            | '\u{3008}'..='\u{3011}' // 〈〉《》「」『』【】
            | '\u{3014}'..='\u{301F}'
            | '\u{30FB}' // ・
            | '\u{FF01}'..='\u{FF0F}' // fullwidth ！＂＃...／
            | '\u{FF1A}'..='\u{FF20}'
            | '\u{FF5E}' | '\u{FF61}'..='\u{FF65}'
    )
}

#[inline]
fn is_decoder_symbol(c: char) -> bool {
    matches!(
        c,
        '\\' | '^' | '[' | ']' | '{' | '}' | '|' | '~' | '=' | '<' | '>' | '%' | '*' | '+' | '`'
    )
}

/// Edge characters stripped from a chunk before asking whether it is an identifier.
#[inline]
fn is_edge(c: char) -> bool {
    matches!(
        c,
        '"' | '\''
            | '('
            | ')'
            | '['
            | ']'
            | '<'
            | '>'
            | ','
            | '.'
            | ';'
            | ':'
            | '!'
            | '?'
            | '\u{2018}'
            ..='\u{201F}'
                | '«'
                | '»'
                | '\u{3001}'
                | '\u{3002}'
                | '\u{FF0C}'
                | '\u{FF01}'
                | '\u{FF1F}'
    )
}

/// Whether a whitespace-separated chunk (edges stripped) is an identifier: neutral
/// evidence, neither language nor junk (docs/design.md, "Boundary with pyWhat").
///
/// * anything with a digit: numbers, dates, times, versions, hashes, keys, leetspeak;
/// * URLs (`://` or a leading `www.`), email addresses, `@mentions` and `#hashtags`;
/// * CTF flags, `prefix{...}`;
/// * eight or more hex letters (`deadbeefcafe`).
pub(crate) fn is_neutral(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() {
        return false;
    }
    if b.iter().any(u8::is_ascii_digit) {
        return true;
    }
    if !s.is_ascii() && s.chars().any(|c| c.is_numeric()) {
        return true;
    }
    if s.contains("://") || (b.len() > 4 && b[..4].eq_ignore_ascii_case(b"www.")) {
        return true;
    }
    if (b[0] == b'@' || b[0] == b'#') && b.len() > 1 {
        return true;
    }
    if let Some(at) = s.find('@') {
        if at > 0 && s[at + 1..].contains('.') {
            return true;
        }
    }
    if let Some(open) = s.find('{') {
        let prefix = &b[..open];
        if (1..=25).contains(&open)
            && b[b.len() - 1] == b'}'
            && prefix[0].is_ascii_alphabetic()
            && prefix
                .iter()
                .all(|&c| c.is_ascii_alphanumeric() || c == b'_')
        {
            return true;
        }
    }
    b.len() >= 8 && b.iter().all(u8::is_ascii_hexdigit)
}

/// Case tracking for one token.
#[derive(Default)]
struct CaseState {
    upper: u16,
    lower: u16,
    /// An upper-case letter followed by another, or ending the token after lower case.
    bad_pairs: u16,
    prev_upper: bool,
    last_upper: bool,
    /// Case runs and whether one of them (other than the first) is a single letter.
    run_len: u16,
    short_run: bool,
    runs: u16,
}

impl CaseState {
    #[inline]
    fn letter(&mut self, c: char) {
        let up = c.is_uppercase();
        let low = c.is_lowercase();
        if !up && !low {
            return;
        }
        if up {
            if self.prev_upper {
                self.bad_pairs += 1;
            }
            self.upper += 1;
        } else {
            self.lower += 1;
        }
        if self.upper + self.lower > 1 && up != self.last_upper {
            if self.run_len == 1 && self.runs > 0 {
                self.short_run = true;
            }
            self.runs += 1;
            self.run_len = 0;
        }
        self.run_len += 1;
        self.prev_upper = up;
        self.last_upper = up;
    }

    fn finish(&self, letters: u16, last_is_s: bool) -> Case {
        if self.upper == 0 && self.lower == 0 {
            return Case::None;
        }
        if self.upper == 0 {
            return Case::Lower;
        }
        if self.lower == 0 {
            return Case::Upper;
        }
        // `URLs`: an acronym's plural
        if self.lower == 1 && last_is_s && self.upper + 1 == letters {
            return Case::Upper;
        }
        let ends_upper = self.last_upper;
        if self.bad_pairs == 0 && !ends_upper {
            return Case::Mixed;
        }
        // Words run together in alternating case: every run (but the first) 2+ letters
        let last_short = self.run_len == 1 && self.runs > 0;
        if letters >= 12 && !self.short_run && !last_short {
            return Case::Mixed;
        }
        Case::Odd
    }
}

/// Runs the pass over `text`.
pub(crate) fn analyze(text: &str) -> Analysis<'_> {
    let text = if text.len() > MAX_BYTES {
        let mut end = MAX_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        &text[..end]
    } else {
        text
    };
    let mut a = Analysis {
        text,
        toks: Vec::with_capacity(text.len() / 5 + 2),
        chars: 0,
        nonspace: 0,
        letters: 0,
        chunks: 0,
        neutral_chunks: 0,
        word_chunks: 0,
        punct: 0,
        decoder_symbols: 0,
        symbols: 0,
        controls: 0,
        cased: 0,
        odd_case: 0,
        unbalanced: false,
        upper_letters: 0,
        lower_letters: 0,
        script_letters: [0; script::COUNT],
    };
    let (mut square, mut curly) = (0i32, 0i32);
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    while pos < bytes.len() {
        // Skip whitespace, counting characters and control characters
        let c = next_char(text, pos);
        if c.is_whitespace() {
            a.chars += 1;
            pos += c.len_utf8();
            continue;
        }
        // A chunk: up to the next whitespace
        let start = pos;
        let mut end = pos;
        while end < bytes.len() {
            let c = next_char(text, end);
            if c.is_whitespace() {
                break;
            }
            end += c.len_utf8();
        }
        let chunk = &text[start..end];
        let chunk_index = a.chunks.min(u16::MAX as u32) as u16;
        a.chunks += 1;
        for c in chunk.chars() {
            a.chars += 1;
            a.nonspace += 1;
            match c {
                '[' => square += 1,
                ']' => square -= 1,
                '{' => curly += 1,
                '}' => curly -= 1,
                _ => {}
            }
            if c.is_control() && !matches!(c, '\t' | '\n' | '\r' | '\u{b}' | '\u{c}') {
                a.controls += 1;
            }
        }
        let core = chunk.trim_matches(is_edge);
        if is_neutral(core) {
            a.neutral_chunks += 1;
            pos = end;
            continue;
        }
        tokenize_chunk(text, start, end, chunk_index, &mut a);
        pos = end;
    }
    a.unbalanced = square != 0 || curly != 0;
    a
}

#[inline]
fn next_char(text: &str, pos: usize) -> char {
    let b = text.as_bytes()[pos];
    if b < 0x80 {
        b as char
    } else {
        text[pos..].chars().next().unwrap_or('\u{FFFD}')
    }
}

/// Splits one non-identifier chunk into word tokens and counts its other characters.
fn tokenize_chunk(text: &str, start: usize, end: usize, chunk: u16, a: &mut Analysis<'_>) {
    let s = &text[start..end];
    let mut cur: Option<(usize, u16, u8, CaseState, bool)> = None; // start, letters, script, case, apostrophe
    let mut last_letter_end = start;
    let mut last_char: char = ' ';
    let mut words_here = 0u32;
    let mut iter = s.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let at = start + i;
        if c.is_alphabetic() {
            let sc = script::of(c);
            let compatible = |prev: u8| prev == sc || (prev | sc) == (script::HAN | script::KANA);
            if let Some((_, _, prev_sc, _, _)) = &cur {
                if !compatible(*prev_sc) {
                    close(
                        &mut cur,
                        last_letter_end,
                        last_char,
                        chunk,
                        a,
                        &mut words_here,
                    );
                }
            }
            let entry = cur.get_or_insert_with(|| (at, 0, sc, CaseState::default(), false));
            if entry.2 != sc && (entry.2 | sc) == (script::HAN | script::KANA) {
                entry.2 = script::KANA;
            }
            entry.1 = entry.1.saturating_add(1);
            entry.3.letter(c);
            a.script_letters[script::index(sc)] += 1;
            last_letter_end = at + c.len_utf8();
            last_char = c;
            continue;
        }
        if is_combining_mark(c) && cur.is_some() {
            last_letter_end = at + c.len_utf8();
            continue;
        }
        if is_apostrophe(c) && cur.is_some() && iter.peek().is_some_and(|(_, n)| n.is_alphabetic())
        {
            if let Some(entry) = cur.as_mut() {
                entry.4 = true;
                // `O'Neill`, `M'Coy`: a capital may follow the apostrophe
                entry.3.prev_upper = false;
            }
            continue;
        }
        close(
            &mut cur,
            last_letter_end,
            last_char,
            chunk,
            a,
            &mut words_here,
        );
        if is_writing_punct(c) {
            a.punct += 1;
        } else if is_decoder_symbol(c) {
            a.decoder_symbols += 1;
        } else if !c.is_control() {
            a.symbols += 1;
        }
    }
    close(
        &mut cur,
        last_letter_end,
        last_char,
        chunk,
        a,
        &mut words_here,
    );
    if words_here > 0 {
        a.word_chunks += 1;
    }
}

fn close(
    cur: &mut Option<(usize, u16, u8, CaseState, bool)>,
    end: usize,
    last_char: char,
    chunk: u16,
    a: &mut Analysis<'_>,
    words_here: &mut u32,
) {
    if let Some((start, letters, sc, case, apostrophe)) = cur.take() {
        a.upper_letters += u32::from(case.upper);
        a.lower_letters += u32::from(case.lower);
        let lower = case.upper == 0;
        let case = if sc & (script::LATIN | script::CYRILLIC | script::GREEK) != 0 {
            case.finish(letters, last_char == 's')
        } else {
            Case::None
        };
        if case != Case::None {
            a.cased += 1;
            if case == Case::Odd {
                a.odd_case += 1;
            }
        }
        a.letters += u32::from(letters);
        *words_here += 1;
        a.toks.push(Tok {
            start: start as u32,
            end: end as u32,
            letters,
            script: sc,
            chunk,
            apostrophe,
            lower,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec::Vec;

    fn words(text: &str) -> Vec<String> {
        let a = analyze(text);
        a.toks
            .iter()
            .map(|t| String::from(&text[t.start as usize..t.end as usize]))
            .collect()
    }

    fn case_of(word: &str) -> Case {
        let a = analyze(word);
        assert_eq!(a.toks.len(), 1);
        if a.cased == 0 {
            Case::None
        } else if a.odd_case == 1 {
            Case::Odd
        } else {
            // Mixed and upper/lower are all "written" cases; tests distinguish via finish()
            let mut st = CaseState::default();
            for c in word.chars() {
                st.letter(c);
            }
            st.finish(a.toks[0].letters, word.ends_with('s'))
        }
    }

    #[test]
    fn tokens_and_identifiers() {
        assert_eq!(
            words("Hello, world! How are you?"),
            ["Hello", "world", "How", "are", "you"]
        );
        assert_eq!(
            words("Don't stop: it's state-of-the-art"),
            ["Don't", "stop", "it's", "state", "of", "the", "art"]
        );
        assert_eq!(
            words("The password is IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR ok"),
            ["The", "password", "is", "ok"]
        );
        assert_eq!(
            words("Mail bee@skerritt.blog or see https://github.com/bee-san."),
            ["Mail", "or", "see"]
        );
        assert_eq!(
            words("Well done! Flag: IW{N0_0ne_can_st0p_y0u}"),
            ["Well", "done", "Flag"]
        );
        assert_eq!(words("flag{this_is_the_flag}"), Vec::<String>::new());
        assert_eq!(
            words("I paid $5.99 at 10:30 on 2024-01-05"),
            ["I", "paid", "at", "on"]
        );
        assert_eq!(words("deadbeefcafe face"), ["face"]);
        assert_eq!(words("東京タワーに行く"), ["東京タワーに行く"]);
        assert_eq!(words("Tokyo東京"), ["Tokyo", "東京"]);
        assert_eq!(words("café naïve"), ["café", "naïve"]);
    }

    #[test]
    fn counts() {
        let a = analyze("Hi!\u{0}there [x {y");
        assert_eq!(a.controls, 1);
        assert!(a.unbalanced);
        assert_eq!(a.decoder_symbols, 2);
        let a = analyze("ok (fine) [x]");
        assert!(!a.unbalanced);
        let a = analyze("a b");
        assert_eq!((a.chars, a.nonspace, a.letters, a.chunks), (3, 2, 2, 2));
    }

    #[test]
    fn cases() {
        assert_eq!(case_of("hello"), Case::Lower);
        assert_eq!(case_of("HELLO"), Case::Upper);
        assert_eq!(case_of("Hello"), Case::Mixed);
        assert_eq!(case_of("GitHub"), Case::Mixed);
        assert_eq!(case_of("iPhone"), Case::Mixed);
        assert_eq!(case_of("McDonald"), Case::Mixed);
        assert_eq!(case_of("URLs"), Case::Upper);
        assert_eq!(case_of("hiTHEREwelcomeTOlinear"), Case::Mixed);
        assert_eq!(case_of("hHeREoUtHo"), Case::Odd);
        assert_eq!(case_of("baLL"), Case::Odd);
        assert_eq!(case_of("NoNeININ"), Case::Odd);
        assert_eq!(case_of("ThE"), Case::Odd);
        assert_eq!(analyze("O'Neill").odd_case, 0);
        assert_eq!(analyze("showed M’Coy").odd_case, 0);
        assert_eq!(case_of("漢字"), Case::None);
    }

    #[test]
    fn truncates_long_text_at_a_char_boundary() {
        let long: String = "é".repeat(MAX_BYTES);
        let a = analyze(&long);
        assert!(a.text.len() <= MAX_BYTES);
        assert_eq!(a.letters as usize, a.text.chars().count());
    }

    #[test]
    fn scripts_majority() {
        let a = analyze("Привет, как дела?");
        assert_eq!(a.main_scripts(), script::CYRILLIC);
        let a = analyze("今日はいい天気ですね");
        assert_eq!(
            a.main_scripts() & (script::HAN | script::KANA),
            script::HAN | script::KANA
        );
        let a = analyze("你好世界");
        assert_eq!(a.main_scripts(), script::HAN);
    }
}
