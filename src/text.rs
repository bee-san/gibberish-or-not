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
    /// Characters of the chunks that aren't identifiers, plus one space between each two:
    /// the length the calibration buckets by, so identifiers don't move a text to another
    /// length bucket.
    pub content_chars: u32,
    /// Letters of cased scripts (Latin, Cyrillic, Greek) in word tokens, how many different
    /// ones (case-folded), and how often the most frequent one occurs.
    pub cased_letters: u32,
    pub distinct_cased: u32,
    pub top_cased: u32,
    /// Word tokens equal (case-folded) to one of the two before them: `NO NO NO NO`.
    pub repeats: u32,
    /// Whether the letters of the word tokens, case-folded and run together, repeat a unit
    /// of at most `PERIOD_MAX` letters three times or more (`INININ`, `prepreprepre`, `and
    /// the and the and the`), or twice when the unit is two words or more (`is the is
    /// the`).
    pub periodic: bool,
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
    // Scripts without spaces between words (Han, kana) and Hangul: a chunk is a whole
    // sentence or phrase, so a number in it (`他工作了6年`, `5년`) doesn't make it an
    // identifier; the number itself is skipped as neutral when the chunk is tokenized
    let cjk = !s.is_ascii()
        && s.chars()
            .any(|c| matches!(script::of(c), script::HAN | script::KANA | script::HANGUL));
    if !cjk && b.iter().any(u8::is_ascii_digit) {
        return true;
    }
    if !cjk && !s.is_ascii() && s.chars().any(|c| c.is_numeric()) {
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
        content_chars: 0,
        cased_letters: 0,
        distinct_cased: 0,
        top_cased: 0,
        repeats: 0,
        periodic: false,
        script_letters: [0; script::COUNT],
    };
    let mut variety = Variety::default();
    let (mut content_chunks, mut content_sum) = (0u32, 0u32);
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
        content_chunks += 1;
        content_sum += chunk.chars().count() as u32;
        tokenize_chunk(text, start, end, chunk_index, &mut a, &mut variety);
        pos = end;
    }
    a.unbalanced = square != 0 || curly != 0;
    a.content_chars = content_sum + content_chunks.saturating_sub(1);
    a.distinct_cased = variety.distinct;
    a.top_cased = variety.top;
    a.repeats = count_repeats(&a);
    a.periodic = is_periodic(&a);
    a
}

/// See [`Analysis::periodic`].
pub(crate) const PERIOD_MAX: usize = 6;
/// Only the first this many letters are looked at for periodicity.
const PERIOD_LETTERS: usize = 256;

fn is_periodic(a: &Analysis<'_>) -> bool {
    let mut buf = ['\0'; PERIOD_LETTERS];
    let mut tok = [0u16; PERIOD_LETTERS];
    let mut n = 0usize;
    'toks: for (ti, t) in a.toks.iter().enumerate() {
        for c in a.text[t.start as usize..t.end as usize].chars() {
            if !c.is_alphabetic() {
                continue;
            }
            if n == PERIOD_LETTERS {
                break 'toks;
            }
            buf[n] = if c.is_ascii() {
                c.to_ascii_lowercase()
            } else {
                c.to_lowercase().next().unwrap_or(c)
            };
            tok[n] = ti.min(u16::MAX as usize) as u16;
            n += 1;
        }
    }
    let letters = &buf[..n];
    (1..=PERIOD_MAX).any(|p| {
        // Three periods, or two of a unit of two or more words (`is the is the`, not
        // `very very`)
        let enough = n >= 3 * p || (n >= 2 * p && tok[p - 1] != tok[0]);
        enough && (p..n).all(|i| letters[i] == letters[i - p])
    })
}

/// The case-folded letters of cased scripts: counts for a–z and for up to 16 others, how
/// many different ones, and the count of the most frequent.
#[derive(Default)]
struct Variety {
    ascii: [u16; 26],
    other: [(u32, u16); 16],
    n_other: usize,
    distinct: u32,
    top: u32,
}

impl Variety {
    #[inline]
    fn add(&mut self, c: char) {
        let n = if c.is_ascii() {
            let i = (c.to_ascii_lowercase() as u32 - 'a' as u32) as usize;
            let slot = &mut self.ascii[i];
            if *slot == 0 {
                self.distinct += 1;
            }
            *slot = slot.saturating_add(1);
            *slot
        } else {
            let u = c.to_lowercase().next().unwrap_or(c) as u32;
            match self.other[..self.n_other].iter_mut().find(|e| e.0 == u) {
                Some(e) => {
                    e.1 = e.1.saturating_add(1);
                    e.1
                }
                None => {
                    self.distinct += 1;
                    if self.n_other < self.other.len() {
                        self.other[self.n_other] = (u, 1);
                        self.n_other += 1;
                    }
                    1
                }
            }
        };
        self.top = self.top.max(u32::from(n));
    }
}

/// Word tokens equal, case-folded, to one of the two tokens before them.
fn count_repeats(a: &Analysis<'_>) -> u32 {
    let word = |i: usize| &a.text[a.toks[i].start as usize..a.toks[i].end as usize];
    let same = |x: &str, y: &str| {
        x.len() == y.len()
            && if x.is_ascii() && y.is_ascii() {
                x.eq_ignore_ascii_case(y)
            } else {
                x.chars()
                    .flat_map(char::to_lowercase)
                    .eq(y.chars().flat_map(char::to_lowercase))
            }
    };
    let mut n = 0u32;
    for i in 1..a.toks.len() {
        let w = word(i);
        if same(w, word(i - 1)) || (i >= 2 && same(w, word(i - 2))) {
            n += 1;
        }
    }
    n
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
fn tokenize_chunk(
    text: &str,
    start: usize,
    end: usize,
    chunk: u16,
    a: &mut Analysis<'_>,
    variety: &mut Variety,
) {
    let s = &text[start..end];
    // Inside a chunk of Han, kana or Hangul (a whole sentence: `is_neutral`), a run of ASCII
    // letters and digits that has a digit is an identifier glued to the text (`密码hunter2`,
    // a UUID next to Hangul): skipped, like a whole identifier chunk. A run of digits alone
    // (`6年`) is skipped as a number below.
    let mut skip_until = 0usize;
    let mut cur: Option<(usize, u16, u8, CaseState, bool)> = None; // start, letters, script, case, apostrophe
    let mut last_letter_end = start;
    let mut last_char: char = ' ';
    let mut words_here = 0u32;
    let mut iter = s.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let at = start + i;
        if i < skip_until {
            continue;
        }
        if c.is_ascii_alphanumeric() && (i == 0 || !s.as_bytes()[i - 1].is_ascii_alphanumeric()) {
            let run = s[i..].bytes().take_while(u8::is_ascii_alphanumeric).count();
            let r = &s.as_bytes()[i..i + run];
            if r.iter().any(u8::is_ascii_digit) && r.iter().any(u8::is_ascii_alphabetic) {
                close(
                    &mut cur,
                    last_letter_end,
                    last_char,
                    chunk,
                    a,
                    &mut words_here,
                );
                skip_until = i + run;
                continue;
            }
        }
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
            if sc & (script::LATIN | script::CYRILLIC | script::GREEK) != 0 {
                a.cased_letters += 1;
                variety.add(c);
            }
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
        // Numbers inside a chunk that isn't an identifier: only in scripts without spaces
        // (`他工作了6年`, see `is_neutral`); neutral, like identifiers
        if c.is_numeric() {
            continue;
        }
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
    fn numbers_in_cjk_sentences() {
        assert_eq!(
            words("他在一家律师事务所工作了6年。"),
            ["他在一家律师事务所工作了", "年"]
        );
        assert_eq!(words("這本舊書值5萬日元。"), ["這本舊書值", "萬日元"]);
        assert_eq!(words("5년 동안"), ["년", "동안"]);
        let a = analyze("他工作了6年");
        assert_eq!(a.symbols, 0);
        assert_eq!(a.neutral_chunks, 0);
        // Latin chunks with digits are still identifiers
        assert_eq!(words("abc123 hello"), ["hello"]);
        // and so are letter-and-digit runs glued to CJK text
        assert_eq!(words("密码hunter2"), ["密码"]);
        assert_eq!(words("パスワードpassword123です"), ["パスワード", "です"]);
        assert_eq!(words("我用macOS开发"), ["我用", "macOS", "开发"]);
    }

    #[test]
    fn content_length_leaves_identifiers_out() {
        assert_eq!(analyze("lunch").content_chars, 5);
        assert_eq!(analyze("x9 7k 2b 4q lunch 9z 3m").content_chars, 5);
        assert_eq!(
            analyze("The password is IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR").content_chars,
            15
        );
        assert_eq!(analyze("  hello   world \n").content_chars, 11);
    }

    #[test]
    fn variety_and_repeats() {
        let a = analyze("zzzzzzzzzzzz");
        assert_eq!((a.cased_letters, a.distinct_cased), (12, 1));
        let a = analyze("NO NO NO NO NO NO");
        assert_eq!((a.distinct_cased, a.repeats), (2, 5));
        let a = analyze("no yes no yes no");
        assert_eq!(a.repeats, 3);
        let a = analyze("The quick brown fox jumps over the lazy dog");
        assert_eq!((a.distinct_cased, a.repeats), (26, 0));
        let a = analyze("Ééé ÉÉÉ");
        assert_eq!((a.distinct_cased, a.repeats), (1, 1));
        let a = analyze("哈哈哈哈");
        assert_eq!(a.cased_letters, 0);
    }

    #[test]
    fn periodic_letters() {
        for junk in [
            "INININ",
            "prepreprepre",
            "and the and the and the",
            "to the to the to the",
            "hahaha",
            "zzz",
            "is the is the",
        ] {
            assert!(analyze(junk).periodic, "{junk}");
        }
        for text in [
            "very very",
            "bye bye",
            "mama",
            "The quick brown fox",
            "very very very good",
            "MEETMEATNOON",
            "banana",
        ] {
            assert!(!analyze(text).periodic, "{text}");
        }
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
