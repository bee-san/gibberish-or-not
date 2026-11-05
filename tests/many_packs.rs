//! Detection with many packs of one script: 13 packs, English and 12 synthetic languages
//! (English's n-grams under a letter substitution each, with English's word list and
//! calibration). English keeps its text and its scores, and junk stays junk
//! (docs/design.md, "Many languages").

use gibberish_or_not::{Detector, Lang, Pack};

const CODES: [&str; 12] = [
    "sv", "da", "fi", "no", "hu", "cs", "ro", "hr", "sk", "sl", "et", "lv",
];

fn english_parts() -> (Vec<u8>, String) {
    let blob = Detector::new().packs()[0].to_bytes();
    let n = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
    (
        blob[8..8 + n].to_vec(),
        String::from_utf8(blob[8 + n..].to_vec()).unwrap(),
    )
}

/// The ALPH section's (offset, alphabet size, entries).
fn alph(model: &[u8]) -> (usize, usize, usize) {
    let u16_at = |o: usize| u16::from_le_bytes([model[o], model[o + 1]]) as usize;
    let u32_at = |o: usize| u32::from_le_bytes(model[o..o + 4].try_into().unwrap()) as usize;
    for i in 0..u16_at(10) {
        let o = 16 + 12 * i;
        if &model[o..o + 4] == b"ALPH" {
            let off = u32_at(o + 4);
            return (off, u16_at(off), u16_at(off + 2));
        }
    }
    panic!("no ALPH section");
}

/// Symbol `s` of language `k` (1..=12): `(5 s + k) mod 26` (5 is coprime with 26).
fn substitute(s: usize, k: usize) -> usize {
    (5 * s + k) % 26
}

/// English's pack as synthetic language `k`.
fn synthetic(k: usize) -> Pack {
    let (mut model, cal) = english_parts();
    model[6..8].copy_from_slice(CODES[k - 1].as_bytes());
    let (off, alpha, n) = alph(&model);
    assert_eq!(alpha, 26);
    for e in 0..n {
        let at = off + 4 + 8 * e + 4;
        let sym = u16::from_le_bytes([model[at], model[at + 1]]) as usize;
        model[at..at + 2].copy_from_slice(&(substitute(sym, k) as u16).to_le_bytes());
    }
    let cal = cal.replace("lang en", &format!("lang {}", CODES[k - 1]));
    Pack::from_parts(&model, &cal).unwrap()
}

/// English text written in synthetic language `k`: each letter is the one whose symbol in
/// language `k` is the English letter's symbol in English.
fn in_language(text: &str, k: usize) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphabetic() {
                let base = if c.is_ascii_uppercase() { b'A' } else { b'a' };
                let s = (c.to_ascii_lowercase() as u8 - b'a') as usize;
                // the letter t with substitute(t, k) == s
                let t = (0..26).find(|&t| substitute(t, k) == s).unwrap();
                (base + t as u8) as char
            } else {
                c
            }
        })
        .collect()
}

fn thirteen() -> Detector {
    let mut d = Detector::new().languages(&[Lang::En]);
    for k in 1..=12 {
        d = d.with_pack(synthetic(k));
    }
    d
}

const SENTENCES: &[&str] = &[
    "The quick brown fox jumps over the lazy dog",
    "It was the best of times, it was the worst of times",
    "Can you send me the report by the end of the day please",
    "Meet me at the usual place at ten",
    "thequickbrownfoxjumpsoverthelazydog",
    "THE TREASURE IS BURIED UNDER THE OLD OAK TREE",
];

#[test]
fn english_keeps_its_text_and_scores_among_thirteen_packs() {
    let d = thirteen();
    let alone = Detector::new().languages(&[Lang::En]);
    assert_eq!(d.packs().len(), 13);
    for s in SENTENCES {
        let r = d.detect(s);
        assert!(r.is_language, "{s}: {r:?}");
        assert_eq!(r.language, Some(Lang::En), "{s}");
        assert_eq!(r.score, alone.detect(s).score, "{s}");
    }
}

#[test]
fn text_of_a_synthetic_language_is_not_english() {
    let d = thirteen();
    let alone = Detector::new().languages(&[Lang::En]);
    for k in [1, 4, 7, 12] {
        for s in SENTENCES {
            let text = in_language(s, k);
            assert_ne!(d.detect(&text).language, Some(Lang::En), "{k} {text}");
            assert!(alone.is_gibberish(&text), "{text}");
        }
    }
}

#[test]
fn junk_stays_junk_among_thirteen_packs() {
    let d = thirteen();
    for junk in [
        "Wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj",
        "q8Zr1Xk0pLm3Vb7Nc5Tw9Hd2Fg4Js6Ky1Ue8Ri0Oa3",
        "xkqzvbnmwjrtplhgfdcsy aeiou qwxz",
        "SGVsbG8gV29ybGQgdGhpcyBpcyBiYXNlNjQ=",
    ] {
        assert!(d.is_gibberish(junk), "{junk}: {:?}", d.detect(junk));
    }
}

#[test]
fn of_two_identical_packs_the_first_registered_reports() {
    let (model, cal) = english_parts();
    let mut clone = model.clone();
    clone[6..8].copy_from_slice(b"sv");
    let clone = Pack::from_parts(&clone, &cal.replace("lang en", "lang sv")).unwrap();
    let d = Detector::new().languages(&[Lang::En]).with_pack(clone);
    let r = d.detect("The quick brown fox jumps over the lazy dog");
    let alone = Detector::new()
        .languages(&[Lang::En])
        .detect("The quick brown fox jumps over the lazy dog");
    assert_eq!(r.language, Some(Lang::En));
    assert_eq!(r.score, alone.score);
}

#[test]
fn a_pregate_rejects_long_texts_before_the_full_pass() {
    let (model, cal) = english_parts();
    let gated = cal.replace("\ngate ", "\npregate 99\ngate ");
    assert!(gated.contains("pregate 99"));
    let d = Detector::empty().with_pack(Pack::from_parts(&model, &gated).unwrap());
    // 12 or more letters: pre-gated
    assert!(d.is_gibberish("The quick brown fox jumps over the lazy dog"));
    // fewer: not
    assert!(!d.is_gibberish("hello world"));
}

#[test]
fn a_rejection_in_one_script_does_not_stop_the_other() {
    // A text in two scripts, 16 Han and 16 Latin letters: both are main scripts, so a Han
    // pack and English are both candidates. A Han pack that rejects everything (its logit,
    // at most 4, never reaches the first anchor, 100) must not change English's verdict.
    let model = std::fs::read("tests/fixtures/hashed-za.bin").unwrap();
    let cal = std::fs::read_to_string("tests/fixtures/hashed-za.calibration.txt").unwrap();
    let rejecting = cal.replace("| 0 1 2 3 4", "| 100 101 102 103 104");
    assert_ne!(rejecting, cal);
    let d = Detector::new()
        .languages(&[Lang::En])
        .with_pack(Pack::from_parts(&model, &rejecting).unwrap());
    let text = "我们明天见今天天气很好我们去公园 meet me at the usual";
    let alone = Detector::new().languages(&[Lang::En]).detect(text);
    assert!(alone.is_language, "{alone:?}");
    assert_eq!(d.detect(text), alone);
}
