//! Behaviour of the public API with the built-in packs.

use gibberish_or_not::{
    detect, is_gibberish, is_password, is_word, Detection, Detector, Lang, Pack, Sensitivity,
};

const ENGLISH: &[&str] = &[
    "The quick brown fox jumps over the lazy dog",
    "Hello, my name is Bee and I like to decode things",
    "It was the best of times, it was the worst of times",
    "Meet me at the usual place at ten",
    "THE TREASURE IS BURIED UNDER THE OLD OAK TREE",
    "MEETMEATTHEUSUALPLACEATTEN",
    "thequickbrownfoxjumpsoverthelazydog",
    "Don't stop believing!",
    "Can you send me the report by the end of the day please",
    "The password is IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR and it expires tomorrow",
    "Well done! Your flag is flag{y0u_f0und_m3}, submit it now",
    "Contact bee@skerritt.blog or see https://github.com/bee-san/Ciphey for details",
    "hello world",
    "call me",
    "attack at dawn",
    "computer",
];

const JUNK: &[&str] = &[
    // Caesar +3, ROT13 of a no-space sentence, Atbash, reversed
    "Wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj",
    "GURDHVPXOEBJASBKWHZCFBIREGURYNMLQBT",
    "Gsv jfrxp yildm ulc qfnkh levi gsv ozab wlt",
    "god yzal eht revo spmuj xof nworb kciuq ehT",
    // Encodings, hashes, random strings
    "SGVsbG8sIFdvcmxkISBUaGlzIGlzIGEgdGVzdCBvZiBjaXBoZXku",
    "736563726574",
    "5f4dcc3b5aa765d61d8327deb882cf99",
    "q8Zr1Xk0pLm3Vb7Nc5Tw9Hd2Fg4Js6Ky1Ue8Ri0Oa3",
    "xqzvkjwpfmhbt",
    "ZXJMNVBQWKLRT FGHJKL",
    // Mojibake and decoder bytes
    "ÐŸÑ€Ð¸Ð²ÐµÑ‚, ÐºÐ°Ðº Ð´ÐµÐ»Ð°?",
    "hello\u{0}world",
    // Mixed-case junk that classical ciphers make of Base64
    "hHeREoUtHo baLL NoNeININ",
    // Identifiers on their own are pyWhat's, not language
    "flag{this_is_the_flag}",
    "https://github.com/bee-san/Ciphey",
    "",
    "   ",
];

#[test]
fn english_is_language_at_medium() {
    for text in ENGLISH {
        let d = Detector::new().detect(text);
        assert!(d.is_language, "{text:?} scored {}", d.score);
        assert_eq!(d.language, Some(Lang::En), "{text:?}");
        assert!(!is_gibberish(text, Sensitivity::Medium), "{text:?}");
    }
}

#[test]
fn junk_is_gibberish_at_high() {
    for text in JUNK {
        let d = Detector::new().sensitivity(Sensitivity::High).detect(text);
        assert!(!d.is_language, "{text:?} scored {}", d.score);
        assert_eq!(d.language, None, "{text:?}");
        assert!(is_gibberish(text, Sensitivity::High), "{text:?}");
    }
}

#[test]
fn sensitivities_nest() {
    // Accepted at Low implies accepted at Medium and High.
    for text in ENGLISH.iter().chain(JUNK) {
        let low = !is_gibberish(text, Sensitivity::Low);
        let medium = !is_gibberish(text, Sensitivity::Medium);
        let high = !is_gibberish(text, Sensitivity::High);
        assert!(!low || medium, "{text:?}");
        assert!(!medium || high, "{text:?}");
    }
}

#[test]
fn score_is_calibrated_to_the_sensitivities() {
    for text in ENGLISH.iter().chain(JUNK) {
        let d = detect(text);
        assert!((0.0..=1.0).contains(&d.score), "{text:?}: {}", d.score);
        for s in [Sensitivity::Low, Sensitivity::Medium, Sensitivity::High] {
            assert_eq!(
                !is_gibberish(text, s),
                d.score >= s.min_score(),
                "{text:?} at {s:?}"
            );
        }
        assert_eq!(
            d.is_language,
            d.score >= Sensitivity::Medium.min_score(),
            "{text:?}"
        );
    }
}

#[test]
fn min_score_overrides_the_sensitivity() {
    let text = "The quick brown fox jumps over the lazy dog";
    let score = detect(text).score;
    assert!(Detector::new().min_score(score).detect(text).is_language);
    assert!(
        !Detector::new()
            .min_score(score + 0.001)
            .detect(text)
            .is_language
            || score >= 1.0
    );
}

#[test]
fn languages_filter() {
    let all: Vec<Lang> = Detector::new().enabled().collect();
    assert!(all.contains(&Lang::En));
    let none = Detector::new().languages(&[]);
    assert_eq!(none.enabled().count(), 0);
    assert!(
        !none
            .detect("The quick brown fox jumps over the lazy dog")
            .is_language
    );
    let en = Detector::new().languages(&[Lang::En]);
    assert_eq!(en.enabled().collect::<Vec<_>>(), [Lang::En]);
}

#[test]
fn packs_round_trip_through_bytes() {
    let en = Detector::new().packs()[0].clone();
    let blob = en.to_bytes();
    let pack = Pack::from_bytes(&blob).expect("a pack's own bytes load");
    assert_eq!(pack.lang(), Lang::En);
    assert_eq!(pack.model_size(), en.model_size());
    assert_eq!(pack.word_count(), en.word_count());
    let custom = Detector::empty().with_pack(pack);
    for text in ENGLISH.iter().chain(JUNK) {
        assert_eq!(
            custom.detect(text),
            Detector::new().languages(&[Lang::En]).detect(text),
            "{text:?}"
        );
    }
}

#[test]
fn bad_packs_are_errors_not_panics() {
    let en = Detector::new().packs()[0].to_bytes();
    assert!(Pack::from_bytes(b"").is_err());
    assert!(Pack::from_bytes(b"GONP").is_err());
    assert!(Pack::from_bytes(&en[..100]).is_err());
    let mut bad = en.clone();
    bad[8] = b'X'; // model magic
    assert!(Pack::from_bytes(&bad).is_err());
    // Every truncation of the model's header and section table is an error, never a panic
    for n in 0..400.min(en.len()) {
        let _ = Pack::from_bytes(&en[..n]);
    }
    // Garbage calibration
    let model_len = u32::from_le_bytes([en[4], en[5], en[6], en[7]]) as usize;
    assert!(Pack::from_parts(&en[8..8 + model_len], "nonsense").is_err());
}

#[test]
fn odd_inputs_dont_panic() {
    let long_word = "a".repeat(10_000);
    let big = "The quick brown fox jumps over the lazy dog. ".repeat(50_000);
    let inputs: Vec<String> = vec![
        String::new(),
        "\u{0}".into(),
        "\u{FFFD}\u{FFFD}".into(),
        "e\u{301}e\u{301}e\u{301} cafe\u{301}".into(),
        "שלום עולם مرحبا".into(),
        "😀😃😄 lol 😂".into(),
        "𝔥𝔢𝔩𝔩𝔬 𝔴𝔬𝔯𝔩𝔡".into(),
        "İSTANBUL ıi Iİ".into(),
        "ǅungla ǈ ǋ".into(),
        long_word,
        big,
        "{{{{[[[[((((".into(),
        "'''''' ’’’ don't'".into(),
        "a'b'c'd'e'f'g'h'i'j'k'l'm'n'o'p".into(),
        "x".repeat(5000) + " " + &"y".repeat(5000),
    ];
    for text in &inputs {
        let d: Detection = detect(text);
        assert!((0.0..=1.0).contains(&d.score), "{}", d.score);
    }
}

#[test]
fn multi_line_text() {
    for text in [
        "Roses are red,\nViolets are blue,\nSugar is sweet,\nAnd so are you.",
        "Dear Bob,\r\n\r\nThe meeting is moved to Thursday.\r\nRegards, Alice",
        "line one of the message\n\tand an indented second line",
    ] {
        assert!(Detector::new().detect(text).is_language, "{text:?}");
    }
}

#[test]
fn long_text_is_cut_at_a_character_boundary() {
    // 3- and 4-byte characters across the 4,096-byte cut
    for unit in ["€", "😀", "a€", "ab😀", "日本"] {
        let text = unit.repeat(5000);
        let d = detect(&text);
        assert!((0.0..=1.0).contains(&d.score));
        let mixed = format!(
            "{}{}",
            "The quick brown fox. ".repeat(190),
            unit.repeat(100)
        );
        let _ = detect(&mixed);
    }
}

#[test]
fn words() {
    for w in ["decode", "Cryptography", "don't", "a", "I", "exuberant"] {
        assert!(is_word(w), "{w}");
    }
    // Runs of one letter are never words (they were URL fragments and noise in the books)
    for w in ["zzzz", "www", "iii", "aa", "xqzvkj", "", "  "] {
        assert!(!is_word(w), "{w:?}");
    }
}

#[test]
fn repetitions_are_gibberish() {
    for text in [
        "zzzzzzzzzzzz",
        "NO NO NO NO NO NO",
        "A A A A A A",
        "INININININININININ",
        "and the and the and the",
        "is the is the",
    ] {
        assert!(is_gibberish(text, Sensitivity::High), "{text}");
    }
    // but not writing that repeats a word
    assert!(!is_gibberish(
        "I am very very tired of this",
        Sensitivity::Medium
    ));
}

#[test]
fn identifiers_dont_make_text_pass_more_easily() {
    // Mostly identifiers and one word: no more language than the word alone (5.x and the
    // first 6.0 candidate let the identifiers' characters make it look longer)
    let alone = detect("lunch").score;
    for text in [
        "x9 7k 2b 4q lunch 9z 3m",
        "W3 5h0uld h4v3 lunch 70g37h3r 50m3 71m3.",
        "lunch 5f4dcc3b5aa765d61d8327deb882cf99",
    ] {
        let s = detect(text).score;
        assert!(s <= alone + 1e-6, "{text}: {s} vs {alone}");
        assert!(is_gibberish(text, Sensitivity::Low), "{text}");
    }
}

#[test]
fn passwords() {
    assert!(is_password("123456"));
    assert!(is_password("password"));
    assert!(is_password("qwerty"));
    assert!(!is_password("this is not a password"));
}
