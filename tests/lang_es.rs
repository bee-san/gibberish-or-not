//! The Spanish pack (docs/languages.md, "Tests"): Spanish text is language and reported as
//! Spanish, Spanish run through ciphers and encodings is not.

use gibberish_or_not::{Detector, Lang, Sensitivity};

/// Ten everyday Spanish sentences, written for this test (not from the benchmark corpus).
const SENTENCES: &[&str] = &[
    "Mañana por la mañana vamos a la playa con mis primos.",
    "¿Dónde está la estación de tren más cercana?",
    "El niño se comió todo el pastel sin decir nada a nadie.",
    "No sé si podré llegar a tiempo a la reunión de esta tarde.",
    "Mi abuela siempre cocinaba arroz con pollo los domingos.",
    "¡Qué alegría verte otra vez después de tantos años!",
    "La contraseña del servidor está escrita en el cuaderno azul.",
    "Nos encontramos en la plaza mayor a las ocho de la noche.",
    "Este libro cuenta la historia de un pueblo pequeño en las montañas.",
    "Por favor, envíame el informe antes del viernes.",
];

fn spanish() -> Detector {
    Detector::new().languages(&[Lang::Es])
}

/// Upper case, as decoders print it.
fn upper(s: &str) -> String {
    s.to_uppercase()
}

/// The letters only, no spaces or punctuation (what classical ciphers leave).
fn spaceless(s: &str) -> String {
    s.chars().filter(|c| c.is_alphabetic()).collect()
}

#[test]
fn spanish_is_spanish_at_medium() {
    for text in SENTENCES {
        let d = spanish().detect(text);
        assert!(d.is_language, "{text:?} scored {}", d.score);
        assert_eq!(d.language, Some(Lang::Es), "{text:?}");
        // With every language: still Spanish
        let d = Detector::new().detect(text);
        assert!(
            d.is_language,
            "{text:?} scored {} with all languages",
            d.score
        );
        assert_eq!(d.language, Some(Lang::Es), "{text:?} with all languages");
    }
}

#[test]
fn upper_case_and_spaceless_spanish_at_high() {
    for text in SENTENCES {
        for form in [upper(text), spaceless(text), upper(&spaceless(text))] {
            let d = spanish().sensitivity(Sensitivity::High).detect(&form);
            assert!(d.is_language, "{form:?} scored {}", d.score);
            assert_eq!(d.language, Some(Lang::Es), "{form:?}");
        }
    }
}

/// Caesar shift of every letter a-z by `k` (accented letters stay, as the corpus's and
/// most tools' Caesar do).
fn caesar(s: &str, k: u8) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' => ((c as u8 - b'a' + k) % 26 + b'a') as char,
            'A'..='Z' => ((c as u8 - b'A' + k) % 26 + b'A') as char,
            _ => c,
        })
        .collect()
}

/// Rail fence: encrypted with `rails`, read back as if it had `wrong` rails.
fn railfence(s: &str, rails: usize, wrong: usize) -> String {
    fn order(n: usize, rails: usize) -> Vec<usize> {
        let cycle: Vec<usize> = (0..rails - 1).chain((1..rails).rev()).collect();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by_key(|&i| (cycle[i % cycle.len()], i));
        idx
    }
    let chars: Vec<char> = s.chars().collect();
    let enc: Vec<char> = order(chars.len(), rails)
        .iter()
        .map(|&i| chars[i])
        .collect();
    let mut out = vec![' '; enc.len()];
    for (c, &pos) in enc.iter().zip(&order(enc.len(), wrong)) {
        out[pos] = *c;
    }
    out.into_iter().collect()
}

/// UTF-8 bytes read as Latin-1 (`niÃ±o`). Text with few accents still reads as itself
/// that way (the corpus reports it as `mojibake_light`, not junk), so the junk below uses
/// short accent-dense text.
fn mojibake(s: &str) -> String {
    s.bytes().map(char::from).collect()
}

#[test]
fn spanish_junk_is_gibberish_at_high() {
    let s = SENTENCES;
    let junk: Vec<String> = vec![
        caesar(s[0], 3),
        caesar(s[2], 11),
        caesar(&spaceless(s[4]).to_uppercase(), 7),
        s[1].chars().rev().collect(),
        s[3].chars().rev().collect(),
        mojibake("¿Qué día es mañana?"),
        mojibake("Él está aquí, ¿sí?"),
        railfence(s[7], 3, 4),
        railfence(&spaceless(s[9]), 4, 3),
        // A wrong Vigenère key (CLAVE read with LLAVE): every fifth letter off
        "Monaña por la mjñana vfmos a la plnya con mis pdimos.".to_string(),
    ];
    for text in &junk {
        let d = spanish().sensitivity(Sensitivity::High).detect(text);
        assert!(!d.is_language, "{text:?} scored {}", d.score);
        let d = Detector::new().sensitivity(Sensitivity::High).detect(text);
        assert!(
            !d.is_language,
            "{text:?} scored {} with all languages",
            d.score
        );
    }
}

#[test]
fn spanish_case_rules() {
    // Ñ and accented capitals lower-case to the pack's letters
    for word in ["ESPAÑA", "CORAZÓN", "PINGÜINO"] {
        let d = spanish().sensitivity(Sensitivity::High).detect(word);
        assert!(d.is_language, "{word:?} scored {}", d.score);
        assert_eq!(d.language, Some(Lang::Es), "{word:?}");
    }
    assert_eq!(Detector::new().word_language("AÑO"), Some(Lang::Es));
}

#[test]
fn english_stays_english() {
    for text in [
        "The quick brown fox jumps over the lazy dog",
        "Meet me at the usual place at ten",
        "Can you send me the report by the end of the day please",
    ] {
        let d = Detector::new().detect(text);
        assert!(d.is_language, "{text:?} scored {}", d.score);
        assert_eq!(d.language, Some(Lang::En), "{text:?}");
    }
}
