//! The library under wasm32-unknown-unknown (node, through wasm-bindgen-test-runner):
//! `cargo test --target wasm32-unknown-unknown --test wasm`.
#![cfg(target_arch = "wasm32")]

use gibberish_or_not::{detect, is_gibberish, is_password, Detector, Lang, Pack, Sensitivity};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn english_and_junk() {
    let d = detect("The quick brown fox jumps over the lazy dog");
    assert!(d.is_language);
    assert_eq!(d.language, Some(Lang::En));
    assert!(!is_gibberish(
        "MEETMEATTHEUSUALPLACEATTEN",
        Sensitivity::Medium
    ));
    assert!(is_gibberish(
        "Wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj",
        Sensitivity::High
    ));
    assert!(is_gibberish(
        "5f4dcc3b5aa765d61d8327deb882cf99",
        Sensitivity::High
    ));
    assert!(is_password("123456"));
}

#[wasm_bindgen_test]
fn deterministic() {
    // Scores use f32 arithmetic only (no libm, no clock, no randomness).
    let d = Detector::new().sensitivity(Sensitivity::High);
    for text in [
        "hello world",
        "Attack at dawn",
        "xqzvkjwpfmhbt",
        "Gsv jfrxp yildm ulc",
    ] {
        let a = d.detect(text);
        let b = d.detect(text);
        assert_eq!(a, b);
    }
}

#[wasm_bindgen_test]
fn packs_load_from_bytes() {
    let blob = Detector::new().packs()[0].to_bytes();
    let pack = Pack::from_bytes(&blob).unwrap();
    let d = Detector::empty().with_pack(pack);
    assert!(
        d.detect("Can you send me the report by the end of the day please")
            .is_language
    );
}

#[wasm_bindgen_test]
fn js_api() {
    use gibberish_or_not::wasm;
    let d = wasm::detect("Hello, my name is Bee and I like to decode things", None);
    assert!(d.is_language());
    assert_eq!(d.language().as_deref(), Some("en"));
    assert!(wasm::is_gibberish(
        "q8Zr1Xk0pLm3Vb7Nc5Tw9Hd2Fg4Js6Ky1Ue8Ri0Oa3",
        Some("high".into())
    ));
    assert!(wasm::languages().contains(&"en".to_string()));
}
