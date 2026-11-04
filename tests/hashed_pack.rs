//! The hashed n-gram model and the paths for scripts without spaces, on a tiny Han pack
//! (tests/fixtures/make_hashed_fixture.py writes it; its calibration scores coverage).

use gibberish_or_not::{Detector, Lang, Pack};

fn detector() -> Detector {
    let model = include_bytes!("fixtures/hashed-za.bin");
    let cal = include_str!("fixtures/hashed-za.calibration.txt");
    let pack = Pack::from_parts(model, cal).expect("the fixture loads");
    assert_eq!(pack.lang(), Lang::from_code("za").unwrap());
    Detector::empty().with_pack(pack)
}

/// (coverage, n-gram mean, rank) of `text` for the fixture pack.
fn features(text: &str) -> (f32, f32, f32) {
    let f = detector().features(text);
    assert_eq!(f.len(), 1, "{text:?}");
    let (_, x, _, rank) = &f[0];
    (x[2], x[1], *rank)
}

#[test]
fn short_han_text_segments_into_words() {
    let (coverage, ngram, rank) = features("我们明天见");
    assert!((coverage - 1.0).abs() < 1e-6, "coverage {coverage}");
    assert!(ngram.is_finite() && rank.is_finite());
    assert!(detector().detect("我们明天见").score > 0.5);
}

#[test]
fn text_the_model_has_seen_ranks_above_shuffled_text() {
    let (_, _, seen) = features("今天天气很好我们去公园散步吧");
    let (_, _, shuffled) = features("吧步散园公去们我好很气天天今");
    assert!(seen > shuffled, "{seen} vs {shuffled}");
}

#[test]
fn long_runs_are_segmented_in_windows() {
    // One token of 120 Han characters: longer than a token the word model looks up whole
    let long: String = "我们明天见今天天气很好我们去公园散步他明天去学校".repeat(5);
    assert!(long.chars().count() > 100);
    let (coverage, _, _) = features(&long);
    assert!(coverage > 0.8, "coverage {coverage}");
}

#[test]
fn junk_has_no_coverage() {
    let (coverage, _, _) = features("鬱鬱鬱鬱鬱鬱鬱鬱");
    assert_eq!(coverage, 0.0);
    assert!(detector().detect("鬱鬱鬱鬱鬱鬱鬱鬱").score < 0.5);
}

#[test]
fn latin_text_is_not_this_packs() {
    assert!(detector().features("hello world").is_empty());
    assert_eq!(detector().detect("hello world").score, 0.0);
}
