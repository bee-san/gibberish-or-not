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
    // Characters the pack has never seen
    let (coverage, _, _) = features("鬱鬮鬯鬲鬳鬴鬵鬶");
    assert_eq!(coverage, 0.0);
    assert!(detector().detect("鬱鬮鬯鬲鬳鬴鬵鬶").score < 0.5);
    // One character repeated is vetoed before any pack scores it
    assert!(detector().features("我我我我我我我我").is_empty());
    assert_eq!(detector().detect("我我我我我我我我").score, 0.0);
}

#[test]
fn latin_text_is_not_this_packs() {
    assert!(detector().features("hello world").is_empty());
    assert_eq!(detector().detect("hello world").score, 0.0);
}

#[test]
fn segmentation_gate_is_in_rank_units() {
    // The fixture's rank offset is 2.0, so its segmentation gate is
    // -1.45 - (2.0 - log10 26) = -2.03 per character. This run of words has a mean n-gram
    // log10 P of about -1.67: above that gate, below the 26-letter value (-1.45) and far
    // below the gate shifted the wrong way (-0.87). Only the right gate segments it.
    let (coverage, ngram, _) = features("我们天气明天");
    assert!(ngram > -2.03 && ngram < -1.45, "{ngram}");
    assert!((coverage - 1.0).abs() < 1e-6, "coverage {coverage}");
}
