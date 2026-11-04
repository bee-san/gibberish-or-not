//! Crafted and corrupted packs are errors, never panics (on 32-bit targets too: CI runs
//! this file under wasmtime, wasm32-wasip1).

use gibberish_or_not::{Detector, Pack};

fn english_blob() -> Vec<u8> {
    Detector::new().packs()[0].to_bytes()
}

/// The model part of a blob, and the calibration text.
fn parts(blob: &[u8]) -> (Vec<u8>, String) {
    let n = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
    (
        blob[8..8 + n].to_vec(),
        String::from_utf8(blob[8 + n..].to_vec()).unwrap(),
    )
}

/// Offset and length of a model section.
fn section(model: &[u8], tag: &[u8; 4]) -> (usize, usize) {
    let n = u16::from_le_bytes([model[10], model[11]]) as usize;
    for i in 0..n {
        let o = 16 + 12 * i;
        if &model[o..o + 4] == tag {
            let off = u32::from_le_bytes(model[o + 4..o + 8].try_into().unwrap()) as usize;
            let len = u32::from_le_bytes(model[o + 8..o + 12].try_into().unwrap()) as usize;
            return (off, len);
        }
    }
    panic!("no section {tag:?}");
}

fn loads_and_detects(model: &[u8], cal: &str) {
    // Either an error, or a pack that detects without panicking
    if let Ok(pack) = Pack::from_parts(model, cal) {
        let d = Detector::empty().with_pack(pack);
        for text in [
            "The quick brown fox jumps over the lazy dog",
            "MEETMEATNOON",
            "xqzvkj",
            "a",
        ] {
            let _ = d.detect(text);
        }
    }
}

#[test]
fn word_bucket_count_that_wraps_on_32_bit() {
    let (mut model, cal) = parts(&english_blob());
    let (off, _) = section(&model, b"WORD");
    let n = u32::from_le_bytes(model[off..off + 4].try_into().unwrap());
    // n + 2^27: on 32-bit, 32 * that wraps to the real slot size
    let crafted = n.wrapping_add(1 << 27);
    model[off..off + 4].copy_from_slice(&crafted.to_le_bytes());
    assert!(Pack::from_parts(&model, &cal).is_err());
}

#[test]
fn section_offset_plus_length_that_overflows() {
    let (mut model, cal) = parts(&english_blob());
    // The first section's length: u32::MAX
    model[16 + 8..16 + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Pack::from_parts(&model, &cal).is_err());
    let (mut model, _) = parts(&english_blob());
    model[16 + 4..16 + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Pack::from_parts(&model, &cal).is_err());
}

#[test]
fn container_length_that_overflows() {
    let mut blob = english_blob();
    blob[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(Pack::from_bytes(&blob).is_err());
}

#[test]
fn alphabet_symbol_out_of_range() {
    let (mut model, cal) = parts(&english_blob());
    let (off, _) = section(&model, b"ALPH");
    let size = u16::from_le_bytes([model[off], model[off + 1]]);
    // The first entry's symbol: the alphabet size (one past the last valid symbol)
    model[off + 4 + 4..off + 4 + 6].copy_from_slice(&size.to_le_bytes());
    assert!(Pack::from_parts(&model, &cal).is_err());
    model[off + 4 + 4..off + 4 + 6].copy_from_slice(&(size - 1).to_le_bytes());
    loads_and_detects(&model, &cal);
}

#[test]
fn bad_calibrations() {
    let (model, cal) = parts(&english_blob());
    // A bucket whose upper bound is below its lower bound
    let swapped = cal.replacen("bucket 4 5 ", "bucket 4 3 ", 1);
    assert_ne!(swapped, cal);
    assert!(Pack::from_parts(&model, &swapped).is_err());
    // A gap between buckets
    let gap = cal.replacen("bucket 4 5 ", "bucket 5 5 ", 1);
    assert!(Pack::from_parts(&model, &gap).is_err());
    // A NaN weight, unordered anchors, a missing field
    let line = cal.lines().find(|l| l.starts_with("bucket 1 ")).unwrap();
    let mut fields: Vec<&str> = line.split(' ').collect();
    let w = fields[3];
    fields[3] = "NaN";
    assert!(Pack::from_parts(&model, &cal.replace(line, &fields.join(" "))).is_err());
    fields[3] = w;
    let n = fields.len();
    fields.swap(n - 1, n - 3);
    assert!(Pack::from_parts(&model, &cal.replace(line, &fields.join(" "))).is_err());
    let short: Vec<&str> = line.split(' ').take(10).collect();
    assert!(Pack::from_parts(&model, &cal.replace(line, &short.join(" "))).is_err());
    assert!(Pack::from_parts(&model, "").is_err());
}

#[test]
fn every_truncation_and_some_bit_flips() {
    let blob = english_blob();
    let (model, cal) = parts(&blob);
    // Truncations of the header, section table and the start of every section
    let mut cuts: Vec<usize> = (0..512).collect();
    for tag in [b"PARM", b"ALPH", b"NGRM", b"WORD"] {
        let (off, len) = section(&model, tag);
        cuts.extend([off, off + 1, off + 7, off + len / 2, off + len - 1]);
    }
    for n in cuts {
        if n <= model.len() {
            loads_and_detects(&model[..n], &cal);
        }
    }
    // Bit flips in the header and section table, deterministic positions
    let mut x = 0x9E37_79B9u32;
    for _ in 0..400 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let pos = (x as usize) % 200;
        let bit = (x >> 24) % 8;
        let mut m = model.clone();
        m[pos] ^= 1 << bit;
        loads_and_detects(&m, &cal);
    }
}
