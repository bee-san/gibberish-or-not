# Changelog

## 6.0.0 (unreleased)

A rewrite: a calibrated, multi-language "is this natural language?" detector, `no_std` and
WASM-ready. Upgrading: MIGRATION.md.

### Breaking

- Removed the BERT path (`GibberishDetector`, `with_model`, `has_enhanced_detection`) and
  the model download API (`download_model*`, `default_model_path`, `model_exists`,
  `check_token_status`, `TokenStatus`, `ModelError`): it never loaded a model
  (bee-san/Ciphey#1031), and it brought in candle, tokenizers (with onig's C library),
  safetensors and reqwest (with openssl).
- Removed `is_string_bad_quality` and `calculate_non_printable_ratio`.
- The binaries `dictionary`, `is_gibberish`, `passwords`, `enhanced_detection`,
  `test_sensitivity` and `download_model` are gone; `gibberish` (feature `cli`) replaces
  them.
- `is_gibberish` decides differently (better: docs/benchmark.md). Space-less text, short
  phrases and text with identifiers in it are language; identifiers on their own are not.

### Added

- `detect`, `Detector` (languages, sensitivity, `min_score`, custom packs, `word_language`),
  `Detection { is_language, language, score }`, `Lang`, `Pack`, `PackError`, `is_word`.
- A calibrated score: 0.5, 0.75 and 0.9 are the High, Medium and Low thresholds, at
  documented junk false-positive rates in every length bucket.
- Language packs, one Cargo feature each (`en` ... `ko`, all on by default): English in
  6.0.0; the other 12 languages land in 6.x under docs/languages.md.
- `no_std` + `alloc` core, `wasm32-unknown-unknown` with default features, JavaScript
  bindings (feature `wasm`: `detect`, `score`, `isGibberish`, `isPassword`, `languages`).
- The benchmark: corpus v1 (bench/corpus/v1, 53,589 positives in 13 languages, 231,251
  negatives, split by source), `gib-bench` (one command for every metric, the classifier
  fit, the CI gate, latency and sizes), frozen baseline verdicts of 5.0.7 and Ciphey's
  English checker.
- CI: fmt, clippy, tests, wasm32 build and wasm-bindgen tests under node, tests under
  wasmtime, the accuracy gate and the size budget.
- docs/design.md, docs/benchmark.md, docs/languages.md, MIGRATION.md, NOTICE.

### Fixed

- `Sensitivity`'s docs had Low and High the wrong way round: Low is the strictest
  (from #9, by @bee-san).
- `.cargo/config.toml` forced clang and mold, which CI runners don't have: every CI run
  failed. Removed.
- The README's merge-conflict markers and unmeasured claims.
