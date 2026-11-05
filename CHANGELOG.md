# Changelog

## Unreleased

- Chinese, Japanese and Korean text: a Han, kana or Hangul character counts 3 times when
  the text's scripts are chosen (Japanese with Latin terms gets Japanese candidates);
  Han and kana tokens in spaced text get the unspaced rules; flags, hex, URLs and emails
  glued to CJK text are identifiers; the dominant-letter veto only applies where cased
  letters are most of the text (`本当に面白い映画でしたwwwwwwwwww` is no longer vetoed).
  English's output is unchanged on every dev and test text.

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
- The binaries `dictionary`, `is_gibberish`, `passwords`, `enhanced_detection` and
  `test_sensitivity` are gone; `gibberish` (feature `cli`) replaces them. The hidden
  `download_with_progress_bar` re-export is gone.
- `is_gibberish` decides differently (better: docs/benchmark.md). Space-less text, short
  phrases and text with identifiers in it are language; identifiers on their own,
  repeated letters or short words, and near-repetitions such as base64 of zero buffers,
  are not. Many Latin-script sentences of languages
  whose packs haven't shipped are accepted as English (MIGRATION.md).

### Added

- `detect`, `Detector` (languages, sensitivity, `min_score`, custom packs, `word_language`),
  `Detection { is_language, language, score }`, `Lang`, `Pack`, `PackError`, `is_word`.
- A calibrated score: 0.5, 0.75 and 0.9 are the High, Medium and Low thresholds, at
  documented junk false-positive rates in every length bucket.
- Language packs, one Cargo feature each (`en` ... `ko`, all on by default): English in
  6.0.0; the other 12 languages land in 6.x under docs/languages.md. With many packs, a
  cheap pre-rank and per-pack gates choose the few that are scored in full, in rank order
  (docs/design.md, "Many languages").
- `no_std` + `alloc` core, `wasm32-unknown-unknown` with default features, JavaScript
  bindings (feature `wasm`: `detect`, `score`, `isGibberish`, `isPassword`, `languages`).
- The benchmark: corpus v1 (bench/corpus/v1, 53,531 positives in 13 languages, 232,349
  negatives, split by source), `gib-bench` (one command for every metric, the classifier
  fit, the CI gate, latency and sizes), frozen baseline verdicts of 5.0.7 and Ciphey's
  English checker.
- `VERSION`; the JavaScript `detect` and `isGibberish` throw on an unknown sensitivity
  name.
- CI: fmt, clippy, tests, wasm32 build and wasm-bindgen tests under node, tests under
  wasmtime (32-bit), the pack writer's self-test, the accuracy gate and the size budget.
- docs/design.md, docs/benchmark.md, docs/languages.md, MIGRATION.md, NOTICE.

### Fixed

- `Sensitivity`'s docs had Low and High the wrong way round: Low is the strictest
  (from #9, by @bee-san).
- `.cargo/config.toml` forced clang and mold, which CI runners don't have: every CI run
  failed. Removed.
- The README's merge-conflict markers and unmeasured claims.
