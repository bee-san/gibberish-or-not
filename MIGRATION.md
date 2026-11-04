# Migrating from 5.x to 6.0

6.0 is a rewrite of how the decision is made; the three functions most code calls keep
their signatures and meaning.

## What stays

| 5.x | 6.0 |
|---|---|
| `is_gibberish(&str, Sensitivity) -> bool` | same signature; better decisions (docs/benchmark.md) |
| `Sensitivity::{Low, Medium, High}` | same meaning: **Low is the strictest, High the most lenient** (the 5.x doc comments had it backwards; [PR #9](https://github.com/bee-san/gibberish-or-not/pull/9)). Now also `Eq`, `Hash`, `Default` (Medium) and `min_score()` |
| `is_password(&str) -> bool` | same, same list |

## What's new

- `detect(text) -> Detection { is_language, language, score }` and the `Detector` builder:
  `Detector::new().languages(&[Lang::En, Lang::Fr]).sensitivity(Sensitivity::Low)`,
  `.min_score(f32)`, `.with_pack(Pack)`, `.detect(text)`, `.is_gibberish(text)`,
  `.word_language(word)`.
- `score`: calibrated so that 0.5, 0.75 and 0.9 are the High, Medium and Low thresholds,
  at the junk false-positive rates of docs/benchmark.md, in every length bucket.
- `Lang`: ISO 639-1 codes (`Lang::En` ... `Lang::Ko`, `Lang::Other([u8; 2])` for packs of
  other languages loaded at run time).
- `is_word(word)`: whether a word is in a compiled-in language's word list (for callers
  with word rules of their own).
- `Pack::from_bytes` / `from_parts` / `to_bytes`: custom or extra language packs.
- One Cargo feature per language (`en`, `es`, ... `ko`), all on by default; `std` (only
  `impl std::error::Error`), `cli` (the `gibberish` binary), `wasm` (JavaScript bindings).
- The library is `no_std` + `alloc` and builds for `wasm32-unknown-unknown` with default
  features.

## What's gone

| 5.x | instead |
|---|---|
| `GibberishDetector`, `with_model`, `has_enhanced_detection` | `Detector`. The BERT path never loaded a model (bee-san/Ciphey#1031) |
| `download_model`, `download_model_with_progress_bar`, `default_model_path`, `model_exists`, `check_token_status`, `TokenStatus`, `ModelError`, the HuggingFace token | nothing to download: the models are embedded |
| `is_string_bad_quality`, `calculate_non_printable_ratio` | `detect` (control characters veto) |
| binaries `dictionary`, `is_gibberish`, `passwords`, `enhanced_detection`, `test_sensitivity`, `download_model` | `gibberish` with `--features cli` |
| dependencies candle, tokenizers (onig, C), safetensors, reqwest (openssl), serde, dirs, chrono, clap, env_logger, lazy_static, encoding_rs, unicode-segmentation | `phf` and `once_cell` (no_std) only |

## Behaviour changes

- Space-less text (`MEETMEATNOON`), short phrases (`call me`), upper case and text with
  identifiers in it (`The password is IFukwKGsFW8...`) are language now.
- Identifiers on their own are not language: a URL, an email address, a hash or a CTF flag
  alone is gibberish to gibberish-or-not (ask pyWhat what it is).
- Wrong cipher keys, near-English Vigenère key-search output, reversed text, encodings,
  hashes, random strings and mojibake are rejected far more often (on the benchmark's test
  split, 5.0.7 accepted 19.1% of the junk at High, 6.0 accepts 0.88%).
- Text in a language whose pack isn't compiled in is gibberish (5.x accepted some French,
  German and Spanish by accident).
- Text longer than 4,096 bytes is judged by its first 4,096.

## Ciphey's exact changes

Measured with Ciphey's own harness (`cargo bench --bench plaintext -- --e2e=3`, master
`9350174d`, held-out split; the full tables are in the 6.0.0 PR):

**1. The dependency** (`Cargo.toml`):

```diff
-gibberish-or-not = "5.0.7"
+gibberish-or-not = "6.0.0"
```

That removes the candle/tokenizers/reqwest/openssl crates (96 of 258 in Ciphey's graph) and
the `half`/`rand` pins "Required for candle-core compatibility". Every `use` of
`is_gibberish`, `is_password` and `Sensitivity` compiles unchanged.

**2. Word lookups** (`src/checkers/english.rs`): `is_dictionary_word` asked `is_gibberish`
about a single word, which in 6.0 is a calibrated judgement of the whole text, not a
dictionary lookup. Ask the word list:

```diff
-use gibberish_or_not::{is_gibberish, Sensitivity};
+use gibberish_or_not::{is_gibberish, is_word, Sensitivity};
 ...
 fn is_dictionary_word(word: &str) -> bool {
-    (4..10).contains(&word.len()) && !is_gibberish(word, Sensitivity::Medium)
+    (4..10).contains(&word.len()) && is_word(word)
 }
```

With 1 and 2 (and nothing else), Athena@Medium's held-out F1 goes from 0.939 to 0.970
(false positives 139 → 49, recall 96.8% → 97.4%), Athena@Low from 0.946 to 0.971 (96 → 37),
and the end-to-end corpora from 268/296 to 271/296 correct (wrong answers 27 → 24). Ciphey's
dependency graph shrinks from 356 to 206 lines of `cargo tree -e normal --prefix none |
sort -u` (294 → 180 crate versions).

**3. Optional: let gibberish-or-not decide** (`is_english` keeps only Ciphey's own vetoes:
control characters and NATO code words, and returns `!is_gibberish(text, sensitivity)`).
That gives the best checker numbers (Athena@Low held-out recall 98.2%, F1 0.979; Caesar
`crack()` correct 91% → 94%, ROT47 93% → 98%, Atbash 94% → 97%), but end to end it is worse
(259/296, wrong answers 36): a search asks
about tens of thousands of short candidates, and 6.0 takes common short words (`ALL`,
`A TO`, T9's word salads) as language, which they are, but in a search they are almost
always junk. Ciphey's short-text rules (single words of 4+ letters, 5+ at Low; two words
that are both words; space-less text that splits into words) encode that search-specific
prior, so keep them: change 2 is the recommended migration, and rules can be retired one
at a time with the harness showing each.

**4. Other languages**: 6.0.0 ships English; the language packs land in 6.x
(docs/languages.md). Ciphey's `is_other_language` word list can go once they do, measured
on the harness's `non_english` category.
