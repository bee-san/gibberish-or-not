# Benchmark

Every accuracy, speed and size claim about gibberish-or-not comes from here, and can be
reproduced with one command.

```sh
. /local/home/skerraut/herd-synth3/shared/ciphey/env.sh   # or any Rust >= 1.85
cargo run --release -p gib-bench -- --corpus v1                  # dev split (the default)
cargo run --release -p gib-bench -- --corpus v1 --split test     # test: once per released model
cargo run --release -p gib-bench -- --corpus v1 --wasm           # + WASM size and latency
cargo run --release -p gib-bench -- --corpus v1 --langs en --examples 5   # misclassified samples
cargo run --release -p gib-bench -- gate                         # the CI accuracy gate
cargo run --release -p gib-bench -- sizes --wasm --check         # the CI size budget
cargo run --release -p gib-bench -- latency                      # latency only
echo 'Gsv jfrxp yildm ulc' | cargo run --release -p gib-bench -- check   # one text, with features
```

`--wasm` needs `wasm-bindgen` 0.2.130, `wasm-opt` (binaryen) and `wasmtime` on the PATH and
the `wasm32-unknown-unknown` and `wasm32-wasip1` targets.

## The corpus

[bench/corpus/v1](../bench/corpus/v1/README.md), frozen: 53,589 positives (natural-language
text in 13 languages: sentences, paragraphs, phrases, short phrases, words, upper case,
space-less, sentences with identifiers) and 231,251 negatives (classical-cipher output of
real text with wrong keys, optimised Vigenère key-search output, encodings, hashes, random
strings in many alphabets, mojibake, and real candidates from Ciphey's searches), split
into train, dev and test **by source** (book, Tatoeba contributor, Ciphey search).
Identifiers alone, code, JSON and passwords are reported but never counted ("info").

- **train** fits the classifier (`gib-bench fit`), **dev** sets the thresholds and is used
  for every tuning run and the CI gate, **test** is for reporting: run once per released
  model version.
- The baselines' verdicts on every sample are frozen in `bench/baselines/v1/`:
  gibberish-or-not 5.0.7's `!is_gibberish`, and Ciphey's English checker (5.0.7 plus
  everything Ciphey layers on it: quadgram scoring of space-less text, short-phrase word
  rules, decoder-output vetoes, masking of flags and secrets), from Ciphey master
  `9350174d` (bench/baselines/README.md).

## Metrics

- **Recall**: the share of the counted languages' positives accepted. **FPR**: the share
  of all negatives accepted. **Precision** and **F1** over the same counts (negatives far
  outnumber positives, as in a search).
- Per sensitivity: accepted means score ≥ 0.9 (Low), 0.75 (Medium) or 0.5 (High).
- **Length buckets** in characters: 1–9, 10–19, 20–49, 50–199, 200+.
- Per category of positive and negative; per language: recall, and which language the
  detector reports (the confusion matrix, at High).
- **Latency**: each sample of up to 256 characters timed on its own (checks under 2 µs
  repeated and averaged), quantiles over samples; with the host's load average, because
  the host is shared.
- **Sizes**: each pack's `model.bin` and `calibration.txt`; the WASM module with every
  default language before and after `wasm-opt -Oz`.

## Calibration targets

`gib-bench fit` sets each length bucket's thresholds at these junk false-positive rates on
the dev split (bench/src/fit.rs). They are below both baselines' dev rates in every bucket.

| characters | High | Medium | Low |
|---|---:|---:|---:|
| 1–9 | 0.10% | 0.08% | 0.03% |
| 10–19 | 0.40% | 0.20% | 0.10% |
| 20–49 | 1.2% | 0.50% | 0.30% |
| 50–199 | 1.5% | 0.50% | 0.20% |
| 200+ | 1.5% | 0.50% | 0% (above the highest dev negative) |

- Where the target allows 25 or more dev false positives in a bucket, the threshold is the
  empirical quantile. Where it allows fewer, one or two samples would set it, which doesn't
  generalise: a split-half check on dev (thresholds from one half of the negatives,
  counted on the other) gave up to 5 times the target. There the top 25 negative logits
  are modelled as an exponential tail (peaks over threshold), and the threshold is where
  the tail reaches the target.
- The four buckets under 10 characters (1–3, 4–5, 6–7, 8–9) share the 1–9 budget, spent
  where it buys the most recall: real short words and junk that happens to be a word
  overlap most in the shortest buckets, so an equal rate per bucket would reject them
  almost all there.
- From 10 characters, Low's thresholds must also meet the target on the junk derived from
  the pack's own language alone, the hardest junk, whose share differs between splits.
- `gib-bench fit` prints the split-half FPR of every bucket next to its target.

The CI gate allows each bucket's dev FPR up to 1.5 × its target + 0.02 points, for all
languages together.

## Current numbers: gibberish-or-not 6.0.0 (English)

Test split, English positives (5,026) against all negatives (90,345). Host: shared 64-core
Linux, Rust 1.99.1, release builds.

| | sensitivity | precision | recall | F1 | false positives | FPR |
|---|---|---:|---:|---:|---:|---:|
| **gibberish-or-not 6.0** | Low | 97.5% | **97.2%** | **0.973** | 126 | **0.14%** |
| | Medium | 93.9% | **98.7%** | **0.962** | 324 | **0.36%** |
| | High | 86.3% | **98.9%** | **0.922** | 789 | **0.87%** |
| 5.0.7 | Low | 86.3% | 70.9% | 0.778 | 566 | 0.63% |
| | Medium | 60.5% | 84.1% | 0.703 | 2,766 | 3.06% |
| | High | 21.1% | 92.1% | 0.344 | 17,287 | 19.1% |
| 5.0.7 + Ciphey's English checker | Low | 95.1% | 95.3% | 0.952 | 248 | 0.27% |
| | Medium | 91.4% | 97.0% | 0.942 | 457 | 0.51% |
| | High | 81.5% | 97.3% | 0.887 | 1,112 | 1.23% |

### By length (test; recall / FPR)

| characters | positives / negatives | sensitivity | gibberish-or-not 6 | 5.0.7 | 5.0.7 + Ciphey English |
|---|---:|---|---|---|---|
| 1–9 | 643 / 11,904 | Low | 81.3% / 0.025% | 41.5% / 0.11% | 78.8% / 0.034% |
| | | Medium | 91.4% / 0.059% | 41.5% / 0.13% | 87.9% / 0.11% |
| | | High | 92.5% / 0.10% | 41.5% / 0.52% | 88.0% / 0.11% |
| 10–19 | 844 / 15,696 | Low | 98.2% / 0.089% | 71.4% / 0.72% | 96.6% / 0.096% |
| | | Medium | 98.7% / 0.12% | 88.4% / 2.47% | 97.3% / 0.18% |
| | | High | 99.2% / 0.22% | 98.9% / 19.2% | 97.4% / 0.32% |
| 20–49 | 1,779 / 30,656 | Low | 99.9% / 0.20% | 73.1% / 0.99% | 97.8% / 0.46% |
| | | Medium | 99.9% / 0.45% | 90.9% / 4.01% | 98.5% / 0.66% |
| | | High | 100.0% / 0.95% | 99.6% / 25.0% | 98.9% / 1.37% |
| 50–199 | 1,703 / 28,428 | Low | 99.8% / 0.17% | 78.5% / 0.47% | 98.2% / 0.31% |
| | | Medium | 100.0% / 0.55% | 90.6% / 3.25% | 98.7% / 0.65% |
| | | High | 100.0% / 1.51% | 99.8% / 22.2% | 99.0% / 1.99% |
| 200+ | 57 / 3,661 | Low | 96.5% / 0.00% | 96.5% / 0.00% | 96.5% / 0.00% |
| | | Medium | 100.0% / 0.11% | 96.5% / 5.74% | 100.0% / 0.74% |
| | | High | 100.0% / 0.66% | 100.0% / 7.05% | 100.0% / 1.72% |

In every bucket and at every sensitivity, 6.0 has a higher recall at a lower FPR than both
baselines, below 10 characters too, except where recall ties at the top: 200+ at Medium and
High against Ciphey's checker (100%, with a lower FPR), and 200+ at Low, where all three
accept 55 of the 57 texts and no junk (the two misses are one Wikipedia sentence about a
Polish village with its IPA transcription, as written and in upper case).

### Positives (test; recall at Low / Medium / High)

| category | n | gibberish-or-not 6 | 5.0.7 | 5.0.7 + Ciphey English |
|---|---:|---|---|---|
| sentence | 1,297 | 99.8% / 100.0% / 100.0% | 97.1% / 98.3% / 99.8% | 98.5% / 99.2% / 99.5% |
| caps (upper case) | 1,207 | 97.8% / 99.0% / 99.2% | 88.6% / 90.1% / 91.2% | 95.7% / 97.5% / 97.8% |
| nospace | 1,083 | 98.6% / 99.4% / 99.8% | 4.9% / 61.6% / 93.9% | 96.0% / 96.6% / 96.9% |
| phrase (1–4 words) | 436 | 97.9% / 98.6% / 98.9% | 75.9% / 77.8% / 78.2% | 95.4% / 96.1% / 96.3% |
| literary (Ciphey's held-out books) | 339 | 100% / 100% / 100% | 99.7% / 100% / 100% | 99.7% / 100% / 100% |
| short (1–3 words, ≤ 12 letters) | 250 | 88.0% / 95.2% / 95.6% | 60.0% / 61.2% / 61.2% | 85.6% / 91.2% / 91.2% |
| word | 197 | 70.6% / 84.8% / 86.3% | 84.8% / 85.3% / 85.8% | 68.5% / 84.8% / 84.8% |
| mixed (with an identifier) | 120 | 99.2% / 100% / 100% | 82.5% / 85.0% / 99.2% | 96.7% / 97.5% / 98.3% |
| paragraph | 44 | 100% / 100% / 100% | 100% / 100% / 100% | 100% / 100% / 100% |
| chat, technical, formal, ctf_text, verse (Ciphey's) | 53 | 100% at every level | 98.1% / 100% / 100% | 98.1% / 100% / 100% |

Single words are the weak spot: a lone word is junk often enough (one string of four
letters in 150 is a dictionary word) that the short buckets' thresholds reject the rarer
ones, at Low most of all. 5.0.7, which accepts any dictionary word, keeps more of them at
Low, with 13 false positives in the 1–9 bucket against 6.0's 3.

### Negatives (test; FPR at Low / Medium / High), the categories that aren't 0 everywhere

| category | n | gibberish-or-not 6 | 5.0.7 | 5.0.7 + Ciphey English |
|---|---:|---|---|---|
| vigenere_partial (key with one letter off) | 3,238 | 0.90% / 2.2% / 4.2% | 11.6% / 39.0% / 75.3% | 6.8% / 12.2% / 28.8% |
| vigenere_opt_quadgram | 145 | 2.8% / 4.1% / 6.9% | 0% / 4.8% / 50.3% | 1.4% / 3.4% / 4.8% |
| vigenere_opt_bigram | 149 | 0.67% / 2.7% / 4.7% | 0% / 8.1% / 52.3% | 0% / 0.67% / 3.4% |
| railfence (wrong key) | 9,942 | 0.42% / 1.6% / 4.7% | 0.32% / 3.2% / 61.5% | 0.15% / 0.24% / 0.67% |
| reversed | 8,066 | 0.12% / 0.25% / 0.77% | 1.5% / 10.3% / 46.6% | 0.087% / 0.29% / 0.84% |
| columnar (wrong key) | 5,932 | 0.067% / 0.19% / 0.37% | 0.32% / 2.9% / 49.8% | 0.051% / 0.084% / 0.30% |
| enc_hex_spaced | 595 | 0.34% / 0.84% / 1.7% | 0% / 0% / 0% | 0% / 0% / 0% |
| unicode_shift (CJK, Hangul) | 1,991 | 0.70% / 0.75% / 0.85% | 0% / 0% / 0.45% | 0% / 0% / 0.15% |
| mojibake | 2,636 | 0.38% / 0.49% / 0.49% | 0.04% / 0.08% / 1.1% | 0% / 0% / 0.15% |
| rot47 | 5,664 | 0% / 0.12% / 0.58% | 0% / 0% / 0.28% | 0% / 0% / 0% |
| search_random (Ciphey's search candidates) | 2,910 | 0.069% / 0.069% / 0.10% | 0.034% / 0.41% / 10.4% | 0% / 0% / 0% |
| substitution | 6,062 | 0.049% / 0.049% / 0.049% | 0.066% / 0.54% / 4.1% | 0% / 0.016% / 0.033% |
| vigenere (wrong key) | 4,651 | 0.043% / 0.043% / 0.043% | 0% / 0.34% / 3.4% | 0% / 0% / 0.022% |
| atbash | 6,114 | 0.033% / 0.049% / 0.049% | 0.033% / 0.18% / 0.80% | 0% / 0.016% / 0.016% |
| caesar | 15,934 | 0.006% / 0.006% / 0.013% | 0.031% / 0.28% / 2.6% | 0% / 0.013% / 0.013% |
| affine, enc_url, rand_upper, rand_words, uuid, base64/32 and the other noise | | 0% everywhere | up to 50% at High | ≤ 0.04% |

The junk 6.0 accepts more often than Ciphey's checker is mostly of two kinds: wrong
transposition keys that leave long runs of the plaintext readable (railfence), and
mojibake or CJK code-point shifts that English-only 6.0.0 reads as English letters among
foreign ones (`enc_hex_spaced` of Cyrillic UTF-8, `屒Ishmael。`). Ciphey's English checker
rejects any non-ASCII letter among ASCII ones; the language packs of 6.x take that role.

The full per-category tables, and per-language tables of all 13 languages' slices (where
English-only 6.0.0 rejects other languages' text, as it should), are in the report that
`gib-bench --corpus v1 --split test` prints.

### Speed (dev split, 87,395 samples of up to 256 characters)

| | p50 | p99 | mean |
|---|---:|---:|---:|
| **gibberish-or-not 6, native** (any sensitivity: one score) | **675 ns** | **13.8 µs** | 1.6 µs |
| junk only / language only | 575 ns / 1.2 µs | 8.6 µs / 20.4 µs | 1.2 µs / 3.0 µs |
| **wasm32-unknown-unknown under node 24** (wasm-bindgen `score()`, string passing included) | 979 ns | 22.7 µs | 1.9 µs |
| **wasm32-wasip1 under wasmtime 50** | 1.1 µs | 18.9 µs | 2.4 µs |
| 5.0.7 `is_gibberish` @Medium (measured when the baselines were frozen) | 5.1 µs | 35.3 µs | 7.0 µs |
| Ciphey's English checker @Medium (same) | 735 ns | 17.0 µs | 1.5 µs |

The host is shared: its load average was 53 during this run (about 25 when the baselines
were frozen), so absolute times can be several times lower on an idle machine. WASM is within
1.7× of native at p50 and p99.

The targets for all 13 languages are p50 ≤ 3 µs and p99 ≤ 30 µs (docs/languages.md, gate
5). Scoring is linear in the number of candidate packs for the n-gram pass only: the word
model runs for the 3 best-ranked packs of the text's script.

### Sizes

| | size | budget |
|---|---:|---:|
| `languages/en/model.bin` + `calibration.txt` | 956 KB | 1 MB |
| WASM module with every default language, `cargo build --release` | 1,277 KB | |
| after `wasm-opt -Oz` | 1,232 KB | 5.5 MB for all 13 |

Dependencies: `phf` (the password list) and `once_cell` (`race`, no_std): 12 crates in
`cargo tree -e normal` (the proc-macro's build-time crates included), against 195 for 5.0.7.

## Other detectors (from bee-san/Ciphey#1031, not re-measured)

On Ciphey's 2,761-sample corpus of the time, thresholds set to the same false-positive count:

| English score | recall | no-space recall | time / check | size |
|---|---:|---:|---:|---:|
| gibberish-or-not 5.0.7 (+ Ciphey's punctuation fix) | 83.4% | 50.0% | 7.8 µs | in binary |
| quadgram log-probability | 96.2% | 100% | 0.14 µs | 1.8 MB |
| DistilBERT (5.x's model download, via ONNX) | 79.0% | 4.5% | 5.5 ms | 268 MB + 96 crates |
| Laya zero-shot (ONNX) | 36.3% | 4.5% | 140 ms | 1.7 GB |
| lingua 1.8 (EN/FR/DE/ES) | 39.5% | 22.7% | 87 µs | ~2.6 MB/language |
| whatlang 0.18 | 54.8% | 13.6% | 50 µs | ~83 KB |

5.x's BERT path itself can't be measured: it never loads the model it downloads (`missing
field hidden_size`, #1031), so 5.x with "enhanced detection" behaves exactly like 5.x
without it.

## In Ciphey

Ciphey's own harness (`cargo bench --bench plaintext -- --e2e=3`, master `9350174d`), before
and after switching to 6.0 the way MIGRATION.md recommends (the dependency, and
`is_dictionary_word` asking `is_word`):

| | before (5.0.7) | after (6.0) |
|---|---|---|
| Athena@Low, held-out: F1, false positives | 0.946, 96 | **0.971, 37** |
| Athena@Medium, held-out | 0.939, 139 | **0.970, 49** |
| Athena@High, held-out | 0.902, 263 | **0.956, 94** |
| English checker@Medium, held-out | 0.901, 133 | **0.933, 43** |
| Athena@Medium, train | 0.941, 107 | **0.968, 43** |
| decoders, held-out (right / wrong / nothing): Caesar | 91% / 0% / 8% | 92% / 0% / 7% |
| ROT47 | 93% / 0% / 7% | 95% / 0% / 5% |
| Atbash | 94% / 0% / 6% | 95% / 0% / 5% |
| end to end, 296 searches: correct / wrong / missed | 268 / 27 / 1 | **271 / 24 / 1** |
| Athena@Medium per check, 3 alternating runs each (median / p99) | 4.4–4.7 µs / 21.9–23.4 µs | 4.5–5.0 µs / 20.7–27.1 µs |
| English checker@Medium per check, same runs | 785–790 ns / 13.0–13.3 µs | 785–790 ns / 10.7–11.9 µs |
| `cargo tree -e normal --prefix none \| sort -u \| wc -l` | 356 | **206** |

## How often the test split was run

The test split is for reporting; model version 6.0.0 was run on it three times, and each
change between runs was made and validated on dev only:

1. First run: Low 97.5% / 0.19%, Medium 97.7% / 0.33%, High 98.2% / 0.86% (recall / FPR).
   Errors showed English Wikipedia sentences that quote names in other scripts rejected.
   Change: words in other scripts are neutral for a pack, as identifiers are; and Low's
   thresholds also meet the target on the pack's own-language junk.
2. Second run: Low 97.8% / 0.18%, Medium 98.4% / 0.36%, High 98.5% / 0.88%. Below 10
   characters the FPR was above Ciphey's checker's (13 against 4 false positives at Low,
   while dev had 0 against 8). Diagnosis on dev: the thresholds of the short buckets were
   order statistics of one or two samples; a split-half check gave 5× the target. Change:
   the tail model and the shared short-text budget above, with the split-half check now
   printed by `gib-bench fit`.
3. Third run: the numbers above.

Looking at the test split three times makes it less of a clean held-out set than one look;
corpus v2 will have a fresh test split. Every later model version gets one test run,
reported in its PR.
