# English (en)

The English pack of gibberish-or-not 6.0.0, in the framework PR.

| file | what | size |
|---|---|---:|
| `model.bin` | dense 4-gram letter model over a–z (475 KB) and 115,000 words in 15 frequency classes (500 KB) | 975,632 bytes |
| `calibration.txt` | n-gram gate, and per length bucket the classifier's weights and score anchors | 3.4 KB |
| `build.py` | builds `model.bin` from the data cache, deterministically | |
| `build.json` | the build's inputs (with sha256), counts, model size and sha256 | |

Budget: 1 MB for `model.bin` + `calibration.txt` (docs/languages.md); used 956 KB.

## Sources (train split only)

| source | licence | used for |
|---|---|---|
| 85 Project Gutenberg books (the books Ciphey counts its quadgram table from) and 296 more English books picked from the Gutenberg catalog with a fixed seed (`gutenberg_books.en_train`, `en_train_extra` in `bench/sources/sources.json`); none is a dev or test book, and the few copyrighted Gutenberg ebooks are left out | public domain in the USA | letter n-gram counts, word counts |
| [Tatoeba](https://tatoeba.org) English sentences of the contributors in the train split (export of 2026-10-31), a fixed-seed sample of 300,000, after dropping the 691 whose letters are those of a dev or test text of the corpus (another contributor's copy) | CC BY 2.0 FR | letter n-gram counts, word counts |
| Grady Ward's Moby crossword list (`crosswd.txt`, Gutenberg ebook 3201) | public domain | 43,207 words the books don't have, in the rarest frequency class |

The exact files and their sha256 are in `build.json` (`inputs`). 124 million letters and
28 million word tokens were counted. Words: every word seen 3+ times (in 2+ sources, or
10+ times in one), lower case with apostrophes kept (`don't`), single letters only `a` and
`i`, no runs of one letter (`www`, `zzzz`), Roman numerals only when mostly written in
lower case: 71,793 words from the books and Tatoeba, then Moby words up to 115,000.

## Rebuild

```sh
python3 bench/scripts/fetch_sources.py          # once: the data cache
python3 languages/en/build.py                   # model.bin, build.json (about 3 min)
cargo run --release -p gib-bench -- fit --lang en
cargo build --release -p gib-bench              # embed the new pack
cargo run --release -p gib-bench -- --corpus v1 --split dev
```

## Model choices (dev split)

- Letters only (spaces and punctuation removed) for the n-grams, so space-less text scores
  like spaced text; a–z only: any other letter is foreign, so mojibake (`Ã©`) counts
  against the text.
- Words from counts plus a spelling list, 115,000 in all. The first build (85 books and
  300,000 Tatoeba sentences) kept 42,000 words; adding about 300 books and the Moby list
  raised dev recall at Medium from 96.3% to 97.2% (High 96.4% → 97.5%) at the same
  false-positive targets.
- `max_word_chars` 16 (each character more is one more word lookup per letter when a token
  is segmented), `seg_min_letters` 6, `lambda` -2.0.
- Calibration targets (bench/src/fit.rs) are below both baselines' dev false-positive
  rates in every length bucket (docs/benchmark.md, "Calibration targets").

## Numbers

Corpus v1, English positives against all junk; full tables in docs/benchmark.md.

| split | Low recall / FPR | Medium | High |
|---|---|---|---|
| dev | 96.3% / 0.076% | 98.8% / 0.23% | 99.0% / 0.58% |
| test | 97.0% / 0.13% | 98.6% / 0.33% | 98.9% / 0.81% |
