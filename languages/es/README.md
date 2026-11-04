# Spanish (es)

The Spanish pack of gibberish-or-not 6.x, built under `docs/languages.md`.

| file | what | size |
|---|---|---:|
| `model.bin` | dense 3-gram letter model over 33 letters (a–z ñ á é í ó ú ü; 37 KB) and 82,674 words in 15 frequency classes (360 KB) | 396,944 bytes |
| `calibration.txt` | n-gram gate and pre-gate, and per length bucket the classifier's weights, score anchors and n-gram gate (`gib-bench fit --lang es`, budget mode, `--fpr-scale` 1) | 3,729 bytes |
| `build.py` | builds `model.bin` from the data cache, deterministically | |
| `build.json` | the build's inputs (739 files with sha256), options, counts, model size and sha256 | |

Budget: 400 KB (409,600 bytes) for `model.bin` + `calibration.txt`; used 400,673 bytes.

## Sources (train split only)

| source | version, sha256 | licence | used |
|---|---|---|---|
| [Tatoeba](https://tatoeba.org) Spanish sentences (`tatoeba-20261031/spa_sentences_detailed.tsv.bz2`), the contributors `packlib.tatoeba_split` puts in train | export of 2026-10-31, `8ebe95f5…e67e` | CC BY 2.0 FR, by the Tatoeba contributors; each sentence's author is on `https://tatoeba.org/en/sentences/show/<id>` | 343,924 sentences: letter and word counts (words weighted twice, see below) |
| The corpus's 13 Spanish Project Gutenberg train books (`gutenberg_books.es`, the first 13 in id order, as `book_split` in `bench/scripts/make_corpus.py` has it), except pg40827 | sha256 per file in `bench/sources/sources.json` and `build.json` | public domain in the USA | 12 books. pg40827 (*La voz de la conseja*, an anthology) is left out: it contains the text of dev book pg53355 (Pérez Galdós, *La novela en el tranvía*) |
| 723 more Spanish Project Gutenberg books (`gutenberg-es/` in the data cache), picked by fixed rules (`extra_books` in `build.py`): every `Text` of `pg_catalog-20261101.csv` whose language is `es` alone, that isn't in the corpus's manifest, a dictionary, grammar, phrase book or reader, or by an author of a corpus dev or test book | fetched 2026-11-04 from `https://www.gutenberg.org/cache/epub/<n>/pg<n>.txt`; sha256 per file in `build.json` | public domain in the USA | 714 books. Skipped: the 7 whose header says "COPYRIGHTED", 2 that are under 25% common Spanish words, and 1 without a plain-text file (`build.json`, `stats.extra_books_skipped`) |
| [Global Voices](https://es.globalvoices.org) en Español, as monolingual text from [OPUS](https://opus.nlpl.eu) GlobalVoices v2018q4 (J. Tiedemann, 2012, *Parallel Data, Tools and Interfaces in OPUS*, LREC 2012; corpus compiled by CASMACAT) | `https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/mono/es.txt.gz`, Last-Modified 2020-11-13, 45,172,682 bytes, `4ebb84ba…3c9e` | CC BY 3.0, by the Global Voices authors and translators (https://globalvoices.org/about/global-voices-attribution-policy/) | 939,558 distinct lines of news and blog posts (2004–2018): letter and word counts |
| Project Gutenberg catalog `gutenberg/pg_catalog-20261101.csv` | `6bf36a85…ab2d` | public domain (catalog data) | which books to fetch (not counted) |

No text of the sources is shipped: `model.bin` holds letter 3-gram probabilities and a
hashed word table (28-bit fingerprints and frequency classes). The NOTICE section for
Spanish (gib-lead adds it at release):

> Spanish (languages/es): Project Gutenberg books (public domain in the USA; the book
> numbers are in languages/es/build.json). Tatoeba (https://tatoeba.org), Spanish sentences,
> export of 2026-10-31: CC BY 2.0 FR, by the Tatoeba contributors. Global Voices
> (https://globalvoices.org) en Español, via OPUS GlobalVoices v2018q4: CC BY 3.0, by the
> Global Voices authors and translators.

Cleaning, before counting:

- **No evaluation text.** Every sentence of every source whose letters are those of a Spanish
  dev or test text of corpus v1 (`packlib.heldout_keys`) is dropped: 482 Tatoeba sentences, 16
  Global Voices lines, 169 book sentences.
- **No English.** Paragraphs (books) and lines (Tatoeba, Global Voices) of 6 or more words
  with more common English words than common Spanish ones are dropped: 111,079 words of
  notes and vocabularies of US school editions, 12,376 words of quoted English headlines.
- Duplicate Tatoeba sentences and Global Voices lines are counted once.

In all, 332 million letters and 70.9 million word tokens were counted (books 49.5 M,
Global Voices 18.9 M, Tatoeba 2.5 M tokens).

## Model

- **Letters**: a–z, ñ, á, é, í, ó, ú, ü, 33 symbols; any other letter is foreign and counts
  against the text (mojibake `Ã±`, other languages' letters). Upper case is found by
  lower-casing (`Ñ` → `ñ`), so no case overrides.
- **N-grams**: dense 3-gram tables (33³ bytes) of interpolated absolute-discounting estimates
  over the letters of each text with everything else removed, so space-less text scores like
  spaced text. A dense 4-gram over 33 (or 27 folded) symbols doesn't fit the budget.
- **Words**: lower case, accents kept; every word seen 3+ times in 2+ pseudo-sources (or 10+
  times in one), single letters only `a e o u y á é ó`, no runs of one letter, no Roman
  numerals written mostly in upper case, no letters outside the alphabet. Frequencies are a
  mixture of the three kinds of source (books 1, Global Voices 1, Tatoeba 2: everyday
  speech, the register of most short texts). The most frequent words up to the size budget,
  then each accented word's accentless spelling (`cancion`, `manana`, `tambien`) half a class
  rarer than the word itself, for text typed or deciphered without accents (classical
  ciphers keep a–z only).
- **Parameters**: `lambda` -2.0, `max_word_chars` 16, `seg_min_letters` 5.

## Model choices (dev split)

Every row: `python3 languages/es/build.py --out X --opt …`, then `gib-bench fit` in budget mode
at `--fpr-scale` 1, measured on corpus v1 dev with en + es against en alone. The comparisons
ran on the framework at c0606c1, whose budget mode rejected every 200+ character text at Low
(fixed in b56b0c5): Low recall in this table is about 4.6 points under the shipped pack's,
the same for every row. "Accentless" is the recall of the 798 dev Spanish positives
that have an accent or ñ, with every accent and ñ removed and `¿¡` dropped: a proxy for text
typed or deciphered without accents (classical ciphers keep a–z only), not part of the
corpus. One dev sample is 0.07 points of recall, so differences under about 0.3 points are
noise. B = `w_tatoeba=2 seg_min_letters=5` (accentless spellings off).

| variant (`--opt`) | model bytes | words | es recall L/M/H | accentless M/H | shared FP rise L/M/H |
|---|---:|---:|---|---|---|
| **shipped**: B + `accentless=2` | 396,944 | 82,674 | **91.7 / 97.6 / 98.0** | **96.6 / 96.6** | +2 / -16 / -75 |
| B | 404,592 | 84,433 | 91.9 / 97.4 / 98.0 | 92.9 / 93.9 | +1 / -17 / -78 |
| B + `accentless=1` (accentless spelling in the word's own class) | 396,944 | 82,674 | 91.4 / 97.3 / 97.7 | 96.6 / 96.6 | +1 / -15 / -75 |
| `w_tatoeba=3 seg_min_letters=5 accentless=2` | 394,704 | 82,154 | 91.7 / 97.5 / 97.8 | 96.6 / 97.0 | +1 / -18 / -74 |
| none (Tatoeba weighted 1, `seg_min_letters` 6) | 404,592 | 84,433 | 90.5 / 97.2 / 97.7 | 93.1 / 94.1 | +0 / -17 / -82 |
| `w_tatoeba=2` (`seg_min_letters` 6) | 404,592 | 84,433 | 91.3 / 97.2 / 97.9 | 92.9 / 93.9 | +0 / -18 / -78 |
| `seg_min_letters=5` (Tatoeba weighted 1) | 404,592 | 84,433 | 91.1 / 97.1 / 97.8 | 93.1 / 94.1 | +1 / -17 / -82 |
| B + `w_gv=0.5` | 404,592 | 84,433 | 91.4 / 97.3 / 98.0 | 93.0 / 94.0 | +1 / -17 / -77 |
| B + `alphabet=folded` (accents folded into their vowels, ñ kept: 27 symbols) | 404,584 | 88,253 | 92.0 / 97.5 / 98.2 | 93.6 / 93.7 | +1 / -9 / -55 |
| B + `alphabet=noü` (ü folded into u: 32 symbols) | 404,592 | 85,177 | 91.9 / 97.4 / 98.0 | 93.2 / 94.2 | +1 / -14 / -71 |
| B + `ngram=hashed order=4 hbits=7,11,14,15` (code points) | 404,576 | 69,397 | 86.9 / 95.8 / 96.8 | 92.6 / 94.1 | +2 / +10 / +23 |
| B + `max_words=60000` | 298,384 | 60,000 | 90.6 / 97.2 / 97.7 | 92.6 / 94.0 | +0 / -17 / -74 |
| B + `lambda=-2.5` | 404,592 | 84,433 | 91.7 / 97.4 / 98.0 | 92.6 / 93.6 | +0 / -18 / -77 |
| B + `lambda=-1.75` | 404,592 | 84,433 | 92.1 / 97.4 / 97.9 | 92.5 / 93.5 | +1 / -19 / -84 |
| B with `seg_min_letters=8` | 404,592 | 84,433 | 92.1 / 97.5 / 97.9 | 92.7 / 93.7 | +1 / -17 / -79 |
| B + `min_count=5` | 404,592 | 84,433 | 91.8 / 97.4 / 98.0 | 94.1 / 94.1 | +1 / -17 / -74 |
| B, Tatoeba only (`extra_books=0 gv=0 corpus_books=0`) | 170,896 | 30,677 | 90.3 / 96.3 / 97.2 | 89.8 / 92.4 | +5 / +7 / +6 |
| B + `gv=0` | 404,592 | 84,433 | 91.7 / 96.9 / 97.7 | 92.7 / 93.2 | +1 / -12 / -46 |
| B + `extra_books=0` | 404,592 | 84,433 | 91.2 / 97.7 / 98.0 | 92.2 / 93.5 | -1 / -18 / -96 |
| shipped + `tatoeba_names=0` (Tatoeba's counts of names it almost always capitalises, Tom, Mary, Muiriel, left out) | 396,944 | 82,670 | 90.7 / 97.5 / 97.7 | 96.7 / 96.7 | +0 / -18 / -92 |
| shipped + `max_word_chars=12` / `=10` | 396,944 | 82,674 | 91.5 / 97.5 / 97.9 ; 91.4 / 97.6 / 98.0 | | ≈ the same |

What decided it:

- The large sources (714 books and Global Voices against Tatoeba alone, both without
  accentless spellings) raise recall by 1.6 / 1.1 / 0.9 points at Low / Medium / High and
  accentless recall by 3.1 / 1.5 points at Medium / High. They also move junk from en to es:
  es outranks en on it and rejects it (see "Numbers").
- Accentless spellings cost nothing measurable on dev, and accentless recall goes from 92.9%
  to 96.6% at Medium.
- Folding the accents scores as well on dev, but it removes fewer of en's false positives
  (the rise at High is -55, against -75), and its n-grams no longer tell `n` from `ñ`.
- The hashed 4-gram, the smaller word list and Tatoeba alone lose recall.
- A held-out check of the news register: 4% of the Global Voices lines (blocks of 10,000
  lines) left out of training, and 2,145 positives built from them the way the corpus builds
  its own (600 sentences, phrases, short phrases, words, caps and space-less forms). With the
  b56b0c5 fit, recall at Low / Medium / High is 93.3 / 95.4 / 96.4% for all of them and
  99.0 / 99.5 / 99.7% for the sentences (79 of the 600 have 200 characters or more). The
  accentless proxy with the shipped pack: 95.6 / 96.6 / 96.9%.

Latency was compared too (`max_word_chars`, `seg_min_letters`, word-list size, a stricter
n-gram gate, `rank_offset`). None of them changes the cost of es much, see "Latency".

## Numbers

Framework 77ecbfa (staged scoring) and 407fd7c, `gib-bench fit --lang es` at `--fpr-scale` 1 (the
same calibration on both; every en + es dev output is byte-identical between them, and with
Latin packs only 407fd7c's per-script candidates are one group, the 77ecbfa code): 21 / 10 / 5
new dev false positives at High / Medium / Low for a budget of 35.4 / 14.2 / 7.1; the
pre-gate and the bucket gates reject 21,895 and 8,597 dev negatives before the full n-gram
pass and the word model. Gates 2–5 are measured on dev with en + es against en alone.

### Gates

| gate | measured | needed | |
|---|---|---|---|
| 1. recall, test (1,690 positives) | Low 93.3%, Medium 95.6%, High 96.4%; sentences at Medium 97.9% | 80 / 90 / 92%; 95% | pass |
| 2. shared junk false positives, dev (70,882 negatives) | Low 54 → 50 (**-4**), Medium 166 → 131 (**-35**), High 409 → 255 (**-154**); `gib-bench gate` GATE_OK | at most +7 / +14 / +35 | pass |
| 3. own junk FPR at Medium, dev (4,094 negatives) | 0.20% (Low 0.049%, High 0.56%) | ≤ 1% | pass |
| 4. confusion, dev | es positives accepted at High reported as es: 99.4%; en positives reported as es: 0.41%; en recall +0.15 / 0 / 0 points | ≥ 90%; ≤ 3%; ≥ -0.2 | pass |
| 5. latency, dev, every pack on main plus es | p50 1.1–1.3 µs, p99 15.0–17.8 µs (6 runs of `gib-bench latency --check` at 77ecbfa and 407fd7c, all LATENCY_OK) | ≤ 3 µs, ≤ 30 µs | pass |
| 6. size | 396,944 + 3,729 = 400,673 bytes; WASM module (en + es) after `wasm-opt -Oz` 1,644.4 KB; SIZES_OK | ≤ 409,600 bytes; ≤ 5.5 MB | pass |

### Dev (corpus v1)

| | Low | Medium | High |
|---|---|---|---|
| es recall (1,413 positives) | 96.3% | 97.7% | 98.6% |
| es sentences (425) | 99.3% | 99.8% | 99.8% |
| for comparison, es recall of 5.0.7 / of Ciphey's English checker (frozen baselines) | 10.3% / 6.4% | 37.8% / 17.6% | 83.1% / 25.2% |

- By category (Low / Medium / High): sentence 99.3 / 99.8 / 99.8%, caps 96.7 / 97.3 / 98.2%,
  nospace 98.3 / 99.7 / 99.7%, phrase 97.3 / 97.3 / 99.1%, mixed 95.0 / 100 / 100%, paragraph
  100%, short 85.7 / 89.8 / 93.9%, word 84.8 / 90.2 / 94.6%.
- By length (Low / Medium / High): 1–9 characters 86.4 / 90.7 / 94.9%, 10–19 94.8 / 96.2 /
  97.2%, 20–49 99.3 / 100 / 100%, 50–199 100%, 200+ 95.7 / 100 / 100%.
- Shared junk goes down: on long texts es ranks first on most Spanish-derived junk and on
  much other Latin junk, rejects it, and stops the scoring there, so English no longer
  accepts it (docs/languages.md, "How detection works with many languages").
- Other languages without a pack: es now claims 28.2% of Portuguese, 11.6% of Italian and
  5.4% of French dev positives, and other languages' recall drops (Italian by 4.6 / 5.6 /
  7.4 points) where es ranks first and rejects. Gate 4 counts only packs on main; the pt,
  it and fr packs win their text back in their PRs.

### Test (corpus v1, run once: 2026-11-05 00:17Z, `model.bin` sha256 `028d18cb…6693`, `calibration.txt` `3b1c5413…316e`)

| | Low | Medium | High |
|---|---|---|---|
| es recall (1,690 positives) | 93.3% | 95.6% | 96.4% |
| es sentences (576) | 97.6% | 97.9% | 98.1% |
| es recall by source: Tatoeba 849 / Wikipedia 444 / Gutenberg 395 | 97.2 / 89.4 / 89.1% | 98.4 / 93.2 / 92.2% | 98.6 / 95.3 / 93.2% |
| for comparison, 5.0.7 / Ciphey's English checker | 8.5% / 5.5% | 38.5% / 19.8% | 88.2% / 30.1% |
| shared junk false positives, en + es against en (90,764 negatives) | 121 → 100 (-21) | 297 → 183 (-114) | 730 → 357 (-373) |
| own junk FPR (4,889) | 0.12% | 0.20% | 0.47% |
| en recall (5,026), en + es against en | ±0 | -0.04 points | ±0 |

- es positives accepted at High reported as es: 98.7%; en positives reported as es: 0.84%.
- By category (Low / Medium / High): sentence 97.6 / 97.9 / 98.1%, caps 96.5 / 97.3 / 98.0%,
  nospace 88.3 / 94.4 / 95.5%, phrase 92.2 / 92.2 / 94.8%, mixed 95.0%, paragraph 95.0 / 100 /
  100%, short 84.6 / 90.1 / 92.3%, word 77.5 / 84.5 / 87.3%.
- What the test split's misses are: Wikipedia names (`Zyras`, `Manazkert`,
  `MATTHIASJÜRGENZIMMERMANN…`), and Old French, Latin and Old Spanish quotations in the
  test book pg73257 (a history of Spanish literature: `bien fu joïz`, `Bis novem noster ||
  populus`, `pitança`). The Old French and Latin samples are labelled Spanish in corpus v1
  (reported to gib-lead for v2).

### Latency

Dev samples of up to 256 characters, shared 64-core host, release builds, `gib-bench latency
--check` in three runs, alternating with the framework head without es (main for this PR):

| framework, host load | en alone p50 / p99 | en + es p50 / p99 |
|---|---|---|
| 77ecbfa, load 49–55 | 935–980 ns / 11.2–12.3 µs | 1.1–1.2 / 15.0–16.0 µs (LATENCY_OK) |
| 407fd7c, load 65 | 1.0 / 13.1–13.5 µs | 1.2–1.3 / 16.0–17.8 µs (LATENCY_OK) |

Thirteen packs, as when every language has shipped (`--synthetic`: copies of en under made-up
letters), 407fd7c at load 88–90: en + 12 synthetic 3.0–3.3 / 15.8–17.3 µs; en + es + 11
synthetic 3.2–3.3 / 18.4–19.5 µs. So es costs about what a synthetic pack does at p50 and
2 µs more at p99. At that load the framework's own 13-pack check is at its 3 µs p50 limit
with or without es.

Before the staged scoring (framework b56b0c5) es added 0.3–0.9 µs at p50 and 8–12 µs at
p99, almost all of it in its full scoring; none of the pack's own settings changed that much
(`max_word_chars` 10–16, `seg_min_letters` 5–8, 20,000 words, `rank_offset`), which led to
the framework change and the new gate 5.

### Sizes

`gib-bench sizes --wasm --no-latency --check`: es 387.6 KB + 3.6 KB; WASM module with en and
es after `wasm-opt -Oz` 1,644.4 KB (en alone 1,239.6 KB); SIZES_OK.

## Rebuild

```sh
python3 languages/es/build.py --fetch       # once: the extra books and Global Voices into the cache
python3 languages/es/build.py               # model.bin, build.json (about 10 min, 0.7 GB RAM)
cargo build --release -p gib-bench
cargo run --release -p gib-bench -- fit --lang es
cargo build --release -p gib-bench          # embed the new calibration
cargo run --release -p gib-bench -- --corpus v1 --split dev
```

`build.py` is deterministic: two fresh runs and a run from cached counts gave the same
`model.bin` (sha256 in `build.json`). `--counts-cache DIR` keeps the counts as JSON between
experiments.
