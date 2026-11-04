# Design

gibberish-or-not answers one question: **is this text natural language, in one of the
languages compiled in?** It returns a verdict, the language and a calibrated score. It is
built for [Ciphey](https://github.com/bee-san/Ciphey), which asks it about every output of
every decoder during a search (a median of ~225 checks per search, up to ~30,000;
bee-san/Ciphey#1031), so it must reject nearly everything, accept the plaintext, and do
both in microseconds, natively and in WASM.

## The boundary with pyWhat

- **gibberish-or-not**: is the text natural language? Identifier-like tokens (numbers,
  hex and Base64 runs, URLs, email addresses, `@mentions`, `#hashtags`, CTF flags) are
  **neutral**: they are left out and the rest of the text decides. They neither make text
  language nor make it gibberish. On its own, an identifier is "not language".
- **[pyWhat](https://github.com/bee-san/pyWhat)**: is the text, or does it contain, a known
  structured identifier (email, hash, key, wallet, URL, CTF flag, ...)? It says nothing
  about natural language.
- **Ciphey (Athena)** composes them: crib → pyWhat full-input identification → JSON/code →
  password → gibberish-or-not. A pyWhat match makes the input plaintext only when it covers
  the whole input; identifiers inside a sentence don't count as words here, so nothing is
  counted twice.

## Architecture

```
text ──► analysis (one pass, language-independent)            src/text.rs
           words, neutral chunks, case, symbols, scripts
           veto: control characters, no letters, odd case ──► score 0
      ──► candidate packs: those of the text's majority script  src/lib.rs
      ──► each candidate: n-gram pass → rank; below its gate: out  src/model.rs
      ──► best 3 by rank (all, under 12 letters): word pass,
           20 features, logistic logit, calibrated score         src/model.rs, src/pack.rs
      ──► best score wins (ties: margin) → Detection { is_language, language, score }
```

- **Analysis** (`text::analyze`). Whitespace splits the text into chunks; a chunk whose
  core (edge punctuation stripped) has a digit, `://`, a leading `www.`, an `@`-address, a
  `prefix{...}` flag shape or 8+ hex letters is neutral. Other chunks are split into word
  tokens (letters, in-word apostrophes, combining marks; split at script changes, Han and
  kana together). In Han, kana and Hangul chunks a number is skipped rather than making the
  whole chunk an identifier: a chunk there is a sentence (`他工作了6年`). It counts writing
  punctuation, decoder symbols (`\^[]{}|~=<>%*+` and the backtick), other symbols, control
  characters, unbalanced brackets, letter variety and repeated words, and classifies each
  token's case: lower, upper, capitals that start lower-case runs (`GitHub`), an
  acronym's plural (`URLs`), long alternating-case runs (`hiTHEREwelcome`), or odd
  (`hHeREoUtHo`). Only the first 4,096 bytes are looked at.
- **Vetoes**: a control character other than tab and line breaks (decoder bytes, never
  writing), no letters outside identifiers, odd case in more than a third of the cased
  words (or in any of three or fewer), one letter repeated (`zzzz`, `A A A A`), or letters
  that repeat a unit of at most 6 letters three times (`INININ`, `and the and the and
  the`), or twice when the unit is two words or more (`is the is the`): score 0. These are
  what decoders make of repeated digits or bytes, and no weight could learn them from data
  that has few of them.
- **N-gram pass** (`model::ngram_pass`): over the letters of the words, with everything
  else removed (word boundaries too, so space-less text scores like spaced text), the mean
  conditional log10 P of each letter given the previous `order - 1` (English: dense 4-gram
  over a–z, interpolated absolute discounting; big alphabets: hashed tables with
  fingerprints and backoff). Letters outside the alphabet are foreign: they break the
  context and count against the text. The **rank** is that mean plus log10 of the alphabet
  size (a uniform model scores 0), foreign letters at -3: comparable across the packs of a
  script.
- **Gate**: each pack's calibration has a rank below which none of the dev split's
  positives fall (minus a margin); texts below it are rejected by that pack without the
  word pass. For English it drops 40% of the dev junk before any word lookup.
- **Word pass** (`model::features_from`): each token is looked up in the pack's word table
  (frequency classes of half a log10 step); tokens with apostrophes whole, then by parts;
  long unknown tokens (6+ letters without spaces, 10+ in spaced text) are split into words
  by Viterbi over the word table, unknown letters at `lambda` (English -2.0 per letter),
  only if their n-gram score is plausible (≥ -1.45 for a 26-letter alphabet, the same rank
  for other packs) and only within a budget of 64 letters per text. Runs of Han or kana
  (no spaces) are segmented in windows of 48 characters, with their own budget of 128. Unknown or rare tokens of text that is otherwise mostly words are checked for
  being one letter off by one from a much more frequent word (`romethinf`, `shem`): the
  signature of a cipher key with one wrong letter.
- **Features** (20, in `pack::FEATURES`): bias; n-gram mean; known-letter coverage; word
  score per letter; odd-case share; decoder-symbol, other-symbol and punctuation shares;
  foreign letters (of the pack's script but not its alphabet: words in other scripts are
  neutral); spaced or not; mean token length; identifier share; unbalanced brackets;
  space-less text entirely covered by known words; n-gram mean, coverage and word score
  again for space-less text only; unknown-token share; unknown lower-case letters in mixed-
  case text; near-miss share; repeated words; letter variety.
- **Classifier and calibration** (`pack::Bucket`): per length bucket (characters of the
  text without its identifiers: 1–3, 4–5, 6–7, 8–9, 10–14, 15–19, 20–29, 30–49, 50–99,
  100–199, 200+), a logistic-regression logit
  (weights fitted on the train split per group of buckets: 1–9, 10–19, 20–49, 50+) mapped
  piecewise-linearly through five anchors to the score: the anchors at score 0.5, 0.75 and
  0.9 are the logits at which the **dev split's junk false-positive rate** reaches the
  High, Medium and Low targets of that bucket (`bench/src/fit.rs`). So a score means a
  documented false-positive rate in each length bucket (stricter for short text), the same
  in every language, and the sensitivities are fixed score thresholds.

## Decisions, and what was rejected

**Character n-grams + word frequencies + a small linear classifier, not a neural model.**
bee-san/Ciphey#1031 measured the candidates on Ciphey's corpus (157 positives, 2,604
negatives; thresholds at the same false-positive count):

| English score | recall | no-space recall | time / check | size |
|---|---:|---:|---:|---:|
| gibberish-or-not 5.0.7 (+ Ciphey's fix 4) | 83.4% | 50.0% | 7.8 µs | in binary |
| quadgram log-probability | 96.2% | 100% | 0.14 µs | 1.8 MB |
| DistilBERT (the 5.x download, via ONNX) | 79.0% (17 FPs) | 4.5% | 5.5 ms | 268 MB + 96 crates |
| Laya zero-shot (ONNX) | 36.3% | 4.5% | 140 ms | 1.7 GB |
| lingua 1.8 (EN/FR/DE/ES) | 39.5% | 22.7% | 87 µs | ~2.6 MB/language |
| whatlang 0.18 | 54.8% | 13.6% | 50 µs | ~83 KB |

At ~30,000 checks per search, 5 ms per check is minutes; and the 5.x BERT path never
loaded at all (`missing field hidden_size`, #1031). Language identifiers (lingua, whatlang,
whichlang) answer "which language", always returning one, not "is this language". The
n-gram + word approach is what wins on the dev split here too (docs/benchmark.md), at
~1 µs.

**Letters-only n-grams** (spaces and punctuation removed), not n-grams with word
boundaries: one model then scores `MEETMEATNOON` and `meet me at noon` alike, and the
word pass carries the boundary information.

**Conditional probabilities with interpolation**, not Ciphey's joint quadgram table: a
proper per-letter likelihood, comparable across packs once offset by the alphabet size.
Dense tables when the alphabet is small (English: 26^4 bytes, collision-free, one lookup
per letter); hashed tables with fingerprints for Han, kana, Hangul and large alphabets.

**A word table of 28-bit fingerprints in two-choice 8-slot buckets** (cuckoo-filled),
parsed from bytes without copying, not `phf` (compile-time only, 16+ bytes per entry for
`&str`), an FST (not `no_std`) or a sorted list (binary search). 115,000 English words
are 500 KB; a lookup is two cache lines and a false match is ~6·10⁻⁸.

**Frequency classes, not a word set.** A lone four-letter string is a dictionary word one
time in 150 (#1031), so "is it a word" can't decide short text; how frequent the words are
can, together with the calibration of the shortest buckets.

**A per-bucket logistic regression calibrated on false-positive rates**, not Platt or
isotonic posteriors. The class balance of a benchmark is arbitrary, so a "probability of
language" would mean nothing at run time; a false-positive rate does: it is what a search
pays. Fixed score thresholds per sensitivity (0.9, 0.75, 0.5) keep `Sensitivity` and a
custom `min_score` simple. Rejected: a tree or MLP over the same features (the linear model
already separates the dev split within the targets; a tree adds size and hides the
calibration).

**Identifiers are neutral**, not evidence either way (the pyWhat boundary): `The password
is IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR` is language, `IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR` alone
is not, and pyWhat says what it is.

**`no_std` + `alloc`.** It costs little: no transcendental functions at run time (scores are
sums of table values; the calibration is piecewise linear; `log10` of alphabet sizes is a
short series), `str::parse::<f32>` and `char::to_lowercase` are in `core`, and the lazily
parsed built-in packs use `once_cell::race::OnceBox` (no locks, no `std`). It buys the
WASM build with default features, embedded targets, and no hidden OS use: no files,
threads, network, environment, process or clock. Rejected: `std` with feature-gated
pieces (easy to regress on), a `libm` dependency (not needed).

**Packs are data with a fixed reader**, not per-language code: a language adds files and
one registry line, the format has one writer (`languages/common/packlib.py`) and one
reader (`src/pack.rs`), and custom packs load at run time (`Pack::from_bytes`). Rejected:
a `build.rs` that discovers `languages/*/` (implicit, and a stray directory would ship),
per-language Rust modules (12 PRs editing shared code).

**Top 3 by n-gram rank, then the full model**, not every pack: with 9 Latin-script
languages, scoring all of them is 3× the time. The n-gram rank puts a text's own
language first; the 3 best keep close languages (Spanish, Portuguese, Italian) all in the
running; texts under 12 letters, where ranks are noisy, get all packs.

**The CLI has no dependencies** (feature `cli`), instead of clap: the library pulls in
nothing for it.

**Removed:** the BERT/candle/tokenizers/safetensors path and the HuggingFace download code
(96 of the 258 crates in Ciphey's dependency graph, and onig's C library that broke
Ciphey's CI), `.cargo/config.toml` (forced clang + mold, which CI runners don't have, so
every run failed), and the 5.x dictionary of unknown licence.

## Limits

- **Junk the corpus lacks** (found in the review of the framework PR, for corpus v2):
  pronounceable pseudo-words from a letter-trigram model, and English with look-alike
  letters of other scripts, are accepted more often than by Ciphey's English checker.
- **Short text is ambiguous.** A wrong Caesar shift of a short word is often another word
  (`Lorry` → `Fills`); the shortest buckets are calibrated strictly, so some real one- and
  two-word texts are rejected at Low.
- **Partly decrypted ciphertext** (a key with one wrong letter) is the hardest junk: it is
  mostly real words. The near-miss feature catches letters off by one; other offsets are
  only caught by coverage.
- **Names-only text** (`READ RUMI'S MATHNAWI`) has few known words.
- **Mixed-language text** is scored by its majority script and the best language; a
  sentence half in two languages can score lower than either half.
- **Only the first 4,096 bytes** are read.
