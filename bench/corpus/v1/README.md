# Benchmark corpus v1 (frozen)

The data gibberish-or-not 6 is fitted, calibrated and measured on. Frozen: never edit these
files; a changed corpus is `v2`. `python3 bench/scripts/make_corpus.py --check` rebuilds it
from the data cache and checks that every file is byte-identical (`MANIFEST.sha256` has the
sha256 of every input and output).

| split | positives | negatives | info | used for |
|---|---:|---:|---:|---|
| train | 14,426 | 70,464 | 1,472 | fitting: each pack's classifier weights (`gib-bench fit`) |
| dev | 17,292 | 70,442 | 1,449 | thresholds and calibration (`gib-bench fit`), every tuning run, the CI gate |
| test | 21,871 | 90,345 | 1,979 | reporting only: run once per released model version |

## Format

`train.tsv`, `dev.tsv`, `test.tsv`: one sample per line,
`id<TAB>lang<TAB>label<TAB>category<TAB>source<TAB>text`.

- `id`: the first 12 hex digits of the text's sha256. Texts are unique across the corpus.
- `lang`: the language of a positive, or the language a negative was derived from (`-`:
  noise that comes from no language).
- `label`: `pos` (natural-language text), `neg` (not language), `info` (reported, never
  counted: see below).
- `category`: what the sample is (lists below).
- `source`: where it came from: `gutenberg:<ebook>`, `tatoeba:<sentence id>`,
  `wikipedia:<lang>`, `ciphey:<file>` or `ciphey:search:<set>/<case>`, `noise`, `vigopt:*`,
  `mixed-junk`.
- `text`, with `\t`, `\n`, `\r` and `\\` escaped, the other control characters (C0, DEL,
  C1) as `\xHH`, and U+2028/U+2029 as `\uHHHH`.

## Splits are by source

One source is in exactly one split, and everything derived from a sample (its upper-case
and space-less forms, its ciphered junk) is in that sample's split:

- **Project Gutenberg**: by book. English: the 85 books Ciphey's quadgram table is counted
  from are train (as are 299 more training books the English pack uses), 4300, 10007 and
  69087 are dev, and 1342, 161, 158 (Jane Austen), 98, 64317, 1250, 26184 and 67098 are
  test. Other languages: the books `fetch_sources.py` picked, 60/20/20 in id order.
- **Tatoeba**: by contributor (60/20/20 by a hash of the user name), so no author is in two
  splits.
- **Wikipedia** (CC BY-SA): test only.
- **Ciphey's harness data**: `curated.tsv` train is train, its held-out half is split
  into dev and test by a hash of each line (each line is its own source);
  `gutenberg.tsv`'s held-out books are test books here, its train books train.
- **Ciphey search captures**: by search (the end-to-end case), 60/20/20.
- **Noise** (random strings, encodings of random bytes): round-robin.

## Positives (`pos`)

Natural-language text in one of the 13 languages (`en es fr de it pt nl pl tr ru zh ja ko`):

| category | what |
|---|---|
| `sentence` | a sentence (Tatoeba, Gutenberg, Wikipedia; Ciphey's other-language samples) |
| `paragraph` | a paragraph of 200–700 characters (Gutenberg) |
| `phrase` | 1–4 consecutive words of a sentence (2–8 characters of Chinese or Japanese) |
| `short` | 1–3 consecutive words, at most 12 letters |
| `word` | one word |
| `caps` | a sentence or phrase in upper case (Latin and Cyrillic scripts) |
| `nospace` | a sentence's letters without spaces or punctuation, upper or lower case (Latin and Cyrillic) |
| `mixed` | a sentence with an identifier inserted (URL, email, hash, CTF flag, number, key, IP, version): identifiers are neutral |
| `chat`, `technical`, `formal`, `ctf_text`, `literary`, `verse` | Ciphey's hand-picked English (`curated.tsv`, `gutenberg.tsv`) |

## Negatives (`neg`)

Derived from every positive base sample, in its split (`lang` = the source language).
English gets every generator that applies; other languages five of them per sample:

| category | what |
|---|---|
| `caesar` | wrong Caesar shifts (two per sample; Cyrillic: over а–я) |
| `rot47`, `atbash`, `affine` | ROT47; Atbash (Latin, Cyrillic); a random affine map `ax+b` (a≠1) |
| `substitution` | a random monoalphabetic substitution (keeps word lengths and letter patterns) |
| `vigenere` | encrypted with a random key, decrypted with another |
| `vigenere_partial` | decrypted with the right key but its last letter off by one: every k-th letter wrong |
| `reversed` | the text backwards |
| `railfence`, `columnar` | encrypted with one transposition key, decrypted with a wrong one |
| `unicode_shift` | Chinese, Japanese, Korean: every code point moved by k inside its block |
| `enc_<encoding>` | one layer of an encoding left: base64, base32, hex, hex_spaced, binary, decimal, octal, url, morse, a1z26, html_entities, unicode_escapes, base58, base85 |
| `mojibake` | the UTF-8 bytes read in another encoding (Latin-1, cp1252, cp1251, KOI8-R, GBK, Shift_JIS, EUC-KR, UTF-16), or the reverse |
| `caesar_mixed` | a wrong Caesar shift with an identifier inserted: still junk |

Language-independent junk:

| category | what |
|---|---|
| `vigenere_opt_bigram`, `vigenere_opt_quadgram` | Vigenère key search output maximising English bigram or quadgram fitness, on random letters and on English ciphertexts attacked with a wrong key length: the near-English junk Ciphey's Vigenère decoder makes |
| `search_random` | an even sample of the texts Ciphey's A* search asked its checkers about on its end-to-end corpora (Ciphey master 9350174d, 1 s per search), minus the answers |
| `rand_upper`, `rand_lower`, `rand_alnum`, `rand_ascii`, `rand_words`, `digits` | random strings |
| `hex_random`, `base64_random`, `base32_random`, `binary_random`, `hash`, `uuid` | encodings of random bytes, hashes of random words |
| `rand_bytes`, `rand_bytes_lossy`, `utf16_random`, `utf16le_as_latin1`, `ascii_as_utf16` | bytes read as Latin-1, UTF-8 (lossy) or UTF-16 |
| `rand_cjk`, `rand_hangul`, `rand_kana`, `rand_cyrillic`, `rand_latin_ext`, `base65536` | random letters of other scripts, and code points from Base65536's blocks |

## Info (`info`: reported, never counted)

- Identifiers on their own (`ctf_flag`, `url`, `email`, `ip`, `identifier`, `password`),
  `json` and `code`: pyWhat's and Ciphey's other checkers' job; on its own an identifier is
  not language (docs/design.md).
- `other_language`: Ciphey's samples in languages outside the 13.
- `search_hard`: search candidates Ciphey's Athena@High accepted. By construction it holds
  near answers that read as language (`The meeting is at 6785 in room 959 ...`).
- `railfence_rotation`, `columnar_rotation`, `*_near_plaintext`: wrong transposition keys
  that leave the plaintext rotated or mostly in place.
- `mojibake_light`: mojibake of text with few non-ASCII letters, which still reads as the
  text.

## Sources and licences

The raw sources are in the data cache, never in the repository; `bench/sources/sources.json`
has each file's URL, date, size, sha256 and licence, and `bench/scripts/fetch_sources.py`
downloads them.

| source | licence | in this corpus |
|---|---|---|
| [Tatoeba](https://tatoeba.org) per-language exports of 2026-10-31 | CC BY 2.0 FR. Attribution: each sample's `source` is its sentence number, `https://tatoeba.org/en/sentences/show/<id>` names its author | sentences, and everything derived from them |
| [Project Gutenberg](https://www.gutenberg.org) books | public domain in the USA | sentences, paragraphs, and derived |
| Wikipedia article introductions, 300 random articles per language, 2026-11-04 | CC BY-SA 4.0: **evaluation only**. They are in the test split and never in a published crate or language pack | sentences, and derived (test) |
| [Ciphey](https://github.com/bee-san/Ciphey) master 9350174d: `benches/data/plaintext/curated.tsv`, `gutenberg.tsv`, `e2e.tsv`, copied unchanged into `ciphey/` | MIT | Ciphey's samples, and derived |
| Ciphey search captures, `ciphey/captured_cases.tsv` (written by a scratch `--capture-cases` mode of Ciphey's plaintext bench) | MIT (Ciphey's output) | `search_random`, `search_hard` |

The corpus is not part of the published crate (Cargo.toml `include`).
