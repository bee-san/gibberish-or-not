<div align="center">

# gibberish-or-not

**Is this text natural language, or gibberish?**

[![Crates.io](https://img.shields.io/crates/v/gibberish-or-not.svg)](https://crates.io/crates/gibberish-or-not)
[![Documentation](https://docs.rs/gibberish-or-not/badge.svg)](https://docs.rs/gibberish-or-not)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

gibberish-or-not tells natural-language text apart from everything else: cipher and decoder
output, encodings, hashes, random strings, mojibake. It says which language and gives a
calibrated score. It is what [Ciphey](https://github.com/bee-san/Ciphey) asks about every
candidate plaintext, thousands of times per search, so a check takes about a microsecond,
the core is `no_std` with no OS, file, thread or network use, and it runs in the browser.

```toml
[dependencies]
gibberish-or-not = "6"
```

```rust
use gibberish_or_not::{detect, is_gibberish, Detector, Lang, Sensitivity};

assert!(!is_gibberish("The quick brown fox jumps over the lazy dog", Sensitivity::Medium));
assert!(!is_gibberish("MEETMEATTHEUSUALPLACEATTEN", Sensitivity::Medium)); // no spaces
assert!(is_gibberish("Wkh txlfn eurzq ira mxsv ryhu", Sensitivity::Medium)); // Caesar +3
assert!(is_gibberish("5f4dcc3b5aa765d61d8327deb882cf99", Sensitivity::High)); // a hash

let d = detect("Hello, my name is Bee and I like to decode things");
assert!(d.is_language);
assert_eq!(d.language, Some(Lang::En));
println!("{:.2}", d.score); // calibrated: 0.5 High, 0.75 Medium, 0.9 Low

// The strictest level, English only
let strict = Detector::new().languages(&[Lang::En]).sensitivity(Sensitivity::Low);
assert!(strict.is_gibberish("Gsv jfrxp yildm ulc"));
```

Upgrading from 5.x: [MIGRATION.md](MIGRATION.md) (`is_gibberish`, `is_password` and
`Sensitivity` are unchanged).

## How good is it

On the frozen benchmark corpus (bench/corpus/v1, test split: 5,026 English texts of every
kind and 90,345 pieces of junk: wrong cipher keys, near-English Vigenère key-search output,
encodings, hashes, random strings, mojibake, real candidates from Ciphey's searches),
against gibberish-or-not 5.0.7 and against 5.0.7 plus everything Ciphey's English checker
adds to it today:

| | Low: recall / false positives | Medium | High |
|---|---|---|---|
| **gibberish-or-not 6.0** | **97.2% / 0.14%** | **98.7% / 0.36%** | **98.9% / 0.87%** |
| 5.0.7 | 70.9% / 0.63% | 84.1% / 3.06% | 92.1% / 19.1% |
| 5.0.7 + Ciphey's English checker | 95.3% / 0.27% | 97.0% / 0.51% | 97.3% / 1.23% |

Per length, kind of text and kind of junk, the speed, the sizes and how to reproduce every
number: [docs/benchmark.md](docs/benchmark.md). In short:

- **Speed**: median 0.7 µs and 99th percentile 14 µs per check (inputs up to 256
  characters, on a shared busy host); WASM within 1.7× of native.
- **Size**: the English pack is 956 KB; the whole WASM module after `wasm-opt -Oz` is
  1.2 MB.
- **5.x's BERT path** never loaded a model (bee-san/Ciphey#1031) and is gone, with its 96
  crates.

## Languages

English ships in 6.0.0. Spanish, French, German, Italian, Portuguese, Dutch, Polish,
Turkish, Russian, Chinese, Japanese and Korean are built under one contract
([docs/languages.md](docs/languages.md)) and ship in 6.x as they pass its gates. Each is
a Cargo feature (`en`, `es`, ... `ko`), all on by default:

```toml
gibberish-or-not = { version = "6", default-features = false, features = ["en"] }
```

Packs for other languages, or your own English, load at run time:
`Detector::new().with_pack(Pack::from_bytes(&bytes)?)`.

## Sensitivity

`Low` is the strictest, `High` the most lenient. Each level is a fixed minimum score (0.9,
0.75, 0.5), calibrated so that it accepts the same small share of junk at every text
length (docs/benchmark.md has the targets). Ciphey checks its classical ciphers' best keys
at Low and everything else at Medium.

## What counts as language

Text in a supported language, in any case, with or without spaces between the words,
short phrases included. Identifiers (numbers, URLs, email addresses, hashes, keys, CTF
flags) are neutral: they are left out and the rest decides, so `The password is
IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR` is language and the password alone is not. Telling what
an identifier is, is [pyWhat](https://github.com/bee-san/pyWhat)'s job.

## How it works

Per language, a character n-gram model, a word list with frequencies (and word
segmentation for text without spaces) and a small logistic classifier, calibrated against
the junk false-positive rate per length; no neural network. [docs/design.md](docs/design.md)
has the architecture, the decisions and the options that lost.

## WASM, no_std, CLI

- `cargo build --target wasm32-unknown-unknown` works with default features. The `wasm`
  feature adds JavaScript bindings: `detect`, `score`, `isGibberish`, `isPassword`,
  `languages`.
- The library is `#![no_std]` with `alloc`; models are embedded with `include_bytes!`.
- `cargo install gibberish-or-not --features cli` gives `gibberish [--sensitivity
  low|medium|high] [--lang en,...] [TEXT ...]` (reads lines from stdin without TEXT).

## Passwords

`is_password(text)` is true for 10,000+ common passwords (Ciphey's password checker uses
it).

## Contributing

Issues and pull requests are welcome. Language packs follow
[docs/languages.md](docs/languages.md); every accuracy claim comes from
`cargo run --release -p gib-bench -- --corpus v1` (docs/benchmark.md).

## License

MIT ([LICENSE](LICENSE)); data sources and their licences in [NOTICE](NOTICE).
