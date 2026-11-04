# Frozen baseline verdicts

What gibberish-or-not 6 is compared with, on every sample of the corpus:

- **5.0.7**: `!gibberish_or_not::is_gibberish(text, s)` of the published 5.0.7, at Low,
  Medium and High (`Sensitivity` has the same meaning in 5.x and 6: Low is the strictest).
- **Ciphey-English**: `Checker::<EnglishChecker>::new().with_sensitivity(s).check(text).is_identified`
  of Ciphey master `9350174d` (2026-11-04): 5.0.7 plus everything Ciphey's English checker
  layers on it (quadgram scoring of space-less text, short-phrase word rules, decoder-output
  vetoes, masking of flags and secrets). It is the English part of Athena, without the
  format checkers (LemmeKnow, JSON, password, code), which answer a different question.

`v1/<split>.tsv` columns: `id`, 5.0.7 at Low/Medium/High, Ciphey-English at Low/Medium/High
(1 = accepted as language), then nanoseconds per check of 5.0.7@Medium and
English@Medium (measured on the same host, same method as gib-bench: checks under 2 µs
repeated). `gib-bench` reads these files, so reports don't need to build 5.0.7 (which
pulls in candle, tokenizers and reqwest).

## How they were made

In a clone of Ciphey at `9350174d` with `ciphey-scratch.patch` applied (it adds
`examples/gon_baselines.rs`, and a `--capture-cases` mode to `benches/plaintext.rs` that
recorded `../corpus/v1/ciphey/captured_cases.tsv`):

```sh
cargo build --release --example gon_baselines
# id<TAB>text for each sample of a split (the corpus's own escaping):
grep -v '^#' bench/corpus/v1/dev.tsv | cut -f1,6 > ids-dev.tsv
target/release/examples/gon_baselines < ids-dev.tsv > dev-verdicts.tsv
```

(`cut -f1,6` is right because the text is the last field and has no raw tabs.) Rust
1.99.1, release profile (fat LTO), on the shared 64-core host.
