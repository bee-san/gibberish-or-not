#!/usr/bin/env python3
"""Builds languages/en/model.bin from the data cache. Deterministic.

    python3 languages/en/build.py [--cache DIR]

Training sources (train split only, see bench/sources/sources.json and README.md here):

* the 85 public-domain Project Gutenberg books Ciphey counts its quadgram table from
  (`gutenberg_books.en_train`) and 299 more English books picked from the catalog with a
  fixed seed (`gutenberg_books.en_train_extra`), none of them a dev or test book;
* the Tatoeba English sentences of the contributors in the train split (CC BY 2.0 FR),
  a fixed-seed sample of TATOEBA_SENTENCES.

Model:

* letters a-z (any other letter is outside the alphabet), dense 4-gram tables of
  interpolated absolute-discounting estimates P(c | previous 3 letters), over the letters of
  each text with everything else removed (word boundaries too, so space-less text scores
  like spaced text);
* the most frequent words seen at least 3 times (in 2 sources, or 10 times in one), lower
  case, apostrophes kept (`don't`), single letters only `a` and `i`, Roman numerals only
  when mostly written in lower case, in frequency classes of half a log10 step; then words
  of Grady Ward's public-domain Moby crossword list (Gutenberg ebook 3201) the books don't
  have, in the rarest class, up to MAX_WORDS in all.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "languages", "common"))
import packlib  # noqa: E402

DEFAULT_CACHE = "/local/home/skerraut/herd-synth3/shared/ciphey/data"
TATOEBA_SENTENCES = 300_000
MAX_WORDS = 115_000
RARE_CLASS = 14  # log10 P = -7.5: words only the spelling list knows
PARAMS = {"lambda": -2.0, "max_word_chars": 16, "seg_min_letters": 6}
ROMAN = re.compile(r"^[ivxlcdm]+$")
TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.U)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GON_DATA_DIR", DEFAULT_CACHE))
    args = ap.parse_args()
    with open(os.path.join(ROOT, "bench", "sources", "sources.json"), encoding="utf-8") as f:
        sources = json.load(f)
    books = sources["gutenberg_books"]["en_train"] + sources["gutenberg_books"]["en_train_extra"]

    texts = []  # (source, text)
    inputs = []
    for n in books:
        rel = f"gutenberg/pg{n}.txt"
        texts.append((f"gutenberg:{n}", packlib.gutenberg_body(os.path.join(args.cache, rel))))
        inputs.append((rel, sources["files"][rel]["sha256"]))
    tat = packlib.tatoeba_train_texts(args.cache, "eng")
    rng = random.Random("gon-en-tatoeba")
    tat = rng.sample(tat, min(TATOEBA_SENTENCES, len(tat)))
    for i, t in enumerate(tat):
        # Tatoeba sentences as pseudo-sources of 10,000 sentences each
        texts.append((f"tatoeba:{i // 10_000}", t))
    rel = "tatoeba-20261031/eng_sentences_detailed.tsv.bz2"
    inputs.append((rel, sources["files"][rel]["sha256"]))

    alphabet = packlib.Alphabet(list("abcdefghijklmnopqrstuvwxyz"))
    counts = [collections.Counter() for _ in range(5)]
    letters = 0
    for _, t in texts:
        for run in alphabet.stream(t):
            s = "".join(chr(97 + x) for x in run)
            letters += len(s)
            counts[1].update(s)
            counts[2].update(zip(s, s[1:]))
            counts[3].update(zip(s, s[1:], s[2:]))
            counts[4].update(zip(s, s[1:], s[2:], s[3:]))
    # Keys as symbol tuples
    sym = {chr(97 + i): i for i in range(26)}
    tup = [collections.Counter()]
    for k in range(1, 5):
        c = collections.Counter()
        for g, n in counts[k].items():
            key = (sym[g],) if k == 1 else tuple(sym[x] for x in g)
            c[key] = n
        tup.append(c)
    dense = packlib.dense_tables(tup, 26, 4)

    words = collections.Counter()
    lower = collections.Counter()
    seen_in = collections.defaultdict(set)
    tokens = 0
    for source, t in texts:
        for chunk in t.split():
            if any(ch.isdigit() for ch in chunk) or "://" in chunk or "@" in chunk:
                continue
            for raw in TOKEN.findall(chunk):
                w = packlib.normalise_word(raw)
                words[w] += 1
                tokens += 1
                if raw == w:
                    lower[w] += 1
                if len(seen_in[w]) < 2:
                    seen_in[w].add(source)
    keep = {}
    for w, n in words.items():
        if n < 3 or (len(seen_in[w]) < 2 and n < 10):
            continue
        if len(w) == 1 and w not in ("a", "i"):
            continue
        if ROMAN.match(w) and len(w) >= 2 and lower[w] < 0.3 * n:
            continue
        keep[w] = n
    classes = packlib.word_classes(keep, tokens, MAX_WORDS)
    rel = "gutenberg/3201/crosswd.txt"
    inputs.append((rel, sources["files"][rel]["sha256"]))
    with open(os.path.join(args.cache, rel), encoding="latin-1") as f:
        spelling = sorted({w.strip().lower() for w in f if w.strip().isalpha() and w.strip().isascii()})
    for w in spelling:
        if len(classes) >= MAX_WORDS:
            break
        if w not in classes and len(w) > 1:
            classes[w] = RARE_CLASS

    out = os.path.join(HERE, "model.bin")
    size = packlib.write_model(out, "en", ["latin"], PARAMS, alphabet=alphabet, dense=dense, words=classes)
    digest = hashlib.sha256(open(out, "rb").read()).hexdigest()
    info = {
        "inputs": inputs,
        "tatoeba_sentences": len(tat),
        "letters": letters,
        "tokens": tokens,
        "distinct_words": len(words),
        "words_kept": len(classes),
        "words_from_books": sum(1 for c in classes.values() if c != RARE_CLASS),
        "model_bytes": size,
        "model_sha256": digest,
        "params": PARAMS,
    }
    with open(os.path.join(HERE, "build.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in info.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
