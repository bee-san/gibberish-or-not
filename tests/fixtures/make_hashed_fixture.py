#!/usr/bin/env python3
"""Writes tests/fixtures/hashed-za.bin: a tiny pack with hashed n-grams over Han characters
and a few words, to test the hashed-model and CJK paths of the library (tests/hashed_pack.rs).
Language code `za` and the text below (written for this test) are arbitrary. Deterministic.

    python3 tests/fixtures/make_hashed_fixture.py
"""
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "languages", "common"))
import packlib  # noqa: E402

TEXT = "我们明天见。今天天气很好。我们去公园散步吧。他明天去学校。你们今天去哪里？" * 20
WORDS = {"我们": 3, "明天": 4, "今天": 4, "天气": 5, "很好": 5, "公园": 6, "散步": 7, "学校": 6, "你们": 4, "哪里": 5, "去": 3, "见": 5, "他": 3}


def main():
    units = [ord(c) for c in TEXT if c.isalpha()]
    counts = [collections.Counter() for _ in range(4)]
    for k in range(1, 4):
        for i in range(len(units) - k + 1):
            counts[k][tuple(units[i:i + k])] += 1
    hashed = packlib.hashed_tables(counts, 3, [10, 10, 10], min_count=1)
    out = os.path.join(HERE, "hashed-za.bin")
    params = {"lambda": -4.0, "max_word_chars": 4, "seg_min_letters": 2, "rank_offset": 2.0}
    n = packlib.write_model(out, "za", ["han"], params, hashed=hashed, words=WORDS)
    print(f"wrote {out} ({n} bytes)")


if __name__ == "__main__":
    main()
