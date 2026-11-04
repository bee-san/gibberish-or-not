"""Shared tooling for building gibberish-or-not language packs (see docs/languages.md).

A language's build script (languages/<code>/build.py) reads its training sources from the
data cache, counts letter n-grams and words with the helpers here, and calls write_model()
to produce languages/<code>/model.bin. `gib-bench fit --lang <code>` then writes
languages/<code>/calibration.txt.

Everything is deterministic: no dict/set iteration order reaches the output, and the only
randomness (cuckoo eviction in the word table) uses random.Random with a fixed seed.

Standard library only. The hash functions must stay identical to src/pack.rs.
"""

import collections
import math
import os
import random
import re
import struct
import unicodedata

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

SCRIPT = {"latin": 1, "cyrillic": 2, "han": 4, "kana": 8, "hangul": 16, "greek": 32, "other": 64}


def fnv1a(data, h=FNV_OFFSET):
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


def mix(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def word_hash(word):
    """The hash the library looks a normalised word up by."""
    return mix(fnv1a(word.encode("utf-8")))


def reduce32(x, n):
    return ((x & 0xFFFFFFFF) * n) >> 32


# --------------------------------------------------------------------- text helpers --

WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.U)
COMBINING = re.compile("[\u0300-\u036f\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20ff\ufe20-\ufe2f]")


def normalise_word(word, case_overrides=None):
    """Lower case (with the pack's overrides first), ’ as ', no combining marks: the form
    the library looks words up in."""
    out = []
    for c in unicodedata.normalize("NFC", word):
        if COMBINING.match(c):
            continue
        if c == "\u2019":
            out.append("'")
            continue
        if case_overrides and c in case_overrides:
            out.append(case_overrides[c])
            continue
        lo = c.lower()
        out.append(lo[0] if lo else c)
    return "".join(out)


def words_of(text, case_overrides=None):
    """The word tokens the library would look up: runs of letters with in-word apostrophes,
    in chunks without digits."""
    out = []
    for chunk in text.split():
        if any(ch.isdigit() for ch in chunk) or "://" in chunk or "@" in chunk:
            continue
        for w in WORD_RE.findall(chunk):
            out.append(normalise_word(w, case_overrides))
    return out


def gutenberg_body(path):
    """A Project Gutenberg book without its header and licence footer."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read().replace("\r\n", "\n")
    start = re.search(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", text, re.M | re.I)
    if start:
        text = text[start.end():]
    end = re.search(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", text, re.M | re.I)
    if end:
        text = text[:end.start()]
    return unicodedata.normalize("NFC", text)


def tatoeba_split(user):
    """The split of a Tatoeba contributor: the same function as bench/scripts/make_corpus.py."""
    import hashlib

    h = hashlib.sha256(("gon-v1|tatoeba-user|" + user).encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / 2.0 ** 64
    return "train" if u < 0.6 else ("dev" if u < 0.8 else "test")


def tatoeba_train_texts(cache, iso3):
    """Sentences of the Tatoeba contributors in the train split."""
    import bz2

    path = os.path.join(cache, "tatoeba-20261031", f"{iso3}_sentences_detailed.tsv.bz2")
    out = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            user = parts[3] if parts[3] != "\\N" else "-"
            if tatoeba_split(user) == "train":
                out.append(unicodedata.normalize("NFC", parts[2]))
    return out


# ------------------------------------------------------------------------- n-grams --


class Alphabet:
    """Characters -> symbols 0..size-1 for a dense n-gram model. Several characters may
    share a symbol (folding); upper case is found through lower-casing at run time."""

    def __init__(self, groups):
        self.map = {}
        for sym, chars in enumerate(groups):
            for c in chars:
                self.map[c] = sym
        self.size = len(groups)

    def stream(self, text, case_overrides=None):
        """Symbol runs of the letters of `text` (other characters dropped, letters outside
        the alphabet split the runs), the way the library scores a text."""
        runs, cur = [], []
        for chunk in text.split():
            if any(ch.isdigit() for ch in chunk) or "://" in chunk or "@" in chunk:
                if cur:
                    runs.append(cur)
                    cur = []
                continue
            for c in chunk:
                if not c.isalpha():
                    continue
                c2 = case_overrides.get(c, c) if case_overrides else c
                s = self.map.get(c2)
                if s is None:
                    lo = c2.lower()
                    s = self.map.get(lo[0] if lo else c2)
                if s is None:
                    if cur:
                        runs.append(cur)
                        cur = []
                    continue
                cur.append(s)
        if cur:
            runs.append(cur)
        return runs


def count_ngrams(runs, order, counts=None):
    """Counts of every n-gram of length 1..order in the symbol runs, as tuples."""
    counts = counts if counts is not None else [collections.Counter() for _ in range(order + 1)]
    for run in runs:
        t = tuple(run)
        for k in range(1, order + 1):
            c = counts[k]
            for i in range(len(t) - k + 1):
                c[t[i:i + k]] += 1
    return counts


def dense_tables(counts, size, order, discount=0.75, scale=0.04):
    """Interpolated absolute-discounting estimates P_k(w | h) for k = 1..order over an
    alphabet of `size` symbols, each table a bytes object of size^k quantised values
    q = round(-log10 P / scale), clamped to 255. Unseen contexts back off to the shorter
    context; the base is uniform."""
    tables = []
    prev = [1.0 / size] * size  # P_{k-1}, indexed by the (k-1)-gram index
    for k in range(1, order + 1):
        ck = counts[k]
        ctx_total = collections.Counter()
        ctx_types = collections.Counter()
        for g, n in ck.items():
            ctx_total[g[:-1]] += n
            ctx_types[g[:-1]] += 1
        cur = [0.0] * (size ** k)
        lower_size = size ** (k - 1)
        for idx in range(size ** k):
            # digits of idx in base size: the n-gram
            ctx_idx, w = divmod(idx, size)
            lower_idx = ctx_idx % (size ** (k - 2)) * size + w if k >= 2 else w
            p_lower = prev[lower_idx] if k >= 2 else 1.0 / size
            cur[idx] = p_lower
        # Contexts that were seen: discounted counts plus the redistributed mass
        for ctx, tot in ctx_total.items():
            ctx_idx = 0
            for s in ctx:
                ctx_idx = ctx_idx * size + s
            gamma = discount * ctx_types[ctx] / tot
            base = ctx_idx * size
            for w in range(size):
                lower_idx = (ctx_idx % (size ** (k - 2)) * size + w) if k >= 2 else w
                p_lower = prev[lower_idx] if k >= 2 else 1.0 / size
                n = ck.get(ctx + (w,), 0)
                cur[base + w] = max(n - discount, 0) / tot + gamma * p_lower
        tables.append(bytes(min(255, max(0, round(-math.log10(max(p, 1e-30)) / scale))) for p in cur))
        prev = cur
        del lower_size
    return tables


def hashed_tables(unit_counts, order, bits, discount=0.75, scale=0.04, min_count=2, backoff=-1.0):
    """For big alphabets (CJK): units are code points and unit_counts[k] is a Counter of
    k-tuples of them. Keeps the most frequent k-grams (count >= min_count) that fit
    2^bits[k] slots at <= 70% load, each with an 8-bit fingerprint; a slot two k-grams hash
    to keeps the more frequent. On a miss the reader backs off one order and adds
    `backoff` (log10). Returns (tables, floor): tables[k-1] = (bits, backoff, slot bytes),
    floor = log10 P of a unit never seen."""
    total_units = sum(unit_counts[1].values())
    floor = math.log10(0.5 / (total_units + len(unit_counts[1]) + 1))
    tables, prev_p = [], {}
    for k in range(1, order + 1):
        ck = unit_counts[k]
        ctx_total, ctx_types = collections.Counter(), collections.Counter()
        for g, n in ck.items():
            ctx_total[g[:-1]] += n
            ctx_types[g[:-1]] += 1
        cap = int((1 << bits[k - 1]) * 0.7)
        keep = sorted((g for g, n in ck.items() if n >= min_count), key=lambda g: (-ck[g], g))[:cap]
        slots = [0] * (1 << bits[k - 1])
        cur_p = {}
        for g in keep:
            tot = ctx_total[g[:-1]]
            if k == 1:
                p = ck[g] / tot
            else:
                lower = prev_p.get(g[1:], 10 ** floor)
                p = max(ck[g] - discount, 0) / tot + discount * ctx_types[g[:-1]] / tot * lower
            cur_p[g] = p
            h = FNV_OFFSET
            for u in g:
                h = fnv1a(struct.pack("<I", u), h)
            h = mix((h + k) & MASK64)
            slot = h & ((1 << bits[k - 1]) - 1)
            if slots[slot] == 0:
                q = min(255, max(1, round(-math.log10(max(p, 1e-30)) / scale)))
                slots[slot] = ((h >> 56) << 8) | q
        tables.append((bits[k - 1], backoff, b"".join(struct.pack("<H", v) for v in slots)))
        prev_p = cur_p
    return tables, floor


# --------------------------------------------------------------------------- words --


def word_classes(counts, total, max_words):
    """The most frequent words (up to max_words) and their frequency class 1..15: class c
    stands for log10 P = -1.0 - 0.5 * (c - 1), the nearest to the word's log10 P."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words]
    out = {}
    for w, n in ranked:
        lp = math.log10(n / total)
        c = int(round((-1.0 - lp) / 0.5)) + 1
        out[w] = min(15, max(1, c))
    return out


CLASS_LOGP = [0.0] + [-1.0 - 0.5 * (c - 1) for c in range(1, 16)]


def word_buckets(h, n_buckets):
    b1 = reduce32(h, n_buckets)
    b2 = reduce32(h >> 32, n_buckets)
    if b2 == b1:
        b2 = (b1 + 1) % n_buckets
    return b1, b2


def word_table(classes, load=0.92, seed=0):
    """A two-choice table of 8-slot buckets: slot = fingerprint (28 bits) << 4 | class.
    Entries are packed at the start of their bucket (the reader stops at an empty slot);
    a word goes to the emptier of its two buckets, and when both are full a random entry
    is moved to its other bucket (cuckoo hashing)."""
    words = sorted(classes)
    n_buckets = max(1, math.ceil(len(words) / (8 * load)))
    while True:
        buckets = [[] for _ in range(n_buckets)]
        rng = random.Random(seed)
        ok = True
        for w in words:
            h = word_hash(w)
            entry = ((mix(h ^ 0x5BD1E995) >> 36) << 4) | classes[w]
            b1, b2 = word_buckets(h, n_buckets)
            cur = (entry, b1, b2)
            placed = False
            for _ in range(1000):
                e, x1, x2 = cur
                if len(buckets[x1]) < 8 and (len(buckets[x1]) <= len(buckets[x2]) or len(buckets[x2]) >= 8):
                    buckets[x1].append(cur)
                    placed = True
                    break
                if len(buckets[x2]) < 8:
                    buckets[x2].append(cur)
                    placed = True
                    break
                tgt = x1 if rng.random() < 0.5 else x2
                j = rng.randrange(8)
                victim = buckets[tgt][j]
                buckets[tgt][j] = cur
                ve, v1, v2 = victim
                other = v2 if v1 == tgt else v1
                cur = (ve, other, tgt)
            if not placed:
                ok = False
                break
        if ok:
            break
        n_buckets = int(n_buckets * 1.05) + 1
    out = bytearray()
    for bucket in buckets:
        entries = [e for e, _, _ in bucket] + [0] * (8 - len(bucket))
        for e in entries:
            out += struct.pack("<I", e)
    return n_buckets, bytes(out)


# --------------------------------------------------------------------------- write --


def write_model(path, lang, scripts, params, alphabet=None, dense=None, hashed=None, words=None,
                case_overrides=None, flags=0):
    """Writes model.bin.

    * lang: two-letter code; scripts: list of SCRIPT names;
    * params: dict with lambda (log10 per unknown letter), max_word_chars, seg_min_letters;
    * alphabet + dense: an Alphabet and dense_tables() output (or hashed: (tables, floor)
      from hashed_tables());
    * words: {normalised word: class} from word_classes(), or None;
    * case_overrides: {upper: lower} applied before Unicode lower-casing (Turkish I -> ı).
    """
    sections = []
    sections.append((b"PARM", struct.pack("<fHH", params["lambda"], params["max_word_chars"], params["seg_min_letters"])))
    if case_overrides:
        sections.append((b"CASE", b"".join(struct.pack("<II", ord(a), ord(b)) for a, b in sorted(case_overrides.items()))))
    if alphabet is not None:
        entries = sorted((ord(c), s) for c, s in alphabet.map.items())
        sections.append((b"ALPH", struct.pack("<HH", alphabet.size, len(entries)) + b"".join(struct.pack("<IHH", c, s, 0) for c, s in entries)))
    if dense is not None:
        body = struct.pack("<BBHff", 0, len(dense), 0, 0.04, 0.0) + b"".join(dense)
        sections.append((b"NGRM", body))
    elif hashed is not None:
        tables, floor = hashed
        body = struct.pack("<BBHff", 1, len(tables), 0, 0.04, floor)
        for bits, backoff, data in tables:
            body += struct.pack("<BBBBf", bits, 0, 0, 0, backoff) + data
        sections.append((b"NGRM", body))
    else:
        raise ValueError("a model needs dense or hashed n-grams")
    if words:
        n_buckets, data = word_table(words)
        head = struct.pack("<II", n_buckets, len(words)) + b"".join(struct.pack("<f", x) for x in CLASS_LOGP)
        sections.append((b"WORD", head + data))
    script_bits = 0
    for s in scripts:
        script_bits |= SCRIPT[s]
    header_len = 16 + 12 * len(sections)
    offset = (header_len + 7) & ~7
    table, blobs = b"", b""
    pad0 = b"\0" * (offset - header_len)
    for tag, body in sections:
        table += tag + struct.pack("<II", offset, len(body))
        padded = body + b"\0" * ((-len(body)) % 8)
        blobs += padded
        offset += len(padded)
    header = b"GONM" + struct.pack("<H", 1) + lang.encode("ascii") + bytes([script_bits, flags]) + struct.pack("<HI", len(sections), 0)
    data = header + table + pad0 + blobs
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def pack_blob(model_path, calibration_path):
    """The single-blob form (Pack::from_bytes): GONP, model length, model, calibration."""
    model = open(model_path, "rb").read()
    cal = open(calibration_path, "rb").read()
    return b"GONP" + struct.pack("<I", len(model)) + model + cal


def _self_test():
    # Also asserted in src/pack.rs (hashes_match_packlib)
    assert fnv1a(b"hello") == 0xA430D84680AABD0B, hex(fnv1a(b"hello"))
    assert mix(0x123456789ABCDEF0) == 0x9629F58E8EC5B906


if __name__ == "__main__":
    _self_test()
