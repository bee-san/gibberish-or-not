#!/usr/bin/env python3
"""Builds the frozen benchmark corpus bench/corpus/v1/ from the data cache and Ciphey's data.

    python3 bench/scripts/fetch_sources.py      # once: fills the data cache
    python3 bench/scripts/make_corpus.py        # writes bench/corpus/v1/{train,dev,test}.tsv

Deterministic: every random choice comes from random.Random(<fixed string>), inputs are
read in sorted order, and the output is sorted. bench/corpus/v1/MANIFEST.sha256 records the
sha256 of every input and output file; `--check` rebuilds into a temporary directory and
compares. v1 is frozen: change it only by making v2.

Each output line is `id<TAB>lang<TAB>label<TAB>category<TAB>source<TAB>text`:

* id: the first 12 hex digits of sha256(text). Texts are unique across the corpus.
* lang: the language of a positive, or the language a negative was derived from; `-` for
  noise that comes from no language.
* label: `pos` (natural-language text), `neg` (not language: junk), `info` (reported but
  counted in no total: identifiers on their own, code, JSON, passwords, other languages).
* category: what the text is (see bench/corpus/v1/README.md).
* source: where it came from. One source (a book, a Tatoeba contributor, a Ciphey search, a
  Wikipedia language edition) is in exactly one split, and every text derived from a sample
  (variants, ciphered junk) is in that sample's split.
* text: with \\t, \\n, \\r and \\\\ escaped.

Splits: train (fitting models and classifiers), dev (thresholds and calibration), test
(reporting, touched once per released model version). Sources, licences and sha256s are in
bench/sources/sources.json and bench/corpus/v1/README.md.

Only the Python standard library is needed.
"""

import argparse
import base64
import bz2
import collections
import difflib
import hashlib
import html
import json
import math
import os
import random
import re
import sys
import tempfile
import unicodedata
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERSION = "v1"
OUT_DIR = os.path.join(ROOT, "bench", "corpus", VERSION)
CIPHEY_DIR = os.path.join(OUT_DIR, "ciphey")
SOURCES_JSON = os.path.join(ROOT, "bench", "sources", "sources.json")
DEFAULT_CACHE = "/local/home/skerraut/herd-synth3/shared/ciphey/data"

LANGS = ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "tr", "ru", "zh", "ja", "ko"]
ISO3 = {"en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por", "nl": "nld",
        "pl": "pol", "tr": "tur", "ru": "rus", "zh": "cmn", "ja": "jpn", "ko": "kor"}
LATIN = {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "tr"}
CYRILLIC = {"ru"}
CJK = {"zh", "ja"}  # no spaces between words
HANGUL = {"ko"}

EN_TRAIN_BOOKS = [11, 1661, 2701, 1400, 345, 120, 1260, 2852, 174, 5230]
EN_DEV_BOOKS = [4300, 10007, 69087]
EN_TEST_BOOKS = [1342, 161, 158, 98, 64317, 1250, 26184, 67098]

# How many base samples to draw per (split, source kind). Short phrases and single words are
# cut from the drawn sentences.
SIZES = {
    "en": {"train": {"tatoeba": 400, "book": 30, "para": 3, "wiki": 0, "phrase": 250, "word": 150, "mixed": 60, "short": 600},
           "dev": {"tatoeba": 600, "book": 100, "para": 10, "wiki": 0, "phrase": 400, "word": 250, "mixed": 120, "short": 400},
           "test": {"tatoeba": 600, "book": 50, "para": 5, "wiki": 300, "phrase": 400, "word": 250, "mixed": 120, "short": 400}},
    "other": {"train": {"tatoeba": 200, "book": 10, "para": 2, "wiki": 0, "phrase": 100, "word": 60, "mixed": 20, "short": 200},
              "dev": {"tatoeba": 300, "book": 25, "para": 4, "wiki": 0, "phrase": 150, "word": 100, "mixed": 40, "short": 150},
              "test": {"tatoeba": 300, "book": 25, "para": 4, "wiki": 150, "phrase": 150, "word": 100, "mixed": 40, "short": 150}},
}
# Non-English bases get this many of their applicable junk generators (all for English).
OTHER_JUNK_PER_BASE = 5

# Ciphey's own other-language samples (curated.tsv, non_english), by language.
CIPHEY_NON_ENGLISH = {
    "Le trésor est caché sous le vieux chêne": "fr", "Nous partirons à l'aube": "fr",
    "Ich habe keine Ahnung": "de", "No entiendo nada": "es", "Ciao, come stai?": "it",
    "Ci vediamo domani sera": "it", "Olá, tudo bem?": "pt", "A senha é segredo": "pt",
    "Goedemorgen, hoe gaat het?": "nl", "Ik hou van je": "nl", "Dzień dobry, jak się masz?": "pl",
    "Привет, как дела?": "ru", "今日はいい天気ですね": "ja", "你好，你好吗？": "zh", "我们明天见": "zh",
    "안녕하세요": "ko", "Bonjour, comment allez-vous aujourd'hui ?": "fr", "Le chat est sur la table": "fr",
    "Je ne sais pas ce que tu veux dire": "fr", "Rendez-vous demain matin devant la gare": "fr",
    "Guten Morgen, wie geht es dir?": "de", "Ich liebe dich": "de", "Das Wetter ist heute sehr schön": "de",
    "Wir treffen uns morgen am Bahnhof": "de", "Hola, ¿cómo estás?": "es", "La casa es muy grande": "es",
    "Nos vemos mañana en la estación": "es", "El tesoro está debajo del árbol": "es",
    "Il fait très beau aujourd'hui": "fr", "Où est la bibliothèque ?": "fr",
    "Der Schlüssel liegt unter der Matte": "de", "Buenas noches a todos": "es",
    "La chiave è sotto il tappeto": "it", "Obrigado pela ajuda": "pt", "Merhaba, nasılsın?": "tr",
    "Встретимся завтра у вокзала": "ru", "こんにちは、お元気ですか": "ja",
}
# Ciphey's curated categories and what they are here.
CIPHEY_CATEGORY = {
    "en_word": ("pos", "word"), "en_phrase": ("pos", "phrase"), "en_sentence": ("pos", "sentence"),
    "en_chat": ("pos", "chat"), "en_technical": ("pos", "technical"), "en_formal": ("pos", "formal"),
    "en_ctf": ("pos", "ctf_text"), "en_paragraph": ("pos", "paragraph"), "en_verse": ("pos", "verse"),
    "en_literary": ("pos", "literary"), "en_nospace": ("pos", "nospace"),
    # Identifiers on their own are pyWhat's, not language (docs/design.md, the boundary)
    "ctf_flag": ("info", "ctf_flag"), "json": ("info", "json"), "code": ("info", "code"),
    "url": ("info", "url"), "email": ("info", "email"), "ip": ("info", "ip"),
    "lemmeknow": ("info", "identifier"), "lemmeknow_ignored": ("info", "identifier"),
    "password": ("info", "password"), "hash": ("neg", "hash"),
}

# --------------------------------------------------------------------------------- utils --


def rng_for(*parts):
    # Fixed seeds: the corpus must be reproducible byte for byte; nothing here is security
    return random.Random("gon-v1|" + "|".join(str(p) for p in parts))  # NOSONAR


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_unit(*parts):
    """A number in [0, 1) from a hash of parts, for split assignment."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def escape(text):
    """TSV escapes: \\t \\n \\r \\\\, \\xHH for the other C0 and C1 control characters and DEL,
    \\u2028 and \\u2029 (line breaks to some tools). Everything else is literal UTF-8."""
    out = []
    for c in text:
        o = ord(c)
        if c == "\\":
            out.append("\\\\")
        elif c == "\t":
            out.append("\\t")
        elif c == "\n":
            out.append("\\n")
        elif c == "\r":
            out.append("\\r")
        elif o < 0x20 or o == 0x7F or 0x80 <= o <= 0x9F:
            out.append(f"\\x{o:02X}")
        elif o in (0x2028, 0x2029):
            out.append(f"\\u{o:04X}")
        else:
            out.append(c)
    return "".join(out)


def unescape(field):
    out, i = [], 0
    while i < len(field):
        c = field[i]
        if c == "\\" and i + 1 < len(field):
            n = field[i + 1]
            if n == "x" and i + 3 < len(field) + 0 and re.fullmatch(r"[0-9A-Fa-f]{2}", field[i + 2:i + 4] or ""):
                out.append(chr(int(field[i + 2:i + 4], 16)))
                i += 4
                continue
            if n == "u" and re.fullmatch(r"[0-9A-Fa-f]{4}", field[i + 2:i + 6] or ""):
                out.append(chr(int(field[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(n, "\\" + n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def letters_of(s):
    return sum(1 for c in s if c.isalpha())


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------------ sources --


def inside(root, rel):
    """`rel` (a path from this script's own lists or the sources manifest) joined to `root`,
    refusing anything that resolves outside `root` (an absolute path, `..`)."""
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(base, rel))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"{rel!r} is outside {base}")
    return path


def gutenberg_body(path):
    """A Project Gutenberg book without its header and licence footer."""
    with open(path, encoding="utf-8", errors="replace") as f:  # NOSONAR - callers pass inside(cache, ...) paths
        text = f.read().replace("\r\n", "\n")
    start = re.search(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*$", text, re.M | re.I)
    if start:
        text = text[start.end():]
    end = re.search(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*$", text, re.M | re.I)
    if end:
        text = text[:end.start()]
    return text


SENT_END = re.compile(r"(?<=[.!?。！？])[\"'”’»)]*\s+")
CJK_SENT_END = re.compile(r"(?<=[。！？!?])")


def book_paragraphs(path):
    body = gutenberg_body(path)
    paras = []
    for block in re.split(r"\n\s*\n", body):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        p = norm_ws(" ".join(lines)).replace("_", "")
        # Headings, tables of contents, illustrations, footnotes, verse numbers
        if len(p) < 40 or p.isupper() or "[" in p or "]" in p or re.search(r"\d{3,}", p):
            continue
        paras.append(p)
    return paras


def split_sentences(p, lang):
    if lang in CJK:
        parts = [s.strip() for s in CJK_SENT_END.split(p)]
    else:
        parts = [s.strip() for s in SENT_END.split(p)]
    return [s for s in parts if s]


def good_sentence(s, lang):
    if lang in CJK:
        n = len(s)
        return 6 <= n <= 80 and letters_of(s) >= 0.7 * n - 3
    words = s.split()
    return 4 <= len(words) <= 30 and 15 <= len(s) <= 260 and letters_of(s) >= 0.6 * len(s)


def tatoeba_rows(cache, lang):
    path = inside(cache, os.path.join("tatoeba-20261031", f"{ISO3[lang]}_sentences_detailed.tsv.bz2"))
    rows = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            sid, _, text, user = parts[0], parts[1], parts[2], parts[3]
            text = norm_ws(text)
            if text:
                rows.append((int(sid), text, user if user != "\\N" else "-"))
    rows.sort()
    return rows


def tatoeba_split(user):
    """Contributors are the sources: 60% train, 20% dev, 20% test by a hash of the name."""
    u = stable_unit("tatoeba-user", user)
    return "train" if u < 0.6 else ("dev" if u < 0.8 else "test")


def book_split(lang, n, books):
    if lang == "en":
        if n in EN_DEV_BOOKS:
            return "dev"
        if n in EN_TEST_BOOKS:
            return "test"
        return "train"
    # Other languages: the books fetch_sources.py picked, in id order: 60/20/20
    i = books.index(n)
    k = len(books)
    if k < 3:
        return "train"
    n_dev = max(1, round(k * 0.2))
    n_test = max(1, round(k * 0.2))
    return "train" if i < k - n_dev - n_test else ("dev" if i < k - n_test else "test")


def wiki_rows(cache, lang):
    path = inside(cache, os.path.join("wikipedia-intros-20261104", f"{lang}.jsonl"))
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:  # NOSONAR - path = inside(cache, ...)
        return [json.loads(l) for l in f if l.strip()]


# -------------------------------------------------------------------------- transforms --

LOWER = "abcdefghijklmnopqrstuvwxyz"
CYR = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"


def caesar(s, k):
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - 97 + k) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - 65 + k) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


def caesar_cyr(s, k):
    out = []
    for c in s:
        lo = c.lower()
        if lo in CYR:
            r = CYR[(CYR.index(lo) + k) % len(CYR)]
            out.append(r.upper() if c != lo else r)
        else:
            out.append(c)
    return "".join(out)


def atbash(s):
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(chr(219 - ord(c)))
        elif "A" <= c <= "Z":
            out.append(chr(155 - ord(c)))
        else:
            out.append(c)
    return "".join(out)


def atbash_cyr(s):
    out = []
    for c in s:
        lo = c.lower()
        if lo in CYR:
            r = CYR[len(CYR) - 1 - CYR.index(lo)]
            out.append(r.upper() if c != lo else r)
        else:
            out.append(c)
    return "".join(out)


def rot47(s):
    return "".join(chr((ord(c) - 33 + 47) % 94 + 33) if 33 <= ord(c) <= 126 else c for c in s)


def affine(s, a, b):
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(chr((a * (ord(c) - 97) + b) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr((a * (ord(c) - 65) + b) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


def substitute(s, perm):
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(perm[ord(c) - 97])
        elif "A" <= c <= "Z":
            out.append(perm[ord(c) - 65].upper())
        else:
            out.append(c)
    return "".join(out)


def substitute_cyr(s, perm):
    out = []
    for c in s:
        lo = c.lower()
        if lo in CYR:
            r = perm[CYR.index(lo)]
            out.append(r.upper() if c != lo else r)
        else:
            out.append(c)
    return "".join(out)


def vigenere(s, key, decrypt):
    k = [ord(c) - 65 for c in key]
    out, i = [], 0
    for c in s:
        if "a" <= c <= "z" or "A" <= c <= "Z":
            base = 65 if c <= "Z" else 97
            shift = (26 - k[i % len(k)]) if decrypt else k[i % len(k)]
            out.append(chr((ord(c) - base + shift) % 26 + base))
            i += 1
        else:
            out.append(c)
    return "".join(out)


def vigenere_cyr(s, key, decrypt):
    out, i = [], 0
    n = len(CYR)
    for c in s:
        lo = c.lower()
        if lo in CYR:
            shift = key[i % len(key)]
            r = CYR[(CYR.index(lo) + (n - shift if decrypt else shift)) % n]
            out.append(r.upper() if c != lo else r)
            i += 1
        else:
            out.append(c)
    return "".join(out)


def zigzag(rails, n, offset=0):
    cycle = list(range(rails - 1)) + list(range(rails - 1, 0, -1))
    return [cycle[(i + offset) % len(cycle)] for i in range(n)]


def rail_encrypt(text, rails):
    chars = list(text)
    z = zigzag(rails, len(chars))
    order = sorted(range(len(chars)), key=lambda i: (z[i], i))
    return "".join(chars[i] for i in order)


def rail_decrypt(text, rails, offset=0):
    n = len(text)
    z = zigzag(rails, n, offset)
    order = sorted(range(n), key=lambda i: (z[i], i))
    out = [""] * n
    for c, pos in zip(text, order):
        out[pos] = c
    return "".join(out)


def columnar_encrypt(text, key):
    cols = len(key)
    order = sorted(range(cols), key=lambda i: (key[i], i))
    return "".join(text[c::cols] for c in order)


def columnar_decrypt(text, key):
    n, cols = len(text), len(key)
    order = sorted(range(cols), key=lambda i: (key[i], i))
    rows = -(-n // cols)
    lengths = [len(range(c, n, cols)) for c in range(cols)]
    out_cols, pos = [None] * cols, 0
    for c in order:
        out_cols[c] = text[pos:pos + lengths[c]]
        pos += lengths[c]
    out = []
    for r in range(rows):
        for c in range(cols):
            if r < len(out_cols[c]):
                out.append(out_cols[c][r])
    return "".join(out)


MORSE = {"A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.",
         "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.",
         "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-",
         "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..", "0": "-----",
         "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....",
         "7": "--...", "8": "---..", "9": "----."}
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def base58(b):
    n = int.from_bytes(b, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + out


ENCODINGS = ["base64", "base32", "hex", "hex_spaced", "binary", "decimal", "octal", "url", "morse",
             "a1z26", "html_entities", "unicode_escapes", "base58", "base85"]


def encode(text, enc):
    b = text.encode("utf-8")
    if enc == "base64":
        return base64.b64encode(b).decode()
    if enc == "base32":
        return base64.b32encode(b).decode()
    if enc == "hex":
        return b.hex()
    if enc == "hex_spaced":
        return " ".join(f"{x:02X}" for x in b)
    if enc == "binary":
        return " ".join(f"{x:08b}" for x in b)
    if enc == "decimal":
        return " ".join(str(x) for x in b)
    if enc == "octal":
        return " ".join(f"{x:o}" for x in b)
    if enc == "url":
        return urllib.parse.quote(text)
    if enc == "morse":
        words = []
        for w in text.split():
            m = " ".join(MORSE[c] for c in w.upper() if c in MORSE)
            if m:
                words.append(m)
        return " / ".join(words)
    if enc == "a1z26":
        words = []
        for w in text.split():
            m = "-".join(str(ord(c) - 96) for c in w.lower() if "a" <= c <= "z")
            if m:
                words.append(m)
        return " ".join(words)
    if enc == "html_entities":
        return "".join(f"&#{ord(c)};" for c in text)
    if enc == "unicode_escapes":
        return "".join(f"\\u{ord(c):04X}" for c in text)
    if enc == "base58":
        return base58(b)
    if enc == "base85":
        return base64.b85encode(b).decode()
    raise ValueError(enc)


def mojibake(text, kind):
    b = text.encode("utf-8")
    if kind == "utf8_as_latin1":
        return b.decode("latin-1")
    if kind == "utf8_as_cp1252":
        return b.decode("cp1252", errors="replace")
    if kind == "utf8_as_gbk":
        return b.decode("gbk", errors="replace")
    if kind == "utf8_as_shift_jis":
        return b.decode("shift_jis", errors="replace")
    if kind == "utf8_as_euc_kr":
        return b.decode("euc_kr", errors="replace")
    if kind == "utf8_as_cp1251":
        return b.decode("cp1251", errors="replace")
    if kind == "utf8_as_koi8_r":
        return b.decode("koi8_r", errors="replace")
    if kind == "cp1251_as_cp1252":
        return text.encode("cp1251", errors="replace").decode("cp1252", errors="replace")
    if kind == "gbk_as_utf8":
        return text.encode("gbk", errors="replace").decode("utf-8", errors="replace")
    if kind == "sjis_as_cp1252":
        return text.encode("shift_jis", errors="replace").decode("cp1252", errors="replace")
    if kind == "utf8_as_utf16le":
        if len(b) % 2:
            b += b" "
        return b.decode("utf-16-le", errors="replace")
    if kind == "utf16le_as_latin1":
        return text.encode("utf-16-le").decode("latin-1")
    raise ValueError(kind)


MOJIBAKE = {
    "latin": ["utf8_as_latin1", "utf8_as_cp1252", "utf8_as_utf16le"],
    "cyrillic": ["utf8_as_latin1", "utf8_as_cp1252", "cp1251_as_cp1252", "utf8_as_koi8_r", "utf8_as_utf16le"],
    "zh": ["utf8_as_latin1", "utf8_as_cp1252", "utf8_as_shift_jis", "gbk_as_utf8", "utf8_as_euc_kr", "utf8_as_utf16le"],
    "ja": ["utf8_as_latin1", "utf8_as_cp1252", "utf8_as_gbk", "sjis_as_cp1252", "utf8_as_euc_kr", "utf8_as_utf16le"],
    "ko": ["utf8_as_latin1", "utf8_as_cp1252", "utf8_as_gbk", "utf8_as_shift_jis", "utf8_as_utf16le"],
}


def unicode_shift(s, k):
    """Every letter's code point moved by k inside its block: what a wrong 'ASCII/Unicode
    shift' key or a misaligned multi-byte decoder makes of CJK and Hangul."""
    out = []
    for c in s:
        o = ord(c)
        if 0x4E00 <= o <= 0x9FFF:
            out.append(chr(0x4E00 + (o - 0x4E00 + k) % (0x9FFF - 0x4E00 + 1)))
        elif 0xAC00 <= o <= 0xD7A3:
            out.append(chr(0xAC00 + (o - 0xAC00 + k) % (0xD7A3 - 0xAC00 + 1)))
        elif 0x3041 <= o <= 0x30FF:
            out.append(chr(0x3041 + (o - 0x3041 + k) % (0x30FF - 0x3041 + 1)))
        else:
            out.append(c)
    return "".join(out)


def is_near_rotation(original, junk, max_shift=3):
    """Whether `junk` is `original` with a few characters moved from one end to the other
    (what a wrong railfence or columnar key sometimes gives): it reads as the plaintext."""
    n = len(original)
    if len(junk) != n or n < 2:
        return False
    for k in range(1, min(max_shift, n - 1) + 1):
        if junk == original[k:] + original[:k] or junk == original[-k:] + original[:-k]:
            return True
    return False


def is_near_plaintext(original, junk, share=0.6):
    """Whether a transposition left most of the plaintext in place: one run of at least
    `share` of it, unchanged (`cn.eUniversity was a wonderful experie`)."""
    if len(original) < 4:
        return False
    m = difflib.SequenceMatcher(None, original, junk, autojunk=False).find_longest_match(0, len(original), 0, len(junk))
    return m.size >= share * len(original)


def junk_label(original, junk, cat):
    """(label, category) of a derived negative. Near rotations of the plaintext, wrong
    transposition keys that leave most of it in place, and text that mojibake changed only
    lightly read as language: reported (info), not counted."""
    if cat in ("railfence", "columnar") and is_near_rotation(original, junk):
        return "info", cat + "_rotation"
    if cat in ("railfence", "columnar") and is_near_plaintext(original, junk):
        return "info", cat + "_near_plaintext"
    if cat == "mojibake":
        changed = sum(1 for c in original if not c.isascii())
        if changed < 0.15 * max(1, len(original)):
            return "info", "mojibake_light"
    return "neg", cat


def script_of(lang):
    if lang in LATIN:
        return "latin"
    if lang in CYRILLIC:
        return "cyrillic"
    return lang  # zh, ja, ko


def junk_generators(lang, text, rng):
    """(category, function) for every junk kind that applies to `text` in `lang`."""
    gens = []
    n = len(text)
    letters = letters_of(text)
    script = script_of(lang)
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    if script == "latin" and ascii_letters >= 3:
        k1 = rng.randint(1, 25)
        gens.append(("caesar", lambda t, k=k1: caesar(t, k)))
        k2 = rng.choice([k for k in range(1, 26) if k != k1])
        gens.append(("caesar", lambda t, k=k2: caesar(t, k)))
        gens.append(("rot47", rot47))
        gens.append(("atbash", atbash))
        a = rng.choice([3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25])
        b = rng.randint(0, 25)
        gens.append(("affine", lambda t, a=a, b=b: affine(t, a, b)))
        perm = list(LOWER)
        rng.shuffle(perm)
        gens.append(("substitution", lambda t, p="".join(perm): substitute(t, p)))
        if ascii_letters >= 10:
            key = "".join(rng.choice(LOWER.upper()) for _ in range(rng.randint(3, 8)))
            wrong = "".join(rng.choice(LOWER.upper()) for _ in range(rng.randint(3, 8)))
            if wrong != key:
                gens.append(("vigenere", lambda t, k=key, w=wrong: vigenere(vigenere(t, k, False), w, True)))
        if ascii_letters >= 20:
            key = "".join(rng.choice(LOWER.upper()) for _ in range(rng.randint(3, 8)))
            partial = key[:-1] + chr((ord(key[-1]) - 65 + 1) % 26 + 65)
            gens.append(("vigenere_partial", lambda t, k=key, p=partial: vigenere(vigenere(t, k, False), p, True)))
    if script == "cyrillic" and letters >= 3:
        k1 = rng.randint(1, 32)
        gens.append(("caesar", lambda t, k=k1: caesar_cyr(t, k)))
        gens.append(("atbash", atbash_cyr))
        perm = list(CYR)
        rng.shuffle(perm)
        gens.append(("substitution", lambda t, p="".join(perm): substitute_cyr(t, p)))
        if letters >= 10:
            key = [rng.randint(0, 32) for _ in range(rng.randint(3, 8))]
            wrong = [rng.randint(0, 32) for _ in range(rng.randint(3, 8))]
            if wrong != key:
                gens.append(("vigenere", lambda t, k=key, w=wrong: vigenere_cyr(vigenere_cyr(t, k, False), w, True)))
    if script in ("zh", "ja", "ko") and letters >= 2:
        k = rng.randint(1, 3000)
        gens.append(("unicode_shift", lambda t, k=k: unicode_shift(t, k)))
    if len(text.strip()) >= 2:
        gens.append(("reversed", lambda t: t[::-1]))
    if n >= 10:
        keys = [(2, 0), (2, 1), (3, 1), (4, 0), (4, 2), (5, 3)]
        rails = rng.choice([2, 3, 4, 5])
        r1, o1 = rng.choice([k for k in keys if k != (rails, 0)])
        gens.append(("railfence", lambda t, r=rails, rr=r1, o=o1: rail_decrypt(rail_encrypt(t, r), rr, o)))
        klen = rng.randint(3, 7)
        key = list(range(klen))
        rng.shuffle(key)
        wrong = list(range(rng.randint(3, 7)))
        rng.shuffle(wrong)
        if wrong != key:
            gens.append(("columnar", lambda t, k=tuple(key), w=tuple(wrong): columnar_decrypt(columnar_encrypt(t, k), w)))
    enc = rng.choice(ENCODINGS if script == "latin" else [e for e in ENCODINGS if e not in ("morse", "a1z26")])
    gens.append(("enc_" + enc, lambda t, e=enc: encode(t, e)))
    if not text.isascii() or script != "latin":
        kind = rng.choice(MOJIBAKE[script])
        gens.append(("mojibake", lambda t, k=kind: mojibake(t, k)))
    return gens


# ------------------------------------------------------------------- n-gram statistics --


class EnglishStats:
    """Letter bigram and quadgram log10 frequencies of the English training books, for the
    optimised Vigenère key search (the near-English junk Ciphey's Vigenère decoder makes)."""

    def __init__(self, cache):
        bi = collections.Counter()
        quad = collections.Counter()
        for n in EN_TRAIN_BOOKS:
            path = inside(cache, os.path.join("gutenberg", f"pg{n}.txt"))
            letters = "".join(c for c in gutenberg_body(path).upper() if "A" <= c <= "Z")
            for i in range(len(letters) - 1):
                bi[letters[i:i + 2]] += 1
            for i in range(len(letters) - 3):
                quad[letters[i:i + 4]] += 1
        tb, tq = sum(bi.values()), sum(quad.values())
        self.bi = [[math.log10((bi.get(chr(65 + a) + chr(65 + b), 0) + 0.5) / tb) for b in range(26)] for a in range(26)]
        self.quad = {k: math.log10(v / tq) for k, v in quad.items()}
        self.quad_floor = math.log10(0.01 / tq)
        self.uni = collections.Counter()
        for k, v in bi.items():
            self.uni[k[0]] += v

    def bigram_fitness(self, s):
        idx = [ord(c) - 65 for c in s]
        return sum(self.bi[a][b] for a, b in zip(idx, idx[1:]))

    def quad_fitness(self, s):
        q, f = self.quad, self.quad_floor
        return sum(q.get(s[i:i + 4], f) for i in range(len(s) - 3))


def optimise_vigenere(ct_letters, key_len, fitness, rng):
    """Coordinate ascent on the key: start from the best shift per column by letter
    frequency, then change one key letter at a time while the fitness improves."""
    english_order = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
    key = []
    for col in range(key_len):
        column = ct_letters[col::key_len]
        best, best_s = None, None
        for s in range(26):
            dec = "".join(chr((ord(c) - 65 - s) % 26 + 65) for c in column)
            counts = collections.Counter(dec)
            score = sum(counts.get(ch, 0) * (26 - i) for i, ch in enumerate(english_order))
            if best is None or score > best:
                best, best_s = score, s
        key.append(best_s)

    def dec(k):
        return "".join(chr((ord(c) - 65 - k[i % key_len]) % 26 + 65) for i, c in enumerate(ct_letters))

    current = fitness(dec(key))
    for _ in range(4):
        improved = False
        for pos in range(key_len):
            for s in range(26):
                if s == key[pos]:
                    continue
                trial = key[:pos] + [s] + key[pos + 1:]
                f = fitness(dec(trial))
                if f > current:
                    current, key, improved = f, trial, True
        if not improved:
            break
    return key


def apply_key_keep_layout(text, key):
    """Decrypts the letters of `text` with `key` (shifts), keeping case and non-letters."""
    out, i = [], 0
    for c in text:
        if "A" <= c <= "Z" or "a" <= c <= "z":
            base = 65 if c <= "Z" else 97
            out.append(chr((ord(c) - base - key[i % len(key)]) % 26 + base))
            i += 1
        else:
            out.append(c)
    return "".join(out)


# ------------------------------------------------------------------------------ corpus --


def is_degenerate(text):
    """A 'sentence' that is one or two words repeated (`Ute ute ute ute ute ute.`): corpus
    noise, not writing."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    return len(words) >= 4 and len(set(words)) <= 2


class Corpus:
    def __init__(self):
        self.rows = []

    def add(self, split, label, lang, cat, source, text):
        if text is None:
            return
        text = text.strip("\n")
        if not text.strip():
            return
        if label == "pos" and is_degenerate(text):
            return
        self.rows.append((split, label, lang, cat, source, text))


def ciphey_samples(corpus):
    """curated.tsv and gutenberg.tsv from Ciphey master (MIT), copied into bench/corpus/v1/ciphey."""
    bases = []
    for name in ("curated.tsv", "gutenberg.tsv"):
        with open(os.path.join(CIPHEY_DIR, name), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                cat, split, text = line.split("\t")
                text = unescape(text)
                if name == "gutenberg.tsv":
                    # The held-out books (1342, 98, 64317) are test books here; the train
                    # ones (1661, 11) train books.
                    split = "test" if split == "heldout" else "train"
                elif split == "heldout":
                    split = "dev" if stable_unit("ciphey-curated", text) < 0.5 else "test"
                source = f"ciphey:{name}"
                if cat == "non_english":
                    lang = CIPHEY_NON_ENGLISH.get(text)
                    if lang:
                        corpus.add(split, "pos", lang, "sentence", source, text)
                        bases.append((split, lang, "sentence", source, text))
                    else:
                        corpus.add(split, "info", "-", "other_language", source, text)
                    continue
                label, mycat = CIPHEY_CATEGORY[cat]
                lang = "en" if label == "pos" else "-"
                corpus.add(split, label, lang, mycat, source, text)
                if label == "pos":
                    bases.append((split, lang, mycat, source, text))
    return bases


def ciphey_captured(corpus):
    path = os.path.join(CIPHEY_DIR, "captured_cases.tsv")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cat, set_, name, text = line.split("\t", 3)
            text = unescape(text)
            u = stable_unit("ciphey-search", set_, name)
            split = "train" if u < 0.6 else ("dev" if u < 0.8 else "test")
            # search_hard is what Athena@High accepted: by construction it holds near
            # answers that read as language ("The meeting is at 6785 in room 959 ..."), so
            # it is reported but not counted; search_random is an even sample of the rest.
            label = "info" if cat == "search_hard" else "neg"
            corpus.add(split, label, "-", cat, f"ciphey:search:{set_}/{name}", text)


def draw_sentences(lang, cache, catalog_books):
    """{split: [(source, text, kind)]} of sentences and paragraphs from every source."""
    out = {"train": [], "dev": [], "test": []}
    sizes = SIZES["en" if lang == "en" else "other"]
    # Tatoeba, by contributor
    by_split = {"train": [], "dev": [], "test": []}
    for sid, text, user in tatoeba_rows(cache, lang):
        if len(text) > 300:
            continue
        # The source is the sentence (https://tatoeba.org/en/sentences/show/<id> gives its
        # author, for the CC BY attribution); the split is by contributor
        by_split[tatoeba_split(user)].append((f"tatoeba:{sid}", text))
    for split in out:
        rows = by_split[split]
        rng = rng_for("tatoeba", lang, split)
        for source, text in rng.sample(rows, min(sizes[split]["tatoeba"], len(rows))):
            out[split].append((source, text, "sentence"))
    # Gutenberg books
    if lang == "en":
        books = EN_TRAIN_BOOKS + EN_DEV_BOOKS + EN_TEST_BOOKS
    else:
        books = catalog_books.get(lang, [])
    for n in books:
        path = inside(cache, os.path.join("gutenberg", f"pg{n}.txt"))
        if not os.path.exists(path):
            continue
        split = book_split(lang, n, books)
        paras = book_paragraphs(path)
        sents = [s for p in paras for s in split_sentences(p, lang) if good_sentence(s, lang)]
        if len(sents) < 20:
            continue
        rng = rng_for("book", lang, n)
        for s in rng.sample(sents, min(sizes[split]["book"], len(sents))):
            out[split].append((f"gutenberg:{n}", s, "sentence"))
        long_paras = [p for p in paras if 200 <= len(p) <= 700]
        for p in rng.sample(long_paras, min(sizes[split]["para"], len(long_paras))):
            out[split].append((f"gutenberg:{n}", p, "paragraph"))
    # Wikipedia (test only: CC BY-SA, evaluation only)
    rows = wiki_rows(cache, lang)
    if rows:
        sents = []
        for r in rows:
            for p in r["text"].split("\n"):
                for s in split_sentences(norm_ws(p), lang):
                    if good_sentence(s, lang):
                        sents.append(s)
        rng = rng_for("wiki", lang)
        for s in rng.sample(sents, min(sizes["test"]["wiki"], len(sents))):
            out["test"].append((f"wikipedia:{lang}", s, "sentence"))
    return out


WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.U)


SCRIPT_TEST = {
    "latin": lambda c: c.isascii() or 0xC0 <= ord(c) <= 0x24F,
    "cyrillic": lambda c: 0x400 <= ord(c) <= 0x52F,
    "ko": lambda c: 0xAC00 <= ord(c) <= 0xD7AF or 0x1100 <= ord(c) <= 0x11FF or 0x3130 <= ord(c) <= 0x318F,
}


def in_script(text, lang):
    """Whether at least 80% of the letters of a word or phrase are in the language's
    script (a Korean sentence's `5G` doesn't make `G` a Korean word)."""
    test = SCRIPT_TEST.get(script_of(lang))
    letters = [c for c in text if c.isalpha()]
    if not letters or test is None:
        return bool(letters)
    return sum(1 for c in letters if test(c)) >= 0.8 * len(letters)


def phrases_and_words(lang, split, sentences, sizes):
    """Short phrases (1-4 words, or 2-8 characters in Chinese and Japanese) and single words
    cut from this split's sentences."""
    rng = rng_for("phrase", lang, split)
    pool = [s for s in sentences if s[2] == "sentence"]
    out = []
    if not pool:
        return out
    for i in range(sizes["phrase"]):
        source, text, _ = pool[rng.randrange(len(pool))]
        if lang in CJK:
            chunks = [c for c in re.split(r"[\s\W\d_]+", text) if len(c) >= 2]
            if not chunks:
                continue
            c = rng.choice(chunks)
            n = rng.randint(2, min(8, len(c)))
            start = rng.randint(0, len(c) - n)
            out.append((source, c[start:start + n], "phrase"))
        else:
            words = WORD_RE.findall(text)
            if not words:
                continue
            n = rng.randint(1, min(4, len(words)))
            start = rng.randint(0, len(words) - n)
            phrase = " ".join(words[start:start + n])
            if in_script(phrase, lang):
                out.append((source, phrase, "phrase" if n > 1 else "word"))
    if lang not in CJK:
        for i in range(sizes.get("short", 0)):
            source, text, _ = pool[rng.randrange(len(pool))]
            words = WORD_RE.findall(text)
            if not words:
                continue
            n = rng.choice([1, 1, 2, 2, 3])
            n = min(n, len(words))
            start = rng.randint(0, len(words) - n)
            phrase = " ".join(words[start:start + n])
            if in_script(phrase, lang) and letters_of(phrase) <= 12:
                out.append((source, phrase, "short"))
        for i in range(sizes["word"]):
            source, text, _ = pool[rng.randrange(len(pool))]
            words = [w for w in WORD_RE.findall(text) if len(w) >= 2 and in_script(w, lang)]
            if words:
                out.append((source, rng.choice(words), "word"))
    return out


IDENT_GENERATORS = None


def identifier(rng):
    kind = rng.choice(["url", "email", "hash", "flag", "number", "secret", "ip", "version"])
    alnum = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if kind == "url":
        return rng.choice(["https://github.com/bee-san/Ciphey", "http://example.com/index.html?q=1",
                           "https://en.wikipedia.org/wiki/Cipher", "www.example.org/path/to/page",
                           "https://docs.rs/gibberish-or-not"])
    if kind == "email":
        return rng.choice(["bee@skerritt.blog", "admin@example.com", "user.name+tag@example.co.uk",
                           "contact@ctf.example.org"])
    if kind == "hash":
        # Benchmark junk shaped like a hash, not security
        if rng.random() < 0.5:
            return hashlib.md5(str(rng.random()).encode(), usedforsecurity=False).hexdigest()
        return hashlib.sha1(str(rng.random()).encode(), usedforsecurity=False).hexdigest()
    if kind == "flag":
        return rng.choice(["flag{", "CTF{", "picoCTF{", "HTB{", "DUCTF{"]) + "".join(rng.choice(alnum + "_") for _ in range(rng.randint(6, 20))) + "}"
    if kind == "number":
        return rng.choice([str(rng.randint(2, 99999)), f"{rng.randint(1, 999)}.{rng.randint(0, 99):02d}",
                           f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}", f"{rng.randint(1990, 2030)}-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}"])
    if kind == "secret":
        return "".join(rng.choice(alnum) for _ in range(rng.randint(16, 40)))
    if kind == "ip":
        return ".".join(str(rng.randint(1, 254)) for _ in range(4))
    return f"v{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"


def with_identifier(text, rng):
    words = text.split(" ")
    pos = rng.randint(0, len(words))
    return " ".join(words[:pos] + [identifier(rng)] + words[pos:])


def variants(lang, base_text, cat, rng):
    """Upper-case and space-less forms, as classical ciphers and Morse leave plaintext."""
    out = []
    if lang in CJK or lang in HANGUL:
        return out
    if cat not in ("sentence", "phrase", "short", "chat", "technical", "formal", "ctf_text", "verse", "literary"):
        return out
    h = int(text_hash(base_text)[:8], 16)
    if h & 1:
        out.append(("caps", base_text.upper()))
    words = base_text.split()
    letters = "".join(c for c in base_text if c.isalpha())
    if h & 2 and len(words) >= 2 and len(letters) >= 8:
        out.append(("nospace", letters.upper() if h & 4 else letters.lower()))
    return out


def noise(corpus, english_texts):
    """Random strings, encodings of random bytes, random CJK/Hangul, mojibake: what wrong
    decoders make of anything. Round-robin over the splits."""
    rng = rng_for("noise")
    counter = [0]

    def add(cat, text):
        counter[0] += 1
        split = ("train", "dev", "test")[counter[0] % 3]
        corpus.add(split, "neg", "-", cat, "noise", text)

    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lower = upper.lower()
    alnum = upper + lower + "0123456789"
    ascii_ = "".join(chr(c) for c in range(0x21, 0x7F))

    def s(alpha, lo, hi):
        return "".join(rng.choice(alpha) for _ in range(rng.randint(lo, hi)))

    for _ in range(600):
        add("rand_upper", s(upper, 1, 48))
        add("rand_lower", s(lower, 1, 48))
        add("rand_alnum", s(alnum, 2, 64))
        add("rand_ascii", s(ascii_, 2, 64))
        add("digits", s("0123456789", 1, 24))
        b = bytes(rng.randrange(256) for _ in range(rng.randint(1, 48)))
        add("hex_random", b.hex())
        add("base64_random", base64.b64encode(b).decode())
        add("base32_random", base64.b32encode(b).decode())
    for _ in range(300):
        words = rng.randint(2, 8)
        add("rand_words", " ".join(s(lower, 1, 8) for _ in range(words)))
        b = bytes(rng.randrange(256) for _ in range(rng.randint(4, 48)))
        add("rand_bytes", b.decode("latin-1"))
        add("rand_bytes_lossy", b.decode("utf-8", errors="replace"))
        add("binary_random", " ".join(f"{x:08b}" for x in b[:12]))
        add("utf16_random", (b if len(b) % 2 == 0 else b + b"\0").decode("utf-16-le", errors="replace"))
        add("rand_cjk", "".join(chr(rng.randint(0x4E00, 0x9FFF)) for _ in range(rng.randint(1, 40))))
        add("rand_hangul", "".join(chr(rng.randint(0xAC00, 0xD7A3)) for _ in range(rng.randint(1, 40))))
        add("rand_kana", "".join(chr(rng.randint(0x3041, 0x30FA)) for _ in range(rng.randint(1, 40))))
        # Base65536 output: code points from its blocks (CJK, CJK Ext A/B, Yi, Hangul, ...)
        blocks = [(0x3400, 0x4DB5), (0x4E00, 0x9FA5), (0xA000, 0xA48C), (0xAC00, 0xD7A3), (0x20000, 0x2A6D6), (0x1D400, 0x1D6A5)]
        lo_hi = [rng.choice(blocks) for _ in range(rng.randint(1, 40))]
        add("base65536", "".join(chr(rng.randint(lo, hi)) for lo, hi in lo_hi))
        add("rand_cyrillic", " ".join(s(CYR, 1, 9) for _ in range(rng.randint(1, 7))))
        add("rand_latin_ext", "".join(chr(rng.randint(0xC0, 0x17F)) for _ in range(rng.randint(2, 30))))
    for _ in range(200):
        w = s(lower, 3, 12).encode()
        algo = rng.choice(["md5", "sha1", "sha256", "sha512"])
        add("hash", hashlib.new(algo, w, usedforsecurity=False).hexdigest())  # junk, not security
        add("uuid", "-".join(s("0123456789abcdef", n, n) for n in (8, 4, 4, 4, 12)))
    for t in english_texts[::7]:
        add("utf16le_as_latin1", mojibake(t, "utf16le_as_latin1"))
        add("ascii_as_utf16", mojibake(t, "utf8_as_utf16le"))


def repetitions(corpus):
    """Repeated letters and repeated short units, what decoders make of repeated digits or
    bytes (`zzzzzzzz`, `NO NO NO NO`, `A A A A`, `abcabcabc`), and near-repetitions: one
    letter with one or two others (`aaaaaaaab`), and base64 or base32 of mostly-zero
    buffers (`AAAAAAAAAAAAAAAAAAAAAQ==`). Every kind in every split."""
    rng = rng_for("repetitions")
    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    words = ["no", "of", "is", "it", "to", "a", "i", "on", "the", "and", "ha", "yes", "an", "in", "at"]
    kinds = 9
    for i in range(150 * kinds):
        split = ("train", "dev", "test")[i % 3]
        kind = (i // 3) % kinds
        if kind == 0:
            text = rng.choice(upper) * rng.randint(3, 60)
        elif kind == 1:
            unit = "".join(rng.choice(upper) for _ in range(rng.randint(2, 3)))
            text = unit * rng.randint(3, 25)
        elif kind == 2:
            unit = rng.choice(upper)
            text = " ".join([unit] * rng.randint(3, 20))
        elif kind == 3:
            text = " ".join([rng.choice(words)] * rng.randint(3, 15))
        elif kind == 4:
            a, b = rng.sample(words, 2)
            text = " ".join([a, b] * rng.randint(2, 10))
        elif kind == 5:
            unit = "".join(rng.choice(upper) for _ in range(rng.randint(1, 3)))
            text = " ".join([unit] * rng.randint(3, 20))
        elif kind == 6:
            # One letter, 8-40 times, with one or two others inserted
            chars = [rng.choice("AIEO" if rng.random() < 0.5 else upper)] * rng.randint(8, 40)
            for _ in range(rng.randint(1, 2)):
                chars.insert(rng.randint(0, len(chars)), rng.choice(upper))
            text = "".join(chars)
        else:
            # base64 (kind 7) or base32 (kind 8) of 6-40 zero bytes and one or two others
            buf = bytearray(rng.randint(6, 40))
            for _ in range(rng.randint(1, 2)):
                buf[rng.randrange(len(buf))] = rng.randint(1, 255)
            text = (base64.b64encode if kind == 7 else base64.b32encode)(bytes(buf)).decode()
        if kind < 7 and rng.random() < 0.5:
            text = text.lower()
        corpus.add(split, "neg", "-", "repetition", "noise:repetition", text)


def optimised_vigenere(corpus, stats, english_by_split):
    """Near-English junk: Vigenère key search output that maximises bigram or quadgram
    fitness, on random letters (no plaintext at all) and on English ciphertexts attacked
    with a wrong key length."""
    for split in ("train", "dev", "test"):
        rng = rng_for("vigopt", split)
        texts = english_by_split[split]
        for i in range(150):
            fit_name, fitness = (("bigram", stats.bigram_fitness), ("quadgram", stats.quad_fitness))[i % 2]
            # Random letters: the search makes them look like English
            n = rng.randint(12, 120)
            ct = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))
            L = rng.randint(3, min(8, max(3, n // 4)))
            key = optimise_vigenere(ct, L, fitness, rng)
            corpus.add(split, "neg", "-", f"vigenere_opt_{fit_name}", "vigopt:random", apply_key_keep_layout(ct, key))
            # English ciphertext, wrong key length (not a multiple or divisor of the real one)
            if not texts:
                continue
            p = rng.choice(texts)
            letters = "".join(c for c in p.upper() if "A" <= c <= "Z")
            if len(letters) < 16:
                continue
            K = rng.randint(3, 7)
            true_key = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(K))
            ct_full = vigenere(p, true_key, False)
            ct_letters = "".join(c for c in ct_full.upper() if "A" <= c <= "Z")
            candidates = [L for L in range(2, 10) if L % K and K % L]
            L = rng.choice(candidates)
            key = optimise_vigenere(ct_letters, L, fitness, rng)
            corpus.add(split, "neg", "en", f"vigenere_opt_{fit_name}", "vigopt:english", apply_key_keep_layout(ct_full, key))


def build(cache):
    corpus = Corpus()
    with open(SOURCES_JSON, encoding="utf-8") as f:
        sources = json.load(f)
    catalog_books = {k: v for k, v in sources.get("gutenberg_books", {}).items() if not k.startswith("en_")}

    bases = ciphey_samples(corpus)  # (split, lang, cat, source, text)
    ciphey_captured(corpus)

    english_by_split = {"train": [], "dev": [], "test": []}
    for split, lang, cat, source, text in bases:
        if lang == "en" and cat in ("sentence", "literary", "chat", "technical", "formal"):
            english_by_split[split].append(text)

    for lang in LANGS:
        sizes = SIZES["en" if lang == "en" else "other"]
        drawn = draw_sentences(lang, cache, catalog_books)
        for split in ("train", "dev", "test"):
            for source, text, kind in drawn[split]:
                if is_degenerate(text):
                    continue
                corpus.add(split, "pos", lang, kind, source, text)
                bases.append((split, lang, kind, source, text))
                if lang == "en" and kind == "sentence":
                    english_by_split[split].append(text)
            for source, text, kind in phrases_and_words(lang, split, [d for d in drawn[split] if not is_degenerate(d[1])], sizes[split]):
                corpus.add(split, "pos", lang, kind, source, text)
                bases.append((split, lang, kind, source, text))
            # Sentences with an identifier in them: still language
            rng = rng_for("mixed", lang, split)
            pool = [d for d in drawn[split] if d[2] == "sentence" and not is_degenerate(d[1])]
            for _ in range(min(sizes[split]["mixed"], len(pool))):
                source, text, _ = pool[rng.randrange(len(pool))]
                corpus.add(split, "pos", lang, "mixed", source, with_identifier(text, rng))

    # Variants and junk derived from every base positive, in the base's split
    derived_bases = []
    for split, lang, cat, source, text in bases:
        rng = rng_for("variants", text)
        for vcat, vtext in variants(lang, text, cat, rng):
            corpus.add(split, "pos", lang, vcat, source, vtext)
            if vcat == "nospace":
                derived_bases.append((split, lang, vtext, source))
    for split, lang, cat, source, text in bases:
        rng = rng_for("junk", text)
        gens = junk_generators(lang, text, rng)
        if lang != "en" and len(gens) > OTHER_JUNK_PER_BASE:
            gens = rng.sample(gens, OTHER_JUNK_PER_BASE)
        for jcat, fn in gens:
            junk = fn(text)
            label, cat = junk_label(text, junk, jcat)
            corpus.add(split, label, lang, cat, source, junk)
    # The usual classical-cipher junk of space-less text: a wrong shift and a wrong rail key
    for split, lang, text, source in derived_bases:
        rng = rng_for("nospace-junk", text)
        if lang in LATIN:
            corpus.add(split, "neg", lang, "caesar", source, caesar(text, rng.randint(1, 25)))
        elif lang in CYRILLIC:
            corpus.add(split, "neg", lang, "caesar", source, caesar_cyr(text, rng.randint(1, 32)))
        corpus.add(split, "neg", lang, "railfence", source, rail_decrypt(rail_encrypt(text, 3), 2, 1))
    # Junk with an identifier in it: still junk
    for split in ("train", "dev", "test"):
        rng = rng_for("mixed-junk", split)
        pool = english_by_split[split]
        for _ in range(min(80, len(pool))):
            t = caesar(rng.choice(pool), rng.randint(1, 25))
            corpus.add(split, "neg", "en", "caesar_mixed", "mixed-junk", with_identifier(t, rng))

    all_english = sorted(t for s in english_by_split.values() for t in s)
    noise(corpus, all_english)
    repetitions(corpus)
    optimised_vigenere(corpus, EnglishStats(cache), english_by_split)
    return corpus


def finalise(corpus):
    """Deduplicates (a text is a positive if any copy is; otherwise the first copy wins in a
    fixed order) and assigns ids."""
    rows = sorted(corpus.rows, key=lambda r: (r[5], {"pos": 0, "info": 1, "neg": 2}[r[1]], r[0], r[3], r[2], r[4]))
    seen = {}
    for r in rows:
        if r[5] not in seen:
            seen[r[5]] = r
    final = []
    for text, (split, label, lang, cat, source, _) in seen.items():
        final.append((text_hash(text)[:12], split, label, lang, cat, source, text))
    ids = collections.Counter(r[0] for r in final)
    assert max(ids.values()) == 1, "id collision"
    final.sort(key=lambda r: (r[1], r[3], r[2], r[4], r[0]))
    return final


def write(final, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "dev", "test"):
        with open(os.path.join(out_dir, f"{split}.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# gibberish-or-not benchmark corpus {VERSION}, split {split}: id<TAB>lang<TAB>label<TAB>category<TAB>source<TAB>text\n")
            for id_, sp, label, lang, cat, source, text in final:
                if sp == split:
                    f.write(f"{id_}\t{lang}\t{label}\t{cat}\t{source}\t{escape(text)}\n")


def manifest(out_dir, cache):
    lines = []
    with open(SOURCES_JSON, encoding="utf-8") as f:
        sources = json.load(f)
    for rel, e in sorted(sources["files"].items()):
        lines.append(f"{e['sha256']}  cache:{rel}")
    for name in sorted(os.listdir(CIPHEY_DIR)):
        if name.endswith(".tsv"):
            lines.append(f"{sha256_file(os.path.join(CIPHEY_DIR, name))}  ciphey/{name}")
    for split in ("train", "dev", "test"):
        lines.append(f"{sha256_file(os.path.join(out_dir, split + '.tsv'))}  {split}.tsv")
    return "\n".join(lines) + "\n"


def summary(final):
    c = collections.Counter((r[1], r[2]) for r in final)
    for split in ("train", "dev", "test"):
        sys.stderr.write(f"{split}: " + ", ".join(f"{l} {c[(split, l)]}" for l in ("pos", "neg", "info")) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GON_DATA_DIR", DEFAULT_CACHE))
    ap.add_argument("--check", action="store_true", help="rebuild into a temporary directory and compare")
    args = ap.parse_args()
    final = finalise(build(args.cache))
    summary(final)
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            write(final, tmp)
            ok = True
            for split in ("train", "dev", "test"):
                a, b = sha256_file(os.path.join(tmp, split + ".tsv")), sha256_file(os.path.join(OUT_DIR, split + ".tsv"))
                print(f"{split}.tsv {'OK' if a == b else 'DIFFERS'} {a}")
                ok &= a == b
            sys.exit(0 if ok else 1)
    write(final, OUT_DIR)
    with open(os.path.join(OUT_DIR, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        f.write(manifest(OUT_DIR, args.cache))


if __name__ == "__main__":
    main()
