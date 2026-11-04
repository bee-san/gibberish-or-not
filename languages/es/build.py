#!/usr/bin/env python3
"""Builds languages/es/model.bin from the data cache. Deterministic.

    python3 languages/es/build.py [--cache DIR]          # model.bin, build.json
    python3 languages/es/build.py --fetch [--cache DIR]  # first: download the extra sources

Training sources (train split only; README.md here has URLs, licences and sha256):

* the Tatoeba Spanish sentences of the contributors in the train split (CC BY 2.0 FR);
* the corpus's 13 Spanish Gutenberg train books (`gutenberg_books.es`, the first 13 in id
  order: bench/scripts/make_corpus.py's book_split) except pg40827, an anthology that holds
  the text of dev book pg53355;
* more Spanish Project Gutenberg books (public domain in the USA), picked from the catalog
  by fixed rules (`extra_books`): Spanish only, not a dictionary, grammar or reader, no
  author of a dev or test book, not copyrighted, mostly Spanish; fetched into
  `gutenberg-es/` of the cache;
* the Spanish side of Global Voices (OPUS GlobalVoices v2018q4, monolingual, CC BY 3.0):
  news and blog posts of 2004-2018, fetched into `globalvoices-es/`.

Training sentences (of every source) whose letters are those of a Spanish dev or test text of
the benchmark corpus (`packlib.heldout_keys`) are left out, so that no evaluation text is
trained on, even when another contributor or book has it too.

Paragraphs (books) and lines (Tatoeba, Global Voices) of 6+ words with more common English
words than common Spanish ones are left out too (notes of US school editions, quoted
English headlines).

Model (the choices compared on dev are in README.md; `--opt key=value` overrides them):

* letters a-z, ñ, á, é, í, ó, ú, ü (any other letter is outside the alphabet); dense 3-gram
  tables of interpolated absolute-discounting estimates P(c | previous 2 letters), over the
  letters of each text with everything else removed (word boundaries too, so space-less
  text scores like spaced text);
* words: lower case, accents kept, frequencies mixed from the three kinds of source (books,
  Global Voices, and Tatoeba weighted twice: everyday speech), every word seen 3+ times in
  2+ pseudo-sources (or 10+ times in one), single letters only a e o u y á é ó, no runs of
  one letter (`packlib.is_letter_run`), no Roman numerals written mostly in upper case, no
  letters outside the alphabet; the most frequent up to the size budget, with the
  accentless spelling of accented words (`cancion`, `manana`) half a class rarer than the
  word, for text typed or deciphered without accents;
* segmentation of space-less tokens from 5 letters.

`--counts-cache DIR` keeps the counts (JSON) between experiments; `--out`, `--info` write
elsewhere. Standard library only. Python 3.9+.
"""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "languages", "common"))
import packlib  # noqa: E402

DEFAULT_CACHE = "/local/home/skerraut/herd-synth3/shared/ciphey/data"
USER_AGENT = "gibberish-or-not-es-pack/1.0 (https://github.com/bee-san/gibberish-or-not; github@skerritt.blog)"

CATALOG = "gutenberg/pg_catalog-20261101.csv"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
EXTRA_BOOKS_DIR = "gutenberg-es"
GV_DIR = "globalvoices-es"
GV_FILE = "GlobalVoices-v2018q4-mono-es.txt.gz"
GV_URL = "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/mono/es.txt.gz"
TATOEBA = "tatoeba-20261031/spa_sentences_detailed.tsv.bz2"

# The shipped model's choices (README.md, "Model choices", has the dev numbers behind them)
DEFAULTS = {
    # Alphabet: "full" (a-z ñ á é í ó ú ü: 33 symbols), "noü" (ü folded into u: 32),
    # "folded" (accents folded into their vowel, ñ kept: 27)
    "alphabet": "full",
    "order": 3,
    "discount": 0.75,
    # N-grams: "dense" (A^order bytes) or "hashed" (code points; per order 2^bits u16 slots)
    "ngram": "dense",
    "hbits": "7,11,14,16,16",
    "hbackoff": -1.0,
    # Rank offset for a hashed model (dense: log10 of the alphabet size, the default)
    "rank_offset": 1.52,
    # Experiments only: a rank offset for the dense model too (0: the default, log10 of
    # the alphabet size, not written)
    "dense_rank_offset": 0.0,
    # Word list
    "max_words": 0,  # 0: as many as fit the size budget
    "min_count": 3,
    "min_count_one_source": 10,
    # Mixture weights of the three kinds of source in the word frequencies
    "w_books": 1.0,
    "w_tatoeba": 2.0,
    "w_gv": 1.0,
    # Accentless variants (`cancion` for `canción`) added in the class of their word plus
    # this many classes (0: off)
    "accentless": 2,
    "extra_books": 1,  # use the extra Gutenberg books
    "gv": 1,  # use Global Voices
    "corpus_books": 1,  # use the corpus's train books
    "tatoeba": 1,
    # Tatoeba's frequency of words it almost always capitalises (names: Tom, Mary, Muiriel):
    # 1 counts it, 0 takes those words' frequency from the other sources only (Tatoeba's
    # sentences are about a few recurring characters)
    "tatoeba_names": 1,
    # Experiments only: leave Global Voices blocks i // 10,000 = 7 (mod 25) out, as a held-out
    # news-register check (the shipped model uses every line)
    "gv_holdout": 0,
    # Leave out paragraphs (books) and lines (Tatoeba, Global Voices) of 6+ words that have
    # more common English words than common Spanish ones: notes and vocabularies of US
    # school editions, quoted English headlines, old Gutenberg headers
    "en_filter": 1,
    # Parameters of the word model
    "lambda": -2.0,
    "max_word_chars": 16,
    "seg_min_letters": 5,
}
# N-grams are counted up to this order (the model uses opt["order"] <= it)
COUNT_ORDER = 4
# model.bin + calibration.txt <= 400 KB (docs/languages.md); calibration.txt is ~3.5 KB
MODEL_BUDGET = 409_600 - 5_000
SINGLE_LETTER_WORDS = {"a", "e", "o", "u", "y", "á", "é", "ó"}
ROMAN = re.compile(r"^[ivxlcdm]+$")
TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.U)
NEUTRAL = re.compile(r"\d|://|@")
# Very common Spanish words: a book whose tokens are less than 25% these is not Spanish
COMMON = set("de la que el en y a los se del las un por con no una su para es al lo como "
             "más o pero sus le ha me si sin sobre este ya entre cuando todo esta ser son "
             "dos también fue había era muy años hasta desde está mi porque qué sólo han yo "
             "hay vez puede todos así nos ni parte tiene él uno donde bien tiempo mismo ese "
             "ahora cada e vida otro después te otros aunque esa eso hace otra gobierno tan "
             "durante siempre día tanto ella tres sí dijo sido gran país según menos".split())
# Very common English words that aren't Spanish words (for en_filter)
COMMON_EN = set("the of and to is that it was for on are with his they be at have this from by "
                "but what some we can out other were all there when up your how said an each "
                "she which do their time if will way about many then them would like so these "
                "her make see him two look more day could go did people my over know than who "
                "been now you not or had in".split())
ALPHABETS = {
    "full": list("abcdefghijklmnopqrstuvwxyz") + ["ñ", "á", "é", "í", "ó", "ú", "ü"],
    "noü": list("abcdefghijklmnopqrstuvwxyz") + ["ñ", "á", "é", "í", "ó", "úü"],
    "folded": ["aá", "b", "c", "d", "eé", "f", "g", "h", "ií", "j", "k", "l", "m", "n", "oó",
               "p", "q", "r", "s", "t", "uúü", "v", "w", "x", "y", "z", "ñ"],
}
FULL = ALPHABETS["full"]
ACCENTLESS = str.maketrans("áéíóúüñ", "aeiouun")


# ------------------------------------------------------------------------- sources --

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:  # NOSONAR - callers pass inside(cache, ...) paths
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - network flakes: retry with backoff
            if i == tries - 1 or getattr(e, "code", None) == 404:
                raise
            sys.stderr.write(f"retry {i + 1} {url}: {e}\n")
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def fetch(cache, rel, url):
    """Downloads url to cache/rel unless it is there (the cache is write-once)."""
    path = packlib.inside(cache, rel)
    if os.path.exists(path):
        return path
    data = http_get(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # NOSONAR - path = inside(cache, rel)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:  # NOSONAR - inside the cache, see above
        f.write(data)
    os.rename(tmp, path)  # NOSONAR - inside the cache, see above
    sys.stderr.write(f"fetched {rel} ({len(data)} bytes)\n")
    time.sleep(1.0)  # be polite to the servers
    return path


def authors_of(field):
    """Author names of a catalog row without dates or roles."""
    out = []
    for a in field.split(";"):
        a = re.sub(r"\[[^\]]*\]", "", a).strip()
        a = re.sub(r",?\s*-?\d{3,4}\??(-\d{3,4}\??)?\s*$", "", a).strip().rstrip(",").strip()
        a = re.sub(r",\s*(BC|AD)$", "", a)
        if a:
            out.append(a)
    return out


def extra_books(cache, sources):
    """The extra Spanish Gutenberg books: every Spanish-only text of the catalog that isn't
    one of the corpus's books, a dictionary, grammar or reader (fetch_sources.py's filter),
    or by an author of a dev or test book (so that dev stays honest for tuning and test for
    reporting); in id order."""
    corpus_books = sources["gutenberg_books"]["es"]
    # Every book the corpus's manifest has (any language, including the copyrighted ones
    # it leaves out of its lists) is not an extra book
    taken = set()
    for rel in sources["files"]:
        m = re.match(r"gutenberg/pg(\d+)\.txt$", rel)
        if m:
            taken.add(int(m.group(1)))
    rows = {}
    with open(packlib.inside(cache, CATALOG), newline="", encoding="utf-8") as f:  # NOSONAR - inside the cache
        for r in csv.DictReader(f):
            rows[int(r["Text#"])] = r
    k = len(corpus_books)
    n_dev = max(1, round(k * 0.2))
    n_test = max(1, round(k * 0.2))
    dev_test = corpus_books[k - n_dev - n_test:]
    banned = set()
    for n in dev_test:
        banned.update(authors_of(rows[n]["Authors"]))
    out = []
    for n in sorted(rows):
        r = rows[n]
        if r["Type"] != "Text" or r["Language"].strip() != "es" or n in taken:
            continue
        subjects = (r["Subjects"] + " " + r["Bookshelves"]).lower()
        if any(w in subjects for w in ("dictionar", "language and languages", "grammar",
                                       "phrase books", "readers")):
            continue
        if set(authors_of(r["Authors"])) & banned:
            continue
        out.append(n)
    return out


def corpus_train_books(sources):
    """The corpus's Spanish train books (book_split in bench/scripts/make_corpus.py)."""
    books = sources["gutenberg_books"]["es"]
    k = len(books)
    n_dev = max(1, round(k * 0.2))
    n_test = max(1, round(k * 0.2))
    return books[:k - n_dev - n_test]


# Books the corpus puts in train that hold the text of a dev or test book
EXCLUDED_BOOKS = {
    40827: "anthology (La voz de la conseja) that contains dev book 53355, Pérez Galdós's "
           "La novela en el tranvía",
}
SENTENCE_END = re.compile(r"(?<=[.!?…;:])\s+|\n\s*\n")


def drop_heldout(text, held):
    """`text` without its sentences whose letters are those of a dev or test text, and how
    many were dropped (a sentence: up to . ! ? … ; : or a blank line)."""
    parts = SENTENCE_END.split(text)
    kept = [x for x in parts if packlib.letters_key(x) not in held]
    if len(kept) == len(parts):
        return text, 0
    return "\n".join(kept), len(parts) - len(kept)


def drop_english(text, paragraphs):
    """`text` without its paragraphs (blank-line separated; or lines) of 6+ words with more
    common English words than common Spanish ones, and how many words were dropped."""
    parts = re.split(r"\n\s*\n", text) if paragraphs else [text]
    kept, dropped = [], 0
    for part in parts:
        toks = [t.lower() for t in TOKEN.findall(part)]
        if len(toks) >= 6:
            en = sum(1 for t in toks if t in COMMON_EN)
            es = sum(1 for t in toks if t in COMMON)
            if en > es:
                dropped += len(toks)
                continue
        kept.append(part)
    if not dropped:
        return text, 0
    return "\n\n".join(kept), dropped


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:  # NOSONAR - callers pass inside(cache, ...) paths
        return f.read()


def is_copyrighted(text):
    return "copyrighted project gutenberg" in text[:20000].lower()


def spanish_share(text):
    toks = [t.lower() for t in TOKEN.findall(text[:200000])]
    return sum(1 for t in toks if t in COMMON) / max(1, len(toks))


# ------------------------------------------------------------------------ counting --

class Counts:
    """N-gram counts over the full alphabet (as strings of representative letters) and word
    counts of one kind of source."""

    def __init__(self, order):
        self.ngrams = [collections.Counter() for _ in range(order + 1)]
        self.words = collections.Counter()
        self.lower = collections.Counter()
        self.sources = {}  # word -> first pseudo-source seen in, or None once seen in 2+
        self.letters = 0
        self.tokens = 0
        self.texts = 0


class Translator(dict):
    """str.translate table for the full alphabet: its letters (any case) to the letter,
    other letters to '|' (they break the n-gram context), everything else deleted."""

    def __missing__(self, code):
        c = chr(code)
        if c.isalpha():
            lo = c.lower()
            lo = lo[0] if lo else c
            v = lo if lo in FULL_SET else "|"
        else:
            v = None
        self[code] = v
        return v


FULL_SET = set(FULL)


def count_text(counts, text, source, translator, order, norm_cache):
    runs, cur = [], []
    for chunk in text.split():
        if NEUTRAL.search(chunk):
            if cur:
                runs.append("".join(cur))
                cur = []
            continue
        cur.append(chunk.translate(translator))
        for raw in TOKEN.findall(chunk):
            w = norm_cache.get(raw)
            if w is None:
                w = packlib.normalise_word(raw)
                norm_cache[raw] = w
            counts.words[w] += 1
            counts.tokens += 1
            if raw == w:
                counts.lower[w] += 1
            s = counts.sources.get(w, source)
            if s is not None and s != source:
                counts.sources[w] = None
            elif w not in counts.sources:
                counts.sources[w] = source
    if cur:
        runs.append("".join(cur))
    ng = counts.ngrams
    for r in runs:
        for s in r.split("|"):
            n = len(s)
            if not n:
                continue
            counts.letters += n
            for k in range(1, order + 1):
                if n >= k:
                    ng[k].update(s[i:i + k] for i in range(n - k + 1))
    counts.texts += 1


def gv_held_out_line(i):
    """Whether Global Voices line i is in the experiment-only held-out blocks."""
    return (i // 10_000) % 25 == 7


def load_sources(args, sources, kinds, opt):
    """(kind, inputs, iterator of (pseudo-source, text)) for each kind of source."""
    cache = args.cache
    out = []
    if "tatoeba" in kinds:
        def tat():
            rows = packlib.tatoeba_train_texts(cache, "spa")
            seen = set()
            held = heldout()
            kept = dropped = english = 0
            for i, t in enumerate(rows):
                if t in seen:
                    continue
                seen.add(t)
                t, d = drop_heldout(t, held)
                dropped += d
                if opt["en_filter"]:
                    t, e = drop_english(t, False)
                    english += e
                if not t.strip():
                    continue
                kept += 1
                yield f"tatoeba:{i // 10_000}", t
            STATS["tatoeba_sentences"] = kept
            STATS["tatoeba_heldout_sentences_dropped"] = dropped
            STATS["tatoeba_english_words_dropped"] = english
        out.append(("tatoeba", [(TATOEBA, sources["files"][TATOEBA]["sha256"])], tat))
    if "corpus_books" in kinds:
        corpus_list = corpus_train_books(sources)
        inputs = [(f"gutenberg/pg{n}.txt", sources["files"][f"gutenberg/pg{n}.txt"]["sha256"])
                  for n in corpus_list]

        def corpus_books(books=tuple(corpus_list)):
            used, skipped, dropped, english = [], [], 0, 0
            held = heldout()
            for n in books:
                if n in EXCLUDED_BOOKS:
                    skipped.append([n, EXCLUDED_BOOKS[n]])
                    continue
                path = packlib.inside(cache, f"gutenberg/pg{n}.txt")
                if is_copyrighted(read_text(path)):
                    skipped.append([n, "copyrighted"])
                    continue
                text = packlib.gutenberg_body(path)
                text, d = drop_heldout(text, held)
                dropped += d
                if opt["en_filter"]:
                    text, e = drop_english(text, True)
                    english += e
                used.append(n)
                yield f"gutenberg:{n}", text
            STATS["corpus_books_english_words_dropped"] = english
            STATS["corpus_books_used"] = used
            STATS["corpus_books_skipped"] = skipped
            STATS["corpus_books_heldout_sentences_dropped"] = dropped
        out.append(("books", inputs, corpus_books))
    if "extra_books" in kinds:
        extra_list = extra_books(cache, sources)
        inputs = [(f"{EXTRA_BOOKS_DIR}/pg{n}.txt", None) for n in extra_list]

        def extra(books=tuple(extra_list)):
            used, skipped, dropped, english = [], [], 0, 0
            held = heldout()
            for n in books:
                path = packlib.inside(cache, f"{EXTRA_BOOKS_DIR}/pg{n}.txt")
                if not os.path.exists(path):
                    skipped.append([n, "no plain-text file"])
                    continue
                if is_copyrighted(read_text(path)):
                    skipped.append([n, "copyrighted"])
                    continue
                text = packlib.gutenberg_body(path)
                share = spanish_share(text)
                if share < 0.25:
                    skipped.append([n, f"not mostly Spanish ({share:.2f} common words)"])
                    continue
                text, d = drop_heldout(text, held)
                dropped += d
                if opt["en_filter"]:
                    text, e = drop_english(text, True)
                    english += e
                used.append(n)
                yield f"gutenberg:{n}", text
            STATS["extra_books_english_words_dropped"] = english
            STATS["extra_books_used"] = used
            STATS["extra_books_skipped"] = skipped
            STATS["extra_books_heldout_sentences_dropped"] = dropped
        out.append(("books", inputs, extra))
    if "gv" in kinds:
        rel = f"{GV_DIR}/{GV_FILE}"

        def gv():
            held = heldout()
            seen = set()
            kept = dropped = english = 0
            path = packlib.inside(cache, rel)
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:  # NOSONAR - inside the cache
                for i, line in enumerate(f):
                    if opt["gv_holdout"] and gv_held_out_line(i):
                        continue
                    t = unicodedata.normalize("NFC", line.strip())
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    t, d = drop_heldout(t, held)
                    dropped += d
                    if opt["en_filter"]:
                        t, e = drop_english(t, False)
                        english += e
                    if not t.strip():
                        continue
                    kept += 1
                    yield f"gv:{i // 10_000}", t
            STATS["gv_sentences"] = kept
            STATS["gv_heldout_sentences_dropped"] = dropped
            STATS["gv_english_words_dropped"] = english
        out.append(("gv", [(rel, None)], gv))
    return out


_HELD = []


def heldout():
    """packlib.heldout_keys: the letters of every es dev and test positive of corpus v1."""
    if not _HELD:
        _HELD.append(packlib.heldout_keys(ROOT, "es"))
    return _HELD[0]


STATS = {}


def count_all(args, sources, kinds, order, opt):
    """Counts per kind of source, cached in --counts-cache when given (for experiments:
    the counts don't depend on the model options)."""
    result = {}
    inputs = []
    for kind, ins, it in load_sources(args, sources, kinds, opt):
        inputs.extend(ins)
        key = None
        if args.counts_cache:
            # The counts depend on the inputs, the held-out texts and the filters only
            state = []
            for rel, _ in ins:
                path = packlib.inside(args.cache, rel)
                state.append([rel, os.path.getsize(path) if os.path.exists(path) else None])
            held = hashlib.sha256("\n".join(sorted(heldout())).encode()).hexdigest()
            ident = json.dumps([kind, state, order, held, opt["en_filter"],
                                opt["gv_holdout"] if kind == "gv" else 0, 6], sort_keys=True)
            key = os.path.join(args.counts_cache,
                               hashlib.sha256(ident.encode()).hexdigest()[:16] + ".json")
            if os.path.exists(key):
                with open(key, encoding="utf-8") as f:  # NOSONAR - the user's own scratch dir
                    c, stats = counts_from_json(json.load(f))
                STATS.update(stats)
                result.setdefault(kind, []).append(c)
                continue
        c = Counts(order)
        tr = Translator()
        norm_cache = {}
        before = dict(STATS)
        for source, text in it():
            count_text(c, text, source, tr, order, norm_cache)
        if key:
            os.makedirs(args.counts_cache, exist_ok=True)  # NOSONAR - the user's own scratch dir
            stats = {k: v for k, v in STATS.items() if before.get(k) != v}
            with open(key + ".tmp", "w", encoding="utf-8") as f:  # NOSONAR - see above
                json.dump(counts_to_json(c, stats), f, ensure_ascii=False)
            os.rename(key + ".tmp", key)  # NOSONAR - see above
        result.setdefault(kind, []).append(c)
    return result, inputs


def counts_to_json(c, stats):
    return {"ngrams": [dict(x) for x in c.ngrams], "words": dict(c.words),
            "lower": dict(c.lower), "sources": c.sources, "letters": c.letters,
            "tokens": c.tokens, "texts": c.texts, "stats": stats}


def counts_from_json(d):
    c = Counts(len(d["ngrams"]) - 1)
    c.ngrams = [collections.Counter(x) for x in d["ngrams"]]
    c.words = collections.Counter(d["words"])
    c.lower = collections.Counter(d["lower"])
    c.sources = d["sources"]
    c.letters, c.tokens, c.texts = d["letters"], d["tokens"], d["texts"]
    return c, d["stats"]


def merged(parts):
    """One Counts from several (n-grams and words summed, sources merged)."""
    if len(parts) == 1:
        return parts[0]
    order = len(parts[0].ngrams) - 1
    m = Counts(order)
    for c in parts:
        for k in range(1, order + 1):
            m.ngrams[k].update(c.ngrams[k])
        m.words.update(c.words)
        m.lower.update(c.lower)
        for w, s in c.sources.items():
            prev = m.sources.get(w, s)
            m.sources[w] = None if (prev is None or s is None or prev != s) else s
            if w not in m.sources:
                m.sources[w] = s
        m.letters += c.letters
        m.tokens += c.tokens
        m.texts += c.texts
    return m


# --------------------------------------------------------------------------- model --

def ngram_tables(kinds, opt):
    groups = ALPHABETS[opt["alphabet"]]
    alphabet = packlib.Alphabet(groups)
    order = opt["order"]
    sym = {}
    for i, g in enumerate(groups):
        for c in g:
            sym[c] = i
    if opt["ngram"] == "hashed":
        # Code points as units (the library looks them up lower-cased, unfolded)
        units = [collections.Counter()]
        for k in range(1, order + 1):
            c = collections.Counter()
            for counts in kinds:
                for g, n in counts.ngrams[k].items():
                    c[tuple(ord(x) for x in g)] += n
            units.append(c)
        bits = [int(b) for b in opt["hbits"].split(",")][:order]
        tables, floor = packlib.hashed_tables(units, order, bits, discount=opt["discount"],
                                              backoff=opt["hbackoff"])
        return None, (tables, floor)
    tup = [collections.Counter()]
    for k in range(1, order + 1):
        c = collections.Counter()
        for counts in kinds:
            for g, n in counts.ngrams[k].items():
                c[tuple(sym[x] for x in g)] += n
        tup.append(c)
    dense = packlib.dense_tables(tup, alphabet.size, order, discount=opt["discount"])
    return alphabet, dense


def word_list(by_kind, opt, budget_words):
    """{word: class}: the kept words of every kind, frequencies mixed by the weights."""
    weights = {"books": opt["w_books"], "tatoeba": opt["w_tatoeba"], "gv": opt["w_gv"]}
    allowed = set("".join(ALPHABETS["full"])) | {"'"}
    totals, tables = {}, {}
    seen_two = set()
    lower = collections.Counter()
    count = collections.Counter()
    for kind, c in by_kind.items():
        totals[kind] = c.tokens
        tables[kind] = c.words
        lower.update(c.lower)
        count.update(c.words)
        for w, s in c.sources.items():
            if s is None:
                seen_two.add(w)
    # Pseudo-sources across kinds: a word in two kinds is in two sources
    kinds_of = collections.Counter()
    for kind, c in by_kind.items():
        for w in c.words:
            kinds_of[w] += 1
    keep = {}
    for w, n in count.items():
        two = w in seen_two or kinds_of[w] >= 2
        if n < opt["min_count"] or (not two and n < opt["min_count_one_source"]):
            continue
        if len(w) == 1 and w not in SINGLE_LETTER_WORDS:
            continue
        if packlib.is_letter_run(w):
            continue
        if ROMAN.match(w) and len(w) >= 2 and lower[w] < 0.3 * n:
            continue
        if any(ch not in allowed for ch in w) or w.startswith("'") or w.endswith("'"):
            continue
        p = 0.0
        wsum = 0.0
        name = (not opt["tatoeba_names"] and "tatoeba" in by_kind
                and by_kind["tatoeba"].lower.get(w, 0) < 0.1 * by_kind["tatoeba"].words.get(w, 0))
        for kind in by_kind:
            if name and kind == "tatoeba":
                continue
            wsum += weights[kind]
            p += weights[kind] * tables[kind].get(w, 0) / max(1, totals[kind])
        if p <= 0.0:
            continue
        keep[w] = p / wsum
    # word_classes ranks by count: use the mixed probability as a pseudo-count
    scale = 1e12
    pseudo = {w: int(round(p * scale)) for w, p in keep.items()}
    pseudo = {w: n for w, n in pseudo.items() if n > 0}
    classes = packlib.word_classes(pseudo, scale, budget_words)
    if opt["accentless"]:
        extra = {}
        for w in sorted(classes, key=lambda x: (classes[x], x)):
            v = w.translate(ACCENTLESS)
            if v != w and v not in classes and v not in extra:
                extra[v] = min(15, classes[w] + opt["accentless"] - 1)
        # Variants take the place of the rarest words
        n_base = budget_words - len(extra)
        if n_base < len(classes):
            ranked = sorted(classes, key=lambda x: (-pseudo[x], x))[:n_base]
            classes = {w: classes[w] for w in ranked}
            extra = {}
            for w in sorted(classes, key=lambda x: (classes[x], x)):
                v = w.translate(ACCENTLESS)
                if v != w and v not in classes and v not in extra:
                    extra[v] = min(15, classes[w] + opt["accentless"] - 1)
        classes.update(extra)
    return classes, len(keep)


def model_size(alphabet, ngrams, n_words, opt):
    """Bytes of model.bin with n_words words (packlib.write_model's layout)."""
    parm = 8
    if alphabet is None:
        parm = 12
        alph = 0
        ngrm = 12 + sum(8 + len(data) for _, _, data in ngrams[0])
    else:
        alph = 4 + 8 * len(alphabet.map)
        ngrm = 12 + sum(len(t) for t in ngrams)
    n_buckets = max(1, math.ceil(n_words / (8 * 0.92)))
    word = 72 + 32 * n_buckets if n_words else 0
    pad = lambda n: n + (-n) % 8  # noqa: E731
    sections = (4 if n_words else 3) - (alphabet is None)
    return (16 + 12 * sections + 7) // 8 * 8 + pad(parm) + (pad(alph) if alph else 0) + pad(ngrm) + pad(word)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GON_DATA_DIR", DEFAULT_CACHE))
    ap.add_argument("--fetch", action="store_true", help="download the extra sources into the cache")
    ap.add_argument("--out", default=os.path.join(HERE, "model.bin"))
    ap.add_argument("--info", default=None, help="where to write build.json (default: next to --out)")
    ap.add_argument("--counts-cache", default=None, help="scratch dir for counts (experiments)")
    ap.add_argument("--opt", action="append", default=[], help="key=value: a model option (DEFAULTS)")
    args = ap.parse_args()
    opt = dict(DEFAULTS)
    for kv in args.opt:
        k, v = kv.split("=", 1)
        if k not in opt:
            sys.exit(f"unknown option {k}")
        opt[k] = type(DEFAULTS[k])(v) if not isinstance(DEFAULTS[k], str) else v
    with open(os.path.join(ROOT, "bench", "sources", "sources.json"), encoding="utf-8") as f:  # NOSONAR - the repository's manifest
        sources = json.load(f)

    if args.fetch:
        for n in extra_books(args.cache, sources):
            try:
                fetch(args.cache, f"{EXTRA_BOOKS_DIR}/pg{n}.txt", GUTENBERG_URL.format(n=n))
            except Exception as e:  # noqa: BLE001 - a book without a plain-text file
                sys.stderr.write(f"pg{n}: {e}\n")
        fetch(args.cache, f"{GV_DIR}/{GV_FILE}", GV_URL)
        return

    kinds = [k for k in ("tatoeba", "corpus_books", "extra_books", "gv") if opt[k]]
    by_kind_parts, inputs = count_all(args, sources, kinds, COUNT_ORDER, opt)
    by_kind = {k: merged(v) for k, v in by_kind_parts.items()}
    # Inputs actually read, with their sha256 (books that were skipped are listed too: the
    # selection reads them)
    rows = []
    for rel, digest in inputs:
        path = packlib.inside(args.cache, rel)
        if not os.path.exists(path):
            continue
        rows.append((rel, digest or sha256_file(path)))
    rows.append((CATALOG, sources["files"][CATALOG]["sha256"]))
    rows.sort()

    alphabet, dense = ngram_tables(list(by_kind.values()), opt)
    budget_words = opt["max_words"]
    if not budget_words:
        lo, hi = 1, 200_000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if model_size(alphabet, dense, mid, opt) <= MODEL_BUDGET:
                lo = mid
            else:
                hi = mid - 1
        budget_words = lo
    classes, candidates = word_list(by_kind, opt, budget_words)
    params = {"lambda": opt["lambda"], "max_word_chars": opt["max_word_chars"],
              "seg_min_letters": opt["seg_min_letters"]}
    if alphabet is None:
        params["rank_offset"] = opt["rank_offset"]
        size = packlib.write_model(args.out, "es", ["latin"], params, hashed=dense, words=classes)
    else:
        if opt["dense_rank_offset"]:
            params["rank_offset"] = opt["dense_rank_offset"]
        size = packlib.write_model(args.out, "es", ["latin"], params, alphabet=alphabet,
                                   dense=dense, words=classes)
    digest = sha256_file(args.out)
    letters = sum(c.letters for c in by_kind.values())
    info = {
        "inputs": rows,
        "options": opt,
        "stats": STATS,
        "letters": letters,
        "letters_by_kind": {k: c.letters for k, c in sorted(by_kind.items())},
        "tokens_by_kind": {k: c.tokens for k, c in sorted(by_kind.items())},
        "distinct_words": len(set().union(*[set(c.words) for c in by_kind.values()])),
        "word_candidates": candidates,
        "words_kept": len(classes),
        "model_bytes": size,
        "model_sha256": digest,
        "params": params,
    }
    info_path = args.info or os.path.join(os.path.dirname(os.path.abspath(args.out)), "build.json")
    with open(info_path, "w", encoding="utf-8") as f:  # NOSONAR - the build's own output
        json.dump(info, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in info.items() if k not in ("inputs", "stats")}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
