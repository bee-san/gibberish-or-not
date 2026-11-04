#!/usr/bin/env python3
"""Downloads the raw sources of the benchmark corpus and the language packs into a data cache.

    python3 bench/scripts/fetch_sources.py [--cache DIR] [--only tatoeba,gutenberg,wikipedia]

The cache defaults to $GON_DATA_DIR, else /local/home/skerraut/herd-synth3/shared/ciphey/data
(the Ciphey program's shared cache). Nothing big is committed to the repository: this script
fetches, and bench/sources/sources.json (committed) records every file's URL, size, sha256
and licence, so a fresh checkout can check that it has the same bytes.

The cache is write-once: a file that is already there is never re-downloaded or changed,
only checked against the manifest. New files are written to a temporary name and renamed.

Sources (see bench/sources/README.md for licences and what each is used for):

* Tatoeba per-language "detailed" exports (sentence id, language, text, contributor, dates),
  CC BY 2.0 FR. One file per language.
* Project Gutenberg plain-text books (public domain in the US) and the catalog CSV the
  non-English books are picked from.
* Wikipedia article introductions picked at random (CC BY-SA 4.0: evaluation only, never
  shipped), with page and revision ids so they can be re-fetched exactly.

Only the Python standard library is needed.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "bench", "sources", "sources.json")
DEFAULT_CACHE = "/local/home/skerraut/herd-synth3/shared/ciphey/data"
USER_AGENT = "gibberish-or-not-bench/1.0 (https://github.com/bee-san/gibberish-or-not; github@skerritt.blog)"

# ISO 639-1 -> Tatoeba's ISO 639-3 code (Mandarin is cmn there).
LANGS = {
    "en": "eng", "es": "spa", "fr": "fra", "de": "deu", "it": "ita", "pt": "por", "nl": "nld",
    "pl": "pol", "tr": "tur", "ru": "rus", "zh": "cmn", "ja": "jpn", "ko": "kor",
}

TATOEBA_DIR = "tatoeba-20261031"
TATOEBA_URL = "https://downloads.tatoeba.org/exports/per_language/{iso3}/{iso3}_sentences_detailed.tsv.bz2"

GUTENBERG_DIR = "gutenberg"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"
CATALOG = "pg_catalog-20261101.csv"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"

WIKI_DIR = "wikipedia-intros-20261104"
WIKI_PAGES = 300

# English: the 85 books Ciphey counts its quadgram table from (src/storage/ngrams/generate.py,
# all train here too), and books that are in neither: dev and test. Jane Austen is test only.
EN_TRAIN = [
    11, 12, 16, 23, 35, 36, 41, 43, 45, 46, 55, 74, 76, 84, 86, 108, 113, 120, 139, 145, 147,
    159, 160, 174, 205, 209, 215, 219, 236, 244, 245, 289, 345, 408, 432, 514, 550, 580, 730,
    766, 768, 829, 834, 844, 863, 885, 902, 910, 1023, 1155, 1228, 1260, 1322, 1400, 1404,
    1661, 1695, 1837, 1952, 2097, 2147, 2148, 2641, 2701, 2781, 2814, 2833, 2852, 2891, 3176,
    3300, 3825, 4217, 4276, 4705, 5230, 5827, 6593, 7370, 8492, 9662, 20203, 25344, 34901,
    37134,
]
EN_DEV = [4300, 10007, 69087]
EN_TEST = [1342, 161, 158, 98, 64317, 1250, 26184, 67098]
# More English training books: picked from the catalog with a fixed seed (pick_books),
# never a dev or test book.
EN_EXTRA_TRAIN = 300
# Other languages: picked from the catalog with a fixed seed (see pick_books).
BOOKS_PER_LANGUAGE = 24
COPYRIGHTED = b"COPYRIGHTED Project Gutenberg"
# Grady Ward's Moby word lists (public domain; Gutenberg ebook 3201): spelling lists that
# fill in rare words the training books don't have.
MOBY_URL = "https://www.gutenberg.org/files/3201/files/{name}"
MOBY_FILES = ["common.txt", "crosswd.txt", "names.txt", "places.txt"]


def inside(root, rel):
    """`rel` (a path from this script's own lists or manifest) joined to `root`, refusing
    anything that resolves outside `root` (an absolute path, `..`)."""
    base = os.path.realpath(root)
    path = os.path.realpath(os.path.join(base, rel))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"{rel!r} is outside {base}")
    return path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:  # NOSONAR: callers pass inside(cache, ...) paths
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}


def save_manifest(manifest):
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    manifest["files"] = dict(sorted(manifest["files"].items()))
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")


def http_get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read(), dict(r.headers)
        except Exception as e:  # noqa: BLE001 - network flakes: retry with backoff
            if i == tries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code == 404):
                raise
            sys.stderr.write(f"retry {i + 1} {url}: {e}\n")
            time.sleep(5 * (i + 1))


def fetch(cache, manifest, rel, url, licence, note=""):
    """Downloads url to cache/rel unless it is there; records it in the manifest."""
    path = inside(cache, rel)
    entry = manifest["files"].get(rel)
    if os.path.exists(path):
        digest = sha256_file(path)
        if entry and entry["sha256"] != digest:
            sys.exit(f"{path}: sha256 {digest} differs from the manifest's {entry['sha256']}")
        if not entry:
            manifest["files"][rel] = {"url": url, "bytes": os.path.getsize(path), "sha256": digest,
                                      "licence": licence, "note": note}
        return path
    data, headers = http_get(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)  # NOSONAR: path = inside(cache, rel)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:  # NOSONAR: inside the cache, see above
        f.write(data)
    os.rename(tmp, path)  # NOSONAR: inside the cache, see above
    digest = hashlib.sha256(data).hexdigest()
    if entry and entry["sha256"] != digest:
        sys.exit(f"{url}: downloaded sha256 {digest} differs from the manifest's {entry['sha256']}")
    manifest["files"][rel] = {"url": url, "bytes": len(data), "sha256": digest, "licence": licence,
                              "last_modified": headers.get("Last-Modified", ""), "note": note}
    sys.stderr.write(f"fetched {rel} ({len(data)} bytes)\n")
    time.sleep(0.5)  # be polite to the servers
    return path


def pick_books(catalog_path, lang, k):
    """k books in `lang` only, Type Text, no dictionaries or language textbooks, by fixed seed."""
    ids = []
    with open(catalog_path, newline="", encoding="utf-8") as f:  # NOSONAR: from fetch(), inside the cache
        for r in csv.DictReader(f):
            if r["Type"] != "Text" or r["Language"].strip() != lang:
                continue
            subjects = (r["Subjects"] + " " + r["Bookshelves"]).lower()
            if any(w in subjects for w in ("dictionar", "language and languages", "grammar",
                                           "phrase books", "readers")):
                continue
            ids.append(int(r["Text#"]))
    ids.sort()
    rng = random.Random(f"gon-v1-gutenberg-{lang}")  # NOSONAR: fixed seed for a reproducible sample, not security
    return sorted(rng.sample(ids, min(k, len(ids))))  # NOSONAR: see above


def wiki_intros(cache, manifest, lang):
    """WIKI_PAGES random article introductions as JSON lines (pageid, revid, title, text)."""
    rel = f"{WIKI_DIR}/{lang}.jsonl"
    path = inside(cache, rel)
    if os.path.exists(path):
        return fetch(cache, manifest, rel, "", "CC BY-SA 4.0 (evaluation only)")
    seen, rows = set(), []
    base = f"https://{lang}.wikipedia.org/w/api.php?"
    while len(rows) < WIKI_PAGES:
        q = {"action": "query", "format": "json", "formatversion": "2", "generator": "random",
             "grnnamespace": "0", "grnlimit": "20", "prop": "extracts|revisions", "rvprop": "ids",
             "explaintext": "1", "exintro": "1", "exlimit": "20"}
        data, _ = http_get(base + urllib.parse.urlencode(q))
        for p in json.loads(data)["query"]["pages"]:
            text = (p.get("extract") or "").strip()
            if p["pageid"] in seen or len(text) < 80:
                continue
            seen.add(p["pageid"])
            rows.append({"pageid": p["pageid"], "revid": p.get("revisions", [{}])[0].get("revid"),
                         "title": p["title"], "text": text})
        time.sleep(1)
    rows = rows[:WIKI_PAGES]
    os.makedirs(os.path.dirname(path), exist_ok=True)  # NOSONAR: path = inside(cache, rel)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:  # NOSONAR: inside the cache, see above
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.rename(tmp, path)  # NOSONAR: inside the cache, see above
    return fetch(cache, manifest, rel,
                 f"https://{lang}.wikipedia.org/w/api.php (generator=random, prop=extracts, exintro)",
                 "CC BY-SA 4.0 (evaluation only)", "random article introductions; re-fetch by revid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GON_DATA_DIR", DEFAULT_CACHE))
    ap.add_argument("--only", default="tatoeba,gutenberg,wikipedia")
    args = ap.parse_args()
    only = set(args.only.split(","))
    manifest = load_manifest()
    try:
        if "tatoeba" in only:
            for code, iso3 in LANGS.items():
                fetch(args.cache, manifest, f"{TATOEBA_DIR}/{iso3}_sentences_detailed.tsv.bz2",
                      TATOEBA_URL.format(iso3=iso3), "CC BY 2.0 FR",
                      f"Tatoeba sentences in {code}: id, lang, text, contributor, added, modified")
        if "gutenberg" in only:
            catalog = fetch(args.cache, manifest, f"{GUTENBERG_DIR}/{CATALOG}", CATALOG_URL,
                            "public domain (catalog data)", "Project Gutenberg catalog")
            taken = set(EN_TRAIN + EN_DEV + EN_TEST)
            extra = [n for n in pick_books(catalog, "en", EN_EXTRA_TRAIN + 40) if n not in taken][:EN_EXTRA_TRAIN]
            books = {"en": EN_TRAIN + EN_DEV + EN_TEST + extra}
            for code in LANGS:
                if code != "en":
                    books[code] = pick_books(catalog, code, BOOKS_PER_LANGUAGE)
            manifest["gutenberg_books"] = {"en_train": EN_TRAIN, "en_train_extra": extra, "en_dev": EN_DEV,
                                           "en_test": EN_TEST, **{k: v for k, v in books.items() if k != "en"}}
            for name in MOBY_FILES:
                fetch(args.cache, manifest, f"{GUTENBERG_DIR}/3201/{name}", MOBY_URL.format(name=name),
                      "public domain (Moby Word Lists, Grady Ward)", f"Moby word list {name}")
            missing = []
            for code, ids in books.items():
                for n in ids:
                    rel = f"{GUTENBERG_DIR}/pg{n}.txt"
                    try:
                        path = fetch(args.cache, manifest, rel, GUTENBERG_URL.format(n=n),
                                     "public domain in the USA (Project Gutenberg)", f"Gutenberg ebook {n} ({code})")
                    except urllib.error.HTTPError as e:
                        if e.code != 404:
                            raise
                        missing.append(n)
                        continue
                    # A few Gutenberg ebooks are copyrighted ("This is a COPYRIGHTED Project
                    # Gutenberg eBook"): never used, for shipping or for evaluation
                    with open(path, "rb") as f:  # NOSONAR: path from fetch(), inside the cache
                        head = f.read(8000)
                    if COPYRIGHTED in head:
                        missing.append(n)
                        manifest["files"][rel]["licence"] = "COPYRIGHTED: excluded from every list"
            # Books without a plain-text file are left out of every list
            for key, ids in manifest["gutenberg_books"].items():
                manifest["gutenberg_books"][key] = [n for n in ids if n not in missing]
        if "wikipedia" in only:
            for code in LANGS:
                wiki_intros(args.cache, manifest, code)
    finally:
        save_manifest(manifest)


if __name__ == "__main__":
    main()
