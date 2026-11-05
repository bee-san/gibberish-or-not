# Raw sources (data cache)

The corpus and the language packs are built from files in a data cache
(`$GON_DATA_DIR`, by default the Ciphey program's shared cache
`/local/home/skerraut/herd-synth3/shared/ciphey/data`). Nothing big is committed:
`python3 bench/scripts/fetch_sources.py` downloads the files, and `sources.json` records
each one's URL, size, sha256, licence and date, so a fresh checkout can check it has the
same bytes. The cache is write-once: files are never changed or deleted.

| cache directory | what | licence | shipped in a pack? |
|---|---|---|---|
| `tatoeba-20261031/<iso3>_sentences_detailed.tsv.bz2` | Tatoeba per-language exports (sentence id, language, text, contributor, dates), 13 languages | CC BY 2.0 FR (attribution: Tatoeba, tatoeba.org) | yes, as counts (n-gram and word frequencies) |
| `gutenberg/pg<n>.txt` | Project Gutenberg plain-text books: English 85 + 296 train, 3 dev, 8 test; up to 24 per other language where Gutenberg has them (`gutenberg_books` in sources.json; copyrighted ebooks are left out) | public domain in the USA | yes, as counts |
| `gutenberg/pg_catalog-20261101.csv` | Gutenberg's catalog, to pick books per language | public domain | no |
| `gutenberg/3201/*.txt` | Grady Ward's Moby word lists (Gutenberg ebook 3201) | public domain | yes (English word list) |
| `wikipedia-intros-20261104/<lang>.jsonl` | 300 random article introductions per language, with page and revision ids | CC BY-SA 4.0 | **no: evaluation only** (test split) |

Language agents add sources the same way: append to `fetch_sources.py`'s lists (or their
own fetch step in `languages/<code>/build.py`), record them in `sources.json` and in the
shared cache's SOURCES.md, and only ship counts from public domain, CC0, MIT/BSD/Apache-2.0
or CC BY sources (docs/languages.md).
