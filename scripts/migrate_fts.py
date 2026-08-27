"""Migrate documents_fts to an external-content FTS5 index.

The original schema stored a full second copy of every document inside the
search index -- 115 MB of the 298 MB database was redundant text. An
external-content index reads from `documents` instead of duplicating it.

Runs against a copy and verifies before swapping. No re-fetching: all the text
is already in `documents`.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signalcatcher.paths import db_path, human


def migrate(src_path: Path) -> None:
    tmp = src_path.with_suffix(".migrating.db")
    for suf in ("", "-wal", "-shm"):
        p = Path(str(tmp) + suf)
        if p.exists():
            p.unlink()

    print(f"source: {src_path}  ({human(src_path.stat().st_size)})")
    src = sqlite3.connect(src_path)
    src.row_factory = sqlite3.Row
    before = {t: src.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("sources", "documents", "claims", "evidence",
                        "embeddings", "scores", "runs", "llm_cache")}
    print("rows:", before)

    dst = sqlite3.connect(tmp)
    dst.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE sources (id TEXT PRIMARY KEY, kind TEXT NOT NULL,
            name TEXT NOT NULL, domain TEXT DEFAULT '', metadata TEXT DEFAULT '{}');
        -- `rid` is an explicit INTEGER PRIMARY KEY so the FTS index has a stable
        -- rowid to point at; external-content FTS5 requires one.
        CREATE TABLE documents (
            rid INTEGER PRIMARY KEY,
            id TEXT NOT NULL UNIQUE, source_id TEXT NOT NULL REFERENCES sources(id),
            url TEXT NOT NULL UNIQUE, title TEXT NOT NULL, published_at TEXT NOT NULL,
            published_ts INTEGER NOT NULL, retrieved_at TEXT NOT NULL, text TEXT NOT NULL,
            text_hash TEXT NOT NULL, date_confidence TEXT NOT NULL,
            date_provenance TEXT DEFAULT '', paywalled INTEGER DEFAULT 0,
            lang TEXT DEFAULT 'en', metadata TEXT DEFAULT '{}');
        CREATE INDEX idx_docs_ts ON documents(published_ts);
        CREATE INDEX idx_docs_source ON documents(source_id, published_ts);
        CREATE INDEX idx_docs_id ON documents(id);
        -- External content: the index tokenises `documents` in place instead of
        -- keeping its own copy of every article.
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            title, text, content='documents', content_rowid='rid',
            tokenize="porter unicode61");
        CREATE TABLE claims (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL,
            kind TEXT NOT NULL, text TEXT NOT NULL, entities TEXT DEFAULT '[]',
            fingerprints TEXT DEFAULT '[]', explicit INTEGER DEFAULT 1,
            salience REAL DEFAULT 0.5, falsifiable INTEGER DEFAULT 0,
            horizon_days INTEGER, metadata TEXT DEFAULT '{}');
        CREATE INDEX idx_claims_doc ON claims(doc_id);
        CREATE TABLE evidence (id TEXT PRIMARY KEY, claim_id TEXT NOT NULL,
            direction TEXT NOT NULL, relation TEXT NOT NULL, doc_id TEXT, url TEXT NOT NULL,
            title TEXT DEFAULT '', published_at TEXT NOT NULL, published_ts INTEGER NOT NULL,
            source_id TEXT, confidence REAL DEFAULT 0.5, rationale TEXT DEFAULT '',
            quote TEXT DEFAULT '', attributes_source INTEGER DEFAULT 0,
            fingerprint_hits TEXT DEFAULT '[]', retriever TEXT DEFAULT '',
            rank INTEGER DEFAULT 0, run_id TEXT);
        CREATE INDEX idx_ev_claim ON evidence(claim_id, direction);
        CREATE TABLE embeddings (key TEXT PRIMARY KEY, model TEXT NOT NULL,
            dim INTEGER NOT NULL, vec BLOB NOT NULL);
        CREATE TABLE scores (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, metric TEXT NOT NULL,
            value REAL, lo REAL, hi REAL, detail TEXT DEFAULT '{}',
            UNIQUE(run_id, entity_type, entity_id, metric));
        CREATE INDEX idx_scores_lookup ON scores(entity_type, entity_id, metric);
        CREATE TABLE runs (id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
            finished_at TEXT, kind TEXT NOT NULL, config TEXT DEFAULT '{}',
            config_hash TEXT DEFAULT '', corpus_docs INTEGER DEFAULT 0, notes TEXT DEFAULT '');
        CREATE TABLE llm_cache (key TEXT PRIMARY KEY, model TEXT NOT NULL,
            response TEXT NOT NULL, created_at TEXT NOT NULL);
    """)

    dcols = ("id, source_id, url, title, published_at, published_ts, retrieved_at, "
             "text, text_hash, date_confidence, date_provenance, paywalled, lang, metadata")
    rows = src.execute(f"SELECT {dcols} FROM documents").fetchall()
    dst.executemany(
        f"INSERT INTO documents ({dcols}) VALUES ({','.join('?' * 14)})",
        [tuple(r) for r in rows])
    print(f"copied {len(rows)} documents")

    for table, cols in [
        ("sources", "id, kind, name, domain, metadata"),
        ("claims", "id, doc_id, kind, text, entities, fingerprints, explicit, salience, "
                   "falsifiable, horizon_days, metadata"),
        ("evidence", "id, claim_id, direction, relation, doc_id, url, title, published_at, "
                     "published_ts, source_id, confidence, rationale, quote, "
                     "attributes_source, fingerprint_hits, retriever, rank, run_id"),
        ("embeddings", "key, model, dim, vec"),
        ("scores", "run_id, entity_type, entity_id, metric, value, lo, hi, detail"),
        ("runs", "id, started_at, finished_at, kind, config, config_hash, corpus_docs, notes"),
        ("llm_cache", "key, model, response, created_at"),
    ]:
        data = src.execute(f"SELECT {cols} FROM {table}").fetchall()
        n = len(cols.split(","))
        dst.executemany(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * n)})",
                        [tuple(r) for r in data])
        print(f"copied {len(data):6d} {table}")

    dst.commit()
    print("rebuilding FTS index from documents ...")
    dst.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    src.close()

    # verify before swapping
    chk = sqlite3.connect(tmp)
    chk.row_factory = sqlite3.Row
    after = {t: chk.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in before}
    hits = chk.execute(
        "SELECT COUNT(*) FROM documents_fts f JOIN documents d ON d.rid=f.rowid "
        "WHERE documents_fts MATCH ?", ('"housing"',)).fetchone()[0]
    chk.close()
    print("rows after:", after)
    print(f"FTS smoke test ('housing'): {hits} hits")
    assert after == before, f"ROW COUNT MISMATCH\n before={before}\n after={after}"
    assert hits > 0, "FTS index returned nothing -- rebuild failed"

    new_size = tmp.stat().st_size
    old_size = src_path.stat().st_size
    backup = src_path.with_suffix(".db.pre-fts-migration")
    shutil.move(str(src_path), str(backup))
    for suf in ("-wal", "-shm"):
        p = Path(str(src_path) + suf)
        if p.exists():
            p.unlink()
    shutil.move(str(tmp), str(src_path))
    for suf in ("-wal", "-shm"):
        p = Path(str(tmp) + suf)
        if p.exists():
            p.unlink()
    print(f"\n{human(old_size)} -> {human(new_size)}  "
          f"(saved {human(old_size - new_size)}, {100*(old_size-new_size)/old_size:.0f}%)")
    print(f"previous database kept at {backup.name} -- delete once you are satisfied")


if __name__ == "__main__":
    migrate(db_path())
