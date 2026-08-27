"""Storage for the pinned corpus.

SQLite with FTS5. Two reasons this beats a search service here: the corpus stays
a single versioned file you can hash and re-run against, and BM25 ranking is
built in. The benchmark's central retrieval primitive is a *time-sliced* search
-- "everything published strictly before date X" -- so that lives here rather
than being bolted on by callers who might forget the cutoff.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import (
    Claim,
    ClaimKind,
    DateConfidence,
    Direction,
    Document,
    Evidence,
    Relation,
    Source,
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    domain TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS documents (
    -- Explicit INTEGER PRIMARY KEY: the FTS index is external-content and needs
    -- a stable rowid to point at.
    rid INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    source_id TEXT NOT NULL REFERENCES sources(id),
    url TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    published_at TEXT NOT NULL,      -- ISO8601 UTC
    published_ts INTEGER NOT NULL,   -- epoch seconds, for fast range slicing
    retrieved_at TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    date_confidence TEXT NOT NULL,
    date_provenance TEXT DEFAULT '',
    paywalled INTEGER DEFAULT 0,
    lang TEXT DEFAULT 'en',
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_docs_ts ON documents(published_ts);
CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source_id, published_ts);
CREATE INDEX IF NOT EXISTS idx_docs_id ON documents(id);

-- External-content FTS: the index tokenises `documents` in place rather than
-- storing its own copy of every article. A standard FTS5 table duplicates the
-- full text, which on a real corpus was 115 MB of the 298 MB database for no
-- benefit. Rows must be kept in sync explicitly (see upsert_document) since
-- external-content tables do not track their content table automatically.
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    text,
    content='documents',
    content_rowid='rid',
    tokenize = "porter unicode61"
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    entities TEXT DEFAULT '[]',
    fingerprints TEXT DEFAULT '[]',
    explicit INTEGER DEFAULT 1,
    salience REAL DEFAULT 0.5,
    falsifiable INTEGER DEFAULT 0,
    horizon_days INTEGER,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_claims_doc ON claims(doc_id);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    direction TEXT NOT NULL,
    relation TEXT NOT NULL,
    doc_id TEXT,
    url TEXT NOT NULL,
    title TEXT DEFAULT '',
    published_at TEXT NOT NULL,
    published_ts INTEGER NOT NULL,
    source_id TEXT,
    confidence REAL DEFAULT 0.5,
    rationale TEXT DEFAULT '',
    quote TEXT DEFAULT '',
    attributes_source INTEGER DEFAULT 0,
    fingerprint_hits TEXT DEFAULT '[]',
    retriever TEXT DEFAULT '',
    rank INTEGER DEFAULT 0,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_ev_claim ON evidence(claim_id, direction);

CREATE TABLE IF NOT EXISTS embeddings (
    key TEXT PRIMARY KEY,   -- 'doc:<id>' or 'claim:<id>'
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,   -- claim | document | source
    entity_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    lo REAL, hi REAL,
    detail TEXT DEFAULT '{}',
    UNIQUE(run_id, entity_type, entity_id, metric)
);
CREATE INDEX IF NOT EXISTS idx_scores_lookup ON scores(entity_type, entity_id, metric);

-- Every score is stamped with the run that produced it. Runs record the config
-- hash and the corpus state, so a number can always be traced to the exact
-- evidence base and settings that produced it.
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    kind TEXT NOT NULL,          -- score | control:date_shift | control:no_retrieval | ...
    config TEXT DEFAULT '{}',
    config_hash TEXT DEFAULT '',
    corpus_docs INTEGER DEFAULT 0,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


class Store:
    def __init__(self, path: str | Path | None = None):
        from .paths import db_path
        self.path = db_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets a corpus build fetch many publications
        # concurrently; callers that share a Store across threads must
        # serialise writes themselves (see scripts/build_corpus.py).
        self.conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------------------------------------------------------------- writes

    def upsert_source(self, s: Source) -> str:
        with self.tx() as c:
            c.execute(
                "INSERT INTO sources (id, kind, name, domain, metadata) VALUES (?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET name=excluded.name, domain=excluded.domain",
                (s.id, s.kind, s.name, s.domain, json.dumps(s.metadata)),
            )
        return s.id

    def upsert_document(self, d: Document) -> bool:
        """Insert a document. Returns True if newly added.

        Re-ingesting the same URL does not duplicate it, so ingestion is
        restartable without corrupting the corpus counts the scores depend on.
        """
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO documents (id, source_id, url, title, published_at, published_ts,"
                " retrieved_at, text, text_hash, date_confidence, date_provenance, paywalled,"
                " lang, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO NOTHING",
                (
                    d.id, d.source_id, d.url, d.title, _iso(d.published_at), _ts(d.published_at),
                    _iso(d.retrieved_at), d.text, d.text_hash, d.date_confidence.value,
                    d.date_provenance, int(d.paywalled), d.lang, json.dumps(d.metadata),
                ),
            )
            added = cur.rowcount > 0
            if added:
                # External-content FTS is not auto-maintained; the index row must
                # be written with the same rowid as the document it describes.
                c.execute(
                    "INSERT INTO documents_fts (rowid, title, text) VALUES (?,?,?)",
                    (cur.lastrowid, d.title, d.text),
                )
        return added

    def add_claims(self, claims: Iterable[Claim]) -> int:
        rows = [
            (
                cl.id, cl.doc_id, cl.kind.value, cl.text, json.dumps(cl.entities),
                json.dumps(cl.fingerprints), int(cl.explicit), cl.salience,
                int(cl.falsifiable), cl.horizon_days, json.dumps(cl.metadata),
            )
            for cl in claims
        ]
        with self.tx() as c:
            c.executemany(
                "INSERT INTO claims (id, doc_id, kind, text, entities, fingerprints, explicit,"
                " salience, falsifiable, horizon_days, metadata) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO NOTHING",
                rows,
            )
        return len(rows)

    def add_evidence(self, items: Iterable[Evidence], run_id: str) -> int:
        rows = [
            (
                e.id, e.claim_id, e.direction.value, e.relation.value, e.doc_id, e.url,
                e.title, _iso(e.published_at), _ts(e.published_at), e.source_id,
                e.confidence, e.rationale, e.quote, int(e.attributes_source),
                json.dumps(e.fingerprint_hits), e.retriever, e.rank, run_id,
            )
            for e in items
        ]
        with self.tx() as c:
            c.executemany(
                "INSERT INTO evidence (id, claim_id, direction, relation, doc_id, url, title,"
                " published_at, published_ts, source_id, confidence, rationale, quote,"
                " attributes_source, fingerprint_hits, retriever, rank, run_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                rows,
            )
        return len(rows)

    def put_score(
        self, run_id: str, entity_type: str, entity_id: str, metric: str,
        value: float | None, lo: float | None = None, hi: float | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO scores (run_id, entity_type, entity_id, metric, value, lo, hi, detail)"
                " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(run_id, entity_type, entity_id, metric)"
                " DO UPDATE SET value=excluded.value, lo=excluded.lo, hi=excluded.hi,"
                " detail=excluded.detail",
                (run_id, entity_type, entity_id, metric, value, lo, hi,
                 json.dumps(detail or {})),
            )

    def start_run(self, run_id: str, kind: str, config: dict[str, Any], config_hash: str) -> str:
        with self.tx() as c:
            c.execute(
                "INSERT INTO runs (id, started_at, kind, config, config_hash, corpus_docs)"
                " VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                (run_id, _iso(datetime.now(timezone.utc)), kind, json.dumps(config),
                 config_hash, self.count_documents()),
            )
        return run_id

    def finish_run(self, run_id: str, notes: str = "") -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE runs SET finished_at=?, notes=? WHERE id=?",
                (_iso(datetime.now(timezone.utc)), notes, run_id),
            )

    # ----------------------------------------------------------------- reads

    def count_documents(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def corpus_span(self) -> tuple[datetime | None, datetime | None]:
        r = self.conn.execute(
            "SELECT MIN(published_ts), MAX(published_ts) FROM documents"
        ).fetchone()
        if not r or r[0] is None:
            return None, None
        return (
            datetime.fromtimestamp(r[0], timezone.utc),
            datetime.fromtimestamp(r[1], timezone.utc),
        )

    def get_document(self, doc_id: str) -> Document | None:
        r = self.conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return _row_to_doc(r) if r else None

    def documents_for_source(self, source_id: str, limit: int = 1000) -> list[Document]:
        rows = self.conn.execute(
            "SELECT * FROM documents WHERE source_id=? ORDER BY published_ts DESC LIMIT ?",
            (source_id, limit),
        ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def get_claims(self, doc_id: str) -> list[Claim]:
        rows = self.conn.execute("SELECT * FROM claims WHERE doc_id=?", (doc_id,)).fetchall()
        return [_row_to_claim(r) for r in rows]

    def get_source(self, source_id: str) -> Source | None:
        r = self.conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        if not r:
            return None
        return Source(id=r["id"], kind=r["kind"], name=r["name"], domain=r["domain"],
                      metadata=json.loads(r["metadata"]))

    def find_source(self, name: str) -> Source | None:
        r = self.conn.execute(
            "SELECT * FROM sources WHERE lower(name)=lower(?)", (name,)
        ).fetchone()
        if not r:
            return None
        return Source(id=r["id"], kind=r["kind"], name=r["name"], domain=r["domain"],
                      metadata=json.loads(r["metadata"]))

    def list_sources(self) -> list[tuple[Source, int]]:
        rows = self.conn.execute(
            "SELECT s.*, COUNT(d.id) n FROM sources s LEFT JOIN documents d ON d.source_id=s.id"
            " GROUP BY s.id ORDER BY n DESC"
        ).fetchall()
        return [
            (Source(id=r["id"], kind=r["kind"], name=r["name"], domain=r["domain"],
                    metadata=json.loads(r["metadata"])), r["n"])
            for r in rows
        ]

    def search(
        self,
        query: str,
        before: datetime | None = None,
        after: datetime | None = None,
        limit: int = 25,
        exclude_source: str | None = None,
        exclude_docs: Iterable[str] = (),
    ) -> list[tuple[Document, float]]:
        """Time-sliced BM25 search. The date bounds are the whole point.

        `before` is strict (<) so a document can never be prior art for itself,
        and `after` is strict (>) for the same reason in the forward direction.
        Excluding the source's own later output keeps a writer from being
        credited with influencing themselves.
        """
        match = _to_fts_query(query)
        if not match:
            return []
        where = ["documents_fts MATCH ?"]
        params: list[Any] = [match]
        if before is not None:
            where.append("d.published_ts < ?")
            params.append(_ts(before))
        if after is not None:
            where.append("d.published_ts > ?")
            params.append(_ts(after))
        excl = list(exclude_docs)
        if excl:
            where.append(f"d.id NOT IN ({','.join('?' * len(excl))})")
            params.extend(excl)
        if exclude_source:
            where.append("d.source_id != ?")
            params.append(exclude_source)
        params.append(limit)
        sql = (
            "SELECT d.*, bm25(documents_fts) AS score FROM documents_fts f "
            "JOIN documents d ON d.rid = f.rowid WHERE " + " AND ".join(where) +
            " ORDER BY score LIMIT ?"
        )
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []  # malformed FTS expression after sanitising; treat as no hits
        # bm25() returns negative numbers, more negative = better. Flip for sanity.
        return [(_row_to_doc(r), -float(r["score"])) for r in rows]

    def search_phrase(
        self, phrase: str, before: datetime | None = None, after: datetime | None = None,
        limit: int = 25, exclude_source: str | None = None,
        exclude_docs: Iterable[str] = (),
    ) -> list[tuple[Document, float]]:
        """Exact-phrase search, for tracking verbatim reuse of a coinage.

        This is the strongest transmission evidence the benchmark has. Semantic
        similarity cannot tell influence apart from two writers independently
        noticing the same thing; a rare phrase reappearing verbatim in someone
        else's work is much harder to explain by coincidence.
        """
        toks = [t for t in "".join(
            ch if ch.isalnum() or ch.isspace() else " " for ch in phrase
        ).split() if t]
        if len(toks) < 2:
            return []
        match = '"' + " ".join(toks) + '"'
        where = ["documents_fts MATCH ?"]
        params: list[Any] = [match]
        if before is not None:
            where.append("d.published_ts < ?")
            params.append(_ts(before))
        if after is not None:
            where.append("d.published_ts > ?")
            params.append(_ts(after))
        excl = list(exclude_docs)
        if excl:
            where.append(f"d.id NOT IN ({','.join('?' * len(excl))})")
            params.extend(excl)
        if exclude_source:
            where.append("d.source_id != ?")
            params.append(exclude_source)
        params.append(limit)
        sql = ("SELECT d.*, bm25(documents_fts) AS score FROM documents_fts f "
               "JOIN documents d ON d.rid = f.rowid WHERE " + " AND ".join(where) +
               " ORDER BY score LIMIT ?")
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(_row_to_doc(r), -float(r["score"])) for r in rows]

    def count_in_window(self, after: datetime | None = None, before: datetime | None = None) -> int:
        where, params = [], []
        if after is not None:
            where.append("published_ts > ?")
            params.append(_ts(after))
        if before is not None:
            where.append("published_ts < ?")
            params.append(_ts(before))
        sql = "SELECT COUNT(*) FROM documents" + (" WHERE " + " AND ".join(where) if where else "")
        return self.conn.execute(sql, params).fetchone()[0]

    # LLM response cache: judging is the dominant cost, and controls re-judge the
    # same claims repeatedly. Caching makes the validation harness affordable.
    def cache_get(self, key: str) -> str | None:
        r = self.conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
        return r["response"] if r else None

    def cache_put(self, key: str, model: str, response: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO llm_cache (key, model, response, created_at) VALUES (?,?,?,?)"
                " ON CONFLICT(key) DO UPDATE SET response=excluded.response",
                (key, model, response, _iso(datetime.now(timezone.utc))),
            )


_FTS_SPECIAL = set('"*():^-')


def _to_fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 OR-query.

    User- and model-generated query strings routinely contain characters FTS5
    treats as operators. Rather than let a stray colon raise, each token is
    quoted and joined with OR; ranking, not the boolean, decides relevance.
    """
    toks = [t for t in "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in q).split()
            if len(t) > 2]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"' for t in toks[:40])


def _row_to_doc(r: sqlite3.Row) -> Document:
    return Document(
        id=r["id"], source_id=r["source_id"], url=r["url"], title=r["title"],
        published_at=_parse(r["published_at"]), text=r["text"],
        date_confidence=DateConfidence(r["date_confidence"]),
        date_provenance=r["date_provenance"], retrieved_at=_parse(r["retrieved_at"]),
        paywalled=bool(r["paywalled"]), lang=r["lang"], metadata=json.loads(r["metadata"]),
    )


def _row_to_claim(r: sqlite3.Row) -> Claim:
    return Claim(
        id=r["id"], doc_id=r["doc_id"], kind=ClaimKind(r["kind"]), text=r["text"],
        entities=json.loads(r["entities"]), fingerprints=json.loads(r["fingerprints"]),
        explicit=bool(r["explicit"]), salience=r["salience"],
        falsifiable=bool(r["falsifiable"]), horizon_days=r["horizon_days"],
        metadata=json.loads(r["metadata"]),
    )
