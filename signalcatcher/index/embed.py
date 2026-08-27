"""Dense embeddings, stored in the corpus file alongside everything else.

Lexical search cannot find the case this benchmark most needs to find: someone
who made the same argument earlier in an entirely different vocabulary. Missing
that prior work is not a neutral miss -- it is scored as originality. Dense
retrieval is the hedge against exactly that failure.

The default backend is a local sentence-transformers model. For a benchmark that
is the right call regardless of cost: a hosted embedding endpoint can be
reweighted or retired underneath you, and then last year's scores are no longer
comparable to this year's. Local weights pin to a version.

If no backend can be initialised the pipeline still runs on lexical retrieval
alone -- but `Embedder.status` records why, and the run report surfaces it,
because a coverage claim that silently depends on a retriever that never ran is
worse than an openly smaller one.
"""

from __future__ import annotations

import os
import struct
from typing import Sequence

import numpy as np

LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
OPENAI_MODEL = "text-embedding-3-small"
BATCH = 64


class Embedder:
    def __init__(
        self, store, backend: str = "local", model: str | None = None,
        api_key: str | None = None,
    ):
        self.store = store
        self.backend = backend
        self.enabled = False
        self.status = "uninitialised"
        self._client = None
        self._st = None

        if backend == "openai":
            self.model = model or OPENAI_MODEL
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                self.status = "no OPENAI_API_KEY"
                return
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=key)
                self._client.embeddings.create(model=self.model, input=["probe"])
            except Exception as exc:  # quota, auth, network
                # Probe at construction. Discovering a dead key thousands of
                # documents into a corpus build, after the run has already
                # reported dense coverage it never had, is the bad outcome.
                self.status = f"openai unavailable: {type(exc).__name__}"
                return
            self.enabled = True
            self.status = f"openai:{self.model}"
            return

        self.model = model or LOCAL_MODEL
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(self.model)
        except Exception as exc:
            self.status = f"local model unavailable: {type(exc).__name__}: {exc}"
            return
        self.enabled = True
        self.status = f"local:{self.model}"

    # -- persistence -------------------------------------------------------

    @staticmethod
    def _pack(v: np.ndarray) -> bytes:
        return struct.pack(f"{len(v)}f", *v.astype(np.float32))

    @staticmethod
    def _unpack(b: bytes, dim: int) -> np.ndarray:
        return np.array(struct.unpack(f"{dim}f", b), dtype=np.float32)

    def get(self, key: str) -> np.ndarray | None:
        r = self.store.conn.execute(
            "SELECT dim, vec FROM embeddings WHERE key=? AND model=?", (key, self.model)
        ).fetchone()
        return self._unpack(r["vec"], r["dim"]) if r else None

    def put(self, key: str, v: np.ndarray) -> None:
        with self.store.tx() as c:
            c.execute(
                "INSERT INTO embeddings (key, model, dim, vec) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET vec=excluded.vec, model=excluded.model",
                (key, self.model, len(v), self._pack(v)),
            )

    # -- embedding ---------------------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> list[np.ndarray] | None:
        if not self.enabled or not texts:
            return None
        clean = [t[:8000] if t and t.strip() else " " for t in texts]
        if self._st is not None:
            vecs = self._st.encode(
                clean, normalize_embeddings=True, batch_size=BATCH,
                show_progress_bar=False,
            )
            return [np.asarray(v, dtype=np.float32) for v in vecs]
        out: list[np.ndarray] = []
        for i in range(0, len(clean), BATCH):
            resp = self._client.embeddings.create(model=self.model, input=clean[i : i + BATCH])
            for item in resp.data:
                v = np.array(item.embedding, dtype=np.float32)
                out.append(v / (np.linalg.norm(v) + 1e-9))
        return out

    def ensure_documents(self, docs, progress=None) -> int:
        """Embed any documents that lack a vector. Returns how many were added."""
        if not self.enabled:
            return 0
        todo = [d for d in docs if self.get(f"doc:{d.id}") is None]
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            # Title plus the opening of the body: the thesis of a piece is almost
            # always stated early, and truncating keeps one long document from
            # dominating its own vector with tangents.
            vecs = self.embed_texts([f"{d.title}\n\n{d.text[:6000]}" for d in batch])
            if vecs is None:
                break
            for d, v in zip(batch, vecs):
                self.put(f"doc:{d.id}", v)
            if progress:
                progress(min(i + BATCH, len(todo)), len(todo))
        return len(todo)

    def embed_one(self, text: str, cache_key: str | None = None) -> np.ndarray | None:
        if cache_key:
            hit = self.get(cache_key)
            if hit is not None:
                return hit
        vecs = self.embed_texts([text])
        if not vecs:
            return None
        if cache_key:
            self.put(cache_key, vecs[0])
        return vecs[0]

    # -- search ------------------------------------------------------------

    def search(
        self, query_vec: np.ndarray, before_ts: int | None = None,
        after_ts: int | None = None, limit: int = 25,
        exclude_source: str | None = None, exclude_docs: Sequence[str] = (),
    ) -> list[tuple[str, float]]:
        """Brute-force cosine over the time slice.

        The date filter is applied in SQL *before* scoring, so the slice is a
        true one: ranking the whole corpus and then discarding out-of-window hits
        would let post-publication documents crowd genuine prior art out of top-k.
        """
        where, params = ["e.key LIKE 'doc:%'", "e.model = ?"], [self.model]
        if before_ts is not None:
            where.append("d.published_ts < ?")
            params.append(before_ts)
        if after_ts is not None:
            where.append("d.published_ts > ?")
            params.append(after_ts)
        if exclude_source:
            where.append("d.source_id != ?")
            params.append(exclude_source)
        excl = list(exclude_docs)
        if excl:
            where.append(f"d.id NOT IN ({','.join('?' * len(excl))})")
            params.extend(excl)
        rows = self.store.conn.execute(
            "SELECT d.id AS doc_id, e.dim, e.vec FROM embeddings e "
            "JOIN documents d ON ('doc:' || d.id) = e.key WHERE " + " AND ".join(where),
            params,
        ).fetchall()
        if not rows:
            return []
        mat = np.vstack([self._unpack(r["vec"], r["dim"]) for r in rows])
        sims = mat @ query_vec
        order = np.argsort(-sims)[:limit]
        return [(rows[i]["doc_id"], float(sims[i])) for i in order]
