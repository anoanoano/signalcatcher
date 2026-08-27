"""Fused, time-sliced candidate retrieval for a single claim.

Recall failures here are not symmetric. If retrieval misses a document that
stated the claim earlier, the claim is scored as *original* -- the benchmark
manufactures a finding rather than merely losing one. So three retrievers with
uncorrelated failure modes are run and fused:

  lexical      BM25 over expanded queries -- precise, but blind to paraphrase
  dense        embedding cosine           -- finds the same idea in other words
  fingerprint  exact phrase match         -- rare, but near-conclusive when hit

Fusion is Reciprocal Rank Fusion, which combines rankings without needing the
retrievers' scores to be on comparable scales (BM25 magnitudes and cosine
similarities are not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ..llm import LLM
from ..models import Claim, Document

RRF_K = 60  # standard damping constant


QUERY_SYSTEM = """\
You write search queries to find any document that states, implies, or argues a \
given claim -- whether it predates the claim or follows it.

Produce queries in DIFFERENT VOCABULARIES. The most important document to find is \
one that made this argument first, in the idiom of its own field and era, using \
none of the words the claim uses. An economist, an activist, a trade journal and \
a philosopher would each phrase the same proposition differently; cover that \
range. Include at least one query using older or more formal terminology.

Each query should be 3-10 content words, no operators, no quotes."""

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


def expand_queries(llm: LLM, claim: Claim, n: int = 5) -> list[str]:
    """Generate vocabulary-diverse queries for a claim, plus the claim itself."""
    out = llm.json(
        QUERY_SYSTEM,
        f"CLAIM: {claim.text}\n"
        f"KIND: {claim.kind.value}\n"
        f"ENTITIES: {', '.join(claim.entities) or 'none'}\n\n"
        f"Write {n} diverse search queries.",
        QUERY_SCHEMA, max_tokens=2000, grounded=False,
    )
    queries = [claim.text]
    if out:
        queries.extend(q for q in out.get("queries", []) if isinstance(q, str) and q.strip())
    seen, uniq = set(), []
    for q in queries:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(q.strip())
    return uniq[: n + 1]


@dataclass
class Candidate:
    doc: Document
    fused_score: float
    retrievers: set[str] = field(default_factory=set)
    fingerprint_hits: list[str] = field(default_factory=list)
    best_rank: int = 999


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    pool_size: int          # documents that existed in the time slice at all
    queries: list[str]
    retrievers_used: list[str]

    @property
    def coverage_note(self) -> str:
        return (f"{self.pool_size} docs in window; retrievers="
                f"{'+'.join(self.retrievers_used) or 'none'}")


def retrieve(
    store,
    claim: Claim,
    queries: Sequence[str],
    before: datetime | None = None,
    after: datetime | None = None,
    embedder=None,
    limit: int = 20,
    per_query: int = 15,
    exclude_source: str | None = None,
    exclude_docs: Sequence[str] = (),
) -> RetrievalResult:
    """Retrieve and fuse candidates for one claim within a time slice."""
    ranked: dict[str, list[int]] = {}       # doc_id -> ranks from each retriever
    docs: dict[str, Document] = {}
    who: dict[str, set[str]] = {}
    fps: dict[str, list[str]] = {}
    used: list[str] = []

    def note(doc: Document, rank: int, retriever: str) -> None:
        docs[doc.id] = doc
        ranked.setdefault(doc.id, []).append(rank)
        who.setdefault(doc.id, set()).add(retriever)

    # 1. lexical
    for q in queries:
        hits = store.search(q, before=before, after=after, limit=per_query,
                            exclude_source=exclude_source, exclude_docs=exclude_docs)
        if hits and "lexical" not in used:
            used.append("lexical")
        for rank, (doc, _) in enumerate(hits):
            note(doc, rank, "lexical")

    # 2. dense
    if embedder is not None and getattr(embedder, "enabled", False):
        qvec = embedder.embed_one(claim.text, cache_key=f"claim:{claim.id}")
        if qvec is not None:
            hits = embedder.search(
                qvec,
                before_ts=int(before.timestamp()) if before else None,
                after_ts=int(after.timestamp()) if after else None,
                limit=per_query, exclude_source=exclude_source, exclude_docs=exclude_docs,
            )
            if hits:
                used.append("dense")
            for rank, (doc_id, _) in enumerate(hits):
                doc = store.get_document(doc_id)
                if doc:
                    note(doc, rank, "dense")

    # 3. fingerprints
    for fp in claim.fingerprints:
        hits = store.search_phrase(fp, before=before, after=after, limit=per_query,
                                   exclude_source=exclude_source, exclude_docs=exclude_docs)
        if hits and "fingerprint" not in used:
            used.append("fingerprint")
        for rank, (doc, _) in enumerate(hits):
            note(doc, rank, "fingerprint")
            fps.setdefault(doc.id, []).append(fp)

    cands = []
    for doc_id, ranks in ranked.items():
        # RRF: many mediocre placements can outweigh one lucky top hit, which is
        # what we want -- agreement across independent retrievers is the signal.
        score = sum(1.0 / (RRF_K + r + 1) for r in ranks)
        if doc_id in fps:
            # A verbatim rare-phrase match is qualitatively better evidence than
            # a ranking coincidence; make sure it is never crowded out of top-k.
            score += 0.5
        cands.append(Candidate(
            doc=docs[doc_id], fused_score=score, retrievers=who[doc_id],
            fingerprint_hits=sorted(set(fps.get(doc_id, []))), best_rank=min(ranks),
        ))
    cands.sort(key=lambda c: -c.fused_score)

    pool = store.count_in_window(after=after, before=before)
    return RetrievalResult(cands[:limit], pool, list(queries), used)
