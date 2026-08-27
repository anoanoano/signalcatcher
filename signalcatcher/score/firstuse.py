"""Locate the true first statement of a claim within its author's own work.

Everything downstream anchors to t1. Surprisal is "how unexpected was this at
t1"; adoption is "how did prevalence move after t1". If a writer has been making
the same argument for years -- and any working writer with a recurring thesis
has -- then the document we happened to extract the claim from is not its
origin, and scoring against that date miscomputes both halves of the measure.
Measured concretely: the "New Axis" piece scored as if coined in Aug 2024, when
the author had used the term in 12 earlier pieces.

Method: cheap filters first (fingerprint phrase search, then dense similarity
over the author's back catalog), then one LLM confirmation pass over the best
candidates only. The LLM question is deliberately strict: the earlier piece must
actually STATE the claim, not merely gesture at the topic, or every recurring
beat of an author's worldview would collapse into their first blog post.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..llm import LLM
from ..models import Claim, Document
from ..textutil import normalize_ws

SIM_FLOOR = 0.60          # below this, an earlier doc is not even a candidate
MAX_CANDIDATES = 6

CONFIRM_SYSTEM = """\
You are checking whether an author already stated a claim in an EARLIER piece of
their own writing.

The bar is: the earlier excerpt must assert the claim itself -- the same
proposition, even in rougher or briefer form. It is NOT enough that the earlier
piece discusses the same subject, shares vocabulary, or contains ideas from
which the claim could later have been developed. Authors circle topics for
years before actually making an argument; circling is not stating.

For each candidate, answer `states_claim` true/false, `confidence` 0..1, and
`quote` -- the verbatim span of the excerpt that states it (empty if false)."""

CONFIRM_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_index": {"type": "integer"},
                    "states_claim": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "quote": {"type": "string"},
                },
                "required": ["candidate_index", "states_claim", "confidence", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


@dataclass
class FirstUse:
    claim_id: str
    anchor_doc_id: str        # document to anchor t1 to
    anchor_date: datetime
    original_doc_id: str      # document the claim was extracted from
    original_date: datetime
    moved_days: int           # how far back the anchor moved (0 = unchanged)
    n_prior_statements: int   # earlier own-docs confirmed to state the claim
    quote: str = ""

    @property
    def moved(self) -> bool:
        return self.moved_days > 0


def find_first_use(
    store, llm: LLM, claim: Claim, doc: Document, embedder=None,
) -> FirstUse:
    """Find the earliest of the author's own documents that states the claim."""
    unchanged = FirstUse(
        claim_id=claim.id, anchor_doc_id=doc.id, anchor_date=doc.published_at,
        original_doc_id=doc.id, original_date=doc.published_at,
        moved_days=0, n_prior_statements=0,
    )

    # -- gather candidate earlier docs by two cheap signals ------------------
    scores: dict[str, float] = {}
    docs_by_id: dict[str, Document] = {}

    for fp in claim.fingerprints:
        for d, _ in store.search_phrase(fp, before=doc.published_at, limit=25):
            if d.source_id == doc.source_id and d.id != doc.id:
                docs_by_id[d.id] = d
                scores[d.id] = max(scores.get(d.id, 0.0), 0.9)  # phrase hit is strong

    if embedder is not None and getattr(embedder, "enabled", False):
        qv = embedder.embed_one(claim.text, cache_key=f"claim:{claim.id}")
        if qv is not None:
            own = [d for d in store.documents_for_source(doc.source_id, limit=2000)
                   if d.published_at < doc.published_at and d.id != doc.id]
            embedder.ensure_documents(own)
            for d in own:
                v = embedder.get(f"doc:{d.id}")
                if v is None:
                    continue
                sim = float(np.dot(v, qv))
                if sim >= SIM_FLOOR:
                    docs_by_id[d.id] = d
                    scores[d.id] = max(scores.get(d.id, 0.0), sim)

    if not scores:
        return unchanged

    # Earliest-first: the whole point is finding the origin, and confirming an
    # early hit lets later candidates only ever *raise* the count, not move t1.
    cands = sorted(docs_by_id.values(), key=lambda d: d.published_at)[:MAX_CANDIDATES]

    # -- one confirmation call over all candidates ---------------------------
    from .adjudicate import best_excerpt
    blocks, excerpts = [], []
    for i, d in enumerate(cands):
        ex = best_excerpt(d.text, claim)
        excerpts.append(ex)
        blocks.append(f"[{i}] {d.title} ({d.published_at.date().isoformat()})\n"
                      f"EXCERPT:\n{ex}")
    out = llm.json(
        CONFIRM_SYSTEM,
        f"CLAIM: {claim.text}\n\nThe author later published this claim on "
        f"{doc.published_at.date().isoformat()}. Candidate earlier pieces by the "
        f"same author:\n\n" + "\n---\n".join(blocks),
        CONFIRM_SCHEMA, max_tokens=6000, grounded=True,
    )
    if not out:
        return unchanged

    confirmed: list[tuple[Document, str]] = []
    for r in out.get("results", []):
        idx = r.get("candidate_index")
        if not isinstance(idx, int) or not (0 <= idx < len(cands)):
            continue
        if not r.get("states_claim") or float(r.get("confidence", 0)) < 0.6:
            continue
        quote = normalize_ws(r.get("quote") or "")
        # Same enforcement as the adjudicator: an unquotable confirmation is
        # a recognition from memory, not a reading of the excerpt.
        if len(quote) >= 12 and quote.lower()[:60] in normalize_ws(excerpts[idx]).lower():
            confirmed.append((cands[idx], quote))

    if not confirmed:
        return unchanged
    earliest, quote = min(confirmed, key=lambda cq: cq[0].published_at)
    return FirstUse(
        claim_id=claim.id, anchor_doc_id=earliest.id,
        anchor_date=earliest.published_at, original_doc_id=doc.id,
        original_date=doc.published_at,
        moved_days=(doc.published_at - earliest.published_at).days,
        n_prior_statements=len(confirmed), quote=quote[:300],
    )
