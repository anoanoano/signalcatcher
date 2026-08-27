"""Claim-driven news ingestion: fetch only the news a claim could show up in.

Bulk-downloading news is the wrong lever. Measured, a broad sweep yields ~553
usable documents an hour, and almost none of them touch any claim actually being
scored -- they pad the pool without improving the measurement. What blocks the
influence score is not corpus size but whether *independent outlets* covering
this specific claim, in the window where uptake would appear, are present at all.

So the corpus is grown per claim: take the claim's entities and distinctive
terms, ask GDELT what news existed in each diffusion window, fetch it, and
collapse syndication. Every document retrieved is one that could change a score.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Sequence

from ..http import Fetcher
from ..llm import LLM
from ..models import Claim, Document, Source
from . import gdelt
from .article import domain_of, fetch_article
from .dedup import cluster_documents

QUERY_SYSTEM = """\
You turn a claim into a news-archive search query.

The query will be run against a wire-service index covering mainstream and trade \
press. Write it the way a NEWS DESK would refer to the subject, not the way an \
essayist would: concrete named entities, organisations, places, statutes, \
products. Drop abstractions, hedges and analytic vocabulary -- they do not \
appear in news copy.

Return 2-3 short queries. Each should be 2-5 words. Quote multi-word proper \
nouns. If the claim is too abstract to have any news footprint at all, return \
an empty list rather than inventing a vague query."""

QUERY_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}

_STOP = {"the", "and", "that", "with", "from", "this", "have", "been", "will",
         "would", "which", "about", "more", "than", "into", "such", "these"}


def fallback_queries(claim: Claim, n: int = 2) -> list[str]:
    """Entity-based queries, for when the model declines or is unavailable."""
    ents = [e for e in claim.entities if len(e) > 3][:3]
    if ents:
        return [f'"{e}"' for e in ents[:n]]
    words = [w for w in re.findall(r"[A-Za-z]{4,}", claim.text)
             if w.lower() not in _STOP][:4]
    return [" ".join(words[:3])] if words else []


def news_queries(llm: LLM | None, claim: Claim) -> list[str]:
    if llm is None:
        return fallback_queries(claim)
    out = llm.json(
        QUERY_SYSTEM,
        f"CLAIM: {claim.text}\nENTITIES: {', '.join(claim.entities) or 'none'}",
        QUERY_SCHEMA, max_tokens=1500, grounded=False,
    )
    qs = [q.strip() for q in (out or {}).get("queries", []) if q and q.strip()]
    return qs or fallback_queries(claim)


def ingest_for_claim(
    store,
    claim: Claim,
    claim_date: datetime,
    windows_days: Sequence[int] = (30, 180, 365),
    llm: LLM | None = None,
    max_per_window: int = 40,
    workers: int = 8,
    include_prior: bool = True,
    progress=None,
) -> dict:
    """Fetch news around one claim, both before and after it.

    Prior news matters as much as later news: without it, a claim the trade
    press had already reported is scored as original, and the benchmark credits
    a columnist for a story someone else broke.
    """
    say = progress or (lambda m: None)
    queries = news_queries(llm, claim)
    if not queries:
        return {"claim": claim.id, "queries": [], "found": 0, "added": 0,
                "note": "claim has no plausible news footprint"}

    spans: list[tuple[datetime, datetime]] = []
    if include_prior:
        spans.append((claim_date - timedelta(days=365), claim_date))
    for d in windows_days:
        spans.append((claim_date, claim_date + timedelta(days=d)))

    arts: dict[str, dict] = {}
    with Fetcher(min_interval=0.3) as f:
        for q in queries:
            for start, end in spans:
                if start < gdelt.COVERAGE_START:
                    continue  # GDELT full-text coverage does not go back further
                for a in gdelt.walk(f, q, start, end, step_days=max(7, (end - start).days),
                                    max_per_step=max_per_window):
                    u = a.get("url")
                    if u and u not in arts:
                        arts[u] = a
    say(f"    {len(queries)} queries -> {len(arts)} candidate articles")
    if not arts:
        return {"claim": claim.id, "queries": queries, "found": 0, "added": 0}

    def grab(a: dict) -> Document | None:
        dom = domain_of(a["url"])
        src = Source(id=Source.make_id("news", dom), kind="news",
                     name=a.get("domain") or dom, domain=dom)
        try:
            with Fetcher(min_interval=0.2) as f:
                return fetch_article(f, a["url"], src,
                                     fallback_date=gdelt.parse_seendate(a))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        docs = [d for d in ex.map(grab, list(arts.values())) if d is not None]

    mapping, clusters = cluster_documents(docs)
    collapsed = sum(c.size - 1 for c in clusters)
    # Keep only cluster canonicals. Storing every wire copy would inflate the
    # corpus and, worse, inflate uptake counts at scoring time.
    keep = [d for d in docs if mapping.get(d.id, d.id) == d.id]

    added = 0
    for d in keep:
        dom = domain_of(d.url)
        store.upsert_source(Source(id=Source.make_id("news", dom), kind="news",
                                   name=dom, domain=dom))
        if store.upsert_document(d):
            added += 1
    say(f"    fetched {len(docs)}, collapsed {collapsed} syndicated, added {added}")
    return {"claim": claim.id, "queries": queries, "found": len(arts),
            "fetched": len(docs), "syndicated_collapsed": collapsed, "added": added}
