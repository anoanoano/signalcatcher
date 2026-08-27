"""Influence: did this claim actually travel, and did it travel from here?

The dimensions are deliberately the ones a newsroom already argues about, so a
score can be handed to an editor and defended in their vocabulary:

  lead time        days between this piece and the next independent source
                   saying the same thing -- the scoop, measured in days
  pickup breadth   how many independent outlets carried the claim, per window
  attribution      how many of them said where it came from
  phrase spread    verbatim reuse of the source's own distinctive wording
  earliness        where this piece sits in the claim's whole recorded lifetime
  lift             pickup relative to what the topic was doing anyway

`lift` is the one that keeps the rest honest. A columnist writing about an
ongoing story will always be followed by more coverage of that story, and
counting those follow-ons as influence would score the busiest desk highest.
So uptake is compared against the topical background: documents that discuss the
same subject in the same window without asserting the claim. Influence is the
excess, not the total.

Retrieval runs per window but adjudication happens once over the union. Judging
each window separately would pay the model six times to re-read the same
documents, and could return inconsistent verdicts for one document across
windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from ..config import Config
from ..index.retrieve import Candidate, expand_queries, retrieve
from ..llm import LLM
from ..models import Claim, Direction, Evidence, Relation
from .adjudicate import adjudicate

UPTAKE_RELATIONS = (Relation.IDENTICAL, Relation.PARAPHRASE, Relation.PARTIAL)
STRONG_UPTAKE = (Relation.IDENTICAL, Relation.PARAPHRASE)


@dataclass
class WindowStats:
    days: int
    uptake_sources: int          # distinct independent sources asserting the claim
    uptake_docs: int
    strong_docs: int             # identical/paraphrase, not merely partial
    topical_docs: int            # same subject, not asserting -- the background rate
    attributed_docs: int
    fingerprint_docs: int
    lift: float                  # uptake relative to topical background

    def to_dict(self) -> dict:
        return {
            "days": self.days, "uptake_sources": self.uptake_sources,
            "uptake_docs": self.uptake_docs, "strong_docs": self.strong_docs,
            "topical_docs": self.topical_docs, "attributed_docs": self.attributed_docs,
            "fingerprint_docs": self.fingerprint_docs, "lift": round(self.lift, 3),
        }


@dataclass
class InfluenceResult:
    claim_id: str
    score: float
    lead_time_days: int | None       # None => nobody independent ever said it
    first_follower: Evidence | None
    windows: list[WindowStats]
    total_uptake_sources: int
    attribution_rate: float
    fingerprint_spread: int
    evidence: list[Evidence] = field(default_factory=list)

    def detail(self) -> dict:
        ff = self.first_follower
        return {
            "lead_time_days": self.lead_time_days,
            "total_uptake_sources": self.total_uptake_sources,
            "attribution_rate": round(self.attribution_rate, 3),
            "fingerprint_spread": self.fingerprint_spread,
            "windows": [w.to_dict() for w in self.windows],
            "first_follower": None if ff is None else {
                "url": ff.url, "title": ff.title,
                "date": ff.published_at.date().isoformat(),
                "relation": ff.relation.value, "quote": ff.quote,
                "attributed": ff.attributes_source,
            },
        }


def _gather_candidates(
    store, claim: Claim, queries: Sequence[str], claim_date: datetime, cfg: Config,
    embedder, source_id: str | None,
) -> tuple[list[Candidate], dict[int, int], list[str]]:
    """Retrieve per window and union.

    Per-window retrieval matters: a single top-k over all future time is
    dominated by whenever the topic peaked, so a handful of early adopters --
    the evidence that actually establishes a lead -- get crowded out by a later
    surge.
    """
    by_id: dict[str, Candidate] = {}
    pool_by_window: dict[int, int] = {}
    retrievers: set[str] = set()
    for days in cfg.windows_days:
        end = claim_date + timedelta(days=days)
        res = retrieve(
            store, claim, queries, after=claim_date, before=end, embedder=embedder,
            limit=cfg.candidates_per_claim, per_query=max(6, cfg.per_query_depth // 2),
            exclude_source=source_id, exclude_docs=[claim.doc_id],
        )
        pool_by_window[days] = res.pool_size
        retrievers.update(res.retrievers_used)
        for c in res.candidates:
            prev = by_id.get(c.doc.id)
            if prev is None or c.fused_score > prev.fused_score:
                by_id[c.doc.id] = c
    cands = sorted(by_id.values(), key=lambda c: -c.fused_score)
    return cands, pool_by_window, sorted(retrievers)


def score_influence(
    store, llm: LLM, claim: Claim, claim_date: datetime, cfg: Config,
    embedder=None, source_id: str | None = None, source_name: str = "",
    run_id: str = "", persist: bool = True, max_candidates: int | None = None,
) -> InfluenceResult:
    queries = expand_queries(llm, claim, n=cfg.queries_per_claim)
    cands, pools, retrievers = _gather_candidates(
        store, claim, queries, claim_date, cfg, embedder, source_id
    )
    cap = max_candidates or cfg.candidates_per_claim
    evidence = adjudicate(llm, claim, claim_date, cands[:cap],
                          Direction.LATER, source_name=source_name)
    if persist and evidence and run_id:
        store.add_evidence(evidence, run_id)

    uptake = [e for e in evidence if e.relation in UPTAKE_RELATIONS and e.confidence >= 0.5]
    topical = [e for e in evidence if e.relation is Relation.TOPICAL]

    windows: list[WindowStats] = []
    for days in cfg.windows_days:
        end = claim_date + timedelta(days=days)
        w_up = [e for e in uptake if e.published_at <= end]
        w_top = [e for e in topical if e.published_at <= end]
        n_up_docs = len(w_up)
        n_top = len(w_top)
        # Share of on-topic documents that actually carry the claim, against a
        # weak prior that they mostly do not. The +2 keeps a single early hit in
        # a quiet window from reading as infinite lift.
        lift = n_up_docs / (n_up_docs + n_top + 2.0) if (n_up_docs or n_top) else 0.0
        windows.append(WindowStats(
            days=days,
            uptake_sources=len({e.source_id for e in w_up if e.source_id}),
            uptake_docs=n_up_docs,
            strong_docs=sum(1 for e in w_up if e.relation in STRONG_UPTAKE),
            topical_docs=n_top,
            attributed_docs=sum(1 for e in w_up if e.attributes_source),
            fingerprint_docs=sum(1 for e in w_up if e.fingerprint_hits),
            lift=lift,
        ))

    first = min(uptake, key=lambda e: e.published_at, default=None)
    lead = (first.published_at - claim_date).days if first else None
    total_sources = len({e.source_id for e in uptake if e.source_id})
    attributed = sum(1 for e in uptake if e.attributes_source)
    attribution_rate = attributed / len(uptake) if uptake else 0.0
    fp_spread = len({e.doc_id for e in uptake if e.fingerprint_hits})

    score = _combine(windows, total_sources, attribution_rate, fp_spread, lead)

    result = InfluenceResult(
        claim_id=claim.id, score=round(score, 4), lead_time_days=lead,
        first_follower=first, windows=windows, total_uptake_sources=total_sources,
        attribution_rate=attribution_rate, fingerprint_spread=fp_spread,
        evidence=evidence,
    )
    if persist and run_id:
        detail = result.detail()
        detail["window_pool_sizes"] = pools
        detail["retrievers"] = retrievers
        store.put_score(run_id, "claim", claim.id, "influence", result.score,
                        None, None, detail)
    return result


def _combine(
    windows: list[WindowStats], total_sources: int, attribution_rate: float,
    fp_spread: int, lead_days: int | None,
) -> float:
    """Fold the components into 0..1, keeping each one's contribution readable.

    Breadth is the backbone. Everything else is a multiplier on it, because a
    claim nobody repeated is not made influential by having been phrased
    distinctively or by carrying a credit line.
    """
    if total_sources == 0:
        return 0.0
    # Diminishing returns: the step from one outlet to three is the interesting
    # one; from thirty to forty is not.
    import math
    breadth = min(1.0, math.log1p(total_sources) / math.log1p(12))
    lift = max((w.lift for w in windows), default=0.0)
    # Early pickup is worth more than late: it is the part hardest to explain by
    # the topic simply becoming popular on its own.
    early = next((w for w in windows if w.days <= 30), None)
    earliness = min(1.0, (early.uptake_sources / 3.0)) if early else 0.0
    fingerprint = min(1.0, fp_spread / 3.0)
    return (
        0.45 * breadth
        + 0.20 * lift
        + 0.15 * earliness
        + 0.10 * fingerprint
        + 0.10 * attribution_rate
    )
