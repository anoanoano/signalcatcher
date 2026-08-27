"""Originality: was this claim already out there before it was published?

The honest difficulty of this measurement is that it is an argument from
silence. Finding prior art proves a claim was not new; *not* finding it proves
only that we did not find it. A benchmark that reports "no prior art -> score
1.0" is really reporting the size of its own corpus, and will reliably rate
obscure claims as brilliant and hand a publisher a number that collapses the
first time someone checks.

So originality is reported as an interval:

    hi = 1 - prior_strength      what the evidence we gathered supports
    lo = hi * coverage           what survives if the unsearched space is
                                 as full of prior art as the searched space

with `coverage` driven mainly by whether the pre-window corpus contained
*topically relevant* material at all. Searching a thousand on-topic documents
and finding no statement of the claim is real evidence of novelty. Searching a
corpus that had nothing on the subject is evidence of nothing, and the interval
widens to say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from ..config import Config
from ..index.retrieve import expand_queries, retrieve
from ..llm import LLM
from ..models import Claim, Direction, Evidence, Relation
from .adjudicate import adjudicate


@dataclass
class OriginalityResult:
    claim_id: str
    score: float           # point estimate, evidence-supported
    lo: float              # lower bound after discounting for search coverage
    hi: float
    coverage: float
    prior_strength: float
    is_synthesis: bool
    best_prior: Evidence | None
    n_prior_matches: int
    n_topical: int
    pool_size: int
    retrievers: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    def detail(self) -> dict:
        bp = self.best_prior
        return {
            "coverage": round(self.coverage, 3),
            "prior_strength": round(self.prior_strength, 3),
            "is_synthesis": self.is_synthesis,
            "n_prior_matches": self.n_prior_matches,
            "n_topical": self.n_topical,
            "pool_size": self.pool_size,
            "retrievers": self.retrievers,
            "best_prior": None if bp is None else {
                "url": bp.url, "title": bp.title,
                "date": bp.published_at.date().isoformat(),
                "relation": bp.relation.value, "confidence": round(bp.confidence, 2),
                "quote": bp.quote,
            },
        }


def compute_coverage(
    pool_size: int, n_topical: int, n_retrievers: int, cfg: Config
) -> float:
    """How much confidence the search itself earns, in 0..1.

    Three multiplicative factors, because they are failure modes rather than
    contributions -- any one of them at zero should collapse the result:

      pool        was there a meaningful body of prior text to search at all
      topical     did the search surface on-subject prior work (the strongest
                  single signal that an absence of matches is informative)
      retrievers  did lexical, dense and fingerprint search all get to vote,
                  or was a whole failure mode left uncovered
    """
    pool = math.log1p(max(pool_size, 0)) / math.log1p(cfg.target_pool)
    pool = min(1.0, pool)
    topical = min(1.0, n_topical / max(cfg.topical_saturation, 1))
    # Even a search that surfaces nothing on-topic retains some value, so this
    # floors at 0.35 rather than zero.
    topical = 0.35 + 0.65 * topical
    retr = 0.55 + 0.45 * min(1.0, n_retrievers / 3.0)
    return round(max(0.0, min(1.0, pool * topical * retr)), 4)


def score_originality(
    store, llm: LLM, claim: Claim, claim_date: datetime, cfg: Config,
    embedder=None, source_id: str | None = None, source_name: str = "",
    run_id: str = "", persist: bool = True,
) -> OriginalityResult:
    queries = expand_queries(llm, claim, n=cfg.queries_per_claim)
    res = retrieve(
        store, claim, queries, before=claim_date, embedder=embedder,
        limit=cfg.candidates_per_claim, per_query=cfg.per_query_depth,
        # A writer restating their own earlier point is not prior art against
        # them; self-citation would otherwise read as unoriginality.
        exclude_source=source_id, exclude_docs=[claim.doc_id],
    )
    evidence = adjudicate(llm, claim, claim_date, res.candidates,
                          Direction.PRIOR, source_name=source_name)
    if persist and evidence and run_id:
        store.add_evidence(evidence, run_id)

    weights = cfg.relation_weights
    strengths = [(weights.get(e.relation.value, 0.0) * e.confidence, e) for e in evidence]
    prior_strength, best = max(strengths, key=lambda x: x[0], default=(0.0, None))
    best_prior = best if prior_strength > 0.1 else None

    n_defeating = sum(1 for e in evidence if e.relation.defeats_novelty and e.confidence >= 0.5)
    n_partial_sources = len({
        e.source_id for e in evidence
        if e.relation is Relation.PARTIAL and e.confidence >= 0.5 and e.source_id
    })
    n_topical = sum(1 for e in evidence
                    if e.relation is not Relation.UNRELATED and e.confidence >= 0.4)

    # A claim whose components are all findable separately, in different places,
    # but which nothing states as a whole, is the textbook original synthesis --
    # and it is exactly the contribution a naive "is this new?" check discards.
    is_synthesis = n_defeating == 0 and n_partial_sources >= 2

    hi = max(0.0, 1.0 - prior_strength)
    coverage = compute_coverage(res.pool_size, n_topical, len(res.retrievers_used), cfg)
    lo = hi * coverage
    # The point estimate leans on the evidence but is pulled toward the pessimistic
    # bound when the search was thin, so a small corpus cannot mint high scores.
    score = lo + (hi - lo) * (0.35 + 0.65 * coverage)

    result = OriginalityResult(
        claim_id=claim.id, score=round(score, 4), lo=round(lo, 4), hi=round(hi, 4),
        coverage=coverage, prior_strength=round(prior_strength, 4),
        is_synthesis=is_synthesis, best_prior=best_prior,
        n_prior_matches=n_defeating, n_topical=n_topical, pool_size=res.pool_size,
        retrievers=res.retrievers_used, evidence=evidence,
    )
    if persist and run_id:
        store.put_score(run_id, "claim", claim.id, "originality", result.score,
                        result.lo, result.hi, result.detail())
    return result
