"""Roll claim-level scores up to documents and sources.

The aggregation deliberately does not lead with a mean. A writer's value is
almost never their average paragraph: one field-defining piece and a hundred
routine ones is a completely different asset from a hundred uniformly competent
ones, and a mean collapses that distinction exactly where it matters most to
someone deciding what a body of work is worth.

So a source is described by a distribution -- its best work, its hit rate, and
its typical work -- and the headline number is explicitly a weighted blend of
those, with the components kept visible underneath.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Sequence


@dataclass
class ClaimScore:
    claim_id: str
    doc_id: str
    text: str
    kind: str
    salience: float
    explicit: bool
    originality: float
    originality_lo: float
    originality_hi: float
    coverage: float
    influence: float
    lead_time_days: int | None
    uptake_sources: int
    is_synthesis: bool

    @property
    def signal(self) -> float:
        """Originality and influence combined into one per-claim number.

        The product, not the sum: an idea that was new but went nowhere and an
        idea that spread but was already common both fail to demonstrate that
        this source contributed something. Only the conjunction does, and a sum
        would let either one alone carry the score.
        """
        return math.sqrt(max(self.originality, 0.0) * max(self.influence, 0.0))


@dataclass
class DocumentScore:
    doc_id: str
    title: str
    url: str
    published: str
    n_claims: int
    originality: float
    influence: float
    signal: float
    best_claim: ClaimScore | None
    claims: list[ClaimScore] = field(default_factory=list)


@dataclass
class SourceScore:
    source_id: str
    name: str
    n_docs: int
    n_claims: int
    headline: float
    peak_signal: float          # best single claim
    hit_rate: float             # share of claims clearing the bar
    typical_signal: float       # median claim
    median_originality: float
    median_influence: float
    mean_coverage: float
    n_synthesis: int
    median_lead_time: float | None
    documents: list[DocumentScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id, "name": self.name, "n_docs": self.n_docs,
            "n_claims": self.n_claims, "headline": round(self.headline, 4),
            "peak_signal": round(self.peak_signal, 4),
            "hit_rate": round(self.hit_rate, 4),
            "typical_signal": round(self.typical_signal, 4),
            "median_originality": round(self.median_originality, 4),
            "median_influence": round(self.median_influence, 4),
            "mean_coverage": round(self.mean_coverage, 4),
            "n_synthesis": self.n_synthesis,
            "median_lead_time_days": self.median_lead_time,
        }


HIT_THRESHOLD = 0.35  # a claim that was both meaningfully new and demonstrably travelled


def aggregate_document(doc_id: str, title: str, url: str, published: str,
                       claims: Sequence[ClaimScore]) -> DocumentScore:
    """Salience-weight claims, so a piece is judged on what it set out to say."""
    claims = list(claims)
    if not claims:
        return DocumentScore(doc_id, title, url, published, 0, 0.0, 0.0, 0.0, None, [])
    wsum = sum(c.salience for c in claims) or float(len(claims))

    def wavg(f) -> float:
        return sum(f(c) * c.salience for c in claims) / wsum

    best = max(claims, key=lambda c: c.signal)
    return DocumentScore(
        doc_id=doc_id, title=title, url=url, published=published, n_claims=len(claims),
        originality=round(wavg(lambda c: c.originality), 4),
        influence=round(wavg(lambda c: c.influence), 4),
        signal=round(wavg(lambda c: c.signal), 4),
        best_claim=best, claims=claims,
    )


def aggregate_source(
    source_id: str, name: str, docs: Sequence[DocumentScore],
    hit_threshold: float = HIT_THRESHOLD,
) -> SourceScore:
    all_claims = [c for d in docs for c in d.claims]
    if not all_claims:
        return SourceScore(source_id, name, len(docs), 0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0, None, list(docs))
    signals = sorted((c.signal for c in all_claims), reverse=True)
    # "Peak" is the mean of the top decile rather than the single maximum, so one
    # lucky judgement cannot define a body of work.
    top_n = max(1, len(signals) // 10)
    peak = sum(signals[:top_n]) / top_n
    hit_rate = sum(1 for s in signals if s >= hit_threshold) / len(signals)
    typical = median(signals)
    leads = [c.lead_time_days for c in all_claims if c.lead_time_days is not None]

    # Weighted toward the top of the distribution on purpose: the case for a
    # source's value rests on its best contributions, tempered by how reliably
    # it produces them.
    headline = 0.5 * peak + 0.3 * hit_rate + 0.2 * typical

    return SourceScore(
        source_id=source_id, name=name, n_docs=len(docs), n_claims=len(all_claims),
        headline=round(headline, 4), peak_signal=round(peak, 4),
        hit_rate=round(hit_rate, 4), typical_signal=round(typical, 4),
        median_originality=round(median([c.originality for c in all_claims]), 4),
        median_influence=round(median([c.influence for c in all_claims]), 4),
        mean_coverage=round(sum(c.coverage for c in all_claims) / len(all_claims), 4),
        n_synthesis=sum(1 for c in all_claims if c.is_synthesis),
        median_lead_time=median(leads) if leads else None,
        documents=list(docs),
    )
