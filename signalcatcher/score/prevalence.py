"""Prevalence curves: how common is a claim in the discourse, over time?

The core object of the redesigned measure. For a claim anchored at t1:

    surprisal = 1 - p(before t1)     how unexpected the claim was when stated
    adoption  = peak p(after) - p(before)   how far discourse moved toward it
    predictive value = surprisal x adoption x sign

The denominator is the part that has to be right. Measuring prevalence against
"all documents in the window" makes the number track the corpus's ingestion
history, not the discourse: this corpus was partly grown claim-by-claim, so
post-t1 windows are stuffed with topically targeted news, and raw prevalence
would rise for every claim regardless of merit. So prevalence is measured
against the claim's TOPICAL NEIGHBOURHOOD -- of documents talking about this
subject at all, what share express this claim? That ratio is invariant to how
much unrelated (or even related-but-only-topical) material happens to be in the
corpus.

Two embedding thresholds, calibrated on this model (bge-small: paraphrases
~0.69, same-topic ~0.55, unrelated ~0.40):

    TOPICAL_FLOOR  0.45   in the neighbourhood at all -> denominator
    EXPRESS_FLOOR  0.62   plausibly expresses the claim -> numerator

EXPRESS_FLOOR is deliberately above the same-topic band: the numerator should
under-count rather than absorb topical chatter. The LLM anticipation pass
(anticipate.py) then adjudicates the borderline band properly for the sign and
for validation of the thresholds.

The author's own documents are excluded from both sides: a writer repeating
their thesis weekly is not discourse moving toward them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ..models import Claim

TOPICAL_FLOOR = 0.45
EXPRESS_FLOOR = 0.62
MIN_NEIGHBOURHOOD = 8   # below this many topical docs, a window's p is unreliable

# Window edges in days relative to t1. Asymmetric on purpose: the pre-period
# establishes a baseline (further subdivision adds noise), while the post-period
# needs resolution to see the shape of adoption.
PRE_EDGES = (-730, -365, 0)
POST_EDGES = (0, 90, 180, 365, 730)


@dataclass
class WindowPoint:
    label: str
    start_days: int
    end_days: int
    n_topical: int
    n_expressing: int
    reliable: bool

    @property
    def p(self) -> float:
        return self.n_expressing / self.n_topical if self.n_topical else 0.0

    def to_dict(self) -> dict:
        return {"label": self.label, "days": [self.start_days, self.end_days],
                "topical": self.n_topical, "expressing": self.n_expressing,
                "p": round(self.p, 4), "reliable": self.reliable}


@dataclass
class PrevalenceCurve:
    claim_id: str
    anchor_date: datetime
    points: list[WindowPoint] = field(default_factory=list)

    @property
    def pre_points(self) -> list[WindowPoint]:
        return [w for w in self.points if w.end_days <= 0 and w.reliable]

    @property
    def post_points(self) -> list[WindowPoint]:
        return [w for w in self.points if w.start_days >= 0 and w.reliable]

    @property
    def baseline(self) -> float | None:
        """Pre-t1 prevalence, pooled across pre-windows (not averaged: windows
        with ten docs and a thousand docs should not weigh equally)."""
        pre = self.pre_points
        if not pre:
            return None
        top = sum(w.n_topical for w in pre)
        exp = sum(w.n_expressing for w in pre)
        return exp / top if top else None

    @property
    def surprisal(self) -> float | None:
        b = self.baseline
        return None if b is None else 1.0 - b

    @property
    def peak_after(self) -> float | None:
        post = self.post_points
        return max((w.p for w in post), default=None) if post else None

    @property
    def adoption(self) -> float | None:
        b, pk = self.baseline, self.peak_after
        if b is None or pk is None:
            return None
        return pk - b

    @property
    def lead_days(self) -> int | None:
        """First post-window where prevalence clears twice baseline (and is not
        trivially tiny). Midpoint of that window, as days after t1."""
        b = self.baseline
        if b is None:
            return None
        floor = max(2 * b, 0.02)
        for w in self.post_points:
            if w.p >= floor and w.n_expressing >= 2:
                return (w.start_days + w.end_days) // 2
        return None

    @property
    def decided(self) -> bool:
        return self.baseline is not None and self.peak_after is not None

    def to_dict(self) -> dict:
        return {
            "anchor_date": self.anchor_date.date().isoformat(),
            "points": [w.to_dict() for w in self.points],
            "baseline": None if self.baseline is None else round(self.baseline, 4),
            "surprisal": None if self.surprisal is None else round(self.surprisal, 4),
            "peak_after": None if self.peak_after is None else round(self.peak_after, 4),
            "adoption": None if self.adoption is None else round(self.adoption, 4),
            "lead_days": self.lead_days,
        }


def compute_curve(
    store, embedder, claim: Claim, anchor_date: datetime,
    exclude_source: str | None = None,
    topical_floor: float = TOPICAL_FLOOR,
    express_floor: float = EXPRESS_FLOOR,
) -> PrevalenceCurve:
    """Similarity of every dated document to the claim, bucketed into windows."""
    qv = embedder.embed_one(claim.text, cache_key=f"claim:{claim.id}")
    curve = PrevalenceCurve(claim_id=claim.id, anchor_date=anchor_date)
    if qv is None:
        return curve

    lo = anchor_date + timedelta(days=PRE_EDGES[0])
    hi = anchor_date + timedelta(days=POST_EDGES[-1])
    where = "d.published_ts >= ? AND d.published_ts < ?"
    params: list = [int(lo.timestamp()), int(hi.timestamp())]
    if exclude_source:
        where += " AND d.source_id != ?"
        params.append(exclude_source)
    rows = store.conn.execute(
        f"SELECT d.published_ts, e.dim, e.vec FROM embeddings e "
        f"JOIN documents d ON ('doc:' || d.id) = e.key WHERE {where}", params,
    ).fetchall()
    if not rows:
        return curve

    sims = np.vstack([embedder._unpack(r["vec"], r["dim"]) for r in rows]) @ qv
    days = np.array([(r["published_ts"] - int(anchor_date.timestamp())) / 86400.0
                     for r in rows])

    edges = list(zip(PRE_EDGES[:-1], PRE_EDGES[1:])) + list(zip(POST_EDGES[:-1], POST_EDGES[1:]))
    for a, b in edges:
        mask = (days >= a) & (days < b)
        w_sims = sims[mask]
        n_top = int((w_sims >= topical_floor).sum())
        n_exp = int((w_sims >= express_floor).sum())
        curve.points.append(WindowPoint(
            label=(f"t{a:+d}..{b:+d}d"), start_days=a, end_days=b,
            n_topical=n_top, n_expressing=n_exp,
            reliable=n_top >= MIN_NEIGHBOURHOOD,
        ))
    return curve
