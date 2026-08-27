"""Predictive value: was the claim unexpected when written, and did the
discourse then move toward it?

This replaces originality-and-influence-as-separate-metrics. They are the prior
and posterior of one measure:

    surprisal   = 1 - p(expressed | topical neighbourhood, before t1)
    adoption    = peak p(expressed | topical neighbourhood, after t1) - baseline
    vindication = signed balance of later evidence: did the record bear the
                  claim out or run against it?  (-1 .. +1)

    predictive_value = surprisal x max(0, adoption) x (1 + vindication) / 2

A claim everyone was already making scores ~0 via surprisal. A claim nobody
ever picked up scores 0 via adoption. A claim the record refuted is crushed by
the vindication factor -- deliberately distinct from the zero of being ignored,
and reported alongside so "wrong" and "unheard" are never conflated.

Both the baseline and the adoption numerators are LLM-adjudicated, not
threshold-crossed: measured on a real claim, documents *expressing* it and
documents merely *discussing the topic* are inseparable in embedding space
(0.63-0.67 for both). Embeddings rank candidates and fix the denominator; the
judge decides expression. Same standard on both sides of t1 -- the baseline
question "was this already being said" is the prior-art question, so the
pre-window reuses the prior-art adjudicator verbatim.

p is therefore "weighted expression rate among the top-K most-plausible
candidates per window", K fixed across windows. It under-counts absolute
prevalence (expressers below rank K are missed) but under-counts it
*consistently*, and every derived quantity is a difference across windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ..config import RELATION_WEIGHTS, Config
from ..llm import LLM
from ..models import Claim, Direction, Document
from .adjudicate import adjudicate
from .anticipate import judge_anticipation
from .firstuse import FirstUse, find_first_use
from ..index.retrieve import Candidate

TOPICAL_FLOOR = 0.45
K_PER_WINDOW = 10
MIN_JUDGED = 4          # fewer judged candidates than this -> window undecided

PRE_WINDOWS = ((-730, -365), (-365, 0))
POST_WINDOWS = ((0, 90), (90, 180), (180, 365), (365, 730), (730, 1460))


@dataclass
class JudgedWindow:
    start_days: int
    end_days: int
    n_topical: int
    n_judged: int
    expression: float        # weighted positive expression rate, 0..1
    support: float           # sum of positive weight x confidence
    refute: float            # sum of |contradicts| weight x confidence
    examples: list[dict] = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.n_judged >= MIN_JUDGED

    @property
    def label(self) -> str:
        return f"t{self.start_days:+d}..{self.end_days:+d}d"

    def to_dict(self) -> dict:
        # `examples` is the adoption trail -- the whole point of the readout is
        # that a reader can see WHERE a claim resurfaced; dropping it here left
        # the report with numbers and no receipts.
        return {"window": self.label, "topical": self.n_topical,
                "judged": self.n_judged, "expression": round(self.expression, 4),
                "support": round(self.support, 3), "refute": round(self.refute, 3),
                "reliable": self.reliable, "examples": self.examples}


@dataclass
class PredictiveResult:
    claim_id: str
    anchor_date: datetime
    first_use: FirstUse
    pre: list[JudgedWindow]
    post: list[JudgedWindow]
    baseline: float | None
    surprisal: float | None
    peak_after: float | None
    adoption: float | None
    vindication: float | None
    predictive_value: float | None
    best_anticipation: dict | None
    contradiction: dict | None

    def to_dict(self) -> dict:
        return {
            "anchor_date": self.anchor_date.date().isoformat(),
            "anchor_moved_back_days": self.first_use.moved_days,
            "windows_pre": [w.to_dict() for w in self.pre],
            "windows_post": [w.to_dict() for w in self.post],
            "baseline": _r(self.baseline), "surprisal": _r(self.surprisal),
            "peak_after": _r(self.peak_after), "adoption": _r(self.adoption),
            "vindication": _r(self.vindication),
            "predictive_value": _r(self.predictive_value),
            "best_anticipation": self.best_anticipation,
            "contradiction": self.contradiction,
        }


def _r(x):
    return None if x is None else round(x, 4)


def _topical_candidates(store, embedder, claim, t_start, t_end,
                        exclude_source) -> tuple[list[Document], int]:
    """Top-K docs by similarity within a window; also the neighbourhood size."""
    qv = embedder.embed_one(claim.text, cache_key=f"claim:{claim.id}")
    if qv is None:
        return [], 0
    where = "d.published_ts >= ? AND d.published_ts < ?"
    params: list = [int(t_start.timestamp()), int(t_end.timestamp())]
    if exclude_source:
        where += " AND d.source_id != ?"
        params.append(exclude_source)
    rows = store.conn.execute(
        f"SELECT d.id, e.dim, e.vec FROM embeddings e "
        f"JOIN documents d ON ('doc:' || d.id) = e.key WHERE {where}", params,
    ).fetchall()
    if not rows:
        return [], 0
    sims = np.vstack([embedder._unpack(r["vec"], r["dim"]) for r in rows]) @ qv
    above = [(float(sims[i]), rows[i]["id"]) for i in range(len(rows))
             if sims[i] >= TOPICAL_FLOOR]
    above.sort(reverse=True)
    docs = [d for d in (store.get_document(doc_id) for _, doc_id in above[:K_PER_WINDOW]) if d]
    return docs, len(above)


def score_predictive(
    store, llm: LLM, claim: Claim, doc: Document, cfg: Config | None = None,
    embedder=None, run_id: str = "", persist: bool = True, progress=None,
    pre_windows=None, post_windows=None,
) -> PredictiveResult:
    """`pre_windows`/`post_windows` override the default horizons. A claim about
    a breaking event needs month-scale windows -- its value is being right in
    week two, not year three -- while a thesis claim needs years. One window set
    per unit, fixed across its claims, so scores stay comparable within a unit."""
    say = progress or (lambda m: None)
    pre_w = tuple(pre_windows) if pre_windows else PRE_WINDOWS
    post_w = tuple(post_windows) if post_windows else POST_WINDOWS
    src_id = doc.source_id

    # ---- Stage 0: anchor at true first use ---------------------------------
    fu = find_first_use(store, llm, claim, doc, embedder=embedder)
    t1 = fu.anchor_date
    if fu.moved:
        say(f"    anchor moved back {fu.moved_days}d to {t1.date()} "
            f"({fu.n_prior_statements} earlier statements by the author)")

    # ---- pre-t1: the baseline, judged with the prior-art adjudicator -------
    pre: list[JudgedWindow] = []
    for a, b in pre_w:
        docs, n_top = _topical_candidates(
            store, embedder, claim, t1 + timedelta(days=a), t1 + timedelta(days=b), src_id)
        if not docs:
            pre.append(JudgedWindow(a, b, n_top, 0, 0.0, 0.0, 0.0))
            continue
        ev = adjudicate(llm, claim, t1,
                        [Candidate(doc=d, fused_score=0.0) for d in docs],
                        Direction.PRIOR)
        if persist and ev and run_id:
            store.add_evidence(ev, run_id)
        w = [RELATION_WEIGHTS.get(e.relation.value, 0.0) * e.confidence for e in ev]
        pre.append(JudgedWindow(a, b, n_top, len(ev),
                                float(np.mean(w)) if w else 0.0,
                                float(np.sum(w)), 0.0))

    # ---- post-t1: adoption and sign, judged with the anticipation taxonomy --
    post: list[JudgedWindow] = []
    all_judged: list[dict] = []
    for a, b in post_w:
        docs, n_top = _topical_candidates(
            store, embedder, claim, t1 + timedelta(days=a), t1 + timedelta(days=b), src_id)
        if not docs:
            post.append(JudgedWindow(a, b, n_top, 0, 0.0, 0.0, 0.0))
            continue
        judged = judge_anticipation(llm, claim, t1, docs)
        all_judged.extend(judged)
        if persist and judged and run_id:
            store.add_evidence([j["evidence"] for j in judged], run_id)
        pos = [max(0.0, j["weight"]) * j["confidence"] for j in judged]
        neg = [abs(j["weight"]) * j["confidence"] for j in judged
               if j["relation"] == "contradicts"]
        post.append(JudgedWindow(
            a, b, n_top, len(judged),
            float(np.mean(pos)) if pos else 0.0,
            float(np.sum(pos)), float(np.sum(neg)),
            examples=[{"title": j["doc"].title[:90],
                       "date": j["doc"].published_at.date().isoformat(),
                       "source": (lambda sr: sr.name if sr else "")(
                           store.get_source(j["doc"].source_id)),
                       "url": j["doc"].url,
                       "relation": j["relation"],
                       "confidence": round(j["confidence"], 2),
                       "quote": j["quote"][:280]}
                      for j in judged if j["weight"] != 0.0],
        ))

    # ---- combine ------------------------------------------------------------
    pre_ok = [w for w in pre if w.reliable]
    post_ok = [w for w in post if w.reliable]
    baseline = (sum(w.expression * w.n_judged for w in pre_ok)
                / sum(w.n_judged for w in pre_ok)) if pre_ok else None
    surprisal = None if baseline is None else 1.0 - baseline
    peak_after = max((w.expression for w in post_ok), default=None) if post_ok else None
    adoption = None if (baseline is None or peak_after is None) else peak_after - baseline

    support = sum(w.support for w in post)
    refute = sum(w.refute for w in post)
    vindication = ((support - refute) / (support + refute)
                   if (support + refute) > 1e-9 else None)

    pv = None
    if surprisal is not None and adoption is not None:
        sign_factor = 1.0 if vindication is None else (1.0 + vindication) / 2.0
        pv = surprisal * max(0.0, adoption) * sign_factor

    strongest = max((j for j in all_judged if j["weight"] > 0),
                    key=lambda j: j["weight"] * j["confidence"], default=None)
    contra = max((j for j in all_judged if j["relation"] == "contradicts"),
                 key=lambda j: j["confidence"], default=None)

    def _ex(j):
        return None if j is None else {
            "title": j["doc"].title[:80], "date": j["doc"].published_at.date().isoformat(),
            "url": j["doc"].url, "relation": j["relation"],
            "confidence": round(j["confidence"], 2), "quote": j["quote"][:200],
        }

    result = PredictiveResult(
        claim_id=claim.id, anchor_date=t1, first_use=fu, pre=pre, post=post,
        baseline=baseline, surprisal=surprisal, peak_after=peak_after,
        adoption=adoption, vindication=vindication, predictive_value=pv,
        best_anticipation=_ex(strongest), contradiction=_ex(contra),
    )
    if persist and run_id:
        store.put_score(run_id, "claim", claim.id, "predictive_value", pv,
                        None, None, result.to_dict())
    return result
