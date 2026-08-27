"""Negative controls and contamination probes.

A number is not a measurement until something could have shown it to be wrong.
Each control below is a prediction that must come true if the benchmark works,
and each targets a specific way this design could be fooling itself.

  date_shift        Score a claim as if published years later, so that genuine
                    prior art is now inside its search window. Originality must
                    FALL. If it does not, the score is not tracking priority --
                    it is tracking how impressive the prose sounds.

  no_retrieval      Re-judge with the evidence removed. Scores must become
                    unstable and uninformative. If they barely move, the judge
                    is answering from training memory, and every "original"
                    verdict is really a statement about what the model has read.

  decoy_source      Score claims from a document the target source never wrote,
                    in the same period and topic. Its influence must be lower.
                    This is the base-rate control: it separates "this writer led
                    the conversation" from "this subject was being written about".

  judge_agreement   Re-judge at a different effort setting. Wide disagreement
                    means the metric is noise dressed as a decimal.

  shuffled_claim    Score a claim against a DIFFERENT document's timeline.
                    Influence must collapse; if a claim looks influential no
                    matter whose publication date it is attached to, the
                    pipeline is measuring topic popularity.

They are cheap to run because every judgement is cached: the unshifted run
usually shares most of its retrieval and adjudication with the controls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence

from ..config import Config
from ..llm import LLM
from ..models import Claim, Document
from ..score.influence import score_influence
from ..score.originality import score_originality


@dataclass
class ControlOutcome:
    name: str
    passed: bool | None          # None => not enough data to decide
    observed: float
    baseline: float
    delta: float
    expectation: str
    n: int
    detail: dict = field(default_factory=dict)

    def line(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[self.passed]
        return (f"[{mark}] {self.name:16} baseline={self.baseline:+.3f} "
                f"observed={self.observed:+.3f} delta={self.delta:+.3f} "
                f"(n={self.n})  expect: {self.expectation}")


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def date_shift_control(
    store, llm: LLM, claims: Sequence[tuple[Claim, datetime]], cfg: Config,
    embedder=None, source_id: str | None = None, source_name: str = "",
    shift_years: int = 4, baseline: Sequence[float] | None = None,
    run_id: str = "",
) -> ControlOutcome:
    """Pretend the claims were published years later than they were.

    Everything the source actually said first now sits in its own past, so a
    working originality metric has to mark it down.
    """
    shifted, base = [], []
    for claim, date in claims:
        later = date + timedelta(days=365 * shift_years)
        base.append(score_originality(
            store, llm, claim, date, cfg, embedder, source_id, source_name,
            run_id=run_id, persist=False).hi)
        shifted.append(score_originality(
            store, llm, claim, later, cfg, embedder,
            # The source's own later work is legitimate prior art for the shifted
            # copy, so the usual self-exclusion is dropped here on purpose.
            source_id=None, source_name=source_name, run_id=run_id, persist=False,
        ).hi)
    obs, bas = _mean(shifted), _mean(base)
    # Compare `hi` (= 1 - prior_strength), the purely evidence-driven term,
    # NOT the reported score. Moving the date forward enlarges the prior window,
    # which raises coverage, which raises the blended score -- so comparing
    # scores would let a coverage gain cancel out the very prior-art effect this
    # control exists to detect, and the control would clear itself.
    undecidable = bas >= 0.999 and obs >= 0.999
    return ControlOutcome(
        name="date_shift",
        passed=None if (not shifted or undecidable) else (obs < bas - 0.02),
        observed=obs, baseline=bas, delta=obs - bas, n=len(shifted),
        expectation=f"prior-art strength rises when dated {shift_years}y later",
        detail={"note": "compares 1-prior_strength, isolated from coverage",
                "undecidable_reason": "no prior art found at either date -- the "
                "corpus contains nothing relevant to these claims"
                if undecidable else ""},
    )


def no_retrieval_control(
    store, llm: LLM, claims: Sequence[tuple[Claim, datetime]], cfg: Config,
    embedder=None, source_id: str | None = None, source_name: str = "",
    baseline: Sequence[float] | None = None, run_id: str = "",
) -> ControlOutcome:
    """Take the evidence away and see whether the score notices.

    This is the contamination probe. A benchmark built on a model that already
    knows the answer will keep producing confident scores with nothing to read;
    a benchmark that is actually reading its corpus cannot.
    """
    class _Blind:
        """A store that answers every retrieval with nothing."""
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def search(self, *a, **kw):
            return []

        def search_phrase(self, *a, **kw):
            return []

    blind = _Blind(store)
    blind_scores, base = [], list(baseline or [])
    for claim, date in claims:
        if baseline is None:
            base.append(score_originality(
                store, llm, claim, date, cfg, embedder, source_id, source_name,
                run_id=run_id, persist=False).score)
        blind_scores.append(score_originality(
            blind, llm, claim, date, cfg, embedder=None, source_id=source_id,
            source_name=source_name, run_id=run_id, persist=False).score)
    obs, bas = _mean(blind_scores), _mean(base)
    if not blind_scores:
        return ControlOutcome("no_retrieval", None, 0.0, 0.0, 0.0,
                              "no claims scored", 0)
    # Two independent ways to pass, either of which shows the evidence was doing
    # the work. (a) withholding it moves the scores; or (b) without it the
    # scores collapse together, because a judge with nothing to read cannot tell
    # claims apart. The failure case is the alarming one: scores that stay put
    # AND stay spread out are being produced from the model's own memory of who
    # said what, which is precisely the contamination this benchmark must avoid.
    spread = max(blind_scores) - min(blind_scores) if len(blind_scores) > 1 else 0.0
    base_spread = (max(base) - min(base)) if len(base) > 1 else 0.0
    moved = abs(obs - bas) > 0.05
    collapsed = spread < max(0.05, base_spread * 0.5)
    return ControlOutcome(
        name="no_retrieval", passed=(moved or collapsed),
        observed=obs, baseline=bas, delta=obs - bas, n=len(blind_scores),
        expectation="withholding evidence must change or flatten the scores",
        detail={"blind_spread": round(spread, 4), "baseline_spread": round(base_spread, 4),
                "passed_via": "score shift" if moved else
                              ("collapse to uniform" if collapsed else "neither")},
    )


def decoy_source_control(
    store, llm: LLM, target: Sequence[tuple[Claim, datetime]],
    decoy: Sequence[tuple[Claim, datetime]], cfg: Config, embedder=None,
    target_source_id: str | None = None, decoy_source_id: str | None = None,
    run_id: str = "", baseline: Sequence[float] | None = None,
) -> ControlOutcome:
    """Compare the target's influence against a contemporaneous other writer.

    Without this, an influence score is unreadable: any claim about a live topic
    is followed by more writing on that topic. The decoy establishes what that
    background looks like for someone who was merely present.
    """
    tgt = list(baseline or [])
    if baseline is None:
        for claim, date in target:
            tgt.append(score_influence(
                store, llm, claim, date, cfg, embedder, target_source_id,
                run_id=run_id, persist=False).score)
    dec = [score_influence(store, llm, claim, date, cfg, embedder, decoy_source_id,
                           run_id=run_id, persist=False).score
           for claim, date in decoy]
    obs, bas = _mean(dec), _mean(tgt)
    # A base-rate comparison against a target that showed no measurable influence
    # tests nothing: 0 vs 0 is not evidence that the metric discriminates. Report
    # it as undecidable rather than letting a vacuous comparison print FAIL (or,
    # worse, PASS) and be read as a verdict on the source.
    if not dec or not tgt or bas <= 0.02:
        return ControlOutcome(
            "decoy_source", None, obs, bas, obs - bas,
            "target influence exceeds a contemporaneous decoy", len(dec),
            {"undecidable_reason": "target influence is ~0, so there is no signal "
                                   "to compare against the decoy"})
    return ControlOutcome(
        name="decoy_source", passed=(bas > obs),
        observed=obs, baseline=bas, delta=obs - bas, n=len(dec),
        expectation="target influence exceeds a contemporaneous decoy",
    )


def shuffled_claim_control(
    store, llm: LLM, claims: Sequence[tuple[Claim, datetime]], cfg: Config,
    embedder=None, source_id: str | None = None, seed: int = 0,
    baseline: Sequence[float] | None = None, run_id: str = "",
) -> ControlOutcome:
    """Attach each claim to a different claim's publication date.

    A claim that scores as influential regardless of when it was supposedly
    published is not being measured for influence; the pipeline is picking up
    how much the subject gets written about.
    """
    if len(claims) < 3:
        return ControlOutcome("shuffled_claim", None, 0.0, 0.0, 0.0,
                              "needs >=3 claims", len(claims))
    rng = random.Random(seed)
    dates = [d for _, d in claims]
    rotated = dates[1:] + dates[:1]
    rng.shuffle(rotated)
    base = list(baseline or [])
    if baseline is None:
        base = [score_influence(store, llm, c, d, cfg, embedder, source_id,
                                run_id=run_id, persist=False).score
                for c, d in claims]
    shuf = [score_influence(store, llm, c, nd, cfg, embedder, source_id,
                            run_id=run_id, persist=False).score
            for (c, _), nd in zip(claims, rotated)]
    obs, bas = _mean(shuf), _mean(base)
    if not shuf or bas <= 0.02:
        return ControlOutcome(
            "shuffled_claim", None, obs, bas, obs - bas,
            "influence falls when claims are given the wrong dates", len(shuf),
            {"undecidable_reason": "baseline influence is ~0, so there is nothing "
                                   "for the shuffle to destroy"})
    return ControlOutcome(
        name="shuffled_claim", passed=(obs < bas),
        observed=obs, baseline=bas, delta=obs - bas, n=len(shuf),
        expectation="influence falls when claims are given the wrong dates",
    )


def judge_agreement_control(
    store, claims: Sequence[tuple[Claim, datetime]], cfg: Config,
    make_llm: Callable[[str], LLM], embedder=None, source_id: str | None = None,
    efforts: tuple[str, str] = ("high", "medium"), run_id: str = "",
) -> ControlOutcome:
    """Re-score at a second effort setting and measure how far the verdicts move."""
    runs: list[list[float]] = []
    for effort in efforts:
        llm = make_llm(effort)
        runs.append([score_originality(store, llm, c, d, cfg, embedder, source_id,
                                       run_id=run_id, persist=False).score
                     for c, d in claims])
    if not runs or not runs[0]:
        return ControlOutcome("judge_agreement", None, 0.0, 0.0, 0.0,
                              "no claims scored", 0)
    diffs = [abs(a - b) for a, b in zip(runs[0], runs[1])]
    mad = _mean(diffs)
    return ControlOutcome(
        name="judge_agreement", passed=(mad < 0.15), observed=mad, baseline=0.0,
        delta=mad, n=len(diffs),
        expectation="mean abs. difference between judges < 0.15",
        detail={"max_diff": round(max(diffs), 3)},
    )
