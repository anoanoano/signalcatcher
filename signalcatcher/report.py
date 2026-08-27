"""Render a run as something a person can audit.

Every headline number is followed by the evidence that produced it -- the dated
prior document that defeated a novelty claim, the first outlet that picked a
claim up, the phrase that travelled. That is the point: a publisher should be
able to disagree with a specific verdict, not just with the arithmetic.
"""

from __future__ import annotations

import json
from typing import TextIO

from .pipeline import RunReport

BAR = "=" * 78
SUB = "-" * 78


def _bar(v: float, width: int = 22) -> str:
    n = int(max(0.0, min(1.0, v)) * width)
    return "#" * n + "." * (width - n)


def render(rep: RunReport, out: TextIO, max_claims: int = 8) -> None:
    s = rep.source
    w = out.write
    w(f"\n{BAR}\nSIGNALCATCHER  |  {s.name}\n{BAR}\n")
    w(f"run {rep.run_id}   config {rep.config_hash}   {rep.elapsed_s}s\n")
    w(f"corpus: {rep.corpus_docs} docs, {rep.corpus_span[0]} .. {rep.corpus_span[1]}\n")
    w(f"retrieval: lexical + fingerprint + dense ({rep.embed_status})\n")

    if rep.warnings:
        w(f"\n{SUB}\nCAVEATS -- read these before quoting any number above\n{SUB}\n")
        for wn in rep.warnings:
            w(f"  ! {wn}\n")

    w(f"\n{SUB}\nSOURCE SCORECARD\n{SUB}\n")
    w(f"  headline signal    {s.headline:.3f}  {_bar(s.headline)}\n")
    w(f"    peak (top 10%)   {s.peak_signal:.3f}  {_bar(s.peak_signal)}   best work\n")
    w(f"    hit rate         {s.hit_rate:.3f}  {_bar(s.hit_rate)}   claims clearing the bar\n")
    w(f"    typical (median) {s.typical_signal:.3f}  {_bar(s.typical_signal)}   routine work\n")
    w(f"\n  median originality {s.median_originality:.3f}\n")
    w(f"  median influence   {s.median_influence:.3f}\n")
    w(f"  search coverage    {s.mean_coverage:.3f}   (how much of the prior record was actually searched)\n")
    w(f"  novel syntheses    {s.n_synthesis}\n")
    if s.median_lead_time is not None:
        w(f"  median lead time   {s.median_lead_time:.0f} days ahead of the next independent source\n")
    w(f"  scored             {s.n_claims} claims across {s.n_docs} documents\n")

    w(f"\n{SUB}\nDOCUMENTS\n{SUB}\n")
    for d in sorted(s.documents, key=lambda d: -d.signal):
        w(f"  {d.published}  signal={d.signal:.3f}  orig={d.originality:.3f}  "
          f"infl={d.influence:.3f}  ({d.n_claims} claims)\n    {d.title[:70]}\n")

    w(f"\n{SUB}\nSTRONGEST CLAIMS  (the receipts)\n{SUB}\n")
    allc = sorted((c for d in s.documents for c in d.claims), key=lambda c: -c.signal)
    for c in allc[:max_claims]:
        tag = "synthesis" if c.is_synthesis else c.kind
        vis = "explicit" if c.explicit else "implicit"
        w(f"\n  [{tag}/{vis}] signal={c.signal:.3f}\n")
        w(f"    {c.text[:150]}\n")
        w(f"    originality {c.originality:.2f} (range {c.originality_lo:.2f}-"
          f"{c.originality_hi:.2f}, coverage {c.coverage:.2f})\n")
        lead = f"{c.lead_time_days}d lead" if c.lead_time_days is not None else "no independent uptake found"
        w(f"    influence   {c.influence:.2f} -- {c.uptake_sources} independent sources, {lead}\n")

    w(f"\n{SUB}\nVALIDATION -- does this measure anything?\n{SUB}\n")
    if not rep.controls:
        w("  (controls not run)\n")
    else:
        for c in rep.controls:
            w(f"  {c.line()}\n")
            why = c.detail.get("undecidable_reason")
            if why:
                w(f"       -> {why}\n")
            via = c.detail.get("passed_via")
            if via:
                w(f"       -> passed via: {via}\n")
        failed = [c for c in rep.controls if c.passed is False]
        undecided = [c for c in rep.controls if c.passed is None]
        w("\n")
        if failed:
            w(f"  {len(failed)} control(s) FAILED. The scores above are not trustworthy\n"
              f"  as evidence of priority or influence until these are explained.\n")
        elif undecided:
            w(f"  {len(undecided)} of {len(rep.controls)} control(s) could not be decided\n"
              f"  on this corpus -- they had no signal to test, which is a statement\n"
              f"  about corpus coverage, not about the source. Treat the scores above\n"
              f"  as provisional until the corpus can decide them.\n")
        else:
            w("  All controls passed: scores respond to dates and to evidence,\n"
              "  and exceed the topical background rate.\n")

    st = rep.llm_stats
    w(f"\n{SUB}\ncost: {st.get('calls',0)} calls, {st.get('cache_hits',0)} cached, "
      f"~${st.get('est_cost_usd',0):.2f}\n{BAR}\n")


def to_json(rep: RunReport) -> str:
    return json.dumps({
        "run_id": rep.run_id,
        "config_hash": rep.config_hash,
        "corpus": {"docs": rep.corpus_docs, "span": rep.corpus_span},
        "embed_status": rep.embed_status,
        "warnings": rep.warnings,
        "source": rep.source.to_dict(),
        "documents": [
            {
                "title": d.title, "url": d.url, "published": d.published,
                "signal": d.signal, "originality": d.originality,
                "influence": d.influence,
                "claims": [
                    {
                        "text": c.text, "kind": c.kind, "explicit": c.explicit,
                        "salience": c.salience, "signal": round(c.signal, 4),
                        "originality": c.originality,
                        "originality_range": [c.originality_lo, c.originality_hi],
                        "coverage": c.coverage, "influence": c.influence,
                        "lead_time_days": c.lead_time_days,
                        "uptake_sources": c.uptake_sources,
                        "is_synthesis": c.is_synthesis,
                    } for c in d.claims
                ],
            } for d in rep.source.documents
        ],
        "controls": [
            {"name": c.name, "passed": c.passed, "observed": c.observed,
             "baseline": c.baseline, "delta": c.delta, "n": c.n,
             "expectation": c.expectation, "detail": c.detail}
            for c in rep.controls
        ],
        "llm": rep.llm_stats,
        "elapsed_s": rep.elapsed_s,
    }, indent=2)
