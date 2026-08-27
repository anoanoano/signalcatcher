"""The anticipation question: does A's content predict B's?

This is deliberately NOT a causal question. Whether B read A is unknowable from
text and is not what is being measured. The question is whether the claim,
stated at t1, semantically anticipates what a later document says -- by stating
it, entailing it, or predicting its direction. Convergence counts exactly as
much as transmission: a writer who saw where the discourse was going without
influencing anyone was still ahead of it.

Relations, and how they weight into prevalence:

    states                   1.0   the document asserts the claim itself
    entails                  0.9   the document's content follows from the
                                   claim, though it never states it -- this is
                                   where credit for IMPLICIT content lands
    anticipates_directionally 0.5  the claim's framework predicts the direction
                                   of what the document reports or argues
    partially_anticipates    0.3   a component, not the substance
    orthogonal               0.0   same neighbourhood, no anticipation
    contradicts             -1.0   the document is evidence the claim was wrong

`contradicts` is what gives the final measure a sign. A confidently wrong claim
must not score zero like a claim nobody engaged with; being refuted by the
subsequent record is the opposite of predicting it.

Weights are stored per-Evidence at judgement time rather than looked up at
aggregation time, so a stored score can always be traced to the exact weights
that produced it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from ..llm import LLM
from ..models import Claim, Direction, Evidence, Relation
from ..textutil import normalize_ws
from .adjudicate import _quote_supported, best_excerpt

ANTICIPATION_WEIGHTS = {
    "states": 1.0,
    "entails": 0.9,
    "anticipates_directionally": 0.5,
    "partially_anticipates": 0.3,
    "orthogonal": 0.0,
    "contradicts": -1.0,
}

# Map onto the legacy Relation enum for storage compatibility; the true relation
# and weight ride in Evidence.rationale-adjacent fields via metadata.
_LEGACY = {
    "states": Relation.IDENTICAL,
    "entails": Relation.SUBSUMES,
    "anticipates_directionally": Relation.PARTIAL,
    "partially_anticipates": Relation.PARTIAL,
    "orthogonal": Relation.TOPICAL,
    "contradicts": Relation.TOPICAL,
}

SYSTEM = """\
You are judging whether a CLAIM, made on a known date, ANTICIPATES the content
of documents published later. This is not about influence or causation -- do not
consider whether the later author read the claim. Ask only: does the later
document's content bear out, follow from, or run against what the claim said?

Relations:
- states: the document asserts the same proposition the claim makes.
- entails: the document's content follows from the claim -- if the claim is
  right, what this document reports or argues is what you would expect, and the
  claim captures WHY. Reserve this for real logical or explanatory connection.
- anticipates_directionally: the claim's framework correctly points toward what
  this document describes, without entailing it. Events developed the way the
  claim's way of seeing things suggests.
- partially_anticipates: the document bears out a component of the claim but not
  its substance.
- orthogonal: same subject area; the claim neither anticipates nor conflicts
  with it.
- contradicts: the document is evidence AGAINST the claim -- events or findings
  ran the other way, or the document credibly refutes it.

Discipline that matters:

ANTICIPATION IS NOT TOPICALITY. A claim about X does not anticipate every later
document about X. Most later documents on the same subject are orthogonal; say
so freely. The measurement is destroyed by generosity, not by strictness.

HINDSIGHT ONLY FLOWS ONE WAY. Judge what the document says against what the
claim said. Do not upgrade a vague claim because you know how things turned out;
the claim must itself carry the content that anticipates.

QUOTE OR DOWNGRADE. For any relation except orthogonal you must supply `quote`:
verbatim text from the document that bears out (or contradicts) the claim. No
quotable span means orthogonal.

`confidence` is 0..1 given only this excerpt."""

SCHEMA = {
    "type": "object",
    "properties": {
        "judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_index": {"type": "integer"},
                    "relation": {"type": "string",
                                 "enum": list(ANTICIPATION_WEIGHTS.keys())},
                    "confidence": {"type": "number"},
                    "quote": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["candidate_index", "relation", "confidence",
                             "quote", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgements"],
    "additionalProperties": False,
}

BATCH = 12


def judge_anticipation(
    llm: LLM, claim: Claim, anchor_date: datetime, candidates: Sequence,
) -> list[dict]:
    """Judge candidate later documents. Returns dicts with the TRUE relation,
    weight, confidence, quote -- plus a legacy Evidence for persistence."""
    out: list[dict] = []
    cands = list(candidates)
    for at in range(0, len(cands), BATCH):
        chunk = cands[at : at + BATCH]
        blocks, excerpts = [], []
        for i, doc in enumerate(chunk):
            ex = best_excerpt(doc.text, claim)
            excerpts.append(ex)
            blocks.append(f"[{i}] {doc.title}\n"
                          f"Published: {doc.published_at.date().isoformat()}\n"
                          f"EXCERPT:\n{ex}")
        resp = llm.json(
            SYSTEM,
            f"CLAIM (made {anchor_date.date().isoformat()}): {claim.text}\n\n"
            f"Later documents:\n\n" + "\n---\n".join(blocks)
            + "\n\nReturn one judgement per candidate, using its index.",
            SCHEMA, max_tokens=10000, grounded=True,
        )
        if not resp:
            continue
        for j in resp.get("judgements", []):
            idx = j.get("candidate_index")
            if not isinstance(idx, int) or not (0 <= idx < len(chunk)):
                continue
            rel = j.get("relation", "orthogonal")
            if rel not in ANTICIPATION_WEIGHTS:
                rel = "orthogonal"
            conf = max(0.0, min(1.0, float(j.get("confidence", 0.5))))
            quote = normalize_ws(j.get("quote") or "")
            # Quote enforcement, as everywhere: without a supporting span in the
            # excerpt, the judgement reverts to orthogonal.
            if rel != "orthogonal" and not _quote_supported(quote, excerpts[idx]):
                rel, conf, quote = "orthogonal", min(conf, 0.3), ""
            doc = chunk[idx]
            out.append({
                "doc": doc,
                "relation": rel,
                "weight": ANTICIPATION_WEIGHTS[rel],
                "confidence": conf,
                "quote": quote[:400],
                "rationale": normalize_ws(j.get("rationale") or "")[:400],
                "evidence": Evidence(
                    id=Evidence.make_id(claim.id, doc.url),
                    claim_id=claim.id, direction=Direction.LATER,
                    relation=_LEGACY[rel], doc_id=doc.id, url=doc.url,
                    title=doc.title, published_at=doc.published_at,
                    source_id=doc.source_id, confidence=conf,
                    rationale=f"[{rel}] " + normalize_ws(j.get("rationale") or "")[:500],
                    quote=quote[:400], retriever="anticipation",
                ),
            })
    return out
