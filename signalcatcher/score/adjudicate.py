"""Judge how a retrieved document relates to a claim.

This is where the benchmark either measures something or fools itself. The judge
model has read most of the internet, so if simply asked "was this idea around
before 2019?" it will answer from memory, with hindsight, and the resulting
score will correlate with fame rather than priority. Three defences:

  1. It only ever sees dated excerpts, and is told to reason from them alone.
  2. It must return a verbatim `quote` from the candidate supporting its verdict.
     A relation asserted without a quote is downgraded on the way out, so
     confident hand-waving cannot score.
  3. `validate/controls.py` re-runs the same judgements with retrieval disabled.
     If the scores survive, the evidence was never doing the work.

Candidates are judged in one batched call per claim, which is not only cheaper
but better: the model can see the whole field at once and say which document is
the *earliest* real statement rather than rating each in isolation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Sequence

from ..index.retrieve import Candidate
from ..llm import LLM
from ..models import Claim, Direction, Evidence, Relation
from ..textutil import normalize_ws

EXCERPT_CHARS = 1400
MAX_CANDIDATES = 14

_WORD = re.compile(r"[a-z0-9]+")


def best_excerpt(text: str, claim: Claim, width: int = EXCERPT_CHARS) -> str:
    """Pick the passage of a candidate most likely to bear on the claim.

    Sending the first N characters instead would systematically hide the match:
    the relevant paragraph of a long essay is rarely at the top, and a judge that
    never sees the evidence returns UNRELATED, which reads downstream as
    originality.
    """
    if len(text) <= width:
        return text
    terms = set(_WORD.findall(claim.text.lower()))
    terms |= {w for fp in claim.fingerprints for w in _WORD.findall(fp.lower())}
    terms |= {w for e in claim.entities for w in _WORD.findall(e.lower())}
    terms = {t for t in terms if len(t) > 3}
    if not terms:
        return text[:width]
    step = max(width // 3, 200)
    best_score, best_at = -1, 0
    for start in range(0, max(len(text) - width, 1), step):
        window = text[start : start + width].lower()
        score = sum(1 for t in terms if t in window)
        if score > best_score:
            best_score, best_at = score, start
    # Snap to a paragraph edge so the excerpt does not begin mid-sentence.
    head = text.rfind("\n\n", max(0, best_at - 300), best_at + 1)
    if head != -1:
        best_at = head + 2
    return text[best_at : best_at + width]


SYSTEM = """\
You are adjudicating, for each candidate document, how it relates to a target \
CLAIM. Judge only what the excerpt shows.

Relations, strongest first:
- identical: the candidate asserts the same proposition, in nearly the same terms
- paraphrase: the candidate asserts the same proposition in different words
- subsumes: the candidate states a more general principle that logically entails \
the claim, so the claim adds nothing new beyond applying it
- partial: the candidate contains a genuine component of the claim, but not the \
whole of it (for example one of two ideas the claim joins together)
- topical: same subject matter, but does not assert the claim
- unrelated: no meaningful connection

Rules that matter:

MERE TOPICALITY IS NOT A MATCH. Two documents about AI regulation do not match \
unless the candidate asserts the same proposition. Most candidates are topical \
or unrelated; say so. Over-matching destroys this measurement.

QUOTE OR DOWNGRADE. For any relation of partial or stronger you must supply \
`quote`: text copied verbatim from the excerpt that carries the claim. If you \
cannot find such a span, the relation is at most topical. Never quote the claim \
back; quote the candidate.

DIRECTION OF ENTAILMENT. `subsumes` requires the candidate to be MORE general \
than the claim. A narrower or merely adjacent statement is `partial`.

ATTRIBUTION. Set `attributes_source` true only if the excerpt visibly credits \
the target source -- naming the author or publication, or quoting them. Judging \
a claim's later spread, an unattributed match still counts as spread; \
attribution is recorded separately, not required.

`confidence` is 0..1 in your relation judgement given only this excerpt. If the \
excerpt is truncated mid-argument, or you are guessing, say so and go low."""

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
                                 "enum": ["identical", "paraphrase", "subsumes",
                                          "partial", "topical", "unrelated"]},
                    "confidence": {"type": "number"},
                    "quote": {"type": "string"},
                    "rationale": {"type": "string"},
                    "attributes_source": {"type": "boolean"},
                },
                "required": ["candidate_index", "relation", "confidence", "quote",
                             "rationale", "attributes_source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgements"],
    "additionalProperties": False,
}


def _quote_supported(quote: str, excerpt: str) -> bool:
    """Check the quote really occurs in what we showed the model."""
    q = normalize_ws(quote).lower()
    if len(q) < 12:
        return False
    hay = normalize_ws(excerpt).lower()
    if q in hay:
        return True
    # Tolerate light elision/typographic drift, but require most of it to land.
    words = q.split()
    if len(words) < 4:
        return False
    hits = sum(1 for i in range(len(words) - 3)
               if " ".join(words[i : i + 4]) in hay)
    return hits >= max(1, (len(words) - 3) // 2)


def adjudicate(
    llm: LLM,
    claim: Claim,
    claim_date: datetime,
    candidates: Sequence[Candidate],
    direction: Direction,
    source_name: str = "",
) -> list[Evidence]:
    """Judge candidates against a claim, returning one Evidence per candidate."""
    cands = list(candidates)[:MAX_CANDIDATES]
    if not cands:
        return []

    when = ("BEFORE the claim was published (candidates are potential prior art)"
            if direction is Direction.PRIOR else
            "AFTER the claim was published (candidates are potential later uptake)")
    blocks = []
    excerpts = []
    for i, c in enumerate(cands):
        ex = best_excerpt(c.doc.text, claim)
        excerpts.append(ex)
        fp = (f"\nVERBATIM PHRASES FROM THE CLAIM'S SOURCE FOUND IN THIS DOCUMENT: "
              f"{c.fingerprint_hits}" if c.fingerprint_hits else "")
        blocks.append(
            f"[{i}] {c.doc.title}\n"
            f"Published: {c.doc.published_at.date().isoformat()}  ({c.doc.url}){fp}\n"
            f"EXCERPT:\n{ex}\n"
        )

    user = (
        f"TARGET CLAIM: {claim.text}\n"
        f"CLAIM KIND: {claim.kind.value}\n"
        f"PUBLISHED BY: {source_name or 'unknown source'} on "
        f"{claim_date.date().isoformat()}\n"
        f"DISTINCTIVE PHRASING FROM THE ORIGINAL: {claim.fingerprints or 'none'}\n\n"
        f"The {len(cands)} candidate documents below were all published {when}.\n\n"
        + "\n---\n".join(blocks)
        + "\n\nReturn one judgement per candidate, using its index."
    )

    out = llm.json(SYSTEM, user, SCHEMA, max_tokens=12000, grounded=True)
    if not out:
        return []

    evidence: list[Evidence] = []
    for j in out.get("judgements", []):
        idx = j.get("candidate_index")
        if not isinstance(idx, int) or not (0 <= idx < len(cands)):
            continue
        c = cands[idx]
        try:
            rel = Relation(j.get("relation", "unrelated"))
        except ValueError:
            rel = Relation.UNRELATED
        conf = max(0.0, min(1.0, float(j.get("confidence", 0.5))))
        quote = normalize_ws(j.get("quote") or "")

        # Enforce the quote requirement rather than trusting the instruction.
        # An unsupported match is the exact failure mode that would let a
        # remembered association masquerade as retrieved evidence.
        if rel in (Relation.IDENTICAL, Relation.PARAPHRASE, Relation.SUBSUMES,
                   Relation.PARTIAL):
            if not _quote_supported(quote, excerpts[idx]):
                rel = Relation.TOPICAL
                conf = min(conf, 0.3)
                quote = ""

        evidence.append(Evidence(
            id=Evidence.make_id(claim.id, c.doc.url),
            claim_id=claim.id, direction=direction, relation=rel,
            doc_id=c.doc.id, url=c.doc.url, title=c.doc.title,
            published_at=c.doc.published_at, source_id=c.doc.source_id,
            confidence=conf, rationale=normalize_ws(j.get("rationale") or "")[:600],
            quote=quote[:400],
            attributes_source=bool(j.get("attributes_source", False)),
            fingerprint_hits=c.fingerprint_hits,
            retriever="+".join(sorted(c.retrievers)), rank=c.best_rank,
        ))
    return evidence
