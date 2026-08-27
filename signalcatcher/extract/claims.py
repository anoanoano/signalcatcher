"""Claim extraction: documents -> the substantive propositions they carry.

Everything downstream depends on this being right. A claim has to be *portable*:
stated so that it can be looked for in some other writer's work, years earlier or
later, without carrying along the vocabulary of the document it came from. A
claim like "the author argues the policy is misguided" is unscoreable -- there is
no way to search for it. "Rent control reduces long-run housing supply by
suppressing new construction" can be found in a 1970s economics paper or a 2024
city-council transcript, and that is the whole game.

Implicit claims are extracted too, and flagged. A newspaper's real contribution
is often a presupposition it never states outright -- the frame it takes for
granted -- and that frame propagates. It is also the noisiest thing here, which
is why `explicit` is stored rather than folded into the score.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..llm import LLM
from ..models import Claim, ClaimKind, Document
from ..textutil import normalize_ws

MAX_CHARS = 60_000  # ~15k tokens; longer documents are chunked

SYSTEM = """\
You extract the substantive intellectual content of a document so it can be \
compared against what was published before and after it.

For each claim you extract, obey these rules:

1. STANDALONE. Write it so a reader who has never seen this document, and does \
not know its author or topic, understands exactly what is being asserted. Name \
the entities. Never write "the author", "this piece", "the report", "he", "it", \
or "the above". A claim mentioning "the study" is useless; name the study.

2. CHECKABLE ELSEWHERE. The claim must be the kind of thing that could \
independently appear in some other writer's work. Do not describe the document \
("the piece is a rebuttal"); state what it asserts about the world.

3. SUBSTANTIVE. Skip pleasantries, admin, self-promotion, housekeeping, \
subscription pitches, and pure summary of someone else's argument unless the \
document endorses, extends or attacks it.

4. ATOMIC. One proposition per claim. Split conjunctions.

Claim kinds:
- fact: an empirical assertion that could be corroborated or refuted
- thesis: a causal, interpretive or evaluative argument
- frame: a coinage, metaphor, category or way of characterising something, \
whose spread is visible in later vocabulary
- prediction: a falsifiable forecast; set horizon_days if a timeframe is implied
- synthesis: an explicit connection drawn between two previously separate \
bodies of work or domains

Also extract IMPLICIT claims: propositions the document presupposes, assumes or \
entails but never states. Set explicit=false for these. Be conservative -- an \
implicit claim must be one the author would readily endorse, not one you can \
merely construct.

fingerprints: 2-6 word phrases copied CHARACTER-FOR-CHARACTER from the document \
that are distinctive to how this claim is worded -- coinages, unusual metaphors, \
memorable formulations. These are used to detect later verbatim reuse, so they \
must be exact substrings of the text, not paraphrases. Prefer rare, specific \
wording. If the claim contains no distinctive phrasing, return an empty list; \
do not invent one.

salience: 0..1, how central the claim is to the document's purpose. Reserve \
values above 0.8 for claims the document exists in order to make.

Return 5-25 claims for a typical article, weighted toward the ones that matter."""

SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["fact", "thesis", "frame", "prediction", "synthesis"]},
                    "explicit": {"type": "boolean"},
                    "salience": {"type": "number"},
                    "falsifiable": {"type": "boolean"},
                    "horizon_days": {"type": ["integer", "null"]},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "fingerprints": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "kind", "explicit", "salience", "falsifiable",
                             "horizon_days", "entities", "fingerprints"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def _chunks(text: str, size: int = MAX_CHARS) -> list[str]:
    """Split on paragraph boundaries so no claim straddles a cut."""
    if len(text) <= size:
        return [text]
    paras = text.split("\n\n")
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > size and cur:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _norm_for_match(s: str) -> str:
    """Fold quotes, dashes and whitespace so verbatim matching is not defeated by
    typographic substitution."""
    s = s.lower()
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"',
                                   "”": '"', "–": "-", "—": "-",
                                   "…": "..."}))
    return re.sub(r"\s+", " ", s).strip()


def validate_fingerprints(fingerprints: Iterable[str], doc_text: str) -> tuple[list[str], int]:
    """Keep only fingerprints that really appear in the document.

    Models paraphrase when asked to quote. An unverified fingerprint is worse
    than none: the diffusion stage treats a fingerprint hit as strong evidence of
    real transmission, so a hallucinated phrase would either match nothing (mild)
    or match generic wording and manufacture influence (not mild). Everything
    that is not an exact substring is dropped.
    """
    hay = _norm_for_match(doc_text)
    kept, dropped = [], 0
    for fp in fingerprints:
        fp = (fp or "").strip().strip('"“”')
        words = fp.split()
        # Very short phrases are not distinctive enough to evidence transmission.
        if len(words) < 2 or len(fp) < 8:
            dropped += 1
            continue
        if _norm_for_match(fp) in hay:
            kept.append(normalize_ws(fp))
        else:
            dropped += 1
    return kept, dropped


def extract_claims(llm: LLM, doc: Document, max_chunks: int = 4) -> tuple[list[Claim], dict]:
    """Extract claims from a document. Returns (claims, diagnostics)."""
    chunks = _chunks(doc.text)[:max_chunks]
    raw_claims: list[dict] = []
    for i, chunk in enumerate(chunks):
        part = f" (part {i + 1} of {len(chunks)})" if len(chunks) > 1 else ""
        user = (
            f"DOCUMENT{part}\n"
            f"Title: {doc.title}\n"
            f"Published: {doc.published_at.date().isoformat()}\n"
            f"---\n{chunk}\n---\n\n"
            "Extract the substantive claims, following the rules exactly."
        )
        # Extraction reads only the document in front of it, so the grounding
        # preamble (which is about dating evidence) does not apply here.
        out = llm.json(SYSTEM, user, SCHEMA, max_tokens=16000, grounded=False)
        if out and out.get("claims"):
            raw_claims.extend(out["claims"])

    claims: list[Claim] = []
    seen: set[str] = set()
    dropped_fp = 0
    for rc in raw_claims:
        text = normalize_ws(rc.get("text") or "")
        if len(text) < 15:
            continue
        key = _norm_for_match(text)
        if key in seen:  # chunk overlap and restatement produce duplicates
            continue
        seen.add(key)
        fps, d = validate_fingerprints(rc.get("fingerprints") or [], doc.text)
        dropped_fp += d
        try:
            kind = ClaimKind(rc.get("kind", "thesis"))
        except ValueError:
            kind = ClaimKind.THESIS
        claims.append(Claim(
            id=Claim.make_id(doc.id, text),
            doc_id=doc.id,
            kind=kind,
            text=text,
            entities=[normalize_ws(e) for e in (rc.get("entities") or [])][:12],
            fingerprints=fps[:6],
            explicit=bool(rc.get("explicit", True)),
            salience=max(0.0, min(1.0, float(rc.get("salience", 0.5)))),
            falsifiable=bool(rc.get("falsifiable", False)),
            horizon_days=rc.get("horizon_days"),
        ))
    diag = {
        "chunks": len(chunks), "raw": len(raw_claims), "kept": len(claims),
        "fingerprints_dropped": dropped_fp,
        "fingerprints_kept": sum(len(c.fingerprints) for c in claims),
    }
    return claims, diag
