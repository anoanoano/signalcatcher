"""What is a source actually worth to a model?

Two questions, measured separately, because they are worth different things to
different buyers:

  INFERENCE-TIME VALUE  Does putting this source in the context window make the
                        model answer better than it otherwise would?

  TRAINING-SET VALUE    Does the model already know this, unaided? For a claim
                        that originated with this source, a correct closed-book
                        answer is evidence the source's contribution is already
                        priced into the weights -- it was trained on, and it
                        stuck.

The measurement only means something because of the third condition. Comparing
"with the source" against "with nothing" mostly measures whether having context
helps, which it always does. So every question is also asked with a *decoy*
context: the same amount of topically matched material from the same period,
written by other people. The source's value is the excess over that.

  inference_value = score(with source) - max(score(closed book), score(decoy))

A source that adds nothing a contemporary wasn't also saying scores near zero,
which is the correct answer even if the writing is excellent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean
from typing import Sequence

from ..config import Config
from ..index.retrieve import expand_queries, retrieve
from ..llm import LLM
from ..models import Claim, Document

QUESTION_SYSTEM = """\
You write a single exam question that can only be answered correctly by someone \
who has read a specific claim.

Requirements:

- The question must NOT contain the answer, and must not paraphrase the claim so \
closely that the wording gives it away. Someone who has not encountered the \
claim should find it genuinely hard.
- It must be specific enough to have a determinate answer -- not "what do people \
think about X".
- It must be answerable in two or three sentences.
- Write `gold` as the correct answer, stated plainly and completely.
- Write `key_points`: the 2-4 things an answer must contain to count as correct.

If the claim is too vague, too self-referential, or too trivially general to \
support such a question, set `usable` to false and leave the other fields empty."""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
        "gold": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["usable", "question", "gold", "key_points"],
    "additionalProperties": False,
}

ANSWER_SYSTEM = """\
Answer the question in at most three sentences, using the provided context if \
any is given. If you do not know, say so plainly rather than guessing -- a \
confident wrong answer is worse than an admission of ignorance."""

GRADE_SYSTEM = """\
Grade a candidate answer against the key points of a known-correct answer.

Return `covered`: how many key points the candidate actually gets right. A point \
half-gestured-at does not count. Contradicting a key point does not count.
Return `score`: covered / total, adjusted down if the answer is confidently \
wrong about anything material.
Return `hedged`: true if the answer declines to commit or says it does not know."""

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "integer"},
        "total": {"type": "integer"},
        "score": {"type": "number"},
        "hedged": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["covered", "total", "score", "hedged", "note"],
    "additionalProperties": False,
}


@dataclass
class ClaimAblation:
    claim_id: str
    claim_text: str
    question: str
    closed_score: float
    source_score: float
    decoy_score: float | None      # None => no decoy context existed to test with
    closed_hedged: bool
    n_decoy_docs: int

    @property
    def decoy_controlled(self) -> bool:
        return self.decoy_score is not None

    @property
    def inference_value(self) -> float:
        """Excess of the source over the best alternative context available.

        When no decoy context could be assembled the comparison falls back to
        closed-book. That is a WEAKER claim -- it shows the source beat the
        model's own knowledge, not that it beat what contemporaries were saying
        -- so `decoy_controlled` is carried alongside and reported separately.
        Averaging an unavailable decoy in as 0.0 would silently inflate the
        source's apparent value, which is exactly the number a buyer would
        challenge first.
        """
        alternative = self.closed_score
        if self.decoy_score is not None:
            alternative = max(alternative, self.decoy_score)
        return self.source_score - alternative

    @property
    def already_internalised(self) -> bool:
        """The model knew it closed-book -- evidence it is already in the weights."""
        return self.closed_score >= 0.6 and not self.closed_hedged


@dataclass
class AblationResult:
    source_name: str
    n_claims: int
    mean_inference_value: float
    mean_closed: float
    mean_source: float
    mean_decoy: float | None       # None => decoy control never ran
    internalised_rate: float
    n_decoy_controlled: int = 0
    mean_inference_value_controlled: float | None = None
    items: list[ClaimAblation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source_name, "n_claims": self.n_claims,
            "mean_inference_value": round(self.mean_inference_value, 4),
            "mean_closed_book": round(self.mean_closed, 4),
            "mean_with_source": round(self.mean_source, 4),
            "mean_with_decoy": None if self.mean_decoy is None
                               else round(self.mean_decoy, 4),
            "n_decoy_controlled": self.n_decoy_controlled,
            "mean_inference_value_controlled":
                None if self.mean_inference_value_controlled is None
                else round(self.mean_inference_value_controlled, 4),
            "decoy_caveat": None if self.n_decoy_controlled == self.n_claims else
                f"only {self.n_decoy_controlled}/{self.n_claims} claims had any "
                f"contemporaneous material from other sources in the corpus to "
                f"compare against; the rest fall back to a closed-book baseline, "
                f"which cannot distinguish this source from its contemporaries",
            "internalised_rate": round(self.internalised_rate, 4),
            "items": [
                {"claim": i.claim_text[:160], "question": i.question,
                 "closed": i.closed_score, "source": i.source_score,
                 "decoy": i.decoy_score, "decoy_controlled": i.decoy_controlled,
                 "inference_value": round(i.inference_value, 4),
                 "already_internalised": i.already_internalised}
                for i in self.items
            ],
        }


def _excerpt_for(doc: Document, claim: Claim, width: int = 4000) -> str:
    from .adjudicate import best_excerpt
    return best_excerpt(doc.text, claim, width=width)


def _grade(llm: LLM, question: str, gold: str, key_points: Sequence[str],
           answer: str) -> tuple[float, bool]:
    out = llm.json(
        GRADE_SYSTEM,
        f"QUESTION: {question}\nKNOWN-CORRECT ANSWER: {gold}\n"
        f"KEY POINTS ({len(key_points)}): {list(key_points)}\n\n"
        f"CANDIDATE ANSWER:\n{answer}",
        GRADE_SCHEMA, max_tokens=2000, grounded=False,
    )
    if not out:
        return 0.0, False
    total = max(1, int(out.get("total") or len(key_points) or 1))
    score = out.get("score")
    if score is None:
        score = (out.get("covered") or 0) / total
    return max(0.0, min(1.0, float(score))), bool(out.get("hedged", False))


def _answer(llm: LLM, question: str, context: str = "") -> str:
    schema = {"type": "object",
              "properties": {"answer": {"type": "string"}},
              "required": ["answer"], "additionalProperties": False}
    user = (f"CONTEXT:\n{context}\n\n---\n\nQUESTION: {question}"
            if context else f"QUESTION: {question}")
    out = llm.json(ANSWER_SYSTEM, user, schema, max_tokens=2000, grounded=False)
    return (out or {}).get("answer", "") or ""


def ablate_claim(
    store, llm: LLM, claim: Claim, doc: Document, cfg: Config,
    embedder=None, source_id: str | None = None, n_decoy: int = 3,
) -> ClaimAblation | None:
    q = llm.json(
        QUESTION_SYSTEM,
        f"CLAIM: {claim.text}\nKIND: {claim.kind.value}\n"
        f"ENTITIES: {', '.join(claim.entities) or 'none'}",
        QUESTION_SCHEMA, max_tokens=2000, grounded=False,
    )
    if not q or not q.get("usable") or not q.get("question"):
        return None
    question, gold = q["question"], q.get("gold", "")
    key_points = q.get("key_points") or [gold]

    # Decoy context: contemporaneous material from other writers on the same
    # subject. Without it, the comparison collapses into "does context help".
    queries = expand_queries(llm, claim, n=3)
    window_start = doc.published_at - timedelta(days=730)
    res = retrieve(store, claim, queries, after=window_start,
                   before=doc.published_at + timedelta(days=1),
                   embedder=embedder, limit=n_decoy, per_query=8,
                   exclude_source=source_id, exclude_docs=[doc.id])
    decoy_docs = [c.doc for c in res.candidates[:n_decoy]]
    decoy_ctx = "\n\n---\n\n".join(
        f"[{d.title} | {d.published_at.date()}]\n{_excerpt_for(d, claim, 3000)}"
        for d in decoy_docs
    )
    source_ctx = f"[{doc.title} | {doc.published_at.date()}]\n{_excerpt_for(doc, claim)}"

    closed_a = _answer(llm, question)
    source_a = _answer(llm, question, source_ctx)
    decoy_a = _answer(llm, question, decoy_ctx) if decoy_ctx else ""

    closed_s, closed_h = _grade(llm, question, gold, key_points, closed_a)
    source_s, _ = _grade(llm, question, gold, key_points, source_a)
    # No contemporaneous material means the decoy condition did not run. Record
    # that as unavailable rather than as a score of zero.
    decoy_s = (_grade(llm, question, gold, key_points, decoy_a)[0]
               if decoy_ctx else None)

    return ClaimAblation(
        claim_id=claim.id, claim_text=claim.text, question=question,
        closed_score=closed_s, source_score=source_s, decoy_score=decoy_s,
        closed_hedged=closed_h, n_decoy_docs=len(decoy_docs),
    )


def ablate_source(
    store, llm: LLM, source_name: str, cfg: Config, embedder=None,
    n_docs: int = 2, claims_per_doc: int = 3, progress=None,
) -> AblationResult:
    say = progress or (lambda m: None)
    src = store.find_source(source_name)
    if src is None:
        raise ValueError(f"unknown source {source_name!r}")
    from ..pipeline import select_documents
    docs, _ = select_documents(store, src.id, n_docs, cfg)

    items: list[ClaimAblation] = []
    for doc in docs:
        claims = store.get_claims(doc.id)
        if not claims:
            from ..extract.claims import extract_claims
            claims, _diag = extract_claims(llm, doc)
            store.add_claims(claims)
        for claim in sorted(claims, key=lambda c: -c.salience)[:claims_per_doc]:
            item = ablate_claim(store, llm, claim, doc, cfg, embedder, src.id)
            if item is None:
                continue
            items.append(item)
            dec = ("n/a " if item.decoy_score is None
                   else f"{item.decoy_score:.2f}")
            say(f"  closed={item.closed_score:.2f} decoy={dec} "
                f"source={item.source_score:.2f}  value={item.inference_value:+.2f}  "
                f"{item.claim_text[:48]}")

    if not items:
        return AblationResult(src.name, 0, 0.0, 0.0, 0.0, None, 0.0, 0, None, [])
    controlled = [i for i in items if i.decoy_controlled]
    return AblationResult(
        source_name=src.name, n_claims=len(items),
        mean_inference_value=mean(i.inference_value for i in items),
        mean_closed=mean(i.closed_score for i in items),
        mean_source=mean(i.source_score for i in items),
        # Averaged over the items where it actually ran, not over all of them.
        mean_decoy=mean(i.decoy_score for i in controlled) if controlled else None,
        internalised_rate=sum(1 for i in items if i.already_internalised) / len(items),
        n_decoy_controlled=len(controlled),
        mean_inference_value_controlled=(
            mean(i.inference_value for i in controlled) if controlled else None),
        items=items,
    )
