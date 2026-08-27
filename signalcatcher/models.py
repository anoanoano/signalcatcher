"""Core data model for the SignalCatcher benchmark.

The unit of analysis is the *claim*: a single substantive proposition carried by a
document. Everything the benchmark measures is measured per-claim and then
aggregated up to documents and sources, so that every headline number can be
unrolled back into specific, dated, citable evidence.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class ClaimKind(str, enum.Enum):
    """What sort of contribution a claim makes.

    These are deliberately the categories an editor would recognise, because the
    scores are meant to be defensible to a publisher, not just to a model.
    """

    FACT = "fact"  # empirical assertion; can be corroborated or debunked
    THESIS = "thesis"  # causal or interpretive argument
    FRAME = "frame"  # a coinage, metaphor, or way of characterising something
    PREDICTION = "prediction"  # falsifiable forecast, ideally with a horizon
    SYNTHESIS = "synthesis"  # a novel connection between existing bodies of work


class Relation(str, enum.Enum):
    """How a retrieved document relates to the claim being scored.

    Ordered from strongest to weakest. `SUBSUMES` matters a great deal for
    originality: a prior source stating a more general principle that entails the
    claim defeats novelty even when no sentence matches.
    """

    IDENTICAL = "identical"
    PARAPHRASE = "paraphrase"
    SUBSUMES = "subsumes"
    PARTIAL = "partial"  # a component of the claim, not the whole
    TOPICAL = "topical"  # same subject, different proposition
    UNRELATED = "unrelated"

    @property
    def defeats_novelty(self) -> bool:
        return self in (Relation.IDENTICAL, Relation.PARAPHRASE, Relation.SUBSUMES)


class Direction(str, enum.Enum):
    PRIOR = "prior"  # evidence published before the document
    LATER = "later"  # evidence published after the document


class DateConfidence(str, enum.Enum):
    """How much we trust a document's publication date.

    Date integrity is load-bearing: the entire benchmark is a claim about who
    said what *first*. A silently wrong date fabricates originality or destroys
    it, so provenance is tracked explicitly rather than assumed.
    """

    EXACT = "exact"  # publisher API gave a timestamp
    DAY = "day"  # date known, time not
    INFERRED = "inferred"  # parsed from URL, sitemap, or archive capture
    UNKNOWN = "unknown"


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


@dataclass
class Source:
    """A publisher or writer: the entity whose value we are ultimately scoring."""

    id: str
    kind: str  # substack | news | blog | forum | academic
    name: str
    domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(kind: str, name: str) -> str:
        return _hash(kind, name.lower())


@dataclass
class Document:
    id: str
    source_id: str
    url: str
    title: str
    published_at: datetime
    text: str
    date_confidence: DateConfidence = DateConfidence.EXACT
    date_provenance: str = ""
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    paywalled: bool = False
    lang: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(url: str) -> str:
        return _hash(url)

    @property
    def text_hash(self) -> str:
        return _hash(self.text)


@dataclass
class Claim:
    """One substantive proposition extracted from a document."""

    id: str
    doc_id: str
    kind: ClaimKind
    text: str  # standalone restatement, readable without the source document
    entities: list[str] = field(default_factory=list)
    # Rare or coined phrases lifted verbatim. Fingerprint reuse downstream is much
    # stronger evidence of actual transmission than mere semantic similarity,
    # which cannot distinguish influence from two writers noticing the same thing.
    fingerprints: list[str] = field(default_factory=list)
    explicit: bool = True  # False => entailed or presupposed, not stated
    salience: float = 0.5  # how central to the document, 0..1
    falsifiable: bool = False
    horizon_days: int | None = None  # for predictions
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, text: str) -> str:
        return _hash(doc_id, text)


@dataclass
class Evidence:
    """A dated document adjudicated against a claim, in one temporal direction."""

    id: str
    claim_id: str
    direction: Direction
    relation: Relation
    doc_id: str | None
    url: str
    title: str
    published_at: datetime
    source_id: str | None = None
    confidence: float = 0.5  # judge's confidence in the relation, 0..1
    rationale: str = ""
    quote: str = ""  # the span that justifies the relation
    attributes_source: bool = False  # later doc explicitly credits our source
    fingerprint_hits: list[str] = field(default_factory=list)
    retriever: str = ""  # bm25 | embed | fingerprint | live
    rank: int = 0

    @staticmethod
    def make_id(claim_id: str, url: str) -> str:
        return _hash(claim_id, url)
