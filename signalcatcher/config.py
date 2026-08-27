"""Tunable constants, in one place so a run can be pinned and reproduced."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


# How much each adjudicated relation counts against a claim's novelty. A prior
# document that *subsumes* the claim nearly defeats it even with no shared
# wording, while `partial` deliberately sits below half: a claim built out of
# components that each existed separately is a synthesis, and syntheses are
# original.
RELATION_WEIGHTS: dict[str, float] = {
    "identical": 1.00,
    "paraphrase": 0.95,
    "subsumes": 0.85,
    "partial": 0.45,
    "topical": 0.05,
    "unrelated": 0.0,
}

# Diffusion windows, in days. Short windows catch the news cycle; the long ones
# are where a claim stops being a talking point and either becomes common
# knowledge or is quietly dropped.
WINDOWS_DAYS: tuple[int, ...] = (7, 30, 90, 180, 365, 1095)


@dataclass
class Config:
    # retrieval
    queries_per_claim: int = 5
    candidates_per_claim: int = 14
    per_query_depth: int = 15

    # originality
    # A pre-window corpus this size is treated as a fully searched neighbourhood.
    # It is a modelling choice, not a fact, and it is why originality is reported
    # as an interval rather than a number.
    target_pool: int = 20_000
    # On-topic prior documents that found nothing constitute real evidence of
    # novelty; this is how many such near-misses count as a well-searched space.
    topical_saturation: int = 4

    # influence
    min_relation_for_uptake: str = "partial"
    independent_source_required: bool = True

    # judging
    model: str = "claude-opus-5"
    effort: str = "high"
    grounded: bool = True

    # embeddings
    embed_backend: str = "local"

    windows_days: tuple[int, ...] = WINDOWS_DAYS
    relation_weights: dict[str, float] = field(
        default_factory=lambda: dict(RELATION_WEIGHTS)
    )

    def hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)
