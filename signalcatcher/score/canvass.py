"""Canvass a text for the handful of claims worth evaluating downstream.

"Boldest, mixed kinds": rank by how much a claim could matter (salience x how
distinctive it is), then force diversity across claim kinds so a unit is not
represented by five variations of one thesis. A prediction, a reported fact and
a frame stress different parts of the measure, and a report showing all of them
demonstrates the machinery rather than one lucky lane."""

from __future__ import annotations

from ..models import Claim


def _boldness(c: Claim) -> float:
    score = c.salience
    if c.fingerprints:          # distinctive phrasing -> trackable, likely bolder
        score += 0.15
    if c.kind.value == "prediction" and c.falsifiable:
        score += 0.15           # checkable predictions are the sharpest tests
    if not c.explicit:
        score -= 0.10           # implicit claims are noisier; admit but demote
    return score


def canvass(claims: list[Claim], n: int = 5) -> list[Claim]:
    ranked = sorted(claims, key=_boldness, reverse=True)
    picked: list[Claim] = []
    seen_kinds: set[str] = set()
    # First pass: best claim of each kind, boldest kinds first.
    for c in ranked:
        if len(picked) >= n:
            break
        if c.kind.value not in seen_kinds:
            picked.append(c)
            seen_kinds.add(c.kind.value)
    # Fill remaining slots by boldness.
    for c in ranked:
        if len(picked) >= n:
            break
        if c not in picked:
            picked.append(c)
    return picked
