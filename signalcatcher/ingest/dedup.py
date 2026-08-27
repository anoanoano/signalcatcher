"""Syndication detection: collapse wire copies into one source.

A single agency story runs verbatim in dozens of outlets. Counted naively that
reads as dozens of independent publishers picking up a claim, and influence --
whose whole point is measuring how widely something spread -- becomes a measure
of how well-syndicated a story was. In a 16-article sample from GDELT, one story
already appeared three times under three domains.

Detection is MinHash over word shingles rather than exact hashing, because
syndicated copies are near-identical, not identical: outlets retitle, top-and-
tail, insert their own boilerplate, and trim paragraphs. Exact matching catches
almost none of them.

Clusters keep the EARLIEST member as canonical, which is also the answer to the
question the benchmark is asking -- who ran it first.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..textutil import shingles

N_HASHES = 64
BANDS = 16          # LSH bands; rows per band = N_HASHES // BANDS
SHINGLE_K = 6
# Syndicated copies typically land above 0.6 estimated Jaccard even after
# retitling and trimming. Genuine independent coverage of the same event, which
# must NOT be collapsed, sits far lower.
SIM_THRESHOLD = 0.55
MIN_SHINGLES = 20   # below this a document is too short to judge


def _hash_family(n: int = N_HASHES) -> list[int]:
    return [int(hashlib.sha256(f"seed{i}".encode()).hexdigest()[:8], 16) for i in range(n)]


_SEEDS = _hash_family()
_MASK = 0xFFFFFFFF


def signature(text: str, k: int = SHINGLE_K) -> list[int] | None:
    """MinHash signature of a document's shingle set."""
    sh = shingles(text, k)
    if len(sh) < MIN_SHINGLES:
        return None
    base = [int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in sh]
    return [min((h ^ seed) & _MASK for h in base) for seed in _SEEDS]


def estimate_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


@dataclass
class Cluster:
    canonical_id: str
    members: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)


def cluster_documents(
    docs: Sequence, threshold: float = SIM_THRESHOLD,
) -> tuple[dict[str, str], list[Cluster]]:
    """Group near-duplicate documents.

    Returns (doc_id -> canonical_doc_id, clusters). Comparing every pair would
    be quadratic, so candidates are first bucketed by LSH bands: documents that
    share any band are compared, the rest are never considered.
    """
    sigs: dict[str, list[int]] = {}
    for d in docs:
        s = signature(d.text)
        if s is not None:
            sigs[d.id] = s

    rows = max(1, N_HASHES // BANDS)
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for doc_id, sig in sigs.items():
        for b in range(BANDS):
            key = (b, tuple(sig[b * rows : (b + 1) * rows]))
            buckets[key].append(doc_id)

    # Union-find over candidate pairs.
    parent: dict[str, str] = {d: d for d in sigs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    checked: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 200:
            continue  # a huge bucket is boilerplate, not a story
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j]) if members[i] < members[j] else (members[j], members[i])
                if pair in checked:
                    continue
                checked.add(pair)
                if estimate_jaccard(sigs[pair[0]], sigs[pair[1]]) >= threshold:
                    union(*pair)

    by_doc = {d.id: d for d in docs}
    groups: dict[str, list[str]] = defaultdict(list)
    for doc_id in sigs:
        groups[find(doc_id)].append(doc_id)

    mapping: dict[str, str] = {}
    clusters: list[Cluster] = []
    for members in groups.values():
        # Earliest publication is canonical -- which is also the priority answer.
        members.sort(key=lambda i: (by_doc[i].published_at, i))
        canonical = members[0]
        for m in members:
            mapping[m] = canonical
        if len(members) > 1:
            clusters.append(Cluster(canonical_id=canonical, members=members))
    return mapping, clusters


def independent_sources(evidence: Iterable, canonical_of: dict[str, str]) -> set:
    """Distinct sources, counting each syndication cluster only once.

    Without this, `uptake_sources` counts a wire story once per outlet that ran
    it and reports one press release as industry-wide agreement.
    """
    seen_clusters: set[str] = set()
    sources: set = set()
    for e in evidence:
        canon = canonical_of.get(e.doc_id or "", e.doc_id)
        if canon in seen_clusters:
            continue
        seen_clusters.add(canon)
        if e.source_id:
            sources.add(e.source_id)
    return sources
