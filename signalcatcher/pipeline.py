"""End-to-end: a source in, an audited scorecard out.

The ordering here is deliberate. Claims are extracted once and reused by every
scorer and every control, because extraction is the one step whose output the
whole run has to agree on -- re-extracting per stage would let the originality
score and the influence score be about subtly different claims.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence

from .config import Config
from .db import Store
from .extract.claims import extract_claims
from .index.embed import Embedder
from .llm import LLM
from .models import Claim, Document
from .score.aggregate import ClaimScore, DocumentScore, SourceScore, aggregate_document, aggregate_source
from .score.influence import score_influence
from .score.originality import score_originality
from .validate.controls import (
    ControlOutcome,
    date_shift_control,
    decoy_source_control,
    judge_agreement_control,
    no_retrieval_control,
    shuffled_claim_control,
)


@dataclass
class RunReport:
    run_id: str
    source: SourceScore
    controls: list[ControlOutcome] = field(default_factory=list)
    corpus_docs: int = 0
    corpus_span: tuple[str, str] = ("", "")
    config_hash: str = ""
    llm_stats: dict = field(default_factory=dict)
    embed_status: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


def _min_forward_days(cfg: Config) -> int:
    return max(cfg.windows_days)


def select_documents(
    store: Store, source_id: str, n: int, cfg: Config, min_chars: int = 3000,
) -> tuple[list[Document], list[str]]:
    """Choose documents that can actually be scored, and say what was excluded.

    A document needs corpus on BOTH sides of it: prior text to test novelty
    against, and enough elapsed time for uptake to have happened. Scoring last
    week's post for three-year influence would return zero and read as "no
    influence" rather than "not yet knowable", so those are excluded and counted
    rather than silently scored as failures.
    """
    warnings: list[str] = []
    _, hi = store.corpus_span()
    now = hi or datetime.now(store.corpus_span()[0].tzinfo if store.corpus_span()[0] else None)
    horizon = now - timedelta(days=_min_forward_days(cfg))

    all_docs = store.documents_for_source(source_id, limit=5000)
    usable = [d for d in all_docs
              if len(d.text) >= min_chars and d.published_at <= horizon]
    too_recent = sum(1 for d in all_docs
                     if len(d.text) >= min_chars and d.published_at > horizon)
    if too_recent:
        warnings.append(
            f"{too_recent} documents excluded: published within "
            f"{_min_forward_days(cfg)}d of the corpus edge, so their long-horizon "
            f"influence is not yet observable"
        )
    if not usable:
        warnings.append(
            "no documents have a full forward window inside this corpus; "
            "falling back to the oldest available documents and reporting "
            "influence over whatever window exists"
        )
        usable = sorted([d for d in all_docs if len(d.text) >= min_chars],
                        key=lambda d: d.published_at)
    # Oldest first: they have the most corpus on both sides of them.
    usable.sort(key=lambda d: d.published_at)
    return usable[:n], warnings


def score_source(
    store: Store,
    llm: LLM,
    source_name: str,
    cfg: Config | None = None,
    n_docs: int = 3,
    max_claims_per_doc: int = 6,
    embedder: Embedder | None = None,
    run_controls: bool = True,
    decoy_source_name: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunReport:
    cfg = cfg or Config()
    t0 = time.time()
    say = progress or (lambda m: None)

    src = store.find_source(source_name)
    if src is None:
        raise ValueError(
            f"no source named {source_name!r}; have: "
            f"{[s.name for s, _ in store.list_sources()][:20]}"
        )

    run_id = f"run_{uuid.uuid4().hex[:10]}"
    store.start_run(run_id, "score", cfg.to_dict(), cfg.hash())
    warnings: list[str] = []

    if embedder is None:
        embedder = Embedder(store, backend=cfg.embed_backend)
    if not embedder.enabled:
        warnings.append(
            f"dense retrieval unavailable ({embedder.status}); running on lexical "
            f"and fingerprint retrieval only, so coverage is lower than it would "
            f"otherwise be and originality scores are correspondingly discounted"
        )

    docs, doc_warnings = select_documents(store, src.id, n_docs, cfg)
    warnings.extend(doc_warnings)
    if not docs:
        raise ValueError(f"source {source_name!r} has no documents long enough to score")

    say(f"embedding corpus for dense retrieval ...")
    if embedder.enabled:
        embedder.ensure_documents(store.documents_for_source(src.id, limit=5000))
        # Everything else in the corpus is the evidence pool; it must be embedded
        # too or dense retrieval can only ever find the source's own work.
        for other, _ in store.list_sources():
            if other.id != src.id:
                embedder.ensure_documents(store.documents_for_source(other.id, limit=5000))

    doc_scores: list[DocumentScore] = []
    scored_claims: list[tuple[Claim, datetime]] = []

    for i, doc in enumerate(docs, 1):
        say(f"[{i}/{len(docs)}] {doc.published_at.date()} {doc.title[:56]}")
        claims = store.get_claims(doc.id)
        if not claims:
            claims, diag = extract_claims(llm, doc)
            store.add_claims(claims)
            say(f"    extracted {diag['kept']} claims "
                f"({diag['fingerprints_kept']} fingerprints kept, "
                f"{diag['fingerprints_dropped']} rejected as non-verbatim)")
        # Score the claims the document most exists to make.
        claims = sorted(claims, key=lambda c: -c.salience)[:max_claims_per_doc]

        cscores: list[ClaimScore] = []
        for claim in claims:
            orig = score_originality(store, llm, claim, doc.published_at, cfg,
                                     embedder, src.id, src.name, run_id)
            infl = score_influence(store, llm, claim, doc.published_at, cfg,
                                   embedder, src.id, src.name, run_id)
            cscores.append(ClaimScore(
                claim_id=claim.id, doc_id=doc.id, text=claim.text,
                kind=claim.kind.value, salience=claim.salience, explicit=claim.explicit,
                originality=orig.score, originality_lo=orig.lo, originality_hi=orig.hi,
                coverage=orig.coverage, influence=infl.score,
                lead_time_days=infl.lead_time_days,
                uptake_sources=infl.total_uptake_sources,
                is_synthesis=orig.is_synthesis,
            ))
            scored_claims.append((claim, doc.published_at))
            say(f"      orig={orig.score:.2f} [{orig.lo:.2f}-{orig.hi:.2f}] "
                f"cov={orig.coverage:.2f}  infl={infl.score:.2f} "
                f"({infl.total_uptake_sources} src)  {claim.text[:52]}")

        ds = aggregate_document(doc.id, doc.title, doc.url,
                                doc.published_at.date().isoformat(), cscores)
        store.put_score(run_id, "document", doc.id, "signal", ds.signal)
        doc_scores.append(ds)

    source_score = aggregate_source(src.id, src.name, doc_scores)
    for metric, value in [("headline", source_score.headline),
                          ("peak_signal", source_score.peak_signal),
                          ("hit_rate", source_score.hit_rate)]:
        store.put_score(run_id, "source", src.id, metric, value)

    controls: list[ControlOutcome] = []
    if run_controls and scored_claims:
        sample = scored_claims[: min(6, len(scored_claims))]
        orig_baseline = [c.originality for d in doc_scores for c in d.claims][: len(sample)]
        infl_baseline = [c.influence for d in doc_scores for c in d.claims][: len(sample)]

        say("running controls: date_shift ...")
        controls.append(date_shift_control(
            store, llm, sample, cfg, embedder, src.id, src.name,
            baseline=orig_baseline, run_id=run_id))

        say("running controls: no_retrieval (contamination probe) ...")
        controls.append(no_retrieval_control(
            store, llm, sample, cfg, embedder, src.id, src.name,
            baseline=orig_baseline, run_id=run_id))

        say("running controls: shuffled_claim ...")
        controls.append(shuffled_claim_control(
            store, llm, sample, cfg, embedder, src.id,
            baseline=infl_baseline, run_id=run_id))

        if decoy_source_name:
            decoy = store.find_source(decoy_source_name)
            if decoy is not None:
                say(f"running controls: decoy_source ({decoy.name}) ...")
                ddocs, _ = select_documents(store, decoy.id, 1, cfg)
                dclaims: list[tuple[Claim, datetime]] = []
                for d in ddocs:
                    dc = store.get_claims(d.id)
                    if not dc:
                        dc, _diag = extract_claims(llm, d)
                        store.add_claims(dc)
                    for c in sorted(dc, key=lambda c: -c.salience)[:3]:
                        dclaims.append((c, d.published_at))
                if dclaims:
                    controls.append(decoy_source_control(
                        store, llm, sample, dclaims, cfg, embedder, src.id,
                        decoy.id, run_id=run_id, baseline=infl_baseline))
            else:
                warnings.append(f"decoy source {decoy_source_name!r} not in corpus")

    store.finish_run(run_id, notes=f"source={src.name}")
    lo, hi = store.corpus_span()
    return RunReport(
        run_id=run_id, source=source_score, controls=controls,
        corpus_docs=store.count_documents(),
        corpus_span=(lo.date().isoformat() if lo else "", hi.date().isoformat() if hi else ""),
        config_hash=cfg.hash(), llm_stats=llm.stats(), embed_status=embedder.status,
        warnings=warnings, elapsed_s=round(time.time() - t0, 1),
    )
