"""How much corpus does the benchmark actually need?

Subsamples the pinned corpus at decreasing rates and re-runs retrieval for a
fixed set of claims. If the evidence that decides a claim survives at 1/8 of the
corpus, then raw size is not the binding constraint and there is no reason to
buy disk. Uses retrieval only -- no LLM calls -- so it is cheap to re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from signalcatcher.db import Store
from signalcatcher.config import Config
from signalcatcher.index.embed import Embedder

# Dense similarity above which a document is plausibly on-topic for a claim.
# Calibrated earlier: paraphrases ~0.69, unrelated ~0.40.
TOPICAL = 0.55


def sampled_pool(store, mod: int, before_ts: int) -> int:
    return store.conn.execute(
        "SELECT COUNT(*) FROM documents WHERE published_ts < ? AND (rid % ?) = 0",
        (before_ts, mod)).fetchone()[0]


def dense_hits(store, emb, qvec, before_ts, mod, limit=20):
    rows = store.conn.execute(
        "SELECT d.rid, d.id, e.dim, e.vec FROM embeddings e "
        "JOIN documents d ON ('doc:' || d.id) = e.key "
        "WHERE d.published_ts < ? AND (d.rid % ?) = 0", (before_ts, mod)).fetchall()
    if not rows:
        return []
    mat = np.vstack([emb._unpack(r["vec"], r["dim"]) for r in rows])
    sims = mat @ qvec
    order = np.argsort(-sims)[:limit]
    return [(rows[i]["id"], float(sims[i])) for i in order]


def main():
    store = Store()
    cfg = Config()
    emb = Embedder(store, backend="local")
    # Embed a working subset so dense retrieval has something to search.
    docs = store.documents_for_source(
        store.find_source("Brian Potter").id, limit=300)
    for other, _ in store.list_sources():
        docs += store.documents_for_source(other.id, limit=150)
    print(f"ensuring embeddings for {len(docs)} docs ...", flush=True)
    emb.ensure_documents(docs)

    claims = []
    for d in docs:
        for c in store.get_claims(d.id):
            claims.append((c, d))
    claims = claims[:12]
    print(f"testing {len(claims)} claims\n")

    from signalcatcher.score.originality import compute_coverage
    print(f"{'sample':>8} {'pool':>8} {'topical':>8} {'best sim':>9} {'coverage':>9}  "
          f"{'top-1 kept':>10}")
    print("-" * 62)
    baseline_top = {}
    for mod in (1, 2, 4, 8, 16):
        pools, tops, sims, covs, kept = [], [], [], [], []
        for c, d in claims:
            qv = emb.embed_one(c.text, cache_key=f"claim:{c.id}")
            if qv is None:
                continue
            bts = int(d.published_at.timestamp())
            hits = dense_hits(store, emb, qv, bts, mod)
            pool = sampled_pool(store, mod, bts)
            n_top = sum(1 for _, s in hits if s >= TOPICAL)
            best = hits[0][1] if hits else 0.0
            pools.append(pool); tops.append(n_top); sims.append(best)
            covs.append(compute_coverage(pool, n_top, 3, cfg))
            if mod == 1:
                baseline_top[c.id] = hits[0][0] if hits else None
            else:
                kept.append(1.0 if (baseline_top.get(c.id) and
                            baseline_top[c.id] in {h[0] for h in hits}) else 0.0)
        pct = f"1/{mod}"
        keep = f"{100*np.mean(kept):.0f}%" if kept else "—"
        print(f"{pct:>8} {int(np.mean(pools)):>8,} {np.mean(tops):>8.1f} "
              f"{np.mean(sims):>9.3f} {np.mean(covs):>9.3f}  {keep:>10}")
    print("\ntop-1 kept = share of claims whose single best piece of evidence at full")
    print("corpus is still retrieved from the subsample")


if __name__ == "__main__":
    main()
