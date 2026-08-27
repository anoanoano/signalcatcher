"""Run the three-unit pilot: canvass each text, ingest targeted news, score every
claim, and write one JSON the report renders from.

Sequential and restartable: every LLM judgement and HTTP fetch is cached, so a
crash resumes at full speed.
"""
from __future__ import annotations

import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signalcatcher.db import Store
from signalcatcher.llm import LLM
from signalcatcher.index.embed import Embedder
from signalcatcher.extract.claims import extract_claims
from signalcatcher.ingest.claim_news import ingest_for_claim
from signalcatcher.score.canvass import canvass
from signalcatcher.score.predictive import score_predictive

LONG_PRE  = ((-730, -365), (-365, 0))
LONG_POST = ((0, 90), (90, 180), (180, 365), (365, 730), (730, 1460))
SHORT_PRE  = ((-365, -90), (-90, 0))
SHORT_POST = ((0, 30), (30, 90), (90, 180), (180, 365))

UNITS = [
    dict(key="new-axis", source="Noah Smith", title_prefix="Sizing up the New Axis",
         horizon="long", label="Geopolitical frame, slow burn"),
    dict(key="gpt4-expectations", source="Gary Marcus",
         title_prefix="What to Expect When You’re Expecting",
         horizon="long", label="Concrete predictions, checkable fast"),
    dict(key="svb-run", source="Noah Smith",
         title_prefix="Why was there a run on Silicon Valley Bank",
         horizon="short", label="Day-one event interpretation"),
]
N_CLAIMS = 5

def main():
    store = Store()
    llm = LLM(store=store, backend="cli", model="opus")
    emb = Embedder(store, backend="local")
    run_id = f"units_{int(time.time())}"
    store.start_run(run_id, "score", {"units": [u["key"] for u in UNITS]}, "units-pilot")
    out = {"run_id": run_id, "units": []}
    t0 = time.time()

    for u in UNITS:
        src = store.find_source(u["source"])
        docs = [d for d in store.documents_for_source(src.id, limit=1200)
                if d.title.startswith(u["title_prefix"])]
        if not docs:
            print(f"!! no doc for {u['key']}", flush=True); continue
        doc = docs[0]
        print(f"\n=== UNIT {u['key']}: {doc.title[:60]} ({doc.published_at.date()}) ===", flush=True)

        claims = store.get_claims(doc.id)
        if not claims:
            claims, diag = extract_claims(llm, doc)
            store.add_claims(claims)
            print(f"  extracted {diag['kept']} claims", flush=True)
        picked = canvass(claims, n=N_CLAIMS)
        print(f"  canvassed {len(picked)} of {len(claims)} claims "
              f"(kinds: {[c.kind.value for c in picked]})", flush=True)

        pre, post = (SHORT_PRE, SHORT_POST) if u["horizon"] == "short" else (LONG_PRE, LONG_POST)
        windows_days = [30, 180] if u["horizon"] == "short" else (180, 365)
        unit = {"key": u["key"], "label": u["label"], "source": src.name,
                "title": doc.title, "url": doc.url,
                "published": doc.published_at.date().isoformat(),
                "horizon": u["horizon"], "claims": []}

        for i, claim in enumerate(picked, 1):
            print(f"  [{i}/{len(picked)}] {claim.kind.value}: {claim.text[:70]}", flush=True)
            try:
                r = ingest_for_claim(store, claim, doc.published_at,
                                     windows_days=windows_days, llm=llm,
                                     max_per_window=25,
                                     progress=lambda m: print(m, flush=True))
                # New sources/docs need vectors before retrieval can see them.
                for s2, _n in store.list_sources():
                    emb.ensure_documents(store.documents_for_source(s2.id, limit=400))
            except Exception:
                print("   news ingest failed (continuing):", traceback.format_exc(limit=1), flush=True)
            try:
                res = score_predictive(store, llm, claim, doc, embedder=emb,
                                       run_id=run_id, persist=True,
                                       pre_windows=pre, post_windows=post,
                                       progress=lambda m: print(m, flush=True))
            except Exception:
                print("   scoring failed:", traceback.format_exc(limit=2), flush=True)
                continue
            d = res.to_dict()
            d["claim"] = {"text": claim.text, "kind": claim.kind.value,
                          "salience": claim.salience, "explicit": claim.explicit,
                          "fingerprints": claim.fingerprints}
            unit["claims"].append(d)
            print(f"      S={d['surprisal']} A={d['adoption']} V={d['vindication']} "
                  f"PV={d['predictive_value']}  [{time.time()-t0:.0f}s]", flush=True)
            Path("data/units_report.json").write_text(json.dumps(out | {"partial": True}, indent=1))
        out["units"].append(unit)
        Path("data/units_report.json").write_text(json.dumps(out, indent=1))

    store.finish_run(run_id)
    Path("data/units_report.json").write_text(json.dumps(out, indent=1))
    print(f"\nDONE in {(time.time()-t0)/60:.0f} min. stats: {llm.stats()}", flush=True)

if __name__ == "__main__":
    main()
