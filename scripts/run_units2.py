"""Second pilot: recognized-great vs low-quality-outlet vs under-the-radar.
Units selected by exact URL (the title-prefix collision in run 1 picked the
wrong Marcus post). Writes data/units2_report.json."""
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
from signalcatcher.models import Document

LONG_PRE=((-730,-365),(-365,0)); LONG_POST=((0,90),(90,180),(180,365),(365,730),(730,1460))
SHORT_PRE=((-365,-90),(-90,0));  SHORT_POST=((0,30),(30,90),(90,180),(180,365))

UNITS=[
 dict(key="media-rarely-lies", url="https://www.astralcodexten.com/p/the-media-very-rarely-lies",
      horizon="long", label="Recognized-great essay, contested thesis"),
 dict(key="techtimes-ai-2023", url="https://www.techtimes.com/articles/285594/20221229/ai-predictions-2023-experts-foresee-shift-autonomous-systems-data-shortage.htm",
      horizon="long", label="Low-tier outlet, aggregated predictions"),
 dict(key="lichtman-ignore", url="https://goodreason.substack.com/p/you-can-ignore-allan-lichtman-and",
      horizon="short", label="Under-the-radar post, checkable within months"),
]

def main():
    store=Store(); llm=LLM(store=store, backend="cli", model="opus")
    emb=Embedder(store, backend="local")
    run_id=f"units2_{int(time.time())}"
    store.start_run(run_id,"score",{"units":[u["key"] for u in UNITS]},"units2-pilot")
    out={"run_id":run_id,"units":[]}; t0=time.time()
    for u in UNITS:
        doc=store.get_document(Document.make_id(u["url"]))
        if doc is None:
            print(f"!! missing doc {u['key']}",flush=True); continue
        src=store.get_source(doc.source_id)
        print(f"\n=== UNIT {u['key']}: {doc.title[:58]} ({doc.published_at.date()}) ===",flush=True)
        claims=store.get_claims(doc.id)
        if not claims:
            claims,diag=extract_claims(llm,doc); store.add_claims(claims)
            print(f"  extracted {diag['kept']} claims",flush=True)
        picked=canvass(claims,n=5)
        print(f"  canvassed {len(picked)} (kinds: {[c.kind.value for c in picked]})",flush=True)
        pre,post=(SHORT_PRE,SHORT_POST) if u["horizon"]=="short" else (LONG_PRE,LONG_POST)
        wdays=[30,90] if u["horizon"]=="short" else (180,365)
        unit={"key":u["key"],"label":u["label"],"source":src.name,"title":doc.title,
              "url":doc.url,"published":doc.published_at.date().isoformat(),
              "horizon":u["horizon"],"claims":[]}
        for i,claim in enumerate(picked,1):
            print(f"  [{i}/{len(picked)}] {claim.kind.value}: {claim.text[:66]}",flush=True)
            try:
                ingest_for_claim(store,claim,doc.published_at,windows_days=wdays,
                                 llm=llm,max_per_window=25,
                                 progress=lambda m:print(m,flush=True))
                for s2,_ in store.list_sources():
                    emb.ensure_documents(store.documents_for_source(s2.id,limit=400))
            except Exception:
                print("   news ingest failed:",traceback.format_exc(limit=1),flush=True)
            try:
                r=score_predictive(store,llm,claim,doc,embedder=emb,run_id=run_id,
                                   persist=True,pre_windows=pre,post_windows=post)
            except Exception:
                print("   scoring failed:",traceback.format_exc(limit=2),flush=True); continue
            d=r.to_dict()
            d["claim"]={"text":claim.text,"kind":claim.kind.value,"salience":claim.salience,
                        "explicit":claim.explicit,"fingerprints":claim.fingerprints}
            unit["claims"].append(d)
            print(f"      S={d['surprisal']} A={d['adoption']} V={d['vindication']} "
                  f"PV={d['predictive_value']} [{time.time()-t0:.0f}s]",flush=True)
            Path("data/units2_report.json").write_text(json.dumps(out|{"partial":True},indent=1))
        out["units"].append(unit)
        Path("data/units2_report.json").write_text(json.dumps(out,indent=1))
    store.finish_run(run_id)
    Path("data/units2_report.json").write_text(json.dumps(out,indent=1))
    print(f"\nDONE in {(time.time()-t0)/60:.0f} min. stats: {llm.stats()}",flush=True)

if __name__=="__main__": main()
