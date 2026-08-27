"""Re-serialize the units run with trails included. All judgements are cached,
so this re-runs scoring without re-paying for it; news ingest is skipped."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signalcatcher.db import Store
from signalcatcher.llm import LLM
from signalcatcher.index.embed import Embedder
from signalcatcher.score.canvass import canvass
from signalcatcher.score.predictive import score_predictive

LONG_PRE=((-730,-365),(-365,0)); LONG_POST=((0,90),(90,180),(180,365),(365,730),(730,1460))
SHORT_PRE=((-365,-90),(-90,0));  SHORT_POST=((0,30),(30,90),(90,180),(180,365))
UNITS=[("new-axis","Noah Smith","Sizing up the New Axis","long","Geopolitical frame, slow burn"),
       ("gpt5-expectations","Gary Marcus","What to Expect When You’re Expecting","long","Concrete predictions, checkable fast"),
       ("svb-run","Noah Smith","Why was there a run on Silicon Valley Bank","short","Day-one event interpretation")]

store=Store(); llm=LLM(store=store, backend="cli", model="opus"); emb=Embedder(store, backend="local")
out={"run_id":"units_rescore","units":[]}; t0=time.time()
for key,src_name,prefix,horizon,label in UNITS:
    src=store.find_source(src_name)
    doc=[d for d in store.documents_for_source(src.id,limit=1200) if d.title.startswith(prefix)][0]
    picked=canvass(store.get_claims(doc.id), n=5)
    pre,post=(SHORT_PRE,SHORT_POST) if horizon=="short" else (LONG_PRE,LONG_POST)
    unit={"key":key,"label":label,"source":src.name,"title":doc.title,"url":doc.url,
          "published":doc.published_at.date().isoformat(),"horizon":horizon,"claims":[]}
    for c in picked:
        r=score_predictive(store,llm,c,doc,embedder=emb,persist=False,
                           pre_windows=pre,post_windows=post)
        d=r.to_dict()
        d["claim"]={"text":c.text,"kind":c.kind.value,"salience":c.salience,
                    "explicit":c.explicit,"fingerprints":c.fingerprints}
        unit["claims"].append(d)
        n_ex=sum(len(w.get("examples") or []) for w in d["windows_post"])
        print(f"  {key} {c.kind.value:10} PV={d['predictive_value']} trail={n_ex} [{time.time()-t0:.0f}s]",flush=True)
    out["units"].append(unit)
Path("data/units_report.json").write_text(json.dumps(out,indent=1))
print(f"done {time.time()-t0:.0f}s cache_hits={llm.cache_hits}/{llm.calls+llm.cache_hits}")
