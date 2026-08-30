"""Rescore runs 1+2 under existence-based surprisal. Post-window judgements are
cached; the new cost is the prior-art search per claim."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signalcatcher.db import Store
from signalcatcher.llm import LLM
from signalcatcher.index.embed import Embedder
from signalcatcher.score.canvass import canvass
from signalcatcher.score.predictive import score_predictive
from signalcatcher.models import Document

LONG_PRE=((-730,-365),(-365,0)); LONG_POST=((0,90),(90,180),(180,365),(365,730),(730,1460))
SHORT_PRE=((-365,-90),(-90,0));  SHORT_POST=((0,30),(30,90),(90,180),(180,365))
UNITS=[  # (file, key, source, title_prefix_or_url, horizon, label)
 ("data/units_report.json","new-axis","Noah Smith","Sizing up the New Axis","long","Geopolitical frame, slow burn"),
 ("data/units_report.json","gpt5-expectations","Gary Marcus","What to Expect When You’re Expecting","long","Concrete predictions, checkable fast"),
 ("data/units_report.json","svb-run","Noah Smith","Why was there a run on Silicon Valley Bank","short","Day-one event interpretation"),
 ("data/units2_report.json","media-rarely-lies","Scott Alexander","The Media Very Rarely Lies","long","Recognized-great essay, contested thesis"),
 ("data/units2_report.json","techtimes-ai-2023","Tech Times","AI Predictions for 2023","long","Low-tier outlet, aggregated predictions"),
 ("data/units2_report.json","lichtman-ignore","Andre Cooper","You Can Ignore Allan Lichtman","short","Under-the-radar post, checkable within months"),
]
store=Store(); llm=LLM(store=store, backend="cli", model="opus"); emb=Embedder(store, backend="local")
outs={}; t0=time.time()
for fpath,key,srcn,prefix,horizon,label in UNITS:
    src=store.find_source(srcn)
    doc=[d for d in store.documents_for_source(src.id,limit=1500) if d.title.startswith(prefix)][0]
    picked=canvass(store.get_claims(doc.id),n=5)
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
        print(f"  {key} {c.kind.value:10} S={d['surprisal']} PV={d['predictive_value']} [{time.time()-t0:.0f}s]",flush=True)
    outs.setdefault(fpath,{"run_id":"rescore-v2","units":[]})["units"].append(unit)
    for fp,data in outs.items():
        Path(fp).write_text(json.dumps(data,indent=1))
print(f"done {time.time()-t0:.0f}s hits={llm.cache_hits}/{llm.calls+llm.cache_hits}",flush=True)
