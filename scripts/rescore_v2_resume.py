"""Resume the v2 rescore: the two units the crash left behind, selected by
exact URL (source names can be renamed by later ingests; URLs cannot)."""
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
UNITS=[
 ("techtimes-ai-2023","https://www.techtimes.com/articles/285594/20221229/ai-predictions-2023-experts-foresee-shift-autonomous-systems-data-shortage.htm",
  "long","Low-tier outlet, aggregated predictions","Tech Times"),
 ("lichtman-ignore","https://goodreason.substack.com/p/you-can-ignore-allan-lichtman-and",
  "short","Under-the-radar post, checkable within months","Andre Cooper"),
]
store=Store(); llm=LLM(store=store, backend="cli", model="opus"); emb=Embedder(store, backend="local")
out=json.loads(Path("data/units2_report.json").read_text())
done={u["key"] for u in out["units"]}
t0=time.time()
for key,url,horizon,label,srcname in UNITS:
    if key in done: continue
    doc=store.get_document(Document.make_id(url)); assert doc, key
    picked=canvass(store.get_claims(doc.id),n=5)
    pre,post=(SHORT_PRE,SHORT_POST) if horizon=="short" else (LONG_PRE,LONG_POST)
    unit={"key":key,"label":label,"source":srcname,"title":doc.title,"url":doc.url,
          "published":doc.published_at.date().isoformat(),"horizon":horizon,"claims":[]}
    for c in picked:
        r=score_predictive(store,llm,c,doc,embedder=emb,persist=False,
                           pre_windows=pre,post_windows=post)
        d=r.to_dict()
        d["claim"]={"text":c.text,"kind":c.kind.value,"salience":c.salience,
                    "explicit":c.explicit,"fingerprints":c.fingerprints}
        unit["claims"].append(d)
        print(f"  {key} {c.kind.value:10} S={d['surprisal']} PV={d['predictive_value']} [{time.time()-t0:.0f}s]",flush=True)
    out["units"].append(unit)
    Path("data/units2_report.json").write_text(json.dumps(out,indent=1))
print(f"done {time.time()-t0:.0f}s hits={llm.cache_hits}/{llm.calls+llm.cache_hits}",flush=True)
