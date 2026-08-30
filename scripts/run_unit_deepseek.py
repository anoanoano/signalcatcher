"""Unit 7: Zvi's 'DeepSeek v3: The Six Million Dollar Model' (2024-12-31),
27 days before the R1 crash. First run under the shell protocol (shell-v1) and
the existence-based surprisal. Short windows -- the story broke in weeks."""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signalcatcher.db import Store
from signalcatcher.llm import LLM
from signalcatcher.index.embed import Embedder
from signalcatcher.extract.claims import extract_claims
from signalcatcher.score.canvass import canvass
from signalcatcher.score.predictive import score_predictive
from signalcatcher.shell import build_shell
from signalcatcher.models import Document

URL="https://thezvi.substack.com/p/deekseek-v3-the-six-million-dollar"
SHORT_PRE=((-365,-90),(-90,0)); SHORT_POST=((0,30),(30,90),(90,180),(180,365))

store=Store(); llm=LLM(store=store, backend="cli", model="opus")
emb=Embedder(store, backend="local")
run_id=f"unit7_{int(time.time())}"
store.start_run(run_id,"score",{"unit":"deepseek-v3"},"shell-v1")
doc=store.get_document(Document.make_id(URL)); assert doc, "unit doc missing"
print(f"UNIT deepseek-v3: {doc.title} ({doc.published_at.date()})",flush=True)
claims=store.get_claims(doc.id)
if not claims:
    claims,diag=extract_claims(llm,doc); store.add_claims(claims)
    print(f"extracted {diag['kept']} claims",flush=True)
picked=canvass(claims,n=5)
print(f"canvassed: {[c.kind.value for c in picked]}",flush=True)

print("\n--- building evidence shell (shell-v1) ---",flush=True)
manifest=build_shell(store,llm,"deepseek-v3",doc,picked,windows_days=(30,180),
                     progress=lambda m:print(m,flush=True))
print("shell channels:",{k:(v.get("status") or "ok") for k,v in manifest["channels"].items()},flush=True)
for s2,_ in store.list_sources():
    emb.ensure_documents(store.documents_for_source(s2.id,limit=400))

unit={"key":"deepseek-v3","label":"Shell-protocol test: intl press + arXiv + reference layer",
      "source":"Zvi Mowshowitz","title":doc.title,"url":doc.url,
      "published":doc.published_at.date().isoformat(),"horizon":"short",
      "manifest_summary":{
        "recipe":manifest["recipe"],
        "intl_added":manifest["channels"]["intl"].get("added"),
        "academic_added":manifest["channels"]["academic"].get("added"),
        "reference":manifest["channels"]["reference"],
        "news_claims":len(manifest["channels"]["news"]["per_claim"])},
      "claims":[]}
t0=time.time()
for i,claim in enumerate(picked,1):
    print(f"\n[{i}/{len(picked)}] {claim.kind.value}: {claim.text[:70]}",flush=True)
    try:
        r=score_predictive(store,llm,claim,doc,embedder=emb,run_id=run_id,persist=True,
                           pre_windows=SHORT_PRE,post_windows=SHORT_POST)
    except Exception:
        print(traceback.format_exc(limit=2),flush=True); continue
    d=r.to_dict()
    d["claim"]={"text":claim.text,"kind":claim.kind.value,"salience":claim.salience,
                "explicit":claim.explicit,"fingerprints":claim.fingerprints}
    unit["claims"].append(d)
    bp=d.get("best_prior")
    print(f"  S={d['surprisal']} (prior_strength={d['prior_strength']}) "
          f"A={d['adoption']} V={d['vindication']} PV={d['predictive_value']} [{time.time()-t0:.0f}s]",flush=True)
    if bp: print(f"  closest prior: {bp['date']} {bp['source'] or '?'} [{bp['relation']}] {bp['title'][:50]}",flush=True)
    Path("data/unit7_report.json").write_text(json.dumps({"units":[unit],"partial":True},indent=1))
Path("data/unit7_report.json").write_text(json.dumps({"units":[unit]},indent=1))
store.finish_run(run_id)
print(f"\nDONE in {(time.time()-t0)/60:.0f} min. {llm.stats()}",flush=True)
