"""Render the SignalCatcher explorer from the unit JSONs.

Structured as a guided path for a colleague meeting the benchmark cold:
question -> one worked example -> the measure -> the judging -> the evidence
base -> explorable results -> validation -> limits. Every number on the page
can be expanded to the dated, quoted evidence that produced it.
"""
from __future__ import annotations

import html as H
import json, sys
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------- data merge
_units: dict = {}
for f in ("data/units_report_v1.json", "data/units_report.json",
          "data/units2_report.json", "data/unit7_report.json"):
    _p = Path(f)
    if _p.exists():
        for u in json.loads(_p.read_text()).get("units", []):
            prev = _units.get(u["key"])
            is_v2 = any(c.get("prior_strength") is not None for c in u["claims"])
            if prev is None or is_v2:
                u["_surprisal_v2"] = is_v2
                _units[u["key"]] = u
ORDER = ["new-axis", "gpt5-expectations", "svb-run", "media-rarely-lies",
         "techtimes-ai-2023", "lichtman-ignore", "deepseek-v3"]
UNITS = sorted(_units.values(), key=lambda u: ORDER.index(u["key"]) if u["key"] in ORDER else 99)
N_CLAIMS = sum(len(u["claims"]) for u in UNITS)

STRESS = {
    "new-axis": "a slow-burn geopolitical frame — value visible only at multi-year horizons",
    "gpt5-expectations": "concrete AI predictions — checkable fast, vindication contested",
    "svb-run": "day-one interpretation of a breaking event — scored on month windows",
    "media-rarely-lies": "a famous, contested essay — engagement without adoption",
    "techtimes-ai-2023": "low-tier aggregated content — tests whether borrowed foresight earns credit",
    "lichtman-ignore": "an unknown writer — tests whether the metric needs reputation",
    "deepseek-v3": "full evidence-shell test — intl press, arXiv and the reference layer in play",
}

# live corpus stats
try:
    from signalcatcher.db import Store
    _s = Store()
    _lo, _hi = _s.corpus_span()
    CORPUS_DOCS = _s.count_documents()
    CORPUS_SPAN = f"{_lo.date().isoformat()} .. {_hi.date().isoformat()}"
    CORPUS_KINDS = {k: (ns, nd) for k, ns, nd in _s.conn.execute(
        "SELECT s2.kind, COUNT(DISTINCT s2.id), COUNT(d.id) FROM sources s2 "
        "JOIN documents d ON d.source_id=s2.id GROUP BY s2.kind")}
except Exception:
    CORPUS_DOCS, CORPUS_SPAN, CORPUS_KINDS = 11000, "2018 .. 2026", {}

REL_LABEL = {
    "states": ("states it", "pos"), "entails": ("entails it", "pos"),
    "anticipates_directionally": ("anticipates", "pos"),
    "partially_anticipates": ("partial", "pos"),
    "orthogonal": ("orthogonal", "mut"), "contradicts": ("contradicts", "neg"),
}

def esc(s): return H.escape(str(s or ""))

def _doc_text(url):
    """Source text for a quoted document, cached; None when unavailable."""
    if url in _DOCTEXT:
        return _DOCTEXT[url]
    t = None
    try:
        from signalcatcher.models import Document as _D
        d = _s.get_document(_D.make_id(url))
        t = d.text if d else None
    except Exception:
        t = None
    _DOCTEXT[url] = t
    return t
_DOCTEXT: dict = {}


def clean_quote(q, limit=200, url=None):
    """Tidy a stored verbatim span for display.

    Judges copy spans character-for-character, so a stored quote can begin or
    end mid-word ("nsensus toward..."), and storage clips long ones. Heuristics
    cannot tell a fragment ("nsensus") from a legitimate lowercase start
    ("our alliance..."), so the repair is done against the source document:
    find the span in the stored text and extend both ends to word boundaries.
    When the document is unavailable the span is shown as stored, with
    ellipses marking any visible clipping.
    """
    q = (q or "").strip()
    if not q:
        return ""
    text = _doc_text(url) if url else None
    if text:
        probe = q[:80]
        i = text.lower().find(probe.lower())
        if i != -1:
            j = i + len(q)
            while i > 0 and (text[i-1].isalnum() or text[i-1] in "'\u2019-"):
                i -= 1
            j = min(j, len(text))
            while j < len(text) and (text[j-1].isalnum()) and (text[j].isalnum() or text[j] in "'\u2019-"):
                j += 1
            q = " ".join(text[i:j].split())
    clipped_tail = False
    if len(q) > limit:
        cut = q[:limit]
        q = cut[: cut.rfind(" ")] if " " in cut else cut
        clipped_tail = True
    head = "&hellip;" if q[:1].islower() else ""
    tail = "&hellip;" if (clipped_tail or q[-1:] not in '.!?\u201d"') else ""
    return head + esc(q) + tail


REL_GLOSS = {
    "identical": "asserts the same proposition",
    "paraphrase": "same proposition, different words",
    "subsumes": "an earlier, more general principle that entails the claim",
    "partial": "contained a component of the claim, not the whole",
    "topical": "same subject; does not assert the claim",
    "states": "asserts the same proposition",
    "entails": "what it reports is what you'd expect if the claim is right",
    "anticipates_directionally": "events developed the way the claim's framework points",
    "partially_anticipates": "bears out a component, not the substance",
    "orthogonal": "same subject; neither confirms nor conflicts",
    "contradicts": "the record ran the other way",
}

def fmt(x, pct=False):
    if x is None: return "&mdash;"
    return f"{100*x:.0f}%" if pct else f"{x:.2f}"

def _pdate(s_):
    y,m,d=s_.split("-"); return _date(int(y),int(m),int(d))

# ---------------------------------------------------------------- components
def window_bars(claim):
    wins = claim["windows_pre"] + claim["windows_post"]
    vals = [w["expression"] for w in wins]
    vmax = max(0.12, max(vals, default=0))
    W, HH, pad, gap = 30, 66, 2, 6
    total_w = len(wins)*W + (len(wins)-1)*gap + 2*pad
    parts = [f'<svg width="{total_w}" height="{HH+30}" role="img" aria-label="expression rate per window">']
    n_pre = len(claim["windows_pre"])
    for i, w in enumerate(wins):
        x = pad + i*(W+gap)
        if not w["reliable"]:
            parts.append(f'<rect x="{x}" y="4" width="{W}" height="{HH-4}" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>')
            parts.append(f'<text x="{x+W/2}" y="{HH/2+4}" text-anchor="middle" font-size="8" fill="var(--ink-3)">thin</text>')
        else:
            h = max(2.5, (HH-4) * w["expression"]/vmax); y = HH - h
            fill = "var(--chart)" if i >= n_pre else "var(--chart-soft)"
            parts.append(f'<rect x="{x}" y="{y}" width="{W}" height="{h}" rx="2" fill="{fill}"/>')
            if w["expression"] > 0:
                parts.append(f'<text x="{x+W/2}" y="{y-3}" text-anchor="middle" font-size="8.5" fill="var(--ink-2)">{100*w["expression"]:.0f}%</text>')
        lab = w["window"].replace("t","").replace("d","")
        parts.append(f'<text x="{x+W/2}" y="{HH+11}" text-anchor="middle" font-size="7.5" fill="var(--ink-3)">{esc(lab)}</text>')
    dx = pad + n_pre*(W+gap) - gap/2
    parts.append(f'<line x1="{dx}" y1="0" x2="{dx}" y2="{HH}" stroke="var(--anno-chart)" stroke-width="1.2" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{dx}" y="{HH+24}" text-anchor="middle" font-size="8" fill="var(--anno)">first use</text>')
    parts.append("</svg>")
    return "".join(parts)

def trail(claim):
    items = [e for w in claim["windows_post"] for e in (w.get("examples") or [])]
    items.sort(key=lambda e: e["date"])
    seen: dict = {}; deduped = []
    for e in items:
        k = e["title"][:60].lower().strip() or e["quote"][:80].lower().strip()
        if k in seen: seen[k]["_copies"] = seen[k].get("_copies", 0) + 1
        else: seen[k] = dict(e); deduped.append(seen[k])
    items = deduped
    if not items:
        return '<p class="note">No later document was judged to express, anticipate or contradict this claim &mdash; within this evidence base, it did not travel.</p>'
    rows = []
    for e in items:
        lab, cls = REL_LABEL.get(e["relation"], (e["relation"], "mut"))
        gl = REL_GLOSS.get(e["relation"], "")
        copies = (f'<span class="trail-src">+{e["_copies"]} syndicated '
                  f'{"copy" if e["_copies"]==1 else "copies"}</span>' if e.get("_copies") else "")
        rows.append(
            f'<div class="trail-item"><div class="trail-head">'
            f'<span class="trail-date">{esc(e["date"])}</span>'
            f'<span class="trail-src">{esc(e.get("source") or "")}</span>'
            f'<span class="chip {cls}" title="{esc(gl)}">{esc(lab)} &middot; {e.get("confidence","")}</span>{copies}</div>'
            f'<div class="trail-title">{esc(e["title"])}</div>'
            + (f'<div class="trail-quote">&ldquo;{clean_quote(e["quote"], 280, e.get("url"))}&rdquo;</div>' if e.get("quote") else "")
            + '</div>')
    return '<div class="trail">' + "".join(rows) + "</div>"

def prior_line(c):
    bp = c.get("best_prior")
    if not bp:
        if c.get("prior_strength") is not None:
            return ('<p class="note"><strong>Closest prior statement:</strong> none found '
                    'by the high-recall search &mdash; novelty undefeated.</p>')
        return ""
    gloss = REL_GLOSS.get(bp["relation"], "")
    return (f'<p class="note"><strong>Closest prior statement</strong> '
            f'(defeats {100*c.get("prior_strength",0):.0f}% of novelty): '
            f'{esc(bp["date"])} &middot; {esc(bp.get("source") or bp["title"][:40])} &middot; '
            f'<span class="rel" title="{esc(gloss)}">{esc(bp["relation"])}</span>'
            + (f' <span class="gloss">({esc(gloss)})</span>' if gloss else "")
            + (f' &mdash; &ldquo;{clean_quote(bp["quote"], 180, bp.get("url"))}&rdquo;' if bp.get("quote") else "")
            + '</p>')

def claim_block(c, idx):
    cl = c["claim"]
    first = (f'<p class="note">First stated {c["anchor_moved_back_days"]}&nbsp;days earlier '
             f'({c["anchor_date"]}) in the author&rsquo;s own work; all clocks start there.</p>'
             if c["anchor_moved_back_days"] else "")
    vind = c["vindication"]
    vtxt = "&mdash;" if vind is None else f"{vind:+.2f}"
    return f'''
<details class="claim">
<summary>
  <span class="ckind">{esc(cl["kind"])}</span>
  <span class="ctext">{esc(cl["text"][:180])}</span>
  <span class="cnums">
    <span title="surprisal">S&nbsp;{fmt(c["surprisal"])}</span>
    <span title="adoption">A&nbsp;{fmt(c["adoption"])}</span>
    <span title="vindication">V&nbsp;{vtxt}</span>
    <span class="cpv" title="predictive value">{fmt(c["predictive_value"])}</span>
  </span>
</summary>
<div class="cbody">
  <p class="claim-full">{esc(cl["text"])}</p>
  {first}
  {prior_line(c)}
  <div class="chart-wrap">{window_bars(c)}</div>
  <p class="note" style="margin-top:.2rem">Expression rate among judged candidates per window
  (outlined&nbsp;= before first use; filled&nbsp;= after).</p>
  <h4>Where it showed up later</h4>
  {trail(c)}
</div>
</details>'''

REL_VIS = {
    "states": ("var(--chart)", "circle"), "entails": ("var(--chart)", "circle"),
    "anticipates_directionally": ("var(--chart-soft)", "circle"),
    "partially_anticipates": ("none", "circle"),
    "contradicts": ("var(--anno-chart)", "diamond"),
}

def unit_timeline(u):
    uid = u["key"].replace("-","_")
    lanes=[]; all_dates=[_pdate(u["published"])]
    for ci,c in enumerate(u["claims"]):
        pts={}
        for w in c["windows_post"]:
            for e in (w.get("examples") or []):
                pts.setdefault(e["url"], e)
        pts=sorted(pts.values(), key=lambda e:e["date"])
        for e in pts: all_dates.append(_pdate(e["date"]))
        all_dates.append(_pdate(c["anchor_date"]))
        lanes.append((ci,c,pts,c["anchor_date"]))
    lo=min(all_dates); hi=max(all_dates)
    span=max((hi-lo).days,120); pad_days=int(span*0.04)
    lo=_date.fromordinal(lo.toordinal()-pad_days); hi=_date.fromordinal(hi.toordinal()+pad_days)
    span=(hi-lo).days
    W=660; LX=118; RX=12; TY=18; lane_h=34; HH=TY+len(lanes)*lane_h+34
    def X(d): return LX+(W-LX-RX)*((_pdate(d) if isinstance(d,str) else d)-lo).days/span
    o=[f'<svg class="tl" width="{W}" height="{HH}" role="img" aria-label="Timeline of later texts engaging each claim">']
    for y in range(lo.year, hi.year+1):
        d=_date(y,1,1)
        if lo<=d<=hi:
            x=X(d)
            o.append(f'<line x1="{x:.0f}" y1="{TY-6}" x2="{x:.0f}" y2="{HH-26}" stroke="var(--rule)" stroke-width="1"/>')
            o.append(f'<text x="{x:.0f}" y="{HH-12}" text-anchor="middle" font-size="9" fill="var(--ink-3)">{y}</text>')
    xp=X(u["published"])
    o.append(f'<line x1="{xp:.0f}" y1="{TY-6}" x2="{xp:.0f}" y2="{HH-26}" stroke="var(--ink-2)" stroke-width="1.4"/>')
    o.append(f'<text x="{xp:.0f}" y="{TY-8}" text-anchor="middle" font-size="8.5" font-weight="600" fill="var(--ink-2)">published</text>')
    for ci,c,pts,anchor in lanes:
        y=TY+ci*lane_h+lane_h//2
        o.append(f'<line x1="{LX}" y1="{y}" x2="{W-RX}" y2="{y}" stroke="var(--rule)" stroke-width="1"/>')
        o.append(f'<text x="{LX-8}" y="{y-2}" text-anchor="end" font-size="8.5" font-weight="600" fill="var(--anno)">{esc(c["claim"]["kind"])}</text>')
        o.append(f'<text x="{LX-8}" y="{y+9}" text-anchor="end" font-size="8" fill="var(--ink-3)">{esc(c["claim"]["text"][:15])}&hellip;</text>')
        if anchor != u["published"]:
            xa=X(anchor)
            o.append(f'<path d="M{xa:.0f},{y-7} L{xa:.0f},{y+7}" stroke="var(--anno-chart)" stroke-width="2"/>')
        for e in pts:
            x=X(e["date"])
            fill,shape=REL_VIS.get(e["relation"],("var(--chart-soft)","circle"))
            stroke='var(--chart)' if fill=="none" else "var(--panel)"
            attrs=(f'data-d="{esc(e["date"])}" data-s="{esc(e.get("source") or "")}" '
                   f'data-r="{esc(e["relation"])}" data-t="{esc(e["title"][:110])}" '
                   f'data-q="{clean_quote(e.get("quote",""), 240, e.get("url"))}" data-u="{esc(e.get("url",""))}" '
                   f'data-unit="{uid}"')
            if shape=="diamond":
                o.append(f'<path class="dot" d="M{x:.0f},{y-6} L{x+6:.0f},{y} L{x:.0f},{y+6} L{x-6:.0f},{y} Z" fill="{fill}" stroke="var(--panel)" stroke-width="1.5" {attrs}/>')
            else:
                o.append(f'<circle class="dot" cx="{x:.0f}" cy="{y}" r="5.5" fill="{fill}" stroke="{stroke}" stroke-width="1.5" {attrs}/>')
    o.append('</svg>')
    legend=('<div class="tl-legend">'
      '<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--chart)"/></svg> states / entails</span>'
      '<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--chart-soft)"/></svg> anticipates</span>'
      '<span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="none" stroke="var(--chart)" stroke-width="1.5"/></svg> partial</span>'
      '<span><svg width="14" height="12"><path d="M7,1 L13,6 L7,11 L1,6 Z" fill="var(--anno-chart)"/></svg> contradicts</span>'
      '<span><svg width="12" height="12"><path d="M6,1 L6,11" stroke="var(--anno-chart)" stroke-width="2"/></svg> first use</span></div>')
    return (f'<figure class="tl-fig"><div class="chart-wrap">{"".join(o)}</div>{legend}'
            f'<figcaption>Every later text judged to engage a claim, placed at its publication date. '
            f'Click a dot for the source and its words.</figcaption></figure>'
            f'<div class="tl-detail" id="det_{uid}" hidden></div>')

def unit_card(u):
    cs = u["claims"]
    def m(key):
        vals = [c[key] for c in cs if c.get(key) is not None]
        return sum(vals)/len(vals) if vals else None
    best = max((c.get("predictive_value") or 0) for c in cs) if cs else 0
    horizon = ("month-scale windows &mdash; scored for being right <em>fast</em>"
               if u["horizon"]=="short" else "year-scale windows &mdash; scored for being right <em>early</em>")
    stress = STRESS.get(u["key"], "")
    return f'''
<div class="unit" id="unit-{esc(u["key"])}">
  <div class="unit-head">
    <p class="eyebrow">stresses: {esc(stress)}</p>
    <h3><a href="{esc(u["url"])}">{esc(u["title"])}</a></h3>
    <p class="unit-meta">{esc(u["source"])} &middot; {esc(u["published"])} &middot; {horizon}</p>
  </div>
  <div class="unit-stats">
    <div><div class="unum">{fmt(m("surprisal"))}</div><div class="ulbl">surprisal</div></div>
    <div><div class="unum">{fmt(m("adoption"))}</div><div class="ulbl">adoption</div></div>
    <div><div class="unum">{fmt(m("vindication"))}</div><div class="ulbl">vindication</div></div>
    <div><div class="unum upv">{fmt(best)}</div><div class="ulbl">best claim PV</div></div>
  </div>
  <p class="note unit-note">Averages over the {len(cs)} canvassed claims; expand each claim for its evidence.</p>
  {unit_timeline(u)}
  {''.join(claim_block(c,i) for i,c in enumerate(cs))}
</div>'''

def summary_table():
    rows=[]
    for u in UNITS:
        cs=u["claims"]
        def m(k):
            v=[c[k] for c in cs if c.get(k) is not None]
            return sum(v)/len(v) if v else None
        best=max((c.get("predictive_value") or 0) for c in cs) if cs else 0
        rows.append((best,f'<tr><td><a href="#unit-{esc(u["key"])}" style="text-decoration:none;color:inherit">{esc(u["title"][:46])}</a>'
             f'<div class="note">{esc(u["source"])}</div></td>'
             f'<td class="num">{fmt(m("surprisal"))}</td><td class="num">{fmt(m("adoption"))}</td>'
             f'<td class="num">{fmt(m("vindication"))}</td><td class="num"><strong>{fmt(best)}</strong></td></tr>'))
    rows.sort(key=lambda r:-r[0])
    return ('<div class="tblwrap"><table><tr><th>text</th><th class="num">S&#772;</th>'
            '<th class="num">A&#772;</th><th class="num">V&#772;</th>'
            '<th class="num">best claim PV</th></tr>'+"".join(r[1] for r in rows)+"</table></div>"
            '<p class="note" style="margin-top:.6rem">Ranked by best claim; averages over each '
            'text&rsquo;s ~5 canvassed claims. Click a row to jump to its evidence.</p>')

# ---------------------------------------------------------------- worked example
def worked_example():
    u = _units.get("new-axis")
    if not u: return ""
    c = max(u["claims"], key=lambda c: c.get("predictive_value") or 0)
    cl = c["claim"]; bp = c.get("best_prior") or {}
    pv = c["predictive_value"]; s_=c["surprisal"]; a_=c["adoption"]; v_=c["vindication"]
    ex = None
    for w in c["windows_post"]:
        for e in (w.get("examples") or []):
            if e["relation"] in ("states","entails","anticipates_directionally"):
                if ex is None or e["date"] > ex["date"]: ex = e
    steps = f'''
  <ol class="steps">
    <li><strong>The text.</strong> {esc(u["source"])} publishes
      <em>{esc(u["title"])}</em> on {esc(u["published"])}. Canvassing extracts its
      boldest claims; this one is a <em>{esc(cl["kind"])}</em>:
      <div class="quote">{esc(cl["text"])}</div>
      <p class="note">Claims are the benchmark&rsquo;s <em>restatements</em>, not
      quotations &mdash; each is rewritten to stand alone, which is why it names its
      author in the third person. Everything downstream searches for this proposition
      in other people&rsquo;s words.</p></li>
    <li><strong>Find its true first use.</strong> A search of the author&rsquo;s own
      back catalog finds the claim was first stated
      <strong>{c["anchor_moved_back_days"]} days earlier</strong>, on
      {esc(c["anchor_date"])} &mdash; the article we happened to extract it from was a
      restatement. Every clock below starts at the true first use.</li>
    <li><strong>Was anyone already saying it?</strong> A high-recall search of everything
      dated before first use (expanded queries, lexical, semantic and verbatim-phrase
      retrieval) surfaces the strongest prior statement:
      <div class="quote">{esc(bp.get("date",""))} &middot; {esc(bp.get("source") or "")} &middot;
      <span class="rel">{esc(bp.get("relation",""))}</span>
      <span class="gloss">({esc(REL_GLOSS.get(bp.get("relation",""),""))})</span>
      {("&mdash; &ldquo;"+clean_quote(bp.get("quote",""),160,bp.get("url"))+"&rdquo;") if bp.get("quote") else ""}</div>
      A <em>partial</em> prior is deliberately weak evidence against novelty: one
      component of the claim existed &mdash; here, supply-chain rebuilding as
      industrial policy &mdash; while the claim&rsquo;s substance (reorganizing the
      economy to <em>prepare for great-power conflict</em>) did not. It therefore
      defeats only {100*c.get("prior_strength",0):.0f}% of the claim&rsquo;s novelty:
      <strong>surprisal = {fmt(s_)}</strong>.</li>
    <li><strong>Did the discourse move toward it?</strong> In time windows after first use,
      a random sample of the claim&rsquo;s topical neighbourhood is judged: does each document
      state, entail, anticipate &mdash; or contradict &mdash; the claim?
      Expression rises from a {fmt(c.get("baseline"),pct=True) if c.get("baseline") is not None else "near-zero"}
      baseline to a peak of {fmt(c.get("peak_after"),pct=True)}:
      <strong>adoption = {fmt(a_)}</strong>.
      {("<div class='quote'>"+esc(ex["date"])+" &middot; "+esc(ex.get("source") or "")+" &middot; <span class='rel'>"+esc(ex.get("relation",""))+"</span> <span class='gloss'>("+esc(REL_GLOSS.get(ex.get("relation",""),""))+")</span> &mdash; &ldquo;"+clean_quote(ex.get("quote",""),200,ex.get("url"))+"&rdquo;</div>") if ex else ""}</li>
    <li><strong>Did the record bear it out?</strong> Weighing every supporting judgement
      against every contradicting one: <strong>vindication = {fmt(v_)}</strong>
      (+1 fully borne out, &minus;1 refuted).</li>
    <li><strong>The score.</strong>
      <span class="rel">PV = {fmt(s_)} &times; {fmt(a_)} &times; (1 + {fmt(v_)})/2 =
      <strong>{fmt(pv)}</strong></span> &mdash; said early, picked up late, borne out.
      Every input above is a dated document you can read.</li>
  </ol>'''
    return steps

# ---------------------------------------------------------------- diagram
PIPELINE_SVG = '''
<figure><svg viewBox="0 0 720 250" style="max-width:100%;height:auto" role="img"
  aria-label="Pipeline: a text is canvassed into claims; each claim is anchored at its first use, then scored by a backward prior-art search (surprisal), forward judged windows (adoption), and the signed balance of later evidence (vindication); the three combine into predictive value.">
<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L8,4 L0,8 Z" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="1.3" font-size="12" font-family="IBM Plex Mono,monospace">
  <rect x="8" y="100" width="90" height="46" rx="5"/>
  <text x="53" y="120" text-anchor="middle" fill="currentColor" stroke="none">a text</text>
  <text x="53" y="136" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">post or corpus</text>
  <line x1="98" y1="123" x2="148" y2="123" marker-end="url(#arr)"/>
  <text x="123" y="114" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">canvass</text>
  <rect x="150" y="100" width="96" height="46" rx="5"/>
  <text x="198" y="120" text-anchor="middle" fill="currentColor" stroke="none">claims</text>
  <text x="198" y="136" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">4&ndash;10 boldest</text>
  <line x1="246" y1="123" x2="296" y2="123" marker-end="url(#arr)"/>
  <text x="271" y="114" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">anchor</text>
  <rect x="298" y="100" width="86" height="46" rx="5" stroke="var(--anno-chart)"/>
  <text x="341" y="120" text-anchor="middle" fill="var(--anno)" stroke="none">t&#8321;</text>
  <text x="341" y="136" text-anchor="middle" fill="var(--anno)" stroke="none" font-size="9.5">true first use</text>
  <line x1="384" y1="112" x2="446" y2="52" marker-end="url(#arr)"/>
  <text x="400" y="62" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">search before t&#8321;</text>
  <line x1="384" y1="123" x2="446" y2="123" marker-end="url(#arr)"/>
  <text x="415" y="114" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">judge after</text>
  <line x1="384" y1="134" x2="446" y2="196" marker-end="url(#arr)"/>
  <text x="404" y="192" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">weigh record</text>
  <rect x="448" y="30" width="150" height="44" rx="5"/>
  <text x="523" y="48" text-anchor="middle" fill="currentColor" stroke="none">SURPRISAL</text>
  <text x="523" y="63" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">1 &minus; strongest prior</text>
  <rect x="448" y="101" width="150" height="44" rx="5"/>
  <text x="523" y="119" text-anchor="middle" fill="currentColor" stroke="none">ADOPTION</text>
  <text x="523" y="134" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">peak p &minus; baseline</text>
  <rect x="448" y="172" width="150" height="44" rx="5"/>
  <text x="523" y="190" text-anchor="middle" fill="currentColor" stroke="none">VINDICATION</text>
  <text x="523" y="205" text-anchor="middle" fill="currentColor" stroke="none" font-size="9.5">support vs refute</text>
  <line x1="598" y1="52" x2="655" y2="112" marker-end="url(#arr)"/>
  <line x1="598" y1="123" x2="648" y2="123" marker-end="url(#arr)"/>
  <line x1="598" y1="194" x2="655" y2="134" marker-end="url(#arr)"/>
  <rect x="650" y="101" width="62" height="44" rx="5" stroke-width="2"/>
  <text x="681" y="127" text-anchor="middle" fill="currentColor" stroke="none">PV</text>
</g></svg>
<figcaption>One claim&rsquo;s path through the benchmark. The backward search asks
<em>was it new</em>; the forward windows ask <em>did discourse move toward it</em>; the
signed weighing asks <em>was it right</em>. Each stage emits dated, quoted receipts.</figcaption></figure>'''

# ---------------------------------------------------------------- page
units_html = summary_table() + "\n".join(unit_card(u) for u in UNITS)
ns_sub, nd_sub = CORPUS_KINDS.get("substack", (34, 8068))
ns_news, nd_news = CORPUS_KINDS.get("news", (0, 0))
ns_ac, nd_ac = CORPUS_KINDS.get("academic", (0, 0))
ns_ref, nd_ref = CORPUS_KINDS.get("reference", (0, 0))

page = f'''<title>SignalCatcher Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{{
  --paper:#F4F5F3; --panel:#FBFBFA; --panel-edge:#E3E6E2;
  --ink:#1D2528; --ink-2:#4A5558; --ink-3:#7A8487;
  --accent:#175A6B; --rule:#D8DCD9;
  --chart:#2a78d6; --chart-soft:#2a78d64d; --anno:#B54A17; --anno-chart:#eb6834;
  --pos:#2E6B3F; --pos-bg:#E4EEE6; --neg:#9B3A1C; --neg-bg:#F4E4DC;
  --mut:#5B6669; --mut-bg:#EAEDEB; --quote-bg:#EDF0EE;
}}
@media (prefers-color-scheme: dark){{
  :root:not([data-theme="light"]){{
    --paper:#13181A; --panel:#1A2124; --panel-edge:#2A3437;
    --ink:#E6EAEA; --ink-2:#ADB8BA; --ink-3:#7E8A8D;
    --accent:#6FB4C6; --rule:#2C3639;
    --chart:#3987e5; --chart-soft:#3987e54d; --anno:#E5854E; --anno-chart:#d95926;
    --pos:#7DC28F; --pos-bg:#1E3226; --neg:#E08663; --neg-bg:#33201A;
    --mut:#8B979A; --mut-bg:#232B2E; --quote-bg:#1E272A;
  }}
}}
:root[data-theme="dark"]{{
  --paper:#13181A; --panel:#1A2124; --panel-edge:#2A3437;
  --ink:#E6EAEA; --ink-2:#ADB8BA; --ink-3:#7E8A8D;
  --accent:#6FB4C6; --rule:#2C3639;
  --chart:#3987e5; --chart-soft:#3987e54d; --anno:#E5854E; --anno-chart:#d95926;
  --pos:#7DC28F; --pos-bg:#1E3226; --neg:#E08663; --neg-bg:#33201A;
  --mut:#8B979A; --mut-bg:#232B2E; --quote-bg:#1E272A;
}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
@media (prefers-reduced-motion: reduce){{html{{scroll-behavior:auto}}}}
body{{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif;font-size:17px;line-height:1.62}}
.wrap{{max-width:52rem;margin:0 auto;padding:3.2rem 1.25rem 5rem}}
h1,h2,h3,h4{{font-family:Fraunces,Georgia,serif;line-height:1.15;text-wrap:balance;margin:0}}
h1{{font-size:2.7rem;font-weight:700}} h2{{font-size:1.6rem;font-weight:600;margin-bottom:.9rem}}
h3{{font-size:1.15rem;font-weight:600}} h4{{font-size:1rem;font-weight:600;margin:1.3rem 0 .5rem}}
p{{margin:0 0 1rem}} a{{color:var(--accent)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:.7rem;font-weight:600;
  letter-spacing:.13em;text-transform:uppercase;color:var(--accent);margin:0 0 .55rem}}
section{{margin-top:3.4rem}}
header.hero{{border-bottom:2px solid var(--ink);padding-bottom:1.7rem}}
.hero .sub{{font-size:1.15rem;color:var(--ink-2);margin-top:1rem;max-width:40rem}}
.hero .meta{{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-3);
  margin-top:1.3rem;display:flex;gap:1.5rem;flex-wrap:wrap}}
nav.toc{{margin-top:1.4rem;font-family:"IBM Plex Mono",monospace;font-size:.76rem;
  display:flex;flex-wrap:wrap;gap:.4rem 1.2rem}}
nav.toc a{{text-decoration:none;color:var(--ink-2)}} nav.toc a:hover{{color:var(--accent)}}
.panel{{background:var(--panel);border:1px solid var(--panel-edge);border-radius:6px;
  padding:1.2rem 1.35rem;margin:1.2rem 0}}
.formula{{font-family:"IBM Plex Mono",monospace;font-size:.85rem;line-height:1.9;
  overflow-x:auto;white-space:pre}}
.formula b{{color:var(--accent)}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums}}
th{{font-family:"IBM Plex Mono",monospace;font-size:.67rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--ink-3);text-align:left;font-weight:600;
  padding:.45rem .7rem .45rem 0;border-bottom:1px solid var(--ink)}}
td{{padding:.5rem .7rem .5rem 0;border-bottom:1px solid var(--rule);vertical-align:top}}
td.num,th.num{{text-align:right;font-family:"IBM Plex Mono",monospace;font-size:.84rem}}
.tblwrap,.chart-wrap{{overflow-x:auto}}
.note{{font-size:.88rem;color:var(--ink-2)}}
.chip{{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.65rem;
  font-weight:600;letter-spacing:.05em;padding:.12rem .45rem;border-radius:3px}}
.chip.pos{{color:var(--pos);background:var(--pos-bg)}}
.chip.neg{{color:var(--neg);background:var(--neg-bg)}}
.chip.mut{{color:var(--mut);background:var(--mut-bg)}}
.rel{{font-family:"IBM Plex Mono",monospace;font-size:.82rem;font-weight:600}}
.gloss{{font-size:.82rem;color:var(--ink-3);font-style:italic}}
.quote{{background:var(--quote-bg);border-left:3px solid var(--accent);
  padding:.7rem 1rem;margin:.6rem 0;font-size:.92rem}}
ol.steps{{padding-left:1.3rem}} ol.steps>li{{margin-bottom:1.1rem}}
.unit{{background:var(--panel);border:1px solid var(--panel-edge);border-radius:8px;
  padding:1.5rem 1.5rem 1rem;margin:1.6rem 0}}
.unit-head h3 a{{color:var(--ink);text-decoration:none}}
.unit-head h3 a:hover{{color:var(--accent)}}
.unit-meta{{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-3);margin:.4rem 0 0}}
.unit-stats{{display:flex;gap:2rem;flex-wrap:wrap;margin:1.2rem 0 .3rem}}
.unum{{font-family:Fraunces,serif;font-size:1.9rem;font-weight:700;line-height:1}}
.upv{{color:var(--accent)}}
.ulbl{{font-family:"IBM Plex Mono",monospace;font-size:.64rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3);margin-top:.3rem}}
.unit-note{{margin-bottom:.6rem}}
details.claim{{border-top:1px solid var(--rule)}}
details.claim summary{{display:flex;gap:.8rem;align-items:baseline;cursor:pointer;
  padding:.75rem .2rem;list-style:none}}
details.claim summary::-webkit-details-marker{{display:none}}
details.claim summary:hover{{background:var(--quote-bg)}}
.ckind{{font-family:"IBM Plex Mono",monospace;font-size:.65rem;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;color:var(--anno);flex:0 0 5.2rem}}
.ctext{{flex:1;font-size:.92rem;line-height:1.4}}
.cnums{{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-2);
  display:flex;gap:.8rem;flex:0 0 auto;font-variant-numeric:tabular-nums}}
.cpv{{color:var(--accent);font-weight:600}}
.cbody{{padding:.4rem .2rem 1.3rem;border-top:1px dashed var(--rule)}}
.claim-full{{font-size:.95rem;font-style:italic;color:var(--ink-2)}}
.trail{{display:flex;flex-direction:column;gap:.8rem}}
.trail-item{{border-left:3px solid var(--chart);padding:.15rem 0 .15rem .9rem}}
.trail-head{{display:flex;gap:.7rem;align-items:baseline;flex-wrap:wrap}}
.trail-date{{font-family:"IBM Plex Mono",monospace;font-size:.74rem;font-weight:600}}
.trail-src{{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-3)}}
.trail-title{{font-size:.88rem;margin-top:.15rem}}
.trail-quote{{font-size:.85rem;color:var(--ink-2);background:var(--quote-bg);
  padding:.5rem .7rem;border-radius:4px;margin-top:.35rem}}
.tl-fig{{margin:1rem 0 .6rem}}
.tl .dot{{cursor:pointer}}
.tl .dot:hover{{stroke:var(--ink);stroke-width:2}}
.tl-legend{{display:flex;gap:1.1rem;flex-wrap:wrap;font-family:"IBM Plex Mono",monospace;
  font-size:.68rem;color:var(--ink-2);margin-top:.4rem;align-items:center}}
.tl-legend span{{display:inline-flex;gap:.35rem;align-items:center}}
.tl-detail{{border:1px solid var(--panel-edge);border-left:3px solid var(--chart);
  border-radius:4px;padding:.7rem .9rem;margin:.3rem 0 .9rem;font-size:.88rem;
  background:var(--quote-bg)}}
.tl-detail .dhead{{display:flex;gap:.7rem;flex-wrap:wrap;align-items:baseline;
  font-family:"IBM Plex Mono",monospace;font-size:.74rem}}
.tl-detail .dq{{color:var(--ink-2);margin-top:.35rem}}
figure{{margin:1.4rem 0}} figcaption{{font-size:.85rem;color:var(--ink-2);margin-top:.7rem;line-height:1.5}}
.caveat li{{margin-bottom:.7rem}}
footer{{margin-top:4rem;padding-top:1.1rem;border-top:1px solid var(--rule);
  font-family:"IBM Plex Mono",monospace;font-size:.71rem;color:var(--ink-3);line-height:2}}
@media (max-width:640px){{ .cnums{{flex-basis:100%;order:3}} }}
</style>

<div class="wrap">
<header class="hero">
  <p class="eyebrow">SignalCatcher &middot; Benchmark Explorer</p>
  <h1>Which text saw further?</h1>
  <p class="sub">A benchmark that scores a piece of writing &mdash; a blog post now, a
  publication&rsquo;s whole corpus at scale &mdash; by one test: were its claims
  <strong>new</strong> when written, did the discourse then <strong>move toward
  them</strong>, and did the record <strong>bear them out</strong>? Every number on this
  page unfolds into the dated, quoted documents that produced it.</p>
  <div class="meta"><span>updated Aug 31, 2026</span><span>{N_CLAIMS} claims &middot; {len(UNITS)} texts</span><span>{CORPUS_DOCS:,} dated documents</span></div>
  <nav class="toc">
    <a href="#s1">01 The question</a><a href="#s2">02 One claim, end to end</a>
    <a href="#s3">03 The three numbers</a><a href="#s4">04 Honest judging</a>
    <a href="#s5">05 The evidence base</a><a href="#s6">06 Seven texts, explored</a>
    <a href="#s7">07 Validation</a><a href="#s8">08 Value to an AI model</a>
    <a href="#s9">09 Limits</a>
  </nav>
</header>

<section id="s1">
  <p class="eyebrow">01 &middot; The question</p>
  <h2>What this measures, and what it refuses to measure</h2>
  <p>Ways of valuing writing usually reduce to attention (clicks, shares), prestige
  (who wrote it, where), or vibes. SignalCatcher tries a different question: treat a text
  as a set of specific, checkable <strong>claims</strong>, and ask where each claim sits
  in time relative to everything else on its subject. A text is valuable here when it
  said things that <em>weren&rsquo;t being said</em>, that discourse <em>later converged
  on</em>, and that the record <em>vindicated</em>.</p>
  <p>Two deliberate refusals shape the design. It does <strong>not</strong> ask whether
  later writers <em>read</em> the text &mdash; causal influence is unknowable from text
  alone, and a writer who saw where things were going without being the cause still saw
  it. And it does <strong>not</strong> reward being right about what everyone already
  expected: a correct prediction that was common knowledge scores zero, on purpose.</p>
  {PIPELINE_SVG}
</section>

<section id="s2">
  <p class="eyebrow">02 &middot; Worked example</p>
  <h2>One claim, end to end</h2>
  <p>The highest-scoring claim in the pilot, walked through every stage. This is the
  whole benchmark in miniature; everything after this section is generalisation.</p>
  {worked_example()}
</section>

<section id="s3">
  <p class="eyebrow">03 &middot; The measure</p>
  <h2>Surprisal &middot; Adoption &middot; Vindication</h2>
  <div class="panel"><div class="formula"><b>surprisal</b>    1 &minus; strength of the closest PRIOR statement found by a hard search   (higher = nobody had said it)
<b>adoption</b>     how much more common the claim became after first use, at peak     (higher = discourse moved toward it)
<b>vindication</b>  signed balance of later evidence: borne out (+1) vs refuted (&minus;1)

<b>predictive value</b> = surprisal &times; adoption &times; (1 + vindication) / 2</div></div>
  <p><strong>Surprisal is an existence test, not a popularity test.</strong> A hard
  search &mdash; query expansion into other vocabularies, lexical, semantic and
  verbatim-phrase retrieval over everything dated before the claim&rsquo;s first use
  &mdash; hunts for anyone who already said it. The single strongest prior defeats
  novelty in proportion to how fully it anticipates the claim, and is shown with every
  claim as a dated, quoted receipt. (An earlier design measured how <em>often</em> the
  claim appeared beforehand; since any specific proposition is rare even among documents
  on its own topic, every claim scored 0.85&ndash;1.0 and the number did no work. That
  design is retired; &sect;07 records how it was caught.)</p>
  <p><strong>Adoption is a rate, correctly.</strong> Discourse <em>moving</em> is a
  frequency phenomenon: in each time window after first use, a random sample of the
  claim&rsquo;s topical neighbourhood is judged, and adoption is the rise in weighted
  expression over the pre-first-use baseline. Random sampling keeps windows of very
  different sizes comparable; the top-ranked candidates are judged too, but feed the
  evidence trail and vindication, not the rate.</p>
  <p><strong>Vindication keeps &ldquo;wrong&rdquo; distinct from &ldquo;ignored.&rdquo;</strong>
  Both score low PV; only one is contradicted by named documents. A claim events refuted
  is crushed by the vindication factor <em>and labeled as refuted</em> &mdash; the two
  failure modes never blur.</p>
  <p class="note">The three are reported separately for every claim because they answer
  different questions: highly surprising but never adopted (ahead of its time, or
  wrong-footed); common and adopted (riding a wave it didn&rsquo;t start); surprising,
  adopted, then refuted. The composite is a summary, never a substitute.</p>
</section>

<section id="s4">
  <p class="eyebrow">04 &middot; The judging</p>
  <h2>How a language-model judge is kept honest</h2>
  <p>Two questions are asked of a judge (Claude, at high reasoning effort); everything
  else is arithmetic. <em>Extraction</em>: what substantive claims does this text make,
  stated or implied &mdash; each restated so it could be searched for in another
  writer&rsquo;s work without the original&rsquo;s vocabulary. <em>Anticipation</em>: for
  each dated document put before it, does that document&rsquo;s content bear out, follow
  from, or run against what the claim said?</p>
  <div class="tblwrap"><table>
    <tr><th>relation (forward)</th><th>meaning</th><th class="num">weight</th></tr>
    <tr><td class="rel" style="color:var(--pos)">states</td><td>asserts the same proposition</td><td class="num">+1.0</td></tr>
    <tr><td class="rel" style="color:var(--pos)">entails</td><td>what it reports is what you&rsquo;d expect if the claim is right &mdash; where <em>implicit</em> content earns credit</td><td class="num">+0.9</td></tr>
    <tr><td class="rel" style="color:var(--pos)">anticipates directionally</td><td>events developed the way the claim&rsquo;s framework points</td><td class="num">+0.5</td></tr>
    <tr><td class="rel" style="color:var(--pos)">partially anticipates</td><td>bears out a component, not the substance</td><td class="num">+0.3</td></tr>
    <tr><td class="rel">orthogonal</td><td>same subject; neither confirms nor conflicts</td><td class="num">0</td></tr>
    <tr><td class="rel" style="color:var(--neg)">contradicts</td><td>the record ran the other way</td><td class="num">&minus;1.0</td></tr>
  </table></div>
  <p class="note" style="margin-top:.6rem">The backward (prior-art) direction uses a
  parallel scale &mdash; identical / paraphrase / subsumes / partial / topical &mdash;
  where <em>subsumes</em> matters most: an earlier, more general principle that entails
  the claim defeats novelty even with no shared wording.</p>
  <h4>Four disciplines, enforced in code rather than requested in prompts</h4>
  <div class="tblwrap"><table>
    <tr><th>discipline</th><th>what it prevents</th></tr>
    <tr><td><strong>Quote or downgrade.</strong> Any verdict stronger than orthogonal must include a verbatim span from the document, checked mechanically against the text; unquotable verdicts are demoted automatically.</td><td>the judge asserting matches from its training memory rather than from the evidence in front of it</td></tr>
    <tr><td><strong>Hindsight flows one way.</strong> The judge sees only dated excerpts and may not use what it remembers about how things turned out; a vague claim is never upgraded because history obliged it.</td><td>scoring fame instead of foresight</td></tr>
    <tr><td><strong>Anchor at true first use.</strong> Before scoring, the author&rsquo;s back catalog is searched for the earliest statement; all clocks start there, not at the article the claim was extracted from.</td><td>crediting a restatement as an origination (measured: one claim&rsquo;s clock moved 761 days)</td></tr>
    <tr><td><strong>Syndication collapses before counting.</strong> Near-duplicate detection folds wire copies into one source, before judging and again at display.</td><td>one AP story in twenty outlets reading as twenty independent adoptions (measured: 26&ndash;31% of fetched news was syndicated)</td></tr>
  </table></div>
</section>

<section id="s5">
  <p class="eyebrow">05 &middot; The evidence base</p>
  <h2>A spine, a shell, and a manifest</h2>
  <p>No corpus can hold &ldquo;the whole discourse,&rdquo; and none needs to: the right
  evidence base is <em>claim-shaped</em>. The architecture is a small persistent
  <strong>spine</strong> &mdash; a fixed panel of {ns_sub} independent writers
  ({nd_sub:,} dated posts) that every evaluation shares, which is what keeps texts
  comparable &mdash; plus a per-text <strong>shell</strong>: evidence fetched by a fixed
  recipe from fixed channels, dense exactly where this text&rsquo;s claims live. A
  <strong>manifest</strong> records every query, hit count and failure, distinguishing
  &ldquo;searched and found nothing&rdquo; from &ldquo;could not search.&rdquo;</p>
  <div class="tblwrap"><table>
    <tr><th>channel</th><th>source</th><th>role</th></tr>
    <tr><td>news</td><td>GDELT index, keyword-searched per claim, prior year included</td><td>pickup beyond the blogosphere; prior art the trade press already broke</td></tr>
    <tr><td>international</td><td>fixed panel of leading dailies via GDELT (Le Monde, FAZ, Spiegel, Corriere, El Pa&iacute;s, El Observador, Asahi, SCMP&hellip;)</td><td>cross-border, cross-language diffusion</td></tr>
    <tr><td>academic</td><td>arXiv abstracts, date-ranged</td><td>scholarly prior art &mdash; the channel that stops aggregators inheriting researchers&rsquo; foresight</td></tr>
    <tr><td>reference</td><td>Wikipedia article-creation dates per language (Wayback first-capture as fallback)</td><td>a datable &ldquo;reached common knowledge&rdquo; milestone, per language</td></tr>
    <tr><td>forum</td><td>Hacker News</td><td>dated discovery and discussion</td></tr>
  </table></div>
  <p style="margin-top:1rem">Current pinned evidence: <strong>{CORPUS_DOCS:,} documents,
  {CORPUS_SPAN}</strong> &mdash; {nd_sub:,} commentary posts from {ns_sub} writers,
  {nd_news:,} news articles from {ns_news} outlets, {nd_ac} academic abstracts,
  {nd_ref} reference-layer entries. Every document carries a publisher timestamp;
  publication date is the load-bearing field of the whole benchmark.</p>
  <p class="note"><strong>Skew, stated plainly:</strong> dense in Anglophone tech,
  economics and policy commentary; news exists mainly around scored claims; no paywalled
  mainstream full text (NYT/WSJ appear only through syndicated echoes). All scores are
  relative to this evidence base and are reported with the search coverage that
  produced them.</p>
</section>

<section id="s6">
  <p class="eyebrow">06 &middot; The results</p>
  <h2>Seven texts, explored</h2>
  <p>Texts chosen to stress different parts of the measure &mdash; long and short
  horizons, famous and unknown authors, original and aggregated content. Each card
  opens with a timeline of every later text that engaged its claims (click a dot for
  the quote), then the claims themselves, each expandable to its receipts.</p>
  {units_html}
</section>

<section id="s7">
  <p class="eyebrow">07 &middot; Validation</p>
  <h2>How we know it measures anything</h2>
  <p>A score is not a measurement until something could have shown it wrong. Controls
  are predictions that must hold if the benchmark works:</p>
  <div class="tblwrap"><table>
    <tr><th>control</th><th>prediction</th><th>result</th></tr>
    <tr><td>Date shift</td><td>claims re-dated 4 years later must find their own era as prior art (prior-art strength 1.00 &rarr; 0.80)</td><td><span class="chip pos">PASS</span></td></tr>
    <tr><td>No retrieval</td><td>withhold the evidence and scores must collapse &mdash; they did, so the judge reads the corpus, not its memory</td><td><span class="chip pos">PASS</span></td></tr>
    <tr><td>Judge agreement</td><td>a different reasoning setting shifts verdicts &le;0.15 (measured 0.11)</td><td><span class="chip pos">PASS</span></td></tr>
    <tr><td>Shuffled dates &middot; decoy writer</td><td>wrong dates must destroy adoption; a contemporaneous decoy must score lower &mdash; now decidable, next scheduled run</td><td><span class="chip mut">PENDING</span></td></tr>
  </table></div>
  <h4>Failures the harness caught in itself</h4>
  <p>The strongest evidence the machinery is falsifiable in practice is that it has
  already falsified parts of itself. Each of these was found by a control, a validity
  check or a receipts audit &mdash; and each fix changed published numbers:</p>
  <ul class="caveat">
    <li><strong>Surprisal compression.</strong> The rate-based design scored all 30 claims 0.81&ndash;1.00. Predicted test: a fact reported everywhere the day before should crash. It did &mdash; the SVB collapse claim fell 0.87&nbsp;&rarr;&nbsp;0.14, receipt attached (a March&nbsp;10 wire paraphrase).</li>
    <li><strong>Aggregator credit.</strong> A content mill&rsquo;s borrowed data-exhaustion thesis scored as novel until the arXiv channel put the actual academic prior in reach; it fell to 0.15 with the paper cited.</li>
    <li><strong>Dense-window bias.</strong> Judging the top-10 of 1,187 candidates is a more extreme selection than top-10 of 20; adoption rates now come from seeded random samples so windows compare.</li>
    <li><strong>A confounded control.</strong> Date-shift originally compared a coverage-blended score, letting a coverage gain cancel the prior-art effect it existed to detect; it now compares the evidence-only term.</li>
    <li><strong>Vacuous verdicts.</strong> Controls with no signal to test reported FAIL (or worse, a decoy condition that never ran was averaged in as 0.0, inflating source value); both now report &ldquo;not decidable,&rdquo; loudly.</li>
    <li><strong>Numbers without receipts.</strong> A serialization bug shipped scores whose evidence trails were silently empty; trails are now a rendered, clickable part of every claim.</li>
  </ul>
</section>

<section id="s8">
  <p class="eyebrow">08 &middot; The AI connection</p>
  <h2>What a text is worth to a model</h2>
  <p>The same claim layer supports a second measurement: grade a model on questions only
  answerable by someone who has read a source&rsquo;s claims &mdash; <em>(a)</em>
  unaided, <em>(b)</em> given contemporaneous writing by others on the same topics,
  <em>(c)</em> given the source itself. Inference-time value is the excess of (c) over
  the best alternative; a correct closed-book answer to a claim that <em>originated</em>
  with the source is evidence its contribution is already priced into the weights.</p>
  <div class="tblwrap"><table>
    <tr><th>source</th><th class="num">model alone</th><th class="num">+ others&rsquo; writing</th><th class="num">+ this source</th><th class="num">inference value</th><th class="num">already in model</th></tr>
    <tr><td>Brian Potter <span class="note">(Construction Physics)</span></td><td class="num">0.54</td><td class="num">n/a*</td><td class="num">0.98</td><td class="num"><strong>+0.44</strong></td><td class="num">33%</td></tr>
    <tr><td>Scott Alexander <span class="note">(Astral Codex Ten)</span></td><td class="num">0.90</td><td class="num">0.80</td><td class="num">0.88</td><td class="num"><strong>&minus;0.05</strong></td><td class="num">100%</td></tr>
  </table></div>
  <p style="margin-top:1rem">The contrast is the finding: the model already knows
  essentially everything ACX argued &mdash; that value was captured at <em>training</em>
  time &mdash; while niche construction reporting still carries large inference-time
  value. Different assets; a single number would hide the distinction that matters most
  to a publisher and a lab alike. The deeper connection: a source that predicts future
  discourse is precisely a source that improves a model &mdash; the publisher-facing and
  lab-facing measurements share one claim layer.</p>
  <p class="note">* No contemporaneous construction coverage existed to build the decoy
  condition, so it is reported as unavailable, never as zero. Measured Aug 2026 on 6
  claims per source; indicative, not stable.</p>
</section>

<section id="s9">
  <p class="eyebrow">09 &middot; Read before quoting</p>
  <h2>Limits and open problems</h2>
  <ul class="caveat">
    <li><strong>{N_CLAIMS} claims across {len(UNITS)} texts</strong> &mdash; a validated pipeline demonstrated at pilot scale. Per-text averages over ~5 claims are indicative; the interesting comparisons are claim-level, receipt-backed.</li>
    <li><strong>Everything is corpus-relative.</strong> Surprisal is bounded by what the evidence base contains; a prior living in a paywalled archive, a podcast, or an untapped language is invisible. The shell recipe narrows this per-claim; it cannot close it.</li>
    <li><strong>The judge is a model</strong>, constrained by grounding, quote-enforcement and measured agreement (&plusmn;0.11) &mdash; individual verdicts can still err in either direction; every one is stored with its quote for audit.</li>
    <li><strong>The recipe is part of the instrument.</strong> A biased &ldquo;where to look&rdquo; biases scores invisibly; the mitigations are a fixed, versioned recipe (shell-v1), manifests, and coverage discounts &mdash; not neutrality.</li>
    <li><strong>Unbuilt:</strong> podcast transcripts as a channel; a broader ring of small publications; decade-scale durability; a gold set of claims with independently known priority to calibrate the judge against; the two pending controls at current scale.</li>
  </ul>
</section>

<footer>
  SignalCatcher &middot; evidence: {CORPUS_DOCS:,} dated documents ({CORPUS_SPAN}) &middot; judge: Claude (Opus), grounded &amp; quote-enforced &middot; recipe shell-v1<br>
  every verdict stored with its verbatim supporting quote &middot; per-unit manifests in data/manifests/ &middot; scores reproducible from pinned corpus + config hash
</footer>
</div>
<div id="tip" style="position:fixed;pointer-events:none;background:var(--ink);color:var(--paper);font-family:'IBM Plex Mono',monospace;font-size:.72rem;padding:.35rem .55rem;border-radius:4px;opacity:0;transition:opacity .12s;z-index:9;white-space:nowrap;max-width:70vw;overflow:hidden;text-overflow:ellipsis"></div>
<script>
(function(){{
  const tip=document.getElementById('tip');
  document.querySelectorAll('.tl .dot').forEach(d=>{{
    d.addEventListener('mousemove',e=>{{
      tip.textContent=d.dataset.d+' \\u00b7 '+(d.dataset.s||'?')+' \\u00b7 '+d.dataset.r;
      tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY-10)+'px'; tip.style.opacity=1;
    }});
    d.addEventListener('mouseleave',()=>tip.style.opacity=0);
    d.addEventListener('click',()=>{{
      const det=document.getElementById('det_'+d.dataset.unit);
      if(!det) return;
      det.hidden=false;
      det.innerHTML='<div class="dhead"><strong>'+d.dataset.d+'</strong><span>'+
        (d.dataset.s||'')+'</span><span class="chip '+(d.dataset.r==='contradicts'?'neg':'pos')+'">'+
        d.dataset.r.replace(/_/g,' ')+'</span></div><div>'+
        (d.dataset.u?('<a href="'+d.dataset.u+'" target="_blank" rel="noopener">'+d.dataset.t+'</a>'):d.dataset.t)+
        '</div>'+(d.dataset.q?('<div class="dq">\\u201c'+d.dataset.q+'\\u201d</div>'):'');
      det.scrollIntoView({{behavior:'smooth',block:'nearest'}});
    }});
  }});
}})();
</script>'''

out = Path(sys.argv[1] if len(sys.argv)>1 else "data/readout_v3.html")
out.write_text(page)
print(f"wrote {out} ({len(page):,} bytes), {N_CLAIMS} claims, {len(UNITS)} units")
