"""Render the units-pilot readout from data/units_report.json.

Static HTML, no framework: every claim row is a <details> whose body holds the
window-by-window prevalence bars and the adoption trail -- the dated, quoted
judgements that let a reader verify the number by eye.
"""
from __future__ import annotations

import html as H
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = json.loads(Path("data/units_report.json").read_text())

REL_LABEL = {
    "states": ("states it", "pos"), "entails": ("entails it", "pos"),
    "anticipates_directionally": ("anticipates", "pos"),
    "partially_anticipates": ("partial", "pos"),
    "orthogonal": ("orthogonal", "mut"), "contradicts": ("contradicts", "neg"),
}

def esc(s): return H.escape(str(s or ""))

def fmt(x, pct=False):
    if x is None: return "&mdash;"
    return f"{100*x:.0f}%" if pct else f"{x:.2f}"

def window_bars(claim):
    """Tiny static bar chart: expression per window, pre in outline, post filled."""
    wins = claim["windows_pre"] + claim["windows_post"]
    vals = [w["expression"] for w in wins]
    vmax = max(0.12, max(vals, default=0))
    W, HH, pad, gap = 30, 66, 2, 6
    total_w = len(wins)*W + (len(wins)-1)*gap + 2*pad
    parts = [f'<svg width="{total_w}" height="{HH+30}" role="img" aria-label="expression rate per window">']
    n_pre = len(claim["windows_pre"])
    for i, w in enumerate(wins):
        x = pad + i*(W+gap)
        h = max(2, (HH-4) * w["expression"]/vmax) if w["reliable"] else 0
        y = HH - h
        if not w["reliable"]:
            parts.append(f'<rect x="{x}" y="4" width="{W}" height="{HH-4}" fill="none" stroke="var(--rule)" stroke-dasharray="3 3"/>')
            parts.append(f'<text x="{x+W/2}" y="{HH/2+4}" text-anchor="middle" font-size="8" fill="var(--ink-3)">thin</text>')
        else:
            fill = "var(--chart)" if i >= n_pre else "var(--chart-soft)"
            parts.append(f'<rect x="{x}" y="{y}" width="{W}" height="{h}" rx="2" fill="{fill}"/>')
            if w["expression"] > 0:
                parts.append(f'<text x="{x+W/2}" y="{y-3}" text-anchor="middle" font-size="8.5" fill="var(--ink-2)">{100*w["expression"]:.0f}%</text>')
        lab = w["window"].replace("t","").replace("d","")
        parts.append(f'<text x="{x+W/2}" y="{HH+11}" text-anchor="middle" font-size="7.5" fill="var(--ink-3)">{esc(lab)}</text>')
    # divider at t=0
    dx = pad + n_pre*(W+gap) - gap/2
    parts.append(f'<line x1="{dx}" y1="0" x2="{dx}" y2="{HH}" stroke="var(--anno-chart)" stroke-width="1.2" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{dx}" y="{HH+24}" text-anchor="middle" font-size="8" fill="var(--anno)">first use</text>')
    parts.append("</svg>")
    return "".join(parts)

def trail(claim):
    items = []
    for w in claim["windows_post"]:
        for ex in w.get("examples") or []:
            items.append(ex)
    items.sort(key=lambda e: e["date"])
    # Residual wire syndication: the ingest collapses copies within a batch, but
    # copies arriving via different claims' fetches slip through. Collapse them
    # here by matching title/quote text, keep the earliest, and say how many
    # copies rode behind it -- the reader should see one pickup, labelled, not
    # three rows implying three independent outlets.
    seen: dict[str, dict] = {}
    deduped = []
    for e in items:
        key = (e["title"][:60].lower().strip(), e["quote"][:80].lower().strip())
        k = key[0] if len(key[0]) > 15 else key[0]+key[1]
        if k in seen:
            seen[k]["_copies"] = seen[k].get("_copies", 0) + 1
        else:
            seen[k] = dict(e)
            deduped.append(seen[k])
    items = deduped
    if not items:
        return '<p class="note">No later document was judged to express, anticipate or contradict this claim &mdash; within this corpus, it did not travel.</p>'
    rows = []
    for e in items:
        lab, cls = REL_LABEL.get(e["relation"], (e["relation"], "mut"))
        src = esc(e.get("source") or "")
        copies = (f'<span class="trail-src">+{e["_copies"]} syndicated '
                  f'{"copy" if e["_copies"]==1 else "copies"}</span>'
                  if e.get("_copies") else "")
        rows.append(
            f'<div class="trail-item"><div class="trail-head">'
            f'<span class="trail-date">{esc(e["date"])}</span>'
            f'<span class="trail-src">{src}</span>'
            f'<span class="chip {cls}">{esc(lab)} &middot; {e.get("confidence","")}</span>{copies}</div>'
            f'<div class="trail-title">{esc(e["title"])}</div>'
            + (f'<div class="trail-quote">&ldquo;{esc(e["quote"])}&rdquo;</div>' if e.get("quote") else "")
            + '</div>')
    return '<div class="trail">' + "".join(rows) + "</div>"

def claim_block(c, idx):
    cl = c["claim"]
    pv = c["predictive_value"]
    first = (f'<p class="note">First stated {c["anchor_moved_back_days"]}&nbsp;days earlier '
             f'({c["anchor_date"]}) in the author&rsquo;s own work; all windows anchored there.</p>'
             if c["anchor_moved_back_days"] else "")
    vind = c["vindication"]
    vtxt = "&mdash; (no engaged evidence)" if vind is None else f"{vind:+.2f}"
    return f'''
<details class="claim">
<summary>
  <span class="ckind">{esc(cl["kind"])}</span>
  <span class="ctext">{esc(cl["text"][:180])}</span>
  <span class="cnums">
    <span title="surprisal">S&nbsp;{fmt(c["surprisal"])}</span>
    <span title="adoption">A&nbsp;{fmt(c["adoption"])}</span>
    <span title="vindication">V&nbsp;{vtxt}</span>
    <span class="cpv" title="predictive value">{fmt(pv)}</span>
  </span>
</summary>
<div class="cbody">
  <p class="claim-full">{esc(cl["text"])}</p>
  {first}
  <div class="chart-wrap">{window_bars(c)}</div>
  <p class="note" style="margin-top:.2rem">Expression rate among the ~10 most-plausible documents per window
  (outlined&nbsp;= before first use; filled&nbsp;= after).</p>
  <h4>Where it showed up later</h4>
  {trail(c)}
</div>
</details>'''

def unit_card(u):
    cs = u["claims"]
    def m(key):
        vals = [c[key] for c in cs if c.get(key) is not None]
        return sum(vals)/len(vals) if vals else None
    best = max((c.get("predictive_value") or 0) for c in cs) if cs else 0
    horizon = ("windows in months &mdash; scored for being right <em>fast</em>"
               if u["horizon"]=="short" else "windows in years &mdash; scored for being right <em>early</em>")
    return f'''
<div class="unit">
  <div class="unit-head">
    <p class="eyebrow">{esc(u["label"])}</p>
    <h3><a href="{esc(u["url"])}">{esc(u["title"])}</a></h3>
    <p class="unit-meta">{esc(u["source"])} &middot; {esc(u["published"])} &middot; {horizon}</p>
  </div>
  <div class="unit-stats">
    <div><div class="unum">{fmt(m("surprisal"))}</div><div class="ulbl">surprisal</div></div>
    <div><div class="unum">{fmt(m("adoption"))}</div><div class="ulbl">adoption</div></div>
    <div><div class="unum">{fmt(m("vindication"))}</div><div class="ulbl">vindication</div></div>
    <div><div class="unum upv">{fmt(best)}</div><div class="ulbl">best claim PV</div></div>
  </div>
  <p class="note unit-note">Averages over the {len(cs)} canvassed claims; expand each for its evidence.</p>
  {''.join(claim_block(c,i) for i,c in enumerate(cs))}
</div>'''

units_html = "\n".join(unit_card(u) for u in DATA["units"])
n_claims = sum(len(u["claims"]) for u in DATA["units"])

page = f'''<title>SignalCatcher Readout</title>
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
body{{margin:0;background:var(--paper);color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif;font-size:17px;line-height:1.62}}
.wrap{{max-width:52rem;margin:0 auto;padding:3.2rem 1.25rem 5rem}}
h1,h2,h3,h4{{font-family:Fraunces,Georgia,serif;line-height:1.15;text-wrap:balance;margin:0}}
h1{{font-size:2.7rem;font-weight:700}} h2{{font-size:1.6rem;font-weight:600;margin-bottom:.9rem}}
h3{{font-size:1.15rem;font-weight:600}} h4{{font-size:1rem;font-weight:600;margin:1.3rem 0 .5rem}}
p{{margin:0 0 1rem}} a{{color:var(--accent)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:.7rem;font-weight:600;
  letter-spacing:.13em;text-transform:uppercase;color:var(--accent);margin:0 0 .55rem}}
section{{margin-top:3.2rem}}
header.hero{{border-bottom:2px solid var(--ink);padding-bottom:1.7rem}}
.hero .sub{{font-size:1.15rem;color:var(--ink-2);margin-top:1rem;max-width:40rem}}
.hero .meta{{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-3);
  margin-top:1.3rem;display:flex;gap:1.5rem;flex-wrap:wrap}}
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
td.num{{text-align:right;font-family:"IBM Plex Mono",monospace;font-size:.84rem}}
.tblwrap,.chart-wrap{{overflow-x:auto}}
.note{{font-size:.88rem;color:var(--ink-2)}}
.chip{{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:.65rem;
  font-weight:600;letter-spacing:.05em;padding:.12rem .45rem;border-radius:3px}}
.chip.pos{{color:var(--pos);background:var(--pos-bg)}}
.chip.neg{{color:var(--neg);background:var(--neg-bg)}}
.chip.mut{{color:var(--mut);background:var(--mut-bg)}}
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
footer{{margin-top:4rem;padding-top:1.1rem;border-top:1px solid var(--rule);
  font-family:"IBM Plex Mono",monospace;font-size:.71rem;color:var(--ink-3);line-height:2}}
@media (max-width:640px){{ .cnums{{flex-basis:100%;order:3}} }}
</style>

<div class="wrap">
<header class="hero">
  <p class="eyebrow">SignalCatcher &middot; Pilot Readout &middot; Three Texts</p>
  <h1>Which text saw further?</h1>
  <p class="sub">The unit of evaluation is <strong>a text</strong> &mdash; here a blog post, at
  larger scale a publication&rsquo;s whole corpus. Each text is canvassed for its 4&ndash;10
  boldest claims, and every claim is scored on three separate axes against a dated
  corpus: was it <strong>surprising</strong> when written, was it <strong>adopted</strong> by
  later discourse, and was it <strong>vindicated</strong> by the record? Expand any claim to
  see exactly where &mdash; and in whose words &mdash; it resurfaced.</p>
  <div class="meta"><span>Aug 27, 2026</span><span>{n_claims} claims scored across 3 texts</span><span>corpus: 8,000+ dated documents, 2018&ndash;2026</span></div>
</header>

<section>
  <p class="eyebrow">01 &middot; How to read the three numbers</p>
  <h2>Surprisal &middot; Adoption &middot; Vindication</h2>
  <div class="panel"><div class="formula"><b>surprisal</b>    1 &minus; how common the claim was BEFORE its author first stated it     (0&ndash;1, higher = more original)
<b>adoption</b>     how much more common it became after, at peak, minus baseline     (higher = discourse moved toward it)
<b>vindication</b>  balance of later evidence: bore it out (+1) vs refuted it (&minus;1)

<b>predictive value</b> = surprisal &times; adoption &times; (1 + vindication) / 2</div></div>
  <p>The three components are reported separately for every claim because they answer
  different questions a reader might care about. A claim can be highly surprising and
  never adopted (ahead of its time, or just wrong-footed); common and adopted (riding a
  wave it didn&rsquo;t start); or surprising, adopted, and then <em>refuted</em> &mdash;
  which the vindication term keeps visibly distinct from being ignored. The composite is a
  summary, not a substitute.</p>
  <p class="note">Prevalence is always measured inside the claim&rsquo;s own topical
  neighbourhood &mdash; of documents discussing this subject at all, what share express this
  claim? &mdash; judged by a grounded language model that must quote its evidence verbatim
  or have its verdict downgraded in code. The author&rsquo;s own writing is excluded, and
  each claim&rsquo;s clock starts at its <em>true first use</em> in the author&rsquo;s back
  catalog, not the article we happened to pull it from.</p>
</section>

<section>
  <p class="eyebrow">02 &middot; The Units</p>
  <h2>Three texts, side by side</h2>
  <p>Chosen to stress different parts of the measure: a slow-burn geopolitical frame, a
  set of concrete near-term predictions, and a day-one interpretation of a breaking
  event scored on <em>month</em>-scale windows rather than years.</p>
  {units_html}
</section>

<section>
  <p class="eyebrow">03 &middot; The Corpus</p>
  <h2>What these scores are measured against</h2>
  <p>A pinned, dated snapshot: <strong>8,000+ documents, 2018&ndash;2026</strong>, from 34
  independent commentary writers (Substack/blogs &mdash; the discourse the units live in)
  plus news press fetched from the GDELT index around each scored claim, with wire-service
  syndication collapsed so twenty copies of one story count once. Every document carries a
  publisher timestamp; every verdict is stored with the verbatim quote that supports it.</p>
  <p class="note"><strong>Skew, stated plainly:</strong> the corpus is dense in tech,
  economics and policy commentary, thin elsewhere; the news layer exists only around
  scored claims. Scores are corpus-relative. Topical-neighbourhood ratios keep them
  meaningful across uneven density, but a claim whose natural audience is outside this
  corpus will under-measure.</p>
</section>

<section>
  <p class="eyebrow">04 &middot; Method checks</p>
  <h2>Controls</h2>
  <div class="tblwrap"><table>
    <tr><th>control</th><th>prediction that must hold</th><th>result</th></tr>
    <tr><td>Date shift</td><td>Claims re-dated 4 years later must find their own era as prior art (1.00 &rarr; 0.80)</td><td><span class="chip pos">PASS</span></td></tr>
    <tr><td>No retrieval</td><td>Withhold the evidence; scores must collapse &mdash; they did, so the judge reads the corpus, not its memory</td><td><span class="chip pos">PASS</span></td></tr>
    <tr><td>Judge agreement</td><td>Different reasoning setting, verdict shift &le;0.15 (measured 0.11)</td><td><span class="chip pos">PASS</span></td></tr>
    <tr><td>Shuffled dates / decoy writer</td><td>Need the multi-unit sample on this page to decide; previously no signal to destroy</td><td><span class="chip mut">PENDING RE-RUN</span></td></tr>
  </table></div>
</section>

<section>
  <p class="eyebrow">05 &middot; Read before quoting</p>
  <h2>Caveats</h2>
  <ul>
    <li style="margin-bottom:.6rem"><strong>Three texts, {n_claims} claims</strong> &mdash; a working pilot, not a survey. Unit-level averages over ~5 claims are indicative, not stable.</li>
    <li style="margin-bottom:.6rem"><strong>Dense-window bias:</strong> each window judges its ~10 most-plausible candidates; windows with many topical documents get a more extreme top-10, biasing late/dense windows upward. The curve shapes survive this; exact percentages should not be quoted as precise.</li>
    <li style="margin-bottom:.6rem"><strong>The judge is a model</strong>, constrained by grounding, quote-enforcement and the controls above &mdash; individual verdicts can still err; each is stored with its quote for audit.</li>
  </ul>
</section>

<footer>SignalCatcher &middot; run {esc(DATA.get("run_id",""))} &middot; judgements: Claude (Opus), grounded &amp; quote-enforced &middot; every claim expandable to its dated evidence above</footer>
</div>'''

out = Path(sys.argv[1] if len(sys.argv)>1 else "data/readout_v2.html")
out.write_text(page)
print(f"wrote {out} ({len(page):,} bytes), {n_claims} claims, {len(DATA['units'])} units")
