"""The evidence shell: a per-unit corpus, assembled by a fixed recipe.

The benchmark cannot host "the whole discourse", and does not need to: the right
evidence base is claim-shaped. For each tested text the shell recipe derives
where to look from the canvassed claims themselves, pulls from a FIXED set of
channels with fixed depth targets, and records a manifest of everything it did.
The recipe being fixed is what keeps units comparable; the manifest is what
keeps a run reproducible after the bulk data is discarded.

Channels (v1):
  spine       the persistent reference panel (blogger cluster) -- always present
  news        GDELT keyword search, syndication-collapsed        (claim_news)
  intl        a fixed panel of leading foreign dailies via GDELT domainis:
  academic    arXiv abstracts, date-ranged                       (arxiv)
  reference   Wikipedia article milestones per language          (wikipedia)
  forum       Hacker News stories                                (hn)

A channel that fails or returns nothing is recorded as such in the manifest --
"searched X, found nothing" and "could not search X" are different facts, and
the coverage discount downstream treats them differently.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

from .http import Fetcher
from .ingest import arxiv, gdelt, hn, wikipedia
from .ingest.article import domain_of, fetch_article
from .ingest.claim_news import ingest_for_claim, news_queries
from .ingest.dedup import cluster_documents
from .models import Source

RECIPE_VERSION = "shell-v1"

INTL_PANEL = [
    "lemonde.fr", "faz.net", "spiegel.de", "corriere.it", "repubblica.it",
    "elpais.com", "elobservador.com.uy", "asahi.com", "japantimes.co.jp",
    "theguardian.com", "scmp.com",
]

TERM_SCHEMA = {
    "type": "object",
    "properties": {
        "news_terms": {"type": "array", "items": {"type": "string"}},
        "academic_terms": {"type": "array", "items": {"type": "string"}},
        "wikipedia_articles": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["news_terms", "academic_terms", "wikipedia_articles"],
    "additionalProperties": False,
}

TERM_SYSTEM = """\
You plan searches for a benchmark that measures whether a text's claims were new
and whether they spread. Given the text's claims, produce:
- news_terms: 2-4 short queries a news desk would use (proper nouns, no jargon)
- academic_terms: 1-3 queries for an academic preprint index, in that field's
  own vocabulary (empty list if no claim has an academic literature)
- wikipedia_articles: 1-3 encyclopedia article TITLES whose existence or
  creation would mark the subject entering common knowledge (canonical names,
  e.g. "DeepSeek", not descriptions)"""


def plan_terms(llm, claims) -> dict:
    text = "\n".join(f"- [{c.kind.value}] {c.text}" for c in claims)
    out = llm.json(TERM_SYSTEM, f"CLAIMS:\n{text}", TERM_SCHEMA,
                   max_tokens=1500, grounded=False)
    return out or {"news_terms": [], "academic_terms": [], "wikipedia_articles": []}


def build_shell(store, llm, unit_key: str, doc, claims, windows_days=(30, 180),
                prior_days: int = 365, progress=None) -> dict:
    """Assemble the shell for one unit and return its manifest."""
    say = progress or (lambda m: None)
    t0 = time.time()
    manifest = {
        "recipe": RECIPE_VERSION, "unit": unit_key, "text_url": doc.url,
        "published": doc.published_at.date().isoformat(),
        "assembled_at": datetime.now().isoformat(timespec="seconds"),
        "channels": {},
    }
    terms = plan_terms(llm, claims)
    manifest["terms"] = terms
    start = doc.published_at - timedelta(days=prior_days)
    end = doc.published_at + timedelta(days=max(windows_days))

    with Fetcher(min_interval=0.35) as f:
        # -- news (per claim, existing machinery: dedup + prior year included)
        ch = {"status": "ok", "per_claim": []}
        for c in claims:
            try:
                r = ingest_for_claim(store, c, doc.published_at,
                                     windows_days=windows_days, llm=llm,
                                     max_per_window=25, progress=say)
                ch["per_claim"].append({k: r.get(k) for k in
                                        ("queries", "found", "syndicated_collapsed", "added")})
            except Exception as e:
                ch["per_claim"].append({"error": type(e).__name__})
        manifest["channels"]["news"] = ch

        # -- international dailies panel (domainis:)
        ch = {"status": "ok", "panel": INTL_PANEL, "hits": {}}
        seen_urls: set[str] = set()
        docs_intl = []
        try:
            from urllib.parse import quote
            for term in (terms.get("news_terms") or [])[:2]:
                for dom in INTL_PANEL:
                    url = (f"{gdelt.API}?query={quote(f'domainis:{dom} ' + term)}"
                           f"&mode=artlist&maxrecords=15"
                           f"&startdatetime={start.strftime('%Y%m%d%H%M%S')}"
                           f"&enddatetime={end.strftime('%Y%m%d%H%M%S')}&format=json")
                    arts = (f.get_json(url) or {}).get("articles") or []
                    ch["hits"][dom] = ch["hits"].get(dom, 0) + len(arts)
                    for a in arts:
                        u = a.get("url")
                        if u and u not in seen_urls:
                            seen_urls.add(u)
                            docs_intl.append(a)
            added = 0
            for a in docs_intl[:120]:
                dom = domain_of(a["url"])
                src = Source(id=Source.make_id("news", dom), kind="news",
                             name=dom, domain=dom)
                store.upsert_source(src)
                try:
                    d = fetch_article(f, a["url"], src,
                                      fallback_date=gdelt.parse_seendate(a))
                except Exception:
                    d = None
                if d and store.upsert_document(d):
                    added += 1
            ch["fetched"] = len(docs_intl)
            ch["added"] = added
        except Exception as e:
            ch["status"] = f"failed: {type(e).__name__}"
        manifest["channels"]["intl"] = ch
        say(f"    intl panel: {sum(ch.get('hits', {}).values())} listed, "
            f"{ch.get('added', 0)} added")

        # -- academic
        ch = {"status": "ok", "queries": [], "added": 0}
        for q in (terms.get("academic_terms") or [])[:3]:
            try:
                r = arxiv.ingest(store, f, q, max_results=40,
                                 start_date=start - timedelta(days=730), end_date=end)
                ch["queries"].append({"q": q, **{k: r[k] for k in ("found", "added")}})
                ch["added"] += r["added"]
            except Exception as e:
                ch["queries"].append({"q": q, "error": type(e).__name__})
        if not terms.get("academic_terms"):
            ch["status"] = "skipped: no academic footprint planned"
        manifest["channels"]["academic"] = ch
        say(f"    academic: {ch.get('added', 0)} abstracts added")

        # -- reference layer (Wikipedia)
        ch = {"status": "ok", "articles": []}
        for title in (terms.get("wikipedia_articles") or [])[:3]:
            try:
                r = wikipedia.ingest_milestones(store, f, title)
                ch["articles"].append(r)
            except Exception as e:
                ch["articles"].append({"title": title, "error": type(e).__name__})
        if ch["articles"] and all(not a.get("milestones") for a in ch["articles"]):
            ch["status"] = "unavailable (blocked or no articles found)"
        manifest["channels"]["reference"] = ch

        # -- forum (HN)
        ch = {"status": "ok", "added": 0}
        try:
            for term in (terms.get("news_terms") or [])[:2]:
                r = hn.ingest_text_posts(store, term, start, end,
                                         max_items=100, min_points=10, fetcher=f)
                ch["added"] += r.get("added", 0)
        except Exception as e:
            ch["status"] = f"failed: {type(e).__name__}"
        manifest["channels"]["forum"] = ch

    manifest["elapsed_s"] = round(time.time() - t0, 1)
    out = Path(f"data/manifests/{unit_key}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, default=str))
    return manifest
