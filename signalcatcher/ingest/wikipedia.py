"""Wikipedia as an adoption milestone, per language.

The creation date of an article -- and of its counterparts in other languages --
is a hard, datable marker that an idea reached the reference layer of a
language's discourse. "The Italian Wikipedia article appeared on X" is exactly
the kind of receipt the benchmark's trails are made of, and the MediaWiki API
serves it keyless.

The article's current intro is ingested as a document dated at CREATION. That is
a deliberate approximation, flagged in provenance: today's text at the creation
date can overstate what was known then. It is admitted anyway because the
milestone itself (an article existed) is what the adoption measurement uses;
the text is context for the judge, and the confidence field carries the doubt.
"""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import quote

from ..http import Fetcher
from ..models import DateConfidence, Document, Source

LANGS = ("en", "de", "fr", "it", "es", "ja")

# Wikimedia's robot policy 403s browser-imitating user agents; it wants tools to
# identify themselves. This adapter therefore bypasses the shared Fetcher's UA
# with an honest one (cache and throttling still come from the Fetcher).
TOOL_UA = "SignalCatcherBenchmark/0.1 (research tool measuring idea diffusion; low volume)"


def _get_json(fetcher: Fetcher, url: str):
    import hashlib, json as _json
    p = fetcher._cache_path(url)
    if fetcher.use_cache and p.exists():
        try:
            return _json.loads(p.read_text())
        except _json.JSONDecodeError:
            pass
    fetcher._throttle(url)
    try:
        r = fetcher.client.get(url, headers={"User-Agent": TOOL_UA})
        r.raise_for_status()
        body = r.text
    except Exception:
        return None
    if fetcher.use_cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    try:
        return _json.loads(body)
    except _json.JSONDecodeError:
        return None


def _api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def _wayback_first_capture(fetcher: Fetcher, lang: str, title: str) -> datetime | None:
    """Earliest Internet Archive capture of the article URL.

    Fallback for when Wikimedia's API is unavailable (their robot policy 403s
    unpredictably by IP). A first capture is a hard LOWER BOUND on the article's
    existence -- it proves the article existed by that date, which is the
    direction the milestone is used in. Provenance records the difference.
    """
    url = f"{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
    body = fetcher.get(f"http://web.archive.org/cdx/search/cdx?url={quote(url)}"
                       f"&output=json&limit=1&fl=timestamp&filter=statuscode:200")
    if not body:
        return None
    try:
        import json as _json
        rows = _json.loads(body)
        ts = rows[1][0]
        return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except (ValueError, IndexError, KeyError):
        return None


def article_timeline(fetcher: Fetcher, title: str, langs=LANGS) -> list[dict]:
    """Creation date of `title`'s article in each language where it exists."""
    out = []
    for lang in langs:
        base = _api(lang)
        # Resolve the local title via langlinks from en, or search directly.
        t = title
        if lang != "en":
            d = _get_json(fetcher,
                f"{_api('en')}?action=query&prop=langlinks&titles={quote(title)}"
                f"&lllang={lang}&format=json&redirects=1") or {}
            pages = (d.get("query") or {}).get("pages") or {}
            links = next(iter(pages.values()), {}).get("langlinks") or []
            # If the API is blocked the lookup returns nothing; proper nouns
            # usually share a title across wikis, so try the same title rather
            # than dropping the language entirely.
            t = (links[0].get("*") if links else None) or title
        d = _get_json(fetcher, 
            f"{base}?action=query&prop=revisions&titles={quote(t)}&rvlimit=1"
            f"&rvdir=newer&rvprop=timestamp&format=json&redirects=1") or {}
        pages = (d.get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), {})
        revs = page.get("revisions") or []
        if revs:
            created = datetime.fromisoformat(revs[0]["timestamp"].replace("Z", "+00:00"))
            prov = "wikipedia:first-revision"
        else:
            created = _wayback_first_capture(fetcher, lang, t)
            prov = "wayback:first-capture (lower bound)"
            if created is None:
                continue
        out.append({"lang": lang, "title": t, "created": created, "provenance": prov,
                    "url": f"https://{lang}.wikipedia.org/wiki/{quote(t.replace(' ', '_'))}"})
    return out


def ingest_milestones(store, fetcher: Fetcher, title: str, langs=LANGS) -> dict:
    src = Source(id=Source.make_id("reference", "Wikipedia"), kind="reference",
                 name="Wikipedia", domain="wikipedia.org")
    store.upsert_source(src)
    timeline = article_timeline(fetcher, title, langs)
    added = 0
    for m in timeline:
        d = _get_json(fetcher, 
            f"{_api(m['lang'])}?action=query&prop=extracts&titles={quote(m['title'])}"
            f"&exintro=1&explaintext=1&format=json&redirects=1") or {}
        pages = (d.get("query") or {}).get("pages") or {}
        extract = next(iter(pages.values()), {}).get("extract") or ""
        if len(extract) < 200:
            continue
        doc = Document(
            id=Document.make_id(m["url"]), source_id=src.id, url=m["url"],
            title=f"Wikipedia ({m['lang']}): {m['title']}",
            published_at=m["created"], text=extract[:12000],
            date_confidence=DateConfidence.INFERRED,
            date_provenance="wikipedia:first-revision (current text, creation date)",
            lang=m["lang"], metadata={"channel": "wikipedia"})
        if store.upsert_document(doc):
            added += 1
    return {"title": title, "milestones": [
        {"lang": m["lang"], "created": m["created"].date().isoformat(), "url": m["url"]}
        for m in timeline], "added": added}
