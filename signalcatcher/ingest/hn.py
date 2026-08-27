"""Hacker News via the Algolia API.

HN earns its place twice over. It is a dated *discovery* index -- it will tell
you which URLs across the whole web were being discussed in any given week going
back to 2007, which is how the corpus finds documents it would not otherwise
know to look for. And the submission itself is hard evidence of transmission
with an exact timestamp: someone read a thing and carried it somewhere else.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..http import Fetcher
from ..models import DateConfidence, Document, Source
from ..textutil import html_to_text

API = "https://hn.algolia.com/api/v1/search_by_date"
PAGE = 100  # Algolia's hard maximum per request


def search_stories(
    fetcher: Fetcher, query: str = "", start: datetime | None = None,
    end: datetime | None = None, max_items: int = 1000, min_points: int = 0,
) -> list[dict]:
    """Date-ranged story search.

    Algolia refuses to page beyond 1000 results for one query, so long ranges
    are walked by *ratcheting the end of the window down* to the oldest hit seen
    so far, rather than by incrementing a page number.
    """
    out: list[dict] = []
    seen: set[str] = set()
    cursor_end = int(end.timestamp()) if end else None
    start_ts = int(start.timestamp()) if start else None

    while len(out) < max_items:
        filters = []
        if start_ts is not None:
            filters.append(f"created_at_i>{start_ts}")
        if cursor_end is not None:
            filters.append(f"created_at_i<{cursor_end}")
        if min_points:
            filters.append(f"points>={min_points}")
        url = (f"{API}?tags=story&hitsPerPage={PAGE}"
               f"&query={_q(query)}&numericFilters={','.join(filters)}")
        data = fetcher.get_json(url)
        hits = (data or {}).get("hits") or []
        if not hits:
            break
        fresh = [h for h in hits if h.get("objectID") not in seen]
        for h in fresh:
            seen.add(h["objectID"])
        out.extend(fresh)
        oldest = min(h["created_at_i"] for h in hits)
        if cursor_end is not None and oldest >= cursor_end:
            break  # window stopped shrinking; no further progress possible
        cursor_end = oldest
        if start_ts is not None and oldest <= start_ts:
            break
    return out[:max_items]


def _q(query: str) -> str:
    from urllib.parse import quote
    return quote(query)


def discover_urls(
    fetcher: Fetcher, query: str = "", start: datetime | None = None,
    end: datetime | None = None, max_items: int = 1000, min_points: int = 10,
) -> list[tuple[str, datetime, dict]]:
    """External URLs discussed on HN in a window, with the discussion timestamp.

    The HN timestamp is a *first-seen* upper bound on the article's publication
    date, not the date itself -- useful as a fallback, never as a substitute.
    """
    stories = search_stories(fetcher, query, start, end, max_items, min_points)
    out = []
    for s in stories:
        url = s.get("url")
        if not url or not url.startswith("http"):
            continue
        seen_at = datetime.fromtimestamp(s["created_at_i"], timezone.utc)
        out.append((url, seen_at, {
            "hn_points": s.get("points"), "hn_comments": s.get("num_comments"),
            "hn_id": s.get("objectID"), "hn_title": s.get("title"),
        }))
    return out


def ingest_text_posts(
    store, query: str = "", start: datetime | None = None, end: datetime | None = None,
    max_items: int = 500, min_points: int = 20, fetcher: Fetcher | None = None,
) -> dict:
    """Ingest Ask HN / Show HN style self-posts, which carry their own text."""
    own = fetcher is None
    fetcher = fetcher or Fetcher()
    try:
        src = Source(id=Source.make_id("forum", "Hacker News"), kind="forum",
                     name="Hacker News", domain="news.ycombinator.com")
        store.upsert_source(src)
        stories = search_stories(fetcher, query, start, end, max_items, min_points)
        added = 0
        for s in stories:
            text = html_to_text(s.get("story_text") or "")
            if len(text) < 300:
                continue
            url = f"https://news.ycombinator.com/item?id={s['objectID']}"
            doc = Document(
                id=Document.make_id(url), source_id=src.id, url=url,
                title=s.get("title") or "", text=text,
                published_at=datetime.fromtimestamp(s["created_at_i"], timezone.utc),
                date_confidence=DateConfidence.EXACT, date_provenance="hn:created_at_i",
                metadata={"hn_points": s.get("points"), "hn_author": s.get("author")},
            )
            if store.upsert_document(doc):
                added += 1
        return {"source_id": src.id, "listed": len(stories), "added": added}
    finally:
        if own:
            fetcher.close()
