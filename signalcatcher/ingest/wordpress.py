"""WordPress REST ingestion, for the large blog archives that predate Substack.

Much of the discourse this benchmark wants to measure happened on WordPress
between roughly 2005 and 2020, and a lot of it is still served by an open
`/wp-json/wp/v2/posts` endpoint with exact `date_gmt` timestamps. Those years
matter disproportionately: they are the *prior art window* for anything written
since, and a corpus that starts in 2021 will score a decade of recycled ideas as
brand new.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..http import Fetcher
from ..models import DateConfidence, Document, Source
from ..textutil import extract_links, html_to_text, normalize_ws

PER_PAGE = 100


def api_root(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith("http"):
        host = f"https://{host}"
    return f"{host}/wp-json/wp/v2"


def available(fetcher: Fetcher, host: str) -> bool:
    probe = fetcher.get_json(f"{api_root(host)}/posts?per_page=1")
    return isinstance(probe, list) and bool(probe)


def list_posts(fetcher: Fetcher, host: str, max_posts: int = 5000) -> list[dict]:
    """Page backwards using a `before` date cursor.

    WordPress refuses deep `page=` offsets on large archives (it 400s past a few
    thousand posts), so paging by date is the only way to reach the early years
    -- which are the years worth having.
    """
    root = api_root(host)
    out: list[dict] = []
    seen: set[int] = set()
    cursor: str | None = None
    while len(out) < max_posts:
        url = f"{root}/posts?per_page={PER_PAGE}&orderby=date&order=desc&_fields=id,date_gmt,link,title,content,excerpt,categories"
        if cursor:
            url += f"&before={cursor}"
        batch = fetcher.get_json(url)
        if not isinstance(batch, list) or not batch:
            break
        fresh = [p for p in batch if p.get("id") not in seen]
        if not fresh:
            break
        for p in fresh:
            seen.add(p["id"])
        out.extend(fresh)
        oldest = min((p.get("date_gmt") or "") for p in batch if p.get("date_gmt"))
        if not oldest or oldest == cursor:
            break  # cursor stopped moving; avoid an infinite loop
        cursor = oldest
    return out[:max_posts]


def ingest(
    store, host: str, name: str | None = None, max_posts: int = 5000,
    fetcher: Fetcher | None = None, min_chars: int = 400, progress=None,
) -> dict:
    own = fetcher is None
    fetcher = fetcher or Fetcher()
    try:
        if not available(fetcher, host):
            return {"host": host, "listed": 0, "added": 0,
                    "error": "no open /wp-json/wp/v2/posts endpoint"}
        posts = list_posts(fetcher, host, max_posts=max_posts)
        domain = api_root(host).split("//", 1)[1].split("/")[0]
        src = Source(id=Source.make_id("blog", domain), kind="blog",
                     name=name or domain, domain=domain)
        store.upsert_source(src)

        added = skipped = 0
        for i, p in enumerate(posts):
            link = p.get("link")
            raw = (p.get("content") or {}).get("rendered") or ""
            text = html_to_text(raw)
            if not link or len(text) < min_chars:
                skipped += 1
                continue
            try:
                dt = datetime.fromisoformat(p["date_gmt"]).replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                skipped += 1
                continue
            doc = Document(
                id=Document.make_id(link), source_id=src.id, url=link,
                title=normalize_ws(html_to_text((p.get("title") or {}).get("rendered") or "")),
                published_at=dt, text=text,
                date_confidence=DateConfidence.EXACT, date_provenance="wp:date_gmt",
                metadata={"wp_id": p.get("id"), "outlinks": extract_links(raw)[:200]},
            )
            if store.upsert_document(doc):
                added += 1
            else:
                skipped += 1
            if progress and (i + 1) % 100 == 0:
                progress(i + 1, len(posts), added)
        return {"host": host, "source_id": src.id, "source_name": src.name,
                "listed": len(posts), "added": added, "skipped": skipped}
    finally:
        if own:
            fetcher.close()
