"""Substack ingestion.

Substack exposes two public JSON endpoints that together give a complete, dated
archive of a publication:

  /api/v1/archive?sort=new&offset=N&limit=L   -> post metadata, paginated
  /api/v1/posts/<slug>                        -> the post, including body_html

The archive listing *has* a `body_html` key but it is always empty. Reading it
there yields a corpus of correctly-dated, entirely textless documents, which
scores as perfectly unoriginal and perfectly uninfluential. The body must come
from the per-post endpoint.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Iterator

from ..http import Fetcher
from ..models import DateConfidence, Document, Source
from ..textutil import extract_links, html_to_text

ARCHIVE_PAGE = 50  # the archive endpoint caps a page at 50 regardless of `limit`


def normalize_publication(pub: str) -> str:
    """Accept 'astralcodexten', a bare domain, or any post URL; return the origin."""
    pub = pub.strip().rstrip("/")
    if not pub.startswith("http"):
        pub = f"https://{pub}" if "." in pub else f"https://{pub}.substack.com"
    m = re.match(r"(https?://[^/]+)", pub)
    return m.group(1) if m else pub


def origin_candidates(pub: str) -> list[str]:
    """Origins to try, in order.

    Custom-domain publications serve the JSON API from exactly one hostname:
    `www.example.com` 200s while the apex `example.com` returns a bare 404 for
    the same path. Guessing wrong looks identical to an empty publication, so
    candidates are probed rather than assumed.
    """
    base = normalize_publication(pub)
    scheme, host = base.split("://", 1)
    cands = [base]
    if host.startswith("www."):
        cands.append(f"{scheme}://{host[4:]}")
    else:
        cands.append(f"{scheme}://www.{host}")
    if not host.endswith(".substack.com"):
        stem = host.removeprefix("www.").split(".")[0]
        cands.append(f"https://{stem}.substack.com")
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_origin(fetcher: Fetcher, pub: str, attempts: int = 2) -> str | None:
    """Return the origin whose archive API actually answers, or None.

    A failed probe is ambiguous: it means either "this is not a Substack" or
    "we are being rate-limited right now". Those must not be conflated -- doing
    so drops real publications from the corpus while reporting success. So the
    whole candidate set is retried after a pause before giving up.
    """
    for attempt in range(attempts):
        for cand in origin_candidates(pub):
            probe = fetcher.get_json(
                f"{cand}/api/v1/archive?sort=new&search=&offset=0&limit=1",
                force=(attempt > 0),
            )
            if isinstance(probe, list):
                return cand
        if attempt + 1 < attempts:
            time.sleep(5.0 * (attempt + 1))
    return None


def list_posts(fetcher: Fetcher, pub_url: str, max_posts: int = 2000) -> list[dict]:
    """Walk the archive backwards in time and return post metadata.

    Substack returns *short pages at arbitrary offsets*: `offset=0&limit=50` may
    hand back 23 items while `offset=50&limit=50` returns a full 50. So a short
    page must not be read as the end of the archive -- doing that silently
    truncates a publication to its most recent weeks and reports success. The
    offset therefore always advances by the requested page size, and only a page
    with *zero* items ends the walk. `limit` above 50 errors, so 50 it is.
    """
    out: list[dict] = []
    seen: set[int] = set()
    offset = 0
    empty_streak = 0
    while len(out) < max_posts:
        url = f"{pub_url}/api/v1/archive?sort=new&search=&offset={offset}&limit={ARCHIVE_PAGE}"
        # The past is append-only, so pages are cacheable; offset 0 is re-fetched
        # every run so newly published posts are picked up.
        batch = fetcher.get_json(url, force=(offset == 0))
        if batch is None:  # transient failure; one page gap should not end the walk
            empty_streak += 1
            if empty_streak >= 2:
                break
            offset += ARCHIVE_PAGE
            continue
        if not batch:
            break  # a genuinely empty page is the end of the archive
        empty_streak = 0
        for post in batch:
            pid = post.get("id")
            if pid not in seen:
                seen.add(pid)
                out.append(post)
        offset += ARCHIVE_PAGE
    return out[:max_posts]


def fetch_post_body(fetcher: Fetcher, pub_url: str, slug: str) -> dict | None:
    return fetcher.get_json(f"{pub_url}/api/v1/posts/{slug}")


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def ingest(
    store,
    publication: str,
    max_posts: int = 2000,
    fetcher: Fetcher | None = None,
    include_paywalled: bool = True,
    progress=None,
) -> dict:
    """Ingest a Substack publication into the pinned corpus."""
    own = fetcher is None
    fetcher = fetcher or Fetcher()
    try:
        pub_url = resolve_origin(fetcher, publication)
        if pub_url is None:
            return {"publication": publication, "listed": 0, "added": 0, "skipped": 0,
                    "error": f"no Substack archive API at any of "
                             f"{origin_candidates(publication)}"}
        posts = list_posts(fetcher, pub_url, max_posts=max_posts)
        if not posts:
            return {"publication": pub_url, "listed": 0, "added": 0, "skipped": 0,
                    "error": "no posts listed (private, renamed, or not a Substack)"}

        name = (posts[0].get("publishedBylines") or [{}])[0].get("name") or pub_url
        domain = pub_url.split("//", 1)[-1]
        src = Source(id=Source.make_id("substack", domain), kind="substack",
                     name=name, domain=domain,
                     metadata={"publication_url": pub_url})
        store.upsert_source(src)

        added = skipped = truncated = 0
        for i, meta in enumerate(posts):
            slug, canonical = meta.get("slug"), meta.get("canonical_url")
            if not slug or not canonical:
                skipped += 1
                continue
            audience = meta.get("audience", "everyone")
            paywalled = audience != "everyone"
            if paywalled and not include_paywalled:
                skipped += 1
                continue

            full = fetch_post_body(fetcher, pub_url, slug)
            body_html = (full or {}).get("body_html") or ""
            text = html_to_text(body_html)
            if len(text) < 200:
                # Paywalled posts return only a teaser. Admitting them would let a
                # truncated stub stand in for an article in prior-art search and
                # manufacture false novelty for whoever wrote the full version.
                truncated += 1
                skipped += 1
                continue

            doc = Document(
                id=Document.make_id(canonical),
                source_id=src.id,
                url=canonical,
                title=meta.get("title") or "",
                published_at=_parse_date(meta["post_date"]),
                text=text,
                date_confidence=DateConfidence.EXACT,
                date_provenance="substack:post_date",
                paywalled=paywalled,
                metadata={
                    "subtitle": meta.get("subtitle") or "",
                    "description": meta.get("description") or "",
                    "audience": audience,
                    "comment_count": meta.get("comment_count"),
                    "reaction_count": meta.get("reaction_count"),
                    "substack_id": meta.get("id"),
                    "outlinks": extract_links(body_html)[:200],
                },
            )
            if store.upsert_document(doc):
                added += 1
            else:
                skipped += 1
            if progress and (i + 1) % 25 == 0:
                progress(i + 1, len(posts), added)

        return {"publication": pub_url, "source_id": src.id, "source_name": name,
                "listed": len(posts), "added": added, "skipped": skipped,
                "truncated_or_paywalled": truncated}
    finally:
        if own:
            fetcher.close()
