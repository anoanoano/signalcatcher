"""arXiv: academic prior art, free Atom API, no key.

Abstracts only -- an abstract states a paper's claims densely enough for
prior-art adjudication, and full PDFs would add megabytes per paper for little
recall. The `published` timestamp is v1 submission time: exactly the priority
date the benchmark needs, and one of the few unambiguous ones on the internet.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote

from ..http import Fetcher
from ..models import DateConfidence, Document, Source

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def search(fetcher: Fetcher, query: str, max_results: int = 40,
           start_date: datetime | None = None, end_date: datetime | None = None) -> list[dict]:
    """`query` is split into quoted all: terms ANDed together; the date range
    goes INTO the query, because sorting newest-first and filtering client-side
    silently misses older papers once the topic becomes popular -- the exact
    papers a prior-art search exists to find."""
    terms = [t for t in query.replace('"', " ").split() if len(t) > 2]
    q = " AND ".join(f'all:"{t}"' for t in terms[:6]) or f'all:"{query}"'
    if start_date or end_date:
        a = (start_date or datetime(1991, 1, 1, tzinfo=timezone.utc)).strftime("%Y%m%d%H%M")
        b = (end_date or datetime.now(timezone.utc)).strftime("%Y%m%d%H%M")
        q += f" AND submittedDate:[{a} TO {b}]"
    url = (f"{API}?search_query={quote(q)}&start=0"
           f"&max_results={min(max_results, 100)}&sortBy=submittedDate&sortOrder=descending")
    body = fetcher.get(url)
    if not body:
        return []
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return []
    out = []
    for e in root.findall("a:entry", NS):
        def t(tag):
            n = e.find(f"a:{tag}", NS)
            return (n.text or "").strip() if n is not None else ""
        try:
            pub = datetime.fromisoformat(t("published").replace("Z", "+00:00"))
        except ValueError:
            continue
        if start_date and pub < start_date:
            continue
        if end_date and pub > end_date:
            continue
        authors = [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)]
        out.append({"id": t("id"), "title": " ".join(t("title").split()),
                    "abstract": " ".join(t("summary").split()),
                    "published": pub, "authors": authors[:12]})
    return out


def ingest(store, fetcher: Fetcher, query: str, max_results: int = 40,
           start_date: datetime | None = None, end_date: datetime | None = None) -> dict:
    src = Source(id=Source.make_id("academic", "arXiv"), kind="academic",
                 name="arXiv", domain="arxiv.org")
    store.upsert_source(src)
    papers = search(fetcher, query, max_results, start_date, end_date)
    added = 0
    for p in papers:
        text = f"{p['title']}\n\nAuthors: {', '.join(p['authors'])}\n\n{p['abstract']}"
        d = Document(id=Document.make_id(p["id"]), source_id=src.id, url=p["id"],
                     title=p["title"][:200], published_at=p["published"], text=text,
                     date_confidence=DateConfidence.EXACT,
                     date_provenance="arxiv:published",
                     metadata={"channel": "arxiv"})
        if store.upsert_document(d):
            added += 1
    return {"query": query, "found": len(papers), "added": added}
