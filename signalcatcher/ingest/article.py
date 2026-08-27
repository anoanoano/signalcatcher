"""Turn an arbitrary web page into a dated Document.

Publication date is the single most load-bearing field in this benchmark: the
whole construct is a claim about who said something *first*. A page that reports
today's date because we parsed a "last updated" stamp will be scored as prior
art for things it actually copied. So dates are extracted from an explicit
preference order and every document records which rule produced its date, with a
confidence the scorer can act on.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from dateutil import parser as dateparser
from selectolax.parser import HTMLParser

from ..http import Fetcher
from ..models import DateConfidence, Document, Source
from ..textutil import extract_links, html_to_text, normalize_ws

# Ordered strongest-first. Structured publisher metadata beats visible text,
# and "published" always beats "modified".
_META_KEYS = [
    ("article:published_time", DateConfidence.EXACT),
    ("og:article:published_time", DateConfidence.EXACT),
    ("datePublished", DateConfidence.EXACT),
    ("date", DateConfidence.DAY),
    ("dc.date.issued", DateConfidence.DAY),
    ("parsely-pub-date", DateConfidence.EXACT),
    ("sailthru.date", DateConfidence.EXACT),
    ("article:modified_time", DateConfidence.INFERRED),
]

_URL_DATE = re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:/|-|$)")


def _try_parse(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw or len(raw) < 4:
        return None
    try:
        dt = dateparser.parse(raw)
    except (ValueError, OverflowError, TypeError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # A date in the future or before the web means we parsed something else.
    now = datetime.now(timezone.utc)
    if dt.year < 1990 or dt > now.replace(year=now.year + 1):
        return None
    return dt.astimezone(timezone.utc)


def _from_jsonld(tree: HTMLParser) -> datetime | None:
    for node in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.text())
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for key in ("datePublished", "dateCreated", "uploadDate"):
                    if key in cur:
                        dt = _try_parse(str(cur[key]))
                        if dt:
                            return dt
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    return None


def extract_date(html: str, url: str) -> tuple[datetime | None, DateConfidence, str]:
    """Return (date, confidence, provenance-rule-that-fired)."""
    tree = HTMLParser(html)

    dt = _from_jsonld(tree)
    if dt:
        return dt, DateConfidence.EXACT, "jsonld:datePublished"

    metas: dict[str, str] = {}
    for node in tree.css("meta"):
        attrs = node.attributes
        key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
        content = attrs.get("content") or ""
        if key and content and key not in metas:
            metas[key] = content
    for key, conf in _META_KEYS:
        if key.lower() in metas:
            dt = _try_parse(metas[key.lower()])
            if dt:
                return dt, conf, f"meta:{key}"

    for node in tree.css("time[datetime]"):
        dt = _try_parse(node.attributes.get("datetime", ""))
        if dt:
            return dt, DateConfidence.DAY, "time[datetime]"

    # URL path dates are usually the publication slug and are hard to fake, but
    # give only day resolution.
    m = _URL_DATE.search(url)
    if m:
        dt = _try_parse(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        if dt:
            return dt, DateConfidence.INFERRED, "url:path-date"

    return None, DateConfidence.UNKNOWN, "none"


_TITLE_SEL = ["meta[property='og:title']", "meta[name='twitter:title']"]


def extract_title(tree: HTMLParser) -> str:
    for sel in _TITLE_SEL:
        node = tree.css_first(sel)
        if node and node.attributes.get("content"):
            return normalize_ws(node.attributes["content"])
    node = tree.css_first("h1") or tree.css_first("title")
    return normalize_ws(node.text()) if node else ""


# Containers that are never the article. Comment threads are the dangerous one:
# a popular post can carry hundreds of them, and left in place they outweigh the
# piece itself -- so the corpus would credit the author with their readers'
# words, and n-gram fingerprints would match on commenter phrasing.
_STRIP_SEL = [
    "#comments", ".comments", ".comment", ".commentlist", ".comment-list",
    ".commentholder", "#respond", "#disqus_thread", ".related", ".related-posts",
    "nav", "header", "footer", "aside", ".sidebar", "#sidebar", ".nav",
    ".share", ".social", ".newsletter-signup", ".subscribe-widget",
    ".advertisement", ".ad", "[aria-hidden=true]",
]

# Prefer the semantic article container; falling back to <body> drags in nav,
# related-links rails and comment threads. Ordered most- to least-specific.
_MAIN_SEL = ["article", "main", "[role=main]", ".post-content", ".entry-content",
             ".pjgm-postcontent", ".article-body", "#article-body", ".story-body",
             ".postbody", ".content-body", ".article__body"]


def _strip_chrome(tree: HTMLParser) -> HTMLParser:
    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()
    for sel in _STRIP_SEL:
        try:
            for node in tree.css(sel):
                node.decompose()
        except Exception:
            continue  # selectolax rejects a few exotic selectors; skip them
    return tree


_DROP_TAGS = ("script", "style", "noscript", "svg", "form", "iframe")


def extract_main_text(html: str) -> str:
    """Best-effort extraction of just the article body."""
    if not html:
        return ""
    tree = _strip_chrome(HTMLParser(html))
    best = ""
    for sel in _MAIN_SEL:
        node = tree.css_first(sel)
        if node:
            text = html_to_text(node.html or "")
            if len(text) > 400:
                return text
            best = max(best, text, key=len)
    # No recognised container. Fall back to the densest text-bearing <div>,
    # which beats taking the whole body on hand-rolled or legacy templates.
    densest = ""
    for node in tree.css("div"):
        text = html_to_text(node.html or "")
        if len(text) > len(densest):
            densest = text
    body = html_to_text((tree.body or tree).html or "")
    # The densest div is preferred only when it is a clear majority of the page,
    # which signals a real content wrapper rather than an arbitrary layout box.
    if densest and len(densest) > 0.5 * len(body):
        return densest
    return max(best, body, key=len)


def fetch_article(
    fetcher: Fetcher, url: str, source: Source, fallback_date: datetime | None = None,
    min_chars: int = 400,
) -> Document | None:
    """Fetch a URL and build a Document, or None if it is unusable."""
    html = fetcher.get(url)
    if not html:
        return None
    text = extract_main_text(html)
    if len(text) < min_chars:
        return None  # paywall teaser, redirect stub, or JS-only page
    dt, conf, prov = extract_date(html, url)
    if dt is None:
        if fallback_date is None:
            return None  # an undated document cannot participate in a priority claim
        dt, conf, prov = fallback_date, DateConfidence.INFERRED, "discovery:first-seen"
    tree = HTMLParser(html)
    return Document(
        id=Document.make_id(url), source_id=source.id, url=url,
        title=extract_title(tree), published_at=dt, text=text,
        date_confidence=conf, date_provenance=prov,
        metadata={"outlinks": extract_links(html)[:200]},
    )


def domain_of(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1).removeprefix("www.").lower() if m else ""
