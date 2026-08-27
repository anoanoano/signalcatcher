"""GDELT DOC 2.0: dated news discovery across ~65 languages, free and keyless.

GDELT returns article *metadata* only -- url, title, domain, and `seendate`.
That is enough on its own to measure the shape of pickup (how many distinct
outlets, how fast), which is the diffusion signal a newsroom cares about. Full
text, when a claim-level judgement needs it, comes from fetching the URLs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from ..http import Fetcher

API = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250      # per request ceiling
COVERAGE_START = datetime(2017, 1, 1, tzinfo=timezone.utc)  # usable full-text coverage


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def article_list(
    fetcher: Fetcher, query: str, start: datetime, end: datetime,
    max_records: int = MAX_RECORDS, country: str = "", language: str = "english",
) -> list[dict]:
    q = query
    if language:
        q += f" sourcelang:{language}"
    if country:
        q += f" sourcecountry:{country}"
    url = (f"{API}?query={quote(q)}&mode=artlist&maxrecords={min(max_records, MAX_RECORDS)}"
           f"&startdatetime={_fmt(start)}&enddatetime={_fmt(end)}&format=json&sort=datedesc")
    data = fetcher.get_json(url)
    return (data or {}).get("articles") or []


def walk(
    fetcher: Fetcher, query: str, start: datetime, end: datetime,
    step_days: int = 7, max_per_step: int = MAX_RECORDS, **kw,
) -> list[dict]:
    """Sweep a long range in fixed windows.

    A single request is capped at 250 articles, so asking for a year at once
    silently returns a biased sample of it. Stepping in weeks keeps the cap from
    truncating any one period, which matters because the diffusion score is
    counting articles *per window*.
    """
    out, seen = [], set()
    cur = start
    while cur < end:
        stop = min(cur + timedelta(days=step_days), end)
        for art in article_list(fetcher, query, cur, stop, max_per_step, **kw):
            u = art.get("url")
            if u and u not in seen:
                seen.add(u)
                out.append(art)
        cur = stop
    return out


def parse_seendate(art: dict) -> datetime | None:
    raw = art.get("seendate") or ""
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def outlet_counts(articles: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a in articles:
        d = (a.get("domain") or "").lower()
        if d:
            counts[d] = counts.get(d, 0) + 1
    return counts
