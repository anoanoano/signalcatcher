"""Polite, cached HTTP.

Corpus builds fetch tens of thousands of URLs and will be interrupted. Every
response is cached on disk keyed by URL, so a re-run costs nothing and the
pinned corpus is genuinely reproducible rather than dependent on a remote host
still serving the same bytes months later.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .paths import cache_dir as _default_cache_dir

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# The cache is pure speed: every response in it has already been parsed into the
# corpus, so evicting an entry costs a re-fetch and nothing else. Left uncapped
# it will happily consume several gigabytes over a large build, which is not a
# trade anyone agreed to -- so it is bounded by default.
DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024  # 512 MB


class Fetcher:
    # Many nominally distinct hosts sit behind one provider (every Substack
    # custom domain, for instance). A purely per-host limit therefore lets N
    # workers hammer one backend N times harder, which is what got this build
    # rate-limited. This shared limiter bounds the whole process as well.
    _global_lock = threading.Lock()
    _global_last = 0.0
    _global_min_interval = 0.08

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        min_interval: float = 0.34,
        timeout: float = 45.0,
        use_cache: bool = True,
        max_cache_bytes: int | None = DEFAULT_MAX_CACHE_BYTES,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = max_cache_bytes
        self._writes_since_sweep = 0
        self.min_interval = min_interval  # per-host rate limit
        self.use_cache = use_cache
        self._last: dict[str, float] = {}
        self.client = httpx.Client(
            headers={"User-Agent": UA, "Accept": "*/*"},
            timeout=timeout,
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.txt"

    def _throttle(self, url: str) -> None:
        host = httpx.URL(url).host or ""
        last = self._last.get(host, 0.0)
        wait = self.min_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()
        with Fetcher._global_lock:
            gap = Fetcher._global_min_interval - (time.monotonic() - Fetcher._global_last)
            if gap > 0:
                time.sleep(gap)
            Fetcher._global_last = time.monotonic()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        reraise=True,
    )
    def _get(self, url: str) -> str:
        self._throttle(url)
        r = self.client.get(url)
        # 404/410 are settled answers; don't burn retries on them. 403 is NOT
        # settled -- edge bot-protection returns it under load, and treating it
        # as final made concurrent corpus builds silently report real
        # publications as nonexistent. Let it back off and retry.
        if r.status_code in (404, 410):
            raise PermanentError(f"{r.status_code} {url}")
        r.raise_for_status()
        return r.text

    def get(self, url: str, *, force: bool = False) -> str | None:
        p = self._cache_path(url)
        if self.use_cache and not force and p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
        try:
            body = self._get(url)
        except PermanentError:
            return None
        except Exception:
            return None
        if self.use_cache:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8", errors="replace")
            self._maybe_sweep()
        return body

    def _maybe_sweep(self) -> None:
        """Enforce the cache ceiling, amortised.

        Stat-ing the whole tree on every write would dominate the cost of a
        build, so the check runs periodically and then evicts oldest-first down
        to 80% of the cap -- leaving headroom so the next few hundred writes do
        not immediately trigger another sweep.
        """
        if self.max_cache_bytes is None:
            return
        self._writes_since_sweep += 1
        if self._writes_since_sweep < 500:
            return
        self._writes_since_sweep = 0
        self.sweep_cache()

    def sweep_cache(self, target_ratio: float = 0.8) -> int:
        """Evict oldest cache entries until under the cap. Returns bytes freed."""
        if self.max_cache_bytes is None:
            return 0
        files = []
        total = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                try:
                    st = f.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, f))
                total += st.st_size
        if total <= self.max_cache_bytes:
            return 0
        target = int(self.max_cache_bytes * target_ratio)
        files.sort()  # oldest first
        freed = 0
        for _mtime, size, f in files:
            if total - freed <= target:
                break
            try:
                f.unlink()
                freed += size
            except OSError:
                continue
        return freed

    def cache_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())

    def get_json(self, url: str, *, force: bool = False) -> Any | None:
        body = self.get(url, force=force)
        if body is None:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None


class PermanentError(Exception):
    """A status that will not improve on retry."""
