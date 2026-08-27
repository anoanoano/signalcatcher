"""Build the pinned corpus from a seed list of sources.

Restartable and idempotent: documents dedupe on URL and every HTTP response is
disk-cached, so re-running fills only what is missing and costs nothing for what
is already there.

Publications are fetched concurrently because they live on different hosts --
the per-host rate limit is what politeness requires, and serialising across
unrelated hosts just wastes wall-clock. SQLite writes are funnelled through one
lock so the workers cannot trip over each other.
"""
from __future__ import annotations

import sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signalcatcher.db import Store
from signalcatcher.http import Fetcher
from signalcatcher.ingest import substack, wordpress

# A deliberately dense, single-conversation cluster: these writers read and argue
# with each other. Diffusion is only measurable inside a community that actually
# transmits, so a broad but disconnected sample would show nothing at all.
SUBSTACKS = [
    # AI / technology commentary
    "astralcodexten.com", "thezvi.substack.com", "www.oneusefulthing.org",
    "garymarcus.substack.com", "importai.substack.com", "www.understandingai.org",
    "www.hyperdimensional.co", "milesbrundage.substack.com", "www.interconnects.ai",
    "www.aisnakeoil.com", "lastweekin.ai", "newsletter.pragmaticengineer.com",
    "www.chinatalk.media", "www.exponentialview.co", "platformer.news",
    "www.transformernews.ai", "epochai.substack.com",
    # economics / policy / housing
    "www.noahpinion.blog", "www.slowboring.com", "www.construction-physics.com",
    "www.derekthompson.org", "www.economicforces.xyz", "briefingday.substack.com",
    "www.city-journal.org", "kevindrum.substack.com", "kyla.substack.com",
    "kareemcarr.substack.com", "kaushikcbasu.substack.com",
    # science / medicine / meta
    "erictopol.substack.com", "www.programmablemutter.com", "nadia.xyz",
    "www.natesilver.net", "www.slowboring.com", "statmodeling.substack.com",
    "experimentalhistory.substack.com", "www.overcomingbias.com",
    "cremieux.xyz", "www.richardhanania.com", "arnoldkling.substack.com",
    "marginalrevolution.substack.com", "goodreason.substack.com",
    "www.thediff.co", "quantian.substack.com", "www.samstack.io",
]
WORDPRESS = [("https://slatestarcodex.com", "Slate Star Codex")]


class LockedStore:
    """Serialise writes; SQLite allows one writer and the workers all write."""
    def __init__(self, store: Store):
        self._s, self._lock = store, threading.Lock()

    def __getattr__(self, name):
        attr = getattr(self._s, name)
        if not callable(attr):
            return attr
        def wrapped(*a, **kw):
            with self._lock:
                return attr(*a, **kw)
        return wrapped


def main(max_posts: int = 900, workers: int = 8) -> None:
    raw = Store("data/corpus.db")
    store = LockedStore(raw)
    t0 = time.time()

    def do_substack(pub):
        with Fetcher(min_interval=0.5) as f:
            return "substack", pub, substack.ingest(store, pub, max_posts=max_posts, fetcher=f)

    def do_wp(item):
        host, name = item
        with Fetcher(min_interval=0.5) as f:
            return "wp", host, wordpress.ingest(store, host, name=name,
                                                max_posts=max_posts, fetcher=f)

    jobs = [(do_substack, p) for p in SUBSTACKS] + [(do_wp, w) for w in WORDPRESS]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, arg): arg for fn, arg in jobs}
        for fut in as_completed(futs):
            try:
                kind, who, r = fut.result()
                print(f"[{time.time()-t0:6.0f}s] {kind:8} {str(who)[:40]:42} "
                      f"listed={r.get('listed',0):4} added={r.get('added',0):4} "
                      f"{r.get('error','')}", flush=True)
            except Exception:
                print(f"FAIL {futs[fut]}\n{traceback.format_exc()}", flush=True)

    lo, hi = raw.corpus_span()
    print(f"\nCORPUS: {raw.count_documents()} docs | {lo.date() if lo else '?'} .. "
          f"{hi.date() if hi else '?'} | {time.time()-t0:.0f}s", flush=True)
    for src, n in raw.list_sources():
        print(f"  {n:5d}  {src.name[:42]:44} {src.domain}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 900, workers=5)
