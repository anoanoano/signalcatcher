"""Where SignalCatcher keeps its data.

The corpus grows without bound as publications are added -- a full build reaches
several gigabytes -- so the location is configurable rather than wired to the
repo. Point it at an external volume and the repository stays a few megabytes of
code.

Resolution order (first wins):
  1. an explicit path passed by the caller
  2. the SIGNALCATCHER_DATA environment variable
  3. ./data, relative to the working directory
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "SIGNALCATCHER_DATA"
DEFAULT = "data"


def data_root(explicit: str | Path | None = None) -> Path:
    root = Path(explicit or os.environ.get(ENV_VAR) or DEFAULT).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path(explicit: str | Path | None = None, root: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return data_root(root) / "corpus.db"


def cache_dir(root: str | Path | None = None) -> Path:
    d = data_root(root) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def usage(root: str | Path | None = None) -> dict[str, int]:
    """Bytes on disk, broken out, so `corpus` can report it without guessing."""
    r = data_root(root)
    out: dict[str, int] = {}
    for name in ("cache",):
        d = r / name
        out[name] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
    db = 0
    for pat in ("corpus.db", "corpus.db-wal", "corpus.db-shm"):
        f = r / pat
        if f.exists():
            db += f.stat().st_size
    out["db"] = db
    out["total"] = sum(out.values())
    return out


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}T"
