"""Snapshot corpora to cold storage, and page them back in on demand.

A live corpus does not belong on a cloud drive: Drive for Desktop caches every
file locally before syncing, so an actively-written database costs the same
local disk *plus* re-uploading a changing binary on every write. Measured, not
assumed -- writing 200 MB to Drive consumed 200 MB locally.

What cloud storage IS good for is the archive. A corpus compresses to roughly
40% of its size and, once snapshotted, never changes. So the working pattern is
one focused corpus at a time on local disk, and every other corpus parked in the
cloud until it is needed. Local disk then holds one investigation, not all of
them.

`VACUUM INTO` is used rather than copying the file: it produces a defragmented,
internally consistent snapshot without having to stop writers or worry about
what is still sitting in the WAL.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import data_root, db_path, human

MANIFEST = "manifest.json"


def default_archive_dir() -> Path | None:
    """Google Drive, if this machine has it mounted."""
    base = Path.home() / "Library" / "CloudStorage"
    if not base.exists():
        return None
    for d in sorted(base.iterdir()):
        if d.name.startswith("GoogleDrive-"):
            target = d / "My Drive" / "signalcatcher-corpora"
            return target
    return None


def _have_zstd() -> bool:
    return shutil.which("zstd") is not None


def _compress(src: Path, dst_base: Path) -> Path:
    """Compress src, preferring zstd, falling back to gzip. Returns the path."""
    if _have_zstd():
        dst = dst_base.with_suffix(dst_base.suffix + ".zst")
        subprocess.run(["zstd", "-10", "-q", "-f", "-o", str(dst), str(src)], check=True)
        return dst
    import gzip
    dst = dst_base.with_suffix(dst_base.suffix + ".gz")
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, length=8 << 20)
    return dst


def _decompress(src: Path, dst: Path) -> None:
    if src.suffix == ".zst":
        if not _have_zstd():
            raise RuntimeError(
                f"{src.name} needs zstd to unpack; install it (brew install zstd)")
        subprocess.run(["zstd", "-d", "-q", "-f", "-o", str(dst), str(src)], check=True)
        return
    import gzip
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=8 << 20)


def _stats(db: Path) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        docs = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        span = c.execute("SELECT MIN(published_at), MAX(published_at) FROM documents").fetchone()
        srcs = c.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        names = [r[0] for r in c.execute(
            "SELECT s.name FROM sources s JOIN documents d ON d.source_id=s.id "
            "GROUP BY s.id ORDER BY COUNT(d.id) DESC LIMIT 25")]
        return {"documents": docs, "sources": srcs,
                "span": [span[0][:10] if span[0] else None,
                         span[1][:10] if span[1] else None],
                "top_sources": names}
    finally:
        c.close()


def snapshot(name: str, archive_dir: str | Path | None = None,
             db: str | Path | None = None, progress=None) -> dict:
    say = progress or (lambda m: None)
    src = db_path(db)
    if not src.exists():
        raise FileNotFoundError(f"no corpus at {src}")
    dest_dir = Path(archive_dir).expanduser() if archive_dir else default_archive_dir()
    if dest_dir is None:
        raise RuntimeError(
            "no archive directory: pass --to, or install/enable Google Drive")
    dest_dir.mkdir(parents=True, exist_ok=True)

    say(f"compacting {human(src.stat().st_size)} corpus ...")
    with tempfile.TemporaryDirectory() as td:
        clean = Path(td) / "corpus.db"
        c = sqlite3.connect(src)
        # VACUUM INTO gives a consistent, defragmented copy while writers run.
        c.execute("VACUUM INTO ?", (str(clean),))
        c.close()
        info = _stats(clean)
        say(f"compressing ({'zstd' if _have_zstd() else 'gzip'}) ...")
        packed = _compress(clean, Path(td) / f"{name}.db")
        raw_size, packed_size = clean.stat().st_size, packed.stat().st_size
        final = dest_dir / packed.name
        say(f"writing to {final} ...")
        shutil.copy2(packed, final)

    entry = {
        "name": name, "file": final.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "raw_bytes": raw_size, "packed_bytes": packed_size, **info,
    }
    man_path = dest_dir / MANIFEST
    man = json.loads(man_path.read_text()) if man_path.exists() else {"snapshots": []}
    man["snapshots"] = [s for s in man["snapshots"] if s["name"] != name] + [entry]
    man_path.write_text(json.dumps(man, indent=2))
    return entry


def list_snapshots(archive_dir: str | Path | None = None) -> list[dict]:
    d = Path(archive_dir).expanduser() if archive_dir else default_archive_dir()
    if d is None or not d.exists():
        return []
    man_path = d / MANIFEST
    if man_path.exists():
        entries = json.loads(man_path.read_text()).get("snapshots", [])
        # Only report what is actually on disk; a manifest can outlive its files.
        return [e for e in entries if (d / e["file"]).exists()]
    return [{"name": f.stem.replace(".db", ""), "file": f.name,
             "packed_bytes": f.stat().st_size}
            for f in sorted(d.glob("*.db.*"))]


def restore(name: str, archive_dir: str | Path | None = None,
            db: str | Path | None = None, force: bool = False,
            progress=None) -> Path:
    say = progress or (lambda m: None)
    d = Path(archive_dir).expanduser() if archive_dir else default_archive_dir()
    if d is None or not d.exists():
        raise RuntimeError("no archive directory found")
    cands = [f for f in d.glob(f"{name}.db.*")] or [f for f in d.glob(f"{name}*")]
    if not cands:
        raise FileNotFoundError(
            f"no snapshot named {name!r} in {d}; have: "
            f"{[s['name'] for s in list_snapshots(d)]}")
    packed = cands[0]
    target = db_path(db)
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists; pass --force to overwrite "
            f"(snapshot it first if it is not already archived)")
    say(f"unpacking {packed.name} ({human(packed.stat().st_size)}) ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".restoring")
    _decompress(packed, tmp)
    # Clear stale WAL/SHM from whatever database used to live here.
    for suf in ("-wal", "-shm"):
        p = Path(str(target) + suf)
        if p.exists():
            p.unlink()
    shutil.move(str(tmp), str(target))
    return target
