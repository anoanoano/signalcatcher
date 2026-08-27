"""Command line entry point."""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .db import Store
from .index.embed import Embedder
from .llm import LLM
from .pipeline import score_source
from .report import render, to_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="signalcatcher", description=__doc__)
    ap.add_argument("--db", default="data/corpus.db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpus", help="show what is in the pinned corpus")

    p = sub.add_parser("ingest", help="add a publication to the corpus")
    p.add_argument("target")
    p.add_argument("--kind", choices=["substack", "wordpress"], default="substack")
    p.add_argument("--max-posts", type=int, default=900)

    p = sub.add_parser("score", help="score a source and run the controls")
    p.add_argument("source", help="source name as listed by `corpus`")
    p.add_argument("--docs", type=int, default=3)
    p.add_argument("--claims", type=int, default=6)
    p.add_argument("--decoy", default=None, help="contemporaneous source for the base-rate control")
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--effort", default="high")
    p.add_argument("--json", default=None, help="also write the full report here")

    args = ap.parse_args(argv)
    store = Store(args.db)

    if args.cmd == "corpus":
        lo, hi = store.corpus_span()
        print(f"{store.count_documents()} documents | {lo} .. {hi}")
        for src, n in store.list_sources():
            print(f"  {n:6d}  {src.kind:9} {src.name[:40]:42} {src.domain}")
        return 0

    if args.cmd == "ingest":
        from .ingest import substack, wordpress
        mod = substack if args.kind == "substack" else wordpress
        r = mod.ingest(store, args.target, max_posts=args.max_posts)
        print(r)
        return 0

    if args.cmd == "score":
        cfg = Config(effort=args.effort)
        llm = LLM(store=store, effort=args.effort)
        embedder = Embedder(store, backend=cfg.embed_backend)
        rep = score_source(
            store, llm, args.source, cfg=cfg, n_docs=args.docs,
            max_claims_per_doc=args.claims, embedder=embedder,
            run_controls=not args.no_controls, decoy_source_name=args.decoy,
            progress=lambda m: print(m, flush=True),
        )
        render(rep, sys.stdout)
        if args.json:
            with open(args.json, "w") as fh:
                fh.write(to_json(rep))
            print(f"\nfull report written to {args.json}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
