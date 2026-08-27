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

    p = sub.add_parser("validate", help="run the full control harness on a source")
    p.add_argument("source")
    p.add_argument("--docs", type=int, default=2)
    p.add_argument("--claims", type=int, default=4)
    p.add_argument("--decoy", default=None)
    p.add_argument("--json", default=None)

    p = sub.add_parser("ablate", help="measure a source's value to a model")
    p.add_argument("source")
    p.add_argument("--docs", type=int, default=2)
    p.add_argument("--claims", type=int, default=3)
    p.add_argument("--json", default=None)

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

    if args.cmd == "ablate":
        import json as _json
        from .score.ablation import ablate_source
        cfg = Config()
        llm = LLM(store=store, effort=cfg.effort)
        embedder = Embedder(store, backend=cfg.embed_backend)
        res = ablate_source(store, llm, args.source, cfg, embedder=embedder,
                            n_docs=args.docs, claims_per_doc=args.claims,
                            progress=lambda m: print(m, flush=True))
        d = res.to_dict()
        print(f"\n{'='*70}\nVALUE TO A MODEL  |  {d['source']}\n{'='*70}")
        print(f"  closed book       {d['mean_closed_book']:.3f}   model unaided")
        decoy = d['mean_with_decoy']
        print(f"  + decoy context   {'n/a  ' if decoy is None else f'{decoy:.3f}'}"
              f"   contemporaries on the same topic")
        print(f"  + THIS source     {d['mean_with_source']:.3f}")
        print(f"  {'-'*40}")
        print(f"  inference value   {d['mean_inference_value']:+.3f}   "
              f"excess over the best alternative context")
        print(f"  internalised      {d['internalised_rate']:.3f}   "
              f"share already answerable from the weights alone")
        print(f"  decoy-controlled  {d['n_decoy_controlled']}/{d['n_claims']} claims")
        if d.get("decoy_caveat"):
            print(f"\n  CAVEAT: {d['decoy_caveat']}")
        if d['internalised_rate'] > 0.8:
            print("\n  Note: nearly every claim is already answerable closed-book.")
            print("  This source's contribution appears to be priced into the model")
            print("  already -- high past training value, little marginal value now.")
        if args.json:
            with open(args.json, "w") as fh:
                _json.dump(d, fh, indent=2)
            print(f"\nfull report written to {args.json}")
        return 0

    if args.cmd == "validate":
        from .validate.controls import judge_agreement_control
        cfg = Config()
        llm = LLM(store=store, effort=cfg.effort)
        embedder = Embedder(store, backend=cfg.embed_backend)
        rep = score_source(
            store, llm, args.source, cfg=cfg, n_docs=args.docs,
            max_claims_per_doc=args.claims, embedder=embedder,
            run_controls=True, decoy_source_name=args.decoy,
            progress=lambda m: print(m, flush=True),
        )
        # Judge agreement needs a second LLM at a different setting, so it runs
        # here rather than inside the scoring pass.
        src = store.find_source(args.source)
        pairs = [(c, d.published_at)
                 for d in store.documents_for_source(src.id, limit=50)
                 for c in sorted(store.get_claims(d.id), key=lambda c: -c.salience)[:3]][:5]
        if pairs:
            print("running controls: judge_agreement ...", flush=True)
            rep.controls.append(judge_agreement_control(
                store, pairs, cfg, make_llm=lambda e: LLM(store=store, effort=e),
                embedder=embedder, source_id=src.id, run_id=rep.run_id))
        render(rep, sys.stdout)
        if args.json:
            with open(args.json, "w") as fh:
                fh.write(to_json(rep))
            print(f"\nfull report written to {args.json}")
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
