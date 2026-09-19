"""Measure a saved population after the fact.

    python -m orchard.analyse --snapshot runs/<name>/snapshots/after-refer-mutual.pt

Loads a snapshot (weights, recent usage, transcript store, curriculum record),
runs the full checkpoint metric suite on the rung you choose -- by default the
one the snapshot was taken at the end of -- and prints the language-properties
scorecard. Nothing is trained. Useful for runs that predate a metric, and for
looking at a cloud run's snapshots on a laptop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--config", default=None,
                   help="config to measure under (default: the one saved in the snapshot)")
    p.add_argument("--phase", default=None,
                   help="rung to measure on (default: the rung the snapshot closed)")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    from .config import Config
    from .metrics import analyse_token_semantics
    from .train import Trainer

    st = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    cfg = Config.from_json(args.config) if args.config else Config.from_dict(st["config"])
    cfg.train.device = args.device
    cfg.log.plot = False
    tag = os.path.splitext(os.path.basename(args.snapshot))[0]
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(args.snapshot)),
                                   "analysis-" + tag)
    tr = Trainer(cfg, out, quiet=True)
    tr.load_snapshot(args.snapshot)

    names = [ph.name for ph in tr.curriculum.phases]
    name = args.phase
    if name is None:
        name = (tr.curriculum.transitions[-1]["from"]
                if tag.startswith("after-") and tr.curriculum.transitions
                else tr.curriculum.phase.name)
    tr.curriculum.index = names.index(name)
    print("measuring %s at episode %d on rung %s ..." % (args.snapshot, tr.episode, name))
    row = tr.checkpoint(final=True)
    sem = analyse_token_semantics(cfg, tr.pop, tr.world, n_samples=400,
                                  device=tr.device, rng=tr.eval_rng,
                                  phase=tr.curriculum.phase)
    props = tr.language_properties(sem)
    print("")
    print("%-22s %-10s %-26s %s" % ("property", "value", "verdict", "note"))
    for q in props:
        v = q["value"]
        vs = ("%.3f" % v) if isinstance(v, float) and v == v else (
            str(v) if isinstance(v, int) else "-")
        print("%-22s %-10s %-26s %s" % (q["property"], vs, q["verdict"], q.get("note", "")))
    with open(os.path.join(out, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump({"snapshot": args.snapshot, "rung": name, "episode": tr.episode,
                   "properties": props,
                   "per_role_structure": row.get("per_role_structure"),
                   "cross_role_overlap": row.get("cross_role_overlap"),
                   "zero_shot": row.get("zero_shot"),
                   "context_consistency": row.get("context_consistency")},
                  fh, indent=1, default=str)
    tr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
