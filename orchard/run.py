"""Command-line entry point.

    python -m orchard.run --config configs/default.json --out runs/main
    python -m orchard.run --config configs/small.json --out runs/nobottleneck --bottleneck off
    python -m orchard.run --smoke                      # spec step 1: scripted agents only
    python -m orchard.run --compare runs/a runs/b      # overlay finished runs

The ablation switches (``--bottleneck on|off``, ``--turnover on|off``) are the
point of the whole exercise: same code, same seed, one mechanism removed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any

import torch

from .config import Config, add_config_args, config_from_args
from .env import (BUYER, FARMER, HonestScriptedAgent, RandomScriptedAgent,
                  run_scripted_episode)
from .metrics import analyse_token_semantics, chance_success_rate
from .render import render_transcript
from .report import write_report
from .train import Trainer
from .world import World


# --------------------------------------------------------------------------
def smoke(cfg: Config) -> int:
    """Spec step 1: exercise the environment with scripted agents, no learning."""
    print("=" * 78)
    print("SMOKE TEST -- environment mechanics only, no neural networks involved")
    print("=" * 78)
    rng = random.Random(0)
    world = World(cfg.world, random.Random(0))
    fa = RandomScriptedAgent(cfg, FARMER, rng)
    ba = RandomScriptedAgent(cfg, BUYER, rng)

    n = 3000
    succ = viable = 0
    modes: dict[str, int] = {}
    for _ in range(n):
        sc = world.sample(held_out=False)
        tr = run_scripted_episode(cfg, sc, fa, ba)
        succ += int(tr.outcome.success)
        viable += int(sc.viable)
        modes[tr.outcome.failure_mode] = modes.get(tr.outcome.failure_mode, 0) + 1
    print("\nrandom agents over %d episodes:" % n)
    print("  viable scenarios : %.3f" % (viable / n))
    print("  success rate     : %.4f   <- this is the chance baseline" % (succ / n))
    print("  outcome mix      : %s" % dict(sorted(modes.items(), key=lambda kv: -kv[1])))

    holder: dict[str, Any] = {}
    ofa = HonestScriptedAgent(cfg, FARMER, rng, lambda: holder["sc"])
    oba = HonestScriptedAgent(cfg, BUYER, rng, lambda: holder["sc"])
    succ = viable = 0
    for _ in range(n):
        sc = world.sample(held_out=False)
        holder["sc"] = sc
        tr = run_scripted_episode(cfg, sc, ofa, oba)
        succ += int(tr.outcome.success)
        viable += int(sc.viable)
    print("\noracle pair (cheats by construction; upper bound only):")
    print("  success rate     : %.4f  of %.3f viable" % (succ / n, viable / n))

    print("\nexample episode, rendered:")
    sc = world.sample(held_out=False)
    tr = run_scripted_episode(cfg, sc, fa, ba)
    print(render_transcript(cfg, tr, indent="  "))
    print("\nenvironment mechanics look sane.")
    return 0


# --------------------------------------------------------------------------
def compare(paths: list[str], out: str) -> int:
    from .plots import write_comparison
    runs: dict[str, Any] = {}
    rows = []
    for p in paths:
        hp = os.path.join(p, "history.json")
        if not os.path.exists(hp):
            print("skipping %s (no history.json)" % p)
            continue
        with open(hp, "r", encoding="utf-8") as fh:
            runs[os.path.basename(p.rstrip("/\\"))] = json.load(fh)
        mp = os.path.join(p, "metrics.jsonl")
        last = None
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        last = json.loads(line)
        if last:
            rows.append((os.path.basename(p.rstrip("/\\")), last))
    if not runs:
        print("nothing to compare")
        return 1
    os.makedirs(out, exist_ok=True)
    png = write_comparison(runs, os.path.join(out, "comparison.png"))

    def num(d, *path, default=float("nan")):
        cur = d
        for k in path:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
        return cur if isinstance(cur, (int, float)) else default

    lines = ["# Ablation comparison", "",
             "Same code, same seed, one mechanism changed. The columns that matter "
             "most are the two ablation ones: *variety naming* is whether a word for "
             "a thing emerged at all, and *channel* is how much of the available "
             "headroom the messages are actually responsible for.",
             "",
             "| run | episodes | variety naming | channel | success | comprehension | "
             "topsim | coherence | transmission | zero-shot |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        st = m.get("stability", {})
        coh = ((st.get("coherence_buyer") or 0) + (st.get("coherence_farmer") or 0)) / 2
        lines.append("| %s | %d | %.3f | %.0f%% | %.3f | %.3f | %.3f | %.3f | %.2f | %.2f |" % (
            name, m.get("episode", 0),
            num(m, "channel_ablation", "intact_variety_acc"),
            100 * (num(m, "channel_ablation", "variety_transfer", default=0.0) or 0.0),
            num(m, "eval_success"), num(m, "comprehension_rate"),
            num(m, "compositionality", "mean"), coh,
            num(m, "intelligibility", "transmission_ratio"),
            num(m, "zero_shot", "retention")))

    lines += ["", "## Vocabulary (addendum section 3)", "",
              "| run | distinct words | words/utterance | multi-atom words | symbols/utterance | "
              "length vs frequency | topsim frequent | topsim rare | drift frequent | drift rare |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        lines.append("| %s | %d | %.2f | %.0f%% | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            name, int(num(m, "words", "distinct_words", default=0)),
            num(m, "words", "mean_words_per_message"),
            100 * (num(m, "words", "multi_atom_word_share", default=0.0) or 0.0),
            num(m, "words", "mean_symbols_per_message"),
            num(m, "length_frequency", "rho_symbols"),
            num(m, "buckets", "frequent", "topsim"),
            num(m, "buckets", "rare", "topsim"),
            num(m, "form_survival", "drift_frequent"),
            num(m, "form_survival", "drift_rare")))
    md = os.path.join(out, "comparison.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote %s and %s" % (png, md))
    return 0


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orchard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--out", type=str, default=None, help="output directory for this run")
    p.add_argument("--smoke", action="store_true",
                   help="environment-only check with scripted agents (spec step 1)")
    p.add_argument("--compare", nargs="+", default=None,
                   help="finished run directories to overlay")
    p.add_argument("--compare-out", type=str, default="runs/comparison")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.compare:
        return compare(args.compare, args.compare_out)

    cfg = config_from_args(args)
    if args.smoke:
        return smoke(cfg)

    out = args.out or os.path.join("runs", cfg.name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    trainer = Trainer(cfg, out, quiet=args.quiet)
    try:
        final = trainer.run()
        trainer.log("")
        trainer.log("running post-hoc token-semantics analysis (spec 6.3)...")
        sem = analyse_token_semantics(cfg, trainer.pop, trainer.world,
                                      n_samples=max(400, cfg.log.topsim_samples * 2),
                                      device=cfg.train.device, rng=trainer.eval_rng)
        with open(os.path.join(out, "token_semantics.json"), "w", encoding="utf-8") as fh:
            json.dump(sem.to_dict(), fh, indent=1)

        # keep the raw token counts available to the report
        from .lexicon import length_frequency, word_stats
        from .metrics import evaluate_success, vocab_stats
        ev = evaluate_success(cfg, trainer.pop, trainer.world, 600,
                              device=cfg.train.device, rng=trainer.eval_rng)
        final["token_counts"] = vocab_stats(cfg, [ev["batch"]])["token_counts"]
        final["word_counts"] = word_stats(cfg, [ev["batch"]])["word_counts"]
        final["length_frequency_rows"] = length_frequency(
            cfg, trainer.pop, trainer.world, device=cfg.train.device).get("rows", [])[:20]
        if trainer.forms is not None:
            final["form_events"] = trainer.forms.report_rows()
            final["form_timeline"] = trainer.forms.timeline()

        path = write_report(
            cfg, out, final=final, chance=trainer.chance, sem=sem,
            archive=trainer.archive, totals=trainer.totals,
            history=trainer.history.to_dict(),
            newborn_reports=trainer.newborn_reports,
            ledger_path=os.path.join(out, "trades.jsonl"),
            wall_minutes=(time.time() - t0) / 60.0)

        trainer.log("")
        trainer.log("wrote final report: %s" % path)
        trainer.log("ledger rows written: %d" % trainer.ledger.n_written)

        # Print the verdict last so it is the thing a human sees.
        from .report import assess
        v = assess(cfg, final, trainer.chance)
        trainer.log("")
        trainer.log("=" * 78)
        trainer.log("VERDICT: %s" % v["verdict"])
        trainer.log(v["summary"])
        trainer.log("=" * 78)
    except KeyboardInterrupt:
        trainer.log("\ninterrupted -- flushing logs")
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
