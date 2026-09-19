"""Run one configuration across several seeds and report the spread.

    python sweep.py --config configs/gpu_small.json --out runs/sweep --seeds 5
    python sweep.py --config configs/gpu_small.json --out runs/ablate --seeds 5 \\
        --arm "bottleneck_on:" --arm "bottleneck_off:--bottleneck off"

Why this exists
---------------
Single runs of this simulation are bimodal. A pair either finds a referential
convention or it does not, and at 40k episodes the same settings produced 76%,
0%, 92% and 6% of the channel headroom across four neighbouring conditions. Any
conclusion drawn from one seed per arm is a coin flip dressed up as a result, so
comparisons go through here and report a mean and a spread.

Runs are launched as separate processes. On one GPU they are run one at a time by
default (they would otherwise contend for it); pass --parallel N to overlap them,
which is usually right on a CPU box with spare cores.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

# What a sweep reports on. Each entry is (label, path into the metrics row).
METRICS = [
    ("variety naming", ("channel_ablation", "intact_variety_acc")),
    ("channel share", ("channel_ablation", "variety_transfer")),
    ("farmer reads buyer", ("channel_ablation", "farmer_reads_transfer")),
    ("buyer reads farmer", ("channel_ablation", "buyer_reads_transfer")),
    ("success rate", ("eval_success",)),
    ("comprehension", ("comprehension_rate",)),
    ("topsim", ("compositionality", "mean")),
    ("distinct words", ("words", "distinct_words")),
    ("symbols/utterance", ("words", "mean_symbols_per_message")),
    ("length vs frequency", ("length_frequency", "rho_symbols")),
]


def dig(d: Any, path) -> Optional[float]:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur if isinstance(cur, (int, float)) and cur == cur else None


def final_metrics(run_dir: str) -> Optional[dict]:
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[-1] if rows else None


def launch(cfg_path: str, out: str, seed: int, extra: list[str],
           quiet: bool = True) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "orchard.run", "--config", cfg_path,
           "--out", out, "--name", os.path.basename(out), "--seed", str(seed)]
    if quiet:
        cmd.append("--quiet")
    cmd += extra
    log = open(out + ".log", "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="directory to put the arms in")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=1,
                    help="how many runs to have going at once (1 for a single GPU)")
    ap.add_argument("--arm", action="append", default=[],
                    metavar="NAME:EXTRA ARGS",
                    help="a named arm; repeat it. Default is one unnamed arm.")
    ap.add_argument("--dry-run", action="store_true")
    args, passthrough = ap.parse_known_args(argv[1:])

    arms: list[tuple[str, list[str]]] = []
    for spec in args.arm or ["base:"]:
        name, _, extra = spec.partition(":")
        arms.append((name.strip() or "base", shlex.split(extra) + passthrough))

    jobs = []
    for name, extra in arms:
        for k in range(args.seeds):
            seed = args.seed0 + k
            out = os.path.join(args.out, "%s_s%d" % (name, seed))
            jobs.append((name, seed, out, extra))

    print("%d arm(s) x %d seeds = %d runs" % (len(arms), args.seeds, len(jobs)))
    for name, seed, out, extra in jobs:
        print("   %-18s seed %d -> %s %s" % (name, seed, out, " ".join(extra)))
    if args.dry_run:
        return 0

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    running: list[tuple[subprocess.Popen, str]] = []
    queue = list(jobs)
    done = 0
    while queue or running:
        while queue and len(running) < max(1, args.parallel):
            name, seed, out, extra = queue.pop(0)
            os.makedirs(out, exist_ok=True)
            running.append((launch(args.config, out, seed, extra), out))
        time.sleep(2.0)
        for proc, out in list(running):
            if proc.poll() is not None:
                running.remove((proc, out))
                done += 1
                print("  [%d/%d] finished %s (exit %s, %.0f min elapsed)"
                      % (done, len(jobs), os.path.basename(out), proc.returncode,
                         (time.time() - t0) / 60))

    report(args.out, arms, args.seeds, args.seed0)
    return 0


def report(root: str, arms, n_seeds: int, seed0: int) -> None:
    print()
    print("=" * 96)
    print("  SWEEP RESULT -- mean +/- spread over %d seeds" % n_seeds)
    print("=" * 96)
    data: dict[str, dict[str, list[float]]] = {}
    for name, _ in arms:
        data[name] = {label: [] for label, _ in METRICS}
        for k in range(n_seeds):
            m = final_metrics(os.path.join(root, "%s_s%d" % (name, seed0 + k)))
            if not m:
                continue
            for label, path in METRICS:
                v = dig(m, path)
                if v is not None:
                    data[name][label].append(v)

    w = max(22, max(len(n) for n, _ in arms) + 4)
    header = "  %-22s" % "" + "".join(("%%-%ds" % w) % n for n, _ in arms)
    print(header)
    for label, _ in METRICS:
        cells = []
        for name, _ in arms:
            vals = data[name][label]
            if not vals:
                cells.append("--")
            elif len(vals) == 1:
                cells.append("%.3f" % vals[0])
            else:
                cells.append("%.3f +/- %.3f" % (statistics.fmean(vals),
                                                statistics.pstdev(vals)))
        print("  %-22s" % label + "".join(("%%-%ds" % w) % c for c in cells))
    print()
    for name, _ in arms:
        vals = data[name]["channel share"]
        if len(vals) > 1:
            print("  %s channel share by seed: %s"
                  % (name, ", ".join("%.2f" % v for v in vals)))
    print("=" * 96)
    with open(os.path.join(root, "sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print("  raw numbers in %s" % os.path.join(root, "sweep.json"))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
