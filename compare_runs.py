"""Side-by-side comparison of two runs on the measures that decide whether a
reward or world change actually did anything.

    python compare_runs.py runs/main runs/main2

Pulls from each run's own artefacts -- metrics.jsonl, token_semantics.json and
the trade ledger -- so it reports what a run actually recorded rather than what a
report was written to say.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional


def load_metrics(run: str) -> Optional[dict[str, Any]]:
    p = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[-1] if rows else None


def load_semantics(run: str) -> dict[str, Any]:
    p = os.path.join(run, "token_semantics.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def ledger_viability(run: str, cap: int = 400_000) -> tuple[int, float, float]:
    """Viable and success rate straight from the ledger, not from a summary."""
    p = os.path.join(run, "trades.jsonl")
    if not os.path.exists(p):
        return 0, float("nan"), float("nan")
    n = v = s = 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            v += bool(r.get("viable"))
            s += bool(r.get("success"))
            if n >= cap:
                break
    if not n:
        return 0, float("nan"), float("nan")
    return n, v / n, s / n


def num(d: Any, *path, default=float("nan")) -> float:
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur if isinstance(cur, (int, float)) else default


def fmt(x: float, pct: bool = False, places: int = 3) -> str:
    if x is None or x != x:
        return "  --  "
    return ("%.0f%%" % (100 * x)) if pct else ("%.*f" % (places, x))


def positional(run: str, role: str) -> list[dict[str, Any]]:
    sem = load_semantics(run)
    return (sem.get("per_position") or {}).get(role, [])


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    runs = argv[1:]
    names = [os.path.basename(r.rstrip("/\\")) for r in runs]
    ms = [load_metrics(r) for r in runs]
    leds = [ledger_viability(r) for r in runs]
    missing = [n for n, m in zip(names, ms) if m is None]
    if missing:
        print("no metrics for: %s" % ", ".join(missing))
        return 1

    w = max(20, max(len(n) for n in names) + 3)
    def row(label, vals):
        print("  %-42s" % label + "".join(("%%%ds" % w) % v for v in vals))

    print("=" * (44 + w * len(runs)))
    print("  RUN COMPARISON")
    print("=" * (44 + w * len(runs)))
    row("", names)
    print()

    print("  -- the four you asked for --")
    row("viable-episode rate (from ledger)",
        [fmt(l[1], pct=True) for l in leds])
    row("task success rate (eval)",
        [fmt(num(m, "eval_success")) for m in ms])
    row("task success rate (from ledger)",
        [fmt(l[2]) for l in leds])
    row("population coherence (buyer)",
        [fmt(num(m, "stability", "coherence_buyer")) for m in ms])
    row("population coherence (farmer)",
        [fmt(num(m, "stability", "coherence_farmer")) for m in ms])
    print()

    print("  -- farmer-slot positional structure --")
    slots = max((len(positional(r, "farmer")) for r in runs), default=0)
    for k in range(slots):
        cells = []
        for r in runs:
            rows_ = positional(r, "farmer")
            if k < len(rows_):
                cells.append("%.3f %s" % (rows_[k]["score"],
                                            rows_[k]["dimension"][:12]))
            else:
                cells.append("  --  ")
        row("slot %d" % k, cells)
    row("mean farmer-slot score",
        [fmt(sum(x["score"] for x in positional(r, "farmer"))
             / max(1, len(positional(r, "farmer")))) for r in runs])
    print()

    print("  -- did the loop close? --")
    row("farmer reads buyer (intact)",
        [fmt(num(m, "channel_ablation", "intact_farmer_reads")) for m in ms])
    row("farmer reads buyer (muted)",
        [fmt(num(m, "channel_ablation", "muted_farmer_reads")) for m in ms])
    row("  -> share carried by channel",
        [fmt(num(m, "channel_ablation", "farmer_reads_transfer"), pct=True) for m in ms])
    row("buyer reads farmer (intact)",
        [fmt(num(m, "channel_ablation", "intact_buyer_reads")) for m in ms])
    row("buyer reads farmer (muted)",
        [fmt(num(m, "channel_ablation", "muted_buyer_reads")) for m in ms])
    row("  -> share carried by channel",
        [fmt(num(m, "channel_ablation", "buyer_reads_transfer"), pct=True) for m in ms])
    print()

    print("  -- context --")
    row("episodes", ["%d" % num(m, "episode", default=0) for m in ms])
    row("farmer names right variety",
        [fmt(num(m, "farmer_variety_acc")) for m in ms])
    row("  -> share carried by channel",
        [fmt(num(m, "channel_ablation", "variety_transfer"), pct=True) for m in ms])
    row("mutual comprehension",
        [fmt(num(m, "comprehension_rate")) for m in ms])
    row("topsim (compositionality)",
        [fmt(num(m, "compositionality", "mean")) for m in ms])
    row("distinct words", ["%d" % num(m, "words", "distinct_words", default=0) for m in ms])
    row("symbols per utterance",
        [fmt(num(m, "words", "mean_symbols_per_message"), places=2) for m in ms])
    row("length vs frequency (rho)",
        [fmt(num(m, "length_frequency", "rho_symbols")) for m in ms])
    row("form drift, frequent (run mean)",
        [fmt(num(m, "form_survival", "drift_frequent")) for m in ms])
    row("form drift, rare (run mean)",
        [fmt(num(m, "form_survival", "drift_rare")) for m in ms])
    row("replacements logged",
        ["%d" % num(m, "form_survival", "n_events", default=0) for m in ms])
    row("births", ["%d" % num(m, "population", "total_births", default=0) for m in ms])
    print("=" * (44 + w * len(runs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
