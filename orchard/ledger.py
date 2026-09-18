"""Output files (spec 6).

Four artefacts, all append-only and all inspectable without re-running anything:

  ``trades.jsonl``   one row per episode: hidden state, every token spoken, both
                     decisions, the outcome and its failure classification, and
                     the reward and money each side ended up with.  Complete
                     enough to reconstruct any trade after the fact.
  ``trades.csv``     the same rows, flattened, for spreadsheet/pandas use.
  ``metrics.jsonl``  one row per checkpoint: every spec-5 measurement.
  ``births.jsonl``   one row per birth: who died, who replaced them, what the
                     newborn was trained on and how well it did against veterans.

Plus ``run.log``, a mirror of everything printed to the console.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Iterable, Optional, Sequence

from .config import Config
from .env import BUYER, FARMER, Decision, Outcome
from .render import compact_transcript_row, render_tokens
from .world import Scenario


def _jsonable(x: Any) -> Any:
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


class RunLogger:
    """Console + file logger.  Everything a human sees during the run is kept."""

    def __init__(self, out_dir: str, quiet: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "run.log")
        self.fh = open(self.path, "a", encoding="utf-8")
        self.quiet = quiet
        self.t0 = time.time()

    def __call__(self, msg: str = "") -> None:
        line = str(msg)
        if not self.quiet:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        self.fh.write(line + "\n")
        self.fh.flush()

    def rule(self, title: str = "", width: int = 78) -> None:
        if title:
            pad = max(0, width - len(title) - 4)
            self(("=" * 3) + " " + title + " " + ("=" * pad))
        else:
            self("=" * width)

    def elapsed(self) -> float:
        return time.time() - self.t0

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


CSV_FIELDS = [
    "episode", "day", "season",
    "farmer_id", "farmer_slot", "farmer_age", "farmer_generation",
    "buyer_id", "buyer_slot", "buyer_age", "buyer_generation",
    "true_farmer_barn", "true_farmer_offered_stock", "true_farmer_offered_quality",
    "true_farmer_reservation",
    "true_buyer_variety", "true_buyer_need", "true_buyer_min_quality", "true_buyer_max_price",
    "viable", "held_out",
    "msg_text",
    "farmer_accept", "farmer_believed_variety", "farmer_believed_qty", "farmer_believed_price",
    "buyer_accept", "buyer_believed_variety", "buyer_believed_qty", "buyer_believed_price",
    "farmer_reads_buyer", "buyer_reads_farmer",
    "success", "failure_mode", "reasons",
    "traded_qty", "traded_price", "trade_value", "farmer_profit", "buyer_savings",
    "farmer_reward", "buyer_reward",
]


class Ledger:
    """The per-episode trade ledger (spec 6.1)."""

    def __init__(self, cfg: Config, out_dir: str, stride: int = 1, write_csv: bool = True):
        self.cfg = cfg
        os.makedirs(out_dir, exist_ok=True)
        self.stride = max(1, stride)
        self.jsonl = open(os.path.join(out_dir, "trades.jsonl"), "a", encoding="utf-8")
        self.csv_fh = None
        self.csv_w = None
        if write_csv:
            path = os.path.join(out_dir, "trades.csv")
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            self.csv_fh = open(path, "a", encoding="utf-8", newline="")
            self.csv_w = csv.DictWriter(self.csv_fh, fieldnames=CSV_FIELDS,
                                        extrasaction="ignore")
            if new:
                self.csv_w.writeheader()
        self.n_written = 0
        self._since_flush = 0

    # ------------------------------------------------------------------
    def row(self, *, episode: int, season: int, scenario: Scenario, farmer, buyer,
            transcript, fd: Decision, bd: Decision, outcome: Outcome) -> dict[str, Any]:
        cfg = self.cfg
        w = cfg.world
        msgs = compact_transcript_row(cfg, transcript)
        return {
            "episode": episode,
            "day": scenario.day,
            "season": season,
            "farmer_id": farmer.agent_id, "farmer_slot": farmer.slot,
            "farmer_age": farmer.age, "farmer_generation": farmer.generation,
            "buyer_id": buyer.agent_id, "buyer_slot": buyer.slot,
            "buyer_age": buyer.age, "buyer_generation": buyer.generation,

            # ground truth -- present in the ledger, never in any agent's input
            "true_farmer_barn": "; ".join(
                "%s:%d@%s" % (w.variety_names[v], scenario.farmer.stocks[v],
                              w.quality_names[scenario.farmer.qualities[v]])
                for v in range(w.n_varieties) if scenario.farmer.stocks[v] > 0) or "empty",
            "true_farmer_stocks": list(scenario.farmer.stocks),
            "true_farmer_qualities": list(scenario.farmer.qualities),
            "true_farmer_offered_stock": scenario.offered_stock,
            "true_farmer_offered_quality": w.quality_names[scenario.offered_quality],
            "true_farmer_reservation": w.price_values[scenario.farmer.reservation],
            "true_buyer_variety": w.variety_names[scenario.buyer.want_variety],
            "true_buyer_need": scenario.buyer.need_qty,
            "true_buyer_min_quality": w.quality_names[scenario.buyer.min_quality],
            "true_buyer_max_price": w.price_values[scenario.buyer.max_price],
            "viable": scenario.viable,
            "held_out": scenario.held_out,

            "msg_symbols": msgs["msg_symbols"],
            "msg_words": msgs["msg_words"],
            "msg_text": msgs["msg_text"],

            "farmer_accept": int(fd.accept),
            "farmer_believed_variety": w.variety_names[fd.variety],
            "farmer_believed_qty": fd.qty,
            "farmer_believed_price": w.price_values[fd.price],
            "buyer_accept": int(bd.accept),
            "buyer_believed_variety": w.variety_names[bd.variety],
            "buyer_believed_qty": bd.qty,
            "buyer_believed_price": w.price_values[bd.price],

            "farmer_reads_buyer": round(outcome.farmer_decode, 3),
            "buyer_reads_farmer": round(outcome.buyer_decode, 3),
            "farmer_belief": (list(transcript.farmer_beliefs.as_tuple())
                              if transcript.farmer_beliefs else None),
            "buyer_belief": (list(transcript.buyer_beliefs.as_tuple())
                             if transcript.buyer_beliefs else None),
            "success": outcome.success,
            "failure_mode": outcome.failure_mode,
            "reasons": ";".join(outcome.reasons),
            "traded_qty": outcome.traded_qty,
            "traded_price": (w.price_values[outcome.traded_price_bin]
                             if outcome.traded_price_bin >= 0 else None),
            "trade_value": round(outcome.trade_value, 4),
            "farmer_profit": round(outcome.farmer_profit, 4),
            "buyer_savings": round(outcome.buyer_savings, 4),
            "farmer_reward": round(outcome.farmer_reward, 5),
            "buyer_reward": round(outcome.buyer_reward, 5),
        }

    def write(self, row: dict[str, Any]) -> None:
        self.jsonl.write(json.dumps(_jsonable(row), separators=(",", ":")) + "\n")
        if self.csv_w is not None:
            self.csv_w.writerow(row)
        self.n_written += 1
        self._since_flush += 1
        if self._since_flush >= self.cfg.log.flush_every:
            self.flush()

    def write_batch(self, batch, pop, episode0: int, season: int) -> int:
        n = 0
        for i in range(len(batch)):
            ep = episode0 + i
            if ep % self.stride:
                continue
            # Only the rows that are actually written get built into objects,
            # which is why a wide stride is cheap on a long run.
            tr = batch.transcript(i)
            self.write(self.row(
                episode=ep, season=season, scenario=batch.scenario(i),
                farmer=pop.farmers[int(batch.f_idx[i])],
                buyer=pop.buyers[int(batch.b_idx[i])],
                transcript=tr, fd=tr.farmer_decision, bd=tr.buyer_decision,
                outcome=tr.outcome))
            n += 1
        return n

    def flush(self) -> None:
        self.jsonl.flush()
        if self.csv_fh:
            self.csv_fh.flush()
        self._since_flush = 0

    def close(self) -> None:
        self.flush()
        try:
            self.jsonl.close()
        finally:
            if self.csv_fh:
                self.csv_fh.close()


class JsonlLog:
    """Generic append-only JSONL sink (metrics, births)."""

    def __init__(self, out_dir: str, name: str):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, name)
        self.fh = open(self.path, "a", encoding="utf-8")
        self.rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        clean = _jsonable(row)
        self.rows.append(clean)
        self.fh.write(json.dumps(clean, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass
