"""The training loop: play, learn, age, die, be replaced, measure, report.

One iteration:
  1. the economy generates a batch of encounters and pairs them off,
  2. the paired agents negotiate (:func:`orchard.rollout.run_episodes`),
  3. the market settles: lots deplete, ledger rows are written, successful
     transcripts go into the bottleneck store,
  4. every agent that played takes a REINFORCE step on its own episodes,
  5. agents that have run out of lifespan die and are replaced by newborns, who
     get their bottleneck apprenticeship and are immediately tested against the
     veterans that predate them,
  6. at checkpoints, the whole spec-5 metric suite runs and a human-readable
     summary is printed.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from .agents import Agent, count_parameters
from .bottleneck import TranscriptStore, train_newborn
from .config import Config
from .economy import Economy
from .env import BUYER, FARMER, ROLE_NAMES
from .ledger import JsonlLog, Ledger, RunLogger
from .metrics import (RollingStat, StabilityTracker, chance_success_rate,
                      channel_ablation, compositionality, detect_degenerate,
                      evaluate_success, intelligibility, newborn_vs_veterans,
                      vocab_stats, zero_shot)
from .curriculum import (CurriculumState, ReferentialWorld, ladder,
                         promotion_for)
from .lexicon import (FormTracker, WordProvenance, bucketed_analysis,
                      length_frequency, word_stats)
from .population import BirthEvent, Population
from .render import render_transcript
from .world import World


def expected_generations(cfg: Config) -> float:
    """How many times each lineage slot will turn over in a run of this length.

    Generations are not a direct knob: an agent ages by the episodes *it* plays,
    which is the run length divided by how many agents share the work, and it dies
    at its lifespan.  This is the inverse the GUI uses to turn "how many
    generations do you want" into an episode budget.
    """
    if not cfg.population.turnover:
        return 0.0
    per_agent = cfg.train.episodes / max(1, cfg.population.n_farmers)
    mean_life = (cfg.population.lifespan_min + cfg.population.lifespan_max) / 2.0
    return per_agent / max(1.0, mean_life)


def episodes_for_generations(generations: float, n_agents: int,
                             mean_lifespan: float) -> int:
    """The inverse of :func:`expected_generations`."""
    return int(round(generations * n_agents * mean_lifespan))


@dataclass
class History:
    episodes: list[int] = field(default_factory=list)
    train_success: list[float] = field(default_factory=list)
    eval_success: list[float] = field(default_factory=list)
    eval_success_viable: list[float] = field(default_factory=list)
    comprehension: list[float] = field(default_factory=list)
    judgement: list[float] = field(default_factory=list)
    variety_acc: list[float] = field(default_factory=list)
    farmer_reads: list[float] = field(default_factory=list)
    buyer_reads: list[float] = field(default_factory=list)
    farmer_reads_transfer: list[float] = field(default_factory=list)
    buyer_reads_transfer: list[float] = field(default_factory=list)
    variety_transfer: list[float] = field(default_factory=list)
    qty_acc: list[float] = field(default_factory=list)
    topsim: list[float] = field(default_factory=list)
    topsim_null: list[float] = field(default_factory=list)
    entropy: list[float] = field(default_factory=list)
    msg_len: list[float] = field(default_factory=list)
    drift: list[float] = field(default_factory=list)
    coherence: list[float] = field(default_factory=list)
    transmission: list[float] = field(default_factory=list)
    zeroshot: list[float] = field(default_factory=list)
    ablation_drop: list[float] = field(default_factory=list)
    information_transfer: list[float] = field(default_factory=list)
    # addendum: open-vocabulary series
    mean_symbols: list[float] = field(default_factory=list)
    mean_words: list[float] = field(default_factory=list)
    distinct_words: list[float] = field(default_factory=list)
    multi_atom_share: list[float] = field(default_factory=list)
    rho_length_frequency: list[float] = field(default_factory=list)
    topsim_frequent: list[float] = field(default_factory=list)
    topsim_rare: list[float] = field(default_factory=list)
    drift_frequent: list[float] = field(default_factory=list)
    drift_rare: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    generations: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) for k, v in self.__dict__.items()}


class Trainer:
    def __init__(self, cfg: Config, out_dir: str, *, quiet: bool = False,
                 resume_note: str = ""):
        self.cfg = cfg
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        cfg.to_json(os.path.join(out_dir, "config.json"))

        torch.manual_seed(cfg.train.seed)
        from .hardware import setup as hw_setup
        dev = hw_setup(cfg)
        # Pin the resolved device back onto the config so everything downstream --
        # agents, the tensor world, saved config.json -- agrees on one answer.
        cfg.train.device = str(dev)
        self.torch_device = dev
        self.rng = random.Random(cfg.train.seed)
        self.device = str(dev)

        self.log = RunLogger(out_dir, quiet=quiet)
        self.ledger = Ledger(cfg, out_dir, stride=cfg.log.ledger_stride)
        self.metrics_log = JsonlLog(out_dir, "metrics.jsonl")
        self.birth_log = JsonlLog(out_dir, "births.jsonl")
        # Phase 1 has no trades to put in the trade ledger, but its rounds are
        # still the record of how the language started, so they get their own.
        self.lineup_log = JsonlLog(out_dir, "lineups.jsonl")

        self.world = World(cfg.world, random.Random(cfg.train.seed + 1))
        # The fast path.  Same distributions, drawn on device in one go; the
        # scalar World stays for metrics probes and for the readable definition.
        self.tensor_world = None
        if cfg.train.vectorised:
            from .batched import TensorWorld
            g = torch.Generator(device=dev)
            g.manual_seed(cfg.train.seed + 11)
            self.tensor_world = TensorWorld(cfg, device=str(dev), generator=g)
        self.pop = Population(cfg, random.Random(cfg.train.seed + 2), device=self.device)
        self.economy = Economy(cfg, self.world, random.Random(cfg.train.seed + 3),
                               n_farms=cfg.population.n_farmers)
        self.store = TranscriptStore(cfg)
        self.stability = StabilityTracker(cfg, self.world, cfg.log.stability_probes,
                                          seed=cfg.train.seed + 4)
        self.forms = FormTracker(cfg, self.world) if cfg.log.track_form_survival else None
        self.provenance = WordProvenance()

        # ---- the curriculum -------------------------------------------------
        self.curriculum = CurriculumState(ladder(cfg))
        if not cfg.curriculum.enabled:
            self.curriculum.index = len(self.curriculum.phases) - 1
        self.referential_world = None
        if cfg.curriculum.enabled:
            g = torch.Generator(device=dev)
            g.manual_seed(cfg.train.seed + 13)
            self.referential_world = ReferentialWorld(cfg, device=str(dev), generator=g)
        self.bottleneck_rng = random.Random(cfg.train.seed + 5)
        self.eval_rng = random.Random(cfg.train.seed + 6)

        self.episode = 0
        self.history = History()
        self.train_success = RollingStat(window=4000)
        self.train_reward = RollingStat(window=4000)
        self.train_comprehension = RollingStat(window=4000)
        self.archive: list[dict[str, Any]] = []      # example transcripts across the run
        self.totals = {"apples_sold": 0, "value": 0.0, "profit": 0.0,
                       "trades": 0, "episodes": 0}
        self.failure_counts: dict[str, int] = {}
        self.chance = chance_success_rate(cfg, self.world)
        self.newborn_reports: list[dict[str, Any]] = []
        self._stop_requested = False
        self.resume_note = resume_note
        self._last_checkpoint_episode = -1
        self.progress_path = os.path.join(out_dir, "progress.json")
        self._last_progress = 0.0
        self._headline: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def curriculum_report(self) -> dict[str, Any]:
        """Where the run got to on the ladder, and what vocabulary came from where."""
        cur = self.curriculum
        names = [p.name for p in cur.phases]
        out: dict[str, Any] = {
            "enabled": self.cfg.curriculum.enabled,
            "phases": names,
            "reached": cur.phase.name,
            "reached_index": cur.phase.index,
            "episodes_in_current_phase": cur.episodes_in_phase,
            "stalled": cur.stalled,
            "transitions": cur.transitions,
            "last_promotion_check": cur.last_report,
            "provenance": self.provenance.summary(names),
        }
        reached = cur.phase.index
        for p in cur.phases[1:reached + 1]:
            out.setdefault("inherited", {})[p.name] = self.provenance.inherited(p.name)
            out.setdefault("new_words", {})[p.name] = self.provenance.new_in_phase(p.name)
        # what each newborn was actually shown, which is what decides transmission
        cov = [r.get("bottleneck", {}).get("word_coverage") or {}
               for r in self.newborn_reports]
        cov = [c for c in cov if c]
        if cov:
            def avg(k):
                vals = [c[k] for c in cov if k in c]
                return sum(vals) / len(vals) if vals else float("nan")
            out["bottleneck_coverage"] = {
                "births_measured": len(cov),
                "common_form_coverage": avg("common_coverage"),
                "rare_form_coverage": avg("rare_coverage"),
                "mean_sample_size": avg("vocabulary_shown"),
            }
        return out

    # ------------------------------------------------------------------
    def write_lineups(self, batch, rb, episode0: int) -> None:
        """One row per lineup round: what was shown, what was said, what was picked."""
        from .render import render_message
        w = self.cfg.world
        stride = max(1, self.cfg.log.ledger_stride)
        L = self.cfg.channel.max_symbols
        for i in range(len(batch)):
            ep = episode0 + i
            if ep % stride:
                continue

            def tup(t):
                return {"variety": w.variety_names[int(t[0])], "quantity": int(t[1]),
                        "quality": w.quality_names[int(t[2])]}

            cands = [tup(rb.meanings[i, k]) for k in range(rb.meanings.shape[1])]
            msg = [int(x) for x in batch.tokens[i, :L]]
            self.lineup_log.write({
                "episode": ep,
                "phase": "refer",
                "informer_id": self.pop.farmers[int(batch.f_idx[i])].agent_id,
                "guesser_id": self.pop.buyers[int(batch.b_idx[i])].agent_id,
                "true_meaning": tup(rb.true_meaning[i]),
                "candidates": cands,
                "target_index": int(rb.target[i]),
                "chosen_index": int(batch.b_dec[i, 8]),
                "correct": bool(batch.res["success"][i]),
                "msg_symbols": msg,
                "msg_text": render_message(self.cfg, msg),
            })

    # ------------------------------------------------------------------
    def phase_sampler(self, phase):
        """How this phase draws evaluation rounds -- lineups, or trades."""
        if phase.referential and self.referential_world is not None:
            return lambda n, held_out=False: self.referential_world.sample(n)
        if self.tensor_world is not None:
            return lambda n, held_out=False: self.tensor_world.sample(
                n, held_out=bool(held_out))
        return None

    def chance_for(self, phase) -> float:
        """The floor this phase has to clear."""
        if phase.referential:
            return 1.0 / max(2, self.cfg.curriculum.n_candidates)
        return self.chance

    def consider_promotion(self, row: dict[str, Any]) -> None:
        """Move up a rung only if this one demonstrably worked."""
        cur = self.curriculum
        if not self.cfg.curriculum.enabled or cur.finished:
            return
        phase = cur.phase
        rule = promotion_for(self.cfg, phase)
        comp = row.get("compositionality", {}) or {}
        abl = row.get("channel_ablation", {}) or {}
        buyer = comp.get("buyer", {}) if isinstance(comp.get("buyer"), dict) else {}
        passed, checks = rule.evaluate(
            success=row.get("eval_success", float("nan")),
            chance=self.chance_for(phase),
            topsim=comp.get("mean", float("nan")),
            null=buyer.get("null_mean", float("nan")),
            transfer=abl.get("information_transfer", float("nan")),
            episodes_in_phase=cur.episodes_in_phase)
        cur.last_report = {"phase": phase.name, "passed": passed, "checks": checks}

        L = self.log
        if passed:
            done_in = cur.episodes_in_phase
            nxt = cur.advance(self.episode, checks)
            cur.transitions[-1]["episodes_in_previous_phase"] = done_in
            L("")
            L("*** PHASE %s -> %s at episode %d ***" % (phase.name, nxt.name, self.episode))
            L("    %s" % nxt.blurb)
            L("    promoted because, after %s episodes in %s:" % ("{:,}".format(done_in), phase.name))
            for name, c in checks.items():
                L("      met: %-24s %s" % (name, c["detail"]))
            L("    the population carries its weights forward; nothing is reinitialised.")
            L("")
            return

        if cur.episodes_in_phase >= rule.max_episodes:
            unmet = [k for k, c in checks.items() if not c["met"]]
            if not cur.stalled:
                cur.stalled = True
                L("")
                L("!!! PHASE %s HAS STALLED at episode %d !!!" % (phase.name, self.episode))
                L("    %s episodes in this phase without meeting:"
                  % "{:,}".format(cur.episodes_in_phase))
                for k in unmet:
                    L("      unmet: %-24s %s" % (k, checks[k]["detail"]))
                L("    Not advancing. Building the next phase on top of a phase that")
                L("    never converged would only reproduce this failure one rung up.")
                if self.cfg.curriculum.on_stall == "stop":
                    L("    on_stall=stop: ending the run here.")
                L("")
            if self.cfg.curriculum.on_stall == "stop":
                self._stop_requested = True

    # ------------------------------------------------------------------
    def write_progress(self, state: str = "running") -> None:
        """A tiny status file, rewritten as the run proceeds.

        Checkpoints are thousands of episodes apart, which is far too coarse for a
        progress bar, so this is written every batch.  It is one small file and a
        few hundred bytes, which is nothing next to a batch of episodes.
        """
        now = time.time()
        if state == "running" and now - self._last_progress < 0.5:
            return
        self._last_progress = now
        elapsed = self.log.elapsed()
        total = max(1, self.cfg.train.episodes)
        rate = self.episode / elapsed if elapsed > 0 else 0.0
        remaining = (total - self.episode) / rate if rate > 0 else float("nan")
        payload = {
            "state": state,
            "episode": self.episode,
            "total_episodes": total,
            "fraction": min(1.0, self.episode / total),
            "elapsed_seconds": round(elapsed, 1),
            "episodes_per_second": round(rate, 1),
            "eta_seconds": (round(remaining, 1) if remaining == remaining else None),
            "day": self.economy.day,
            "births": len(self.pop.births),
            "generation": max((a.generation for a in self.pop.all_agents()), default=0),
            "rolling_success": round(self.train_success.mean, 4),
            "trades": self.totals["trades"],
            "apples_sold": self.totals["apples_sold"],
            "headline": self._headline,
        }
        try:
            tmp = self.progress_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.progress_path)
        except Exception:
            pass          # a status file must never be able to kill a run

    # ------------------------------------------------------------------
    def banner(self) -> None:
        c = self.cfg
        L = self.log
        L.rule("ORCHARD: emergent language in an apple-trading world")
        L("run name           : %s" % c.name)
        L("output directory   : %s" % os.path.abspath(self.out_dir))
        L("episodes           : %d  (batch %d -> %d updates)"
          % (c.train.episodes, c.train.batch_size,
             c.train.episodes // max(1, c.train.batch_size)))
        L("population         : %d farmers, %d buyers"
          % (c.population.n_farmers, c.population.n_buyers))
        L("turnover           : %s%s" % (
            "ON" if c.population.turnover else "OFF",
            "  (lifespan %d-%d episodes)" % (c.population.lifespan_min, c.population.lifespan_max)
            if c.population.turnover else ""))
        L("bottleneck         : %s%s" % (
            "ON" if c.bottleneck.enabled else "OFF",
            "  (%d samples, %d epochs)" % (c.bottleneck.n_samples, c.bottleneck.epochs)
            if c.bottleneck.enabled else ""))
        L("channel            : %d content tokens + <eos>, <= %d tokens/turn, %d turns"
          % (c.channel.vocab_size, c.channel.max_msg_len, c.channel.n_turns))
        L("world              : %d varieties, qty 1-%d, %d quality levels, %d price bins"
          % (c.world.n_varieties, c.world.max_qty, c.world.n_quality, c.world.n_price_bins))
        L("meaning space      : %d distinct private states per role"
          % (c.world.n_varieties * c.world.max_qty * c.world.n_quality * c.world.n_price_bins))
        L("economy            : %s, %d encounters/day, season = %d days"
          % ("persistent lots" if c.economy.persistent_inventory else "per-episode sampling",
             c.economy.episodes_per_day, c.economy.season_days))
        L("held-out combos    : %d (variety, quantity) pairs reserved for zero-shot"
          % len(self.world.holdout))
        L("algorithm          : %s" % (
            "straight-through Gumbel-softmax on message tokens + REINFORCE on the "
            "trade decision" if c.train.algo == "gumbel"
            else "REINFORCE with a learned value baseline"))
        L("agent brain        : %d-layer transformer, d=%d, %d params, RANDOMLY INITIALISED"
          % (c.model.n_layers, c.model.d_model,
             count_parameters(self.pop.farmers[0].net)))
        L("expected turnover  : ~%.1f generations per lineage" % expected_generations(c))
        from .hardware import describe
        L("hardware           : %s" % describe(self.torch_device, c))
        L("chance success rate: %.4f  (two uniformly random agents)" % self.chance)
        if self.resume_note:
            L(self.resume_note)
        L.rule()
        L("")

    # ------------------------------------------------------------------
    def on_birth(self, newborn: Agent, ev: BirthEvent) -> None:
        info = train_newborn(self.cfg, newborn, self.store, self.bottleneck_rng,
                             device=self.device)
        ev.bottleneck = info
        # Spec 5.5: test the newborn the moment it comes out of the bottleneck,
        # before it has played a single live episode.
        probe = newborn_vs_veterans(self.cfg, self.pop, self.world, newborn,
                                    self.cfg.log.intelligibility_episodes // 2,
                                    device=self.device, rng=self.eval_rng)
        rec = ev.to_dict()
        rec["at_birth_vs_veterans"] = probe
        self.birth_log.write(rec)
        self.newborn_reports.append(rec)
        self.log("  [birth] %s %s gen %d replaces agent %d (age %d, success %.3f)"
                 % (ROLE_NAMES[ev.role], newborn.name, ev.generation,
                    ev.replaced_agent_id, ev.replaced_age, ev.replaced_success_rate))
        if info.get("n_samples"):
            self.log("          bottleneck: %d transcripts from generations %s, "
                     "token acc %.3f, decision acc %.3f"
                     % (info["n_samples"], info["teacher_generations"],
                        info["token_accuracy"] or 0.0, info["decision_accuracy"] or 0.0))
        else:
            self.log("          bottleneck: %s" % info.get("skipped", "disabled"))
        sr = probe.get("success_rate")
        if sr == sr:
            self.log("          straight out of the bottleneck vs veterans: success %.3f" % sr)

    # ------------------------------------------------------------------
    def checkpoint(self, final: bool = False) -> dict[str, Any]:
        cfg, L = self.cfg, self.log
        # A checkpoint fired at exactly train.episodes would otherwise be repeated
        # by the final one, with nothing changed in between.  Re-measuring drift
        # against a snapshot taken seconds earlier guarantees 0.000, and that row
        # is the one the report reads.
        if final and self._last_checkpoint_episode == self.episode:
            if self.metrics_log.rows:
                self.metrics_log.rows[-1]["final"] = True
                return self.metrics_log.rows[-1]
        self._last_checkpoint_episode = self.episode
        t0 = time.time()
        phase = self.curriculum.phase
        sampler = self.phase_sampler(phase)
        ev = evaluate_success(cfg, self.pop, self.world,
                              max(200, cfg.log.intelligibility_episodes),
                              device=self.device, rng=self.eval_rng,
                              phase=phase, sampler=sampler)
        comp = compositionality(cfg, self.pop, self.world,
                                n_samples=cfg.log.topsim_samples,
                                device=self.device, rng=self.eval_rng)
        vocab = vocab_stats(cfg, [ev["batch"]])
        stab = self.stability.measure(self.pop, self.world, device=self.device)
        zs = zero_shot(cfg, self.pop, self.world, cfg.log.zeroshot_episodes,
                       device=self.device, rng=self.eval_rng)
        abl = channel_ablation(cfg, self.pop, self.world, cfg.log.ablation_episodes,
                               device=self.device, rng=self.eval_rng,
                               phase=phase, sampler=sampler)
        newborn_age = max(200, cfg.population.lifespan_min // 6)
        intel = intelligibility(cfg, self.pop, self.world,
                                cfg.log.intelligibility_episodes,
                                newborn_age=newborn_age, device=self.device,
                                rng=self.eval_rng)
        # ---- addendum section 3 ------------------------------------------
        words = word_stats(cfg, [ev["batch"]])
        lenfreq = length_frequency(cfg, self.pop, self.world, device=self.device)
        buckets = bucketed_analysis(cfg, self.pop, self.world, device=self.device,
                                    rng=self.eval_rng)
        forms = (self.forms.observe(self.pop, self.episode, device=self.device)
                 if self.forms is not None else {})

        comp_pop = self.pop.composition()
        flags = detect_degenerate(cfg, ev["success_rate"], self.chance, vocab, comp,
                                  self.episode, words=words, ablation=abl)

        row = {
            "episode": self.episode,
            "phase": phase.name,
            "phase_index": phase.index,
            "episodes_in_phase": self.curriculum.episodes_in_phase,
            "chance_for_phase": self.chance_for(phase),
            "day": self.economy.day,
            "season": self.economy.season,
            "wall_seconds": round(self.log.elapsed(), 1),
            "train_success_rolling": self.train_success.mean,
            "train_reward_rolling": self.train_reward.mean,
            "eval_success": ev["success_rate"],
            "eval_success_on_viable": ev["success_rate_on_viable"],
            "comprehension_rate": ev["comprehension_rate"],
            "comprehension_on_viable": ev["comprehension_on_viable"],
            "judgement_rate": ev["judgement_rate"],
            "farmer_reads_buyer": ev["farmer_reads_buyer"],
            "buyer_reads_farmer": ev["buyer_reads_farmer"],
            "farmer_variety_acc": ev["farmer_variety_acc"],
            "farmer_qty_acc": ev["farmer_qty_acc"],
            "farmer_price_acc": ev["farmer_price_acc"],
            "train_comprehension_rolling": self.train_comprehension.mean,
            "eval_reward": ev["mean_reward"],
            "chance_success": self.chance,
            "compositionality": {k: v for k, v in comp.items() if k != "per_agent"},
            "vocab": {k: v for k, v in vocab.items() if k != "token_counts"},
            "words": {k: v for k, v in words.items() if k != "word_counts"},
            "length_frequency": {k: v for k, v in lenfreq.items() if k != "rows"},
            "buckets": buckets,
            "form_survival": forms,
            "stability": stab,
            "zero_shot": zs,
            "channel_ablation": abl,
            "intelligibility": intel,
            "population": comp_pop,
            "economy": {"restocks": self.economy.restocks, "soldouts": self.economy.soldouts},
            "totals": dict(self.totals),
            "failure_modes": dict(sorted(self.failure_counts.items(),
                                         key=lambda kv: -kv[1])[:10]),
            "degenerate_flags": flags,
            "final": final,
        }
        self.provenance.observe(phase.name, self.episode, words.get("word_counts", {}))
        row["vocabulary_provenance"] = self.provenance.summary(
            [p.name for p in self.curriculum.phases])
        self.metrics_log.write(row)

        h = self.history
        h.episodes.append(self.episode)
        h.train_success.append(self.train_success.mean)
        h.eval_success.append(ev["success_rate"])
        h.eval_success_viable.append(ev["success_rate_on_viable"])
        h.comprehension.append(ev["comprehension_rate"])
        h.judgement.append(ev["judgement_rate"])
        h.variety_acc.append(ev["farmer_variety_acc"])
        h.farmer_reads.append(ev["farmer_reads_buyer"])
        h.buyer_reads.append(ev["buyer_reads_farmer"])
        h.farmer_reads_transfer.append(abl.get("farmer_reads_transfer", float("nan")))
        h.buyer_reads_transfer.append(abl.get("buyer_reads_transfer", float("nan")))
        h.qty_acc.append(ev["farmer_qty_acc"])
        h.variety_transfer.append(abl.get("variety_transfer", float("nan")))
        h.topsim.append(comp["mean"])
        h.topsim_null.append(comp["buyer"]["null_mean"])
        h.entropy.append(vocab["token_entropy_bits"])
        h.msg_len.append(vocab["mean_msg_len"])
        h.drift.append(stab["drift"])
        h.coherence.append((stab["coherence_buyer"] + stab["coherence_farmer"]) / 2)
        h.transmission.append(intel["transmission_ratio"])
        h.zeroshot.append(zs["retention"])
        h.ablation_drop.append(abl.get("comprehension_drop", float("nan")))
        h.information_transfer.append(abl.get("information_transfer", float("nan")))
        h.mean_symbols.append(words["mean_symbols_per_message"])
        h.mean_words.append(words["mean_words_per_message"])
        h.distinct_words.append(float(words["distinct_words"]))
        h.multi_atom_share.append(words["multi_atom_word_share"])
        h.rho_length_frequency.append(lenfreq.get("rho_symbols", float("nan")))
        h.topsim_frequent.append(buckets.get("frequent", {}).get("topsim", float("nan")))
        h.topsim_rare.append(buckets.get("rare", {}).get("topsim", float("nan")))
        h.drift_frequent.append(forms.get("drift_frequent_interval", float("nan")))
        h.drift_rare.append(forms.get("drift_rare_interval", float("nan")))
        h.reward.append(ev["mean_reward"])
        h.generations.append(comp_pop["farmers"]["mean_generation"])

        self._headline = {
            "episode": self.episode,
            "success": round(ev["success_rate"], 4),
            "comprehension": round(ev["comprehension_rate"], 4),
            "variety_naming": round(ev["farmer_variety_acc"], 4),
            "farmer_reads": round(ev["farmer_reads_buyer"], 4),
            "buyer_reads": round(ev["buyer_reads_farmer"], 4),
            "channel_transfer": (round(abl["variety_transfer"], 4)
                                 if isinstance(abl.get("variety_transfer"), float)
                                 and abl["variety_transfer"] == abl["variety_transfer"]
                                 else None),
            "topsim": (round(comp["mean"], 4) if comp["mean"] == comp["mean"] else None),
            "distinct_words": words["distinct_words"],
            "symbols_per_utterance": round(words["mean_symbols_per_message"], 2),
            "flags": flags,
        }
        self.write_progress()
        self.print_summary(row, ev, comp, vocab, stab, zs, intel, comp_pop, flags,
                           time.time() - t0, abl, words, lenfreq, buckets, forms)
        self.archive_examples(ev["batch"])
        self.consider_promotion(row)
        return row

    # ------------------------------------------------------------------
    def print_summary(self, row, ev, comp, vocab, stab, zs, intel, comp_pop, flags,
                      secs, abl=None, words=None, lenfreq=None, buckets=None,
                      forms=None) -> None:
        L = self.log
        cfg = self.cfg
        pct = 100.0 * self.episode / max(1, cfg.train.episodes)
        L("")
        L.rule("CHECKPOINT  episode %d / %d  (%.1f%%)  phase %s"
               % (self.episode, cfg.train.episodes, pct, self.curriculum.phase.name))
        L("phase %d of %d: %s  (%s episodes in this phase)"
          % (self.curriculum.phase.index + 1, len(self.curriculum.phases),
             self.curriculum.phase.blurb,
             "{:,}".format(self.curriculum.episodes_in_phase)))
        L("elapsed %.1f min   (metrics took %.1fs)" % (self.log.elapsed() / 60.0, secs))

        L("")
        L("TRADE PERFORMANCE")
        L("  success rate      : %.3f eval   %.3f rolling-train   (chance %.4f)"
          % (ev["success_rate"], self.train_success.mean,
             self.chance_for(self.curriculum.phase)))
        L("  on viable deals   : %.3f   (%.0f%% of encounters are viable)"
          % (ev["success_rate_on_viable"], 100 * ev["viable_frac"]))
        L("  reading each other : farmer reads buyer %.3f | buyer reads farmer %.3f"
          % (ev["farmer_reads_buyer"], ev["buyer_reads_farmer"]))
        L("  reference accuracy: variety %.3f, quantity %.3f, price-in-range %.3f"
          % (ev["farmer_variety_acc"], ev["farmer_qty_acc"], ev["farmer_price_acc"]))
        L("                      (can the farmer name what the buyer asked for? "
          "chance is %.3f / %.3f)"
          % (1.0 / cfg.world.n_varieties, 1.0 / cfg.world.max_qty))
        L("  comprehension     : %.3f  (beliefs matched and described an executable "
          "deal, whether or not they traded)" % ev["comprehension_rate"])
        L("  viability judged  : %.3f  (both sides correctly decided whether a deal "
          "was possible)" % ev["judgement_rate"])
        L("  mean reward       : %.3f" % ev["mean_reward"])
        if len(self.history.eval_success) > 1:
            prev = self.history.eval_success[-2]
            L("  trend             : %+.3f since last checkpoint" % (ev["success_rate"] - prev))
        if self.failure_counts:
            top = sorted(self.failure_counts.items(), key=lambda kv: -kv[1])[:4]
            tot = sum(self.failure_counts.values()) or 1
            L("  top failure modes : " + ", ".join("%s %.0f%%" % (k, 100 * v / tot)
                                                   for k, v in top))

        L("")
        L("LANGUAGE")
        L("  compositionality  : topsim %.3f  (buyer %.3f / farmer %.3f, shuffled null %.3f)"
          % (comp["mean"], comp["buyer"]["mean"], comp["farmer"]["mean"],
             comp["buyer"]["null_mean"]))
        L("  vocabulary        : %d/%d tokens used, entropy %.2f bits (%.0f%% of max)"
          % (vocab["tokens_used"], vocab["vocab_size"], vocab["token_entropy_bits"],
             100 * vocab["token_entropy_norm"]))
        L("  message length    : %.2f of %d allowed, %d distinct messages, %.0f%% silent"
          % (vocab["mean_msg_len"], vocab["max_msg_len"], vocab["distinct_messages"],
             100 * vocab["silent_frac"]))
        L("  most used tokens  : " + ", ".join(
            "tok%d %.0f%%" % (t["token"], 100 * t["share"]) for t in vocab["top_tokens"][:6]))
        L("  stability         : drift %.3f, %.0f%% of probe messages unchanged"
          % (stab["drift"] if stab["drift"] == stab["drift"] else float("nan"),
             100 * (stab["identical_frac"] if stab["identical_frac"] == stab["identical_frac"] else 0)))
        L("  shared code       : coherence buyer %.3f / farmer %.3f  (1 = all agents say "
          "the same thing for the same meaning)"
          % (stab["coherence_buyer"], stab["coherence_farmer"]))

        if abl and abl.get("n"):
            L("")
            L("IS THE CHANNEL ACTUALLY CARRYING INFORMATION?  (scrambled-channel control)")
            L("  comprehension     : %.3f intact | %.3f scrambled | %.3f muted"
              % (abl["intact_comprehension"], abl["scrambled_comprehension"],
                 abl.get("muted_comprehension", float("nan"))))
            L("  success           : %.3f intact | %.3f scrambled | %.3f muted"
              % (abl["intact_success"], abl["scrambled_success"],
                 abl.get("muted_success", float("nan"))))
            L("  variety naming    : %.3f intact | %.3f scrambled | %.3f muted"
              % (abl.get("intact_variety_acc", float("nan")),
                 abl.get("scrambled_variety_acc", float("nan")),
                 abl.get("muted_variety_acc", float("nan"))))
            L("                      %.0f%% of the headroom above silence, of which "
              "%.0f%% needs the actual symbols (the rest is utterance length)"
              % (100 * (abl.get("variety_transfer") or 0.0),
                 100 * (abl.get("variety_transfer_content") or 0.0)))
            L("  quantity naming   : %.3f intact | %.3f scrambled | %.3f muted "
              "(%.0f%% of headroom)"
              % (abl.get("intact_qty_acc", float("nan")),
                 abl.get("scrambled_qty_acc", float("nan")),
                 abl.get("muted_qty_acc", float("nan")),
                 100 * (abl.get("qty_transfer") or 0.0)))
            L("  farmer reads buyer: %.3f intact | %.3f muted  (%.0f%% of headroom)"
              % (abl.get("intact_farmer_reads", float("nan")),
                 abl.get("muted_farmer_reads", float("nan")),
                 100 * (abl.get("farmer_reads_transfer") or 0.0)))
            L("  buyer reads farmer: %.3f intact | %.3f muted  (%.0f%% of headroom)"
              % (abl.get("intact_buyer_reads", float("nan")),
                 abl.get("muted_buyer_reads", float("nan")),
                 100 * (abl.get("buyer_reads_transfer") or 0.0)))
            L("  information transfer: %.3f of the available headroom on full "
              "comprehension" % abl.get("information_transfer", float("nan")))
            L("  viability judged  : %.3f intact -> %.3f scrambled (drop %.3f)"
              % (abl["intact_judgement"], abl["scrambled_judgement"], abl["judgement_drop"]))

        if words:
            L("")
            L("VOCABULARY")
            L("  words             : %d distinct, entropy %.2f bits, %.2f words per "
              "utterance" % (words["distinct_words"], words["word_entropy_bits"],
                             words["mean_words_per_message"]))
            L("  word shape        : %.2f atoms per word (longest %d), %.0f%% of words "
              "are multi-atom compounds"
              % (words["mean_word_len_atoms"], words["max_word_len_atoms"],
                 100 * words["multi_atom_word_share"]))
            L("  utterance length  : %.2f of %d symbols allowed, %.0f%% run to the cap, "
              "%.0f%% silent"
              % (words["mean_symbols_per_message"], words["max_symbols_allowed"],
                 100 * words["at_length_cap_frac"], 100 * words["silent_frac"]))
            L("  structure marks   : hyphen %.0f%% of symbols, space %.0f%%"
              % (100 * words["hyphen_share"], 100 * words["space_share"]))
            if words["top_words"]:
                L("  commonest words   : " + ", ".join(
                    "%s %.0f%%" % (w["word"], 100 * w["share"])
                    for w in words["top_words"][:6]))
        if lenfreq and lenfreq.get("n"):
            L("  length vs frequency: rho %.3f in symbols, %.3f in words   "
              "(negative = commoner meanings get shorter forms)"
              % (lenfreq.get("rho_symbols", float("nan")),
                 lenfreq.get("rho_words", float("nan"))))
            L("                      commonest third %.2f symbols, rarest third %.2f"
              % (lenfreq.get("mean_symbols_frequent", float("nan")),
                 lenfreq.get("mean_symbols_rare", float("nan"))))
        if buckets and "frequent" in buckets and "rare" in buckets:
            f, r = buckets["frequent"], buckets["rare"]
            L("  frequent meanings : topsim %.3f, coherence %.3f, %.2f symbols"
              % (f.get("topsim", float("nan")), f.get("coherence", float("nan")),
                 f.get("mean_symbols", float("nan"))))
            L("  rare meanings     : topsim %.3f, coherence %.3f, %.2f symbols"
              % (r.get("topsim", float("nan")), r.get("coherence", float("nan")),
                 r.get("mean_symbols", float("nan"))))
        if forms and forms.get("drift_rare") == forms.get("drift_rare"):
            L("  form drift        : frequent %.3f vs rare %.3f   (%d replacements so "
              "far, %d rebuilt from commoner parts)"
              % (forms.get("drift_frequent", float("nan")),
                 forms.get("drift_rare", float("nan")),
                 forms.get("n_events", 0), forms.get("n_regularised", 0)))

        L("")
        L("GENERALISATION")
        L("  zero-shot         : seen %.3f -> unseen %.3f  (retention %.2f, %d held-out combos)"
          % (zs["seen_success"], zs["unseen_success"], zs["retention"], zs["n_holdout_combos"]))
        L("  cross-generation  : veteran-veteran %.3f, newcomer-mixed %.3f (ratio %.2f)"
          % (intel["veteran_veteran"], intel["newcomer_mixed"], intel["transmission_ratio"]))

        L("")
        L("POPULATION")
        for label, d in (("farmers", comp_pop["farmers"]), ("buyers", comp_pop["buyers"])):
            L("  %-8s n=%d  mean age %.0f  generations %s"
              % (label, d["n"], d["mean_age"], d["generations"]))
        L("  births %d, deaths %d" % (comp_pop["total_births"], comp_pop["total_deaths"]))
        L("  market: %d apples sold, value %.1f, farmer profit %.1f, %d restocks, %d sell-outs"
          % (self.totals["apples_sold"], self.totals["value"], self.totals["profit"],
             self.economy.restocks, self.economy.soldouts))

        if flags:
            L("")
            L("  !! DEGENERATE-OUTCOME WARNINGS !!")
            for f in flags:
                L("     - " + f)

        L("")
        L("EXAMPLE TRANSCRIPTS FROM THIS PERIOD")
        self.print_examples(ev["batch"])
        L.rule()
        L("")

    def _pick_examples(self, batch, n: int) -> list[int]:
        succ = [i for i, o in enumerate(batch.outcomes) if o.success]
        fail = [i for i, o in enumerate(batch.outcomes) if not o.success]
        picks: list[int] = []
        for pool in (succ, fail):
            for i in pool[:max(1, n // 2 + 1)]:
                if len(picks) < n:
                    picks.append(i)
        return picks

    def print_examples(self, batch) -> None:
        for i in self._pick_examples(batch, self.cfg.log.n_example_transcripts):
            self.log(render_transcript(self.cfg, batch.transcript(i), indent="    "))
            self.log("")

    def archive_examples(self, batch) -> None:
        for i in self._pick_examples(batch, 2):
            self.archive.append({
                "episode": self.episode,
                "rendered": render_transcript(self.cfg, batch.transcript(i), indent="    "),
                "success": batch.outcomes[i].success,
            })

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg, L = self.cfg, self.log
        self.banner()
        B = cfg.train.batch_size
        next_ckpt = cfg.log.checkpoint_every

        while self.episode < cfg.train.episodes and not self._stop_requested:
            n = min(B, cfg.train.episodes - self.episode)
            phase = self.curriculum.phase
            f_idx, b_idx = self.pop.pair(n, device=self.device)
            if phase.referential:
                # The lineup game: no market, no stock, no price -- just meanings.
                scen = self.referential_world.sample(n)
            elif not phase.use_market:
                # Price and budget, but scenarios drawn fresh rather than held as
                # depleting inventory; the economy is the last thing introduced.
                scen = self.tensor_world.sample(n)
            elif self.tensor_world is not None:
                scen = self.economy.make_batch_tensor(
                    n, len(self.pop.farmers), len(self.pop.buyers),
                    self.tensor_world, f_idx, b_idx)
            else:
                scen, f_idx, b_idx = self.economy.make_batch(
                    n, len(self.pop.farmers), len(self.pop.buyers))
            frac = self.episode / max(1, cfg.train.episodes)
            if cfg.train.algo == "gumbel":
                from .gumbel import run_and_update_gumbel
                batch, _ = run_and_update_gumbel(
                    cfg, scen, self.pop.farmers, self.pop.buyers, f_idx, b_idx,
                    frac_done=frac, device=self.device, phase=phase)
            else:
                from .rollout import run_episodes
                batch = run_episodes(cfg, scen, self.pop.farmers, self.pop.buyers,
                                     f_idx, b_idx, device=self.device, phase=phase)

            self.pop.record_episode_participation(f_idx, b_idx, batch)
            if phase.use_market and batch.res is not None:
                settle = self.economy.settle_tensor(f_idx, batch.res)
            elif phase.use_market:
                settle = self.economy.settle(f_idx, batch.outcomes)
            else:
                settle = {"apples_sold": 0, "value": 0.0, "profit": 0.0, "soldout": 0}
            self.totals["apples_sold"] += settle["apples_sold"]
            self.totals["value"] += settle["value"]
            self.totals["profit"] += settle["profit"]
            self.totals["episodes"] += n
            if batch.res is not None and not phase.referential:
                from .batched import failure_modes
                for mode in failure_modes(batch.res):
                    self.failure_counts[mode] = self.failure_counts.get(mode, 0) + 1
                succ = batch.success_t
                self.totals["trades"] += int(succ.sum())
                self.train_success.extend(succ.float().tolist())
                self.train_comprehension.extend(batch.comprehended_t.float().tolist())
            elif batch.res is not None:
                # The lineup game has one outcome that matters: did the guess land.
                succ = batch.success_t
                hits = int(succ.sum())
                self.failure_counts["lineup_hit"] = (
                    self.failure_counts.get("lineup_hit", 0) + hits)
                self.failure_counts["lineup_miss"] = (
                    self.failure_counts.get("lineup_miss", 0) + (len(batch) - hits))
                self.train_success.extend(succ.float().tolist())
                self.train_comprehension.extend(succ.float().tolist())
            else:
                for o in batch.outcomes:
                    self.failure_counts[o.failure_mode] = (
                        self.failure_counts.get(o.failure_mode, 0) + 1)
                    self.totals["trades"] += int(o.success)
                self.train_success.extend(float(o.success) for o in batch.outcomes)
                self.train_comprehension.extend(
                    float(o.comprehended) for o in batch.outcomes)
            self.train_reward.extend(
                (o.farmer_reward + o.buyer_reward) / 2 for o in batch.outcomes)

            if phase.referential:
                self.write_lineups(batch, scen, self.episode)
            else:
                self.ledger.write_batch(batch, self.pop, self.episode,
                                        self.economy.season)
            self.store.add_batch(batch, self.pop.farmers, self.pop.buyers, self.episode)

            if cfg.train.algo != "gumbel":
                from .rollout import update_agents
                update_agents(cfg, batch, self.pop.farmers, self.pop.buyers,
                              frac_done=frac, device=self.device)

            self.episode += n
            self.curriculum.episodes_in_phase += n
            self.pop.turn_over(self.episode, on_birth=self.on_birth)
            self.write_progress()

            if self.episode >= next_ckpt:
                row = self.checkpoint()
                self.maybe_plot()
                self.write_interim_report(row)
                next_ckpt += cfg.log.checkpoint_every

        final = self.checkpoint(final=True)
        self.maybe_plot()
        self.ledger.flush()
        self.write_progress(state="analysing")
        return final

    def write_interim_report(self, row: dict[str, Any]) -> None:
        """Rewrite report.md at every checkpoint.

        A long run that is interrupted, or that a human wants to read while it is
        still going, should not be left with nothing but raw JSONL.  The final
        report overwrites this with the full-run version.
        """
        try:
            from .metrics import analyse_token_semantics, evaluate_success, vocab_stats
            from .report import write_report
            sem = analyse_token_semantics(
                self.cfg, self.pop, self.world,
                n_samples=max(200, self.cfg.log.topsim_samples),
                device=self.device, rng=self.eval_rng)
            ev = evaluate_success(self.cfg, self.pop, self.world, 400,
                                  device=self.device, rng=self.eval_rng)
            row = dict(row)
            row["token_counts"] = vocab_stats(self.cfg, [ev["batch"]])["token_counts"]
            row["word_counts"] = word_stats(self.cfg, [ev["batch"]])["word_counts"]
            row["length_frequency_rows"] = length_frequency(
                self.cfg, self.pop, self.world, device=self.device).get("rows", [])[:20]
            if self.forms is not None:
                row["form_events"] = self.forms.report_rows()
                row["form_timeline"] = self.forms.timeline()
            row["curriculum"] = self.curriculum_report()
            write_report(self.cfg, self.out_dir, final=row, chance=self.chance, sem=sem,
                         archive=self.archive, totals=self.totals,
                         history=self.history.to_dict(),
                         newborn_reports=self.newborn_reports,
                         ledger_path=os.path.join(self.out_dir, "trades.jsonl"),
                         wall_minutes=self.log.elapsed() / 60.0)
            with open(os.path.join(self.out_dir, "token_semantics.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(sem.to_dict(), fh, indent=1)
            with open(os.path.join(self.out_dir, "history.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(self.history.to_dict(), fh, indent=1)
        except Exception as exc:          # never let reporting kill a run
            self.log("  [report] interim write skipped: %s" % exc)

    def maybe_plot(self) -> None:
        if not self.cfg.log.plot:
            return
        try:
            from .plots import write_plots
            write_plots(self.history, os.path.join(self.out_dir, "plots"), self.chance)
        except Exception as exc:               # plotting must never kill a run
            self.log("  [plot] skipped: %s" % exc)

    def close(self) -> None:
        self.write_progress(state="finished")
        self.ledger.close()
        self.metrics_log.close()
        self.birth_log.close()
        self.lineup_log.close()
        with open(os.path.join(self.out_dir, "history.json"), "w", encoding="utf-8") as fh:
            json.dump(self.history.to_dict(), fh, indent=1)
        self.log.close()
