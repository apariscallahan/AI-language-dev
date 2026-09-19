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
from .conventions import PopulationUsage
from .curriculum import (CurriculumState, ReferentialWorld, evaluate_rung, ladder,
                         promotion_for, rung_budget)
from .lexicon import (FormTracker, WordProvenance, bucketed_analysis,
                      cross_role_overlap, length_frequency, live_encoding, word_stats)
from .metrics import phase_evidence
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
    mean_life = (cfg.population.lifespan_min + cfg.population.lifespan_max) / 2.0
    if cfg.population.lifespan_unit == "updates":
        # every agent takes part in (nearly) every update
        return (cfg.train.episodes / max(1, cfg.train.batch_size)) / max(1.0, mean_life)
    per_agent = cfg.train.episodes / max(1, cfg.population.n_farmers)
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
    # per-role and cross-role series
    positional_farmer: list[float] = field(default_factory=list)
    positional_buyer: list[float] = field(default_factory=list)
    coherence_cross: list[float] = field(default_factory=list)
    cross_role_overlap: list[float] = field(default_factory=list)
    mean_word_len: list[float] = field(default_factory=list)
    at_length_cap: list[float] = field(default_factory=list)
    phase_index: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) for k, v in self.__dict__.items()}


class Trainer:
    def __init__(self, cfg: Config, out_dir: str, *, quiet: bool = False,
                 resume_note: str = "", started_utc: Optional[str] = None):
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
        # every promotion check, passed or not, with the criteria it applied
        self.promotion_log = JsonlLog(out_dir, "promotions.jsonl")

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
        # what the population has recently been saying: drives the coining cost
        # and the convention bonus
        self.usage = PopulationUsage(cfg)
        # How hard the speaker pressures bite, 0..1. A language has to exist
        # before it can be economised: charged from the first episode, even the
        # old 0.012 per symbol drove the lineup's describer towards silence long
        # before the lineup had taken off (ladder2 took off at ~80k episodes).
        # So the costs are off for the whole first rung and fully on from its
        # promotion onwards -- including haggle, where the earlier vocabulary
        # exploded. (A version that ramped them up with the first rung's success
        # capped it: with lineups that need every field named, success stalled at
        # ~0.42 with the costs 66% on, where a run with them ~off passed at 0.62.)
        self.cost_gate = 0.0 if cfg.curriculum.enabled else 1.0
        self.rung_success = RollingStat(window=2000)
        self._batch_no = 0
        self._next_check = cfg.curriculum.check_every
        self._next_grow: Optional[int] = None
        self.community_log: list[dict[str, Any]] = []

        # ---- the curriculum -------------------------------------------------
        self.curriculum = CurriculumState(ladder(cfg))
        if not cfg.curriculum.enabled:
            self.curriculum.index = len(self.curriculum.phases) - 1
        elif cfg.curriculum.start_phase:
            names = [p.name for p in self.curriculum.phases]
            self.curriculum.index = names.index(cfg.curriculum.start_phase)
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
        self.started_utc = started_utc or time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                                        time.gmtime())
        from .transcripts import TranscriptWriter
        self.transcripts = TranscriptWriter(cfg, out_dir)
        self._beat = (time.time(), 0)
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
            "stop_report": cur.stop_report,
            "community_growth": self.community_log,
            "budgets": {p.name: list(rung_budget(self.cfg, p)) for p in cur.phases},
            "transitions": cur.transitions,
            "last_promotion_check": cur.last_report,
            "promotion_checks_run": cur.checks_run,
            "provenance": self.provenance.summary(names),
        }
        reached = cur.phase.index
        for p in cur.phases[1:reached + 1]:
            out.setdefault("inherited", {})[p.name] = self.provenance.inherited(p.name)
            out.setdefault("new_words", {})[p.name] = self.provenance.new_in_phase(p.name)
        # what each newborn was actually shown, which is what decides transmission
        bns = [r.get("bottleneck", {}) or {} for r in self.newborn_reports]
        cov = [b.get("word_coverage") or {} for b in bns]
        cov = [c for c in cov if c]
        if cov:
            def avg(k):
                vals = [c[k] for c in cov if k in c]
                return sum(vals) / len(vals) if vals else float("nan")
            sizes = [b.get("n_samples", 0) for b in bns if b.get("n_samples")]
            stores = [b.get("store_size", 0) for b in bns if b.get("n_samples")]
            shares = [b.get("store_coverage", 0.0) for b in bns if b.get("n_samples")]
            out["bottleneck_coverage"] = {
                "births_measured": len(cov),
                "common_form_coverage": avg("common_coverage"),
                "rare_form_coverage": avg("rare_coverage"),
                "vocabulary_shown": avg("vocabulary_shown"),
                "vocabulary_in_population": avg("vocabulary_in_population"),
                "mean_transcripts_shown": (sum(sizes) / len(sizes)) if sizes else 0.0,
                "mean_store_size": (sum(stores) / len(stores)) if stores else 0.0,
                "mean_share_of_store": (sum(shares) / len(shares)) if shares else 0.0,
                "fixed_cap": self.cfg.bottleneck.n_samples,
                "coverage_setting": self.cfg.bottleneck.coverage,
                "max_samples": self.cfg.bottleneck.max_samples,
            }
        # newborn token accuracy, by role -- the farmer-side bottleneck check
        by_role: dict[str, list[float]] = {"farmer": [], "buyer": []}
        silent: dict[str, int] = {"farmer": 0, "buyer": 0}
        for r in self.newborn_reports:
            b = r.get("bottleneck", {}) or {}
            lbl = r.get("role", "?")
            if lbl not in by_role:
                continue
            if b.get("token_accuracy") is not None:
                by_role[lbl].append(float(b["token_accuracy"]))
            elif b.get("n_samples"):
                silent[lbl] += 1
        out["newborn_token_accuracy"] = {
            k: {"births": len(v), "mean": (sum(v) / len(v)) if v else None,
                "min": min(v) if v else None, "max": max(v) if v else None,
                "births_with_nothing_to_say": silent[k]}
            for k, v in by_role.items()}
        return out

    # ------------------------------------------------------------------
    def write_lineups(self, batch, rb, episode0: int, phase) -> None:
        """One row per lineup / mutual round: what was shown, said and picked."""
        from .curriculum import H_BELIEF, H_CHOICE, MutualBatch
        from .env import BUYER as _B, FARMER as _F
        from .render import render_message
        w = self.cfg.world
        stride = max(1, self.cfg.log.ledger_stride)
        L = self.cfg.channel.max_symbols

        def tup(t):
            return {"variety": w.variety_names[int(t[0])], "quantity": int(t[1]),
                    "quality": w.quality_names[int(t[2])]}

        f_ids = [self.pop.farmers[int(x)].agent_id for x in batch.f_idx.tolist()]
        b_ids = [self.pop.buyers[int(x)].agent_id for x in batch.b_idx.tolist()]
        for i in range(len(batch)):
            ep = episode0 + i
            if ep % stride:
                continue
            if phase.order:
                msg = [int(x) for x in batch.tokens[i, :L]]
                self.lineup_log.write({
                    "episode": ep, "phase": phase.name,
                    "farmer_id": f_ids[i], "buyer_id": b_ids[i],
                    "want_variety": w.variety_names[int(rb.want_variety[i])],
                    "need_qty": int(rb.need_qty[i]),
                    "filled_variety": w.variety_names[int(batch.f_dec[i, 1])],
                    "filled_qty": int(batch.f_dec[i, 2]),
                    "correct": bool(batch.res["success"][i]),
                    "msg_symbols": msg,
                    "msg_text": render_message(self.cfg, msg),
                })
                continue
            if isinstance(rb, MutualBatch):
                msgs = [[int(x) for x in batch.tokens[i, t * L:(t + 1) * L]]
                        for t in range(phase.n_turns)]
                rep = list(H_BELIEF[:3])
                self.lineup_log.write({
                    "episode": ep, "phase": phase.name,
                    "farmer_id": f_ids[i], "buyer_id": b_ids[i],
                    "farmer_meaning": tup(rb.f_meaning[i]),
                    "buyer_meaning": tup(rb.b_meaning[i]),
                    "farmer_report": [int(x) for x in batch.f_dec[i, rep]],
                    "buyer_report": [int(x) for x in batch.b_dec[i, rep]],
                    "farmer_ok": bool(batch.res["farmer_report_ok"][i]),
                    "buyer_ok": bool(batch.res["buyer_report_ok"][i]),
                    "correct": bool(batch.res["success"][i]),
                    "msg_symbols": msgs,
                    "msg_text": " | ".join(render_message(self.cfg, m) for m in msgs),
                })
                continue
            cands = [tup(rb.meanings[i, k]) for k in range(rb.meanings.shape[1])]
            msg = [int(x) for x in batch.tokens[i, :L]]
            guess = batch.b_dec if phase.guesser == _B else batch.f_dec
            self.lineup_log.write({
                "episode": ep,
                "phase": phase.name,
                "informer": "farmer" if phase.informer == _F else "buyer",
                "informer_id": f_ids[i] if phase.informer == _F else b_ids[i],
                "guesser_id": b_ids[i] if phase.informer == _F else f_ids[i],
                "true_meaning": tup(rb.true_meaning[i]),
                "candidates": cands,
                "target_index": int(rb.target[i]),
                "chosen_index": int(guess[i, H_CHOICE]),
                "correct": bool(batch.res["success"][i]),
                "msg_symbols": msg,
                "msg_text": render_message(self.cfg, msg),
            })

    # ------------------------------------------------------------------
    def phase_sampler(self, phase):
        """How one view of a phase draws rounds -- lineups, mutual pairs, or trades."""
        rw = self.referential_world
        if phase.referential and rw is not None:
            return lambda n, held_out=False: rw.sample(n, informer=phase.informer,
                                                       held_out=bool(held_out))
        if phase.mutual and rw is not None:
            return lambda n, held_out=False: rw.sample_mutual(n, held_out=bool(held_out))
        if self.tensor_world is not None:
            return lambda n, held_out=False: self.tensor_world.sample(
                n, held_out=bool(held_out))
        return None

    def _context_consistency(self, phase) -> dict[str, Any]:
        """Only meaningful once buyers have spoken in a trade context too."""
        if phase.tuples:
            return {"n": 0, "consistency": float("nan"),
                    "note": "buyers have only spoken in lineups so far"}
        try:
            from .metrics import context_consistency
            return context_consistency(self.cfg, self.pop, device=self.device)
        except Exception as exc:
            return {"n": 0, "consistency": float("nan"), "error": str(exc)}

    def language_properties(self, sem=None) -> list[dict[str, Any]]:
        from .properties import scorecard
        return scorecard(self.cfg, self.metrics_log.rows, self.curriculum_report(), sem)

    def zero_shot_sampler(self, phase):
        """Seen versus held-out rounds, drawn the *same* way.

        Held-out lineups use independent distractors (a near-miss cluster built
        around a held-out target would give the target away), so the seen rounds
        they are compared with must use independent distractors too.
        """
        rw = self.referential_world
        if phase.referential and rw is not None:
            return lambda n, held_out=False: rw.sample(n, informer=phase.informer,
                                                       held_out=bool(held_out),
                                                       hard_frac=0.0)
        return self.phase_sampler(phase)

    def chance_for(self, phase) -> float:
        """The floor this phase has to clear (NaN: measured, not analytic)."""
        if phase.referential:
            return 1.0 / max(2, self.cfg.curriculum.n_candidates)
        if phase.mutual or phase.order:
            return float("nan")           # measured against a muted channel instead
        return self.chance

    def gather_evidence(self, phase, *, light: bool) -> dict[str, Any]:
        lg = self.cfg.log
        n_eval = lg.ablation_episodes // (2 if light else 1)
        return phase_evidence(
            self.cfg, self.pop, self.world, phase, sampler_for=self.phase_sampler,
            n_eval=max(200, n_eval), n_topsim=max(60, lg.topsim_samples // (2 if light else 1)),
            n_semantics=max(200, lg.topsim_samples * 2), chance=self.chance_for(phase),
            device=self.device, rng=self.eval_rng)

    def consider_promotion(self, evidence: dict[str, Any], source: str) -> None:
        """Move up a rung only if this one demonstrably worked -- judged per role
        where the rung is about both roles."""
        cur = self.curriculum
        if not self.cfg.curriculum.enabled or cur.finished:
            return
        phase = cur.phase
        lo, hi = rung_budget(self.cfg, phase)
        passed, checks = evaluate_rung(self.cfg, phase, evidence, cur.episodes_in_phase)
        if cur.index > 0 and not self.pop.full_size:
            # every rung after the first is judged on the whole community
            p = self.cfg.population
            checks["community at full size"] = {
                "met": False,
                "detail": "%d of %d farmers, %d of %d buyers" % (
                    len(self.pop.farmers), p.n_farmers, len(self.pop.buyers), p.n_buyers)}
            passed = False
        cur.checks_run += 1
        cur.last_report = {"phase": phase.name, "passed": passed, "checks": checks,
                           "episode": self.episode, "source": source}
        slim = {k: v for k, v in evidence.items() if not k.startswith("_")}
        for sp in slim.get("speakers", {}).values():
            sp.pop("positional_rows", None)
        self.promotion_log.write({"episode": self.episode, "phase": phase.name,
                                  "episodes_in_phase": cur.episodes_in_phase,
                                  "source": source, "passed": passed,
                                  "checks": checks, "evidence": slim})

        L = self.log
        if passed:
            done_in = cur.episodes_in_phase
            nxt = cur.advance(self.episode, checks)
            self.rung_success = RollingStat(window=2000)
            self.cost_gate = 1.0
            cur.transitions[-1]["episodes_in_previous_phase"] = done_in
            try:
                cur.transitions[-1]["snapshot"] = self.save_snapshot("after-" + phase.name)
            except Exception as exc:          # a snapshot must never kill a run
                self.log("  [snapshot] skipped: %s" % exc)
            cur.transitions[-1]["evidence"] = slim
            A = L.always
            A("")
            A("*** PHASE %s -> %s at episode %d ***" % (phase.name, nxt.name, self.episode))
            A("    %s" % nxt.blurb)
            A("    promoted because, after %s episodes in %s (budget %s-%s):"
              % ("{:,}".format(done_in), phase.name, "{:,}".format(lo), "{:,}".format(hi)))
            for name, c in checks.items():
                A("      met: %-40s %s" % (name, c["detail"]))
            A("    the population carries its weights forward; nothing is reinitialised.")
            A("")
            return

        if cur.episodes_in_phase >= hi:
            unmet = [k for k, c in checks.items() if not c["met"]]
            if not cur.stalled:
                cur.stalled = True
                cur.stop_report = {
                    "phase": phase.name, "episode": self.episode,
                    "episodes_in_phase": cur.episodes_in_phase, "max_episodes": hi,
                    "unmet": {k: checks[k]["detail"] for k in unmet},
                    "met": {k: c["detail"] for k, c in checks.items() if c["met"]},
                    "action": self.cfg.curriculum.on_stall,
                }
                A = L.always
                A("")
                A("!!! RUNG %s EXCEEDED ITS BUDGET at episode %d !!!" % (phase.name, self.episode))
                A("    %s episodes in this rung (max %s) without meeting:"
                  % ("{:,}".format(cur.episodes_in_phase), "{:,}".format(hi)))
                for k in unmet:
                    A("      unmet: %-40s %s" % (k, checks[k]["detail"]))
                for k, c in checks.items():
                    if c["met"]:
                        A("      met:   %-40s %s" % (k, c["detail"]))
                A("    Not advancing. Building the next rung on top of one that")
                A("    never converged would only reproduce this failure one rung up.")
                if self.cfg.curriculum.on_stall == "stop":
                    A("    on_stall=stop: ending the run here and writing the report.")
                A("")
            if self.cfg.curriculum.on_stall == "stop":
                self._stop_requested = True

    # ------------------------------------------------------------------
    # snapshots: the whole population and everything it has been saying
    # ------------------------------------------------------------------
    def save_snapshot(self, tag: str) -> str:
        """Write everything needed to carry on from here.

        Taken at every promotion (``after-<rung>.pt``) and at every checkpoint
        (``latest.pt``). A later rung can then be iterated on without replaying
        the ones below it, and a pre-empted cloud machine loses at most one
        checkpoint interval.
        """
        from dataclasses import asdict

        def agent_state(a):
            return {"agent_id": a.agent_id, "role": a.role, "slot": a.slot,
                    "generation": a.generation, "birth_episode": a.birth_episode,
                    "lifespan": a.lifespan, "age": a.age, "days_alive": a.days_alive,
                    "updates": a.updates,
                    "n_success": a.n_success, "n_episodes": a.n_episodes,
                    "reward_sum": a.reward_sum, "net": a.net.state_dict(),
                    "opt": a.opt.state_dict()}
        u = self.usage
        state = {
            "version": 1, "episode": self.episode, "config": self.cfg.to_dict(),
            "curriculum": {"index": self.curriculum.index,
                           "episodes_in_phase": self.curriculum.episodes_in_phase,
                           "transitions": self.curriculum.transitions,
                           "checks_run": self.curriculum.checks_run},
            "cost_gate": self.cost_gate, "batch_no": self._batch_no,
            "next_grow": self._next_grow, "community_log": self.community_log,
            "farmers": [agent_state(a) for a in self.pop.farmers],
            "buyers": [agent_state(a) for a in self.pop.buyers],
            "next_id": self.pop._next_id, "deaths": self.pop.deaths,
            "births": [asdict(e) for e in self.pop.births],
            "usage": {"scale": u.scale, "words": dict(u.words), "word_total": u.word_total,
                      "forms": {k: dict(v) for k, v in u.forms.items()},
                      "form_total": dict(u.form_total), "episodes": u.episodes},
            "store": {"buf": self.store._buf, "pos": self.store._pos,
                      "total_added": self.store.total_added,
                      "meaning_counts": dict(self.store.meaning_counts)},
            "newborn_reports": self.newborn_reports,
            "totals": self.totals, "failure_counts": self.failure_counts,
        }
        d = os.path.join(self.out_dir, "snapshots")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "%s.pt" % tag)
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        return path

    def load_snapshot(self, path: str) -> None:
        """Continue from a snapshot, under *this* trainer's configuration."""
        from collections import Counter, defaultdict

        from .agents import make_agent
        from .population import BirthEvent
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.episode = int(st["episode"])
        cur = self.curriculum
        cur.index = int(st["curriculum"]["index"])
        cur.episodes_in_phase = int(st["curriculum"]["episodes_in_phase"])
        cur.transitions = list(st["curriculum"]["transitions"])
        cur.checks_run = int(st["curriculum"]["checks_run"])
        self.cost_gate = float(st["cost_gate"])
        self._batch_no = int(st["batch_no"])
        self._next_grow = st["next_grow"]
        self.community_log = list(st["community_log"])

        def restore(rec):
            a = make_agent(self.cfg, agent_id=rec["agent_id"], role=rec["role"],
                           slot=rec["slot"], generation=rec["generation"],
                           birth_episode=rec["birth_episode"], lifespan=rec["lifespan"],
                           device=self.device)
            a.net.load_state_dict(rec["net"])
            try:
                a.opt.load_state_dict(rec["opt"])
            except Exception:
                pass                    # e.g. a changed learning rate: fresh moments
            for k in ("age", "days_alive", "n_success", "n_episodes", "reward_sum"):
                setattr(a, k, rec[k])
            a.updates = int(rec.get("updates", 0))
            return a
        self.pop.farmers = [restore(r) for r in st["farmers"]]
        self.pop.buyers = [restore(r) for r in st["buyers"]]
        self.pop._next_id = int(st["next_id"])
        self.pop.deaths = int(st["deaths"])
        self.pop.births = [BirthEvent(**b) for b in st["births"]]
        u = self.usage
        us = st["usage"]
        u.scale, u.word_total, u.episodes = us["scale"], us["word_total"], us["episodes"]
        u.words = defaultdict(float, us["words"])
        u.forms = defaultdict(lambda: defaultdict(float),
                              {k: defaultdict(float, v) for k, v in us["forms"].items()})
        u.form_total = defaultdict(float, us["form_total"])
        so = st["store"]
        self.store._buf = list(so["buf"])[:self.store.capacity]
        self.store._pos = int(so["pos"]) % max(1, self.store.capacity)
        self.store.total_added = int(so["total_added"])
        self.store.meaning_counts = Counter(so["meaning_counts"])
        self.newborn_reports = list(st["newborn_reports"])
        self.totals = dict(st["totals"])
        self.failure_counts = dict(st["failure_counts"])
        self._next_check = self.episode + self.cfg.curriculum.check_every
        self.resume_note = ("resumed from     : %s at episode %d, rung %s"
                            % (path, self.episode, cur.phase.name))

    def maybe_grow(self) -> None:
        """Newcomers join once the founders have a working language."""
        if self.pop.full_size:
            return
        if self._next_grow is None:
            if self.cfg.curriculum.enabled and self.curriculum.index == 0:
                return                     # the founders are still inventing it
            self._next_grow = self.episode
        if self.episode < self._next_grow:
            return
        p = self.cfg.population
        for role, target in ((FARMER, p.n_farmers), (BUYER, p.n_buyers)):
            if len(self.pop.pool(role)) < target:
                self.pop.add_newcomer(role, self.episode, on_birth=self.on_birth)
        self._next_grow = self.episode + p.grow_every
        self.community_log.append({"episode": self.episode, "phase": self.curriculum.phase.name,
                                   "farmers": len(self.pop.farmers),
                                   "buyers": len(self.pop.buyers)})
        if self.pop.full_size:
            self.log.always("  [community] full size: %d farmers, %d buyers at episode %s"
                            % (len(self.pop.farmers), len(self.pop.buyers),
                               "{:,}".format(self.episode)))

    def update_cost_gate(self, succ: torch.Tensor) -> None:
        """Speaker costs: off through the first rung, on once it has been passed."""
        cur = self.curriculum
        if cur.index > 0 or not self.cfg.curriculum.enabled:
            self.cost_gate = 1.0

    def maybe_check_promotion(self) -> None:
        """The light, frequent check -- so a rung that has worked is left promptly."""
        cur = self.curriculum
        if (not self.cfg.curriculum.enabled or cur.finished
                or self.episode < self._next_check):
            return
        self._next_check = self.episode + self.cfg.curriculum.check_every
        lo, hi = rung_budget(self.cfg, cur.phase)
        if cur.episodes_in_phase < lo and cur.episodes_in_phase < hi:
            return                      # cannot pass yet; do not pay for the probe
        ev = self.gather_evidence(cur.phase, light=True)
        self.consider_promotion(ev, source="check")

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
        L("started            : %s" % self.started_utc)
        L("output directory   : %s" % os.path.abspath(self.out_dir))
        L("episodes           : %d  (batch %d -> %d updates)"
          % (c.train.episodes, c.train.batch_size,
             c.train.episodes // max(1, c.train.batch_size)))
        if c.population.founders_farmers or c.population.founders_buyers:
            L("population         : founded by %d farmers, %d buyers; grows to %d + %d "
              "after the first rung (one of each every %s episodes)"
              % (len(self.pop.farmers), len(self.pop.buyers), c.population.n_farmers,
                 c.population.n_buyers, "{:,}".format(c.population.grow_every)))
        else:
            L("population         : %d farmers, %d buyers"
              % (c.population.n_farmers, c.population.n_buyers))
        L("turnover           : %s%s" % (
            "ON" if c.population.turnover else "OFF",
            "  (lifespan %d-%d %s)" % (c.population.lifespan_min, c.population.lifespan_max,
                                       c.population.lifespan_unit)
            if c.population.turnover else ""))
        bn = c.bottleneck
        regime = ("fixed cap of %d transcripts" % bn.n_samples if bn.n_samples > 0
                  else "%.0f%% of the store, up to %d transcripts"
                  % (100 * bn.coverage, bn.max_samples))
        L("bottleneck         : %s%s" % (
            "ON" if bn.enabled else "OFF",
            "  (%s; store holds %d; %d epochs)" % (regime, bn.store_capacity, bn.epochs)
            if bn.enabled else ""))
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
        if c.curriculum.enabled:
            L("curriculum         : " + " -> ".join(
                "%s [%s-%s]" % (p.name, "{:,}".format(rung_budget(c, p)[0]),
                                "{:,}".format(rung_budget(c, p)[1])
                                if rung_budget(c, p)[1] < 10**11 else "open")
                for p in self.curriculum.phases))
            L("                     (on a rung's max without meeting its criteria: %s)"
              % c.curriculum.on_stall)
        L("speaker pressures  : %.3f per symbol, up to %.3f per novel word, %.3f for "
          "matching the population's convention" % (c.reward.symbol_cost,
                                                     c.reward.rarity_cost,
                                                     c.reward.convention))
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
                                    device=self.device, rng=self.eval_rng,
                                    phase=self.curriculum.phase,
                                    sampler_for=self.phase_sampler)
        rec = ev.to_dict()
        rec["at_birth_vs_veterans"] = probe
        rec["phase"] = self.curriculum.phase.name
        self.birth_log.write(rec)
        self.newborn_reports.append(rec)
        if ev.kind == "newcomer":
            self.log("  [join] %s %s joins the community (now %d farmers, %d buyers)"
                     % (ROLE_NAMES[ev.role], newborn.name, len(self.pop.farmers),
                        len(self.pop.buyers)))
        else:
            self.log("  [birth] %s %s gen %d replaces agent %d (age %d, success %.3f)"
                     % (ROLE_NAMES[ev.role], newborn.name, ev.generation,
                        ev.replaced_agent_id, ev.replaced_age, ev.replaced_success_rate))
        if info.get("n_samples"):
            def acc(x):
                return "%.3f" % x if x is not None else "n/a"
            self.log("          bottleneck: %d of %d stored transcripts (%s), from "
                     "generations %s; token acc %s over %d own tokens, decision acc %s"
                     % (info["n_samples"], info["store_size"],
                        ", ".join("%s %d" % kv for kv in
                                  info.get("phases_in_curriculum", {}).items()),
                        info["teacher_generations"], acc(info["token_accuracy"]),
                        info.get("own_token_targets", 0), acc(info["decision_accuracy"])))
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
        views = phase.views()
        sampler = self.phase_sampler(views[0])
        n_ev = max(200, cfg.log.intelligibility_episodes)
        # every view of the rung (both describers, in a swap): sampled play is what
        # the vocabulary statistics are computed over
        evs = [evaluate_success(cfg, self.pop, self.world, max(100, n_ev // len(views)),
                                device=self.device, rng=self.eval_rng,
                                phase=v, sampler=self.phase_sampler(v)) for v in views]
        ev = dict(evs[0])
        for k in ("success_rate", "success_rate_on_viable", "comprehension_rate",
                  "mean_reward"):
            vals = [e[k] for e in evs if e[k] == e[k]]
            ev[k] = sum(vals) / len(vals) if vals else float("nan")
        batches = [e["batch"] for e in evs]
        evidence = self.gather_evidence(phase, light=False)
        comp = evidence["_compositionality"]
        vocab = vocab_stats(cfg, batches)
        stab = self.stability.measure(self.pop, self.world, device=self.device, phase=phase)
        zs = zero_shot(cfg, self.pop, self.world, cfg.log.zeroshot_episodes,
                       device=self.device, rng=self.eval_rng, phase=views[0],
                       sampler=self.zero_shot_sampler(views[0]),
                       chance=(self.chance_for(views[0])
                               if self.chance_for(views[0]) == self.chance_for(views[0])
                               else 0.0),
                       n_holdout=(int(self.referential_world.holdout.shape[0])
                                  if phase.tuples and self.referential_world is not None
                                  else None))
        abl = channel_ablation(cfg, self.pop, self.world, cfg.log.ablation_episodes,
                               device=self.device, rng=self.eval_rng,
                               phase=views[0], sampler=sampler)
        newborn_age = (max(20, cfg.population.lifespan_min // 6)
                       if cfg.population.lifespan_unit == "updates"
                       else max(200, cfg.population.lifespan_min // 6))
        intel = intelligibility(cfg, self.pop, self.world,
                                cfg.log.intelligibility_episodes,
                                newborn_age=newborn_age, device=self.device,
                                rng=self.eval_rng, phase=phase,
                                sampler_for=self.phase_sampler)
        # ---- addendum section 3 ------------------------------------------
        words = word_stats(cfg, batches)
        overlap = cross_role_overlap(cfg, batches)
        qty_live = live_encoding(cfg, batches, field=1, given=0)
        lenfreq = length_frequency(cfg, self.pop, self.world, device=self.device,
                                   phase=phase)
        buckets = bucketed_analysis(cfg, self.pop, self.world, device=self.device,
                                    rng=self.eval_rng, phase=phase)
        forms = (self.forms.observe(self.pop, self.episode, device=self.device,
                                    phase=phase)
                 if self.forms is not None else {})
        speakers = {k: {kk: vv for kk, vv in v.items()}
                    for k, v in evidence.get("speakers", {}).items()}

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
            "rung_evidence": {k: v for k, v in evidence.items() if not k.startswith("_")},
            "per_role_structure": {
                k: {"topsim": v.get("topsim"), "null": v.get("null"),
                    "positional": v.get("positional"),
                    "posdis": v.get("posdis"), "bosdis": v.get("bosdis"),
                    "field_coverage": v.get("field_coverage"),
                    "per_field_coverage": v.get("per_field_coverage"),
                    "slots": v.get("positional_rows", [])}
                for k, v in speakers.items()},
            "context_consistency": self._context_consistency(phase),
            "cross_role_overlap": overlap,
            "quantity_encoding_live": qty_live,
            "usage": self.usage.summary(),
            "speaker_cost_gate": self.cost_gate,
            "started_utc": self.started_utc,
            "final": final,
        }
        for sp in row["rung_evidence"].get("speakers", {}).values():
            sp.pop("positional_rows", None)
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
        h.coherence.append(stab["coherence"])
        h.coherence_cross.append(stab.get("coherence_cross", float("nan")))
        h.positional_farmer.append(speakers.get("farmer", {}).get("positional", float("nan")))
        h.positional_buyer.append(speakers.get("buyer", {}).get("positional", float("nan")))
        h.cross_role_overlap.append(overlap.get("weighted_overlap", float("nan")))
        h.mean_word_len.append(words["mean_word_len_atoms"])
        h.at_length_cap.append(words["at_length_cap_frac"])
        h.phase_index.append(float(phase.index))
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
        self.checkpoint_headline(row)
        self.print_summary(row, ev, comp, vocab, stab, zs, intel, comp_pop, flags,
                           time.time() - t0, abl, words, lenfreq, buckets, forms)
        self.archive_examples(ev["batch"])
        if not final:
            self.consider_promotion(evidence, source="checkpoint")
            self._next_check = self.episode + cfg.curriculum.check_every
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

        ev_rung = row.get("rung_evidence") or {}
        if ev_rung:
            L("")
            L("RUNG EVIDENCE  (%s)" % self.curriculum.phase.name)
            for v in ev_rung.get("views", []):
                who = ("%s describes -> %s decodes" % (v["informer"], v["guesser"])
                       if v.get("informer") else "both")
                L("  %-32s success %.3f (muted %.3f), %.0f%% of headroom over silence"
                  % (who, v["success"], v["muted_success"], 100 * (v["transfer"]
                     if v["transfer"] == v["transfer"] else float("nan"))))
            for lbl in ("farmer", "buyer"):
                if "%s_report" % lbl in ev_rung:
                    L("  %s reports partner's tuple: %.3f (muted %.3f)"
                      % (lbl, ev_rung["%s_report" % lbl], ev_rung["muted_%s_report" % lbl]))
            for lbl, sp in (row.get("per_role_structure") or {}).items():
                L("  %-6s speaking: topsim %.3f vs null %.3f, positional structure %.3f, "
                  "field coverage %.3f %s"
                  % (lbl, sp.get("topsim", float("nan")), sp.get("null", float("nan")),
                     sp.get("positional", float("nan")),
                     sp.get("field_coverage", float("nan")) if sp.get("field_coverage")
                     is not None else float("nan"),
                     [round(x, 2) for x in (sp.get("per_field_coverage") or [])]))
            ov = row.get("cross_role_overlap") or {}
            L("  cross-role overlap: %.3f weighted (farmer %.0f%% / buyer %.0f%% of word "
              "tokens are shared forms; Jaccard %.3f)"
              % (ov.get("weighted_overlap", float("nan")),
                 100 * ov.get("farmer_share_shared", float("nan")),
                 100 * ov.get("buyer_share_shared", float("nan")),
                 ov.get("jaccard_types", float("nan"))))
            q = row.get("quantity_encoding_live") or {}
            if q.get("n", 0) >= 50:
                L("  quantity in live messages: %.3f bits beyond variety (shuffled-null "
                  "corrected, %d messages)" % (q["excess_bits"], q["n"]))

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
        L("  shared code       : coherence buyer %.3f / farmer %.3f / across roles %.3f  "
          "(1 = all agents say the same thing for the same meaning)"
          % (stab["coherence_buyer"], stab["coherence_farmer"],
             stab.get("coherence_cross", float("nan"))))
        L("  speaker pressures : %s" % ("on" if row.get("speaker_cost_gate", 1.0) >= 1.0
                                          else "off until the first rung is passed"))
        us = row.get("usage") or {}
        if us:
            L("  recent usage      : %d word types in circulation, %d established; "
              "%d meanings with a convention"
              % (us.get("recent_word_types", 0), us.get("established_types", 0),
                 us.get("meanings_with_convention", 0)))

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
        if zs.get("suppressed"):
            L("  zero-shot         : seen %.3f -> unseen %.3f  (retention not reported: %s)"
              % (zs["seen_success"], zs["unseen_success"], zs["suppressed"]))
        else:
            L("  zero-shot         : seen %.3f -> unseen %.3f  (retention %.2f, %d held-out "
              "combos)" % (zs["seen_success"], zs["unseen_success"], zs["retention"],
                           zs["n_holdout_combos"]))
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

    def _format_examples(self, batch, n: int) -> list[tuple[bool, str]]:
        from .transcripts import format_round, pick_examples
        phase = getattr(batch, "phase", None) or self.curriculum.phase
        out = []
        for i in pick_examples(batch, n):
            try:
                text = "\n".join(format_round(self.cfg, phase, batch, i, pop=self.pop,
                                               episode=self.episode))
            except Exception as exc:
                text = "(could not format round %d: %s)" % (i, exc)
            ok = bool(batch.res["success"][i]) if batch.res is not None else False
            out.append((ok, text))
        return out

    def print_examples(self, batch) -> None:
        for _, text in self._format_examples(batch, self.cfg.log.n_example_transcripts):
            self.log(text)
            self.log("")

    def archive_examples(self, batch) -> None:
        for ok, text in self._format_examples(batch, 2):
            self.archive.append({"episode": self.episode, "rendered": text, "success": ok})

    # ------------------------------------------------------------------
    def heartbeat(self, force: bool = False) -> None:
        """One status line, printed even in quiet mode, every heartbeat_seconds."""
        every = self.cfg.log.heartbeat_seconds
        now = time.time()
        t0, e0 = self._beat
        if not force and (every <= 0 or now - t0 < every):
            return
        self._beat = (now, self.episode)
        rate = (self.episode - e0) / max(1e-9, now - t0)
        total = self.cfg.train.episodes
        eta = (total - self.episode) / rate / 3600 if rate > 0 else float("nan")
        cur = self.curriculum
        gpu = ""
        if self.torch_device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.torch_device) / 1e9
            cap = torch.cuda.get_device_properties(self.torch_device).total_memory / 1e9
            torch.cuda.reset_peak_memory_stats(self.torch_device)
            gpu = " | GPU peak %.1f / %.0f GB" % (peak, cap)
        self.log.always(
            "[%s] %s / %s episodes (%.1f%%) | rung %s (%d/%d, %s in rung) | %s eps/s | "
            "ETA %.1f h | rolling success %.3f | community %d+%d | births %d%s"
            % (time.strftime("%H:%M:%S UTC", time.gmtime()), "{:,}".format(self.episode),
               "{:,}".format(total), 100.0 * self.episode / max(1, total), cur.phase.name,
               cur.phase.index + 1, len(cur.phases), "{:,}".format(cur.episodes_in_phase),
               "{:,.0f}".format(rate), eta, self.train_success.mean,
               len(self.pop.farmers), len(self.pop.buyers), len(self.pop.births), gpu))

    def checkpoint_headline(self, row: dict[str, Any]) -> None:
        """Two lines per checkpoint for the terminal; the full block goes to run.log."""
        def f(x, fmt="%.3f"):
            return fmt % x if isinstance(x, (int, float)) and x == x else "n/a"
        ev = row.get("rung_evidence") or {}
        views = ev.get("views") or []
        succ = " / ".join("%s (muted %s)" % (f(v.get("success")), f(v.get("muted_success")))
                          for v in views) or f(row.get("eval_success"))
        roles = []
        for lbl, sp in (row.get("per_role_structure") or {}).items():
            cov = sp.get("per_field_coverage") or []
            roles.append("%s coverage %s [%s]" % (lbl, f(sp.get("field_coverage")),
                                                  " ".join(f(x, "%.2f") for x in cov)))
        st = row.get("stability") or {}
        w = row.get("words") or {}
        ov = row.get("cross_role_overlap") or {}
        self.log.always(
            "[checkpoint %s] rung %s | success %s | channel %s of headroom | %s"
            % ("{:,}".format(self.episode), row.get("phase"), succ, f(ev.get("transfer"), "%.2f"),
               "; ".join(roles) or "no speakers probed"))
        self.log.always(
            "    coherence farmer %s buyer %s across %s | overlap %s | %s words, %s atoms/word, "
            "%s words/utterance, %s at buffer end"
            % (f(st.get("coherence_farmer")), f(st.get("coherence_buyer")),
               f(st.get("coherence_cross")), f(ov.get("weighted_overlap")),
               w.get("distinct_words", "n/a"), f(w.get("mean_word_len_atoms"), "%.2f"),
               f(w.get("mean_words_per_message"), "%.2f"),
               f(100 * w.get("at_length_cap_frac", float("nan")), "%.0f%%")))

    def batch_size_for(self, rung) -> int:
        scale = float((self.cfg.train.rung_batch_scale or {}).get(rung.name, 1))
        return max(1, int(round(self.cfg.train.batch_size * scale)))

    # ------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        cfg, L = self.cfg, self.log
        self.banner()
        every = max(1, cfg.log.checkpoint_every)
        next_ckpt = (self.episode // every + 1) * every
        L.always("run %s started %s -> %s  (full log: %s)"
                 % (cfg.name, self.started_utc, os.path.abspath(self.out_dir),
                    os.path.join(os.path.abspath(self.out_dir), "run.log")))

        while self.episode < cfg.train.episodes and not self._stop_requested:
            rung = self.curriculum.phase
            B = self.batch_size_for(rung)
            n = min(B, cfg.train.episodes - self.episode)
            # A swap rung alternates the describer batch by batch, so every agent
            # spends half its time describing and half decoding.
            phase = (rung.with_informer(FARMER if self._batch_no % 2 == 0 else BUYER)
                     if rung.swaps else rung)
            self._batch_no += 1
            f_idx, b_idx = self.pop.pair(n, device=self.device)
            if phase.referential:
                # The lineup game: no market, no stock, no price -- just meanings.
                scen = self.referential_world.sample(n, informer=phase.informer)
            elif phase.mutual:
                scen = self.referential_world.sample_mutual(n)
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
                    frac_done=frac, device=self.device, phase=phase, usage=self.usage,
                    cost_scale=self.cost_gate)
            else:
                from .rollout import run_episodes
                batch = run_episodes(cfg, scen, self.pop.farmers, self.pop.buyers,
                                     f_idx, b_idx, device=self.device, phase=phase)

            self.pop.record_episode_participation(f_idx, b_idx, batch)
            if batch.res is not None:
                self.update_cost_gate(batch.res["success"])
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
            lineup_like = phase.referential or phase.mutual or phase.order
            if batch.res is not None and not lineup_like:
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
                tag = "mutual" if phase.mutual else ("order" if phase.order else "lineup")
                self.failure_counts[tag + "_hit"] = (
                    self.failure_counts.get(tag + "_hit", 0) + hits)
                self.failure_counts[tag + "_miss"] = (
                    self.failure_counts.get(tag + "_miss", 0) + (len(batch) - hits))
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

            if lineup_like:
                self.write_lineups(batch, scen, self.episode, phase)
            else:
                self.ledger.write_batch(batch, self.pop, self.episode,
                                        self.economy.season)
            self.store.add_batch(batch, self.pop.farmers, self.pop.buyers, self.episode)
            self.transcripts.write_batch(batch, phase, self.episode, self.pop)

            if cfg.train.algo != "gumbel":
                from .rollout import update_agents
                update_agents(cfg, batch, self.pop.farmers, self.pop.buyers,
                              frac_done=frac, device=self.device)

            self.episode += n
            self.curriculum.episodes_in_phase += n
            self.pop.turn_over(self.episode, on_birth=self.on_birth)
            self.maybe_grow()
            self.write_progress()
            self.heartbeat()

            if self.episode >= next_ckpt:
                row = self.checkpoint()
                self.maybe_plot()
                self.write_interim_report(row)
                if cfg.log.snapshot_every_checkpoint:
                    try:
                        self.save_snapshot("latest")
                    except Exception as exc:
                        self.log("  [snapshot] skipped: %s" % exc)
                next_ckpt += cfg.log.checkpoint_every
            else:
                self.maybe_check_promotion()

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
            phase = self.curriculum.phase
            sem = analyse_token_semantics(
                self.cfg, self.pop, self.world,
                n_samples=max(200, self.cfg.log.topsim_samples),
                device=self.device, rng=self.eval_rng, phase=phase)
            batches = [evaluate_success(self.cfg, self.pop, self.world,
                                        400 // len(phase.views()), device=self.device,
                                        rng=self.eval_rng, phase=v,
                                        sampler=self.phase_sampler(v))["batch"]
                       for v in phase.views()]
            row = dict(row)
            row["token_counts"] = vocab_stats(self.cfg, batches)["token_counts"]
            row["word_counts"] = word_stats(self.cfg, batches)["word_counts"]
            row["length_frequency_rows"] = length_frequency(
                self.cfg, self.pop, self.world, device=self.device,
                phase=phase).get("rows", [])[:20]
            if self.forms is not None:
                row["form_events"] = self.forms.report_rows()
                row["form_timeline"] = self.forms.timeline()
            row["curriculum"] = self.curriculum_report()
            row["language_properties"] = self.language_properties(sem)
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
        self.promotion_log.close()
        self.transcripts.close()
        with open(os.path.join(self.out_dir, "history.json"), "w", encoding="utf-8") as fh:
            json.dump(self.history.to_dict(), fh, indent=1)
        self.log.close()
