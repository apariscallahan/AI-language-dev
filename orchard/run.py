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
import glob
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
def benchmark(cfg: Config, n_batches: int = 12) -> int:
    """Time this machine on this configuration and print what the run will cost.

    Worth doing before starting anything long on a rented box: it reports real
    episodes per second for these exact settings, an estimate for the configured
    episode count, and peak GPU memory, so a run that will not fit or will not
    finish is obvious in under a minute.
    """
    import time

    import torch

    from .agents import count_parameters, make_agent, sequence_len
    from .batched import TensorWorld
    from .env import BUYER, FARMER
    from .hardware import describe, setup
    from .population import Population
    from .train import expected_generations

    dev = setup(cfg)
    cfg.train.device = str(dev)
    torch.manual_seed(cfg.train.seed)

    print("=" * 78)
    print("BENCHMARK -- %s" % cfg.name)
    print("=" * 78)
    print("  %s" % describe(dev, cfg))

    pop = Population(cfg, random.Random(0), device=str(dev))
    # Time the community at the size it will spend most of the run at, not the
    # founders it starts from.
    while not pop.full_size:
        for role, target in ((FARMER, cfg.population.n_farmers),
                             (BUYER, cfg.population.n_buyers)):
            if len(pop.pool(role)) < target:
                pop.add_newcomer(role, 0)
    per_agent = count_parameters(pop.farmers[0].net)
    n_agents = cfg.population.n_farmers + cfg.population.n_buyers
    print("  agents            : %d farmers + %d buyers = %d"
          % (cfg.population.n_farmers, cfg.population.n_buyers, n_agents))
    print("  brain             : d=%d, %d layers, %s params each, %s in total"
          % (cfg.model.d_model, cfg.model.n_layers, "{:,}".format(per_agent),
             "{:,}".format(per_agent * n_agents)))
    print("  weights + Adam    : %.2f GB" % (per_agent * n_agents * 4 * 3 / 1e9))
    print("  sequence length   : %d  (%d symbols x %d turns of dialogue)"
          % (sequence_len(cfg), cfg.channel.max_symbols, cfg.channel.n_turns))
    print("  batch             : %d episodes -> %d per agent per step"
          % (cfg.train.batch_size,
             cfg.train.batch_size // max(1, cfg.population.n_farmers)))

    tw = TensorWorld(cfg, device=str(dev),
                     generator=torch.Generator(device=dev).manual_seed(0))
    B = cfg.train.batch_size
    f_idx, b_idx = pop.pair(B, device=str(dev))

    from .gumbel import run_and_update_gumbel

    from .curriculum import (ReferentialWorld, convention_applies, costs_apply,
                             phase_named)
    from .conventions import PopulationUsage
    rw = ReferentialWorld(cfg, device=str(dev),
                          generator=torch.Generator(device=dev).manual_seed(1))
    # The speaker's own terms are host-side Python (edit distances against the
    # population's recent forms) and are a real share of an update on the rungs
    # that have them on, so the benchmark plays each rung with the gates that
    # rung would actually run under. Timing them off flattered every rung from
    # `name-all` up. The warm-up step is what gives the record enough support
    # for the convention bonus to be live by the time anything is timed.
    usage = PopulationUsage(cfg)

    def one_step(phase, n):
        fi, bi = pop.pair(n, device=str(dev))
        if phase.referential:
            sb = rw.sample(n, informer=phase.informer)
        elif phase.mutual:
            sb = rw.sample_mutual(n)
        else:
            sb = tw.sample(n)
        run_and_update_gumbel(cfg, sb, pop.farmers, pop.buyers, fi, bi,
                              update=100, device=str(dev), phase=phase,
                              usage=usage,
                              cost_scale=1.0 if costs_apply(cfg, phase) else 0.0,
                              convention_scale=(1.0 if convention_applies(cfg, phase)
                                                else 0.0))

    # Rungs differ a lot in cost: one speaking turn in the lineup, the whole
    # dialogue in the market. Time a light, a middle and the heaviest rung, at
    # full community size (an upper bound: founders are cheaper), at the
    # configured batch -- the same measurement on every device.
    rates = {}
    scales = (1,)
    for name in ("name-fruit", "mutual", "market"):
        phase = phase_named(cfg, name)
        base = int(B * float((cfg.train.rung_batch_scale or {}).get(name, 1)))
        cells = []
        for s in scales:
            n = base * s
            try:
                one_step(phase, n)                 # warm-up
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                for _ in range(n_batches):
                    one_step(phase, n)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                cells.append("batch %s: out of memory" % "{:,}".format(n))
                continue
            dt = (time.time() - t0) / n_batches
            if s == 1:
                rates[name] = n / dt
            mem = (", peak %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9)
                   if dev.type == "cuda" else "")
            cells.append("batch %s: %.2f s/update, %s eps/s%s"
                         % ("{:,}".format(n), dt, "{:,.0f}".format(n / dt), mem))
        print("\n  %-13s %s" % (name, "\n                ".join(cells)))
    def hours(name):
        r = rates.get(name)
        return cfg.train.episodes / r / 3600 if r else float("nan")
    print("\n  configured run    : %s episodes -> ~%.1f hours at the middle rung's rate "
          "(%.1f at a naming rung's, %.1f at the market's)"
          % ("{:,}".format(cfg.train.episodes), hours("mutual"), hours("name-fruit"),
             hours("market")))
    n_ck = max(1, cfg.train.episodes // max(1, cfg.train.batch_size)
               // max(1, cfg.log.checkpoint_every_updates))
    print("  plus %d checkpoints; the metric suite replays episodes three times "
          "for the\n  channel ablation, so allow roughly %.0f%% on top."
          % (n_ck, 15))
    print("  generations       : ~%.1f lineage turnovers" % expected_generations(cfg))
    print("=" * 78)
    print("  If the hours are wrong, change --episodes, or --batch-size for speed.")
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
def holdout_report(cfg: Config, path: str) -> int:
    """Which field generalises to combinations nobody trained on, field by field.

    The promotion gate reports one number -- the mean over (fruit, colour,
    quality) -- and at `mutual` that mean is systematically pessimistic. The
    reserved set is a Latin square: exactly one quality is withheld from every
    (fruit, colour) pair, so a held-out round asks for precisely the quality
    that pair never showed, and a listener that has fit the training
    distribution is pushed away from it. That field cannot be got right, and it
    is averaged in with two that can.

    So this prints the breakdown from a finished snapshot, for a run that has
    already promoted past the only rungs where the gate is evaluated.
    """
    import random as _random

    from .metrics import phase_evidence
    from .train import Trainer

    # Score it under the settings it was trained with, not under this command
    # line: the report is about the snapshot, and the holdout is derived from
    # the world's field sizes.
    saved = (torch.load(path, map_location="cpu", weights_only=False)
             .get("config"))
    if saved:
        keep = cfg.train.device
        cfg = Config.from_dict(saved, allow_legacy=True)
        if keep and keep != "auto":
            cfg.train.device = keep
    out = os.path.join(os.path.dirname(os.path.abspath(path)), "_holdout_report")
    trainer = Trainer(cfg, out, quiet=True)
    trainer.load_snapshot(path)
    phase = trainer.curriculum.phase
    if not getattr(phase, "whole", False):
        # The gate runs at `name-all` and `mutual` only; a later snapshot has to
        # be scored on the last rung that measured this.
        cand = [p for p in trainer.curriculum.phases if getattr(p, "whole", False)]
        if not cand:
            print("no rung in this ladder measures held-out combinations")
            return 1
        phase = cand[-1]
        print("this snapshot stopped on a rung that does not measure held-out "
              "combinations; scoring it on `%s`, the last one that does\n" % phase.name)
    ev = phase_evidence(
        cfg, trainer.pop, trainer.world, phase,
        sampler_for=trainer.phase_sampler, n_eval=cfg.log.zeroshot_episodes,
        n_topsim=cfg.log.topsim_samples, n_semantics=cfg.log.topsim_samples,
        chance=trainer.chance_for(phase), device=cfg.train.device,
        rng=_random.Random(0), holdout_sampler_for=trainer.holdout_sampler,
        holdout_floor_for=trainer.holdout_floor)
    acc, base = ev.get("holdout_field_acc"), ev.get("seen_field_acc")
    floors = trainer.holdout_floor(phase) or (float("nan"), float("nan"))
    print("rung %s, %d held-out combinations of %d"
          % (phase.name, len(trainer.referential_world.holdout.held),
             cfg.world.n_varieties * cfg.world.n_colors * cfg.world.n_quality))
    each = ev.get("holdout_field_ratios") or []
    print("%-11s %8s %8s %9s" % ("field", "held-out", "trained", "transfers"))
    prod_h = prod_s = 1.0
    for i, name in enumerate(("fruit", "colour", "quality")):
        if not acc or i >= len(acc):
            break
        b = base[i] if base and i < len(base) else float("nan")
        prod_h *= acc[i]
        prod_s *= b
        r = "%9.3f" % each[i] if i < len(each) else "      n/a"
        note = ""
        if acc[i] < floors[0]:
            note = "  <- below the %.2f a message-blind guesser gets" % floors[0]
        print("%-11s %8.3f %8.3f%s%s" % (name, acc[i], b, r, note))
    print()
    print("%-11s %8.3f %8.3f %9.3f   each field once, over %.2f/%.2f"
          % ("mean", ev.get("holdout_fields", float("nan")),
             ev.get("seen_fields", float("nan")),
             ev.get("holdout_field_ratio", float("nan")), floors[0], floors[1]))
    print()
    # Independent fields would multiply. Where they do not, the code is right
    # about each field on its own and wrong about them together -- which is what
    # a Latin-square holdout produces: get the fruit and the colour right and
    # the training distribution has ruled out the one quality that is the answer.
    print("%-11s %8.3f %8.3f   (one side, all three at once)"
          % ("conjunction", ev.get("holdout_side", float("nan")),
             ev.get("seen_side", float("nan"))))
    print("%-11s %8.3f %8.3f   if the three fields were independent"
          % ("  expected", prod_h, prod_s))
    print("%-11s %8.3f %8.3f   (both sides, all three)"
          % ("whole round", ev.get("holdout_success", float("nan")),
             ev.get("seen_success", float("nan"))))
    trainer.close()
    return 0


def list_snapshots(where: str) -> int:
    """What is in each snapshot, and is it safe to resume from?

    Printed before choosing one to carry on from: which rung it stopped on, how
    far in, how big the community was, and -- the one that is not obvious --
    whether its two seats are still one pool. Below
    ``curriculum.split_roles_at`` they have to be, and a snapshot written by a
    run that resumed before that was guaranteed holds two sets of copies that
    have been training apart.
    """
    import torch

    from .curriculum import ladder
    from .train import Trainer

    names = [ph.name for ph in ladder(Config())]
    if os.path.isfile(where):
        paths = [where]
    else:
        paths = sorted(p for p in glob.glob(os.path.join(where, "**", "*.pt"),
                                            recursive=True))
    if not paths:
        print("no snapshots under %s" % os.path.abspath(where))
        print("(runs keep them in <run>/snapshots/; pass a run folder, that "
              "folder, or a single .pt)")
        return 1
    print("%-34s %-12s %-9s %-9s %s"
          % ("snapshot", "rung", "update", "pool", "state"))
    affected = 0
    for path in paths:
        try:
            st = torch.load(path, map_location="cpu", weights_only=False)
            i = int(st["curriculum"]["index"])
            split = Trainer._pool_had_split(st)
            affected += int(split)
            print("%-34s %-12s %-9s %-9s %s"
                  % (os.path.relpath(path, where if os.path.isdir(where) else "."),
                     names[i] if i < len(names) else i,
                     "{:,}".format(int(st.get("updates", 0))),
                     "%d+%d" % (len(st["farmers"]), len(st["buyers"])),
                     "two sets of copies, drifted apart" if split else "one pool"))
        except Exception as exc:               # a half-written .pt must not stop the list
            print("%-34s %s" % (os.path.relpath(path), "unreadable: %s" % exc))
    if affected:
        print("")
        print("%d of these were written by a run that had resumed before the pool "
              "aliasing was fixed." % affected)
        print("Resuming one is allowed: the farmer copies are kept, the buyer ones "
              "dropped, and the run says so.")
        print("A snapshot marked `one pool` is the cleaner place to carry on from "
              "if you have one at a rung you are happy to redo from.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orchard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--out", type=str, default=None, help="output directory for this run")
    p.add_argument("--smoke", action="store_true",
                   help="environment-only check with scripted agents (spec step 1)")
    p.add_argument("--compare", nargs="+", default=None,
                   help="finished run directories to overlay")
    p.add_argument("--snapshots", nargs="?", type=str, const="runs", default=None,
                   metavar="PATH",
                   help="list the snapshots under PATH (a run folder, a snapshots "
                        "folder, or one .pt; default runs/) with the rung, update and "
                        "community each holds and whether its pool is intact, then exit")
    p.add_argument("--compare-out", type=str, default="runs/comparison")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", type=str, default=None,
                   help="continue from a snapshot (runs/<name>/snapshots/*.pt) under "
                        "the configuration given here; rung, weights, usage and the "
                        "transcript store all carry over")
    p.add_argument("--resume-at", type=str, default=None, metavar="RUNG",
                   help="with --resume: put the curriculum back on this rung, "
                        "keeping the weights, the community and the store. "
                        "`after-<rung>.pt` holds a curriculum already pointing at "
                        "the rung after, so this is how a rung is run again once "
                        "something it depends on has changed")
    p.add_argument("--holdout-report", type=str, default=None, metavar="SNAPSHOT",
                   help="score one snapshot on the held-out combinations, field "
                        "by field, and exit -- which of fruit, colour and quality "
                        "generalises to combinations nobody trained on")
    p.add_argument("--benchmark", nargs="?", type=int, const=12, default=None,
                   metavar="N",
                   help="time N batches on this machine and print what the "
                        "configured run will cost, then exit")
    args = p.parse_args(argv)

    if args.compare:
        return compare(args.compare, args.compare_out)
    if args.snapshots:
        return list_snapshots(args.snapshots)

    cfg = config_from_args(args)
    if args.holdout_report:
        return holdout_report(cfg, args.holdout_report)
    if args.smoke:
        return smoke(cfg)
    if args.benchmark:
        return benchmark(cfg, args.benchmark)

    # Default folder: when it started (UTC), then the run's name, e.g.
    # runs/2026-09-18_14-03-12UTC_orchard -- sorts by time, says what it is.
    now = time.gmtime()
    out = args.out or os.path.join(
        "runs", "%s_%s" % (time.strftime("%Y-%m-%d_%H-%M-%SUTC", now), cfg.name))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    trainer = Trainer(cfg, out, quiet=args.quiet,
                      started_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", now))
    if args.resume:
        trainer.load_snapshot(args.resume)
        if args.resume_at:
            trainer.rewind_to(args.resume_at)
    elif args.resume_at:
        print("--resume-at needs --resume: it moves the curriculum of a snapshot")
        return 2
    try:
        final = trainer.run()
        trainer.log("")
        trainer.log("running post-hoc token-semantics analysis (spec 6.3)...")
        # Analysed under the rung the run ended on: who speaks, and about what,
        # depends on it.
        phase = trainer.curriculum.phase
        sem = analyse_token_semantics(cfg, trainer.pop, trainer.world,
                                      n_samples=max(400, cfg.log.topsim_samples * 2),
                                      device=cfg.train.device, rng=trainer.eval_rng,
                                      phase=phase)
        with open(os.path.join(out, "token_semantics.json"), "w", encoding="utf-8") as fh:
            json.dump(sem.to_dict(), fh, indent=1)

        # keep the raw token counts available to the report
        from .lexicon import length_frequency, word_stats
        from .metrics import evaluate_success, vocab_stats
        batches = [evaluate_success(cfg, trainer.pop, trainer.world,
                                    600 // len(phase.views()), device=cfg.train.device,
                                    rng=trainer.eval_rng, phase=v,
                                    sampler=trainer.phase_sampler(v))["batch"]
                   for v in phase.views()]
        final["token_counts"] = vocab_stats(cfg, batches)["token_counts"]
        final["word_counts"] = word_stats(cfg, batches)["word_counts"]
        final["length_frequency_rows"] = length_frequency(
            cfg, trainer.pop, trainer.world, device=cfg.train.device,
            phase=phase).get("rows", [])[:20]
        if trainer.forms is not None:
            final["form_events"] = trainer.forms.report_rows()
            final["form_timeline"] = trainer.forms.timeline()
        final["curriculum"] = trainer.curriculum_report()
        final["language_properties"] = trainer.language_properties(sem)

        path = write_report(
            cfg, out, final=final, chance=trainer.chance, sem=sem,
            archive=trainer.archive, totals=trainer.totals,
            history=trainer.history.to_dict(),
            newborn_reports=trainer.newborn_reports,
            ledger_path=os.path.join(out, "trades.jsonl"),
            wall_minutes=(time.time() - t0) / 60.0)

        trainer.log.always("")
        trainer.log.always("wrote final report: %s" % path)
        trainer.log("ledger rows written: %d" % trainer.ledger.n_written)

        # Print the verdict last so it is the thing a human sees.
        from .report import assess
        v = assess(cfg, final, trainer.chance)
        trainer.log.always("")
        trainer.log.always("=" * 78)
        trainer.log.always("VERDICT: %s" % v["verdict"])
        trainer.log.always(v["summary"])
        trainer.log.always("=" * 78)
    except KeyboardInterrupt:
        trainer.log("\ninterrupted -- flushing logs")
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
