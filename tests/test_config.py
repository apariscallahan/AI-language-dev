"""One method, many scales: guards against the runs drifting apart again.

The CPU and GPU versions drifted because the method was written down in several
places -- code defaults, a dozen JSON presets, a GUI's own presets -- and because
schedules were counted in episodes, which mean a different amount of learning at
every batch size. These tests pin both down: presets may only change scale, and
every schedule is in training updates.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.config import (LEGACY_KEYS, PRESET_EXTRA_KEYS, PRESET_KEYS, Config,
                            flat_keys, method_changes, validate)
from orchard.conventions import PopulationUsage
from testscale import method_at_test_scale

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class TestPresetsChangeScaleOnly(unittest.TestCase):
    def test_every_preset_sets_only_what_a_preset_may(self):
        paths = sorted(CONFIGS.glob("*.json"))
        self.assertTrue(paths, "no presets found")
        for path in paths:
            d = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(d.get("name"), path.stem, path.name)
            allowed = PRESET_KEYS | PRESET_EXTRA_KEYS.get(path.stem, frozenset())
            extra = sorted(set(flat_keys(d)) - allowed)
            self.assertEqual(extra, [], "%s changes the method: %s -- change the code "
                             "default instead, or list it in PRESET_EXTRA_KEYS with a "
                             "reason" % (path.name, extra))
            validate(Config.from_json(str(path)))

    def test_scale_presets_report_no_method_change(self):
        for name in ("gpu_small", "gpu_community", "gpu_full"):
            cfg = Config.from_json(str(CONFIGS / ("%s.json" % name)))
            self.assertEqual(method_changes(cfg), {}, name)

    def test_the_community_preset_is_the_defaults(self):
        cfg = Config.from_json(str(CONFIGS / "gpu_community.json"))
        self.assertEqual(cfg.to_dict(), Config().to_dict())

    def test_a_method_change_is_named(self):
        cfg = Config()
        cfg.reward.symbol_cost = 0.5
        cfg.population.n_farmers = 3            # scale: not a method change
        self.assertEqual(method_changes(cfg), {"reward.symbol_cost": (0.03, 0.5)})


class TestSchedulesCountUpdates(unittest.TestCase):
    SCHEDULE = re.compile(r"every|anneal|budget|grow|half_life|lifespan")
    NOT_LEARNING = {"snapshot_every_checkpoint", "flush_every"}   # output only

    def test_every_schedule_is_in_updates(self):
        cfg = Config()
        for sect in ("curriculum", "population", "train", "reward", "log", "bottleneck"):
            for f in dataclasses.fields(getattr(cfg, sect)):
                if not self.SCHEDULE.search(f.name) or f.name in self.NOT_LEARNING:
                    continue
                if f.name.startswith("lifespan_"):
                    continue                    # documented as updates; tested below
                self.assertTrue(f.name.endswith("_updates"),
                                "%s.%s looks like a schedule but is not in updates"
                                % (sect, f.name))

    def test_old_episode_keys_are_refused_with_their_replacement(self):
        for key, why in LEGACY_KEYS.items():
            sect, k = key.split(".")
            with self.assertRaises(KeyError) as ctx:
                Config.from_dict({sect: {k: 1}})
            self.assertIn(key, str(ctx.exception))
            Config.from_dict({sect: {k: 1}}, allow_legacy=True)   # an old snapshot

    def test_usage_memory_decays_per_update_not_per_episode(self):
        for batch in (64, 4096):
            cfg = method_at_test_scale()
            u = PopulationUsage(cfg)
            for _ in range(cfg.reward.usage_half_life_updates):
                u.observe({}, batch)
            self.assertAlmostEqual(u.scale, 0.5, places=6, msg="batch %d" % batch)


class TestGrowthDoesNotSpendTheBudget(unittest.TestCase):
    def test_a_rung_budget_counts_from_full_size(self):
        from orchard.train import Trainer
        cfg = method_at_test_scale()
        cfg.world.max_qty = 4
        cfg.channel.max_symbols = 6
        cfg.model.d_model, cfg.model.d_ff = 32, 64
        cfg.population.n_farmers = cfg.population.n_buyers = 3
        cfg.population.founders_farmers = cfg.population.founders_buyers = 1
        cfg.population.grow_every_updates = 2
        cfg.population.turnover = False
        cfg.bottleneck.enabled = False
        cfg.curriculum.start_phase = "refer-swap"
        cfg.curriculum.on_stall = "hold"
        cfg.train.batch_size = 16
        cfg.train.episodes = 16 * 6
        cfg.log.checkpoint_every_updates = 10 ** 6
        cfg.log.plot = False
        for k in ("intelligibility_episodes", "zeroshot_episodes", "ablation_episodes"):
            setattr(cfg.log, k, 40)
        cfg.log.topsim_samples = 20
        cfg.log.stability_probes = 4
        out = tempfile.mkdtemp(prefix="orchard_grow_")
        try:
            torch.manual_seed(0)
            t = Trainer(cfg, out, quiet=True)
            t.run()
            t.close()
            self.assertEqual(t.updates, 6)
            self.assertTrue(t.pop.full_size)
            # 1+1 -> 2+2 after update 1, -> 3+3 after update 3: updates 4-6 count
            self.assertEqual(t.curriculum.updates_in_phase, 3)
            self.assertEqual(t.curriculum.episodes_in_phase, 16 * 6)
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestGradientCheckpointingChangesNothing(unittest.TestCase):
    """On by default for the GPU; must give the same update as without it."""

    def test_same_weights_after_an_update(self):
        from orchard.agents import make_agent
        from orchard.batched import TensorWorld
        from orchard.curriculum import ReferentialWorld, phase_named
        from orchard.env import BUYER, FARMER
        from orchard.gumbel import run_and_update_gumbel

        base = method_at_test_scale()
        base.model.d_model, base.model.d_ff = 32, 64
        base.channel.max_symbols = 6
        rw = ReferentialWorld(base, generator=torch.Generator().manual_seed(0))
        tw = TensorWorld(base, generator=torch.Generator().manual_seed(0))
        for name in ("refer-mutual", "haggle"):
            phase = phase_named(base, name)
            scen = rw.sample_mutual(24) if phase.mutual else tw.sample(24)
            results = []
            for ckpt in (False, True):
                cfg = copy.deepcopy(base)
                cfg.train.grad_checkpoint = ckpt
                torch.manual_seed(5)
                f = [make_agent(cfg, agent_id=i, role=FARMER, slot=i, generation=0,
                                birth_episode=0, lifespan=10 ** 9) for i in range(2)]
                b = [make_agent(cfg, agent_id=10 + i, role=BUYER, slot=i, generation=0,
                                birth_episode=0, lifespan=10 ** 9) for i in range(2)]
                i = torch.arange(24)
                torch.manual_seed(9)
                import torch.utils.checkpoint as tuc
                real, calls = tuc.checkpoint, []

                def counted(*a, **k):
                    calls.append(1)
                    return real(*a, **k)
                tuc.checkpoint = counted
                try:
                    run_and_update_gumbel(cfg, scen, f, b, i % 2, (i // 2) % 2,
                                          update=100, phase=phase)
                finally:
                    tuc.checkpoint = real
                self.assertEqual(bool(calls), ckpt, "checkpointing on=%s ran %d times"
                                 % (ckpt, len(calls)))
                results.append([p.detach().clone() for a in f + b
                                for p in a.net.parameters()])
            for p0, p1 in zip(*results):
                self.assertTrue(torch.allclose(p0, p1, atol=1e-5), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
