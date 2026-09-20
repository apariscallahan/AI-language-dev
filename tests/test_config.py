"""One configuration, on every device: guards against versions drifting apart.

The CPU and GPU versions drifted because the settings were written down in
several places -- code defaults, a dozen JSON presets, a GUI's own presets --
because schedules were counted in episodes, which mean a different amount of
learning at every batch size, and because the CPU was tested at one size while
the GPU ran at another. These tests pin all three down: there is one
configuration, a run may change only its length, seed, device and output,
every schedule is in training updates, and there is no device-specific
arithmetic.
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

from orchard.config import (EXPERIMENT_KEYS, LEGACY_KEYS, RUN_KEYS, Config, flat_keys,
                            method_changes, validate)
from orchard.conventions import PopulationUsage

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ROOT = Path(__file__).resolve().parents[1]


class TestOneConfiguration(unittest.TestCase):
    def test_config_files_are_named_experiments_only(self):
        for path in sorted(CONFIGS.glob("*.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(d.get("name"), path.stem, path.name)
            self.assertIn(path.stem, EXPERIMENT_KEYS,
                          "%s is not a named experiment: the configuration lives in "
                          "orchard/config.py, not in a second copy" % path.name)
            extra = sorted(set(flat_keys(d)) - RUN_KEYS - EXPERIMENT_KEYS[path.stem])
            self.assertEqual(extra, [], "%s changes more than its experiment: %s"
                             % (path.name, extra))
            validate(Config.from_json(str(path)))

    def test_sizes_are_part_of_what_is_simulated(self):
        # a run that tested one size on a CPU and ran another on a GPU was two
        # versions; now a size change is reported like any other change
        cfg = Config()
        cfg.population.n_farmers = 48
        cfg.train.batch_size = 4096
        cfg.model.d_model = 96
        self.assertEqual(set(method_changes(cfg)),
                         {"population.n_farmers", "train.batch_size", "model.d_model"})

    def test_run_settings_are_not_changes(self):
        cfg = Config()
        cfg.name, cfg.train.episodes, cfg.train.seed = "x", 1000, 7
        cfg.train.device, cfg.train.grad_checkpoint = "cpu", True
        cfg.log.heartbeat_seconds = 5
        self.assertEqual(method_changes(cfg), {})

    def test_no_device_specific_arithmetic(self):
        cfg = Config()
        self.assertFalse(hasattr(cfg.train, "amp"))
        self.assertFalse(hasattr(cfg.train, "tf32"))
        for f in ("orchard/agents.py", "orchard/gumbel.py", "orchard/rollout.py",
                  "orchard/bottleneck.py", "orchard/curriculum.py", "orchard/batched.py"):
            src = (ROOT / f).read_text(encoding="utf-8")
            for word in ("autocast", "bfloat16", "is_cuda", "allow_tf32"):
                self.assertNotIn(word, src, "%s: %s" % (f, word))
        from orchard.hardware import setup
        setup(cfg)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_the_launcher_uses_the_configuration(self):
        sh = (ROOT / "cloud_run.sh").read_text(encoding="utf-8")
        self.assertNotIn("configs/gpu_", sh)
        self.assertNotIn(b"\r\n", (ROOT / "cloud_run.sh").read_bytes(),
                         "cloud_run.sh must keep Unix line endings")


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
            cfg = Config()
            u = PopulationUsage(cfg)
            for _ in range(cfg.reward.usage_half_life_updates):
                u.observe({}, batch)
            self.assertAlmostEqual(u.scale, 0.5, places=6, msg="batch %d" % batch)


class TestGrowthDoesNotSpendTheBudget(unittest.TestCase):
    def test_a_rung_budget_counts_from_full_size(self):
        from orchard.train import Trainer
        cfg = Config()
        cfg.world.max_qty = 4
        cfg.channel.max_symbols = 6
        cfg.model.d_model, cfg.model.d_ff = 32, 64
        cfg.population.n_farmers = cfg.population.n_buyers = 3
        cfg.population.founders_farmers = cfg.population.founders_buyers = 1
        cfg.population.grow_every_updates = 2
        cfg.population.turnover = False
        cfg.bottleneck.enabled = False
        cfg.curriculum.start_phase = "name-all"
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

        base = Config()
        base.model.d_model, base.model.d_ff = 32, 64
        base.channel.max_symbols = 6
        rw = ReferentialWorld(base, generator=torch.Generator().manual_seed(0))
        tw = TensorWorld(base, generator=torch.Generator().manual_seed(0))
        for name in ("mutual", "haggle"):
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


class TestTheCodeCanForm(unittest.TestCase):
    """Guards for what stopped the lineup code forming on the GPU."""

    def test_no_hindsight_while_a_code_has_to_form(self):
        # A listener told the answer learns that the (still random) messages
        # carry nothing, and the speaker's gradient dies with it: with hindsight
        # from the start the lineup never left chance.
        from orchard import gumbel
        from orchard.agents import make_agent
        from orchard.curriculum import ReferentialWorld, ladder, phase_named
        from orchard.env import BUYER, FARMER
        cfg = Config()
        cfg.model.d_model, cfg.model.d_ff = 32, 64
        cfg.channel.max_symbols = 4
        names = [p.name for p in ladder(cfg)]
        self.assertLess(names.index("name-all"), names.index(cfg.train.hindsight_from_rung))
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        f = [make_agent(cfg, agent_id=0, role=FARMER, slot=0, generation=0,
                        birth_episode=0, lifespan=10 ** 9)]
        b = [make_agent(cfg, agent_id=1, role=BUYER, slot=0, generation=0,
                        birth_episode=0, lifespan=10 ** 9)]
        z = torch.zeros(16, dtype=torch.long)
        calls = []
        real = gumbel.hindsight_targets
        gumbel.hindsight_targets = lambda *a, **k: calls.append(a[1].name) or real(*a, **k)
        try:
            for name in ("name-fruit", "name-all", "mutual"):
                ph = phase_named(cfg, name)
                for view in ph.views():
                    scen = (rw.sample_mutual(16) if ph.mutual
                            else rw.sample(16, informer=view.informer))
                    gumbel.run_and_update_gumbel(cfg, scen, f, b, z, z, phase=view)
        finally:
            gumbel.hindsight_targets = real
        self.assertEqual(calls, ["mutual"])

    def test_degenerate_flags_after_the_run_has_settled(self):
        # this branch only runs at a settled checkpoint and once crashed a run
        from orchard.metrics import detect_degenerate
        cfg = Config()
        vocab = {"token_entropy_norm": 0.5, "tokens_used": 10, "silent_frac": 0.0,
                 "mean_msg_len": 2.0}
        flags = detect_degenerate(cfg, 0.25, 0.25, vocab, {"mean": 0.0},
                                  10 * cfg.log.checkpoint_every_updates,
                                  ablation={"information_transfer": 0.0})
        self.assertTrue(any("NO COMPOSITIONAL STRUCTURE" in f for f in flags))
        self.assertTrue(any("CHANCE" in f for f in flags))
        self.assertEqual(detect_degenerate(cfg, 0.2, float("nan"), vocab, {"mean": 0.5}, 10 ** 6),
                         [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
