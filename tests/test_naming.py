"""The naming curriculum: one field at a time, then all of them, then held out.

These guard the things the ladder is *for*. A lineup that varies one field has
to vary only that field; the combinations set aside must never be trained on
anywhere, by anybody; a code that names whole things must fail the productivity
gate however well it scores on what it drilled; and the two roles must be one
population speaking one language until trading starts.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.config import Config
from orchard.curriculum import (ASK_ALL, ReferentialWorld, evaluate_rung, ladder,
                                phase_named, rung_budget)
from orchard.env import BUYER, FARMER
from orchard.population import Population
from orchard.world import ComboHoldout, World

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rungs import _swap_evidence          # noqa: E402


class TestTheLadderTeachesOneFieldAtATime(unittest.TestCase):
    def test_the_rungs_come_in_the_intended_order(self):
        cfg = Config()
        names = [p.name for p in ladder(cfg)]
        self.assertEqual(names[:6], ["name-fruit", "name-color", "name-quality",
                                     "name-all", "describe-one", "mutual"])
        self.assertEqual([p.query for p in ladder(cfg)[:3]], [0, 1, 2])
        self.assertIsNone(phase_named(cfg, "name-all").query)
        self.assertTrue(phase_named(cfg, "describe-one").mixed_query)

    def test_a_query_round_varies_only_the_field_it_asks_about(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        for field in range(3):
            rb = rw.sample(512, query=field)
            others = [f for f in range(3) if f != field]
            for f in others:
                col = rb.meanings[:, :, f]
                self.assertTrue(bool((col == col[:, :1]).all()),
                                "a round about field %d also varied field %d"
                                % (field, f))
            asked = rb.meanings[:, :, field]
            for k in range(asked.shape[1]):
                for j in range(k + 1, asked.shape[1]):
                    self.assertTrue(bool((asked[:, k] != asked[:, j]).all()),
                                    "two candidates share the asked-about value")
            self.assertTrue(bool((rb.query == field).all()))

    def test_the_describer_is_told_which_field_is_asked(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(1))
        rb = rw.sample(64, query=1)
        obs = rb.obs(cfg, rb.informer)
        self.assertTrue(bool((obs[:, 3] == 1).all()))
        rb = rw.sample(64)                    # an open round asks for the lot
        self.assertTrue(bool((rb.obs(cfg, rb.informer)[:, 3] == ASK_ALL).all()))

    def test_a_mixed_round_asks_every_field_and_keeps_one_width(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        rb = rw.sample(1024, mixed_query=True)
        self.assertEqual(set(rb.query.tolist()), {0, 1, 2})
        for i in range(0, 1024, 97):
            rows = [tuple(r) for r in rb.meanings[i].tolist()]
            self.assertEqual(len(rows), len(set(rows)))


class TestHeldOutCombinations(unittest.TestCase):
    def test_the_set_is_balanced_over_every_field(self):
        cfg = Config()
        h = ComboHoldout(cfg.world, cfg.world.holdout_combo_frac, cfg.world.holdout_seed)
        self.assertGreater(len(h), 0)
        for name, counts in h.counts().items():
            self.assertEqual(len(set(counts)), 1,
                             "%s is withheld unevenly: %s" % (name, counts))
        # and every value still appears in training
        for i in range(3):
            self.assertEqual(len({c[i] for c in h.training}),
                             len({c[i] for c in h.combos}))

    def test_nothing_trains_on_a_held_out_combination(self):
        cfg = Config()
        h = ComboHoldout(cfg.world, cfg.world.holdout_combo_frac, cfg.world.holdout_seed)
        # the lineup rungs
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(3))
        for kw in ({}, {"query": 0}, {"query": 1}, {"query": 2}, {"mixed_query": True}):
            rb = rw.sample(512, **kw)
            self.assertEqual(int(rb.held_out.sum()), 0, kw)
        # the trading world, scalar and tensor
        w = World(cfg.world, random.Random(0), holdout=h)
        for _ in range(500):
            sc = w.sample()
            self.assertNotIn((sc.buyer.want_variety, sc.buyer.want_color,
                              sc.buyer.min_quality), h)
            for v in range(cfg.world.n_varieties):
                for c in range(cfg.world.n_colors):
                    if sc.farmer.stock_of(v, c) > 0:
                        self.assertNotIn((v, c, sc.farmer.quality_of(v, c)), h,
                                         "a reserved lot turned up in the barn")
        from orchard.batched import TensorWorld
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(4))
        self.assertEqual(int(tw.sample(2048).held_out.sum()), 0)

    def test_held_out_rounds_can_be_drawn_for_evaluation(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(5))
        for kw in ({}, {"query": 1}, {"mixed_query": True}):
            rb = rw.sample(256, held_out=True, **kw)
            self.assertTrue(bool(rb.held_out.all()), kw)

    def test_a_fused_code_cannot_pass_a_naming_rung(self):
        """Perfect on what it drilled, at chance on the rest: not promoted."""
        cfg = Config()
        rung = phase_named(cfg, "name-all")
        lo, _ = rung_budget(cfg, rung)
        ok, checks = evaluate_rung(cfg, rung, _swap_evidence(True, True, holdout=0.40), lo)
        self.assertFalse(ok)
        self.assertFalse(checks["describes combinations it never trained on"]["met"])
        ok, _ = evaluate_rung(cfg, rung, _swap_evidence(True, True, holdout=0.85), lo)
        self.assertTrue(ok)


class TestOnePopulationUntilTrading(unittest.TestCase):
    def test_both_seats_are_the_same_agents_and_never_the_same_one(self):
        cfg = Config()
        pop = Population(cfg, random.Random(0))
        self.assertTrue(pop.shared)
        self.assertIs(pop.farmers, pop.buyers)
        f, b = pop.pair(64)
        self.assertTrue(bool((f != b).all()), "an agent was paired with itself")

    def test_the_split_gives_both_roles_the_same_language(self):
        cfg = Config()
        pop = Population(cfg, random.Random(0))
        n = pop.split_roles(episode=0)
        self.assertEqual(n, len(pop.farmers))
        self.assertFalse(pop.shared)
        self.assertIsNot(pop.farmers, pop.buyers)
        for f, b in zip(pop.farmers, pop.buyers):
            self.assertTrue(torch.equal(f.net.token_head.weight, b.net.token_head.weight))
            self.assertEqual(b.net.role, BUYER)
            self.assertEqual(f.net.role, FARMER)

    def test_the_split_happens_where_trading_starts(self):
        cfg = Config()
        names = [p.name for p in ladder(cfg)]
        at = names.index(cfg.curriculum.split_roles_at)
        self.assertEqual(cfg.curriculum.split_roles_at, "order")
        self.assertTrue(all(not ladder(cfg)[i].trading and not ladder(cfg)[i].order
                            for i in range(at)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
