"""The tensor path must agree with the scalar path exactly.

The scalar implementation in env.py is the definition of the rules and stays
readable; the tensor one exists so a GPU is not waiting on the interpreter. If
they ever disagree, the scalar one is right. These tests are what make it safe to
run the fast path by default.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.batched import ScenarioBatch, TensorWorld, resolve_batch
from orchard.config import Config
from testscale import method_at_test_scale
from orchard.env import BUYER, FARMER, Beliefs, Decision, buyer_obs, farmer_obs, resolve
from orchard.world import World


def cfg_small() -> Config:
    cfg = method_at_test_scale()
    cfg.world.n_varieties = 3
    cfg.world.max_qty = 8
    cfg.world.n_price_bins = 8
    cfg.world.reservation_max_bin = 6
    cfg.world.budget_min_bin = 1
    cfg.world.zipf_alpha = 0.3
    return cfg


class TestScenarioBatch(unittest.TestCase):
    def test_derived_flags_match_the_scalar_scenario(self):
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(0))
        sb = tw.sample(512)
        for i in range(0, 512, 7):
            sc = sb.scenario(i)
            self.assertEqual(int(sb.offered_stock[i]), sc.offered_stock)
            self.assertEqual(int(sb.offered_quality[i]), sc.offered_quality)
            self.assertEqual(bool(sb.variety_ok[i]), sc.variety_ok)
            self.assertEqual(bool(sb.stock_ok[i]), sc.stock_ok)
            self.assertEqual(bool(sb.quality_ok[i]), sc.quality_ok)
            self.assertEqual(bool(sb.price_ok[i]), sc.price_ok)
            self.assertEqual(bool(sb.viable[i]), sc.viable)

    def test_observations_match(self):
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(1))
        sb = tw.sample(256)
        f = sb.obs(cfg, FARMER)
        b = sb.obs(cfg, BUYER)
        for i in range(0, 256, 11):
            sc = sb.scenario(i)
            self.assertEqual(tuple(int(x) for x in f[i]), farmer_obs(sc, cfg))
            self.assertEqual(tuple(int(x) for x in b[i]), buyer_obs(sc, cfg))

    def test_holdout_is_respected_in_both_directions(self):
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(2))
        train = tw.sample(4000, held_out=False)
        self.assertFalse(bool(train.held_out.any()),
                         "a reserved request leaked into training")
        test = tw.sample(2000, held_out=True)
        self.assertTrue(bool(test.held_out.all()),
                        "zero-shot draw returned non-reserved requests")

    def test_batch_size_is_exact(self):
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(3))
        for n in (1, 7, 333):
            self.assertEqual(len(tw.sample(n)), n)

    def test_distributions_match_the_scalar_world(self):
        """Same marginals, drawn a different way."""
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(4))
        sb = tw.sample(40000)
        ref = World(cfg.world, random.Random(4))
        scal = [ref.sample() for _ in range(40000)]

        self.assertAlmostEqual(float(sb.viable.float().mean()),
                               sum(s.viable for s in scal) / len(scal), delta=0.02)
        self.assertAlmostEqual(float(sb.need_qty.float().mean()),
                               sum(s.buyer.need_qty for s in scal) / len(scal), delta=0.15)
        self.assertAlmostEqual(float(sb.offered_stock.float().mean()),
                               sum(s.offered_stock for s in scal) / len(scal), delta=0.2)
        self.assertAlmostEqual(float(sb.reservation.float().mean()),
                               sum(s.farmer.reservation for s in scal) / len(scal), delta=0.15)

    def test_the_two_sides_stay_independent(self):
        """The property the whole world design exists to protect."""
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(5))
        sb = tw.sample(40000)
        nv = cfg.world.n_varieties
        base = max(float((sb.want_variety == v).float().mean()) for v in range(nv))
        biggest_line = sb.stocks.argmax(dim=1)
        from_barn = float((sb.want_variety == biggest_line).float().mean())
        self.assertLess(from_barn, base + 0.03,
                        "the farm's own stock predicts the shopper's request")


class TestResolveAgreement(unittest.TestCase):
    def _compare(self, cfg, n=1500, seed=0, with_beliefs=True):
        g = torch.Generator().manual_seed(seed)
        tw = TensorWorld(cfg, generator=g)
        sb = tw.sample(n)
        w = cfg.world
        f_dec = torch.stack([
            torch.randint(0, 2, (n,), generator=g),
            torch.randint(0, w.n_varieties, (n,), generator=g),
            torch.randint(0, w.max_qty + 1, (n,), generator=g),
            torch.randint(0, w.n_price_bins, (n,), generator=g)], dim=1)
        b_dec = torch.stack([
            torch.randint(0, 2, (n,), generator=g),
            torch.randint(0, w.n_varieties, (n,), generator=g),
            torch.randint(0, w.max_qty + 1, (n,), generator=g),
            torch.randint(0, w.n_price_bins, (n,), generator=g)], dim=1)
        bel = lambda: torch.stack([
            torch.randint(0, w.n_varieties, (n,), generator=g),
            torch.randint(0, w.max_qty + 1, (n,), generator=g),
            torch.randint(0, w.n_quality, (n,), generator=g),
            torch.randint(0, w.n_price_bins, (n,), generator=g)], dim=1)
        f_bel, b_bel = (bel(), bel()) if with_beliefs else (None, None)
        f_sym = torch.randint(0, 9, (n,), generator=g)
        b_sym = torch.randint(0, 9, (n,), generator=g)

        res = resolve_batch(cfg, sb, f_dec, b_dec, f_sym, b_sym,
                            f_bel=f_bel, b_bel=b_bel)
        for i in range(n):
            sc = sb.scenario(i)
            fd = Decision(*[int(x) for x in f_dec[i]])
            bd = Decision(*[int(x) for x in b_dec[i]])
            fb = Beliefs(*[int(x) for x in f_bel[i]]) if with_beliefs else None
            bb = Beliefs(*[int(x) for x in b_bel[i]]) if with_beliefs else None
            o = resolve(cfg, sc, fd, bd, int(f_sym[i]), int(b_sym[i]),
                        f_beliefs=fb, b_beliefs=bb)
            self.assertAlmostEqual(float(res["farmer_reward"][i]), o.farmer_reward,
                                   places=4, msg="farmer reward differs at %d" % i)
            self.assertAlmostEqual(float(res["buyer_reward"][i]), o.buyer_reward,
                                   places=4, msg="buyer reward differs at %d" % i)
            self.assertEqual(bool(res["success"][i]), o.success)
            self.assertEqual(bool(res["comprehended"][i]), o.comprehended)
            self.assertEqual(bool(res["both_judged"][i]), o.both_judged_viability)
            self.assertAlmostEqual(float(res["farmer_decode"][i]), o.farmer_decode,
                                   places=5)
            self.assertAlmostEqual(float(res["buyer_decode"][i]), o.buyer_decode,
                                   places=5)
            if o.success:
                self.assertEqual(int(res["traded_qty"][i]), o.traded_qty)
                self.assertAlmostEqual(float(res["trade_value"][i]), o.trade_value,
                                       places=4)
                self.assertAlmostEqual(float(res["farmer_profit"][i]), o.farmer_profit,
                                       places=4)

    def test_matches_scalar_with_beliefs(self):
        self._compare(cfg_small(), seed=0)

    def test_matches_scalar_without_beliefs(self):
        cfg = cfg_small()
        cfg.reward.belief_heads = False
        self._compare(cfg, n=800, seed=1, with_beliefs=False)

    def test_matches_scalar_with_tolerances(self):
        cfg = cfg_small()
        cfg.reward.qty_tol = 1
        cfg.reward.belief_qty_tol = 2
        cfg.reward.belief_price_tol = 0
        self._compare(cfg, n=800, seed=2)

    def test_matches_scalar_on_a_bigger_world(self):
        cfg = method_at_test_scale()      # the original spec's bigger world
        cfg.world.n_varieties, cfg.world.max_qty = 4, 20
        cfg.world.n_price_bins, cfg.world.reservation_max_bin = 12, 9
        cfg.world.zipf_alpha = 0.9
        self._compare(cfg, n=600, seed=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
