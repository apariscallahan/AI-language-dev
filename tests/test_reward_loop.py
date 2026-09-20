"""The communication loop must close in both directions.

These guard the fix for a measured asymmetry: the buyer used to collect 91% of its
comprehension score (2.204 of 2.428) just by restating its own want and need, so
it had no reason to listen, and *neither* role had any reward term for having been
understood. A speaker with no stake in being understood has no gradient telling it
to be informative, whatever else the system does.
"""
from __future__ import annotations

import random
import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchard.config import Config
from orchard.env import (BUYER, FARMER, Beliefs, Decision, decode_hits,
                         decode_score, resolve)
from orchard.world import World


def cfg_small() -> Config:
    cfg = Config()
    cfg.world.n_varieties = 3
    cfg.world.max_qty = 8
    cfg.world.n_price_bins = 8
    cfg.world.reservation_max_bin = 6
    cfg.world.budget_min_bin = 1
    cfg.world.zipf_alpha = 0.3
    return cfg


def truth(sc, role) -> Beliefs:
    if role == FARMER:
        b = sc.buyer
        return Beliefs(b.want_variety, b.need_qty, b.min_quality, b.max_price,
                       color=b.want_color)
    return Beliefs(sc.buyer.want_variety, sc.offered_stock, sc.offered_quality,
                   sc.farmer.reservation, color=sc.offered_color)


def blind(sc, cfg, role) -> Beliefs:
    """The best an agent can do from its own half plus the world's marginals."""
    if role == FARMER:
        return Beliefs(0, 1, 0, cfg.world.n_price_bins - 2, color=0)
    return Beliefs(sc.buyer.want_variety, cfg.world.max_qty // 2,
                   cfg.world.n_quality - 1, 1, color=sc.buyer.want_color)


def good_deal(sc) -> Decision:
    if not sc.viable:
        return Decision(0, sc.deal_variety, 0, 0)
    lo, hi = sc.zopa
    return Decision(1, sc.deal_variety, sc.deal_qty, (lo + hi) // 2)


class TestLoopCloses(unittest.TestCase):
    """Each role must be paid for the OTHER having read it."""

    def _stakes(self, cfg, n=6000, seed=0):
        w = World(cfg.world, random.Random(seed))
        f_stake, b_stake = [], []
        for _ in range(n):
            sc = w.sample()
            d = good_deal(sc)
            ft, bt = truth(sc, FARMER), truth(sc, BUYER)
            fb, bb = blind(sc, cfg, FARMER), blind(sc, cfg, BUYER)
            # the farmer's reward, varying ONLY whether the buyer read the farmer
            heard = resolve(cfg, sc, d, d, f_beliefs=ft, b_beliefs=bt).farmer_reward
            unheard = resolve(cfg, sc, d, d, f_beliefs=ft, b_beliefs=bb).farmer_reward
            f_stake.append(heard - unheard)
            # the buyer's reward, varying ONLY whether the farmer read the buyer
            heard = resolve(cfg, sc, d, d, f_beliefs=ft, b_beliefs=bt).buyer_reward
            unheard = resolve(cfg, sc, d, d, f_beliefs=fb, b_beliefs=bt).buyer_reward
            b_stake.append(heard - unheard)
        return statistics.fmean(f_stake), statistics.fmean(b_stake)

    def test_both_roles_are_paid_for_being_understood(self):
        cfg = cfg_small()
        f, b = self._stakes(cfg)
        self.assertGreater(f, 0.05, "the farmer gains nothing from being understood")
        self.assertGreater(b, 0.05, "the buyer gains nothing from being understood")

    def test_the_two_stakes_are_comparable(self):
        """Neither side should have a much weaker reason to be informative."""
        cfg = cfg_small()
        f, b = self._stakes(cfg)
        ratio = min(f, b) / max(f, b)
        self.assertGreater(ratio, 0.5,
                           "one role's stake in being understood is less than half "
                           "the other's (%.3f vs %.3f)" % (f, b))

    def test_being_understood_is_separable_from_trade_outcome(self):
        """The term must not just be joint trade success under another name.

        Hold the deal fixed and unsuccessful, and the reward should still move
        with whether the partner read us.
        """
        cfg = cfg_small()
        w = World(cfg.world, random.Random(3))
        moved = 0
        for _ in range(500):
            sc = w.sample()
            walk = Decision(0, 0, 0, 0)          # nobody trades, outcome identical
            a = resolve(cfg, sc, walk, walk, f_beliefs=truth(sc, FARMER),
                        b_beliefs=truth(sc, BUYER))
            b = resolve(cfg, sc, walk, walk, f_beliefs=truth(sc, FARMER),
                        b_beliefs=blind(sc, cfg, BUYER))
            self.assertEqual(a.success, b.success)
            if a.farmer_reward != b.farmer_reward:
                moved += 1
        self.assertGreater(moved, 400,
                           "with the trade held fixed, being understood changed "
                           "nothing -- the term is redundant with trade outcome")


class TestDecodingNeedsTheChannel(unittest.TestCase):
    def test_neither_role_can_read_the_other_from_its_own_state(self):
        cfg = cfg_small()
        w = World(cfg.world, random.Random(1))
        n = 6000
        got = {FARMER: [], BUYER: []}
        blind_got = {FARMER: [], BUYER: []}
        for _ in range(n):
            sc = w.sample()
            for role in (FARMER, BUYER):
                got[role].append(decode_score(truth(sc, role), sc, role, cfg))
                blind_got[role].append(decode_score(blind(sc, cfg, role), sc, role, cfg))
        for role, name in ((FARMER, "farmer"), (BUYER, "buyer")):
            perfect = statistics.fmean(got[role])
            uninformed = statistics.fmean(blind_got[role])
            self.assertAlmostEqual(perfect, 1.0, places=6)
            self.assertLess(uninformed, 0.75,
                            "%s can already read the other without listening "
                            "(%.3f)" % (name, uninformed))
            self.assertGreater(perfect - uninformed, 0.25,
                               "%s gains too little from actually listening" % name)

    def test_every_scored_field_is_hidden_from_the_agent_scoring_it(self):
        """A field the agent could see would make this a free lottery again."""
        cfg = cfg_small()
        w = World(cfg.world, random.Random(2))
        for _ in range(200):
            sc = w.sample()
            # the farmer is asked only about buyer-side facts
            f = truth(sc, FARMER).as_tuple()
            self.assertEqual(f, (sc.buyer.want_variety, sc.buyer.need_qty,
                                 sc.buyer.min_quality, sc.buyer.max_price))
            # the buyer is asked only about farmer-side facts: how much of the
            # lot there is, its quality and colour, and the farmer's floor price
            hits = decode_hits(truth(sc, BUYER), sc, BUYER, cfg)
            self.assertEqual(len(hits), 4)
            self.assertTrue(all(hits))

    def test_tolerances_are_respected(self):
        cfg = cfg_small()
        cfg.reward.belief_qty_tol = 1
        w = World(cfg.world, random.Random(4))
        sc = w.sample()
        b = truth(sc, FARMER)
        near = Beliefs(b.variety, b.qty + 1, b.quality, b.price)
        far = Beliefs(b.variety, b.qty + 3, b.quality, b.price)
        self.assertTrue(decode_hits(near, sc, FARMER, cfg)[1])
        self.assertFalse(decode_hits(far, sc, FARMER, cfg)[1])


class TestWorldOverlap(unittest.TestCase):
    """Task 2: deals should usually be possible, without either side leaking."""

    def test_a_healthy_majority_of_rounds_are_viable(self):
        cfg = cfg_small()
        w = World(cfg.world, random.Random(5))
        n = 20000
        scen = [w.sample() for _ in range(n)]
        rate = sum(s.viable for s in scen) / n
        self.assertGreater(rate, 0.62, "too few deals are possible to practise on")
        self.assertLess(rate, 0.88, "walking away has stopped being a real outcome")

    def test_no_single_cause_dominates_the_failures(self):
        cfg = cfg_small()
        w = World(cfg.world, random.Random(6))
        scen = [w.sample() for _ in range(20000)]
        nv = [s for s in scen if not s.viable]
        self.assertGreater(len(nv), 100)
        causes = [
            sum(not s.variety_ok for s in nv),
            sum(s.variety_ok and not s.stock_ok for s in nv),
            sum(s.variety_ok and s.stock_ok and not s.quality_ok for s in nv),
            sum(s.variety_ok and s.stock_ok and s.quality_ok and not s.price_ok
                for s in nv),
        ]
        self.assertLess(max(causes) / len(nv), 0.6,
                        "no-deal has collapsed onto a single reason")
        self.assertTrue(all(c > 0 for c in causes), "a failure mode never occurs")

    def test_both_sides_stay_independent_after_the_overlap_widening(self):
        """Widening overlap must not be done by correlating the two draws."""
        cfg = cfg_small()
        w = World(cfg.world, random.Random(7))
        n = 20000
        scen = [w.sample() for _ in range(n)]
        nv = cfg.world.n_varieties
        base = max(sum(s.buyer.want_variety == v for s in scen) for v in range(nv)) / n
        from_barn = sum(
            s.buyer.want_variety == max(
                range(nv), key=lambda v: sum(s.farmer.stock_of(v, c)
                                             for c in range(cfg.world.n_colors)))
            for s in scen) / n
        self.assertLess(from_barn, base + 0.03,
                        "the farmer's own stock now predicts what the buyer wants")


class TestCheckpointing(unittest.TestCase):
    def test_the_final_checkpoint_is_not_a_duplicate(self):
        """A repeat measurement at the same episode reads as zero drift.

        The loop fires a checkpoint at exactly train.episodes and the final one
        then fired again with nothing changed in between, so the form-drift
        measurement compared a snapshot against itself and reported 0.000 -- next
        to a cumulative count of 136 replacements, which made no sense.
        """
        import os
        import shutil
        import tempfile

        import torch

        from orchard.train import Trainer

        cfg = cfg_small()
        cfg.channel.atomic_vocab = 8
        cfg.channel.max_symbols = 3
        cfg.channel.n_turns = 2
        cfg.model.d_model = 32
        cfg.model.d_ff = 64
        cfg.population.n_farmers = cfg.population.n_buyers = 1
        cfg.population.turnover = False
        cfg.bottleneck.enabled = False
        cfg.train.episodes = 128
        cfg.train.batch_size = 64
        cfg.log.checkpoint_every_updates = 1
        cfg.log.topsim_samples = 20
        cfg.log.intelligibility_episodes = 40
        cfg.log.zeroshot_episodes = 40
        cfg.log.ablation_episodes = 40
        cfg.log.stability_probes = 4
        cfg.log.plot = False

        out = tempfile.mkdtemp(prefix="orchard_ckpt_")
        try:
            torch.manual_seed(0)
            t = Trainer(cfg, out, quiet=True)
            t.run()
            episodes = [r["episode"] for r in t.metrics_log.rows]
            t.close()
            self.assertEqual(len(episodes), len(set(episodes)),
                             "a checkpoint was written twice at the same episode: %s"
                             % episodes)
            self.assertTrue(t.metrics_log.rows[-1]["final"])
        finally:
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
