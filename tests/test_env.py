"""Spec step 1: prove the environment is right before any learning exists.

The critical tests here are the ones that would let the whole experiment be a
lie if they failed:

* agents never receive the other side's private state,
* a successful trade is impossible without mutual understanding,
* the reward cannot be farmed by a fixed convention that ignores the world.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchard.config import Config
from orchard.env import (BUYER, FARMER, Decision, HonestScriptedAgent,
                         RandomScriptedAgent, buyer_obs, farmer_obs, resolve,
                         run_scripted_episode, speaker_of_turn)
from orchard.world import K_EMPTY, World, n_obs_slots, obs_schema


def small_cfg() -> Config:
    cfg = Config()
    cfg.world.max_qty = 8
    cfg.world.n_varieties = 3
    cfg.world.n_price_bins = 8
    cfg.world.reservation_max_bin = 4
    cfg.world.budget_min_bin = 3
    cfg.channel.atomic_vocab = 10
    cfg.channel.max_symbols = 3
    cfg.channel.n_turns = 4
    return cfg


class TestObservationSeparation(unittest.TestCase):
    def test_each_role_sees_only_its_own_state(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(0))
        n = n_obs_slots(cfg.world)
        for _ in range(200):
            sc = w.sample()
            fo, bo = farmer_obs(sc, cfg), buyer_obs(sc, cfg)
            self.assertEqual(len(fo), n)
            self.assertEqual(len(bo), n)
            self.assertEqual(fo[:len(sc.farmer.as_tuple())], sc.farmer.as_tuple())
            self.assertEqual(bo[:len(sc.buyer.as_tuple())], sc.buyer.as_tuple())
            for tup, role in ((fo, FARMER), (bo, BUYER)):
                for i, kind in enumerate(obs_schema(cfg.world, role)):
                    if kind == K_EMPTY:
                        self.assertEqual(tup[i], 0, "padding slot carries a value")

    def test_obs_does_not_leak_viability(self):
        """Neither observation determines whether a deal is possible."""
        cfg = small_cfg()
        w = World(cfg.world, random.Random(1))
        # A barn is detailed enough that two identical ones are rare, so the
        # honest test holds one side fixed and varies the other: neither half of
        # the world settles whether a deal is on.
        from orchard.world import Scenario
        held_f = w.sample_farmer()
        seen = {Scenario(farmer=held_f, buyer=w.sample_buyer()).viable
                for _ in range(400)}
        self.assertEqual(seen, {True, False},
                         "one barn always gives the same answer on viability")
        held_b = w.sample_buyer()
        seen = {Scenario(farmer=w.sample_farmer(), buyer=held_b).viable
                for _ in range(400)}
        self.assertEqual(seen, {True, False},
                         "one shopping list always gives the same answer on viability")

    def test_knowing_one_side_does_not_predict_the_other(self):
        """The property the whole experiment rests on (see the note in world.py).

        Requests are deliberately Zipfian, so the *marginal* is far from uniform
        and a mute agent scores well above 1/n by always guessing the common
        case.  That is fine and is corrected for by the scrambled-channel
        control.  What must not happen is the farmer doing better than that
        marginal by looking at its own barn -- an earlier sampler coerced half of
        all encounters to be compatible and handed out exactly that free lift.
        """
        cfg = small_cfg()
        w = World(cfg.world, random.Random(2))
        n = 20000
        scen = [w.sample() for _ in range(n)]
        nv, nq = cfg.world.n_varieties, cfg.world.max_qty

        # the unconditional base rate: always name the commonest request
        base_v = max(sum(s.buyer.want_variety == v for s in scen)
                     for v in range(nv)) / n
        base_q = max(sum(s.buyer.need_qty == q for s in scen)
                     for q in range(1, nq + 1)) / n

        # the best the farmer can do using its own private state
        def held(s, v):        # the barn's stock of a fruit, over all its colours
            return sum(s.farmer.stock_of(v, c) for c in range(cfg.world.n_colors))

        from_barn = sum(s.buyer.want_variety == max(range(nv),
                                                    key=lambda v: held(s, v))
                        for s in scen) / n
        # conditional accuracy: for each barn shape, guess that barn's commonest request
        best_by_barn: dict[tuple, dict[int, int]] = {}
        for s in scen:
            key = tuple(1 if held(s, v) > 0 else 0 for v in range(nv))
            best_by_barn.setdefault(key, {})
            d = best_by_barn[key]
            d[s.buyer.want_variety] = d.get(s.buyer.want_variety, 0) + 1
        conditional = sum(max(d.values()) for d in best_by_barn.values()) / n

        self.assertLess(from_barn, base_v + 0.03,
                        "the farmer can guess the wanted variety from its own barn")
        self.assertLess(conditional, base_v + 0.03,
                        "the barn shape predicts the request better than the base rate")
        self.assertLess(base_v, 0.75, "requests are so skewed there is nothing to say")
        self.assertLess(base_q, 0.5, "quantities are so skewed there is nothing to say")

    def test_requests_are_skewed_so_length_pressure_has_something_to_act_on(self):
        """Addendum 2.2: a uniform world gives word length nothing to track."""
        cfg = small_cfg()
        cfg.world.zipf_alpha = 0.9          # the skew is a setting; check it bites
        w = World(cfg.world, random.Random(3))
        table = w.meaning_table()
        self.assertGreater(len(table), 8)
        ratio = table[0][1] / table[-1][1]
        self.assertGreater(ratio, 3.0,
                           "meaning frequencies are too flat for a Zipf effect")
        probs = [p for _, p in table]
        self.assertAlmostEqual(sum(probs), 1.0, places=6)


class TestTurnOrder(unittest.TestCase):
    def test_buyer_opens_and_alternates(self):
        self.assertEqual(speaker_of_turn(0), BUYER)
        self.assertEqual(speaker_of_turn(1), FARMER)
        self.assertEqual(speaker_of_turn(2), BUYER)


class TestResolution(unittest.TestCase):
    def test_oracle_pair_succeeds_on_viable_scenarios(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(2))
        rng = random.Random(2)
        n_viable = n_success = 0
        holder = {}
        fa = HonestScriptedAgent(cfg, FARMER, rng, lambda: holder["sc"])
        ba = HonestScriptedAgent(cfg, BUYER, rng, lambda: holder["sc"])
        for _ in range(500):
            sc = w.sample()
            holder["sc"] = sc
            tr = run_scripted_episode(cfg, sc, fa, ba)
            if sc.viable:
                n_viable += 1
                n_success += int(tr.outcome.success)
            else:
                self.assertFalse(tr.outcome.success)
                self.assertEqual(tr.outcome.failure_mode, "correct_no_deal")
        self.assertGreater(n_viable, 50)
        self.assertEqual(n_success, n_viable,
                         "an oracle pair must convert every viable scenario")

    def test_one_sided_perfection_is_not_enough(self):
        """A farmer who knows the right answer still fails if the buyer disagrees."""
        cfg = small_cfg()
        w = World(cfg.world, random.Random(3))
        sc = None
        while sc is None or not sc.viable:
            sc = w.sample()
        lo, hi = sc.zopa
        right = Decision(1, sc.deal_variety, sc.deal_qty, (lo + hi) // 2)
        wrong_qty = (sc.deal_qty % cfg.world.max_qty) + 1
        wrong = Decision(1, sc.deal_variety, wrong_qty, (lo + hi) // 2)
        out = resolve(cfg, sc, right, wrong)
        self.assertFalse(out.success)
        self.assertEqual(out.failure_mode, "qty_mismatch")

    def test_random_pairs_almost_never_succeed(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(4))
        rng = random.Random(4)
        fa = RandomScriptedAgent(cfg, FARMER, rng)
        ba = RandomScriptedAgent(cfg, BUYER, rng)
        successes = sum(run_scripted_episode(cfg, w.sample(), fa, ba).outcome.success
                        for _ in range(2000))
        self.assertLess(successes / 2000.0, 0.02,
                        "chance-level success is too high; the task is too easy")

    def test_fixed_convention_cannot_beat_communication(self):
        """A pair that always accepts a hardcoded deal must score far below an oracle.

        This is the anti-shortcut test from spec 1.3: if a state-independent policy
        scored well, the task would not require language at all.
        """
        cfg = small_cfg()
        w = World(cfg.world, random.Random(5))
        fixed = Decision(1, 0, 1, cfg.world.n_price_bins // 2)
        fixed_score = 0.0
        oracle_score = 0.0
        n = 2000
        for _ in range(n):
            sc = w.sample()
            fixed_score += resolve(cfg, sc, fixed, fixed).farmer_reward
            if sc.viable:
                lo, hi = sc.zopa
                d = Decision(1, sc.deal_variety, sc.deal_qty, (lo + hi) // 2)
            else:
                d = Decision(0, 0, 0, 0)
            oracle_score += resolve(cfg, sc, d, d).farmer_reward
        self.assertLess(fixed_score / n, 0.5 * (oracle_score / n),
                        "a state-independent convention scores too well")

    def test_blanket_rejection_is_a_poor_strategy(self):
        """The trap that sank the first reward design.

        Refusing every deal collects the correct-no-deal payout on unviable
        scenarios at no risk.  If that strategy were competitive, agents would
        park there and the channel would never acquire meaning, so it must score
        far below an oracle -- and below a pair that merely understands each other.
        """
        cfg = small_cfg()
        w = World(cfg.world, random.Random(13))
        always_no = Decision(0, 0, 0, 0)
        reject_score = oracle_score = 0.0
        n = 3000
        for _ in range(n):
            sc = w.sample()
            reject_score += resolve(cfg, sc, always_no, always_no).farmer_reward
            if sc.viable:
                lo, hi = sc.zopa
                d = Decision(1, sc.deal_variety, sc.deal_qty, (lo + hi) // 2)
            else:
                d = Decision(0, 0, 0, 0)
            oracle_score += resolve(cfg, sc, d, d).farmer_reward
        self.assertLess(reject_score / n, 0.4 * (oracle_score / n),
                        "blanket rejection scores too close to playing well")

    def test_understanding_is_rewarded_even_when_the_deal_is_refused(self):
        """Comprehension credit must not be gated behind accepting."""
        cfg = small_cfg()
        w = World(cfg.world, random.Random(14))
        sc = None
        while sc is None or not sc.viable:
            sc = w.sample()
        lo, hi = sc.zopa
        clued = Decision(0, sc.deal_variety, sc.deal_qty, (lo + hi) // 2)
        clueless = Decision(0, (sc.deal_variety + 1) % cfg.world.n_varieties,
                            (sc.deal_qty % cfg.world.max_qty) + 1,
                            (hi + 1) % cfg.world.n_price_bins)
        a = resolve(cfg, sc, clued, clued).farmer_reward
        b = resolve(cfg, sc, clueless, clueless).farmer_reward
        self.assertGreater(a, b, "an agent that understood the deal must score higher "
                                 "than one that did not, even when both walk away")

    def test_reward_is_symmetric_on_task_terms(self):
        cfg = small_cfg()
        cfg.reward.economics = 0.0
        w = World(cfg.world, random.Random(6))
        for _ in range(300):
            sc = w.sample()
            lo, hi = sc.zopa
            d = Decision(1, sc.deal_variety, sc.deal_qty, max(0, min(cfg.world.n_price_bins - 1, (lo + hi) // 2)))
            out = resolve(cfg, sc, d, d)
            self.assertAlmostEqual(out.farmer_reward, out.buyer_reward, places=6)

    def test_economics_is_zero_sum_in_price(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(7))
        sc = None
        while sc is None or not sc.viable or sc.zopa[1] - sc.zopa[0] < 2:
            sc = w.sample()
        lo, hi = sc.zopa
        cheap = resolve(cfg, sc, Decision(1, sc.deal_variety, sc.deal_qty, lo),
                        Decision(1, sc.deal_variety, sc.deal_qty, lo))
        dear = resolve(cfg, sc, Decision(1, sc.deal_variety, sc.deal_qty, hi),
                       Decision(1, sc.deal_variety, sc.deal_qty, hi))
        self.assertTrue(cheap.success and dear.success)
        self.assertGreater(dear.farmer_reward, cheap.farmer_reward)
        self.assertGreater(cheap.buyer_reward, dear.buyer_reward)

    def test_symbol_cost_applies_to_the_speaker_only(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(8))
        sc = w.sample()
        d = Decision(0, 0, 0, 0)
        a = resolve(cfg, sc, d, d, farmer_cost=0.0, buyer_cost=0.0)
        b = resolve(cfg, sc, d, d, farmer_cost=0.15, buyer_cost=0.0)
        self.assertAlmostEqual(a.farmer_reward - b.farmer_reward, 0.15, places=6)
        self.assertAlmostEqual(a.buyer_reward, b.buyer_reward, places=6)


class TestWorld(unittest.TestCase):
    def test_holdout_never_sampled_in_training(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(9))
        self.assertGreater(len(w.holdout), 0)
        for _ in range(3000):
            sc = w.sample(held_out=False)
            self.assertFalse(sc.held_out)
            self.assertFalse(w.is_held_out(sc.buyer.want_variety, sc.buyer.want_color,
                                           sc.buyer.min_quality))

    def test_holdout_sampling_returns_held_out(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(10))
        got = [w.sample(held_out=True) for _ in range(200)]
        self.assertTrue(all(s.held_out for s in got))

    def test_viability_rate_is_balanced(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(11))
        rate = sum(w.sample().viable for _ in range(4000)) / 4000.0
        self.assertGreater(rate, 0.25, "too few viable scenarios to learn accepting")
        self.assertLess(rate, 0.80, "too few unviable scenarios to learn rejecting")


class TestTranscript(unittest.TestCase):
    def test_utterances_strip_pad_and_respect_length(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(12))
        rng = random.Random(12)
        tr = run_scripted_episode(cfg, w.sample(),
                                  RandomScriptedAgent(cfg, FARMER, rng),
                                  RandomScriptedAgent(cfg, BUYER, rng))
        self.assertEqual(len(tr.tokens), cfg.channel.dialogue_len)
        for turn in range(cfg.channel.n_turns):
            utt = tr.utterance(cfg, turn)
            self.assertLessEqual(len(utt), cfg.channel.max_msg_len)
            self.assertNotIn(cfg.channel.pad_id, utt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
