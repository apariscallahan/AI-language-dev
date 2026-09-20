"""Tests for the curriculum and for the widened transmission bottleneck.

The two properties worth protecting here are the ones that would quietly undo
each change: that a phase transition really does carry the weights forward rather
than starting again, and that a newborn really does see the common vocabulary
rather than a sliver of it.
"""
from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.agents import make_agent, sequence_len
from orchard.bottleneck import StoredEpisode, TranscriptStore
from orchard.batched import TensorWorld
from orchard.config import Config
from orchard.curriculum import (H_CHOICE, N_HEADS, CurriculumState, Phase,
                                ReferentialWorld, ladder, phase_named, phase_schema,
                                promotion_for, resolve_referential)
from orchard.env import BUYER, FARMER
from orchard.gumbel import run_and_update_gumbel
from orchard.world import K_EMPTY, n_obs_slots


def cfg_small() -> Config:
    cfg = Config()
    cfg.world.n_varieties = 3
    cfg.world.max_qty = 8
    cfg.world.n_price_bins = 8
    cfg.world.reservation_max_bin = 6
    cfg.world.budget_min_bin = 1
    cfg.channel.atomic_vocab = 16
    cfg.channel.max_symbols = 4
    cfg.channel.n_turns = 4
    cfg.model.d_model = 48
    cfg.model.d_ff = 96
    return cfg


def agents(cfg, n=2):
    f = [make_agent(cfg, agent_id=i, role=FARMER, slot=i, generation=0,
                    birth_episode=0, lifespan=10**9) for i in range(n)]
    b = [make_agent(cfg, agent_id=50 + i, role=BUYER, slot=i, generation=0,
                    birth_episode=0, lifespan=10**9) for i in range(n)]
    return f, b


class TestLadder(unittest.TestCase):
    def test_the_hard_task_comes_last(self):
        cfg = cfg_small()
        phases = ladder(cfg)
        self.assertTrue(phases[0].referential)
        self.assertFalse(phases[0].use_price)
        self.assertFalse(phases[0].use_market)
        # the full economy is the final rung, not the first
        self.assertTrue(phases[-1].use_market)
        self.assertEqual(phases[-1].name, "market")
        # difficulty is monotone in each dimension it adds
        self.assertLessEqual(phases[0].n_turns, phases[1].n_turns)
        self.assertLessEqual(phases[1].n_turns, phases[2].n_turns)
        for i, p in enumerate(phases):
            self.assertEqual(p.index, i)
            self.assertLessEqual(p.n_turns, cfg.channel.n_turns)

    def test_the_informer_speaks_first_in_the_lineup(self):
        p = ladder(cfg_small())[0]
        self.assertEqual(p.speaker_of_turn(0), FARMER)
        # and the buyer opens once there is something to ask for
        self.assertEqual(phase_named(cfg_small(), "haggle").speaker_of_turn(0), BUYER)

    def test_only_relevant_heads_are_scored(self):
        cfg = cfg_small()
        refer, haggle = ladder(cfg)[0], phase_named(cfg, "haggle")
        self.assertEqual(refer.active_heads(BUYER, cfg), [H_CHOICE])
        self.assertEqual(refer.active_heads(FARMER, cfg), [],
                         "the informer has no decision in a lineup game")
        self.assertIn(0, haggle.active_heads(BUYER, cfg))
        self.assertNotIn(H_CHOICE, haggle.active_heads(BUYER, cfg))


class TestOneArchitectureEveryPhase(unittest.TestCase):
    """Weights can only carry forward if nothing about the shape changes."""

    def test_layout_fits_every_role_in_every_phase(self):
        cfg = cfg_small()
        n = n_obs_slots(cfg.world, cfg)
        for phase in ladder(cfg):
            for role in (FARMER, BUYER):
                schema = phase_schema(cfg, role, phase)
                self.assertEqual(len(schema), n,
                                 "%s/%s schema is not the shared layout"
                                 % (phase.name, role))

    def test_the_lineup_needs_the_most_room(self):
        cfg = cfg_small()
        cfg.curriculum.n_candidates = 6
        self.assertGreaterEqual(n_obs_slots(cfg.world, cfg), 3 * 6)

    def test_weights_carry_across_every_transition(self):
        """The same modules keep training; nothing is reinitialised."""
        cfg = cfg_small()
        torch.manual_seed(0)
        f, b = agents(cfg)
        ids = [id(a.net) for a in f + b]
        before = f[0].net.token_head.weight.detach().clone()

        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(1))
        B = 128
        i = torch.arange(B)
        fi, bi = i % 2, torch.div(i, 2, rounding_mode="floor") % 2

        for phase in ladder(cfg):
            if phase.referential:
                scen = rw.sample(B)
            elif phase.mutual:
                scen = rw.sample_mutual(B)
            else:
                scen = tw.sample(B)
            batch, stats = run_and_update_gumbel(cfg, scen, f, b, fi, bi,
                                                 update=200, phase=phase)
            self.assertEqual(stats.policy_loss, stats.policy_loss)   # finite
            self.assertEqual(tuple(batch.f_dec.shape), (B, N_HEADS))

        self.assertEqual(ids, [id(a.net) for a in f + b],
                         "a phase transition replaced an agent's network")
        self.assertFalse(torch.allclose(before, f[0].net.token_head.weight.detach()),
                         "training did not actually move the carried weights")

    def test_unused_turns_stay_empty(self):
        cfg = cfg_small()
        torch.manual_seed(1)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        B = 64
        i = torch.arange(B)
        batch, _ = run_and_update_gumbel(
            cfg, rw.sample(B), f, b, i % 2, torch.div(i, 2, rounding_mode="floor") % 2,
            update=0, phase=ladder(cfg)[0])
        L = cfg.channel.max_symbols
        for turn in range(1, cfg.channel.n_turns):
            seg = batch.tokens[:, turn * L:(turn + 1) * L]
            self.assertTrue(bool((seg == cfg.channel.pad_id).all()),
                            "a one-turn phase wrote into turn %d" % turn)


class TestLineupGame(unittest.TestCase):
    def test_distractors_always_differ_from_the_target(self):
        """Otherwise a correct guess could be luck rather than information."""
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(3))
        rb = rw.sample(2000)
        true = rb.true_meaning
        K = cfg.curriculum.n_candidates
        for k in range(K):
            same = (rb.meanings[:, k] == true).all(dim=1)
            is_target = rb.target == k
            self.assertTrue(bool((same == is_target).all()),
                            "a distractor duplicated the target")

    def test_the_target_position_carries_no_information(self):
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(4))
        rb = rw.sample(20000)
        K = cfg.curriculum.n_candidates
        for k in range(K):
            share = float((rb.target == k).float().mean())
            self.assertAlmostEqual(share, 1.0 / K, delta=0.02)

    def test_informer_and_guesser_see_different_things(self):
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(5))
        rb = rw.sample(256)
        f = rb.obs(cfg, FARMER)
        g = rb.obs(cfg, BUYER)
        n = n_obs_slots(cfg.world, cfg)
        self.assertEqual(tuple(f.shape), (256, n))
        self.assertEqual(tuple(g.shape), (256, n))
        # the informer sees exactly the thing it must describe and which field
        # is being asked about -- and nothing at all about the lineup
        self.assertTrue(bool((f[:, :3] == rb.true_meaning).all()))
        self.assertTrue(bool((f[:, 3] == rb.query).all()))
        self.assertTrue(bool((f[:, 4:] == 0).all()), "informer saw the lineup")
        # the guesser sees the candidates and the query, not the answer
        self.assertTrue(bool((g[:, :3 * rb.meanings.shape[1]]
                              == rb.meanings.reshape(256, -1)).all()))

    def test_scoring_is_the_guess(self):
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(6))
        rb = rw.sample(500)
        zero = torch.zeros(500, dtype=torch.long)
        right = resolve_referential(cfg, rb, rb.target, zero, zero)
        wrong = resolve_referential(
            cfg, rb, (rb.target + 1) % cfg.curriculum.n_candidates, zero, zero)
        self.assertTrue(bool(right["success"].all()))
        self.assertFalse(bool(wrong["success"].any()))
        self.assertGreater(float(right["farmer_reward"].mean()),
                           float(wrong["farmer_reward"].mean()),
                           "the informer is not paid for being understood")
        self.assertGreater(float(right["buyer_reward"].mean()),
                           float(wrong["buyer_reward"].mean()))

    def test_symbols_still_cost(self):
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(7))
        rb = rw.sample(100)
        zero = torch.zeros(100, dtype=torch.long)
        five = torch.full((100,), 5, dtype=torch.long)
        # the resolvers are handed the length cost itself, already computed
        quiet = resolve_referential(cfg, rb, rb.target, zero.float(), zero.float())
        chatty = resolve_referential(cfg, rb, rb.target, five.float(), zero.float())
        self.assertAlmostEqual(
            float(quiet["farmer_reward"].mean() - chatty["farmer_reward"].mean()),
            5.0, places=5)
        self.assertAlmostEqual(
            float(quiet["buyer_reward"].mean() - chatty["buyer_reward"].mean()),
            0.0, places=5)


class TestTheGuesserActuallyListens(unittest.TestCase):
    """The bug this guards against made the lineup unlearnable and looked fine.

    The encoder is causal, so a candidate's slots sit early in the sequence and
    cannot attend forward to the message. A choice head that scored hidden states
    at those slots was therefore scoring something that had heard nothing: the
    guess was independent of what was said, success sat at chance forever, and
    every other metric looked unremarkable.
    """

    def _net(self, cfg):
        from orchard.agents import CommNet
        torch.manual_seed(0)
        return CommNet(cfg, BUYER).eval()

    def test_the_choice_changes_when_the_message_changes(self):
        cfg = cfg_small()
        net = self._net(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        rb = rw.sample(128)
        obs = rb.obs(cfg, BUYER)
        schema = phase_schema(cfg, BUYER, ladder(cfg)[0])
        D = cfg.channel.dialogue_len

        silent = torch.full((128, D), cfg.channel.pad_id, dtype=torch.long)
        spoken = silent.clone()
        spoken[:, :3] = torch.randint(0, cfg.channel.atomic_vocab, (128, 3))

        with torch.no_grad():
            a = net.all_heads(net.encode(obs, silent, schema=schema)[:, -1], obs)[H_CHOICE]
            b = net.all_heads(net.encode(obs, spoken, schema=schema)[:, -1], obs)[H_CHOICE]
        self.assertEqual(tuple(a.shape), (128, cfg.curriculum.n_candidates))
        self.assertFalse(torch.allclose(a, b, atol=1e-7),
                         "the guess does not depend on the message at all")

    def test_the_choice_changes_when_the_candidates_change(self):
        cfg = cfg_small()
        net = self._net(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(1))
        obs_a = rw.sample(128).obs(cfg, BUYER)
        obs_b = rw.sample(128).obs(cfg, BUYER)
        schema = phase_schema(cfg, BUYER, ladder(cfg)[0])
        toks = torch.full((128, cfg.channel.dialogue_len), cfg.channel.pad_id,
                          dtype=torch.long)
        toks[:, :3] = 1
        with torch.no_grad():
            a = net.all_heads(net.encode(obs_a, toks, schema=schema)[:, -1], obs_a)[H_CHOICE]
            b = net.all_heads(net.encode(obs_b, toks, schema=schema)[:, -1], obs_b)[H_CHOICE]
        self.assertFalse(torch.allclose(a, b, atol=1e-7),
                         "the guess ignores which candidates are on offer")

    def test_gradient_reaches_the_informer_through_the_guess(self):
        """The informer has no decision of its own; this path is all it has."""
        cfg = cfg_small()
        torch.manual_seed(2)
        f, b = agents(cfg, 1)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        grads = {}
        orig = b[0].opt.step

        def capture():
            g = f[0].net.token_head.weight.grad
            grads["informer"] = float(g.abs().sum()) if g is not None else 0.0
            return orig()
        b[0].opt.step = capture

        B = 128
        i = torch.arange(B)
        run_and_update_gumbel(cfg, rw.sample(B), f, b, i * 0, i * 0,
                              update=0, phase=ladder(cfg)[0])
        self.assertGreater(grads.get("informer", 0.0), 0.0,
                           "no gradient reached the informer's message head")


class TestPromotion(unittest.TestCase):
    def _rule(self, cfg=None):
        cfg = cfg or cfg_small()
        return promotion_for(cfg, ladder(cfg)[0])

    def test_all_criteria_must_hold(self):
        rule = self._rule()
        good = dict(success=0.9, chance=0.25, topsim=0.4, null=0.0, transfer=0.5,
                    updates_in_phase=10 ** 6)
        passed, _ = rule.evaluate(**good)
        self.assertTrue(passed)
        for spoil, val in (("success", 0.26), ("topsim", 0.0), ("transfer", 0.0),
                           ("updates_in_phase", 0)):
            bad = dict(good)
            bad[spoil] = val
            passed, checks = rule.evaluate(**bad)
            self.assertFalse(passed, "promoted despite %s = %s" % (spoil, val))

    def test_a_nan_never_counts_as_met(self):
        rule = self._rule()
        passed, checks = rule.evaluate(
            success=float("nan"), chance=0.25, topsim=float("nan"),
            null=float("nan"), transfer=float("nan"), updates_in_phase=10 ** 6)
        self.assertFalse(passed)
        self.assertFalse(checks["success above floor"]["met"])
        self.assertFalse(checks["channel actually carries"]["met"])

    def test_success_alone_is_not_enough(self):
        """A pair can score on base rates without saying anything."""
        rule = self._rule()
        passed, _ = rule.evaluate(success=0.95, chance=0.25, topsim=0.0, null=0.0,
                                  transfer=0.0, updates_in_phase=10 ** 6)
        self.assertFalse(passed)

    def test_state_advances_and_records_why(self):
        cfg = cfg_small()
        st = CurriculumState(ladder(cfg))
        self.assertEqual(st.phase.name, "name-fruit")
        st.episodes_in_phase = 999
        st.updates_in_phase = 99
        st.advance(1234, {"success above floor": {"met": True, "detail": "0.9"}})
        self.assertEqual(st.phase.name, "name-color")
        self.assertEqual(st.episodes_in_phase, 0)
        self.assertEqual(st.updates_in_phase, 0)
        self.assertEqual(len(st.transitions), 1)
        self.assertEqual(st.transitions[0]["from"], "name-fruit")
        self.assertEqual(st.transitions[0]["episode"], 1234)
        self.assertIn("criteria", st.transitions[0])

    def test_it_stops_at_the_last_rung(self):
        cfg = cfg_small()
        st = CurriculumState(ladder(cfg))
        for _ in range(len(ladder(cfg)) + 3):
            st.advance(0, {})
        self.assertEqual(st.phase.name, "market")
        self.assertTrue(st.finished)


# ==========================================================================
class TestTheStoreActuallyFills(unittest.TestCase):
    """A silent regression that killed every population run.

    The tensor path keeps its outcome in ``batch.res`` and builds no per-episode
    Outcome objects, so a store that iterated ``batch.outcomes`` filed nothing.
    Newborns then got an empty curriculum and started from random weights, and
    with turnover on the population reset itself every lifespan. Nothing failed
    loudly; the runs just never learned.
    """

    def _run(self, cfg, phase, scen_fn):
        torch.manual_seed(0)
        f, b = agents(cfg, 2)
        store = TranscriptStore(cfg)
        B = 256
        i = torch.arange(B)
        fi, bi = i % 2, torch.div(i, 2, rounding_mode="floor") % 2
        for _ in range(3):
            batch, _ = run_and_update_gumbel(cfg, scen_fn(B), f, b, fi, bi,
                                             update=500, phase=phase)
            store.add_batch(batch, f, b, 0)
        return store, batch

    def test_it_fills_from_a_trading_batch(self):
        cfg = cfg_small()
        cfg.bottleneck.only_successful = False      # untrained agents rarely succeed
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(0))
        store, _ = self._run(cfg, ladder(cfg)[-1], tw.sample)
        self.assertGreater(len(store), 0, "the transcript store stayed empty")
        self.assertGreater(len(store.meaning_counts), 1,
                           "every stored episode got the same meaning key")

    def test_it_fills_from_a_lineup_batch(self):
        cfg = cfg_small()
        cfg.bottleneck.only_successful = False
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(1))
        store, _ = self._run(cfg, ladder(cfg)[0], rw.sample)
        self.assertGreater(len(store), 0,
                           "the store stayed empty in the referential phase")

    def test_only_successful_is_respected(self):
        cfg = cfg_small()
        cfg.bottleneck.only_successful = True
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(2))
        store, batch = self._run(cfg, ladder(cfg)[-1], tw.sample)
        # whatever landed in the store, it should be no more than the successes
        self.assertLessEqual(len(store), 3 * int(batch.res["success"].sum()) + 1)

    def test_meanings_come_from_the_right_place_in_each_phase(self):
        cfg = cfg_small()
        cfg.bottleneck.only_successful = False
        store = TranscriptStore(cfg)
        torch.manual_seed(3)
        f, b = agents(cfg, 2)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(3))
        B = 128
        i = torch.arange(B)
        rb = rw.sample(B)
        batch, _ = run_and_update_gumbel(
            cfg, rb, f, b, i % 2, torch.div(i, 2, rounding_mode="floor") % 2,
            update=500, phase=ladder(cfg)[0])
        for j in (0, 5, 40):
            got = store.meaning_of(batch, j)
            want = (int(rb.true_meaning[j][0]), int(rb.true_meaning[j][1]))
            self.assertEqual(got, want,
                             "a lineup round was filed under the wrong meaning")


class TestBottleneckCoverage(unittest.TestCase):
    """Change 2: common forms must transmit reliably; only rare ones may be lost."""

    def _store(self, cfg, n_common=4000, n_rare=20):
        st = TranscriptStore(cfg)
        c = cfg.channel
        D = c.dialogue_len

        def add(word_atoms, meaning, count):
            toks = torch.full((D,), c.pad_id, dtype=torch.long)
            for j, a in enumerate(word_atoms[:D]):
                toks[j] = a
            for _ in range(count):
                st._buf.append(StoredEpisode(
                    f_obs=torch.zeros(4, dtype=torch.long),
                    b_obs=torch.zeros(4, dtype=torch.long),
                    tokens=toks.clone(),
                    active=torch.ones(D, dtype=torch.bool),
                    f_dec=torch.zeros(8, dtype=torch.long),
                    b_dec=torch.zeros(8, dtype=torch.long),
                    episode=0, f_generation=0, b_generation=0, meaning=meaning))
                st.meaning_counts[meaning] += 1

        add([1, 2], (0, 1), n_common)      # a form used constantly
        add([7, 9], (2, 8), n_rare)        # a form used almost never
        return st

    def test_a_newborn_sees_essentially_the_whole_parent_generation(self):
        cfg = cfg_small()
        st = self._store(cfg)
        want = min(cfg.bottleneck.max_samples,
                   int(cfg.bottleneck.coverage * len(st)))
        got = st.sample(want, random.Random(0))
        self.assertGreaterEqual(len(got) / len(st), 0.95,
                                "the sample is a sliver, not a generation")

    def test_common_forms_transmit_and_rare_ones_are_the_ones_at_risk(self):
        cfg = cfg_small()
        st = self._store(cfg, n_common=4000, n_rare=8)
        want = min(cfg.bottleneck.max_samples, int(cfg.bottleneck.coverage * len(st)))
        sample = st.sample(want, random.Random(1))
        cov = st.word_coverage(cfg, sample)
        self.assertGreaterEqual(cov["common_coverage"], 0.99,
                                "a common form was at risk of being lost")
        self.assertGreaterEqual(cov["common_forms_shown"], cov["common_forms"])

    def test_a_small_cap_is_what_used_to_endanger_common_forms(self):
        """The old behaviour, kept reachable so the contrast is testable."""
        cfg = cfg_small()
        cfg.bottleneck.n_samples = 40          # the old small-sample regime
        st = self._store(cfg, n_common=4000, n_rare=8)
        small = st.sample(cfg.bottleneck.n_samples, random.Random(2))
        self.assertEqual(len(small), 40)
        rare_seen = sum(1 for it in small if it.meaning == (2, 8))
        self.assertLessEqual(rare_seen, 2,
                             "a 40-sample draw should rarely contain the rare form")

    def test_coverage_default_is_effectively_complete(self):
        cfg = cfg_small()
        self.assertGreaterEqual(cfg.bottleneck.coverage, 0.95)
        self.assertGreaterEqual(cfg.bottleneck.max_samples, 10000)
        self.assertEqual(cfg.bottleneck.n_samples, 0,
                         "a hard cap would put common forms back at risk")


if __name__ == "__main__":
    unittest.main(verbosity=2)
