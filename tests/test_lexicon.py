"""Tests for the open-vocabulary channel and its analyses (addendum).

The properties worth protecting here are the ones that would quietly invalidate
the vocabulary claims: that any symbol sequence parses, that the cost really is
paid per symbol, that the newborn curriculum is skewed the way it says it is, and
that "a word emerged" is distinguished from "the hyphen was never used".
"""
from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.agents import make_agent
from orchard.bottleneck import StoredEpisode, TranscriptStore
from orchard.config import Config
from testscale import method_at_test_scale
from orchard.env import BUYER, FARMER, Decision, parse_words, resolve, word_text
from orchard.lexicon import (FormTracker, bucketed_analysis, length_frequency,
                             reference_buyer_obs, split_meanings, word_stats,
                             word_usage_flags)
from orchard.population import Population
from orchard.rollout import run_episodes
from orchard.world import World


def small_cfg() -> Config:
    cfg = method_at_test_scale()
    cfg.world.n_varieties = 3
    cfg.world.max_qty = 8
    cfg.world.n_price_bins = 6
    cfg.world.reservation_max_bin = 4
    cfg.world.budget_min_bin = 1
    cfg.channel.atomic_vocab = 10
    cfg.channel.max_symbols = 6
    cfg.channel.n_turns = 4
    cfg.model.d_model = 32
    cfg.model.d_ff = 64
    cfg.population.n_farmers = 2
    cfg.population.n_buyers = 2
    return cfg


class TestSymbolStream(unittest.TestCase):
    def test_ids_do_not_collide(self):
        c = small_cfg().channel
        ids = {c.hyphen_id, c.space_id, c.end_id, c.pad_id}
        self.assertEqual(len(ids), 4)
        self.assertTrue(all(i >= c.atomic_vocab for i in ids))
        self.assertEqual(c.n_emittable, c.atomic_vocab + 3)
        self.assertEqual(c.n_symbol_ids, c.atomic_vocab + 4)
        self.assertFalse(c.is_atom(c.hyphen_id))
        self.assertTrue(c.is_atom(c.atomic_vocab - 1))

    def test_any_symbol_sequence_parses(self):
        """Lenient by design: the agent may put marks anywhere, including nowhere."""
        cfg = small_cfg()
        c = cfg.channel
        rng = random.Random(0)
        for _ in range(3000):
            n = rng.randint(0, c.max_symbols)
            syms = [rng.randrange(c.pad_id + 1) for _ in range(n)]
            words = parse_words(cfg, syms)          # must not raise
            for w in words:
                self.assertGreater(len(w), 0)
                for atom in w:
                    self.assertTrue(c.is_atom(atom))

    def test_structural_marks_group_atoms(self):
        cfg = small_cfg()
        c = cfg.channel
        H, S, E = c.hyphen_id, c.space_id, c.end_id
        self.assertEqual(parse_words(cfg, [1, H, 2, S, 3, E]), [(1, 2), (3,)])
        self.assertEqual(parse_words(cfg, [1, 2, E]), [(1, 2)])       # no marks: one word
        self.assertEqual(parse_words(cfg, [1, S, 2, S, 3, E]), [(1,), (2,), (3,)])
        self.assertEqual(parse_words(cfg, [E]), [])
        self.assertEqual(parse_words(cfg, [S, S, H, 4, E]), [(4,)])   # malformed, still fine

    def test_open_vocabulary_is_actually_open(self):
        """The point of the redesign: more words than there are atoms."""
        cfg = small_cfg()
        c = cfg.channel
        # a turn of max_symbols can spell words of up to that many atoms
        self.assertGreater(c.atomic_vocab ** 2, c.atomic_vocab)
        seen = set()
        rng = random.Random(1)
        for _ in range(4000):
            syms = [rng.randrange(c.end_id) for _ in range(c.max_symbols)]
            seen.update(parse_words(cfg, syms))
        multi = [w for w in seen if len(w) > 1]
        self.assertGreater(len(seen), c.atomic_vocab,
                           "vocabulary is not open: no more words than atoms")
        self.assertTrue(multi, "no multi-atom words are even reachable")


class TestLengthCost(unittest.TestCase):
    def test_every_emitted_symbol_is_charged(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(0))
        sc = w.sample()
        d = Decision(0, 0, 0, 0)
        base = resolve(cfg, sc, d, d, 0, 0).farmer_reward
        for n in (1, 3, 5):
            r = resolve(cfg, sc, d, d, farmer_tokens=n, buyer_tokens=0)
            self.assertAlmostEqual(base - r.farmer_reward, n * cfg.reward.symbol_cost,
                                   places=6)

    def test_hyphens_and_spaces_cost_the_same_as_atoms(self):
        """Otherwise structure would be free and agents would pad with it."""
        c = small_cfg().channel
        self.assertTrue(c.costed(0))
        self.assertTrue(c.costed(c.hyphen_id))
        self.assertTrue(c.costed(c.space_id))
        self.assertFalse(c.costed(c.end_id), "ending a message must be free")
        self.assertFalse(c.costed(c.pad_id))


class TestFrequencySkew(unittest.TestCase):
    def _store(self, cfg, counts):
        st = TranscriptStore(cfg)
        for meaning, n in counts.items():
            for _ in range(n):
                st._buf.append(StoredEpisode(
                    f_obs=torch.zeros(4, dtype=torch.long),
                    b_obs=torch.zeros(4, dtype=torch.long),
                    tokens=torch.zeros(4, dtype=torch.long),
                    active=torch.zeros(4, dtype=torch.bool),
                    f_dec=torch.zeros(4, dtype=torch.long),
                    b_dec=torch.zeros(4, dtype=torch.long),
                    episode=0, f_generation=0, b_generation=0, meaning=meaning))
                st.meaning_counts[meaning] += 1
        return st

    def test_skew_controls_what_a_newborn_sees(self):
        cfg = small_cfg()
        counts = {(0, 1): 600, (0, 2): 250, (1, 1): 100, (2, 7): 50}
        shares = {}
        for skew in (0.0, 1.0, 2.0):
            cfg.bottleneck.frequency_skew = skew
            st = self._store(cfg, counts)
            got = Counter(x.meaning for x in st.sample(200, random.Random(0)))
            shares[skew] = {k: v / 200 for k, v in got.items()}

        rare = (2, 7)
        common = (0, 1)
        # flat sampling gives the rare meaning roughly its type share
        self.assertGreater(shares[0.0][rare], 0.10)
        # natural proportion tracks the store
        self.assertAlmostEqual(shares[1.0][common], 0.6, delta=0.12)
        # heavy skew all but erases the rare meaning -- this is the vocabulary-loss lever
        self.assertLess(shares[2.0].get(rare, 0.0), 0.03)
        self.assertGreater(shares[2.0][common], shares[1.0][common])

    def test_world_requests_are_zipfian(self):
        cfg = small_cfg()
        cfg.world.zipf_alpha = 0.9
        w = World(cfg.world, random.Random(0))
        table = w.meaning_table()
        self.assertGreater(table[0][1] / table[-1][1], 3.0)
        # and the analytic table matches what actually gets drawn
        n = 20000
        emp = Counter(w.meaning_key(w.sample().buyer) for _ in range(n))
        for key, prob in table[:3]:
            self.assertAlmostEqual(emp[key] / n, prob, delta=0.02)

    def test_uniform_world_when_alpha_is_zero(self):
        cfg = small_cfg()
        cfg.world.zipf_alpha = 0.0
        w = World(cfg.world, random.Random(0))
        probs = [p for _, p in w.meaning_table()]
        self.assertAlmostEqual(max(probs), min(probs), places=6)


class TestWordAnalysis(unittest.TestCase):
    def _batch(self, cfg, n=24):
        torch.manual_seed(0)
        farmers = [make_agent(cfg, agent_id=i, role=FARMER, slot=i, generation=0,
                              birth_episode=0, lifespan=10**9) for i in range(2)]
        buyers = [make_agent(cfg, agent_id=9 + i, role=BUYER, slot=i, generation=0,
                             birth_episode=0, lifespan=10**9) for i in range(2)]
        w = World(cfg.world, random.Random(0))
        scen = w.sample_batch(n, held_out=False)
        f_idx = torch.randint(0, 2, (n,))
        b_idx = torch.randint(0, 2, (n,))
        return run_episodes(cfg, scen, farmers, buyers, f_idx, b_idx), farmers, buyers, w

    def test_word_stats_are_self_consistent(self):
        cfg = small_cfg()
        batch, _, _, _ = self._batch(cfg)
        ws = word_stats(cfg, [batch])
        self.assertGreaterEqual(ws["distinct_words"], 1)
        self.assertGreaterEqual(ws["mean_word_len_atoms"], 1.0)
        self.assertLessEqual(ws["mean_symbols_per_message"], cfg.channel.max_symbols)
        self.assertLessEqual(ws["multi_atom_word_share"], 1.0)
        self.assertAlmostEqual(
            sum(w["share"] for w in ws["top_words"]),
            sum(ws["word_counts"][w["word"]] for w in ws["top_words"])
            / sum(ws["word_counts"].values()), places=6)

    def test_degenerate_hyphen_and_space_usage_are_distinguished(self):
        """The addendum asks for these to be called out separately, not conflated."""
        cfg = small_cfg()
        no_hyphen = {"hyphen_share": 0.0, "space_share": 0.2, "at_length_cap_frac": 0.1,
                     "distinct_words": 9, "word_entropy_bits": 3.0}
        no_space = {"hyphen_share": 0.2, "space_share": 0.0, "at_length_cap_frac": 0.1,
                    "distinct_words": 9, "word_entropy_bits": 3.0}
        babble = {"hyphen_share": 0.2, "space_share": 0.2, "at_length_cap_frac": 0.9,
                  "distinct_words": 9, "word_entropy_bits": 3.0}
        healthy = {"hyphen_share": 0.15, "space_share": 0.2, "at_length_cap_frac": 0.2,
                   "distinct_words": 20, "word_entropy_bits": 3.5}
        self.assertTrue(any("WORD FORMATION" in f for f in word_usage_flags(cfg, no_hyphen)))
        self.assertTrue(any("SEGMENTATION" in f for f in word_usage_flags(cfg, no_space)))
        self.assertTrue(any("BABBLING" in f for f in word_usage_flags(cfg, babble)))
        self.assertEqual(word_usage_flags(cfg, healthy), [])

    def test_meaning_probe_is_deterministic_and_on_meaning(self):
        cfg = small_cfg()
        a = reference_buyer_obs(cfg, (2, 5))
        b = reference_buyer_obs(cfg, (2, 5))
        self.assertEqual(a, b)
        self.assertEqual(a[0], 2)
        self.assertEqual(a[1], 5)

    def test_buckets_split_the_meaning_space(self):
        cfg = small_cfg()
        w = World(cfg.world, random.Random(0))
        frequent, rare = split_meanings(w, 0.5)
        self.assertTrue(frequent and rare)
        self.assertEqual(set(frequent) & set(rare), set())
        pf = sum(w.meaning_prob(k) for k in frequent)
        pr = sum(w.meaning_prob(k) for k in rare)
        self.assertGreater(pf, pr, "the frequent bucket should carry more mass")
        self.assertGreater(len(rare), len(frequent),
                           "Zipf means few common meanings and many rare ones")

    def test_length_frequency_and_buckets_run_end_to_end(self):
        cfg = small_cfg()
        torch.manual_seed(0)
        pop = Population(cfg, random.Random(0))
        w = World(cfg.world, random.Random(0))
        lf = length_frequency(cfg, pop, w)
        self.assertEqual(lf["n"], len(w.meaning_table()))
        self.assertIn("rho_symbols", lf)
        bk = bucketed_analysis(cfg, pop, w, rng=random.Random(0))
        self.assertIn("frequent", bk)
        self.assertIn("rare", bk)

    def test_form_tracker_records_and_detects_change(self):
        cfg = small_cfg()
        torch.manual_seed(0)
        pop = Population(cfg, random.Random(0))
        w = World(cfg.world, random.Random(0))
        ft = FormTracker(cfg, w)
        first = ft.observe(pop, 0)
        self.assertEqual(first["n_events"], 0, "nothing to compare against yet")
        # perturb every buyer so the forms must move, then look again
        with torch.no_grad():
            for a in pop.buyers:
                a.net.token_head.weight.add_(torch.randn_like(a.net.token_head.weight) * 3.0)
        second = ft.observe(pop, 1000)
        self.assertGreater(second["n_events"], 0, "a large weight change produced no drift")
        self.assertTrue(all(e.old_form is not None for e in ft.events))
        rows = ft.report_rows()
        self.assertTrue(rows)
        self.assertIn("regularised", rows[0])
        timeline = ft.timeline()
        self.assertTrue(timeline)
        self.assertIn("trail", timeline[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
