"""Tests for the innate lexicon: one name per meaning, one meaning per name.

What each class protects:

* the meaning a name is for is the asked-about field's value, and nothing where
  a whole lot is asked for;
* the speaker's own lexicon term pays a settled, distinct name, pays nothing for
  one form used for everything, pulls a form that co-occurs more with one meaning
  towards that meaning, and is per speaker;
* an utterance counts as "a word" -- and so as an ostensive lesson -- only when it
  begins with the speaker's established, unshared name (names are words since
  2026-09-30; `test_language_faculty.py` covers what that changed);
* the naming objective is positive on a distinct code, and has a live gradient
  where the meanings' distributions are all but identical (information alone
  has none there);
* the words-only view of a gestured turn is the gesture-free layout;
* the ostensive lesson fires only on established names, and its statistics and
  state survive a snapshot round trip.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard import gesture as G
from orchard.config import Config
from orchard.conventions import (PopulationUsage, SpeakerLexicon, naming_keys,
                                 naming_mutual_information)
from orchard.curriculum import ASK_ALL, ReferentialWorld, phase_named
from orchard.env import BUYER, FARMER, parse_words
from orchard.gumbel import run_and_update_gumbel
from orchard.world import N_LOT_FIELDS

from test_curriculum import agents, cfg_small


def pairing(B, n=2):
    i = torch.arange(B)
    return i % n, torch.div(i, n, rounding_mode="floor") % n


def teach(lex: SpeakerLexicon, agent: int, key, form, times: int) -> None:
    lex.observe([agent] * times, [key] * times, [tuple(form)] * times)


class TestWhatANameIsFor(unittest.TestCase):
    def test_the_meaning_is_the_asked_about_value_and_nothing_for_a_whole_lot(self):
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        ph = phase_named(cfg, "name-quantity").with_informer(FARMER)
        rb = rw.sample(32, informer=FARMER, query=3)
        keys = naming_keys(cfg, ph, FARMER, rb.obs(cfg, FARMER))
        for i, k in enumerate(keys):
            self.assertEqual(k, (3, int(rb.true_meaning[i, 3])))
        rb = rw.sample(32, informer=FARMER, query=ASK_ALL)
        self.assertTrue(all(k is None for k in naming_keys(cfg, ph, FARMER, rb.obs(cfg, FARMER))))
        # the guesser is not naming anything
        self.assertTrue(all(k is None for k in naming_keys(cfg, ph, BUYER, rb.obs(cfg, BUYER))))
        # nor is anyone in `mutual` (a whole lot each) or the farmer at market (a barn)
        mutual = phase_named(cfg, "mutual")
        mb = rw.sample_mutual(8)
        self.assertTrue(all(k is None for k in naming_keys(cfg, mutual, FARMER, mb.obs(cfg, FARMER))))


class TestTheSpeakersOwnLexicon(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_small()
        self.cfg.reward.lexicon = 1.0
        self.cfg.reward.lexicon_min_support = 3

    def test_a_settled_distinct_name_is_paid_and_a_shared_one_is_not(self):
        lex = SpeakerLexicon(self.cfg)
        apple, banana = (0, 0), (0, 1)
        teach(lex, 7, apple, (3,), 10)
        teach(lex, 7, banana, (5,), 10)
        bonus, used = lex.terms([7, 7, 7], [apple, apple, banana], [(3,), (5,), (5,)])
        self.assertAlmostEqual(bonus[0], 1.0, places=5, msg="my name for apple, said of apple")
        self.assertAlmostEqual(bonus[1], -1.0, places=5, msg="my name for banana, said of apple")
        self.assertAlmostEqual(bonus[2], 1.0, places=5)
        self.assertEqual(used, [True, False, True])
        # one form for everything: nothing to earn, and it is not a word
        lex2 = SpeakerLexicon(self.cfg)
        teach(lex2, 7, apple, (3,), 10)
        teach(lex2, 7, banana, (3,), 10)
        bonus, used = lex2.terms([7, 7], [apple, banana], [(3,), (3,)])
        self.assertAlmostEqual(bonus[0], 0.0, places=5)
        self.assertAlmostEqual(bonus[1], 0.0, places=5)
        self.assertEqual(used, [False, False])

    def test_a_form_that_leans_towards_one_meaning_is_pulled_to_it(self):
        """The positive feedback a modal-form term lacks: mostly-shared usage with a
        small asymmetry still rewards the asymmetric form when said for its meaning."""
        lex = SpeakerLexicon(self.cfg)
        apple, banana = (0, 0), (0, 1)
        teach(lex, 1, apple, (3,), 8)
        teach(lex, 1, apple, (9,), 2)          # said of apple sometimes
        teach(lex, 1, banana, (3,), 10)        # never said of banana
        bonus, used = lex.terms([1, 1], [apple, banana], [(9,), (9,)])
        self.assertGreater(bonus[0], 0.0, "saying 9 of apple should earn: it leans apple")
        self.assertLess(bonus[1], 0.0, "saying 9 of banana should be charged")
        self.assertGreater(bonus[0], bonus[1])
        # and the mostly-shared form leans the other way: charged a little when
        # said of apple, paid a little when said of banana, where it is purer
        bonus, _ = lex.terms([1, 1], [apple, banana], [(3,), (3,)])
        self.assertLess(bonus[0], 0.0)
        self.assertGreater(bonus[1], 0.0)
        self.assertLess(abs(bonus[0]), 0.5)

    def test_the_lexicon_is_per_speaker(self):
        lex = SpeakerLexicon(self.cfg)
        apple = (0, 0)
        teach(lex, 1, apple, (3,), 10)
        bonus, used = lex.terms([1, 2], [apple, apple], [(3,), (3,)])
        self.assertAlmostEqual(bonus[0], 1.0, places=5)
        self.assertEqual(bonus[1], 0.0, "another speaker has no name for it yet")
        self.assertEqual(used, [True, False])
        self.assertEqual(set(lex.summary()), {"1"})

    def test_a_name_needs_support_and_a_whole_lot_round_has_no_name(self):
        lex = SpeakerLexicon(self.cfg)
        teach(lex, 1, (0, 0), (3,), 2)         # below min_support
        bonus, used = lex.terms([1], [(0, 0)], [(3,)])
        self.assertEqual(bonus, [0.0])
        self.assertEqual(used, [False])
        bonus, used = lex.terms([1], [None], [(3,)])
        self.assertEqual((bonus, used), ([0.0], [False]))
        self.assertEqual(lex.summary()["1"]["meanings_named"], 0)

    def test_state_survives_a_round_trip(self):
        lex = SpeakerLexicon(self.cfg)
        teach(lex, 4, (2, 1), (1, 2), 5)
        lex._decay(3)
        other = SpeakerLexicon(self.cfg)
        other.load_state(lex.state())
        self.assertEqual(other.names(4), lex.names(4))
        self.assertAlmostEqual(other.support(4, (2, 1)), lex.support(4, (2, 1)), places=6)


class TestTheNamingObjective(unittest.TestCase):
    def test_a_distinct_code_scores_full_and_a_shared_one_zero(self):
        E, n = 8, 90
        keys = [(0, i % 3) for i in range(n)]
        have = torch.ones(n, dtype=torch.bool)
        lp = torch.full((n, E), -30.0)
        for i in range(n):
            lp[i, i % 3] = 0.0
        v = naming_mutual_information(torch.log_softmax(lp, -1), have, keys)
        self.assertAlmostEqual(float(v), math.log(3) + 1.0, places=3)
        same = torch.log_softmax(torch.zeros(n, E), -1)
        self.assertAlmostEqual(float(naming_mutual_information(same, have, keys)), 0.0, places=5)

    def test_the_gradient_is_alive_next_to_symmetry(self):
        """Information has a zero gradient at the symmetric point; the separation
        term does not, and an untrained speaker starts next to that point."""
        E, n = 8, 90
        keys = [(0, i % 3) for i in range(n)]
        have = torch.ones(n, dtype=torch.bool)
        torch.manual_seed(0)
        base = torch.full((n, E), 1.0 / E)
        lp = (base + 1e-4 * torch.randn(n, E)).clamp(min=1e-6).log().requires_grad_(True)
        v = naming_mutual_information(lp, have, keys)
        v.backward()
        self.assertGreater(float(lp.grad.abs().max()), 1e-4)
        # ...and pushes the meanings apart: a step along the gradient raises it
        with torch.no_grad():
            after = naming_mutual_information(lp + 0.5 * lp.grad, have, keys)
        self.assertGreater(float(after), float(v))

    def test_naming_the_field_instead_of_the_value_earns_nothing(self):
        """A speaker that says one thing for every fruit and another for every
        quantity has told the fields apart, which is in its observation anyway,
        and no value apart, which is the point. Measured: the across-meanings
        form of this objective paid exactly that code 1.5-2.1 on name-quantity."""
        E, n = 8, 120
        keys = [((0, i % 3) if i % 2 == 0 else (3, i % 5)) for i in range(n)]
        have = torch.ones(n, dtype=torch.bool)
        lp = torch.full((n, E), -30.0)
        for i, k in enumerate(keys):
            lp[i, 0 if k[0] == 0 else 1] = 0.0            # one symbol per *field*
        lp = torch.log_softmax(lp, -1)
        self.assertAlmostEqual(float(naming_mutual_information(lp, have, keys)), 0.0, places=4)
        # ...whereas one symbol per value within each field scores in full
        lp = torch.full((n, E), -30.0)
        for i, k in enumerate(keys):
            lp[i, k[1] if k[0] == 0 else 3 + k[1]] = 0.0
        lp = torch.log_softmax(lp, -1)
        v = float(naming_mutual_information(lp, have, keys))
        self.assertAlmostEqual(v, 0.5 * (math.log(3) + 1.0) + 0.5 * (math.log(5) + 1.0), places=3)

    def test_rows_without_a_meaning_or_without_a_symbol_are_left_out(self):
        E = 6
        lp = torch.log_softmax(torch.randn(6, E), -1)
        self.assertIsNone(naming_mutual_information(lp, torch.ones(6, dtype=torch.bool),
                                                    [None] * 6))
        self.assertIsNone(naming_mutual_information(lp, torch.zeros(6, dtype=torch.bool),
                                                    [(0, i % 2) for i in range(6)]))
        self.assertIsNone(naming_mutual_information(lp, torch.ones(6, dtype=torch.bool),
                                                    [(0, 0)] * 6), "one meaning: nothing to tell apart")


class TestTheOstensiveLesson(unittest.TestCase):
    def test_the_words_only_view_is_the_gesture_free_layout(self):
        cfg = cfg_small()
        c = cfg.channel
        L = c.max_msg_len
        g = G.gesture_id(cfg, 3, 2)
        toks = torch.full((2, c.dialogue_len), c.pad_id, dtype=torch.long)
        toks[0, :4] = torch.tensor([g, 5, c.space_id, 7][:min(4, L)])
        toks[1, :3] = torch.tensor([5, c.space_id, 7][:min(3, L)])
        view = G.without_gestures(cfg, toks, toks, [0])
        self.assertEqual(view[0, :L].tolist(), toks[1, :L].tolist(),
                         "a gestured turn without its gesture is the same words at the same slots")
        self.assertEqual(view[1].tolist(), toks[1].tolist(), "an ungestured turn is untouched")
        # the soft (one-hot) form shifts the same way
        NT = G.n_token_ids(cfg)
        soft = torch.nn.functional.one_hot(toks, NT).float()
        sview = G.without_gestures(cfg, toks, soft, [0])
        self.assertEqual(sview[0].argmax(-1).tolist(), view[0].tolist())
        self.assertTrue(torch.equal(sview[1], soft[1]))

    def test_a_lesson_needs_an_established_name(self):
        """Fresh speakers have no names, so no turn is a word and no lesson fires;
        a speaker with settled names has its name recognised as one."""
        cfg = cfg_small()
        cfg.channel.max_symbols = 6
        torch.manual_seed(5)
        f, b = agents(cfg)
        usage = PopulationUsage(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(5))
        ph = phase_named(cfg, "name-fruit").with_informer(FARMER)
        fi, bi = pairing(64)
        batch, st = run_and_update_gumbel(cfg, rw.sample(64, informer=FARMER, query=0), f, b,
                                          fi, bi, phase=ph, usage=usage, gesture_share=1.0)
        self.assertEqual(st.words_used, 0.0)
        self.assertTrue(bool((batch.res["farmer_lexicon"] == 0).all()))
        # settle names for the two describers by hand -- in a lexicon holding
        # nothing else, so what each is paid is exact -- then play again
        usage = PopulationUsage(cfg)
        for a in f:
            for v in range(cfg.world.n_varieties):
                teach(usage.lexicon, a.agent_id, (0, v), (v,), 10)
        rb = rw.sample(64, informer=FARMER, query=0)
        batch, st = run_and_update_gumbel(cfg, rb, f, b, fi, bi, phase=ph, usage=usage,
                                          gesture_share=1.0)
        # whichever turns happened to *begin with* the settled name count as words
        L = cfg.channel.max_msg_len
        said_name = []
        for i in range(64):
            turn = G.strip_gestures(cfg, [t for t in batch.tokens[i, :L].tolist()
                                          if t != cfg.channel.pad_id])
            words = parse_words(cfg, [t for t in turn if t < cfg.channel.end_id])
            if words and words[0] == (int(rb.true_meaning[i, 0]),):
                said_name.append((i, len(words)))
        self.assertAlmostEqual(st.words_used, len(said_name) / 64.0, places=6)
        # a settled, distinct name said of its meaning is paid in full when it
        # is said alone, and shared out over the words when it is not
        for i, n in said_name:
            self.assertAlmostEqual(float(batch.res["farmer_lexicon"][i]),
                                   cfg.reward.lexicon / n, places=5)

    def test_the_prior_and_the_lesson_are_method_settings(self):
        cfg = cfg_small()
        cfg.reward.lexicon = 0.0
        cfg.reward.lexicon_mi = 0.0
        cfg.gesture.ostensive_coef = 0.0
        torch.manual_seed(1)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(1))
        ph = phase_named(cfg, "name-fruit").with_informer(FARMER)
        fi, bi = pairing(32)
        batch, st = run_and_update_gumbel(cfg, rw.sample(32, informer=FARMER, query=0), f, b,
                                          fi, bi, phase=ph, usage=PopulationUsage(cfg),
                                          gesture_share=1.0)
        self.assertEqual(st.naming_signal, 0.0)
        self.assertEqual(st.lexicon_bonus, 0.0)
        self.assertEqual(len(batch), 32)


if __name__ == "__main__":
    unittest.main()
