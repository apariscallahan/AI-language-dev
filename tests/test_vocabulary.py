"""Tests for what the 2026-10-01 run was missing: a vocabulary fit to hand on.

That run passed every naming rung at its first check and then sat in `mutual`:
each founder had 8 words for 27 meanings, six words for nine quantities, not
one word in common, descriptions that did not stop, and a scaffold doing all
the describing. What each class here protects:

* one word per meaning: the matching target gives every meaning an atom of its
  own, a fresh lexicon becomes one-to-one under it, and a lexicon that already
  is one feels nothing;
* one dialect: a listener remembers an elder's word only where it understood,
  the memory fades, and a junior with a lexicon of its own ends up saying its
  elder's words;
* numbers are exact: near rounds take the nearest values, a coarse number code
  wins random rounds and loses near ones, and the gate asks for the near ones;
* the scaffold fades: at zero it changes nothing, the trainer withdraws it in
  one rung and that rung cannot be passed until it is gone, a newborn is born
  into the scaffold of the moment, and practice with it becomes the speaker's
  own habit -- without bending a word;
* the market: a request in words points at a barn row, the farmer's lexicon
  names that row's parts with the naming rungs' words, what a listener
  understood reaches its state causally, and generation and the full pass
  agree on every rung;
* the vocabulary probe counts what is said, and `name-all` is gated on it.
"""
from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from orchard.agents import CommNet, dialogue_offset, lexicon_offsets, make_agent
from orchard.batched import TensorWorld
from orchard.config import Config, validate
from orchard.conventions import (PopulationUsage, imitation_loss, lexicon_exclusivity,
                                 one_to_one)
from orchard.curriculum import (ASK_ALL, NUMERAL_FIELDS, ReferentialWorld, evaluate_rung,
                                phase_named, phase_schema, rung_budget)
from orchard.env import BUYER, FARMER
from orchard.gumbel import run_and_update_gumbel
from orchard.rollout import run_episodes
from orchard.world import K_PRICE, K_QTY, N_LOT_FIELDS, lot_spans

from test_rungs import _swap_evidence


def founders(cfg, n=2):
    """One pool in both seats, as below the split: agent 0 is the elder."""
    return [make_agent(cfg, agent_id=i, role=FARMER, slot=i, generation=0,
                       birth_episode=0, lifespan=10 ** 9) for i in range(n)]


def distinct(table: torch.Tensor) -> int:
    return len(set(table.argmax(-1).tolist()))


def settle(net, steps=80):
    """Train a lexicon under exclusivity alone until it is one-to-one."""
    opt = torch.optim.Adam(net.parameters(), lr=net.cfg.train.lr)
    for _ in range(steps):
        opt.zero_grad()
        lexicon_exclusivity(net.lexicon_table()).backward()
        opt.step()


# ==========================================================================
class TestOneWordPerMeaning(unittest.TestCase):
    def test_the_matching_gives_every_meaning_an_atom_of_its_own(self):
        torch.manual_seed(0)
        table = torch.randn(27, 32)
        table[5] = table[4] + 0.01          # two meanings with the very same claims
        table[9, :] = 0.0
        table[9, 3] = table[2, 3] = 9.0     # ...and two certain of one atom
        got = one_to_one(table)
        self.assertEqual(tuple(got.shape), (27,))
        self.assertEqual(len(set(got.tolist())), 27, "two meanings were given the same atom")
        self.assertIn(3, (int(got[2]), int(got[9])), "the contested atom went to neither")
        # nearest: a table that is already one-to-one is given back as it is
        own = torch.full((27, 32), -5.0)
        perm = torch.randperm(32)[:27]
        own[torch.arange(27), perm] = 5.0
        self.assertTrue(torch.equal(one_to_one(own), perm))
        # every word said with certainty, three meanings to an atom: still parted
        stuck = torch.full((27, 32), -10.0)
        stuck[torch.arange(27), torch.arange(27) // 3] = 10.0
        self.assertEqual(len(set(one_to_one(stuck).tolist())), 27)

    def test_the_assignment_is_the_cheapest_one(self):
        """Exact, not approximate: checked against every permutation."""
        import itertools
        import numpy as np
        from orchard.conventions import _assign
        rng = np.random.default_rng(0)
        for n, m in ((4, 4), (3, 6), (5, 5)):
            for _ in range(10):
                cost = rng.normal(size=(n, m))
                got = _assign(cost)
                self.assertEqual(len(set(got.tolist())), n)
                best = min(sum(cost[i, c] for i, c in enumerate(cols))
                           for cols in itertools.permutations(range(m), n))
                self.assertAlmostEqual(float(cost[np.arange(n), got].sum()), best, places=9)

    def test_fewer_atoms_than_meanings_are_shared_out_evenly(self):
        """The `duality` inventory: 8 atoms for 27 meanings, at most 4 each."""
        torch.manual_seed(1)
        got = one_to_one(torch.randn(27, 8))
        self.assertLessEqual(int(torch.bincount(got, minlength=8).max()), 4)

    def test_a_word_heard_from_an_elder_beats_a_word_made_up(self):
        """The junior's own word for one meaning sits on the atom its elder
        uses for another. The heard word gets the atom; the made-up one moves."""
        torch.manual_seed(2)
        own = torch.full((27, 32), -8.0)
        perm = torch.randperm(32)[:27]
        own[torch.arange(27), perm] = 8.0
        heard = torch.zeros(27, 32)
        heard[0, perm[5]] = 12.0
        self.assertEqual(int(one_to_one(own)[0]), int(perm[0]), "nothing heard: it keeps its own")
        got = one_to_one(own, heard)
        self.assertEqual(int(got[0]), int(perm[5]))
        self.assertNotEqual(int(got[5]), int(perm[5]))
        self.assertEqual(len(set(got.tolist())), 27)

    def test_a_fresh_lexicon_becomes_one_to_one(self):
        """Measured before this existed, with information plus separation as
        the objective: 17, 16 and 18 distinct words of 27, and no further --
        two meanings both certain of one atom have no gradient left to part
        them. The matching's target does not saturate."""
        for seed in (0, 1, 2):
            cfg = Config()
            torch.manual_seed(seed)
            net = CommNet(cfg, FARMER)
            self.assertLess(distinct(net.lexicon_table()), 20, "born one-to-one: nothing tested")
            settle(net, steps=40)
            table = net.lexicon_table()
            self.assertEqual(distinct(table), 27, "seed %d" % seed)
            self.assertGreater(float(torch.softmax(table, -1).max(-1).values.min()), 0.5,
                               "a meaning's word is not the one it says most of the time")

    def test_a_one_to_one_lexicon_is_left_alone(self):
        cfg = Config()
        table = torch.full((27, 32), -6.0)
        table[torch.arange(27), torch.randperm(32)[:27]] = 6.0
        table.requires_grad_(True)
        lexicon_exclusivity(table).backward()
        self.assertLess(float(table.grad.abs().max()), 1e-3)
        # ...and a shared word is not: one of the two meanings is moved off it
        clash = table.detach().clone()
        clash[1] = clash[0]
        clash.requires_grad_(True)
        lexicon_exclusivity(clash).backward()
        moved = clash.grad[:2].abs().sum(-1)
        self.assertGreater(float(moved.max()), 0.5)
        self.assertLess(float(moved.min()), 1e-3, "the meaning that keeps the atom was moved too")
        self.assertEqual(sum(lot_spans(cfg.world)), 27)

    def test_the_table_is_every_value_of_every_field_in_lot_order(self):
        cfg = Config()
        torch.manual_seed(3)
        net = CommNet(cfg, FARMER)
        spans, offs = lot_spans(cfg.world), lexicon_offsets(cfg)
        table = net.lexicon_table()
        self.assertEqual(tuple(table.shape), (sum(spans), cfg.channel.atomic_vocab))
        ph = phase_named(cfg, "name-all").with_informer(FARMER)
        schema = phase_schema(cfg, FARMER, ph)
        obs = torch.zeros((1, len(schema)), dtype=torch.long)
        obs[0, :N_LOT_FIELDS] = torch.tensor([2, 1, 3, 7, 4])
        words = net.speaker_lexicon.part_words(net.lot_concepts(obs, schema))[0]
        for f in range(N_LOT_FIELDS):
            self.assertEqual(int(words[f]), int(table[offs[f] + int(obs[0, f])].argmax()))

    def test_every_speaker_that_played_is_trained_on_it(self):
        cfg = Config()
        cfg.train.device = "cpu"
        torch.manual_seed(0)
        pool = founders(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        ph = phase_named(cfg, "name-fruit").with_informer(FARMER)
        idx = torch.arange(32) % 2
        _, st = run_and_update_gumbel(cfg, rw.sample(32, informer=FARMER, mix=ph.mix), pool, pool,
                                      idx, 1 - idx, phase=ph, usage=PopulationUsage(cfg))
        self.assertGreater(st.lexicon_exclusive, 0.5, "a fresh lexicon is far from one-to-one")
        cfg.reward.lexicon_exclusive = 0.0
        cfg.reward.lexicon_imitate = 0.0
        _, st = run_and_update_gumbel(cfg, rw.sample(32, informer=FARMER, mix=ph.mix), pool, pool,
                                      idx, 1 - idx, phase=ph, usage=PopulationUsage(cfg))
        self.assertEqual(st.lexicon_exclusive, 0.0)


# ==========================================================================
class TestOneDialect(unittest.TestCase):
    def play(self, cfg, pool, rung="name-fruit", n=64, seed=0, informer=FARMER):
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(seed))
        ph = phase_named(cfg, rung)
        if ph.referential:
            ph = ph.with_informer(informer)
            scen = rw.sample(n, informer=informer, mix=ph.mix)
        else:
            scen = rw.sample_mutual(n)
        idx = torch.arange(n) % 2
        batch, st = run_and_update_gumbel(cfg, scen, pool, pool, idx, 1 - idx, phase=ph,
                                          usage=PopulationUsage(cfg),
                                          generator=torch.Generator().manual_seed(seed))
        return scen, batch, st

    def test_a_listener_remembers_an_elders_word_where_it_understood(self):
        cfg = Config()
        cfg.train.device = "cpu"
        torch.manual_seed(0)
        pool = founders(cfg)
        scen, batch, _ = self.play(cfg, pool)
        elder, junior = (a.net.speaker_lexicon.heard for a in pool)
        self.assertEqual(float(elder.sum()), 0.0, "the eldest has nobody to learn a word from")
        # the junior listened where it sat in the guesser's seat (the buyer's
        # here) and remembers one word for each round it won
        won = batch.res["success"] & (batch.b_idx == 1)
        self.assertGreater(int(won.sum()), 0)
        self.assertAlmostEqual(float(junior.sum()), float(won.sum()), places=3)
        # ...filed under the fruit that was asked about, as the atom it heard
        offs = lexicon_offsets(cfg)
        self.assertEqual(float(junior[offs[1]:].sum()), 0.0, "only fruit was asked about")
        c = cfg.channel
        first = batch.tokens[:, :c.max_msg_len]
        for r in won.nonzero().squeeze(-1).tolist()[:8]:
            atom = next(int(t) for t in first[r].tolist() if t < c.atomic_vocab)
            self.assertGreater(float(junior[int(scen.true_meaning[r, 0]), atom]), 0.0)

    def test_in_a_description_it_remembers_the_word_it_took_for_each_field_it_got_right(self):
        """`mutual` asks about no single field, and a newcomer plays nothing
        else: what it remembers there is, field by field, the word its own
        reader took to name the field, where its report of the field was right."""
        from orchard.curriculum import H_REPORT
        cfg = Config()
        cfg.train.device = "cpu"
        cfg.train.lr = 0.0                       # nothing learns: the reading can be redone
        torch.manual_seed(0)
        pool = founders(cfg)
        mutual = phase_named(cfg, "mutual")
        scen, batch, _ = self.play(cfg, pool, rung="mutual", n=96)
        elder, junior = (a.net.speaker_lexicon.heard for a in pool)
        self.assertEqual(float(elder.sum()), 0.0)
        offs = lexicon_offsets(cfg)
        want = torch.zeros_like(junior)
        for role, idx, dec, truth in ((FARMER, batch.f_idx, batch.f_dec, scen.b_meaning),
                                      (BUYER, batch.b_idx, batch.b_dec, scen.f_meaning)):
            mine = idx == 1                                  # the junior, in this seat
            with torch.no_grad():
                hd = pool[1].net.listen(batch.tokens, None, mutual.self_mask(cfg, role))
            for f in range(N_LOT_FIELDS):
                ok = mine & (dec[:, H_REPORT[f]] == truth[:, f]) & (hd.said[:, f] >= 0)
                want.index_put_((offs[f] + truth[ok, f], hd.said[ok, f]),
                                torch.ones(int(ok.sum())), accumulate=True)
        self.assertGreater(float(want.sum()), 0.0, "nothing was understood: nothing tested")
        self.assertTrue(torch.allclose(junior, want, atol=1e-4))

    def test_guessing_which_word_named_what_still_finds_the_word(self):
        """Cross-situational learning. A listener that has no idea which of the
        five words named which field files each field's value under a word
        picked at random -- the right one a fifth of the time, and otherwise a
        word for another field's value, a different one each time. The memory
        still peaks on the right word, and a junior with a lexicon of its own
        still ends up with its elder's."""
        cfg = Config()
        torch.manual_seed(0)
        elder, junior = CommNet(cfg, FARMER), CommNet(cfg, FARMER)
        settle(elder)
        settle(junior)
        words = elder.lexicon_table().argmax(-1)
        spans, offs = lot_spans(cfg.world), lexicon_offsets(cfg)
        opt = torch.optim.Adam(junior.parameters(), lr=cfg.train.lr)
        g = torch.Generator().manual_seed(0)
        keep = 0.5 ** (1.0 / cfg.reward.lexicon_imitate_half_life_updates)
        heard = junior.speaker_lexicon.heard
        R = cfg.reward
        n = 64
        for _ in range(200):
            heard.mul_(keep)
            lots = torch.stack([torch.randint(0, sp, (n,), generator=g) for sp in spans], 1)
            said = torch.stack([words[offs[f] + lots[:, f]] for f in range(N_LOT_FIELDS)], 1)
            for f in range(N_LOT_FIELDS):
                took = said[torch.arange(n), torch.randint(0, N_LOT_FIELDS, (n,), generator=g)]
                heard.index_put_((offs[f] + lots[:, f], took), torch.ones(n), accumulate=True)
            table = junior.lexicon_table()
            loss = (R.lexicon_exclusive * lexicon_exclusivity(table, heard)
                    + R.lexicon_imitate * imitation_loss(table, heard))
            opt.zero_grad()
            loss.backward()
            opt.step()
        right = heard[torch.arange(27), words] / heard.sum(-1)
        self.assertLess(float(right.max()), 0.5, "the guesses were not guesses")
        self.assertTrue(bool((heard.argmax(-1) == words).all()),
                        "the memory does not peak on the right word")
        mine = junior.lexicon_table().argmax(-1)
        self.assertEqual(int((mine == words).sum()), 27)

    def test_the_memory_fades_with_every_update(self):
        cfg = Config()
        cfg.train.device = "cpu"
        torch.manual_seed(0)
        pool = founders(cfg)
        # the eldest hears nothing new from anyone, so what it holds only fades
        memory = pool[0].net.speaker_lexicon.heard
        memory.fill_(3.0)
        self.play(cfg, pool)
        keep = 0.5 ** (1.0 / cfg.reward.lexicon_imitate_half_life_updates)
        self.assertTrue(torch.allclose(memory, torch.full_like(memory, 3.0 * keep), atol=1e-5))
        self.play(cfg, pool, rung="mutual", n=16)
        self.assertTrue(torch.allclose(memory, torch.full_like(memory, 3.0 * keep * keep),
                                       atol=1e-5))

    def test_only_meanings_that_were_heard_are_pulled(self):
        torch.manual_seed(0)
        table = torch.randn(27, 32, requires_grad=True)
        heard = torch.zeros(27, 32)
        heard[4, 7] = 3.0
        heard[9, 2], heard[9, 5] = 1.0, 1.0
        imitation_loss(table, heard).backward()
        moved = table.grad.abs().sum(-1) > 0
        self.assertEqual(moved.nonzero().squeeze(-1).tolist(), [4, 9])
        self.assertLess(float(table.grad[4, 7]), 0.0, "the heard word is pulled up")
        self.assertLess(float(table.grad[9, 2]), 0.0)
        self.assertEqual(float(imitation_loss(table, torch.zeros(27, 32))), 0.0)

    def test_a_junior_with_words_of_its_own_ends_up_saying_its_elders(self):
        """The 2026-10-01 founders agreed on 0 of 27 words at the end of every
        naming rung. Here the junior already has a one-to-one lexicon of its
        own -- the hard case -- and hears a tenth of the meanings per update."""
        cfg = Config()
        torch.manual_seed(0)
        elder, junior = CommNet(cfg, FARMER), CommNet(cfg, FARMER)
        settle(elder)
        settle(junior)
        words = elder.lexicon_table().argmax(-1)
        self.assertLess(int((junior.lexicon_table().argmax(-1) == words).sum()), 8)
        opt = torch.optim.Adam(junior.parameters(), lr=cfg.train.lr)
        g = torch.Generator().manual_seed(0)
        keep = 0.5 ** (1.0 / cfg.reward.lexicon_imitate_half_life_updates)
        heard = junior.speaker_lexicon.heard
        R = cfg.reward
        for _ in range(150):
            heard.mul_(keep)
            rows = (torch.rand(27, generator=g) < 0.1).nonzero().squeeze(-1)
            heard[rows, words[rows]] += 1.0
            table = junior.lexicon_table()
            loss = (R.lexicon_exclusive * lexicon_exclusivity(table)
                    + R.lexicon_imitate * imitation_loss(table, heard))
            opt.zero_grad()
            loss.backward()
            opt.step()
        mine = junior.lexicon_table().argmax(-1)
        self.assertEqual(int((mine == words).sum()), 27)
        self.assertEqual(len(set(mine.tolist())), 27)


# ==========================================================================
class TestNumbersAreExact(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.rw = ReferentialWorld(self.cfg, generator=torch.Generator().manual_seed(0))

    def test_a_near_round_takes_the_nearest_values(self):
        for f in NUMERAL_FIELDS:
            span = lot_spans(self.cfg.world)[f]
            rb = self.rw.sample(2000, query=f, near=1.0)
            t = rb.true_meaning[:, f]
            vals = rb.meanings[:, :, f]
            self.assertTrue(bool((vals[torch.arange(2000), rb.target] == t).all()))
            d = (vals - t.unsqueeze(1)).abs()
            inside = (t > 0) & (t < span - 1)
            self.assertTrue(bool((d[inside].max(1).values == 1).all()),
                            "inside the range the wrong candidates are one more and one less")
            self.assertTrue(bool((d[~inside].max(1).values == 2).all()),
                            "at an end they are the next two")
            # every candidate distinct, as in any round
            self.assertTrue(bool((vals.sort(1).values.diff(dim=1) > 0).all()))

    def test_the_share_is_the_setting_and_only_numbers_have_neighbours(self):
        f = NUMERAL_FIELDS[0]

        def near_share(rb):
            d = (rb.meanings[:, :, f] - rb.true_meaning[:, f].unsqueeze(1)).abs()
            return float((d.max(1).values <= 2).float().mean())
        far = near_share(self.rw.sample(4000, query=f, near=0.0))
        mixed = near_share(self.rw.sample(4000, query=f))
        self.assertLess(far, 0.25)
        self.assertAlmostEqual(mixed, far + self.cfg.curriculum.numeral_near_frac * (1 - far),
                               delta=0.05)
        self.assertEqual(self.rw.near_frac(0, 1.0), 0.0, "a fruit has no neighbours")
        self.assertEqual(self.rw.near_frac(f), self.cfg.curriculum.numeral_near_frac)

    def test_a_coarse_number_code_wins_random_rounds_and_loses_near_ones(self):
        """Six words for nine quantities -- the 2026-10-01 founders' -- read by
        a listener that knows the code perfectly."""
        f = NUMERAL_FIELDS[0]
        word = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4, 5])          # 9 values, 6 words

        def success(near):
            rb = self.rw.sample(6000, query=f, near=near)
            vals = rb.meanings[:, :, f]
            same = word[vals] == word[rb.true_meaning[:, f]].unsqueeze(1)
            return float((1.0 / same.sum(1).float()).mean())       # guess among the matches
        self.assertGreater(success(0.0), 0.85)
        self.assertLess(success(1.0), self.cfg.curriculum.numeral_min_near)

    def test_a_hard_rounds_near_miss_in_a_number_is_one_step_away(self):
        rb = self.rw.sample(4000, query=ASK_ALL, near=1.0)
        m = rb.meanings
        for f in NUMERAL_FIELDS:
            for a in range(m.shape[1]):
                for b in range(a + 1, m.shape[1]):
                    only = ((m[:, a] != m[:, b]).sum(-1) == 1) & (m[:, a, f] != m[:, b, f])
                    if bool(only.any()):
                        step = (m[:, a, f] - m[:, b, f]).abs()[only]
                        # (two near misses of one anchor in *different* fields
                        # never differ in one field alone, so these are anchor
                        # against near miss -- or independent draws, rarely)
                        self.assertGreater(float((step == 1).float().mean()), 0.9)

    def test_a_rung_that_plays_number_rounds_is_judged_on_near_ones(self):
        cfg = self.cfg
        for name, kinds in (("name-quantity", (3,)), ("name-price", (3, 4)),
                            ("name-all", (3, 4))):
            rung = phase_named(cfg, name)
            lo = max(rung_budget(cfg, rung)[0], 10 ** 6)
            ev = _swap_evidence(True, True)
            ok, checks = evaluate_rung(cfg, rung, ev, lo)
            self.assertTrue(ok, {k: c for k, c in checks.items() if not c["met"]})
            what = {3: "tells neighbouring quantities apart", 4: "tells neighbouring prices apart"}
            self.assertEqual({k for k in checks if k.startswith("tells")},
                             {what[k] for k in kinds})
            # one describer's numbers are only roughly right
            ev["near_miss"][kinds[0]] = {"success": 0.85, "each": [0.95, 0.74]}
            ok, checks = evaluate_rung(cfg, rung, ev, lo)
            self.assertFalse(checks[what[kinds[0]]]["met"])
            # not measured is not passed
            ev = _swap_evidence(True, True)
            del ev["near_miss"]
            ok, checks = evaluate_rung(cfg, rung, ev, lo)
            self.assertFalse(ok)
            self.assertIn("not measured", checks[what[kinds[0]]]["detail"])
        fruit = phase_named(cfg, "name-fruit")
        _, checks = evaluate_rung(cfg, fruit, _swap_evidence(True, True), 10 ** 6)
        self.assertFalse(any(k.startswith("tells") for k in checks))


# ==========================================================================
class TestTheScaffoldFades(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        torch.manual_seed(0)
        self.net = CommNet(self.cfg, FARMER)
        self.rw = ReferentialWorld(self.cfg, generator=torch.Generator().manual_seed(0))
        self.ph = phase_named(self.cfg, "name-all").with_informer(FARMER)
        self.schema = phase_schema(self.cfg, FARMER, self.ph)

    def test_it_holds_once_enough_has_been_said(self):
        """Say as much as was asked, and no more: on the 2026-10-01 run nothing
        said when to stop, and `mutual`'s descriptions ran to 8.3 words."""
        cfg, net, c = self.cfg, self.net, self.cfg.channel
        S = c.space_id
        one = self.rw.sample(2, informer=FARMER, query=3).obs(cfg, FARMER)
        toks = torch.full((2, c.dialogue_len), c.pad_id, dtype=torch.long)
        toks[:, 0] = 5
        push, _ = net.turn_so_far(one, self.schema, toks, [0, 1])
        self.assertEqual(push.tolist(), [[0, -1], [0, -1]], "one field: one word, then hold")
        whole = self.rw.sample(1, informer=FARMER, query=ASK_ALL).obs(cfg, FARMER)
        settle(net)                                   # five distinct words for the five parts
        w = net.speaker_lexicon.part_words(net.lot_concepts(whole, self.schema))[0].tolist()
        toks = torch.full((1, c.dialogue_len), c.pad_id, dtype=torch.long)
        seq = sum([[a, S] for a in w], [])[:-1]
        toks[0, :len(seq)] = torch.tensor(seq)
        push, _ = net.turn_so_far(whole, self.schema, toks, list(range(len(seq) + 1)))
        self.assertEqual(push[0].tolist(), [0, 1, 1, 1, 1, 1, 1, 1, 1, -1],
                         "go on while parts are unnamed, hold when all five are")
        # the push and the hold, as logits
        h = torch.randn(1, cfg.model.d_model)
        lex = net.speaker_lexicon
        g = float(lex.go_on)
        with torch.no_grad():
            base = net.speak(h, whole, self.schema)
            go = net.speak(h, whole, self.schema, (torch.tensor([1]), None)) - base
            hold = net.speak(h, whole, self.schema, (torch.tensor([-1]), None)) - base
        self.assertAlmostEqual(float(go[0, c.space_id]), g, places=4)
        self.assertAlmostEqual(float(go[0, c.end_id]), -g, places=4)
        self.assertAlmostEqual(float(hold[0, c.space_id]), -g, places=4)
        self.assertAlmostEqual(float(hold[0, c.end_id]), 0.0, places=5)
        # atoms enough for every meaning: a word is one atom, so hold means stop
        self.assertAlmostEqual(float(hold[0, c.hyphen_id]), -g, places=4)
        self.assertEqual(float(hold[0, :c.atomic_vocab].abs().sum()), 0.0)

    def test_with_fewer_atoms_than_meanings_a_word_may_go_on(self):
        cfg = Config()
        cfg.channel.atomic_vocab = 8
        net = CommNet(cfg, FARMER)
        hold = net.speaker_lexicon._hold
        self.assertEqual(float(hold[cfg.channel.space_id]), -1.0)
        self.assertEqual(float(hold[cfg.channel.hyphen_id]), 0.0)

    def test_at_zero_it_changes_nothing(self):
        cfg, net = self.cfg, self.net
        obs = self.rw.sample(8, informer=FARMER, mix=(0.1,) * 5 + (0.5,)).obs(cfg, FARMER)
        h = torch.randn(8, cfg.model.d_model)
        turn = (torch.tensor([1, -1, 1, 0, 1, -1, 0, 1]),
                torch.rand(8, N_LOT_FIELDS) < 0.5)
        with torch.no_grad():
            on, (base, own, asked_for, plan) = net.speak(h, obs, self.schema, turn, habit=True)
            bare = net.speak(h, obs, self.schema)
            self.assertGreater(float((on - bare).abs().max()), 1.0, "the scaffold does nothing")
            self.assertGreater(float((asked_for - own).abs().max()), 0.3)
            self.assertTrue(torch.equal(base, net.speaker_lexicon.gate(net.token_head(h))))
            net.set_scaffold(0.0)
            off, (_, own0, asked_for0, plan0) = net.speak(h, obs, self.schema, turn, habit=True)
            lex, part = net.speaker_lexicon(h, net.lot_concepts(obs, self.schema))
        self.assertTrue(torch.allclose(off, base + lex, atol=1e-5),
                        "with the scaffold gone the speaker says what its own policy says: "
                        "its token head, and the word for the part it chose itself")
        self.assertTrue(torch.allclose(part, own0, atol=1e-6))
        self.assertTrue(torch.allclose(own, own0, atol=1e-6),
                        "the choice underneath does not depend on the scaffold")
        # what the scaffold asks for is the same however much of it is left
        self.assertTrue(torch.allclose(asked_for, asked_for0, atol=1e-6))
        self.assertTrue(torch.equal(plan, plan0))
        for name in ("go_on", "inhibit", "ask", "scaffold"):
            self.assertFalse(getattr(net.speaker_lexicon, name).requires_grad)
            self.assertIn("speaker_lexicon." + name, net.state_dict())
        net.set_scaffold(7.0)
        self.assertEqual(float(net.speaker_lexicon.scaffold), 1.0)

    def lessons(self, obs, toks, positions):
        """The two halves of `train.scaffold_distil`, for these states: KL of
        the choice of part the scaffold asks for against the speaker's own, and
        of the next symbol it asks for against the token head's, over what the
        grammar allows."""
        from orchard.env import MASKED, grammar_allowed
        cfg, net, c = self.cfg, self.net, self.cfg.channel
        off = dialogue_offset(cfg)
        mask = self.ph.self_mask(cfg, FARMER)
        part, after = [], []
        for p in positions:
            h = net.encode(obs, toks, upto=off + p, schema=self.schema, self_mask=mask)[:, -1]
            st = net.turn_so_far(obs, self.schema, toks, [p])
            _, (base, own, asked_for, plan) = net.speak(
                h, obs, self.schema, (st[0][:, 0], st[1][:, 0]), habit=True)
            allowed = grammar_allowed(cfg, toks[:, p - 1] if p else toks[:, p], p)
            if bool(allowed[0, 0]):                      # an atom is due: which part?
                t_ = asked_for.detach()
                part.append((t_ * ((t_ + 1e-9).log() - (own + 1e-9).log())).sum(-1).mean())
            else:                                        # a word was said: what next?
                lp = F.log_softmax(plan.masked_fill(~allowed, MASKED), -1)
                lq = F.log_softmax(base.masked_fill(~allowed, MASKED), -1)
                after.append((lp.exp() * torch.where(allowed, lp - lq,
                                                     torch.zeros_like(lp))).sum(-1).mean())
        return part, after

    def test_practice_becomes_the_speakers_own_habit(self):
        """What `train.scaffold_distil` trains. Before: the speaker's own choice
        of part is uniform whatever it was asked, and its token head neither
        goes on while parts are unnamed nor stops when all five are."""
        cfg, net, c = self.cfg, self.net, self.cfg.channel
        settle(net)
        words_before = net.speaker_lexicon.say.weight.detach().clone()
        n = 60
        off = dialogue_offset(cfg)
        mask = self.ph.self_mask(cfg, FARMER)
        silent = torch.full((n, c.dialogue_len), c.pad_id, dtype=torch.long)

        def asked():
            rb = self.rw.sample(n, informer=FARMER, mix=(0.2,) * N_LOT_FIELDS + (0.0,))
            return rb.obs(cfg, FARMER), rb.query

        def described():
            """Whole lots, with the first k parts' words said (k = 1..5)."""
            obs = self.rw.sample(n, informer=FARMER, query=ASK_ALL).obs(cfg, FARMER)
            w = net.speaker_lexicon.part_words(net.lot_concepts(obs, self.schema))
            toks = silent.clone()
            k = 1 + torch.arange(n) % N_LOT_FIELDS
            for r in range(n):
                seq = sum([[int(w[r, f]), c.space_id] for f in range(int(k[r]))], [])[:-1]
                toks[r, :len(seq)] = torch.tensor(seq)
            return obs, toks, k

        def measure():
            with torch.no_grad():
                obs, q = asked()
                h = net.encode(obs, silent, upto=off, schema=self.schema, self_mask=mask)[:, -1]
                own = net.speak(h, obs, self.schema, habit=True)[1][1]
                on_asked = float(own[torch.arange(n), q].mean())
                obs, toks, k = described()
                go, stop = [], []
                for words in range(1, N_LOT_FIELDS + 1):
                    rows = k == words
                    p = 2 * words - 1
                    h = net.encode(obs[rows], toks[rows], upto=off + p, schema=self.schema,
                                   self_mask=mask)[:, -1]
                    head = torch.softmax(net.token_head(h)[:, [c.hyphen_id, c.space_id,
                                                              c.end_id]], -1)
                    (stop if words == N_LOT_FIELDS else go).append(
                        head[:, 2 if words == N_LOT_FIELDS else 1].mean())
            return on_asked, float(torch.stack(go).mean()), float(torch.stack(stop).mean())
        before = measure()
        self.assertLess(before[0], 0.3)
        self.assertLess(before[1], 0.6)
        self.assertLess(before[2], 0.6)
        opt = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)
        for step in range(120):
            obs, _ = asked()
            part, _ = self.lessons(obs, silent, [0])
            obs, toks, k = described()
            _, after = self.lessons(obs, toks, [1])        # after the first word: go on
            for words in (3, N_LOT_FIELDS):                 # ...and later, and at the end
                rows = k == words
                after += self.lessons(obs[rows], toks[rows], [2 * words - 1])[1]
            opt.zero_grad()
            (torch.stack(part).mean() + torch.stack(after).mean()).backward()
            if step == 0:
                self.assertIsNone(net.speaker_lexicon.say.weight.grad,
                                  "practising a description bent a word")
                self.assertGreater(float(net.speaker_lexicon.query.weight.grad.abs().sum()), 0.0)
                grad = net.token_head.weight.grad
                self.assertEqual(float(grad[:c.atomic_vocab].abs().sum()), 0.0,
                                 "the token head was taught atoms")
                self.assertGreater(float(grad[c.space_id].abs().sum()), 0.0)
            opt.step()
        after_ = measure()
        self.assertGreater(after_[0], 0.8, "its own choice of part: %.2f -> %.2f"
                           % (before[0], after_[0]))
        self.assertGreater(after_[1], 0.8, "going on while parts are unnamed: %.2f -> %.2f"
                           % (before[1], after_[1]))
        self.assertGreater(after_[2], 0.8, "stopping when all five are named: %.2f -> %.2f"
                           % (before[2], after_[2]))
        self.assertTrue(torch.equal(net.speaker_lexicon.say.weight, words_before))

    def test_in_the_naming_rungs_a_word_is_the_lexicons_alone(self):
        """The token head sees the context, so it is a second place a word can
        live. Measured on a CPU run: a junior's lexicon had taken its elder's
        word for a fruit while its token head, 6.5 nats up on the junior's own
        old word in exactly that context, went on saying that one."""
        cfg, net, c = self.cfg, self.net, self.cfg.channel
        A = c.atomic_vocab
        obs = self.rw.sample(16, informer=FARMER, query=0).obs(cfg, FARMER)
        h = torch.randn(16, cfg.model.d_model)
        self.assertEqual(float(net.speaker_lexicon.own_atoms), 0.0, "a brain is born naming")
        with torch.no_grad():
            before = net.speak(h, obs, self.schema)
            net.token_head.weight[:A] += 5.0 * torch.randn(A, cfg.model.d_model)
            after = net.speak(h, obs, self.schema)
            self.assertTrue(torch.allclose(before[:, :A], after[:, :A], atol=1e-5),
                            "the token head had a say in which atom")
            self.assertTrue(torch.equal(before[:, A:], after[:, A:]))
            # whether to go on, start another word or stop is still its own
            net.token_head.weight[c.end_id] += 1.0
            self.assertFalse(torch.allclose(net.speak(h, obs, self.schema)[:, c.end_id],
                                            after[:, c.end_id]))
            after = net.speak(h, obs, self.schema)
            net.set_own_atoms(True)
            free = net.speak(h, obs, self.schema)
        self.assertGreater(float((free[:, :A] - after[:, :A]).abs().mean()), 1.0)
        self.assertTrue(torch.allclose(free[:, :A] - after[:, :A], net.token_head(h)[:, :A],
                                       atol=1e-4))
        self.assertIn("speaker_lexicon.own_atoms", net.state_dict())

    def test_the_head_gets_its_atoms_back_in_the_market(self):
        from orchard.train import Trainer
        cfg = Config()
        cfg.train.device, cfg.log.plot = "cpu", False
        with tempfile.TemporaryDirectory() as tmp:
            tr = Trainer(cfg, tmp, quiet=True)
            try:
                names = [p.name for p in tr.curriculum.phases]
                for i, name in enumerate(names):
                    tr.curriculum.index = i
                    want = i >= names.index(cfg.curriculum.own_atoms_from_rung)
                    self.assertEqual(tr.own_atoms_now(), want, name)
                    tr.apply_scaffold()
                    for a in tr.pop.all_agents():
                        self.assertEqual(float(a.net.speaker_lexicon.own_atoms), float(want))
                self.assertEqual(cfg.curriculum.own_atoms_from_rung, "order")
                self.assertFalse(tr.curriculum.phases[names.index("mutual")].trading)
                # fewer atoms than meanings: a word needs more than the lexicon's one
                tr.curriculum.index = 0
                cfg.channel.atomic_vocab = 8
                self.assertTrue(tr.own_atoms_now())
            finally:
                tr.close()

    def test_the_scaffold_overrules_whatever_the_speaker_has_learned(self):
        """`name-fruit` asks about nothing but the fruit, and a speaker that
        learns "name the fruit" there must still answer a question about the
        colour when one first comes. Measured without the bound, and with a
        teacher that was the pupil plus a nudge: at update 50 a speaker asked
        about colour, quality, quantity or price named the fruit."""
        cfg, net = self.cfg, self.net
        lex = net.speaker_lexicon
        obs = self.rw.sample(64, informer=FARMER, query=1).obs(cfg, FARMER)     # colour asked
        h = torch.randn(64, cfg.model.d_model)
        concepts = net.lot_concepts(obs, self.schema)
        asked = F.one_hot(torch.full((64,), 1), N_LOT_FIELDS)

        def set_on(part):
            """A learned choice of ``part`` as hard as a choice can be: its raw
            score far up, every other part's far down, whatever the state."""
            want = torch.full((N_LOT_FIELDS,), -50.0)
            want[part] = 50.0
            with torch.no_grad():
                lex.query.weight.zero_()
                lex.query.bias.copy_(torch.linalg.pinv(lex.part_key) @ want
                                     * cfg.model.d_model ** 0.5)
        set_on(0)                                              # "the fruit"
        with torch.no_grad():
            _, att, own, asked_for = lex(h, concepts, None, asked, habit=True)
        self.assertGreater(float(own[:, 0].min()), 0.9, "the pupil is not sure of the fruit")
        self.assertGreater(float(att[:, 1].min()), 0.8, "the scaffold was overruled")
        self.assertGreater(float(asked_for[:, 1].min()), 0.99,
                           "the lesson follows the pupil, not the question")
        # with no question, the order among the parts not yet named is its own
        named = torch.zeros(64, N_LOT_FIELDS)
        named[:, 0] = 1
        set_on(3)
        with torch.no_grad():
            _, att, own, asked_for = lex(h, concepts, named, torch.zeros(64, N_LOT_FIELDS),
                                         habit=True)
        self.assertGreater(float(asked_for[:, 3].min()), 0.9)
        self.assertLess(float(asked_for[:, 0].max()), 0.01, "a part already named is asked for")

    def test_a_scaffolded_speakers_symbols_are_lessons_for_its_own_policy(self):
        cfg = self.cfg
        cfg.train.device = "cpu"
        pool = founders(cfg)
        idx = torch.arange(32) % 2
        scen = self.rw.sample(32, informer=FARMER, mix=self.ph.mix)
        _, st = run_and_update_gumbel(cfg, scen, pool, pool, idx, 1 - idx, phase=self.ph,
                                      usage=PopulationUsage(cfg))
        self.assertGreater(st.scaffold_distil, 0.05)
        for a in pool:
            a.net.set_scaffold(0.0)
        _, st = run_and_update_gumbel(cfg, scen, pool, pool, idx, 1 - idx, phase=self.ph,
                                      usage=PopulationUsage(cfg))
        self.assertEqual(st.scaffold_distil, 0.0, "nothing left to practise against")
        # ...except in the rung it is withdrawn in, where the lesson runs to the
        # end: a schedule does not know how long a habit takes to form
        fade = phase_named(cfg, cfg.curriculum.scaffold_fade_rung)
        _, st = run_and_update_gumbel(cfg, self.rw.sample_mutual(32), pool, pool, idx, 1 - idx,
                                      phase=fade, usage=PopulationUsage(cfg))
        self.assertGreater(st.scaffold_distil, 0.0)
        # and never above it: the market has neither the scaffold nor the lesson
        tw = TensorWorld(cfg, device="cpu", generator=torch.Generator().manual_seed(0))
        _, st = run_and_update_gumbel(cfg, tw.sample(32), pool, pool, idx, 1 - idx,
                                      phase=phase_named(cfg, "order"),
                                      usage=PopulationUsage(cfg))
        self.assertEqual(st.scaffold_distil, 0.0)

    def test_the_trainer_withdraws_it_in_one_rung(self):
        from orchard.train import Trainer
        cfg = Config()
        cfg.train.device, cfg.log.plot = "cpu", False
        cu = cfg.curriculum
        with tempfile.TemporaryDirectory() as tmp:
            tr = Trainer(cfg, tmp, quiet=True)
            try:
                names = [p.name for p in tr.curriculum.phases]
                at = names.index(cu.scaffold_fade_rung)
                self.assertEqual(cu.scaffold_fade_rung, "mutual",
                                 "the last rung over bare lots, and the only one below the "
                                 "market in which the second seat describes")
                B = cfg.train.batch_size
                for i in range(at):
                    tr.curriculum.index = i
                    tr.curriculum.episodes_in_phase = 10 ** 6 * B
                    self.assertEqual(tr.scaffold_now(), 1.0, names[i])
                tr.curriculum.index = at
                hold, fade = cu.scaffold_hold_updates, cu.scaffold_fade_updates
                for done, want in ((0, 1.0), (hold, 1.0), (hold + fade // 2, 0.5),
                                   (hold + fade, 0.0), (hold + fade + 500, 0.0)):
                    tr.curriculum.episodes_in_phase = done * B
                    # the promotion clock stands still while the community
                    # arrives; the scaffold goes on being withdrawn
                    tr.curriculum.updates_in_phase = 0
                    self.assertAlmostEqual(tr.scaffold_now(), want, places=6)
                self.assertEqual(tr.apply_scaffold(), 0.0)
                for a in tr.pop.all_agents():
                    self.assertEqual(float(a.net.speaker_lexicon.scaffold), 0.0)
                for i in range(at + 1, len(names)):
                    tr.curriculum.index = i
                    tr.curriculum.episodes_in_phase = 0
                    self.assertEqual(tr.scaffold_now(), 0.0,
                                     "%s runs with the scaffold on" % names[i])
                    self.assertFalse(tr.curriculum.phases[i].tuples and i > at)
                # the gate is told what was left when it measured
                tr.curriculum.index = at
                tr.curriculum.episodes_in_phase = (hold + fade // 2) * B
                self.assertEqual(tr.apply_scaffold(), 0.5)
            finally:
                tr.close()

    def test_a_snapshot_from_before_it_could_be_withdrawn(self):
        """Its `ask` / `go_on` / `inhibit` were learned and load, by name, into
        what are now constants; and its speakers have no policy of their own,
        so resumed in `mutual` -- where the 2026-10-01 run stopped, 630 updates
        in -- the scaffold is gone from the first update with nothing in its
        place. The strengths are put back and the run says what will happen."""
        from orchard.train import Trainer

        def trainer(d):
            cfg = Config()
            cfg.train.device, cfg.log.plot = "cpu", False
            tr = Trainer(cfg, d, quiet=True)
            said = []
            tr.log.always = lambda msg, *a, **k: said.append(str(msg))
            return tr, said

        with tempfile.TemporaryDirectory() as tmp:
            tr, _ = trainer(tmp + "/a")
            names = [q.name for q in tr.curriculum.phases]
            tr.curriculum.index = names.index("mutual")
            tr.curriculum.episodes_in_phase = 630 * tr.cfg.train.batch_size
            tr.episode = tr.curriculum.episodes_in_phase
            path = tr.save_snapshot("now")
            tr.close()
            st = torch.load(path, map_location="cpu", weights_only=False)
            for rec in st["farmers"] + st["buyers"]:
                net = rec["net"]
                net["speaker_lexicon.ask"] = torch.tensor(4.24)
                net["speaker_lexicon.go_on"] = torch.tensor(5.05)
                net["speaker_lexicon.inhibit"] = torch.tensor(3.78)
                for k in ("part_key", "scaffold", "heard", "own_atoms"):
                    net.pop("speaker_lexicon." + k, None)
            before = tmp + "/before.pt"
            torch.save(st, before)

            tr, said = trainer(tmp + "/b")
            try:
                tr.load_snapshot(before)
                for a in tr.pop.all_agents():
                    lex = a.net.speaker_lexicon
                    self.assertEqual((float(lex.ask), float(lex.go_on), float(lex.inhibit)),
                                     (lex.ASK, lex.GO_ON, lex.INHIBIT))
                self.assertEqual(sum("put back" in x for x in said), 1, said)
                self.assertEqual(tr.scaffold_now(), 0.0)
                self.assertTrue(tr.warn_unschooled())
                self.assertTrue(any("WARNING" in x and "`mutual`" in x for x in said), said)
                # wound back below the rung it is withdrawn in, the naming
                # rungs can still teach what is missing
                tr.rewind_to("name-fruit")
                self.assertEqual(tr.scaffold_now(), 1.0)
                self.assertFalse(tr.warn_unschooled())
            finally:
                tr.close()

            tr, said = trainer(tmp + "/c")           # one written by this version
            try:
                tr.load_snapshot(path)
                self.assertFalse(tr.warn_unschooled())
                self.assertFalse(any("put back" in x or "WARNING" in x for x in said), said)
            finally:
                tr.close()

    def test_the_rung_it_fades_in_cannot_be_passed_until_it_is_gone(self):
        from test_rungs import _report_evidence
        cfg = self.cfg
        rung = phase_named(cfg, cfg.curriculum.scaffold_fade_rung)
        lo = rung_budget(cfg, rung)[0]
        ev = _report_evidence(rung)
        ok, checks = evaluate_rung(cfg, rung, ev, lo)
        self.assertTrue(ok, {k: c for k, c in checks.items() if not c["met"]})
        ev["scaffold"] = 0.25
        ok, checks = evaluate_rung(cfg, rung, ev, lo)
        self.assertFalse(ok)
        self.assertEqual([k for k, c in checks.items() if not c["met"]],
                         ["describes without the scaffold"])
        del ev["scaffold"]
        ok, checks = evaluate_rung(cfg, rung, ev, lo)
        self.assertFalse(ok, "unmeasured passed")
        self.assertIn("not measured", checks["describes without the scaffold"]["detail"])
        # no other rung is asked -- and a naming rung is, if it fades there
        _, checks = evaluate_rung(cfg, phase_named(cfg, "name-all"),
                                  _swap_evidence(True, True), 10 ** 6)
        self.assertNotIn("describes without the scaffold", checks)
        cfg.curriculum.scaffold_fade_rung = "name-all"
        ev = _swap_evidence(True, True)
        ev["scaffold"] = 0.5
        ok, checks = evaluate_rung(cfg, phase_named(cfg, "name-all"), ev, 10 ** 6)
        self.assertEqual([k for k, c in checks.items() if not c["met"]],
                         ["describes without the scaffold"])

    def test_the_scaffold_has_to_be_gone_before_the_market(self):
        cfg = Config()
        cfg.curriculum.scaffold_fade_rung = "order"
        with self.assertRaises(AssertionError):
            validate(cfg)
        for name in ("name-all", "mutual"):
            cfg.curriculum.scaffold_fade_rung = name
            validate(cfg)


# ==========================================================================
class TestTheMarketFaculty(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.train.device = "cpu"
        torch.manual_seed(0)
        self.farmer = make_agent(self.cfg, agent_id=0, role=FARMER, slot=0, generation=0,
                                 birth_episode=0, lifespan=10 ** 9)
        self.buyer = make_agent(self.cfg, agent_id=1, role=BUYER, slot=0, generation=0,
                                birth_episode=0, lifespan=10 ** 9)
        self.tw = TensorWorld(self.cfg, device="cpu", generator=torch.Generator().manual_seed(0))

    def said(self, obs, fruit, colour, sure=8.0):
        """A reader's reading of a request for (fruit, colour); nothing on the rest."""
        w = self.cfg.world
        out = []
        for span, value in ((w.n_varieties, fruit), (w.n_colors, colour)):
            lg = torch.zeros((obs.shape[0], span))
            lg[torch.arange(obs.shape[0]), value] = sure
            out.append(F.log_softmax(lg, -1))
        for span in lot_spans(w)[2:]:
            out.append(torch.full((obs.shape[0], span), -torch.log(torch.tensor(float(span)))))
        return tuple(out)

    def test_a_request_in_words_points_at_a_barn_row(self):
        cfg, net = self.cfg, self.farmer.net
        offer = phase_named(cfg, "offer")
        obs = self.tw.sample(64).obs(cfg, FARMER)
        self.assertTrue(net.is_barn(phase_schema(cfg, FARMER, offer)))
        rows = net.barn_rows(obs)
        g = torch.Generator().manual_seed(1)
        pick = torch.randint(0, rows.shape[1], (64,), generator=g)
        want = rows[torch.arange(64), pick]
        h = torch.randn(64, cfg.model.d_model)
        with torch.no_grad():
            blind = net.row_attention(h, obs)
            att = net.row_attention(h, obs, self.said(obs, want[:, 0], want[:, 1]))
            many = net.row_attention(h.unsqueeze(1).expand(-1, 3, -1), obs,
                                     self.said(obs, want[:, 0], want[:, 1]))
        self.assertLess(float(blind.max(-1).values.mean()), 0.2,
                        "sixteen rows and nothing heard: no row stands out at birth")
        self.assertTrue(bool((att.argmax(-1) == pick).all()))
        self.assertGreater(float(att.max(-1).values.mean()), 0.95)
        self.assertEqual(tuple(many.shape), (64, 3, rows.shape[1]))
        self.assertTrue(torch.allclose(many[:, 1], att, atol=1e-5))
        # with the faculty off the words point at nothing
        net.reads_rows = False
        with torch.no_grad():
            off = net.row_attention(h, obs, self.said(obs, want[:, 0], want[:, 1]))
        self.assertTrue(torch.allclose(off, blind, atol=1e-6))

    def test_the_farmer_names_that_rows_parts_with_the_naming_rungs_words(self):
        """A lot in a barn is named with the words for a lot in the hand: the
        row's fruit, colour, quality and stock, and the floor price."""
        cfg, net = self.cfg, self.farmer.net
        settle(net)
        obs = self.tw.sample(32).obs(cfg, FARMER)
        rows = net.barn_rows(obs)
        pick = torch.arange(32) % rows.shape[1]
        want = rows[torch.arange(32), pick]
        heard = self.said(obs, want[:, 0], want[:, 1], sure=20.0)
        h = torch.randn(32, cfg.model.d_model)
        table, offs = net.lexicon_table().argmax(-1), lexicon_offsets(cfg)
        price = obs[:, 4 * net.n_cells]
        with torch.no_grad():
            concepts = net.barn_concepts(h, obs, heard)
            words = net.speaker_lexicon.part_words(concepts)
        self.assertEqual(tuple(concepts.shape), (32, N_LOT_FIELDS, cfg.model.d_model))
        for j in range(4):
            self.assertTrue(bool((words[:, j] == table[offs[j] + want[:, j]]).all()),
                            "part %d of the row is not named with its own word" % j)
        self.assertTrue(bool((words[:, 4] == table[offs[4] + price]).all()))
        # and it is what speak() says: the lexicon's term, on a barn
        schema = phase_schema(cfg, FARMER, phase_named(cfg, "offer"))
        with torch.no_grad():
            term = net.speak(h, obs, schema, heard=heard) - net.token_head(h)
            many = net.speak(h.unsqueeze(1).expand(-1, 4, -1), obs, schema, heard=heard)
        self.assertGreater(float(term[:, :cfg.channel.atomic_vocab].abs().max()), 0.1)
        self.assertEqual(float(term[:, cfg.channel.atomic_vocab:].abs().max()), 0.0)
        self.assertEqual(tuple(many.shape), (32, 4, cfg.channel.n_emittable))
        net.barn_lexicon = False
        with torch.no_grad():
            self.assertTrue(torch.equal(net.speak(h, obs, schema, heard=heard), net.token_head(h)))

    def test_what_was_understood_reaches_the_state_and_only_from_what_was_heard(self):
        cfg, net = self.cfg, self.buyer.net
        c = cfg.channel
        offer = phase_named(cfg, "offer")
        mask = offer.self_mask(cfg, BUYER)             # the buyer opens; the farmer answers
        L = c.max_msg_len
        toks = torch.full((3, c.dialogue_len), c.pad_id, dtype=torch.long)
        toks[:, 0:3] = torch.tensor([4, c.space_id, 9])           # the buyer's own words
        toks[:, L:L + 5] = torch.tensor([7, c.space_id, 2, c.hyphen_id, 11])   # the farmer's
        hd = net.listen(toks, None, mask)
        self.assertEqual(tuple(hd.meaning.shape), (3, c.dialogue_len, cfg.model.d_model))
        at = hd.meaning.abs().sum(-1)[0] > 0
        self.assertEqual(at.nonzero().squeeze(-1).tolist(), [L, L + 2, L + 4],
                         "a meaning sits at each atom of a heard word, and nowhere else")
        self.assertTrue(torch.allclose(hd.meaning[0, L + 2], hd.meaning[0, L + 4]),
                        "the two atoms of one word carry that word's one meaning")
        # causal: a word's meaning is the same whatever is said after it
        more = toks.clone()
        more[:, L + 5:L + 8] = torch.tensor([c.space_id, 20, c.end_id])
        self.assertTrue(torch.allclose(net.listen(more, None, mask).meaning[:, :L + 5],
                                       hd.meaning[:, :L + 5], atol=1e-6))
        # it changes the state, through the input; the reader is not trained by it
        obs = self.tw.sample(3).obs(cfg, BUYER)
        schema = phase_schema(cfg, BUYER, offer)
        with_it = net.encode(obs, toks, schema=schema, self_mask=mask)[:, -1]
        without = net.encode(obs, toks, schema=schema, self_mask=mask, heard=None)[:, -1]
        self.assertGreater(float((with_it - without).abs().max()), 1e-4)
        with_it.sum().backward()
        self.assertIsNone(net.reader.word_class.weight.grad)
        self.assertGreater(float(net.variety_emb.weight.grad.abs().sum()), 0.0)
        net.hears_meaning = False
        self.assertIsNone(net.listen(toks, None, mask).meaning)

    def test_generation_and_the_full_pass_agree_on_every_rung(self):
        """A newborn's lessons score the symbols a full pass says the speaker
        would emit; generation emits them one prefix at a time. With heard
        meanings at the input and a barn row found by what had been heard when
        the farmer spoke, the two must still be the same numbers -- on a lot,
        on a barn, and where the other party speaks again afterwards."""
        cfg = self.cfg
        c = cfg.channel
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        zero = torch.zeros(6, dtype=torch.long)
        off = dialogue_offset(cfg)
        for name in ("name-all", "mutual", "offer", "bargain"):
            phase = phase_named(cfg, name)
            if phase.referential:
                phase = phase.with_informer(FARMER)
                scen = rw.sample(6, informer=FARMER, mix=phase.mix)
            elif phase.mutual:
                scen = rw.sample_mutual(6)
            else:
                scen = self.tw.sample(6)
            with torch.no_grad():
                batch = run_episodes(cfg, scen, [self.farmer], [self.buyer], zero, zero,
                                     phase=phase, generator=torch.Generator().manual_seed(1))
            toks = batch.tokens
            for role, agent in ((FARMER, self.farmer), (BUYER, self.buyer)):
                pos = phase.own_positions(cfg, role)
                if not pos:
                    continue
                schema, mask = phase_schema(cfg, role, phase), phase.self_mask(cfg, role)
                obs = scen.obs(cfg, role)
                with torch.no_grad():
                    full, _, dec, _ = agent.net.full_pass(
                        obs, toks, phase.read_positions(cfg, role), schema=schema, self_mask=mask)
                    heads = agent.net.decision_logits(obs, toks, schema=schema, self_mask=mask)
                    for i, p in enumerate(pos):
                        step, _ = agent.net.next_token_logits(obs, toks, off + p, schema=schema,
                                                              self_mask=mask)
                        self.assertTrue(
                            torch.allclose(step, full[:, i], atol=2e-4),
                            "%s, %s, slot %d: generation and the full pass disagree by %.4f"
                            % (name, "farmer" if role == FARMER else "buyer", p,
                               float((step - full[:, i]).abs().max())))
                for a, b in zip(dec, heads):
                    self.assertTrue(torch.allclose(a, b, atol=2e-4), name)

    def test_the_market_trains_through_all_of_it(self):
        """One update of `offer` with every new piece in the graph: no error,
        finite loss, and a gradient reaches the farmer's lexicon from a barn."""
        cfg = self.cfg
        offer = phase_named(cfg, "offer")
        idx = torch.zeros(16, dtype=torch.long)
        before = self.farmer.net.speaker_lexicon.say.weight.detach().clone()
        for a in (self.farmer, self.buyer):
            a.net.set_scaffold(0.0)
        _, st = run_and_update_gumbel(cfg, self.tw.sample(16), [self.farmer], [self.buyer],
                                      idx, idx, phase=offer, usage=PopulationUsage(cfg))
        self.assertTrue(st.policy_loss == st.policy_loss)
        self.assertFalse(torch.equal(self.farmer.net.speaker_lexicon.say.weight, before))
        # and with gradient checkpointing, which re-runs the encoder in backward
        cfg.train.grad_checkpoint = True
        _, st = run_and_update_gumbel(cfg, self.tw.sample(16), [self.farmer], [self.buyer],
                                      idx, idx, phase=offer, usage=PopulationUsage(cfg))
        self.assertTrue(st.policy_loss == st.policy_loss)


# ==========================================================================
class TestTheVocabularyProbe(unittest.TestCase):
    def pop(self, cfg, nets):
        from orchard.population import Population
        import random
        pop = Population(cfg, random.Random(0), device="cpu")
        for a, net in zip(pop.farmers, nets):
            a.net.load_state_dict(net.state_dict())
        return pop

    def test_it_counts_what_each_agent_says_and_whether_they_agree(self):
        from orchard.metrics import vocabulary
        cfg = Config()
        cfg.train.device = "cpu"
        torch.manual_seed(0)
        a, b = CommNet(cfg, FARMER), CommNet(cfg, FARMER)
        fresh = vocabulary(cfg, self.pop(cfg, [a, b]), n_contexts=3)
        self.assertEqual(fresh["meanings"], 27)
        self.assertEqual(fresh["speakers"], 2)
        self.assertLess(fresh["distinct_min"], 0.8, "two fresh lexicons are not one-to-one")
        self.assertLess(fresh["agreement_min"], 0.5, "two fresh lexicons do not agree")
        settle(a)
        same = vocabulary(cfg, self.pop(cfg, [a, copy.deepcopy(a)]), n_contexts=3)
        self.assertEqual(same["distinct_fewest"], 27)
        self.assertEqual(same["agreement_min"], 1.0)
        self.assertGreater(same["consistency"], 0.95)
        self.assertEqual(len(same["agents"]), 2)
        words = next(iter(same["agents"].values()))["words"]
        self.assertEqual(len(words), 27)
        self.assertIn("quantity=8", words)

    def test_above_the_naming_rungs_it_reads_the_lexicons(self):
        """Nobody is asked about one field after `name-all`, and with the
        scaffold gone a speaker's own policy was only ever practised on whole
        lots. Asked about one field in `mutual`, four speakers with identical
        lexicons read "17 of 27 words, 63% shared". What they hold is what
        their lexicons hold, whatever is left of the scaffold."""
        from orchard.metrics import vocabulary
        cfg = Config()
        cfg.train.device = "cpu"
        torch.manual_seed(0)
        a = CommNet(cfg, FARMER)
        settle(a)
        pop = self.pop(cfg, [a, copy.deepcopy(a)])
        for agent in pop.all_agents():
            agent.net.set_scaffold(0.0)            # no help answering the question
        asked = vocabulary(cfg, pop, n_contexts=3)
        held = vocabulary(cfg, pop, source="lexicon")
        self.assertEqual(asked["source"], "said")
        self.assertEqual(held["source"], "lexicon")
        self.assertLess(asked["distinct_fewest"], 27, "an untaught speaker answered the question")
        self.assertEqual(held["distinct_fewest"], 27)
        self.assertEqual(held["agreement_min"], 1.0)
        self.assertGreater(held["consistency"], 0.5)
        table = a.lexicon_table().argmax(-1).tolist()
        words = next(iter(held["agents"].values()))["words"]
        self.assertEqual(list(words.values()), ["a%d" % x for x in table])

    def test_name_all_hands_on_one_word_per_meaning_and_one_dialect(self):
        cfg = Config()
        rung = phase_named(cfg, "name-all")
        lo = rung_budget(cfg, rung)[0]
        # the 2026-10-01 founders: 8 words for 27 meanings, none in common
        ev = _swap_evidence(True, True)
        ev["vocabulary"] = {"meanings": 27, "speakers": 2, "distinct_fewest": 8,
                            "distinct_min": 8 / 27.0, "agreement_min": 0.0,
                            "agreement_mean": 0.0}
        ok, checks = evaluate_rung(cfg, rung, ev, lo)
        self.assertFalse(ok)
        self.assertEqual({k for k, c in checks.items() if not c["met"]},
                         {"one word per meaning", "one dialect"})
        self.assertIn("8 distinct words for 27 meanings", checks["one word per meaning"]["detail"])
        # not measured is not passed, and an error is shown
        ev["vocabulary"] = {"error": "KeyError('x')"}
        ok, checks = evaluate_rung(cfg, rung, ev, lo)
        self.assertFalse(checks["one word per meaning"]["met"])
        self.assertIn("KeyError", checks["one word per meaning"]["detail"])
        # the single-field rungs are not asked: their fields are not all taught
        _, checks = evaluate_rung(cfg, phase_named(cfg, "name-price"),
                                  _swap_evidence(True, True), 10 ** 6)
        self.assertNotIn("one dialect", checks)


if __name__ == "__main__":
    unittest.main()
