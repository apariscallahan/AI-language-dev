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
        self.assertEqual(names[:5], ["name-fruit", "name-color", "name-quality",
                                     "name-all", "mutual"])
        self.assertEqual([p.primary for p in ladder(cfg)[:4]], [0, 1, 2, 3])
        self.assertTrue(phase_named(cfg, "name-all").whole)

    def test_each_rung_adds_a_kind_of_round_and_keeps_the_ones_below_it(self):
        """The scaffolding: nothing a rung taught is dropped by the next one."""
        cfg = Config()
        rungs = ladder(cfg)[:4]
        for i, p in enumerate(rungs):
            self.assertEqual(p.primary, i)
            self.assertGreater(p.mix[i], 0.0, "%s does not draw its own kind" % p.name)
            self.assertEqual(set(p.rehearsed), set(range(i)),
                             "%s should rehearse %s" % (p.name, list(range(i))))
            self.assertAlmostEqual(sum(p.mix), 1.0, places=6)
            for j in range(i + 1, 4):
                self.assertEqual(p.mix[j], 0.0,
                                 "%s draws a kind it has not taught" % p.name)

    def test_a_rung_draws_its_kinds_in_the_proportions_it_asks_for(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        for p in ladder(cfg)[:4]:
            rb = rw.sample(4096, mix=p.mix)
            for kind, want in enumerate(p.mix):
                got = float((rb.query == kind).float().mean())
                self.assertAlmostEqual(got, want, places=2,
                                       msg="%s: %s rounds" % (p.name, kind))

    def test_costs_and_newcomers_wait_for_the_words_to_exist(self):
        """Neither pressure applies while a rung's words are still being invented."""
        from orchard.curriculum import costs_apply, growth_applies
        cfg = Config()
        for p in ladder(cfg):
            if p.naming and p.primary < 3 or p.name == "name-all":
                self.assertFalse(costs_apply(cfg, p),
                                 "%s charges for speaking" % p.name)
                self.assertFalse(growth_applies(cfg, p),
                                 "%s grows the community" % p.name)
        self.assertTrue(growth_applies(cfg, phase_named(cfg, "mutual")))
        self.assertTrue(costs_apply(cfg, phase_named(cfg, "market")))

    def test_agreement_arrives_at_the_first_rung_that_invents_no_word(self):
        """The convention bonus follows the same rule as the costs, one rung up.

        A pressure to reuse a word is off while the rung still has to invent
        one, and on at the first rung that only reuses them. For the costs that
        is `offer`; for agreement it is `name-all`, which invents nothing --
        fruit, colour and quality were each invented and promoted below it, and
        its own job is to say three of them at once.

        It used to wait for the community at `mutual`, which left four rungs in
        which nothing paid a speaker for saying the same thing twice. The run
        that was measured there had two founders sharing no form at all
        (within-role coherence 0.15-0.17) and a speaker so unsure of its own
        words that sampled play showed 686 of them over a 64-meaning world.
        """
        from orchard.curriculum import convention_applies, costs_apply
        cfg = Config()
        first = lambda f: next(p for p in ladder(cfg) if f(cfg, p))
        self.assertLess(first(convention_applies).index, first(costs_apply).index,
                        "agreement waits for the costs again")
        for p in ladder(cfg):
            if p.naming and p.primary < 3:
                self.assertFalse(convention_applies(cfg, p),
                                 "%s pays for agreeing on a word it is still "
                                 "inventing" % p.name)
        self.assertTrue(convention_applies(cfg, phase_named(cfg, "name-all")),
                        "nothing pays for reusing a word on the rung whose whole "
                        "job is to reuse three of them")

    def test_the_costs_are_a_per_rung_rule_not_a_threshold(self):
        """`ask-qty` and `quote` sit above `mutual` and still have a word to
        invent, so no single threshold gets this right: at `offer` it spares
        them but also spares `mutual` and `order`, which invent nothing; at
        `mutual` it charges them while they are still naming quantity and price.
        """
        from orchard.curriculum import costs_apply
        cfg = Config()
        want = {"name-fruit": False, "name-color": False, "name-quality": False,
                "name-all": False, "mutual": True, "ask-qty": False,
                "order": True, "quote": False, "offer": True, "judge": True,
                "haggle": True, "bargain": True, "market": True}
        for p in ladder(cfg):
            self.assertEqual(costs_apply(cfg, p), want[p.name],
                             "%s: costs %s, expected %s"
                             % (p.name, costs_apply(cfg, p), want[p.name]))
        # and the floor still holds them off entirely if a run wants that
        cfg.reward.costs_from_rung = "market"
        self.assertFalse(costs_apply(cfg, phase_named(cfg, "mutual")))
        self.assertTrue(costs_apply(cfg, phase_named(cfg, "market")))

    def test_the_costs_price_a_fused_name_above_a_compositional_one(self):
        """Why `mutual` gets them: it is the first rung with no lineup, so
        nothing else there forces a message to decompose.

        The pressure has to survive the obvious objection -- that a length cost
        just makes everything shorter, and the shortest code is the collapse
        this project keeps rediscovering. It does not, because the collapse
        cannot carry the meaning space: the task forbids what the cost would
        otherwise reward. That is the difference from the convention bonus,
        whose collapse was both cheap and well paid.
        """
        import torch
        from orchard.env import length_cost
        cfg = Config()
        c, w = cfg.channel, cfg.world
        meanings = w.n_varieties * w.n_colors * w.n_quality

        def cost_of(words):
            out = []
            for word in words:
                for j, a in enumerate(word):
                    if j:
                        out.append(c.hyphen_id)
                    out.append(a)
                out.append(c.space_id)
            out[-1] = c.end_id
            toks = torch.full((1, c.dialogue_len), c.pad_id, dtype=torch.long)
            toks[0, :len(out)] = torch.tensor(out)
            v = float(length_cost(cfg, toks, list(range(c.max_msg_len)))[0])
            return v, c.atomic_vocab ** sum(len(x) for x in words)

        one_atom, room = cost_of([[1]])
        self.assertLess(room, meanings,
                        "one atom can encode the whole world, so the cheapest "
                        "code is the collapse and this pressure is unsafe")
        fused_2, room_2 = cost_of([[1, 2]])
        split_2, _ = cost_of([[1], [2]])
        fused_3, _ = cost_of([[1, 2, 3]])
        split_3, room_3 = cost_of([[1], [2], [3]])
        self.assertGreaterEqual(room_2, meanings)
        self.assertGreaterEqual(room_3, meanings)
        self.assertGreater(fused_2, split_2 * 2,
                           "a fused two-atom label is not meaningfully dearer "
                           "than two short words (%.4f vs %.4f)" % (fused_2, split_2))
        self.assertGreater(fused_3, split_3 * 2,
                           "a fused three-atom label is not meaningfully dearer "
                           "than three short words (%.4f vs %.4f)" % (fused_3, split_3))

    def test_the_costs_wait_for_every_rung_that_invents_a_word(self):
        """Length and rarity are pressures on a word that exists."""
        from orchard.curriculum import costs_apply
        cfg = Config()
        invents = [p for p in ladder(cfg)
                   if p.referential or (p.order and p.asks_first in
                                        ("quantity", "price"))]
        for p in invents:
            self.assertFalse(costs_apply(cfg, p),
                             "%s charges for speaking while it is still "
                             "inventing %s" % (p.name, p.asks_first or "words"))
        last = max(p.index for p in invents)
        self.assertTrue(costs_apply(cfg, ladder(cfg)[last + 1]),
                        "the costs never come on")

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

    def test_the_split_happens_where_the_two_roles_start_to_differ(self):
        """One pool for every rung that is one language in two seats."""
        cfg = Config()
        names = [p.name for p in ladder(cfg)]
        at = names.index(cfg.curriculum.split_roles_at)
        self.assertEqual(cfg.curriculum.split_roles_at, "haggle")
        # nothing below the split pays the two roles differently
        self.assertTrue(all(not ladder(cfg)[i].trading for i in range(at)))
        # and the request rungs, which run in both directions, are below it
        for n in ("ask-qty", "order", "quote", "offer"):
            self.assertLess(names.index(n), at, "%s is played by split roles" % n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
