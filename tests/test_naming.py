"""The naming curriculum: one field at a time, then all of them, then held out.

These guard the things the ladder is *for*. A lineup that varies one field has
to vary only that field; every field of a lot -- quantity and price included --
gets a naming rung before anything is traded; the combinations set aside must
never be trained on anywhere, by anybody; a code that names whole things must
fail the productivity gate however well it scores on what it drilled; and the
two roles must be one population speaking one language until trading starts.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.config import Config
from orchard.curriculum import (ASK_ALL, N_KINDS, ReferentialWorld, evaluate_rung,
                                ladder, phase_named, phase_schema, rung_budget)
from orchard.env import BUYER, FARMER, buyer_obs
from orchard.population import Population
from orchard.world import (LOT_FIELDS, N_LOT_FIELDS, QUERY_ALL, ComboHoldout, World,
                           lot_spans)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rungs import _swap_evidence          # noqa: E402

NAMING = ["name-fruit", "name-color", "name-quality", "name-quantity", "name-price"]


class TestTheLadderTeachesOneFieldAtATime(unittest.TestCase):
    def test_the_rungs_come_in_the_intended_order(self):
        cfg = Config()
        names = [p.name for p in ladder(cfg)]
        self.assertEqual(names[:7], NAMING + ["name-all", "mutual"])
        self.assertEqual([p.primary for p in ladder(cfg)[:6]], [0, 1, 2, 3, 4, ASK_ALL])
        self.assertTrue(phase_named(cfg, "name-all").whole)

    def test_every_field_of_a_lot_is_named_before_anything_is_traded(self):
        """The rule the ladder is built on: no trading rung invents a word."""
        cfg = Config()
        names = [p.name for p in ladder(cfg)]
        for kind, field in enumerate(LOT_FIELDS):
            rung = next(p for p in ladder(cfg) if p.referential and p.primary == kind)
            self.assertLess(names.index(rung.name), names.index("mutual"),
                            "%s is first named after the community arrives" % field)
        self.assertEqual(N_KINDS, N_LOT_FIELDS + 1)

    def test_each_rung_adds_a_kind_of_round_and_keeps_the_ones_below_it(self):
        """The scaffolding: nothing a rung taught is dropped by the next one."""
        cfg = Config()
        rungs = ladder(cfg)[:N_LOT_FIELDS]
        for i, p in enumerate(rungs):
            self.assertEqual(p.primary, i)
            self.assertGreater(p.mix[i], 0.0, "%s does not draw its own kind" % p.name)
            self.assertEqual(set(p.rehearsed), set(range(i)),
                             "%s should rehearse %s" % (p.name, list(range(i))))
            self.assertAlmostEqual(sum(p.mix), 1.0, places=6)
            for j in range(i + 1, N_KINDS):
                self.assertEqual(p.mix[j], 0.0,
                                 "%s draws a kind it has not taught" % p.name)
        whole = phase_named(cfg, "name-all")
        self.assertEqual(set(whole.rehearsed), set(range(N_LOT_FIELDS)))
        self.assertGreater(whole.mix[ASK_ALL], 0.5)

    def test_a_rung_draws_its_kinds_in_the_proportions_it_asks_for(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        for p in ladder(cfg)[:N_LOT_FIELDS + 1]:
            rb = rw.sample(4096, mix=p.mix)
            for kind, want in enumerate(p.mix):
                got = float((rb.query == kind).float().mean())
                self.assertAlmostEqual(got, want, places=2,
                                       msg="%s: %s rounds" % (p.name, kind))

    def test_a_lot_has_five_fields_and_quantity_includes_none(self):
        """A farmer has to be able to say "none of that", so 0 is a quantity."""
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        rb = rw.sample(4096, mix=(0.2,) * N_LOT_FIELDS + (0.0,))
        self.assertEqual(rb.meanings.shape[-1], N_LOT_FIELDS)
        spans = lot_spans(cfg.world)
        for f in range(N_LOT_FIELDS):
            vals = set(rb.meanings[:, :, f].flatten().tolist())
            self.assertEqual(vals, set(range(spans[f])), LOT_FIELDS[f])
        self.assertIn(0, set(rb.meanings[:, :, 3].flatten().tolist()))

    def test_costs_and_newcomers_wait_for_the_words_to_exist(self):
        """Neither pressure applies while a rung's words are still being invented."""
        from orchard.curriculum import costs_apply, growth_applies
        cfg = Config()
        for p in ladder(cfg):
            if p.referential:
                self.assertFalse(costs_apply(cfg, p),
                                 "%s charges for speaking" % p.name)
                self.assertFalse(growth_applies(cfg, p),
                                 "%s grows the community" % p.name)
        self.assertTrue(growth_applies(cfg, phase_named(cfg, "mutual")))
        self.assertTrue(costs_apply(cfg, phase_named(cfg, "market")))

    def test_agreement_arrives_once_every_word_exists(self):
        """The convention bonus is paid from the last naming rung on.

        The founders keep a dialect each through the single-field rungs, so
        something has to pay them -- and then the community that arrives at
        `mutual` -- to settle on one word per meaning. It must not wait for the
        costs, which are a different pressure and are ramped in later.
        """
        from orchard.curriculum import convention_applies, costs_apply, growth_applies
        cfg = Config()
        first = lambda f: next(p for p in ladder(cfg) if f(cfg, p))
        self.assertEqual(first(convention_applies).name, "name-all")
        self.assertLessEqual(first(convention_applies).index, first(growth_applies).index)
        self.assertLess(first(convention_applies).index, first(costs_apply).index,
                        "agreement waits for the costs again")
        for p in ladder(cfg):
            if p.referential and not p.whole:
                self.assertFalse(convention_applies(cfg, p),
                                 "%s pays two founders to agree with each other" % p.name)

    def test_the_costs_wait_for_every_rung_that_invents_a_word(self):
        """Length and rarity are pressures on a word that exists."""
        from orchard.curriculum import costs_apply
        cfg = Config()
        invents = [p for p in ladder(cfg) if p.referential]
        for p in invents:
            self.assertFalse(costs_apply(cfg, p),
                             "%s charges for speaking while it is still inventing words"
                             % p.name)
        last = max(p.index for p in invents)
        self.assertTrue(costs_apply(cfg, ladder(cfg)[last + 1]),
                        "the costs never come on")
        self.assertEqual(ladder(cfg)[last + 1].name, cfg.reward.costs_from_rung)

    def test_a_query_round_varies_only_the_field_it_asks_about(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        for field in range(N_LOT_FIELDS):
            rb = rw.sample(512, query=field)
            others = [f for f in range(N_LOT_FIELDS) if f != field]
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
        self.assertTrue(bool((obs[:, N_LOT_FIELDS] == 1).all()))
        rb = rw.sample(64)                    # an open round asks for the lot
        self.assertTrue(bool((rb.obs(cfg, rb.informer)[:, N_LOT_FIELDS] == ASK_ALL).all()))

    def test_a_mixed_round_asks_every_field_and_keeps_one_width(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        rb = rw.sample(1024, mixed_query=True)
        self.assertEqual(set(rb.query.tolist()), set(range(N_LOT_FIELDS)))
        for i in range(0, 1024, 97):
            rows = [tuple(r) for r in rb.meanings[i].tolist()]
            self.assertEqual(len(rows), len(set(rows)))


class TestOneLayoutForEveryLot(unittest.TestCase):
    """A request is a lot, seen exactly as the naming describer sees a lot.

    This is what lets the words invented in the naming game place an order
    without being relearned: slot for slot, the buyer's observation in the
    market is the observation a `name-all` describer had for the same lot.
    """

    def test_the_request_is_the_lot_layout(self):
        cfg = Config()
        w = World(cfg.world, random.Random(0))
        for _ in range(50):
            sc = w.sample()
            b = sc.buyer
            self.assertEqual(buyer_obs(sc, cfg)[:N_LOT_FIELDS + 1],
                             (b.want_variety, b.want_color, b.min_quality, b.need_qty,
                              b.max_price, QUERY_ALL))

    def test_the_describer_and_the_requester_share_a_schema(self):
        cfg = Config()
        describer = phase_named(cfg, "name-all").with_informer(BUYER)
        for name in ("order", "offer", "judge", "haggle", "market"):
            self.assertEqual(phase_schema(cfg, BUYER, phase_named(cfg, name)),
                             phase_schema(cfg, BUYER, describer), name)
        self.assertEqual(phase_schema(cfg, FARMER, phase_named(cfg, "mutual")),
                         phase_schema(cfg, BUYER, describer))

    def test_the_barn_is_lot_rows_in_a_random_order(self):
        """The lot a buyer asked about must be found by content, not position."""
        cfg = Config()
        w = World(cfg.world, random.Random(0))
        from orchard.env import farmer_obs
        from orchard.world import n_cells
        cells = n_cells(cfg.world)
        firsts = set()
        for _ in range(60):
            sc = w.sample()
            obs = farmer_obs(sc, cfg)
            rows = [obs[4 * i:4 * i + 4] for i in range(cells)]
            seen = set()
            for fruit, colour, quality, stock in rows:
                seen.add((fruit, colour))
                self.assertEqual(stock, sc.farmer.stock_of(fruit, colour))
                if stock:
                    self.assertEqual(quality, sc.farmer.quality_of(fruit, colour))
            self.assertEqual(len(seen), cells, "a cell is missing or repeated")
            self.assertEqual(obs[4 * cells], sc.farmer.reservation)
            firsts.add(rows[0][:2])
        self.assertGreater(len(firsts), 4, "the barn's first row is always the same cell")


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
        for kw in [{}, {"mixed_query": True}] + [{"query": q} for q in range(N_LOT_FIELDS)]:
            rb = rw.sample(512, **kw)
            self.assertEqual(int(rb.held_out.sum()), 0, kw)
        mb = rw.sample_mutual(512)
        self.assertEqual(int(rw.is_held_out(mb.f_meaning).sum()), 0)
        self.assertEqual(int(rw.is_held_out(mb.b_meaning).sum()), 0)
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
        for kw in ({}, {"query": 1}, {"query": 4}, {"mixed_query": True}):
            rb = rw.sample(256, held_out=True, **kw)
            self.assertTrue(bool(rb.held_out.all()), kw)
        mb = rw.sample_mutual(256, held_out=True)
        self.assertTrue(bool(rw.is_held_out(mb.f_meaning).all()))

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
        # and the report rungs, which run in both directions, are below it
        for n in ("order", "offer", "judge"):
            self.assertLess(names.index(n), at, "%s is played by split roles" % n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
