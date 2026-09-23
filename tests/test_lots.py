"""The lot layout and the mechanisms that came with it.

Each class guards one thing that the redesign rests on and that could quietly
come undone:

* a report rung's checkpoint produces the per-role, per-field evidence its
  promotion is judged on (no test ran a checkpoint on `mutual` or `order`
  before, and a bug in exactly that code once went unnoticed);
* a snapshot of one pool is restored as one pool, never as two copies;
* a newborn's apprenticeship really withholds a share of the meanings;
* the speaker costs wait for the first costed rung to work, then ramp in;
* a completed sale depletes the lot it came out of, not the fruit's first cell;
* the report rungs read their truths off a real ScenarioBatch.
"""
from __future__ import annotations

import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.batched import TensorWorld
from orchard.bottleneck import TranscriptStore, train_newborn
from orchard.config import Config
from orchard.curriculum import (H_REPORT, ReferentialWorld, field_truth, ladder,
                                phase_named, phase_schema, report_spec)
from orchard.env import BUYER, FARMER
from orchard.gumbel import run_and_update_gumbel
from orchard.world import LOT_FIELDS, QUERY_ALL, n_cells

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_curriculum import agents, cfg_small          # noqa: E402


def tiny_run_cfg(start: str) -> Config:
    cfg = cfg_small()
    cfg.channel.max_symbols = 6
    cfg.model.d_model, cfg.model.d_ff = 32, 64
    cfg.population.n_farmers = cfg.population.n_buyers = 2
    cfg.population.founders_farmers = cfg.population.founders_buyers = 2
    cfg.population.turnover = False
    cfg.bottleneck.enabled = False
    cfg.curriculum.start_phase = start
    cfg.curriculum.on_stall = "hold"
    cfg.train.batch_size = 16
    cfg.train.episodes = 16 * 3
    cfg.train.device = "cpu"
    cfg.log.checkpoint_every_updates = 3
    cfg.log.plot = False
    for k in ("intelligibility_episodes", "zeroshot_episodes", "ablation_episodes"):
        setattr(cfg.log, k, 32)
    cfg.log.topsim_samples = 16
    cfg.log.stability_probes = 4
    cfg.log.word_analysis_samples = 20
    return cfg


class TestReportRungCheckpoints(unittest.TestCase):
    """A checkpoint on a report rung yields exactly the evidence its gate reads."""

    def _checkpoint(self, start):
        from orchard.train import Trainer
        cfg = tiny_run_cfg(start)
        out = tempfile.mkdtemp(prefix="orchard_lots_")
        try:
            torch.manual_seed(0)
            t = Trainer(cfg, out, quiet=True)
            t.run()
            row = t.metrics_log.rows[-1]
            last = t.curriculum.last_report
            t.close()
            return row, last
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def _check(self, row, last, names_for):
        ev = row["rung_evidence"]
        for lbl, role in (("farmer", FARMER), ("buyer", BUYER)):
            names = names_for(role)
            if not names:
                self.assertNotIn(lbl + "_report", ev, "a vacuous report for a silent role")
                continue
            self.assertEqual(tuple(ev[lbl + "_field_names"]), tuple(names))
            self.assertEqual(len(ev[lbl + "_fields_intact"]), len(names))
            self.assertEqual(len(ev[lbl + "_field_transfer"]), len(names))
            for k in ("_report", "_new", "_new_transfer", "_report_transfer"):
                self.assertIn(lbl + k, ev)
            self.assertIn("muted_" + lbl + "_new", ev)
        for k in ("holdout_field_ratio", "holdout_field_success", "seen_field_success",
                  "transfer"):
            self.assertIn(k, ev)
        # and the promotion check ran on it, per field
        self.assertTrue(last.get("checks"), "no promotion check was run")
        self.assertTrue(any("carries" in k for k in last["checks"]),
                        list(last["checks"]))
        self.assertIn("reports combinations it never trained on", last["checks"])

    def test_mutual(self):
        row, last = self._checkpoint("mutual")
        self.assertEqual(row["phase"], "mutual")
        self._check(row, last, lambda role: LOT_FIELDS)
        self.assertIn("both decode in the same round", last["checks"])

    def test_order(self):
        cfg = Config()
        order = phase_named(cfg, "order")
        row, last = self._checkpoint("order")
        self.assertEqual(row["phase"], "order")
        self._check(row, last, order.report_names)

    def test_judge(self):
        cfg = Config()
        judge = phase_named(cfg, "judge")
        row, last = self._checkpoint("judge")
        self._check(row, last, judge.report_names)
        self.assertIn("farmer decodes: deal arrives", last["checks"])
        self.assertIn("buyer decodes: still carries stock", last["checks"])


class TestOnePoolStaysOnePool(unittest.TestCase):
    """A shared pool written to a snapshot comes back as one pool.

    Restoring the farmer and buyer lists separately turned one population into
    two copies with the same ids; they drifted apart from the first update and
    the messages collapsed to fruit-only within a checkpoint.
    """

    def _trainer(self, d, name):
        from orchard.train import Trainer
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 2
        cfg.train.episodes = 256
        cfg.train.batch_size = 64
        cfg.train.device = "cpu"
        cfg.log.plot = False
        return Trainer(cfg, d + "/" + name, quiet=True)

    def test_a_shared_pool_is_restored_as_one_pool(self):
        with tempfile.TemporaryDirectory() as d:
            tr = self._trainer(d, "a")
            self.assertTrue(tr.pop.shared)
            tr.curriculum.index = [p.name for p in tr.curriculum.phases].index("mutual")
            path = tr.save_snapshot("t")
            tr2 = self._trainer(d, "b")
            tr2.load_snapshot(path)
            self.assertTrue(tr2.pop.shared)
            self.assertIs(tr2.pop.farmers, tr2.pop.buyers)
            f, b = tr2.pop.pair(32)
            self.assertFalse(bool((f == b).any()))
            tr.close()
            tr2.close()

    def test_two_copies_from_the_old_bug_are_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            tr = self._trainer(d, "a")
            tr.curriculum.index = [p.name for p in tr.curriculum.phases].index("mutual")
            path = tr.save_snapshot("t")
            st = torch.load(path, map_location="cpu", weights_only=False)
            # the old bug: the same ids, but the "buyer" copies have drifted
            st["shared"] = False
            for rec in st["buyers"]:
                rec["net"] = {k: v + 0.5 for k, v in rec["net"].items()}
            torch.save(st, path)
            tr2 = self._trainer(d, "b")
            tr2.load_snapshot(path)
            self.assertIs(tr2.pop.farmers, tr2.pop.buyers)
            for a, b in zip(tr.pop.farmers, tr2.pop.farmers):
                self.assertTrue(torch.equal(a.net.token_head.weight,
                                            b.net.token_head.weight))
            tr.close()
            tr2.close()

    def test_a_snapshot_after_the_split_stays_split(self):
        with tempfile.TemporaryDirectory() as d:
            tr = self._trainer(d, "a")
            tr.curriculum.index = [p.name for p in tr.curriculum.phases].index("haggle")
            tr.maybe_split_roles(tr.curriculum.phase, log=lambda *_: None)
            self.assertFalse(tr.pop.shared)
            path = tr.save_snapshot("t")
            tr2 = self._trainer(d, "b")
            tr2.load_snapshot(path)
            self.assertFalse(tr2.pop.shared)
            self.assertIsNot(tr2.pop.farmers, tr2.pop.buyers)
            self.assertEqual(len(tr2.pop.buyers), len(tr.pop.buyers))
            tr.close()
            tr2.close()


class TestTheBottleneckWithholdsMeanings(unittest.TestCase):
    def _store(self, cfg):
        cfg.bottleneck.only_successful = False
        cfg.bottleneck.epochs = 1
        torch.manual_seed(2)
        f, b = agents(cfg)
        store = TranscriptStore(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        B = 256
        i = torch.arange(B)
        fi, bi = i % 2, torch.div(i, 2, rounding_mode="floor") % 2
        for _ in range(2):
            batch, _ = run_and_update_gumbel(cfg, rw.sample_mutual(B), f, b, fi, bi,
                                             phase=phase_named(cfg, "mutual"), update=500)
            store.add_batch(batch, f, b, 0)
        return store

    def test_a_share_of_the_combinations_is_kept_from_a_newborn(self):
        cfg = cfg_small()
        cfg.bottleneck.meaning_holdout = 0.25
        store = self._store(cfg)
        kinds = {it.meaning for it in store._buf}
        self.assertGreater(len(kinds), 8)
        self.assertTrue(all(len(k) == 3 for k in kinds), "a meaning is a combination")
        newborn = agents(cfg, 1)[0][0]
        info = train_newborn(cfg, newborn, store, random.Random(0))
        self.assertEqual(info["withheld_meanings"], int(round(0.25 * len(kinds))))
        self.assertGreater(info["withheld_meanings"], 0)
        self.assertLess(info["n_samples"], len(store))
        for m in info["withheld"]:
            self.assertIn(tuple(m), kinds)

    def test_zero_withholds_nothing(self):
        cfg = cfg_small()
        cfg.bottleneck.meaning_holdout = 0.0
        store = self._store(cfg)
        newborn = agents(cfg, 1)[0][0]
        info = train_newborn(cfg, newborn, store, random.Random(0))
        self.assertEqual(info["withheld_meanings"], 0)
        self.assertEqual(info["n_samples"], len(store))


class TestTheCostsWaitThenRamp(unittest.TestCase):
    def _trainer(self, d, start):
        from orchard.train import Trainer
        cfg = tiny_run_cfg(start)
        cfg.reward.costs_ramp_updates = 4
        return Trainer(cfg, d, quiet=True)

    def test_off_in_the_naming_rungs_waiting_in_mutual_on_after(self):
        with tempfile.TemporaryDirectory() as d:
            t = self._trainer(d, "name-all")
            t.update_cost_gate()
            self.assertEqual(t.cost_gate, 0.0)
            t.close()
        with tempfile.TemporaryDirectory() as d:
            t = self._trainer(d, "mutual")
            t.update_cost_gate()
            self.assertEqual(t.cost_gate, 0.0, "charged before the rung works")
            # the rung starts working: rolling success clears the floor
            t.rung_success.extend([1.0] * 300)
            t.updates = 100
            t.update_cost_gate()
            self.assertEqual(t._costs_ramp_start, 100)
            self.assertEqual(t.cost_gate, 0.0)
            t.updates = 102
            t.update_cost_gate()
            self.assertAlmostEqual(t.cost_gate, 0.5)
            t.updates = 110
            t.update_cost_gate()
            self.assertEqual(t.cost_gate, 1.0)
            # and the state survives a snapshot
            path = t.save_snapshot("t")
            st = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(st["costs_ramp_start"], 100)
            t.close()
        with tempfile.TemporaryDirectory() as d:
            t = self._trainer(d, "order")
            t.update_cost_gate()
            self.assertEqual(t.cost_gate, 1.0, "a later rung has the costs on")
            t.close()


class TestSalesDepleteTheRightLot(unittest.TestCase):
    def test_tensor_settlement_uses_the_deal_cell(self):
        from orchard.economy import Economy
        from orchard.world import World
        cfg = cfg_small()
        cfg.economy.persistent_inventory = True
        world = World(cfg.world, random.Random(0))
        eco = Economy(cfg, world, random.Random(0), n_farms=2)
        C = cfg.world.n_colors
        before = list(eco.inventories[1].remaining)
        cell = next(c for c in range(len(before)) if before[c] > 0)
        res = {"success": torch.tensor([True]), "traded_qty": torch.tensor([1]),
               "trade_value": torch.tensor([1.5]), "farmer_profit": torch.tensor([0.5]),
               "deal_cell": torch.tensor([cell])}
        eco.settle_tensor(torch.tensor([1]), res)
        after = eco.inventories[1].remaining
        self.assertEqual(after[cell], before[cell] - 1)
        for c in range(len(before)):
            if c != cell:
                self.assertEqual(after[c], before[c])

    def test_scalar_settlement_matches(self):
        from orchard.economy import Economy
        from orchard.env import Decision, resolve
        from orchard.world import World
        cfg = cfg_small()
        world = World(cfg.world, random.Random(1))
        eco = Economy(cfg, world, random.Random(1), n_farms=1)
        scen, f_idx, _ = eco.make_batch(40, 1, 1)
        sc = next(s for s in scen if s.viable)
        lo, hi = sc.zopa
        d = Decision(1, sc.deal_variety, sc.deal_qty, (lo + hi) // 2)
        o = resolve(cfg, sc, d, d)
        self.assertTrue(o.success)
        self.assertEqual(o.traded_cell, sc.deal_cell)
        before = list(eco.inventories[0].remaining)
        eco.settle(torch.tensor([0]), [o])
        self.assertEqual(eco.inventories[0].remaining[sc.deal_cell],
                         max(0, before[sc.deal_cell] - o.traded_qty))


class TestReportTruths(unittest.TestCase):
    def test_every_field_reads_off_a_real_batch(self):
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(0))
        sb = tw.sample(64)
        for name in ("order", "offer", "judge"):
            spec = report_spec(cfg, phase_named(cfg, name), sb)
            for role in (FARMER, BUYER):
                for fname, head, truth in spec[role]:
                    self.assertEqual(tuple(truth.shape), (64,), fname)
                    self.assertTrue(torch.equal(truth, field_truth(cfg, sb, fname)))
        # the farmer reports the request on the same heads mutual used
        spec = report_spec(cfg, phase_named(cfg, "order"), sb)
        self.assertEqual([h for _, h, _ in spec[FARMER]], list(H_REPORT))
        self.assertTrue(torch.equal(spec[FARMER][0][2], sb.want_variety))
        self.assertTrue(torch.equal(spec[FARMER][3][2], sb.need_qty))
        # a lot the farmer does not carry is reported as stock 0
        spec = report_spec(cfg, phase_named(cfg, "offer"), sb)
        stock = dict((n, t) for n, _, t in spec[BUYER])["stock"]
        self.assertTrue(bool(((stock == 0) == ~sb.variety_ok).all()))

    def test_the_farmer_can_find_the_lot_it_was_asked_about(self):
        """Every request's lot is somewhere in the farmer's rows, by content."""
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(1))
        sb = tw.sample(128)
        obs = sb.obs(cfg, FARMER)
        cells = n_cells(cfg.world)
        rows = obs[:, :4 * cells].reshape(128, cells, 4)
        hit = (rows[:, :, 0] == sb.want_variety.unsqueeze(1)) & \
              (rows[:, :, 1] == sb.want_color.unsqueeze(1))
        self.assertTrue(bool((hit.sum(1) == 1).all()), "a requested lot is not exactly one row")
        found = rows[hit]
        self.assertTrue(torch.equal(found[:, 3], sb.offered_stock))
        self.assertTrue(torch.equal(found[:, 2][sb.variety_ok], sb.offered_quality[sb.variety_ok]))
        # and the buyer's request sits in the lot layout, asking for all of it
        b = sb.obs(cfg, BUYER)
        self.assertTrue(torch.equal(b[:, :5], sb.request))
        self.assertTrue(bool((b[:, 5] == QUERY_ALL).all()))


class TestConventionsKeyOnWhatWasAsked(unittest.TestCase):
    def test_a_lot_asked_for_its_colour_is_not_the_whole_lot(self):
        from orchard.conventions import PopulationUsage
        cfg = cfg_small()
        u = PopulationUsage(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        colour = rw.sample(8, query=1)
        whole = rw.sample(8)
        k1 = u._keys(phase_named(cfg, "name-color"), FARMER, colour.obs(cfg, FARMER))
        k2 = u._keys(phase_named(cfg, "name-all"), FARMER, whole.obs(cfg, FARMER))
        self.assertTrue(all(k[-1] == 1 for k in k1))
        self.assertTrue(all(k[-1] == QUERY_ALL for k in k2))
        # a buyer's request in the market shares the whole-lot convention
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(0))
        k3 = u._keys(phase_named(cfg, "order"), BUYER, tw.sample(8).obs(cfg, BUYER))
        self.assertTrue(all(k[0] == "tuple" and k[-1] == QUERY_ALL for k in k3))
        self.assertEqual(len(k3[0]), len(k2[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTheBarnLookup(unittest.TestCase):
    """The farmer's one new skill in the market has a structure to express it.

    Measured, supervised, with the answer given: the plain network left the
    stock of the asked-for lot at the base rate after 800 steps (0.27-0.34)
    while one cross-attention step to the barn rows reached 1.00 by step 500.
    """

    def test_it_is_only_active_on_a_barn(self):
        from orchard.agents import CommNet
        cfg = cfg_small()
        net = CommNet(cfg, FARMER)
        for name in ("order", "offer", "judge", "haggle", "market"):
            self.assertTrue(net.is_barn(phase_schema(cfg, FARMER, phase_named(cfg, name))), name)
            self.assertFalse(net.is_barn(phase_schema(cfg, BUYER, phase_named(cfg, name))), name)
        for name in ("name-fruit", "name-all", "mutual"):
            for role in (FARMER, BUYER):
                self.assertFalse(net.is_barn(phase_schema(cfg, role, phase_named(cfg, name))))

    def test_it_barely_moves_a_hidden_state_at_birth(self):
        """Weights that never saw a barn must keep their listening at `order`."""
        from orchard.agents import CommNet
        cfg = cfg_small()
        torch.manual_seed(0)
        net = CommNet(cfg, FARMER).eval()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(0))
        obs = tw.sample(6).obs(cfg, FARMER)
        toks = torch.full((6, cfg.channel.dialogue_len), cfg.channel.pad_id, dtype=torch.long)
        with torch.no_grad():
            on = net.encode(obs, toks)
            cfg.model.barn_lookup = False
            off = net.encode(obs, toks)
        rel = float((on - off).norm(dim=-1).mean() / off.norm(dim=-1).mean())
        self.assertLess(rel, 0.1, "the untrained lookup rewrote the hidden state (%.3f)" % rel)
        self.assertGreater(rel, 0.0, "the lookup is not applied to a barn at all")

    def test_it_learns_the_lookup_when_told_the_answer(self):
        """A short supervised drill on what `offer` asks: the request's fruit and
        colour, and the stock and quality of that lot. The plain network stays at
        the base rate on the last two for as long as anyone has run it."""
        import torch.nn.functional as F
        from orchard.agents import CommNet
        cfg = cfg_small()
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(1))
        offer = phase_named(cfg, "offer")
        schema = phase_schema(cfg, FARMER, offer)
        mask = offer.self_mask(cfg, FARMER)
        c = cfg.channel

        def batch(n=128):
            sb = tw.sample(n)
            toks = torch.full((n, c.dialogue_len), c.pad_id, dtype=torch.long)
            toks[:, 0] = sb.want_variety
            toks[:, 1] = c.space_id
            toks[:, 2] = 4 + sb.want_color
            toks[:, 3] = c.end_id
            return sb.obs(cfg, FARMER), toks, (
                sb.want_variety, sb.want_color,
                sb.offered_stock.clamp(0, cfg.world.max_qty), sb.offered_quality)

        def heads(net, h):
            return (net.belief_variety_head(h), net.belief_color_head(h),
                    net.belief_qty_head(h), net.belief_quality_head(h))

        torch.manual_seed(0)
        net = CommNet(cfg, FARMER)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(300):
            obs, toks, ys = batch()
            h = net.encode(obs, toks, schema=schema, self_mask=mask)[:, -1]
            loss = sum(F.cross_entropy(lg, y) for lg, y in zip(heads(net, h), ys))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            obs, toks, ys = batch(256)
            h = net.encode(obs, toks, schema=schema, self_mask=mask)[:, -1]
            acc = [float((lg.argmax(-1) == y).float().mean()) for lg, y in zip(heads(net, h), ys)]
        self.assertGreater(acc[2], 0.6, "stock of the asked-for lot not found (%.2f)" % acc[2])
        self.assertGreater(acc[3], 0.8, "quality of the asked-for lot not found (%.2f)" % acc[3])
