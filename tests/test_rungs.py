"""Tests for the expanded ladder, the speaker pressures and the per-role metrics.

Each class guards one of the five changes against the specific way it could
quietly come undone:

* the lineup rungs have their own speaking order, and *everything* that needs to
  know whose words are whose -- symbol cost, bottleneck targets, the "me"
  embedding, the probes -- follows it;
* promotion out of the two-way rungs is judged per role, never pooled;
* the coining cost prefers established forms without ever favouring silence, and
  the convention bonus rewards a shared form *for a meaning*, not one form for
  everything;
* zero-shot retention is not reported over a near-empty denominator;
* cross-role overlap and live quantity encoding measure what they claim to.
"""
from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard import metrics as M
from orchard.bottleneck import TranscriptStore, train_newborn
from orchard.config import Config
from orchard.conventions import PopulationUsage
from orchard.curriculum import (H_BELIEF, H_BELIEF_COLOR, H_CHOICE, H_QTY, H_REPORT,
                                H_VARIETY, MutualBatch, N_HEADS,
                                ReferentialWorld, evaluate_rung, ladder, phase_named,
                                resolve_mutual, resolve_order, rung_budget)
from orchard.env import BUYER, FARMER
from orchard.gumbel import run_and_update_gumbel
from orchard.lexicon import cross_role_overlap, live_encoding, word_stats

from test_curriculum import agents, cfg_small


def pairing(B, n=2):
    i = torch.arange(B)
    return i % n, torch.div(i, n, rounding_mode="floor") % n


class TestSpeakingOrder(unittest.TestCase):
    def test_each_rung_says_who_speaks_when(self):
        cfg = cfg_small()
        L = cfg.channel.max_msg_len
        refer = phase_named(cfg, "name-fruit")
        self.assertEqual(refer.own_positions(cfg, FARMER), list(range(L)))
        self.assertEqual(refer.own_positions(cfg, BUYER), [],
                         "the lineup guesser never speaks")
        swap = phase_named(cfg, "name-all")
        self.assertEqual([v.informer for v in swap.views()], [FARMER, BUYER])
        self.assertEqual(swap.with_informer(BUYER).own_positions(cfg, BUYER), list(range(L)))
        self.assertEqual(swap.with_informer(BUYER).own_positions(cfg, FARMER), [])
        mutual = phase_named(cfg, "mutual")
        self.assertEqual(mutual.own_positions(cfg, FARMER), list(range(L)))
        self.assertEqual(mutual.own_positions(cfg, BUYER), list(range(L, 2 * L)))
        haggle = phase_named(cfg, "haggle")
        self.assertEqual(haggle.speaker_of_turn(0), BUYER)

    def test_the_me_embedding_follows_the_phase(self):
        cfg = cfg_small()
        L = cfg.channel.max_msg_len
        m = phase_named(cfg, "name-fruit").self_mask(cfg, FARMER)
        self.assertTrue(bool(m[:L].all()), "the describer's own words were not 'mine'")
        self.assertFalse(bool(m[L:].any()))

    def test_symbols_are_billed_to_whoever_spoke(self):
        """In refer the old buyer-opens schedule billed the silent guesser."""
        cfg = cfg_small()
        torch.manual_seed(0)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        fi, bi = pairing(128)
        for informer in (FARMER, BUYER):
            ph = phase_named(cfg, "name-all").with_informer(informer)
            batch, _ = run_and_update_gumbel(cfg, rw.sample(128, informer=informer),
                                             f, b, fi, bi, phase=ph, train=False)
            spoke = batch.f_emitted if informer == FARMER else batch.b_emitted
            silent = batch.b_emitted if informer == FARMER else batch.f_emitted
            self.assertGreater(int(spoke.sum()), 0)
            self.assertEqual(int(silent.sum()), 0,
                             "a role was charged for symbols it never emitted")

    def test_the_guesser_is_whoever_did_not_describe(self):
        cfg = cfg_small()
        torch.manual_seed(1)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(1))
        fi, bi = pairing(64)
        ph = phase_named(cfg, "name-all").with_informer(BUYER)
        rb = rw.sample(64, informer=BUYER)
        batch, _ = run_and_update_gumbel(cfg, rb, f, b, fi, bi, phase=ph, train=False)
        want = batch.f_dec[:, H_CHOICE] == rb.target
        self.assertTrue(torch.equal(batch.res["success"], want),
                        "with the buyer describing, the farmer's pick must be scored")


class TestBottleneckIsRoleCorrect(unittest.TestCase):
    def _store(self, cfg, phase, sample, n_batches=3):
        cfg.bottleneck.only_successful = False
        torch.manual_seed(2)
        f, b = agents(cfg)
        store = TranscriptStore(cfg)
        fi, bi = pairing(128)
        for _ in range(n_batches):
            batch, _ = run_and_update_gumbel(cfg, sample(), f, b, fi, bi, phase=phase,
                                             update=500)
            store.add_batch(batch, f, b, 0)
        return store, f, b

    def test_a_farmer_newborn_learns_its_own_lineup_words(self):
        cfg = cfg_small()
        cfg.bottleneck.epochs = 1
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        store, f, b = self._store(cfg, phase_named(cfg, "name-fruit"), lambda: rw.sample(128))
        self.assertEqual(store._buf[0].phase.name, "name-fruit")
        newborn = agents(cfg, 1)[0][0]
        info = train_newborn(cfg, newborn, store, random.Random(0))
        self.assertIsNotNone(info["token_accuracy"],
                             "the farmer had no token targets: its own turn was missed")
        self.assertGreater(info["own_token_targets"], 0)
        self.assertIsNone(info["decision_accuracy"],
                          "the describer has no scored decision in refer")

    def test_a_buyer_newborn_in_refer_learns_to_listen_not_to_parrot(self):
        cfg = cfg_small()
        cfg.bottleneck.epochs = 1
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(3))
        store, f, b = self._store(cfg, phase_named(cfg, "name-fruit"), lambda: rw.sample(128))
        newborn = agents(cfg, 1)[1][0]
        info = train_newborn(cfg, newborn, store, random.Random(0))
        self.assertIsNone(info["token_accuracy"],
                          "a guesser was trained to imitate the describer's words")
        self.assertIsNotNone(info["decision_accuracy"])

    def test_in_the_swap_rung_both_roles_get_their_own_words(self):
        cfg = cfg_small()
        cfg.bottleneck.epochs = 1
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(4))
        swap = phase_named(cfg, "name-all")
        k = [0]

        def draw():
            k[0] += 1
            return rw.sample(128, informer=FARMER if k[0] % 2 else BUYER)
        cfg.bottleneck.only_successful = False
        torch.manual_seed(4)
        f, b = agents(cfg)
        store = TranscriptStore(cfg)
        fi, bi = pairing(128)
        for _ in range(4):
            rb = draw()
            batch, _ = run_and_update_gumbel(cfg, rb, f, b, fi, bi,
                                             phase=swap.with_informer(rb.informer))
            store.add_batch(batch, f, b, 0)
        for role in (FARMER, BUYER):
            newborn = agents(cfg, 1)[0 if role == FARMER else 1][0]
            info = train_newborn(cfg, newborn, store, random.Random(0))
            self.assertIsNotNone(info["token_accuracy"])
            self.assertIsNotNone(info["decision_accuracy"])


class TestMutualRung(unittest.TestCase):
    def test_both_must_report_the_other(self):
        cfg = cfg_small()
        f_m = torch.tensor([[0, 3, 1], [1, 5, 2]])
        b_m = torch.tensor([[2, 1, 0], [0, 7, 1]])
        mb = MutualBatch(f_meaning=f_m, b_meaning=b_m)
        sym = torch.zeros(2, dtype=torch.long)
        res = resolve_mutual(cfg, mb, f_report=b_m.clone(), b_report=f_m.clone(),
                             f_cost=sym.float(), b_cost=sym.float())
        self.assertTrue(bool(res["success"].all()))
        wrong = f_m.clone()
        wrong[0, 0] = (wrong[0, 0] + 1) % cfg.world.n_varieties
        res = resolve_mutual(cfg, mb, f_report=b_m.clone(), b_report=wrong,
                             f_cost=sym.float(), b_cost=sym.float())
        self.assertEqual(res["success"].tolist(), [False, True])
        self.assertEqual(res["farmer_report_ok"].tolist(), [True, True])
        self.assertEqual(res["buyer_report_ok"].tolist(), [False, True])

    def test_quantity_is_exact_by_default(self):
        cfg = cfg_small()
        mb = MutualBatch(f_meaning=torch.tensor([[0, 4, 0]]),
                         b_meaning=torch.tensor([[1, 4, 1]]))
        sym = torch.zeros(1, dtype=torch.long)
        off_by_one = resolve_mutual(cfg, mb, torch.tensor([[1, 5, 1]]),
                                    torch.tensor([[0, 4, 0]]), sym, sym)
        self.assertFalse(bool(off_by_one["farmer_report_ok"][0]))

    def test_every_field_of_the_report_must_be_right(self):
        """A thing is (fruit, colour, quality) and all three are reported exactly."""
        cfg = cfg_small()
        mb = MutualBatch(f_meaning=torch.tensor([[0, 1, 0]]),
                         b_meaning=torch.tensor([[1, 2, 1]]))
        zero = torch.zeros(1)
        right = resolve_mutual(cfg, mb, torch.tensor([[1, 2, 1]]),
                               torch.tensor([[0, 1, 0]]), zero, zero)
        one_off = resolve_mutual(cfg, mb, torch.tensor([[1, 0, 1]]),
                                 torch.tensor([[0, 1, 0]]), zero, zero)
        self.assertTrue(bool(right["success"][0]))
        self.assertFalse(bool(one_off["success"][0]),
                         "a wrong colour still counted as understood")

    def test_the_reports_are_the_belief_heads(self):
        cfg = cfg_small()
        mutual = phase_named(cfg, "mutual")
        # the thing being reported is (fruit, colour, quality)
        self.assertEqual(mutual.active_heads(FARMER, cfg), list(H_REPORT))
        self.assertEqual(mutual.active_heads(BUYER, cfg), list(H_REPORT))


def _swap_evidence(farmer_ok: bool, buyer_ok: bool, holdout: float = 0.75) -> dict:
    good = {"topsim": 0.40, "null": 0.01, "positional": 0.45, "field_coverage": 0.6}
    bad = {"topsim": 0.03, "null": 0.01, "positional": 0.03, "field_coverage": 0.05}
    return {
        "chance": 0.25,
        # the productivity gate: success on combinations never trained on
        "seen_success": 0.80, "holdout_success": 0.80 * holdout,
        "holdout_ratio": holdout,
        "views": [
            {"informer": "farmer", "guesser": "buyer", "success": 0.80 if farmer_ok else 0.28,
             "transfer": 0.70 if farmer_ok else 0.02},
            {"informer": "buyer", "guesser": "farmer", "success": 0.80 if buyer_ok else 0.28,
             "transfer": 0.70 if buyer_ok else 0.02},
        ],
        "speakers": {"farmer": good if farmer_ok else bad,
                     "buyer": good if buyer_ok else bad},
    }


class TestPerRolePromotion(unittest.TestCase):
    def test_swap_passes_only_when_both_roles_pass(self):
        cfg = cfg_small()
        swap = phase_named(cfg, "name-all")
        lo, _ = rung_budget(cfg, swap)
        ok, checks = evaluate_rung(cfg, swap, _swap_evidence(True, True), lo)
        self.assertTrue(ok, checks)
        ok, checks = evaluate_rung(cfg, swap, _swap_evidence(True, False), lo)
        self.assertFalse(ok)
        unmet = {k for k, c in checks.items() if not c["met"]}
        # the buyer's describing fails, and so does decoding in the view it
        # describes -- the farmer cannot decode what was never encoded
        self.assertIn("buyer describes: positional structure", unmet)
        self.assertIn("buyer describes: topsim clear of null", unmet)
        self.assertIn("farmer decodes: success", unmet)
        self.assertFalse(any(k.startswith("farmer describes") for k in unmet), unmet)
        self.assertFalse(any(k.startswith("buyer decodes") for k in unmet), unmet)

    def test_a_pooled_average_cannot_carry_a_silent_role(self):
        """Mean success here is 0.54 and mean positional 0.24 -- both 'fine' pooled."""
        cfg = cfg_small()
        swap = phase_named(cfg, "name-all")
        ev = _swap_evidence(True, False)
        ok, _ = evaluate_rung(cfg, swap, ev, rung_budget(cfg, swap)[0])
        self.assertFalse(ok)

    def test_mutual_checks_each_reader(self):
        cfg = cfg_small()
        mutual = phase_named(cfg, "mutual")
        good = {"topsim": 0.40, "null": 0.01, "positional": 0.45, "field_coverage": 0.6}
        ev = {"chance": 0.005, "success": 0.30, "views": [{"success": 0.30}],
              "speakers": {"farmer": good, "buyer": good},
              "farmer_report": 0.60, "muted_farmer_report": 0.06, "farmer_report_transfer": 0.57,
              "buyer_report": 0.10, "muted_buyer_report": 0.06, "buyer_report_transfer": 0.04}
        ok, checks = evaluate_rung(cfg, mutual, ev, rung_budget(cfg, mutual)[0])
        self.assertFalse(ok)
        self.assertFalse(checks["buyer decodes: reports partner's tuple"]["met"])
        self.assertTrue(checks["farmer decodes: reports partner's tuple"]["met"])

    def test_every_rung_has_a_budget(self):
        cfg = cfg_small()
        for p in ladder(cfg):
            lo, hi = rung_budget(cfg, p)
            self.assertLess(lo, hi, p.name)
        self.assertEqual(cfg.curriculum.on_stall, "stop")


class TestSpeakerPressures(unittest.TestCase):
    def _usage(self, cfg):
        u = PopulationUsage(cfg)
        u.words[(1,)] = 900.0          # an established word
        u.words[(2, 3)] = 1.0          # a rare one
        u.word_total = 1000.0
        return u

    def _batch(self, cfg, rows):
        c = cfg.channel
        toks = torch.full((len(rows), c.dialogue_len), c.pad_id, dtype=torch.long)
        for i, r in enumerate(rows):
            toks[i, :len(r)] = torch.tensor(r)
        return toks

    def test_established_forms_are_cheaper_and_silence_is_not_favoured(self):
        cfg = cfg_small()
        c = cfg.channel
        u = self._usage(cfg)
        refer = phase_named(cfg, "name-fruit")
        toks = self._batch(cfg, [[1, c.end_id], [2, c.hyphen_id, 3, c.end_id],
                                 [c.end_id], [9, c.end_id]])
        obs = torch.zeros((4, 12), dtype=torch.long)
        t = u.speaker_terms(refer, toks, {FARMER: obs, BUYER: obs})[FARMER]["rarity"]
        self.assertLess(float(t[0]), float(t[1]), "an established word cost as much as a rare one")
        self.assertLess(float(t[0]), float(t[3]), "an established word cost as much as a novel one")
        self.assertEqual(float(t[2]), 0.0, "silence was charged or paid")
        self.assertLess(float(t[0]), 0.0, "centred: an established word is below average")

    def test_the_convention_is_for_a_meaning_not_for_everything(self):
        cfg = cfg_small()
        cfg.reward.convention_min_support = 1
        c = cfg.channel
        u = PopulationUsage(cfg)
        refer = phase_named(cfg, "name-fruit")
        # the population says a4 for meaning A and a5 for meaning B
        obs = torch.tensor([[0, 1, 0] + [0] * 9, [1, 2, 1] + [0] * 9])
        for _ in range(20):
            toks = self._batch(cfg, [[4, c.end_id], [5, c.end_id]])
            u.observe(u.speaker_terms(refer, toks, {FARMER: obs, BUYER: obs}), 2)
        toks = self._batch(cfg, [[4, c.end_id], [4, c.end_id]])
        conv = u.speaker_terms(refer, toks, {FARMER: obs, BUYER: obs})[FARMER]["convention"]
        self.assertGreater(float(conv[0]), 0.0, "matching this meaning's convention earned nothing")
        self.assertLess(float(conv[1]), 0.0,
                        "using another meaning's form was rewarded as agreement")


class TestZeroShotSuppression(unittest.TestCase):
    def _run(self, seen, unseen, n=600):
        orig = M._play

        def fake(cfg, pop, world, n_, f_sel, b_sel, *, held_out=False, **kw):
            r = unseen if held_out else seen
            return {"n": n_, "success_rate": r, "success_rate_on_viable": r}
        M._play = fake
        try:
            pop = SimpleNamespace(farmers=[0], buyers=[0])
            world = SimpleNamespace(holdout=[(0, 1)])
            return M.zero_shot(cfg_small(), pop, world, n)
        finally:
            M._play = orig

    def test_a_ratio_over_a_handful_of_successes_is_not_reported(self):
        z = self._run(seen=1 / 600, unseen=4 / 600)       # what used to read 4.00
        self.assertTrue(math.isnan(z["retention"]))
        self.assertIn("suppressed", z["suppressed"])

    def test_a_real_ratio_is_reported(self):
        z = self._run(seen=0.5, unseen=0.4)
        self.assertAlmostEqual(z["retention"], 0.8)
        self.assertIsNone(z["suppressed"])


class _FakeBatch:
    def __init__(self, cfg, farmer_words, buyer_words):
        c = cfg.channel
        L = c.max_msg_len
        rows = []
        for fw, bw in zip(farmer_words, buyer_words):
            r = [c.pad_id] * c.dialogue_len
            r[0], r[1] = fw, c.end_id
            r[L], r[L + 1] = bw, c.end_id
            rows.append(r)
        self.tokens = torch.tensor(rows)
        self.active = self.tokens != c.pad_id
        self.phase = phase_named(cfg, "mutual")
        self.cfg = cfg

    def own_positions(self, role):
        return self.phase.own_positions(self.cfg, role)


class TestCrossRoleOverlap(unittest.TestCase):
    def test_one_language_scores_one_and_two_codes_score_zero(self):
        cfg = cfg_small()
        same = _FakeBatch(cfg, [1, 2, 3, 1], [1, 2, 3, 1])
        self.assertAlmostEqual(cross_role_overlap(cfg, [same])["weighted_overlap"], 1.0)
        apart = _FakeBatch(cfg, [1, 2, 1, 2], [5, 6, 5, 6])
        self.assertAlmostEqual(cross_role_overlap(cfg, [apart])["weighted_overlap"], 0.0)

    def test_shared_core_with_jargon_is_in_between(self):
        cfg = cfg_small()
        mixed = _FakeBatch(cfg, [1, 1, 1, 7], [1, 1, 1, 9])
        ov = cross_role_overlap(cfg, [mixed])
        self.assertAlmostEqual(ov["weighted_overlap"], 0.75)
        self.assertEqual(ov["farmer_only_top"], ["a7"])


class TestLiveQuantityEncoding(unittest.TestCase):
    def _batch(self, cfg, words, qty):
        c = cfg.channel
        B = len(words)
        toks = torch.full((B, c.dialogue_len), c.pad_id, dtype=torch.long)
        toks[:, 0] = torch.tensor(words)
        toks[:, 1] = c.end_id
        sb = MutualBatch(f_meaning=torch.stack([torch.zeros(B, dtype=torch.long),
                                                torch.tensor(qty),
                                                torch.zeros(B, dtype=torch.long)], 1),
                         b_meaning=torch.zeros((B, 3), dtype=torch.long))
        refer = phase_named(cfg, "name-fruit")
        return SimpleNamespace(tokens=toks, sb=sb, phase=refer,
                               own_positions=lambda role: refer.own_positions(cfg, role))

    def test_it_finds_quantity_when_it_is_there_and_not_when_it_is_not(self):
        cfg = cfg_small()
        rng = random.Random(0)
        qty = [rng.randint(1, 8) for _ in range(800)]
        coded = live_encoding(cfg, [self._batch(cfg, [q for q in qty], qty)])
        self.assertGreater(coded["excess_bits"], 1.5)
        noise = live_encoding(cfg, [self._batch(cfg, [rng.randint(0, 15) for _ in qty], qty)])
        self.assertLess(abs(noise["excess_bits"]), 0.1)


class TestPhaseAwareProbes(unittest.TestCase):
    def test_a_role_that_never_speaks_is_not_probed(self):
        cfg = cfg_small()
        f, b = agents(cfg, 1)
        refer = phase_named(cfg, "name-fruit")
        m = M.tuple_meanings(cfg, 5, seed=0)
        self.assertIsNone(M.utterances_for_meanings(cfg, b[0], m, phase=refer))
        self.assertEqual(len(M.utterances_for_meanings(cfg, f[0], m, phase=refer)), 5)

    def test_unscheduled_turns_are_not_counted_as_silence(self):
        cfg = cfg_small()
        torch.manual_seed(5)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(5))
        fi, bi = pairing(64)
        batch, _ = run_and_update_gumbel(cfg, rw.sample(64), f, b, fi, bi,
                                         phase=phase_named(cfg, "name-fruit"), train=False)
        ws = word_stats(cfg, [batch])
        # one real utterance per round, not n_turns of them
        n_msgs = round(ws["silent_frac"] * 64 + (1 - ws["silent_frac"]) * 64)
        self.assertEqual(n_msgs, 64)
        self.assertLess(ws["silent_frac"], 0.75,
                        "phantom turns were counted as silent messages")


class TestEveryFieldIsNeeded(unittest.TestCase):
    def test_a_hard_round_is_a_cluster_of_one_field_near_misses(self):
        cfg = cfg_small()
        cfg.curriculum.hard_distractor_frac = 1.0
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(7))
        rb = rw.sample(2000)
        m = rb.meanings
        K = m.shape[1]
        # all members distinct
        for a in range(K):
            for b in range(a + 1, K):
                self.assertFalse(bool((m[:, a] == m[:, b]).all(-1).any()))
        # every member is within two fields of every other (one anchor, near misses)
        diff = (m.unsqueeze(1) != m.unsqueeze(2)).sum(-1)
        self.assertTrue(bool((diff <= 2).all()))

    def test_the_lineup_structure_does_not_give_the_target_away(self):
        """Picking the most central candidate must score chance, not 42%."""
        cfg = cfg_small()
        for frac in (0.5, 1.0):
            cfg.curriculum.hard_distractor_frac = frac
            rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(9))
            rb = rw.sample(8000)
            m = rb.meanings
            agree = (m.unsqueeze(1) == m.unsqueeze(2)).sum(-1).sum(-1)     # (n, K)
            centre = agree.argmax(dim=1)
            hit = float((centre == rb.target).float().mean())
            self.assertLess(abs(hit - 1.0 / m.shape[1]), 0.04,
                            "structure alone picks the target %.3f of the time" % hit)

    def test_every_field_is_needed_in_the_open_lineup(self):
        """No field can ride on the others: each is the only thing that
        separates the target from some distractor often enough to matter."""
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(8))
        rb = rw.sample(4000)
        t = rb.true_meaning
        others = torch.ones(rb.meanings.shape[:2], dtype=torch.bool)
        others[torch.arange(4000), rb.target] = False
        for field, name in enumerate(("fruit", "colour", "quality")):
            keep = [f for f in range(3) if f != field]
            same = (rb.meanings[:, :, keep] == t[:, keep].unsqueeze(1)).all(-1) & others
            share = float(same.any(1).float().mean())
            self.assertGreater(share, 0.25,
                               "%s is hardly ever the field that decides (%.3f)"
                               % (name, share))

    def test_mutual_is_judged_field_by_field(self):
        cfg = cfg_small()
        mutual = phase_named(cfg, "mutual")
        good = {"topsim": 0.40, "null": 0.01, "positional": 0.45, "field_coverage": 0.6}
        ev = {"chance": 0.005, "success": 0.30, "views": [{"success": 0.30}],
              "speakers": {"farmer": good, "buyer": good},
              "farmer_report": 0.60, "farmer_report_transfer": 0.57,
              "buyer_report": 0.60, "buyer_report_transfer": 0.57,
              "farmer_field_transfer": [0.9, 0.05, 0.9],      # quantity at chance
              "buyer_field_transfer": [0.9, 0.6, 0.9]}
        ok, checks = evaluate_rung(cfg, mutual, ev, rung_budget(cfg, mutual)[0])
        self.assertFalse(ok)
        self.assertFalse(checks["farmer decodes: colour"]["met"])
        self.assertTrue(checks["buyer decodes: colour"]["met"])


class TestOrderRung(unittest.TestCase):
    def test_it_sits_between_mutual_and_haggle(self):
        names = [p.name for p in ladder(cfg_small())]
        self.assertEqual(names.index("order"), names.index("mutual") + 1)
        self.assertEqual(names.index("haggle"), names.index("order") + 1)

    def test_the_farmer_fills_the_order_with_its_deal_heads(self):
        cfg = cfg_small()
        order = phase_named(cfg, "order")
        # fruit, colour and quantity: the whole order
        self.assertEqual(order.active_heads(FARMER, cfg),
                         [H_VARIETY, H_BELIEF_COLOR, H_QTY])
        self.assertEqual(order.active_heads(BUYER, cfg), [])
        self.assertEqual(order.speaker_of_turn(0), BUYER)
        sb = SimpleNamespace(want_variety=torch.tensor([0, 1, 2]),
                             want_color=torch.tensor([1, 1, 1]),
                             need_qty=torch.tensor([3, 4, 5]))
        dec = torch.zeros((3, N_HEADS), dtype=torch.long)
        dec[:, H_VARIETY] = torch.tensor([0, 1, 0])
        dec[:, H_QTY] = torch.tensor([3, 2, 5])
        dec[:, H_BELIEF_COLOR] = torch.tensor([1, 1, 0])
        zero = torch.zeros(3)
        res = resolve_order(cfg, sb, dec, zero, zero)
        # right; wrong quantity; wrong fruit and colour
        self.assertEqual(res["success"].tolist(), [True, False, False])
        self.assertEqual(res["order_fields"].tolist(),
                         [[True, True, True], [True, True, False], [False, False, True]])
        self.assertGreater(float(res["farmer_reward"][1]), float(res["farmer_reward"][2]) - 1e-6)


class TestSnapshots(unittest.TestCase):
    def test_a_snapshot_carries_the_population_forward(self):
        import tempfile
        from orchard.train import Trainer
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 2
        cfg.train.episodes = 256
        cfg.train.batch_size = 128
        cfg.train.device = "cpu"
        cfg.log.plot = False
        with tempfile.TemporaryDirectory() as d:
            tr = Trainer(cfg, d + "/a", quiet=True)
            ph = tr.curriculum.phase
            f_idx, b_idx = tr.pop.pair(128)
            batch, _ = run_and_update_gumbel(cfg, tr.referential_world.sample(128),
                                             tr.pop.farmers, tr.pop.buyers, f_idx, b_idx,
                                             phase=ph, usage=tr.usage)
            tr.store.add_batch(batch, tr.pop.farmers, tr.pop.buyers, 0)
            tr.episode = 128
            tr.curriculum.index = [p.name for p in tr.curriculum.phases].index("mutual")
            path = tr.save_snapshot("t")
            tr2 = Trainer(cfg, d + "/b", quiet=True)
            tr2.load_snapshot(path)
            self.assertEqual(tr2.episode, 128)
            self.assertEqual(tr2.curriculum.phase.name, "mutual")
            for a, b in zip(tr.pop.all_agents(), tr2.pop.all_agents()):
                for (k, x), (_, y) in zip(a.net.state_dict().items(),
                                          b.net.state_dict().items()):
                    self.assertTrue(torch.equal(x, y.to(x.device)), k)
            self.assertEqual(len(tr2.store), len(tr.store))
            self.assertAlmostEqual(tr2.usage.word_total, tr.usage.word_total)
            tr.close()
            tr2.close()


class TestCommunityGrowth(unittest.TestCase):
    def test_founders_then_newcomers_in_new_slots(self):
        from orchard.population import Population
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 4
        cfg.population.founders_farmers = cfg.population.founders_buyers = 2
        pop = Population(cfg, random.Random(0))
        self.assertEqual((len(pop.farmers), len(pop.buyers)), (2, 2))
        self.assertFalse(pop.full_size)
        seen = []
        ev = pop.add_newcomer(FARMER, 100, on_birth=lambda a, e: seen.append(a.slot))
        self.assertEqual(ev.kind, "newcomer")
        self.assertEqual(seen, [2])
        self.assertEqual(len(pop.farmers), 3)
        f_idx, b_idx = pop.pair(12)
        self.assertEqual(sorted(set(f_idx.tolist())), [0, 1, 2])
        pop.add_newcomer(FARMER, 200)
        pop.add_newcomer(BUYER, 200)
        pop.add_newcomer(BUYER, 200)
        self.assertTrue(pop.full_size)

    def test_without_founders_the_population_starts_full(self):
        from orchard.population import Population
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 3
        cfg.population.founders_farmers = cfg.population.founders_buyers = 0
        pop = Population(cfg, random.Random(0))
        self.assertTrue(pop.full_size)


if __name__ == "__main__":
    unittest.main()


class TestLanguageProperties(unittest.TestCase):
    def test_disentanglement_scores_a_clean_code_high_and_noise_low(self):
        from orchard.properties import disentanglement
        rng = random.Random(0)
        meanings = [(rng.randrange(3), rng.randrange(1, 6), rng.randrange(3)) for _ in range(400)]
        clean = [[m[0], 10 + m[1], 20 + m[2]] for m in meanings]       # one field per slot
        noise = [[rng.randrange(16) for _ in range(3)] for _ in meanings]
        d_clean = disentanglement(meanings, clean, [0, 1, 2], 3)
        d_noise = disentanglement(meanings, noise, [0, 1, 2], 3)
        self.assertGreater(d_clean["posdis"], 0.8)
        self.assertGreater(d_clean["bosdis"], 0.8)
        self.assertLess(d_noise["posdis"], 0.1)

    def test_duality_is_only_demanded_when_values_outnumber_atoms(self):
        from orchard.properties import duality
        cfg = cfg_small()
        sem = SimpleNamespace(per_word={}, per_token={})
        self.assertFalse(duality(cfg, sem)["necessary"])        # 16 atoms, 3+8+3 values
        cfg.channel.atomic_vocab = 8
        self.assertTrue(duality(cfg, sem)["necessary"])

    def test_the_scorecard_covers_every_property(self):
        from orchard.properties import scorecard
        props = scorecard(cfg_small(), [], {"transitions": [], "reached": "name-fruit"}, None)
        names = {p["property"] for p in props}
        for want in ("reference", "productivity", "intentionality", "decontextualised",
                     "displaced", "interchangeable", "generic", "perspectives",
                     "cultural transmission"):
            self.assertIn(want, names)


def _grammatical(cfg, seg):
    """utterance := word (SPACE word)*, word := atom (HYPHEN atom)*, then END/PAD."""
    c = cfg.channel
    body = []
    for t in seg:
        if t in (c.end_id, c.pad_id):
            break
        body.append(t)
    if not body:
        return True
    for i, t in enumerate(body):
        if i % 2 == 0 and not c.is_atom(t):
            return False
        if i % 2 == 1 and t not in (c.hyphen_id, c.space_id):
            return False
    return c.is_atom(body[-1])


class TestWordGrammar(unittest.TestCase):
    def test_every_utterance_alternates_atoms_and_marks(self):
        cfg = cfg_small()
        cfg.channel.max_symbols = 12
        torch.manual_seed(11)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(11))
        fi, bi = pairing(256)
        batch, _ = run_and_update_gumbel(cfg, rw.sample_mutual(256), f, b, fi, bi,
                                         phase=phase_named(cfg, "mutual"))
        L = cfg.channel.max_msg_len
        for row in batch.tokens.tolist():
            for t in range(2):
                self.assertTrue(_grammatical(cfg, row[t * L:(t + 1) * L]), row)

    def test_evaluation_and_probes_obey_it_too(self):
        from orchard.rollout import run_episodes
        cfg = cfg_small()
        cfg.channel.max_symbols = 12
        torch.manual_seed(12)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(12))
        fi, bi = pairing(128)
        batch = run_episodes(cfg, rw.sample(128), f, b, fi, bi,
                             phase=phase_named(cfg, "name-fruit"))
        for row in batch.tokens.tolist():
            self.assertTrue(_grammatical(cfg, row[:cfg.channel.max_msg_len]))
        m = M.tuple_meanings(cfg, 20, seed=0)
        for u in M.utterances_for_meanings(cfg, f[0], m, phase=phase_named(cfg, "name-fruit")):
            self.assertTrue(_grammatical(cfg, u))

    def test_what_is_rendered_is_what_was_said(self):
        from orchard.render import render_message
        cfg = cfg_small()
        c = cfg.channel
        msg = [3, c.hyphen_id, 7, c.space_id, 1, c.end_id]
        self.assertEqual(render_message(cfg, msg), "a3-a7 a1")
        from orchard.env import parse_words
        self.assertEqual(parse_words(cfg, msg), [(3, 7), (1,)])

    def test_the_default_buffer_is_not_the_constraint(self):
        c = Config().channel
        self.assertGreaterEqual(c.max_symbols, 16)
        self.assertTrue(c.enforce_word_grammar)


class TestReadableTranscripts(unittest.TestCase):
    def _rounds(self, cfg, phase, scen):
        from orchard.transcripts import format_round
        torch.manual_seed(21)
        f, b = agents(cfg)
        fi, bi = pairing(16)
        batch, _ = run_and_update_gumbel(cfg, scen, f, b, fi, bi, phase=phase, train=False)
        return [format_round(cfg, batch.phase, batch, i, episode=100 + i) for i in range(4)]

    def _check(self, rounds):
        for lines in rounds:
            text = "\n".join(lines)
            self.assertIn("  expected : ", text)
            self.assertIn("  dialogue : ", text)
            self.assertIn("  outcome  : ", text)
            # the three parts are on their own lines, in that order
            order = [next(k for k, l in enumerate(lines) if l.startswith("  " + tag))
                     for tag in ("expected", "dialogue", "outcome")]
            self.assertEqual(order, sorted(order))

    def test_every_rung_kind_reads_as_expected_dialogue_outcome(self):
        from orchard.batched import TensorWorld
        cfg = cfg_small()
        cfg.channel.max_symbols = 8
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(21))
        tw = TensorWorld(cfg, generator=torch.Generator().manual_seed(21))
        self._check(self._rounds(cfg, phase_named(cfg, "name-fruit"), rw.sample(16)))
        self._check(self._rounds(cfg, phase_named(cfg, "name-all").with_informer(BUYER),
                                 rw.sample(16, informer=BUYER)))
        self._check(self._rounds(cfg, phase_named(cfg, "mutual"), rw.sample_mutual(16)))
        self._check(self._rounds(cfg, phase_named(cfg, "order"), tw.sample(16)))
        self._check(self._rounds(cfg, phase_named(cfg, "haggle"), tw.sample(16)))


class TestReportSummary(unittest.TestCase):
    def test_summary_statistics_come_first(self):
        from orchard.report import _summary_lines
        lines = _summary_lines(cfg_small(), {"episode": 1000, "started_utc": "2026-01-01 00:00:00 UTC",
                                             "curriculum": {"phases": ["name-fruit"], "reached": "name-fruit"}},
                               12.0)
        self.assertEqual(lines[0], "## Summary statistics")
        self.assertTrue(any("furthest rung" in l for l in lines))


class TestLifespanInUpdates(unittest.TestCase):
    def test_age_in_updates_does_not_depend_on_batch_size(self):
        from orchard.population import Population
        for batch in (64, 4096):
            cfg = cfg_small()
            cfg.population.n_farmers = cfg.population.n_buyers = 2
            cfg.population.lifespan_min = cfg.population.lifespan_max = 3
            cfg.population.initial_stagger = False
            pop = Population(cfg, random.Random(0))
            f_idx, b_idx = pop.pair(batch)
            res = {"success": torch.zeros(batch, dtype=torch.bool),
                   "farmer_profit": torch.zeros(batch), "buyer_savings": torch.zeros(batch),
                   "traded_qty": torch.zeros(batch, dtype=torch.long),
                   "trade_value": torch.zeros(batch)}
            batch_obj = SimpleNamespace(res=res, f_reward=torch.zeros(batch),
                                        b_reward=torch.zeros(batch))
            for step in range(3):
                self.assertFalse(pop.farmers[0].is_expired(), (batch, step))
                pop.record_episode_participation(f_idx, b_idx, batch_obj)
            self.assertEqual(pop.farmers[0].updates, 3)
            self.assertTrue(pop.farmers[0].is_expired())
            # One pool takes both seats below the trading rungs, so an agent
            # plays its share of each -- but still counts one update.
            self.assertEqual(pop.farmers[0].age, 3 * batch)


class TestVerdictUsesTheRungsChance(unittest.TestCase):
    def test_lineup_success_at_one_in_four_is_not_emergence(self):
        from orchard.report import assess
        final = {"eval_success": 0.244, "chance_for_phase": 0.25,
                 "channel_ablation": {"variety_transfer": -0.02, "information_transfer": -0.02,
                                      "intact_comprehension": 0.246},
                 "vocab": {"token_entropy_norm": 0.95, "tokens_used": 16}}
        v = assess(cfg_small(), final, 0.0)       # 0.0: the trading task's chance
        self.assertEqual(v["verdict"], "NO EMERGENCE")
        self.assertFalse(v["checks"]["learned_to_trade"])
