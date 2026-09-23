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
from orchard.curriculum import (H_ACCEPT, H_BELIEF, H_BELIEF_COLOR, H_CHOICE, H_QTY,
                                H_REPORT, H_VARIETY, MutualBatch, N_HEADS,
                                ReferentialBatch, ReferentialWorld, evaluate_rung,
                                hindsight_targets, resolve_referential,
                                ladder, phase_named, resolve_mutual, resolve_request,
                                rung_budget)
from orchard.env import BUYER, FARMER
from orchard.gumbel import run_and_update_gumbel
from orchard.lexicon import cross_role_overlap, live_encoding, word_stats
from orchard.population import Population

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
    """The report is read out of the heads the rung scores, and only those.

    Every caller used to slice the columns itself. They disagreed: the reward
    read `H_BELIEF[:3]`, whose middle head is a belief about *quantity*, and
    scored it against the colour -- so the middle field of every mutual round
    could only be right by luck, and the rung could never have been passed.
    These tests hand `resolve_mutual` whole decision rows, so the selection is
    part of what is tested.
    """

    def _rows(self, reports):
        """(B, N_HEADS) decisions carrying `reports` in the scored heads."""
        dec = torch.zeros((len(reports), N_HEADS), dtype=torch.long)
        for i, r in enumerate(reports):
            for k, h in enumerate(H_REPORT):
                dec[i, h] = int(r[k])
            # the head the reward used to read, filled with something that would
            # pass for a colour if anyone read it again
            dec[i, H_BELIEF[1]] = 8
        return dec

    def test_both_must_report_the_other(self):
        cfg = cfg_small()
        f_m = torch.tensor([[0, 3, 1], [1, 2, 2]])
        b_m = torch.tensor([[2, 1, 0], [0, 3, 1]])
        mb = MutualBatch(f_meaning=f_m, b_meaning=b_m)
        sym = torch.zeros(2)
        res = resolve_mutual(cfg, mb, self._rows(b_m), self._rows(f_m), sym, sym)
        self.assertTrue(bool(res["success"].all()),
                        "a perfect report did not count: the reward read other heads")
        wrong = f_m.clone()
        wrong[0, 0] = (wrong[0, 0] + 1) % cfg.world.n_varieties
        res = resolve_mutual(cfg, mb, self._rows(b_m), self._rows(wrong), sym, sym)
        self.assertEqual(res["success"].tolist(), [False, True])
        self.assertEqual(res["farmer_report_ok"].tolist(), [True, True])
        self.assertEqual(res["buyer_report_ok"].tolist(), [False, True])

    def test_every_field_of_the_report_must_be_right(self):
        """A thing is (fruit, colour, quality) and all three are reported exactly."""
        cfg = cfg_small()
        f_m, b_m = torch.tensor([[0, 1, 0]]), torch.tensor([[1, 2, 1]])
        mb = MutualBatch(f_meaning=f_m, b_meaning=b_m)
        zero = torch.zeros(1)
        right = resolve_mutual(cfg, mb, self._rows(b_m), self._rows(f_m), zero, zero)
        self.assertTrue(bool(right["success"][0]))
        for field in range(3):
            off = b_m.clone()
            off[0, field] = (off[0, field] + 1) % 3
            res = resolve_mutual(cfg, mb, self._rows(off), self._rows(f_m), zero, zero)
            self.assertFalse(bool(res["success"][0]),
                             "a wrong %s still counted as understood"
                             % ("fruit", "colour", "quality")[field])

    def test_the_reward_reads_the_heads_the_rung_scores(self):
        cfg = cfg_small()
        mutual = phase_named(cfg, "mutual")
        self.assertEqual(mutual.active_heads(FARMER, cfg), list(H_REPORT))
        self.assertEqual(mutual.active_heads(BUYER, cfg), list(H_REPORT))
        # a report that is perfect in the scored heads and junk everywhere else
        f_m, b_m = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 1]])
        mb = MutualBatch(f_meaning=f_m, b_meaning=b_m)
        f_dec, b_dec = self._rows(b_m), self._rows(f_m)
        junk = [h for h in range(N_HEADS) if h not in H_REPORT]
        f_dec[:, junk] = 7
        b_dec[:, junk] = 7
        zero = torch.zeros(1)
        res = resolve_mutual(cfg, mb, f_dec, b_dec, zero, zero)
        self.assertTrue(bool(res["success"][0]),
                        "the reward depends on a head the rung never trains")

    def test_hindsight_teaches_the_same_heads(self):
        cfg = cfg_small()
        mutual = phase_named(cfg, "mutual")
        mb = MutualBatch(f_meaning=torch.tensor([[1, 2, 3]]),
                         b_meaning=torch.tensor([[2, 3, 1]]))
        tgt = hindsight_targets(cfg, mutual, mb)
        self.assertEqual(sorted(tgt[FARMER]), sorted(H_REPORT))
        self.assertEqual(sorted(tgt[BUYER]), sorted(H_REPORT))
        # and each head is taught the field it is scored on
        for k, h in enumerate(H_REPORT):
            self.assertEqual(int(tgt[FARMER][h][0]), int(mb.b_meaning[0, k]))


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


class TestTheConventionBonusCannotPayForACollapse(unittest.TestCase):
    """The term exists to make a population agree. It must not also decide
    *what* they agree on, and above all it must not pay for agreeing on less.

    Contrasting against the *average* other meaning's form did exactly that: a
    compositional code's forms resemble each other -- that is what sharing a
    morpheme means -- so it read as undistinctive and was taxed, while a
    collapsed code that named one field and dropped the rest was paid more than
    the code it replaced. A GPU run at `mutual`, where the task signal starts at
    zero and nothing else shapes what is said, collapsed onto exactly that: 7
    words of 1.0 atoms, one word per utterance, coherence 0.92, field coverage
    [0.83, 0.13, 0.05]. The contrast is now against the closest other form.
    """

    def _codes(self, cfg):
        import itertools
        sp = cfg.channel.space_id
        combos = [c for c in itertools.product(range(3), repeat=3)]
        rng = random.Random(0)
        tags = {m: (rng.randrange(cfg.channel.atomic_vocab),
                    rng.randrange(cfg.channel.atomic_vocab)) for m in combos}
        return combos, {
            "compositional": lambda m: (m[0], sp, 3 + m[1], sp, 6 + m[2]),
            "arbitrary": lambda m: tags[m],
            "collapse: one field, one atom": lambda m: (m[0],),
            "collapse: one form for everything": lambda m: (0,),
        }

    def _earned(self, cfg, combos, form_of, n_contrast):
        """The real contrast, over the modal forms of a sample of other meanings."""
        from orchard.conventions import similarity
        modal = {m: form_of(m) for m in combos}
        rng = random.Random(3)
        out = []
        for _ in range(20):
            for m in combos:
                u = modal[m]
                pool = [o for o in combos if o != m]
                others = [similarity(u, modal[o])
                          for o in rng.sample(pool, min(n_contrast, len(pool)))]
                base = max(others) if others else 0.0
                out.append(cfg.reward.convention * (similarity(u, modal[m]) - base))
        return sum(out) / len(out)

    def test_a_collapsed_code_earns_nothing(self):
        cfg = cfg_small()
        combos, codes = self._codes(cfg)
        n = cfg.reward.convention_contrast_samples
        earned = {k: self._earned(cfg, combos, f, n) for k, f in codes.items()}
        for name, v in earned.items():
            if name.startswith("collapse"):
                self.assertAlmostEqual(
                    v, 0.0, places=3,
                    msg="%s earns %+.4f -- the bonus pays to drop a field" % (name, v))
        self.assertGreater(
            earned["compositional"], 0.01,
            "a compositional code earns nothing either, so the term says nothing")
        for name, v in earned.items():
            if name.startswith("collapse"):
                self.assertGreater(
                    earned["compositional"], v + 0.01,
                    "%s is paid as well as a compositional code" % name)

    def test_the_sample_is_big_enough_to_find_a_near_neighbour(self):
        """The contrast takes the closest of a *sample*, so too small a sample
        misses the near neighbour that makes a collapsed code worth nothing."""
        cfg = cfg_small()
        combos, codes = self._codes(cfg)
        collapse = codes["collapse: one field, one atom"]
        configured = self._earned(cfg, combos, collapse,
                                  cfg.reward.convention_contrast_samples)
        self.assertLess(configured, 0.01,
                        "at %d contrast samples a collapsed code still earns %+.4f"
                        % (cfg.reward.convention_contrast_samples, configured))
        self.assertGreater(self._earned(cfg, combos, collapse, 1), configured,
                           "the sample size does not affect the contrast at all, "
                           "which means it is not taking the closest")

    def test_the_live_term_agrees_with_all_that(self):
        """Through `speaker_terms`, not a reimplementation of it."""
        cfg = cfg_small()
        cfg.reward.convention_min_support = 1
        c = cfg.channel
        phase = phase_named(cfg, "name-all")
        obs = torch.tensor([[i % 3, (i // 3) % 3, 0, 3] + [0] * 8 for i in range(9)])

        def run(form_of):
            u = PopulationUsage(cfg)
            toks = torch.full((9, c.dialogue_len), c.pad_id, dtype=torch.long)
            for i in range(9):
                f = list(form_of(i)) + [c.end_id]
                toks[i, :len(f)] = torch.tensor(f)
            for _ in range(8):
                u.observe(u.speaker_terms(phase, toks, {FARMER: obs, BUYER: obs}), 9)
            t = u.speaker_terms(phase, toks, {FARMER: obs, BUYER: obs})[FARMER]
            return float(t["convention"].mean())

        varied = run(lambda i: (i, c.hyphen_id, 4 + (i % 3)))   # a form per meaning
        same = run(lambda i: (4,))                              # one form for all
        self.assertAlmostEqual(same, 0.0, places=3,
                               msg="one form for every meaning earned %+.4f" % same)
        self.assertGreater(varied, same,
                           "saying something different per meaning earned no more "
                           "than saying one thing for all of them")


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


class TestTheRequestRungs(unittest.TestCase):
    """The trade half of the ladder adds one field at a time, like the naming half."""

    def test_each_request_rung_adds_one_field_and_keeps_the_rest(self):
        cfg = cfg_small()
        names = [p.name for p in ladder(cfg)]
        self.assertEqual(names[names.index("mutual"):],
                         ["mutual", "ask-qty", "order", "quote", "offer",
                          "judge", "haggle", "bargain", "market"])
        asked = {n: phase_named(cfg, n).ask for n in ("ask-qty", "order", "quote")}
        self.assertEqual(asked["ask-qty"], ("quantity",))
        self.assertEqual(asked["order"], ("fruit", "colour", "quantity"))
        self.assertEqual(asked["quote"], ("fruit", "colour", "quantity", "price"))
        for a, b in (("ask-qty", "order"), ("order", "quote")):
            self.assertTrue(set(asked[a]) < set(asked[b]),
                            "%s drops a field %s had" % (b, a))
        # and each one is judged on the field it introduced
        for n, field in (("ask-qty", "quantity"), ("order", "fruit"),
                         ("quote", "price"), ("offer", "stock"), ("judge", "deal")):
            self.assertEqual(phase_named(cfg, n).asks_first, field)

    def test_quantity_and_price_are_asked_for_before_they_are_negotiated(self):
        """The two fields no naming rung teaches get a rung of their own first."""
        cfg = cfg_small()
        names = [p.name for p in ladder(cfg)]
        for field, rung in (("quantity", "ask-qty"), ("price", "quote")):
            first = next(p.name for p in ladder(cfg)
                         if p.order and field in p.ask)
            self.assertEqual(first, rung)
            self.assertLess(names.index(rung), names.index("haggle"),
                            "%s is first asked for after haggle needs it" % field)

    def test_the_farmer_fills_the_order_with_its_deal_heads(self):
        cfg = cfg_small()
        order = phase_named(cfg, "order")
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
        res = resolve_request(cfg, order, sb, {FARMER: dec, BUYER: dec * 0}, zero, zero)
        # right; wrong quantity; wrong fruit and colour
        self.assertEqual(res["success"].tolist(), [True, False, False])
        self.assertEqual(res["order_fields"].tolist(),
                         [[True, True, True], [True, True, False], [False, False, True]])
        self.assertGreater(float(res["farmer_reward"][1]), float(res["farmer_reward"][2]) - 1e-6)

    def test_always_accepting_cannot_pass_the_judge_rung(self):
        """The failure `haggle` actually had: accept everything, score the base rate."""
        cfg = Config()
        judge = phase_named(cfg, "judge")
        self.assertEqual(judge.active_heads(BUYER, cfg), [H_ACCEPT])
        self.assertEqual(judge.active_heads(FARMER, cfg), [])
        base = 0.68                       # roughly the share of viable rounds

        def ev(intact):
            return {"request_first": intact, "muted_request_first": base,
                    "request_fields_intact": [intact], "request_fields_muted": [base],
                    "request_field_transfer": [(intact - base) / (1 - base)],
                    "success": intact, "chance": base,
                    "transfer": (intact - base) / (1 - base),
                    "topsim": 1.0, "null": 0.0}

        ok, checks = evaluate_rung(cfg, judge, ev(base), 10 ** 6)
        self.assertFalse(ok, "a pair that accepts everything passed `judge`")
        self.assertFalse(checks["deal arrives"]["met"])
        # and a pair that actually reads the answer does pass
        ok, checks = evaluate_rung(cfg, judge, ev(0.90), 10 ** 6)
        self.assertTrue(ok, [n for n, d in checks.items() if not d["met"]])

    def test_a_binary_decision_is_judged_on_its_gain_over_silence(self):
        """2x the muted rate is not a reachable bar when silence already scores 0.68."""
        from orchard.curriculum import _headroom_floor
        cfg = Config()
        self.assertGreater(_headroom_floor(0.68, cfg.curriculum.min_field_transfer), 0.68)
        self.assertLess(_headroom_floor(0.68, cfg.curriculum.min_field_transfer), 1.0)
        # a field silence rarely gets right keeps the ordinary absolute floor
        self.assertLess(_headroom_floor(0.15, cfg.curriculum.min_field_transfer),
                        cfg.curriculum.order_min_success)

    def test_the_answer_rung_runs_the_other_way(self):
        """In `offer` the farmer says what it holds and the buyer has to report it."""
        cfg = cfg_small()
        offer = phase_named(cfg, "offer")
        self.assertTrue(offer.answers)
        self.assertEqual(offer.active_heads(BUYER, cfg),
                         [H_BELIEF[1], H_BELIEF[2], H_BELIEF[3]])
        self.assertEqual(offer.active_heads(FARMER, cfg), [])
        # the buyer asks first, so the farmer knows which lot to describe
        self.assertEqual(offer.speaker_of_turn(0), BUYER)
        self.assertEqual(offer.speaker_of_turn(1), FARMER)
        sb = SimpleNamespace(offered_stock=torch.tensor([2, 5, 0]),
                             offered_quality=torch.tensor([1, 2, 3]),
                             reservation=torch.tensor([0, 1, 2]))
        dec = torch.zeros((3, N_HEADS), dtype=torch.long)
        dec[:, H_BELIEF[1]] = torch.tensor([2, 5, 1])
        dec[:, H_BELIEF[2]] = torch.tensor([1, 0, 3])
        dec[:, H_BELIEF[3]] = torch.tensor([0, 1, 2])
        zero = torch.zeros(3)
        res = resolve_request(cfg, offer, sb, {BUYER: dec, FARMER: dec * 0}, zero, zero)
        self.assertEqual(res["success"].tolist(), [True, False, False])
        # the buyer is the one being scored here, so it is the one credited
        self.assertGreater(float(res["buyer_decode"][0]), 0.0)
        self.assertEqual(float(res["farmer_decode"][0]), 0.0)


# ==========================================================================
class TestEveryRungIsReachable(unittest.TestCase):
    """The ladder is only a ladder if every rung can be played and passed.

    The trading rungs went a long time without either check -- no run had ever
    got that far -- so a rung could demand evidence nothing produced, or crash
    on its own scenario shape, and nothing would say so until a GPU run had
    spent a day getting there.
    """

    def _perfect(self, cfg, phase):
        """Evidence from an imaginary rung that worked perfectly."""
        roles = ["farmer", "buyer"]
        spk = {r: {"topsim": 1.0, "null": 0.0, "positional": 1.0,
                   "field_coverage": 1.0} for r in roles}
        ev = {
            "phase": phase.name, "speakers": spk,
            "success": 1.0, "chance": 0.0, "transfer": 1.0,
            "topsim": 1.0, "null": 0.0,
            "holdout_success": 1.0, "seen_success": 1.0, "holdout_ratio": 1.0,
            "views": [{"success": 1.0, "transfer": 1.0, "muted_success": 0.0,
                       "guesser": r, "informer": r} for r in roles],
            "by_kind": {k: {"success": 1.0} for k in range(4)},
            "request_first": 1.0, "muted_request_first": 0.0,
            "request_first_transfer": 1.0,
            "request_fields_intact": [1.0] * 4, "request_fields_muted": [0.0] * 4,
            "request_field_transfer": [1.0] * 4,
        }
        for r in roles:
            ev[r + "_report"] = 1.0
            ev["muted_" + r + "_report"] = 0.0
            ev[r + "_report_transfer"] = 1.0
            ev[r + "_field_transfer"] = [1.0, 1.0, 1.0]
        return ev

    def test_every_rung_passes_on_perfect_evidence(self):
        cfg = Config()
        for phase in ladder(cfg):
            ok, checks = evaluate_rung(cfg, phase, self._perfect(cfg, phase), 10 ** 6)
            unmet = [n for n, d in checks.items() if not d["met"]]
            self.assertTrue(ok, "%s can never be promoted; unmet: %s"
                            % (phase.name, unmet))
            self.assertTrue(checks, "%s is promoted without checking anything"
                            % phase.name)

    def test_no_rung_passes_on_evidence_of_nothing(self):
        cfg = Config()
        nan = float("nan")
        blank = {"speakers": {"farmer": {}, "buyer": {}}, "views": []}
        for phase in ladder(cfg):
            ok, _ = evaluate_rung(cfg, phase, dict(blank), 10 ** 6)
            self.assertFalse(ok, "%s promotes on missing evidence" % phase.name)
            ok, _ = evaluate_rung(cfg, phase, dict(blank, success=nan), 0)
            self.assertFalse(ok, "%s promotes immediately" % phase.name)

    def test_every_rung_plays_a_batch_and_learns_from_it(self):
        """One real training step per rung, on that rung's own scenario shape."""
        from orchard.batched import TensorWorld
        from orchard.curriculum import ReferentialWorld
        from orchard.gumbel import run_and_update_gumbel
        cfg = cfg_small()
        cfg.train.batch_size = 8
        torch.manual_seed(0)
        farmers, buyers = agents(cfg, 2)
        gen = torch.Generator().manual_seed(0)
        rw = ReferentialWorld(cfg, generator=gen)
        tw = TensorWorld(cfg, device="cpu", generator=gen)
        n = 8
        idx = torch.arange(n) % 2
        for phase in ladder(cfg):
            if phase.referential:
                scen = rw.sample(n, informer=phase.informer, mix=phase.mix)
            elif phase.mutual:
                scen = rw.sample_mutual(n)
            else:
                scen = tw.sample(n)
            batch, stats = run_and_update_gumbel(
                cfg, scen, farmers, buyers, idx, idx, update=0, phase=phase)
            self.assertEqual(len(batch), n, "%s played nothing" % phase.name)
            for who in ("farmer_reward", "buyer_reward"):
                r = batch.res[who] if isinstance(batch.res, dict) else None
                if r is not None:
                    self.assertTrue(bool(torch.isfinite(r).all()),
                                    "%s produced a non-finite %s" % (phase.name, who))
            self.assertTrue(all(v == v for v in vars(stats).values()
                                if isinstance(v, float)),
                            "%s produced a NaN statistic" % phase.name)


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


class TestResumeKeepsOnePool(unittest.TestCase):
    """Below `split_roles_at` the two seats are one list, and that is an
    identity `Population.pair` relies on: it seats index i opposite a
    *different* index and calls that "never against itself". Restoring the two
    saved lists separately made two copies of every founder -- same agent_id,
    same weights, then their own gradients -- and one rung later the first
    newcomer appended to `farmers` alone, so `pair` handed out a buyer index
    the buyer list did not have (IndexError, in the rollout, 4,500 updates in).
    """

    def _cfg(self):
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 4
        cfg.population.founders_farmers = cfg.population.founders_buyers = 2
        cfg.train.batch_size = 32
        cfg.train.device = "cpu"
        cfg.log.plot = False
        return cfg

    def _round_trip(self, cfg, rung):
        import tempfile
        from orchard.train import Trainer
        d = tempfile.mkdtemp()
        tr = Trainer(cfg, d + "/a", quiet=True)
        tr.curriculum.index = [p.name for p in tr.curriculum.phases].index(rung)
        tr.maybe_split_roles(tr.curriculum.phase, log=lambda *_: None)
        path = tr.save_snapshot("t")
        tr2 = Trainer(cfg, d + "/b", quiet=True)
        tr2.load_snapshot(path)
        tr.close()
        return tr, tr2

    def test_a_pooled_rung_comes_back_as_one_list(self):
        cfg = self._cfg()
        _, tr2 = self._round_trip(cfg, "mutual")     # below `haggle`
        self.assertTrue(tr2.pop.shared, "the resumed pool stopped being shared")
        self.assertIs(tr2.pop.farmers, tr2.pop.buyers,
                      "the two seats came back as two lists of copies")
        tr2.close()

    def test_a_newcomer_after_resuming_grows_both_seats(self):
        """The crash, in miniature."""
        cfg = self._cfg()
        _, tr2 = self._round_trip(cfg, "mutual")
        tr2.pop.add_newcomer(FARMER, 10)
        self.assertEqual((len(tr2.pop.farmers), len(tr2.pop.buyers)), (3, 3))
        f_idx, b_idx = tr2.pop.pair(24)
        self.assertLess(int(f_idx.max()), len(tr2.pop.farmers))
        self.assertLess(int(b_idx.max()), len(tr2.pop.buyers))
        self.assertTrue(all(int(a) != int(b) for a, b in zip(f_idx, b_idx)),
                        "an agent was seated opposite itself")
        tr2.close()

    def test_a_split_rung_comes_back_as_two(self):
        cfg = self._cfg()
        tr, tr2 = self._round_trip(cfg, "haggle")    # at the split
        self.assertFalse(tr.pop.shared, "the split never fired")
        self.assertFalse(tr2.pop.shared)
        self.assertIsNot(tr2.pop.farmers, tr2.pop.buyers)
        self.assertEqual([a.agent_id for a in tr2.pop.farmers],
                         [a.agent_id for a in tr.pop.farmers])
        self.assertEqual([a.agent_id for a in tr2.pop.buyers],
                         [a.agent_id for a in tr.pop.buyers])
        tr2.close()

    def test_pairing_says_so_rather_than_indexing_off_the_end(self):
        from orchard.population import Population
        cfg = self._cfg()
        pop = Population(cfg, random.Random(0))
        self.assertTrue(pop.shared)
        pop.buyers = list(pop.farmers)[:1]          # what the resume used to do
        with self.assertRaises(AssertionError) as caught:
            pop.pair(8)
        self.assertIn("shared", str(caught.exception))

    def test_a_snapshot_from_an_affected_run_is_named_as_such(self):
        """A run that resumed under the bug wrote two lists of drifted copies
        with matching ids. The loader keeps one and says what it dropped,
        because the other's training is being discarded."""
        import tempfile
        from orchard.train import Trainer
        cfg = self._cfg()
        d = tempfile.mkdtemp()
        tr = Trainer(cfg, d + "/a", quiet=True)
        tr.curriculum.index = [p.name for p in tr.curriculum.phases].index("mutual")
        path = tr.save_snapshot("t")
        healthy = torch.load(path, map_location="cpu", weights_only=False)
        self.assertFalse(Trainer._pool_had_split(healthy),
                         "a shared pool was read as duplicated")
        # what the bug produced: same ids, weights drifted apart
        split = dict(healthy)
        split["buyers"] = [dict(r) for r in healthy["buyers"]]
        first = split["buyers"][0]
        first["net"] = {k: v.clone() for k, v in first["net"].items()}
        k0 = next(iter(first["net"]))
        first["net"][k0] = first["net"][k0] + 1.0
        self.assertTrue(Trainer._pool_had_split(split),
                        "drifted copies were read as one pool")
        tr.close()

    def test_the_snapshot_listing_says_which_are_safe(self):
        """`--snapshots` is how you choose one to resume from, so it has to tell
        a healthy pool from two sets of copies."""
        import io, tempfile
        from contextlib import redirect_stdout
        from orchard.run import list_snapshots
        from orchard.train import Trainer
        cfg = self._cfg()
        d = tempfile.mkdtemp()
        tr = Trainer(cfg, d + "/run", quiet=True)
        tr.curriculum.index = [p.name for p in tr.curriculum.phases].index("mutual")
        path = tr.save_snapshot("latest")
        tr.close()
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(list_snapshots(d), 0)
        out = buf.getvalue()
        self.assertIn("mutual", out)
        self.assertIn("one pool", out)
        self.assertNotIn("drifted", out)
        # a single file, and a directory with nothing in it
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(list_snapshots(path), 0)
        self.assertIn("one pool", buf.getvalue())
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(list_snapshots(tempfile.mkdtemp()), 1)
        self.assertIn("no snapshots", buf.getvalue())

    def test_throughput_is_measured_from_where_this_process_started(self):
        """A resumed run inherits an episode count but not a wall clock.

        Dividing the whole count by this process's elapsed time reported a rate
        it had never reached: the first heartbeat after resuming an 18.4M-episode
        run read 137,215 eps/s against a real 700, and `progress.json` -- which
        a progress bar reads -- kept a version of that error all run.
        """
        import json, os, tempfile, time
        from orchard.train import Trainer
        cfg = self._cfg()
        d = tempfile.mkdtemp()
        tr = Trainer(cfg, d + "/a", quiet=True)
        tr.episode = 18_436_096
        path = tr.save_snapshot("t")
        tr.close()
        tr2 = Trainer(cfg, d + "/b", quiet=True)
        tr2.load_snapshot(path)
        self.assertEqual(tr2._beat[1], tr2.episode,
                         "the heartbeat window still starts at episode 0")
        self.assertEqual(tr2._episode_at_start, tr2.episode)
        time.sleep(0.05)
        played = 4096
        tr2.episode += played
        tr2.write_progress()
        with open(os.path.join(d, "b", "progress.json")) as fh:
            rate = json.load(fh)["episodes_per_second"]
        self.assertLess(rate, 10 * played,
                        "throughput still counts the episodes it inherited: %.0f" % rate)
        tr2.close()

    def test_a_fresh_run_measures_everything_it_played(self):
        import tempfile
        from orchard.train import Trainer
        cfg = self._cfg()
        tr = Trainer(cfg, tempfile.mkdtemp(), quiet=True)
        self.assertEqual(tr._episode_at_start, 0)
        self.assertEqual(tr._beat[1], 0)
        tr.close()

    def test_pooled_at_follows_the_split(self):
        from orchard.curriculum import pooled_at
        cfg = self._cfg()
        at = cfg.curriculum.split_roles_at
        for phase in ladder(cfg):
            want = phase.index < phase_named(cfg, at).index
            self.assertEqual(pooled_at(cfg, phase), want, phase.name)
        cfg.curriculum.split_roles_at = ""
        self.assertFalse(pooled_at(cfg, phase_named(cfg, "name-all")))


class TestTheStoreStaysOnTheHost(unittest.TestCase):
    """A newborn's apprenticeship stacks what it sampled and moves *that* to the
    device, so the store is host-side by construction -- `add_batch` copies each
    batch off the device in one go. Resuming mapped the whole snapshot onto the
    training device, store included, so the buffer then held device tensors from
    before the resume and host tensors from after. `train_newborn` groups its
    sample by rung, so nothing failed until one rung held both: the first birth
    after a resume taken mid-rung died with "Expected all tensors to be on the
    same device"."""

    def _cfg(self):
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 4
        cfg.population.founders_farmers = cfg.population.founders_buyers = 2
        cfg.train.batch_size = 32
        cfg.train.device = "cpu"
        cfg.log.plot = False
        return cfg

    def test_a_snapshot_is_read_on_the_host_whatever_the_run_trains_on(self):
        """The line that was wrong, asserted so that a CPU box can still see it.

        It read the file onto `self.device`, which on a CPU box *is* the host --
        so the bug was invisible here and only ever appeared on the GPU. The
        trainer is therefore given a device it is not on, and the file still has
        to come to the host; each consumer places what it needs from there
        (`load_state_dict` copies across devices for the nets and their
        optimisers alike).
        """
        import tempfile
        import orchard.agents as A
        import orchard.train as T
        cfg = self._cfg()
        d = tempfile.mkdtemp()
        tr = T.Trainer(cfg, d + "/a", quiet=True)
        path = tr.save_snapshot("t")
        tr.close()
        seen = {}
        real_load, real_make = T.torch.load, A.make_agent

        def spy(p, *a, **kw):
            seen["map_location"] = kw.get("map_location", "<positional>")
            return real_load(p, *a, **kw)

        def on_host(*a, **kw):                 # the agents stay where this box can hold them
            kw["device"] = "cpu"
            return real_make(*a, **kw)

        T.torch.load, A.make_agent = spy, on_host
        try:
            tr2 = T.Trainer(cfg, d + "/b", quiet=True)
            tr2.device = "cuda:7"              # a device this box does not have
            tr2.load_snapshot(path)
            tr2.close()
        finally:
            T.torch.load, A.make_agent = real_load, real_make
        self.assertEqual(seen.get("map_location"), "cpu",
                         "the snapshot was read onto the training device (%s), which "
                         "puts the host-side transcript store on the card"
                         % seen.get("map_location"))

    def test_the_store_comes_back_on_the_host(self):
        import tempfile
        from orchard.train import Trainer
        cfg = self._cfg()
        d = tempfile.mkdtemp()
        tr = Trainer(cfg, d + "/a", quiet=True)
        ph = tr.curriculum.phase
        f_idx, b_idx = tr.pop.pair(32)
        batch, _ = run_and_update_gumbel(cfg, tr.referential_world.sample(32),
                                         tr.pop.farmers, tr.pop.buyers,
                                         f_idx, b_idx, phase=ph, usage=tr.usage)
        tr.store.add_batch(batch, tr.pop.farmers, tr.pop.buyers, 0)
        self.assertGreater(len(tr.store), 0, "nothing was stored to check")
        path = tr.save_snapshot("t")
        tr.close()
        tr2 = Trainer(cfg, d + "/b", quiet=True)
        tr2.load_snapshot(path)
        for s in tr2.store._buf:
            for name in s.TENSOR_FIELDS:
                t = getattr(s, name)
                self.assertEqual(t.device.type, "cpu",
                                 "%s came back on %s" % (name, t.device))
        self.assertEqual(tr2.store.to_host(), 0, "the store was not already host-side")
        tr2.close()

    def test_a_store_holding_two_rungs_can_still_teach_a_newborn(self):
        """The grouping that hid the bug: `train_newborn` batches per rung, so a
        store spanning rungs has to work for every group it makes."""
        import tempfile
        from orchard.train import Trainer
        cfg = self._cfg()
        cfg.bottleneck.only_successful = False   # untrained agents succeed at nothing
        tr = Trainer(cfg, tempfile.mkdtemp(), quiet=True)
        names = [p.name for p in tr.curriculum.phases]
        for rung in ("name-all", "mutual"):
            tr.curriculum.index = names.index(rung)
            ph = tr.curriculum.phase
            scen = (tr.referential_world.sample_mutual(32) if ph.mutual
                    else tr.referential_world.sample(32))
            f_idx, b_idx = tr.pop.pair(32)
            batch, _ = run_and_update_gumbel(cfg, scen, tr.pop.farmers, tr.pop.buyers,
                                            f_idx, b_idx, phase=ph, usage=tr.usage)
            tr.store.add_batch(batch, tr.pop.farmers, tr.pop.buyers, 0)
        rungs = {s.phase.name for s in tr.store._buf if s.phase is not None}
        self.assertGreaterEqual(len(rungs), 2, "only one rung reached the store: %s" % rungs)
        ev = tr.pop.add_newcomer(FARMER, 1, on_birth=tr.on_birth)   # the crashing call
        self.assertEqual(ev.kind, "newcomer")
        tr.close()


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


class TestNoEmptyTurns(unittest.TestCase):
    """Silence is the muted control; no speaker may say it.

    Measured on the GPU: with no length cost, speakers went silent in 24-55% of
    lineup rounds, because silence is the shortest, most reliable message there
    is. It is also exactly what the muted control feeds the listener, so a code
    that used it as a word was invisible to every channel measurement.
    """

    def _scheduled_turns(self, cfg, phase, tokens, active):
        """Every (row, first position) of a turn the rung actually scheduled."""
        L = cfg.channel.max_msg_len
        for turn in range(min(phase.n_turns, cfg.channel.n_turns)):
            p = turn * L
            for i in range(tokens.shape[0]):
                if bool(active[i, p]):
                    yield i, int(tokens[i, p])

    def test_no_training_batch_on_any_rung_starts_a_turn_with_silence(self):
        from orchard.batched import TensorWorld
        cfg = cfg_small()
        torch.manual_seed(21)
        f, b = agents(cfg)
        gen = torch.Generator().manual_seed(21)
        rw = ReferentialWorld(cfg, generator=gen)
        tw = TensorWorld(cfg, device="cpu", generator=gen)
        n = 96
        fi, bi = pairing(n)
        for phase in ladder(cfg):
            if phase.referential:
                scen = rw.sample(n, informer=phase.informer, mix=phase.mix)
            elif phase.mutual:
                scen = rw.sample_mutual(n)
            else:
                scen = tw.sample(n)
            batch, _ = run_and_update_gumbel(cfg, scen, f, b, fi, bi, phase=phase)
            for i, first in self._scheduled_turns(cfg, phase, batch.tokens, batch.active):
                self.assertTrue(cfg.channel.is_atom(first),
                                "%s: a turn opened with %d, not a word" % (phase.name, first))

    def test_evaluation_play_is_held_to_it_too(self):
        from orchard.rollout import run_episodes
        cfg = cfg_small()
        torch.manual_seed(22)
        f, b = agents(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(22))
        fi, bi = pairing(128)
        phase = phase_named(cfg, "name-fruit")
        batch = run_episodes(cfg, rw.sample(128), f, b, fi, bi, phase=phase)
        for i, first in self._scheduled_turns(cfg, phase, batch.tokens, batch.active):
            self.assertTrue(cfg.channel.is_atom(first))

    def test_silence_is_only_possible_when_the_config_asks_for_it(self):
        from orchard.env import grammar_allowed
        cfg = Config()
        prev = torch.zeros(4, dtype=torch.long)
        self.assertFalse(bool(grammar_allowed(cfg, prev, 0)[:, cfg.channel.end_id].any()))
        cfg.channel.allow_silence = True
        self.assertTrue(bool(grammar_allowed(cfg, prev, 0)[:, cfg.channel.end_id].all()))
        # and an utterance can still stop after its first word either way
        cfg.channel.allow_silence = False
        after = grammar_allowed(cfg, torch.tensor([3, 3, 3, 3]), 1)
        self.assertTrue(bool(after[:, cfg.channel.end_id].all()))

    def test_a_silent_lesson_in_the_store_is_skipped_not_taught(self):
        """A resumed run carries silent turns from before the rule; they must not
        reach a newborn as targets on a masked logit (cross-entropy ~1e9)."""
        cfg = cfg_small()
        cfg.bottleneck.epochs = 1
        cfg.bottleneck.only_successful = False
        torch.manual_seed(23)
        f, b = agents(cfg)
        phase = phase_named(cfg, "name-fruit")
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(23))
        fi, bi = pairing(64)
        batch, _ = run_and_update_gumbel(cfg, rw.sample(64), f, b, fi, bi,
                                         phase=phase, update=500)
        # every describer turn made silent, the way the store of an older run holds it
        L = cfg.channel.max_msg_len
        for p in phase.own_positions(cfg, phase.informer):
            k = p % L
            batch.tokens[:, p] = cfg.channel.end_id if k == 0 else cfg.channel.pad_id
            batch.active[:, p] = (k == 0)
        store = TranscriptStore(cfg)
        store.add_batch(batch, f, b, 0)
        newborn = agents(cfg, 1)[0][0]
        info = train_newborn(cfg, newborn, store, random.Random(0))
        for name, v in info.items():
            if isinstance(v, float) and "loss" in name:
                self.assertTrue(math.isfinite(v) and v < 1e3,
                                "%s = %r: a silent lesson was taught" % (name, v))
        for prm in newborn.net.parameters():
            self.assertTrue(bool(torch.isfinite(prm).all()))


class TestTheLineupPaysForGettingClose(unittest.TestCase):
    """A guess is paid for how much of the thing it got.

    Every other rung pays `decode` per field; the lineup paid nothing at all for
    a near miss, so `name-all` -- the rung that needs three fields in one
    utterance -- had no staircase between "one field" and "all of them". The run
    that stalled there named each field on its own at 0.74 / 0.87 / 0.97 and all
    three at once at 0.60, with utterances 1.5 words long.
    """

    def _round(self, cfg, pick):
        m = torch.tensor([[[1, 2, 3], [1, 2, 0], [0, 0, 0]]])   # target, near miss, wild
        rb = ReferentialBatch(meanings=m, target=torch.tensor([0]),
                              query=torch.tensor([3]),
                              held_out=torch.tensor([False]), day=0, informer=FARMER)
        zero = torch.zeros(1)
        return resolve_referential(cfg, rb, torch.tensor([pick]), zero, zero)

    def test_a_near_miss_beats_a_wild_miss(self):
        cfg = cfg_small()
        right = float(self._round(cfg, 0)["farmer_reward"][0])
        near = float(self._round(cfg, 1)["farmer_reward"][0])
        wild = float(self._round(cfg, 2)["farmer_reward"][0])
        self.assertGreater(right, near)
        self.assertGreater(near, wild, "two fields of three paid the same as none")
        self.assertTrue(self._round(cfg, 0)["success"][0])
        self.assertFalse(self._round(cfg, 1)["success"][0],
                         "partial credit must not count as success")

    def test_it_can_be_turned_off(self):
        cfg = cfg_small()
        cfg.reward.refer_partial = 0.0
        self.assertAlmostEqual(float(self._round(cfg, 1)["farmer_reward"][0]),
                               float(self._round(cfg, 2)["farmer_reward"][0]), places=6)


class TestNobodyDiesWhileTheCodeIsBeingInvented(unittest.TestCase):
    """Turnover costs half a two-agent pool, and there is nothing to transmit yet."""

    def test_deaths_wait_for_the_rung_newcomers_arrive_in(self):
        from orchard.curriculum import growth_applies, turnover_applies
        cfg = Config()
        for p in ladder(cfg):
            if p.naming and p.referential:
                self.assertFalse(turnover_applies(cfg, p),
                                 "%s kills a founder mid-invention" % p.name)
            # deaths and newcomers start together: a death is only survivable
            # once there is a community to absorb it
            self.assertEqual(turnover_applies(cfg, p), growth_applies(cfg, p),
                             "%s: turnover and growth disagree" % p.name)

    def test_the_switch_still_turns_it_all_off(self):
        from orchard.curriculum import turnover_applies
        cfg = Config()
        cfg.population.turnover = False
        self.assertFalse(any(turnover_applies(cfg, p) for p in ladder(cfg)))

    def test_a_cohort_that_outlived_its_span_does_not_die_at_once(self):
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 4
        cfg.population.founders_farmers = cfg.population.founders_buyers = 4
        torch.manual_seed(0)
        pop = Population(cfg, random.Random(0))
        for a in pop.farmers:
            a.updates = 10 ** 5                     # long past any lifespan
        self.assertTrue(all(a.is_expired() for a in pop.farmers))
        pop.restagger()
        self.assertFalse(any(a.is_expired() for a in pop.farmers),
                         "every agent would have died in the same update")
        deaths = sorted(a.lifespan - a.updates for a in pop.farmers)
        self.assertGreater(deaths[-1] - deaths[0], 0, "the deaths are not staggered")


class TestExplorationIsRestoredEachRung(unittest.TestCase):
    def test_the_anneals_count_updates_in_the_rung(self):
        from orchard.gumbel import gumbel_tau
        cfg = cfg_small()
        self.assertTrue(cfg.train.anneal_per_rung)
        # a rung that starts 2,000 updates into the run still starts warm
        self.assertGreater(gumbel_tau(cfg, 0), gumbel_tau(cfg, cfg.train.tau_anneal_updates))
        self.assertAlmostEqual(gumbel_tau(cfg, 0), cfg.train.gumbel_tau, places=6)

    def test_the_trainer_hands_the_rungs_own_clock_to_the_anneal(self):
        import inspect
        from orchard.gumbel import run_and_update_gumbel
        from orchard import train as T
        self.assertIn("phase_update", inspect.signature(run_and_update_gumbel).parameters)
        src = inspect.getsource(T.Trainer.run)
        self.assertIn("phase_update", src,
                      "the training loop never passes the rung's own update count")
        self.assertIn("updates_in_phase", src)


class TestMeasurementSeatsPairsLikeTraining(unittest.TestCase):
    """A promotion check must ask the question training asks.

    Training never seats an agent opposite itself (`Population.pair`); the
    evaluation drew the two seats independently, so with two founders half of
    every check was an agent reading its own words. Two founders who had each
    invented a dialect the other could read scored 0.92 in training and 0.60 in
    the check, and `name-fruit` ran out its budget with a working code.
    """

    def _pool(self, n):
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = n
        cfg.population.founders_farmers = cfg.population.founders_buyers = n
        torch.manual_seed(0)
        return cfg, Population(cfg, random.Random(0))

    def _seats(self, cfg, pop, n_eps=400):
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        phase = phase_named(cfg, "name-fruit")
        out = M._play(cfg, pop, None, n_eps, list(range(len(pop.farmers))),
                      list(range(len(pop.buyers))), rng=random.Random(0),
                      phase=phase, sampler=lambda n, held_out=False: rw.sample(n))
        return out["pairing"]

    def test_training_never_seats_an_agent_opposite_itself(self):
        cfg, pop = self._pool(2)
        self.assertTrue(pop.shared)
        f, b = pop.pair(64)
        self.assertFalse(bool((f == b).any()))

    def test_neither_does_the_evaluation(self):
        for n in (2, 3):
            cfg, pop = self._pool(n)
            f, b = self._seats(cfg, pop)
            self.assertFalse(bool((f == b).any()),
                             "a %d-agent pool was measured talking to itself" % n)
            # and every agent still takes both seats
            self.assertEqual(set(f.tolist()), set(range(n)))
            self.assertEqual(set(b.tolist()), set(range(n)))

    def test_split_roles_are_left_alone(self):
        """Once farmers and buyers are different agents, index i is two agents."""
        cfg, pop = self._pool(2)
        pop.split_roles(0)
        self.assertFalse(pop.shared)
        f, b = self._seats(cfg, pop)
        self.assertTrue(bool((f == b).any()),
                        "farmer i and buyer i are different agents and may meet")


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


class TestAPerfectSpeakerPasses(unittest.TestCase):
    """The bars are measured, so a flawless describer has to clear them.

    ``TestEveryRungIsReachable`` hands ``evaluate_rung`` an evidence dict of
    ones, which proves the *rule* can pass but never asks whether the
    *measurements* can produce those numbers. They could not. ``name-all`` is
    the only rung judged on message structure, and it is also the rung that
    mixes queries most -- 70% whole things, 30% single fields -- so 30% of its
    probes asked a perfect describer for one field and then scored its one-word
    answer against all three. A flawless, fully compositional, noise-free
    speaker measured that way reached field coverage 0.33-0.49 against a 0.30
    bar and topsim 0.32 against its own shuffled null. A real run cannot beat
    a perfect one, so the rung could not be left.
    """

    def _cfg(self):
        cfg = Config()
        cfg.model.d_model, cfg.model.d_ff = 48, 96
        return cfg

    def _perfect_speaker(self, cfg):
        """Name exactly the field(s) asked for, one short word each."""
        from orchard.curriculum import ASK_ALL
        sp = cfg.channel.space_id
        block = [0, cfg.world.n_varieties,
                 cfg.world.n_varieties + cfg.world.n_colors]

        def speak(m):
            query = m[3]
            if query != ASK_ALL:
                return [block[query] + m[query]]
            return [block[0] + m[0], sp, block[1] + m[1], sp, block[2] + m[2]]
        return speak

    def _measure(self, cfg, phase, n):
        from orchard.properties import field_coverage
        from orchard.world import K_EMPTY, K_FIELD
        view = phase.views()[0]
        kinds = M.phase_kinds(cfg, FARMER, view)
        real = [i for i, k in enumerate(kinds) if k not in (K_EMPTY, K_FIELD)]
        speak = self._perfect_speaker(cfg)
        meanings = M.tuple_meanings(cfg, n, seed=7, phase=view,
                                    query=M.probe_query(view))
        msgs = [speak(m) for m in meanings]
        cov = field_coverage(meanings, msgs, real, rng=random.Random(0))["coverage"]
        ts = M.topographic_similarity(meanings, msgs, cfg, FARMER, metric="hamming",
                                      rng=random.Random(0), kinds=kinds, n_null=2)
        return cov, ts["topsim"] - ts["null_mean"]

    def test_a_flawless_describer_clears_the_bars_it_is_judged_on(self):
        cfg = self._cfg()
        c = cfg.curriculum
        for phase in ladder(cfg):
            if not (phase.swaps and phase.whole):
                continue            # only these rungs are judged on structure
            for n in (100, 200):
                cov, gap = self._measure(cfg, phase, n)
                # Both metrics run to 1.0, and this speaker is flawless: it
                # should be near the top of the scale, not a whisker above the
                # bar. A real code is always worse than this one, so whatever
                # margin is missing here is missing from every run.
                self.assertGreaterEqual(
                    cov, 0.75,
                    "%s: a perfect describer covers only %.3f of each field over "
                    "%d probes (bar %.2f) -- no real code can beat it"
                    % (phase.name, cov, n, c.min_field_coverage))
                self.assertGreaterEqual(
                    gap, 0.75,
                    "%s: a perfect describer is only %.3f clear of its own null "
                    "over %d probes (bar %.2f)"
                    % (phase.name, gap, n, c.min_topsim_over_null))

    def test_the_bars_do_not_move_with_the_probe_count(self):
        """Coverage is a plug-in estimate; its *ceiling* must not follow the sample.

        Normalised by H(field) it did: the same perfect code read 0.50 over 100
        probes and 0.93 over 800, so the light promotion check (half the probes)
        was strictly harder to pass than the checkpoint one.
        """
        cfg = self._cfg()
        phase = phase_named(cfg, "name-all")
        scores = [self._measure(cfg, phase, n)[0] for n in (100, 200, 400)]
        self.assertLess(max(scores) - min(scores), 0.1,
                        "field coverage moved %.3f with the probe count alone: %s"
                        % (max(scores) - min(scores), scores))

    def test_a_describer_that_says_nothing_useful_still_fails(self):
        """The debias must not turn the bar into a formality."""
        from orchard.properties import field_coverage
        from orchard.world import K_EMPTY, K_FIELD
        cfg = self._cfg()
        phase = phase_named(cfg, "name-all")
        view = phase.views()[0]
        kinds = M.phase_kinds(cfg, FARMER, view)
        real = [i for i, k in enumerate(kinds) if k not in (K_EMPTY, K_FIELD)]
        meanings = M.tuple_meanings(cfg, 200, seed=7, phase=view,
                                    query=M.probe_query(view))
        rng = random.Random(3)
        noise = [[rng.randrange(cfg.channel.atomic_vocab) for _ in range(3)]
                 for _ in meanings]
        cov = field_coverage(meanings, noise, real, rng=random.Random(0))["coverage"]
        self.assertLess(cov, cfg.curriculum.min_field_coverage,
                        "a message unrelated to the meaning covered %.3f of each field"
                        % cov)

    def test_the_probes_ask_the_kind_the_rung_is_promoted_on(self):
        from orchard.curriculum import ASK_ALL
        cfg = self._cfg()
        for phase in ladder(cfg):
            view = phase.views()[0]
            q = M.probe_query(view)
            if not phase.tuples:
                self.assertIsNone(q, "%s has no query slot to fix" % phase.name)
                continue
            if phase.referential:
                self.assertEqual(q, phase.primary,
                                 "%s probes a kind it is not promoted on" % phase.name)
            asked = {m[3] for m in M.tuple_meanings(cfg, 40, seed=1, phase=view, query=q)}
            self.assertEqual(asked, {q},
                             "%s probed a mixture: %s" % (phase.name, sorted(asked)))
        self.assertEqual(M.probe_query(phase_named(cfg, "name-all").views()[0]), ASK_ALL)

    def test_a_probe_feeds_the_observation_the_rung_feeds(self):
        """The query slot has its own embedding table, so a probe that writes a
        value the rung never writes is measuring an off-distribution speaker.
        ``mutual`` pads that slot; the naming rungs fill it with the query."""
        from orchard.curriculum import ReferentialWorld
        cfg = self._cfg()
        gen = torch.Generator().manual_seed(0)
        rw = ReferentialWorld(cfg, generator=gen)
        for phase in ladder(cfg):
            if not phase.tuples:
                continue
            view = phase.views()[0]
            if phase.mutual:
                real = rw.sample_mutual(8).obs(cfg, FARMER)
            else:
                real = rw.sample(8, informer=view.informer, mix=view.mix).obs(
                    cfg, view.informer)
            probe = M.tuple_meanings(cfg, 8, seed=2, phase=view,
                                     query=M.probe_query(view))
            played = set(real[:, 3].tolist())
            self.assertIn(probe[0][3], played,
                          "%s probes with query slot %d, which the rung never "
                          "puts there (it plays %s)"
                          % (phase.name, probe[0][3], sorted(played)))


class TestTheConventionTermCanAffordToRun(unittest.TestCase):
    """It is charged per episode against a sample of other meanings' forms, so
    at a GPU batch it runs tens of thousands of edit distances per update --
    the whole cost of the term, and pure host-side Python while the device
    waits. Both the distance and the sample size were tuned for that; neither
    may change what is computed."""

    def _ref_edit(self, a, b):
        """The textbook version the tightened one replaced."""
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def test_the_tightened_edit_distance_is_the_same_distance(self):
        from orchard.conventions import _edit
        rng = random.Random(7)
        for _ in range(4000):
            a = [rng.randrange(6) for _ in range(rng.randrange(0, 10))]
            b = [rng.randrange(6) for _ in range(rng.randrange(0, 10))]
            self.assertEqual(_edit(a, b), self._ref_edit(a, b),
                             "edit(%s, %s) changed" % (a, b))
        # the cases the rolling row is easiest to get wrong on
        self.assertEqual(_edit([], []), 0)
        self.assertEqual(_edit([1, 2, 3], []), 3)
        self.assertEqual(_edit([], [1, 2]), 2)
        self.assertEqual(_edit([1, 2, 3], [1, 2, 3]), 0)
        self.assertEqual(_edit([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(_edit([1, 2, 3], [1, 3]), 1)

    def test_the_contrast_sample_is_a_knob_and_is_honoured(self):
        cfg = cfg_small()
        cfg.reward.convention_min_support = 1
        c = cfg.channel
        phase = phase_named(cfg, "name-all")
        seen = {}
        for n in (1, 8):
            cfg.reward.convention_contrast_samples = n
            u = PopulationUsage(cfg)
            obs = torch.tensor([[i % 3, (i // 3) % 3, i % 2, 3] + [0] * 8
                                for i in range(12)])
            toks = torch.full((12, c.dialogue_len), c.pad_id, dtype=torch.long)
            for i in range(12):
                toks[i, :2] = torch.tensor([4 + i % 5, c.end_id])
            for _ in range(6):
                u.observe(u.speaker_terms(phase, toks, {FARMER: obs, BUYER: obs}), 12)
            calls = []
            import orchard.conventions as C
            real = C.similarity
            C.similarity = lambda a, b: (calls.append(1), real(a, b))[1]
            try:
                u.speaker_terms(phase, toks, {FARMER: obs, BUYER: obs})
            finally:
                C.similarity = real
            seen[n] = len(calls)
        self.assertLess(seen[1], seen[8],
                        "the contrast sample size did nothing: %s" % seen)


class TestSharedPoolIsNotComparedWithItself(unittest.TestCase):
    """Below ``split_roles_at`` one pool fills both seats, so "the two roles"
    are the same agents. Comparing them measured self-agreement: with two
    founders half of every cross pair was an agent against itself, and the
    checkpoint line read cross-role coherence 0.56 and overlap 0.87 for a pair
    that shared no form at all (within-role coherence 0.16)."""

    def test_cross_role_coherence_skips_the_same_agent_in_the_other_seat(self):
        cfg = cfg_small()
        tracker = M.StabilityTracker(cfg, M.World(cfg.world, random.Random(0)),
                                     n_probes=6, seed=1)
        torch.manual_seed(0)
        pop = Population(cfg, random.Random(0))
        self.assertTrue(pop.shared, "the naming rungs are meant to share one pool")
        self.assertIs(pop.farmers, pop.buyers)
        phase = phase_named(cfg, "name-all")
        out = tracker.measure(pop, M.World(cfg.world, random.Random(0)), phase=phase)
        within = [out["coherence_farmer"], out["coherence_buyer"]]
        cross = out["coherence_cross"]
        # With two founders and self-pairs included, cross is pinned at
        # 1 - d/2 -- always about halfway to 1 however foreign the two codes
        # are. Excluding them, it can only be the honest between-agent number.
        self.assertTrue(cross != cross or cross <= max(within) + 0.15,
                        "cross-role coherence %.3f sits above the within-role "
                        "numbers %s: self-pairs are still in it" % (cross, within))

    def test_overlap_is_not_answered_while_one_pool_fills_both_seats(self):
        cfg = cfg_small()
        batch = _FakeBatch(cfg, [1, 2, 1, 2], [1, 2, 1, 2])
        shared = cross_role_overlap(cfg, [batch], shared_pool=True)
        self.assertTrue(shared["weighted_overlap"] != shared["weighted_overlap"],
                        "a shared pool was scored as if it were two codes")
        self.assertIn("note", shared)
        split = cross_role_overlap(cfg, [batch], shared_pool=False)
        self.assertAlmostEqual(split["weighted_overlap"], 1.0, places=6)


class TestConventionsKnowWhatWasAsked(unittest.TestCase):
    """A convention is a form *for a meaning*, and on a rung that asks different
    questions about the same thing, the question is part of the meaning. Keyed
    on the tuple alone, ``name-all``'s conventions blended the answers to "what
    fruit?" and "what is it?" into one modal form."""

    def test_the_same_tuple_asked_differently_is_a_different_convention(self):
        cfg = cfg_small()
        c = cfg.channel
        u = PopulationUsage(cfg)
        phase = phase_named(cfg, "name-all")
        thing = [1, 2, 0]
        whole = torch.tensor([thing + [3] + [0] * 8])      # ASK_ALL
        fruit = torch.tensor([thing + [0] + [0] * 8])      # just the fruit
        keys_whole = u._keys(phase, FARMER, whole)
        keys_fruit = u._keys(phase, FARMER, fruit)
        self.assertNotEqual(keys_whole, keys_fruit,
                            "one key served two questions about the same thing")

    def test_a_rung_with_no_query_slot_is_unchanged(self):
        cfg = cfg_small()
        u = PopulationUsage(cfg)
        market = phase_named(cfg, "market")
        obs = torch.zeros((2, 40), dtype=torch.long)
        keys = u._keys(market, BUYER, obs)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])

    def test_the_key_reads_kind_then_question_then_meaning(self):
        cfg = cfg_small()
        u = PopulationUsage(cfg)
        phase = phase_named(cfg, "name-all")
        obs = torch.tensor([[1, 2, 0, 3] + [0] * 8])
        self.assertEqual(u._keys(phase, FARMER, obs)[0], ("tuple", 3, 1, 2, 0))

    def test_a_resumed_run_forgets_conventions_in_the_older_key_format(self):
        """Snapshots written before the key carried the question are shorter.

        Kept, they sit in the contrast set for a couple of half-lives, scoring
        speakers against the modal forms of meanings those keys no longer
        denote. Word counts are keyed by the word and are untouched.
        """
        cfg = cfg_small()
        u = PopulationUsage(cfg)
        old_key = ("tuple", 1, 2, 0)            # no question in it
        new_key = ("tuple", 3, 1, 2, 0)
        for k in (old_key, new_key):
            u.forms[k][(4,)] = 20.0
            u.form_total[k] = 20.0
        u.words[(4,)] = 40.0
        u.word_total = 40.0
        self.assertEqual(u.drop_stale_forms(), 1)
        self.assertIn(new_key, u.form_total)
        self.assertNotIn(old_key, u.form_total)
        self.assertNotIn(old_key, u.forms)
        self.assertEqual(u.word_total, 40.0, "the word counts were disturbed")
        self.assertEqual(u.drop_stale_forms(), 0, "it is not idempotent")

    def test_every_kind_of_meaning_has_one_key_length(self):
        """`drop_stale_forms` is only safe if arity is fixed per meaning kind."""
        cfg = cfg_small()
        u = PopulationUsage(cfg)
        want = u.key_arities()
        for phase in ladder(cfg):
            for view in phase.views():
                for role in (FARMER, BUYER):
                    if not view.speaks(cfg, role):
                        continue
                    obs = torch.zeros((1, 40), dtype=torch.long)
                    key = u._keys(view, role, obs)[0]
                    self.assertEqual(len(key), want[key[0]],
                                     "%s/%s writes a %r key of length %d, not %d"
                                     % (phase.name, role, key[0], len(key),
                                        want[key[0]]))


# ==========================================================================
class TestProductivityIsJudgedOnFieldsNotTheConjunction(unittest.TestCase):
    """`mutual` scores a round only when three fields land on each of two novel
    meanings. Per-field accuracy therefore enters the held-out number to the
    sixth power, and the ratio the gate reads is no longer a measure of whether
    the code generalises -- it is that measure raised to a power that crushes
    every partial result into the noise around zero."""

    def test_the_conjunction_hides_a_plainly_productive_code(self):
        # What the exponent does to a code that generalises at 0.73 per field
        # when it manages 0.80 on what it trained on.
        per_field_ratio = 0.73 / 0.80
        joint_ratio = (0.73 ** 6) / (0.80 ** 6)
        self.assertGreater(per_field_ratio, 0.90)
        self.assertLess(joint_ratio, 0.60)      # fails the bar it should clear

    def test_the_gate_reads_the_fields_when_they_are_there(self):
        cfg = cfg_small()
        phase = phase_named(cfg, "mutual")
        self.assertTrue(phase.whole)
        ev = _mutual_evidence(cfg, holdout_fields=0.73, seen_fields=0.80,
                              holdout_success=0.73 ** 6, seen_success=0.80 ** 6,
                              holdout_field_ratio=(0.73 - 0.28) / (0.80 - 0.25))
        _, checks = evaluate_rung(cfg, phase, ev, updates_in_phase=10 ** 6)
        check = checks["describes combinations it never trained on"]
        self.assertTrue(check["met"], check["detail"])
        self.assertIn("per field", check["detail"])

    def test_a_memorised_code_still_fails(self):
        """The fix must not be a lower bar: a code sitting at the floor for
        meanings it never saw reads 0.00, not the base rate it scores anyway."""
        cfg = cfg_small()
        phase = phase_named(cfg, "mutual")
        ev = _mutual_evidence(cfg, holdout_fields=0.28, seen_fields=0.80,
                              holdout_success=0.0, seen_success=0.80 ** 6,
                              holdout_field_ratio=0.0)
        _, checks = evaluate_rung(cfg, phase, ev, updates_in_phase=10 ** 6)
        self.assertFalse(
            checks["describes combinations it never trained on"]["met"])

    def test_a_rung_without_field_reports_keeps_the_old_check(self):
        """Lineup rungs score one K-way choice, so there is no exponent to
        remove and the joint ratio is still the right number."""
        cfg = cfg_small()
        phase = phase_named(cfg, "name-all")
        ev = _mutual_evidence(cfg, holdout_fields=float("nan"),
                              seen_fields=float("nan"), holdout_success=0.80,
                              seen_success=0.85, holdout_field_ratio=float("nan"))
        _, checks = evaluate_rung(cfg, phase, ev, updates_in_phase=10 ** 6)
        detail = checks["describes combinations it never trained on"]["detail"]
        self.assertNotIn("per field", detail)


class TestTheFloorComesFromThePoolBeingScored(unittest.TestCase):
    """A quarter of 64 combinations is 16 rows, and 16 rows need not be balanced.
    Scoring a per-field number against 1/4 would credit a message-blind guesser
    with whatever skew the reserved pool happens to have."""

    def test_a_skewed_pool_has_a_higher_floor(self):
        # fruit is 0 in six of eight rows; colour and quality are balanced.
        pool = torch.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3],
                             [0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 0]])
        floor = M.pool_field_floor(pool)
        # fruit 6/8, colour 2/8, quality 2/8
        self.assertAlmostEqual(floor, (0.75 + 0.25 + 0.25) / 3, places=6)
        self.assertGreater(floor, 0.25)

    def test_a_balanced_pool_sits_at_one_over_the_span(self):
        pool = torch.tensor([[f, c, q] for f in range(4)
                             for c in range(4) for q in range(4)])
        self.assertAlmostEqual(M.pool_field_floor(pool), 0.25, places=6)

    def test_an_empty_pool_is_not_a_crash(self):
        self.assertNotEqual(M.pool_field_floor(torch.zeros((0, 3), dtype=torch.long)),
                            M.pool_field_floor(torch.zeros((0, 3), dtype=torch.long)))


def _mutual_evidence(cfg, **over):
    """A `phase_evidence` row that passes every check but the productivity one."""
    role = {"field_coverage": 1.0, "per_field_coverage": [1.0, 1.0, 1.0],
            "topsim": 0.9, "topsim_null": 0.1, "topsim_over_null": 0.9,
            "positional_structure": 0.9, "report": 0.9}
    ev = {
        "success": 0.9, "chance": float("nan"), "transfer": 1.0,
        "holdout_ratio": (over.get("holdout_success", 0.0)
                          / max(1e-9, over.get("seen_success", 1.0))),
        "per_role_structure": {"farmer": dict(role), "buyer": dict(role)},
        "speakers": {"farmer": dict(role), "buyer": dict(role)},
        "views": [{"success": 0.9, "transfer": 1.0}],
        "mutual_report": 0.9, "farmer_report": 0.9, "buyer_report": 0.9,
        "muted_success": 0.05, "by_kind": {},
    }
    ev.update(over)
    return ev


import tempfile


# ==========================================================================
class TestASnapshotDecidesItsOwnArchitecture(unittest.TestCase):
    """The shape-deciding settings have exactly one valid reading on a resume:
    the one the weights were trained under. Leaving them to the command line
    turns a forgotten `--config` into sixty `size mismatch` lines that name
    tensors and never name the setting that is wrong."""

    def _snapshot_at(self, d, d_model, d_ff, batch_size=None):
        from orchard.train import Trainer
        cfg = cfg_small()
        cfg.population.n_farmers = cfg.population.n_buyers = 2
        cfg.model.d_model, cfg.model.d_ff = d_model, d_ff
        if batch_size is not None:
            cfg.train.batch_size = batch_size
        cfg.train.device = "cpu"
        cfg.log.plot = False
        tr = Trainer(cfg, d + "/a", quiet=True)
        tr.episode = 128
        path = tr.save_snapshot("t")
        tr.close()
        return path

    def test_a_wider_snapshot_loads_under_the_narrow_default(self):
        """The exact failure a resume without its preset produced: a 96-wide
        run, resumed under the 48-wide default."""
        from orchard.train import Trainer
        with tempfile.TemporaryDirectory() as d:
            path = self._snapshot_at(d, 96, 384)
            cfg = cfg_small()                      # d_model 48, d_ff 96
            cfg.population.n_farmers = cfg.population.n_buyers = 2
            cfg.train.device = "cpu"
            cfg.log.plot = False
            self.assertEqual(cfg.model.d_model, 48)
            tr2 = Trainer(cfg, d + "/b", quiet=True)
            tr2.load_snapshot(path)                # must not raise
            self.assertEqual(tr2.cfg.model.d_model, 96)
            self.assertEqual(tr2.cfg.model.d_ff, 384)
            for a in tr2.pop.all_agents():
                self.assertEqual(a.net.d_model, 96)
            tr2.close()

    def test_it_leaves_alone_what_a_resume_may_change(self):
        """Community size and batch size are not architecture: a resume is
        allowed to shrink a 32-agent run to 8, which is how this one is run."""
        from orchard.train import Trainer
        with tempfile.TemporaryDirectory() as d:
            path = self._snapshot_at(d, 96, 384)
            cfg = cfg_small()
            cfg.population.n_farmers = cfg.population.n_buyers = 2
            cfg.train.device = "cpu"
            cfg.train.batch_size = 77
            cfg.log.plot = False
            tr2 = Trainer(cfg, d + "/b", quiet=True)
            tr2.load_snapshot(path)
            self.assertEqual(tr2.cfg.train.batch_size, 77)
            tr2.close()

    def test_a_shape_nobody_listed_still_fails_legibly(self):
        """`_check_shapes` is the backstop for a field missing from ARCH_KEYS:
        the message has to name the configuration difference, not the tensors."""
        import orchard.train as T
        from orchard.train import Trainer
        with tempfile.TemporaryDirectory() as d:
            path = self._snapshot_at(d, 96, 384)
            cfg = cfg_small()
            cfg.population.n_farmers = cfg.population.n_buyers = 2
            cfg.train.device = "cpu"
            cfg.log.plot = False
            tr2 = Trainer(cfg, d + "/b", quiet=True)
            was = T.ARCH_KEYS
            T.ARCH_KEYS = frozenset()        # as if nobody had listed d_model
            try:
                with self.assertRaises(RuntimeError) as got:
                    tr2.load_snapshot(path)
            finally:
                T.ARCH_KEYS = was
            msg = str(got.exception)
            self.assertIn("model.d_model", msg)
            self.assertIn("--config", msg)
            tr2.close()

    def test_it_says_when_the_rest_of_the_settings_differ(self):
        """Adopting the architecture silently would turn a loud crash into a
        quiet one: the forgotten `--config` that used to stop the run would
        instead carry on at the default batch of 256 where it had been training
        at 4096, and nothing would say so."""
        from orchard.train import Trainer
        with tempfile.TemporaryDirectory() as d:
            path = self._snapshot_at(d, 96, 384, batch_size=4096)
            cfg = cfg_small()
            cfg.population.n_farmers = cfg.population.n_buyers = 2
            cfg.train.device = "cpu"
            cfg.train.batch_size = 256          # the snapshot was written at 4096
            cfg.log.plot = False
            said = []
            tr2 = Trainer(cfg, d + "/b", quiet=True)
            tr2.log.always = lambda m, *a: said.append(m % a if a else m)
            tr2.load_snapshot(path)
            joined = "\n".join(said)
            self.assertIn("not trained with", joined)
            self.assertIn("train.batch_size", joined)
            tr2.close()


# ==========================================================================
class TestTheHoldoutAsksForTheOneQualityItRuledOut(unittest.TestCase):
    """The reserved set is a Latin square: for every (fruit, colour) pair
    exactly one quality is withheld, and a held-out round asks for precisely
    that value. A listener that has fit the training distribution has learned
    that value cannot occur there, so one field can sit near zero for a reason
    that has nothing to do with whether the code is compositional -- and the
    mean over three fields cannot say which."""

    def test_every_cell_withholds_exactly_one_quality(self):
        from collections import defaultdict

        from orchard.world import ComboHoldout
        w = Config().world
        h = ComboHoldout(w, w.holdout_combo_frac, w.holdout_seed)
        cells = defaultdict(list)
        for (f, c, q) in h.held:
            cells[(f, c)].append(q)
        self.assertEqual(len(cells), w.n_varieties * w.n_colors)
        self.assertTrue(all(len(v) == 1 for v in cells.values()))

    def test_one_dead_field_pulls_the_conjunction_to_zero_not_the_mean(self):
        """Why the whole-round number cannot be read as a productivity failure:
        two fields generalising beautifully and one at the floor still scores
        essentially nothing as a conjunction."""
        fruit, colour, quality = 0.95, 0.80, 0.02
        mean = (fruit + colour + quality) / 3
        joint = (fruit * colour * quality) ** 2      # both sides, three fields
        self.assertGreater(mean, 0.55)
        self.assertLess(joint, 0.001)                # prints as 0.000

    def test_the_breakdown_is_carried_out_of_the_evaluation(self):
        vec = M._report_field_vec({"farmer_report_fields": [0.9, 0.8, 0.0],
                                   "buyer_report_fields": [1.0, 0.8, 0.1]})
        self.assertEqual(len(vec), 3)
        self.assertAlmostEqual(vec[0], 0.95, places=6)
        self.assertAlmostEqual(vec[1], 0.80, places=6)
        self.assertAlmostEqual(vec[2], 0.05, places=6)

    def test_a_round_that_reports_no_fields_has_no_breakdown(self):
        self.assertIsNone(M._report_field_vec({"success_rate": 0.5}))
        self.assertIsNone(M._report_field_vec(None))

    def test_a_ratio_of_two_numbers_at_the_floor_is_not_a_pass(self):
        """Both at chance is not "generalises perfectly": it is no signal at
        all, and noise reads 1.00 as readily as 0.00."""
        cfg = cfg_small()
        phase = phase_named(cfg, "mutual")
        ev = _mutual_evidence(cfg, holdout_fields=0.262, seen_fields=0.259,
                              holdout_success=0.0, seen_success=0.0,
                              holdout_field_ratio=float("nan"))
        _, checks = evaluate_rung(cfg, phase, ev, updates_in_phase=10 ** 6)
        self.assertFalse(
            checks["describes combinations it never trained on"]["met"])
