"""Tests for the gesture channel and the innate word classes (orchard/gesture.py).

What each class protects:

* a gesture is a symbol id of its own, never a word, never costed, never parsed;
* a gesture can only show what its maker can see -- the value comes off the
  speaker's own observation, and a farmer at market can show only its floor;
* the world withdraws gestures over a naming rung and never allows one in any
  measurement, so every gate stays word-only;
* the token policy is not credited for a gesture slot, and a newborn is never
  taught to emit one -- though it reads them like everyone else;
* the factored lineup choice is the field's belief head on a single-field round;
* gestures are a change of method, not of architecture.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from orchard import gesture as G
from orchard.agents import make_agent
from orchard.bottleneck import TranscriptStore, train_newborn
from orchard.config import ARCH_KEYS, Config
from orchard.curriculum import ReferentialWorld, ladder, phase_named, phase_schema
from orchard.env import BUYER, FARMER, length_cost, parse_words
from orchard.gumbel import run_and_update_gumbel
from orchard.lexicon import word_stats
from orchard.render import render_message
from orchard.rollout import run_episodes
from orchard.world import LOT_FIELDS, N_LOT_FIELDS, lot_spans, n_cells

from test_curriculum import agents, cfg_small


def pairing(B, n=2):
    i = torch.arange(B)
    return i % n, torch.div(i, n, rounding_mode="floor") % n


def gestured_batch(cfg, phase_name="name-quantity", n=96, seed=3, share=1.0):
    torch.manual_seed(seed)
    f, b = agents(cfg)
    rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(seed))
    ph = phase_named(cfg, phase_name)
    fi, bi = pairing(n)
    batch, stats = run_and_update_gumbel(cfg, rw.sample(n, informer=ph.informer, mix=ph.mix),
                                         f, b, fi, bi, phase=ph, gesture_share=share)
    return cfg, f, b, ph, batch, stats


class TestGestureIds(unittest.TestCase):
    def test_gesture_ids_follow_the_symbols_and_do_not_collide(self):
        cfg = cfg_small()
        c = cfg.channel
        self.assertEqual(G.first_gesture_id(cfg), c.n_symbol_ids)
        self.assertEqual(G.n_gesture_ids(cfg), sum(lot_spans(cfg.world)))
        self.assertEqual(G.n_token_ids(cfg), c.n_symbol_ids + G.n_gesture_ids(cfg))
        for sym in range(c.n_symbol_ids):
            self.assertFalse(G.is_gesture(cfg, sym))
            self.assertIsNone(G.gesture_meaning(cfg, sym))
        seen = set()
        for field, span in enumerate(lot_spans(cfg.world)):
            for v in range(span):
                tok = G.gesture_id(cfg, field, v)
                self.assertTrue(G.is_gesture(cfg, tok))
                self.assertEqual(G.gesture_meaning(cfg, tok), (field, v))
                self.assertNotIn(tok, seen)
                seen.add(tok)
        self.assertEqual(len(seen), G.n_gesture_ids(cfg))

    def test_a_gesture_is_not_a_word_and_costs_nothing_as_a_symbol(self):
        cfg = cfg_small()
        cfg.channel.max_symbols = 8
        c = cfg.channel
        g = G.gesture_id(cfg, 3, 5)
        turn = [g, 2, c.hyphen_id, 7, c.space_id, 1, c.end_id]
        self.assertEqual(parse_words(cfg, turn), [(2, 7), (1,)])
        self.assertFalse(c.costed(g))
        self.assertEqual(G.strip_gestures(cfg, turn), turn[1:])
        L = c.max_msg_len
        toks = torch.full((2, L), c.pad_id, dtype=torch.long)
        toks[0, :len(turn)] = torch.tensor(turn)
        toks[1, :len(turn) - 1] = torch.tensor(turn[1:])
        cost = length_cost(cfg, toks, list(range(L)))
        self.assertAlmostEqual(float(cost[0]), float(cost[1]), places=6,
                               msg="the gesture slot was charged for")

    def test_gestures_are_rendered_as_what_they_are(self):
        cfg = cfg_small()
        c = cfg.channel
        text = render_message(cfg, [G.gesture_id(cfg, 3, 3), 1, c.end_id])
        self.assertIn("[3 fingers]", text)
        self.assertIn("a1", text)
        self.assertIn("[1 finger]", render_message(cfg, [G.gesture_id(cfg, 3, 1), 4]))
        self.assertIn("for @", render_message(cfg, [G.gesture_id(cfg, 4, 0), 4]))
        self.assertIn("[points: %s]" % cfg.world.variety_names[1],
                      render_message(cfg, [G.gesture_id(cfg, 0, 1), 4]))
        # a gesture alone is not silence: the hands said something
        self.assertNotEqual(render_message(cfg, [G.gesture_id(cfg, 1, 2)]), "<silence>")


class TestAGestureShowsOnlyWhatTheSpeakerSees(unittest.TestCase):
    def test_lot_speakers_may_show_any_field_and_the_farmer_at_market_only_its_floor(self):
        cfg = cfg_small()
        for name in ("name-fruit", "name-all", "mutual"):
            ph = phase_named(cfg, name)
            for role in (FARMER, BUYER):
                self.assertEqual(G.gesture_columns(cfg, ph, role),
                                 [(j, j) for j in range(N_LOT_FIELDS)])
        for name in ("order", "haggle", "market"):
            ph = phase_named(cfg, name)
            self.assertEqual(G.gesture_columns(cfg, ph, BUYER),
                             [(j, j) for j in range(N_LOT_FIELDS)])
            cols = G.gesture_columns(cfg, ph, FARMER)
            self.assertEqual(cols, [(4, 4 * n_cells(cfg.world))],
                             "the farmer may hold up fingers for its floor price only")
            # and that column *is* the reservation in the farmer's schema
            from orchard.world import K_PRICE, farmer_schema
            self.assertEqual(farmer_schema(cfg.world)[cols[0][1]], K_PRICE)

    def test_the_value_shown_is_read_off_the_observation(self):
        cfg = cfg_small()
        ph = phase_named(cfg, "name-quantity")
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(5))
        rb = rw.sample(32, informer=FARMER, query=3)
        obs = rb.obs(cfg, FARMER)
        for field in range(N_LOT_FIELDS):
            choice = torch.full((32,), 1 + field, dtype=torch.long)
            ids = G.gesture_tokens_for(cfg, ph, FARMER, obs, choice)
            for i in range(32):
                self.assertEqual(G.gesture_meaning(cfg, int(ids[i])),
                                 (field, int(obs[i, field])))
        none = G.gesture_tokens_for(cfg, ph, FARMER, obs, torch.zeros(32, dtype=torch.long))
        self.assertTrue(bool((none == cfg.channel.pad_id).all()))
        # a seat that cannot show a field gets nothing for asking
        market = phase_named(cfg, "market")
        fobs = torch.zeros((4, obs.shape[1]), dtype=torch.long)
        fobs[:, 4 * n_cells(cfg.world)] = 2
        ids = G.gesture_tokens_for(cfg, market, FARMER, fobs, torch.full((4,), 1, dtype=torch.long))
        self.assertTrue(bool((ids == cfg.channel.pad_id).all()), "a barn has no fruit to point at")
        ids = G.gesture_tokens_for(cfg, market, FARMER, fobs, torch.full((4,), 5, dtype=torch.long))
        self.assertEqual(G.gesture_meaning(cfg, int(ids[0])), (4, 2))
        mask = G.gesture_option_mask(cfg, market, FARMER)
        self.assertEqual(mask.tolist(), [True, False, False, False, False, True])


class TestTheWorldDecidesWhen(unittest.TestCase):
    def test_gestures_are_withdrawn_over_a_naming_rung_and_stay_small_afterwards(self):
        cfg = cfg_small()
        g = cfg.gesture
        for ph in ladder(cfg):
            if ph.invents:
                self.assertAlmostEqual(G.gesture_share(cfg, ph, 0), g.share_start)
                self.assertAlmostEqual(G.gesture_share(cfg, ph, g.anneal_updates // 2),
                                       (g.share_start + g.share_end) / 2)
                self.assertAlmostEqual(G.gesture_share(cfg, ph, g.anneal_updates), g.share_end)
                self.assertAlmostEqual(G.gesture_share(cfg, ph, 10 * g.anneal_updates),
                                       g.share_end)
            else:
                self.assertAlmostEqual(G.gesture_share(cfg, ph, 0), g.share_reuse)
                self.assertAlmostEqual(G.gesture_share(cfg, ph, 5000), g.share_reuse)
        cfg.gesture.enabled = False
        for ph in ladder(cfg):
            self.assertEqual(G.gesture_share(cfg, ph, 0), 0.0)

    def test_availability_is_a_share_of_rounds(self):
        torch.manual_seed(0)
        self.assertFalse(bool(G.draw_availability(0.0, 500, "cpu").any()))
        self.assertTrue(bool(G.draw_availability(1.0, 500, "cpu").all()))
        share = float(G.draw_availability(0.3, 20000, "cpu").float().mean())
        self.assertAlmostEqual(share, 0.3, delta=0.02)


class TestGesturesInPlay(unittest.TestCase):
    def test_a_training_batch_can_open_a_turn_with_a_gesture(self):
        cfg = cfg_small()
        cfg.channel.max_symbols = 6
        cfg, f, b, ph, batch, stats = gestured_batch(cfg)
        L = cfg.channel.max_msg_len
        firsts = batch.tokens[:, 0].tolist()
        gestured = [i for i, t in enumerate(firsts) if G.is_gesture(cfg, t)]
        self.assertGreater(len(gestured), 10, "with every round allowing one, "
                           "an untrained speaker gestures in ~5/6 of turns")
        self.assertLess(len(gestured), len(firsts), "and not in all of them")
        for i in gestured:
            # the slot is the world's, not the token policy's act
            self.assertFalse(bool(batch.active[i, 0]))
            rest = [t for t in batch.tokens[i, 1:L].tolist() if t != cfg.channel.pad_id]
            self.assertTrue(rest and cfg.channel.is_atom(rest[0]),
                            "the spoken part still opens with a word")
            self.assertFalse(any(G.is_gesture(cfg, t) for t in rest),
                             "one gesture per turn, at its start")
            # a gesture never shows anything but the describer's own lot
            field, value = G.gesture_meaning(cfg, firsts[i])
            self.assertEqual(value, int(batch.sb.true_meaning[i, field]))
        for i in range(len(firsts)):
            if i not in gestured:
                self.assertTrue(bool(batch.active[i, 0]))
                self.assertTrue(cfg.channel.is_atom(firsts[i]))
        self.assertAlmostEqual(stats.gesture_available, 1.0)
        self.assertAlmostEqual(stats.gesture_used, len(gestured) / len(firsts), places=6)
        self.assertEqual(sum(stats.gesture_fields), len(gestured))
        # the gesture cost was charged to the describer, per gesture
        res = batch.res
        self.assertTrue(torch.allclose(res["farmer_gesture_cost"],
                                       cfg.gesture.cost * torch.tensor(
                                           [1.0 if i in gestured else 0.0
                                            for i in range(len(firsts))])))
        self.assertTrue(bool((res["buyer_gesture_cost"] == 0).all()),
                        "the guesser never spoke, so it never gestured")

    def test_a_gestured_turn_is_a_spoken_turn_without_its_gesture_in_the_statistics(self):
        cfg = cfg_small()
        cfg.channel.max_symbols = 6
        cfg, f, b, ph, batch, stats = gestured_batch(cfg)
        ws = word_stats(cfg, [batch])
        n_turns = int((batch.tokens[:, 0] != cfg.channel.pad_id).sum())
        self.assertEqual(round(ws["silent_frac"] * n_turns), 0)
        # every symbol counted is a spoken one
        L = cfg.channel.max_msg_len
        spoken = [sum(1 for t in row if cfg.channel.costed(t))
                  for row in batch.tokens[:, :L].tolist()]
        self.assertAlmostEqual(ws["mean_symbols_per_message"], sum(spoken) / len(spoken),
                               places=6)

    def test_without_a_share_nothing_gestures_and_measurement_never_does(self):
        cfg = cfg_small()
        cfg, f, b, ph, batch, stats = gestured_batch(cfg, share=0.0)
        self.assertFalse(any(G.is_gesture(cfg, t) for t in batch.tokens.flatten().tolist()))
        self.assertEqual(stats.gesture_fields, [])
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(9))
        fi, bi = pairing(128)
        for name in ("name-quantity", "name-all"):
            phase = phase_named(cfg, name)
            for mode in ("intact", "scrambled", "muted"):
                ev = run_episodes(cfg, rw.sample(128, informer=phase.informer), f, b, fi, bi,
                                  phase=phase, channel_mode=mode)
                self.assertFalse(any(G.is_gesture(cfg, t) for t in ev.tokens.flatten().tolist()),
                                 "%s/%s: a measurement emitted a gesture" % (name, mode))

    def test_the_listener_is_taught_the_gestured_field(self):
        cfg = cfg_small()
        cfg.channel.max_symbols = 6
        cfg, f, b, ph, batch, stats = gestured_batch(cfg)
        L = cfg.channel.max_msg_len
        field, value = G.gestured_fields(cfg, batch.tokens, [0])
        for i, t in enumerate(batch.tokens[:, 0].tolist()):
            m = G.gesture_meaning(cfg, t)
            if m is None:
                self.assertEqual(int(field[i]), -1)
            else:
                self.assertEqual((int(field[i]), int(value[i])), m)
        # ...and the supervision reached the guesser's report heads: their
        # weights moved in the update (the guesser scores only the choice, so
        # without a gesture term nothing but the factored choice would touch them)
        cfg2 = cfg_small()
        cfg2.channel.max_symbols = 6
        cfg2.model.factored_choice = False
        torch.manual_seed(3)
        f2, b2 = agents(cfg2)
        before = {n: p.detach().clone() for n, p in b2[1].net.named_parameters()
                  if n.startswith("belief_qty_head")}
        rw = ReferentialWorld(cfg2, generator=torch.Generator().manual_seed(3))
        ph2 = phase_named(cfg2, "name-quantity").with_informer(FARMER)
        fi, bi = pairing(96)
        run_and_update_gumbel(cfg2, rw.sample(96, informer=FARMER, query=3), f2, b2, fi, bi,
                              phase=ph2, gesture_share=1.0)
        moved = any(not torch.equal(before[n], p.detach())
                    for n, p in b2[1].net.named_parameters() if n in before)
        self.assertTrue(moved, "the gestured field's head was not taught")
        cfg2.gesture.supervise_coef = 0.0
        torch.manual_seed(3)
        f3, b3 = agents(cfg2)
        before = {n: p.detach().clone() for n, p in b3[1].net.named_parameters()
                  if n.startswith("belief_qty_head")}
        rw = ReferentialWorld(cfg2, generator=torch.Generator().manual_seed(3))
        run_and_update_gumbel(cfg2, rw.sample(96, informer=FARMER, query=3), f3, b3, fi, bi,
                              phase=ph2, gesture_share=1.0)
        still = all(torch.equal(before[n], p.detach())
                    for n, p in b3[1].net.named_parameters() if n in before)
        self.assertTrue(still, "with the term off and the pointer listener, nothing "
                        "should touch a head the rung does not score")

    def test_a_newborn_reads_gestures_but_is_never_taught_to_make_one(self):
        cfg = cfg_small()
        cfg.channel.max_symbols = 6
        cfg.bottleneck.only_successful = False
        cfg.bottleneck.epochs = 1
        cfg.bottleneck.meaning_holdout = 0.0
        cfg, f, b, ph, batch, stats = gestured_batch(cfg)
        store = TranscriptStore(cfg)
        store.add_batch(batch, f, b, episode=0)
        self.assertGreater(len(store), 0)
        own = ph.own_positions(cfg, FARMER)
        n_active = int(batch.active[:, own].sum())
        n_gest = sum(1 for t in batch.tokens[:, 0].tolist() if G.is_gesture(cfg, t))
        self.assertGreater(n_gest, 0)
        newborn = make_agent(cfg, agent_id=99, role=FARMER, slot=0, generation=1,
                             birth_episode=1, lifespan=10 ** 9)
        info = train_newborn(cfg, newborn, store, random.Random(0), roles=(FARMER, BUYER))
        self.assertNotIn("skipped", info)
        self.assertLessEqual(info["own_token_targets"], n_active,
                             "a gesture slot became a token lesson")
        self.assertGreater(info["own_token_targets"], 0)
        self.assertIsNotNone(info["token_accuracy"])


class TestInnateWordClasses(unittest.TestCase):
    def test_the_factored_choice_is_the_field_head_on_a_single_field_round(self):
        cfg = cfg_small()
        torch.manual_seed(1)
        f, b = agents(cfg)
        net = b[0].net
        ph = phase_named(cfg, "name-quantity").with_informer(FARMER)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(1))
        rb = rw.sample(16, informer=FARMER, query=3)
        obs = rb.obs(cfg, BUYER)
        toks = torch.full((16, cfg.channel.dialogue_len), cfg.channel.pad_id, dtype=torch.long)
        toks[:, 0] = 3
        toks[:, 1] = cfg.channel.end_id
        with torch.no_grad():
            h = net.encode(obs, toks, schema=phase_schema(cfg, BUYER, ph),
                           self_mask=ph.self_mask(cfg, BUYER))[:, -1]
            choice = net.choice_logits(h, obs)
            qty_lp = F.log_softmax(net.belief_qty_head(h), dim=-1)
        K = cfg.curriculum.n_candidates
        for i in range(16):
            cands = rb.meanings[i]
            expect = torch.stack([qty_lp[i, int(cands[k, 3])] for k in range(K)])
            diff = choice[i] - expect
            self.assertTrue(torch.allclose(diff, diff[0].expand_as(diff), atol=1e-5),
                            "on a quantity round the choice must be the quantity head "
                            "up to a per-round constant")
        # the pointer listener is still there as the control
        cfg.model.factored_choice = False
        torch.manual_seed(1)
        f, b = agents(cfg)
        with torch.no_grad():
            h = b[0].net.encode(obs, toks, schema=phase_schema(cfg, BUYER, ph),
                                self_mask=ph.self_mask(cfg, BUYER))[:, -1]
            self.assertEqual(tuple(b[0].net.choice_logits(h, obs).shape), (16, K))

    def test_every_rung_still_plays_with_gestures_on(self):
        from orchard.batched import TensorWorld
        cfg = cfg_small()
        cfg.channel.max_symbols = 6
        torch.manual_seed(7)
        f, b = agents(cfg)
        gen = torch.Generator().manual_seed(7)
        rw = ReferentialWorld(cfg, generator=gen)
        tw = TensorWorld(cfg, device="cpu", generator=gen)
        n = 48
        fi, bi = pairing(n)
        for phase in ladder(cfg):
            if phase.referential:
                scen = rw.sample(n, informer=phase.informer, mix=phase.mix)
            elif phase.mutual:
                scen = rw.sample_mutual(n)
            else:
                scen = tw.sample(n)
            batch, stats = run_and_update_gumbel(cfg, scen, f, b, fi, bi, phase=phase,
                                                 gesture_share=0.5)
            self.assertEqual(len(batch), n)
            L = cfg.channel.max_msg_len
            for turn in range(min(phase.n_turns, cfg.channel.n_turns)):
                role = phase.speaker_of_turn(turn)
                allowed = {fld for fld, _ in G.gesture_columns(cfg, phase, role)}
                for t in batch.tokens[:, turn * L].tolist():
                    m = G.gesture_meaning(cfg, t)
                    if m is not None:
                        self.assertIn(m[0], allowed,
                                      "%s: a %s gestured a field it cannot see"
                                      % (phase.name, "farmer" if role == FARMER else "buyer"))


class TestTheFactoredListenerCanLearn(unittest.TestCase):
    def test_a_fixed_compositional_code_is_learned_supervised(self):
        """The capacity check before blaming the architecture: told the answer, the
        factored listener learns a one-atom-per-quantity code on quantity rounds."""
        cfg = cfg_small()
        cfg.channel.max_symbols = 6
        torch.manual_seed(0)
        f, b = agents(cfg)
        net = b[0].net
        ph = phase_named(cfg, "name-quantity").with_informer(FARMER)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        schema = phase_schema(cfg, BUYER, ph)
        mask = ph.self_mask(cfg, BUYER)
        c = cfg.channel

        def rounds(n):
            rb = rw.sample(n, informer=FARMER, query=3)
            obs = rb.obs(cfg, BUYER)
            toks = torch.full((n, c.dialogue_len), c.pad_id, dtype=torch.long)
            toks[:, 0] = rb.true_meaning[:, 3]          # atom q names quantity q
            toks[:, 1] = c.end_id
            return rb, obs, toks

        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(60):                         # measured: 1.00 by step 40
            rb, obs, toks = rounds(64)
            h = net.encode(obs, toks, schema=schema, self_mask=mask)[:, -1]
            loss = F.cross_entropy(net.choice_logits(h, obs), rb.target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        rb, obs, toks = rounds(512)
        with torch.no_grad():
            h = net.encode(obs, toks, schema=schema, self_mask=mask)[:, -1]
            acc = float((net.choice_logits(h, obs).argmax(-1) == rb.target).float().mean())
            # and the head it went through now reads the quantity off the message
            qty = float((net.belief_qty_head(h).argmax(-1) == rb.true_meaning[:, 3]).float().mean())
        self.assertGreater(acc, 0.85, "the factored listener could not learn a clean code: %.3f" % acc)
        self.assertGreater(qty, 0.85, "the quantity head did not learn what the choice did: %.3f" % qty)


class TestGesturesAreMethodNotArchitecture(unittest.TestCase):
    def test_switching_gestures_off_changes_no_parameter_shape(self):
        on, off = cfg_small(), cfg_small()
        off.gesture.enabled = False
        a = make_agent(on, agent_id=0, role=FARMER, slot=0, generation=0, birth_episode=0,
                       lifespan=1)
        b = make_agent(off, agent_id=0, role=FARMER, slot=0, generation=0, birth_episode=0,
                       lifespan=1)
        self.assertEqual({k: tuple(v.shape) for k, v in a.net.state_dict().items()},
                         {k: tuple(v.shape) for k, v in b.net.state_dict().items()})
        self.assertIn("model.factored_choice", ARCH_KEYS)
        self.assertNotIn("gesture.enabled", ARCH_KEYS)

    def test_a_disabled_run_never_gestures_whatever_the_share(self):
        cfg = cfg_small()
        cfg.gesture.enabled = False
        cfg, f, b, ph, batch, stats = gestured_batch(cfg, share=1.0)
        self.assertFalse(any(G.is_gesture(cfg, t) for t in batch.tokens.flatten().tolist()))
        self.assertEqual(stats.gesture_share, 0.0)


if __name__ == "__main__":
    unittest.main()
