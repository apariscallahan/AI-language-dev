"""Tests for the agent network and the rollout/update machinery.

The load-bearing test is ``test_prefix_and_full_pass_agree``: the whole training
loop's efficiency rests on the claim that a prefix forward during rollout and a
masked full forward during the update produce identical logits.  If causality
ever broke, training would silently optimise the wrong log-probabilities.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from orchard.agents import (CommNet, count_parameters, dialogue_offset, make_agent,
                            own_dialogue_positions, sequence_len)
from orchard.config import Config
from orchard.env import BUYER, FARMER
from orchard.rollout import read_positions_for, run_episodes
from orchard.world import World


def tiny_cfg() -> Config:
    cfg = Config()
    cfg.world.max_qty = 6
    cfg.world.n_varieties = 3
    cfg.world.n_price_bins = 6
    cfg.world.reservation_max_bin = 3
    cfg.world.budget_min_bin = 2
    cfg.channel.atomic_vocab = 8
    cfg.channel.max_symbols = 3
    cfg.channel.n_turns = 4
    cfg.model.d_model = 32
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.d_ff = 64
    cfg.train.batch_size = 16
    return cfg


def build(cfg, n_f=2, n_b=2):
    farmers = [make_agent(cfg, agent_id=i, role=FARMER, slot=i, generation=0,
                          birth_episode=0, lifespan=10**9) for i in range(n_f)]
    buyers = [make_agent(cfg, agent_id=100 + i, role=BUYER, slot=i, generation=0,
                         birth_episode=0, lifespan=10**9) for i in range(n_b)]
    return farmers, buyers


class TestNetwork(unittest.TestCase):
    def test_shapes_and_size(self):
        cfg = tiny_cfg()
        net = CommNet(cfg, FARMER)
        self.assertEqual(net.seq_len, sequence_len(cfg))
        obs = torch.zeros((5, net.n_obs), dtype=torch.long)
        toks = torch.full((5, cfg.channel.dialogue_len), cfg.channel.pad_id, dtype=torch.long)
        h = net.encode(obs, toks)
        self.assertEqual(tuple(h.shape), (5, net.seq_len, cfg.model.d_model))
        self.assertLess(count_parameters(net), 500_000)

    def test_two_agents_are_independently_initialised(self):
        cfg = tiny_cfg()
        a, b = CommNet(cfg, FARMER), CommNet(cfg, FARMER)
        self.assertFalse(torch.allclose(a.token_head.weight, b.token_head.weight),
                         "agents must not share or copy weights at birth")

    def test_prefix_and_full_pass_agree(self):
        """Causality check: prefix logits == full-sequence logits at the same index."""
        cfg = tiny_cfg()
        torch.manual_seed(0)
        net = CommNet(cfg, BUYER).eval()
        D = cfg.channel.dialogue_len
        from orchard.env import buyer_obs
        probe_world = World(cfg.world, random.Random(7))
        obs = torch.tensor([buyer_obs(probe_world.sample(), cfg) for _ in range(4)],
                           dtype=torch.long)
        toks = torch.randint(0, cfg.channel.atomic_vocab, (4, D))

        read_pos = read_positions_for(cfg, BUYER)
        with torch.no_grad():
            full_logits, _, _, _ = net.full_pass(obs, toks, read_pos)
            for j, p in enumerate(own_dialogue_positions(cfg, BUYER)):
                seq_pos = dialogue_offset(cfg) + p
                prefix_logits, _ = net.next_token_logits(obs, toks, seq_pos)
                self.assertTrue(
                    torch.allclose(prefix_logits, full_logits[:, j], atol=1e-5),
                    "prefix forward and full causal forward disagree at slot %d" % p)

    def test_future_tokens_cannot_influence_the_past(self):
        cfg = tiny_cfg()
        torch.manual_seed(1)
        net = CommNet(cfg, FARMER).eval()
        D = cfg.channel.dialogue_len
        obs = torch.zeros((2, net.n_obs), dtype=torch.long)
        a = torch.randint(0, cfg.channel.atomic_vocab, (2, D))
        b = a.clone()
        b[:, -1] = (b[:, -1] + 1) % cfg.channel.vocab_size
        with torch.no_grad():
            ha = net.encode(obs, a)
            hb = net.encode(obs, b)
        last = dialogue_offset(cfg) + D - 1
        self.assertTrue(torch.allclose(ha[:, :last], hb[:, :last], atol=1e-6))
        self.assertFalse(torch.allclose(ha[:, -1], hb[:, -1], atol=1e-6))


class TestRollout(unittest.TestCase):
    def test_rollout_structure(self):
        cfg = tiny_cfg()
        torch.manual_seed(2)
        farmers, buyers = build(cfg)
        w = World(cfg.world, random.Random(2))
        scen = w.sample_batch(12, held_out=False)
        f_idx = torch.randint(0, len(farmers), (12,))
        b_idx = torch.randint(0, len(buyers), (12,))
        batch = run_episodes(cfg, scen, farmers, buyers, f_idx, b_idx)

        self.assertEqual(tuple(batch.tokens.shape), (12, cfg.channel.dialogue_len))
        self.assertEqual(len(batch.outcomes), 12)
        # no content token may follow EOS within a turn
        L = cfg.channel.max_msg_len
        for i in range(12):
            for turn in range(cfg.channel.n_turns):
                seen_eos = False
                for k in range(L):
                    tok = int(batch.tokens[i, turn * L + k])
                    if seen_eos:
                        self.assertEqual(tok, cfg.channel.pad_id)
                    if tok == cfg.channel.eos_id:
                        seen_eos = True
        # active mask must line up with non-PAD slots
        self.assertTrue(bool(((batch.tokens != cfg.channel.pad_id) == batch.active).all()))

    def test_agents_only_ever_see_their_own_observation(self):
        """Instrument the nets and assert the farmer's obs never reaches the buyer."""
        cfg = tiny_cfg()
        torch.manual_seed(3)
        farmers, buyers = build(cfg, 1, 1)
        w = World(cfg.world, random.Random(3))
        scen = w.sample_batch(8, held_out=False)
        seen = {FARMER: [], BUYER: []}

        def patch(agent):
            orig = agent.net.embed
            role = agent.role

            def wrapper(obs, tokens, schema=None, self_mask=None, upto=None):
                seen[role].append(obs.clone())
                return orig(obs, tokens, schema, self_mask, upto)
            agent.net.embed = wrapper

        for a in farmers + buyers:
            patch(a)
        f_idx = torch.zeros(8, dtype=torch.long)
        b_idx = torch.zeros(8, dtype=torch.long)
        run_episodes(cfg, scen, farmers, buyers, f_idx, b_idx)

        from orchard.env import buyer_obs, farmer_obs
        expect_f = {farmer_obs(s, cfg) for s in scen}
        expect_b = {buyer_obs(s, cfg) for s in scen}
        got_f = {tuple(int(v) for v in row) for t in seen[FARMER] for row in t}
        got_b = {tuple(int(v) for v in row) for t in seen[BUYER] for row in t}
        self.assertTrue(got_f <= expect_f, "farmer network saw a tuple that is not its own")
        self.assertTrue(got_b <= expect_b, "buyer network saw a tuple that is not its own")

    def test_muted_control_really_silences_the_other_party(self):
        """The three channel controls must differ in exactly the intended way.

        Scrambling preserves an utterance's *shape* and replaces only its content,
        which matters because with an open vocabulary length is itself a channel.
        Muting removes both. Confusing the two would make the headline transfer
        number wrong in a way no other test would catch -- and did, until the
        scrambled baseline was seen sitting above the world's base rate.
        """
        cfg = tiny_cfg()
        torch.manual_seed(6)
        farmers, buyers = build(cfg, 1, 1)
        w = World(cfg.world, random.Random(6))
        scen = w.sample_batch(24, held_out=False)
        f_idx = torch.zeros(24, dtype=torch.long)
        b_idx = torch.zeros(24, dtype=torch.long)

        seen = {"farmer": [], "buyer": []}

        def patch(agent, label):
            orig = agent.net.embed

            def wrapper(obs, tokens, schema=None, self_mask=None, upto=None):
                seen[label].append(tokens.clone())
                return orig(obs, tokens, schema, self_mask, upto)
            agent.net.embed = wrapper

        patch(farmers[0], "farmer")
        patch(buyers[0], "buyer")
        run_episodes(cfg, scen, farmers, buyers, f_idx, b_idx, channel_mode="muted")

        c = cfg.channel
        # turn 0 belongs to the buyer, so those slots are what the farmer hears
        buyer_slots = list(range(0, c.max_symbols))
        atoms = set(range(c.atomic_vocab))
        for view in seen["farmer"]:
            for row in view:
                for j in buyer_slots:
                    self.assertNotIn(int(row[j]), atoms,
                                     "muted control leaked an atom to the listener")

    def test_scrambled_control_keeps_shape_and_drops_content(self):
        cfg = tiny_cfg()
        torch.manual_seed(7)
        farmers, buyers = build(cfg, 1, 1)
        w = World(cfg.world, random.Random(7))
        scen = w.sample_batch(24, held_out=False)
        f_idx = torch.zeros(24, dtype=torch.long)
        b_idx = torch.zeros(24, dtype=torch.long)

        heard = []
        orig = farmers[0].net.embed

        def wrapper(obs, tokens, schema=None, self_mask=None, upto=None):
            heard.append(tokens.clone())
            return orig(obs, tokens, schema, self_mask, upto)
        farmers[0].net.embed = wrapper
        batch = run_episodes(cfg, scen, farmers, buyers, f_idx, b_idx,
                             channel_mode="scrambled")

        c = cfg.channel
        said = batch.tokens[:, :c.max_symbols]
        view = heard[-1][:, :c.max_symbols]
        # same stopping point -- END and PAD land in the same places
        for i in range(said.shape[0]):
            for j in range(c.max_symbols):
                a, b = int(said[i, j]), int(view[i, j])
                if a in (c.end_id, c.pad_id) or b in (c.end_id, c.pad_id):
                    self.assertEqual(a, b, "scrambling changed the message shape")

    def test_unknown_channel_mode_is_rejected(self):
        cfg = tiny_cfg()
        farmers, buyers = build(cfg, 1, 1)
        w = World(cfg.world, random.Random(8))
        with self.assertRaises(ValueError):
            run_episodes(cfg, w.sample_batch(2, held_out=False), farmers, buyers,
                         torch.zeros(2, dtype=torch.long),
                         torch.zeros(2, dtype=torch.long), channel_mode="nonsense")

    def test_greedy_rollout_is_deterministic(self):
        cfg = tiny_cfg()
        torch.manual_seed(5)
        farmers, buyers = build(cfg)
        w = World(cfg.world, random.Random(5))
        scen = w.sample_batch(8, held_out=False)
        f_idx = torch.zeros(8, dtype=torch.long)
        b_idx = torch.zeros(8, dtype=torch.long)
        a = run_episodes(cfg, scen, farmers, buyers, f_idx, b_idx, greedy=True)
        b = run_episodes(cfg, scen, farmers, buyers, f_idx, b_idx, greedy=True)
        self.assertTrue(torch.equal(a.tokens, b.tokens))
        self.assertTrue(torch.equal(a.f_dec, b.f_dec))


if __name__ == "__main__":
    unittest.main(verbosity=2)
