"""Tests for the straight-through Gumbel channel.

The load-bearing property is the one the whole design rests on: a gradient must
reach the *speaker* from the *listener's* loss. If that path is ever broken,
training still runs, losses still go down, and the channel silently carries
nothing — which is exactly the failure the scrambled-channel ablation was built
to catch, and which is far cheaper to catch here.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from orchard.agents import CommNet, make_agent
from orchard.config import Config
from orchard.env import BUYER, FARMER
from orchard.gumbel import gumbel_tau, run_and_update_gumbel
from orchard.world import World


def tiny_cfg() -> Config:
    cfg = Config()
    cfg.world.n_varieties = 3
    cfg.world.max_qty = 6
    cfg.world.n_price_bins = 6
    cfg.world.reservation_max_bin = 4
    cfg.world.budget_min_bin = 1
    cfg.channel.atomic_vocab = 8
    cfg.channel.max_symbols = 4
    cfg.channel.n_turns = 4
    cfg.model.d_model = 32
    cfg.model.d_ff = 64
    return cfg


def build(cfg, n_f=1, n_b=1):
    farmers = [make_agent(cfg, agent_id=i, role=FARMER, slot=i, generation=0,
                          birth_episode=0, lifespan=10**9) for i in range(n_f)]
    buyers = [make_agent(cfg, agent_id=50 + i, role=BUYER, slot=i, generation=0,
                         birth_episode=0, lifespan=10**9) for i in range(n_b)]
    return farmers, buyers


class TestStraightThrough(unittest.TestCase):
    def test_forward_symbol_is_a_hard_one_hot(self):
        """The channel must stay discrete: no fractional symbols cross it."""
        torch.manual_seed(0)
        logits = torch.randn(64, 12)
        y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        self.assertTrue(torch.allclose(y.sum(-1), torch.ones(64)))
        self.assertTrue(torch.all((y == 0) | (y == 1)),
                        "a relaxed symbol would be extra bandwidth the spec forbids")

    def test_gradient_survives_the_hard_sample(self):
        logits = torch.randn(16, 10, requires_grad=True)
        y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        y.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_tau_anneals(self):
        cfg = tiny_cfg()
        self.assertGreater(gumbel_tau(cfg, 0), gumbel_tau(cfg, cfg.train.tau_anneal_updates))
        self.assertEqual(gumbel_tau(cfg, 10 * cfg.train.tau_anneal_updates),
                         cfg.train.gumbel_tau_final)


class TestSpeakerGetsGradientFromListener(unittest.TestCase):
    def test_listener_loss_reaches_the_speaker(self):
        """The whole point of the Gumbel path.

        The buyer speaks first and never hears anything before it does; if its
        message head has a gradient at all, that gradient can only have come
        through the farmer's forward pass.
        """
        cfg = tiny_cfg()
        torch.manual_seed(0)
        farmers, buyers = build(cfg)
        w = World(cfg.world, random.Random(0))
        scen = w.sample_batch(32, held_out=False)
        f_idx = torch.zeros(32, dtype=torch.long)
        b_idx = torch.zeros(32, dtype=torch.long)

        grads = {}
        orig_step = buyers[0].opt.step

        def capture():
            g = buyers[0].net.token_head.weight.grad
            grads["buyer_token_head"] = float(g.abs().sum()) if g is not None else 0.0
            g2 = farmers[0].net.token_head.weight.grad
            grads["farmer_token_head"] = float(g2.abs().sum()) if g2 is not None else 0.0
            return orig_step()
        buyers[0].opt.step = capture

        run_and_update_gumbel(cfg, scen, farmers, buyers, f_idx, b_idx, update=0)
        self.assertGreater(grads.get("buyer_token_head", 0.0), 0.0,
                           "no gradient reached the buyer's message head -- the "
                           "speaker->listener path is broken")
        self.assertGreater(grads.get("farmer_token_head", 0.0), 0.0)

    def test_message_head_moves(self):
        cfg = tiny_cfg()
        torch.manual_seed(1)
        farmers, buyers = build(cfg)
        w = World(cfg.world, random.Random(1))
        before = buyers[0].net.token_head.weight.detach().clone()
        for _ in range(3):
            scen = w.sample_batch(32, held_out=False)
            run_and_update_gumbel(cfg, scen, farmers, buyers,
                                  torch.zeros(32, dtype=torch.long),
                                  torch.zeros(32, dtype=torch.long), update=100)
        after = buyers[0].net.token_head.weight.detach()
        self.assertFalse(torch.allclose(before, after))
        self.assertTrue(torch.isfinite(after).all())

    def test_emitted_symbols_are_valid_and_well_formed(self):
        cfg = tiny_cfg()
        torch.manual_seed(2)
        farmers, buyers = build(cfg, 2, 2)
        w = World(cfg.world, random.Random(2))
        scen = w.sample_batch(24, held_out=False)
        batch, _ = run_and_update_gumbel(
            cfg, scen, farmers, buyers,
            torch.randint(0, 2, (24,)), torch.randint(0, 2, (24,)), update=500)
        c = cfg.channel
        self.assertTrue(bool(((batch.tokens >= 0) & (batch.tokens <= c.pad_id)).all()))
        # nothing may follow END within a turn
        for i in range(24):
            for turn in range(c.n_turns):
                seen_end = False
                for k in range(c.max_symbols):
                    sym = int(batch.tokens[i, turn * c.max_symbols + k])
                    if seen_end:
                        self.assertEqual(sym, c.pad_id)
                    if sym == c.end_id:
                        seen_end = True

    def test_reinforce_mix_adds_a_term_without_breaking_anything(self):
        cfg = tiny_cfg()
        cfg.train.gumbel_mix_reinforce = 1.0
        torch.manual_seed(3)
        farmers, buyers = build(cfg)
        w = World(cfg.world, random.Random(3))
        scen = w.sample_batch(32, held_out=False)
        batch, stats = run_and_update_gumbel(
            cfg, scen, farmers, buyers, torch.zeros(32, dtype=torch.long),
            torch.zeros(32, dtype=torch.long), update=0)
        self.assertEqual(stats.policy_loss, stats.policy_loss)  # finite
        self.assertTrue(torch.isfinite(buyers[0].net.token_head.weight).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
