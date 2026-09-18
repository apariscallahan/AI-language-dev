"""Agent brains: small, randomly-initialised causal transformers.

NO PRETRAINING ANYWHERE.  Every weight in this file is initialised by PyTorch's
default random init at the moment an agent is born.  There is no text corpus, no
tokenizer trained on language, no pretrained embedding table.  The "vocabulary"
is ``range(vocab_size)`` -- integer ids into a randomly-initialised lookup table
whose only source of meaning is the reinforcement signal from trading.

Sequence layout
---------------
Every agent, both roles, sees one fixed-length sequence::

    idx 0            BOS (+ role embedding)
    idx 1..4         its own four private observation fields, one per position
    idx 5            SEP
    idx 6..6+D-1     the dialogue, D = n_turns * max_msg_len, PAD where unspoken
    idx 6+D          DECIDE

The four observation positions carry a *field* embedding plus a *value*
embedding, so the network knows which number is which; the role embedding tells
it whether field 2 means "stock I hold" or "quantity I need".  The dialogue
positions carry a token embedding plus a self/other speaker embedding -- an agent
therefore always knows which words were its own.

The model is causal (masked self-attention), which makes two things identical:

* rollout, where we feed the prefix that exists so far and read the last
  position, and
* the update, where we feed the whole finished episode *once* and read the
  logits at every position the agent spoke from.

That equivalence is what makes training affordable: one forward per agent per
batch in the backward pass instead of one per emitted token.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .env import BUYER, FARMER, speaker_of_turn
from .world import (K_EMPTY, K_PRICE, K_QTY, K_QUALITY, K_VARIETY, n_obs_slots,
                    obs_schema)

# Sequence layout.  The number of observation slots is whatever the world's
# schema needs (a farm with several varieties has more to look at than a buyer
# with one shopping list), and the shorter role is padded, so both roles share
# one layout and one set of position indices.
SLOT_BOS, SLOT_SEP, SLOT_DIALOGUE, SLOT_DECIDE = 0, 1, 2, 3
N_FIXED_SLOT_TYPES = 4
N_SLOT_TYPES = N_FIXED_SLOT_TYPES + 5        # + one per field kind


def dialogue_offset(cfg: Config) -> int:
    """Index of the first dialogue slot: BOS + the role-padded obs slots + SEP."""
    return 1 + n_obs_slots(cfg.world) + 1


def sequence_len(cfg: Config) -> int:
    return dialogue_offset(cfg) + cfg.channel.dialogue_len + 1


def own_dialogue_positions(cfg: Config, role: int) -> list[int]:
    """Dialogue-buffer indices (0..D-1) at which ``role`` is the speaker."""
    L = cfg.channel.max_msg_len
    out: list[int] = []
    for turn in range(cfg.channel.n_turns):
        if speaker_of_turn(turn) == role:
            out.extend(range(turn * L, (turn + 1) * L))
    return out


def speaker_self_mask(cfg: Config, role: int) -> torch.Tensor:
    """(D,) bool: True where this agent is the speaker."""
    D = cfg.channel.dialogue_len
    m = torch.zeros(D, dtype=torch.bool)
    m[own_dialogue_positions(cfg, role)] = True
    return m


class CommNet(nn.Module):
    """Policy + value network for one agent.  Randomly initialised, always."""

    def __init__(self, cfg: Config, role: int):
        super().__init__()
        self.cfg = cfg
        self.role = role
        m, c, w = cfg.model, cfg.channel, cfg.world
        d = m.d_model
        self.d_model = d
        self.seq_len = sequence_len(cfg)

        self.schema = obs_schema(w, role)
        self.n_obs = len(self.schema)
        self.dialogue_offset = dialogue_offset(cfg)

        self.tok_emb = nn.Embedding(c.n_token_ids, d)
        self.variety_emb = nn.Embedding(w.n_varieties, d)
        self.qty_emb = nn.Embedding(w.max_qty + 1, d)
        self.quality_emb = nn.Embedding(w.n_quality, d)
        self.price_emb = nn.Embedding(w.n_price_bins, d)
        self.empty_emb = nn.Embedding(1, d)
        self.slot_emb = nn.Embedding(N_SLOT_TYPES, d)
        # One embedding per observation *position*, so "stock of GREEN" is a
        # different thing to look at from "stock of GOLD" even though both are
        # quantities.
        self.obs_pos_emb = nn.Embedding(self.n_obs, d)
        self.speaker_emb = nn.Embedding(2, d)     # 0 = me, 1 = the other party
        self.role_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(self.seq_len, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=m.n_heads, dim_feedforward=m.d_ff,
            dropout=m.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=m.n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        self.token_head = nn.Linear(d, c.n_emittable)
        self.accept_head = nn.Linear(d, 2)
        self.variety_head = nn.Linear(d, w.n_varieties)
        self.decide_qty_head = nn.Linear(d, w.max_qty + 1)
        self.decide_price_head = nn.Linear(d, w.n_price_bins)
        self.value_head = nn.Linear(d, 1)

        self.register_buffer("_self_mask", speaker_self_mask(cfg, role), persistent=False)
        causal = torch.triu(torch.full((self.seq_len, self.seq_len), float("-inf")), diagonal=1)
        self.register_buffer("_causal", causal, persistent=False)

        self.apply(self._init)

    @staticmethod
    def _init(mod: nn.Module) -> None:
        if isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    # ------------------------------------------------------------------
    def embed(self, obs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """obs: (B,4) long -> (B, seq_len, d).

        ``tokens`` is either (B,D) integer ids, or (B,D,n_token_ids) of
        per-slot weights.  The float form is what the straight-through Gumbel
        channel passes: the forward values are still exact one-hots, so the
        message that crosses the channel is genuinely discrete, but the lookup
        becomes a differentiable matrix product and a gradient can reach the
        speaker that produced it.
        """
        B = obs.shape[0]
        d = self.d_model
        dev = obs.device
        parts = []

        bos = self.slot_emb.weight[SLOT_BOS] + self.role_emb.weight[self.role]
        parts.append(bos.expand(B, 1, d))

        tables = {K_VARIETY: self.variety_emb, K_QTY: self.qty_emb,
                  K_QUALITY: self.quality_emb, K_PRICE: self.price_emb}
        cols = []
        for i, kind in enumerate(self.schema):
            if kind == K_EMPTY:
                vec = self.empty_emb.weight[0].expand(B, d)
            else:
                vec = tables[kind](obs[:, i])
            cols.append(vec
                        + self.slot_emb.weight[N_FIXED_SLOT_TYPES + kind]
                        + self.obs_pos_emb.weight[i])
        parts.append(torch.stack(cols, dim=1))

        parts.append(self.slot_emb.weight[SLOT_SEP].expand(B, 1, d))

        spk = torch.where(self._self_mask, 0, 1).to(dev)              # (D,)
        tok_vec = (self.tok_emb(tokens) if tokens.dtype == torch.long
                   else tokens @ self.tok_emb.weight)
        dial = (tok_vec
                + self.speaker_emb(spk).unsqueeze(0)
                + self.slot_emb.weight[SLOT_DIALOGUE])
        parts.append(dial)

        parts.append(self.slot_emb.weight[SLOT_DECIDE].expand(B, 1, d))

        x = torch.cat(parts, dim=1)
        return x + self.pos_emb.weight.unsqueeze(0)

    def encode(self, obs: torch.Tensor, tokens: torch.Tensor,
               upto: Optional[int] = None) -> torch.Tensor:
        """Hidden states for the prefix of length ``upto`` (default: whole sequence)."""
        x = self.embed(obs, tokens)
        n = self.seq_len if upto is None else upto
        x = x[:, :n]
        mask = self._causal[:n, :n]
        h = self.encoder(x, mask=mask)
        return self.norm(h)

    # ------------------------------------------------------------------
    def next_token_logits(self, obs: torch.Tensor, tokens: torch.Tensor,
                          seq_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits for the token that will occupy ``seq_pos``, plus that state's value."""
        h = self.encode(obs, tokens, upto=seq_pos)[:, -1]
        return self.token_head(h), self.value_head(h).squeeze(-1)

    def decision_logits(self, obs: torch.Tensor, tokens: torch.Tensor):
        h = self.encode(obs, tokens)[:, -1]
        return (self.accept_head(h), self.variety_head(h),
                self.decide_qty_head(h), self.decide_price_head(h),
                self.value_head(h).squeeze(-1))

    def full_pass(self, obs: torch.Tensor, tokens: torch.Tensor,
                  read_positions: torch.Tensor):
        """One causal forward over the finished episode.

        ``read_positions`` are sequence indices whose hidden state produced this
        agent's own tokens (i.e. ``DIALOGUE_OFFSET + p - 1`` for each own
        dialogue slot ``p``).  Returns token logits and values at those
        positions, plus the four decision logits and the value at DECIDE.
        """
        h = self.encode(obs, tokens)                        # (B, L, d)
        hr = h[:, read_positions]                           # (B, K, d)
        tok_logits = self.token_head(hr)                    # (B, K, V+1)
        tok_values = self.value_head(hr).squeeze(-1)        # (B, K)
        hd = h[:, -1]
        dec = (self.accept_head(hd), self.variety_head(hd),
               self.decide_qty_head(hd), self.decide_price_head(hd))
        dec_value = self.value_head(hd).squeeze(-1)
        return tok_logits, tok_values, dec, dec_value


@dataclass
class Agent:
    """A living agent: its brain, its optimiser, and its lifecycle bookkeeping."""
    agent_id: int
    role: int
    slot: int                     # index into the population list ("lineage slot")
    generation: int
    net: CommNet
    opt: torch.optim.Optimizer
    birth_episode: int
    lifespan: int
    age: int = 0                  # episodes this agent has participated in
    days_alive: int = 0
    # running tallies, reported in per-generation summaries
    n_success: int = 0
    n_episodes: int = 0
    reward_sum: float = 0.0
    apples_traded: int = 0
    value_traded: float = 0.0
    profit: float = 0.0
    bottleneck_info: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "%s%d/g%d" % ("F" if self.role == FARMER else "B", self.slot, self.generation)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_episodes if self.n_episodes else 0.0

    def is_expired(self) -> bool:
        return self.age >= self.lifespan


def make_agent(cfg: Config, *, agent_id: int, role: int, slot: int, generation: int,
               birth_episode: int, lifespan: int, device: str = "cpu") -> Agent:
    net = CommNet(cfg, role).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)
    return Agent(agent_id=agent_id, role=role, slot=slot, generation=generation,
                 net=net, opt=opt, birth_episode=birth_episode, lifespan=lifespan)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
