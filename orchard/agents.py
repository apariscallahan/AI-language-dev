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
    return 1 + n_obs_slots(cfg.world, cfg) + 1


def sequence_len(cfg: Config) -> int:
    return dialogue_offset(cfg) + cfg.channel.dialogue_len + 1


def own_dialogue_positions(cfg: Config, role: int) -> list[int]:
    """Dialogue-buffer indices (0..D-1) at which ``role`` speaks *in the trading task*.

    This is the buyer-opens schedule only. Anything that can run in a lineup
    rung must ask the phase instead (:meth:`orchard.curriculum.Phase.own_positions`):
    there the farmer describes first, and using this schedule there silently
    swaps whose words are whose.
    """
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

        self.schema = obs_schema(w, role, cfg)
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
        # What this agent thinks the OTHER party's private situation is.  Read at
        # the same position as the decision, from the same state, but scored
        # against the other's hidden facts rather than against the deal -- this is
        # what makes "did you understand me" a thing either side can be paid for.
        self.belief_variety_head = nn.Linear(d, w.n_varieties)
        self.belief_qty_head = nn.Linear(d, w.max_qty + 1)
        self.belief_quality_head = nn.Linear(d, w.n_quality)
        self.belief_price_head = nn.Linear(d, w.n_price_bins)
        # Which candidate in the lineup (referential phase only).  This is a
        # pointer rather than a flat classifier: it scores the hidden state *at
        # each candidate's own slots*, so "compare the message against this
        # candidate" is something the attention can express directly instead of a
        # relational trick the network has to discover from nothing.  Present in
        # every phase so the architecture -- and the carried weights -- never
        # change at a curriculum boundary.
        self.choice_proj = nn.Linear(d, d)
        # Both sides of the match are normalised before the dot product.  Without
        # this the candidate side is a sum of three freshly-initialised embeddings
        # (norm ~0.24) against a query of norm ~6.8, which put the choice logits at
        # std 0.03 where every other head sits near 1.0 -- a policy so close to
        # uniform that the gradient could not move it, and the lineup game sat
        # exactly at chance no matter how long it ran.
        self.choice_ln_cand = nn.LayerNorm(d)
        self.choice_ln_query = nn.LayerNorm(d)
        self.n_candidates = max(2, cfg.curriculum.n_candidates)
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
    def embed(self, obs: torch.Tensor, tokens: torch.Tensor,
              schema: "list[int] | None" = None,
              self_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """obs: (B,4) long -> (B, seq_len, d).

        ``tokens`` is either (B,D) integer ids, or (B,D,n_token_ids) of
        per-slot weights.  The float form is what the straight-through Gumbel
        channel passes: the forward values are still exact one-hots, so the
        message that crosses the channel is genuinely discrete, but the lookup
        becomes a differentiable matrix product and a gradient can reach the
        speaker that produced it.

        ``self_mask`` (D,) marks the dialogue slots this agent produced. It
        defaults to the trading schedule; a phase with a different speaking order
        passes its own, so an agent's own words are always embedded as its own.
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
        for i, kind in enumerate(schema if schema is not None else self.schema):
            if kind == K_EMPTY:
                vec = self.empty_emb.weight[0].expand(B, d)
            else:
                vec = tables[kind](obs[:, i])
            cols.append(vec
                        + self.slot_emb.weight[N_FIXED_SLOT_TYPES + kind]
                        + self.obs_pos_emb.weight[i])
        parts.append(torch.stack(cols, dim=1))

        parts.append(self.slot_emb.weight[SLOT_SEP].expand(B, 1, d))

        mine = self._self_mask if self_mask is None else self_mask
        spk = torch.where(mine.to(dev), 0, 1)                            # (D,)
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
               upto: Optional[int] = None,
               schema: "list[int] | None" = None,
               self_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Hidden states for the prefix of length ``upto`` (default: whole sequence)."""
        x = self.embed(obs, tokens, schema, self_mask)
        n = self.seq_len if upto is None else upto
        x = x[:, :n]
        mask = self._causal[:n, :n]
        # bf16 autocast on the transformer layers only, and only on CUDA: that
        # is where the arithmetic is, and keeping embeddings, heads and losses in
        # fp32 means nothing downstream has to know. (The flag used to be set by
        # the GPU presets and read by nothing.)
        amp = self.cfg.train.amp and x.is_cuda

        def run(t):
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return self.encoder(t, mask=mask).float()
            return self.encoder(t, mask=mask)
        if self.cfg.train.grad_checkpoint and self.training and x.requires_grad:
            from torch.utils.checkpoint import checkpoint
            h = checkpoint(run, x, use_reentrant=False)
        else:
            h = run(x)
        return self.norm(h)

    # ------------------------------------------------------------------
    def next_token_logits(self, obs: torch.Tensor, tokens: torch.Tensor,
                          seq_pos: int, schema=None, self_mask=None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits for the token that will occupy ``seq_pos``, plus that state's value."""
        h = self.encode(obs, tokens, upto=seq_pos, schema=schema,
                        self_mask=self_mask)[:, -1]
        return self.token_head(h), self.value_head(h).squeeze(-1)

    def decision_logits(self, obs: torch.Tensor, tokens: torch.Tensor, schema=None,
                        self_mask=None):
        """Every discrete head, then the value.  Order matches curriculum.py's
        head indices, so callers can slice the first N_HEADS and trust it."""
        h = self.encode(obs, tokens, schema=schema, self_mask=self_mask)[:, -1]
        return self.all_heads(h, obs) + (self.value_head(h).squeeze(-1),)

    def decision_heads(self, h: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.accept_head(h), self.variety_head(h),
                self.decide_qty_head(h), self.decide_price_head(h))

    def belief_heads(self, h: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.belief_variety_head(h), self.belief_qty_head(h),
                self.belief_quality_head(h), self.belief_price_head(h))

    def candidate_embeddings(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, K, d) -- each lineup candidate embedded from its own three fields.

        Built from the raw observation rather than from hidden states, because the
        encoder is causal: a candidate sits early in the sequence and cannot
        attend forward to the message. Scoring it against a hidden state taken at
        a candidate slot would therefore be scoring it against something that has
        not heard anything, which is exactly how the first version of this head
        managed to be entirely independent of what was said.
        """
        K = self.n_candidates
        vecs = []
        for k in range(K):
            i = 3 * k
            if i + 2 >= obs.shape[1]:
                vecs.append(torch.zeros_like(vecs[0]) if vecs else
                            self.empty_emb.weight[0].expand(obs.shape[0], self.d_model))
                continue
            # Outside the lineup phase these slots hold trading fields whose
            # ranges do not match these tables, and the head's output is unused.
            # Clamping keeps the lookup legal rather than making every call site
            # have to know which phase it is in.
            vecs.append(
                self.variety_emb(obs[:, i].clamp(0, self.variety_emb.num_embeddings - 1))
                + self.qty_emb(obs[:, i + 1].clamp(0, self.qty_emb.num_embeddings - 1))
                + self.quality_emb(
                    obs[:, i + 2].clamp(0, self.quality_emb.num_embeddings - 1)))
        return torch.stack(vecs, dim=1)

    def choice_logits(self, h_last: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """(B, K) -- how well each candidate matches what was just heard.

        A dot product between a projection of the final hidden state (which has
        seen the whole message) and each candidate's embedding: the standard
        listener for a signalling game, and the one structure that makes
        "does this description fit this candidate" directly expressible.
        """
        cand = self.choice_ln_cand(self.candidate_embeddings(obs))   # (B, K, d)
        q = self.choice_ln_query(self.choice_proj(h_last)).unsqueeze(-1)
        return torch.bmm(cand, q).squeeze(-1) / math.sqrt(self.d_model)

    def all_heads(self, h: torch.Tensor, obs: Optional[torch.Tensor] = None
                  ) -> tuple[torch.Tensor, ...]:
        """Every discrete output, in the fixed order curriculum.py indexes."""
        choice = (self.choice_logits(h, obs) if obs is not None
                  else h.new_zeros((h.shape[0], self.n_candidates)))
        return self.decision_heads(h) + self.belief_heads(h) + (choice,)

    def full_pass(self, obs: torch.Tensor, tokens: torch.Tensor,
                  read_positions: torch.Tensor, schema=None, self_mask=None):
        """One causal forward over the finished episode.

        ``read_positions`` are sequence indices whose hidden state produced this
        agent's own tokens (i.e. ``DIALOGUE_OFFSET + p - 1`` for each own
        dialogue slot ``p``).  Returns token logits and values at those
        positions, plus the four decision logits and the value at DECIDE.
        """
        h = self.encode(obs, tokens, schema=schema, self_mask=self_mask)   # (B, L, d)
        hr = h[:, read_positions]                           # (B, K, d)
        tok_logits = self.token_head(hr)                    # (B, K, V+1)
        tok_values = self.value_head(hr).squeeze(-1)        # (B, K)
        hd = h[:, -1]
        dec = self.all_heads(hd, obs)
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
    if str(device) == "auto":                      # resolve the config sentinel
        from .hardware import resolve_device
        device = resolve_device("auto")
    net = CommNet(cfg, role).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)
    return Agent(agent_id=agent_id, role=role, slot=slot, generation=generation,
                 net=net, opt=opt, birth_episode=birth_episode, lifespan=lifespan)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
