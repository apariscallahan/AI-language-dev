"""Batched episode rollout and the policy-gradient update.

Training algorithm: **REINFORCE with a learned value baseline** (spec 2.3's first
recommendation).  It was chosen over Gumbel-softmax because every action in this
environment is discrete -- message tokens *and* the four-part trade decision --
and because REINFORCE keeps the channel genuinely discrete during training.  A
relaxed channel would let gradients carry information the tokens themselves do
not, which is precisely the infinite-bandwidth cheat spec 2.2 warns about.

Variance reduction used here, all standard:
  * a per-state learned baseline (the value head), giving A = R - V(s);
  * batch standardisation of episode returns;
  * an entropy bonus on both the token and the decision policies, annealed, so
    the vocabulary does not collapse before it means anything.

Returns are undiscounted and episodic: the whole episode's reward is credited to
every action the agent took in it, which is what spec 2.3 asks for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from .agents import Agent, dialogue_offset, own_dialogue_positions
from .config import Config
from .env import (BUYER, FARMER, Beliefs, Decision, Outcome, Transcript,
                  buyer_obs, farmer_obs, length_cost, resolve, speaker_of_turn)
from .world import Scenario


# --------------------------------------------------------------------------
@dataclass
class BatchRollout:
    """One batch of finished negotiations.

    On the vectorised path the outcome lives in ``res`` as tensors and the
    per-episode :class:`~orchard.env.Outcome` objects are built only for the
    episodes something actually asks for -- the ledger's stride, a printed
    transcript.  Materialising all of them was costing one dataclass construction
    per episode, which on a GPU is the whole batch time.
    """
    scenarios: list[Scenario]
    tokens: torch.Tensor                 # (B, D) emitted dialogue, PAD-filled
    active: torch.Tensor                 # (B, D) bool: slot was a real sampled action
    f_obs: torch.Tensor                  # (B, 4)
    b_obs: torch.Tensor                  # (B, 4)
    f_idx: torch.Tensor                  # (B,) index into the farmer agent list
    b_idx: torch.Tensor                  # (B,) index into the buyer agent list
    f_dec: torch.Tensor                  # (B, N_HEADS) accept, variety, qty, price,
    b_dec: torch.Tensor                  # the belief fields, the lineup choice, colour
    f_reward: torch.Tensor               # (B,)
    b_reward: torch.Tensor               # (B,)
    outcomes: list[Outcome] = field(default_factory=list)
    f_emitted: torch.Tensor | None = None
    b_emitted: torch.Tensor | None = None
    f_cost: torch.Tensor | None = None       # what each paid for its length
    b_cost: torch.Tensor | None = None
    # vectorised path only
    res: dict[str, torch.Tensor] | None = None
    sb: Any = None
    n: int = 0

    def __len__(self) -> int:
        return self.n or len(self.scenarios)

    # ---- tensor views, so summaries never need the dataclasses ----------
    def _t(self, key: str) -> torch.Tensor:
        if self.res is not None:
            return self.res[key]
        raise AttributeError("no tensor outcome on this rollout")

    @property
    def vectorised(self) -> bool:
        return self.res is not None

    @property
    def success_t(self) -> torch.Tensor:
        if self.res is not None:
            return self.res["success"]
        return torch.tensor([o.success for o in self.outcomes])

    @property
    def comprehended_t(self) -> torch.Tensor:
        if self.res is not None:
            return self.res["comprehended"]
        return torch.tensor([o.comprehended for o in self.outcomes])

    @property
    def judged_t(self) -> torch.Tensor:
        if self.res is not None:
            return self.res["both_judged"]
        return torch.tensor([o.both_judged_viability for o in self.outcomes])

    @property
    def farmer_decode_t(self) -> torch.Tensor:
        if self.res is not None:
            return self.res["farmer_decode"]
        return torch.tensor([o.farmer_decode for o in self.outcomes])

    @property
    def buyer_decode_t(self) -> torch.Tensor:
        if self.res is not None:
            return self.res["buyer_decode"]
        return torch.tensor([o.buyer_decode for o in self.outcomes])

    @property
    def viable_t(self) -> torch.Tensor:
        if self.sb is not None:
            return self.sb.viable
        return torch.tensor([s.viable for s in self.scenarios])

    def scenario(self, i: int) -> Scenario:
        if self.scenarios:
            return self.scenarios[i]
        return self.sb.scenario(i)

    def outcome(self, i: int) -> Outcome:
        """Build one Outcome on demand (ledger rows, printed transcripts)."""
        if self.outcomes:
            return self.outcomes[i]
        from .env import resolve
        fd, fb = split_decision(self.f_dec[i])
        bd, bb = split_decision(self.b_dec[i])
        return resolve(self.cfg_ref, self.sb.scenario(i), fd, bd,
                       float(self.f_cost[i]), float(self.b_cost[i]),
                       f_beliefs=fb, b_beliefs=bb)

    cfg_ref: Any = None
    # The curriculum phase this batch was played under (with its informer, for a
    # swap rung). Whose words are whose depends on it.
    phase: Any = None

    def own_positions(self, role: int) -> list[int]:
        """Dialogue slots ``role`` produced in this batch."""
        if self.phase is not None:
            return self.phase.own_positions(self.cfg_ref, role)
        return own_dialogue_positions(self.cfg_ref, role)

    def transcript(self, i: int) -> Transcript:
        return Transcript(tokens=[int(t) for t in self.tokens[i]],
                          scenario=self.scenario(i),
                          farmer_decision=split_decision(self.f_dec[i])[0],
                          buyer_decision=split_decision(self.b_dec[i])[0],
                          farmer_beliefs=split_decision(self.f_dec[i])[1],
                          buyer_beliefs=split_decision(self.b_dec[i])[1],
                          outcome=(self.outcomes[i] if self.outcomes
                                   else (self.outcome(i) if self.res is not None else None)))

    def obs_for_role(self, role: int) -> torch.Tensor:
        return self.f_obs if role == FARMER else self.b_obs

    def idx_for_role(self, role: int) -> torch.Tensor:
        return self.f_idx if role == FARMER else self.b_idx

    def reward_for_role(self, role: int) -> torch.Tensor:
        return self.f_reward if role == FARMER else self.b_reward

    def dec_for_role(self, role: int) -> torch.Tensor:
        return self.f_dec if role == FARMER else self.b_dec


N_OUTPUTS = 8          # accept, variety, qty, price | belief x4


def n_outputs(cfg) -> int:
    """Sampled discrete outputs per agent: 4 for the deal, 4 more for the belief."""
    return 8 if cfg.reward.belief_heads else 4


def split_decision(row) -> tuple[Decision, Optional[Beliefs]]:
    """Unpack one agent's discrete outputs into a deal and (maybe) a belief."""
    from .curriculum import H_BELIEF_COLOR
    v = [int(x) for x in row]
    if len(v) < 8:
        return Decision(*v[:4]), None
    colour = v[H_BELIEF_COLOR] if len(v) > H_BELIEF_COLOR else 0
    return Decision(*v[:4]), Beliefs(*v[4:8], color=colour)


def belief_columns(dec: "torch.Tensor") -> "torch.Tensor":
    """(B, 5) the belief heads in the order decode_hits reads them, colour last."""
    from .curriculum import H_BELIEF, H_BELIEF_COLOR
    cols = list(H_BELIEF) + [H_BELIEF_COLOR]
    return dec[:, cols]


_GROUP_CACHE: dict[tuple, list[tuple[int, torch.Tensor]]] = {}


def group_by_agent_static(n: int, n_agents: int, stride_offset: int,
                          device: torch.device) -> list[tuple[int, torch.Tensor]]:
    """Index sets for a stride pairing, computed once and reused.

    With :meth:`orchard.population.Population.pair` handing episode i to agent
    ``i % n_agents``, every agent owns a fixed stride slice.  That makes the
    groups constants rather than something to be derived from a tensor each
    step -- which matters because deriving them reads the tensor on the host and
    stalls the device.
    """
    key = (n, n_agents, stride_offset, str(device))
    hit = _GROUP_CACHE.get(key)
    if hit is None:
        hit = [(a, torch.arange(a, n, n_agents, device=device))
               for a in range(n_agents)]
        hit = [(a, ep) for a, ep in hit if ep.numel() > 0]
        _GROUP_CACHE[key] = hit
    return hit


def group_by_agent(idx: torch.Tensor, mask: Optional[torch.Tensor] = None
                   ) -> list[tuple[int, torch.Tensor]]:
    """[(agent_list_index, episode_indices)] -- lets each agent run as one minibatch."""
    if mask is not None:
        valid = mask.nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            return []
        sub = idx[valid]
    else:
        valid = torch.arange(idx.shape[0], device=idx.device)
        sub = idx
    out = []
    for a in torch.unique(sub).tolist():
        out.append((int(a), valid[sub == a]))
    return out


# --------------------------------------------------------------------------
@torch.no_grad()
def run_episodes(cfg: Config, scenarios: Sequence[Scenario],
                 farmers: Sequence[Agent], buyers: Sequence[Agent],
                 f_idx: torch.Tensor, b_idx: torch.Tensor,
                 *, device: str = "cpu", greedy: bool = False,
                 channel_mode: str = "intact", phase=None,
                 generator: Optional[torch.Generator] = None) -> BatchRollout:
    """Play a batch of negotiations to completion.

    No gradients are taken here -- only the sampled discrete actions are kept.
    The update pass recomputes log-probabilities in a single causal forward per
    agent, which is exact (the model is causal, so a prefix forward and a masked
    full forward give identical logits) and far cheaper than retaining the graph.

    ``channel_mode`` selects the causal control used by
    :func:`orchard.metrics.channel_ablation`.  In every case each agent still says
    exactly what it would have said; only what *reaches the other party* changes.

      ``"intact"``     nothing is altered.
      ``"scrambled"``  content is replaced by uniform random atoms, but the
                       message keeps its shape -- same length, same stopping
                       point.  Isolates what the *symbols* carry.
      ``"muted"``      the other party is heard as having said nothing at all.
                       Removes length as well as content, and is therefore the
                       honest baseline for "how much is the channel worth".

    The distinction matters because with an open vocabulary utterance length is
    itself a usable channel, so a scrambled message is not a silent one.
    """
    if channel_mode not in ("intact", "scrambled", "muted"):
        raise ValueError("unknown channel_mode %r" % (channel_mode,))
    from .batched import ScenarioBatch, resolve_batch
    from .curriculum import (H_BELIEF, H_CHOICE, N_HEADS, MutualBatch,
                             ReferentialBatch, ladder, phase_schema,
                             resolve_referential, resolve_reports)
    c = cfg.channel
    if phase is None:
        phase = ladder(cfg)[-1]
    referential = isinstance(scenarios, ReferentialBatch)
    mutual = isinstance(scenarios, MutualBatch)
    tensor_in = referential or mutual or isinstance(scenarios, ScenarioBatch)
    if referential and scenarios.informer != phase.informer:
        phase = phase.with_informer(scenarios.informer)
    schema_of = {FARMER: phase_schema(cfg, FARMER, phase),
                 BUYER: phase_schema(cfg, BUYER, phase)}
    mask_of = {FARMER: phase.self_mask(cfg, FARMER, device),
               BUYER: phase.self_mask(cfg, BUYER, device)}
    B = len(scenarios)
    D = c.dialogue_len

    if tensor_in:
        f_obs = scenarios.obs(cfg, FARMER)
        b_obs = scenarios.obs(cfg, BUYER)
    else:
        f_obs = torch.tensor([farmer_obs(s, cfg) for s in scenarios],
                             dtype=torch.long, device=device)
        b_obs = torch.tensor([buyer_obs(s, cfg) for s in scenarios],
                             dtype=torch.long, device=device)
    tokens = torch.full((B, D), c.pad_id, dtype=torch.long, device=device)
    active = torch.zeros((B, D), dtype=torch.bool, device=device)
    # Per-role views of the dialogue.  One shared object unless a control is on.
    altered = channel_mode != "intact"
    views = ({FARMER: tokens.clone(), BUYER: tokens.clone()} if altered
             else {FARMER: tokens, BUYER: tokens})

    # who plays which episode, worked out once (reading the index is a device sync)
    groups_of = {FARMER: group_by_agent(f_idx), BUYER: group_by_agent(b_idx)}
    for turn in range(min(phase.n_turns, c.n_turns)):
        role = phase.speaker_of_turn(turn)
        other = BUYER if role == FARMER else FARMER
        pool = farmers if role == FARMER else buyers
        idx = f_idx if role == FARMER else b_idx
        obs = f_obs if role == FARMER else b_obs
        alive = torch.ones(B, dtype=torch.bool, device=device)

        for k in range(c.max_msg_len):
            if not bool(alive.any()):
                break
            p = turn * c.max_msg_len + k
            seq_pos = dialogue_offset(cfg) + p
            logits = torch.zeros((B, c.n_emittable), device=device)
            # finished utterances are computed too and masked to PAD below:
            # cheaper on a GPU than finding the live ones at every step
            for a_i, ep in groups_of[role]:
                lg, _ = pool[a_i].net.next_token_logits(
                    obs[ep], views[role][ep], seq_pos, schema=schema_of[role],
                    self_mask=mask_of[role])
                logits[ep] = lg
            from .env import MASKED, grammar_allowed
            allowed = grammar_allowed(cfg, tokens[:, p - 1] if k > 0 else tokens[:, p], k)
            logits = logits.masked_fill(~allowed, MASKED)
            if greedy:
                tok = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            tok = torch.where(alive, tok, torch.full_like(tok, c.pad_id))
            tokens[:, p] = tok
            active[:, p] = alive
            if altered:
                views[role][:, p] = tok            # the speaker hears itself correctly
                if channel_mode == "scrambled":
                    noise = torch.randint(0, c.atomic_vocab, (B,), device=device,
                                          generator=generator)
                    # keep the shape (where it stopped), destroy the content
                    views[other][:, p] = torch.where(
                        tok == c.pad_id, torch.full_like(tok, c.pad_id),
                        torch.where(tok == c.end_id, tok, noise))
                else:                              # muted: heard as immediate silence
                    views[other][:, p] = torch.full_like(
                        tok, c.end_id if k == 0 else c.pad_id)
            alive = alive & (tok != c.eos_id)

    decs: dict[int, torch.Tensor] = {}
    for role in (FARMER, BUYER):
        pool = farmers if role == FARMER else buyers
        idx = f_idx if role == FARMER else b_idx
        obs = f_obs if role == FARMER else b_obs
        out = torch.zeros((B, N_HEADS), dtype=torch.long, device=device)
        for a_i, ep in groups_of[role]:
            heads = pool[a_i].net.decision_logits(
                obs[ep], views[role][ep], schema=schema_of[role],
                self_mask=mask_of[role])[:N_HEADS]
            for col, lg in enumerate(heads):
                if greedy:
                    out[ep, col] = lg.argmax(dim=-1)
                else:
                    out[ep, col] = torch.multinomial(
                        F.softmax(lg, dim=-1), 1, generator=generator).squeeze(-1)
        decs[role] = out

    # Addendum 2.1: atoms, hyphens and spaces are all charged for; ending is free,
    # because brevity should not be taxed.
    content = (tokens < c.end_id)
    zeros = torch.zeros(B, dtype=torch.long, device=device)
    f_pos = phase.own_positions(cfg, FARMER)
    b_pos = phase.own_positions(cfg, BUYER)
    f_emitted = content[:, f_pos].sum(dim=1) if f_pos else zeros
    b_emitted = content[:, b_pos].sum(dim=1) if b_pos else zeros
    f_len = length_cost(cfg, tokens, f_pos)
    b_len = length_cost(cfg, tokens, b_pos)

    outcomes: list[Outcome] = []
    res = None
    if referential:
        res = resolve_referential(cfg, scenarios, decs[phase.guesser][:, H_CHOICE],
                                  f_len, b_len)
        f_rew, b_rew = res["farmer_reward"], res["buyer_reward"]
    elif phase.reporting and tensor_in:
        res = resolve_reports(cfg, phase, scenarios, decs, f_len, b_len)
        f_rew, b_rew = res["farmer_reward"], res["buyer_reward"]
    elif tensor_in:
        use_bel = cfg.reward.belief_heads
        res = resolve_batch(cfg, scenarios, decs[FARMER][:, :4], decs[BUYER][:, :4],
                            f_len, b_len,
                            f_bel=belief_columns(decs[FARMER]) if use_bel else None,
                            b_bel=belief_columns(decs[BUYER]) if use_bel else None)
        f_rew, b_rew = res["farmer_reward"], res["buyer_reward"]
    else:
        f_rew = torch.zeros(B)
        b_rew = torch.zeros(B)
        for i, sc in enumerate(scenarios):
            fd, fb = split_decision(decs[FARMER][i])
            bd, bb = split_decision(decs[BUYER][i])
            o = resolve(cfg, sc, fd, bd, float(f_len[i]), float(b_len[i]),
                        f_beliefs=fb, b_beliefs=bb)
            outcomes.append(o)
            f_rew[i] = o.farmer_reward
            b_rew[i] = o.buyer_reward
        f_rew, b_rew = f_rew.to(device), b_rew.to(device)

    return BatchRollout(
        scenarios=[] if tensor_in else list(scenarios), tokens=tokens, active=active,
        f_obs=f_obs, b_obs=b_obs, f_idx=f_idx, b_idx=b_idx,
        f_dec=decs[FARMER], b_dec=decs[BUYER],
        f_reward=f_rew, b_reward=b_rew,
        outcomes=outcomes, f_emitted=f_emitted, b_emitted=b_emitted,
        f_cost=f_len, b_cost=b_len,
        res=res, sb=scenarios if tensor_in else None, n=B, cfg_ref=cfg, phase=phase)


# --------------------------------------------------------------------------
def read_positions_for(cfg: Config, role: int, device: str = "cpu") -> torch.Tensor:
    """Sequence indices whose hidden state emits this role's own dialogue slots."""
    return torch.tensor([dialogue_offset(cfg) + p - 1 for p in own_dialogue_positions(cfg, role)],
                        dtype=torch.long, device=device)


def anneal(start: float, end: float, done: float, over: float) -> float:
    """Linear from ``start`` to ``end`` over the first ``over`` units (updates)."""
    if over <= 0:
        return end
    t = min(1.0, max(0.0, done / over))
    return start + (end - start) * t


@dataclass
class UpdateStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    token_entropy: float = 0.0
    decision_entropy: float = 0.0
    grad_norm: float = 0.0
    n_agents: int = 0
    n_actions: int = 0
