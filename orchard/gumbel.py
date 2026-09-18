"""Straight-through Gumbel-softmax channel -- spec 2.3's sanctioned alternative.

Why this exists
---------------
With pure REINFORCE the speaker's tokens are credited only through the partner's
eventual reward, and the scrambled-channel ablation showed the consequence
plainly: after 24k episodes, destroying every message in flight cost almost
nothing, because almost nothing was being communicated.  The variance of the
score-function estimator over a 3-4 token discrete action space, credited by a
single scalar at the end of the episode, is simply too high at this scale.

Straight-through Gumbel-softmax fixes the *estimator* without softening the
*channel*:

* In the forward pass the emitted token is an exact one-hot.  The partner
  receives one discrete symbol from a fixed vocabulary, exactly as before.  There
  is no extra bandwidth, which is the cheat spec 2.2 warns about -- a relaxed
  message would let real-valued information leak across.
* In the backward pass the hard sample is replaced by the soft Gumbel-softmax
  distribution, so the listener's loss can push directly on the speaker's logits.

The trade decision stays discrete and stays on REINFORCE.  There is no sensible
relaxation of "accept or walk away", and the speaker picks up gradient through
the *listener's* policy-gradient term -- which is where the useful signal lives:
"change what you said so that the decision that just worked becomes more likely".

The cost is that an episode must be built as a single graph spanning both
agents, so rollout and update are one function here rather than two.  Episodes
are short and the networks are tiny, so the graph is small.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .agents import Agent, dialogue_offset, own_dialogue_positions
from .config import Config
from .env import (BUYER, FARMER, Decision, Outcome, buyer_obs, farmer_obs,
                  resolve, speaker_of_turn)
from .rollout import BatchRollout, UpdateStats, anneal, group_by_agent
from .world import Scenario


def gumbel_tau(cfg: Config, frac_done: float) -> float:
    t = cfg.train
    return anneal(t.gumbel_tau, t.gumbel_tau_final, frac_done, t.gumbel_tau_anneal_frac)


def run_and_update_gumbel(cfg: Config, scenarios: Sequence[Scenario],
                          farmers: Sequence[Agent], buyers: Sequence[Agent],
                          f_idx: torch.Tensor, b_idx: torch.Tensor, *,
                          frac_done: float = 0.0, device: str = "cpu",
                          train: bool = True,
                          generator: Optional[torch.Generator] = None
                          ) -> tuple[BatchRollout, UpdateStats]:
    """Play a batch with a reparameterised message channel, then learn from it."""
    c = cfg.channel
    t = cfg.train
    B = len(scenarios)
    D = c.dialogue_len
    NT = c.n_token_ids
    tau = gumbel_tau(cfg, frac_done)

    f_obs = torch.tensor([farmer_obs(s, cfg) for s in scenarios], dtype=torch.long, device=device)
    b_obs = torch.tensor([buyer_obs(s, cfg) for s in scenarios], dtype=torch.long, device=device)
    obs_of = {FARMER: f_obs, BUYER: b_obs}
    idx_of = {FARMER: f_idx, BUYER: b_idx}
    pool_of = {FARMER: farmers, BUYER: buyers}

    pad_onehot = F.one_hot(torch.tensor(c.pad_id, device=device), NT).float()
    soft = pad_onehot.view(1, 1, NT).expand(B, D, NT).clone()
    tokens = torch.full((B, D), c.pad_id, dtype=torch.long, device=device)
    active = torch.zeros((B, D), dtype=torch.bool, device=device)
    token_entropy_terms: list[torch.Tensor] = []
    token_logp_terms: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {
        FARMER: [], BUYER: []}

    # ---- the conversation ------------------------------------------------
    for turn in range(c.n_turns):
        role = speaker_of_turn(turn)
        pool, idx, obs = pool_of[role], idx_of[role], obs_of[role]
        alive = torch.ones(B, dtype=torch.bool, device=device)

        for k in range(c.max_msg_len):
            if not bool(alive.any()):
                break
            p = turn * c.max_msg_len + k
            seq_pos = dialogue_offset(cfg) + p
            logits = torch.zeros((B, c.n_emittable), device=device)
            for a_i, ep in group_by_agent(idx, alive):
                h = pool[a_i].net.encode(obs[ep], soft[ep], upto=seq_pos)[:, -1]
                logits = logits.index_copy(0, ep, pool[a_i].net.token_head(h))

            y = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            tok = y.argmax(dim=-1)
            tok = torch.where(alive, tok, torch.full_like(tok, c.pad_id))
            y_full = torch.cat([y, torch.zeros((B, 1), device=device)], dim=-1)
            row = torch.where(alive.unsqueeze(-1), y_full, pad_onehot.view(1, NT))
            soft = soft.index_copy(1, torch.tensor([p], device=device), row.unsqueeze(1))

            tokens[:, p] = tok.detach()
            active[:, p] = alive
            lp = F.log_softmax(logits, dim=-1)
            token_entropy_terms.append(-(lp.exp() * lp).sum(-1) * alive.float())
            if t.gumbel_mix_reinforce > 0:
                chosen = lp.gather(-1, tok.clamp(max=c.n_emittable - 1).unsqueeze(-1)).squeeze(-1)
                token_logp_terms[role].append((chosen, alive.float()))
            alive = alive & (tok != c.eos_id)

    # ---- the decisions: still discrete, still REINFORCE ------------------
    dec_sampled: dict[int, torch.Tensor] = {}
    dec_logp: dict[int, torch.Tensor] = {}
    dec_ent: dict[int, torch.Tensor] = {}
    dec_value: dict[int, torch.Tensor] = {}
    for role in (FARMER, BUYER):
        pool, idx, obs = pool_of[role], idx_of[role], obs_of[role]
        out = torch.zeros((B, 4), dtype=torch.long, device=device)
        logp_sum = torch.zeros(B, device=device)
        ent_sum = torch.zeros(B, device=device)
        val = torch.zeros(B, device=device)
        for a_i, ep in group_by_agent(idx):
            net = pool[a_i].net
            h = net.encode(obs[ep], soft[ep])[:, -1]
            heads = (net.accept_head(h), net.variety_head(h),
                     net.decide_qty_head(h), net.decide_price_head(h))
            val = val.index_copy(0, ep, net.value_head(h).squeeze(-1))
            lps = torch.zeros(ep.shape[0], device=device)
            ents = torch.zeros(ep.shape[0], device=device)
            for col, lg in enumerate(heads):
                lp = F.log_softmax(lg, dim=-1)
                with torch.no_grad():
                    a = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
                out[ep, col] = a
                lps = lps + lp.gather(-1, a.unsqueeze(-1)).squeeze(-1)
                ents = ents + (-(lp.exp() * lp).sum(-1))
            logp_sum = logp_sum.index_copy(0, ep, lps)
            ent_sum = ent_sum.index_copy(0, ep, ents)
        dec_sampled[role] = out
        dec_logp[role] = logp_sum
        dec_ent[role] = ent_sum
        dec_value[role] = val

    # ---- resolve ---------------------------------------------------------
    content = (tokens < c.end_id)
    f_emitted = content[:, own_dialogue_positions(cfg, FARMER)].sum(dim=1)
    b_emitted = content[:, own_dialogue_positions(cfg, BUYER)].sum(dim=1)
    outcomes: list[Outcome] = []
    f_rew = torch.zeros(B, device=device)
    b_rew = torch.zeros(B, device=device)
    for i, sc in enumerate(scenarios):
        fd = Decision(*[int(v) for v in dec_sampled[FARMER][i]])
        bd = Decision(*[int(v) for v in dec_sampled[BUYER][i]])
        o = resolve(cfg, sc, fd, bd, int(f_emitted[i]), int(b_emitted[i]))
        outcomes.append(o)
        f_rew[i] = o.farmer_reward
        b_rew[i] = o.buyer_reward

    batch = BatchRollout(
        scenarios=list(scenarios), tokens=tokens, active=active,
        f_obs=f_obs, b_obs=b_obs, f_idx=f_idx, b_idx=b_idx,
        f_dec=dec_sampled[FARMER].detach(), b_dec=dec_sampled[BUYER].detach(),
        f_reward=f_rew.detach(), b_reward=b_rew.detach(), outcomes=outcomes,
        f_emitted=f_emitted, b_emitted=b_emitted)

    stats = UpdateStats()
    if not train:
        return batch, stats

    # ---- one joint objective ---------------------------------------------
    ent_tok_coef = anneal(t.entropy_coef, t.entropy_coef_final, frac_done,
                          t.entropy_anneal_frac)
    ent_dec_coef = anneal(t.decision_entropy_coef, t.decision_entropy_coef_final,
                          frac_done, t.entropy_anneal_frac)
    rew_of = {FARMER: f_rew, BUYER: b_rew}
    loss = torch.zeros((), device=device)
    value_loss_total = 0.0
    for role in (FARMER, BUYER):
        R = rew_of[role]
        if t.normalise_adv and R.numel() > 1:
            R = (R - R.mean()) / (R.std(unbiased=False) + 1e-6)
        adv = (R - dec_value[role]).detach()
        # The speaker's gradient arrives through this term: dec_logp depends on
        # the soft message, which depends on the other agent's token logits.
        loss = loss + (-(adv * dec_logp[role]).mean())
        vl = ((dec_value[role] - R) ** 2).mean()
        loss = loss + t.value_coef * vl
        loss = loss - ent_dec_coef * dec_ent[role].mean()
        value_loss_total += float(vl.detach())
        if t.gumbel_mix_reinforce > 0:
            for chosen, mask in token_logp_terms[role]:
                denom = mask.sum().clamp(min=1.0)
                loss = loss + t.gumbel_mix_reinforce * (-(adv * chosen * mask).sum() / denom)

    if token_entropy_terms:
        ent_stack = torch.stack(token_entropy_terms, dim=1)
        mask = active.float()[:, :ent_stack.shape[1]]
        tok_ent = (ent_stack * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss - ent_tok_coef * tok_ent
        stats.token_entropy = float(tok_ent.detach())

    seen = []
    for role in (FARMER, BUYER):
        for a_i, _ in group_by_agent(idx_of[role]):
            seen.append(pool_of[role][a_i])
    for a in seen:
        a.opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = 0.0
    for a in seen:
        gn += float(torch.nn.utils.clip_grad_norm_(a.net.parameters(), t.grad_clip))
        a.opt.step()

    stats.policy_loss = float(loss.detach())
    stats.value_loss = value_loss_total
    stats.decision_entropy = float(
        sum(dec_ent[r].mean().detach() for r in (FARMER, BUYER)) / 2)
    stats.n_agents = len(seen)
    stats.grad_norm = gn / max(1, len(seen))
    stats.n_actions = int(active.sum())
    return batch, stats
