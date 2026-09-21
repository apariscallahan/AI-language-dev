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

from .agents import Agent, dialogue_offset
from .config import Config
from .env import (BUYER, FARMER, MASKED, Beliefs, Decision, Outcome, buyer_obs,
                  farmer_obs, grammar_allowed, length_cost, resolve,
                  speaker_of_turn)
from .batched import ScenarioBatch, resolve_batch
from .curriculum import (H_BELIEF, H_CHOICE, N_HEADS, MutualBatch, Phase,
                         ReferentialBatch, hindsight_applies, hindsight_targets, ladder,
                         phase_schema, resolve_mutual, resolve_request,
                         resolve_referential)
from .rollout import (BatchRollout, UpdateStats, anneal, belief_columns,
                      group_by_agent, n_outputs, split_decision)
from .world import Scenario


def _count(content: torch.Tensor, positions: list[int]) -> torch.Tensor:
    if not positions:
        return torch.zeros(content.shape[0], dtype=torch.long, device=content.device)
    return content[:, positions].sum(dim=1)


def gumbel_tau(cfg: Config, update: int) -> float:
    """Relaxation temperature after ``update`` training updates of the run."""
    t = cfg.train
    return anneal(t.gumbel_tau, t.gumbel_tau_final, update, t.tau_anneal_updates)


def run_and_update_gumbel(cfg: Config, scenarios,
                          farmers: Sequence[Agent], buyers: Sequence[Agent],
                          f_idx: torch.Tensor, b_idx: torch.Tensor, *,
                          update: int = 0, phase_update: Optional[int] = None,
                          device: str = "cpu",
                          train: bool = True, phase: Optional[Phase] = None,
                          generator: Optional[torch.Generator] = None,
                          usage=None, cost_scale: float = 1.0
                          ) -> tuple[BatchRollout, UpdateStats]:
    """Play a batch with a reparameterised message channel, then learn from it.

    ``scenarios`` is either a list of :class:`~orchard.world.Scenario` -- the
    readable path -- or a :class:`~orchard.batched.ScenarioBatch`, which keeps the
    whole batch on device and never enters the interpreter per episode.

    ``usage`` is the population's recent-usage record
    (:class:`orchard.conventions.PopulationUsage`). When given, speakers pay the
    coining cost and earn the convention bonus, and -- if training -- the batch is
    folded into it afterwards.

    ``cost_scale`` in [0, 1] scales the speaker's costs -- the length and coining
    costs, and the convention bonus only if ``reward.convention_gated``. The
    trainer raises it at ``reward.costs_from_rung`` and holds it at 1 from then
    on (see ``Trainer.update_cost_gate``).

    ``phase_update`` is how far into its own rung the run is. The temperature and
    entropy anneals count that rather than the whole run when
    ``train.anneal_per_rung`` is on, so a rung that starts late still explores.
    """
    c = cfg.channel
    t = cfg.train
    if phase is None:
        phase = ladder(cfg)[-1]              # the full market task
    referential = isinstance(scenarios, ReferentialBatch)
    mutual = isinstance(scenarios, MutualBatch)
    batched = referential or mutual or isinstance(scenarios, ScenarioBatch)
    if referential and scenarios.informer != phase.informer:
        phase = phase.with_informer(scenarios.informer)
    schema_of = {FARMER: phase_schema(cfg, FARMER, phase),
                 BUYER: phase_schema(cfg, BUYER, phase)}
    # Which slots are "mine" follows the phase's speaking order, not the trading
    # task's buyer-opens order.
    mask_of = {FARMER: phase.self_mask(cfg, FARMER, device),
               BUYER: phase.self_mask(cfg, BUYER, device)}
    B = len(scenarios)
    D = c.dialogue_len
    NT = c.n_token_ids
    # How far into its *own* rung the run is, which is what the anneals count
    # when `train.anneal_per_rung` is on.
    anneal_at = update if phase_update is None else int(phase_update)
    tau = gumbel_tau(cfg, anneal_at)

    if batched:
        f_obs = scenarios.obs(cfg, FARMER)
        b_obs = scenarios.obs(cfg, BUYER)
    else:
        f_obs = torch.tensor([farmer_obs(s, cfg) for s in scenarios],
                             dtype=torch.long, device=device)
        b_obs = torch.tensor([buyer_obs(s, cfg) for s in scenarios],
                             dtype=torch.long, device=device)
    obs_of = {FARMER: f_obs, BUYER: b_obs}
    idx_of = {FARMER: f_idx, BUYER: b_idx}
    pool_of = {FARMER: farmers, BUYER: buyers}
    # Which episodes each agent plays, computed once. Deriving it reads the index
    # tensor on the host, which on a GPU is a device sync; doing that at every
    # symbol step for every agent was the dominant cost with large populations.
    groups_of = {FARMER: group_by_agent(f_idx), BUYER: group_by_agent(b_idx)}

    pad_onehot = F.one_hot(torch.tensor(c.pad_id, device=device), NT).float()
    soft = pad_onehot.view(1, 1, NT).expand(B, D, NT).clone()
    tokens = torch.full((B, D), c.pad_id, dtype=torch.long, device=device)
    active = torch.zeros((B, D), dtype=torch.bool, device=device)
    token_entropy_terms: list[torch.Tensor] = []
    token_logp_terms: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {
        FARMER: [], BUYER: []}
    want_token_logp = (t.gumbel_mix_reinforce > 0 or t.shaping_reinforce > 0
                       or t.convention_reinforce > 0)

    # ---- the conversation ------------------------------------------------
    # Phases that use fewer turns simply leave the later dialogue slots empty,
    # which keeps one sequence layout -- and therefore one set of weights -- valid
    # across every rung of the curriculum.
    for turn in range(min(phase.n_turns, c.n_turns)):
        role = phase.speaker_of_turn(turn)
        pool, idx, obs = pool_of[role], idx_of[role], obs_of[role]
        alive = torch.ones(B, dtype=torch.bool, device=device)

        for k in range(c.max_msg_len):
            # Deliberately no `if not alive.any(): break`.  That reads a tensor on
            # the host and so synchronises the device on every symbol step, which
            # costs far more than the work it would skip.  Finished utterances are
            # masked to PAD below and contribute nothing.
            p = turn * c.max_msg_len + k
            seq_pos = dialogue_offset(cfg) + p
            logits = torch.zeros((B, c.n_emittable), device=device)
            for a_i, ep in groups_of[role]:
                # only the conversation so far, gathered per agent: the gathered
                # copy is what the backward pass keeps, so it must not carry the
                # empty remainder of the buffer
                h = pool[a_i].net.encode(obs[ep], soft[:, :p][ep], upto=seq_pos,
                                         schema=schema_of[role],
                                         self_mask=mask_of[role])[:, -1]
                logits = logits.index_copy(0, ep, pool[a_i].net.token_head(h))

            allowed = grammar_allowed(cfg, tokens[:, p - 1] if k > 0 else tokens[:, p], k)
            logits = logits.masked_fill(~allowed, MASKED)
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
            if want_token_logp:
                chosen = lp.gather(-1, tok.clamp(max=c.n_emittable - 1).unsqueeze(-1)).squeeze(-1)
                token_logp_terms[role].append((chosen, alive.float()))
            alive = alive & (tok != c.eos_id)
            # Stop early once every utterance in the batch has ended. Checked every
            # few symbols only: the check reads the device, and the buffer is long.
            if k % 4 == 3 and not bool(alive.any()):
                break

    # ---- the decisions: still discrete, still REINFORCE ------------------
    dec_sampled: dict[int, torch.Tensor] = {}
    dec_logp: dict[int, torch.Tensor] = {}
    dec_ent: dict[int, torch.Tensor] = {}
    dec_value: dict[int, torch.Tensor] = {}
    use_hindsight = train and batched and hindsight_applies(cfg, phase)
    targets = (hindsight_targets(cfg, phase, scenarios)
               if use_hindsight else {FARMER: {}, BUYER: {}})
    head_lp: dict[int, dict[int, torch.Tensor]] = {FARMER: {}, BUYER: {}}
    for role in (FARMER, BUYER):
        pool, idx, obs = pool_of[role], idx_of[role], obs_of[role]
        scored = set(phase.active_heads(role, cfg))
        out = torch.zeros((B, N_HEADS), dtype=torch.long, device=device)
        logp_sum = torch.zeros(B, device=device)
        ent_sum = torch.zeros(B, device=device)
        val = torch.zeros(B, device=device)
        for a_i, ep in groups_of[role]:
            net = pool[a_i].net
            h = net.encode(obs[ep], soft[ep], schema=schema_of[role],
                           self_mask=mask_of[role])[:, -1]
            heads = net.all_heads(h, obs[ep])
            val = val.index_copy(0, ep, net.value_head(h).squeeze(-1))
            lps = torch.zeros(ep.shape[0], device=device)
            ents = torch.zeros(ep.shape[0], device=device)
            for col, lg in enumerate(heads):
                lp = F.log_softmax(lg, dim=-1)
                with torch.no_grad():
                    a = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
                out[ep, col] = a
                if col in targets[role]:
                    full = head_lp[role].get(col)
                    if full is None:
                        full = torch.zeros((B, lp.shape[-1]), device=device)
                    head_lp[role][col] = full.index_copy(0, ep, lp)
                if col not in scored:
                    continue          # sampled for shape, not scored: no gradient
                # The belief heads are what make "were you understood" scoreable,
                # but they are also four more sampled actions; weighting them
                # keeps the loop without doubling the noise on the deal decision.
                wgt = 1.0 if col < 4 else cfg.reward.belief_grad_weight
                lps = lps + wgt * lp.gather(-1, a.unsqueeze(-1)).squeeze(-1)
                ents = ents + (-(lp.exp() * lp).sum(-1))
            # Entropy is averaged over the heads in play, not summed: adding heads
            # must not silently raise the exploration bonus.
            ents = ents / max(1, len(scored))
            logp_sum = logp_sum.index_copy(0, ep, lps)
            ent_sum = ent_sum.index_copy(0, ep, ents)
        dec_sampled[role] = out
        dec_logp[role] = logp_sum
        dec_ent[role] = ent_sum
        dec_value[role] = val

    # ---- resolve ---------------------------------------------------------
    # Everyone pays for the symbols they emitted -- counted from the phase's
    # speaking order. Counting from the trading order billed the lineup's
    # describer for nothing and its silent guesser for everything.
    content = (tokens < c.end_id)
    f_emitted = _count(content, phase.own_positions(cfg, FARMER))
    b_emitted = _count(content, phase.own_positions(cfg, BUYER))
    # What each speaker pays for the length of what it said: short words are
    # pressed for, saying several of them is nearly free.
    f_len = length_cost(cfg, tokens, phase.own_positions(cfg, FARMER))
    b_len = length_cost(cfg, tokens, phase.own_positions(cfg, BUYER))
    outcomes: list[Outcome] = []
    res = None
    if referential:
        res = resolve_referential(cfg, scenarios,
                                  dec_sampled[phase.guesser][:, H_CHOICE],
                                  f_len, b_len)
        f_rew, b_rew = res["farmer_reward"], res["buyer_reward"]
    elif mutual:
        res = resolve_mutual(cfg, scenarios, dec_sampled[FARMER],
                             dec_sampled[BUYER], f_len, b_len)
        f_rew, b_rew = res["farmer_reward"], res["buyer_reward"]
    elif phase.order and batched:
        res = resolve_request(cfg, phase, scenarios, dec_sampled, f_len, b_len)
        f_rew, b_rew = res["farmer_reward"], res["buyer_reward"]
    elif batched:
        # One pass over the batch instead of B trips through the interpreter.
        use_bel = cfg.reward.belief_heads
        res = resolve_batch(
            cfg, scenarios, dec_sampled[FARMER][:, :4], dec_sampled[BUYER][:, :4],
            f_len, b_len,
            f_bel=belief_columns(dec_sampled[FARMER]) if use_bel else None,
            b_bel=belief_columns(dec_sampled[BUYER]) if use_bel else None)
        f_rew, b_rew = res["farmer_reward"], res["buyer_reward"]
    else:
        f_rew = torch.zeros(B, device=device)
        b_rew = torch.zeros(B, device=device)
        for i, sc in enumerate(scenarios):
            fd, fb = split_decision(dec_sampled[FARMER][i])
            bd, bb = split_decision(dec_sampled[BUYER][i])
            o = resolve(cfg, sc, fd, bd, float(f_len[i]), float(b_len[i]),
                        f_beliefs=fb, b_beliefs=bb)
            outcomes.append(o)
            f_rew[i] = o.farmer_reward
            b_rew[i] = o.buyer_reward

    # ---- the speaker's own terms: brevity, coining, convention ------------
    g = float(min(1.0, max(0.0, cost_scale)))
    shape = {FARMER: -g * f_len, BUYER: -g * b_len}
    agree = {FARMER: torch.zeros(B, device=device), BUYER: torch.zeros(B, device=device)}
    if g < 1.0:
        # the resolvers charged the full length cost; hand back the gated share
        f_rew = f_rew + (1.0 - g) * f_len
        b_rew = b_rew + (1.0 - g) * b_len
        if res is not None:
            res["farmer_reward"], res["buyer_reward"] = f_rew, b_rew
    terms = {}
    if usage is not None:
        conv_on = g > 0 or not cfg.reward.convention_gated
        terms = usage.speaker_terms(phase, tokens, obs_of, rarity=g > 0,
                                    convention=conv_on)
        for role, d in terms.items():
            d["rarity"] = g * d["rarity"]
            if cfg.reward.convention_gated:
                d["convention"] = g * d["convention"]
            extra = d["convention"] - d["rarity"]
            shape[role] = shape[role] - d["rarity"]
            agree[role] = d["convention"]
            if role == FARMER:
                f_rew = f_rew + extra
            else:
                b_rew = b_rew + extra
        if res is not None:
            for role, label in ((FARMER, "farmer"), (BUYER, "buyer")):
                d = terms.get(role)
                zero = torch.zeros(B, device=device)
                res[label + "_rarity_cost"] = d["rarity"] if d else zero
                res[label + "_convention"] = d["convention"] if d else zero
            res["farmer_reward"], res["buyer_reward"] = f_rew, b_rew
        if train:
            usage.observe(terms, B)

    batch = BatchRollout(
        scenarios=[] if batched else list(scenarios), tokens=tokens, active=active,
        f_obs=f_obs, b_obs=b_obs, f_idx=f_idx, b_idx=b_idx,
        f_dec=dec_sampled[FARMER].detach(), b_dec=dec_sampled[BUYER].detach(),
        f_reward=f_rew.detach(), b_reward=b_rew.detach(), outcomes=outcomes,
        f_emitted=f_emitted, b_emitted=b_emitted, f_cost=f_len, b_cost=b_len,
        res=res, sb=scenarios if batched else None, n=B, cfg_ref=cfg, phase=phase)

    stats = UpdateStats()
    if not train:
        return batch, stats

    # ---- one joint objective ---------------------------------------------
    ent_tok_coef = anneal(t.entropy_coef, t.entropy_coef_final, anneal_at,
                          t.entropy_anneal_updates)
    ent_dec_coef = anneal(t.decision_entropy_coef, t.decision_entropy_coef_final,
                          anneal_at, t.entropy_anneal_updates)
    rew_of = {FARMER: f_rew, BUYER: b_rew}
    loss = torch.zeros((), device=device)
    value_loss_total = 0.0
    for role in (FARMER, BUYER):
        R = rew_of[role]
        scale = 1.0
        if t.normalise_adv and R.numel() > 1:
            scale = float(R.std(unbiased=False)) + 1e-6
            R = (R - R.mean()) / scale
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
        if (t.shaping_reinforce > 0 or t.convention_reinforce > 0) and token_logp_terms[role]:
            # Brevity, coining and convention are the speaker's alone and depend
            # only on what it said, so they are credited to its token choices
            # directly, in the same units as the task advantage (divided by the
            # same reward scale) so raising a cost really does raise its pull.
            sh = shape[role]
            adv_s = ((sh - sh.mean()) / scale).detach()
            ag = agree[role]
            adv_c = ((ag - ag.mean()) / scale).detach()
            for chosen, mask in token_logp_terms[role]:
                denom = mask.sum().clamp(min=1.0)
                loss = loss + t.shaping_reinforce * (-(adv_s * chosen * mask).sum() / denom)
                if t.convention_reinforce > 0:
                    loss = loss + t.convention_reinforce * (
                        -(adv_c * chosen * mask).sum() / denom)

    # hindsight feedback: every scored head is pulled towards the outcome
    if use_hindsight:
        for role in (FARMER, BUYER):
            for col, tgt in targets[role].items():
                lp = head_lp[role].get(col)
                if lp is None:
                    continue
                tgt = tgt.long().clamp(0, lp.shape[-1] - 1)
                loss = loss + t.hindsight_coef * F.nll_loss(lp, tgt)

    if token_entropy_terms:
        ent_stack = torch.stack(token_entropy_terms, dim=1)
        mask = active.float()[:, :ent_stack.shape[1]]
        tok_ent = (ent_stack * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss - ent_tok_coef * tok_ent
        stats.token_entropy = float(tok_ent.detach())

    seen = []
    for role in (FARMER, BUYER):
        for a_i, _ in groups_of[role]:
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
