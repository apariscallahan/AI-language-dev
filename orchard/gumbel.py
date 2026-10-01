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
from .curriculum import (H_BELIEF, H_BELIEF_COLOR, H_CHOICE, H_REPORT, N_HEADS,
                         MutualBatch, Phase, ReferentialBatch, hindsight_applies,
                         hindsight_targets, ladder, phase_schema, resolve_referential,
                         resolve_reports)
from .gesture import (GESTURE_NONE, draw_availability, gesture_option_mask,
                      gesture_tokens_for, gestured_fields, n_token_ids,
                      without_gestures)
from .rollout import (BatchRollout, UpdateStats, anneal, belief_columns,
                      group_by_agent, n_outputs, split_decision)
from .world import N_LOT_FIELDS, Scenario


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
                          usage=None, cost_scale: float = 1.0,
                          convention_scale: Optional[float] = None,
                          gesture_share: float = 0.0
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

    ``gesture_share`` is the share of this batch's rounds in which a speaker may
    open a turn with a gesture (orchard/gesture.py). The trainer sets it from
    the rung's schedule; every measurement leaves it at 0, so nothing about the
    language is ever judged on a gesture.
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
    NT = n_token_ids(cfg)
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

    # ---- gestures (orchard/gesture.py) -------------------------------------
    # The world says in which rounds a gesture is possible; in those, the speaker
    # decides at the start of each of its turns whether to make one, and about
    # which field. A gesture takes the turn's first slot, is not the token
    # policy's action (so it is inactive for the token terms and masked from a
    # newborn's lesson), and the speaker pays for it.
    gest_on = bool(cfg.gesture.enabled) and gesture_share > 0
    avail = (draw_availability(gesture_share, B, device, generator) if gest_on
             else torch.zeros(B, dtype=torch.bool, device=device))
    gest_logp = {FARMER: torch.zeros(B, device=device), BUYER: torch.zeros(B, device=device)}
    gest_ent = {FARMER: torch.zeros(B, device=device), BUYER: torch.zeros(B, device=device)}
    gest_n = {FARMER: torch.zeros(B, device=device), BUYER: torch.zeros(B, device=device)}
    turn_starts: dict[int, list[int]] = {FARMER: [], BUYER: []}
    gest_field_counts = torch.zeros(1 + N_LOT_FIELDS, dtype=torch.long, device=device)
    onehot_table = torch.eye(NT, device=device) if gest_on else None

    # ---- positive signalling (`reward.lexicon_mi`) --------------------------
    # The distribution over the first symbol a speaker *speaks* in its first
    # turn (after its gesture, if it made one), kept per role so that after the
    # conversation the mutual information between the asked-about meaning and
    # that symbol can be taken over the batch. Grouped by meaning, so the
    # variation rewarded is in the asked-about field, not in the rest of the lot.
    mi_on = train and cfg.reward.lexicon_mi > 0
    first_lp = {r: torch.zeros((B, c.n_emittable), device=device) for r in (FARMER, BUYER)}
    have_first = {r: torch.zeros(B, dtype=torch.bool, device=device) for r in (FARMER, BUYER)}
    mi_keys: dict[int, list] = {}
    if mi_on:
        from .conventions import naming_keys
        for r in (FARMER, BUYER):
            if phase.speaks(cfg, r):
                mi_keys[r] = naming_keys(cfg, phase, r, obs_of[r])

    # ---- the conversation ------------------------------------------------
    # Phases that use fewer turns simply leave the later dialogue slots empty,
    # which keeps one sequence layout -- and therefore one set of weights -- valid
    # across every rung of the curriculum.
    for turn in range(min(phase.n_turns, c.n_turns)):
        role = phase.speaker_of_turn(turn)
        pool, idx, obs = pool_of[role], idx_of[role], obs_of[role]
        alive = torch.ones(B, dtype=torch.bool, device=device)

        turn_starts[role].append(turn * c.max_msg_len)
        for k in range(c.max_msg_len):
            # Deliberately no `if not alive.any(): break`.  That reads a tensor on
            # the host and so synchronises the device on every symbol step, which
            # costs far more than the work it would skip.  Finished utterances are
            # masked to PAD below and contribute nothing.
            p = turn * c.max_msg_len + k
            seq_pos = dialogue_offset(cfg) + p
            logits = torch.zeros((B, c.n_emittable), device=device)
            g_logits = (torch.zeros((B, 1 + N_LOT_FIELDS), device=device)
                        if gest_on and k == 0 else None)
            for a_i, ep in groups_of[role]:
                # only the conversation so far, gathered per agent: the gathered
                # copy is what the backward pass keeps, so it must not carry the
                # empty remainder of the buffer
                net = pool[a_i].net
                h = net.encode(obs[ep], soft[:, :p][ep], upto=seq_pos,
                               schema=schema_of[role], self_mask=mask_of[role])[:, -1]
                # which parts this turn has named, and are some still unnamed?
                turn_st = net.turn_so_far(obs[ep], schema_of[role], tokens[ep], [p])
                logits = logits.index_copy(
                    0, ep, net.speak(h, obs[ep], schema_of[role],
                                     None if turn_st is None
                                     else (turn_st[0][:, 0], turn_st[1][:, 0])))
                if g_logits is not None:
                    g_logits = g_logits.index_copy(0, ep, net.gesture_head(h))

            allowed = grammar_allowed(cfg, tokens[:, p - 1] if k > 0 else tokens[:, p], k)
            logits = logits.masked_fill(~allowed, MASKED)
            y = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
            tok = y.argmax(dim=-1)
            tok = torch.where(alive, tok, torch.full_like(tok, c.pad_id))
            y_full = torch.cat([y, torch.zeros((B, NT - c.n_emittable), device=device)],
                               dim=-1)
            row = torch.where(alive.unsqueeze(-1), y_full, pad_onehot.view(1, NT))

            gestured = torch.zeros(B, dtype=torch.bool, device=device)
            if g_logits is not None:
                # Which options this seat has at all, and only "none" in a round
                # the world allows no gesture in. A round without the choice is
                # not trained on it.
                opts = gesture_option_mask(cfg, phase, role, device).unsqueeze(0).expand(B, -1)
                none_only = torch.zeros_like(opts)
                none_only[:, GESTURE_NONE] = True
                allowed_g = torch.where(avail.unsqueeze(1), opts, none_only)
                g_lp = F.log_softmax(g_logits.masked_fill(~allowed_g, MASKED), dim=-1)
                with torch.no_grad():
                    g_choice = torch.multinomial(g_lp.exp(), 1, generator=generator).squeeze(-1)
                # The value shown is read off the speaker's own observation, so
                # a gesture can only ever reveal what its maker can see.
                g_tok = gesture_tokens_for(cfg, phase, role, obs, g_choice)
                gestured = g_tok != c.pad_id
                had_choice = avail.float()
                gest_logp[role] = gest_logp[role] + (
                    g_lp.gather(-1, g_choice.unsqueeze(-1)).squeeze(-1) * had_choice)
                gest_ent[role] = gest_ent[role] + (-(g_lp.exp() * g_lp).sum(-1)) * had_choice
                gest_n[role] = gest_n[role] + gestured.float()
                gest_field_counts = gest_field_counts + torch.bincount(
                    g_choice[gestured], minlength=1 + N_LOT_FIELDS)
                # the gesture takes the slot: an exact one-hot, no gradient --
                # its content is the world's, only the choice was the agent's
                tok = torch.where(gestured, g_tok, tok)
                row = torch.where(gestured.unsqueeze(-1), onehot_table[g_tok], row)
            # what the token policy actually did here (a gesture is not its act)
            acted = alive & ~gestured
            soft = soft.index_copy(1, torch.tensor([p], device=device), row.unsqueeze(1))

            tokens[:, p] = tok.detach()
            active[:, p] = acted
            lp = F.log_softmax(logits, dim=-1)
            if mi_on and role in mi_keys and k <= 1:
                take = acted & ~have_first[role]
                first_lp[role] = torch.where(take.unsqueeze(-1), lp, first_lp[role])
                have_first[role] = have_first[role] | take
            token_entropy_terms.append(-(lp.exp() * lp).sum(-1) * acted.float())
            if want_token_logp:
                chosen = lp.gather(-1, tok.clamp(max=c.n_emittable - 1).unsqueeze(-1)).squeeze(-1)
                token_logp_terms[role].append((chosen, acted.float()))
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
    # Which heads' log-probabilities to keep for a supervised term: hindsight's
    # (read with the innate reader, like every decision).
    keep_lp = {r: set(targets[r]) for r in (FARMER, BUYER)}
    # ...and, when the other party may have gestured, the five report heads as
    # the *transformer alone* reads them, one of which is taught the gestured
    # value (see below). Not through the innate reader: it cannot see a
    # gesture, so for it the lesson would be the gestured value from the words
    # alone, on every gestured round, babble included -- hindsight by another
    # name. Measured on `name-fruit` with fresh agents: that term alone pushed
    # the reader's "no word names it" score up and its gain down in 10 of 12
    # batches, several times harder than everything else put together. The
    # reader learns words from the ostensive lesson, which waits for a name.
    supervise_gestures = train and gest_on and cfg.gesture.supervise_coef > 0
    gest_lp: dict[int, dict[int, torch.Tensor]] = {FARMER: {}, BUYER: {}}
    belief_cols = set(H_BELIEF) | {H_BELIEF_COLOR}
    for role in (FARMER, BUYER):
        pool, idx, obs = pool_of[role], idx_of[role], obs_of[role]
        scored = set(phase.active_heads(role, cfg))
        out = torch.zeros((B, N_HEADS), dtype=torch.long, device=device)
        logp_sum = torch.zeros(B, device=device)
        ent_sum = torch.zeros(B, device=device)
        val = torch.zeros(B, device=device)
        for a_i, ep in groups_of[role]:
            net = pool[a_i].net
            # indexed once: the backward pass keeps what goes in, so a second
            # gather would keep a second copy of the soft dialogue
            soft_ep = soft[ep]
            h = net.encode(obs[ep], soft_ep, schema=schema_of[role],
                           self_mask=mask_of[role])[:, -1]
            # the innate reader's reading of the other party's words, through
            # the same soft one-hots, so it too carries a gradient to the speaker
            lex = net.read_words(soft_ep, tokens[ep], mask_of[role])
            heads = net.all_heads(h, obs[ep], lex=lex)
            if supervise_gestures:
                for col, lg in zip(H_REPORT, net.report_logits(h)):
                    full = gest_lp[role].get(col)
                    if full is None:
                        full = torch.zeros((B, lg.shape[-1]), device=device)
                    gest_lp[role][col] = full.index_copy(0, ep, F.log_softmax(lg, dim=-1))
            val = val.index_copy(0, ep, net.value_head(h).squeeze(-1))
            lps = torch.zeros(ep.shape[0], device=device)
            ents = torch.zeros(ep.shape[0], device=device)
            for col, lg in enumerate(heads):
                lp = F.log_softmax(lg, dim=-1)
                with torch.no_grad():
                    a = torch.multinomial(lp.exp(), 1, generator=generator).squeeze(-1)
                out[ep, col] = a
                if col in keep_lp[role]:
                    full = head_lp[role].get(col)
                    if full is None:
                        full = torch.zeros((B, lp.shape[-1]), device=device)
                    head_lp[role][col] = full.index_copy(0, ep, lp)
                if col not in scored:
                    continue          # sampled for shape, not scored: no gradient
                # The belief heads are what make "were you understood" scoreable,
                # but they are also four more sampled actions; weighting them
                # keeps the loop without doubling the noise on the deal decision.
                # (The lineup choice is not a belief head and is not weighted.)
                wgt = cfg.reward.belief_grad_weight if col in belief_cols else 1.0
                lps = lps + wgt * lp.gather(-1, a.unsqueeze(-1)).squeeze(-1)
                ents = ents + (-(lp.exp() * lp).sum(-1))
            # Entropy is averaged over the heads in play, not summed: adding heads
            # must not silently raise the exploration bonus.
            ents = ents / max(1, len(scored))
            logp_sum = logp_sum.index_copy(0, ep, lps)
            ent_sum = ent_sum.index_copy(0, ep, ents)
        dec_sampled[role] = out
        # Whether to gesture is one of the speaker's decisions, trained on the
        # round's reward, which includes the cost of having done it -- but not
        # against the value head (see the loss below), so it is kept apart.
        dec_logp[role] = logp_sum
        dec_ent[role] = ent_sum + gest_ent[role]
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
    elif phase.reporting and batched:
        # mutual, order, offer, judge: what one side holds, read off the other's heads
        res = resolve_reports(cfg, phase, scenarios, dec_sampled, f_len, b_len)
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
            fd, fb = split_decision(dec_sampled[FARMER][i], cfg.reward.belief_heads)
            bd, bb = split_decision(dec_sampled[BUYER][i], cfg.reward.belief_heads)
            o = resolve(cfg, sc, fd, bd, float(f_len[i]), float(b_len[i]),
                        f_beliefs=fb, b_beliefs=bb)
            outcomes.append(o)
            f_rew[i] = o.farmer_reward
            b_rew[i] = o.buyer_reward

    # ---- gesturing takes effort -------------------------------------------
    # Charged whatever the rung: this is what makes a word worth more than the
    # gesture it replaces once the word works, and it is small enough that the
    # gesture is still worth making while it does not.
    if gest_on and cfg.gesture.cost > 0:
        f_g = cfg.gesture.cost * gest_n[FARMER]
        b_g = cfg.gesture.cost * gest_n[BUYER]
        f_rew = f_rew - f_g
        b_rew = b_rew - b_g
        if res is not None:
            res["farmer_reward"], res["buyer_reward"] = f_rew, b_rew
            res["farmer_gesture_cost"], res["buyer_gesture_cost"] = f_g, b_g

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
        # The convention bonus has its own gate (`reward.convention_from_rung`):
        # agreeing is not economising. The rarity cost stays with the costs.
        cg = g if convention_scale is None else float(min(1.0, max(0.0, convention_scale)))
        conv_on = cg > 0 or not cfg.reward.convention_gated
        # Who spoke each episode, for the speaker's own lexicon (the innate
        # one-name-per-meaning prior, `reward.lexicon`): never gated, so it is
        # on from the first round of the first rung.
        # (Composition and word order read the same lexicon, so each of the
        # three needs the speakers named; gating this on `reward.lexicon` alone
        # switched the other two off whenever the lexicon bonus was.)
        ids_of = None
        if cfg.reward.lexicon > 0 or cfg.reward.compose > 0 or cfg.reward.word_order > 0:
            ids_of = {r: [pool_of[r][i].agent_id for i in idx_of[r].tolist()]
                      for r in (FARMER, BUYER) if phase.speaks(cfg, r)}
        terms = usage.speaker_terms(phase, tokens, obs_of, rarity=g > 0,
                                    convention=conv_on, agent_ids=ids_of)
        for role, d in terms.items():
            d["rarity"] = g * d["rarity"]
            if cfg.reward.convention_gated:
                d["convention"] = cg * d["convention"]
            extra = d["convention"] + d["lexicon"] + d["compose"] - d["rarity"]
            shape[role] = shape[role] - d["rarity"]
            # all three are pressures to *agree* -- with the community, with
            # oneself, and with one's own words when describing a whole thing --
            # and reach the words by the same route
            agree[role] = d["convention"] + d["lexicon"] + d["compose"]
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
                res[label + "_lexicon"] = d["lexicon"] if d else zero
                res[label + "_compose"] = d["compose"] if d else zero
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
        # The speaker's own acts -- its gesture and its symbols -- are credited
        # against a baseline that cannot see them: the batch's mean reward. The
        # value head is read at DECIDE, after the whole dialogue, so it sees
        # the gesture and the words; as a baseline for them it predicts their
        # effect and subtracts it, and once it fits, the gesture's cost and the
        # symbols' task advantage cancel out of their own gradient.
        adv_own = (R - R.mean()).detach()
        loss = loss + (-(adv_own * gest_logp[role]).mean())
        vl = ((dec_value[role] - R) ** 2).mean()
        loss = loss + t.value_coef * vl
        loss = loss - ent_dec_coef * dec_ent[role].mean()
        value_loss_total += float(vl.detach())
        # A token term is a sequence's log-probability times its advantage,
        # summed over the tokens and averaged over the episodes. (It was the mean
        # over whichever episodes were still talking at each step, so the one
        # long utterance still going at step twenty weighed as much as the whole
        # batch's first symbols.)
        n_ep = float(max(1, B))
        if t.gumbel_mix_reinforce > 0:
            for chosen, mask in token_logp_terms[role]:
                loss = loss + t.gumbel_mix_reinforce * (
                    -(adv_own * chosen * mask).sum() / n_ep)
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
                loss = loss + t.shaping_reinforce * (-(adv_s * chosen * mask).sum() / n_ep)
                if t.convention_reinforce > 0:
                    loss = loss + t.convention_reinforce * (
                        -(adv_c * chosen * mask).sum() / n_ep)

    # hindsight feedback: every scored head is pulled towards the outcome
    if use_hindsight:
        for role in (FARMER, BUYER):
            for col, tgt in targets[role].items():
                lp = head_lp[role].get(col)
                if lp is None:
                    continue
                tgt = tgt.long().clamp(0, lp.shape[-1] - 1)
                loss = loss + t.hindsight_coef * F.nll_loss(lp, tgt)

    # Positive signalling: apple rounds should sound alike and unlike banana
    # rounds, by the speaker's own lights. Exact, listener-free, from update one.
    if mi_on:
        from .conventions import naming_mutual_information
        mi_total = 0.0
        for role, keys in mi_keys.items():
            mi = naming_mutual_information(first_lp[role], have_first[role], keys)
            if mi is None:
                continue
            loss = loss - cfg.reward.lexicon_mi * mi
            mi_total += float(mi.detach())
        stats.naming_signal = mi_total / (len(mi_keys) or 1)

    # The ostensive lesson (`gesture.ostensive_coef`): the parent points at the
    # apple and says "apple". On a round where the other party gestured *and*
    # said its established name for the gestured meaning (the speaker's own
    # lexicon says which utterances are names), the listener is shown the turn
    # without the gesture -- the words shifted to where a gesture-free turn's
    # words sit -- and its head for that field is taught the gestured value: a
    # labelled example of the word, from the words alone. Babble is not a
    # lesson, which is what keeps this from teaching a listener that words
    # carry nothing before any word exists.
    ost = float(cfg.gesture.ostensive_coef)
    if train and gest_on and ost > 0 and terms:
        for role in (FARMER, BUYER):
            other = BUYER if role == FARMER else FARMER
            d = terms.get(other)
            if d is None or not turn_starts[other]:
                continue
            field, value = gestured_fields(cfg, tokens, turn_starts[other])
            # The name the speaker said is its name for the field it was *asked*
            # about, so the lesson is about that field only: a speaker naming
            # the fruit while pointing at the colour does not teach the colour.
            # (It used to, from the fruit word, and the error reached the reader
            # and -- through the straight-through channel -- the speaker too.)
            asked = torch.tensor([k[0] if k is not None else -1
                                  for k in (d.get("_lexicon_keys") or [None] * B)],
                                 dtype=torch.long, device=device)
            lesson = d["word_used"] & (field >= 0) & (field == asked)
            if not bool(lesson.any()):
                continue
            words_only = without_gestures(cfg, tokens, soft, turn_starts[other])
            words_ids = without_gestures(cfg, tokens, tokens, turn_starts[other])
            pool, obs = pool_of[role], obs_of[role]
            for a_i, ep in groups_of[role]:
                sel = ep[lesson[ep]]
                if sel.numel() == 0:
                    continue
                net = pool[a_i].net
                w_sel = words_only[sel]           # indexed once: one copy for backward
                h = net.encode(obs[sel], w_sel, schema=schema_of[role],
                               self_mask=mask_of[role])[:, -1]
                heads = net.report_logits(h, net.read_words(
                    w_sel, words_ids[sel], mask_of[role]))
                for j in range(N_LOT_FIELDS):
                    rows = field[sel] == j
                    if not bool(rows.any()):
                        continue
                    lg = heads[j][rows]
                    tgt = value[sel][rows].clamp(0, lg.shape[-1] - 1)
                    loss = loss + ost * F.cross_entropy(lg, tgt) * (
                        float(rows.sum()) / float(lesson.sum()))

    # A gesture is the answer, shown: the listener's head for that field is
    # taught to read it. Not hindsight -- the answer is *in the message*, so this
    # cannot teach a listener that the message carries nothing -- and it reaches
    # the speaker's words through the straight-through channel, pulling them
    # towards whatever the listener already reads as that value.
    if supervise_gestures:
        coef = float(cfg.gesture.supervise_coef)
        for role in (FARMER, BUYER):
            other = BUYER if role == FARMER else FARMER
            if not turn_starts[other]:
                continue
            field, value = gestured_fields(cfg, tokens, turn_starts[other])
            for j in range(N_LOT_FIELDS):
                sel = field == j
                if not bool(sel.any()):
                    continue
                lp = gest_lp[role].get(H_REPORT[j])
                if lp is None:
                    continue
                tgt = value[sel].clamp(0, lp.shape[-1] - 1)
                loss = loss + coef * F.nll_loss(lp[sel], tgt)

    if token_entropy_terms:
        # Each step's entropy is already masked to the episodes whose token
        # policy acted there, so the mean per token is the sum over the total.
        # (It used to be re-masked by `active[:, :n_steps]`, which lines steps
        # up with turn-0 positions: every turn after the first was compared
        # with padding, and the second speaker's entropy was dropped.)
        ent_stack = torch.stack(token_entropy_terms, dim=1)
        tok_ent = ent_stack.sum() / active.float().sum().clamp(min=1.0)
        loss = loss - ent_tok_coef * tok_ent
        stats.token_entropy = float(tok_ent.detach())

    # Every agent that played takes one optimiser step. Below
    # `curriculum.split_roles_at` the farmer and buyer seats are one list, so
    # listing both seats named every agent twice, and each took two Adam steps
    # on the same gradient -- about twice the intended step size on every
    # pooled rung, halving at the split. (`train.lr` is now the step those
    # rungs were run and validated at; see its comment.)
    seen, seen_ids = [], set()
    for role in (FARMER, BUYER):
        for a_i, _ in groups_of[role]:
            a = pool_of[role][a_i]
            if id(a) not in seen_ids:
                seen_ids.add(id(a))
                seen.append(a)
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
    if terms:
        stats.lexicon_bonus = float(sum(float(d["lexicon"].sum()) for d in terms.values())) / B
        used_n = sum(int(d["word_used"].sum()) for d in terms.values())
        stats.words_used = used_n / float(B * max(1, len(terms)))
        # describing a whole lot (`reward.compose`): how many of the lot's
        # fields the speaker named with its own word, and how consistent the
        # order was -- the numbers that showed `name-all` failing
        cs = [d["_compose_stats"] for d in terms.values() if d.get("_compose_stats")]
        n_desc = sum(s["descriptions"] for s in cs)
        if n_desc:
            stats.descriptions = n_desc
            stats.compose_bonus = sum(s["bonus_sum"] for s in cs) / n_desc
            stats.names_reused = (sum(s["fields_reused"] for s in cs)
                                  / float(n_desc * N_LOT_FIELDS))
            n_pairs = sum(s["order_pairs"] for s in cs)
            stats.order_pairs = n_pairs
            stats.order_agreement = (sum(s["order_agreed"] for s in cs) / n_pairs
                                     if n_pairs else 0.0)
    if gest_on:
        n_turns = sum(len(v) for v in turn_starts.values())
        allowed_turns = float(avail.sum()) * n_turns
        used = float(gest_n[FARMER].sum() + gest_n[BUYER].sum())
        stats.gesture_share = float(gesture_share)
        stats.gesture_available = float(avail.float().mean())
        stats.gesture_used = used / allowed_turns if allowed_turns > 0 else 0.0
        stats.gesture_fields = [int(x) for x in gest_field_counts[1:].tolist()]
    return batch, stats
