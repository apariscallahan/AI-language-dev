"""The transmission bottleneck: iterated learning for newborns (spec 4).

When an agent dies its replacement does not inherit weights.  Instead it gets a
short supervised apprenticeship on a **deliberately small sample** of recent
successful trades by the living population, and only then joins the RL loop.

The sample is also **skewed toward what was common** (addendum 2.3).  Drawing
episodes in proportion to how often each meaning actually came up is what a
learner's experience is really like: a newborn sees hundreds of ordinary trades
and may see a given unusual one never.  It therefore reliably generalises the
systematic pattern for common cases, and often simply cannot reproduce whatever
narrow form a rare case happened to acquire -- which is where vocabulary loss and
the regularisation of rare forms come from, with no separate forgetting
mechanism.  ``BottleneckConfig.frequency_skew`` controls how hard this bites:
1.0 is natural proportion, 0.0 flattens it so rare meanings are as well
represented as common ones.

The smallness is the whole mechanism.  Kirby-style iterated learning produces
systematic structure precisely because each generation must reconstruct the whole
language from a fraction of it: an idiosyncratic lookup table cannot survive the
squeeze, whereas a compositional code can be inferred from a handful of examples
and regenerated in full.  Making ``n_samples`` large would remove the pressure
and turn this into plain cloning -- which is exactly why it is the headline knob
in the on/off comparison (spec 9).

What the newborn learns is standard cross-entropy:
  * its own message tokens, teacher-forced against what the retiring generation
    said in the same position of the same conversation, and
  * its trade decision, against what that generation decided.
It never sees the other party's private observation -- only its own half of the
transcript, exactly as in live play.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from collections import Counter

from .agents import Agent, own_dialogue_positions
from .config import Config
from .env import BUYER, FARMER
from .rollout import BatchRollout, read_positions_for


@dataclass
class StoredEpisode:
    """One successful negotiation, kept in a form either role can learn from."""
    f_obs: torch.Tensor      # (4,)
    b_obs: torch.Tensor      # (4,)
    tokens: torch.Tensor     # (D,)
    active: torch.Tensor     # (D,) bool
    f_dec: torch.Tensor      # (4,)
    b_dec: torch.Tensor      # (4,)
    episode: int
    f_generation: int
    b_generation: int
    meaning: tuple[int, int] = (0, 0)   # (wanted variety, needed quantity)

    def obs_for(self, role: int) -> torch.Tensor:
        return self.f_obs if role == FARMER else self.b_obs

    def dec_for(self, role: int) -> torch.Tensor:
        return self.f_dec if role == FARMER else self.b_dec

    def generation_of(self, role: int) -> int:
        return self.f_generation if role == FARMER else self.b_generation


class TranscriptStore:
    """Ring buffer of recent (by default: successful) episodes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.capacity = cfg.bottleneck.store_capacity
        self._buf: list[StoredEpisode] = []
        self._pos = 0
        self.total_added = 0
        self.meaning_counts: Counter = Counter()

    def __len__(self) -> int:
        return len(self._buf)

    def add_batch(self, batch: BatchRollout, farmers, buyers, episode: int) -> int:
        added = 0
        for i, o in enumerate(batch.outcomes):
            if self.cfg.bottleneck.only_successful and not o.success:
                continue
            sc = batch.scenarios[i]
            item = StoredEpisode(
                f_obs=batch.f_obs[i].detach().clone(),
                b_obs=batch.b_obs[i].detach().clone(),
                tokens=batch.tokens[i].detach().clone(),
                active=batch.active[i].detach().clone(),
                f_dec=batch.f_dec[i].detach().clone(),
                b_dec=batch.b_dec[i].detach().clone(),
                episode=episode,
                f_generation=farmers[int(batch.f_idx[i])].generation,
                b_generation=buyers[int(batch.b_idx[i])].generation,
                meaning=(sc.buyer.want_variety, sc.buyer.need_qty),
            )
            if len(self._buf) < self.capacity:
                self._buf.append(item)
            else:
                old = self._buf[self._pos]
                self.meaning_counts[old.meaning] -= 1
                if self.meaning_counts[old.meaning] <= 0:
                    del self.meaning_counts[old.meaning]
                self._buf[self._pos] = item
                self._pos = (self._pos + 1) % self.capacity
            self.meaning_counts[item.meaning] += 1
            added += 1
            self.total_added += 1
        return added

    def sample(self, n: int, rng: random.Random) -> list[StoredEpisode]:
        """Draw the newborn's curriculum, skewed by meaning frequency.

        Sampling episodes uniformly already reproduces the parent generation's
        natural proportions, so ``frequency_skew == 1`` is the "what actually
        happened" case.  The weight ``count ** (skew - 1)`` bends it from there:
        0 flattens to one-per-meaning-type, above 1 concentrates further on the
        common cases.
        """
        if not self._buf:
            return []
        skew = self.cfg.bottleneck.frequency_skew
        if n >= len(self._buf) and abs(skew - 1.0) < 1e-9:
            return list(self._buf)
        if abs(skew - 1.0) < 1e-9:
            return rng.sample(self._buf, n)

        weights = [max(self.meaning_counts.get(it.meaning, 1), 1) ** (skew - 1.0)
                   for it in self._buf]
        total = sum(weights)
        if total <= 0:
            return rng.sample(self._buf, min(n, len(self._buf)))
        k = min(n, len(self._buf))
        # weighted sampling without replacement, so the curriculum is not padded
        # with duplicates of the single commonest trade
        pool = list(range(len(self._buf)))
        w = list(weights)
        picked: list[StoredEpisode] = []
        for _ in range(k):
            tot = sum(w)
            if tot <= 0:
                break
            x = rng.random() * tot
            acc = 0.0
            for j, wj in enumerate(w):
                acc += wj
                if x <= acc:
                    picked.append(self._buf[pool[j]])
                    w[j] = 0.0
                    break
        return picked

    def meaning_profile(self, items: list[StoredEpisode]) -> dict[str, Any]:
        """What the newborn was actually shown, for the birth log."""
        seen = Counter(it.meaning for it in items)
        live = len(self.meaning_counts)
        return {
            "distinct_meanings_shown": len(seen),
            "distinct_meanings_available": live,
            "coverage": len(seen) / live if live else 0.0,
            "top_meanings": [{"meaning": list(k), "n": v} for k, v in seen.most_common(5)],
            "unseen_meanings": [list(k) for k in self.meaning_counts if k not in seen][:10],
        }


def train_newborn(cfg: Config, agent: Agent, store: TranscriptStore,
                  rng: random.Random, *, device: str = "cpu") -> dict[str, Any]:
    """Run the newborn's supervised apprenticeship.  Returns a log record."""
    bc = cfg.bottleneck
    info: dict[str, Any] = {
        "enabled": bc.enabled,
        "requested_samples": bc.n_samples,
        "store_size": len(store),
        "n_samples": 0,
        "epochs": 0,
        "final_token_loss": None,
        "final_decision_loss": None,
        "token_accuracy": None,
        "decision_accuracy": None,
        "teacher_generations": {},
    }
    if not bc.enabled:
        info["skipped"] = "bottleneck disabled"
        return info

    samples = store.sample(bc.n_samples, rng)
    if len(samples) < 8:
        info["skipped"] = "not enough successful transcripts yet"
        return info

    role = agent.role
    obs = torch.stack([s.obs_for(role) for s in samples]).to(device)
    toks = torch.stack([s.tokens for s in samples]).to(device)
    act = torch.stack([s.active for s in samples]).to(device)
    dec = torch.stack([s.dec_for(role) for s in samples]).to(device)

    gens: dict[int, int] = {}
    for s in samples:
        g = s.generation_of(role)
        gens[g] = gens.get(g, 0) + 1
    info["teacher_generations"] = {str(k): v for k, v in sorted(gens.items())}
    info["n_samples"] = len(samples)
    info["epochs"] = bc.epochs
    info["frequency_skew"] = bc.frequency_skew
    info["meaning_coverage"] = store.meaning_profile(samples)
    info["sample_episode_span"] = [min(s.episode for s in samples),
                                   max(s.episode for s in samples)]

    own_pos = own_dialogue_positions(cfg, role)
    read_pos = read_positions_for(cfg, role, device)
    target_tokens = toks[:, own_pos]
    target_mask = act[:, own_pos]
    # PAD is not an emittable id; park masked-out slots at 0 and drop them from the loss.
    safe_targets = torch.where(target_mask, target_tokens, torch.zeros_like(target_tokens))

    opt = torch.optim.Adam(agent.net.parameters(), lr=bc.lr)
    n = len(samples)
    order = list(range(n))
    tok_loss_val = dec_loss_val = 0.0
    tok_acc = dec_acc = 0.0

    agent.net.train()
    for _ in range(bc.epochs):
        rng.shuffle(order)
        ep_tok = ep_dec = 0.0
        ep_tok_acc = ep_dec_acc = 0.0
        nb = 0
        for start in range(0, n, bc.batch_size):
            sel = torch.tensor(order[start:start + bc.batch_size], dtype=torch.long,
                               device=device)
            tok_logits, _, dec_logits, _ = agent.net.full_pass(obs[sel], toks[sel], read_pos)
            m = target_mask[sel]
            denom = m.sum().clamp(min=1)
            ce = F.cross_entropy(
                tok_logits.reshape(-1, tok_logits.shape[-1]),
                safe_targets[sel].reshape(-1), reduction="none").reshape(m.shape)
            tok_loss = (ce * m).sum() / denom
            pred = tok_logits.argmax(-1)
            ep_tok_acc += float(((pred == target_tokens[sel]) & m).sum() / denom)

            dec_loss = torch.zeros((), device=device)
            corr = torch.ones(sel.shape[0], dtype=torch.bool, device=device)
            for col, lg in enumerate(dec_logits):
                dec_loss = dec_loss + F.cross_entropy(lg, dec[sel][:, col])
                corr &= (lg.argmax(-1) == dec[sel][:, col])
            dec_loss = dec_loss / len(dec_logits)
            ep_dec_acc += float(corr.float().mean())

            loss = bc.token_loss_weight * tok_loss + bc.decision_loss_weight * dec_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.net.parameters(), cfg.train.grad_clip)
            opt.step()

            ep_tok += float(tok_loss.detach())
            ep_dec += float(dec_loss.detach())
            nb += 1
        nb = max(nb, 1)
        tok_loss_val, dec_loss_val = ep_tok / nb, ep_dec / nb
        tok_acc, dec_acc = ep_tok_acc / nb, ep_dec_acc / nb
    agent.net.eval()

    # Hand the agent back a fresh RL optimiser -- the apprenticeship optimiser's
    # moments are about a different objective and should not carry over.
    agent.opt = torch.optim.Adam(agent.net.parameters(), lr=cfg.train.lr)

    info["final_token_loss"] = round(tok_loss_val, 4)
    info["final_decision_loss"] = round(dec_loss_val, 4)
    info["token_accuracy"] = round(tok_acc, 4)
    info["decision_accuracy"] = round(dec_acc, 4)
    return info
