"""The transmission bottleneck: iterated learning for newborns (spec 4).

When an agent dies its replacement does not inherit weights.  Instead it gets a
short supervised apprenticeship on a **deliberately small sample** of recent
successful trades by the living population, and only then joins the RL loop.

How much a newborn sees
-----------------------
Nearly all of it.  An earlier version drew a few hundred transcripts, which had
the asymmetry backwards: with a sample that small, a form used in 2% of trades
might appear a handful of times or not at all, so *common* vocabulary was at risk
of being lost, not just obscure vocabulary.  Real transmission does not look like
that.  Children reliably acquire essentially everything the adults around them
use with any regularity; loss and drift are marginal phenomena at the rare end.

So the sample is now a near-complete pass over the parent generation's recent
successful trades (``coverage``, default all of them).  The asymmetry then falls
out of the statistics instead of being imposed by a cap: a form used in 1% of
trades still appears hundreds of times in a 40,000-transcript sample and
transmits reliably, while one used in 0.01% may genuinely not appear at all, or
appear once and not be learned.  Only the second kind is at real risk, which is
the point.

Sampling stays proportional to how often each meaning actually came up
(``frequency_skew``, 1.0 = natural proportion), so the *composition* of a
newborn's experience still mirrors the parent generation's.  What changed is that
it is no longer artificially thin.

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
    f_dec: torch.Tensor      # (8,) deal + belief
    b_dec: torch.Tensor      # (8,)
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

    def meaning_of(self, batch, i: int) -> tuple[int, int]:
        """The (variety, quantity) this episode was about."""
        sb = batch.sb
        if sb is None:
            sc = batch.scenarios[i]
            return (sc.buyer.want_variety, sc.buyer.need_qty)
        if hasattr(sb, "want_variety"):                     # a trading batch
            return (int(sb.want_variety[i]), int(sb.need_qty[i]))
        m = sb.true_meaning[i]                              # a lineup round
        return (int(m[0]), int(m[1]))

    def _push(self, batch, i: int, farmers, buyers, episode: int) -> None:
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
            meaning=self.meaning_of(batch, i),
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
        self.total_added += 1

    def add_batch(self, batch: BatchRollout, farmers, buyers, episode: int) -> int:
        """File this batch's usable transcripts for the next generation to learn from.

        The tensor path carries its outcome in ``batch.res`` and builds no
        per-episode Outcome objects, so iterating ``batch.outcomes`` silently
        stored nothing at all -- every newborn then got an empty curriculum and
        started from random weights, which with turnover on kept resetting the
        population. Hence the explicit branch, and the test that guards it.
        """
        if batch.res is not None:
            keep = batch.res["success"]
            if not self.cfg.bottleneck.only_successful:
                keep = torch.ones_like(keep)
            idx = keep.nonzero(as_tuple=True)[0].tolist()
            for i in idx:
                self._push(batch, i, farmers, buyers, episode)
            return len(idx)

        added = 0
        for i, o in enumerate(batch.outcomes):
            if self.cfg.bottleneck.only_successful and not o.success:
                continue
            self._push(batch, i, farmers, buyers, episode)
            added += 1
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

    def word_coverage(self, cfg, items: list[StoredEpisode]) -> dict[str, Any]:
        """How much of the population's vocabulary this curriculum contains.

        This is the number that decides whether a form transmits.  A word the
        newborn never sees cannot be learned; a word it sees hundreds of times
        will be.  Reported per birth so the frequent/rare asymmetry is auditable
        rather than assumed.
        """
        from .env import parse_words, word_text
        def words_of(pool):
            c: Counter = Counter()
            for it in pool:
                for w in parse_words(cfg, [int(x) for x in it.tokens]):
                    c[word_text(cfg, w)] += 1
            return c
        shown = words_of(items)
        whole = words_of(self._buf)
        if not whole:
            return {}
        total = sum(whole.values())
        # a form is "common" if it is more than 1 in 1000 of all word tokens
        common = {w for w, n in whole.items() if n / total >= 1e-3}
        rare = set(whole) - common
        seen_enough = {w for w, n in shown.items() if n >= 3}
        return {
            "vocabulary_in_population": len(whole),
            "vocabulary_shown": len(shown),
            "common_forms": len(common),
            "common_forms_shown": len(common & set(shown)),
            "common_forms_learnable": len(common & seen_enough),
            "rare_forms": len(rare),
            "rare_forms_shown": len(rare & set(shown)),
            "common_coverage": (len(common & seen_enough) / len(common)) if common else 1.0,
            "rare_coverage": (len(rare & seen_enough) / len(rare)) if rare else 1.0,
        }

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
        "coverage": bc.coverage,
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

    # 0 means "derive from coverage"; a positive n_samples forces an explicit cap,
    # which is mostly useful for reproducing the old, lossy behaviour.
    want = (bc.n_samples if bc.n_samples > 0
            else min(bc.max_samples, max(1, int(round(bc.coverage * len(store))))))
    samples = store.sample(want, rng)
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
    info["store_coverage"] = len(samples) / max(1, len(store))
    info["epochs"] = bc.epochs
    info["frequency_skew"] = bc.frequency_skew
    info["meaning_coverage"] = store.meaning_profile(samples)
    info["word_coverage"] = store.word_coverage(cfg, samples)
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
