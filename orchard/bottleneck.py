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

Where the squeeze comes from: the newborn is an apprentice, not a clone.
It sees transcripts, never weights, for a few epochs, and must rebuild the
parents' code from the forms that actually occur in them -- and a share of the
(fruit, colour, quality) *meanings* in the store (``meaning_holdout``) is kept
from it entirely, so those it has to put together from parts it did see. That
is the bottleneck of iterated learning: a language transmits intact only if it
is made of reusable parts. A positive ``n_samples`` restores a hard cap for the
Kirby-style on/off comparison (spec 9); the report states which regime a run
was in rather than assuming.

What the newborn learns is standard cross-entropy:
  * its own message tokens, teacher-forced against what the retiring generation
    said in the same position of the same conversation, and
  * the decisions its role actually made in that phase, against what that
    generation decided.
It never sees the other party's private observation -- only its own half of the
transcript, exactly as in live play.

Whose half is whose depends on the phase
----------------------------------------
Each stored transcript remembers the curriculum phase it was played in. "My
tokens" are the slots *this role* produced under *that phase's* speaking order,
read with that phase's observation layout, and the decision targets are only the
heads that phase scores for this role. An earlier version used the trading
task's buyer-opens order for everything, so in the lineup game -- where the
farmer describes first -- a farmer newborn's targets were the empty slots of a
turn nobody spoke (token accuracy 0.000 at every farmer birth), and a buyer
newborn was trained to imitate the *farmer's* words.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from collections import Counter

from .agents import Agent
from .config import Config
from .env import BUYER, FARMER, MASKED, grammar_mask_for_positions
from .rollout import BatchRollout


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
    meaning: tuple = (0, 0, 0)          # the (fruit, colour, quality) it was about
    phase: Any = None                   # the curriculum phase it was played in

    def obs_for(self, role: int) -> torch.Tensor:
        return self.f_obs if role == FARMER else self.b_obs

    def dec_for(self, role: int) -> torch.Tensor:
        return self.f_dec if role == FARMER else self.b_dec

    def generation_of(self, role: int) -> int:
        return self.f_generation if role == FARMER else self.b_generation


class _HostBatch:
    """A host-side slice of a batch that looks enough like one for ``_push``."""

    def __init__(self, cpu: dict, batch):
        self.__dict__.update(cpu)
        self._batch = batch
        self.row = (0, 0)
        self.meanings = None
        self.phase = getattr(batch, "phase", None)

    @property
    def sb(self):
        return self._batch.sb

    @property
    def scenarios(self):
        return self._batch.scenarios


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

    def meaning_of(self, batch, i: int) -> tuple:
        """The (fruit, colour, quality) combination this episode was about.

        The combination is the unit the held-out set is drawn over, so it is
        also the unit a newborn's apprenticeship can withhold
        (``bottleneck.meaning_holdout``).
        """
        if isinstance(batch, _HostBatch):
            if batch.meanings is not None:
                m = batch.meanings[batch.row[0]]
                return tuple(int(x) for x in m[:3])
            i = batch.row[1]                 # the episode's index in the full batch
        sb = batch.sb
        if sb is None:
            b = batch.scenarios[i].buyer
            return (b.want_variety, b.want_color, b.min_quality)
        if hasattr(sb, "want_variety"):                     # a trading batch
            return (int(sb.want_variety[i]), int(sb.want_color[i]), int(sb.min_quality[i]))
        m = sb.true_meaning[i]                              # a lineup / mutual round
        return tuple(int(x) for x in m[:3])

    def _phase_of(self, batch):
        from .curriculum import ladder
        ph = getattr(batch, "phase", None)
        return ph if ph is not None else ladder(self.cfg)[-1]

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
            phase=self._phase_of(batch),
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
            idx = keep.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                return 0
            # Everything the store keeps, gathered and moved to the host in one go.
            # Cloning six tiny tensors per episode on a GPU was thousands of kernel
            # launches per batch at cloud batch sizes.
            cpu = {k: getattr(batch, k)[idx].detach().cpu()
                   for k in ("f_obs", "b_obs", "tokens", "active", "f_dec", "b_dec",
                             "f_idx", "b_idx")}
            rows = idx.tolist()
            view = _HostBatch(cpu, batch)
            sb = batch.sb
            if sb is not None and hasattr(sb, "want_variety"):
                view.meanings = torch.stack([sb.want_variety[idx], sb.want_color[idx],
                                             sb.min_quality[idx]], dim=1).cpu().tolist()
            elif sb is not None:
                view.meanings = sb.true_meaning[idx][:, :3].cpu().tolist()
            for j, i in enumerate(rows):
                view.row = (j, i)
                self._push(view, j, farmers, buyers, episode)
            return len(rows)

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
    # The bottleneck proper: a share of the *meanings* in the store is withheld
    # from this newborn altogether, so it has to reconstruct those from parts
    # it did see. Seeing every meaning is a near-clone; a language whose forms
    # only survive when every combination is shown is not a compositional one,
    # and this is the pressure iterated learning is known to exert.
    withheld: set = set()
    if bc.meaning_holdout > 0 and samples:
        kinds = sorted({s.meaning for s in samples})
        k = int(round(bc.meaning_holdout * len(kinds)))
        if 0 < k < len(kinds):
            withheld = set(rng.sample(kinds, k))
            samples = [s for s in samples if s.meaning not in withheld]
    info["withheld_meanings"] = len(withheld)
    info["withheld_share"] = bc.meaning_holdout if withheld else 0.0
    info["withheld"] = [list(m) for m in sorted(withheld)][:24]
    if len(samples) < 8:
        info["skipped"] = "not enough successful transcripts yet"
        return info

    role = agent.role
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

    # One group per phase (and per describer, in the swap rung): each has its own
    # speaking order, observation layout and scored heads.
    from .curriculum import ladder, phase_schema
    groups: dict[Any, list[StoredEpisode]] = {}
    for s in samples:
        groups.setdefault(s.phase if s.phase is not None else ladder(cfg)[-1], []).append(s)

    plans = []
    for ph, items in groups.items():
        own_pos = ph.own_positions(cfg, role)
        heads = ph.active_heads(role, cfg)
        if not own_pos and not heads:
            continue                     # this role neither spoke nor decided here
        obs = torch.stack([s.obs_for(role) for s in items]).to(device)
        toks = torch.stack([s.tokens for s in items]).to(device)
        act = torch.stack([s.active for s in items]).to(device)
        dec = torch.stack([s.dec_for(role) for s in items]).to(device)
        if own_pos:
            tgt = toks[:, own_pos]
            m = act[:, own_pos]
            safe = torch.where(m, tgt, torch.zeros_like(tgt))
            gram = grammar_mask_for_positions(cfg, toks, own_pos)
            # A lesson the grammar now forbids is not taught. The store can hold
            # transcripts from before a change to the medium -- a resumed run
            # carries silent turns from before silence was ruled out -- and its
            # target would sit on a masked logit, where the cross-entropy is ~1e9.
            m = m & gram.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        else:
            tgt = m = safe = gram = None
        plans.append({
            "phase": ph, "obs": obs, "toks": toks, "dec": dec, "heads": heads,
            "targets": tgt, "mask": m, "safe": safe, "grammar": gram, "n": len(items),
            "read_pos": ph.read_positions(cfg, role, device) if own_pos else None,
            "schema": phase_schema(cfg, role, ph),
            "self_mask": ph.self_mask(cfg, role, device),
        })
    info["phases_in_curriculum"] = {
        ("%s/%s-describes" % (p["phase"].name, "farmer" if p["phase"].informer == FARMER
                              else "buyer") if p["phase"].swaps else p["phase"].name): p["n"]
        for p in plans}
    info["own_token_targets"] = int(sum(int(p["mask"].sum()) for p in plans
                                        if p["mask"] is not None))
    if not plans:
        info["skipped"] = "nothing in the store that this role said or decided"
        return info

    opt = torch.optim.Adam(agent.net.parameters(), lr=bc.lr)
    tok_loss_val = dec_loss_val = 0.0
    tok_acc = dec_acc = None

    agent.net.train()
    for _ in range(bc.epochs):
        # every minibatch of every group, in one shuffled order
        work = []
        for gi, p in enumerate(plans):
            order = list(range(p["n"]))
            rng.shuffle(order)
            for start in range(0, p["n"], bc.batch_size):
                work.append((gi, order[start:start + bc.batch_size]))
        rng.shuffle(work)
        ep_tok = ep_dec = 0.0
        n_tok_b = n_dec_b = 0
        tok_hit = tok_n = 0.0
        dec_hit = dec_n = 0.0
        for gi, idx in work:
            p = plans[gi]
            sel = torch.tensor(idx, dtype=torch.long, device=device)
            read_pos = (p["read_pos"] if p["read_pos"] is not None
                        else torch.zeros(0, dtype=torch.long, device=device))
            tok_logits, _, dec_logits, _ = agent.net.full_pass(
                p["obs"][sel], p["toks"][sel], read_pos, schema=p["schema"],
                self_mask=p["self_mask"])
            loss = torch.zeros((), device=device)
            if p["mask"] is not None:
                m = p["mask"][sel]
                denom = m.sum().clamp(min=1)
                # the same word grammar the speaker is held to in live play
                tok_logits = tok_logits.masked_fill(~p["grammar"][sel], MASKED)
                ce = F.cross_entropy(
                    tok_logits.reshape(-1, tok_logits.shape[-1]),
                    p["safe"][sel].reshape(-1), reduction="none").reshape(m.shape)
                tok_loss = (ce * m).sum() / denom
                pred = tok_logits.argmax(-1)
                tok_hit += float(((pred == p["targets"][sel]) & m).sum())
                tok_n += float(m.sum())
                loss = loss + bc.token_loss_weight * tok_loss
                ep_tok += float(tok_loss.detach())
                n_tok_b += 1
            if p["heads"]:
                d = p["dec"][sel]
                dl = torch.zeros((), device=device)
                corr = torch.ones(sel.shape[0], dtype=torch.bool, device=device)
                for col in p["heads"]:
                    lg = dec_logits[col]
                    dl = dl + F.cross_entropy(lg, d[:, col])
                    corr &= (lg.argmax(-1) == d[:, col])
                dl = dl / len(p["heads"])
                dec_hit += float(corr.float().sum())
                dec_n += float(sel.shape[0])
                loss = loss + bc.decision_loss_weight * dl
                ep_dec += float(dl.detach())
                n_dec_b += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.net.parameters(), cfg.train.grad_clip)
            opt.step()
        tok_loss_val = ep_tok / max(1, n_tok_b)
        dec_loss_val = ep_dec / max(1, n_dec_b)
        tok_acc = (tok_hit / tok_n) if tok_n else None
        dec_acc = (dec_hit / dec_n) if dec_n else None
    agent.net.eval()

    # Hand the agent back a fresh RL optimiser -- the apprenticeship optimiser's
    # moments are about a different objective and should not carry over.
    agent.opt = torch.optim.Adam(agent.net.parameters(), lr=cfg.train.lr)

    # None, not 0.0, when there was nothing of that kind to learn: a buyer born
    # during ``refer`` never speaks there, and a zero would read as a failure.
    info["final_token_loss"] = round(tok_loss_val, 4) if tok_acc is not None else None
    info["final_decision_loss"] = round(dec_loss_val, 4) if dec_acc is not None else None
    info["token_accuracy"] = round(tok_acc, 4) if tok_acc is not None else None
    info["decision_accuracy"] = round(dec_acc, 4) if dec_acc is not None else None
    return info
