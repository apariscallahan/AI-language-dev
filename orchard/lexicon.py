"""Vocabulary analysis for the open channel (addendum section 3).

The original metric suite measures whether *a* code emerged.  These measure
whether what emerged looks like a *vocabulary*:

  3.1 length <-> frequency   -> :func:`length_frequency`
  3.2 word-like units        -> :func:`word_stats`
  3.3 rare vs frequent       -> :func:`bucketed_analysis`
  3.4 form survival          -> :class:`FormTracker`

All of it is measurement.  Nothing here tells an agent how to speak; every
"word" is whatever fell out of where the agent chose to put its space marks.

Probing convention
------------------
A meaning is a (variety, quantity) request -- the thing the buyer has to name.
To turn one into an actual observation the other two buyer fields have to be
filled in, and they are held at fixed reference values so that the probe is a
deterministic function of the meaning and stays comparable across checkpoints
and across generations.  Messages are decoded greedily for the same reason.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch

from .config import Config
from .env import BUYER, FARMER, parse_words, word_text
from .metrics import (_spearman, entropy_bits, levenshtein, normalised_levenshtein,
                      topographic_similarity, utterances_for_meanings)
from .population import Population
from .world import World


# ==========================================================================
# probing
# ==========================================================================
def reference_buyer_obs(cfg: Config, key: tuple[int, int]) -> tuple[int, ...]:
    """A buyer observation expressing meaning ``key``, with the other fields pinned."""
    from .world import n_obs_slots
    variety, qty = key
    ref_quality = 0
    ref_price = cfg.world.n_price_bins - 2
    vals = (variety, qty, ref_quality, ref_price)
    return tuple(vals) + (0,) * (n_obs_slots(cfg.world) - len(vals))


def probe_messages(cfg: Config, pop: Population, keys: Sequence[tuple[int, int]],
                   *, device: str = "cpu", agents: Optional[Sequence] = None
                   ) -> dict[tuple[int, int], list[list[int]]]:
    """For each meaning, every buyer's greedy utterance for it."""
    obs = [reference_buyer_obs(cfg, k) for k in keys]
    pool = agents if agents is not None else pop.pool(BUYER)
    out: dict[tuple[int, int], list[list[int]]] = {k: [] for k in keys}
    for agent in pool:
        msgs = utterances_for_meanings(cfg, agent, obs, device=device)
        for k, m in zip(keys, msgs):
            out[k].append(m)
    return out


def consensus_message(msgs: Sequence[Sequence[int]]) -> list[int]:
    """The population's most common form for a meaning (ties broken by first seen)."""
    if not msgs:
        return []
    counts = Counter(tuple(m) for m in msgs)
    return list(counts.most_common(1)[0][0])


# ==========================================================================
# 3.2  word-like units
# ==========================================================================
def word_stats(cfg: Config, batches: Sequence[Any]) -> dict[str, Any]:
    """Word inventory and shape, plus the degenerate-usage checks the addendum asks for."""
    c = cfg.channel
    words: Counter[tuple[int, ...]] = Counter()
    word_len: list[int] = []
    per_msg_words: list[int] = []
    per_msg_symbols: list[int] = []
    n_hyphen = n_space = n_atom = n_msgs = n_empty = 0
    maxed_out = 0

    for batch in batches:
        toks = batch.tokens
        B = toks.shape[0]
        for turn in range(c.n_turns):
            seg = toks[:, turn * c.max_symbols:(turn + 1) * c.max_symbols]
            for i in range(B):
                syms = [int(t) for t in seg[i] if int(t) != c.pad_id]
                emitted = [t for t in syms if c.costed(t)]
                n_msgs += 1
                per_msg_symbols.append(len(emitted))
                if not emitted:
                    n_empty += 1
                if len(emitted) >= c.max_symbols:
                    maxed_out += 1
                n_hyphen += sum(1 for t in emitted if t == c.hyphen_id)
                n_space += sum(1 for t in emitted if t == c.space_id)
                n_atom += sum(1 for t in emitted if c.is_atom(t))
                ws = parse_words(cfg, syms)
                per_msg_words.append(len(ws))
                for w in ws:
                    words[w] += 1
                    word_len.append(len(w))

    total_words = sum(words.values()) or 1
    multi = sum(n for w, n in words.items() if len(w) > 1)
    top = words.most_common(12)
    return {
        "distinct_words": len(words),
        "word_entropy_bits": entropy_bits(words.values()),
        "mean_word_len_atoms": sum(word_len) / len(word_len) if word_len else 0.0,
        "max_word_len_atoms": max(word_len) if word_len else 0,
        "multi_atom_word_share": multi / total_words,
        "mean_words_per_message": sum(per_msg_words) / len(per_msg_words) if per_msg_words else 0.0,
        "mean_symbols_per_message": (sum(per_msg_symbols) / len(per_msg_symbols)
                                     if per_msg_symbols else 0.0),
        "max_symbols_allowed": c.max_symbols,
        "at_length_cap_frac": maxed_out / max(n_msgs, 1),
        "silent_frac": n_empty / max(n_msgs, 1),
        "hyphen_share": n_hyphen / max(n_atom + n_hyphen + n_space, 1),
        "space_share": n_space / max(n_atom + n_hyphen + n_space, 1),
        "top_words": [{"word": word_text(cfg, w), "atoms": list(w), "count": n,
                       "share": n / total_words} for w, n in top],
        "word_counts": {word_text(cfg, w): n for w, n in words.items()},
    }


def word_usage_flags(cfg: Config, ws: dict[str, Any]) -> list[str]:
    """Distinguish the failure modes the addendum tells us not to conflate."""
    flags: list[str] = []
    if ws["hyphen_share"] < 0.01:
        flags.append("NO WORD FORMATION: the hyphen is essentially unused, so every "
                     "word is a single atom (acceptable, but multi-atom words never "
                     "emerged)")
    if ws["space_share"] < 0.01:
        flags.append("NO WORD SEGMENTATION: the space is essentially unused, so each "
                     "turn is one undivided word")
    if ws["at_length_cap_frac"] > 0.6:
        flags.append("LENGTH-CAP BABBLING: %.0f%% of utterances run to the cap; the "
                     "symbol cost is not biting" % (100 * ws["at_length_cap_frac"]))
    if ws["distinct_words"] > 0 and ws["word_entropy_bits"] < 1.0:
        flags.append("WORD COLLAPSE: word entropy %.2f bits -- effectively one or two "
                     "forms in use" % ws["word_entropy_bits"])
    return flags


# ==========================================================================
# 3.1  length <-> frequency
# ==========================================================================
def length_frequency(cfg: Config, pop: Population, world: World, *,
                     device: str = "cpu") -> dict[str, Any]:
    """Do commoner meanings get shorter messages?  (Addendum 2.2's prediction.)

    Reported as a correlation, not eyeballed: Spearman between how often a
    meaning occurs and the length of the message used for it, in symbols and
    again in words.  A negative correlation is the human pattern.
    """
    table = world.meaning_table()
    if len(table) < 6:
        return {"n": len(table)}
    keys = [k for k, _ in table]
    probs = [p for _, p in table]
    msgs = probe_messages(cfg, pop, keys, device=device)

    sym_len, word_len, rows = [], [], []
    for k, p in zip(keys, probs):
        forms = msgs[k]
        cons = consensus_message(forms)
        n_sym = sum(1 for t in cons if cfg.channel.costed(t))
        n_word = len(parse_words(cfg, cons))
        sym_len.append(float(n_sym))
        word_len.append(float(n_word))
        rows.append({"meaning": list(k), "prob": p, "symbols": n_sym, "words": n_word,
                     "form": " ".join(word_text(cfg, w) for w in parse_words(cfg, cons))})
    lp = [math.log(p) for p in probs]
    return {
        "n": len(keys),
        "rho_symbols": _spearman(lp, sym_len),
        "rho_words": _spearman(lp, word_len),
        "mean_symbols_frequent": _mean([r["symbols"] for r in rows[:max(1, len(rows) // 3)]]),
        "mean_symbols_rare": _mean([r["symbols"] for r in rows[-max(1, len(rows) // 3):]]),
        "mean_words_frequent": _mean([r["words"] for r in rows[:max(1, len(rows) // 3)]]),
        "mean_words_rare": _mean([r["words"] for r in rows[-max(1, len(rows) // 3):]]),
        "rows": rows,
    }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


# ==========================================================================
# 3.3  rare vs frequent buckets
# ==========================================================================
def split_meanings(world: World, quantile: float = 0.5
                   ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Frequent and rare halves of the meaning space, by cumulative probability."""
    table = world.meaning_table()
    if not table:
        return [], []
    total = sum(p for _, p in table)
    frequent, rare, acc = [], [], 0.0
    for k, p in table:
        if acc < quantile * total:
            frequent.append(k)
        else:
            rare.append(k)
        acc += p
    if not rare:
        rare = [table[-1][0]]
    if not frequent:
        frequent = [table[0][0]]
    return frequent, rare


def bucketed_analysis(cfg: Config, pop: Population, world: World, *,
                      device: str = "cpu", rng: Optional[random.Random] = None
                      ) -> dict[str, Any]:
    """Compositionality, agreement and length, computed separately per bucket.

    A global average hides precisely the effect the addendum predicts -- frequent
    meanings settling on short, possibly irregular forms while rare ones stay
    long, volatile and compositional -- so the split is the point.
    """
    rng = rng or random.Random(0)
    frequent, rare = split_meanings(world, cfg.log.rare_frequent_split)
    out: dict[str, Any] = {"n_frequent": len(frequent), "n_rare": len(rare)}
    for label, keys in (("frequent", frequent), ("rare", rare)):
        if len(keys) < 4:
            out[label] = {"n": len(keys)}
            continue
        msgs = probe_messages(cfg, pop, keys, device=device)
        meanings = [reference_buyer_obs(cfg, k) for k in keys]
        consensus = [consensus_message(msgs[k]) for k in keys]
        ts = topographic_similarity(meanings, consensus, cfg, BUYER, n_null=2, rng=rng)
        # agreement across the population for the same meaning
        spread = []
        for k in keys:
            forms = msgs[k]
            pairs = [normalised_levenshtein(forms[i], forms[j])
                     for i in range(len(forms)) for j in range(i + 1, len(forms))]
            if pairs:
                spread.append(sum(pairs) / len(pairs))
        out[label] = {
            "n": len(keys),
            "topsim": ts["topsim"],
            "null": ts["null_mean"],
            "coherence": 1.0 - _mean(spread) if spread else float("nan"),
            "mean_symbols": _mean([sum(1 for t in c if cfg.channel.costed(t))
                                   for c in consensus]),
            "mean_words": _mean([len(parse_words(cfg, c)) for c in consensus]),
        }
    return out


# ==========================================================================
# 3.4  generational form survival
# ==========================================================================
@dataclass
class FormEvent:
    episode: int
    meaning: list[int]
    bucket: str
    prob: float
    old_form: str
    new_form: str
    old_words: int
    new_words: int
    old_compositionality: float
    new_compositionality: float
    generations: list[int]
    regularised: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class FormTracker:
    """Follows the word used for specific meanings across generational turnover.

    Rare meanings are expected to be volatile: a newborn's bottleneck sample is
    dominated by common trades (addendum 2.3), so whatever idiosyncratic form a
    rare meaning picked up is often simply never shown to the next generation,
    and has to be rebuilt out of whatever parts *are* well attested.  When a form
    is replaced by one made of commoner words, that is the direct analogue of an
    irregular verb levelling out, and it is logged as such.
    """

    def __init__(self, cfg: Config, world: World):
        self.cfg = cfg
        self.world = world
        self.frequent, self.rare = split_meanings(world, cfg.log.rare_frequent_split)
        self.keys = self.frequent + self.rare
        self.bucket = {k: "frequent" for k in self.frequent}
        self.bucket.update({k: "rare" for k in self.rare})
        self.prob = {k: world.meaning_prob(k) for k in self.keys}
        self.history: dict[tuple[int, int], list[tuple[int, str]]] = defaultdict(list)
        self.events: list[FormEvent] = []
        self._last: dict[tuple[int, int], list[int]] = {}

    # ------------------------------------------------------------------
    def _compositionality(self, words: Sequence[tuple[int, ...]],
                          inventory: Counter) -> float:
        """How much of this form is built from parts that are common elsewhere."""
        if not words:
            return 0.0
        shared = sum(1 for w in words if inventory[w] >= 3)
        return shared / len(words)

    def observe(self, pop: Population, episode: int, *, device: str = "cpu"
                ) -> dict[str, Any]:
        msgs = probe_messages(self.cfg, pop, self.keys, device=device)
        consensus = {k: consensus_message(msgs[k]) for k in self.keys}

        inventory: Counter = Counter()
        for k in self.keys:
            for w in parse_words(self.cfg, consensus[k]):
                inventory[w] += 1

        gens = sorted({a.generation for a in pop.all_agents()})
        drift = {"frequent": [], "rare": []}
        changed = {"frequent": 0, "rare": 0}
        counted = {"frequent": 0, "rare": 0}

        for k in self.keys:
            cur = consensus[k]
            bucket = self.bucket[k]
            text = " ".join(word_text(self.cfg, w) for w in parse_words(self.cfg, cur))
            self.history[k].append((episode, text))
            prev = self._last.get(k)
            if prev is not None:
                d = normalised_levenshtein(prev, cur)
                drift[bucket].append(d)
                counted[bucket] += 1
                if prev != cur:
                    changed[bucket] += 1
                    old_words = parse_words(self.cfg, prev)
                    new_words = parse_words(self.cfg, cur)
                    oc = self._compositionality(old_words, inventory)
                    nc = self._compositionality(new_words, inventory)
                    if d > 0.34:          # a real replacement, not a one-symbol wobble
                        self.events.append(FormEvent(
                            episode=episode, meaning=list(k), bucket=bucket,
                            prob=self.prob[k],
                            old_form=" ".join(word_text(self.cfg, w) for w in old_words) or "<silence>",
                            new_form=text or "<silence>",
                            old_words=len(old_words), new_words=len(new_words),
                            old_compositionality=round(oc, 3),
                            new_compositionality=round(nc, 3),
                            generations=gens,
                            regularised=bool(nc > oc + 1e-9)))
            self._last[k] = cur

        return {
            "drift_frequent": _mean(drift["frequent"]),
            "drift_rare": _mean(drift["rare"]),
            "changed_frequent": changed["frequent"] / max(counted["frequent"], 1),
            "changed_rare": changed["rare"] / max(counted["rare"], 1),
            "n_events": len(self.events),
            "n_regularised": sum(1 for e in self.events if e.regularised),
            "generations_alive": gens,
        }

    # ------------------------------------------------------------------
    def report_rows(self, limit: int = 12) -> list[dict[str, Any]]:
        """The most interesting replacements: rare meanings rebuilt from common parts."""
        ev = sorted(self.events,
                    key=lambda e: (not e.regularised, e.bucket != "rare", e.prob))
        return [e.to_dict() for e in ev[:limit]]

    def timeline(self, n_meanings: int = 6) -> list[dict[str, Any]]:
        """A few meanings' forms over the whole run, commonest and rarest."""
        picks = self.frequent[:n_meanings // 2] + self.rare[-(n_meanings - n_meanings // 2):]
        out = []
        for k in picks:
            hist = self.history.get(k, [])
            trail = []
            for ep, text in hist:
                if not trail or trail[-1][1] != text:
                    trail.append((ep, text))
            out.append({"meaning": list(k), "bucket": self.bucket[k],
                        "prob": round(self.prob[k], 4),
                        "trail": [{"episode": e, "form": t or "<silence>"} for e, t in trail]})
        return out
