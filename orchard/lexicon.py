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
A trading meaning is a (fruit, quantity) pair; a naming meaning is a lot. To
turn one into an actual observation the remaining fields are held at fixed
reference values, so the probe is a deterministic function of the meaning and
stays comparable across checkpoints and generations. Messages are decoded
greedily for the same reason.

*Who* is probed depends on the phase. In the trading rungs it is the buyer, who
opens by naming what it wants. In the naming rungs it is whoever describes: both
roles, since the describer alternates. An earlier version always probed buyers
with a shopping-list observation, which in a rung where only farmers described
read "the population's form for this meaning" off agents that never spoke.

Greedy forms are the policy's *mode*. When a policy is still high-entropy the
mode can be one repeated symbol for a whole row of meanings while sampled play
varies -- so a greedy form table on its own is not evidence about what live
messages encode; :func:`live_encoding` measures that directly.
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
# The values the other fields of a lot are pinned at when a probe sweeps one.
REF_QTY = 3
REF_PRICE = 2


def reference_buyer_obs(cfg: Config, key: tuple[int, ...]) -> tuple[int, ...]:
    """A buyer observation expressing meaning ``key``, with the other fields pinned.

    A request is a lot -- (fruit, colour, quality, quantity, price) -- then the
    asked-about slot, "all of it". ``key`` may be a whole lot, a (fruit, colour,
    quality) combination, or a trading meaning (fruit, quantity).
    """
    from .world import QUERY_ALL, n_obs_slots
    w = cfg.world
    ref_price = min(REF_PRICE, w.n_price_bins - 1)
    ref_qty = min(REF_QTY, w.max_qty)
    if len(key) >= 5:
        fruit, colour, quality, qty, price = key[:5]
    elif len(key) >= 3:                 # a combination: pin quantity and price
        fruit, colour, quality = key[:3]
        qty, price = ref_qty, ref_price
    else:                               # a trading meaning: (fruit, quantity)
        fruit, qty = key[0], key[1]
        colour, quality, price = 0, 0, ref_price
    vals = (fruit, colour, quality, qty, price, QUERY_ALL)
    return tuple(vals) + (0,) * (n_obs_slots(cfg.world, cfg) - len(vals))


def reference_obs(cfg: Config, key: tuple[int, ...], phase=None) -> tuple[int, ...]:
    """The probe observation for ``key``: a shopping list, or a thing to describe.

    In a naming rung the key is the (fruit, colour, quality) combination itself,
    and the observation carries the field the rung asks about as well -- that is
    part of what the describer sees, so a probe without it would be asking a
    different question.
    """
    from .world import n_obs_slots
    if phase is None or not phase.tuples:
        return reference_buyer_obs(cfg, key)
    from .curriculum import ASK_ALL
    q = ASK_ALL
    if getattr(phase, "query", None) is not None:
        q = int(phase.query)
    lot = tuple(reference_buyer_obs(cfg, key)[:5])
    vals = lot + (q,)
    return vals + (0,) * (n_obs_slots(cfg.world, cfg) - len(vals))


def meaning_keys(cfg: Config, world: World, phase=None) -> list[tuple]:
    """The meanings a probe should sweep, for this rung.

    In a naming rung: every trained (fruit, colour, quality) combination at a
    reference quantity and price, plus one combination swept over every
    quantity and every price -- so the form for each field value is followed
    without probing all 3,000-odd lots.
    """
    if phase is not None and phase.tuples:
        w = cfg.world
        combos = [tuple(c) for c in world.holdout.training]
        ref_qty, ref_price = min(REF_QTY, w.max_qty), min(REF_PRICE, w.n_price_bins - 1)
        keys = [c + (ref_qty, ref_price) for c in combos]
        if combos:
            f0, c0, q0 = combos[0]
            keys += [(f0, c0, q0, n, ref_price) for n in range(w.max_qty + 1) if n != ref_qty]
            keys += [(f0, c0, q0, ref_qty, p) for p in range(w.n_price_bins) if p != ref_price]
        return keys
    return [k for k, _ in world.meaning_table()]


def probe_plan(cfg: Config, phase=None) -> list[tuple[int, Any]]:
    """(role, view) pairs whose first utterance is "the form" for a meaning."""
    if phase is None or not phase.tuples:
        from .curriculum import ladder
        return [(BUYER, phase if phase is not None else ladder(cfg)[-1])]
    out = []
    for role in (FARMER, BUYER):
        for v in phase.views():
            if v.speaks(cfg, role):
                out.append((role, v))
                break
    return out


def probe_messages(cfg: Config, pop: Population, keys: Sequence[tuple[int, int]],
                   *, device: str = "cpu", agents: Optional[Sequence] = None,
                   phase=None) -> dict[tuple[int, int], list[list[int]]]:
    """For each meaning, every describing agent's greedy utterance for it."""
    from .metrics import opening_context
    obs = [reference_obs(cfg, k, phase) for k in keys]
    out: dict[tuple[int, int], list[list[int]]] = {k: [] for k in keys}
    for role, view in probe_plan(cfg, phase):
        pool = (agents if agents is not None else pop.pool(role))
        if not pop.shared:        # one pool fills both seats below the trading rungs
            pool = [a for a in pool if a.role == role]
        ctx = opening_context(cfg, pop, None, view, device) if (
            view.turns_of(cfg, role) and view.turns_of(cfg, role)[0] > 0
            and view.tuples) else None
        for agent in pool:
            msgs = utterances_for_meanings(cfg, agent, obs, context=ctx, device=device,
                                           phase=view, role=role)
            if msgs is None:
                continue
            for k, m in zip(keys, msgs):
                out[k].append(m)
    return out


def live_encoding(cfg: Config, batches: Sequence[Any], field: int = 3,
                  given: int = 0, n_shuffles: int = 5, seed: int = 0) -> dict[str, Any]:
    """Does what speakers *actually said* carry a field, beyond another field?

    Plug-in mutual information between a message and a meaning field is badly
    inflated when most messages are unique, so this reports the excess over a
    shuffled null (the field permuted within each value of ``given``) -- the
    honest "bits the message carries about quantity once variety is known".
    Fields index a lot: 0 fruit, 1 colour, 2 quality, 3 quantity, 4 price.
    Computed on sampled play, not greedy probes.
    """
    rows: list[tuple[tuple[int, ...], int, int]] = []
    for batch in batches:
        ph = getattr(batch, "phase", None)
        sb = getattr(batch, "sb", None)
        if sb is None:
            continue
        for role in (FARMER, BUYER):
            pos = batch.own_positions(role)
            if not pos:
                continue
            if ph is not None and ph.tuples:
                if hasattr(sb, "meaning_of"):
                    mean = sb.meaning_of(role)
                elif getattr(sb, "informer", None) == role:
                    mean = sb.true_meaning
                else:
                    continue
            elif role == BUYER and hasattr(sb, "want_variety"):
                mean = sb.request                 # the buyer's request is a lot
            else:
                continue
            L = cfg.channel.max_msg_len
            first = pos[:L]
            toks = batch.tokens[:, first].tolist()
            for msg, m in zip(toks, mean.tolist()):
                u = tuple(t for t in msg if t != cfg.channel.pad_id)
                rows.append((u, int(m[given]), int(m[field])))
    if len(rows) < 50:
        return {"n": len(rows)}

    def mi(pairs):
        n = len(pairs)
        cx = Counter(a for a, _ in pairs)
        cy = Counter(b for _, b in pairs)
        cxy = Counter(pairs)
        return sum(c / n * math.log2((c / n) / ((cx[x] / n) * (cy[y] / n)))
                   for (x, y), c in cxy.items())

    rng = random.Random(seed)
    groups: dict[int, list[tuple[tuple[int, ...], int]]] = defaultdict(list)
    for u, g, f in rows:
        groups[g].append((u, f))
    real = null = 0.0
    for g, pairs in groups.items():
        w = len(pairs) / len(rows)
        real += w * mi(pairs)
        ys = [f for _, f in pairs]
        acc = 0.0
        for _ in range(n_shuffles):
            rng.shuffle(ys)
            acc += mi(list(zip([u for u, _ in pairs], ys)))
        null += w * acc / n_shuffles
    return {"n": len(rows), "mi_bits": real, "null_bits": null,
            "excess_bits": real - null}


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
        act = getattr(batch, "active", None)
        for turn in range(c.n_turns):
            seg = toks[:, turn * c.max_symbols:(turn + 1) * c.max_symbols]
            # A turn the phase never schedules is all PAD. Counting it as a
            # "silent message" padded every lineup statistic with phantom silence.
            spoken = (act[:, turn * c.max_symbols].tolist() if act is not None
                      else [True] * B)
            for i in range(B):
                if not spoken[i]:
                    continue
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


def role_word_counts(cfg: Config, batches: Sequence[Any]) -> dict[str, Counter]:
    """Word tokens each role actually emitted, over sampled play."""
    out = {"farmer": Counter(), "buyer": Counter()}
    for batch in batches:
        toks = batch.tokens.tolist()
        L = cfg.channel.max_msg_len
        for role, label in ((FARMER, "farmer"), (BUYER, "buyer")):
            pos = batch.own_positions(role)
            if not pos:
                continue
            turns = sorted({p // L for p in pos})
            for row in toks:
                for t in turns:
                    seg = [x for x in row[t * L:(t + 1) * L] if x != cfg.channel.pad_id]
                    for w in parse_words(cfg, seg):
                        out[label][word_text(cfg, w)] += 1
    return out


def cross_role_overlap(cfg: Config, batches: Sequence[Any],
                       shared_pool: bool = False) -> dict[str, Any]:
    """One community language, or two mutually foreign codes?

    ``shared_pool`` says the two seats are being filled from one pool of agents
    (everything below ``curriculum.split_roles_at``). There are no two codes to
    compare there -- the same agents speak in both seats -- so the question is
    reported as not yet askable rather than answered with a number that is
    really "does an agent use the same words in either chair". It read 0.87
    through the naming rungs of a run whose two founders shared no form at all.

    Computed over the words each role actually emitted in sampled play:

    * ``weighted_overlap`` -- histogram intersection of the two roles' word
      distributions, sum over words of min(p_farmer, p_buyer). 1.0 is one shared
      vocabulary used in the same proportions; 0.0 is two disjoint codes. This
      is the headline.
    * ``farmer_share_shared`` / ``buyer_share_shared`` -- the fraction of each
      role's word tokens that are forms the other role also uses. High on both
      with a modest type-level Jaccard is exactly "one language with
      role-specific jargon at the edges".
    * ``jaccard_types`` -- shared forms over all forms, unweighted, which the
      long tail of one-off coinages drags down.
    """
    counts = role_word_counts(cfg, batches)
    f, b = counts["farmer"], counts["buyer"]
    nf, nb = sum(f.values()), sum(b.values())
    out: dict[str, Any] = {"farmer_word_tokens": nf, "buyer_word_tokens": nb,
                           "farmer_types": len(f), "buyer_types": len(b),
                           "shared_pool": bool(shared_pool)}
    if shared_pool:
        out.update({"weighted_overlap": float("nan"), "jaccard_types": float("nan"),
                    "farmer_share_shared": float("nan"),
                    "buyer_share_shared": float("nan"),
                    "note": "one pool fills both seats: there are not two codes yet"})
        return out
    if not nf or not nb:
        out.update({"weighted_overlap": float("nan"), "jaccard_types": float("nan"),
                    "farmer_share_shared": float("nan"), "buyer_share_shared": float("nan"),
                    "note": "only one role spoke in the evaluated play"})
        return out
    shared = set(f) & set(b)
    out["shared_types"] = len(shared)
    out["jaccard_types"] = len(shared) / len(set(f) | set(b))
    out["weighted_overlap"] = sum(min(f[w] / nf, b[w] / nb) for w in shared)
    out["farmer_share_shared"] = sum(f[w] for w in shared) / nf
    out["buyer_share_shared"] = sum(b[w] for w in shared) / nb
    out["farmer_only_top"] = [w for w, _ in f.most_common() if w not in b][:5]
    out["buyer_only_top"] = [w for w, _ in b.most_common() if w not in f][:5]
    out["shared_top"] = [w for w, _ in (f + b).most_common() if w in shared][:8]
    return out


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
                     device: str = "cpu", phase=None) -> dict[str, Any]:
    """Do commoner meanings get shorter messages?  (Addendum 2.2's prediction.)

    Reported as a correlation, not eyeballed: Spearman between how often a
    meaning occurs and the length of the message used for it, in symbols and
    again in words.  A negative correlation is the human pattern.
    """
    if phase is not None and phase.tuples:
        # Things are drawn uniformly in the naming rungs, so there is no "commoner
        # meaning" for length to track. Saying so beats reporting a correlation
        # computed over a flat distribution.
        return {"n": 0, "rho_symbols": float("nan"), "rho_words": float("nan"),
                "note": "meanings are uniform in the naming rungs: nothing for "
                        "length to track"}
    table = world.meaning_table()
    if len(table) < 6:
        return {"n": len(table)}
    keys = [k for k, _ in table]
    probs = [p for _, p in table]
    msgs = probe_messages(cfg, pop, keys, device=device, phase=phase)

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
def split_meanings(world: World, quantile: float = 0.5, phase=None
                   ) -> tuple[list[tuple], list[tuple]]:
    """Frequent and rare halves of the meaning space, by cumulative probability.

    Things are drawn uniformly in the naming rungs, so there is no rare half to
    contrast: both lists come back empty and the analyses that use them say so
    rather than splitting a flat distribution down the middle.
    """
    if phase is not None and phase.tuples:
        return [], []
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
                      device: str = "cpu", rng: Optional[random.Random] = None,
                      phase=None) -> dict[str, Any]:
    """Compositionality, agreement and length, computed separately per bucket.

    A global average hides precisely the effect the addendum predicts -- frequent
    meanings settling on short, possibly irregular forms while rare ones stay
    long, volatile and compositional -- so the split is the point.
    """
    rng = rng or random.Random(0)
    frequent, rare = split_meanings(world, cfg.log.rare_frequent_split, phase)
    out: dict[str, Any] = {"n_frequent": len(frequent), "n_rare": len(rare)}
    for label, keys in (("frequent", frequent), ("rare", rare)):
        if len(keys) < 4:
            out[label] = {"n": len(keys)}
            continue
        msgs = probe_messages(cfg, pop, keys, device=device, phase=phase)
        meanings = [reference_obs(cfg, k, phase) for k in keys]
        consensus = [consensus_message(msgs[k]) for k in keys]
        from .metrics import phase_kinds
        role, view = probe_plan(cfg, phase)[0]
        ts = topographic_similarity(meanings, consensus, cfg, role, n_null=2, rng=rng,
                                    kinds=phase_kinds(cfg, role, view))
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
        # Run-level totals.  A single interval's drift is noisy and, at the final
        # checkpoint, can be a guaranteed zero; the honest summary of "how much did
        # this bucket move over the run" is the accumulation.
        self._drift_sum = {"frequent": 0.0, "rare": 0.0}
        self._drift_n = {"frequent": 0, "rare": 0}
        self._changed = {"frequent": 0, "rare": 0}
        # Who was probed. Crossing into a rung with different speakers is not a
        # form "changing": the comparison restarts, and the boundary is logged.
        self._regime: Optional[tuple] = None
        self.regime_changes: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _compositionality(self, words: Sequence[tuple[int, ...]],
                          inventory: Counter) -> float:
        """How much of this form is built from parts that are common elsewhere."""
        if not words:
            return 0.0
        shared = sum(1 for w in words if inventory[w] >= 3)
        return shared / len(words)

    def observe(self, pop: Population, episode: int, *, device: str = "cpu",
                phase=None) -> dict[str, Any]:
        regime = tuple((r, "tuple" if v.tuples else "trade")
                       for r, v in probe_plan(self.cfg, phase))
        # In a naming rung the meanings are the (fruit, colour, quality)
        # combinations themselves, and they are all equally common.
        if phase is not None and phase.tuples:
            keys = meaning_keys(self.cfg, self.world, phase)
            bucket = {k: "frequent" for k in keys}
            prob = {k: 1.0 / max(1, len(keys)) for k in keys}
        else:
            keys, bucket, prob = self.keys, self.bucket, self.prob
        if self._regime is not None and regime != self._regime:
            self._last = {}
            self.regime_changes.append({"episode": episode,
                                        "phase": phase.name if phase else "market"})
        self._regime = regime
        msgs = probe_messages(self.cfg, pop, keys, device=device, phase=phase)
        consensus = {k: consensus_message(msgs[k]) for k in keys}

        inventory: Counter = Counter()
        for k in keys:
            for w in parse_words(self.cfg, consensus[k]):
                inventory[w] += 1

        gens = sorted({a.generation for a in pop.all_agents()})
        drift = {"frequent": [], "rare": []}
        changed = {"frequent": 0, "rare": 0}
        counted = {"frequent": 0, "rare": 0}

        for k in keys:
            cur = consensus[k]
            b_of_k = bucket[k]
            text = " ".join(word_text(self.cfg, w) for w in parse_words(self.cfg, cur))
            self.history[k].append((episode, text))
            prev = self._last.get(k)
            if prev is not None:
                d = normalised_levenshtein(prev, cur)
                drift[b_of_k].append(d)
                counted[b_of_k] += 1
                if prev != cur:
                    changed[b_of_k] += 1
                    old_words = parse_words(self.cfg, prev)
                    new_words = parse_words(self.cfg, cur)
                    oc = self._compositionality(old_words, inventory)
                    nc = self._compositionality(new_words, inventory)
                    if d > 0.34:          # a real replacement, not a one-symbol wobble
                        self.events.append(FormEvent(
                            episode=episode, meaning=list(k), bucket=b_of_k,
                            prob=prob[k],
                            old_form=" ".join(word_text(self.cfg, w) for w in old_words) or "<silence>",
                            new_form=text or "<silence>",
                            old_words=len(old_words), new_words=len(new_words),
                            old_compositionality=round(oc, 3),
                            new_compositionality=round(nc, 3),
                            generations=gens,
                            regularised=bool(nc > oc + 1e-9)))
            self._last[k] = cur

        for b in ("frequent", "rare"):
            self._drift_sum[b] += sum(drift[b])
            self._drift_n[b] += counted[b]
            self._changed[b] += changed[b]

        def overall(b: str) -> float:
            n = self._drift_n[b]
            return self._drift_sum[b] / n if n else float("nan")

        return {
            # this checkpoint only
            "drift_frequent_interval": _mean(drift["frequent"]),
            "drift_rare_interval": _mean(drift["rare"]),
            # over the whole run so far -- what the report should quote
            "drift_frequent": overall("frequent"),
            "drift_rare": overall("rare"),
            "changed_frequent": self._changed["frequent"] / max(self._drift_n["frequent"], 1),
            "changed_rare": self._changed["rare"] / max(self._drift_n["rare"], 1),
            "n_events": len(self.events),
            "n_events_frequent": sum(1 for e in self.events if e.bucket == "frequent"),
            "n_events_rare": sum(1 for e in self.events if e.bucket == "rare"),
            "n_regularised": sum(1 for e in self.events if e.regularised),
            "n_comparisons": self._drift_n["frequent"] + self._drift_n["rare"],
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


# ==========================================================================
# where a word came from: the referential phase, or the negotiation phases
# ==========================================================================
class WordProvenance:
    """Tracks which curriculum phase each word first appeared and settled in.

    Some of the structure visible at the end of a curriculum run is simply
    inherited from the lineup game -- the agents already had words for varieties
    and quantities before a price was ever mentioned. Saying "negotiation produced
    this" without checking would be claiming a cause that is not there. So every
    word is stamped with the phase it was first seen in and the phase it became
    regular in, and the report separates the two populations.
    """

    def __init__(self, settle_count: int = 20):
        self.settle_count = settle_count
        self.first_phase: dict[str, str] = {}
        self.first_episode: dict[str, int] = {}
        self.counts_by_phase: dict[str, Counter] = defaultdict(Counter)
        self.settled_phase: dict[str, str] = {}

    def observe(self, phase_name: str, episode: int, word_counts: dict[str, int]) -> None:
        for word, n in word_counts.items():
            if word not in self.first_phase:
                self.first_phase[word] = phase_name
                self.first_episode[word] = episode
            self.counts_by_phase[phase_name][word] += int(n)
            if (word not in self.settled_phase
                    and self.counts_by_phase[phase_name][word] >= self.settle_count):
                self.settled_phase[word] = phase_name

    # ------------------------------------------------------------------
    def summary(self, phases: Sequence[str]) -> dict[str, Any]:
        total_by_phase = {p: sum(self.counts_by_phase[p].values()) for p in phases}
        out: dict[str, Any] = {"phases": list(phases), "n_words": len(self.first_phase)}
        for p in phases:
            first = [w for w, f in self.first_phase.items() if f == p]
            settled = [w for w, f in self.settled_phase.items() if f == p]
            out[p] = {
                "first_appeared": len(first),
                "settled_here": len(settled),
                "word_tokens": total_by_phase[p],
            }
        return out

    def new_in_phase(self, phase_name: str, limit: int = 15) -> list[dict[str, Any]]:
        """Words that first showed up in this phase, commonest first.

        For the negotiation phases this is the place to look for anything like
        offer / counter-offer / accept / reject vocabulary that the lineup game
        had no reason to invent.
        """
        rows = [(w, self.counts_by_phase[phase_name][w])
                for w, f in self.first_phase.items() if f == phase_name]
        rows.sort(key=lambda kv: -kv[1])
        return [{"word": w, "count": n, "settled": self.settled_phase.get(w) == phase_name,
                 "first_episode": self.first_episode[w]} for w, n in rows[:limit]]

    def inherited(self, later_phase: str, earlier: Sequence[str] = ()) -> dict[str, Any]:
        """How much of a rung's vocabulary was already in use before it.

        ``earlier`` is every rung that came before this one. It used to default to
        one rung called "refer", which stopped existing three renames ago, so
        every rung reported that none of its words were inherited -- the opposite
        of what the run was doing, and the exact question the report asks.
        """
        used_later = set(self.counts_by_phase[later_phase])
        before = set(earlier)
        from_earlier = {w for w in used_later if self.first_phase.get(w) in before}
        return {
            "words_in_use": len(used_later),
            "inherited": len(from_earlier),
            "inherited_from": sorted(before),
            "new_here": len(used_later) - len(from_earlier),
            "inherited_share": (len(from_earlier) / len(used_later)) if used_later else 0.0,
        }
