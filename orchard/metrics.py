"""Metrics and analysis (spec 5), built alongside the simulation rather than after.

Without these you cannot tell an emergent grammar from noise that happens to
correlate with reward, so every one of spec 5's six measurements is implemented
here and computed at every checkpoint:

  5.1 task success rate            -> :class:`RollingStat`, :func:`evaluate_success`
  5.2 compositionality (topsim)    -> :func:`topographic_similarity`
  5.3 vocabulary usage             -> :func:`vocab_stats`
  5.4 stability over time          -> :class:`StabilityTracker`
  5.5 cross-generation intelligibility -> :func:`intelligibility`
  5.6 zero-shot generalisation     -> :func:`zero_shot`

Plus the post-hoc token->meaning analysis (:class:`TokenSemantics`) that spec 6.3
allows for *annotating* transcripts after the fact.  Nothing here assigns meaning
in advance; it only measures correlations that training produced.

Sampling convention: measurements of the meaning->message *mapping* (topsim,
stability, semantics) use greedy decoding so the mapping is a function; measures
of *performance* (success, intelligibility, zero-shot) use sampling, matching the
conditions the agents were actually trained under.
"""
from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

try:
    from scipy.stats import spearmanr as _spearmanr
except Exception:                                    # pragma: no cover
    _spearmanr = None

from .agents import Agent, dialogue_offset
from .config import Config
from .env import BUYER, FARMER, buyer_obs, farmer_obs, obs_for, speaker_of_turn
from .population import Population
from .rollout import run_episodes
from .world import (K_COLOR, K_EMPTY, K_FIELD, K_PRICE, K_QTY, K_QUALITY, K_VARIETY,
                    KIND_NAMES,
                    Scenario, World, field_labels, field_spans, obs_schema)


# ==========================================================================
# small utilities
# ==========================================================================
class RollingStat:
    def __init__(self, window: int = 2000):
        self.buf: deque[float] = deque(maxlen=window)

    def add(self, x: float) -> None:
        self.buf.append(float(x))

    def extend(self, xs: Iterable[float]) -> None:
        for x in xs:
            self.buf.append(float(x))

    @property
    def mean(self) -> float:
        return sum(self.buf) / len(self.buf) if self.buf else 0.0

    def __len__(self) -> int:
        return len(self.buf)


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalised_levenshtein(a: Sequence[int], b: Sequence[int]) -> float:
    m = max(len(a), len(b))
    return levenshtein(a, b) / m if m else 0.0


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho, with a dependency-free fallback if scipy is unavailable."""
    if len(x) < 3:
        return float("nan")
    if _spearmanr is not None:
        # A constant input (every message identical, e.g. before training does
        # anything) has no defined correlation; NaN is the right answer and the
        # callers already handle it, so the warning is just noise.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = _spearmanr(x, y).statistic
        return float(rho) if rho == rho else float("nan")

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def entropy_bits(counts: Iterable[int]) -> float:
    cs = [c for c in counts if c > 0]
    total = sum(cs)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in cs)


# ==========================================================================
# meaning-space distances  (spec 5.2)
# ==========================================================================
def meaning_distance(a: Sequence[int], b: Sequence[int], cfg: Config, role: int,
                     metric: str = "hamming", kinds: Optional[Sequence[int]] = None
                     ) -> float:
    """Distance between two private states, over that role's real fields only.

    ``kinds`` overrides the trading-task schema -- a lineup describer's meaning
    is a (variety, quantity, quality) tuple, not a shopping list.
    """
    kinds = list(kinds) if kinds is not None else obs_schema(cfg.world, role)
    if metric == "hamming":
        return float(sum(1 for x, y, k in zip(a, b, kinds)
                         if k not in (K_EMPTY, K_FIELD) and x != y))
    if metric == "l1":
        # keeps the ordinal structure of quantity and price that Hamming discards
        spans = _spans_for(cfg, kinds)
        return float(sum(abs(x - y) / sp
                         for x, y, k, sp in zip(a, b, kinds, spans)
                         if k not in (K_EMPTY, K_FIELD)))
    raise ValueError("unknown meaning metric %r" % (metric,))


def _spans_for(cfg: Config, kinds: Sequence[int]) -> list[int]:
    w = cfg.world
    from .world import field_spans_by_kind
    spans = field_spans_by_kind(w)
    return [spans[k] for k in kinds]


def topographic_similarity(meanings: Sequence[Sequence[int]],
                           messages: Sequence[Sequence[int]],
                           cfg: Config, role: int, *, metric: str = "hamming",
                           n_null: int = 3, rng: Optional[random.Random] = None,
                           kinds: Optional[Sequence[int]] = None
                           ) -> dict[str, float]:
    """Brighton & Kirby topological similarity.

    Spearman correlation between pairwise distance in meaning space and pairwise
    distance in message space.  High positive rho = similar meanings get similar
    messages = compositional structure.  A shuffled-message null is reported
    alongside so a small positive rho is not over-read.
    """
    n = len(meanings)
    if n < 4:
        return {"topsim": float("nan"), "null_mean": float("nan"),
                "null_std": float("nan"), "z": float("nan"), "n": n}
    md: list[float] = []
    sd: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            md.append(meaning_distance(meanings[i], meanings[j], cfg, role, metric, kinds))
            sd.append(float(levenshtein(messages[i], messages[j])))
    rho = _spearman(md, sd)

    rng = rng or random.Random(0)
    nulls = []
    idx = list(range(n))
    for _ in range(max(0, n_null)):
        rng.shuffle(idx)
        perm = [messages[k] for k in idx]
        nd = []
        for i in range(n):
            for j in range(i + 1, n):
                nd.append(float(levenshtein(perm[i], perm[j])))
        nulls.append(_spearman(md, nd))
    nulls = [v for v in nulls if v == v]
    nm = sum(nulls) / len(nulls) if nulls else 0.0
    ns = (math.sqrt(sum((v - nm) ** 2 for v in nulls) / len(nulls)) if len(nulls) > 1 else 0.0)
    z = (rho - nm) / ns if ns > 1e-9 and rho == rho else float("nan")
    return {"topsim": rho, "null_mean": nm, "null_std": ns, "z": z, "n": n}


# ==========================================================================
# generating an agent's utterance for a given meaning (greedy = deterministic)
# ==========================================================================
@torch.no_grad()
def greedy_turn(cfg: Config, agent: Agent, obs: torch.Tensor, tokens: torch.Tensor,
                turn: int, schema=None, self_mask=None) -> torch.Tensor:
    """Fill dialogue ``turn`` for a whole batch by greedy decoding.  Mutates ``tokens``."""
    c = cfg.channel
    B = obs.shape[0]
    alive = torch.ones(B, dtype=torch.bool, device=obs.device)
    for k in range(c.max_msg_len):
        if not bool(alive.any()):
            break
        p = turn * c.max_msg_len + k
        logits, _ = agent.net.next_token_logits(obs, tokens, dialogue_offset(cfg) + p,
                                                schema=schema, self_mask=self_mask)
        from .env import MASKED, grammar_allowed
        allowed = grammar_allowed(cfg, tokens[:, p - 1] if k > 0 else tokens[:, p], k)
        tok = logits.masked_fill(~allowed, MASKED).argmax(dim=-1)
        tok = torch.where(alive, tok, torch.full_like(tok, c.pad_id))
        tokens[:, p] = tok
        alive = alive & (tok != c.eos_id)
    return tokens


def _strip(cfg: Config, row: Sequence[int]) -> list[int]:
    """Drop the padding from one already-on-host row of symbols.

    ``row`` must not be a device tensor: iterating one element by element
    synchronises the device on every symbol. Callers with a batch bring the
    whole thing across once (see :func:`utterances_for_meanings`).
    """
    return [int(t) for t in row if int(t) != cfg.channel.pad_id]


@torch.no_grad()
def utterances_for_meanings(cfg: Config, agent: Agent, meanings: Sequence[Sequence[int]],
                            *, context: Optional[torch.Tensor] = None,
                            device: str = "cpu", phase=None,
                            role: Optional[int] = None) -> Optional[list[list[int]]]:
    """This agent's own first utterance for each meaning, greedily decoded.

    The first speaker's utterance is a pure function of its private observation.
    A later speaker's also depends on what was said to it, so ``context``
    supplies one fixed opening for every probe -- holding the conversation
    constant so the variation measured is the probed agent's own.

    ``role`` is the seat being probed, which is not always the agent's own role:
    below the trading rungs one pool of agents fills both seats, so the seat the
    view describes from is what decides the schema and whose words are whose.

    ``phase`` decides who speaks when, what the observation slots mean, and which
    slots are "mine". Without it this is the trading task (buyer opens). Returns
    ``None`` if the agent's role does not speak in that phase: a lineup guesser
    has no utterance to probe, and pretending otherwise is how per-meaning forms
    got read off agents that had never said a word in that rung.
    """
    from .curriculum import ladder, phase_schema
    c = cfg.channel
    ph = phase if phase is not None else ladder(cfg)[-1]
    seat = agent.role if role is None else role
    turns = ph.turns_of(cfg, seat)
    if not turns:
        return None
    turn = turns[0]
    n = len(meanings)
    L = c.max_msg_len
    obs = torch.tensor([list(m) for m in meanings], dtype=torch.long, device=device)
    tokens = torch.full((n, c.dialogue_len), c.pad_id, dtype=torch.long, device=device)
    if turn > 0:
        if context is None:
            context = torch.full((turn * L,), c.pad_id, dtype=torch.long, device=device)
            context[0] = c.eos_id
        tokens[:, :turn * L] = context[:turn * L].unsqueeze(0)
    greedy_turn(cfg, agent, obs, tokens, turn, schema=phase_schema(cfg, seat, ph),
                self_mask=ph.self_mask(cfg, seat, device))
    # One transfer for the batch. Per row it was one device synchronisation per
    # symbol -- 24 of them per probe, times every probe and every agent, at
    # every promotion check.
    pad = c.pad_id
    return [[t for t in row if t != pad]
            for row in tokens[:, turn * L:(turn + 1) * L].tolist()]


def phase_kinds(cfg: Config, role: int, phase=None) -> list[int]:
    from .curriculum import ladder, phase_schema
    return phase_schema(cfg, role, phase if phase is not None else ladder(cfg)[-1])


def phase_labels(cfg: Config, role: int, phase=None) -> list[str]:
    """Human names for a speaker's observation slots in this phase."""
    if phase is not None and phase.tuples:
        names = ["variety", "quantity", "quality"]
        kinds = phase_kinds(cfg, role, phase)
        return names + ["-"] * (len(kinds) - len(names))
    return field_labels(cfg.world, role)


def tuple_meanings(cfg: Config, n: int, seed: int = 0, phase=None,
                   held_out: bool = False, query: Optional[int] = None
                   ) -> list[tuple[int, ...]]:
    """(fruit, colour, quality) things to describe, plus the field being asked about.

    The describer's observation in a naming rung is the thing *and* the query, so
    a probe that left the query out would be asking about the wrong rung: in
    ``name-color`` every probe has to say "colour" for the answer to mean
    anything.

    ``query`` fixes the question every probe asks.  The structure metrics pass
    the rung's own ``primary`` kind, because they score a message against the
    *whole* tuple and have no way to know a probe only asked for one field of
    it: on a mixed rung like ``name-all`` (70% whole things, 30% single fields)
    a perfect describer answers 30% of the probes with one word, and those
    answers are read as two fields randomly dropped.  Left to the mixture, that
    alone caps a flawless speaker's field coverage at 0.7 of the estimator's own
    ceiling.  ``None`` keeps the rung's mixture, which is what the report wants
    when it is describing what the rung actually plays.
    """
    from .curriculum import ASK_ALL, ReferentialWorld
    from .world import n_obs_slots
    g = torch.Generator()
    g.manual_seed(seed)
    rw = ReferentialWorld(cfg, device="cpu", generator=g)
    rows = rw._draw(n, held_out=held_out).tolist()
    width = n_obs_slots(cfg.world, cfg)
    if query is not None:
        asks = [int(query)] * n
    else:
        mix = getattr(phase, "mix", None) or (0.0, 0.0, 0.0, 1.0)
        total = float(sum(mix)) or 1.0
        # The probe asks the fields the rung asks, as often as the rung asks them.
        asks = [k for k, w in enumerate(mix) for _ in range(int(round(n * w / total)))]
        asks = (asks + [ASK_ALL] * n)[:n]
        perm = torch.randperm(n, generator=g).tolist()
        asks = [asks[i] for i in perm]
    out = []
    for i, r in enumerate(rows):
        row = tuple(r) + (asks[i],)
        out.append(row + (0,) * (width - len(row)))
    return out


def probe_query(phase) -> Optional[int]:
    """What goes in the query slot of a structure probe on this rung.

    A naming rung is promoted on the kind of round it introduces -- everything
    else about it is rehearsal -- so its describer's code is measured on that
    kind too, and the probe asks ``phase.primary``.

    ``mutual`` has the same observation *schema* (a tuple and a query slot) but
    never fills the query in: :meth:`MutualBatch.obs` pads it with zeros,
    because both sides simply hold a thing and nobody is asked about a field.
    The probes were writing ASK_ALL there, which is a value that slot never
    takes in training, and ``K_FIELD`` has its own embedding table -- so every
    structure number on ``mutual`` was read off an observation the speaker had
    never been trained on. The probe feeds what the rung feeds.

    ``None`` outside the tuple rungs, where there is no query slot at all.
    """
    if phase is None or not getattr(phase, "tuples", False):
        return None
    if getattr(phase, "mutual", False):
        return 0
    return int(getattr(phase, "primary", 3))


def sample_meanings(cfg: Config, world: World, role: int, n: int,
                    *, held_out: bool | None = False, phase=None,
                    seed: Optional[int] = None,
                    query: Optional[int] = None) -> list[tuple[int, ...]]:
    if phase is not None and phase.tuples:
        return tuple_meanings(cfg, n, seed if seed is not None
                              else random.Random().randrange(1 << 30),
                              phase=phase, held_out=bool(held_out), query=query)
    from .world import n_obs_slots
    width = n_obs_slots(cfg.world, cfg)
    out = []
    for _ in range(n):
        sc = world.sample(held_out=held_out)
        o = tuple(obs_for(role, sc, cfg))
        out.append(o + (0,) * max(0, width - len(o)))
    return out


@torch.no_grad()
def farmer_context_message(cfg: Config, pop: Population, world: World,
                           device: str = "cpu") -> torch.Tensor:
    """A single fixed Buyer opening, used as the constant context for Farmer probes."""
    c = cfg.channel
    buyer = pop.buyers[0]
    sc = world.sample(held_out=False)
    obs = torch.tensor([buyer_obs(sc, cfg)], dtype=torch.long, device=device)
    tokens = torch.full((1, c.dialogue_len), c.pad_id, dtype=torch.long, device=device)
    greedy_turn(cfg, buyer, obs, tokens, 0)
    return tokens[0, :c.max_msg_len].clone()


@torch.no_grad()
def opening_context(cfg: Config, pop: Population, world: World, phase=None,
                    device: str = "cpu") -> Optional[torch.Tensor]:
    """One fixed opening turn, for probing whoever speaks second in ``phase``."""
    if phase is None or not phase.tuples:
        return farmer_context_message(cfg, pop, world, device)
    first = phase.speaker_of_turn(0)
    agent = pop.pool(first)[0]
    m = sample_meanings(cfg, world, first, 1, phase=phase, seed=12345)
    msg = utterances_for_meanings(cfg, agent, m, device=device, phase=phase)
    c = cfg.channel
    out = torch.full((c.max_msg_len,), c.pad_id, dtype=torch.long, device=device)
    if msg:
        u = msg[0][:c.max_msg_len]
        if u:
            out[:len(u)] = torch.tensor(u, dtype=torch.long, device=device)
    return out


def speaking_roles(cfg: Config, phase=None) -> list[tuple[int, str]]:
    """The (role, label) pairs that actually talk in this phase, over all its views."""
    from .curriculum import ladder
    ph = phase if phase is not None else ladder(cfg)[-1]
    out = []
    for role, label in ((BUYER, "buyer"), (FARMER, "farmer")):
        if any(v.speaks(cfg, role) for v in ph.views()):
            out.append((role, label))
    return out


def speaker_view(phase, cfg: Config, role: int):
    """The view of ``phase`` in which ``role`` speaks (its describer view in a swap)."""
    from .curriculum import ladder
    ph = phase if phase is not None else ladder(cfg)[-1]
    for v in ph.views():
        if v.speaks(cfg, role):
            return v
    return None


# ==========================================================================
# 5.2  compositionality, per role
# ==========================================================================
def compositionality(cfg: Config, pop: Population, world: World, *,
                     n_samples: int = 200, device: str = "cpu",
                     rng: Optional[random.Random] = None, phase=None,
                     n_null: int = 3) -> dict[str, Any]:
    """Topographic similarity of each role's utterances, under ``phase``.

    A role that does not speak in the phase is reported as NaN with
    ``silent`` set, rather than probed anyway -- a lineup guesser never talks.

    Every probe asks the rung's own kind of round (:func:`probe_query`), so a
    describer is measured on the job the rung is promoted for rather than on a
    blend of that and the rehearsal rounds mixed in beneath it.
    """
    rng = rng or random.Random(0)
    out: dict[str, Any] = {}
    talking = dict((lbl, r) for r, lbl in speaking_roles(cfg, phase))
    for role, label in ((BUYER, "buyer"), (FARMER, "farmer")):
        view = speaker_view(phase, cfg, role) if phase is not None else None
        if phase is not None and label not in talking:
            out[label] = {"mean": float("nan"), "max": float("nan"),
                          "mean_l1": float("nan"), "null_mean": float("nan"),
                          "per_agent": [], "silent": True}
            continue
        kinds = phase_kinds(cfg, role, view)
        ctx = opening_context(cfg, pop, world, view, device)
        meanings = sample_meanings(cfg, world, role, n_samples, phase=view,
                                   seed=rng.randrange(1 << 30),
                                   query=probe_query(view))
        per_agent = []
        from .properties import disentanglement
        real = [i for i, k in enumerate(kinds) if k not in (K_EMPTY, K_FIELD)]
        pool = list(pop.pool(role))
        cap = max(1, cfg.log.max_agents_probed)
        if len(pool) > cap:
            pool = random.Random(cfg.train.seed + len(pool)).sample(pool, cap)
        from .env import parse_words
        lexicon: set = set()
        for agent in pool:
            msgs = utterances_for_meanings(cfg, agent, meanings, context=ctx,
                                           device=device, phase=view, role=role)
            r = topographic_similarity(meanings, msgs, cfg, role, metric="hamming",
                                       rng=rng, kinds=kinds, n_null=n_null)
            r_l1 = topographic_similarity(meanings, msgs, cfg, role, metric="l1",
                                          n_null=0, rng=rng, kinds=kinds)
            dis = disentanglement(meanings, msgs, real, cfg.channel.max_msg_len)
            from .properties import field_coverage
            cov = field_coverage(meanings, msgs, real, rng=rng)
            mine = {tuple(w) for m in msgs for w in parse_words(cfg, m)}
            lexicon |= mine
            per_agent.append({"agent": agent.name, "generation": agent.generation,
                              "topsim": r["topsim"], "null": r["null_mean"],
                              "z": r["z"], "topsim_l1": r_l1["topsim"],
                              "posdis": dis["posdis"], "bosdis": dis["bosdis"],
                              "field_coverage": cov["coverage"],
                              "distinct_forms": len({tuple(m) for m in msgs}),
                              "distinct_words": len(mine),
                              "n_probes": len(msgs),
                              "per_field": cov["per_field"]})
        vals = [a["topsim"] for a in per_agent if a["topsim"] == a["topsim"]]
        l1s = [a["topsim_l1"] for a in per_agent if a["topsim_l1"] == a["topsim_l1"]]
        nulls = [a["null"] for a in per_agent if a["null"] == a["null"]]
        def avg(key):
            xs = [a[key] for a in per_agent if a[key] == a[key]]
            return sum(xs) / len(xs) if xs else float("nan")
        out[label] = {
            "mean": sum(vals) / len(vals) if vals else float("nan"),
            "max": max(vals) if vals else float("nan"),
            "mean_l1": sum(l1s) / len(l1s) if l1s else float("nan"),
            "null_mean": sum(nulls) / len(nulls) if nulls else float("nan"),
            "posdis": avg("posdis"),
            "bosdis": avg("bosdis"),
            "field_coverage": avg("field_coverage"),
            # How many different things a speaker says at all, over probes it
            # answers deterministically. Beside the sampled word count this
            # separates a large lexicon from a speaker that is merely unsure:
            # a flawless 12-word code, emitted with 98% per-symbol accuracy,
            # shows up as ~170 distinct sampled words.
            "distinct_forms": avg("distinct_forms"),
            "distinct_words": avg("distinct_words"),
            "lexicon_size": len(lexicon),
            "n_probes": per_agent[0]["n_probes"] if per_agent else 0,
            "per_field_coverage": [
                (sum(a["per_field"][i] for a in per_agent) / len(per_agent))
                for i in range(len(per_agent[0]["per_field"]))] if per_agent else [],
            "per_agent": per_agent,
        }
    both = [out[k]["mean"] for k in ("buyer", "farmer") if out[k]["mean"] == out[k]["mean"]]
    out["mean"] = sum(both) / len(both) if both else float("nan")
    nulls = [out[k]["null_mean"] for k in ("buyer", "farmer")
             if out[k]["null_mean"] == out[k]["null_mean"]]
    out["null_mean"] = sum(nulls) / len(nulls) if nulls else float("nan")
    return out


# ==========================================================================
# 5.3  vocabulary usage
# ==========================================================================
def vocab_stats(cfg: Config, batches: Sequence[Any]) -> dict[str, Any]:
    """Token/message statistics over one or more rollout batches."""
    c = cfg.channel
    tok_counts: Counter[int] = Counter()
    msg_counts: Counter[tuple] = Counter()
    lengths: list[int] = []
    eos_first = 0
    n_utt = 0
    for batch in batches:
        toks = batch.tokens
        B = toks.shape[0]
        for turn in range(c.n_turns):
            seg = toks[:, turn * c.max_msg_len:(turn + 1) * c.max_msg_len]
            for i in range(B):
                utt = [int(t) for t in seg[i] if int(t) != c.pad_id]
                n_utt += 1
                content = [t for t in utt if c.is_atom(t)]
                tok_counts.update(content)
                msg_counts[tuple(content)] += 1
                lengths.append(len(content))
                if utt and utt[0] == c.end_id:
                    eos_first += 1
    used = len(tok_counts)
    h_tok = entropy_bits(tok_counts.values())
    h_msg = entropy_bits(msg_counts.values())
    max_h = math.log2(c.atomic_vocab) if c.atomic_vocab > 1 else 1.0
    top = tok_counts.most_common(10)
    total = sum(tok_counts.values()) or 1
    return {
        "tokens_used": used,
        "vocab_size": c.atomic_vocab,
        "token_entropy_bits": h_tok,
        "token_entropy_norm": h_tok / max_h if max_h else 0.0,
        "message_entropy_bits": h_msg,
        "distinct_messages": len(msg_counts),
        "mean_msg_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_msg_len": c.max_symbols,
        "silent_frac": eos_first / n_utt if n_utt else 0.0,
        "top_tokens": [{"token": t, "count": n, "share": n / total} for t, n in top],
        "atomic_vocab": c.atomic_vocab,
        "token_counts": dict(tok_counts),
    }


# ==========================================================================
# 5.4  stability of the meaning->message mapping over time
# ==========================================================================
class StabilityTracker:
    """Re-asks the same fixed probe meanings every checkpoint (spec 5.4).

    Probes are per *meaning kind* -- a trading request, a farmer's barn, a lineup
    tuple -- and only the roles that actually speak in the current phase are
    asked. Coherence is "do different agents say the same thing for the same
    meaning"; with both roles describing tuples it is also measured *across*
    roles (``coherence_cross``), which is the one-language-or-two question.
    """

    def __init__(self, cfg: Config, world: World, n_probes: int, seed: int = 99):
        self.cfg = cfg
        self.n_probes = n_probes
        self.seed = seed
        probe_rng = random.Random(seed)
        probe_world = World(cfg.world, probe_rng)
        self.probes = {
            ("request", BUYER): [buyer_obs(probe_world.sample(held_out=False), cfg)
                                 for _ in range(n_probes)],
            ("barn", FARMER): [farmer_obs(probe_world.sample(held_out=False), cfg)
                               for _ in range(n_probes)],
        }
        self._tuples = tuple_meanings(cfg, n_probes, seed=seed)
        self.last: dict[tuple, list[list[int]]] = {}   # (agent_id, kind) -> messages

    def probes_for(self, phase, role: int) -> tuple[str, list]:
        if phase is None or not phase.tuples:
            kind = "request" if role == BUYER else "barn"
            return kind, self.probes[(kind, role)]
        return "tuple", self._tuples

    @torch.no_grad()
    def measure(self, pop: Population, world: World, device: str = "cpu",
                phase=None) -> dict[str, Any]:
        drift_scores: list[float] = []
        identical: list[float] = []
        coherence: dict[str, float] = {}
        by_role: dict[str, list[list[list[int]]]] = {}
        kind_of: dict[str, str] = {}
        talking = dict((lbl, r) for r, lbl in speaking_roles(self.cfg, phase))

        for role, label in ((BUYER, "buyer"), (FARMER, "farmer")):
            if phase is not None and label not in talking:
                coherence[label] = float("nan")
                continue
            view = speaker_view(phase, self.cfg, role) if phase is not None else None
            kind, probes = self.probes_for(view, role)
            ctx = opening_context(self.cfg, pop, world, view, device)
            all_msgs: list[list[list[int]]] = []
            for agent in pop.pool(role):
                msgs = utterances_for_meanings(self.cfg, agent, probes, context=ctx,
                                               role=role,
                                               device=device, phase=view)
                if msgs is None:
                    continue
                key = (agent.agent_id, kind)
                all_msgs.append(msgs)
                prev = self.last.get(key)
                if prev is not None and len(prev) == len(msgs):
                    ds = [normalised_levenshtein(a, b) for a, b in zip(prev, msgs)]
                    drift_scores.extend(ds)
                    identical.extend([1.0 if a == b else 0.0 for a, b in zip(prev, msgs)])
                self.last[key] = msgs

            # population coherence: do *different* agents say the same thing for the
            # same meaning?  This is what makes it a shared code rather than N codes.
            pair_d: list[float] = []
            for i in range(len(all_msgs)):
                for j in range(i + 1, len(all_msgs)):
                    for a, b in zip(all_msgs[i], all_msgs[j]):
                        pair_d.append(normalised_levenshtein(a, b))
            coherence[label] = 1.0 - (sum(pair_d) / len(pair_d)) if pair_d else float("nan")
            by_role[label] = all_msgs
            kind_of[label] = kind

        cross = float("nan")
        if ("farmer" in by_role and "buyer" in by_role
                and kind_of["farmer"] == kind_of["buyer"]):
            # Below `split_roles_at` one pool fills both seats, so row i of each
            # list is the *same agent* in the other seat. Comparing it with
            # itself asks whether an agent agrees with itself, which it does,
            # and with two founders that was half of every pair: cross read
            # 0.56 where the honest number -- the two founders sharing no form
            # at all -- was 0.16. Training never seats an agent opposite
            # itself, and neither does this.
            shared = getattr(pop, "shared", False)
            d = [normalised_levenshtein(a, b)
                 for i, fm in enumerate(by_role["farmer"])
                 for j, bm in enumerate(by_role["buyer"])
                 if not (shared and i == j)
                 for a, b in zip(fm, bm)]
            cross = 1.0 - sum(d) / len(d) if d else float("nan")
        vals = [v for v in coherence.values() if v == v]
        return {
            "drift": sum(drift_scores) / len(drift_scores) if drift_scores else float("nan"),
            "identical_frac": sum(identical) / len(identical) if identical else float("nan"),
            "coherence_buyer": coherence.get("buyer", float("nan")),
            "coherence_farmer": coherence.get("farmer", float("nan")),
            "coherence": sum(vals) / len(vals) if vals else float("nan"),
            "coherence_cross": cross,
            "n_compared": len(drift_scores),
        }


# ==========================================================================
# performance-style evaluations
# ==========================================================================
@torch.no_grad()
def _play(cfg: Config, pop: Population, world: World, n: int,
          f_sel: Sequence[int], b_sel: Sequence[int], *, held_out: bool | None = False,
          device: str = "cpu", rng: Optional[random.Random] = None,
          scenarios: Optional[Sequence[Scenario]] = None,
          pairing: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
          channel_mode: str = "intact", phase=None, sampler=None) -> dict[str, Any]:
    if not f_sel or not b_sel or n <= 0:
        return {"n": 0, "success_rate": float("nan"), "mean_reward": float("nan")}
    rng = rng or random.Random(0)
    if scenarios is not None:
        scen = scenarios
    elif sampler is not None:
        scen = sampler(n, held_out)          # the phase supplies its own world
    else:
        scen = [world.sample(held_out=held_out) for _ in range(n)]
    if pairing is not None:
        f_idx, b_idx = pairing
    else:
        fs, bs = list(f_sel), list(b_sel)
        f_list = [rng.choice(fs) for _ in range(n)]
        # Seat the pairs the way training does. Until the roles split, one pool
        # fills both seats and training never puts an agent opposite itself
        # (Population.pair); drawing the seats independently did, half the time
        # with two founders. That asked every promotion check a question training
        # never asks -- can an agent read its *own* words -- and two founders who
        # had each invented a dialect the other could read scored 0.92 in
        # training and 0.60 here, and a working rung ran out its budget.
        shared = getattr(pop, "shared", False)
        b_list = [rng.choice([b for b in bs if b != f] or bs) if shared else rng.choice(bs)
                  for f in f_list]
        f_idx = torch.tensor(f_list, dtype=torch.long)
        b_idx = torch.tensor(b_list, dtype=torch.long)
    batch = run_episodes(cfg, scen, pop.farmers, pop.buyers, f_idx, b_idx, device=device,
                         channel_mode=channel_mode, phase=phase)
    # Tensor views, so a 4096-episode evaluation is a few reductions rather than
    # 4096 attribute lookups on dataclasses that had to be built first.
    succ_t = batch.success_t
    referential = batch.sb is not None and not hasattr(batch.sb, "viable")
    mutual = referential and batch.res is not None and "farmer_report_ok" in batch.res
    viable_t = (torch.ones_like(succ_t) if referential
                else batch.viable_t.to(succ_t.device))
    succ = int(succ_t.sum())
    viable = int(viable_t.sum())
    succ_viable = int((succ_t & viable_t).sum())
    # "comprehended" = the two independently-produced beliefs matched each other
    # and described an executable deal, whether or not the pair chose to trade.
    # It separates "did the message get through" from "did they want the deal".
    comp_t = batch.comprehended_t
    comp = int(comp_t.sum())
    comp_viable = int((comp_t & viable_t).sum())
    judged = int(batch.judged_t.sum())

    # Reference accuracy, per dimension.  Joint comprehension is a conjunction of
    # four things and stays near zero long after the first real words appear, so
    # this is the metric that actually shows a vocabulary arriving: can the farmer
    # name the variety the buyer asked for, and the number they asked for?  Both
    # are facts only the buyer holds.
    if referential:
        # In the lineup game "naming the right thing" is the guess itself.
        n_var_hits = n_qty_hits = n_price_hits = int(succ_t.sum())
    elif batch.sb is not None:
        sb = batch.sb
        n_var_hits = int((batch.f_dec[:, 1] == sb.want_variety).sum())
        n_qty_hits = int((batch.f_dec[:, 2] == sb.need_qty).sum())
        n_price_hits = int(((batch.f_dec[:, 3] >= sb.reservation)
                            & (batch.f_dec[:, 3] <= sb.max_price)).sum())
    else:
        n_var_hits = sum(int(batch.f_dec[i, 1]) == scen[i].buyer.want_variety
                         for i in range(n))
        n_qty_hits = sum(int(batch.f_dec[i, 2]) == scen[i].buyer.need_qty
                         for i in range(n))
        n_price_hits = sum(scen[i].price_in_zopa(int(batch.f_dec[i, 3]))
                           for i in range(n))

    # The closed loop, per direction: each agent states what it believes the other
    # party's private situation to be, and this is how often it is right.  Unlike
    # the deal-decision measures above, neither side can move these without the
    # channel -- every field is one the answering agent cannot observe.
    f_decode = float(batch.farmer_decode_t.float().mean())
    b_decode = float(batch.buyer_decode_t.float().mean())
    extra: dict[str, Any] = {}
    res = batch.res if isinstance(batch.res, dict) else None
    if res is not None and "order_fields" in res:
        # A request rung: how often each asked-for field arrived, and how often
        # the one this rung introduced did. One conjunction would hide a field
        # sitting at chance, which is how the old `haggle` failure looked.
        of = res["order_fields"].float()
        extra["request_fields"] = [float(x) for x in of.mean(0)]
        extra["request_first"] = float(res["order_first"].float().mean())
    if mutual:
        extra["farmer_report"] = float(batch.res["farmer_report_ok"].float().mean())
        extra["buyer_report"] = float(batch.res["buyer_report_ok"].float().mean())
        ff, bf = batch.res["farmer_fields"].float(), batch.res["buyer_fields"].float()
        extra["farmer_report_fields"] = [float(x) for x in ff.mean(0)]
        extra["buyer_report_fields"] = [float(x) for x in bf.mean(0)]
    return {
        **extra,
        "n": n,
        "success_rate": succ / n,
        "success_rate_on_viable": succ_viable / viable if viable else float("nan"),
        "comprehension_rate": comp / n,
        "comprehension_on_viable": comp_viable / viable if viable else float("nan"),
        "judgement_rate": judged / n,
        "farmer_reads_buyer": f_decode,
        "buyer_reads_farmer": b_decode,
        "farmer_variety_acc": n_var_hits / n,
        "farmer_qty_acc": n_qty_hits / n,
        "farmer_price_acc": n_price_hits / n,
        "viable_frac": viable / n,
        "mean_reward": float((batch.f_reward.mean() + batch.b_reward.mean()) / 2),
        "batch": batch,
        "scenarios": scen,
        "pairing": (f_idx, b_idx),
    }


def evaluate_success(cfg: Config, pop: Population, world: World, n: int,
                     device: str = "cpu", rng: Optional[random.Random] = None,
                     phase=None, sampler=None) -> dict[str, Any]:
    f_all = list(range(len(pop.farmers)))
    b_all = list(range(len(pop.buyers)))
    return _play(cfg, pop, world, n, f_all, b_all, held_out=False, device=device,
                 rng=rng, phase=phase, sampler=sampler)


# ---- 5.6 zero-shot generalisation ---------------------------------------
ZS_MIN_SEEN_SUCCESSES = 20
ZS_MIN_SEEN_RATE = 0.05


def zero_shot(cfg: Config, pop: Population, world: World, n: int,
              device: str = "cpu", rng: Optional[random.Random] = None,
              phase=None, sampler=None, n_holdout: Optional[int] = None,
              chance: float = 0.0) -> dict[str, Any]:
    """Success on (variety, quantity) combinations never sampled during training.

    Retention is unseen / seen, which is meaningless when "seen" is a handful of
    lucky rounds: 4 successes against 1 reads as 4.00, which is how a run with no
    working language reported perfect-plus generalisation. Below
    ``ZS_MIN_SEEN_SUCCESSES`` successes, or a seen rate under ``ZS_MIN_SEEN_RATE``,
    retention is suppressed (NaN) and the reason recorded. Lineup rungs have no
    held-out combinations at all, so there it is simply not applicable.
    """
    base = {"n": n, "n_holdout_combos": len(world.holdout) if n_holdout is None else n_holdout,
            "context": "lineup tuples" if (phase is not None and phase.tuples) else "trades"}
    if phase is not None and phase.tuples and not n_holdout:
        return {**base, "seen_success": float("nan"), "unseen_success": float("nan"),
                "seen_success_on_viable": float("nan"),
                "unseen_success_on_viable": float("nan"), "retention": float("nan"),
                "suppressed": "not applicable: no lineup tuples are held out"}
    f_all = list(range(len(pop.farmers)))
    b_all = list(range(len(pop.buyers)))
    seen = _play(cfg, pop, world, n, f_all, b_all, held_out=False, device=device, rng=rng,
                 phase=phase, sampler=sampler)
    unseen = _play(cfg, pop, world, n, f_all, b_all, held_out=True, device=device, rng=rng,
                   phase=phase, sampler=sampler)
    s_rate, u_rate = seen["success_rate"], unseen["success_rate"]
    n_seen_succ = int(round(s_rate * seen["n"])) if s_rate == s_rate else 0
    retention = float("nan")
    why = None
    if s_rate != s_rate:
        why = "no evaluation"
    elif n_seen_succ < ZS_MIN_SEEN_SUCCESSES or s_rate < ZS_MIN_SEEN_RATE:
        why = ("suppressed: only %d successes (rate %.3f) on seen combinations; need "
               "%d and %.2f for the ratio to mean anything"
               % (n_seen_succ, s_rate, ZS_MIN_SEEN_SUCCESSES, ZS_MIN_SEEN_RATE))
    elif s_rate < chance + 0.10:
        # unseen / seen is ~1 when both are chance: that is no language, not
        # perfect generalisation
        why = ("suppressed: seen success %.3f is at chance (%.3f); there is nothing yet "
               "to generalise" % (s_rate, chance))
    else:
        retention = u_rate / s_rate
    return {
        **base,
        "seen_success": s_rate,
        "unseen_success": u_rate,
        "seen_success_on_viable": seen["success_rate_on_viable"],
        "unseen_success_on_viable": unseen["success_rate_on_viable"],
        "seen_successes": n_seen_succ,
        "retention": retention,
        "suppressed": why,
    }


def _headroom(intact: float, scrambled: float) -> float:
    """Share of the *available* improvement that the channel is responsible for.

    Raw accuracy flatters a mute agent in a Zipfian world, where always naming
    the commonest variety already scores well.  This asks instead: of the gap
    between what silence gets you and perfect, how much did the messages close?
    """
    room = 1.0 - scrambled
    if room <= 1e-9 or intact != intact or scrambled != scrambled:
        return float("nan")
    return (intact - scrambled) / room


def channel_ablation(cfg: Config, pop: Population, world: World, n: int,
                     device: str = "cpu", rng: Optional[random.Random] = None,
                     phase=None, sampler=None) -> dict[str, Any]:
    """Causal test: does the channel actually carry information?

    The same scenarios and the same pairings are played twice.  In the second
    run every agent still emits exactly what its policy chooses, but what the
    *other* party receives is replaced by uniform random tokens of the same
    shape.  Any gap between the two runs is caused by the content of the
    messages and nothing else.

    This is the metric that cannot be fooled by reward shaping: a pair that has
    learned to exploit base rates rather than to talk will score identically in
    both conditions.
    """
    rng = rng or random.Random(0)
    f_all = list(range(len(pop.farmers)))
    b_all = list(range(len(pop.buyers)))
    intact = _play(cfg, pop, world, n, f_all, b_all, device=device, rng=rng,
                   phase=phase, sampler=sampler)
    if intact.get("scenarios") is None:
        return {"n": 0}
    same = dict(scenarios=intact["scenarios"], pairing=intact["pairing"],
                phase=phase, sampler=sampler)
    scrambled = _play(cfg, pop, world, n, f_all, b_all, device=device, rng=rng,
                      channel_mode="scrambled", **same)
    muted = _play(cfg, pop, world, n, f_all, b_all, device=device, rng=rng,
                  channel_mode="muted", **same)
    def drop(key: str) -> float:
        a, b = intact.get(key, float("nan")), scrambled.get(key, float("nan"))
        if a != a or b != b:
            return float("nan")
        return a - b
    rel = float("nan")
    if intact["comprehension_rate"] > 1e-9:
        rel = 1.0 - muted["comprehension_rate"] / intact["comprehension_rate"]
    # Fraction of the *available headroom* the channel captures, measured against
    # silence.  A mute agent already scores something by guessing the common case,
    # so raw accuracy flatters it; this does not.
    transfer = _headroom(intact["comprehension_rate"], muted["comprehension_rate"])
    return {
        "n": n,
        "intact_success": intact["success_rate"],
        "scrambled_success": scrambled["success_rate"],
        "intact_comprehension": intact["comprehension_rate"],
        "scrambled_comprehension": scrambled["comprehension_rate"],
        "intact_judgement": intact["judgement_rate"],
        "scrambled_judgement": scrambled["judgement_rate"],
        "muted_success": muted["success_rate"],
        "muted_comprehension": muted["comprehension_rate"],
        "muted_judgement": muted["judgement_rate"],
        "intact_variety_acc": intact["farmer_variety_acc"],
        "scrambled_variety_acc": scrambled["farmer_variety_acc"],
        "muted_variety_acc": muted["farmer_variety_acc"],
        "intact_qty_acc": intact["farmer_qty_acc"],
        "scrambled_qty_acc": scrambled["farmer_qty_acc"],
        "muted_qty_acc": muted["farmer_qty_acc"],
        # against silence: everything the channel is worth
        # how much of each *direction* of the loop the channel is responsible for
        "intact_farmer_reads": intact["farmer_reads_buyer"],
        "muted_farmer_reads": muted["farmer_reads_buyer"],
        "intact_buyer_reads": intact["buyer_reads_farmer"],
        "muted_buyer_reads": muted["buyer_reads_farmer"],
        "farmer_reads_transfer": _headroom(intact["farmer_reads_buyer"],
                                           muted["farmer_reads_buyer"]),
        "buyer_reads_transfer": _headroom(intact["buyer_reads_farmer"],
                                          muted["buyer_reads_farmer"]),
        "variety_transfer": _headroom(intact["farmer_variety_acc"],
                                      muted["farmer_variety_acc"]),
        "qty_transfer": _headroom(intact["farmer_qty_acc"], muted["farmer_qty_acc"]),
        # against scrambling: what the symbols carry beyond mere utterance length
        "variety_transfer_content": _headroom(intact["farmer_variety_acc"],
                                              scrambled["farmer_variety_acc"]),
        "length_only_variety": _headroom(scrambled["farmer_variety_acc"],
                                         muted["farmer_variety_acc"]),
        **({"request_fields_intact": intact["request_fields"],
            "request_fields_muted": muted["request_fields"],
            "request_field_transfer": [_headroom(a, m) for a, m in zip(
                intact["request_fields"], muted["request_fields"])],
            "request_first": intact["request_first"],
            "muted_request_first": muted["request_first"],
            "request_first_transfer": _headroom(intact["request_first"],
                                                muted["request_first"])}
           if "request_fields" in intact else {}),
        **({"intact_farmer_report": intact["farmer_report"],
            "muted_farmer_report": muted["farmer_report"],
            "intact_buyer_report": intact["buyer_report"],
            "muted_buyer_report": muted["buyer_report"],
            "farmer_report_transfer": _headroom(intact["farmer_report"],
                                                muted["farmer_report"]),
            "buyer_report_transfer": _headroom(intact["buyer_report"],
                                               muted["buyer_report"]),
            # per field: variety, quantity, quality
            "farmer_fields_intact": intact["farmer_report_fields"],
            "farmer_fields_muted": muted["farmer_report_fields"],
            "buyer_fields_intact": intact["buyer_report_fields"],
            "buyer_fields_muted": muted["buyer_report_fields"],
            "farmer_field_transfer": [_headroom(a, m) for a, m in zip(
                intact["farmer_report_fields"], muted["farmer_report_fields"])],
            "buyer_field_transfer": [_headroom(a, m) for a, m in zip(
                intact["buyer_report_fields"], muted["buyer_report_fields"])]}
           if "farmer_report" in intact else {}),
        "intact_reward": intact["mean_reward"],
        "scrambled_reward": scrambled["mean_reward"],
        "success_drop": drop("success_rate"),
        "comprehension_drop": drop("comprehension_rate"),
        "judgement_drop": drop("judgement_rate"),
        "reward_drop": drop("mean_reward"),
        "relative_comprehension_loss": rel,
        "information_transfer": transfer,
    }


# ---- decontextualisation ------------------------------------------------------
@torch.no_grad()
def context_consistency(cfg: Config, pop: Population, *, n: int = 45,
                        device: str = "cpu") -> dict[str, Any]:
    """Does a buyer name a meaning the same way in two different contexts?

    The same (variety, quantity, quality) is probed twice per buyer: as the
    describer in a lineup (tuple layout, speaking first) and as the requester
    in a trade (request layout, price held at a reference value). Similarity of
    the two forms for the *same* meaning, minus their similarity for different
    meanings, is how context-free the form-meaning pairing is.
    """
    from .conventions import similarity
    from .curriculum import ladder
    from .lexicon import reference_buyer_obs
    # Taken from the ladder, not by name: this asked for "refer-swap" for long
    # enough that the rungs were renamed under it, and the KeyError was swallowed
    # by the caller, so the measure silently reported nothing for every run.
    rungs = ladder(cfg)
    naming = next((p for p in rungs if p.swaps and p.whole), None)
    trade = next((p for p in rungs if p.trading), None)
    if naming is None or trade is None:
        return {"n": 0, "consistency": float("nan"),
                "note": "the ladder has no naming rung or no trading rung"}
    lineup = naming.with_informer(BUYER)
    tup = tuple_meanings(cfg, n, seed=4242)
    req = []
    for m in tup:
        r = list(reference_buyer_obs(cfg, (m[0], m[1])))
        r[2] = m[2]
        req.append(tuple(r))
    same, diff = [], []
    for agent in pop.buyers:
        a = utterances_for_meanings(cfg, agent, tup, device=device, phase=lineup,
                                    role=lineup.informer)
        b = utterances_for_meanings(cfg, agent, req, device=device, phase=trade,
                                    role=BUYER)
        if a is None or b is None:
            continue
        strip = lambda u: [t for t in u if t < cfg.channel.end_id]
        a, b = [strip(u) for u in a], [strip(u) for u in b]
        for i in range(len(a)):
            same.append(similarity(a[i], b[i]))
            j = (i + 1 + i % 7) % len(a)
            if tup[j] != tup[i]:
                diff.append(similarity(a[i], b[j]))
    if not same:
        return {"n": 0, "consistency": float("nan")}
    s, d = sum(same) / len(same), (sum(diff) / len(diff) if diff else 0.0)
    return {"n": len(same), "same_meaning": s, "different_meaning": d, "consistency": s - d}


def _report_fields(res) -> float:
    """Mean per-field report accuracy over both roles, or NaN if not a mutual round."""
    vals = []
    for k in ("farmer_report_fields", "buyer_report_fields"):
        v = (res or {}).get(k)
        if v:
            vals.extend(float(x) for x in v)
    return sum(vals) / len(vals) if vals else float("nan")


def _report_side(res) -> float:
    """How often ONE side got all three fields at once, averaged over the roles.

    Sits between the per-field numbers and the whole round, and the gap between
    it and the product of the per-field rates is the whole story on held-out
    combinations: independent fields would multiply, and these do not.
    """
    vals = [(res or {}).get(k) for k in ("farmer_report", "buyer_report")]
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _report_field_vec(res) -> "list[float] | None":
    """(fruit, colour, quality) report accuracy, averaged over the two roles.

    The mean over all six numbers hides which field is carrying it, and on
    held-out combinations that is the whole question: the reserved set is a
    Latin square, so for every (fruit, colour) pair exactly one quality is
    withheld and the held-out round asks for precisely the value the training
    distribution says cannot occur there. A listener that has fit that
    distribution is pushed *away* from the right quality, which can hold one
    field near zero while the other two generalise perfectly well.
    """
    rows = [(res or {}).get(k) for k in ("farmer_report_fields", "buyer_report_fields")]
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = min(len(r) for r in rows)
    return [sum(float(r[i]) for r in rows) / len(rows) for i in range(n)]


def pool_field_floor(pool) -> float:
    """Per field, the best a listener that hears nothing can do on this pool.

    Combinations are drawn uniformly from a pool, but a *pool* need not have
    uniform field marginals: the reserved quarter is 16 rows of 64, and one
    fruit can easily be six of them. A guesser that ignores the message and
    always says the commonest value scores that share, so it is the floor a
    per-field number has to be read against -- exactly as ``round_chance`` is
    the floor for a lineup. Returns the mean of that share over the three
    fields.
    """
    try:
        n = int(pool.shape[0])
    except Exception:
        return float("nan")
    if n <= 0:
        return float("nan")
    shares = []
    for col in range(int(pool.shape[1])):
        counts = torch.bincount(pool[:, col].reshape(-1).long())
        shares.append(float(counts.max()) / n)
    return sum(shares) / len(shares) if shares else float("nan")


# ---- the evidence a curriculum rung is judged on ---------------------------
def phase_evidence(cfg: Config, pop: Population, world: World, phase, *,
                   sampler_for, n_eval: int, n_topsim: int, n_semantics: int,
                   chance: float, device: str = "cpu",
                   rng: Optional[random.Random] = None,
                   holdout_sampler_for=None,
                   holdout_floor_for=None,
                   kind_sampler_for=None) -> dict[str, Any]:
    """Everything :func:`orchard.curriculum.evaluate_rung` needs, per view and per role.

    * per view (both describers, in a swap rung): intact / muted success and the
      share of headroom over a muted channel -- so each role's *decoding* is
      measured in the view where it is the one decoding;
    * per speaking role: topographic similarity against its own shuffled null,
      and positional structure (mean slot->field strength), so each role's
      *describing* is measured on its own utterances;
    * in the mutual rung: each role's accuracy at reporting the other's thing,
      intact and muted;
    * success on **held-out combinations** -- ones no agent was ever trained on.
      A code with reusable parts describes them; a fused one, however well
      drilled, cannot, so this is the test that separates the two.
    """
    rng = rng or random.Random(0)
    f_all = list(range(len(pop.farmers)))
    b_all = list(range(len(pop.buyers)))
    out: dict[str, Any] = {"phase": phase.name, "views": [], "speakers": {},
                           "chance": chance}
    succ, trans = [], []
    for v in phase.views():
        abl = channel_ablation(cfg, pop, world, n_eval, device=device, rng=rng,
                               phase=v, sampler=sampler_for(v))
        if not abl.get("n"):
            continue
        row = {"informer": None, "guesser": None,
               "success": abl["intact_success"], "muted_success": abl["muted_success"],
               "scrambled_success": abl["scrambled_success"],
               "transfer": abl["information_transfer"], "n": abl["n"]}
        if v.referential:
            row["informer"] = "farmer" if v.informer == FARMER else "buyer"
            row["guesser"] = "farmer" if v.guesser == FARMER else "buyer"
        for k in ("farmer_report", "buyer_report"):
            if "intact_" + k in abl:
                out[k] = abl["intact_" + k]
                out["muted_" + k] = abl["muted_" + k]
                out[k + "_transfer"] = abl[k + "_transfer"]
        for k in ("farmer_field_transfer", "buyer_field_transfer",
                  "farmer_fields_intact", "buyer_fields_intact",
                  "farmer_fields_muted", "buyer_fields_muted",
                  "request_fields_intact", "request_fields_muted",
                  "request_field_transfer", "request_first",
                  "muted_request_first", "request_first_transfer"):
            if k in abl:
                out[k] = abl[k]
        out["views"].append(row)
        succ.append(row["success"])
        trans.append(row["transfer"])
    good = [x for x in succ if x == x]
    out["success"] = sum(good) / len(good) if good else float("nan")
    good = [x for x in trans if x == x]
    out["transfer"] = sum(good) / len(good) if good else float("nan")

    # Held-out combinations, played exactly like the rung itself.
    out["holdout_success"] = float("nan")
    out["holdout_ratio"] = float("nan")
    out["holdout_fields"] = float("nan")
    out["seen_fields"] = float("nan")
    out["holdout_field_acc"] = None
    out["seen_field_acc"] = None
    out["holdout_field_ratios"] = None
    out["holdout_side"] = float("nan")
    out["seen_side"] = float("nan")
    out["holdout_field_ratio"] = float("nan")
    if holdout_sampler_for is not None:
        hs, seen = [], []
        hf, sf = [], []
        hv, sv = [], []
        hd, sd = [], []
        for v in phase.views():
            sam = holdout_sampler_for(v)
            plain = sampler_for(v)
            if sam is None:
                continue
            try:
                h = evaluate_success(cfg, pop, world, max(128, n_eval // 2),
                                     device=device, rng=rng, phase=v, sampler=sam)
                # the comparison run: seen combinations, same lineup shape
                s = evaluate_success(cfg, pop, world, max(128, n_eval // 2),
                                     device=device, rng=rng, phase=v,
                                     sampler=lambda n, _ho=False, _p=plain: _p(n, held_out=False))
            except Exception:
                continue
            if h.get("n"):
                hs.append(h["success_rate"])
                seen.append(s["success_rate"])
                # Where a round is scored as a conjunction, the conjunction is a
                # terrible estimator of whether the code generalises: `mutual`
                # wants three fields right on each of two novel meanings, so a
                # per-field shortfall is raised to the sixth power. Per-field
                # accuracy on the same rounds answers the same question without
                # the exponent -- the argument the request rungs already make
                # ("which field arrived, not the conjunction").
                a, b = _report_fields(h), _report_fields(s)
                if a == a:
                    hf.append(a)
                if b == b:
                    sf.append(b)
                va, vb = _report_field_vec(h), _report_field_vec(s)
                if va:
                    hv.append(va)
                if vb:
                    sv.append(vb)
                da, db = _report_side(h), _report_side(s)
                if da == da:
                    hd.append(da)
                if db == db:
                    sd.append(db)
        if hs:
            out["holdout_success"] = sum(hs) / len(hs)
            base = sum(seen) / len(seen) if seen else float("nan")
            out["seen_success"] = base
            out["holdout_ratio"] = (out["holdout_success"] / base
                                    if base == base and base > 1e-9 else float("nan"))
        for key, rows in (("holdout_field_acc", hv), ("seen_field_acc", sv)):
            if rows:
                n = min(len(r) for r in rows)
                out[key] = [sum(r[i] for r in rows) / len(rows) for i in range(n)]
        if hd:
            out["holdout_side"] = sum(hd) / len(hd)
        if sd:
            out["seen_side"] = sum(sd) / len(sd)
        if hf and sf:
            out["holdout_fields"] = sum(hf) / len(hf)
            out["seen_fields"] = sum(sf) / len(sf)
            h_floor = s_floor = 0.0
            if holdout_floor_for is not None:
                got = holdout_floor_for(phase)
                if got is not None:
                    h_floor, s_floor = got
                    h_floor = 0.0 if h_floor != h_floor else h_floor
                    s_floor = 0.0 if s_floor != s_floor else s_floor
            # Headroom over what a message-blind guesser gets, so a memorised
            # code reads 0 rather than the base rate it would score anyway.
            # Per field, then averaged -- not a ratio of the two means. The
            # ratio of means weights each field by its headroom, so the field
            # the language learned best also counts most towards whether the
            # language generalises, and one strong field can carry two weak
            # ones over the bar. Measured on the run that promoted out of
            # `mutual`: fruit transferred 0.887 of its headroom, colour 0.406
            # and quality 0.411, and the ratio of means read 0.617 against a
            # 0.60 bar where each field counted once reads 0.568. That is the
            # same masking the per-role and per-kind gates already refuse --
            # "a pooled average would let a fluent farmer carry a buyer that
            # never learned to speak".
            ha = out.get("holdout_field_acc") or []
            sa = out.get("seen_field_acc") or []
            ratios = []
            for i in range(min(len(ha), len(sa))):
                # A ratio between two numbers both at the floor is noise, and
                # noise reads 1.00 as readily as 0.00. A field the language
                # never learned has nothing to say about generalising, so it is
                # left out rather than counted as a pass or a failure.
                if sa[i] - s_floor > 0.05:
                    ratios.append(max(0.0, min(1.0, (ha[i] - h_floor)
                                               / (sa[i] - s_floor))))
            if ratios:
                out["holdout_field_ratios"] = ratios
                out["holdout_field_ratio"] = sum(ratios) / len(ratios)
            elif out["seen_fields"] - s_floor > 0.05:
                out["holdout_field_ratio"] = max(0.0, min(
                    1.0, (out["holdout_fields"] - h_floor)
                    / (out["seen_fields"] - s_floor)))
    # Each kind of round in the rung's mixture, scored on its own. A rung that
    # adds colour to fruit is promoted on colour and has to show it can still do
    # fruit; one number over both would let either hide behind the other.
    out["by_kind"] = {}
    if kind_sampler_for is not None and len(getattr(phase, "kinds", ())) > 1:
        from .curriculum import ASK_ALL, round_chance
        for kind in phase.kinds:
            rates, held = [], []
            for v in phase.views():
                sam = kind_sampler_for(v, kind)
                if sam is None:
                    continue
                try:
                    r = evaluate_success(cfg, pop, world, max(128, n_eval // 3),
                                         device=device, rng=rng, phase=v, sampler=sam)
                    # The same kind of round, on combinations nobody trained on.
                    # A lineup rung has no per-field report to read -- success is
                    # one K-way choice -- so this is the only way to ask of it
                    # what `mutual` is asked: does *this field* generalise. It is
                    # also the field-by-field view its own productivity gate has
                    # never had, and `name-all` is judged by that gate.
                    h = evaluate_success(cfg, pop, world, max(128, n_eval // 3),
                                         device=device, rng=rng, phase=v,
                                         sampler=lambda n, _ho=False, _s=sam:
                                         _s(n, held_out=True))
                except Exception:
                    continue
                if r.get("n"):
                    rates.append(r["success_rate"])
                if h.get("n"):
                    held.append(h["success_rate"])
            if rates:
                row = {"success": sum(rates) / len(rates), "views": len(rates)}
                if held:
                    ch = round_chance(cfg, kind)
                    row["holdout"] = sum(held) / len(held)
                    # Only a whole-thing round is a clean productivity test.
                    # A single-field lineup holds every field but the queried
                    # one fixed, and the Latin square reserves exactly one
                    # value of that field for each such cell -- so the target
                    # is the ONLY reserved candidate and can be told from the
                    # rest without understanding a word. Measured: 1 of 3 on a
                    # colour or quality round, 3 of 3 on a whole thing. The
                    # number is still worth having, but it is not comparable
                    # with one from a round where every candidate is reserved.
                    row["holdout_clean"] = bool(kind == ASK_ALL)
                    head = row["success"] - ch
                    # As everywhere else: over the headroom above a guesser, so a
                    # memorised code reads 0.00 rather than the chance it scores
                    # anyway, and too little headroom reads as no answer at all.
                    row["transfers"] = (max(0.0, min(1.0, (row["holdout"] - ch) / head))
                                        if head > 0.05 else float("nan"))
                out["by_kind"][int(kind)] = row

    if (phase.mutual or phase.order) and out["views"]:
        # no analytic chance for "report / fill a whole tuple": silence is the floor
        out["chance"] = out["views"][0]["muted_success"]

    comp = compositionality(cfg, pop, world, n_samples=n_topsim, device=device,
                            rng=rng, phase=phase, n_null=2)
    sem = analyse_token_semantics(cfg, pop, world, n_samples=n_semantics,
                                  device=device, rng=rng, phase=phase)
    for role, label in speaking_roles(cfg, phase):
        c = comp.get(label, {})
        out["speakers"][label] = {
            "topsim": c.get("mean", float("nan")),
            "null": c.get("null_mean", float("nan")),
            "posdis": c.get("posdis", float("nan")),
            "bosdis": c.get("bosdis", float("nan")),
            "field_coverage": c.get("field_coverage", float("nan")),
            "per_field_coverage": c.get("per_field_coverage", []),
            "distinct_forms": c.get("distinct_forms", float("nan")),
            "distinct_words": c.get("distinct_words", float("nan")),
            "lexicon_size": c.get("lexicon_size", 0),
            "n_probes": c.get("n_probes", 0),
            "positional": positional_structure(sem.per_position.get(label, [])),
            "positional_rows": sem.per_position.get(label, []),
        }
    out["topsim"] = comp.get("mean", float("nan"))
    out["null"] = comp.get("null_mean", float("nan"))
    out["_compositionality"] = comp
    return out


# ---- 5.5 cross-generation intelligibility -------------------------------
def _play_views(cfg, pop, world, n, f_sel, b_sel, *, device, rng, phase, sampler_for):
    """_play over every view of a phase (both describers in a swap), pooled."""
    if phase is None:
        return _play(cfg, pop, world, n, f_sel, b_sel, device=device, rng=rng)
    views = phase.views()
    rs = [_play(cfg, pop, world, max(1, n // len(views)), f_sel, b_sel, device=device,
                rng=rng, phase=v, sampler=sampler_for(v) if sampler_for else None)
          for v in views]
    rs = [r for r in rs if r.get("n")]
    if not rs:
        return {"n": 0, "success_rate": float("nan"), "success_rate_on_viable": float("nan")}
    out = dict(rs[0])
    for k in ("success_rate", "success_rate_on_viable"):
        vals = [r[k] for r in rs if r[k] == r[k]]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    out["n"] = sum(r["n"] for r in rs)
    return out


def intelligibility(cfg: Config, pop: Population, world: World, n: int,
                    *, newborn_age: int, device: str = "cpu",
                    rng: Optional[random.Random] = None, phase=None,
                    sampler_for=None) -> dict[str, Any]:
    """Can recent arrivals trade with agents that predate them?

    Compares newcomer-with-veteran success against veteran-with-veteran success.
    A ratio near 1 means the code transmits; a ratio near 0 means veterans share a
    private cipher that newcomers cannot acquire.
    """
    rng = rng or random.Random(0)
    # ``newborn_age`` is in training updates, like lifespans
    new_f = [i for i, a in enumerate(pop.farmers) if a.updates <= newborn_age]
    new_b = [i for i, a in enumerate(pop.buyers) if a.updates <= newborn_age]
    vet_f = [i for i, a in enumerate(pop.farmers) if a.updates > newborn_age]
    vet_b = [i for i, a in enumerate(pop.buyers) if a.updates > newborn_age]

    res: dict[str, Any] = {"n_newborn_farmers": len(new_f), "n_newborn_buyers": len(new_b),
                           "n_veteran_farmers": len(vet_f), "n_veteran_buyers": len(vet_b)}
    kw = dict(device=device, rng=rng, phase=phase, sampler_for=sampler_for)
    vv = _play_views(cfg, pop, world, n, vet_f, vet_b, **kw) if vet_f and vet_b else None
    nv = _play_views(cfg, pop, world, n, new_f, vet_b, **kw) if new_f and vet_b else None
    vn = _play_views(cfg, pop, world, n, vet_f, new_b, **kw) if vet_b and new_b and vet_f else None

    res["veteran_veteran"] = vv["success_rate"] if vv else float("nan")
    res["newborn_farmer_veteran_buyer"] = nv["success_rate"] if nv else float("nan")
    res["veteran_farmer_newborn_buyer"] = vn["success_rate"] if vn else float("nan")
    mixed = [v for v in (res["newborn_farmer_veteran_buyer"],
                         res["veteran_farmer_newborn_buyer"]) if v == v]
    res["newcomer_mixed"] = sum(mixed) / len(mixed) if mixed else float("nan")
    if vv and vv["success_rate"] > 0 and mixed:
        res["transmission_ratio"] = res["newcomer_mixed"] / vv["success_rate"]
    else:
        res["transmission_ratio"] = float("nan")
    return res


def newborn_vs_veterans(cfg: Config, pop: Population, world: World, newborn: Agent,
                        n: int, *, device: str = "cpu",
                        rng: Optional[random.Random] = None, phase=None,
                        sampler_for=None) -> dict[str, Any]:
    """Spec 5.5, measured at the moment of birth: a fresh agent, straight out of the
    bottleneck, against the agents that were already alive.  It has never played a
    live episode, so any success is transmitted knowledge, not co-adaptation."""
    rng = rng or random.Random(0)
    if newborn.role == FARMER:
        f_sel = [i for i, a in enumerate(pop.farmers) if a.agent_id == newborn.agent_id]
        b_sel = [i for i, a in enumerate(pop.buyers) if a.birth_episode < newborn.birth_episode]
    else:
        b_sel = [i for i, a in enumerate(pop.buyers) if a.agent_id == newborn.agent_id]
        f_sel = [i for i, a in enumerate(pop.farmers) if a.birth_episode < newborn.birth_episode]
    if not f_sel or not b_sel:
        return {"n": 0, "success_rate": float("nan")}
    # Tested on the game the population is currently playing. Testing a
    # lineup-rung newborn on the full market scored every one of them 0.000.
    r = _play_views(cfg, pop, world, n, f_sel, b_sel, device=device, rng=rng,
                    phase=phase, sampler_for=sampler_for)
    return {"n": r["n"], "success_rate": r["success_rate"],
            "success_rate_on_viable": r.get("success_rate_on_viable", float("nan")),
            "phase": phase.name if phase is not None else "market"}


# ==========================================================================
# post-hoc token semantics  (spec 6.3 / 6.4 -- inferred, never asserted)
# ==========================================================================
def _bin_value(cfg: Config, kind: int, value: int, n_bins: int = 5) -> int:
    """Coarsen a field so mutual information is estimable from a few hundred samples."""
    w = cfg.world
    if kind in (K_VARIETY, K_QUALITY):
        return int(value)
    if kind == K_QTY:
        return min(n_bins - 1, int(value * n_bins / max(w.max_qty + 1, 1)))
    if kind == K_PRICE:
        return min(n_bins - 1, int(value * n_bins / max(w.n_price_bins, 1)))
    return 0


def _mutual_information(pairs: Sequence[tuple[int, int]]) -> tuple[float, float]:
    """MI(x;y) in bits and H(y) in bits, for discrete x, y."""
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    jx: Counter[int] = Counter()
    jy: Counter[int] = Counter()
    jxy: Counter[tuple[int, int]] = Counter()
    for x, y in pairs:
        jx[x] += 1
        jy[y] += 1
        jxy[(x, y)] += 1
    mi = 0.0
    for (x, y), c in jxy.items():
        pxy = c / n
        px = jx[x] / n
        py = jy[y] / n
        mi += pxy * math.log2(pxy / (px * py))
    return max(mi, 0.0), entropy_bits(jy.values())


@dataclass
class TokenSemantics:
    """Inferred, post-hoc token <-> meaning-dimension associations.

    This is an *analysis output*, produced only after training, and is always
    rendered with a ``likely:`` hedge.  The raw token ids remain the record of
    what was actually said (spec 6.3).
    """
    cfg: Config
    per_token: dict[int, dict[str, Any]] = field(default_factory=dict)
    per_word: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_position: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # word classes: which separate words specialise to which field, per role
    word_classes: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_samples: int = 0

    def hint(self, tok: int) -> str:
        rec = self.per_token.get(tok)
        if not rec or rec["score"] < 0.05:
            return ""
        return "%s=%s, %.2f" % (rec["dimension"], rec["typical"], rec["score"])

    def word_hint(self, word) -> str:
        """Inferred association for a whole word, falling back to its first atom."""
        from .env import word_text
        rec = self.per_word.get(word_text(self.cfg, tuple(word)))
        if rec and rec["score"] >= 0.05:
            return "%s=%s, %.2f" % (rec["dimension"], rec["typical"], rec["score"])
        return self.hint(int(word[0])) if len(word) == 1 else ""

    def to_dict(self) -> dict[str, Any]:
        return {"n_samples": self.n_samples,
                "per_token": {str(k): v for k, v in sorted(self.per_token.items())},
                "per_word": self.per_word,
                "per_position": self.per_position,
                "word_classes": self.word_classes}


def analyse_token_semantics(cfg: Config, pop: Population, world: World, *,
                            n_samples: int = 600, device: str = "cpu",
                            rng: Optional[random.Random] = None,
                            phase=None) -> TokenSemantics:
    """Correlate emitted tokens with the speaker's private fields.

    For every content token we compute the mutual information between "this token
    appeared in the utterance" and each (coarsened) field of the speaker's own
    observation, normalised by that field's entropy.  The winning field is
    reported as a *likely* association together with the value most often present
    when the token is used.  We do the same per message position, which is what
    would reveal word-order structure: field X tends to be named in slot 1, Y in
    slot 2.

    Nothing here is given to the agents.  It is measurement after the fact, and
    it is rendered with a ``likely:`` hedge everywhere it is shown.
    """
    rng = rng or random.Random(0)
    ts = TokenSemantics(cfg=cfg, n_samples=n_samples)
    talking = dict((lbl, r) for r, lbl in speaking_roles(cfg, phase))

    for role, label in ((BUYER, "buyer"), (FARMER, "farmer")):
        if phase is not None and label not in talking:
            continue                    # silent in this phase: nothing to analyse
        view = speaker_view(phase, cfg, role) if phase is not None else None
        kinds = phase_kinds(cfg, role, view)
        labels = phase_labels(cfg, role, view)
        real = [i for i, k in enumerate(kinds) if k not in (K_EMPTY, K_FIELD)]
        ctx = opening_context(cfg, pop, world, view, device)
        meanings = sample_meanings(cfg, world, role, n_samples, phase=view,
                                   seed=rng.randrange(1 << 30),
                                   query=probe_query(view))
        agents = pop.pool(role)
        msgs: list[list[int]] = [[] for _ in meanings]
        # one batched greedy decode per agent over its share of the meanings
        for a_i, agent in enumerate(agents):
            idx = list(range(a_i, len(meanings), len(agents)))
            if not idx:
                continue
            got = utterances_for_meanings(cfg, agent, [meanings[j] for j in idx],
                                          role=role,
                                          context=ctx, device=device, phase=view)
            for j, m in zip(idx, got):
                msgs[j] = m

        binned = [[_bin_value(cfg, kinds[i], m[i]) for i in range(len(kinds))]
                  for m in meanings]

        for tok in range(cfg.channel.atomic_vocab):
            present = [1 if tok in msg else 0 for msg in msgs]
            if sum(present) < 8 or sum(present) > len(present) - 8:
                continue
            best = None
            for i in real:
                mi, hy = _mutual_information(
                    list(zip(present, [b[i] for b in binned])))
                score = mi / hy if hy > 1e-9 else 0.0
                if best is None or score > best[1]:
                    best = (i, score)
            if best is None:
                continue
            i, score = best
            vals = [meanings[j][i] for j in range(len(msgs)) if present[j]]
            prev = ts.per_token.get(tok)
            if prev is None or score > prev["score"]:
                ts.per_token[tok] = {
                    "dimension": labels[i], "score": round(float(score), 4),
                    "typical": _describe_value(cfg, kinds[i], vals), "role": label,
                    "usage": round(sum(present) / len(present), 3),
                }

        # ---- words, not just atoms (addendum 3.2) ----------------------
        from .env import parse_words, word_text
        word_lists = [parse_words(cfg, m) for m in msgs]
        seen_words: Counter = Counter()
        for wl in word_lists:
            seen_words.update(set(wl))
        for word, count in seen_words.items():
            if count < 8 or count > len(msgs) - 8:
                continue
            present = [1 if word in set(wl) else 0 for wl in word_lists]
            best = None
            for i in real:
                mi, hy = _mutual_information(
                    list(zip(present, [b[i] for b in binned])))
                score = mi / hy if hy > 1e-9 else 0.0
                if best is None or score > best[1]:
                    best = (i, score)
            if best is None:
                continue
            i, score = best
            vals = [meanings[j][i] for j in range(len(msgs)) if present[j]]
            key = word_text(cfg, word)
            prev = ts.per_word.get(key)
            if prev is None or score > prev["score"]:
                ts.per_word[key] = {
                    "dimension": labels[i], "score": round(float(score), 4),
                    "typical": _describe_value(cfg, kinds[i], vals), "role": label,
                    "atoms": len(word),
                    "usage": round(sum(present) / len(present), 3),
                }

        # ---- word classes: noun-, adjective-, numeral-like words ------------
        # A word "belongs" to a field when it predicts that field strongly
        # (normalised MI >= 0.3). The question the user of this report cares
        # about is whether separate, space-separated words carry separate fields
        # -- a variety word next to a quality word -- rather than one compound
        # naming the whole meaning.
        word_field = {k: r["dimension"] for k, r in ts.per_word.items()
                      if r.get("role") == label and r["score"] >= 0.3}
        classes: dict[str, list[str]] = {}
        for k, f in word_field.items():
            classes.setdefault(f, []).append(k)
        multi = sum(1 for wl in word_lists
                    if len({word_field.get(word_text(cfg, w)) for w in wl} - {None}) >= 2)
        ts.word_classes[label] = {
            "classes": {f: sorted(ws)[:8] for f, ws in classes.items()},
            "n_classes": len(classes),
            "multi_class_share": multi / max(1, len(word_lists)),
            "mean_words_per_message": sum(len(wl) for wl in word_lists) / max(1, len(word_lists)),
        }

        rows = []
        for k in range(cfg.channel.max_msg_len):
            toks_at_k = [(msg[k] if k < len(msg) else cfg.channel.pad_id)
                         for msg in msgs]
            best = None
            for i in real:
                mi, hy = _mutual_information(
                    list(zip(toks_at_k, [b[i] for b in binned])))
                score = mi / hy if hy > 1e-9 else 0.0
                if best is None or score > best[1]:
                    best = (i, score)
            used = sum(1 for t in toks_at_k if t < cfg.channel.end_id) / max(1, len(msgs))
            rows.append({"position": k, "dimension": labels[best[0]],
                         "score": round(float(best[1]), 4),
                         "distinct_tokens": len(set(toks_at_k)),
                         "used": round(used, 3)})
        ts.per_position[label] = rows

    return ts


def positional_structure(rows: Sequence[dict[str, Any]], min_used: float = 0.2) -> float:
    """One number per role from the per-slot table: mean slot->field strength.

    Averaged over the slots the language actually uses (a symbol there in at
    least ``min_used`` of utterances), so a terse code is not penalised for the
    slots it leaves empty -- and an empty code scores NaN rather than zero.
    """
    live = [r["score"] for r in rows if r.get("used", 1.0) >= min_used]
    return sum(live) / len(live) if live else float("nan")


def _describe_value(cfg: Config, kind: int, vals: Sequence[int]) -> str:
    if not vals:
        return "?"
    w = cfg.world
    if kind == K_VARIETY:
        return w.variety_names[Counter(vals).most_common(1)[0][0]]
    if kind == K_QUALITY:
        return w.quality_names[Counter(vals).most_common(1)[0][0]]
    mean = sum(vals) / len(vals)
    if kind == K_QTY:
        return "~%.1f apples" % mean
    if kind == K_PRICE:
        i = int(round(mean))
        return "~%.2f" % (w.price_values[i] if 0 <= i < w.n_price_bins else mean)
    return "-"


# ==========================================================================
# degenerate-outcome detection  (spec 6.2: never let a failing run look fine)
# ==========================================================================
def detect_degenerate(cfg: Config, success_rate: float, chance_rate: float,
                      vocab: dict[str, Any], comp: dict[str, Any],
                      updates_done: int, words: Optional[dict[str, Any]] = None,
                      ablation: Optional[dict[str, Any]] = None) -> list[str]:
    """``chance_rate`` is the current rung's (NaN where it has none: no flag)."""
    flags: list[str] = []
    settled = updates_done >= 2 * cfg.log.checkpoint_every_updates
    if settled and chance_rate == chance_rate and success_rate <= chance_rate * 1.5:
        flags.append("SUCCESS RATE AT OR NEAR CHANCE (%.3f vs chance %.3f)"
                     % (success_rate, chance_rate))
    if vocab["token_entropy_norm"] < 0.15:
        flags.append("VOCABULARY COLLAPSE: normalised token entropy %.3f"
                     % vocab["token_entropy_norm"])
    if vocab["tokens_used"] <= 2:
        flags.append("ONLY %d TOKEN(S) IN USE" % vocab["tokens_used"])
    if vocab["silent_frac"] > 0.9:
        flags.append("CHANNEL UNUSED: %.0f%% of utterances are empty"
                     % (100 * vocab["silent_frac"]))
    if vocab["mean_msg_len"] >= cfg.channel.max_msg_len - 0.05:
        flags.append("MAXIMAL-LENGTH BABBLING: mean message length %.2f of %d"
                     % (vocab["mean_msg_len"], cfg.channel.max_msg_len))
    m = comp.get("mean", float("nan"))
    if settled and m == m and m < 0.05:
        flags.append("NO COMPOSITIONAL STRUCTURE: topsim %.3f after %d updates"
                     % (m, updates_done))
    if settled and ablation:
        t = ablation.get("information_transfer")
        if isinstance(t, float) and t == t and t < 0.05:
            flags.append("CHANNEL CARRIES NOTHING: scrambling every message costs "
                         "%.1f%% of the available headroom -- performance is base "
                         "rates, not communication" % (100 * t))
    if words:
        from .lexicon import word_usage_flags
        flags.extend(word_usage_flags(cfg, words))
    return flags


def chance_success_rate(cfg: Config, world: World, n: int = 4000,
                        seed: int = 7) -> float:
    """Empirical success rate of two uniformly random agents -- the honest baseline."""
    from .env import RandomScriptedAgent, run_scripted_episode
    rng = random.Random(seed)
    w = World(cfg.world, random.Random(seed))
    fa = RandomScriptedAgent(cfg, FARMER, rng)
    ba = RandomScriptedAgent(cfg, BUYER, rng)
    hits = sum(run_scripted_episode(cfg, w.sample(held_out=False), fa, ba).outcome.success
               for _ in range(n))
    return hits / n
