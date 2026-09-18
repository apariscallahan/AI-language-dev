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
from .world import (K_EMPTY, K_PRICE, K_QTY, K_QUALITY, K_VARIETY, KIND_NAMES,
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
                     metric: str = "hamming") -> float:
    """Distance between two private states, over that role's real fields only."""
    kinds = obs_schema(cfg.world, role)
    if metric == "hamming":
        return float(sum(1 for x, y, k in zip(a, b, kinds)
                         if k != K_EMPTY and x != y))
    if metric == "l1":
        # keeps the ordinal structure of quantity and price that Hamming discards
        spans = field_spans(cfg.world, role)
        return float(sum(abs(x - y) / sp
                         for x, y, k, sp in zip(a, b, kinds, spans) if k != K_EMPTY))
    raise ValueError("unknown meaning metric %r" % (metric,))


def topographic_similarity(meanings: Sequence[Sequence[int]],
                           messages: Sequence[Sequence[int]],
                           cfg: Config, role: int, *, metric: str = "hamming",
                           n_null: int = 3, rng: Optional[random.Random] = None
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
            md.append(meaning_distance(meanings[i], meanings[j], cfg, role, metric))
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
                turn: int) -> torch.Tensor:
    """Fill dialogue ``turn`` for a whole batch by greedy decoding.  Mutates ``tokens``."""
    c = cfg.channel
    B = obs.shape[0]
    alive = torch.ones(B, dtype=torch.bool, device=obs.device)
    for k in range(c.max_msg_len):
        if not bool(alive.any()):
            break
        p = turn * c.max_msg_len + k
        logits, _ = agent.net.next_token_logits(obs, tokens, dialogue_offset(cfg) + p)
        tok = logits.argmax(dim=-1)
        tok = torch.where(alive, tok, torch.full_like(tok, c.pad_id))
        tokens[:, p] = tok
        alive = alive & (tok != c.eos_id)
    return tokens


def _strip(cfg: Config, row: Sequence[int]) -> list[int]:
    return [int(t) for t in row if int(t) != cfg.channel.pad_id]


@torch.no_grad()
def utterances_for_meanings(cfg: Config, agent: Agent, meanings: Sequence[Sequence[int]],
                            *, context: Optional[torch.Tensor] = None,
                            device: str = "cpu") -> list[list[int]]:
    """This agent's own first utterance for each meaning, greedily decoded.

    For a Buyer (who opens) the utterance is a pure function of its private
    observation.  For a Farmer, the reply also depends on what the Buyer just
    said, so ``context`` supplies one fixed opening message for every probe --
    holding the conversation constant so the variation measured is the Farmer's.
    """
    c = cfg.channel
    n = len(meanings)
    obs = torch.tensor([list(m) for m in meanings], dtype=torch.long, device=device)
    tokens = torch.full((n, c.dialogue_len), c.pad_id, dtype=torch.long, device=device)
    turn = 0 if agent.role == BUYER else 1
    if agent.role == FARMER:
        if context is None:
            context = torch.full((c.max_msg_len,), c.eos_id, dtype=torch.long, device=device)
            context[0] = c.eos_id
        tokens[:, :c.max_msg_len] = context.unsqueeze(0)
    greedy_turn(cfg, agent, obs, tokens, turn)
    L = c.max_msg_len
    return [_strip(cfg, tokens[i, turn * L:(turn + 1) * L]) for i in range(n)]


def sample_meanings(cfg: Config, world: World, role: int, n: int,
                    *, held_out: bool | None = False) -> list[tuple[int, int, int, int]]:
    out = []
    for _ in range(n):
        sc = world.sample(held_out=held_out)
        out.append(obs_for(role, sc, cfg))
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


# ==========================================================================
# 5.2  compositionality, per role
# ==========================================================================
def compositionality(cfg: Config, pop: Population, world: World, *,
                     n_samples: int = 200, device: str = "cpu",
                     rng: Optional[random.Random] = None) -> dict[str, Any]:
    rng = rng or random.Random(0)
    ctx = farmer_context_message(cfg, pop, world, device)
    out: dict[str, Any] = {}
    for role, label in ((BUYER, "buyer"), (FARMER, "farmer")):
        meanings = sample_meanings(cfg, world, role, n_samples)
        per_agent = []
        for agent in pop.pool(role):
            msgs = utterances_for_meanings(cfg, agent, meanings,
                                           context=ctx if role == FARMER else None,
                                           device=device)
            r = topographic_similarity(meanings, msgs, cfg, role, metric="hamming", rng=rng)
            r_l1 = topographic_similarity(meanings, msgs, cfg, role, metric="l1",
                                          n_null=0, rng=rng)
            per_agent.append({"agent": agent.name, "generation": agent.generation,
                              "topsim": r["topsim"], "null": r["null_mean"],
                              "z": r["z"], "topsim_l1": r_l1["topsim"]})
        vals = [a["topsim"] for a in per_agent if a["topsim"] == a["topsim"]]
        l1s = [a["topsim_l1"] for a in per_agent if a["topsim_l1"] == a["topsim_l1"]]
        nulls = [a["null"] for a in per_agent if a["null"] == a["null"]]
        out[label] = {
            "mean": sum(vals) / len(vals) if vals else float("nan"),
            "max": max(vals) if vals else float("nan"),
            "mean_l1": sum(l1s) / len(l1s) if l1s else float("nan"),
            "null_mean": sum(nulls) / len(nulls) if nulls else float("nan"),
            "per_agent": per_agent,
        }
    both = [out[k]["mean"] for k in ("buyer", "farmer") if out[k]["mean"] == out[k]["mean"]]
    out["mean"] = sum(both) / len(both) if both else float("nan")
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
    """Re-asks the same fixed probe meanings every checkpoint (spec 5.4)."""

    def __init__(self, cfg: Config, world: World, n_probes: int, seed: int = 99):
        self.cfg = cfg
        probe_rng = random.Random(seed)
        probe_world = World(cfg.world, probe_rng)
        self.probes = {
            BUYER: [buyer_obs(probe_world.sample(held_out=False), cfg)
                    for _ in range(n_probes)],
            FARMER: [farmer_obs(probe_world.sample(held_out=False), cfg)
                     for _ in range(n_probes)],
        }
        self.last: dict[int, list[list[int]]] = {}       # agent_id -> messages

    @torch.no_grad()
    def measure(self, pop: Population, world: World, device: str = "cpu") -> dict[str, Any]:
        ctx = farmer_context_message(self.cfg, pop, world, device)
        drift_scores: list[float] = []
        identical: list[float] = []
        coherence: dict[str, float] = {}
        current: dict[int, list[list[int]]] = {}

        for role, label in ((BUYER, "buyer"), (FARMER, "farmer")):
            probes = self.probes[role]
            all_msgs: list[list[list[int]]] = []
            for agent in pop.pool(role):
                msgs = utterances_for_meanings(
                    self.cfg, agent, probes,
                    context=ctx if role == FARMER else None, device=device)
                current[agent.agent_id] = msgs
                all_msgs.append(msgs)
                prev = self.last.get(agent.agent_id)
                if prev is not None and len(prev) == len(msgs):
                    ds = [normalised_levenshtein(a, b) for a, b in zip(prev, msgs)]
                    drift_scores.extend(ds)
                    identical.extend([1.0 if a == b else 0.0 for a, b in zip(prev, msgs)])

            # population coherence: do *different* agents say the same thing for the
            # same meaning?  This is what makes it a shared code rather than N codes.
            pair_d: list[float] = []
            for i in range(len(all_msgs)):
                for j in range(i + 1, len(all_msgs)):
                    for a, b in zip(all_msgs[i], all_msgs[j]):
                        pair_d.append(normalised_levenshtein(a, b))
            coherence[label] = 1.0 - (sum(pair_d) / len(pair_d)) if pair_d else float("nan")

        self.last = current
        return {
            "drift": sum(drift_scores) / len(drift_scores) if drift_scores else float("nan"),
            "identical_frac": sum(identical) / len(identical) if identical else float("nan"),
            "coherence_buyer": coherence.get("buyer", float("nan")),
            "coherence_farmer": coherence.get("farmer", float("nan")),
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
          channel_mode: str = "intact") -> dict[str, Any]:
    if not f_sel or not b_sel or n <= 0:
        return {"n": 0, "success_rate": float("nan"), "mean_reward": float("nan")}
    rng = rng or random.Random(0)
    scen = list(scenarios) if scenarios is not None else         [world.sample(held_out=held_out) for _ in range(n)]
    if pairing is not None:
        f_idx, b_idx = pairing
    else:
        f_idx = torch.tensor([rng.choice(list(f_sel)) for _ in range(n)], dtype=torch.long)
        b_idx = torch.tensor([rng.choice(list(b_sel)) for _ in range(n)], dtype=torch.long)
    batch = run_episodes(cfg, scen, pop.farmers, pop.buyers, f_idx, b_idx, device=device,
                         channel_mode=channel_mode)
    # Tensor views, so a 4096-episode evaluation is a few reductions rather than
    # 4096 attribute lookups on dataclasses that had to be built first.
    succ_t = batch.success_t
    viable_t = batch.viable_t.to(succ_t.device)
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
    if batch.sb is not None:
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
    return {
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
                     device: str = "cpu", rng: Optional[random.Random] = None
                     ) -> dict[str, Any]:
    f_all = list(range(len(pop.farmers)))
    b_all = list(range(len(pop.buyers)))
    return _play(cfg, pop, world, n, f_all, b_all, held_out=False, device=device, rng=rng)


# ---- 5.6 zero-shot generalisation ---------------------------------------
def zero_shot(cfg: Config, pop: Population, world: World, n: int,
              device: str = "cpu", rng: Optional[random.Random] = None) -> dict[str, Any]:
    """Success on (variety, quantity) combinations never sampled during training."""
    f_all = list(range(len(pop.farmers)))
    b_all = list(range(len(pop.buyers)))
    seen = _play(cfg, pop, world, n, f_all, b_all, held_out=False, device=device, rng=rng)
    unseen = _play(cfg, pop, world, n, f_all, b_all, held_out=True, device=device, rng=rng)
    gap = float("nan")
    if seen["success_rate"] == seen["success_rate"] and seen["success_rate"] > 0:
        gap = unseen["success_rate"] / seen["success_rate"]
    return {
        "seen_success": seen["success_rate"],
        "unseen_success": unseen["success_rate"],
        "seen_success_on_viable": seen["success_rate_on_viable"],
        "unseen_success_on_viable": unseen["success_rate_on_viable"],
        "retention": gap,
        "n": n,
        "n_holdout_combos": len(world.holdout),
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
                     device: str = "cpu", rng: Optional[random.Random] = None
                     ) -> dict[str, Any]:
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
    intact = _play(cfg, pop, world, n, f_all, b_all, device=device, rng=rng)
    if not intact.get("scenarios"):
        return {"n": 0}
    same = dict(scenarios=intact["scenarios"], pairing=intact["pairing"])
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
        "intact_reward": intact["mean_reward"],
        "scrambled_reward": scrambled["mean_reward"],
        "success_drop": drop("success_rate"),
        "comprehension_drop": drop("comprehension_rate"),
        "judgement_drop": drop("judgement_rate"),
        "reward_drop": drop("mean_reward"),
        "relative_comprehension_loss": rel,
        "information_transfer": transfer,
    }


# ---- 5.5 cross-generation intelligibility -------------------------------
def intelligibility(cfg: Config, pop: Population, world: World, n: int,
                    *, newborn_age: int, device: str = "cpu",
                    rng: Optional[random.Random] = None) -> dict[str, Any]:
    """Can recent arrivals trade with agents that predate them?

    Compares newcomer-with-veteran success against veteran-with-veteran success.
    A ratio near 1 means the code transmits; a ratio near 0 means veterans share a
    private cipher that newcomers cannot acquire.
    """
    rng = rng or random.Random(0)
    new_f = [i for i, a in enumerate(pop.farmers) if a.age <= newborn_age]
    new_b = [i for i, a in enumerate(pop.buyers) if a.age <= newborn_age]
    vet_f = [i for i, a in enumerate(pop.farmers) if a.age > newborn_age]
    vet_b = [i for i, a in enumerate(pop.buyers) if a.age > newborn_age]

    res: dict[str, Any] = {"n_newborn_farmers": len(new_f), "n_newborn_buyers": len(new_b),
                           "n_veteran_farmers": len(vet_f), "n_veteran_buyers": len(vet_b)}
    vv = _play(cfg, pop, world, n, vet_f, vet_b, device=device, rng=rng) if vet_f and vet_b else None
    nv = _play(cfg, pop, world, n, new_f, vet_b, device=device, rng=rng) if new_f and vet_b else None
    vn = _play(cfg, pop, world, n, vet_f, new_b, device=device, rng=rng) if vet_b and new_b and vet_f else None

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
                        rng: Optional[random.Random] = None) -> dict[str, Any]:
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
    r = _play(cfg, pop, world, n, f_sel, b_sel, device=device, rng=rng)
    return {"n": r["n"], "success_rate": r["success_rate"],
            "success_rate_on_viable": r["success_rate_on_viable"]}


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
                "per_position": self.per_position}


def analyse_token_semantics(cfg: Config, pop: Population, world: World, *,
                            n_samples: int = 600, device: str = "cpu",
                            rng: Optional[random.Random] = None) -> TokenSemantics:
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
    ctx = farmer_context_message(cfg, pop, world, device)

    for role, label in ((BUYER, "buyer"), (FARMER, "farmer")):
        kinds = obs_schema(cfg.world, role)
        labels = field_labels(cfg.world, role)
        real = [i for i, k in enumerate(kinds) if k != K_EMPTY]
        meanings = sample_meanings(cfg, world, role, n_samples)
        agents = pop.pool(role)
        msgs: list[list[int]] = []
        for i, m in enumerate(meanings):
            agent = agents[i % len(agents)]
            msgs.append(utterances_for_meanings(
                cfg, agent, [m], context=ctx if role == FARMER else None,
                device=device)[0])

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
            rows.append({"position": k, "dimension": labels[best[0]],
                         "score": round(float(best[1]), 4),
                         "distinct_tokens": len(set(toks_at_k))})
        ts.per_position[label] = rows

    return ts


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
                      episodes_done: int, words: Optional[dict[str, Any]] = None,
                      ablation: Optional[dict[str, Any]] = None) -> list[str]:
    flags: list[str] = []
    settled = episodes_done >= max(2000, cfg.log.checkpoint_every * 2)
    if settled and success_rate <= chance_rate * 1.5:
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
        flags.append("NO COMPOSITIONAL STRUCTURE: topsim %.3f after %d episodes"
                     % (m, episodes_done))
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
