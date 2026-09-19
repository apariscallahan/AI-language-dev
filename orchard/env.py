"""The trading episode: message exchange, trade resolution, reward.

Two design rules from the spec are enforced here and nowhere else:

1. **Strict observation separation** (spec 1.2).  ``farmer_obs`` / ``buyer_obs``
   are the *only* way state reaches an agent, and each returns exactly that
   agent's own four private fields.  Nothing else about the scenario is ever
   handed to a network.  ``tests/test_env.py`` asserts this.

2. **Success requires mutual understanding** (spec 1.3).  A trade succeeds only
   when the two independently-produced decisions agree with each other *and* the
   agreed deal is actually executable against both sides' hidden constraints.
   Neither agent can produce a successful trade alone, however cleverly it plays.

The environment itself contains no learning and no torch; it is pure bookkeeping,
which keeps it testable with scripted dummy agents (spec 7 step 1).
"""
from __future__ import annotations

import torch

from dataclasses import dataclass, field
from typing import Any

from .config import Config, RewardConfig
from .world import Scenario, n_obs_slots

FARMER = 0
BUYER = 1
ROLE_NAMES = {FARMER: "farmer", BUYER: "buyer"}


# --------------------------------------------------------------------------
# Observations -- the enforced information boundary
# --------------------------------------------------------------------------
def _pad(cfg: Config, values: tuple[int, ...]) -> tuple[int, ...]:
    """Both roles share one sequence layout, so the shorter tuple is padded."""
    return tuple(values) + (0,) * (n_obs_slots(cfg.world) - len(values))


def farmer_obs(scenario: Scenario, cfg: Config) -> tuple[int, ...]:
    """Stock per variety, quality per variety, reservation price -- the whole barn."""
    return _pad(cfg, scenario.farmer.as_tuple())


def buyer_obs(scenario: Scenario, cfg: Config) -> tuple[int, ...]:
    """Wanted variety, needed quantity, minimum quality, budget ceiling."""
    return _pad(cfg, scenario.buyer.as_tuple())


def obs_for(role: int, scenario: Scenario, cfg: Config) -> tuple[int, ...]:
    return farmer_obs(scenario, cfg) if role == FARMER else buyer_obs(scenario, cfg)


def speaker_of_turn(turn: int) -> int:
    """Buyer opens (they are the one with a request), then strict alternation."""
    return BUYER if turn % 2 == 0 else FARMER


# --------------------------------------------------------------------------
# Decisions and outcomes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Decision:
    """What one agent independently believes was agreed, emitted simultaneously."""
    accept: int      # 0 = walk away, 1 = do the deal
    variety: int
    qty: int         # 0..max_qty
    price: int       # price bin

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.accept, self.variety, self.qty, self.price)


@dataclass(frozen=True)
class Beliefs:
    """What one agent says the OTHER party's private situation is.

    This is a claim about facts the speaker cannot see, so it can only be right if
    the other party told it something and it read that correctly.  Scoring it is
    what closes the communication loop in both directions: the reader is paid for
    reading, and the party who was read is paid for having been readable.
    """
    variety: int
    qty: int
    quality: int
    price: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.variety, self.qty, self.quality, self.price)


def decode_hits(beliefs: Beliefs, sc: Scenario, role: int,
                cfg: Config) -> list[bool]:
    """Per-field: did this agent correctly recover the other party's state?

    The farmer is asked about the buyer's shopping list; the buyer is asked about
    what is actually in the barn for the line it came for.  Nothing here is
    visible to the agent being scored, and nothing is derivable from its own
    half of the world -- the two sides are drawn independently (see world.py).
    """
    R = cfg.reward
    if role == FARMER:
        b = sc.buyer
        return [
            beliefs.variety == b.want_variety,
            abs(beliefs.qty - b.need_qty) <= R.belief_qty_tol,
            beliefs.quality == b.min_quality,
            abs(beliefs.price - b.max_price) <= R.belief_price_tol,
        ]
    # The buyer reports on the line it asked about, so stock 0 means "he does not
    # carry it" -- getting that right is itself a thing the farmer had to convey.
    return [
        abs(beliefs.qty - sc.offered_stock) <= R.belief_qty_tol,
        beliefs.quality == sc.offered_quality,
        abs(beliefs.price - sc.farmer.reservation) <= R.belief_price_tol,
    ]


def decode_score(beliefs: Beliefs, sc: Scenario, role: int, cfg: Config) -> float:
    hits = decode_hits(beliefs, sc, role, cfg)
    return sum(hits) / len(hits)


@dataclass
class Outcome:
    success: bool
    failure_mode: str
    reasons: list[str]
    farmer_reward: float
    buyer_reward: float
    # what, if anything, actually changed hands
    traded_qty: int = 0
    traded_price_bin: int = -1
    traded_variety: int = -1
    farmer_profit: float = 0.0      # currency, not reward
    buyer_savings: float = 0.0      # currency the buyer kept vs. their ceiling
    trade_value: float = 0.0        # currency, price * qty
    # diagnostics
    both_accept: bool = False
    # How well each side read the other, 0..1.  These are the closed-loop numbers:
    # farmer_decode is also what the buyer is paid for being understood, and
    # buyer_decode is what the farmer is paid for being understood.
    farmer_decode: float = 0.0
    buyer_decode: float = 0.0
    farmer_decode_hits: tuple = ()
    buyer_decode_hits: tuple = ()
    # Would this episode have succeeded had both agents accepted?  Isolates
    # "did they understand each other" from "did they choose to trade".
    comprehended: bool = False
    both_judged_viability: bool = False
    agree_variety: bool = False
    agree_qty: bool = False
    agree_price: bool = False
    farmer_correct: tuple[bool, bool, bool] = (False, False, False)
    buyer_correct: tuple[bool, bool, bool] = (False, False, False)
    reward_terms: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
def _agent_correctness(decision: Decision, sc: Scenario) -> tuple[bool, bool, bool]:
    """Per-dimension: is *this* agent's belief actually right about the joint deal?

    Every one of these three requires a fact the agent does not itself hold:
      variety  -- the farmer must learn what the buyer wants / the buyer must
                  learn what the farmer has,
      quantity -- the farmer must learn the buyer's need (and it must fit stock),
      price    -- the bin must sit inside the zone of possible agreement, whose
                  two ends are one private fact each.
    So correctness credit is never obtainable without information transfer.
    """
    ok_variety = sc.variety_ok and decision.variety == sc.deal_variety
    ok_qty = (decision.qty == sc.buyer.need_qty) and (decision.qty <= sc.offered_stock) \
        and decision.qty >= 1
    ok_price = sc.price_in_zopa(decision.price)
    return (ok_variety, ok_qty, ok_price)


def _classify(sc: Scenario, fd: Decision, bd: Decision, agree: dict[str, bool],
              success: bool) -> tuple[str, list[str]]:
    """Primary failure label plus every contributing reason (spec 6.1)."""
    if success:
        return "success", []
    reasons: list[str] = []
    both_accept = bool(fd.accept and bd.accept)

    if not both_accept:
        if fd.accept != bd.accept:
            reasons.append("one_sided_accept")
            primary = "one_sided_accept"
        elif sc.viable:
            reasons.append("missed_deal")
            primary = "missed_deal"
        else:
            # both rejected an unviable scenario -- this is the *correct* answer and is
            # never routed here (handled as correct_no_deal), but keep the label total.
            reasons.append("correct_no_deal")
            primary = "correct_no_deal"
        return primary, reasons

    # Both accepted but it did not come off.
    if not agree["variety"]:
        reasons.append("variety_mismatch")
    if not agree["qty"]:
        reasons.append("qty_mismatch")
    if not agree["price"]:
        reasons.append("price_mismatch")

    agreed_qty = (fd.qty + bd.qty) // 2
    agreed_price = (fd.price + bd.price) // 2
    if not sc.variety_ok:
        reasons.append("variety_not_stocked")
    if agreed_qty > sc.offered_stock:
        reasons.append("insufficient_stock")
    if agreed_qty != sc.buyer.need_qty:
        reasons.append("qty_not_what_buyer_needed")
    if agreed_price > sc.buyer.max_price:
        reasons.append("over_budget")
    if agreed_price < sc.farmer.reservation:
        reasons.append("below_reservation")
    if not sc.quality_ok:
        reasons.append("quality_below_requirement")
    if not reasons:
        reasons.append("unclassified")

    # Priority order: a breakdown in mutual understanding outranks an infeasible deal,
    # because that is the linguistically interesting failure.
    for key in ("variety_mismatch", "qty_mismatch", "price_mismatch"):
        if key in reasons:
            return key, reasons
    return reasons[0], reasons


def resolve(cfg: Config, sc: Scenario, fd: Decision, bd: Decision,
            farmer_tokens: int = 0, buyer_tokens: int = 0,
            f_beliefs: Beliefs | None = None,
            b_beliefs: Beliefs | None = None) -> Outcome:
    """Score one finished negotiation.

    ``farmer_tokens`` / ``buyer_tokens`` are the symbols that agent emitted and
    is charged for: atoms, hyphens and spaces, but not the end-of-message mark.
    """
    R: RewardConfig = cfg.reward
    prices = cfg.world.price_values
    price_span = max(prices[-1] - prices[0], 1e-9)

    both_accept = bool(fd.accept and bd.accept)
    agree = {
        "variety": fd.variety == bd.variety,
        "qty": abs(fd.qty - bd.qty) <= R.qty_tol,
        "price": abs(fd.price - bd.price) <= R.price_tol,
    }
    mutual = all(agree.values())

    f_correct = _agent_correctness(fd, sc)
    b_correct = _agent_correctness(bd, sc)

    agreed_variety = fd.variety
    agreed_qty = (fd.qty + bd.qty) // 2
    agreed_price = (fd.price + bd.price) // 2

    executable = (
        sc.variety_ok
        and agreed_variety == sc.deal_variety
        and 1 <= agreed_qty <= sc.offered_stock
        and abs(agreed_qty - sc.buyer.need_qty) <= R.qty_tol
        and sc.quality_ok
        and sc.farmer.reservation <= agreed_price <= sc.buyer.max_price
    )
    success = both_accept and mutual and executable

    fr = 0.0
    br = 0.0
    terms: dict[str, Any] = {}

    # ---- understanding, scored unconditionally -------------------------
    # The four decision outputs are read as "here is what I believe the deal is",
    # and that belief is scored whether or not the agent goes on to accept.  An
    # earlier version gated this behind both parties accepting; the population
    # promptly discovered that refusing every deal was safe, and with the belief
    # heads then receiving no gradient at all the channel never acquired meaning.
    # Scoring understanding separately from the accept/reject gamble keeps a dense
    # signal on comprehension while leaving "should this deal happen?" as its own
    # decision.  Nothing here is obtainable without information transfer: see
    # _agent_correctness.
    n_agree = sum(agree.values())
    agree_r = R.agree_per_dim * n_agree
    f_corr_r = R.correct_per_dim * sum(f_correct)
    b_corr_r = R.correct_per_dim * sum(b_correct)
    # Judging whether a deal is possible at all is the fourth thing neither agent
    # can work out alone: the farmer must learn the buyer's variety, need, minimum
    # quality and budget; the buyer must learn the farmer's stock, quality and
    # cost.  Scoring it per agent is what gives the accept/reject head a gradient
    # of its own instead of leaving it to stumble onto joint success by accident.
    f_judge = R.judgement * float(bool(fd.accept) == sc.viable)
    b_judge = R.judgement * float(bool(bd.accept) == sc.viable)
    fr += agree_r + f_corr_r + f_judge
    br += agree_r + b_corr_r + b_judge
    terms["agree"] = agree_r
    terms["farmer_correct"] = f_corr_r
    terms["buyer_correct"] = b_corr_r
    terms["farmer_judgement"] = f_judge
    terms["buyer_judgement"] = b_judge

    # ---- the closed loop: read the other, and be readable ----------------
    f_hits = decode_hits(f_beliefs, sc, FARMER, cfg) if f_beliefs else []
    b_hits = decode_hits(b_beliefs, sc, BUYER, cfg) if b_beliefs else []
    f_decode = (sum(f_hits) / len(f_hits)) if f_hits else 0.0
    b_decode = (sum(b_hits) / len(b_hits)) if b_hits else 0.0
    if f_hits or b_hits:
        # Each agent is paid twice over: once for reading the other, and once for
        # having been read.  The second term is the one that gives a speaker any
        # reason to be informative rather than merely to trade well.
        fr += R.decode * f_decode + R.understood * b_decode
        br += R.decode * b_decode + R.understood * f_decode
        terms["farmer_decode"] = R.decode * f_decode
        terms["buyer_decode"] = R.decode * b_decode
        terms["farmer_understood"] = R.understood * b_decode
        terms["buyer_understood"] = R.understood * f_decode

    # ---- the trade decision itself -------------------------------------
    if both_accept:
        if success:
            fr += R.success
            br += R.success
            terms["success"] = R.success
            # Economics: strictly zero-sum in price, so farmers push up and buyers push
            # down inside the ZOPA.  Quantity is not a free choice (it is the buyer's
            # need), so it is deliberately left out of the reward and reported as
            # currency in the ledger instead.
            pv = prices[agreed_price]
            margin = (pv - prices[sc.farmer.reservation]) / price_span
            surplus = (prices[sc.buyer.max_price] - pv) / price_span
            fr += R.economics * margin
            br += R.economics * surplus
            terms["farmer_econ"] = R.economics * margin
            terms["buyer_econ"] = R.economics * surplus
        elif not sc.viable:
            fr += R.bad_deal
            br += R.bad_deal
            terms["bad_deal"] = R.bad_deal
    elif fd.accept != bd.accept:
        fr += R.one_sided_accept
        br += R.one_sided_accept
        terms["one_sided_accept"] = R.one_sided_accept
    else:  # both rejected
        if sc.viable:
            fr += R.missed_deal
            br += R.missed_deal
            terms["missed_deal"] = R.missed_deal
        else:
            fr += R.correct_no_deal
            br += R.correct_no_deal
            terms["correct_no_deal"] = R.correct_no_deal

    # Addendum 2.1: every symbol the speaker emitted is charged for -- atoms,
    # hyphens and spaces alike.  Paid per episode, so meanings that come up often
    # pay it often, which is the whole of the Zipf mechanism in 2.2.
    fr -= R.symbol_cost * farmer_tokens
    br -= R.symbol_cost * buyer_tokens
    terms["farmer_symbol_cost"] = -R.symbol_cost * farmer_tokens
    terms["buyer_symbol_cost"] = -R.symbol_cost * buyer_tokens

    correct_no_deal = (not both_accept) and (fd.accept == bd.accept) and (not sc.viable)
    if success:
        mode, reasons = "success", []
    elif correct_no_deal:
        mode, reasons = "correct_no_deal", []
    else:
        mode, reasons = _classify(sc, fd, bd, agree, success)

    out = Outcome(
        success=success, failure_mode=mode, reasons=reasons,
        farmer_reward=fr, buyer_reward=br,
        both_accept=both_accept,
        farmer_decode=f_decode, buyer_decode=b_decode,
        farmer_decode_hits=tuple(f_hits), buyer_decode_hits=tuple(b_hits),
        comprehended=bool(mutual and executable),
        both_judged_viability=bool(bool(fd.accept) == sc.viable
                                   and bool(bd.accept) == sc.viable),
        agree_variety=agree["variety"], agree_qty=agree["qty"], agree_price=agree["price"],
        farmer_correct=f_correct, buyer_correct=b_correct,
        reward_terms=terms,
    )
    if success:
        pv = prices[agreed_price]
        out.traded_qty = agreed_qty
        out.traded_price_bin = agreed_price
        out.traded_variety = agreed_variety
        out.trade_value = pv * agreed_qty
        out.farmer_profit = (pv - prices[sc.farmer.reservation]) * agreed_qty
        out.buyer_savings = (prices[sc.buyer.max_price] - pv) * agreed_qty
    return out


# --------------------------------------------------------------------------
# Transcript container
# --------------------------------------------------------------------------
def grammar_allowed(cfg: Config, prev: torch.Tensor, k: int) -> torch.Tensor:
    """(B, n_emittable) bool: which symbols may come next in a turn.

    ``prev`` is the previous symbol of this turn (ignored at ``k == 0``). At the
    start: an atom, or END (silence). After an atom: HYPHEN, SPACE or END. After a
    HYPHEN or SPACE: an atom. In the last slot of the buffer an atom is followed
    by END, so no utterance ends on a dangling mark.
    """
    c = cfg.channel
    E = c.n_emittable
    dev = prev.device
    if not c.enforce_word_grammar:
        return torch.ones((prev.shape[0], E), dtype=torch.bool, device=dev)
    start, atoms, after_atom, closing = _grammar_rows(c.atomic_vocab, c.hyphen_id,
                                                      c.space_id, c.end_id, E, str(dev))
    if k == 0:
        return start.unsqueeze(0).expand(prev.shape[0], E)
    after = closing if k >= c.max_symbols - 1 else after_atom
    prev_atom = (prev >= 0) & (prev < c.atomic_vocab)
    return torch.where(prev_atom.unsqueeze(1), after.unsqueeze(0), atoms.unsqueeze(0))


_GRAMMAR_CACHE: dict = {}


def _grammar_rows(A: int, hyphen: int, space: int, end: int, E: int, dev: str):
    """The four fixed legal-next-symbol rows, built once per device (they are
    consulted at every symbol step of every rollout)."""
    key = (A, hyphen, space, end, E, dev)
    hit = _GRAMMAR_CACHE.get(key)
    if hit is None:
        atoms = torch.zeros(E, dtype=torch.bool, device=dev)
        atoms[:A] = True
        start = atoms.clone()
        start[end] = True
        after_atom = torch.zeros(E, dtype=torch.bool, device=dev)
        after_atom[[hyphen, space, end]] = True
        closing = torch.zeros(E, dtype=torch.bool, device=dev)
        closing[end] = True
        hit = _GRAMMAR_CACHE[key] = (start, atoms, after_atom, closing)
    return hit


def grammar_mask_for_positions(cfg: Config, tokens: torch.Tensor,
                               positions: "list[int]") -> torch.Tensor:
    """(B, P, n_emittable) legal-symbol masks for teacher forcing at ``positions``
    (dialogue indices), each judged from the previous symbol in its own turn."""
    L = cfg.channel.max_msg_len
    out = []
    for p in positions:
        k = p % L
        prev = tokens[:, p - 1] if k > 0 else torch.zeros_like(tokens[:, 0])
        out.append(grammar_allowed(cfg, prev, k))
    if not out:
        return torch.zeros((tokens.shape[0], 0, cfg.channel.n_emittable),
                           dtype=torch.bool, device=tokens.device)
    return torch.stack(out, dim=1)


MASKED = -1e9     # a logit no sample can land on; finite, so entropies stay finite


def parse_words(cfg: Config, symbols: "list[int]") -> list[tuple[int, ...]]:
    """Split a symbol sequence into words of atoms.  Never raises, never rejects."""
    c = cfg.channel
    words: list[tuple[int, ...]] = []
    current: list[int] = []
    for sym in symbols:
        if sym == c.space_id:
            if current:
                words.append(tuple(current))
                current = []
        elif sym == c.hyphen_id:
            continue          # joins whatever surrounds it; nothing to record
        elif c.is_atom(sym):
            current.append(sym)
        # END and PAD terminate / pad and carry no content
    if current:
        words.append(tuple(current))
    return words


def word_text(cfg: Config, word: "tuple[int, ...]") -> str:
    return "-".join("a%d" % t for t in word)


@dataclass
class Transcript:
    """Everything said in one episode, plus who said it.

    ``tokens`` is a flat list of length ``n_turns * max_msg_len``; slot
    ``turn*max_msg_len + k`` holds the k-th token of that turn (PAD after EOS).
    """
    tokens: list[int]
    scenario: Scenario
    farmer_decision: Decision | None = None
    buyer_decision: Decision | None = None
    farmer_beliefs: "Beliefs | None" = None
    buyer_beliefs: "Beliefs | None" = None
    outcome: Outcome | None = None

    def turn_tokens(self, cfg: Config, turn: int) -> list[int]:
        L = cfg.channel.max_msg_len
        return self.tokens[turn * L:(turn + 1) * L]

    def utterance(self, cfg: Config, turn: int) -> list[int]:
        """The turn's tokens with the PAD tail stripped (EOS kept)."""
        raw = self.turn_tokens(cfg, turn)
        pad = cfg.channel.pad_id
        return [t for t in raw if t != pad]

    def role_tokens(self, cfg: Config, role: int) -> list[int]:
        out: list[int] = []
        for turn in range(cfg.channel.n_turns):
            if speaker_of_turn(turn) == role:
                out.extend(self.utterance(cfg, turn))
        return out

    def words(self, cfg: Config, turn: int) -> list[tuple[int, ...]]:
        """Segment one turn into words (addendum 3.2).

        Deliberately lenient.  The agent is free to put HYPHEN and SPACE
        anywhere, including nowhere and everywhere, so this never rejects a
        message: split on SPACE, split each piece on HYPHEN, drop empty pieces.
        A stray leading hyphen costs the speaker a symbol and changes nothing
        else -- whether agents learn to use these marks consistently is measured
        (see metrics.word_stats), not imposed.
        """
        return parse_words(cfg, self.utterance(cfg, turn))

    def role_words(self, cfg: Config, role: int) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []
        for turn in range(cfg.channel.n_turns):
            if speaker_of_turn(turn) == role:
                out.extend(self.words(cfg, turn))
        return out


# --------------------------------------------------------------------------
# Scripted agents, used to validate the environment before any learning exists
# (spec 7 step 1) and as a chance-level baseline afterwards.
# --------------------------------------------------------------------------
class RandomScriptedAgent:
    """Emits uniformly random tokens and a uniformly random decision."""

    def __init__(self, cfg: Config, role: int, rng):
        self.cfg, self.role, self.rng = cfg, role, rng

    def speak(self, obs, history) -> list[int]:
        """Random symbols, including the structural ones, then END."""
        c = self.cfg.channel
        n_atoms = self.rng.randint(1, max(1, c.max_symbols // 2))
        syms = [self.rng.randrange(c.atomic_vocab)]
        for _ in range(n_atoms - 1):
            syms.append(self.rng.choice((c.hyphen_id, c.space_id)))
            syms.append(self.rng.randrange(c.atomic_vocab))
        syms = syms[:c.max_symbols - 1]
        if syms and not c.is_atom(syms[-1]):
            syms = syms[:-1]
        syms.append(c.end_id)
        return syms[:c.max_symbols]

    def decide(self, obs, history) -> Decision:
        w = self.cfg.world
        return Decision(accept=self.rng.randint(0, 1),
                        variety=self.rng.randrange(w.n_varieties),
                        qty=self.rng.randint(0, w.max_qty),
                        price=self.rng.randrange(w.n_price_bins))

    def believe(self, obs, history) -> Beliefs:
        w = self.cfg.world
        return Beliefs(variety=self.rng.randrange(w.n_varieties),
                       qty=self.rng.randint(0, w.max_qty),
                       quality=self.rng.randrange(w.n_quality),
                       price=self.rng.randrange(w.n_price_bins))


class HonestScriptedAgent:
    """An oracle pair used only in tests: ignores the channel and plays the true deal.

    This exists to prove the environment *can* be solved and that ``resolve``
    scores a correct joint play as a success.  It is never used in training -- it
    cheats by construction (it is handed the scenario), which is exactly what the
    learned agents are forbidden from doing.
    """

    def __init__(self, cfg: Config, role: int, rng, scenario_ref):
        self.cfg, self.role, self.rng = cfg, role, rng
        self.scenario_ref = scenario_ref

    def speak(self, obs, history) -> list[int]:
        return [self.cfg.channel.end_id]

    def decide(self, obs, history) -> Decision:
        sc: Scenario = self.scenario_ref()
        if not sc.viable:
            return Decision(0, 0, 0, 0)
        lo, hi = sc.zopa
        return Decision(1, sc.deal_variety, sc.deal_qty, (lo + hi) // 2)

    def believe(self, obs, history) -> Beliefs:
        sc: Scenario = self.scenario_ref()
        if self.role == FARMER:
            b = sc.buyer
            return Beliefs(b.want_variety, b.need_qty, b.min_quality, b.max_price)
        return Beliefs(sc.buyer.want_variety, sc.offered_stock,
                       sc.offered_quality, sc.farmer.reservation)


def run_scripted_episode(cfg: Config, scenario: Scenario, farmer_agent, buyer_agent) -> Transcript:
    """Drive one episode with objects exposing ``speak``/``decide`` (no torch)."""
    c = cfg.channel
    tokens = [c.pad_id] * c.dialogue_len
    history: list[tuple[int, list[int]]] = []

    for turn in range(c.n_turns):
        role = speaker_of_turn(turn)
        agent = farmer_agent if role == FARMER else buyer_agent
        said = agent.speak(obs_for(role, scenario, cfg), history)
        said = list(said)[: c.max_msg_len]
        for k, tok in enumerate(said):
            tokens[turn * c.max_msg_len + k] = tok
            if tok == c.end_id:
                break
        history.append((role, said))

    fd = farmer_agent.decide(farmer_obs(scenario, cfg), history)
    bd = buyer_agent.decide(buyer_obs(scenario, cfg), history)
    fb = farmer_agent.believe(farmer_obs(scenario, cfg), history)
    bb = buyer_agent.believe(buyer_obs(scenario, cfg), history)

    def n_costed(role: int) -> int:
        n = 0
        for turn in range(c.n_turns):
            if speaker_of_turn(turn) != role:
                continue
            for sym in tokens[turn * c.max_symbols:(turn + 1) * c.max_symbols]:
                if c.costed(sym):
                    n += 1
        return n

    outcome = resolve(cfg, scenario, fd, bd, n_costed(FARMER), n_costed(BUYER),
                      f_beliefs=fb, b_beliefs=bb)
    return Transcript(tokens=tokens, scenario=scenario, farmer_decision=fd,
                      buyer_decision=bd, outcome=outcome,
                      farmer_beliefs=fb, buyer_beliefs=bb)
