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
            farmer_tokens: int = 0, buyer_tokens: int = 0) -> Outcome:
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
        n = self.rng.randint(1, c.max_symbols)
        syms = [self.rng.randrange(c.end_id) for _ in range(n - 1)]
        syms.append(c.end_id)
        return syms[:c.max_symbols]

    def decide(self, obs, history) -> Decision:
        w = self.cfg.world
        return Decision(accept=self.rng.randint(0, 1),
                        variety=self.rng.randrange(w.n_varieties),
                        qty=self.rng.randint(0, w.max_qty),
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

    def n_costed(role: int) -> int:
        n = 0
        for turn in range(c.n_turns):
            if speaker_of_turn(turn) != role:
                continue
            for sym in tokens[turn * c.max_symbols:(turn + 1) * c.max_symbols]:
                if c.costed(sym):
                    n += 1
        return n

    outcome = resolve(cfg, scenario, fd, bd, n_costed(FARMER), n_costed(BUYER))
    return Transcript(tokens=tokens, scenario=scenario, farmer_decision=fd,
                      buyer_decision=bd, outcome=outcome)
