"""A curriculum: learn to refer before learning to haggle.

Why this exists
---------------
Dropped straight into the full trading task, agents have to solve five things at
once before any of them pays off even once: emit a stable non-arbitrary signal,
put true private information into it, have the other side decode it, close the
loop so that decoding changes a decision, and get the trade arithmetic (budget,
viability, quantity) right as well. The most recent run showed exactly what that
costs -- success 0.000 at every checkpoint, comprehension 0.000 throughout, and
scrambling the channel costing nothing, because there was nothing to scramble.

So the task is built up in stages, and a stage is only left behind once it has
actually worked:

  1a. ``refer``        A lineup game. The informer (the farmer) sees one meaning
                       and describes it; the guesser sees several candidates and
                       picks. No price, no budget, no negotiation, no market.
                       Success is 1/K by chance, a gradient RL can actually climb.
  1b. ``refer-swap``   The same game, but the roles alternate: every agent has to
                       both describe and decode. Promotion needs *each* role to
                       clear the bar on its own -- a farmer that talks and a buyer
                       that only listens is exactly the one-way code this rung
                       exists to rule out.
  1c. ``refer-mutual`` Both agents hold a private meaning and each has to report
                       the other's -- every field, quantity exactly. Bidirectional
                       exchange, still with no price and no accept/reject.
  1d. ``order``        The buyer states what it wants; the farmer must fill the order
                       exactly with its *deal* decision. The first rung that uses the
                       deal heads, and the one thing it adds is acting on what was
                       heard. Still no price, no accept/reject.
  2.  ``haggle``       Price and budget appear, so there is a real accept/reject
                       with a payoff -- but still one message each and then decide.
  3.  ``bargain``      The same, with multiple turns, so counter-offers are possible.
  4.  ``market``       The full economy: persistent stock, restocking, several
                       goods, viability. The last thing agents meet.

Each rung adds one thing. Each has a minimum and a maximum episode budget: a rung
that meets its criteria is left promptly, and one that runs past its maximum
without meeting them stops the run with a report instead of burning the rest of
the budget on a rung that is not converging.

Who speaks when is a property of the rung, not of the code base. Everything that
needs to know which dialogue slots an agent produced -- the symbol cost, the
bottleneck's targets, the speaker embedding, the probes that extract per-meaning
forms -- asks the phase (:meth:`Phase.own_positions`), because the fixed
buyer-opens schedule of the trading task is wrong for every lineup rung.

Weights carry across phase boundaries -- the population that learned to refer is
the population that learns to haggle. Nothing is reinitialised at a transition.
(The transmission bottleneck still applies normally to newborns *within* a phase;
that is a different mechanism and is untouched here.)

Every phase shares one sequence layout, one channel and one set of heads, so
"carry the weights forward" is literally the same modules continuing to train.
Phases that use fewer turns simply leave the later dialogue slots empty.

Promotion is on evidence, not on a schedule: see :class:`Promotion`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import torch

from .config import Config
from .env import BUYER, FARMER
from .world import K_COLOR, K_EMPTY, K_FIELD, K_QUALITY, K_VARIETY

# Head layout shared by every phase.  Sampling all of them keeps tensor shapes
# constant across a transition; each phase says which ones actually count.
H_ACCEPT, H_VARIETY, H_QTY, H_PRICE = 0, 1, 2, 3
H_BELIEF = (4, 5, 6, 7)          # the other party's (fruit, quantity, quality, price)
H_CHOICE = 8                     # which candidate in a lineup
H_BELIEF_COLOR = 9               # the other party's colour -- appended, so 0..8 kept
N_HEADS = 10
# Reporting a thing: (fruit, colour, quality), the three fields a meaning has.
H_REPORT = (H_BELIEF[0], H_BELIEF_COLOR, H_BELIEF[2])

# The request rungs. A buyer's order is built up one field at a time -- the same
# scaffolding the naming rungs use, carried into the trade format -- and the
# answer rungs run it the other way, with the farmer describing the lot the
# buyer asked about. Each name maps to the head that has to carry it.
ASK_HEADS = {"fruit": H_VARIETY, "colour": H_BELIEF_COLOR,
             "quantity": H_QTY, "price": H_PRICE}
ANSWER_HEADS = {"stock": H_BELIEF[1], "quality": H_BELIEF[2],
                "reservation": H_BELIEF[3], "deal": H_ACCEPT}


# ==========================================================================
KIND_REFER, KIND_SWAP, KIND_MUTUAL, KIND_TRADE = "refer", "swap", "mutual", "trade"
KIND_ORDER = "order"


def other_role(role: int) -> int:
    return BUYER if role == FARMER else FARMER


@dataclass(frozen=True)
class Phase:
    """One rung of the ladder."""
    name: str
    index: int
    n_turns: int
    informer_first: bool          # who opens: the describer, or the buyer
    use_price: bool               # is there a budget and an accept/reject payoff
    use_market: bool              # persistent stock, restocking, viability
    blurb: str
    kind: str = KIND_TRADE
    # Who describes in a lineup rung, alternated batch by batch via
    # :meth:`with_informer` so every agent does both jobs.
    informer: int = FARMER
    # What kinds of round this rung is made of, as weights over
    # (fruit, colour, quality, all-fields). A naming rung *adds* a kind and keeps
    # rehearsing the ones below it: switching outright cost the run its fruit
    # code and its gradient at once -- the messages still carried fruit
    # (coverage 0.40) and nothing else, at chance, for 500 updates.
    mix: tuple = (0.0, 0.0, 0.0, 1.0)
    # The kind this rung introduces: 0 fruit, 1 colour, 2 quality, 3 all fields.
    # Promotion is judged on *this* kind, so acing the rehearsal cannot carry a
    # rung whose new job is not being done.
    primary: int = 3
    # A request rung (``KIND_ORDER``): which fields of the order have to arrive,
    # and which of them this rung introduces. Like the naming rungs, a request
    # rung *adds* a field and keeps the earlier ones in play, and promotion is
    # judged on the one it added -- a conjunction of four fields would otherwise
    # hide which one is at chance, which is exactly how `haggle` failed before
    # (quality 0.88, variety 0.52, quantity 0.20 -- and one number, 0.07).
    ask: tuple = ()
    asks_first: str = ""          # the field this rung introduces
    # Which way the request runs. False: the buyer orders and the farmer fills
    # it. True: the farmer answers about the lot that was asked for, and the
    # buyer has to report what it was told -- the direction `haggle` needs and
    # no rung below it ever trained.
    answers: bool = False

    # ---- what kind of game this is --------------------------------------
    @property
    def referential(self) -> bool:
        """A lineup round: one describer, one guesser, a choice among K."""
        return self.kind in (KIND_REFER, KIND_SWAP)

    @property
    def mutual(self) -> bool:
        return self.kind == KIND_MUTUAL

    @property
    def swaps(self) -> bool:
        return self.kind == KIND_SWAP

    @property
    def order(self) -> bool:
        return self.kind == KIND_ORDER

    @property
    def tuples(self) -> bool:
        """Played over bare (fruit, colour, quality) things, not farms and requests."""
        return self.kind in (KIND_REFER, KIND_SWAP, KIND_MUTUAL)

    @property
    def naming(self) -> bool:
        """A rung whose whole job is learning to name things."""
        return self.referential or self.mutual

    @property
    def query(self) -> "int | None":
        """The single field this rung is about, if it is about one."""
        return None if self.primary >= ASK_ALL else self.primary

    @property
    def mixed_query(self) -> bool:
        """Does this rung ask about different fields in different rounds?"""
        return sum(1 for w in self.mix if w > 0) > 1

    @property
    def kinds(self) -> tuple:
        """The kinds of round this rung draws, commonest first."""
        return tuple(sorted((i for i, w in enumerate(self.mix) if w > 0),
                            key=lambda i: -self.mix[i]))

    @property
    def rehearsed(self) -> tuple:
        """The kinds carried over from earlier rungs -- what must not be lost."""
        return tuple(k for k in self.kinds if k != self.primary)

    @property
    def whole(self) -> bool:
        """Is this rung's own job to name a whole (fruit, colour, quality)?"""
        return self.naming and self.primary >= ASK_ALL

    @property
    def ask_heads(self) -> dict:
        """The head each asked-for field has to arrive in."""
        table = ANSWER_HEADS if self.answers else ASK_HEADS
        return {f: table[f] for f in self.ask}

    @property
    def reporter(self) -> int:
        """Who has to get the fields right: the side that was told them."""
        return BUYER if self.answers else FARMER

    @property
    def trading(self) -> bool:
        return self.kind == KIND_TRADE

    @property
    def guesser(self) -> int:
        return other_role(self.informer)

    def with_informer(self, role: int) -> "Phase":
        return replace(self, informer=role)

    def views(self) -> list["Phase"]:
        """The concrete games this rung is made of. A swap rung is two."""
        if self.swaps:
            return [self.with_informer(FARMER), self.with_informer(BUYER)]
        return [self]

    # ---- who says what, where -------------------------------------------
    def speaker_of_turn(self, turn: int) -> int:
        """Whose turn it is. In the lineup rungs the describer talks first."""
        first = self.informer if self.informer_first else BUYER
        return first if turn % 2 == 0 else other_role(first)

    def turns_of(self, cfg: Config, role: int) -> list[int]:
        return [t for t in range(min(self.n_turns, cfg.channel.n_turns))
                if self.speaker_of_turn(t) == role]

    def speaks(self, cfg: Config, role: int) -> bool:
        return bool(self.turns_of(cfg, role))

    def own_positions(self, cfg: Config, role: int) -> list[int]:
        """Dialogue-buffer indices (0..D-1) that ``role`` produces in this phase."""
        L = cfg.channel.max_msg_len
        out: list[int] = []
        for t in self.turns_of(cfg, role):
            out.extend(range(t * L, (t + 1) * L))
        return out

    def self_mask(self, cfg: Config, role: int, device=None) -> torch.Tensor:
        """(D,) bool: True where ``role`` is the speaker -- the "me" embedding."""
        m = torch.zeros(cfg.channel.dialogue_len, dtype=torch.bool)
        pos = self.own_positions(cfg, role)
        if pos:
            m[pos] = True
        return m if device is None else m.to(device)

    def read_positions(self, cfg: Config, role: int, device="cpu") -> torch.Tensor:
        """Sequence indices whose hidden state emits this role's own slots."""
        from .agents import dialogue_offset
        off = dialogue_offset(cfg)
        return torch.tensor([off + p - 1 for p in self.own_positions(cfg, role)],
                            dtype=torch.long, device=device)

    def active_heads(self, role: int, cfg: Config) -> list[int]:
        """Which outputs are scored, and therefore which ones get a gradient.

        Sampling a head nobody scores only adds noise to the policy-gradient
        term, so each phase names what it actually uses.
        """
        if self.referential:
            # The describer only speaks; it has no decision to make, and its
            # gradient arrives entirely through the guesser's forward pass.
            return [H_CHOICE] if role == self.guesser else []
        if self.mutual:
            # Report the partner's thing: fruit, colour, quality.
            return list(H_REPORT)
        if self.order:
            # Only the side that has to act on what it heard has a decision; the
            # side holding the facts just says them.
            return list(self.ask_heads.values()) if role == self.reporter else []
        heads = [H_ACCEPT, H_VARIETY, H_QTY]
        if self.use_price:
            heads.append(H_PRICE)
        if cfg.reward.belief_heads:
            heads.extend(H_BELIEF)
        return heads

    def meaning_kind(self, role: int) -> str:
        """What a speaker's utterance is about, for pooling conventions.

        Every lineup rung describes the same kind of thing -- a (variety,
        quantity, quality) tuple -- whichever role is talking, so a farmer's word
        and a buyer's word for the same tuple are the same convention.
        """
        if self.tuples:
            return "tuple"
        return "request" if role == BUYER else "barn"


def ladder(cfg: Config) -> list[Phase]:
    """The rungs, easiest first.  ``n_turns`` never exceeds ``channel.n_turns``.

    The first five teach naming, one field at a time and then together, before
    anything is traded.  Each is a lineup: the describer sees one thing, the
    guesser sees the candidates and picks.  Everyone takes both seats, because
    the describer alternates batch by batch and (below ``split_roles_at``) both
    seats are filled from one pool of agents speaking one language.
    """
    full = max(2, cfg.channel.n_turns)
    return [
        Phase("name-fruit", 0, 1, True, False, False,
              "lineup game over fruit alone: the candidates share colour and "
              "quality, so only the fruit needs saying",
              kind=KIND_SWAP, mix=(1.0, 0.0, 0.0, 0.0), primary=0),
        Phase("name-color", 1, 1, True, False, False,
              "colour rounds added to fruit ones: a word for a colour and nothing "
              "else, while the fruit words stay in use and stay needed",
              kind=KIND_SWAP, mix=(0.4, 0.6, 0.0, 0.0), primary=1),
        Phase("name-quality", 2, 1, True, False, False,
              "quality rounds added: every round still asks about one field, but "
              "which field changes, so a word has to mean the same thing wherever "
              "it appears", kind=KIND_SWAP, mix=(0.25, 0.25, 0.5, 0.0), primary=2),
        Phase("name-all", 3, 1, True, False, False,
              "rounds where the candidates differ in any field, mostly one-field "
              "near misses, so the whole (fruit, colour, quality) has to be named "
              "at once -- with single-field rounds still mixed in",
              kind=KIND_SWAP, mix=(0.1, 0.1, 0.1, 0.7), primary=3),
        Phase("mutual", 4, 2, True, False, False,
              "both hold a private thing and each must report the other's; "
              "no price, no accept/reject", kind=KIND_MUTUAL),
        Phase("ask-qty", 5, 1, False, False, False,
              "the first order: the buyer says how many it needs and the farmer "
              "has to fill that number. Quantity is the one field the naming "
              "rungs never asked for, and the rung that follows needs it",
              kind=KIND_ORDER, ask=("quantity",), asks_first="quantity"),
        Phase("order", 6, 1, False, False, False,
              "the whole order: fruit, colour and quantity together, so the words "
              "from the naming rungs have to work in a request",
              kind=KIND_ORDER, ask=("fruit", "colour", "quantity"),
              asks_first="fruit"),
        Phase("quote", 7, 1, False, False, False,
              "the order now carries the price the buyer will pay, so every "
              "price bin needs a word before any price has to be agreed",
              kind=KIND_ORDER, ask=("fruit", "colour", "quantity", "price"),
              asks_first="price"),
        Phase("offer", 8, 2, False, False, False,
              "the other direction: the buyer asks about a lot and the farmer "
              "answers with what it holds -- how much, what quality, what it "
              "wants for it -- and the buyer has to report what it was told",
              kind=KIND_ORDER, ask=("stock", "quality", "reservation"),
              asks_first="stock", answers=True),
        Phase("judge", 9, 2, False, False, False,
              "the same dialogue, and now one decision: is this deal any good? "
              "The buyer has to weigh what it was told against what it needs, "
              "with nothing yet riding on the answer",
              kind=KIND_ORDER, ask=("deal",), asks_first="deal", answers=True),
        Phase("haggle", 10, 2, False, True, False,
              "both sides now decide: budget and reservation make a deal "
              "refusable, and it only counts if both name the same one"),
        Phase("bargain", 11, full, False, True, False,
              "several turns, so counter-offers are possible"),
        Phase("market", 12, full, False, True, True,
              "the full economy: stock, restocking, viability"),
    ]


def phase_named(cfg: Config, name: str) -> Phase:
    for p in ladder(cfg):
        if p.name == name:
            return p
    raise KeyError("no phase called %r" % (name,))


def hindsight_applies(cfg: Config, phase: Phase) -> bool:
    """Is hindsight feedback on in this rung? (``train.hindsight_from_rung``)"""
    if cfg.train.hindsight_coef <= 0:
        return False
    return phase.index >= phase_named(cfg, cfg.train.hindsight_from_rung).index


def costs_apply(cfg: Config, phase: Phase) -> bool:
    """Does the speaker pay for what it says in this rung? (``reward.costs_from_rung``)"""
    return phase.index >= phase_named(cfg, cfg.reward.costs_from_rung).index


def convention_applies(cfg: Config, phase: Phase) -> bool:
    """Is a speaker paid for using the community's word? (``reward.convention_from_rung``)"""
    return phase.index >= phase_named(cfg, cfg.reward.convention_from_rung).index


def growth_applies(cfg: Config, phase: Phase) -> bool:
    """May newcomers join in this rung? (``population.grow_from_rung``)"""
    return phase.index >= phase_named(cfg, cfg.population.grow_from_rung).index


def pooled_at(cfg: Config, phase: Phase) -> bool:
    """Do both seats come from one pool in this rung? (``curriculum.split_roles_at``)

    Below the split this is an *identity*, not a coincidence: index i of
    ``Population.farmers`` and index i of ``Population.buyers`` are the same
    object, which is why :meth:`Population.pair` can seat i opposite a
    different index and know it has not seated an agent against itself.
    Anything that rebuilds the two lists has to preserve it.
    """
    at = cfg.curriculum.split_roles_at
    if not (cfg.curriculum.enabled and at):
        return False
    return phase.index < phase_named(cfg, at).index


def turnover_applies(cfg: Config, phase: Phase) -> bool:
    """Do agents die of old age in this rung? (``population.turnover_from_rung``)"""
    if not cfg.population.turnover:
        return False
    return phase.index >= phase_named(cfg, cfg.population.turnover_from_rung).index


def rung_budget(cfg: Config, phase: Phase) -> tuple[int, int]:
    """(minimum, maximum) training updates this rung may take."""
    c = cfg.curriculum
    b = (c.rung_budget_updates or {}).get(phase.name) or c.default_rung_updates
    return int(b[0]), int(b[1])


# ==========================================================================
@dataclass
class Promotion:
    """What a phase has to show before the next one is allowed to start.

    All three have to hold at the same checkpoint.  Success alone is not enough --
    a pair can score on base rates without saying anything -- so the channel
    control has to show that muting the messages actually costs something, and
    topological similarity has to be clear of its own shuffled null.
    """
    min_success: float
    min_success_over_chance: float
    min_topsim_over_null: float
    min_channel_transfer: float
    min_updates: int
    max_updates: int

    def evaluate(self, *, success: float, chance: float, topsim: float,
                 null: float, transfer: float, updates_in_phase: int
                 ) -> tuple[bool, dict[str, Any]]:
        def ok(x) -> bool:
            return isinstance(x, float) and x == x

        checks = {
            "long enough in phase": (updates_in_phase >= self.min_updates,
                                     "%d of %d updates" % (updates_in_phase,
                                                           self.min_updates)),
            "success above floor": (ok(success) and success >= self.min_success,
                                    "%.3f, need %.3f" % (success if ok(success) else float("nan"),
                                                         self.min_success)),
            "success above chance": (
                ok(success) and ok(chance)
                and (success >= self.min_success_over_chance * chance if chance > 1e-9
                     else success >= self.min_success),
                "%.3f vs chance %.3f, need %.1fx" % (
                    success if ok(success) else float("nan"),
                    chance if ok(chance) else float("nan"),
                    self.min_success_over_chance)),
            "topsim clear of null": (
                ok(topsim) and ok(null) and (topsim - null) >= self.min_topsim_over_null,
                "%.3f vs null %.3f, need +%.2f" % (
                    topsim if ok(topsim) else float("nan"),
                    null if ok(null) else float("nan"), self.min_topsim_over_null)),
            "channel actually carries": (
                ok(transfer) and transfer >= self.min_channel_transfer,
                "%.2f of headroom, need %.2f" % (
                    transfer if ok(transfer) else float("nan"),
                    self.min_channel_transfer)),
        }
        passed = all(v[0] for v in checks.values())
        return passed, {k: {"met": v[0], "detail": v[1]} for k, v in checks.items()}


def round_chance(cfg: Config, kind: Optional[int]) -> float:
    """One in how many candidates, by luck alone, in a round of this kind.

    A lineup cannot be wider than the field it varies: over four colours a
    colour round is one in three only because three candidates were asked for.
    """
    K = max(2, cfg.curriculum.n_candidates)
    if kind is not None and int(kind) < ASK_ALL:
        span = (cfg.world.n_varieties, cfg.world.n_colors, cfg.world.n_quality)
        K = min(K, span[int(kind)])
    return 1.0 / K


def promotion_for(cfg: Config, phase: Phase) -> Promotion:
    c = cfg.curriculum
    lo, hi = rung_budget(cfg, phase)
    if phase.referential:
        # Judged on the kind the rung introduces, not on its mixture: by then the
        # rehearsal is easy, and pooled success would carry a rung that never did
        # its own job. What the rehearsal has to show is checked separately.
        chance = round_chance(cfg, phase.primary)
        return Promotion(
            # The floor and the "clear of chance" multiple are the same question
            # asked twice; the multiple was hardcoded here, so tuning
            # `min_success_over_chance` moved one bar and not the other.
            min_success=max(c.refer_min_success, c.min_success_over_chance * chance),
            min_success_over_chance=c.min_success_over_chance,
            min_topsim_over_null=c.min_topsim_over_null,
            min_channel_transfer=c.min_channel_transfer,
            min_updates=lo, max_updates=hi)
    if phase.order:
        return Promotion(
            min_success=c.order_min_success,
            min_success_over_chance=c.min_success_over_chance,
            min_topsim_over_null=c.min_topsim_over_null,
            min_channel_transfer=c.min_channel_transfer,
            min_updates=lo, max_updates=hi)
    if phase.mutual:
        return Promotion(
            min_success=c.mutual_min_success,
            min_success_over_chance=c.min_success_over_chance,
            min_topsim_over_null=c.min_topsim_over_null,
            min_channel_transfer=c.min_channel_transfer,
            min_updates=lo, max_updates=hi)
    return Promotion(
        min_success=c.trade_min_success,
        min_success_over_chance=c.min_success_over_chance,
        min_topsim_over_null=c.min_topsim_over_null,
        min_channel_transfer=c.min_channel_transfer,
        min_updates=lo, max_updates=hi)


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) and x == x else float("nan")


def _fmt(x: float) -> str:
    return "%.3f" % x if x == x else "n/a"


def _headroom_floor(muted: float, share: float) -> float:
    """The score that takes ``share`` of the headroom left above silence."""
    if muted != muted:
        return 0.0
    return muted + (1.0 - muted) * share


def evaluate_request_rung(cfg: Config, phase: Phase, ev: dict[str, Any],
                          updates_in_phase: int, rule: "Promotion"
                          ) -> tuple[bool, dict[str, Any]]:
    """Has an order arrived, field by field?

    Judged on the field this rung introduced -- the rest were introduced below
    it and are checked separately, so a rung cannot pass on work it did last
    time, and cannot fail invisibly because one field of four is at chance.
    """
    c = cfg.curriculum
    k = c.min_success_over_chance
    checks: dict[str, tuple[bool, str]] = {}
    checks["long enough in rung"] = (
        updates_in_phase >= rule.min_updates,
        "%d of %d updates" % (updates_in_phase, rule.min_updates))

    new = phase.asks_first or (phase.ask[0] if phase.ask else "")
    got, muted = _num(ev.get("request_first")), _num(ev.get("muted_request_first"))
    # An absolute floor, plus a real gain over silence. "Twice the muted rate" is
    # the bar everywhere else, but it is unreachable for a field a mute agent
    # already gets most of the time -- `judge` is one binary decision whose base
    # rate is ~0.68, and 1.36 is not a score. The gain over silence, which the
    # per-field check below states as a share of the headroom, is the honest form
    # of the same question and is what an always-accept policy fails.
    floor = max(rule.min_success, _headroom_floor(muted, c.min_field_transfer))
    checks["%s arrives" % new] = (
        got == got and got >= floor,
        "%s, need %.2f (silence alone scores %s)" % (_fmt(got), floor, _fmt(muted)))

    per = ev.get("request_field_transfer") or []
    intact = ev.get("request_fields_intact") or []
    for name, t, a in zip(phase.ask, per, intact):
        t, a = _num(t), _num(a)
        label = ("still carries %s" % name if name != new else "%s carries" % name)
        checks[label] = (
            t == t and t >= c.min_field_transfer,
            "%s right, %s of the headroom over a muted channel, need %.2f"
            % (_fmt(a), _fmt(t), c.min_field_transfer))

    if len(phase.ask) > 1:
        succ, chance = _num(ev.get("success")), _num(ev.get("chance"))
        checks["the whole order arrives"] = (
            succ == succ and (chance != chance or succ >= k * chance),
            "%s complete, against %s with the channel muted" % (_fmt(succ), _fmt(chance)))

    tr = _num(ev.get("transfer"))
    checks["channel carries"] = (
        tr == tr and tr >= c.min_channel_transfer,
        "%s of the headroom, need %.2f" % (_fmt(tr), c.min_channel_transfer))
    ts, nl = _num(ev.get("topsim")), _num(ev.get("null"))
    checks["topsim clear of null"] = (
        ts == ts and nl == nl and ts - nl >= c.min_topsim_over_null,
        "%s vs null %s, need +%.2f" % (_fmt(ts), _fmt(nl), c.min_topsim_over_null))

    passed = all(v[0] for v in checks.values())
    return passed, {n: {"met": v[0], "detail": v[1]} for n, v in checks.items()}


def evaluate_rung(cfg: Config, phase: Phase, ev: dict[str, Any],
                  updates_in_phase: int) -> tuple[bool, dict[str, Any]]:
    """Has this rung demonstrably worked?  Returns (passed, named checks).

    ``ev`` is :func:`orchard.metrics.phase_evidence`'s output. ``refer`` and the
    trading rungs are judged exactly as before, on pooled numbers. The two rungs
    that exist to make communication two-way are judged per role, and every
    role has to clear every bar on its own: a pooled average would let a
    fluent farmer carry a buyer that never learned to speak, which is the
    precise failure the previous run showed (farmer positional structure 0.03
    against the buyer's 0.39).
    """
    c = cfg.curriculum
    rule = promotion_for(cfg, phase)
    if phase.order:
        return evaluate_request_rung(cfg, phase, ev, updates_in_phase, rule)
    if not (phase.swaps or phase.mutual):
        spk = ev.get("speakers", {})
        # the describer's structure in the lineup; both speakers' otherwise
        if phase.referential:
            d = spk.get("farmer" if phase.informer == FARMER else "buyer", {})
            topsim, null = _num(d.get("topsim")), _num(d.get("null"))
        else:
            topsim, null = _num(ev.get("topsim")), _num(ev.get("null"))
        return rule.evaluate(success=_num(ev.get("success")), chance=_num(ev.get("chance")),
                             topsim=topsim, null=null,
                             transfer=_num(ev.get("transfer")),
                             updates_in_phase=updates_in_phase)

    checks: dict[str, tuple[bool, str]] = {}
    checks["long enough in rung"] = (
        updates_in_phase >= rule.min_updates,
        "%d of %d updates" % (updates_in_phase, rule.min_updates))
    k = c.min_success_over_chance
    # Structure is only a fair bar where a round turns on all three fields at
    # once. A rung that varies one field has nothing to be compositional *about*:
    # what it has to show is that the word works, and keeps working on
    # combinations never trained on (the held-out check below).
    whole_thing = phase.swaps and phase.whole
    for role, label in ((FARMER, "farmer"), (BUYER, "buyer")):
        d = ev.get("speakers", {}).get(label, {})
        ts, nl, pos = _num(d.get("topsim")), _num(d.get("null")), _num(d.get("positional"))
        if whole_thing or phase.mutual:
            checks["%s describes: topsim clear of null" % label] = (
                ts == ts and nl == nl and ts - nl >= c.min_topsim_over_null,
                "%s vs null %s, need +%.2f" % (_fmt(ts), _fmt(nl), c.min_topsim_over_null))
            checks["%s describes: positional structure" % label] = (
                pos == pos and pos >= c.min_positional_structure,
                "%s, need %.2f" % (_fmt(pos), c.min_positional_structure))
            # Positional structure is fooled by redundancy -- a13-a13-a13-a13
            # scores 1.0 because every slot names the fruit. Coverage asks how
            # much of *each* field the messages carry, corrected for chance.
            cov = _num(d.get("field_coverage"))
            checks["%s describes: covers every field" % label] = (
                cov == cov and cov >= c.min_field_coverage,
                "%s of each field's information on average, need %.2f"
                % (_fmt(cov), c.min_field_coverage))
        if phase.swaps:
            # the view in which this role is the one that has to decode
            view = next((v for v in ev.get("views", [])
                         if v.get("guesser") == label), {})
            succ, chance = _num(view.get("success")), _num(ev.get("chance"))
            tr = _num(view.get("transfer"))
            floor = rule.min_success
            checks["%s decodes: success" % label] = (
                succ == succ and succ >= floor and succ >= k * chance,
                "%s as guesser, need %.3f and %.1fx chance %.3f"
                % (_fmt(succ), floor, k, chance))
            checks["%s decodes: channel carries" % label] = (
                tr == tr and tr >= c.min_channel_transfer,
                "%s of headroom over a muted channel, need %.2f"
                % (_fmt(tr), c.min_channel_transfer))
        else:
            acc = _num(ev.get("%s_report" % label))
            muted = _num(ev.get("muted_%s_report" % label))
            tr = _num(ev.get("%s_report_transfer" % label))
            checks["%s decodes: reports partner's tuple" % label] = (
                acc == acc and acc >= c.mutual_min_report,
                "%s exact (quantity within %d), need %.2f; muted %s"
                % (_fmt(acc), cfg.reward.belief_qty_tol, c.mutual_min_report, _fmt(muted)))
            checks["%s decodes: channel carries" % label] = (
                tr == tr and tr >= c.min_channel_transfer,
                "%s of headroom over a muted channel, need %.2f"
                % (_fmt(tr), c.min_channel_transfer))
            # Every field, not just the tuple: a code that carries quality and a
            # little variety can clear a whole-tuple bar with quantity at chance,
            # and the next rungs need quantity exactly.
            ft = ev.get("%s_field_transfer" % label) or []
            for name, t in zip(("fruit", "colour", "quality"), ft):
                t = _num(t)
                checks["%s decodes: %s" % (label, name)] = (
                    t == t and t >= c.min_field_transfer,
                    "%s of headroom over a muted channel, need %.2f"
                    % (_fmt(t), c.min_field_transfer))
    # Nothing learned lower down may be dropped to pick this rung up. Every kind
    # of round the rung rehearses is scored on its own and has to stay clearly
    # above chance: forgetting the fruit words while learning colour is not
    # progress, and one pooled number hides it.
    by_kind = ev.get("by_kind") or {}
    for kind in phase.rehearsed:
        d = by_kind.get(kind, by_kind.get(str(kind)))
        if not d:
            continue
        s, ch = _num(d.get("success")), round_chance(cfg, kind)
        # A detector for forgetting, not a second promotion: the kind was
        # promoted at the full bar once already, and a rung spends its first
        # stretch exploring (`train.anneal_per_rung`), which shakes every word a
        # little. What must not happen is a field sliding back to chance, so the
        # bar is the share of the headroom the channel checks use everywhere.
        # (At the full bar, the one run that exercised this passed at 0.670
        # against 0.667 -- a coin toss away from stalling a rung that worked.)
        floor = _headroom_floor(ch, c.min_channel_transfer)
        checks["still names %s" % ROUND_NAMES[kind]] = (
            s == s and s >= floor,
            "%s on %s rounds, need %.2f -- %.2f of the headroom over chance %.3f"
            % (_fmt(s), ROUND_NAMES[kind], floor, c.min_channel_transfer, ch))
    if phase.mutual:
        succ, chance = _num(ev.get("success")), _num(ev.get("chance"))
        checks["both decode in the same round"] = (
            succ == succ and succ >= rule.min_success
            and (chance != chance or succ >= k * chance),
            "%s, need %.3f and %.1fx the muted rate %s"
            % (_fmt(succ), rule.min_success, k, _fmt(chance)))
    whole = phase.whole
    if whole:
        # The productivity gate, on the rungs that describe a whole thing. Every
        # candidate in a held-out round is a combination nobody ever trained on,
        # so a code that names whole things has nothing to say about any of them,
        # however well it scores on the ones it drilled; one with reusable parts
        # describes them as easily as anything else. A rung that varies a single
        # field is not asked for this: it has not been taught the rest.
        hs, seen = _num(ev.get("holdout_success")), _num(ev.get("seen_success"))
        ratio = _num(ev.get("holdout_ratio"))
        chance = _num(ev.get("chance"))
        above_chance = chance != chance or (hs == hs and hs >= k * chance)
        checks["describes combinations it never trained on"] = (
            ratio == ratio and ratio >= c.min_holdout_ratio and above_chance,
            "held-out %s vs seen %s = %s of it, need %.2f%s"
            % (_fmt(hs), _fmt(seen), _fmt(ratio), c.min_holdout_ratio,
               "" if above_chance else "; and above %.1fx chance %s" % (k, _fmt(chance))))
    passed = all(v[0] for v in checks.values())
    return passed, {name: {"met": v[0], "detail": v[1]} for name, v in checks.items()}


# ==========================================================================
# observation schemas -- one fixed layout, different content per phase
# ==========================================================================
def guesser_schema(cfg: Config) -> list[int]:
    """The lineup: K candidate things, three fields each, then the asked field."""
    return ([K_VARIETY, K_COLOR, K_QUALITY] * cfg.curriculum.n_candidates) + [K_FIELD]


def informer_schema(cfg: Config) -> list[int]:
    """One thing to describe, and which of its fields is being asked about."""
    return [K_VARIETY, K_COLOR, K_QUALITY, K_FIELD]


def phase_schema(cfg: Config, role: int, phase: Phase) -> list[int]:
    """Field kinds for each observation slot in this phase, padded to the layout."""
    from .world import buyer_schema, farmer_schema, n_obs_slots
    if phase.referential:
        base = informer_schema(cfg) if role == phase.informer else guesser_schema(cfg)
    elif phase.mutual:
        base = informer_schema(cfg)                 # each holds one thing
    elif role == FARMER:
        base = farmer_schema(cfg.world)
    else:
        base = buyer_schema(cfg.world)
    n = n_obs_slots(cfg.world, cfg)
    return base[:n] + [K_EMPTY] * max(0, n - len(base))


# ==========================================================================
# the lineup game
# ==========================================================================
ASK_ALL = 3          # the query slot's value when the whole thing is asked for
ROUND_NAMES = ("fruit", "colour", "quality", "whole things")


@dataclass
class ReferentialBatch:
    """A batch of lineup rounds, as tensors.

    ``meanings`` is (B, K, 3) -- K candidate (fruit, colour, quality) things per
    round, already shuffled -- ``target`` says which one the informer was shown,
    and ``query`` says which field the round turns on: 0 fruit, 1 colour, 2
    quality, or ``ASK_ALL`` when the candidates differ in any of them.

    A query round is where an adjective can pay for itself: every candidate
    shares the other two fields, so the only thing worth saying is the value of
    the one that differs.
    """
    meanings: torch.Tensor        # (B, K, 3)
    target: torch.Tensor          # (B,)
    query: torch.Tensor           # (B,) 0..2 or ASK_ALL
    held_out: torch.Tensor        # (B,) bool -- is the target a reserved combination
    day: int = 0
    informer: int = FARMER        # who sees the target; the other sees the lineup

    def __len__(self) -> int:
        return int(self.target.shape[0])

    @property
    def device(self) -> torch.device:
        return self.target.device

    @property
    def true_meaning(self) -> torch.Tensor:
        """(B, 3) -- what the informer sees."""
        idx = self.target.view(-1, 1, 1).expand(-1, 1, 3)
        return self.meanings.gather(1, idx).squeeze(1)

    def obs(self, cfg: Config, role: int) -> torch.Tensor:
        from .world import n_obs_slots
        n = n_obs_slots(cfg.world, cfg)
        B = len(self)
        if role == self.informer:
            x = torch.cat([self.true_meaning, self.query.unsqueeze(1)], dim=1)
        else:
            x = torch.cat([self.meanings.reshape(B, -1), self.query.unsqueeze(1)], dim=1)
        if x.shape[1] < n:
            x = torch.cat([x, torch.zeros((B, n - x.shape[1]), dtype=torch.long,
                                          device=x.device)], dim=1)
        return x[:, :n]


class ReferentialWorld:
    """Draws lineups over the same (fruit, colour, quality) things the market trades.

    Two kinds of round:

    * a **query round** (``query`` is a field): the candidates share every field
      but one, so the describer has to convey that one value and nothing else.
      This is what the naming rungs are built from, one field at a time and then
      mixed.
    * an **open round** (``query`` is ``ASK_ALL``): the candidates differ in any
      combination of fields, mostly as an anchor plus one-field near misses, so
      every field has to be named at once.

    Held-out combinations never appear as a training target.  ``held_out=True``
    draws them deliberately: that is the productivity test, and nothing in a
    fused code can pass it.
    """

    def __init__(self, cfg: Config, device: str = "cpu",
                 generator: Optional[torch.Generator] = None, holdout=None):
        from .world import ComboHoldout
        self.cfg = cfg
        self.device = torch.device(device)
        self.gen = generator
        self.holdout = holdout or ComboHoldout(cfg.world, cfg.world.holdout_combo_frac,
                                               cfg.world.holdout_seed)
        w = cfg.world
        self.spans = (w.n_varieties, w.n_colors, w.n_quality)
        held = torch.zeros(self.spans, dtype=torch.bool)
        for (f, c, q) in self.holdout.held:
            held[f, c, q] = True
        self.held_mask = held.to(self.device)
        train = self.holdout.training
        self.train_combos = torch.tensor(train, dtype=torch.long,
                                         device=self.device).reshape(-1, 3)
        self.held_combos = self.holdout.tensor(self.device)

    # ------------------------------------------------------------------
    def is_held_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.held_mask[x[:, 0], x[:, 1], x[:, 2]]

    def _draw(self, n: int, held_out: bool = False) -> torch.Tensor:
        """(n, 3) things, from the training combinations or the reserved ones."""
        pool = self.held_combos if held_out else self.train_combos
        if pool.shape[0] == 0:
            pool = self.train_combos
        idx = torch.randint(0, pool.shape[0], (n,), device=self.device, generator=self.gen)
        return pool[idx].clone()

    def _n_candidates(self, query: Optional[int]) -> int:
        """A lineup cannot be wider than the field it varies."""
        K = max(2, self.cfg.curriculum.n_candidates)
        if query is not None and query != ASK_ALL:
            K = min(K, self.spans[query])
        return K

    def chance(self, phase) -> float:
        """One in how many, by luck alone, on the kind this rung is judged on."""
        return round_chance(self.cfg, getattr(phase, "primary", None))

    def sample_mutual(self, n: int, held_out: bool = False) -> "MutualBatch":
        """Two private things per round, drawn independently."""
        return MutualBatch(f_meaning=self._draw(n, held_out),
                           b_meaning=self._draw(n, held_out))

    # ------------------------------------------------------------------
    def _query_round(self, n: int, field: int, held_out: bool) -> tuple:
        """Candidates that share every field but ``field``, which they all differ in.

        Every candidate is a combination that *could* be the answer: the one
        value of the field reserved for this pair of other fields is left out of
        the lineup entirely. Including it would hand the guesser a candidate it
        could rule out without listening -- it is never anybody's target.
        """
        sp = self.spans[field]
        K = self._n_candidates(field)
        base = self._draw(n, held_out)                       # (n, 3): the target
        rows = torch.arange(n, device=self.device)
        # every value of the queried field, with the other two held fixed
        grid = base.unsqueeze(1).repeat(1, sp, 1)
        grid[:, :, field] = torch.arange(sp, device=self.device).unsqueeze(0)
        reserved = self.is_held_out(grid.reshape(-1, 3)).view(n, sp)
        is_base = grid[:, :, field] == base[:, field].unsqueeze(1)
        # rank: real alternatives first, then reserved ones, never the target
        score = (torch.rand(n, sp, device=self.device, generator=self.gen)
                 + reserved.float() * 2.0 + is_base.float() * 4.0)
        pick = torch.argsort(score, dim=1)[:, :K - 1]
        others = grid.gather(1, pick.unsqueeze(-1).expand(-1, -1, 3))[:, :, field]
        slot = torch.randint(0, K, (n,), device=self.device, generator=self.gen)
        is_target = torch.arange(K, device=self.device).unsqueeze(0) == slot.unsqueeze(1)
        vals = torch.empty((n, K), dtype=torch.long, device=self.device)
        vals[is_target] = base[:, field]
        vals[~is_target] = others.reshape(-1)
        cand = base.unsqueeze(1).repeat(1, K, 1)
        cand[:, :, field] = vals
        return cand, slot

    def _open_round(self, n: int, held_out: bool, hard_frac: float) -> tuple:
        """Candidates that differ in any field: an anchor plus near misses, mostly.

        In a held-out round *every* candidate is a reserved combination, not just
        the target. Otherwise the target would be the only unfamiliar thing in
        the lineup and could be picked out by its novelty alone -- which is
        precisely the memorising reader this test exists to fail.
        """
        K = self._n_candidates(None)
        cand = torch.stack([self._draw(n, held_out) for _ in range(K)], dim=1)
        rows = torch.arange(n, device=self.device)
        for k in range(1, K):                 # a distractor must not be the target
            for _ in range(8):
                same = (cand[:, k] == cand[:, 0]).all(dim=1)
                if not bool(same.any()):
                    break
                cand[:, k] = torch.where(same.unsqueeze(1), self._draw(n, held_out),
                                         cand[:, k])
        target = torch.randint(0, K, (n,), device=self.device, generator=self.gen)
        perm = cand.clone()
        perm[rows, target] = cand[:, 0]
        perm[rows, 0] = cand[rows, target]
        # The independently drawn rounds are cleaned here; the clusters below
        # keep their own invariants (a near miss must stay a near miss, so it
        # cannot be replaced by an unrelated thing).
        perm = self._dedup(perm, target, held_out=held_out)
        # Hard rounds: the whole lineup is one thing plus near misses of it, each
        # differing in a single field, so no one field can carry the round alone.
        if hard_frac > 0:
            hard = torch.rand(n, device=self.device, generator=self.gen) < hard_frac
            h = int(hard.sum())
            if h:
                anchor = perm[hard][torch.arange(h, device=self.device),
                                    target[hard]]
                cl = self._cluster(anchor, K)
                # Shuffle the whole cluster and take the target uniformly from
                # it. Leaving the anchor as the target would make the target the
                # one candidate the others are all near misses *of*, and "pick
                # the most central" would find it half the time with the channel
                # muted -- which is exactly what an earlier version did.
                order = torch.argsort(torch.rand(h, K, device=self.device,
                                                 generator=self.gen), dim=1)
                perm[hard] = cl.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
                target[hard] = torch.randint(0, K, (h,), device=self.device,
                                             generator=self.gen)
        return perm, target

    def _dedup(self, cand: torch.Tensor, target: torch.Tensor,
               held_out: bool = False) -> torch.Tensor:
        """Every candidate distinct, and -- in training -- none of them reserved.

        Two identical wrong options are not wrong in the same way twice: they
        make the lineup narrower than it looks and the chance rate a lie. And a
        reserved combination must never be the thing described in training,
        which any member of a cluster may end up being.
        """
        n, K, _ = cand.shape
        for _ in range(16):
            bad = torch.zeros(n, K, dtype=torch.bool, device=self.device)
            for a in range(K):
                for b in range(a + 1, K):
                    bad[:, b] |= (cand[:, a] == cand[:, b]).all(dim=1)
            if not held_out:
                bad |= self.is_held_out(cand.reshape(-1, 3)).view(n, K)
            if not bool(bad.any()):
                break
            r, k = bad.nonzero(as_tuple=True)
            cand[r, k] = self._draw(r.shape[0], held_out=held_out)
        return cand

    def _cluster(self, anchor: torch.Tensor, K: int) -> torch.Tensor:
        """(n, K, 3): the anchor, then K-1 of its one-field near misses.

        Built by enumerating every combination one field away from the anchor and
        drawing K-1 distinct ones that are not reserved, rather than by nudging
        at random until it works: a hard round that quietly failed to be a
        cluster would make the rung easier than it looks.
        """
        n = anchor.shape[0]
        rows = torch.arange(n, device=self.device)
        # every one-field neighbour of the anchor: (n, M, 3)
        cols = []
        for field, span in enumerate(self.spans):
            for step in range(1, span):
                cand = anchor.clone()
                cand[:, field] = (anchor[:, field] + step) % span
                cols.append(cand)
        nb = torch.stack(cols, dim=1)
        M = nb.shape[1]
        # reserved neighbours are not available: score them to the back
        score = torch.rand(n, M, device=self.device, generator=self.gen)
        score = score + self.is_held_out(nb.reshape(-1, 3)).view(n, M).float()
        pick = torch.argsort(score, dim=1)[:, :K - 1]
        members = [anchor.unsqueeze(1),
                   nb.gather(1, pick.unsqueeze(-1).expand(-1, -1, 3))]
        return torch.cat(members, dim=1)

    def sample(self, n: int, informer: int = FARMER, held_out: bool = False,
               hard_frac: Optional[float] = None, mix: "tuple | None" = None,
               query: "int | None" = None, mixed_query: bool = False
               ) -> ReferentialBatch:
        """One batch of lineup rounds, mixed as the rung asks.

        ``mix`` weights the four kinds of round -- fruit, colour, quality, all
        fields -- and is how a rung adds a field without dropping the ones
        below it. ``query`` forces a single kind (used by the probes, which ask
        about one field at a time).
        """
        if query is not None:
            mix = tuple(1.0 if i == int(query) else 0.0 for i in range(4))
        elif mixed_query and mix is None:
            mix = (1 / 3, 1 / 3, 1 / 3, 0.0)
        elif mix is None:
            mix = (0.0, 0.0, 0.0, 1.0)
        total = float(sum(mix)) or 1.0
        counts = [int(round(n * w / total)) for w in mix]
        counts[max(range(4), key=lambda i: counts[i])] += n - sum(counts)

        K = min([self._n_candidates(f) for f in range(3)] + [self._n_candidates(None)])
        cand = torch.zeros((n, K, 3), dtype=torch.long, device=self.device)
        target = torch.zeros(n, dtype=torch.long, device=self.device)
        q = torch.zeros(n, dtype=torch.long, device=self.device)
        at = 0
        for kind, k in enumerate(counts):
            if k <= 0:
                continue
            sl = slice(at, at + k)
            if kind == ASK_ALL:
                p = (self.cfg.curriculum.hard_distractor_frac if hard_frac is None
                     else hard_frac)
                c, t = self._open_round(k, held_out, 0.0 if held_out else p)
            else:
                c, t = self._query_round(k, kind, held_out)
            cand[sl] = c[:, :K]
            target[sl] = t
            q[sl] = kind
            at += k
        # Shuffle so a round's kind is not its position in the batch: the agents
        # are paired by index, and a sorted batch would hand each agent one kind.
        order = torch.randperm(n, device=self.device, generator=self.gen)
        cand, target, q = cand[order], target[order], q[order]
        rows = torch.arange(n, device=self.device)
        truth = cand[rows, target]
        return ReferentialBatch(meanings=cand, target=target, query=q,
                                held_out=self.is_held_out(truth), day=0,
                                informer=informer)


def resolve_referential(cfg: Config, rb: ReferentialBatch, choice: torch.Tensor,
                        f_cost: torch.Tensor,
                        b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """Score a batch of lineup rounds.

    Both roles are paid for the same thing -- did the guess land -- because in a
    lineup game being understood and understanding are the same event. The symbol
    cost still applies, so brevity is still worth something.
    """
    R = cfg.reward
    correct = choice == rb.target
    rows = torch.arange(choice.shape[0], device=choice.device)
    # How much of the thing the guess actually got: the fields the chosen
    # candidate shares with the target. In a round that turns on one field this
    # is a constant and does nothing; in a round that turns on all three it is
    # the difference between a near miss and a wild one.
    shared = (rb.meanings[rows, choice] == rb.meanings[rows, rb.target]).float()
    partial = shared.mean(dim=1)
    reward = (R.refer_success * correct.float() + R.refer_miss * (~correct).float()
              + R.refer_partial * partial)
    # Each role pays for the symbols *it* emitted. Callers count those from the
    # phase's own speaker schedule; the old fixed buyer-opens schedule billed the
    # lineup's describer nothing, which is how 38% of utterances ended up at the
    # length cap.
    f = reward - f_cost.float()
    b = reward - b_cost.float()
    zero_l = torch.zeros_like(choice)
    zero_f = torch.zeros_like(f)
    # The trade-shaped fields are reported as zero rather than omitted, so every
    # consumer of a result dict -- the population tally, the economy, the ledger --
    # works unchanged in a phase where nothing is bought or sold.
    return {
        "farmer_reward": f, "buyer_reward": b,
        "success": correct, "comprehended": correct,
        "both_judged": correct,
        "farmer_decode": partial, "buyer_decode": partial,
        "choice": choice, "target": rb.target,
        "both_accept": correct,
        "agree_variety": correct, "agree_qty": correct, "agree_price": correct,
        "agreed_variety": zero_l, "agreed_qty": zero_l, "agreed_price": zero_l,
        "traded_qty": zero_l, "trade_value": zero_f,
        "farmer_profit": zero_f, "buyer_savings": zero_f,
        "correct_no_deal": torch.zeros_like(correct),
        "missed_deal": torch.zeros_like(correct),
        "one_sided": torch.zeros_like(correct),
        "bad_deal": torch.zeros_like(correct),
    }


# ==========================================================================
# the mutual game
# ==========================================================================
@dataclass
class MutualBatch:
    """Both parties hold one private (variety, quantity, quality) meaning."""
    f_meaning: torch.Tensor       # (B, 3)
    b_meaning: torch.Tensor       # (B, 3)
    day: int = 0

    def __len__(self) -> int:
        return int(self.f_meaning.shape[0])

    @property
    def device(self) -> torch.device:
        return self.f_meaning.device

    def meaning_of(self, role: int) -> torch.Tensor:
        return self.f_meaning if role == FARMER else self.b_meaning

    @property
    def true_meaning(self) -> torch.Tensor:
        return self.f_meaning

    def obs(self, cfg: Config, role: int) -> torch.Tensor:
        from .world import n_obs_slots
        n = n_obs_slots(cfg.world, cfg)
        x = self.meaning_of(role)
        pad = torch.zeros((len(self), max(0, n - x.shape[1])), dtype=torch.long,
                          device=x.device)
        return torch.cat([x, pad], dim=1)[:, :n]


def report_fields(cfg: Config, report: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(B, 3) bool -- was each of (fruit, colour, quality) reported correctly."""
    return report == truth


def resolve_mutual(cfg: Config, mb: MutualBatch, f_dec: torch.Tensor,
                   b_dec: torch.Tensor, f_cost: torch.Tensor,
                   b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """Score a batch of mutual rounds.

    ``f_dec`` is the farmer's whole decision row; its report of the buyer's
    thing is read out of ``H_REPORT`` here, and nowhere else. Every caller used
    to slice the columns itself, and they disagreed: the reward read
    ``H_BELIEF[:3]``, whose middle head is a belief about *quantity*, and scored
    it against the colour -- a field the rung trains on a different head
    (``H_BELIEF_COLOR``) and which therefore could only ever be right by luck.

    Each side is paid for reading the other (``decode``) and for being read
    (``understood``) field by field, which is what gives each message a
    gradient, plus the full round bonus only when both reports are right at once.
    """
    R = cfg.reward
    rep = list(H_REPORT)
    f_report, b_report = f_dec[:, rep], b_dec[:, rep]
    f_fields = report_fields(cfg, f_report, mb.b_meaning)
    b_fields = report_fields(cfg, b_report, mb.f_meaning)
    f_ok, b_ok = f_fields.all(dim=1), b_fields.all(dim=1)
    both = f_ok & b_ok
    f_frac, b_frac = f_fields.float().mean(1), b_fields.float().mean(1)
    joint = R.refer_success * both.float() + R.refer_miss * (~both).float()
    f = joint + R.decode * f_frac + R.understood * b_frac - f_cost.float()
    b = joint + R.decode * b_frac + R.understood * f_frac - b_cost.float()
    zero_l = torch.zeros_like(f_cost, dtype=torch.long)
    zero_f = torch.zeros_like(f)
    return {
        "farmer_reward": f, "buyer_reward": b,
        "success": both, "comprehended": both, "both_judged": both,
        "farmer_decode": f_frac, "buyer_decode": b_frac,
        "farmer_report_ok": f_ok, "buyer_report_ok": b_ok,
        "farmer_fields": f_fields, "buyer_fields": b_fields,
        "both_accept": both,
        "agree_variety": f_fields[:, 0] & b_fields[:, 0],
        "agree_qty": f_fields[:, 1] & b_fields[:, 1],
        "agree_price": both,
        "agreed_variety": zero_l, "agreed_qty": zero_l, "agreed_price": zero_l,
        "traded_qty": zero_l, "trade_value": zero_f,
        "farmer_profit": zero_f, "buyer_savings": zero_f,
        "correct_no_deal": torch.zeros_like(both), "missed_deal": torch.zeros_like(both),
        "one_sided": torch.zeros_like(both), "bad_deal": torch.zeros_like(both),
    }


# ==========================================================================
# hindsight: what each scored head should have said, once the round is over
# ==========================================================================
def hindsight_targets(cfg: Config, phase: Phase, scen) -> dict[int, dict[int, torch.Tensor]]:
    """{role: {head: (B,) correct class}} for the heads this rung scores.

    After a round both parties learn how it came out: which candidate was meant,
    what the other's meaning was, what the buyer actually wanted, what the farmer
    actually had. That is feedback about the *outcome*, never about which words
    to use -- the form of the language stays entirely the agents' own.

    Why it is needed: with only a sampled right/wrong reward, the listener is
    never told the answer, and the speaker's only gradient runs through the
    listener's current reading of the message. Once every slot is read as
    variety, nothing points towards spending a slot on quantity. Measured: after
    three rungs the messages carried 1.5 bits of variety and 0.01-0.05 bits of
    quantity or quality, with repetition like ``a13-a13-a13-a13``. A supervised
    target on the listener's head flows back through the straight-through
    channel into the speaker for every field the listener has to recover.
    """
    out: dict[int, dict[int, torch.Tensor]] = {FARMER: {}, BUYER: {}}
    w = cfg.world
    if phase.referential:
        out[phase.guesser][H_CHOICE] = scen.target
    elif phase.mutual:
        for role, other in ((FARMER, scen.b_meaning), (BUYER, scen.f_meaning)):
            for i, h in enumerate(H_REPORT):
                out[role][h] = other[:, i]
    elif phase.order and hasattr(scen, "want_variety"):
        # Exactly the fields this rung asks for, in the direction it asks them.
        for name, head in phase.ask_heads.items():
            out[phase.reporter][head] = request_truth(cfg, scen, name)
    elif hasattr(scen, "want_variety"):
        # A trading rung: both sides state the deal, so both are told what it was.
        for role in (FARMER, BUYER):
            out[role][H_VARIETY] = scen.want_variety
            out[role][H_QTY] = scen.need_qty
            out[role][H_ACCEPT] = scen.viable.long()
        out[FARMER][H_BELIEF_COLOR] = scen.want_color
        if cfg.reward.belief_heads:
            b = H_BELIEF
            out[FARMER][b[0]] = scen.want_variety
            out[FARMER][b[1]] = scen.need_qty
            out[FARMER][b[2]] = scen.min_quality
            out[FARMER][b[3]] = scen.max_price
            out[BUYER][b[1]] = scen.offered_stock.clamp(0, w.max_qty)
            out[BUYER][b[2]] = scen.offered_quality
            out[BUYER][b[3]] = scen.reservation
    # only heads the rung actually scores for that role
    return {r: {h: t for h, t in d.items() if h in phase.active_heads(r, cfg)}
            for r, d in out.items()}


# ==========================================================================
# the order rung
# ==========================================================================
def request_truth(cfg: Config, sb, name: str) -> torch.Tensor:
    """The true value of one request field, from whichever side holds it."""
    if name == "fruit":
        return sb.want_variety
    if name == "colour":
        return sb.want_color
    if name == "quantity":
        return sb.need_qty
    if name == "price":                       # the most the buyer will pay
        return sb.max_price
    if name == "stock":                       # of the lot that was asked about
        return sb.offered_stock.clamp(0, cfg.world.max_qty)
    if name == "quality":
        return sb.offered_quality
    if name == "reservation":                 # the least the farmer will take
        return sb.reservation
    if name == "deal":                        # is this one worth doing at all
        return sb.viable.long()
    raise KeyError(name)


def resolve_request(cfg: Config, phase, sb, dec: dict, f_cost: torch.Tensor,
                    b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """Did what one side holds privately arrive intact at the other?

    Every request rung is the same event -- one side says facts only it has, the
    other has to put them in its decision heads -- and they differ only in which
    fields are asked for and which way round. Both parties are paid, because
    being understood and understanding are one event here, plus partial credit
    per field so a farmer that gets the fruit right and the number wrong is told
    which half worked.
    """
    R = cfg.reward
    answer = dec[phase.reporter]
    oks = [answer[:, head] == request_truth(cfg, sb, name)
           for name, head in phase.ask_heads.items()]
    fields = torch.stack(oks, dim=1)
    both = fields.all(dim=1)
    # the field this rung introduced, reported separately so promotion can be
    # judged on it rather than on a conjunction that hides it
    by_name = dict(zip(phase.ask_heads, oks))
    first = by_name.get(phase.asks_first, oks[0])
    var_ok = by_name.get("fruit", both)
    qty_ok = by_name.get("quantity", by_name.get("stock", both))
    frac = fields.float().mean(1)
    base = (R.refer_success * both.float() + R.refer_miss * (~both).float()
            + R.decode * frac)
    f = base - f_cost.float()
    b = base - b_cost.float()
    zero_l = torch.zeros_like(f_cost, dtype=torch.long)
    zero_f = torch.zeros_like(f)
    no = torch.zeros_like(both)
    return {
        "farmer_reward": f, "buyer_reward": b,
        "success": both, "comprehended": both, "both_judged": both,
        "farmer_decode": frac if phase.reporter == FARMER else zero_f,
        "buyer_decode": frac if phase.reporter == BUYER else zero_f,
        "order_fields": fields, "order_first": first,
        "both_accept": both, "agree_variety": var_ok, "agree_qty": qty_ok,
        "agree_price": both,
        # Only what this rung actually asked for; an unasked head is an untrained
        # sample, and a log is worse for carrying one than for carrying nothing.
        "agreed_variety": (answer[:, H_VARIETY] if "fruit" in phase.ask else zero_l),
        "agreed_qty": (answer[:, H_QTY] if "quantity" in phase.ask else zero_l),
        "agreed_price": zero_l,
        "traded_qty": zero_l, "trade_value": zero_f,
        "farmer_profit": zero_f, "buyer_savings": zero_f,
        "correct_no_deal": no, "missed_deal": no, "one_sided": no, "bad_deal": no,
    }


# ==========================================================================
@dataclass
class CurriculumState:
    """Where the run is on the ladder, and how it got there."""
    phases: list[Phase]
    index: int = 0
    episodes_in_phase: int = 0
    updates_in_phase: int = 0             # what the rung budgets count
    transitions: list[dict[str, Any]] = field(default_factory=list)
    stalled: bool = False
    last_report: dict[str, Any] = field(default_factory=dict)
    stop_report: dict[str, Any] = field(default_factory=dict)
    checks_run: int = 0

    @property
    def phase(self) -> Phase:
        return self.phases[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.phases) - 1

    def advance(self, episode: int, checks: dict[str, Any]) -> Phase:
        old = self.phase
        self.index = min(self.index + 1, len(self.phases) - 1)
        self.episodes_in_phase = 0
        self.updates_in_phase = 0
        self.stalled = False
        self.transitions.append({
            "episode": episode, "from": old.name, "to": self.phase.name,
            "episodes_in_previous_phase": None, "updates_in_previous_phase": None,
            "criteria": checks,
        })
        return self.phase
