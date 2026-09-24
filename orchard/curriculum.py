"""A curriculum: learn to name every field before learning to trade.

Why this exists
---------------
Dropped straight into the full trading task, agents have to solve five things at
once before any of them pays off even once: emit a stable non-arbitrary signal,
put true private information into it, have the other side decode it, close the
loop so that decoding changes a decision, and get the trade arithmetic (budget,
viability, quantity) right as well. Measured: success 0.000 at every checkpoint
of such a run, comprehension 0.000 throughout, and scrambling the channel
costing nothing, because there was nothing to scramble.

So the task is built up in rungs, and a rung is only left behind once it has
demonstrably worked. One rule shapes the whole ladder: **every word is invented
in a naming rung, and every later rung only reuses words.** The ladder that
preceded this one named three fields and left quantity and price to be invented
in the trading rungs -- where hindsight feedback, the speaker costs and the
convention bonus were already on, which are exactly the conditions the naming
rungs show stop a code from forming. Quantity never arrived.

A thing to talk about is a **lot**: (fruit, colour, quality, quantity, price).
A buyer's request is a lot, a farmer's barn is a list of lots, and the naming
game describes lots -- one observation layout everywhere, so the words a
population invents in the naming game are, slot for slot, the words it places
an order with.

  naming    ``name-fruit``, ``name-color``, ``name-quality``, ``name-quantity``,
            ``name-price``: a lineup game, one field at a time. The describer
            sees one lot and which field is asked about; the guesser sees three
            candidates that differ in that field and picks. Each rung *adds* a
            kind of round and keeps rehearsing the ones below it. Both seats are
            filled from one pool, and the describer alternates, so every agent
            does both jobs and there is one language.
            ``name-all``: the candidates differ in any field, so the whole lot has
            to be named at once. Judged on combinations never trained on.
  mutual    Both hold a private lot and each must report the other's, field by
            field. The community arrives here.
  order     The buyer states its request (a lot, in the trading layout) and the
            farmer -- looking at its barn -- must report it. Every word is
            inherited; what is new is the farmer listening with a barn in view.
  offer     The other direction too: the farmer answers with what it holds of
            the lot that was asked for -- how many, what quality, at what floor
            -- and the buyer must report that. The first rung where a farmer
            has to find a lot in its barn by the words it heard.
  judge     The same dialogue, and now both decide whether the deal is any good.
  haggle    Budget and floor make a deal refusable; it only counts if both name
            the same one and it is executable. The pool splits into farmers and
            buyers here.
  bargain   Several turns, so counter-offers are possible.
  market    The full economy: persistent stock, restocking, viability.

Each rung has a minimum and a maximum budget in training updates: a rung that
meets its criteria is left promptly, and one that runs past its maximum without
meeting them stops the run with a report instead of burning the rest of the
budget on a rung that is not converging.

Who speaks when is a property of the rung, not of the code base. Everything that
needs to know which dialogue slots an agent produced -- the symbol cost, the
bottleneck's targets, the speaker embedding, the probes that extract per-meaning
forms -- asks the phase (:meth:`Phase.own_positions`), because the fixed
buyer-opens schedule of the trading task is wrong for every lineup rung.

Weights carry across phase boundaries -- the population that learned to name is
the population that learns to haggle. Nothing is reinitialised at a transition.
Every phase shares one sequence layout, one channel and one set of heads, so
"carry the weights forward" is literally the same modules continuing to train.
Phases that use fewer turns simply leave the later dialogue slots empty.

Promotion is on evidence, not on a schedule: see :class:`Promotion` and
:func:`evaluate_rung`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

import torch

from .config import Config
from .env import BUYER, FARMER
from .world import (K_COLOR, K_EMPTY, K_FIELD, K_PRICE, K_QTY, K_QUALITY, K_VARIETY,
                    LOT_FIELDS, LOT_KINDS, MEANING_FIELDS, N_LOT_FIELDS, QUERY_ALL,
                    lot_spans)

# Head layout shared by every phase.  Sampling all of them keeps tensor shapes
# constant across a transition; each phase says which ones actually count.
H_ACCEPT, H_VARIETY, H_QTY, H_PRICE = 0, 1, 2, 3
H_BELIEF = (4, 5, 6, 7)          # the other party's (fruit, quantity, quality, price)
H_CHOICE = 8                     # which candidate in a lineup
H_BELIEF_COLOR = 9               # the other party's colour -- appended, so 0..8 kept
N_HEADS = 10
# Reporting a lot: (fruit, colour, quality, quantity, price) -- the five fields a
# lot has, in the one order they are held everywhere, each on its belief head.
H_REPORT = (H_BELIEF[0], H_BELIEF_COLOR, H_BELIEF[2], H_BELIEF[1], H_BELIEF[3])

# Every field a report rung can ask for, and the head it has to arrive in.
# The first five are the buyer's request (reported by the farmer); the next
# three are the farmer's lot (reported by the buyer); "deal" is whether the
# round is worth doing at all (either side).
REQUEST_FIELDS = LOT_FIELDS                          # fruit, colour, quality, quantity, price
LOT_ANSWER_FIELDS = ("stock", "lot-quality", "reservation")
DEAL_FIELD = "deal"
FIELD_HEADS = {
    "fruit": H_BELIEF[0], "colour": H_BELIEF_COLOR, "quality": H_BELIEF[2],
    "quantity": H_BELIEF[1], "price": H_BELIEF[3],
    "stock": H_BELIEF[1], "lot-quality": H_BELIEF[2], "reservation": H_BELIEF[3],
    "deal": H_ACCEPT,
}
# Fields that are a (fruit, colour, quality) combination's parts: the ones the
# held-out gate can be measured on.
COMBO_FIELDS = MEANING_FIELDS

# The kinds of round a naming rung is made of: one per lot field, then "all".
ASK_ALL = QUERY_ALL
N_KINDS = N_LOT_FIELDS + 1
ROUND_NAMES = LOT_FIELDS + ("whole lots",)


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
    # (fruit, colour, quality, quantity, price, all-fields). A naming rung *adds*
    # a kind and keeps rehearsing the ones below it: switching outright cost the
    # run its fruit code and its gradient at once -- the messages still carried
    # fruit (coverage 0.40) and nothing else, at chance, for 500 updates.
    mix: tuple = (0.0,) * N_LOT_FIELDS + (1.0,)
    # The kind this rung introduces: 0..4 one field, ASK_ALL the whole lot.
    # Promotion is judged on *this* kind, so acing the rehearsal cannot carry a
    # rung whose new job is not being done.
    primary: int = ASK_ALL
    # A report rung (``KIND_ORDER``): the fields each role has to report, indexed
    # by role (farmer, buyer), and which of them this rung introduces. Like the
    # naming rungs, a report rung *adds* fields and keeps the earlier ones in
    # play, and it is judged on the ones it added, field by field -- a
    # conjunction would hide which one is at chance, which is exactly how
    # `haggle` failed before (quality 0.88, variety 0.52, quantity 0.20 -- and
    # one number, 0.07).
    reports: tuple = ((), ())
    new: tuple = ((), ())

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
        """A report rung played in the trading world (order, offer, judge)."""
        return self.kind == KIND_ORDER

    @property
    def reporting(self) -> bool:
        """A rung scored on what one side reports of the other's private lot."""
        return self.mutual or self.order

    @property
    def tuples(self) -> bool:
        """Played over bare lots, not farms and requests."""
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
    def invents(self) -> bool:
        """Does this rung still have a *word* to invent, rather than reuse?

        The lineup rungs each invent their field's words, and `name-all` has to
        find the five-word utterance where one used to do. Every rung from
        `mutual` up recombines what exists: a lot is described with the words
        the naming rungs built, whether it is held, asked for or offered. This
        is what decides whether the speaker pays for length and for novelty
        (:func:`costs_apply`), because a price on a word is a pressure on a
        word that exists: charged while one is still being invented, the
        cheapest way to be brief is to say the same short nothing.
        """
        return self.referential

    @property
    def whole(self) -> bool:
        """Is this rung's own job to name a whole lot?"""
        return self.naming and self.primary >= ASK_ALL

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

    # ---- what each role has to report -----------------------------------
    def report_names(self, role: int) -> tuple:
        """The fields ``role`` has to report about the other party in this rung."""
        if self.mutual:
            return LOT_FIELDS
        if self.order:
            return tuple(self.reports[role])
        return ()

    def new_names(self, role: int) -> tuple:
        """The reported fields this rung introduced for ``role``."""
        if self.mutual:
            return LOT_FIELDS
        if self.order:
            return tuple(self.new[role])
        return ()

    def report_heads(self, role: int) -> dict:
        """field name -> the head it has to arrive in, for ``role``."""
        if self.mutual:
            return dict(zip(LOT_FIELDS, H_REPORT))
        return {n: FIELD_HEADS[n] for n in self.report_names(role)}

    def reporting_roles(self) -> list[int]:
        return [r for r in (FARMER, BUYER) if self.report_names(r)]

    def lot_speakers(self, cfg: Config) -> list[int]:
        """Roles whose utterances are about a lot they can see (probe-able).

        In the naming rungs everyone describes a lot; in the trading world only
        the buyer's observation *is* a lot -- the farmer looks at a barn, and its
        words are about whichever lot was asked for, which no probe of its own
        observation can hold fixed.
        """
        # Over every view: in a swap rung the buyer describes in the second
        # view and would otherwise be taken for a barn speaker and never
        # probed, which left `name-all` with an n/a structure bar it could not
        # clear.
        roles = [r for r in (FARMER, BUYER)
                 if any(v.speaks(cfg, r) for v in self.views())]
        if self.tuples:
            return roles
        return [r for r in roles if r == BUYER]

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
        if self.reporting:
            # Report the other party's lot, or the fields this rung asks for.
            return list(self.report_heads(role).values())
        heads = [H_ACCEPT, H_VARIETY, H_QTY]
        if self.use_price:
            heads.append(H_PRICE)
        if cfg.reward.belief_heads:
            heads.extend(H_BELIEF)
        return heads

    def meaning_kind(self, role: int) -> str:
        """What a speaker's utterance is about, for pooling conventions.

        Every naming rung describes a lot, whichever role is talking, and a
        buyer's request in the market is a lot too, so all of those feed and are
        measured against one convention per lot. A farmer's words in the market
        are about whichever lot was asked for, so they are keyed on the barn.
        """
        if self.tuples or role == BUYER:
            return "tuple"
        return "barn"


def ladder(cfg: Config) -> list[Phase]:
    """The rungs, easiest first.  ``n_turns`` never exceeds ``channel.n_turns``.

    The first six teach naming, one field at a time and then together, before
    anything is traded.  Each is a lineup: the describer sees one lot, the
    guesser sees the candidates and picks.  Everyone takes both seats, because
    the describer alternates batch by batch and (below ``split_roles_at``) both
    seats are filled from one pool of agents speaking one language.
    """
    full = max(2, cfg.channel.n_turns)
    one = lambda i: tuple(1.0 if k == i else 0.0 for k in range(N_KINDS))
    request = tuple(REQUEST_FIELDS)
    lot = tuple(LOT_ANSWER_FIELDS)
    return [
        Phase("name-fruit", 0, 1, True, False, False,
              "lineup game over fruit alone: the candidates share every other "
              "field, so only the fruit needs saying",
              kind=KIND_SWAP, mix=one(0), primary=0),
        Phase("name-color", 1, 1, True, False, False,
              "colour rounds added to fruit ones: a word for a colour and nothing "
              "else, while the fruit words stay in use and stay needed",
              kind=KIND_SWAP, mix=(0.4, 0.6, 0.0, 0.0, 0.0, 0.0), primary=1),
        Phase("name-quality", 2, 1, True, False, False,
              "quality rounds added: every round still asks about one field, but "
              "which field changes, so a word has to mean the same thing wherever "
              "it appears", kind=KIND_SWAP, mix=(0.25, 0.25, 0.5, 0.0, 0.0, 0.0),
              primary=2),
        Phase("name-quantity", 3, 1, True, False, False,
              "quantity rounds added: a word for each number, 0 to the most a barn "
              "holds, invented here so that no trading rung ever has to",
              kind=KIND_SWAP, mix=(0.15, 0.15, 0.2, 0.5, 0.0, 0.0), primary=3),
        Phase("name-price", 4, 1, True, False, False,
              "price rounds added: a word for each price bin, so every field a "
              "deal turns on has a word before any deal is attempted",
              kind=KIND_SWAP, mix=(0.12, 0.12, 0.13, 0.13, 0.5, 0.0), primary=4),
        Phase("name-all", 5, 1, True, False, False,
              "rounds where the candidates differ in any field, mostly one-field "
              "near misses, so the whole lot has to be named at once -- with "
              "single-field rounds still mixed in",
              kind=KIND_SWAP, mix=(0.06, 0.06, 0.06, 0.06, 0.06, 0.7), primary=ASK_ALL),
        Phase("mutual", 6, 2, True, False, False,
              "both hold a private lot and each must report the other's, field by "
              "field; the community arrives; no price to agree, no accept/reject",
              kind=KIND_MUTUAL),
        Phase("order", 7, 1, False, False, False,
              "the buyer states its request -- the same five fields, the same "
              "layout as a lot -- and the farmer, looking at its barn, has to "
              "report it. Every word is inherited; the farmer's job is to listen "
              "with a barn in view", kind=KIND_ORDER,
              reports=(request, ()), new=(request, ())),
        Phase("offer", 8, 2, False, False, False,
              "the other direction too: the farmer answers with what it holds of "
              "the lot that was asked for -- how many, what quality, at what floor "
              "-- and the buyer has to report that. The first rung where a farmer "
              "finds a lot in its barn by the words it heard", kind=KIND_ORDER,
              reports=(request, lot), new=((), lot)),
        Phase("judge", 9, 2, False, False, False,
              "the same dialogue, and now both decide: is this deal any good? Each "
              "weighs what it was told against what it holds, with nothing yet "
              "riding on the answer", kind=KIND_ORDER,
              reports=(request + (DEAL_FIELD,), lot + (DEAL_FIELD,)),
              new=((DEAL_FIELD,), (DEAL_FIELD,))),
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
    """Does the speaker pay for what it says in this rung?

    Two conditions, because the rule is per rung and a threshold only happens
    to say it. Not before ``reward.costs_from_rung`` -- a floor, so a run can
    hold them off entirely -- and not on a rung that still has a word to invent
    (:attr:`Phase.invents`). With every field named below `mutual` the two
    agree; the earlier ladder had trading rungs that still named quantity and
    price, and no threshold could spare those without sparing `mutual`.
    """
    if phase.index < phase_named(cfg, cfg.reward.costs_from_rung).index:
        return False
    return not phase.invents


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
        K = min(K, lot_spans(cfg.world)[int(kind)])
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


def _describe(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def measures_holdout(phase: Phase) -> bool:
    """Does this rung's gate ask about combinations nobody ever trained on?

    The lineup rung that names a whole lot, and every report rung whose reports
    include the fields a reserved combination is made of (`mutual`, `order`,
    `offer`, `judge`). A rung that varies one field is not asked: it has not
    been taught the rest.
    """
    if phase.referential:
        return bool(phase.whole)
    return any(n in COMBO_FIELDS for r in (FARMER, BUYER) for n in phase.report_names(r))


def evaluate_report_rung(cfg: Config, phase: Phase, ev: dict[str, Any],
                         updates_in_phase: int, rule: "Promotion"
                         ) -> tuple[bool, dict[str, Any]]:
    """Has what one side holds arrived at the other, field by field?

    Judged per role and per field, never pooled: every reported field has to
    clear the channel bar on its own (labelled "still carries" if an earlier rung
    introduced it, so forgetting is visible as forgetting), the fields this rung
    introduced have to arrive together, above an absolute floor *and* a real
    gain over silence -- "twice the muted rate" is not a reachable bar for a
    binary decision silence already gets 68% of the time -- and, where the
    report includes a (fruit, colour, quality) combination, the combinations
    never trained on have to be reported nearly as well as the trained ones.
    """
    c = cfg.curriculum
    k = c.min_success_over_chance
    checks: dict[str, tuple[bool, str]] = {}
    checks["long enough in rung"] = (
        updates_in_phase >= rule.min_updates,
        "%d of %d updates" % (updates_in_phase, rule.min_updates))

    # Speakers whose meaning is a lot are held to structure.
    for role, label in ((FARMER, "farmer"), (BUYER, "buyer")):
        if role not in phase.lot_speakers(cfg):
            continue
        d = ev.get("speakers", {}).get(label, {})
        ts, nl = _num(d.get("topsim")), _num(d.get("null"))
        checks["%s describes: topsim clear of null" % label] = (
            ts == ts and nl == nl and ts - nl >= c.min_topsim_over_null,
            "%s vs null %s, need +%.2f" % (_fmt(ts), _fmt(nl), c.min_topsim_over_null))
        if phase.mutual:
            pos = _num(d.get("positional"))
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

    for role, label in ((FARMER, "farmer"), (BUYER, "buyer")):
        names = phase.report_names(role)
        if not names:
            continue
        new = set(phase.new_names(role))
        per = ev.get("%s_field_transfer" % label) or []
        intact = ev.get("%s_fields_intact" % label) or []
        for i, name in enumerate(names):
            t = _num(per[i]) if i < len(per) else float("nan")
            a = _num(intact[i]) if i < len(intact) else float("nan")
            what = ("%s carries" % name) if name in new else ("still carries %s" % name)
            checks["%s decodes: %s" % (label, what)] = (
                t == t and t >= c.min_field_transfer,
                "%s right, %s of the headroom over a muted channel, need %.2f"
                % (_fmt(a), _fmt(t), c.min_field_transfer))
        if new:
            got = _num(ev.get("%s_new" % label))
            muted = _num(ev.get("muted_%s_new" % label))
            floor = rule.min_success if len(new) > 1 else 0.0
            floor = max(floor, _headroom_floor(muted, c.min_field_transfer))
            what = "the other's lot" if phase.mutual else _describe(
                [n for n in names if n in new])
            checks["%s decodes: %s arrives" % (label, what)] = (
                got == got and got >= floor,
                "%s exact, need %.2f (silence alone scores %s)"
                % (_fmt(got), floor, _fmt(muted)))

    if phase.mutual:
        succ, chance = _num(ev.get("success")), _num(ev.get("chance"))
        checks["both decode in the same round"] = (
            succ == succ and succ >= rule.min_success
            and (chance != chance or succ >= k * chance),
            "%s, need %.3f and %.1fx the muted rate %s"
            % (_fmt(succ), rule.min_success, k, _fmt(chance)))

    tr = _num(ev.get("transfer"))
    checks["channel carries"] = (
        tr == tr and tr >= c.min_channel_transfer,
        "%s of the headroom, need %.2f" % (_fmt(tr), c.min_channel_transfer))

    # The productivity gate, per field: a combination nobody ever trained on has
    # to be reported nearly as well as the trained ones. Measured on the three
    # fields a combination is made of; quantity and price are never held out.
    # Each field's ratio is taken over the headroom above a message-blind
    # guesser and the ratios are averaged -- never a ratio of means, which
    # weights each field by its headroom and lets the one the language learned
    # best carry two it did not -- and each is named, so the log says which.
    if measures_holdout(phase):
        ratio = _num(ev.get("holdout_field_ratio"))
        hs = _num(ev.get("holdout_fields", ev.get("holdout_field_success")))
        seen = _num(ev.get("seen_fields", ev.get("seen_field_success")))
        each = ev.get("holdout_field_ratios") or []
        names = tuple(ev.get("holdout_field_names") or COMBO_FIELDS)
        per = (" [" + ", ".join("%s %s" % (n, _fmt(_num(r))) for n, r in zip(names, each))
               + "]") if each else ""
        checks["describes combinations it never trained on"] = (
            ratio == ratio and ratio >= c.min_holdout_ratio,
            "held-out %s vs seen %s per field = %s of the headroom%s, need %.2f "
            "(the whole round: %s vs %s)"
            % (_fmt(hs), _fmt(seen), _fmt(ratio), per, c.min_holdout_ratio,
               _fmt(_num(ev.get("holdout_success"))), _fmt(_num(ev.get("seen_success")))))

    passed = all(v[0] for v in checks.values())
    return passed, {n: {"met": v[0], "detail": v[1]} for n, v in checks.items()}


def evaluate_rung(cfg: Config, phase: Phase, ev: dict[str, Any],
                  updates_in_phase: int) -> tuple[bool, dict[str, Any]]:
    """Has this rung demonstrably worked?  Returns (passed, named checks).

    ``ev`` is :func:`orchard.metrics.phase_evidence`'s output. The trading rungs
    are judged on pooled numbers. Every rung that exists to make communication
    two-way is judged per role, and every role has to clear every bar on its
    own: a pooled average would let a fluent farmer carry a buyer that never
    learned to speak, which is the precise failure an earlier run showed (farmer
    positional structure 0.03 against the buyer's 0.39).
    """
    c = cfg.curriculum
    rule = promotion_for(cfg, phase)
    if phase.reporting:
        return evaluate_report_rung(cfg, phase, ev, updates_in_phase, rule)
    if not phase.swaps:
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
    # Structure is only a fair bar where a round turns on every field at once.
    # A rung that varies one field has nothing to be compositional *about*:
    # what it has to show is that the word works, and keeps working on
    # combinations never trained on (the held-out check below).
    whole_thing = phase.whole
    for role, label in ((FARMER, "farmer"), (BUYER, "buyer")):
        d = ev.get("speakers", {}).get(label, {})
        ts, nl, pos = _num(d.get("topsim")), _num(d.get("null")), _num(d.get("positional"))
        if whole_thing:
            checks["%s describes: topsim clear of null" % label] = (
                ts == ts and nl == nl and ts - nl >= c.min_topsim_over_null,
                "%s vs null %s, need +%.2f" % (_fmt(ts), _fmt(nl), c.min_topsim_over_null))
            checks["%s describes: positional structure" % label] = (
                pos == pos and pos >= c.min_positional_structure,
                "%s, need %.2f" % (_fmt(pos), c.min_positional_structure))
            cov = _num(d.get("field_coverage"))
            checks["%s describes: covers every field" % label] = (
                cov == cov and cov >= c.min_field_coverage,
                "%s of each field's information on average, need %.2f"
                % (_fmt(cov), c.min_field_coverage))
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
        floor = _headroom_floor(ch, c.min_channel_transfer)
        checks["still names %s" % ROUND_NAMES[kind]] = (
            s == s and s >= floor,
            "%s on %s rounds, need %.2f -- %.2f of the headroom over chance %.3f"
            % (_fmt(s), ROUND_NAMES[kind], floor, c.min_channel_transfer, ch))
    if whole_thing:
        # The productivity gate, on the rungs that describe a whole lot. Every
        # candidate in a held-out round is a combination nobody ever trained on,
        # so a code that names whole things has nothing to say about any of them,
        # however well it scores on the ones it drilled; one with reusable parts
        # describes them as easily as anything else. A rung that varies a single
        # field is not asked for this: it has not been taught the rest.
        hs, seen = _num(ev.get("holdout_success")), _num(ev.get("seen_success"))
        ratio = _num(ev.get("holdout_ratio"))
        chance = _num(ev.get("chance"))
        above_chance = chance != chance or (hs == hs and hs >= k * chance)
        # Where the rung scores a round as a conjunction, judge the ratio on the
        # fields, not on the conjunction (the report rungs do, in
        # `evaluate_report_rung`; a lineup rung scores one K-way choice and has
        # no exponent to remove, so this branch is normally the joint one).
        # `mutual` needed three fields right on each of two novel meanings, so
        # per-field accuracy entered this number to the sixth power -- five
        # fields make it the tenth: a code generalising at 0.73 per field against 0.80
        # trained -- a per-field ratio of 0.91, plainly productive -- scores
        # 0.15 against 0.26 as a whole round, a joint ratio of 0.58 that fails
        # a bar it should clear; and 0.60 per field reads 0.05/0.26 = 0.18,
        # which is not distinguishable from a code that generalises not at all.
        # The per-field ratio is the same question with the exponent removed,
        # normalised by the headroom over a message-blind guesser so a memorised
        # code reads 0.00 rather than the base rate it scores anyway.
        f_ratio = _num(ev.get("holdout_field_ratio"))
        if f_ratio == f_ratio:
            hf, sf = _num(ev.get("holdout_fields")), _num(ev.get("seen_fields"))
            # Per field, named: one field carrying two weak ones is exactly what
            # this gate must not let through, and an average cannot show it.
            each = ev.get("holdout_field_ratios") or []
            names = tuple(ev.get("holdout_field_names") or COMBO_FIELDS)
            per = (" [" + ", ".join("%s %s" % (n, _fmt(_num(r)))
                                    for n, r in zip(names, each)) + "]") if each else ""
            checks["describes combinations it never trained on"] = (
                f_ratio >= c.min_holdout_ratio,
                "held-out %s vs seen %s per field = %s of the headroom%s, need "
                "%.2f (the whole round: %s vs %s)"
                % (_fmt(hf), _fmt(sf), _fmt(f_ratio), per, c.min_holdout_ratio,
                   _fmt(hs), _fmt(seen)))
        else:
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
    """The lineup: K candidate lots, five fields each, then the asked field."""
    return (list(LOT_KINDS) * cfg.curriculum.n_candidates) + [K_FIELD]


def informer_schema(cfg: Config) -> list[int]:
    """One lot to describe, and which of its fields is being asked about."""
    return list(LOT_KINDS) + [K_FIELD]


def phase_schema(cfg: Config, role: int, phase: Phase) -> list[int]:
    """Field kinds for each observation slot in this phase, padded to the layout."""
    from .world import buyer_schema, farmer_schema, n_obs_slots
    if phase.referential:
        base = informer_schema(cfg) if role == phase.informer else guesser_schema(cfg)
    elif phase.mutual:
        base = informer_schema(cfg)                 # each holds one lot
    elif role == FARMER:
        base = farmer_schema(cfg.world)
    else:
        base = buyer_schema(cfg.world)              # a request is a lot
    n = n_obs_slots(cfg.world, cfg)
    return base[:n] + [K_EMPTY] * max(0, n - len(base))


# ==========================================================================
# the lineup game
# ==========================================================================
@dataclass
class ReferentialBatch:
    """A batch of lineup rounds, as tensors.

    ``meanings`` is (B, K, 5) -- K candidate lots per round, already shuffled --
    ``target`` says which one the informer was shown, and ``query`` says which
    field the round turns on: 0..4 one field of the lot (fruit, colour,
    quality, quantity, price), or ``ASK_ALL`` when the candidates differ in any
    of them.

    A query round is where an adjective can pay for itself: every candidate
    shares the other fields, so the only thing worth saying is the value of the
    one that differs.
    """
    meanings: torch.Tensor        # (B, K, 5)
    target: torch.Tensor          # (B,)
    query: torch.Tensor           # (B,) 0..4 or ASK_ALL
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
        """(B, 5) -- the lot the informer sees."""
        idx = self.target.view(-1, 1, 1).expand(-1, 1, self.meanings.shape[-1])
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
    """Draws lineups over the same lots the market trades.

    Two kinds of round:

    * a **query round** (``query`` is a field): the candidates share every field
      but one, so the describer has to convey that one value and nothing else.
      This is what the naming rungs are built from, one field at a time and then
      mixed.
    * an **open round** (``query`` is ``ASK_ALL``): the candidates differ in any
      combination of fields, mostly as an anchor plus one-field near misses, so
      every field has to be named at once.

    Held-out (fruit, colour, quality) combinations never appear as a training
    target. ``held_out=True`` draws them deliberately: that is the productivity
    test, and nothing in a fused code can pass it. Quantity (0 to the most a
    barn holds -- 0 is what a farmer says of a lot it does not carry) and price
    are drawn uniformly over all their values.
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
        self.spans = lot_spans(w)
        held = torch.zeros((w.n_varieties, w.n_colors, w.n_quality), dtype=torch.bool)
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
        """(n, 5) lots, from the training combinations or the reserved ones."""
        pool = self.held_combos if held_out else self.train_combos
        if pool.shape[0] == 0:
            pool = self.train_combos
        idx = torch.randint(0, pool.shape[0], (n,), device=self.device, generator=self.gen)
        combo = pool[idx]
        qty = torch.randint(0, self.spans[3], (n,), device=self.device, generator=self.gen)
        price = torch.randint(0, self.spans[4], (n,), device=self.device, generator=self.gen)
        return torch.cat([combo, qty.unsqueeze(1), price.unsqueeze(1)], dim=1)

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
        """Two private lots per round, drawn independently."""
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
        W = N_LOT_FIELDS
        base = self._draw(n, held_out)                       # (n, 5): the target
        # every value of the queried field, with the other fields held fixed
        grid = base.unsqueeze(1).repeat(1, sp, 1)
        grid[:, :, field] = torch.arange(sp, device=self.device).unsqueeze(0)
        reserved = self.is_held_out(grid.reshape(-1, W)).view(n, sp)
        is_base = grid[:, :, field] == base[:, field].unsqueeze(1)
        # rank: real alternatives first, then reserved ones, never the target
        score = (torch.rand(n, sp, device=self.device, generator=self.gen)
                 + reserved.float() * 2.0 + is_base.float() * 4.0)
        pick = torch.argsort(score, dim=1)[:, :K - 1]
        others = grid.gather(1, pick.unsqueeze(-1).expand(-1, -1, W))[:, :, field]
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
        W = N_LOT_FIELDS
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
        # Hard rounds: the whole lineup is one lot plus near misses of it, each
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
                perm[hard] = cl.gather(1, order.unsqueeze(-1).expand(-1, -1, W))
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
        n, K, W = cand.shape
        for _ in range(16):
            bad = torch.zeros(n, K, dtype=torch.bool, device=self.device)
            for a in range(K):
                for b in range(a + 1, K):
                    bad[:, b] |= (cand[:, a] == cand[:, b]).all(dim=1)
            if not held_out:
                bad |= self.is_held_out(cand.reshape(-1, W)).view(n, K)
            if not bool(bad.any()):
                break
            r, k = bad.nonzero(as_tuple=True)
            cand[r, k] = self._draw(r.shape[0], held_out=held_out)
        return cand

    def _cluster(self, anchor: torch.Tensor, K: int) -> torch.Tensor:
        """(n, K, 5): the anchor, then K-1 of its one-field near misses.

        The *field* a near miss differs in is drawn uniformly, and only then a
        value of it. Drawing uniformly over all one-field neighbours instead
        weighted each field by how many values it has: with nine quantities
        against four fruits, fruit decided an open round 7% of the time and a
        describer could drop it almost for free. Every field has to be the one
        that decides often enough to be worth saying.
        """
        n = anchor.shape[0]
        W = N_LOT_FIELDS
        rows = torch.arange(n, device=self.device)
        spans = torch.tensor(self.spans, dtype=torch.long, device=self.device)
        picks: list[torch.Tensor] = []
        fields_used: list[torch.Tensor] = []
        for _ in range(K - 1):
            cand = anchor.clone()
            chosen = torch.zeros(n, dtype=torch.long, device=self.device)
            pending = torch.ones(n, dtype=torch.bool, device=self.device)
            for _attempt in range(24):
                field = torch.randint(0, W, (n,), device=self.device, generator=self.gen)
                span = spans[field]
                step = 1 + (torch.rand(n, device=self.device, generator=self.gen)
                            * (span - 1).float()).long().clamp(max=(span - 2).clamp(min=0))
                new = anchor.clone()
                new[rows, field] = (anchor[rows, field] + step) % span
                # a reserved combination is never a candidate; nor is a repeat;
                # and the near misses differ from the anchor in *different*
                # fields, so a round asks two fields of the describer, not one
                bad = self.is_held_out(new)
                for prev, f_prev in zip(picks, fields_used):
                    bad = bad | (new == prev).all(dim=1) | (field == f_prev)
                cand = torch.where(pending.unsqueeze(1), new, cand)
                chosen = torch.where(pending, field, chosen)
                pending = pending & bad
                if not bool(pending.any()):
                    break
            picks.append(cand)
            fields_used.append(chosen)
        return torch.cat([anchor.unsqueeze(1)] + [p.unsqueeze(1) for p in picks], dim=1)

    def sample(self, n: int, informer: int = FARMER, held_out: bool = False,
               hard_frac: Optional[float] = None, mix: "tuple | None" = None,
               query: "int | None" = None, mixed_query: bool = False
               ) -> ReferentialBatch:
        """One batch of lineup rounds, mixed as the rung asks.

        ``mix`` weights the kinds of round -- one per lot field, then all fields
        -- and is how a rung adds a field without dropping the ones below it.
        ``query`` forces a single kind (used by the probes, which ask about one
        field at a time).
        """
        if query is not None:
            mix = tuple(1.0 if i == int(query) else 0.0 for i in range(N_KINDS))
        elif mixed_query and mix is None:
            mix = (1.0 / N_LOT_FIELDS,) * N_LOT_FIELDS + (0.0,)
        elif mix is None:
            mix = (0.0,) * N_LOT_FIELDS + (1.0,)
        mix = tuple(mix) + (0.0,) * (N_KINDS - len(mix))
        total = float(sum(mix)) or 1.0
        counts = [int(round(n * w / total)) for w in mix]
        counts[max(range(N_KINDS), key=lambda i: counts[i])] += n - sum(counts)

        K = min([self._n_candidates(f) for f in range(N_LOT_FIELDS)]
                + [self._n_candidates(None)])
        W = N_LOT_FIELDS
        cand = torch.zeros((n, K, W), dtype=torch.long, device=self.device)
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
    # How much of the lot the guess actually got: the fields the chosen
    # candidate shares with the target. In a round that turns on one field this
    # is a constant and does nothing; in a round that turns on all of them it is
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
    """Both parties hold one private lot."""
    f_meaning: torch.Tensor       # (B, 5)
    b_meaning: torch.Tensor       # (B, 5)
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
        B = len(self)
        q = torch.full((B, 1), ASK_ALL, dtype=torch.long, device=x.device)
        x = torch.cat([x, q], dim=1)
        pad = torch.zeros((B, max(0, n - x.shape[1])), dtype=torch.long,
                          device=x.device)
        return torch.cat([x, pad], dim=1)[:, :n]


def report_fields(cfg: Config, report: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(B, n) bool -- was each reported field right."""
    return report == truth


# ==========================================================================
# report rungs: what one side holds, read off the other's heads
# ==========================================================================
def field_truth(cfg: Config, sb, name: str) -> torch.Tensor:
    """The true value of one reportable field, from whichever side holds it."""
    w = cfg.world
    if name == "fruit":
        return sb.want_variety
    if name == "colour":
        return sb.want_color
    if name == "quality":                     # the least the buyer will take
        return sb.min_quality
    if name == "quantity":
        return sb.need_qty
    if name == "price":                       # the most the buyer will pay
        return sb.max_price
    if name == "stock":                       # of the lot that was asked about
        return sb.offered_stock.clamp(0, w.max_qty)
    if name == "lot-quality":
        return sb.offered_quality
    if name == "reservation":                 # the least the farmer will take
        return sb.reservation
    if name == DEAL_FIELD:                    # is this one worth doing at all
        return sb.viable.long()
    raise KeyError(name)


# older names, kept so an analysis written against them still runs
request_truth = field_truth


def report_spec(cfg: Config, phase: Phase, scen) -> dict[int, list[tuple]]:
    """{role: [(field name, head, truth (B,)), ...]} -- what each role must report."""
    out: dict[int, list[tuple]] = {FARMER: [], BUYER: []}
    if phase.mutual:
        for role, other in ((FARMER, scen.b_meaning), (BUYER, scen.f_meaning)):
            out[role] = [(LOT_FIELDS[i], H_REPORT[i], other[:, i])
                         for i in range(N_LOT_FIELDS)]
        return out
    for role in (FARMER, BUYER):
        out[role] = [(name, head, field_truth(cfg, scen, name))
                     for name, head in phase.report_heads(role).items()]
    return out


def resolve_reports(cfg: Config, phase: Phase, scen, dec: dict, f_cost: torch.Tensor,
                    b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """Score a batch of report rounds -- `mutual`, `order`, `offer`, `judge`.

    Every report rung is the same event: one side holds facts the other cannot
    see, and the other has to put them in its heads. Each side is paid for
    reading (``decode``, per field) and for being read (``understood``, per
    field), which is what gives each message a gradient, plus the round bonus
    only when every reporting side got everything right at once.

    ``dec[role]`` is the role's whole decision row; the report is read out of
    the heads the rung scores here, and nowhere else. Every caller used to slice
    the columns itself, and they disagreed: the reward once read a belief about
    *quantity* and scored it against the colour, a field the rung trained on a
    different head, so it could only ever be right by luck.
    """
    R = cfg.reward
    spec = report_spec(cfg, phase, scen)
    B = f_cost.shape[0]
    dev = f_cost.device
    fields: dict[int, torch.Tensor] = {}
    frac: dict[int, torch.Tensor] = {}
    ok: dict[int, torch.Tensor] = {}
    new_ok: dict[int, torch.Tensor] = {}
    for role in (FARMER, BUYER):
        rows = spec[role]
        if not rows:
            fields[role] = torch.zeros((B, 0), dtype=torch.bool, device=dev)
            frac[role] = torch.zeros(B, device=dev)
            ok[role] = torch.ones(B, dtype=torch.bool, device=dev)
            new_ok[role] = torch.ones(B, dtype=torch.bool, device=dev)
            continue
        got = torch.stack([dec[role][:, head] == truth for _, head, truth in rows], dim=1)
        fields[role] = got
        frac[role] = got.float().mean(1)
        ok[role] = got.all(dim=1)
        new = set(phase.new_names(role))
        cols = [i for i, (name, _, _) in enumerate(rows) if name in new]
        new_ok[role] = got[:, cols].all(dim=1) if cols else ok[role]
    both = ok[FARMER] & ok[BUYER]
    joint = R.refer_success * both.float() + R.refer_miss * (~both).float()
    f = joint + R.decode * frac[FARMER] + R.understood * frac[BUYER] - f_cost.float()
    b = joint + R.decode * frac[BUYER] + R.understood * frac[FARMER] - b_cost.float()
    zero_l = torch.zeros_like(f_cost, dtype=torch.long)
    zero_f = torch.zeros_like(f)
    no = torch.zeros_like(both)
    f_names = [name for name, _, _ in spec[FARMER]]
    b_names = [name for name, _, _ in spec[BUYER]]
    return {
        "farmer_reward": f, "buyer_reward": b,
        "success": both, "comprehended": both, "both_judged": both,
        "farmer_decode": frac[FARMER], "buyer_decode": frac[BUYER],
        "farmer_report_ok": ok[FARMER], "buyer_report_ok": ok[BUYER],
        "farmer_fields": fields[FARMER], "buyer_fields": fields[BUYER],
        "farmer_new_ok": new_ok[FARMER], "buyer_new_ok": new_ok[BUYER],
        "farmer_field_names": f_names, "buyer_field_names": b_names,
        "both_accept": both,
        "agree_variety": both, "agree_qty": both, "agree_price": both,
        "agreed_variety": zero_l, "agreed_qty": zero_l, "agreed_price": zero_l,
        "traded_qty": zero_l, "trade_value": zero_f,
        "farmer_profit": zero_f, "buyer_savings": zero_f,
        "correct_no_deal": no, "missed_deal": no, "one_sided": no, "bad_deal": no,
    }


def resolve_mutual(cfg: Config, mb: MutualBatch, f_dec: torch.Tensor,
                   b_dec: torch.Tensor, f_cost: torch.Tensor,
                   b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """The mutual rung, scored as a report round (both report the other's lot)."""
    phase = next(p for p in ladder(cfg) if p.mutual)
    return resolve_reports(cfg, phase, mb, {FARMER: f_dec, BUYER: b_dec}, f_cost, b_cost)


def resolve_request(cfg: Config, phase, sb, dec: dict, f_cost: torch.Tensor,
                    b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """A report rung in the trading world (order, offer, judge)."""
    return resolve_reports(cfg, phase, sb, dec, f_cost, b_cost)


# ==========================================================================
# hindsight: what each scored head should have said, once the round is over
# ==========================================================================
def hindsight_targets(cfg: Config, phase: Phase, scen) -> dict[int, dict[int, torch.Tensor]]:
    """{role: {head: (B,) correct class}} for the heads this rung scores.

    After a round both parties learn how it came out: which candidate was meant,
    what the other's lot was, what the buyer actually wanted, what the farmer
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

    And why it waits (``train.hindsight_from_rung``): while no code exists yet, a
    listener told the answer learns, correctly, that the messages carry nothing,
    and the speaker's gradient dies with it. It joins once every word exists.
    """
    out: dict[int, dict[int, torch.Tensor]] = {FARMER: {}, BUYER: {}}
    w = cfg.world
    if phase.referential:
        out[phase.guesser][H_CHOICE] = scen.target
    elif phase.reporting:
        for role, rows in report_spec(cfg, phase, scen).items():
            for _, head, truth in rows:
                out[role][head] = truth
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
