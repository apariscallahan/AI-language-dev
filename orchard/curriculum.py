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
from .world import K_COLOR, K_EMPTY, K_FIELD, K_PRICE, K_QTY, K_QUALITY, K_VARIETY

# Head layout shared by every phase.  Sampling all of them keeps tensor shapes
# constant across a transition; each phase says which ones actually count.
H_ACCEPT, H_VARIETY, H_QTY, H_PRICE = 0, 1, 2, 3
H_BELIEF = (4, 5, 6, 7)          # the other party's (fruit, quantity, quality, price)
H_CHOICE = 8                     # which candidate in a lineup
H_BELIEF_COLOR = 9               # the other party's colour -- appended, so 0..8 kept
N_HEADS = 10
# Reporting a thing: (fruit, colour, quality), the three fields a meaning has.
H_REPORT = (H_BELIEF[0], H_BELIEF_COLOR, H_BELIEF[2])


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
    # Which field a lineup round turns on: 0 fruit, 1 colour, 2 quality; None
    # means the candidates differ in any of them. ``mixed_query`` draws it per
    # round instead -- the rung where one word has to serve whichever field is
    # asked, which is where an adjective earns its keep.
    query: "int | None" = None
    mixed_query: bool = False

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
            # The farmer fills the order with its deal decision; the buyer only asks.
            return [H_VARIETY, H_BELIEF_COLOR, H_QTY] if role == FARMER else []
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
              "quality, so only the fruit needs saying", kind=KIND_SWAP, query=0),
        Phase("name-color", 1, 1, True, False, False,
              "the same, over colour alone: same fruit, same quality, different "
              "colours -- a word for a colour and nothing else", kind=KIND_SWAP, query=1),
        Phase("name-quality", 2, 1, True, False, False,
              "the same, over quality alone", kind=KIND_SWAP, query=2),
        Phase("name-all", 3, 1, True, False, False,
              "candidates differing in any field, mostly one-field near misses: "
              "the whole (fruit, colour, quality) has to be named at once",
              kind=KIND_SWAP),
        Phase("describe-one", 4, 1, True, False, False,
              "the asked-about field changes round by round, so one word has to "
              "mean a colour wherever it appears -- including on combinations "
              "never seen in training", kind=KIND_SWAP, mixed_query=True),
        Phase("mutual", 5, 2, True, False, False,
              "both hold a private thing and each must report the other's; "
              "no price, no accept/reject", kind=KIND_MUTUAL),
        Phase("order", 6, 1, False, False, False,
              "trading begins: the buyer asks for a fruit, a colour and a "
              "quantity; the farmer must fill the order exactly", kind=KIND_ORDER),
        Phase("haggle", 7, 2, False, True, False,
              "price and budget appear; one message each, then accept or walk"),
        Phase("bargain", 8, full, False, True, False,
              "several turns, so counter-offers are possible"),
        Phase("market", 9, full, False, True, True,
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


def promotion_for(cfg: Config, phase: Phase) -> Promotion:
    c = cfg.curriculum
    lo, hi = rung_budget(cfg, phase)
    if phase.referential:
        span = (cfg.world.n_varieties, cfg.world.n_colors, cfg.world.n_quality)
        K = max(2, c.n_candidates)
        if phase.query is not None:
            K = min(K, span[phase.query])
        elif phase.mixed_query:
            K = min([K] + [span[f] for f in range(3)])
        chance = 1.0 / K
        return Promotion(
            min_success=max(c.refer_min_success, 2.0 * chance),
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
    whole_thing = phase.swaps and phase.query is None and not phase.mixed_query
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
    if phase.mutual:
        succ, chance = _num(ev.get("success")), _num(ev.get("chance"))
        checks["both decode in the same round"] = (
            succ == succ and succ >= rule.min_success
            and (chance != chance or succ >= k * chance),
            "%s, need %.3f and %.1fx the muted rate %s"
            % (_fmt(succ), rule.min_success, k, _fmt(chance)))
    whole = (phase.mutual or (phase.swaps and phase.query is None
                              and not phase.mixed_query))
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
        """One in how many, by luck alone, on this rung."""
        if phase.mixed_query:
            return 1.0 / min(self._n_candidates(f) for f in range(3))
        return 1.0 / self._n_candidates(phase.query)

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
               hard_frac: Optional[float] = None,
               query: "int | None" = None, mixed_query: bool = False
               ) -> ReferentialBatch:
        """One batch of lineup rounds.

        ``query`` fixes the field the round turns on (a naming rung);
        ``mixed_query`` draws it per round (the rung where one word has to serve
        whichever field is asked); neither means an open round.
        """
        if mixed_query:
            # One lineup width for every field, so a round cannot be told apart by
            # how many candidates it has.
            K = min(self._n_candidates(f) for f in range(3))
            q = torch.randint(0, 3, (n,), device=self.device, generator=self.gen)
            cand = torch.zeros((n, K, 3), dtype=torch.long, device=self.device)
            target = torch.zeros(n, dtype=torch.long, device=self.device)
            for f in range(3):
                m = q == f
                k = int(m.sum())
                if not k:
                    continue
                c, t = self._query_round(k, f, held_out)
                cand[m] = c[:, :K]
                target[m] = t
        elif query is None:
            p = self.cfg.curriculum.hard_distractor_frac if hard_frac is None else hard_frac
            cand, target = self._open_round(n, held_out, 0.0 if held_out else p)
            q = torch.full((n,), ASK_ALL, dtype=torch.long, device=self.device)
        else:
            q = torch.full((n,), int(query), dtype=torch.long, device=self.device)
            cand, target = self._query_round(n, int(query), held_out)
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
    reward = R.refer_success * correct.float() + R.refer_miss * (~correct).float()
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
        "farmer_decode": correct.float(), "buyer_decode": correct.float(),
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


def resolve_mutual(cfg: Config, mb: MutualBatch, f_report: torch.Tensor,
                   b_report: torch.Tensor, f_cost: torch.Tensor,
                   b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """Score a batch of mutual rounds.

    ``f_report`` is the farmer's (variety, quantity, quality) report of the
    buyer's meaning, and vice versa. Each side is paid for reading the other
    (``decode``) and for being read (``understood``) field by field, which is
    what gives each message a gradient, plus the full round bonus only when both
    reports are right at once.
    """
    R = cfg.reward
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
    elif hasattr(scen, "want_variety"):
        out[FARMER][H_VARIETY] = scen.want_variety
        out[FARMER][H_QTY] = scen.need_qty
        out[FARMER][H_BELIEF_COLOR] = scen.want_color
        if not phase.order:
            out[BUYER][H_VARIETY] = scen.want_variety
            out[BUYER][H_QTY] = scen.need_qty
            for role in (FARMER, BUYER):
                out[role][H_ACCEPT] = scen.viable.long()
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
def resolve_order(cfg: Config, sb, f_dec: torch.Tensor, f_cost: torch.Tensor,
                  b_cost: torch.Tensor) -> dict[str, torch.Tensor]:
    """Did the farmer's deal decision fill the buyer's order exactly?

    Scored on the deal heads (variety, quantity), which no earlier rung used.
    Both parties are paid for the round -- being understood and understanding
    are one event here too -- plus partial credit per field, so a farmer that
    gets the variety right but the quantity wrong is told which half worked.
    """
    R = cfg.reward
    var_ok = f_dec[:, H_VARIETY] == sb.want_variety
    col_ok = f_dec[:, H_BELIEF_COLOR] == sb.want_color
    qty_ok = f_dec[:, H_QTY] == sb.need_qty
    both = var_ok & col_ok & qty_ok
    fields = torch.stack([var_ok, col_ok, qty_ok], dim=1)
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
        "farmer_decode": frac, "buyer_decode": zero_f,
        "order_fields": fields,
        "both_accept": both, "agree_variety": var_ok, "agree_qty": qty_ok,
        "agree_price": both,
        "agreed_variety": f_dec[:, 1], "agreed_qty": f_dec[:, 2], "agreed_price": zero_l,
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
