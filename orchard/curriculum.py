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
from .world import K_EMPTY, K_PRICE, K_QTY, K_QUALITY, K_VARIETY

# Head layout shared by every phase.  Sampling all of them keeps tensor shapes
# constant across a transition; each phase says which ones actually count.
H_ACCEPT, H_VARIETY, H_QTY, H_PRICE = 0, 1, 2, 3
H_BELIEF = (4, 5, 6, 7)
H_CHOICE = 8
N_HEADS = 9


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
    # Who describes in a lineup rung. Fixed to the farmer in ``refer``; alternated
    # batch by batch in ``refer-swap`` via :meth:`with_informer`.
    informer: int = FARMER

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
        """Played over bare (variety, quantity, quality) tuples, not farms and requests."""
        return self.kind in (KIND_REFER, KIND_SWAP, KIND_MUTUAL)

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
            # Report the partner's (variety, quantity, quality). No price here.
            return list(H_BELIEF[:3])
        if self.order:
            # The farmer fills the order with its deal decision; the buyer only asks.
            return [H_VARIETY, H_QTY] if role == FARMER else []
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
    """The phases, easiest first.  ``n_turns`` never exceeds ``channel.n_turns``."""
    full = max(2, cfg.channel.n_turns)
    return [
        Phase("refer", 0, 1, True, False, False,
              "lineup game: the farmer describes one meaning, the buyer picks it "
              "out of several", kind=KIND_REFER),
        Phase("refer-swap", 1, 1, True, False, False,
              "the same lineup game with roles alternating: everyone describes "
              "and everyone decodes", kind=KIND_SWAP),
        Phase("refer-mutual", 2, 2, True, False, False,
              "both hold a private meaning and each must report the other's; "
              "no price, no accept/reject", kind=KIND_MUTUAL),
        Phase("order", 3, 1, False, False, False,
              "the buyer asks for a variety and a quantity; the farmer must fill "
              "the order exactly with its deal decision", kind=KIND_ORDER),
        Phase("haggle", 4, 2, False, True, False,
              "price and budget appear; one message each, then accept or walk"),
        Phase("bargain", 5, full, False, True, False,
              "several turns, so counter-offers are possible"),
        Phase("market", 6, full, False, True, True,
              "the full economy: stock, restocking, viability"),
    ]


def phase_named(cfg: Config, name: str) -> Phase:
    for p in ladder(cfg):
        if p.name == name:
            return p
    raise KeyError("no phase called %r" % (name,))


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
        chance = 1.0 / max(2, c.n_candidates)
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
    for role, label in ((FARMER, "farmer"), (BUYER, "buyer")):
        d = ev.get("speakers", {}).get(label, {})
        ts, nl, pos = _num(d.get("topsim")), _num(d.get("null")), _num(d.get("positional"))
        checks["%s describes: topsim clear of null" % label] = (
            ts == ts and nl == nl and ts - nl >= c.min_topsim_over_null,
            "%s vs null %s, need +%.2f" % (_fmt(ts), _fmt(nl), c.min_topsim_over_null))
        checks["%s describes: positional structure" % label] = (
            pos == pos and pos >= c.min_positional_structure,
            "%s, need %.2f" % (_fmt(pos), c.min_positional_structure))
        # Positional structure is fooled by redundancy -- a13-a13-a13-a13 scores
        # 1.0 because every slot names the variety. Coverage asks how much of
        # *each* field the messages carry, corrected for chance.
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
            for name, t in zip(("variety", "quantity", "quality"), ft):
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
    passed = all(v[0] for v in checks.values())
    return passed, {name: {"met": v[0], "detail": v[1]} for name, v in checks.items()}


# ==========================================================================
# observation schemas -- one fixed layout, different content per phase
# ==========================================================================
def guesser_schema(cfg: Config) -> list[int]:
    """The lineup: K candidate meanings, three fields each."""
    return [K_VARIETY, K_QTY, K_QUALITY] * cfg.curriculum.n_candidates


def informer_schema(cfg: Config) -> list[int]:
    """One meaning to describe."""
    return [K_VARIETY, K_QTY, K_QUALITY]


def phase_schema(cfg: Config, role: int, phase: Phase) -> list[int]:
    """Field kinds for each observation slot in this phase, padded to the layout."""
    from .world import buyer_schema, farmer_schema, n_obs_slots
    if phase.referential:
        base = informer_schema(cfg) if role == phase.informer else guesser_schema(cfg)
    elif phase.mutual:
        base = informer_schema(cfg)                 # each holds one meaning
    elif role == FARMER:
        base = farmer_schema(cfg.world)
    else:
        base = buyer_schema(cfg.world)
    n = n_obs_slots(cfg.world, cfg)
    return base[:n] + [K_EMPTY] * max(0, n - len(base))


# ==========================================================================
# the lineup game
# ==========================================================================
@dataclass
class ReferentialBatch:
    """A batch of lineup rounds, as tensors.

    ``meanings`` is (B, K, 3) -- K candidate (variety, quantity, quality) tuples
    per round, already shuffled -- and ``target`` says which one the informer was
    actually shown.
    """
    meanings: torch.Tensor        # (B, K, 3)
    target: torch.Tensor          # (B,)
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
            x = self.true_meaning
        else:
            x = self.meanings.reshape(B, -1)
        if x.shape[1] < n:
            x = torch.cat([x, torch.zeros((B, n - x.shape[1]), dtype=torch.long,
                                          device=x.device)], dim=1)
        return x[:, :n]


class ReferentialWorld:
    """Draws lineups over the same attribute space the trading task uses.

    Candidates are drawn from the same marginals as real requests, so a code
    learned here is a code about the same things -- and the distractors are forced
    to differ from the target, so picking correctly always requires information
    that only the informer had.
    """

    def __init__(self, cfg: Config, device: str = "cpu",
                 generator: Optional[torch.Generator] = None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.gen = generator
        from .batched import TensorWorld
        self.tw = TensorWorld(cfg, device=device, generator=generator)
        self.holdout = self._pick_holdout()

    def _pick_holdout(self) -> torch.Tensor:
        """(H, 3) tuples never used in training: the productivity test.

        A code that names variety, quantity and quality separately can describe
        a combination it has never seen; a lookup table of whole tuples cannot.
        Every value of every field still occurs in training -- only these
        particular *combinations* are withheld.
        """
        import random as _random
        frac = self.cfg.curriculum.holdout_tuple_frac
        w = self.cfg.world
        nq = int(self.tw._qty_cdf.numel())
        allt = [(v, q, u) for v in range(w.n_varieties) for q in range(1, nq + 1)
                for u in range(w.n_quality)]
        k = int(round(frac * len(allt)))
        if k <= 0:
            return torch.zeros((0, 3), dtype=torch.long, device=self.device)
        rng = _random.Random(w.holdout_seed + 17)
        pick = rng.sample(allt, k)
        return torch.tensor(pick, dtype=torch.long, device=self.device)

    def is_held_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.holdout.numel() == 0:
            return torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        return (x.unsqueeze(1) == self.holdout.unsqueeze(0)).all(-1).any(-1)

    def _raw(self, n: int) -> torch.Tensor:
        w = self.cfg.world
        variety = self.tw._categorical(self.tw._variety_cdf, n)
        qty = self.tw._categorical(self.tw._qty_cdf, n) + 1
        quality = self.tw._skew_low(0, w.n_quality - 1, n)
        return torch.stack([variety, qty, quality], dim=1)

    def _draw(self, n: int, held_out: bool = False) -> torch.Tensor:
        """(n, 3) meanings: variety, quantity, quality -- from the training
        combinations, or only from the held-out ones."""
        if held_out and self.holdout.numel():
            idx = torch.randint(0, self.holdout.shape[0], (n,), device=self.device,
                                generator=self.gen)
            return self.holdout[idx].clone()
        x = self._raw(n)
        for _ in range(16):
            bad = self.is_held_out(x)
            if not bool(bad.any()):
                break
            x[bad] = self._raw(int(bad.sum()))
        return x

    def sample_mutual(self, n: int, held_out: bool = False) -> "MutualBatch":
        """Two private meanings per round, drawn independently."""
        return MutualBatch(f_meaning=self._draw(n, held_out), b_meaning=self._draw(n, held_out))

    def _near_miss(self, anchor: torch.Tensor) -> torch.Tensor:
        """Copies of ``anchor`` with exactly one field changed (a random field),
        never landing on a held-out combination."""
        n = anchor.shape[0]
        w = self.cfg.world
        span = torch.tensor([w.n_varieties, int(self.tw._qty_cdf.numel()), w.n_quality],
                            device=self.device)
        lo = torch.tensor([0, 1, 0], device=self.device)
        rows = torch.arange(n, device=self.device)
        out = anchor.clone()
        todo = torch.ones(n, dtype=torch.bool, device=self.device)
        for _ in range(16):
            if not bool(todo.any()):
                break
            idx = todo.nonzero(as_tuple=True)[0]
            m = idx.shape[0]
            field = torch.randint(0, 3, (m,), device=self.device, generator=self.gen)
            fresh = self._raw(m)
            cand = anchor[idx].clone()
            r = torch.arange(m, device=self.device)
            cand[r, field] = fresh[r, field]
            # a fresh value equal to the anchor's steps to a neighbouring value
            same = (cand == anchor[idx]).all(dim=1)
            if bool(same.any()):
                f = field[same]
                v = cand[same, f] - lo[f]
                cand[same, f] = (v + 1) % span[f] + lo[f]
            out[idx] = cand
            todo[idx] = self.is_held_out(cand)
        return out

    def _cluster(self, n: int) -> torch.Tensor:
        """(n, K, 3) lineups of an anchor and K-1 one-field near misses of it,
        in random order, all distinct."""
        K = self.cfg.curriculum.n_candidates
        anchor = self._draw(n)
        members = [anchor] + [self._near_miss(anchor) for _ in range(K - 1)]
        cl = torch.stack(members, dim=1)
        for _ in range(16):              # two near misses can coincide: redo one
            dup = torch.zeros(n, K, dtype=torch.bool, device=self.device)
            for a in range(K):
                for b in range(a + 1, K):
                    dup[:, b] |= (cl[:, a] == cl[:, b]).all(dim=1)
            if not bool(dup.any()):
                break
            r, k = dup.nonzero(as_tuple=True)
            cl[r, k] = self._near_miss(anchor[r])
        order = torch.argsort(torch.rand(n, K, device=self.device, generator=self.gen), dim=1)
        return cl.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))

    def sample(self, n: int, informer: int = FARMER, held_out: bool = False,
               hard_frac: Optional[float] = None) -> ReferentialBatch:
        K = self.cfg.curriculum.n_candidates
        cand = torch.stack([self._draw(n, held_out and k == 0) for k in range(K)], dim=1)
        # Distractors must differ from the target, or a "correct" guess could be
        # luck rather than information.  Re-draw any duplicate of slot 0.
        for k in range(1, K):
            for _ in range(8):
                same = (cand[:, k] == cand[:, 0]).all(dim=1)
                if not bool(same.any()):
                    break
                cand[:, k] = torch.where(same.unsqueeze(1), self._draw(n), cand[:, k])
        # Put the target in a random position so its index carries no information.
        target = torch.randint(0, K, (n,), device=self.device, generator=self.gen)
        perm = cand.clone()
        ar = torch.arange(n, device=self.device)
        perm[ar, target] = cand[:, 0]
        perm[ar, 0] = cand[ar, target]
        # Hard rounds. With independent candidates, variety and quality alone
        # single out the target 69% of the time, so the code that emerged carried
        # quality, a little variety and no quantity at all -- and every trading
        # rung needs quantity exactly. In a hard round the whole lineup is one
        # anchor plus near misses of it, each differing in one field, so every
        # field has to be named. The target is uniform among the members: the
        # first version built the near misses around the *target*, which made it
        # the one candidate the others clustered around, and the guesser found it
        # 42% of the time with the channel muted (chance 25%).
        p = self.cfg.curriculum.hard_distractor_frac if hard_frac is None else hard_frac
        if p > 0 and not held_out:
            hard = torch.rand(n, device=self.device, generator=self.gen) < p
            h = int(hard.sum())
            if h:
                perm[hard] = self._cluster(h)
        return ReferentialBatch(meanings=perm, target=target, day=0, informer=informer)


def resolve_referential(cfg: Config, rb: ReferentialBatch, choice: torch.Tensor,
                        f_symbols: torch.Tensor,
                        b_symbols: torch.Tensor) -> dict[str, torch.Tensor]:
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
    f = reward - R.symbol_cost * f_symbols.float()
    b = reward - R.symbol_cost * b_symbols.float()
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
    """(B, 3) bool: which of variety / quantity / quality were reported right.

    Quantity is judged with ``curriculum.mutual_qty_tol`` (default exact),
    because the order and trading rungs need the exact quantity.
    """
    tol = cfg.curriculum.mutual_qty_tol
    return torch.stack([report[:, 0] == truth[:, 0],
                        (report[:, 1] - truth[:, 1]).abs() <= tol,
                        report[:, 2] == truth[:, 2]], dim=1)


def resolve_mutual(cfg: Config, mb: MutualBatch, f_report: torch.Tensor,
                   b_report: torch.Tensor, f_symbols: torch.Tensor,
                   b_symbols: torch.Tensor) -> dict[str, torch.Tensor]:
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
    f = joint + R.decode * f_frac + R.understood * b_frac - R.symbol_cost * f_symbols.float()
    b = joint + R.decode * b_frac + R.understood * f_frac - R.symbol_cost * b_symbols.float()
    zero_l = torch.zeros_like(f_symbols)
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
            for i, h in enumerate(H_BELIEF[:3]):
                out[role][h] = other[:, i]
    elif hasattr(scen, "want_variety"):
        out[FARMER][H_VARIETY] = scen.want_variety
        out[FARMER][H_QTY] = scen.need_qty
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
def resolve_order(cfg: Config, sb, f_dec: torch.Tensor, f_symbols: torch.Tensor,
                  b_symbols: torch.Tensor) -> dict[str, torch.Tensor]:
    """Did the farmer's deal decision fill the buyer's order exactly?

    Scored on the deal heads (variety, quantity), which no earlier rung used.
    Both parties are paid for the round -- being understood and understanding
    are one event here too -- plus partial credit per field, so a farmer that
    gets the variety right but the quantity wrong is told which half worked.
    """
    R = cfg.reward
    var_ok = f_dec[:, 1] == sb.want_variety
    qty_ok = f_dec[:, 2] == sb.need_qty
    both = var_ok & qty_ok
    fields = torch.stack([var_ok, qty_ok], dim=1)
    frac = fields.float().mean(1)
    base = (R.refer_success * both.float() + R.refer_miss * (~both).float()
            + R.decode * frac)
    f = base - R.symbol_cost * f_symbols.float()
    b = base - R.symbol_cost * b_symbols.float()
    zero_l = torch.zeros_like(f_symbols)
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
