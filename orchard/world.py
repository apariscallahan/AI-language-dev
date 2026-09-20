"""The apple world: hidden state, scenario sampling, and what makes a deal viable.

Spec section 1.2 is the load-bearing part of this file.  The *only* reason
language is necessary in this simulation is that the Farmer and the Buyer each
hold private facts the other needs and cannot observe.  This module defines that
asymmetry; :mod:`orchard.env` enforces it.

  Farmer knows : how much of **each fruit** is in the barn, the colour and
                 quality of each, and the lowest price they will accept
  Buyer knows  : which fruit they want, in which colour, how many, the minimum
                 quality they will take, and the most they can pay

Fruit and colour are separate fields, and a lot is a (fruit, colour, quality)
combination.  A quarter of those combinations are never trained on anywhere in
the project (see :class:`ComboHoldout`), so a code that fuses the three into one
name per thing cannot describe them, and one that names the parts can.

Why farms carry several varieties
---------------------------------
An earlier version gave each farm one variety and then *coerced* roughly half of
all encounters to be compatible, so that viable deals were common enough to
learn from.  The scrambled-channel ablation caught what that did: it made the
buyer's wanted variety predictable from the farmer's own stock, and the farmer
could score 0.67 on "which variety do they want" -- against a chance rate of
0.33 -- without listening to anything.  Worse, with one variety per farm the
farmer's best answer is always "the one I have", so the variety dimension could
never reward listening even in principle.

Stocking several varieties removes both problems at once.  The buyer's wanted
variety is drawn independently and uniformly, so no amount of staring at one's
own barn predicts it; and the farmer now has a real choice to get right.  Every
marginal here is independent across the two agents, so nothing about one side's
private state shifts the odds on the other's.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

from .config import WorldConfig

# Field kinds, used everywhere an observation has to be embedded, binned,
# measured or printed.  Keeping one schema means adding a field changes one place.
K_EMPTY, K_VARIETY, K_QTY, K_QUALITY, K_PRICE, K_COLOR, K_FIELD = 0, 1, 2, 3, 4, 5, 6
KIND_NAMES = {K_EMPTY: "empty", K_VARIETY: "fruit", K_QTY: "quantity",
              K_QUALITY: "quality", K_PRICE: "price", K_COLOR: "colour",
              K_FIELD: "asked-about field"}
# The three fields a thing is described by, in the order they are held
# everywhere (observations, meanings, held-out combinations, reports).
MEANING_KINDS = (K_VARIETY, K_COLOR, K_QUALITY)
MEANING_FIELDS = ("fruit", "colour", "quality")


@dataclass(frozen=True)
class FarmerState:
    """Private to the Farmer: the whole barn."""
    # One lot per (fruit, colour), flattened fruit-major: cell (f, c) is at
    # f * n_colors + c. A shopper who wants green pears is asking about one cell,
    # which is why the barn has to be laid out this way -- with a single colour
    # per fruit, two thirds of shoppers could not be served by anybody and
    # refusing every deal beat trading.
    stocks: tuple[int, ...]        # per (fruit, colour); 0 means "none of that"
    qualities: tuple[int, ...]     # per (fruit, colour) (meaningless where stock is 0)
    reservation: int               # price bin: the lowest per-unit price they take
    n_colors: int = 1

    def cell(self, variety: int, color: int) -> int:
        return variety * self.n_colors + color

    def stock_of(self, variety: int, color: int = 0) -> int:
        return self.stocks[self.cell(variety, color)]

    def quality_of(self, variety: int, color: int = 0) -> int:
        return self.qualities[self.cell(variety, color)]

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.stocks) + tuple(self.qualities) + (self.reservation,)


@dataclass(frozen=True)
class BuyerState:
    """Private to the Buyer: one shopping list."""
    want_variety: int
    want_color: int
    need_qty: int
    min_quality: int
    max_price: int                 # price bin: the most they will pay per unit

    def as_tuple(self) -> tuple[int, ...]:
        return (self.want_variety, self.want_color, self.need_qty, self.min_quality,
                self.max_price)


# --------------------------------------------------------------------------
# Observation schema
# --------------------------------------------------------------------------
def farmer_schema(cfg: WorldConfig) -> list[int]:
    cells = cfg.n_varieties * cfg.n_colors
    return [K_QTY] * cells + [K_QUALITY] * cells + [K_PRICE]


def buyer_schema(cfg: WorldConfig) -> list[int]:
    return [K_VARIETY, K_COLOR, K_QTY, K_QUALITY, K_PRICE]


def n_obs_slots(cfg: WorldConfig, full: "object | None" = None) -> int:
    """Slots in the shared observation layout.

    One layout serves every role in every phase, with the shorter ones padded.
    That is what lets a population carry its weights across a curriculum
    transition: the architecture does not change, only what is written into it.
    The lineup game needs the most room -- three fields per candidate.
    """
    n = max(len(farmer_schema(cfg)), len(buyer_schema(cfg)))
    if full is not None and getattr(full, "curriculum", None) is not None:
        n = max(n, 3 * full.curriculum.n_candidates)
    return n


def obs_schema(cfg: WorldConfig, role: int, full: "object | None" = None) -> list[int]:
    """Field kinds for each observation slot, padded with K_EMPTY."""
    from .env import FARMER
    base = farmer_schema(cfg) if role == FARMER else buyer_schema(cfg)
    n = n_obs_slots(cfg, full)
    return base[:n] + [K_EMPTY] * max(0, n - len(base))


def field_labels(cfg: WorldConfig, role: int) -> list[str]:
    """Human names for the slots, used by metrics and the report."""
    from .env import FARMER
    if role == FARMER:
        cells = ["%s_%s" % (c, v) for v in cfg.variety_names for c in cfg.color_names]
        names = (["stock_" + x for x in cells] + ["quality_" + x for x in cells]
                 + ["reservation"])
    else:
        names = ["want_fruit", "want_colour", "need_qty", "min_quality", "max_price"]
    n = n_obs_slots(cfg)
    return names[:n] + ["-"] * max(0, n - len(names))


def field_spans_by_kind(cfg: WorldConfig) -> dict[int, int]:
    """Value range of each field kind, so distances can be normalised per field."""
    return {K_EMPTY: 1, K_VARIETY: max(cfg.n_varieties - 1, 1),
            K_QTY: max(cfg.max_qty, 1), K_QUALITY: max(cfg.n_quality - 1, 1),
            K_PRICE: max(cfg.n_price_bins - 1, 1),
            K_COLOR: max(cfg.n_colors - 1, 1), K_FIELD: len(MEANING_KINDS)}


def field_spans(cfg: WorldConfig, role: int) -> list[int]:
    """Value range per slot, so distances can be normalised per field."""
    spans = field_spans_by_kind(cfg)
    return [spans[k] for k in obs_schema(cfg, role)]


# --------------------------------------------------------------------------
class ComboHoldout:
    """The (fruit, colour, quality) combinations no agent is ever trained on.

    One set for the whole project: the naming rungs never show them, no barn
    holds them and no shopper asks for them, so success on them at evaluation
    time is a clean test of whether the code has reusable parts.

    The set is **balanced by construction**: exactly one quality is withheld from
    every (fruit, colour) lot, and each quality is withheld from the same number
    of lots. Every fruit, colour and quality therefore appears equally often in
    training and in the held-out set -- what is withheld is always a *pairing*.
    That matters for more than fairness: an unbalanced set makes some lots rarer
    in the barn *and* rarer in shopping lists at the same time, and a farmer can
    then predict the request from its own shelves, which is exactly the
    independence the world exists to protect.
    """

    def __init__(self, cfg: WorldConfig, frac: float, seed: int = 0):
        self.cfg = cfg
        self.combos = [(f, c, q) for f in range(cfg.n_varieties)
                       for c in range(cfg.n_colors) for q in range(cfg.n_quality)]
        self.held: set[tuple[int, int, int]] = set()
        per_cell = int(round(frac * cfg.n_quality))
        per_cell = max(0, min(cfg.n_quality - 1, per_cell))
        if per_cell <= 0:
            return
        rng = random.Random(seed)
        n = cfg.n_varieties
        if per_cell == 1 and cfg.n_colors == n and cfg.n_quality == n:
            # A random Latin square: exactly one combination withheld from every
            # pair of fields, in every direction.
            rows = list(range(n))
            cols = list(range(n))
            syms = list(range(n))
            rng.shuffle(rows)
            rng.shuffle(cols)
            rng.shuffle(syms)
            self.held = {(rows[f], cols[c], syms[(f + c) % n])
                         for f in range(n) for c in range(n)}
            return
        cells = [(f, c) for f in range(cfg.n_varieties) for c in range(cfg.n_colors)]
        need = per_cell * len(cells)
        # spread the withheld qualities evenly over the quality values
        quals = [q for q in range(cfg.n_quality)] * (need // cfg.n_quality + 1)
        quals = quals[:need]
        for _ in range(200):
            rng.shuffle(quals)
            out, ok = set(), True
            for i, (f, c) in enumerate(cells):
                picks = set(quals[i * per_cell:(i + 1) * per_cell])
                if len(picks) != per_cell:        # the same quality twice in one lot
                    ok = False
                    break
                out.update((f, c, q) for q in picks)
            if ok:
                self.held = out
                return
        self.held = {(f, c, quals[i]) for i, (f, c) in enumerate(cells)}

    def __contains__(self, combo) -> bool:
        return tuple(int(x) for x in combo) in self.held

    def __len__(self) -> int:
        return len(self.held)

    @property
    def training(self) -> list[tuple[int, int, int]]:
        return [c for c in self.combos if c not in self.held]

    def counts(self) -> dict[str, list[int]]:
        """How many held-out combinations each value of each field appears in."""
        out = {}
        for i, name in enumerate(MEANING_FIELDS):
            span = (self.cfg.n_varieties, self.cfg.n_colors, self.cfg.n_quality)[i]
            out[name] = [sum(1 for c in self.held if c[i] == v) for v in range(span)]
        return out

    def tensor(self, device="cpu"):
        import torch
        return torch.tensor(sorted(self.held), dtype=torch.long, device=device).reshape(-1, 3)


@dataclass(frozen=True)
class Scenario:
    """One episode's ground truth.  Agents never see this -- only their own half."""
    farmer: FarmerState
    buyer: BuyerState
    day: int = 0
    held_out: bool = False

    # ---- what is actually on offer for what the buyer came for ----------
    @property
    def offered_stock(self) -> int:
        """What the barn holds of exactly what the shopper came for."""
        return self.farmer.stock_of(self.buyer.want_variety, self.buyer.want_color)

    @property
    def offered_quality(self) -> int:
        return self.farmer.quality_of(self.buyer.want_variety, self.buyer.want_color)

    @property
    def offered_color(self) -> int:
        return self.buyer.want_color

    @property
    def variety_ok(self) -> bool:
        """The barn has that fruit in that colour at all."""
        return self.offered_stock > 0

    @property
    def color_ok(self) -> bool:
        return self.variety_ok

    @property
    def stock_ok(self) -> bool:
        return self.offered_stock >= self.buyer.need_qty

    @property
    def quality_ok(self) -> bool:
        return self.variety_ok and self.offered_quality >= self.buyer.min_quality

    @property
    def price_ok(self) -> bool:
        return self.farmer.reservation <= self.buyer.max_price

    @property
    def viable(self) -> bool:
        """Is there any (quantity, price) both parties would rationally take?"""
        return (self.variety_ok and self.color_ok and self.stock_ok
                and self.quality_ok and self.price_ok)

    @property
    def deal_variety(self) -> int:
        return self.buyer.want_variety

    @property
    def deal_qty(self) -> int:
        return self.buyer.need_qty

    @property
    def zopa(self) -> tuple[int, int]:
        """Zone of possible agreement on the price bin (inclusive)."""
        return (self.farmer.reservation, self.buyer.max_price)

    def price_in_zopa(self, price_bin: int) -> bool:
        lo, hi = self.zopa
        return lo <= price_bin <= hi

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day, "held_out": self.held_out, "viable": self.viable,
            "farmer_stocks": list(self.farmer.stocks),
            "farmer_qualities": list(self.farmer.qualities),
            "farmer_reservation": self.farmer.reservation,
            "buyer_want_variety": self.buyer.want_variety,
            "buyer_want_color": self.buyer.want_color,
            "buyer_need_qty": self.buyer.need_qty,
            "buyer_min_quality": self.buyer.min_quality,
            "buyer_max_price": self.buyer.max_price,
        }


# --------------------------------------------------------------------------
class World:
    """Samples scenarios and owns the train/held-out split.

    Every farmer field is drawn independently of every buyer field.  That is the
    property the whole experiment rests on: no agent can do better than the
    marginal base rate on anything the other side privately holds, so any
    above-chance performance has to have come through the channel.
    """

    def __init__(self, cfg: WorldConfig, rng: random.Random | None = None,
                 holdout: "ComboHoldout | None" = None):
        self.cfg = cfg
        self.rng = rng or random.Random(0)
        self.day = 0
        self._mass_cache = None
        self.holdout = holdout or ComboHoldout(cfg, cfg.holdout_combo_frac, cfg.holdout_seed)

    # ------------------------------------------------------------------
    def is_held_out(self, variety: int, color: int, quality: int) -> bool:
        return (variety, color, quality) in self.holdout

    # ------------------------------------------------------------------
    # Marginals.  These raise how often a deal is possible, and make some
    # meanings far commoner than others, without creating *any* dependence
    # between the two sides -- each field is still drawn from a fixed
    # distribution that the other agent's state does not touch.
    def _skew_high(self, lo: int, hi: int) -> int:
        return max(self.rng.randint(lo, hi), self.rng.randint(lo, hi))

    def _skew_low(self, lo: int, hi: int) -> int:
        return min(self.rng.randint(lo, hi), self.rng.randint(lo, hi))

    def _zipf_weights(self, n: int, alpha: float) -> list[float]:
        """Zipf-like weights over ``n`` ranked items, normalised (addendum 2.2).

        A uniform world gives the message-length cost nothing to bite on: if no
        meaning is commoner than any other, no meaning is worth a shorter word.
        Real vocabularies are short where they are used most, so the world has to
        have a "most" in the first place.
        """
        if alpha <= 0:
            return [1.0 / n] * n
        w = [1.0 / ((i + 1) ** alpha) for i in range(n)]
        tot = sum(w)
        return [x / tot for x in w]

    def _zipf_draw(self, n: int, alpha: float, offset: int = 0) -> int:
        """Draw a rank 0..n-1 from the Zipf weights, then shift by ``offset``."""
        w = self._zipf_weights(n, alpha)
        x = self.rng.random()
        acc = 0.0
        for i, p in enumerate(w):
            acc += p
            if x <= acc:
                return i + offset
        return n - 1 + offset

    def variety_probs(self) -> list[float]:
        return self._zipf_weights(self.cfg.n_varieties, self.cfg.zipf_alpha_variety)

    def need_ceiling(self) -> int:
        """The largest order a shopper ever places."""
        return max(1, int(round(self.cfg.need_max_frac * self.cfg.max_qty)))

    def qty_probs(self) -> list[float]:
        """P(need_qty = q) for q = 1..max_qty (zero above the order ceiling)."""
        w = self._zipf_weights(self.need_ceiling(), self.cfg.zipf_alpha)
        return w + [0.0] * (self.cfg.max_qty - len(w))

    def sample_farmer(self) -> FarmerState:
        c, r = self.cfg, self.rng
        stocks, quals = [], []
        floor = max(1, int(round(c.stock_floor_frac * c.max_qty)))
        for v in range(c.n_varieties):
            for col in range(c.n_colors):
                if r.random() >= c.p_stocked:
                    stocks.append(0)
                    quals.append(0)
                    continue
                # A lot is itself a (fruit, colour, quality) combination, so the
                # held-out ones must not sit in the barn either.
                q = None
                for _ in range(32):
                    cand = self._skew_high(0, c.n_quality - 1)
                    for _ in range(c.quality_bias):
                        cand = max(cand, self._skew_high(0, c.n_quality - 1))
                    if not self.is_held_out(v, col, cand):
                        q = cand
                        break
                if q is None:            # every quality of this lot is reserved
                    stocks.append(0)
                    quals.append(0)
                    continue
                stocks.append(self._skew_high(floor, c.max_qty))
                quals.append(q)
        return FarmerState(stocks=tuple(stocks), qualities=tuple(quals),
                           n_colors=c.n_colors,
                           reservation=self._skew_low(0, c.reservation_max_bin))

    def _shopper_quality(self) -> int:
        c = self.cfg
        q = self._skew_low(0, c.n_quality - 1)
        for _ in range(c.quality_bias):
            q = min(q, self._skew_low(0, c.n_quality - 1))
        return q

    def sample_buyer(self) -> BuyerState:
        """The buyer's request carries the world's frequency skew.

        Requests are what the language has to name, so this is where Zipf lives.
        By default the skew is on *quantity* only -- see the note on
        ``WorldConfig.zipf_alpha_variety`` for the measurement behind that.  The
        farmer's barn stays broadly distributed: skewing it too would only make
        deals rarer without adding anything for word length to track.
        """
        c, r = self.cfg, self.rng
        want = self._zipf_draw(c.n_varieties, c.zipf_alpha_variety)
        col, q = r.randrange(c.n_colors), self._shopper_quality()
        for _ in range(32):          # never shop for a held-out combination
            if not self.is_held_out(want, col, q):
                break
            col, q = r.randrange(c.n_colors), self._shopper_quality()
        return BuyerState(
            want_variety=want,
            want_color=col,
            need_qty=self._zipf_draw(self.need_ceiling(), c.zipf_alpha, offset=1),
            min_quality=q,
            max_price=self._skew_high(c.budget_min_bin, c.n_price_bins - 1),
        )

    # ------------------------------------------------------------------
    # Meaning identity and frequency (addendum 2.2 / 2.4 analyses)
    # ------------------------------------------------------------------
    def meaning_key(self, buyer: BuyerState) -> tuple[int, int]:
        """The coarse "concept" a request expresses: which fruit, how many.

        Quantity is used raw rather than bucketed: it is the dimension the
        frequency skew acts on most strongly, and bucketing would hide exactly
        the rare-meaning behaviour the addendum asks us to look for. Colour is
        uniform, so it scales every meaning equally and is left out here.
        """
        return (buyer.want_variety, buyer.need_qty)

    def meaning_prob(self, key: tuple[int, int]) -> float:
        """How often this meaning comes up, analytically."""
        v, q = key
        pv = self.variety_probs()[v]
        pq = self.qty_probs()[q - 1]
        return pv * pq

    def meaning_table(self) -> list[tuple[tuple[int, int], float]]:
        """Every meaning with its probability, commonest first."""
        out = [((v, q), self.meaning_prob((v, q)))
               for v in range(self.cfg.n_varieties)
               for q in range(1, self.cfg.max_qty + 1)]
        out = [(k, p) for k, p in out if p > 0]
        out.sort(key=lambda kv: -kv[1])
        return out

    # ------------------------------------------------------------------
    def sample(self, *, held_out: bool | None = False, max_tries: int = 200) -> Scenario:
        """Draw one encounter.

        ``held_out=False`` (training) rejects scenarios touching a reserved
        (variety, quantity) combination; ``True`` requires one; ``None`` accepts
        either.
        """
        c, r = self.cfg, self.rng
        for _ in range(max_tries):
            buyer = self.sample_buyer()
            if held_out is True:
                # A held-out request, for evaluation only: sample_buyer refuses
                # to make one, so take a held-out combination directly.
                combos = sorted(self.holdout.held)
                if not combos:
                    break
                f, col, q = combos[r.randrange(len(combos))]
                buyer = BuyerState(want_variety=f, want_color=col,
                                   need_qty=buyer.need_qty, min_quality=q,
                                   max_price=buyer.max_price)
            ho = self.is_held_out(buyer.want_variety, buyer.want_color, buyer.min_quality)
            if held_out is False and ho:
                continue
            return Scenario(farmer=self.sample_farmer(), buyer=buyer, day=self.day,
                            held_out=ho)
        return Scenario(farmer=self.sample_farmer(), buyer=self.sample_buyer(),
                        day=self.day, held_out=False)

    def sample_batch(self, n: int, **kw) -> list[Scenario]:
        return [self.sample(**kw) for _ in range(n)]

    def advance_day(self) -> None:
        self.day += 1

    # ------------------------------------------------------------------
    def describe_farmer(self, f: FarmerState) -> str:
        c = self.cfg
        parts = []
        for v in range(c.n_varieties):
            for col in range(c.n_colors):
                if f.stock_of(v, col) > 0:
                    parts.append("%s %s x%d (%s)" % (
                        c.color_names[col], c.variety_names[v], f.stock_of(v, col),
                        c.quality_names[f.quality_of(v, col)]))
        barn = ", ".join(parts) if parts else "nothing"
        return "%s; will not sell below %.2f" % (barn, c.price_values[f.reservation])

    def describe_buyer(self, b: BuyerState) -> str:
        c = self.cfg
        return "wants %s %s x%d, quality >= %s, cannot pay above %.2f" % (
            c.color_names[b.want_color], c.variety_names[b.want_variety], b.need_qty,
            c.quality_names[b.min_quality], c.price_values[b.max_price])

