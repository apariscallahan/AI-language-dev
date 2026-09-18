"""The apple world: hidden state, scenario sampling, and what makes a deal viable.

Spec section 1.2 is the load-bearing part of this file.  The *only* reason
language is necessary in this simulation is that the Farmer and the Buyer each
hold private facts the other needs and cannot observe.  This module defines that
asymmetry; :mod:`orchard.env` enforces it.

  Farmer knows : how much of **each variety** is in the barn, the quality of
                 each, and the lowest price they will accept
  Buyer knows  : which single variety they want, how many, the minimum quality
                 they will take, and the most they can pay

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
K_EMPTY, K_VARIETY, K_QTY, K_QUALITY, K_PRICE = 0, 1, 2, 3, 4
KIND_NAMES = {K_EMPTY: "empty", K_VARIETY: "variety", K_QTY: "quantity",
              K_QUALITY: "quality", K_PRICE: "price"}


@dataclass(frozen=True)
class FarmerState:
    """Private to the Farmer: the whole barn."""
    stocks: tuple[int, ...]        # per variety, 0 means "not stocking that one"
    qualities: tuple[int, ...]     # per variety (meaningless where stock is 0)
    reservation: int               # price bin: the lowest per-apple price they take

    def stock_of(self, variety: int) -> int:
        return self.stocks[variety]

    def quality_of(self, variety: int) -> int:
        return self.qualities[variety]

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.stocks) + tuple(self.qualities) + (self.reservation,)


@dataclass(frozen=True)
class BuyerState:
    """Private to the Buyer: one shopping list."""
    want_variety: int
    need_qty: int
    min_quality: int
    max_price: int                 # price bin: the most they will pay per apple

    def as_tuple(self) -> tuple[int, ...]:
        return (self.want_variety, self.need_qty, self.min_quality, self.max_price)


# --------------------------------------------------------------------------
# Observation schema
# --------------------------------------------------------------------------
def farmer_schema(cfg: WorldConfig) -> list[int]:
    return [K_QTY] * cfg.n_varieties + [K_QUALITY] * cfg.n_varieties + [K_PRICE]


def buyer_schema(cfg: WorldConfig) -> list[int]:
    return [K_VARIETY, K_QTY, K_QUALITY, K_PRICE]


def n_obs_slots(cfg: WorldConfig) -> int:
    """Both roles use one sequence layout, so the shorter one is padded."""
    return max(len(farmer_schema(cfg)), len(buyer_schema(cfg)))


def obs_schema(cfg: WorldConfig, role: int) -> list[int]:
    """Field kinds for each observation slot, padded with K_EMPTY."""
    from .env import FARMER
    base = farmer_schema(cfg) if role == FARMER else buyer_schema(cfg)
    return base + [K_EMPTY] * (n_obs_slots(cfg) - len(base))


def field_labels(cfg: WorldConfig, role: int) -> list[str]:
    """Human names for the slots, used by metrics and the report."""
    from .env import FARMER
    if role == FARMER:
        names = (["stock_" + v for v in cfg.variety_names]
                 + ["quality_" + v for v in cfg.variety_names]
                 + ["reservation"])
    else:
        names = ["want_variety", "need_qty", "min_quality", "max_price"]
    return names + ["-"] * (n_obs_slots(cfg) - len(names))


def field_spans(cfg: WorldConfig, role: int) -> list[int]:
    """Value range per slot, so distances can be normalised per field."""
    spans = {K_EMPTY: 1, K_VARIETY: max(cfg.n_varieties - 1, 1),
             K_QTY: max(cfg.max_qty, 1), K_QUALITY: max(cfg.n_quality - 1, 1),
             K_PRICE: max(cfg.n_price_bins - 1, 1)}
    return [spans[k] for k in obs_schema(cfg, role)]


# --------------------------------------------------------------------------
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
        return self.farmer.stock_of(self.buyer.want_variety)

    @property
    def offered_quality(self) -> int:
        return self.farmer.quality_of(self.buyer.want_variety)

    @property
    def variety_ok(self) -> bool:
        return self.offered_stock > 0

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
        return self.variety_ok and self.stock_ok and self.quality_ok and self.price_ok

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

    def __init__(self, cfg: WorldConfig, rng: random.Random | None = None):
        self.cfg = cfg
        self.rng = rng or random.Random(0)
        self.day = 0
        self._mass_cache = None
        self.holdout = self._build_holdout()

    # ------------------------------------------------------------------
    def _build_holdout(self) -> set[tuple[int, int]]:
        """Reserve (variety, quantity) requests for zero-shot testing.

        Balanced by construction: every variety loses the same *number* of
        quantities.  A random subset would leave the training distribution over
        varieties slightly non-uniform, and a farmer could then score above the
        base rate by always naming the variety that happens to be asked for most
        -- the same kind of free reward the sampler rewrite was meant to remove.
        """
        c = self.cfg
        per_variety = int(round(c.holdout_frac * c.max_qty))
        if per_variety <= 0 or per_variety >= c.max_qty:
            return set()
        rng = random.Random(c.holdout_seed)
        offsets = rng.sample(range(c.max_qty), per_variety)
        out: set[tuple[int, int]] = set()
        for v in range(c.n_varieties):
            for i, off in enumerate(offsets):
                # rotate per variety so no single quantity is held out everywhere
                out.add((v, ((off + v * per_variety) % c.max_qty) + 1))
        return out

    def is_held_out(self, variety: int, qty: int) -> bool:
        return (variety, qty) in self.holdout

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
        for _ in range(c.n_varieties):
            if r.random() < c.p_stocked:
                stocks.append(self._skew_high(floor, c.max_qty))
                q = self._skew_high(0, c.n_quality - 1)
                for _ in range(c.quality_bias):
                    q = max(q, self._skew_high(0, c.n_quality - 1))
                quals.append(q)
            else:
                stocks.append(0)
                quals.append(0)
        return FarmerState(stocks=tuple(stocks), qualities=tuple(quals),
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
        return BuyerState(
            want_variety=self._zipf_draw(c.n_varieties, c.zipf_alpha_variety),
            need_qty=self._zipf_draw(self.need_ceiling(), c.zipf_alpha, offset=1),
            min_quality=self._shopper_quality(),
            max_price=self._skew_high(c.budget_min_bin, c.n_price_bins - 1),
        )

    # ------------------------------------------------------------------
    # Meaning identity and frequency (addendum 2.2 / 2.4 analyses)
    # ------------------------------------------------------------------
    def meaning_key(self, buyer: BuyerState) -> tuple[int, int]:
        """The coarse "concept" a request expresses: which variety, how many.

        Quantity is used raw rather than bucketed: it is the dimension the
        frequency skew acts on most strongly, and bucketing would hide exactly
        the rare-meaning behaviour the addendum asks us to look for.
        """
        return (buyer.want_variety, buyer.need_qty)

    def meaning_prob(self, key: tuple[int, int]) -> float:
        """How often this meaning comes up, analytically, excluding held-out ones."""
        v, q = key
        if self.is_held_out(v, q):
            return 0.0
        pv = self.variety_probs()[v]
        pq = self.qty_probs()[q - 1]
        raw = pv * pq
        return raw / max(self._live_mass(), 1e-12)

    def _live_mass(self) -> float:
        if getattr(self, "_mass_cache", None) is not None:
            return self._mass_cache
        vp, qp = self.variety_probs(), self.qty_probs()
        total = sum(vp[v] * qp[q - 1]
                    for v in range(self.cfg.n_varieties)
                    for q in range(1, self.cfg.max_qty + 1)
                    if not self.is_held_out(v, q))
        self._mass_cache = total
        return total

    def meaning_table(self) -> list[tuple[tuple[int, int], float]]:
        """Every trainable meaning with its probability, commonest first."""
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
        farmer = buyer = None
        for _ in range(max_tries):
            buyer = self.sample_buyer()
            # The held-out predicate looks only at the *request*.  Making it depend
            # on the farmer too would couple the two sides through the rejection
            # step, which is exactly the independence this sampler exists to keep.
            ho = self.is_held_out(buyer.want_variety, buyer.need_qty)
            if held_out is True and not ho:
                continue
            if held_out is False and ho:
                continue
            farmer = self.sample_farmer()
            return Scenario(farmer=farmer, buyer=buyer, day=self.day, held_out=ho)
        return Scenario(farmer=self.sample_farmer(), buyer=buyer, day=self.day,
                        held_out=False)

    def sample_batch(self, n: int, **kw) -> list[Scenario]:
        return [self.sample(**kw) for _ in range(n)]

    def advance_day(self) -> None:
        self.day += 1

    # ------------------------------------------------------------------
    def describe_farmer(self, f: FarmerState) -> str:
        c = self.cfg
        parts = []
        for v in range(c.n_varieties):
            if f.stocks[v] > 0:
                parts.append("%s x%d (%s)" % (c.variety_names[v], f.stocks[v],
                                              c.quality_names[f.qualities[v]]))
        barn = ", ".join(parts) if parts else "nothing"
        return "%s; will not sell below %.2f" % (barn, c.price_values[f.reservation])

    def describe_buyer(self, b: BuyerState) -> str:
        c = self.cfg
        return "wants %s x%d, quality >= %s, cannot pay above %.2f" % (
            c.variety_names[b.want_variety], b.need_qty,
            c.quality_names[b.min_quality], c.price_values[b.max_price])

