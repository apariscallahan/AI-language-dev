"""Tensorised world sampling and trade resolution, for running on a GPU.

The scalar versions in :mod:`orchard.world` and :mod:`orchard.env` are the
reference implementation and stay the readable definition of the rules.  They are
also, on a GPU, the whole bottleneck: they do one Python call and one dataclass
construction *per episode*, so a batch of 4096 costs 4096 interpreter round trips
before a single kernel launches.  Everything here is the same arithmetic done once
over the whole batch, on whatever device the run is using.

``tests/test_batched.py`` asserts the two agree exactly -- same rewards, same
success flags, same failure counts -- on random batches.  If they ever diverge,
the scalar version is right and this one is wrong.

Nothing here changes the rules.  It is the same world and the same reward, moved
off the interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .config import Config
from .env import BUYER, FARMER
from .world import BuyerState, FarmerState, Scenario


# ==========================================================================
# scenarios as tensors
# ==========================================================================
@dataclass
class ScenarioBatch:
    """A whole market day's hidden state, as tensors on one device."""
    stocks: torch.Tensor          # (B, V*C), fruit-major cells
    qualities: torch.Tensor       # (B, V*C)
    reservation: torch.Tensor     # (B,)
    want_variety: torch.Tensor    # (B,)
    want_color: torch.Tensor      # (B,)
    need_qty: torch.Tensor        # (B,)
    min_quality: torch.Tensor     # (B,)
    max_price: torch.Tensor       # (B,)
    held_out: torch.Tensor        # (B,) bool
    n_colors: int = 1
    day: int = 0

    # ---- derived, computed once ---------------------------------------
    def __post_init__(self) -> None:
        n_colors = self.n_colors
        idx = (self.want_variety * n_colors + self.want_color).unsqueeze(1)
        self.offered_stock = self.stocks.gather(1, idx).squeeze(1)
        self.offered_quality = self.qualities.gather(1, idx).squeeze(1)
        self.offered_color = self.want_color
        self.variety_ok = self.offered_stock > 0
        self.color_ok = self.variety_ok
        self.stock_ok = self.offered_stock >= self.need_qty
        self.quality_ok = self.variety_ok & (self.offered_quality >= self.min_quality)
        self.price_ok = self.reservation <= self.max_price
        self.viable = (self.variety_ok & self.color_ok & self.stock_ok
                       & self.quality_ok & self.price_ok)

    def __len__(self) -> int:
        return int(self.want_variety.shape[0])

    @property
    def device(self) -> torch.device:
        return self.want_variety.device

    def scenario(self, i: int, day: int | None = None) -> Scenario:
        """One episode as a plain :class:`Scenario`, for logging and rendering.

        Built only for the episodes actually written to the ledger, which is why
        the ledger's stride is worth setting on a long run.
        """
        return Scenario(
            farmer=FarmerState(
                stocks=tuple(int(x) for x in self.stocks[i]),
                qualities=tuple(int(x) for x in self.qualities[i]),
                n_colors=self.n_colors,
                reservation=int(self.reservation[i])),
            buyer=BuyerState(
                want_variety=int(self.want_variety[i]),
                want_color=int(self.want_color[i]),
                need_qty=int(self.need_qty[i]),
                min_quality=int(self.min_quality[i]),
                max_price=int(self.max_price[i])),
            day=self.day if day is None else day,
            held_out=bool(self.held_out[i]))

    def obs(self, cfg: Config, role: int) -> torch.Tensor:
        """The role's observation rows, padded to the shared slot layout."""
        from .world import n_obs_slots
        n = n_obs_slots(cfg.world)
        B = len(self)
        if role == FARMER:
            parts = [self.stocks, self.qualities, self.reservation.unsqueeze(1)]
        else:
            parts = [self.want_variety.unsqueeze(1), self.want_color.unsqueeze(1),
                     self.need_qty.unsqueeze(1), self.min_quality.unsqueeze(1),
                     self.max_price.unsqueeze(1)]
        x = torch.cat(parts, dim=1)
        if x.shape[1] < n:
            x = torch.cat([x, torch.zeros((B, n - x.shape[1]), dtype=torch.long,
                                          device=x.device)], dim=1)
        return x


class TensorWorld:
    """Samples whole batches of scenarios directly on the device.

    Same distributions as :class:`orchard.world.World`, drawn with torch instead
    of the ``random`` module so that a batch is a handful of kernels rather than
    thousands of interpreter steps.  Both sides are still drawn independently of
    each other -- that property is the point of the world design and is asserted
    in the tests.
    """

    def __init__(self, cfg: Config, device: str = "cpu",
                 generator: Optional[torch.Generator] = None, holdout=None):
        self.cfg = cfg
        w = cfg.world
        self.device = torch.device(device)
        self.gen = generator
        self.day = 0

        from .world import ComboHoldout, World
        holdout = holdout or ComboHoldout(w, w.holdout_combo_frac, w.holdout_seed)
        ref = World(w, holdout=holdout)      # reuse the Zipf definitions
        self.ref = ref
        self.holdout = holdout
        self.need_ceiling = ref.need_ceiling()

        self._variety_cdf = self._cdf(ref.variety_probs())
        self._qty_cdf = self._cdf(ref._zipf_weights(self.need_ceiling, w.zipf_alpha))

        # (fruit, colour, quality) -> is this combination never trained on?
        held = torch.zeros((w.n_varieties, w.n_colors, w.n_quality), dtype=torch.bool)
        for (f, c_, q) in holdout.held:
            held[f, c_, q] = True
        self.combo_held = held.to(self.device)
        self._held_combos = holdout.tensor(self.device)

    # ------------------------------------------------------------------
    def _cdf(self, probs) -> torch.Tensor:
        t = torch.tensor(probs, dtype=torch.float32, device=self.device)
        return torch.cumsum(t / t.sum(), dim=0)

    def _categorical(self, cdf: torch.Tensor, n: int) -> torch.Tensor:
        u = torch.rand(n, device=self.device, generator=self.gen)
        return torch.searchsorted(cdf, u.contiguous()).clamp_(max=cdf.numel() - 1)

    def _randint(self, lo: int, hi: int, n: int) -> torch.Tensor:
        """Uniform integers in [lo, hi] inclusive, matching random.randint."""
        if hi <= lo:
            return torch.full((n,), lo, dtype=torch.long, device=self.device)
        return torch.randint(lo, hi + 1, (n,), device=self.device,
                             generator=self.gen, dtype=torch.long)

    def _skew_high(self, lo: int, hi: int, n: int) -> torch.Tensor:
        return torch.maximum(self._randint(lo, hi, n), self._randint(lo, hi, n))

    def _skew_low(self, lo: int, hi: int, n: int) -> torch.Tensor:
        return torch.minimum(self._randint(lo, hi, n), self._randint(lo, hi, n))

    # ------------------------------------------------------------------
    def sample(self, n: int, *, held_out: bool = False) -> ScenarioBatch:
        w = self.cfg.world
        V = w.n_varieties

        C = w.n_colors
        cells = V * C
        stocked = torch.rand((n, cells), device=self.device,
                             generator=self.gen) < w.p_stocked
        floor = max(1, int(round(w.stock_floor_frac * w.max_qty)))
        stocks = torch.maximum(
            torch.randint(floor, w.max_qty + 1, (n, cells), device=self.device,
                          generator=self.gen),
            torch.randint(floor, w.max_qty + 1, (n, cells), device=self.device,
                          generator=self.gen))
        def draw_quality() -> torch.Tensor:
            """The barn's quality draw, skewed high exactly as the scalar world's."""
            q = torch.maximum(
                torch.randint(0, w.n_quality, (n, cells), device=self.device,
                              generator=self.gen),
                torch.randint(0, w.n_quality, (n, cells), device=self.device,
                              generator=self.gen))
            for _ in range(w.quality_bias):
                q = torch.maximum(q, torch.maximum(
                    torch.randint(0, w.n_quality, (n, cells), device=self.device,
                                  generator=self.gen),
                    torch.randint(0, w.n_quality, (n, cells), device=self.device,
                                  generator=self.gen)))
            return q

        quals = draw_quality()
        # A lot is a (fruit, colour, quality) combination too, so held-out ones
        # must not sit in the barn: redraw the quality, and empty the lot if every
        # quality of it is reserved.
        f_ix = (torch.arange(cells, device=self.device) // C).unsqueeze(0).expand(n, cells)
        c_ix = (torch.arange(cells, device=self.device) % C).unsqueeze(0).expand(n, cells)
        for _ in range(32):
            bad = self.combo_held[f_ix, c_ix, quals] & stocked
            if not bool(bad.any()):
                break
            quals = torch.where(bad, draw_quality(), quals)
        stocked = stocked & ~self.combo_held[f_ix, c_ix, quals]
        stocks = torch.where(stocked, stocks, torch.zeros_like(stocks))
        quals = torch.where(stocked, quals, torch.zeros_like(quals))
        reservation = self._skew_low(0, w.reservation_max_bin, n)

        want = self._categorical(self._variety_cdf, n)
        need = self._categorical(self._qty_cdf, n) + 1
        want_c = torch.randint(0, w.n_colors, (n,), device=self.device, generator=self.gen)
        min_q = self._skew_low(0, w.n_quality - 1, n)
        for _ in range(w.quality_bias):
            min_q = torch.minimum(min_q, self._skew_low(0, w.n_quality - 1, n))
        max_p = self._skew_high(w.budget_min_bin, w.n_price_bins - 1, n)

        if held_out:
            if self._held_combos is None or not len(self._held_combos):
                raise ValueError("no reserved combinations exist; holdout_combo_frac is 0")
            pick = self._held_combos[torch.randint(0, self._held_combos.shape[0], (n,),
                                                   device=self.device, generator=self.gen)]
            want, want_c, min_q = pick[:, 0], pick[:, 1], pick[:, 2]
        else:
            # Reserved requests are re-drawn rather than filtered out, so the batch
            # stays exactly the size asked for.
            for _ in range(32):
                bad = self.combo_held[want, want_c, min_q]
                if not bool(bad.any()):
                    break
                want = torch.where(bad, self._categorical(self._variety_cdf, n), want)
                want_c = torch.where(bad, torch.randint(0, w.n_colors, (n,), device=self.device,
                                                        generator=self.gen), want_c)
                min_q = torch.where(bad, self._skew_low(0, w.n_quality - 1, n), min_q)

        return ScenarioBatch(stocks=stocks, qualities=quals,
                             reservation=reservation,
                             want_variety=want, want_color=want_c, need_qty=need,
                             min_quality=min_q,
                             max_price=max_p,
                             held_out=self.combo_held[want, want_c, min_q],
                             n_colors=C, day=self.day)


# ==========================================================================
# reward, over a whole batch
# ==========================================================================
def _decode_hits(cfg: Config, sb: ScenarioBatch, bel: torch.Tensor,
                 role: int) -> torch.Tensor:
    """(B, n_fields) bool -- the tensor form of :func:`orchard.env.decode_hits`."""
    R = cfg.reward
    v, q, k, p = bel[:, 0], bel[:, 1], bel[:, 2], bel[:, 3]
    col = bel[:, 4] if bel.shape[1] > 4 else None
    if role == FARMER:
        hits = [
            v == sb.want_variety,
            (q - sb.need_qty).abs() <= R.belief_qty_tol,
            k == sb.min_quality,
            (p - sb.max_price).abs() <= R.belief_price_tol,
        ]
        if col is not None:
            hits.append(col == sb.want_color)
        return torch.stack(hits, dim=1)
    hits = [
        (q - sb.offered_stock).abs() <= R.belief_qty_tol,
        k == sb.offered_quality,
        (p - sb.reservation).abs() <= R.belief_price_tol,
    ]
    if col is not None:
        hits.append(col == sb.offered_color)
    return torch.stack(hits, dim=1)


def _correctness(cfg: Config, sb: ScenarioBatch, dec: torch.Tensor) -> torch.Tensor:
    """(B, 3) bool -- the tensor form of ``_agent_correctness``."""
    R = cfg.reward
    variety, qty, price = dec[:, 1], dec[:, 2], dec[:, 3]
    ok_v = sb.variety_ok & (variety == sb.want_variety)
    ok_q = (qty == sb.need_qty) & (qty <= sb.offered_stock) & (qty >= 1)
    ok_p = (price >= sb.reservation) & (price <= sb.max_price)
    return torch.stack([ok_v, ok_q, ok_p], dim=1)


def resolve_batch(cfg: Config, sb: ScenarioBatch, f_dec: torch.Tensor,
                  b_dec: torch.Tensor, f_cost: torch.Tensor,
                  b_cost: torch.Tensor, *, f_bel: Optional[torch.Tensor] = None,
                  b_bel: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
    """Score a whole batch of negotiations.  Mirrors :func:`orchard.env.resolve`."""
    R = cfg.reward
    w = cfg.world
    dev = sb.device
    prices = torch.tensor(w.price_values, dtype=torch.float32, device=dev)
    span = max(float(prices[-1] - prices[0]), 1e-9)

    f_acc = f_dec[:, 0].bool()
    b_acc = b_dec[:, 0].bool()
    both = f_acc & b_acc

    agree_v = f_dec[:, 1] == b_dec[:, 1]
    agree_q = (f_dec[:, 2] - b_dec[:, 2]).abs() <= R.qty_tol
    agree_p = (f_dec[:, 3] - b_dec[:, 3]).abs() <= R.price_tol
    mutual = agree_v & agree_q & agree_p
    n_agree = (agree_v.long() + agree_q.long() + agree_p.long()).float()

    f_corr = _correctness(cfg, sb, f_dec)
    b_corr = _correctness(cfg, sb, b_dec)

    agreed_v = f_dec[:, 1]
    agreed_q = torch.div(f_dec[:, 2] + b_dec[:, 2], 2, rounding_mode="floor")
    agreed_p = torch.div(f_dec[:, 3] + b_dec[:, 3], 2, rounding_mode="floor")

    executable = (
        sb.variety_ok
        & sb.color_ok
        & (agreed_v == sb.want_variety)
        & (agreed_q >= 1) & (agreed_q <= sb.offered_stock)
        & ((agreed_q - sb.need_qty).abs() <= R.qty_tol)
        & sb.quality_ok
        & (agreed_p >= sb.reservation) & (agreed_p <= sb.max_price))
    success = both & mutual & executable

    fr = torch.zeros(len(sb), device=dev)
    br = torch.zeros(len(sb), device=dev)

    agree_r = R.agree_per_dim * n_agree
    fr += agree_r + R.correct_per_dim * f_corr.sum(1).float()
    br += agree_r + R.correct_per_dim * b_corr.sum(1).float()
    fr += R.judgement * (f_acc == sb.viable).float()
    br += R.judgement * (b_acc == sb.viable).float()

    if f_bel is not None and b_bel is not None:
        f_hits = _decode_hits(cfg, sb, f_bel, FARMER)
        b_hits = _decode_hits(cfg, sb, b_bel, BUYER)
        f_decode = f_hits.float().mean(1)
        b_decode = b_hits.float().mean(1)
        fr += R.decode * f_decode + R.understood * b_decode
        br += R.decode * b_decode + R.understood * f_decode
    else:
        f_decode = torch.zeros(len(sb), device=dev)
        b_decode = torch.zeros(len(sb), device=dev)

    pv = prices[agreed_p.clamp(0, w.n_price_bins - 1)]
    margin = (pv - prices[sb.reservation]) / span
    surplus = (prices[sb.max_price] - pv) / span
    succ_f = success.float()
    fr += succ_f * (R.success + R.economics * margin)
    br += succ_f * (R.success + R.economics * surplus)

    bad_deal = both & (~success) & (~sb.viable)
    fr += R.bad_deal * bad_deal.float()
    br += R.bad_deal * bad_deal.float()

    one_sided = f_acc != b_acc
    fr += R.one_sided_accept * one_sided.float()
    br += R.one_sided_accept * one_sided.float()

    both_reject = (~f_acc) & (~b_acc)
    missed = both_reject & sb.viable
    correct_no = both_reject & (~sb.viable)
    fr += R.missed_deal * missed.float() + R.correct_no_deal * correct_no.float()
    br += R.missed_deal * missed.float() + R.correct_no_deal * correct_no.float()

    fr -= f_cost.float()
    br -= b_cost.float()

    traded_qty = torch.where(success, agreed_q, torch.zeros_like(agreed_q))
    trade_value = torch.where(success, pv * traded_qty.float(),
                              torch.zeros_like(pv))
    return {
        "farmer_reward": fr, "buyer_reward": br,
        "success": success, "both_accept": both,
        "comprehended": mutual & executable,
        "both_judged": (f_acc == sb.viable) & (b_acc == sb.viable),
        "farmer_decode": f_decode, "buyer_decode": b_decode,
        "agree_variety": agree_v, "agree_qty": agree_q, "agree_price": agree_p,
        "farmer_correct": f_corr, "buyer_correct": b_corr,
        "agreed_variety": agreed_v, "agreed_qty": agreed_q, "agreed_price": agreed_p,
        "traded_qty": traded_qty, "trade_value": trade_value,
        "farmer_profit": torch.where(
            success, (pv - prices[sb.reservation]) * traded_qty.float(),
            torch.zeros_like(pv)),
        "buyer_savings": torch.where(
            success, (prices[sb.max_price] - pv) * traded_qty.float(),
            torch.zeros_like(pv)),
        "correct_no_deal": correct_no,
        "missed_deal": missed, "one_sided": one_sided, "bad_deal": bad_deal,
    }


def failure_modes(res: dict[str, torch.Tensor]) -> list[str]:
    """Primary outcome label per episode, matching the scalar classifier's order."""
    n = res["success"].shape[0]
    out = ["unclassified"] * n
    succ = res["success"].tolist()
    cnd = res["correct_no_deal"].tolist()
    one = res["one_sided"].tolist()
    missed = res["missed_deal"].tolist()
    av = res["agree_variety"].tolist()
    aq = res["agree_qty"].tolist()
    ap = res["agree_price"].tolist()
    both = res["both_accept"].tolist()
    for i in range(n):
        if succ[i]:
            out[i] = "success"
        elif cnd[i]:
            out[i] = "correct_no_deal"
        elif one[i]:
            out[i] = "one_sided_accept"
        elif missed[i]:
            out[i] = "missed_deal"
        elif both[i] and not av[i]:
            out[i] = "variety_mismatch"
        elif both[i] and not aq[i]:
            out[i] = "qty_mismatch"
        elif both[i] and not ap[i]:
            out[i] = "price_mismatch"
        else:
            out[i] = "infeasible_deal"
    return out
