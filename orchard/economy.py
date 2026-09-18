"""The economy loop: market days, seasons, inventories and replenishment (spec 1.4).

A *farm* is a lineage slot, not an agent: the orchard outlives whoever is running
it, so an inventory belongs to the slot and survives the farmer's death.

  * Each farm holds an **inventory**: a quantity and quality per variety, plus a
    cost price.  All of it is private to whoever farms it.
  * A **day** is a block of encounters at the market.  Every encounter that day
    sees the same start-of-day inventory, because they happen at the same market.
  * Sales are settled at the end of the batch and deplete the variety that sold;
    a farm with nothing left restocks immediately.
  * Every **season** (a few days) every farm replenishes with a fresh random
    inventory.

Buyers draw a fresh private need for every encounter, independently of the farm
they are about to meet.  That independence is deliberate and load-bearing --
see the note at the top of :mod:`orchard.world`.

Setting ``persistent_inventory = False`` falls back to sampling both sides fresh
per episode, which is the simpler regime used for the early build-order steps.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .config import Config
from .world import BuyerState, FarmerState, Scenario, World


@dataclass
class Inventory:
    state: FarmerState
    drawn_day: int
    remaining: list[int]
    sold_total: int = 0

    @property
    def empty(self) -> bool:
        return sum(self.remaining) <= 0


class Economy:
    """Generates encounters and keeps the market's books."""

    def __init__(self, cfg: Config, world: World, rng: random.Random, n_farms: int):
        self.cfg = cfg
        self.world = world
        self.rng = rng
        self.n_farms = n_farms
        self.day = -1
        self.season = 0
        self.restocks = 0
        self.soldouts = 0
        self.inventories: list[Inventory] = [self._draw(f) for f in range(n_farms)]

    # ------------------------------------------------------------------
    def _draw(self, farm: int) -> Inventory:
        """Fresh random variety/quantity/quality, avoiding held-out combinations."""
        saved = self.world.rng
        self.world.rng = self.rng
        try:
            st = self.world.sample_farmer()
        finally:
            self.world.rng = saved
        self.restocks += 1
        return Inventory(state=st, drawn_day=max(self.day, 0),
                         remaining=list(st.stocks))

    def _visible(self, farm: int) -> FarmerState:
        """What the farmer can offer right now: the inventory at its remaining levels."""
        inv = self.inventories[farm]
        st = inv.state
        return FarmerState(stocks=tuple(inv.remaining), qualities=st.qualities,
                           reservation=st.reservation)

    # ------------------------------------------------------------------
    def begin_day(self) -> None:
        self.day += 1
        if self.day % max(1, self.cfg.economy.season_days) == 0:
            self.season = self.day // max(1, self.cfg.economy.season_days)
            for f in range(self.n_farms):
                self.inventories[f] = self._draw(f)

    def _buyer_need(self) -> BuyerState:
        """Drawn with no reference whatsoever to the farm being visited."""
        saved = self.world.rng
        self.world.rng = self.rng
        try:
            for _ in range(200):
                b = self.world.sample_buyer()
                if not self.world.is_held_out(b.want_variety, b.need_qty):
                    return b
        finally:
            self.world.rng = saved
        return b

    # ------------------------------------------------------------------
    def make_batch(self, n: int, n_farmers: int, n_buyers: int
                   ) -> tuple[list[Scenario], torch.Tensor, torch.Tensor]:
        """Produce ``n`` encounters, advancing market days as it goes."""
        per_day = max(1, self.cfg.economy.episodes_per_day)
        if not self.cfg.economy.persistent_inventory:
            scen = [self.world.sample(held_out=False) for _ in range(n)]
            f_idx = torch.tensor([self.rng.randrange(n_farmers) for _ in range(n)],
                                 dtype=torch.long)
            b_idx = torch.tensor([self.rng.randrange(n_buyers) for _ in range(n)],
                                 dtype=torch.long)
            for _ in range(max(1, n // per_day)):
                self.begin_day()
            return scen, f_idx, b_idx

        scen: list[Scenario] = []
        f_list: list[int] = []
        b_list: list[int] = []
        while len(scen) < n:
            self.begin_day()
            for _ in range(min(per_day, n - len(scen))):
                farm = self.rng.randrange(n_farmers)
                buyer_i = self.rng.randrange(n_buyers)
                farmer_state = self._visible(farm)
                buyer_state = self._buyer_need()
                held = self.world.is_held_out(buyer_state.want_variety,
                                              buyer_state.need_qty)
                scen.append(Scenario(farmer=farmer_state, buyer=buyer_state,
                                     day=self.day, held_out=held))
                f_list.append(farm)
                b_list.append(buyer_i)
        return (scen,
                torch.tensor(f_list, dtype=torch.long),
                torch.tensor(b_list, dtype=torch.long))

    # ------------------------------------------------------------------
    def settle(self, f_idx: torch.Tensor, outcomes: Sequence[Any]) -> dict[str, float]:
        """Deplete inventories by completed sales; restock any farm that sells out."""
        stats = {"apples_sold": 0, "value": 0.0, "profit": 0.0, "soldout": 0}
        for i, o in enumerate(outcomes):
            if not o.success:
                continue
            stats["apples_sold"] += o.traded_qty
            stats["value"] += o.trade_value
            stats["profit"] += o.farmer_profit
            if not self.cfg.economy.persistent_inventory:
                continue
            farm = int(f_idx[i])
            inv = self.inventories[farm]
            v = o.traded_variety
            sold = min(o.traded_qty, inv.remaining[v])
            inv.remaining[v] -= sold
            inv.sold_total += sold
            if inv.empty:
                self.soldouts += 1
                stats["soldout"] += 1
                self.inventories[farm] = self._draw(farm)
        return stats

    # ------------------------------------------------------------------
    # tensor path
    # ------------------------------------------------------------------
    def inventory_tensors(self, device) -> tuple:
        """Current per-farm stock, quality and cost, as (n_farms, V) / (n_farms,)."""
        stocks = torch.tensor([inv.remaining for inv in self.inventories],
                              dtype=torch.long, device=device)
        quals = torch.tensor([list(inv.state.qualities) for inv in self.inventories],
                             dtype=torch.long, device=device)
        res = torch.tensor([inv.state.reservation for inv in self.inventories],
                           dtype=torch.long, device=device)
        return stocks, quals, res

    def make_batch_tensor(self, n: int, n_farmers: int, n_buyers: int,
                          tensor_world, f_idx, b_idx, *, held_out: bool = False):
        """A ScenarioBatch: buyer halves drawn fresh, farmer halves read off the farms.

        The two sides are assembled separately and never consult each other, which
        is the same independence the scalar path guarantees -- it is just done with
        one gather instead of one Python object per episode.
        """
        from .batched import ScenarioBatch
        sb = tensor_world.sample(n, held_out=held_out)
        if not self.cfg.economy.persistent_inventory:
            sb.day = self.day
            for _ in range(max(1, n // max(1, self.cfg.economy.episodes_per_day))):
                self.begin_day()
            return sb
        for _ in range(max(1, n // max(1, self.cfg.economy.episodes_per_day))):
            self.begin_day()
        stocks, quals, res = self.inventory_tensors(sb.want_variety.device)
        return ScenarioBatch(
            stocks=stocks[f_idx].clamp(min=0), qualities=quals[f_idx],
            reservation=res[f_idx], want_variety=sb.want_variety,
            need_qty=sb.need_qty, min_quality=sb.min_quality,
            max_price=sb.max_price, held_out=sb.held_out, day=self.day)

    def settle_tensor(self, f_idx, res: dict) -> dict[str, float]:
        """Deplete farms by the batch's completed sales, without a Python loop."""
        import torch as _t
        succ = res["success"]
        qty = res["traded_qty"] * succ.long()
        stats = {
            "apples_sold": int(qty.sum()),
            "value": float((res["trade_value"] * succ.float()).sum()),
            "profit": float((res["farmer_profit"] * succ.float()).sum()),
            "soldout": 0,
        }
        if not self.cfg.economy.persistent_inventory:
            return stats
        variety = res["agreed_variety"]
        flat = f_idx.long() * self.cfg.world.n_varieties + variety.long()
        n_cells = self.n_farms * self.cfg.world.n_varieties
        sold = _t.zeros(n_cells, dtype=_t.long, device=qty.device)
        sold.scatter_add_(0, flat, qty)
        sold = sold.view(self.n_farms, self.cfg.world.n_varieties).tolist()
        for farm in range(self.n_farms):
            inv = self.inventories[farm]
            for v in range(self.cfg.world.n_varieties):
                if sold[farm][v]:
                    inv.remaining[v] = max(0, inv.remaining[v] - sold[farm][v])
                    inv.sold_total += sold[farm][v]
            if inv.empty:
                self.soldouts += 1
                stats["soldout"] += 1
                self.inventories[farm] = self._draw(farm)
        return stats

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        w = self.cfg.world
        return {
            "day": self.day, "season": self.season,
            "restocks": self.restocks, "soldouts": self.soldouts,
            "inventories": [
                {"remaining": dict(zip(w.variety_names, inv.remaining)),
                 "cost": w.price_values[inv.state.reservation]}
                for inv in self.inventories],
        }
