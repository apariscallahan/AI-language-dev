"""Population, ageing, death and birth (spec 3).

The population is a fixed number of *lineage slots* per role.  A slot always
holds exactly one living agent.  When that agent's lifespan runs out it is
removed and a brand-new randomly-initialised agent takes the slot, with the
slot's generation counter incremented.  Weights are never inherited -- "generation"
counts population turnover, not genetic descent, exactly as the spec specifies.

Deaths are deliberately staggered.  If the whole population turned over at once
there would be no veterans left to learn from and the language would restart from
noise every cycle; the point of overlapping generations is that at any moment
some agents already know the code and some must acquire it, which is the pressure
towards a code that is *learnable by a stranger* rather than a private cipher.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch

from .agents import Agent, make_agent
from .config import Config
from .env import BUYER, FARMER, ROLE_NAMES


@dataclass
class BirthEvent:
    episode: int
    agent_id: int
    role: int
    slot: int
    generation: int
    replaced_agent_id: int
    replaced_age: int
    replaced_success_rate: float
    lifespan: int
    bottleneck: dict[str, Any] = field(default_factory=dict)
    kind: str = "replacement"          # or "newcomer": a new slot, nobody replaced

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["role"] = ROLE_NAMES[self.role]
        return d


class Population:
    """Owns every living agent and the birth/death clock."""

    def __init__(self, cfg: Config, rng: random.Random, device: str = "cpu"):
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self._next_id = 0
        self.births: list[BirthEvent] = []
        self.deaths = 0

        p = cfg.population
        nf = p.founders_farmers if p.founders_farmers > 0 else p.n_farmers
        nb = p.founders_buyers if p.founders_buyers > 0 else p.n_buyers
        # Until trading begins there are no roles: one pool of agents takes both
        # seats of a lineup, so there is one language rather than two that have
        # to be reconciled. ``farmers`` and ``buyers`` are the same list, and
        # pairing never puts an agent opposite itself.
        self.shared = bool(cfg.curriculum.enabled and cfg.curriculum.split_roles_at)
        if self.shared:
            n = max(nf, nb)
            self.farmers: list[Agent] = [self._spawn(FARMER, s, 0, 0, initial=True)
                                         for s in range(n)]
            self.buyers: list[Agent] = self.farmers
        else:
            self.farmers = [self._spawn(FARMER, s, 0, 0, initial=True)
                            for s in range(nf)]
            self.buyers = [self._spawn(BUYER, s, 0, 0, initial=True)
                           for s in range(nb)]

    @property
    def full_size(self) -> bool:
        p = self.cfg.population
        if self.shared:
            return len(self.farmers) >= max(p.n_farmers, p.n_buyers)
        return len(self.farmers) >= p.n_farmers and len(self.buyers) >= p.n_buyers

    # ------------------------------------------------------------------
    def split_roles(self, episode: int) -> int:
        """Trading begins: copy every agent into a farmer and a buyer.

        Both roles therefore start out fluent in the one language the pool
        learned, which is the point of keeping them together until now. The copy
        carries the weights *and* the optimiser state; only the role embedding
        each one uses differs from here on.
        """
        import copy as _copy
        if not self.shared:
            return 0
        pool = self.farmers
        farmers, buyers = [], []
        for slot, a in enumerate(pool):
            twin = _copy.deepcopy(a)
            twin.agent_id = self._next_id
            self._next_id += 1
            twin.role = BUYER
            twin.net.role = BUYER
            a.slot = twin.slot = slot
            farmers.append(a)
            buyers.append(twin)
        self.farmers, self.buyers = farmers, buyers
        self.shared = False
        return len(pool)

    def add_newcomer(self, role: int, episode: int,
                     on_birth: Optional[Callable[[Agent, "BirthEvent"], None]] = None
                     ) -> "BirthEvent":
        """Grow the community by one agent of ``role``, in a new lineage slot."""
        pool = self.pool(role)
        slot = len(pool)
        agent = self._spawn(role, slot, 1, episode)
        ev = BirthEvent(episode=episode, agent_id=agent.agent_id, role=role, slot=slot,
                        generation=agent.generation, replaced_agent_id=-1, replaced_age=0,
                        replaced_success_rate=float("nan"), lifespan=agent.lifespan,
                        kind="newcomer")
        pool.append(agent)
        if on_birth is not None:
            on_birth(agent, ev)
        agent.bottleneck_info = ev.bottleneck
        self.births.append(ev)
        return ev

    # ------------------------------------------------------------------
    def _sample_lifespan(self, initial: bool) -> int:
        p = self.cfg.population
        span = self.rng.randint(p.lifespan_min, p.lifespan_max)
        if initial and p.initial_stagger:
            # Spread the founding cohort's first deaths over the whole lifespan
            # window instead of bunching them at the same episode.
            span = self.rng.randint(max(1, p.lifespan_min // 8), span)
        return span

    def _spawn(self, role: int, slot: int, generation: int, episode: int,
               initial: bool = False) -> Agent:
        agent = make_agent(self.cfg, agent_id=self._next_id, role=role, slot=slot,
                           generation=generation, birth_episode=episode,
                           lifespan=self._sample_lifespan(initial), device=self.device)
        self._next_id += 1
        return agent

    # ------------------------------------------------------------------
    def pool(self, role: int) -> list[Agent]:
        return self.farmers if role == FARMER else self.buyers

    def all_agents(self) -> list[Agent]:
        return list(self.farmers) if self.shared else self.farmers + self.buyers

    def pair(self, n: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Farmer/Buyer pairings for ``n`` episodes (spec 1.1: a marketplace).

        Episode i goes to farmer ``i % n_farmers`` and buyer
        ``(i // n_farmers) % n_buyers``.  Because scenarios are drawn i.i.d., this
        is distributionally the same marketplace as drawing each pairing at
        random, but it gives every agent exactly its share of the batch instead of
        a multinomial count -- lower gradient variance -- and it makes each
        agent's slice of the batch a compile-time constant, which is what lets the
        rollout run without stalling the device to ask who plays what.
        """
        nf, nb = len(self.farmers), len(self.buyers)
        i = torch.arange(n, device=device)
        if self.shared:
            # One pool: seat A is agent i % n, seat B is a *different* agent, and
            # over a batch every ordered pair comes up equally often.
            if nf < 2:
                return i % nf, i % nf
            a = i % nf
            step = 1 + torch.div(i, nf, rounding_mode="floor") % (nf - 1)
            return a, (a + step) % nf
        return i % nf, torch.div(i, nf, rounding_mode="floor") % nb

    # ------------------------------------------------------------------
    def record_episode_participation(self, f_idx: torch.Tensor, b_idx: torch.Tensor,
                                     batch) -> None:
        """Age every agent by the episodes it actually played, and tally its results."""
        if getattr(batch, "res", None) is not None:
            return self._record_from_tensors(f_idx, b_idx, batch)
        seats = ((f_idx, self.farmers), (b_idx, self.buyers))
        if self.shared:      # one pool in both seats: it took part in one update
            seats = ((torch.cat([f_idx, b_idx]), self.farmers),)
        for idx, pool in seats:
            for a_i in set(int(x) for x in idx.tolist()):
                pool[a_i].updates += 1            # took part in this update
        for i in range(len(batch)):
            o = batch.outcomes[i]
            fa = self.farmers[int(f_idx[i])]
            ba = self.buyers[int(b_idx[i])]
            for a, rew in ((fa, o.farmer_reward), (ba, o.buyer_reward)):
                a.age += 1
                a.n_episodes += 1
                a.reward_sum += rew
                a.n_success += int(o.success)
            if o.success:
                fa.apples_traded += o.traded_qty
                ba.apples_traded += o.traded_qty
                fa.value_traded += o.trade_value
                ba.value_traded += o.trade_value
                fa.profit += o.farmer_profit
                ba.profit += o.buyer_savings

    def _record_from_tensors(self, f_idx: torch.Tensor, b_idx: torch.Tensor,
                             batch) -> None:
        """The same tallies, as a handful of scatter_adds instead of B iterations."""
        res = batch.res
        succ = res["success"]
        seats = ((self.farmers, f_idx, batch.f_reward, res["farmer_profit"]),
                 (self.buyers, b_idx, batch.b_reward, res["buyer_savings"]))
        if self.shared:
            # The same agents fill both seats, so their episodes and rewards are
            # the two seats added together -- and it is still *one* update each.
            seats = ((self.farmers, torch.cat([f_idx, b_idx]),
                      torch.cat([batch.f_reward, batch.b_reward]),
                      torch.cat([res["farmer_profit"], res["buyer_savings"]])),)
            succ = torch.cat([succ, succ])
        for pool, idx, rew, money in seats:
            n = len(pool)
            idx = idx.long()
            counts = torch.zeros(n, device=idx.device).scatter_add_(
                0, idx, torch.ones_like(idx, dtype=torch.float))
            rewards = torch.zeros(n, device=idx.device).scatter_add_(0, idx, rew.float())
            successes = torch.zeros(n, device=idx.device).scatter_add_(
                0, idx, succ.float())
            qty, val = res["traded_qty"], res["trade_value"]
            if self.shared:
                qty, val = torch.cat([qty, qty]), torch.cat([val, val])
            apples = torch.zeros(n, device=idx.device).scatter_add_(
                0, idx, (qty * succ.long()).float())
            value = torch.zeros(n, device=idx.device).scatter_add_(
                0, idx, val * succ.float())
            profit = torch.zeros(n, device=idx.device).scatter_add_(
                0, idx, money * succ.float())
            for a, c, r, sx, ap, va, pf in zip(
                    pool, counts.tolist(), rewards.tolist(), successes.tolist(),
                    apples.tolist(), value.tolist(), profit.tolist()):
                a.age += int(c)
                a.n_episodes += int(c)
                if c > 0:
                    a.updates += 1            # took part in this update
                a.reward_sum += r
                a.n_success += int(sx)
                a.apples_traded += int(ap)
                a.value_traded += va
                a.profit += pf

    # ------------------------------------------------------------------
    def turn_over(self, episode: int,
                  on_birth: Optional[Callable[[Agent, BirthEvent], None]] = None
                  ) -> list[BirthEvent]:
        """Retire expired agents and install newborns.  No-op when turnover is off."""
        if not self.cfg.population.turnover:
            return []
        events: list[BirthEvent] = []
        for role in (FARMER, BUYER):
            pool = self.pool(role)
            for slot, agent in enumerate(pool):
                if not agent.is_expired():
                    continue
                newborn = self._spawn(role, slot, agent.generation + 1, episode)
                ev = BirthEvent(
                    episode=episode, agent_id=newborn.agent_id, role=role, slot=slot,
                    generation=newborn.generation, replaced_agent_id=agent.agent_id,
                    replaced_age=agent.age,
                    replaced_success_rate=agent.success_rate,
                    lifespan=newborn.lifespan)
                pool[slot] = newborn
                self.deaths += 1
                if on_birth is not None:
                    on_birth(newborn, ev)
                newborn.bottleneck_info = ev.bottleneck
                events.append(ev)
                self.births.append(ev)
        return events

    # ------------------------------------------------------------------
    def veterans(self, role: int, born_before: int) -> list[int]:
        """Indices of agents in ``role``'s pool that were already alive at an episode."""
        return [i for i, a in enumerate(self.pool(role)) if a.birth_episode < born_before]

    def composition(self) -> dict[str, Any]:
        def summarise(pool: Sequence[Agent]) -> dict[str, Any]:
            return {
                "n": len(pool),
                "ages": [a.age for a in pool],
                "generations": [a.generation for a in pool],
                "mean_age": sum(a.age for a in pool) / max(1, len(pool)),
                "mean_generation": sum(a.generation for a in pool) / max(1, len(pool)),
                "max_generation": max((a.generation for a in pool), default=0),
                "success_rates": [round(a.success_rate, 3) for a in pool],
            }
        return {"farmers": summarise(self.farmers), "buyers": summarise(self.buyers),
                "total_births": len(self.births), "total_deaths": self.deaths}
