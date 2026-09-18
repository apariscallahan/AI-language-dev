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
        self.farmers: list[Agent] = [self._spawn(FARMER, s, 0, 0, initial=True)
                                     for s in range(p.n_farmers)]
        self.buyers: list[Agent] = [self._spawn(BUYER, s, 0, 0, initial=True)
                                    for s in range(p.n_buyers)]

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
        return self.farmers + self.buyers

    def pair(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Random Farmer/Buyer pairings for ``n`` episodes (spec 1.1: a marketplace)."""
        f = torch.tensor([self.rng.randrange(len(self.farmers)) for _ in range(n)],
                         dtype=torch.long)
        b = torch.tensor([self.rng.randrange(len(self.buyers)) for _ in range(n)],
                         dtype=torch.long)
        return f, b

    # ------------------------------------------------------------------
    def record_episode_participation(self, f_idx: torch.Tensor, b_idx: torch.Tensor,
                                     batch) -> None:
        """Age every agent by the episodes it actually played, and tally its results."""
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
