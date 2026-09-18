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

  1. ``refer``   A lineup game. The informer sees one meaning and describes it;
                 the guesser sees several candidate meanings and picks. No price,
                 no budget, no negotiation, no market. Success is 1/K by chance,
                 which is a gradient an RL agent can actually climb.
  2. ``haggle``  Price and budget appear, so there is a real accept/reject with a
                 payoff -- but still one message each and then decide.
  3. ``bargain`` The same, with multiple turns, so counter-offers become possible.
  4. ``market``  The full economy: persistent stock, restocking, several goods,
                 viability. This is the existing trading task, and it is the last
                 thing agents meet rather than the first.

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

from dataclasses import dataclass, field
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

    @property
    def referential(self) -> bool:
        return self.name == "refer"

    def speaker_of_turn(self, turn: int) -> int:
        """Whose turn it is.  In the lineup game the informer talks first."""
        if self.informer_first:
            return FARMER if turn % 2 == 0 else BUYER
        return BUYER if turn % 2 == 0 else FARMER

    def active_heads(self, role: int, cfg: Config) -> list[int]:
        """Which outputs are scored, and therefore which ones get a gradient.

        Sampling a head nobody scores only adds noise to the policy-gradient
        term, so each phase names what it actually uses.
        """
        if self.referential:
            # The informer only speaks; it has no decision to make, and its
            # gradient arrives entirely through the guesser's forward pass.
            return [H_CHOICE] if role == BUYER else []
        heads = [H_ACCEPT, H_VARIETY, H_QTY]
        if self.use_price:
            heads.append(H_PRICE)
        if cfg.reward.belief_heads:
            heads.extend(H_BELIEF)
        return heads


def ladder(cfg: Config) -> list[Phase]:
    """The phases, easiest first.  ``n_turns`` never exceeds ``channel.n_turns``."""
    full = max(2, cfg.channel.n_turns)
    return [
        Phase("refer", 0, 1, True, False, False,
              "lineup game: describe one meaning, pick it out of several"),
        Phase("haggle", 1, 2, False, True, False,
              "price and budget appear; one message each, then accept or walk"),
        Phase("bargain", 2, full, False, True, False,
              "several turns, so counter-offers are possible"),
        Phase("market", 3, full, False, True, True,
              "the full economy: stock, restocking, viability"),
    ]


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
    min_episodes: int
    max_episodes: int

    def evaluate(self, *, success: float, chance: float, topsim: float,
                 null: float, transfer: float, episodes_in_phase: int
                 ) -> tuple[bool, dict[str, Any]]:
        def ok(x) -> bool:
            return isinstance(x, float) and x == x

        checks = {
            "long enough in phase": (episodes_in_phase >= self.min_episodes,
                                     "%d of %d episodes" % (episodes_in_phase,
                                                            self.min_episodes)),
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
    if phase.referential:
        chance = 1.0 / max(2, c.n_candidates)
        return Promotion(
            min_success=max(c.refer_min_success, 2.0 * chance),
            min_success_over_chance=c.min_success_over_chance,
            min_topsim_over_null=c.min_topsim_over_null,
            min_channel_transfer=c.min_channel_transfer,
            min_episodes=c.min_episodes_per_phase,
            max_episodes=c.max_episodes_per_phase)
    return Promotion(
        min_success=c.trade_min_success,
        min_success_over_chance=c.min_success_over_chance,
        min_topsim_over_null=c.min_topsim_over_null,
        min_channel_transfer=c.min_channel_transfer,
        min_episodes=c.min_episodes_per_phase,
        max_episodes=c.max_episodes_per_phase)


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
        base = informer_schema(cfg) if role == FARMER else guesser_schema(cfg)
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
        if role == FARMER:
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

    def _draw(self, n: int) -> torch.Tensor:
        """(n, 3) meanings: variety, quantity, quality."""
        w = self.cfg.world
        variety = self.tw._categorical(self.tw._variety_cdf, n)
        qty = self.tw._categorical(self.tw._qty_cdf, n) + 1
        quality = self.tw._skew_low(0, w.n_quality - 1, n)
        return torch.stack([variety, qty, quality], dim=1)

    def sample(self, n: int) -> ReferentialBatch:
        K = self.cfg.curriculum.n_candidates
        cand = torch.stack([self._draw(n) for _ in range(K)], dim=1)   # (n, K, 3)
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
        return ReferentialBatch(meanings=perm, target=target, day=0)


def resolve_referential(cfg: Config, rb: ReferentialBatch, choice: torch.Tensor,
                        informer_symbols: torch.Tensor,
                        guesser_symbols: torch.Tensor) -> dict[str, torch.Tensor]:
    """Score a batch of lineup rounds.

    Both roles are paid for the same thing -- did the guess land -- because in a
    lineup game being understood and understanding are the same event. The symbol
    cost still applies, so brevity is still worth something.
    """
    R = cfg.reward
    correct = choice == rb.target
    reward = R.refer_success * correct.float() + R.refer_miss * (~correct).float()
    f = reward - R.symbol_cost * informer_symbols.float()
    b = reward - R.symbol_cost * guesser_symbols.float()
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
@dataclass
class CurriculumState:
    """Where the run is on the ladder, and how it got there."""
    phases: list[Phase]
    index: int = 0
    episodes_in_phase: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)
    stalled: bool = False
    last_report: dict[str, Any] = field(default_factory=dict)

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
        self.stalled = False
        self.transitions.append({
            "episode": episode, "from": old.name, "to": self.phase.name,
            "episodes_in_previous_phase": None, "criteria": checks,
        })
        return self.phase
