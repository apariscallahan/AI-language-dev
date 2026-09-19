"""Configuration for the orchard emergent-language simulation.

Everything tunable lives here.  Nothing in the simulation should hardcode a
population size, vocabulary size, message length, lifespan, etc. -- the whole
scientific point of the project (spec section 7) is running the *same* code with
different settings and comparing, so all of it is config-driven and serialisable
to JSON.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------
@dataclass
class WorldConfig:
    n_varieties: int = 4              # RED / GREEN / GOLD / RUSSET
    n_quality: int = 3                # LOW / MED / HIGH
    max_qty: int = 20                 # quantities 1..max_qty  (spec 1.4: big enough that
                                      # memorising whole scenarios is infeasible)
    n_price_bins: int = 12            # "continuous-ish" price discretised onto a fine grid
    price_min: float = 1.0
    price_step: float = 0.5

    # Farmer stock is sampled uniformly in [1, max_qty].
    # Farmer reservation (cost) price bin sampled in [0, reservation_max_bin].
    reservation_max_bin: int = 9
    # Buyer budget ceiling bin sampled in [budget_min_bin, n_price_bins-1].
    budget_min_bin: int = 1

    # Probability that a farm stocks any given variety at all.  Together with the
    # skewed-but-independent marginals in World, this is what keeps roughly half
    # of encounters worth doing WITHOUT making either side's private state
    # predictable from the other's -- see the note at the top of world.py.
    p_stocked: float = 0.90
    # Both sides are drawn fresh and independently every round -- that is what
    # keeps the private information genuinely private.  These shift the *marginals*
    # so the two distributions overlap often enough that closing a deal is the
    # common case, without ever making one side's draw depend on the other's.
    stock_floor_frac: float = 0.0   # farms carry at least this fraction of max_qty
    quality_bias: int = 0           # extra skew: farms up, shoppers down
    # A shop stocks more than any one shopper asks for.  This is the cleanest way
    # to make deals common: it lifts P(stock >= need) a long way while leaving the
    # farmer's stock broadly spread, so the buyer still cannot guess it without
    # being told.  Narrowing the *stock* range instead would raise viability just
    # as well and quietly make the farmer's state predictable, which is the thing
    # this whole world design exists to prevent.
    need_max_frac: float = 0.65     # shoppers ask for up to this fraction of max_qty

    # Held-out (variety, quantity) combinations, never sampled during training,
    # used for the zero-shot generalisation metric (spec 5.6).
    holdout_frac: float = 0.10
    holdout_seed: int = 1234

    # Zipf-like skew on which meanings actually come up (addendum 2.2).  The
    # length/frequency pressure has nothing to act on in a uniform world: if every
    # meaning is equally common, no meaning is worth a short word.
    #
    # The skew is split by dimension because the two behave very differently.
    # Quantity has many values, so even a strong skew leaves plenty of mass spread
    # across the rest, and it supplies most of the frequency *range* that the
    # length analysis needs.  Variety has only a handful, so skewing it hands a
    # non-listening agent a large free score and -- measured, not guessed -- wipes
    # out reference learning entirely: 64% of the channel headroom with a uniform
    # variety marginal, 0% at alpha 0.8.  Default is therefore a strong skew on
    # quantity and none on variety.  Both remain tunable, and setting
    # zipf_alpha_variety above 0 is a good way to reproduce that finding.
    zipf_alpha: float = 0.9            # over requested quantities
    zipf_alpha_variety: float = 0.0    # over requested varieties

    @property
    def price_values(self) -> list[float]:
        return [self.price_min + i * self.price_step for i in range(self.n_price_bins)]

    @property
    def variety_names(self) -> list[str]:
        base = ["RED", "GREEN", "GOLD", "RUSSET", "BRAMLEY", "PIPPIN", "FUJI", "GALA",
                "COX", "BRAEBURN", "JAZZ", "ENVY", "EMPIRE", "COMICE", "DISCOVERY", "SPARTAN"]
        # Labels are for humans reading reports; the agents only ever see indices.
        return (base + ["V%d" % i for i in range(len(base), self.n_varieties)])[: self.n_varieties]

    @property
    def quality_names(self) -> list[str]:
        base = ["LOW", "MED", "HIGH", "PRIME"]
        return base[: self.n_quality]


# --------------------------------------------------------------------------
# Communication channel  (addendum section 1, superseding original spec 2.2)
# --------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    """An open vocabulary built from a small closed set of meaningless atoms.

    The original design gave agents 20-40 fixed tokens and one choice per slot.
    That can only ever produce a code: the vocabulary cannot grow, words cannot
    get longer or shorter, and there is nothing for word-formation to happen in.

    Here an agent emits a *stream of symbols*, one per step, from

        {atom_0 .. atom_{A-1}}  u  {HYPHEN, SPACE, END}

    A **word** is atoms joined by HYPHEN; an **utterance** (one turn) is words
    separated by SPACE:

        utterance := word (SPACE word)*        word := atom (HYPHEN atom)*

    That shape is part of the medium, like letters being written in words, and
    with ``enforce_word_grammar`` it is enforced at every step: after an atom the
    speaker may continue the word (HYPHEN), start a new word (SPACE) or stop
    (END); after a HYPHEN or SPACE it must say an atom. So every junction between
    two atoms is an explicit choice between "same word" and "next word", and a
    transcript reads exactly as it was emitted -- ``a3-a7 a1`` is a two-atom word
    and a one-atom word. (Without the rule, bare atoms ran together into one
    "word" while the HYPHEN symbol did nothing, and reports printed hyphens the
    agents had never emitted.)

    Nothing about *which* atoms form words, or where words split, is given: no
    meaning is assigned to any atom, HYPHEN or SPACE.

    ``max_symbols`` is a buffer size, not a pressure. It is set well above what
    any meaning needs; what keeps utterances short is a *cost*
    (``RewardConfig.symbol_cost``), and the report flags it if utterances ever
    actually reach the buffer's end.
    """
    atomic_vocab: int = 36     # meaningless atoms; ids 0 .. atomic_vocab-1
                               # HYPHEN = atomic_vocab       joins atoms into a word
                               # SPACE  = atomic_vocab + 1   separates words
                               # END    = atomic_vocab + 2   ends the utterance
                               # PAD    = atomic_vocab + 3   never emitted; fills the slot
    max_symbols: int = 24      # buffer per turn; generous on purpose (the cost sets length)
    n_turns: int = 6           # alternating turns per negotiation; buyer speaks first
    enforce_word_grammar: bool = True   # atoms and HYPHEN/SPACE must alternate

    # ---- symbol ids ----------------------------------------------------
    @property
    def hyphen_id(self) -> int:
        return self.atomic_vocab

    @property
    def space_id(self) -> int:
        return self.atomic_vocab + 1

    @property
    def end_id(self) -> int:
        return self.atomic_vocab + 2

    @property
    def pad_id(self) -> int:
        return self.atomic_vocab + 3

    @property
    def n_emittable(self) -> int:
        """Size of the output head: atoms + hyphen + space + end."""
        return self.atomic_vocab + 3

    @property
    def n_symbol_ids(self) -> int:
        """Size of the embedding table (everything above, plus PAD)."""
        return self.atomic_vocab + 4

    @property
    def dialogue_len(self) -> int:
        return self.n_turns * self.max_symbols

    def is_atom(self, sym: int) -> bool:
        return 0 <= sym < self.atomic_vocab

    def is_structural(self, sym: int) -> bool:
        return sym in (self.hyphen_id, self.space_id)

    def costed(self, sym: int) -> bool:
        """Symbols the speaker pays for: atoms, hyphens and spaces.

        Ending an utterance is free -- brevity should not be taxed.
        """
        return 0 <= sym < self.end_id

    # ---- names the rest of the codebase still uses ----------------------
    # The machinery below (rollout, agents, metrics) is written against a generic
    # "emit symbols until END" loop, so these aliases let the channel change shape
    # without touching it.
    @property
    def vocab_size(self) -> int:
        return self.atomic_vocab

    @property
    def eos_id(self) -> int:
        return self.end_id

    @property
    def max_msg_len(self) -> int:
        return self.max_symbols

    @property
    def n_token_ids(self) -> int:
        return self.n_symbol_ids


# --------------------------------------------------------------------------
# Agent network
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.0


# --------------------------------------------------------------------------
# Reward shaping  (spec 1.3)
# --------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """All reward terms depend on *mutual* outcomes, never on privileged access.

    The partial-credit terms exist because the fully-joint success event
    (both accept AND agree on variety AND quantity AND price AND the deal is
    actually feasible) has probability ~1e-3 under random play, which is far too
    sparse for REINFORCE to bootstrap from.  Every partial term still requires
    genuine information transfer: the two agents must *agree with each other*,
    and each holds only half of what is needed to agree correctly.
    """
    # ---- the two halves of a closed communication loop --------------------
    # Measured before these existed: the buyer got 91% of its comprehension score
    # (2.204 of 2.428) just by naming its own want and need, so it had no reason
    # to listen; and neither role had any term at all for being *understood*.
    # Whatever an agent said, its reward was the same as long as the trade came
    # out the same way, which left nothing teaching either side to be informative.
    #
    # Each agent now states what it believes the other party's private situation
    # to be, and that statement is scored against the truth.  Then:
    #   decode      pays an agent for reading the other correctly
    #   understood  pays an agent for having been read correctly
    # The second is the one that was missing.  It is symmetric, it is per-message
    # rather than per-trade, and neither term is obtainable without the channel:
    # every field scored is one the scoring agent cannot observe.
    # Off restores the pre-belief-head agent exactly: four decision outputs, no
    # belief statement sampled, no decode/understood terms.  Kept switchable
    # because adding the heads also doubles the sampled action space, and those
    # two effects have to be separable when something regresses.
    belief_heads: bool = True
    # How strongly the belief heads pull on the speaker's gradient.  They are the
    # point of the closed loop, but they also double the sampled action space, and
    # the score-function term over them is high-variance; below 1 keeps the loop
    # without letting it drown the deal-decision signal.
    belief_grad_weight: float = 1.0
    # The lineup game pays both sides for the same event, because there being
    # understood and understanding are the same thing.
    refer_success: float = 1.0
    refer_miss: float = -0.1

    decode: float = 0.45            # I worked out your situation
    understood: float = 0.45        # you worked out mine
    belief_qty_tol: int = 1         # counts as read correctly if within this
    belief_price_tol: int = 1

    success: float = 1.5            # viable deal, both accept, beliefs agree, feasible
    correct_no_deal: float = 0.25   # not viable, both reject  (the right answer)
    agree_per_dim: float = 0.05     # the two agents' beliefs match, per dimension
    correct_per_dim: float = 0.10   # this agent's deal decision is right, per dimension
                                    # (halved when decode/understood arrived: the
                                    # comprehension signal now lives there instead)
    judgement: float = 0.25         # this agent's accept/reject matches whether a deal
                                    # was actually possible -- the fourth comprehension
                                    # dimension, and the one that trains the accept head
    one_sided_accept: float = -0.10 # one accepts, one rejects
    missed_deal: float = -0.10      # viable but both rejected
    bad_deal: float = -0.15         # not viable but both accepted
    symbol_cost: float = 0.03       # per emitted symbol -- atoms, hyphens and spaces
                                    # all count (addendum 2.1).  A soft pressure toward
                                    # brevity on top of the hard per-turn cap, because
                                    # people do not routinely max out the longest
                                    # sentence they could physically produce.  It is
                                    # also the whole of the Zipf mechanism (2.2): the
                                    # cost is paid once per episode, so meanings that
                                    # come up often pay it far more often and feel far
                                    # more pressure to shorten.
    economics: float = 0.25         # farmer margin / buyer surplus (zero-sum in price)

    qty_tol: int = 0                # tolerance when comparing believed quantities
    price_tol: int = 0              # tolerance when comparing believed price bins

    # ---- conventions (charged or paid to the speaker only) ---------------
    # Coining: a word costs more the rarer it is in the population's recent
    # usage. Rarity runs from 0 for established forms (at least
    # ``rarity_common_share`` of recent word tokens) to 1 at or below
    # ``rarity_novel_share`` -- including forms nobody has used -- log-linear
    # between, and each word is charged ``rarity_cost`` x (its rarity minus the
    # batch's mean word rarity). Centred, so it steers towards established forms
    # without ever making silence the cheap option. A price, not a ban.
    rarity_cost: float = 0.05
    rarity_common_share: float = 0.01
    rarity_novel_share: float = 1e-4
    # Agreeing: paid for saying what the population currently says for this
    # meaning (1 - normalised edit distance to the modal utterance), not just
    # for being understood by this one partner. Only counts once the
    # convention has ``convention_min_support`` recent uses behind it.
    convention: float = 0.15
    convention_min_support: int = 12
    # Whether the convention bonus waits for a working channel like the costs do.
    # Measured: ungated, even strongly weighted, it raised coherence among six
    # speakers from random weights only to ~0.2 and did not get their lineup off
    # chance -- a population that size needs founding small (see
    # population.founders_*), after which the bonus is fully on anyway.
    convention_gated: bool = True
    usage_half_life: int = 20_000    # episodes; how "recent" recent usage is


# --------------------------------------------------------------------------
# Economy loop  (spec 1.4)
# --------------------------------------------------------------------------
@dataclass
class EconomyConfig:
    persistent_inventory: bool = True   # farms hold a lot across days and deplete it
    episodes_per_day: int = 8           # encounters at one market day
    season_days: int = 3                # days between full replenishment


# --------------------------------------------------------------------------
# Curriculum
# --------------------------------------------------------------------------
@dataclass
class CurriculumConfig:
    """Learn to refer before learning to haggle.

    The full trading task is too conjunctive to bootstrap from random weights --
    measured: success 0.000 at every checkpoint of a 10k-episode run, with the
    channel carrying nothing.  The ladder in orchard/curriculum.py starts with a
    lineup game whose chance rate is 1/K rather than ~0, and only moves on when a
    phase has demonstrably worked.
    """
    enabled: bool = True
    n_candidates: int = 4            # lineup size in the referential phase

    # ---- promotion, on evidence rather than on a schedule -----------------
    # Fallback budget for a rung not named in ``rung_budgets``.
    min_episodes_per_phase: int = 20_000
    max_episodes_per_phase: int = 400_000
    # (minimum, maximum) episodes per rung. A rung that meets its criteria after
    # its minimum is left at the next check; one that reaches its maximum
    # without meeting them ends the run with a report (see ``on_stall``).
    rung_budgets: dict = field(default_factory=lambda: {
        "refer": [20_000, 250_000],
        "refer-swap": [20_000, 250_000],
        "refer-mutual": [20_000, 300_000],
        "order": [20_000, 300_000],
        "haggle": [20_000, 300_000],
        "bargain": [20_000, 300_000],
        "market": [20_000, 10**12],
    })
    # Promotion is checked this often -- a light probe of just the evidence the
    # rung needs -- rather than only at the (much heavier) full checkpoints.
    check_every: int = 5_000
    # Start partway up the ladder (a rung name), e.g. to exercise later rungs.
    # Empty = the bottom rung, which is what every real run should use.
    start_phase: str = ""
    refer_min_success: float = 0.55    # vs 1/n_candidates by chance
    trade_min_success: float = 0.15
    min_success_over_chance: float = 2.0
    min_topsim_over_null: float = 0.10
    min_channel_transfer: float = 0.25
    # per-role bars in refer-swap and refer-mutual
    min_positional_structure: float = 0.15   # mean slot->field strength, each role
    mutual_min_report: float = 0.30          # each role reports the other's tuple
    mutual_min_success: float = 0.10         # both do, in the same round
    mutual_qty_tol: int = 0                  # quantity must be reported exactly
    # each role, each field (variety, quantity, quality): share of headroom over
    # a muted channel, so no field can ride on the others
    min_field_transfer: float = 0.25
    order_min_success: float = 0.50          # farmer fills the buyer's order exactly
    # swap and mutual: mean over fields of I(message; field) / H(field), chance-
    # corrected, for each describing role
    min_field_coverage: float = 0.30
    # Share of lineup rounds that are "hard": one anchor plus near misses of it,
    # each differing in one field, target uniform among them. At 0.75, quantity
    # is needed to pick the target in ~46% of rounds (31% with independent
    # candidates), variety in ~36%, quality in ~33%.
    hard_distractor_frac: float = 0.75
    # Share of (variety, quantity, quality) combinations never used in the lineup
    # rungs, so describing one is a test of productivity, not recall.
    holdout_tuple_frac: float = 0.1
    # If a rung never hits threshold inside its budget, advancing anyway would
    # just rebuild the same failure one rung up.  "stop" ends the run and writes
    # the report; "hold" keeps training and flags it loudly.
    on_stall: str = "stop"


# --------------------------------------------------------------------------
# Population / lifecycle  (spec 3)
# --------------------------------------------------------------------------
@dataclass
class PopulationConfig:
    n_farmers: int = 8
    n_buyers: int = 8
    # A community can be founded small and grow to n_farmers / n_buyers. With
    # founders > 0 the run starts with that many of each, and once the first
    # curriculum rung has been passed a newcomer of each role joins every
    # ``grow_every`` episodes. Newcomers are born like any newborn -- random
    # weights, then the transmission bottleneck on the community's transcripts --
    # so they learn the existing language rather than inventing one. Measured:
    # six speakers and six listeners from random weights kept six private,
    # drifting codes and the lineup never left chance in 200k episodes, where two
    # and two invent one in ~80k. 0 = start at full size.
    founders_farmers: int = 0
    founders_buyers: int = 0
    grow_every: int = 10_000
    turnover: bool = True                 # master switch for birth/death (spec 9)
    lifespan_min: int = 6000              # in lifespan_unit (below)
    lifespan_max: int = 12000
    # What an agent's age counts. "episodes": episodes it played. "updates":
    # training updates it took part in -- how much it has actually learned, the
    # same at any batch size or population size. With "episodes", a 4,096-episode
    # batch shared by 2 founders ages each founder 2,048 episodes per update, 16x
    # the CPU runs': founders lived ~50 updates and never learned the lineup.
    lifespan_unit: str = "episodes"
    # At t=0 every agent would otherwise die at the same time; stagger the first
    # cohort's lifespans so deaths are spread out rather than synchronised.
    initial_stagger: bool = True


# --------------------------------------------------------------------------
# Transmission bottleneck / iterated learning  (spec 4)
# --------------------------------------------------------------------------
@dataclass
class BottleneckConfig:
    enabled: bool = True                  # master switch (spec 9)
    # How much of the parent generation a newborn gets to see, as a fraction of
    # everything in the store.  Real children acquire essentially all of the
    # vocabulary that adults around them use regularly; loss is a marginal
    # phenomenon at the rare end, not a broad one.  A small fixed sample gets that
    # backwards -- it puts common forms at risk too.  At full coverage a form used
    # in 1% of trades still appears hundreds of times and transmits reliably,
    # while one used in 0.01% may genuinely not appear at all.  That asymmetry is
    # the thing worth modelling, and it falls out of coverage rather than a cap.
    coverage: float = 1.0
    max_samples: int = 40_000             # a ceiling for tractability, not a squeeze
    n_samples: int = 0                    # 0 = derive from coverage; >0 forces a cap
    epochs: int = 3
    batch_size: int = 256
    lr: float = 1e-3
    store_capacity: int = 40_000          # ring buffer of recent successful episodes
    only_successful: bool = True          # learn from trades that worked
    # How strongly the newborn's sample favours common meanings (addendum 2.3).
    #   1.0 = whatever the parent generation actually did, in proportion
    #         (so a newborn sees many common trades and few rare ones)
    #   0.0 = flat across meaning types, rare cases as well represented as common
    #   >1  = even more dominated by the common cases
    # This is the lever for vocabulary loss and regularisation across
    # generations, so it is a first-class knob rather than a constant.
    frequency_skew: float = 1.0
    token_loss_weight: float = 1.0
    decision_loss_weight: float = 1.0


# --------------------------------------------------------------------------
# RL training  (spec 2.3)
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # "gumbel" -- straight-through Gumbel-softmax on the message tokens, with
    #             REINFORCE retained for the (genuinely discrete, un-relaxable)
    #             trade decision.  Spec 2.3 permits this and it is the default
    #             because pure REINFORCE could not get information across the
    #             channel at this scale: see the scrambled-channel ablation.
    # "reinforce" -- score-function estimator for message tokens too.  Kept, and
    #             runnable, because the comparison is informative.
    algo: str = "gumbel"
    gumbel_tau: float = 1.5
    gumbel_tau_final: float = 0.5
    gumbel_tau_anneal_frac: float = 0.6
    # Straight-through Gumbel gives the symbol policy a gradient from the
    # listener, but *not* from the episode return -- so the per-symbol length cost
    # never reaches it and utterances run to the cap.  This mixes a score-function
    # term back in over the symbols, which is the direct path for "shorter is
    # better".  0 disables it and reproduces the babbling.
    gumbel_mix_reinforce: float = 1.0
    # Speaker-only terms (symbol cost, coining cost, convention) reach the
    # speaker's token choices through this score-function term, in the same
    # units as the task advantage. Through the Gumbel path they have no route at
    # all -- the straight-through gradient only carries what the listener did.
    shaping_reinforce: float = 0.5
    # Batch multiplier per rung, e.g. {"refer": 2}. Rungs with one short turn use
    # little memory, so a larger batch there buys lower-noise updates for almost
    # no extra time per update. Absent rungs use 1.
    rung_batch_scale: dict = field(default_factory=dict)
    # The convention bonus gets its own coefficient on the same route: it has to
    # be strong enough to seed a shared code before the task pays anything,
    # whereas the costs have to be weak enough not to silence a young channel.
    convention_reinforce: float = 1.0
    # Hindsight feedback: after each round the scored heads are also trained
    # towards the outcome (the target, the partner's meaning, the order), and the
    # gradient reaches the speaker through the straight-through channel.
    hindsight_coef: float = 1.0
    episodes: int = 200_000
    batch_size: int = 64                  # episodes per policy-gradient update
    lr: float = 3e-4
    grad_clip: float = 1.0
    value_coef: float = 0.5
    # Measured on the lineup game, everything else held fixed: 0.05 reached 0.473
    # against a 0.25 chance rate, 0.01 reached 0.618.  A large exploration bonus
    # keeps the symbol policy near-uniform long after it should have committed.
    entropy_coef: float = 0.01            # on message tokens; annealed
    entropy_coef_final: float = 0.002
    entropy_anneal_frac: float = 0.5      # fraction of the run over which it anneals
    decision_entropy_coef: float = 0.02
    decision_entropy_coef_final: float = 0.002
    normalise_adv: bool = True
    seed: int = 0

    # ---- where and how it runs -------------------------------------------
    # "auto" picks cuda when a GPU is visible and cpu otherwise, which is what you
    # want for a script that has to run on a laptop and on a cloud box unchanged.
    device: str = "auto"
    torch_threads: int = 4
    # Sample scenarios and score trades as whole batches of tensors rather than
    # one Python call per episode.  On a GPU the scalar path is the entire
    # bottleneck -- a batch of 4096 costs 4096 interpreter round trips before a
    # kernel launches.  tests/test_batched.py asserts the two agree exactly.
    vectorised: bool = True
    # bfloat16 autocast for the forward passes.  bf16 rather than fp16 because it
    # needs no loss scaling and these are tiny models where range matters more
    # than precision.  Ignored on CPU without bf16 support.
    amp: bool = False
    compile: bool = False           # torch.compile the agent networks
    # Recompute encoder activations in the backward pass instead of keeping them.
    # The Gumbel path builds one graph spanning every symbol step of an episode,
    # so activation memory grows as batch x sequence x width x symbol-steps and is
    # what limits big configurations long before parameter count does.  Costs
    # roughly 30% more compute and buys back most of that memory.
    grad_checkpoint: bool = False
    tf32: bool = True               # allow TF32 matmuls on Ampere and later
    # Episodes generated per optimiser step.  A GPU wants this an order of
    # magnitude larger than a CPU does; see configs/gpu.json.
    log_every_batches: int = 0      # 0 = quiet between checkpoints


# --------------------------------------------------------------------------
# Logging / evaluation cadence  (spec 5, 6)
# --------------------------------------------------------------------------
@dataclass
class LogConfig:
    ledger_stride: int = 1           # write every Nth episode to the trade ledger
    checkpoint_every: int = 5_000    # episodes between metric checkpoints
    summary_every: int = 5_000       # episodes between human-readable console summaries
    n_example_transcripts: int = 3
    topsim_samples: int = 200        # scenarios sampled for topological similarity
    # Topsim is O(samples^2) per agent; with a large community, probe a fixed
    # random sample of agents per role instead of every one.
    max_agents_probed: int = 8
    stability_probes: int = 32       # fixed probe meanings re-queried each checkpoint
    intelligibility_episodes: int = 400
    zeroshot_episodes: int = 600
    ablation_episodes: int = 600
    word_analysis_samples: int = 400   # messages sampled for word-unit statistics
    rare_frequent_split: float = 0.5   # quantile splitting rare from frequent meanings
    track_form_survival: bool = True   # follow specific meanings across generations
    plot: bool = True
    flush_every: int = 200           # ledger flush cadence (episodes)
    # Overwrite snapshots/latest.pt at every checkpoint (promotions always
    # snapshot). Resume with ``python -m orchard.run --resume <file>``.
    snapshot_every_checkpoint: bool = True
    # Every Nth episode is written to transcripts.txt as expected / dialogue /
    # outcome lines. 0 turns the file off.
    transcript_stride: int = 50
    # With --quiet (as cloud_run.sh runs), print one status line this often, plus
    # rung transitions, checkpoint headlines and the verdict. 0 = never.
    heartbeat_seconds: int = 60


@dataclass
class Config:
    name: str = "default"
    world: WorldConfig = field(default_factory=WorldConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Config":
        cfg = Config()
        for f in dataclasses.fields(Config):
            if f.name not in d:
                continue
            val = d[f.name]
            cur = getattr(cfg, f.name)
            if dataclasses.is_dataclass(cur) and isinstance(val, dict):
                for k, v in val.items():
                    if not hasattr(cur, k):
                        raise KeyError("unknown config key %s.%s" % (f.name, k))
                    setattr(cur, k, v)
            else:
                setattr(cfg, f.name, val)
        return cfg

    @staticmethod
    def from_json(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return Config.from_dict(json.load(fh))


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------
_BOOL = {"1": True, "0": False, "true": True, "false": False,
         "on": True, "off": False, "yes": True, "no": False}


def _tobool(s: str) -> bool:
    key = str(s).strip().lower()
    if key not in _BOOL:
        raise argparse.ArgumentTypeError("expected a boolean, got %r" % (s,))
    return _BOOL[key]


def add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None, help="path to a JSON config file")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--bottleneck", type=_tobool, default=None,
                   help="on/off: iterated-learning transmission bottleneck for newborns")
    p.add_argument("--turnover", type=_tobool, default=None,
                   help="on/off: population birth/death")
    p.add_argument("--n-farmers", type=int, default=None)
    p.add_argument("--n-buyers", type=int, default=None)
    p.add_argument("--atomic-vocab", type=int, default=None)
    p.add_argument("--max-symbols", type=int, default=None)
    p.add_argument("--symbol-cost", type=float, default=None)
    p.add_argument("--zipf-alpha", type=float, default=None)
    p.add_argument("--zipf-alpha-variety", type=float, default=None)
    p.add_argument("--gumbel-mix", type=float, default=None)
    p.add_argument("--frequency-skew", type=float, default=None)
    p.add_argument("--n-turns", type=int, default=None)
    p.add_argument("--lifespan-min", type=int, default=None)
    p.add_argument("--lifespan-max", type=int, default=None)
    p.add_argument("--bottleneck-samples", type=int, default=None)
    p.add_argument("--bottleneck-coverage", type=float, default=None)
    p.add_argument("--curriculum", type=_tobool, default=None,
                   help="on/off: the referential-then-trading curriculum")
    p.add_argument("--on-stall", type=str, default=None, choices=["hold", "stop"])
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--summary-every", type=int, default=None)
    p.add_argument("--ledger-stride", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="auto (default), cpu, cuda, or cuda:N")
    p.add_argument("--amp", type=_tobool, default=None,
                   help="on/off: bfloat16 autocast")
    p.add_argument("--compile", type=_tobool, default=None,
                   help="on/off: torch.compile the agent networks")
    p.add_argument("--vectorised", type=_tobool, default=None,
                   help="on/off: tensor world and reward (leave on for GPU)")
    p.add_argument("--algo", type=str, default=None, choices=["gumbel", "reinforce"])
    p.add_argument("--persistent-inventory", type=_tobool, default=None,
                   help="on/off: farms hold a depleting lot across market days")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override any config field, e.g. --set world.max_qty=8")


def _coerce(cur: Any, raw: str) -> Any:
    if isinstance(cur, (dict, list)):
        return json.loads(raw)            # e.g. --set 'curriculum.rung_budgets={"refer": [0, 50000]}'
    if isinstance(cur, bool):
        return _tobool(raw)
    if isinstance(cur, int):
        return int(raw)
    if isinstance(cur, float):
        return float(raw)
    return raw


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_json(args.config) if getattr(args, "config", None) else Config()

    simple = [
        ("name", cfg, "name"),
        ("episodes", cfg.train, "episodes"),
        ("batch_size", cfg.train, "batch_size"),
        ("seed", cfg.train, "seed"),
        ("lr", cfg.train, "lr"),
        ("bottleneck", cfg.bottleneck, "enabled"),
        ("turnover", cfg.population, "turnover"),
        ("n_farmers", cfg.population, "n_farmers"),
        ("n_buyers", cfg.population, "n_buyers"),
        ("atomic_vocab", cfg.channel, "atomic_vocab"),
        ("max_symbols", cfg.channel, "max_symbols"),
        ("symbol_cost", cfg.reward, "symbol_cost"),
        ("zipf_alpha", cfg.world, "zipf_alpha"),
        ("zipf_alpha_variety", cfg.world, "zipf_alpha_variety"),
        ("gumbel_mix", cfg.train, "gumbel_mix_reinforce"),
        ("frequency_skew", cfg.bottleneck, "frequency_skew"),
        ("n_turns", cfg.channel, "n_turns"),
        ("lifespan_min", cfg.population, "lifespan_min"),
        ("lifespan_max", cfg.population, "lifespan_max"),
        ("bottleneck_samples", cfg.bottleneck, "n_samples"),
        ("bottleneck_coverage", cfg.bottleneck, "coverage"),
        ("curriculum", cfg.curriculum, "enabled"),
        ("on_stall", cfg.curriculum, "on_stall"),
        ("checkpoint_every", cfg.log, "checkpoint_every"),
        ("summary_every", cfg.log, "summary_every"),
        ("ledger_stride", cfg.log, "ledger_stride"),
        ("threads", cfg.train, "torch_threads"),
        ("device", cfg.train, "device"),
        ("amp", cfg.train, "amp"),
        ("compile", cfg.train, "compile"),
        ("vectorised", cfg.train, "vectorised"),
        ("algo", cfg.train, "algo"),
        ("persistent_inventory", cfg.economy, "persistent_inventory"),
    ]
    for arg_name, section, key in simple:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(section, key, val)

    if getattr(args, "no_plot", False):
        cfg.log.plot = False

    for override in getattr(args, "set", []) or []:
        if "=" not in override or "." not in override.split("=", 1)[0]:
            raise SystemExit("--set expects SECTION.KEY=VALUE, got %r" % (override,))
        path, raw = override.split("=", 1)
        sect_name, key = path.split(".", 1)
        if not hasattr(cfg, sect_name):
            raise SystemExit("unknown config section %r" % (sect_name,))
        sect = getattr(cfg, sect_name)
        if not hasattr(sect, key):
            raise SystemExit("unknown config key %s.%s" % (sect_name, key))
        setattr(sect, key, _coerce(getattr(sect, key), raw))

    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    c = cfg.channel
    assert c.atomic_vocab >= 4, "need a real inventory of atoms"
    assert c.max_symbols >= 2, "a turn needs room for at least a short word"
    assert c.n_turns >= 2, "need at least one turn each way"
    assert cfg.world.zipf_alpha >= 0.0
    assert cfg.world.zipf_alpha_variety >= 0.0
    assert cfg.bottleneck.frequency_skew >= 0.0
    assert cfg.curriculum.n_candidates >= 2
    assert cfg.curriculum.on_stall in ("hold", "stop")
    assert cfg.population.lifespan_unit in ("episodes", "updates")
    assert 0.0 < cfg.bottleneck.coverage <= 1.0
    assert cfg.model.d_model % cfg.model.n_heads == 0
    w = cfg.world
    assert w.n_varieties >= 2 and w.n_quality >= 2 and w.max_qty >= 2
    assert 0 <= w.reservation_max_bin < w.n_price_bins
    assert 0 <= w.budget_min_bin < w.n_price_bins
    p = cfg.population
    assert p.n_farmers >= 1 and p.n_buyers >= 1
    assert p.lifespan_min <= p.lifespan_max
    assert cfg.train.algo in ("gumbel", "reinforce")
    assert cfg.train.device == "auto" or cfg.train.device.split(":")[0] in ("cpu", "cuda")
    assert cfg.economy.episodes_per_day >= 1
    assert cfg.economy.season_days >= 1
