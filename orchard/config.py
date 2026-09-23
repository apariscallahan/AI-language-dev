"""Configuration for the orchard emergent-language simulation.

Everything tunable lives here.  Nothing in the simulation should hardcode a
population size, vocabulary size, message length, lifespan, etc. -- the whole
scientific point of the project (spec section 7) is running the *same* code with
different settings and comparing, so all of it is config-driven and serialisable
to JSON.

**These defaults are the configuration -- the only one.** Every run uses them,
on a GPU or on a CPU, sizes included: the same agents, the same brains, the same
batch, the same arithmetic (fp32 everywhere; there is no GPU-only precision
mode). A GPU runs it faster; that is the only difference. A CPU check therefore
tests exactly what a GPU run does, which is why the sizes are ones a CPU can
test: 2 + 2 founders growing to 6 + 6, 48-wide brains, batch 256 -- the scale at
which the CPU runs that worked were made.

A run may change only how long it runs, its seed, its device and its output
(``RUN_KEYS``). Anything else is printed as a method change in the run header and
the report. ``configs/`` holds only named experiments (``EXPERIMENT_KEYS``), each
changing the few settings that define it; ``tests/test_config.py`` enforces both.

Anything whose meaning is "an amount of learning" -- rung budgets, how often
promotion is checked, checkpoints, annealing, community growth, lifespans -- is
counted in **training updates**, never episodes. An episode count means a
different amount of learning at every batch size and every population size;
counting lifespans in episodes once killed each founder after ~50 updates on a
GPU because 2 founders shared a 4,096-episode batch.
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
    # A thing to talk about is a lot: (fruit, colour, quality, quantity, price).
    # The first three make a *combination* -- 4 x 4 x 4 = 64 of them, of which a
    # quarter are never trained on (holdout_combo_frac). Fruit and colour are
    # separate fields on purpose: a code can only describe a combination it has
    # never seen if it names them separately, which is what makes an adjective
    # worth inventing. Quantity (0 to max_qty) and price (n_price_bins) are named
    # in the naming rungs too, so no trading rung ever has to invent a word.
    # The three fields are the same size on purpose. It lets the held-out set be
    # a Latin square -- one quality withheld from every (fruit, colour) lot, one
    # colour from every (fruit, quality), one fruit from every (colour, quality)
    # -- so a lineup that varies one field always has exactly three trainable
    # candidates. With unequal fields some lineups contain a combination that is
    # never anyone's target, and a guesser can rule it out without listening.
    n_varieties: int = 4              # fruit types: APPLE / BANANA / PEAR / PLUM
    n_colors: int = 4                 # RED / YELLOW / GREEN / PURPLE
    n_quality: int = 4                # LOW / MED / HIGH / PRIME
    max_qty: int = 8                  # quantities 1..max_qty
    n_price_bins: int = 6             # price discretised onto a grid
    price_min: float = 1.0
    price_step: float = 0.5

    # Farmer stock is sampled uniformly in [1, max_qty].
    # Farmer reservation (cost) price bin sampled in [0, reservation_max_bin].
    reservation_max_bin: int = 4
    # Buyer budget ceiling bin sampled in [budget_min_bin, n_price_bins-1].
    budget_min_bin: int = 1

    # Probability that a farm stocks any given variety at all.  Together with the
    # skewed-but-independent marginals in World, this is what keeps roughly half
    # of encounters worth doing WITHOUT making either side's private state
    # predictable from the other's -- see the note at the top of world.py.
    # Per (fruit, colour) lot. The barn has n_varieties x n_colors of them, so a
    # shopper's exact request is usually -- not always -- servable: 0.85 puts the
    # share of rounds where a deal is possible at ~0.68, inside the 55-71% band
    # the earlier sweeps found workable (higher makes "just accept" pay), and
    # leaves the reasons a deal falls through spread: no such lot 0.49,
    # quality 0.22, too few 0.21, price 0.09.
    p_stocked: float = 0.85
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
    # Share of (fruit, colour, quality) combinations never trained on anywhere,
    # so describing one is a test of productivity, not recall. Rounded to whole
    # qualities per (fruit, colour) lot, which is what makes the set balanced:
    # every fruit, colour and quality appears equally often in training and in
    # the held-out set (see world.ComboHoldout).
    holdout_combo_frac: float = 0.25
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
    zipf_alpha: float = 0.3            # over requested quantities
    zipf_alpha_variety: float = 0.0    # over requested varieties

    @property
    def price_values(self) -> list[float]:
        return [self.price_min + i * self.price_step for i in range(self.n_price_bins)]

    @property
    def variety_names(self) -> list[str]:
        base = ["APPLE", "BANANA", "PEAR", "PLUM", "CHERRY", "MELON", "FIG", "MANGO",
                "PEACH", "QUINCE", "LEMON", "GRAPE"]
        # Labels are for humans reading reports; the agents only ever see indices.
        return (base + ["F%d" % i for i in range(len(base), self.n_varieties)])[: self.n_varieties]

    @property
    def color_names(self) -> list[str]:
        base = ["RED", "YELLOW", "GREEN", "PURPLE", "ORANGE", "BROWN"]
        return (base + ["C%d" % i for i in range(len(base), self.n_colors)])[: self.n_colors]

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
    # Meaningless atoms; ids 0 .. atomic_vocab-1. There are 27 field values to
    # name (4 fruits, 4 colours, 4 qualities, 9 quantities, 6 prices); with 32
    # atoms every value *can* have an atom of its own, and whether a population
    # reuses atoms across fields (homonyms told apart by context) or builds
    # multi-atom words instead is something to measure, not to force. Fewer
    # atoms than values makes duality of patterning necessary; that is the
    # `duality` experiment in configs/, not the baseline.
    atomic_vocab: int = 32
                               # HYPHEN = atomic_vocab       joins atoms into a word
                               # SPACE  = atomic_vocab + 1   separates words
                               # END    = atomic_vocab + 2   ends the utterance
                               # PAD    = atomic_vocab + 3   never emitted; fills the slot
    max_symbols: int = 24      # buffer per turn; generous on purpose (the cost sets length)
    n_turns: int = 4           # alternating turns per negotiation; buyer speaks first
    enforce_word_grammar: bool = True   # atoms and HYPHEN/SPACE must alternate
    # Whether a turn may be empty. It may not. Measured on the GPU, speakers with
    # no length cost at all went silent in 24-55% of lineup rounds while the code
    # was forming: silence is the shortest message there is -- one decision, with
    # nothing after it to get wrong -- so {silence, a3, a7, a12} names four fruits
    # more reliably than any spoken code. But silence is also exactly what the
    # muted control feeds the listener (END, then nothing), and a word identical
    # to the control cannot be measured by it: whatever silence meant would count
    # as zero in every channel number. So a turn is at least one word, the way a
    # word is atoms joined by hyphens -- a property of the medium, not a rule
    # about which words to use. The muted control keeps its meaning: nothing a
    # speaker can say sounds like it.
    allow_silence: bool = False

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
    d_model: int = 48
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 96
    dropout: float = 0.0
    # One cross-attention step from every hidden state to the barn's rows, keyed
    # on each row's (fruit, colour) and valued on its (quality, stock), added to
    # the hidden state through a zero-initialised projection. It is the barn's
    # analogue of the lineup's candidate pointer: "compare what was heard against
    # each row" made directly expressible, instead of a lookup the network has
    # to discover among sixteen shuffled rows. Measured, supervised, with the
    # answer given: without it the stock of the asked-for lot stayed at the
    # base rate after 800 steps (0.27-0.34; quality 0.5); with it both reached
    # 1.00 by step 500. Only active when the observation is a barn.
    barn_lookup: bool = True


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
    # What a wrong guess is still paid, per field it shares with the target. The
    # lineup was the one rung in the ladder with no partial credit -- every other
    # one pays `decode` per field -- so a guess that got two fields of three was
    # worth exactly as much as one that got none, and nothing rewarded a message
    # for narrowing the field down. Measured on the run that stalled at
    # `name-all`: each field named on its own scored 0.74 / 0.87 / 0.97, all
    # three at once 0.60, with utterances 1.5 words long where three were needed.
    # This is the staircase from "one field" to "all of them".
    refer_partial: float = 0.45
    refer_miss: float = -0.1

    decode: float = 0.45            # I worked out your situation
    understood: float = 0.45        # you worked out mine
    belief_qty_tol: int = 1         # counts as read correctly if within this
    belief_price_tol: int = 1

    success: float = 1.5            # viable deal, both accept, beliefs agree, feasible
    correct_no_deal: float = 0.25   # not viable, both reject  (the right answer)
    agree_per_dim: float = 0.05     # the two agents' beliefs match, per dimension
    correct_per_dim: float = 0.20   # this agent's deal decision is right, per dimension
    judgement: float = 0.25         # this agent's accept/reject matches whether a deal
                                    # was actually possible -- the fourth comprehension
                                    # dimension, and the one that trains the accept head
    one_sided_accept: float = -0.10 # one accepts, one rejects
    missed_deal: float = -0.10      # viable but both rejected
    bad_deal: float = -0.15         # not viable but both accepted
    # Length costs, split so that *words* are pressed to be short while an
    # utterance may hold several of them. A fused name for a whole (fruit,
    # colour, quality) needs one long word; a compositional one needs two or
    # three short words, and must not be taxed for it.
    atom_cost: float = 0.03         # per atom after the first in each word
    word_cost: float = 0.005        # per word -- deliberately much smaller
    # Per emitted symbol -- atoms, hyphens and spaces all count (addendum 2.1).
    # A small flat charge on top of the two above, because the cheapest way to
    # repeat a word is with spaces, and a speaker with the word cost alone
    # repeated one word twelve times to the buffer end ("a1 a1 a1 ...": 0.06
    # under the word cost, 0.29 with this). It is also part of the Zipf
    # mechanism (2.2): paid once per episode, so meanings that come up often pay
    # it far more often and feel far more pressure to shorten. A five-word
    # request costs 0.09 under it, against a task reward above 1.
    symbol_cost: float = 0.01
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
    convention: float = 0.30
    convention_min_support: int = 12
    # Whether the convention bonus waits for a working channel like the costs do.
    # Measured: ungated, even strongly weighted, it raised coherence among six
    # speakers from random weights only to ~0.2 and did not get their lineup off
    # chance -- a population that size needs founding small (see
    # population.founders_*), after which the bonus is fully on anyway.
    convention_gated: bool = True
    # The rung from which the speaker pays for length, rarity and coining, and is
    # paid for agreeing. Off below it. Measured: with the costs on from the
    # second rung, the population collapsed onto one one-atom word (coherence
    # 1.000, 1.0 atoms per word, 17 distinct words among 15 speakers) and colour
    # never left chance -- the cheapest way to agree, before a word for colour
    # exists, is for everyone to say the same short nothing.
    # The rule that follows: the costs stay off while a rung still has to
    # *invent* a word, and come on at the first rung that only reuses them.
    # Every field now has a naming rung, so that is `mutual` -- and even there
    # they wait for the rung to be working (`costs_ramp_trigger`) and are ramped
    # in (`costs_ramp_updates`): switched fully on at the transition, the mutual
    # rung climbed at half the pace of one with them off.
    costs_from_rung: str = "mutual"
    # Within the rung named above, the costs come on only once the rung's
    # rolling success has reached this multiple of its promotion floor, and then
    # rise linearly from 0 to full over `costs_ramp_updates` updates; from the
    # next rung on they are simply on. 0 turns the trigger off (on at once).
    costs_ramp_trigger: float = 1.0
    costs_ramp_updates: int = 200
    # The rung from which the convention bonus pays a speaker for using the
    # community's word for a meaning -- separately from the costs above, because
    # it is a pressure to *agree*, not to economise, and it cannot punish
    # inventing: a form only counts once it has `convention_min_support` recent
    # uses behind it. It comes on at the last naming rung, when every word
    # exists: the founders keep a dialect each through the single-field rungs
    # (nobody dies there), and something has to pay the two of them, and then
    # the community that arrives at `mutual`, to settle on one word per meaning.
    # The rarity cost stays with the costs: it charges a *new* word.
    convention_from_rung: str = "name-all"
    # How many other meanings' conventions a form is contrasted against when the
    # bonus is computed (a fixed sample per batch): the bonus pays similarity to
    # this meaning's modal form *minus* similarity to theirs, so one form for
    # everything earns nothing. More is a steadier baseline at more cost.
    convention_contrast_samples: int = 16
    # How "recent" the population's recent usage is, in training updates. (It
    # was 20,000 episodes: ~80 updates at the CPU runs' batch of 256, but only
    # ~5 at a GPU batch of 4,096 -- the coining cost and convention bonus were
    # chasing a 16x shorter memory of the language on the GPU.)
    usage_half_life_updates: int = 80


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
    n_candidates: int = 3            # lineup size: every field has at least 3 values

    # ---- promotion, on evidence rather than on a schedule -----------------
    # (minimum, maximum) training updates per rung. A rung that meets its
    # criteria after its minimum is left at the next check; one that reaches its
    # maximum without meeting them ends the run with a report (``on_stall``).
    # The CPU runs took the lineup off at ~550 updates. Rungs not named use
    # ``default_rung_updates``.
    rung_budget_updates: dict = field(default_factory=lambda: {
        # The first code forms suddenly and late -- ~300-600 updates on the CPU
        # runs, 525 and 1,525 on two GPU runs -- so the first rung gets the room.
        "name-fruit": [80, 2000],
        "name-color": [80, 1500],
        "name-quality": [80, 1500],
        "name-quantity": [80, 1500],
        "name-price": [80, 1500],
        "name-all": [80, 2500],
        "mutual": [80, 3500],
        # The report rungs in the trading world: every word is inherited, so
        # each should be far cheaper than the naming rung that invented it.
        "order": [80, 1500],
        "offer": [80, 2500],
        "judge": [80, 2000],
        "haggle": [80, 3500],
        "bargain": [80, 3500],
        "market": [80, 10**9],
    })
    default_rung_updates: list = field(default_factory=lambda: [80, 2500])
    # Promotion is checked this often -- a light probe of just the evidence the
    # rung needs -- rather than only at the (much heavier) full checkpoints.
    check_every_updates: int = 25
    # Start partway up the ladder (a rung name), e.g. to exercise later rungs.
    # Empty = the bottom rung, which is what every real run should use.
    start_phase: str = ""
    refer_min_success: float = 0.45    # vs 1/n_candidates by chance
    # Held-out combinations are never trained on, so success on them is the test
    # of whether the code has reusable parts rather than one name per thing. A
    # naming rung is not passed until held-out success reaches this share of
    # success on trained combinations (and is itself clear of chance): a fused
    # code scores at chance here however well drilled it is.
    min_holdout_ratio: float = 0.60
    trade_min_success: float = 0.15
    min_success_over_chance: float = 2.0
    min_topsim_over_null: float = 0.10
    min_channel_transfer: float = 0.25
    # per-role bars in the swap and mutual rungs
    min_positional_structure: float = 0.15   # mean slot->field strength, each role
    # Each role reports the other's whole lot -- five fields exactly -- in
    # `mutual`; the per-field bars (min_field_transfer) are the evidence, this
    # conjunction is the floor. Was 0.30 for a three-field thing.
    mutual_min_report: float = 0.25
    mutual_min_success: float = 0.08         # both do, in the same round
    # The first rung at which the population is split into farmers and buyers.
    # Below it everyone is one pool speaking one language, taking both sides of
    # the lineup; at the split each agent is copied into a farmer and a buyer,
    # so both roles start out fluent in the same language.
    # The rung where the one pool becomes farmers and buyers. Everything below
    # it is one language in two seats -- the request rungs included, since they
    # run in both directions (`quote` has the buyer saying prices, `offer` the
    # farmer) and one pool learns both from the same words. The split exists so
    # the two sides can diverge in *strategy*, which only starts to matter where
    # selling and buying pay differently: `haggle`.
    split_roles_at: str = "haggle"
    # each role, each reported field: share of headroom over a muted channel,
    # so no field can ride on the others
    min_field_transfer: float = 0.25
    # In a report rung, the fields the rung introduced have to arrive *together*
    # at least this often (five of them exactly, in `order`), on top of the
    # per-field bars and a real gain over silence.
    order_min_success: float = 0.25
    # swap and mutual: mean over fields of I(message; field) / H(field), chance-
    # corrected, for each describing role
    min_field_coverage: float = 0.30
    # Share of open lineup rounds that are "hard": one anchor plus near misses of
    # it, each differing in one field (a different field each), target uniform
    # among them, so no field can ride on the others. With three candidates at
    # most two fields can decide a round, so with five fields each is the one
    # that decides in roughly a quarter of hard rounds; the remaining rounds
    # draw the candidates independently and are easy. Was 0.75 with three
    # fields, where each field decided about a third of the rounds.
    hard_distractor_frac: float = 0.9
    # If a rung never hits threshold inside its budget, advancing anyway would
    # just rebuild the same failure one rung up.  "stop" ends the run and writes
    # the report; "hold" keeps training and flags it loudly.
    on_stall: str = "stop"


# --------------------------------------------------------------------------
# Population / lifecycle  (spec 3)
# --------------------------------------------------------------------------
@dataclass
class PopulationConfig:
    n_farmers: int = 6
    n_buyers: int = 6
    # A community can be founded small and grow to n_farmers / n_buyers. With
    # founders > 0 the run starts with that many of each, and once the first
    # curriculum rung has been passed a newcomer of each role joins every
    # ``grow_every_updates`` updates. Newcomers are born like any newborn -- random
    # weights, then the transmission bottleneck on the community's transcripts --
    # so they learn the existing language rather than inventing one. Measured:
    # six speakers and six listeners from random weights kept six private,
    # drifting codes and the lineup never left chance in 200k episodes, where two
    # and two invent one in ~80k. 0 = start at full size.
    founders_farmers: int = 2
    founders_buyers: int = 2
    # The rung from which newcomers start arriving. Every newborn is taught from
    # the store of what the community has said, so growing during a rung whose
    # words do not exist yet fills the community with apprentices of a code that
    # is about to be replaced: the GPU run grew 2 -> 15 across the colour rung
    # and stayed at chance throughout. The founders take the naming rungs; the
    # community arrives to inherit a language that already works.
    grow_from_rung: str = "mutual"
    # The rung from which agents start dying of old age, which is the same one
    # newcomers start arriving in. Turnover exists to force a code a stranger can
    # learn; while the founders are still inventing it there is no stranger and
    # nothing to transmit, and a death costs half of a two-agent pool. Measured
    # on the run that stalled: six replacements in 3,200 updates, the first at
    # update 205 -- before the first code had formed -- and success rose after
    # each newborn settled and decayed between.
    turnover_from_rung: str = "mutual"
    grow_every_updates: int = 40
    turnover: bool = True                 # master switch for birth/death (spec 9)
    # In training updates the agent took part in: how much it has learned, the
    # same at any batch size or population size.
    lifespan_min: int = 900
    lifespan_max: int = 1600
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
    # The share of the (fruit, colour, quality) combinations in its sample that a
    # newborn is *not* shown at all. It has to name those from the parts it did
    # see, which is what makes the bottleneck a pressure towards a language
    # built from reusable parts rather than one name per thing. 0 shows a
    # newborn every combination the store holds (a near-clone).
    meaning_holdout: float = 0.25
    token_loss_weight: float = 1.0
    decision_loss_weight: float = 1.0


# --------------------------------------------------------------------------
# RL training  (spec 2.3)
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Training is straight-through Gumbel-softmax on the message tokens with
    # REINFORCE on the (genuinely discrete) decisions -- orchard/gumbel.py, the
    # only training path. Pure REINFORCE could not get information across the
    # channel at this scale (see the scrambled-channel ablation) and was removed
    # rather than left to drift out of date.
    gumbel_tau: float = 1.5
    gumbel_tau_final: float = 0.5
    tau_anneal_updates: int = 1000
    # Whether the temperature and entropy anneals count updates *in the current
    # rung* rather than since the run began. They ran once, globally, and the
    # ladder has since grown to thirteen rungs: everything was at its floor by
    # update 1,000, so `name-all` -- which starts around 2,000 and has to find
    # three-word utterances where one used to do -- explored nothing at all. On
    # the run that stalled there, the only new forms came from newborns, and
    # success rose each time one settled and decayed in between.
    anneal_per_rung: bool = True        # temperature reaches its final value here
    # Straight-through Gumbel gives the symbol policy a gradient from the
    # listener, but *not* from the episode return -- so the per-symbol length cost
    # never reaches it and utterances run to the cap.  This mixes a score-function
    # term back in over the symbols, which is the direct path for "shorter is
    # better".  0 disables it and reproduces the babbling.
    gumbel_mix_reinforce: float = 0.1
    # Speaker-only terms (symbol cost, coining cost, convention) reach the
    # speaker's token choices through this score-function term, in the same
    # units as the task advantage. Through the Gumbel path they have no route at
    # all -- the straight-through gradient only carries what the listener did.
    shaping_reinforce: float = 0.2
    # Batch multiplier per rung, e.g. {"refer": 2}. Absent rungs use 1. Empty:
    # every rung runs at the batch the lineup was shown to form a code at.
    rung_batch_scale: dict = field(default_factory=dict)
    # The convention bonus gets its own coefficient on the same route: it has to
    # be strong enough to seed a shared code before the task pays anything,
    # whereas the costs have to be weak enough not to silence a young channel.
    convention_reinforce: float = 1.0
    # Hindsight feedback: after each round the scored heads are also trained
    # towards the outcome (the target, the partner's meaning, the order), and the
    # gradient reaches the speaker through the straight-through channel.
    hindsight_coef: float = 1.0
    # ...but only from this rung up. While no code exists yet, a listener told
    # the answer learns, correctly, that the messages carry nothing: it spreads
    # its guesses evenly (choice-logit spread 0.5 -> 0.17 in 100 updates) and
    # the speaker's gradient through it dies with it. Measured on the lineup,
    # CPU and GPU alike: with hindsight on from the start the code never formed
    # (chance after 2,500 updates); without it, it formed at ~550 updates. So
    # every rung where a word has to form from nothing -- all six naming rungs --
    # runs without it, and it joins at `mutual`, where every word exists and the
    # listener's job is to put five of them into five heads.
    hindsight_from_rung: str = "mutual"
    episodes: int = 6_000_000             # the run's ceiling (~23k updates); rung budgets stop it earlier
    batch_size: int = 256                 # episodes per update (x rung_batch_scale)
    lr: float = 3e-4
    grad_clip: float = 1.0
    value_coef: float = 0.5
    # Measured on the lineup game, everything else held fixed: 0.05 reached 0.473
    # against a 0.25 chance rate, 0.01 reached 0.618.  A large exploration bonus
    # keeps the symbol policy near-uniform long after it should have committed.
    entropy_coef: float = 0.01            # on message tokens; annealed
    entropy_coef_final: float = 0.002
    entropy_anneal_updates: int = 800     # entropy bonuses reach their final value here
    decision_entropy_coef: float = 0.02
    decision_entropy_coef_final: float = 0.002
    normalise_adv: bool = True
    seed: int = 0

    # ---- where and how it runs -------------------------------------------
    # "auto" picks cuda when a GPU is visible and cpu otherwise, which is what you
    # want for a script that has to run on a laptop and on a cloud box unchanged.
    device: str = "auto"
    torch_threads: int = 4
    # Everything runs in fp32 on every device: there is no bf16 or TF32 mode (both
    # were removed so a GPU can never compute something a CPU check does not; a
    # code forms here from ~0.02-logit signals).
    #
    # Recompute encoder activations in the backward pass instead of keeping them.
    # Memory only -- tests/test_config.py checks the update is the same -- so a
    # run may switch it (it is a RUN_KEY). Not needed at this size (under 1 GB);
    # it is what made batch 4,096 fit on a 24 GB card.
    grad_checkpoint: bool = False


# --------------------------------------------------------------------------
# Logging / evaluation cadence  (spec 5, 6)
# --------------------------------------------------------------------------
@dataclass
class LogConfig:
    ledger_stride: int = 500         # write every Nth episode to the trade ledger
    checkpoint_every_updates: int = 100   # full metric checkpoint + report + snapshot
    n_example_transcripts: int = 3
    topsim_samples: int = 200        # scenarios sampled for topological similarity
    # Topsim is O(samples^2) per agent; with a large community, probe a fixed
    # random sample of agents per role instead of every one.
    max_agents_probed: int = 8
    stability_probes: int = 32       # fixed probe meanings re-queried each checkpoint
    intelligibility_episodes: int = 1024
    zeroshot_episodes: int = 1024
    ablation_episodes: int = 1024
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
    transcript_stride: int = 2000
    # With --quiet (as cloud_run.sh runs), print one status line this often, plus
    # rung transitions, checkpoint headlines and the verdict. 0 = never.
    heartbeat_seconds: int = 60


@dataclass
class Config:
    name: str = "orchard"
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
    def from_dict(d: dict[str, Any], allow_legacy: bool = False) -> "Config":
        """Build a config from (a subset of) its dict form.

        ``allow_legacy`` skips keys that older versions wrote (see
        ``LEGACY_KEYS``) -- for reading an old run's snapshot. Otherwise such a
        key is an error that names its replacement, so an old config file cannot
        silently lose a setting.
        """
        cfg = Config()
        for f in dataclasses.fields(Config):
            if f.name not in d:
                continue
            val = d[f.name]
            cur = getattr(cfg, f.name)
            if dataclasses.is_dataclass(cur) and isinstance(val, dict):
                for k, v in val.items():
                    key = "%s.%s" % (f.name, k)
                    if key in LEGACY_KEYS:
                        if allow_legacy:
                            continue
                        raise KeyError("%s is no longer a setting: %s" % (key, LEGACY_KEYS[key]))
                    if not hasattr(cur, k):
                        raise KeyError("unknown config key %s" % key)
                    setattr(cur, k, v)
            else:
                setattr(cfg, f.name, val)
        return cfg

    @staticmethod
    def from_json(path: str, allow_legacy: bool = False) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return Config.from_dict(json.load(fh), allow_legacy=allow_legacy)


# Rungs older ladders had. Their budgets are ignored; their snapshots do not load
# (the observation layout changed with them).
RETIRED_RUNGS = frozenset({"refer", "refer-swap", "refer-mutual", "ask-qty", "quote"})

# Settings older versions had, and what replaced them. Everything that means an
# amount of learning moved from episodes to training updates.
_UPDATES = "counted in training updates now; use %s"
LEGACY_KEYS = {
    "curriculum.rung_budgets": _UPDATES % "curriculum.rung_budget_updates",
    "curriculum.min_episodes_per_phase": _UPDATES % "curriculum.default_rung_updates",
    "curriculum.max_episodes_per_phase": _UPDATES % "curriculum.default_rung_updates",
    "curriculum.check_every": _UPDATES % "curriculum.check_every_updates",
    "population.grow_every": _UPDATES % "population.grow_every_updates",
    "population.lifespan_unit": "lifespans are always in training updates",
    "reward.usage_half_life": _UPDATES % "reward.usage_half_life_updates",
    "train.gumbel_tau_anneal_frac": _UPDATES % "train.tau_anneal_updates",
    "train.entropy_anneal_frac": _UPDATES % "train.entropy_anneal_updates",
    "log.checkpoint_every": _UPDATES % "log.checkpoint_every_updates",
    "log.summary_every": "the console summary comes with each checkpoint",
    "train.log_every_batches": "the heartbeat (log.heartbeat_seconds) replaced it",
    "train.algo": "Gumbel-softmax is the only training path",
    "train.vectorised": "the tensor world is the only training path",
    "train.compile": "torch.compile was never wired in",
    "curriculum.holdout_tuple_frac": "renamed: world.holdout_combo_frac (a third of the "
                                     "(fruit, colour, quality) combinations)",
    "world.holdout_frac": "renamed: world.holdout_combo_frac",
    "curriculum.mutual_qty_tol": "the mutual rung reports (fruit, colour, quality), "
                                 "each exactly; there is no quantity in a thing",
    "train.amp": "removed: every device computes in fp32, so a CPU check is a GPU run",
    "train.tf32": "removed: every device computes in fp32, so a CPU check is a GPU run",
}


# --------------------------------------------------------------------------
# What a run may change, in two tiers
# --------------------------------------------------------------------------
# How long, which seed, which device, how much output. Nothing here changes
# what an agent sees, learns from or is judged on.
RUN_KEYS = frozenset({
    "name", "train.episodes", "train.seed", "train.device", "train.torch_threads",
    "train.grad_checkpoint",                 # memory only; same update (tested)
    "curriculum.on_stall", "curriculum.start_phase",
    "log.ledger_stride", "log.transcript_stride", "log.heartbeat_seconds",
    "log.plot", "log.flush_every", "log.snapshot_every_checkpoint",
    "log.n_example_transcripts",
})

# **Scale**: how big, not what. A GPU can afford a larger community, wider
# brains and bigger batches than a laptop, and a preset in ``configs/`` may say
# so -- but it changes nothing about the world, the ladder, the rewards or the
# schedules, all of which are counted in training updates and therefore mean the
# same thing at any batch size. The run header prints the scale it is running at
# next to the method line, so the two can never be confused, and
# ``tests/test_config.py`` checks that every preset leaves the method alone.
SCALE_KEYS = frozenset({
    "population.n_farmers", "population.n_buyers",
    "model.d_model", "model.n_layers", "model.n_heads", "model.d_ff",
    "train.batch_size", "train.rung_batch_scale",
    "bottleneck.batch_size",
    "log.checkpoint_every_updates", "log.intelligibility_episodes",
    "log.zeroshot_episodes", "log.ablation_episodes", "log.topsim_samples",
    "log.stability_probes", "log.max_agents_probed", "log.word_analysis_samples",
})

# Named experiments in configs/, and the settings each is allowed to change.
# They are method changes by design, and are reported as such.
EXPERIMENT_KEYS = {
    # more meanings than atoms, so an atom cannot stand for a whole meaning
    "duality": frozenset({"world.n_varieties", "channel.atomic_vocab",
                          "channel.max_symbols"}),
}


def flat_keys(d: dict[str, Any], prefix: str = "") -> list[str]:
    """``{"train": {"lr": 1}}`` -> ``["train.lr"]`` (dict-valued fields stay whole)."""
    out = []
    for k, v in d.items():
        key = prefix + k
        sect = getattr(Config(), k, None) if not prefix else None
        if isinstance(v, dict) and dataclasses.is_dataclass(sect):
            out.extend(flat_keys(v, key + "."))
        else:
            out.append(key)
    return out


def method_changes(cfg: "Config") -> dict[str, tuple[Any, Any]]:
    """Every setting outside ``RUN_KEYS`` and ``SCALE_KEYS`` that differs.

    ``{"reward.symbol_cost": (default, this run's)}``. This is what the run
    header and the report call a *method* change: something that alters what is
    simulated rather than how big it is. Size differences are reported
    separately by :func:`scale_changes`, so a big run and a small one can be
    compared, and neither can quietly become a different experiment.
    """
    base, mine = Config().to_dict(), cfg.to_dict()
    out = {}
    for key in flat_keys(mine):
        if key in RUN_KEYS or key in SCALE_KEYS:
            continue
        a, b = base, mine
        for part in key.split("."):
            a, b = a[part], b[part]
        if a != b:
            out[key] = (a, b)
    return out


def scale_changes(cfg: "Config") -> dict[str, tuple[Any, Any]]:
    """Every size this run differs from the reference scale in."""
    base, mine = Config().to_dict(), cfg.to_dict()
    out = {}
    for key in sorted(SCALE_KEYS):
        a, b = base, mine
        try:
            for part in key.split("."):
                a, b = a[part], b[part]
        except KeyError:
            continue
        if a != b:
            out[key] = (a, b)
    return out


def scale_summary(cfg: "Config") -> str:
    """One line: the community, the brain and the batch this run uses."""
    p, m, t = cfg.population, cfg.model, cfg.train
    pool = max(p.n_farmers, p.n_buyers)
    start = max(p.founders_farmers, p.founders_buyers) or pool
    return ("pool of %d growing to %d, then %d + %d once trading starts; "
            "d=%d x %d layers, batch %s, %s episodes"
            % (start, pool, p.n_farmers, p.n_buyers, m.d_model, m.n_layers,
               "{:,}".format(t.batch_size), "{:,}".format(t.episodes)))


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
    p.add_argument("--checkpoint-every-updates", type=int, default=None)
    p.add_argument("--ledger-stride", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="auto (default), cpu, cuda, or cuda:N")
    p.add_argument("--persistent-inventory", type=_tobool, default=None,
                   help="on/off: farms hold a depleting lot across market days")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override any config field, e.g. --set world.max_qty=8")


def _coerce(cur: Any, raw: str) -> Any:
    if isinstance(cur, (dict, list)):
        return json.loads(raw)   # e.g. --set 'curriculum.rung_budget_updates={"refer": [80, 4000]}'
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
        ("checkpoint_every_updates", cfg.log, "checkpoint_every_updates"),
        ("ledger_stride", cfg.log, "ledger_stride"),
        ("threads", cfg.train, "torch_threads"),
        ("device", cfg.train, "device"),
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
    assert 0.0 < cfg.bottleneck.coverage <= 1.0
    assert cfg.model.d_model % cfg.model.n_heads == 0
    w = cfg.world
    assert w.n_varieties >= 2 and w.n_quality >= 2 and w.max_qty >= 2
    assert 0 <= w.reservation_max_bin < w.n_price_bins
    assert 0 <= w.budget_min_bin < w.n_price_bins
    p = cfg.population
    assert p.n_farmers >= 1 and p.n_buyers >= 1
    assert p.lifespan_min <= p.lifespan_max
    from .curriculum import phase_named
    phase_named(cfg, cfg.train.hindsight_from_rung)          # must name a rung
    phase_named(cfg, cfg.reward.costs_from_rung)
    phase_named(cfg, cfg.reward.convention_from_rung)
    phase_named(cfg, cfg.population.grow_from_rung)
    from .curriculum import ladder
    rungs = {p.name for p in ladder(cfg)}
    for name, (lo, hi) in dict(cfg.curriculum.rung_budget_updates).items():
        assert 0 <= int(lo) <= int(hi), "rung %s: budget must be (min, max) updates" % name
        # A budget for a rung that does not exist is silently ignored, and the
        # rung it was meant for quietly falls back to the default. Three separate
        # settings in this project outlived the rung they named. (Rungs an older
        # ladder had are tolerated so an old run's config.json still loads.)
        assert name in rungs or name in RETIRED_RUNGS, (
            "curriculum.rung_budget_updates names %r, which is not a rung: %s"
            % (name, ", ".join(sorted(rungs))))
    if cfg.curriculum.start_phase:
        assert cfg.curriculum.start_phase in rungs, (
            "curriculum.start_phase is %r, which is not a rung: %s"
            % (cfg.curriculum.start_phase, ", ".join(sorted(rungs))))
    assert cfg.curriculum.check_every_updates >= 1
    assert cfg.log.checkpoint_every_updates >= 1
    assert cfg.population.grow_every_updates >= 1
    assert cfg.train.device == "auto" or cfg.train.device.split(":")[0] in ("cpu", "cuda")
    assert cfg.economy.episodes_per_day >= 1
    assert cfg.economy.season_days >= 1
