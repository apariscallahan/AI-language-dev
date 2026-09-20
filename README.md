# Orchard: emergent language in an apple-trading world

Two populations of small neural agents — **Farmers** who grow apples and **Buyers**
who need them — have to invent a language in order to trade. Nobody starts with
one. Every agent is a randomly initialised transformer; the "words" are integer
ids into a random embedding table; the only thing that shapes them is the outcome
of trades, population turnover, and what each new generation manages to pick up
from the one before it.

**No pretrained model, no pretrained embedding, and no text corpus is used
anywhere in this project.** If a component ever looks like it needs real-world
language data, that is a design bug, not a shortcut — see `orchard/agents.py`.

This implements [`orchard_language_emergence_spec.md`](orchard_language_emergence_spec.md)
and its follow-up [`additional-improvements-1.md`](additional-improvements-1.md),
which replaces the fixed-token channel with an open vocabulary.

---

## Quick start

```bash
bash cloud_run.sh
```

on a GPU box, or, identically but slower, on any machine:

```bash
python -m orchard.run
```

**[CLOUD.md](CLOUD.md)** covers what the terminal shows, resuming, and changing
a setting. `cloud_run.sh` only checks that a GPU is visible, writes to
`runs/<UTC start time>_orchard/`, and resumes from that folder's latest snapshot
if it is run again.

### One method, chosen scale

**What is simulated lives in one place: the defaults in `orchard/config.py`** --
the world, the ladder, the rewards, the channel and every schedule. A GPU and a
CPU run the same code in the same fp32 arithmetic, with no device-specific path,
so a CPU check tests what a GPU run does.

**How big it runs is a separate, declared choice.** The presets in `configs/`
change the community, the brain, the batch, the run length and the amount of
output -- and nothing else:

| preset | community | brain | batch | episodes |
|---|---|---|---|---|
| *(none)* | 2 -> 6, then 6 + 6 | d48, 2 layers, 55k | 256 | 6M |
| `gpu_small` | 2 -> 12, then 12 + 12 | d64, 2 layers, 124k | 1,024 | 20M |
| `gpu_community` | 2 -> 32, then 32 + 32 | d96, 3 layers, 374k | 4,096 | 60M |
| `gpu_large` | 2 -> 64, then 64 + 64 | d128, 4 layers, 849k | 4,096 | 120M |

Every agent is the same shape at every scale: a pre-norm causal transformer
(GELU feed-forward at four times the model width, learned positions, one
attention mask over observation and dialogue) reading its private situation and
everything said so far, with a head per decision. Randomly initialised, always.

The run header prints the two separately -- a `scale` line and a `method` line
-- so a big run and a small one can be compared, and neither can quietly become
a different experiment. `tests/test_config.py` fails if a preset touches the
method, and checks that every preset runs the same ladder, the same world and
the same held-out set. `configs/duality.json` is a declared experiment (12
fruits against 8 atoms) and says so in its header.

**Everything that means an amount of learning is counted in training updates**
(one update = one batch): rung budgets, how often promotion is checked,
checkpoints, the temperature and entropy anneals, community growth, lifespans,
and how long the population remembers what it has been saying.

### What makes it fast on a GPU

These change speed and memory only; the same code runs on a CPU.

- **Tensor world and reward.** Scenarios are sampled and trades scored as whole
  batches of tensors (`orchard/batched.py`); per-episode Python used to cap a
  large batch at ~6,700 episodes/sec. `tests/test_batched.py` asserts the tensor
  versions agree exactly with the scalar ones in `world.py` and `env.py`, which
  remain the readable definition of the rules.
- **Fixed-stride pairings**, so each agent's slice of the batch is a constant and
  the rollout never stalls the device to ask who plays what.
- **Prefix-only embedding**, slicing soft tokens before gathering, grouping
  agents once per batch, and one host copy per batch for the bottleneck store.
- **Gradient checkpointing** (`train.grad_checkpoint`, off at this size): memory
  only, and `tests/test_config.py` checks it gives the same update. It is what
  let a 4,096 batch fit on a 24 GB card.

`python -m orchard.run --benchmark` times the light, middle and heaviest rungs at
full community size on whatever device it finds, and estimates the run's hours.

### Comparing anything: use several seeds

```bash
python sweep.py --out runs/ablation --seeds 5 --arm "bottleneck_on:" --arm "bottleneck_off:--bottleneck off"
```

**Do not draw conclusions from single runs of this simulation.** It is bimodal: a
population either finds a referential convention or it does not. Four
neighbouring conditions at 40k episodes produced 76%, 0%, 92% and 6% of the
channel headroom -- a spread that swamps any effect worth measuring. `sweep.py`
runs each arm across seeds and reports mean, spread and the per-seed values, so
the bimodality is visible rather than averaged into a misleading single number.
Use `--parallel 1` on a single GPU. (Turning the bottleneck off is a method
change, and each arm's report says so.)

### Other commands

```bash
python -m orchard.run --smoke
```

Runs the environment with scripted agents and no learning at all. It prints the
chance-level success rate (essentially zero), confirms an oracle pair can convert
every viable scenario, and shows one rendered episode.

```bash
python -m orchard.analyse --snapshot runs/<run>/snapshots/latest.pt
```

Re-measures a snapshot after the fact (older snapshots load too).

```bash
python -m unittest discover -s tests
```

---

## What makes language necessary here

Language is only needed when one party holds something the other cannot see and
cannot guess. That is built in explicitly and enforced in code:

A thing in this world is a **(fruit, colour, quality)** combination: 4 fruits,
4 colours, 4 qualities, 64 in all. The three fields are separate on purpose --
that is what makes an adjective worth inventing, because a code can only
describe a combination it has never met if it names the parts.

| Farmer privately knows | Buyer privately knows |
|---|---|
| how much of **each (fruit, colour) lot** is in the barn | which fruit, in which colour, they want |
| the quality of each lot | the minimum quality they will accept |
| the lowest price they will take | how many they need, and the most they can pay |

A deal is possible only if the barn has that fruit in that colour, in enough
quantity, at acceptable quality, within budget. **Neither agent can determine
that alone.**
Both then independently declare what they think was agreed, and the trade succeeds
only if those declarations match *each other* and describe a deal that is actually
executable. One agent being right is never enough.

### A quarter of the combinations are never trained on

Sixteen of the 64 combinations are reserved, and nothing in the project ever
trains on them: no lineup describes one, no barn stocks one, no shopper asks for
one. They are chosen as a **Latin square** -- one quality withheld from every
(fruit, colour) lot, one colour from every (fruit, quality), one fruit from every
(colour, quality) -- which makes the set balanced in every direction. Two things
follow, and both matter:

* every fruit, colour and quality still appears constantly in training, so there
  is always something to generalise *from*; what is withheld is a pairing, never
  a value;
* a lineup that varies one field always has exactly three candidates that could
  be the answer. An unbalanced set leaves lineups containing a combination that
  is never anybody's target, and a guesser can then rule it out **without
  listening** -- which is how an earlier version scored 0.42 with the channel
  muted.

Success on the reserved combinations is the productivity test, and it gates
promotion (below). A code that gives each thing its own name scores at chance
there however well it has drilled the rest; a code with reusable parts does not.

### The property everything rests on

Every farmer field is drawn independently of every buyer field. No amount of
staring at your own barn tells you what the customer wants.

This was got wrong once and it is worth recording. An earlier sampler forced
roughly half of all encounters to be compatible so that viable deals would be
common enough to learn from. That made the buyer's wanted variety predictable from
the farmer's own stock — the farmer could score 0.67 against a 0.33 base rate
without listening to anything. Worse, when a farm held only one variety, the
farmer's best answer was always "the one I have", so that dimension could never
reward listening even in principle. Farms now carry a multi-variety inventory and
nothing is coerced. `tests/test_env.py::test_knowing_one_side_does_not_predict_the_other`
exists so this cannot come back unnoticed.

### The control that cannot be fooled

Every checkpoint plays the same scenarios **three times**, with the same pairings
and the same scenarios. Only what reaches the other party changes:

| condition | what the listener hears | what it isolates |
|---|---|---|
| **intact** | the message | — |
| **scrambled** | random atoms, *same length and stopping point* | what the symbols carry |
| **muted** | silence | everything the channel is worth |

The muted condition exists because an earlier version used scrambling alone, and
scrambled accuracy sat at 0.48 in a world whose base rate was 0.33 — something was
still getting through. It was utterance **length**, which scrambling preserves and
which, with an open vocabulary, is a usable channel in its own right. Transfer is
therefore reported against silence, with scrambled-versus-muted showing how much
of the work length alone was doing.

A pair exploiting base rates rather than talking scores identically in all three
conditions. This is what caught the sampler bug above, and what the verdict in
every report leans on hardest.

---

## The channel: an open vocabulary

Following the addendum, agents do not choose from a fixed word list. They emit a
**stream of symbols**, one at a time, from

```
{ a0 … a35 }  ∪  { HYPHEN, SPACE, END }
```

- a **word** is atoms joined by `HYPHEN` — `a7-a22-a3` is one word;
- an **utterance** (one turn) is words separated by `SPACE` — `a7-a22 a3` is two;
- `HYPHEN` and `SPACE` are structural marks and mean nothing themselves, exactly
  as no atom means anything at the start.

That shape is part of the medium and is enforced at every step
(`channel.enforce_word_grammar`): after an atom the speaker must choose `HYPHEN`
(same word), `SPACE` (next word) or `END`; after a mark it must say an atom. So
every junction between two atoms is an explicit "same word / next word" choice,
and a transcript reads exactly as it was emitted. (An earlier version let bare
atoms run together into one word while `HYPHEN` did nothing, and printed hyphens
the agents had never emitted.)

*Which* atoms make words, and where words split, is entirely the agents' own.
Ideally separate words come to name separate fields -- a variety word (noun-like)
next to a quality word (adjective-like) -- and the report measures exactly that
("word classes"); nothing requires it.

The vocabulary is open — far more possible words than atoms — while the channel
stays discrete. `channel.max_symbols` (24 per turn) is a buffer, not a limit
anyone should feel: the report flags any utterance that reaches it.

### What keeps utterances short is a cost, not a rule

Length is charged **per atom after the first in a word** (`reward.atom_cost`,
0.03), plus a much smaller charge **per word** (`reward.word_cost`, 0.005).
Ending a message is free, because brevity should not be taxed.

The split is deliberate. A fused name for a whole (fruit, colour, quality) is one
long word; naming the parts is two or three short ones. Charging every symbol
equally would tax the compositional utterance for being longer overall -- so
words are pressed to be short, while saying several of them costs almost
nothing. Three atoms as one word cost 0.065; the same three atoms as two words
cost 0.040.

In the trading rungs this is also the Zipf mechanism: requests follow a Zipf-like
frequency distribution, so a meaning that comes up constantly pays its length
cost constantly, while a rare one barely pays it at all. Nothing rewards "short
words for common things" directly; it is a prediction, and `report.md` reports
the correlation rather than eyeballing it. (In the naming rungs things are drawn
uniformly, so there is nothing for length to track, and the report says so.)

---

## Closing the loop: reading, and being read

A speaker only has a reason to be informative if something it cares about depends
on having been understood. For a long time nothing did, and it was costing the
farmer side most of its signal.

Measured on the reward function directly, with no trained agents involved:

| | score from own state alone | with the other's facts | gain from listening |
|---|---|---|---|
| farmer | 1.385 | 2.428 | **1.044** of 3 |
| buyer | 2.204 | 2.428 | **0.224** of 3 |

The buyer was collecting 91% of its comprehension reward simply by restating the
want and need it already held — no listening required. And neither role had *any*
term for being understood: swap a partner between "decoded perfectly" and "ignored
the message" and the only thing that moved was the joint trade outcome.

So each agent now also states **what it believes the other party's private
situation to be** — the farmer about the buyer's shopping list, the buyer about
what is actually in the barn for the line it came for — and that statement is
scored against the truth. Two reward terms follow from it:

- `reward.decode` pays an agent for having read the other correctly;
- `reward.understood` pays an agent for having *been* read correctly.

The second is the one that was missing. It is per-message rather than per-trade,
it is symmetric, and every field it scores is one the answering agent cannot
observe, so neither term is obtainable without the channel. After the change both
roles have a comparable stake in being understood (0.211 / 0.243) and comparable
gains from listening (0.540 / 0.469, previously 1.044 / 0.224).

`tests/test_reward_loop.py` guards all of this, including a test that holds the
trade fixed and checks the reward still moves with whether the partner read you —
otherwise the term would just be trade success under another name.

## Making deals common enough to practise

Both sides are drawn fresh and independently every round; that independence is
what keeps the private information private, and it is not negotiable. But
independence alone left only **56.6%** of rounds viable, so buyers spent nearly
half their time practising correct refusals.

The obvious fix — correlate the farmer's stock with the buyer's wanted variety —
would have raised viability and destroyed the experiment, since the farmer could
then predict the request from its own barn. Instead the *marginals* were widened,
and the lever that worked best was `world.need_max_frac`: **a shop stocks more
than any one shopper asks for.** That lifts P(stock ≥ need) a long way while
leaving the farmer's stock broadly spread and therefore still unguessable.

Viability is now **69.7%** with the remaining 30% spread across all four causes
(variety not stocked 34%, not enough of it 28%, quality too low 23%, price gap
15%), so walking away stays a real, multi-reason outcome rather than a rare edge
case. Narrowing the stock range instead would have hit the same viability while
pushing the buyer's blind-guess baseline from 0.57 to 0.70.

## Two findings worth knowing before you change anything

Both came out of the channel ablation rather than from reasoning, both are easy to
reintroduce by accident, and both have a test or a config comment guarding them.

### Frequency skew and learnability pull against each other

The addendum asks for a skewed meaning distribution so that word length has
something to track. Skew turns out to be directly antagonistic to getting any
language at all, and the effect is large.

Skewing **which variety is wanted** is the worst case. With three varieties at
`zipf_alpha = 0.8` the commonest is wanted ~55% of the time, so "always name the
common one" is available immediately while learning to listen needs two agents to
co-adapt first. The constant policy wins and nothing ever leaves it. Measured at
1v1 over 40k episodes, everything else held fixed:

| skew | turn length | farmer names the right variety | carried by the channel |
|---|---|---|---|
| none | 3 symbols | 0.814 | **64%** |
| none | 6 symbols | 0.649 | 24% |
| variety, 0.8 | 3 symbols | 0.554 | **0%** |
| variety, 0.8 | 6 symbols | 0.560 | **0%** |

So the skew is split by dimension — `zipf_alpha` for requested **quantity**,
`zipf_alpha_variety` (default 0) for variety. But skewing *quantity* alone still
costs a great deal. Two seeds each, uniform variety, 3-symbol turns:

| quantity skew | seed | variety naming (intact → muted) | carried by the channel |
|---|---|---|---|
| 0.0 | 0 | 0.957 → 0.342 | **93%** |
| 0.0 | 1 | 0.802 → 0.359 | **69%** |
| 0.5 | 0 | 0.359 → 0.361 | 0% |
| 0.5 | 1 | 0.537 → 0.335 | 30% |

Two mechanisms are at work: a skewed marginal hands a mute agent a larger free
score (the best constant guess on quantity goes from 0.14 to 0.28 as α goes 0→0.9),
and concentrating demand on small quantities makes `stock ≥ need` nearly always
true, which raises viability and deepens the "always accept" attractor.

The shipped default is `zipf_alpha = 0.3` — a ~1.9× frequency range across
meanings, enough for the length analysis to have something to measure, mild enough
to still train. `runs/uniform` is an α=0 control run alongside the main set so the
trade-off is visible in the results rather than only asserted here. **The strong-
skew regime the addendum envisages did not train at this scale**, and that is
reported as a finding rather than worked around.

Note the secondary effect in the first table: a longer per-turn cap costs
transmission on its own. The cap has since been raised to a generous buffer
(24 symbols per turn) because a small cap does worse damage: at 4 symbols, 100%
of utterances were hitting it once every field had to be named.

### Straight-through Gumbel cannot feel a length cost on its own

Under ST-Gumbel the symbol policy gets gradient only through the listener's
decision. The episode return — and therefore the per-symbol cost — reaches it
merely as a scalar reweighting of that term, which is far too weak to teach an
agent to stop talking. The result was unmistakable: 84% of utterances ran to the
cap, and the single commonest "word" in the whole language was
`a8-a8-a8-a8-a8-a8`, one atom repeated six times.

`train.gumbel_mix_reinforce` mixes a score-function term back in over the symbols,
restoring the direct "shorter is better" path. It works — raising the cost with
the mix on drove utterances from 3.83 symbols (93% at the cap) down to 1.20 (16%)
— but the score-function term is itself high-variance and too much of it costs
transmission, so the default is a small 0.1. Set it to 0 to reproduce the
babbling, or turn it up to watch agents go quiet.

---

## The curriculum: learn to refer before learning to haggle

Dropped straight into the full trading task from random weights, agents have to
solve five things at once before any of them pays off even once — emit a stable
signal, put true private information in it, have the other side decode it, close
the loop so decoding changes a decision, and get the trade arithmetic right as
well. A run at that setting produced success 0.000 at *every* checkpoint,
comprehension 0.000 throughout, and a channel whose scrambling cost nothing.

So the task is built up, and a rung is only left behind once it has worked:

Four of the nine rungs are about naming, and nothing is traded until they are
done. Each is a lineup: the describer sees one thing and which field it is being
asked about, the guesser sees the candidates and picks. The describer alternates
batch by batch, and below the trading rungs **both seats are filled from one pool
of agents**, so there is one language rather than two that have to be reconciled
afterwards.

**A naming rung adds a kind of round; it never swaps to one.** `name-color` is
60% colour rounds and 40% fruit rounds, so the fruit words stay in use and stay
needed while the colour words are being invented. Swapping outright was tried
and cost the run both things at once: the messages still carried fruit
(coverage 0.40, 0.00, 0.00) because nothing asked for anything else, and colour
sat at chance for 500 updates with almost no gradient to move it.

Each rung is **promoted on the kind it introduces** -- by then the rehearsal is
easy, and one pooled number would let a rung pass on work it did last time --
and it must also show it **still names** everything below it, scored round-kind
by round-kind. Forgetting fruit to learn colour is not progress.

| rung | what is added | mixture | chance rate |
|---|---|---|---|
| `name-fruit` | a lineup whose candidates share colour and quality and differ only in fruit: only the fruit needs saying | all fruit | 1/3 |
| `name-color` | colour rounds -- same fruit, same quality, different colours. A word for a colour and nothing else. | 60% colour, 40% fruit | 1/3 |
| `name-quality` | quality rounds. Every round still asks one field, but which field changes, so a word has to mean the same thing wherever it appears. | 50/25/25 | 1/3 |
| `name-all` | rounds where the candidates differ in any field, mostly one-field near misses, so the whole (fruit, colour, quality) is named at once | 70% all, 10% each single field | 1/3 |
| `mutual` | both hold a private thing and each must report the other's; still no price, no accept/reject | measured (muted channel) |
| `order` | trading begins and the pool splits into farmers and buyers, each carrying the language it learned: the buyer asks for a fruit, a colour and a quantity, and the farmer must fill the order exactly | measured (muted channel) |
| `haggle` | price and budget, so accept/reject has a payoff — still one message each | ~0 |
| `bargain` | several turns, so counter-offers become possible | ~0 |
| `market` | the full economy: persistent stock, restocking, viability | ~0 |

**The single-field rungs come first because they are learnable from nothing.**
A code has to exist before it can be made compositional: `name-fruit` needs one
word per fruit and nothing else, and the rungs that follow reuse those words
rather than starting again.

**`name-all` and `mutual` are gated on the held-out combinations** -- success
there must reach 60% of success on trained ones (`curriculum.min_holdout_ratio`)
and be clear of chance. The single-field rungs are not asked for it: they have
not been taught the other fields. That gate is the pressure toward reusable
parts; everything else in the ladder is opportunity.

Each rung adds one thing. Alternating describers exist because a single fixed one
produces a one-way code: in the run that motivated it, the farmer's utterances
had positional structure 0.03 while the buyer's had 0.39, and every farmer
newborn's token accuracy was 0.000.

The point of the first rung is that 1/K is a gradient RL can climb, where the
full task's success probability from random weights is about 1e-3.

**Weights carry across every transition.** The population that learned to refer
is the population that learns to haggle — nothing is reinitialised at a boundary.
That works because all four phases share one sequence layout, one channel and one
set of heads; a phase that uses fewer turns just leaves the later dialogue slots
empty. (The transmission bottleneck still applies normally to newborns *within* a
phase. That is a separate mechanism and is untouched.)

**Promotion is on evidence, not on a schedule.** All of these have to hold at the
same check before the next rung starts:

- success clear of that rung's chance rate (and above an absolute floor),
- topological similarity clear of its own shuffled null,
- the channel control showing a real drop when messages are muted.

In the lineup rungs and `mutual` every one of these is checked **per role**,
never pooled: each role's own utterances must show topsim over null *and*
positional structure (`curriculum.min_positional_structure`), and each role must
decode in the view where it is the one decoding, or report the other's thing
(`mutual`, `curriculum.mutual_min_report`). A pooled
average would let a fluent partner carry a role that never learned to speak.

Success alone is not enough, because a pair can score on base rates without
saying anything. Every check, passed or not, is written to `promotions.jsonl`.

**Every rung has a budget** (`curriculum.rung_budget_updates`, min and max
training updates). Promotion is checked every `curriculum.check_every_updates`
updates with a light probe, so a rung that works is left promptly. After the
first rung, the budget only starts counting once the community is at full size
-- the rung cannot pass before then, and a bigger community would otherwise stall
on growth alone. A rung that reaches its maximum without
meeting its criteria **stops the run** (`curriculum.on_stall`, default `stop`)
and the report names every unmet criterion. Building the next rung on top of one
that never converged would only reproduce the failure a rung higher.

**Who speaks when belongs to the rung.** In the lineup rungs the describer opens,
so everything that needs to know whose words are whose — the symbol cost, the
bottleneck's training targets, the "these were my words" embedding, and the
probes that extract per-meaning forms — asks the rung. The earlier fixed
buyer-opens schedule billed a silent guesser for the describer's symbols, trained
buyer newborns to imitate farmer words, and gave farmer newborns no targets at
all.

### Hindsight feedback

After each round, the heads a rung scores are also trained towards the outcome:
the lineup target, the partner's actual meaning, the order that was placed, the
other trader's actual situation (`train.hindsight_coef`). This is feedback about
*what happened*, never about which words to use, and it reaches the speaker
through the straight-through channel for every field the listener has to
recover. Without it the code locked into naming variety alone -- 1.5 bits of
variety and 0.01-0.05 bits of quantity or quality in live messages, e.g.
`a13-a13-a13-a13` -- because a listener that only ever hears "right" or "wrong"
never learns what it should have read, and a speaker whose every slot is read
as variety gets no gradient towards anything else.

**It starts at `mutual`** (`train.hindsight_from_rung`), not before. While
no code exists yet, a listener told the answer learns -- correctly -- that the
messages carry nothing: it spreads its guesses evenly (the spread of its choice
logits fell from 0.5 to 0.17 in 100 updates) and the speaker's gradient, which
runs through the listener, dies with it. With hindsight on from the first rung
the lineup code never formed, on the CPU and on the GPU (still at chance after
2,500 updates); without it, it formed at ~550 updates. So the two rungs where a
code has to form from nothing -- every naming rung -- run without it, and it
joins where it was meant
to help: drawing quantity and quality out of a code that already carries
variety.

Structure is judged by **field coverage** (how much of each field the messages
carry, chance-corrected), not just positional structure, which that
variety-only code scored at 1.00 by naming the variety in every slot.

### Telling inherited structure from new structure

Some of the vocabulary visible at the end was inherited from the lineup game
rather than caused by negotiation pressure. Every word is stamped with the phase
it first appeared in and the phase it settled in, so the report separates
"structure the referential game already produced" from "structure negotiation
specifically added" — and lists the words that first appeared in a negotiation
phase, which is where anything like offer / counter-offer / accept / refuse
vocabulary would show up.

## Speaker pressures: brevity, established forms, shared conventions

Three terms are paid to or charged to the *speaker* only. All three are reward
terms, not restrictions: nothing ever stops an agent from saying anything.

| knob | what it does |
|---|---|
| `reward.symbol_cost` (0.03) | per emitted atom, hyphen or space |
| `reward.rarity_cost` (0.05) | per word, scaled by how rare the form is in the population's recent usage (`usage_half_life_updates`), centred on the batch so it favours established forms without ever favouring silence |
| `reward.convention` (0.15) | for matching the population's current form *for this meaning*, minus the similarity to other meanings' forms, so one form for everything earns nothing |
| `train.shaping_reinforce` | how strongly these reach the speaker's token choices |

A language has to exist before it can be economised. Charged from the first
episode, even a small symbol cost drives the lineup's describer to silence long
before the lineup takes off, and ramping them in with the first rung's success
capped that success once lineups demanded every field be named (0.42 with the
costs two-thirds on, against 0.62 with them essentially off). So all three are
off for the first rung and fully on from its promotion onwards.

**Growing the community.** Six speakers and six listeners from random weights
never got the lineup off chance in 200k episodes: each farmer kept its own
drifting code (coherence 0.04-0.09), and even a strong convention bonus only
lifted that to ~0.2. Two and two invent a code in ~80k. So a large community is
*founded* small (`population.founders_farmers/_buyers`) and grows after the
first rung: a newcomer of each role joins every `population.grow_every_updates`
updates, born like any newborn -- random weights, then the transmission
bottleneck on the community's transcripts -- so it learns the existing language
instead of inventing another. Every rung after the first waits for, and is
judged on, the full community.

The report measures what they are for: distinct words, atoms per word, share of
utterances at the length cap, coherence within each role and across roles, and
**cross-role vocabulary overlap**: the histogram intersection of the farmer's and
the buyer's word use (1.0 = one shared vocabulary, 0.0 = two foreign codes).

## Generations, and what gets lost

Agents age, die at a randomised lifespan (900-1,600 training updates), and are
replaced by newborns with fresh
random weights. Deaths are staggered, so at any moment some agents already know the
language and some must acquire it. A code that only works between two co-adapted
agents fails to transmit and is selected against.

A newborn's apprenticeship (the **transmission bottleneck**) is supervised learning
on the parent generation's recent successful trades. It sees **nearly all of them**
(`bottleneck.coverage`, default 1.0, up to `max_samples`), not a few hundred.

That sizing is the point. An earlier version drew a small fixed sample — as few as
43–90 transcripts in practice — and that had the asymmetry backwards: with a sample
that thin, a form used in 2% of trades might appear a handful of times or not at
all, so *common* vocabulary was at risk of being lost, not just obscure vocabulary.
Real transmission does not look like that. Children reliably acquire essentially
everything the adults around them use with any regularity; loss and drift are
marginal phenomena at the rare end.

With near-complete coverage the asymmetry falls out of the statistics instead of
being imposed by a cap: a form used in 1% of trades still appears hundreds of times
in a 40,000-transcript sample and transmits reliably, while one used in 0.01% may
genuinely not appear at all. Only the second kind is at real risk. Sampling stays
proportional to how often each meaning actually came up
(`bottleneck.frequency_skew`), so the *composition* of a newborn's experience still
mirrors the parent generation's — it is simply no longer artificially thin.

Every birth records what vocabulary it was actually shown, and the report gives
retention for common and rare forms **separately** rather than as an aggregate, so
the asymmetry is visible rather than assumed. When a rare meaning's form is lost and
rebuilt out of words that are common elsewhere, that is the shape of an irregular
verb levelling out, and `FormTracker` logs it with before/after examples.

This is why metrics are bucketed into frequent and rare meanings. A global average
hides exactly this effect.

---

## Layout

```
orchard/
  config.py      every knob, JSON-serialisable; nothing is hardcoded
  world.py       private state, Zipfian requests, the independence property
  economy.py     market days, seasons, multi-variety inventories, replenishment
  env.py         episode mechanics, word parsing, trade resolution, reward
  agents.py      the randomly-initialised transformer policies
  batched.py     the tensor world and reward the training loop uses
  rollout.py     batched play (probes and evaluation)
  gumbel.py      training: straight-through Gumbel channel + REINFORCE decisions
  curriculum.py  the ladder of rungs, their worlds, and promotion
  conventions.py the population's recent usage: coining cost, convention bonus
  population.py  ageing, death, birth, generation counting
  bottleneck.py  iterated learning, frequency-skewed curriculum
  metrics.py     spec section 5: success, topsim, entropy, stability,
                 cross-generation intelligibility, zero-shot, channel ablation
  lexicon.py     addendum section 3: words, length↔frequency, buckets, form survival
  ledger.py      trades.jsonl / trades.csv / metrics.jsonl / births.jsonl / run.log
  render.py      human-readable transcripts (placeholder names only)
  report.py      the final report and its verdict
  plots.py       matplotlib figures, with a dependency-free SVG fallback
  properties.py  the language-properties scorecard (disentanglement, duality, ...)
  transcripts.py transcripts.txt: expected / dialogue / outcome for every round
  analyse.py     re-measure a snapshot after the fact
  run.py         CLI (also --benchmark and --resume)
configs/         named experiments only; the configuration is config.py's defaults
tests/           including test_config.py: one configuration on every device
```

### Why Gumbel-softmax

Spec 2.3 offers REINFORCE or Gumbel-softmax and asks the implementer to document
the choice. Gumbel-softmax on the message symbols is the only training path; the
pure-REINFORCE path was removed rather than left to fall out of date (it is in
the git history).

Pure REINFORCE was tried first and the ablation showed it failing: after 24k
episodes, destroying every message in flight cost almost nothing, because almost
nothing was getting through. Crediting a multi-symbol discrete utterance with one
scalar at the end of an episode is too high-variance at this scale.

Straight-through Gumbel fixes the *estimator* without softening the *channel*. The
emitted symbol is still an exact one-hot in the forward pass — the partner receives
one discrete symbol, with no extra bandwidth, which is the infinite-bandwidth cheat
spec 2.2 warns about. Only the backward pass uses the relaxation. The trade decision
stays discrete and stays on REINFORCE. No babbling or auto-encoding pretraining was
needed.

---

## Running the experiment

The scientific point is the comparison, so the mechanisms toggle from the command
line and the same code runs either way:

```bash
python -m orchard.run --out runs/main
python -m orchard.run --out runs/nobottleneck --bottleneck off
python -m orchard.run --out runs/noturnover  --turnover off
python -m orchard.run --compare runs/main runs/nobottleneck runs/noturnover
```

Any field is overridable: `--set world.zipf_alpha=0 --set bottleneck.frequency_skew=2`.
A change to anything other than scale is printed in the run header and the
report as a method change.

### Output

| file | what is in it |
|---|---|
| `trades.jsonl` / `.csv` | every episode: hidden state, the full symbol transcript, its word segmentation, both decisions, outcome, failure classification, rewards and money |
| `metrics.jsonl` | every checkpoint's full metric suite |
| `births.jsonl` | every birth: what the newborn was trained on, which meanings it never saw, how it fared against veterans |
| `run.log` | the complete console history |
| `plots/metrics.svg`, `plots/vocabulary.svg` (+ `.png`) | progress over the run |
| `report.md` | final metrics, the inferred dictionary, example transcripts early/middle/late, economic totals, and a threshold-computed verdict |
| `transcripts.txt` | sampled rounds, each as an expected / dialogue / outcome block |
| `promotions.jsonl` | every promotion check, passed or not, with its evidence |
| `snapshots/` | `latest.pt` each checkpoint and `after-<rung>.pt` at each promotion, for `--resume` |
| `progress.json` | rewritten every batch: episode, update, rate, ETA, headline numbers |

Two runs can be put side by side on the measures that decide whether a change did
anything:

```bash
python compare_runs.py runs/main runs/main2
```

It reads each run's own `metrics.jsonl`, `token_semantics.json` and ledger, so it
reports what the run recorded rather than what a report was written to say.

Reports are rewritten at every checkpoint, so a long run can be read while it is
still going and an interrupted one is never left with only raw JSONL.

### Reading a report honestly

The verdict is computed from fixed thresholds, not written by hand, so a mediocre
run cannot be talked up. It can come back as `NO EMERGENCE`, `DEGENERATE CODE`,
`NON-COMPOSITIONAL SIGNALLING`, `PARTIALLY COMPOSITIONAL` or
`COMPOSITIONAL LANGUAGE`, and the evidence for it is listed. Degenerate outcomes
are flagged loudly during the run — success stuck at chance, vocabulary collapse,
length-cap babbling, a channel that carries nothing, and the two distinct
word-structure failures (the hyphen never used, or the space never used).

Reward shaping is disclosed in every report. Fully sparse success has probability
around 1e-3 under random play, which REINFORCE cannot bootstrap from, so partial
credit is given for *mutual agreement* and for decisions matching the joint ground
truth. Every shaped term still requires information neither agent holds alone — but
it does mean success rate alone is not proof of language, which is why the ablation
and the topsim/coherence figures sit beside it.
