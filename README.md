# Orchard: emergent language in a fruit-trading world

Two populations of small neural agents — **Farmers** who grow fruit and **Buyers**
who need it — have to invent a language in order to trade. Nobody starts with
one. Every agent is a randomly initialised transformer; the "words" are integer
ids into a random embedding table; the only things that shape them are the
outcomes of trades, population turnover, and what each new generation manages to
pick up from the one before it.

**No pretrained model, no pretrained embedding, and no text corpus is used
anywhere in this project.** If a component ever looks like it needs real-world
language data, that is a design bug, not a shortcut — see `orchard/agents.py`.

To run it, see **[CLOUD.md](CLOUD.md)**. This document is the project: what was
asked for, what is built, why each piece is the way it is, what has been
measured, and what has not.

---

## Contents

1. [The brief this implements](#1-the-brief-this-implements)
2. [The world, and why language is necessary](#2-the-world-and-why-language-is-necessary)
3. [The channel: an open vocabulary](#3-the-channel-an-open-vocabulary)
4. [Reward: closing the communication loop](#4-reward-closing-the-communication-loop)
5. [The curriculum](#5-the-curriculum)
6. [Speaker pressures and the community](#6-speaker-pressures-and-the-community)
7. [Generations and the transmission bottleneck](#7-generations-and-the-transmission-bottleneck)
8. [Training](#8-training)
9. [What is measured](#9-what-is-measured)
10. [Output](#10-output)
11. [Findings, with the evidence](#11-findings-with-the-evidence)
12. [Status: what is validated and what is not](#12-status-what-is-validated-and-what-is-not)
13. [Performance and engineering](#13-performance-and-engineering)
14. [One method, declared scale](#14-one-method-declared-scale)
15. [Layout](#15-layout)

---

## 1. The brief this implements

The project was specified in two documents, now folded into this one: an
original spec and an addendum that replaced its communication channel. Both are
reproduced here in substance, because they are the requirements the code is
answerable to.

### 1.1 The original brief

> Build a Python simulation in which two populations of small neural agents —
> Farmers who grow and sell fruit and Buyers who purchase it for a household —
> must communicate to trade. Neither population starts with any language. All
> agents begin with randomly initialised, untrained neural network policies and a
> discrete, meaningless token vocabulary. Communication has to be invented from
> scratch through repeated interaction, reinforcement learning, population
> turnover across generations, and a transmission bottleneck between generations.
> The research goal is to observe whether a compositional, stable language
> emerges, and to produce hard output logging every trade and every utterance so
> this can be inspected afterward.

It is a research simulation, not a product: correctness, inspectability and
logging come before performance or polish, and each layer had to work and be
tested before the next was added.

**The critical constraint**, stated in the brief and honoured throughout: no
pretrained language model, no pretrained embeddings, no text corpus. Every agent
brain is a small randomly-initialised network trained only by reinforcement
learning on interactions inside the simulation, plus the supervised
apprenticeship a newborn gets from its own population's transcripts.

The brief's specific requirements:

| § | requirement | where it lives now |
|---|---|---|
| 1.1 | discrete-time market; farmers and buyers paired each day | `economy.py`, `env.py` |
| 1.2 | private information neither side can observe: varieties, quality, quantity, the buyer's need and budget; **price is not fixed by the world** and must be agreed through the channel | `world.py`, [§2](#2-the-world-and-why-language-is-necessary) |
| 1.3 | a trade succeeds only if **both** agents' final understanding matches *and* the deal is executable; both are rewarded, both penalised for miscommunication | `env.resolve`, [§4](#4-reward-closing-the-communication-loop) |
| 1.4 | stock replenishes by season, buyers get fresh needs, so the meaning space is too large to memorise — this is the pressure toward compositionality | `economy.py`, [§2](#2-the-world-and-why-language-is-necessary) |
| 2.1 | small transformer per agent; input embeddings for private observation, incoming messages and role; separate heads for the next message symbol and for the trade decision; randomly initialised at birth | `agents.CommNet` |
| 2.2 | **discrete** channel, no continuous message vectors (which would be an infinite-bandwidth cheat); bounded length; bounded turns; tokens with no pre-assigned meaning | superseded by the addendum, [§3](#3-the-channel-an-open-vocabulary) |
| 2.3 | policy-gradient training; REINFORCE or Gumbel-softmax, **document which and why** | [§8](#8-training) |
| 3 | 8–20 agents per role, ages, randomised staggered lifespans, death and replacement by randomly-initialised newborns, generation counting | `population.py`, [§7](#7-generations-and-the-transmission-bottleneck) |
| 4 | the transmission bottleneck: a newborn is trained supervised on a *limited* sample of recent successful transcripts before being let loose | `bottleneck.py`, [§7](#7-generations-and-the-transmission-bottleneck) |
| 5 | metrics over time: success rate, topological similarity, vocabulary/entropy stats, stability, cross-generation intelligibility, zero-shot generalisation | `metrics.py`, [§9](#9-what-is-measured) |
| 6 | per-episode trade ledger with full transcripts; per-checkpoint human-readable summaries; placeholder token rendering (never hand-assigned meanings); a final report with an honest assessment | `ledger.py`, `render.py`, `report.py`, [§10](#10-output) |
| 7 | build order: environment first with scripted agents, then a single learning pair, then the metric, then a population, then turnover, then the bottleneck, then scale | `--smoke` is step 1; the rest is history |
| 8 | Python 3.10+, PyTorch, JSONL/CSV ledger, matplotlib plots, modular repo, everything configurable rather than hardcoded | `config.py` — every knob, JSON-serialisable |

Two requirements are deliberately *not* met as literally written, and both are
reported rather than hidden:

- **Turns.** The brief suggests 6–10 alternating turns. The ladder uses 1, 2 or
  4 turns depending on the rung (`channel.n_turns` = 4 at the top). More turns
  multiply the hardest cost in the system — one forward pass per agent per
  symbol step — for a negotiation that does not yet need them.
- **Quantities 1–20.** `world.max_qty` is 8. The meaning space is made large by
  the five-field product (4 fruits × 4 colours × 4 qualities × 9 quantities × 6
  prices = 3,456 lots) rather than by a long quantity range, and 8 already makes
  memorisation infeasible.

### 1.2 The addendum: an open vocabulary

The addendum replaced §2.2's fixed 20–40 token list, whose problem it stated
plainly: a fixed tiny vocabulary produces *a code*, not a language — it cannot
have more words, longer sentences, or vocabulary growth, because the vocabulary
is fixed by construction.

What it asked for instead:

- **A small inventory of meaningless atomic tokens**, playing the structural
  role phonemes play — a closed set combined into an open-ended vocabulary.
- **Words are hyphenated atoms**: `a7-a22-a3` is one three-atom word. The hyphen
  is a structural symbol the agent emits, not a meaningful token, with no cap on
  word length beyond the message buffer.
- **Words are separated by a space symbol**; one turn's message is a sentence of
  one or more words.
- **Generation is autoregressive** over `{atoms} ∪ {hyphen, space, end}` — a
  genuine sequence-generation problem, not single-token selection.
- **A generous per-turn buffer**, not a tight cap: large enough for multi-word
  phrases, and to be relaxed once training is validated.
- **Do not hand-segment words.** Where hyphens and spaces go is the agent's
  choice, and whether multi-atom words emerge at all is something to *detect*,
  not to force.

And three pressures, explicitly **costs, not rules** — the point being that
human-like vocabulary properties should emerge under pressure the same way
compositionality is supposed to:

| addendum § | pressure | intended outcome | implementation |
|---|---|---|---|
| 2.1 | a length cost per turn | no absurdly long sentences, without a hard ban | `reward.symbol_cost`, `reward.atom_cost`, `reward.word_cost` — see [§6](#6-speaker-pressures-and-the-community) |
| 2.2 | the length cost paid *per episode*, against a skewed meaning distribution | common meanings get short words (Zipf) | `world.zipf_alpha`; the correlation is reported, not assumed |
| 2.3 | a newborn's sample dominated by frequent meanings | no broadly-useless over-specific words; rare forms are at risk | `bottleneck.frequency_skew`, `bottleneck.coverage`, `bottleneck.meaning_holdout` |
| 2.4 | generational drift of obscure forms — *not* a bespoke mechanism, but a prediction to instrument | irregular forms levelling out into compositional ones | `lexicon.FormTracker`, reported with before/after examples |

Plus the analyses in its §3: length–frequency correlation, word-like unit
detection from the agent's own space symbol, per-bucket (frequent vs rare)
stability and compositionality, and generational form-survival tracking. All
four are in `lexicon.py` and appear in the report.

The addendum also warned about two failure modes, both of which are now detected
and named explicitly: **degenerate long babbling** (utterances at the buffer end
carrying noise) and **degenerate hyphen/space usage** (the hyphen never used, or
the space never used).

---

## 2. The world, and why language is necessary

Language is only needed when one party holds something the other cannot see and
cannot guess. That is built in explicitly and enforced in code.

### Everything is a lot

The unit of everything that is ever talked about is a **lot**:

```
(fruit, colour, quality, quantity, price)
```

| field | values |
|---|---|
| fruit | APPLE, BANANA, PEAR, PLUM |
| colour | RED, YELLOW, GREEN, PURPLE |
| quality | LOW, MED, HIGH, PRIME |
| quantity | 0–8 (0 is "none of that", which a farmer has to be able to say) |
| price | six bins from 1.00 to 3.50 in steps of 0.50 |

A buyer's request is a lot: the fruit and colour it wants, the lowest quality it
will take, how many it needs, the most it will pay. A farmer's barn is a list of
lots: each stocked (fruit, colour) cell with its quality and how many are left,
all at the farm's one floor price. And the naming game ([§5](#5-the-curriculum))
describes lots. So there is **one observation layout** for "a thing to talk
about" — five slots in that order, plus a slot saying which field is being asked
about — and it is the same five slots whether the thing is the lot a naming round
asks about or the request a buyer brings to market. The words a population
invents in the naming game are, slot for slot, the words it places an order
with; nothing has to be relearned when trading starts, and no field has to be
*invented* under trading conditions.

That last point is why the lot exists. The design before this one named three
fields — (fruit, colour, quality) — and left quantity and price to be invented
in the trading rungs, where hindsight feedback, the speaker costs and the
convention bonus were already on. Those are exactly the conditions the naming
rungs show stop a code from forming, and quantity never arrived
([§11](#11-findings-with-the-evidence)).

| the Farmer privately knows | the Buyer privately knows |
|---|---|
| for each (fruit, colour) cell: its quality and how many are left | which fruit, in which colour, it wants |
| the lowest per-unit price it will take | the minimum quality it will accept |
| | how many it needs, and the most it can pay |

Price is never set by the world — it has to be proposed and agreed.

A deal is possible only if the barn has that fruit in that colour, in enough
quantity, at acceptable quality, within budget. **Neither agent can determine
that alone.** Both then independently declare what they think was agreed, and
the trade succeeds only if those declarations match *each other* and describe a
deal that is actually executable. One agent being right is never enough.

The barn has one cell per (fruit, colour) — 16 cells, each stocked with
probability `world.p_stocked` (0.85) — and the farmer sees it as **rows in a
random order** that changes every encounter. So the only way to find "the lot
the buyer asked about" is to match its fruit and colour against the words that
were heard: a lookup by content, which is what a transformer's attention does
well, rather than by an arithmetic cell index. (With a single colour per fruit,
two thirds of shoppers could not be served by anybody and refusing every deal
beat trading; that is why there are colours at all.)

### A quarter of the combinations are never trained on

Sixteen of the 64 (fruit, colour, quality) combinations are reserved
(`world.holdout_combo_frac` = 0.25), and nothing in the project ever trains on
them: no lineup describes one, no barn stocks one, no shopper asks for one. They
are chosen as a **Latin square** — one quality withheld from every (fruit,
colour) pair, one colour from every (fruit, quality), one fruit from every
(colour, quality) — which makes the set balanced in every direction. Two things
follow, and both matter:

* every fruit, colour and quality still appears constantly in training, so there
  is always something to generalise *from*; what is withheld is a pairing, never
  a value;
* a lineup that varies one field always has exactly three candidates that could
  be the answer. An unbalanced set leaves lineups containing a combination that
  is never anybody's target, and a guesser can then rule it out **without
  listening** — which is how an earlier version scored 0.42 against a chance rate
  of 0.33 with the channel muted.

The Latin square requires the three fields to be the same size, which is why
there are four of each. Quantity and price are never held out: every value of
each is trained, and what is tested is whether the *combination* generalises.
Success on the reserved combinations is the productivity test, and it gates
promotion ([§5](#5-the-curriculum)). A code that gives each thing its own name
scores at chance there however well it has drilled the rest; a code with
reusable parts does not.

### The property everything rests on

Every farmer field is drawn independently of every buyer field. No amount of
staring at your own barn tells you what the customer wants.

This was got wrong once and it is worth recording. An earlier sampler forced
roughly half of all encounters to be compatible so that viable deals would be
common enough to learn from. That made the buyer's wanted variety predictable
from the farmer's own stock — the farmer could score 0.67 against a 0.33 base
rate without listening to anything. Worse, when a farm held only one variety, the
farmer's best answer was always "the one I have", so that dimension could never
reward listening even in principle. Farms now carry a multi-lot inventory and
nothing is coerced.
`tests/test_env.py::test_knowing_one_side_does_not_predict_the_other` exists so
this cannot come back unnoticed.

### Making deals common enough to practise

Both sides are drawn fresh and independently every round; that independence is
not negotiable. But independence alone left only **56.6%** of rounds viable, so
buyers spent nearly half their time practising correct refusals.

The obvious fix — correlating the farmer's stock with the buyer's wanted variety
— would have raised viability and destroyed the experiment. Instead the
*marginals* were widened, and the lever that worked best was
`world.need_max_frac` (0.65): **a shop stocks more than any one shopper asks
for.** That lifts P(stock ≥ need) a long way while leaving the farmer's stock
broadly spread and therefore still unguessable.

Viability is now **67.7%** (measured over 200,000 scenarios at the shipped
defaults; `--smoke` reports ~0.68 on its own sample). The remaining third fails
for reasons that overlap — of failed rounds, 72% have quality too low, 68% not
enough stock, 47% the fruit/colour not stocked at all, 13% a price gap — so
walking away stays a real, multi-reason outcome rather than a rare edge case.
Narrowing the stock range instead would have hit the same viability while
pushing the buyer's blind-guess baseline from 0.57 to 0.70. The world's
statistics did not change when the barn became rows of lots; only its layout
did.

### The control that cannot be fooled

Every checkpoint plays the same scenarios **three times**, with the same pairings
and the same scenarios. Only what reaches the other party changes:

| condition | what the listener hears | what it isolates |
|---|---|---|
| **intact** | the message | — |
| **scrambled** | random atoms, *same length and stopping point* | what the symbols carry |
| **muted** | silence | everything the channel is worth |

The muted condition exists because an earlier version used scrambling alone, and
scrambled accuracy sat at 0.48 in a world whose base rate was 0.33 — something
was still getting through. It was utterance **length**, which scrambling
preserves and which, with an open vocabulary, is a usable channel in its own
right. Transfer is therefore reported against silence, with scrambled-versus-muted
showing how much of the work length alone was doing.

Because muted means silence, **no agent is allowed to be silent**: a speaker
that used silence as a word would be saying something the control cannot tell
apart from saying nothing, and whatever that word meant would count as zero in
every channel number.

A pair exploiting base rates rather than talking scores identically in all three
conditions. This is what caught the sampler bug above, and what the verdict in
every report leans on hardest.

---

## 3. The channel: an open vocabulary

Agents do not choose from a fixed word list. They emit a **stream of symbols**,
one at a time, from

```
{ a0 … a31 }  ∪  { HYPHEN, SPACE, END }
```

(`channel.atomic_vocab` = 32 atoms; 36 token ids in all once padding is counted,
plus 27 **gesture** ids that sit after them in the embedding table and that the
token head can never emit — see [§5](#gestures-the-scaffold-a-word-forms-on).)

- a **word** is atoms joined by `HYPHEN` — `a7-a2-a3` is one word;
- an **utterance** (one turn) is words separated by `SPACE` — `a7-a2 a3` is two;
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
Ideally separate words come to name separate fields — a fruit word (noun-like)
beside a quality word (adjective-like) and a number — and the report measures
exactly that ("word classes"); nothing requires it.

**Why 32 atoms.** There are 27 field values to name (4 fruits, 4 colours, 4
qualities, 9 quantities, 6 prices). With 32 atoms every value *can* have an atom
of its own, and whether a population reuses atoms across fields (homonyms, told
apart by context) or builds multi-atom words instead is something to measure,
not to force. Fewer atoms than values would make duality of patterning
*necessary*; that is the `duality` experiment in `configs/`, not the baseline.
The vocabulary is open either way — far more possible words than atoms — while
the channel stays discrete. `channel.max_symbols` (24 per turn) is a **buffer,
not a limit** anyone should feel: a five-word request is nine symbols; the report
flags any utterance that reaches the buffer end, and the share there should be
~0.

**A turn is at least one word** (`channel.allow_silence` = false): the first
symbol of every turn must be an atom. That is the one hard rule besides the word
grammar, and it exists because silence is taken — it is exactly what the muted
control feeds the listener ([§2](#the-control-that-cannot-be-fooled)). See
[§11](#silence-was-the-shortest-word-and-it-was-the-control) for how that was
found.

### What keeps utterances short is a cost, not a rule

Length is charged **per atom after the first in a word** (`reward.atom_cost`,
0.03), a much smaller charge **per word** (`reward.word_cost`, 0.005), and a
small flat charge **per symbol** (`reward.symbol_cost`, 0.005). Ending a
message is free, because brevity should not be taxed.

The split is deliberate. A fused name for a whole lot is one long word; naming
the parts is five short ones. Charging every symbol equally would tax the
compositional utterance for being longer overall — so words are pressed to be
short, while saying several of them costs little. The flat per-symbol charge is
there for repetition: with the word cost alone, a speaker repeated one word
twelve times to the buffer end (`a1 a1 a1 …` costs 0.06 under the word cost and
0.29 with the symbol cost). A five-word request costs 0.115 in all, against a
task reward above 1.

In the trading rungs this is also the Zipf mechanism: requests follow a Zipf-like
frequency distribution, so a meaning that comes up constantly pays its length
cost constantly, while a rare one barely pays it at all. Nothing rewards "short
words for common things" directly; it is a prediction, and `report.md` reports
the correlation rather than eyeballing it. (In the naming rungs lots are drawn
uniformly, so there is nothing for length to track, and the report says so.)

---

## 4. Reward: closing the communication loop

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
term for being understood: swap a partner between "decoded perfectly" and
"ignored the message" and the only thing that moved was the joint trade outcome.

So each agent now also states **what it believes the other party's private
situation to be** — the farmer about the buyer's request, the buyer about
what is actually in the barn for the lot it came for — and that statement is
scored against the truth. Two reward terms follow from it:

- `reward.decode` (0.45) pays an agent for having read the other correctly;
- `reward.understood` (0.45) pays an agent for having *been* read correctly.

The second is the one that was missing. It is per-message rather than per-trade,
it is symmetric, and every field it scores is one the answering agent cannot
observe, so neither term is obtainable without the channel. After the change both
roles have a comparable stake in being understood (0.211 / 0.243) and comparable
gains from listening (0.540 / 0.469, previously 1.044 / 0.224).

`tests/test_reward_loop.py` guards all of this, including a test that holds the
trade fixed and checks the reward still moves with whether the partner read you —
otherwise the term would just be trade success under another name.

The rest of the reward is partial credit for *mutual agreement*: joint success
(1.5), a correct walk-away (0.25), per-dimension agreement and correctness,
judgement, and penalties for one-sided acceptance, missed deals and bad deals.
Every shaped term still requires information neither agent holds alone. This is
disclosed in every report, because it means success rate alone is not proof of
language — which is why the ablation and the topsim/coherence figures sit beside
it.

The same two terms — read the other, be read — are what every rung below trading
pays too ([§5](#5-the-curriculum)): a report round pays `decode` per field the
reader got right and `understood` per field the speaker was read right on, plus
the whole-round bonus only when everything arrived at once.

---

## 5. The curriculum

Dropped straight into the full trading task from random weights, agents have to
solve five things at once before any of them pays off even once — emit a stable
signal, put true private information in it, have the other side decode it, close
the loop so decoding changes a decision, and get the trade arithmetic right as
well. A run at that setting produced success 0.000 at *every* checkpoint,
comprehension 0.000 throughout, and a channel whose scrambling cost nothing.

So the task is built up over **thirteen rungs**, and a rung is only left behind
once it has demonstrably worked. Two rules shape the whole ladder: **each rung
adds exactly one thing and keeps everything below it in play**, and **every word
is invented in a naming rung; every later rung only reuses words.**

### The naming rungs

Six of the thirteen are about naming, and nothing is traded until they are done.
Each is a lineup: the describer sees one lot and which field it is being asked
about, the guesser sees three candidates and picks. The describer alternates
batch by batch, so every agent does both jobs — a single fixed describer produces
a one-way code (in the run that motivated it, the farmer's utterances had
positional structure 0.03 while the buyer's had 0.39, and every farmer newborn's
token accuracy was 0.000).

Until `haggle` **both seats are filled from one pool of agents**, so there is
one language rather than two that have to be reconciled afterwards. `farmers`
and `buyers` are literally the same list, and everything that rebuilds the
population has to keep it that way (`curriculum.pooled_at`): restoring the two
saved lists separately, as resuming used to, made two copies of every founder
that then trained apart -- twice the reported population, two languages, and an
`IndexError` one rung later when the first newcomer grew only one of the lists.
An agent is never seated opposite itself — not in training (`Population.pair`), and not in
any measurement either. The measurements used to draw the two seats
independently, so with two founders half of every promotion check was an agent
reading its *own* words, which training never asks for. Two founders who had
each invented a dialect the other could read scored 0.92 in training and 0.60
in the check, and `name-fruit` ran out its budget with a working code.

Two founders who never die will each keep a dialect if nothing pays them not
to, and nothing did: the convention bonus used to wait for the community at
`mutual`, so all the naming rungs ran with no term anywhere rewarding a speaker
for saying the same thing twice. Measured at `name-all`: within-role coherence
0.15–0.17, which is two codes with no form in common. Bringing the bonus on at
`name-all` ([§6](#6-speaker-pressures-and-the-community)) did not mend it: with
two speakers "the community's word" is whichever of two words was said last, and
on the 2026-10-01 run the founders had **0 of 27 words in common** at the end of
every naming rung, `name-all` included. What mends it is that a learner takes
its elders' words
([below](#a-vocabulary-fit-to-hand-on)). Deaths used to paper over the dialects
earlier — a newborn apprenticed to the survivor inherited its words — at the
cost of half the population, which is why nobody dies before `mutual`.

**Cross-role coherence and cross-role overlap cannot be read while the pool is
shared.** Row *i* of "the farmers" and row *i* of "the buyers" are the same
agent in the other chair, so comparing them asks whether an agent agrees with
itself. With two founders that was half of every cross pair, which pins
`coherence across` at about halfway to 1 however foreign the two codes are — it
read 0.56 for the pair whose honest number was 0.16. Self-pairs are now
excluded from cross-role coherence, and cross-role vocabulary overlap is
reported as not yet askable until the roles split.

**A naming rung adds a kind of round; it never swaps to one.** `name-color` is
60% colour rounds and 40% fruit rounds, so the fruit words stay in use and stay
needed while the colour words are being invented. Swapping outright was tried and
cost the run both things at once: the messages still carried fruit (field
coverage 0.40, 0.00, 0.00) because nothing asked for anything else, and colour
sat at chance for 500 updates with almost no gradient to move it.

Each rung is **promoted on the kind of round it introduces** — by then the
rehearsal is easy, and one pooled number would let a rung pass on work it did
last time — and it must also show it **still names** everything below it, scored
kind by kind. Forgetting fruit to learn colour is not progress.

| rung | what is added | mixture of rounds | chance |
|---|---|---|---|
| `name-fruit` | a lineup whose candidates share every other field and differ only in fruit: only the fruit needs saying | all fruit | 1/3 |
| `name-color` | colour rounds — same everything else, different colours. A word for a colour and nothing else. | 60% colour, 40% fruit | 1/3 |
| `name-quality` | quality rounds. Every round still asks one field, but which field changes, so a word has to mean the same thing wherever it appears. | 50% quality, 25% fruit, 25% colour | 1/3 |
| `name-quantity` | quantity rounds: a word for each number, 0 to 8. Invented here so that no trading rung ever has to. | 50% quantity, the rest rehearsed | 1/3 |
| `name-price` | price rounds: a word for each price bin, so every field a deal turns on has a word before any deal is attempted | 50% price, the rest rehearsed | 1/3 |
| `name-all` | rounds where the candidates differ in any field, mostly one-field near misses, so the whole lot is named at once | 70% all fields, 6% each single field | 1/3 |

**A guess is paid for how much of the lot it got.** A lineup round pays
`reward.refer_partial` (0.45) for each field the chosen candidate shares with
the target, on top of the whole-round bonus. Every other rung on the ladder
already paid per field; the lineup did not, so a guess that got most of the
fields was worth exactly as much as one that got none, and nothing rewarded a
message for *narrowing the field down*. That left `name-all` — which needs every
field in one utterance — with no staircase between "one field" and "all of
them". A run stalled there at 0.60 while naming each field on its own at 0.74,
0.87 and 0.97, with utterances 1.5 words long where three were needed. With the
staircase the next run passed it at 0.73–0.80. Partial credit does not count as
success: promotion still needs the exact pick.

**The single-field rungs come first because they are learnable from nothing.** A
code has to exist before it can be made compositional: `name-fruit` needs one
word per fruit and nothing else, and the rungs that follow reuse those words
rather than starting again. The point of the first rung is that 1/3 is a gradient
RL can climb, where the full task's success probability from random weights is
about 1e-3 (`--smoke` measures it as 0.0000).

**Hard rounds.** `curriculum.hard_distractor_frac` (0.9) of all-field rounds are
built as an anchor plus one-field near misses — each near miss differing from the
anchor in a *different* field, and the field drawn uniformly — so every field has
to be named. Two things were got wrong here before. The first version built the
near misses *around the target*, which made the target the most central
candidate — 42% success with the channel muted against 25% chance; the cluster
is now shuffled and the target drawn uniformly from it, and a test checks that
"pick the most central candidate" scores chance. The second drew the near miss
uniformly over all one-field neighbours, which weighted each field by how many
values it has: with nine quantities against four fruits, fruit decided an open
round 7% of the time and a describer could drop it almost for free. With three
candidates at most two fields can decide a round, so with five fields each is the
deciding one in roughly a quarter of hard rounds, and a test checks that no
field is favoured.

### The report rungs

Between naming and trading sit four rungs that are one event with the roles
turned up one at a time: **one side holds a lot, and the other has to put it in
its heads** — the five belief heads, one per field, the same heads `haggle` will
score. Every word is inherited from the naming rungs; what each rung adds is
something to *do* with the words.

| rung | what is added | who reports | turns |
|---|---|---|---|
| `mutual` | both hold a private lot and each must report the other's, all five fields. The community arrives; newcomers, deaths and hindsight feedback switch on. | both | 2 |
| `order` | the buyer states its request — the same five fields, the same layout — and the farmer, **looking at its barn**, reports it. Nothing new to say; what is new is listening with a barn in view. | farmer | 1 |
| `offer` | the other direction too: the farmer answers with what it holds of the lot that was asked for — how many, what quality, at what floor — and the buyer reports that. The first rung where a farmer **finds a lot in its barn** by the words it heard, and describes a lot from a barn row rather than from the naming layout. | both | 2 |
| `judge` | the same dialogue, and now **both decide** whether the deal is any good, each weighing what it was told against what it holds, with nothing yet riding on the answer | both | 2 |

Like a naming rung, each is **judged on the fields it introduced** and has to
show it **still carries** the ones below it, field by field against a muted
channel. A conjunction of ten fields would hide which one is at chance, and that
is exactly what the old ladder did: `haggle` reported 0.07 success, and the
diagnosis — quality 0.88, variety 0.52, **quantity 0.20** — had to be dug out by
hand afterwards.

`judge` exists because of the way `haggle` failed: **always accept**, plus
base-rate guessing, for 7% success and a channel carrying 0.00–0.02. Roughly 68%
of rounds are worth doing, so accepting everything scores 0.68 and looks like
competence. `judge` is judged on the **gain over silence** rather than on the
raw rate — a pair that accepts everything scores exactly what a mute pair scores,
which is zero of the headroom, and cannot pass. ("Twice the chance rate", the
bar everywhere else, is not a reachable number when silence already scores
0.68.)

### The trading rungs

| rung | what is added | turns | chance |
|---|---|---|---|
| `haggle` | the pool splits into farmers and buyers, and the deal starts paying: both sides must name the same one, and it only counts if it is actually executable | 2 | ~0 |
| `bargain` | several turns, so counter-offers become possible | 4 | ~0 |
| `market` | the full economy: persistent stock, restocking, viability | 4 | ~0 |

**The role split moved to `haggle`.** Everything below it is one language in two
seats: the report rungs run in both directions, and one pool learns both from
the same words. The split exists so the two sides can diverge in *strategy*,
which only starts to matter where selling and buying pay differently.

**Agreeing is not bargaining.** A trade needs both sides to name the same
price, and the first run to reach `haggle` (2026-10-01, locally) found the
cheapest way to do it: both named 2.50 in 99% of rounds, whatever the floor and
the limit they had just told each other. That agrees every time, fits 0.83 of
the deals that exist, passes the rung, and is not a negotiation
([§11](#11-findings-with-the-evidence), item 17). Success cannot tell the two
apart, so every checkpoint of a trading rung now prints the commonest price
each side names and, in the rounds where that price does not fit both limits,
how often one that does is named instead (`price named … follows the limits`:
0 is one price whatever the limits are, 1 is a price that follows them). It is
a measurement, not yet a bar.

**Weights carry across every transition.** The population that learned to name is
the population that learns to haggle — nothing is reinitialised at a boundary.
That works because every rung shares one sequence layout, one channel and one set
of heads; a rung that uses fewer turns just leaves the later dialogue slots empty.
(The transmission bottleneck still applies normally to newborns *within* a rung.
That is a separate mechanism and is untouched.)

### Promotion is on evidence, not on a schedule

All of these have to hold at the same check before the next rung starts:

- success clear of that rung's chance rate (at least `min_success_over_chance` =
  2× chance) and above an absolute floor (`refer_min_success` = 0.45);
- topological similarity clear of its own shuffled null (`min_topsim_over_null` =
  0.10);
- the channel control showing a real drop when messages are muted
  (`min_channel_transfer` = 0.25 of the headroom);
- on mixed rungs, every rehearsed kind of round still clear of chance, for each
  describer;
- on `name-all`, **structure** — positional structure ≥ 0.15 and **field
  coverage** ≥ 0.30, and ≥ 0.25 for every field — and **the reserved
  combinations understood**, field by field, at least 60% as well as trained ones
  over the headroom a message-blind reader leaves (`min_holdout_ratio`);
- on every rung that plays rounds on a quantity or a price, **numbers told from
  their neighbours**: rounds whose wrong candidates are the nearest values won
  0.80 of the time, each describer on its own (`numeral_min_near`);
- on `name-all`, **the vocabulary it hands on**: the speaker with the fewest
  distinct words has one for 0.95 of the meanings, and the two speakers furthest
  apart say the same word for 0.90 of them (`min_vocabulary_distinct`,
  `min_vocabulary_agreement`), both read off what the agents say when asked;
- on `mutual`, **the description scaffold gone**
  ([below](#a-vocabulary-fit-to-hand-on)).

In the lineup rungs and the report rungs every one of these is checked **per
role**, never pooled: each role's own utterances must show topsim over null and
positional structure, and each role must decode in the view where it is the one
decoding, or report the other's lot. A pooled average would let a fluent partner
carry a role that never learned to speak.

On a report rung the checks are **per field** as well: every field a role reports
must carry `min_field_transfer` (0.25) of the headroom over a muted channel —
labelled "still carries" if an earlier rung introduced it, so forgetting is
visible as forgetting — the fields the rung introduced must arrive together above
an absolute floor *and* a real gain over silence, and the **held-out gate is per
field**: the three fields of a reserved combination have to be reported nearly
as well as those of a trained one, as a share of the headroom. That last form is
deliberate. A listener whose heads have learned the training set's joint puts no
mass on a reserved combination however compositional the *language* is, so the
whole-round ratio sits near zero — the `mutual` rung stalled on exactly that,
with per-field coverage 0.96 / 0.75 / 0.77 and a whole-round held-out ratio of
0.02 ([§11](#11-findings-with-the-evidence)).

**In the lineup the held-out gate is per field too.** The whole round was kept
there on the argument that a K-way choice has no exponent to remove, but it
cannot tell a code that reuses all three words from one that reuses one. Two
reserved combinations always differ in at least two fields (the reserved set is
a Latin square), so with three candidates a code that says the fruit with its
own word — and colour-and-quality with one fused word per fruit — picks the
target 0.81 of the time on reserved combinations against 1.00 on trained ones:
a "ratio" the old gate passed. The guesser's five belief heads are what the
factored choice sums, so the choice can be taken apart
(`metrics.lineup_field_scores`): for every candidate that differs from the
target in a field, does that field's head give the target's value more weight?
A reading that ignores the message scores exactly 0.5 whatever it does with the
candidates it can see (the target's place in the lineup is uniform, so every
pair of values is compared once each way), so each field's ratio is taken over
the headroom above 0.5. The ratios are averaged per role and the weaker role is
the one judged; the fruit-only code reads fruit 1.0, colour 0.0, quality 0.0
and fails. (The whole round remains the test only for the pointer listener,
`model.factored_choice` off, whose heads do not make the choice.) A rung that
varies one field is not asked, and no longer prints a held-out number: its
held-out rounds show a reserved target among trained distractors, so the target
is the one unfamiliar candidate — the novelty cue every held-out round is built
to deny.

Field coverage is the check that matters most, because positional structure is
fooled by redundancy: a variety-only code like `a13-a13-a13-a13` scores 1.00 on
it by naming the variety in every slot. Coverage asks how much of *each* field
a one-piece reader recovers from the messages: for every symbol slot, every
word position and the bag of words, a lookup table is fitted on half the probes
and scored on the other half, as the share of the headroom above always
guessing the commonest value; a field's coverage is the best piece, and the
rung's is the mean over fields. Cross-validation is what keeps that honest — a
piece that only fits the probes it was fitted on predicts nothing on the rest —
and reading the message in pieces is what makes it measurable at all: a lot
takes 3,456 values, so over a few hundred probes nearly every message of a
compositional code is unique and any whole-message statistic sits at its
ceiling whatever the message means.

**A measured bar has to be measured on the right thing.** `name-all` is the only
rung judged on message structure, and it is also the rung that mixes questions
most — 70% whole things, 30% single fields. The structure probes used to follow
that mixture while `meaning_distance` and field coverage both ignore the query
slot, so 30% of the probes asked a perfect describer for one field and then
scored its one-word answer against all three. Two consequences, both measured on
a flawless, noise-free, fully compositional speaker:

| | perfect speaker scored | bar |
|---|---|---|
| field coverage, probes following the mixture, whole-message MI ÷ H(field) | 0.33 (100 probes) / 0.49 (200) | 0.30 |
| topsim over null, probes following the mixture | 0.32 | 0.10 |
| field coverage, probes asking the rung's own kind, read in pieces | **1.00** at 100, 200 and 400 probes | 0.30 |
| topsim over null, probes asking the rung's own kind | **1.00** | 0.10 |

No real code beats a perfect one, so the rung could not be left. Two fixes:

* **The structure probes ask the kind of round the rung is promoted on**
  (`metrics.probe_query`), which is the rule promotion already follows
  everywhere else. The same function fixed a second probe that was off
  distribution: `mutual` has a query slot in its schema but never fills it in
  (`MutualBatch.obs` pads it), while the probes wrote ASK_ALL there — and
  `K_FIELD` has its own embedding table, so every structure number on `mutual`
  was read off an observation its speakers had never been trained on.
* **Field coverage is read off the pieces of a message, cross-validated**,
  not off the whole message. The whole-message statistic was
  `I(message; field)` over its shuffled null, and both are plug-in estimates
  inflated by however many distinct messages there are: in the limit where
  every probe gets its own message both reach `H(field)` whatever the message
  means. On the three-field world that only moved the ceiling with the sample
  (the same perfect code read 0.50 over 100 probes and 0.93 over 800), which
  dividing by the headroom the null leaves repaired. On a five-field lot it is
  fatal: a compositional code gives each of 3,456 lots its own message, so at
  any affordable probe count the headroom is gone and a flawless describer read
  **0.00** at 100 probes, 0.34 at 200 and 0.84 at 400. Read in pieces it reads
  1.00 at every one of them; a code that gets each field right three times in
  four reads 0.78 at every one of them; a holistic code — one arbitrary word
  per lot — reads 0.04, a random message 0.03–0.05, and a code that names the
  fruit in every slot reads 0.20 (one field of five). A perfect code with the
  words in a random order reads 1.00 too, off the bag of words.

`tests/test_rungs.py::TestAPerfectSpeakerPasses` holds both: a flawless
describer, run through the real measurement functions rather than a dict of
ones, has to score at least 0.75 on every structure bar it is judged on.

Positional structure is still pooled across the agents of a role
(`analyse_token_semantics` splits the probes between them), so two dialects that
put different fields in the same slot depress it — measured, two *perfect*
dialects score 0.30 where one scores 0.40, against a 0.15 bar. Both numbers were
measured before two fixes that each held a perfect code down: colour had no case
in the value binning, so the colour slot read as noise and no word was ever
labelled a colour, and space and hyphen slots counted as content slots with
strength 0. It is not the binding check and is left as is.

Success alone is never enough, because a pair can score on base rates without
saying anything. Every check, passed or not, is written to `promotions.jsonl`.

**Three holes the 2026-09-29 run found in these gates**, each now closed and
tested (`tests/test_language_faculty.py`):

* **The productivity test compared an easy round with a hard one.** Held-out
  rounds drew their candidates independently; the "trained" rounds they were
  compared with were the rung's ordinary ones, 90% hard near misses. And
  independent candidates almost always differ in quantity and price, which are
  never held out. The run showed "held-out 0.89 vs trained 0.66" for a code that
  barely named fruit or colour. Simulated on the real sampler, a flawless code
  for quantity and price *alone* scored 0.98 held-out against 0.69 trained, a
  "productivity" ratio of 1.43, without a word for anything a reserved
  combination is made of. On a whole-lot rung the test is now its own round
  (`ReferentialWorld._combo_round`): every candidate shares one quantity and one
  price and is a distinct combination from the same pool, all reserved or all
  trained, so only fruit, colour and quality can decide it. The comparison is
  the same round on trained combinations. A code without the combination
  scores chance on both; one with it scores in full.
* **Coverage was gated as a mean**, so three covered fields could carry two at
  zero. A flawless code for quality, quantity and price alone scored 0.79
  success and 0.60 coverage against bars of 0.667 and 0.30. `name-all` now also
  needs every field on its own (`min_field_coverage_each`, 0.25).
* **Most rungs were never checked for forgetting.** The frequent light check
  never measured the rehearsed kinds, and `evaluate_rung` skipped what was not
  measured, so every rung promoted between checkpoints (`name-quality`,
  `name-quantity` and `name-price` on that run) went up without a "still names"
  line. Every check measures them now, and an unmeasured kind is unmet.

**More holes, from a code review on 2026-09-30**, each closed and tested
(`tests/test_rungs.py`, `tests/test_config.py`):

* **The muted baseline was not a lower bound.** A muted turn is END in the
  first slot, which no speaker is allowed to say, so no listener was ever
  trained on it and it can read *below* guessing blind. In `judge` about 0.68
  of deals are worth doing: a listener that accepts whatever it hears, against a
  silence it happens to read as 50% accept, "carried" 0.36 of the headroom with
  nothing in the words. Every report-rung headroom is now taken over the better
  of silence and a reader that ignores the message (the commonest value's share,
  or the commonest combination's for a conjunction), and an unmeasured baseline
  fails the check instead of flooring it at zero.
* **`mutual`'s per-side floor was the wrong setting.** Each side's report of
  the other's lot is held to `curriculum.mutual_min_report`; the both-in-one-round
  bar (`mutual_min_success`, a quarter as high) had been standing in for it.
* **Pooled where it should be per role or per view.** The held-out ratio
  averaged the two roles before dividing, and "still names" averaged the two
  describers, so a seat that generalised or remembered could carry one that did
  not. Both are now judged on the weaker one.
* **Failures were swallowed.** An exception while measuring the held-out or
  rehearsal rounds became a silent "n/a", and a measurement with one view
  missing counted as complete. The error is now in the check's detail, and every
  view has to be measured.
* **The reserved set could make a lineup ruled out without listening.** A
  lineup that varies one field can only be as wide as that field's unreserved
  values; past that, a reserved lot sat in it as a distractor that is never the
  answer. The `duality` preset (12 fruits) had (fruit, quality) pairs with one
  unreserved colour, so its colour rounds all did this at the default three
  candidates. The construction for worlds that are not n × n × n now spreads
  the reserved qualities over residues, so every pair of fields withholds as few
  values as the counts allow (the default 4 × 4 × 4 Latin square is unchanged),
  and `validate` refuses a `holdout_combo_frac` that reserves nothing, a lineup
  wider than the unreserved values, and unequal farmer and buyer counts (one
  pool is split into both). Snapshots now record their reserved set, and a
  resume keeps it.

**Every rung has a budget** (`curriculum.rung_budget_updates`, in training
updates): 80–2,000 for `name-fruit` (the first code forms suddenly and late:
525 and 1,525 updates on two GPU runs), 80–1,500 for the other single-field
rungs and `order`, 80–2,500 for `name-all` and `offer`, 80–2,000 for `judge`,
80–3,500 for `mutual`, `haggle` and `bargain`, open for `market`.
Promotion is checked every 25 updates with a light probe, so a rung that works is
left promptly. A rung whose community is still filling up does not spend its
budget, and cannot pass, until everyone has arrived. A rung that reaches its
maximum without meeting its criteria **stops the run** (`curriculum.on_stall`,
default `stop`) and the report names every unmet criterion. Building the next
rung on top of one that never converged would only reproduce the failure a rung
higher.

**Who speaks when belongs to the rung.** In the lineup rungs the describer opens,
so everything that needs to know whose words are whose — the speaker costs, the
bottleneck's training targets, the "these were my words" embedding, and the
probes that extract per-meaning forms — asks the rung. The earlier fixed
buyer-opens schedule billed a silent guesser for the describer's symbols, trained
buyer newborns to imitate farmer words, and gave farmer newborns no targets at
all.

### Hindsight feedback

After each round, the heads a rung scores are also trained towards the outcome:
the lineup target, the partner's actual lot, the request that was placed, the
other trader's actual situation (`train.hindsight_coef` = 1.0). This is feedback
about *what happened*, never about which words to use, and it reaches the speaker
through the straight-through channel for every field the listener has to recover.
Without it the code locked into naming variety alone — 1.5 bits of variety and
0.01–0.05 bits of quantity or quality in live messages — because a listener that
only ever hears "right" or "wrong" never learns what it should have read, and a
speaker whose every slot is read as variety gets no gradient towards anything
else.

**It starts at `mutual`** (`train.hindsight_from_rung`), not before. While no
code exists yet, a listener told the answer learns — correctly — that the
messages carry nothing: it spreads its guesses evenly (the spread of its choice
logits fell from 0.5 to 0.17 in 100 updates) and the speaker's gradient, which
runs through the listener, dies with it. With hindsight on from the first rung
the lineup code never formed, on the CPU or the GPU (still at chance after 2,500
updates); without it, it formed at ~550 updates. So every rung where a word has
to form from nothing — all six naming rungs — runs without it, and it joins
where every word exists and the listener's job is to put five of them into five
heads.

### Gestures: the scaffold a word forms on

The run of 2026-09-24 (`gpu_community`, 2 + 2 founders, the five-field ladder)
passed `name-fruit`, `name-color` and `name-quality` and then sat in
`name-quantity` for 850 updates at exactly chance — 0.32–0.34 against 0.333,
the channel carrying −2% to +1% of the headroom, the rehearsed fields still
named at 0.91 / 0.83 / 0.68. The transcripts show what that looks like: on a
quantity round the describer says one arbitrary atom (`a15`, `a25`, `a24`) and
the guesser picks at random. Nine number words had to break symmetry from
nothing, through a listener whose reading of the message was itself random, so
the straight-through gradient reaching the speaker had no consistent direction
to point in. Four fruit words took 1,775 updates to form that way; nine number
words did not form at all.

Humans do not learn words from words alone. A parent points at the apple while
saying "apple"; a trader holds up three fingers while naming a quantity. The
gesture puts the referent and the word in front of the listener at the same
moment, and it is what makes the word learnable. So the simulation now has a
**gesture channel** beside the spoken one (`orchard/gesture.py`,
`GestureConfig`):

- A speaker may open a turn with **one iconic gesture**, occupying the turn's
  first dialogue slot: *fingers* for a quantity (0–8) or a price bin, *pointing*
  at an exemplar for a fruit, a colour or a quality. Its meaning is given by the
  world, as a real gesture's is — 27 gesture ids, one per (field, value) — and
  it is **truthful by construction**: the value shown is read off the speaker's
  own observation, never the scenario, so a gesture can only reveal what its
  maker can see. A describer, either party in `mutual` and the buyer at market
  look at a lot and may show any of its five fields; the farmer at market looks
  at a barn and can hold up fingers for its floor price only — "how many of
  what you asked for" depends on having understood the request, and a gesture
  the world computed for it would be the world doing the understanding.
- **The world decides when.** Gesturing is possible in a share of rounds
  (`gesture.share_start` = 1.0 at the start of every rung that still has a
  field's words to form — lots of pointing before there are words — withdrawn
  linearly to `share_end` = 0 over `anneal_updates` = 600), then a small
  standing share in every rung that only reuses words (`share_reuse` = 0.1):
  fingers are part of a market, and whether speakers still bother with them
  once the words work is something to measure. `name-all` is a reuse rung
  here: it forms no word, only the description that puts the words together.
  It used to get the full schedule, and its speakers pointed at one part and
  said one word ("[points: PEAR] a26"); a gesture carries one field, standing
  in for a word the speaker already had. The words are shaped from the
  first update regardless, by the innate lexicon below, which does not go
  through the listener.
- **The speaker decides whether.** A sixth head (`CommNet.gesture_head`),
  sampled from the same hidden state that emits the turn's first symbol and
  trained by REINFORCE like the decisions, chooses none or one of the fields
  its seat can show; a round the world allows no gesture in is masked to none
  and not trained on. Each gesture costs `gesture.cost` = 0.02 — small against
  a round's reward of ~1, so gesturing is worth it while the word fails and
  worth stopping once the word works.
- **The listener is taught what the gesture showed** (`gesture.supervise_coef`
  = 0.5): its belief head for the gestured field is pulled towards the gestured
  value. This is *not* hindsight, and the reason hindsight is off in the
  naming rungs does not apply: hindsight tells a listener the answer while the
  message carries nothing, so the listener learns — correctly — to ignore the
  message. Here the answer is *in* the message. The term also reaches the
  speaker through the straight-through channel, pulling its words towards
  whatever the listener already reads as that value.
- **The ostensive lesson** (`gesture.ostensive_coef` = 1.0): the parent points
  at the apple *and says "apple"*, and the child learns the word. On a round
  where the speaker both gestured and said its established name for the
  gestured meaning — the speaker's own lexicon, below, says which utterances
  are names — the listener is shown the turn *without* the gesture, the words
  shifted to where a gesture-free turn's words sit, and its head for that
  field is taught the gestured value: a labelled example of the word, from the
  words alone. Babble is not a lesson, so this cannot teach a listener that
  words carry nothing before any word exists. The lesson reaches the speaker's
  word through the straight-through channel too.

Why that should help the *words*, and not just replace them: the gesture and
the atoms share the dialogue, the token table and the listener's readout. A
listener that has learned "this slot says 3 fingers → quantity 3" has a
readout for quantity that the speaker's atoms pass through too, so on a
word-only round the gradient on an atom says "become whatever this listener
reads as 3" — a consistent direction across episodes for the same meaning, which
is exactly what symmetry breaking needs and exactly what an untrained readout
cannot supply. With the factored listener below, that readout *is* the
quantity head, so the choice on a quantity round is the same head the gesture
trains.

What a gesture is not: it is not a word, and **nothing about the language is
measured on it**. A gesture is not parsed as a word, not costed as a symbol,
not counted in utterance length; a newborn is never taught to emit one (the
slot is masked from its token lesson) though it learns to read them like
everyone else; and `run_episodes` — every probe, every promotion check, the
intact / scrambled / muted ablation, the held-out test — never emits one. A
rung is left only when the *words* carry what the gestures used to. What the
hands did is recorded separately: how often the world allowed a gesture, how
often the speakers made one and about which field, per rung, on the checkpoint
line (`gestures possible 74%, used 89%`), in `metrics.jsonl`, in the plots and
in the report's §3g. A scaffold doing its job is used heavily while the word is
forming and dropped once it works; speakers still reaching for their fingers in
a rung whose words pass the gate are saying the gesture is cheaper than the
word for them.

### One name per meaning: the innate lexicon

A child brings two assumptions to a new word: it names one sort of thing, and a
thing that already has a name is not what it names. Nothing about *which*
sounds go with which things is innate; that a name is for one meaning and a
meaning has one name is. So when there are four fruits there should very soon
be four names — not one form for everything, and not a new form every time.

Before this the only pressure on the words in the naming rungs was the gradient
through the listener, and a listener whose reading of the channel is random
gives that gradient no consistent direction. That is a fact about *speakers*
that was missing, not about communities: the population-level convention bonus
([§6](#6-speaker-pressures-and-the-community)) waits for `name-all` because a
speaker that has not settled its own names has nothing to agree with anyone
about. Two speaker-side terms now supply it, on from the first round of the
first rung, never gated, per agent (`orchard/conventions.py`):

- **Positive signalling** (`reward.lexicon_mi` = 2.0): the mutual information,
  in the speaker's own policy, between the value it was asked about and the
  first symbol it speaks — taken *within the asked-about field* — plus the
  mean pairwise separation of the values' first-symbol distributions. Apple
  rounds should sound alike and unlike banana rounds, by the speaker's own
  lights: exact, listener-free, and it names no symbol. This is the bias of
  Eccles et al. (2019) applied per meaning rather than per observation, so the
  variation rewarded is in the asked-about field and not in the colour of the
  apple being described.
- **The speaker's own lexicon** (`reward.lexicon` = 0.30): a decayed record,
  per speaker, of the forms it has used for each (field, value) it was asked
  about. An utterance is paid for being closer to this meaning's recent forms
  than to any other meaning's — the mutual-exclusivity charge runs across
  fields, so a fruit name and a colour name are pressed apart too — and a form
  counts as *a name* once it has `lexicon_min_support` = 3 recent uses and is
  not also the speaker's name for something else. That last judgement is what
  gates the ostensive lesson above, and what the checkpoint line reports as
  `names 4/4`: distinct names over meanings named. Since 2026-09-30 a form is a
  *word*, not a whole utterance, and a name is paid in full only when said once
  and no longer than two atoms
  ([below](#the-language-faculty-words-word-classes-and-composition)).

Three things were got wrong on the way, each measured on `name-fruit` at batch
256 with two founders, and each is a test now (`tests/test_lexicon_prior.py`):

| version | what happened |
|---|---|
| lexicon bonus against each meaning's *modal* form | both speakers said one form for every fruit and stayed there for 60 updates: the shared form was charged equally on every fruit, which says "not that" but never "something different for each" |
| information alone | 0.00 bits for 90 updates, with or without gestures: mutual information has a zero *gradient* where every meaning's distribution is the same, which is where an untrained speaker starts. The pairwise separation has full-size gradient at any asymmetry, and is what breaks it |
| information and separation across *all* meanings | on `name-quantity` from scratch both speakers settled on **one name per field** — every fruit one form, every quantity another — and the objective read 1.5–2.1 with every value at chance: 139 of the 190 meaning pairs lie across fields, and *which field was asked* is in the speaker's observation. Within the field, the shortcut is gone |

With the within-field objective at 2.0 and the distribution-based lexicon,
`name-fruit` at batch 256 goes from chance to **0.98–0.99 word-only success in
90 updates**, both founders holding three distinct names, with gestures on or
off; at 0.5 the names stayed shared; the same population with neither term
had no code after 90, and the GPU run had needed 1,775. On `name-quantity`
played from scratch — four fields and twenty values at once, which the real
ladder never asks, and with about 7 rows per quantity value per update — the
signal climbs to 1.33 with gestures against 0.82 without over 150 updates and
the quantity names begin to separate; the estimate the objective is taken on
has 16× the rows at the GPU batch. The scaffold is meant to be dropped:
gesturing costs 0.02 a time and words cost nothing here, so once the words
work the gesture head has nothing to earn, and the world withdraws the
possibility over the rung in any case.

### The language faculty: words, word classes and composition

The run of 2026-09-29 (`gpu_community`, 2 + 2 founders, with the gestures and
the innate lexicon above) climbed the five single-field rungs in 550 updates:
`name-fruit` 100, `name-color` 100, `name-quality` 150, `name-quantity` 100 —
where the run before it had sat at chance for 850 — and `name-price` 100. Then
it stalled in `name-all` for 1,300 updates: whole-lot success rose from 0.51 to
0.65 and stopped there, a hair under the 0.667 bar, with field coverage at
0.20–0.29 from the first checkpoint to the last and utterances of 1.9 words.

Every word existed. During `name-all` the rehearsed single-field rounds scored
**fruit 1.00, colour 1.00, quality 1.00, quantity 0.98, price 0.96** — each
field could be named alone. What never happened was putting the words
together. Across the last 1,500 whole-lot rounds in the ledger, a speaker's own
word for the lot's colour appeared in **0 of 760** of its descriptions, its
word for the quality in **0 of 760**; the other speaker used its own words for
2–12% of fields. The whole-lot rounds had grown a second, holistic code — 595
words first seen in `name-all`, compounds like `a0-a25` and `a12-a25` that
mostly meant a quantity, 9% of the rung's words inherited from the rungs that
invented them — and it carried about 1.4 fields' worth of the lot.

Nothing in the design asked for anything else. The innate lexicon applied only
where one field is asked about; a whole lot was a new context for the
speaker's policy, with no pressure to reuse what it said elsewhere. And the
listener read everything through one pooled state, so a concatenation of words
it had only ever heard alone was as foreign to it as any new utterance, and a
speaker who tried one was not understood for it.

The response gives the agents more of a language faculty. It is the debated
premise of the nativist side of the argument — that children bring innate
structure to language — made concrete and switchable, so that what it does and
does not buy can be measured. **What is innate is the architecture of words and
the kinds of thing they name; which atoms make which word, which word names
what, whether a description names every field, and in what order are all left
to the agents.**

| innate | where | what it gives |
|---|---|---|
| a name is a short *word*, said once, and names one thing in every field | `reward.lexicon`, `reward.lexicon_name_atoms` (`SpeakerLexicon`) | four fruits, four colours, four qualities, nine numbers and six prices get 27 distinct words, not homonyms told apart by how often they are repeated |
| describing a thing means naming its parts | `reward.compose` (0.30) | a whole lot is described with the speaker's own words for its fields |
| phrases have a consistent order | `reward.word_order` (0.15) | each pair of fields comes in the speaker's usual order; which order is learned |
| words are read one at a time, as nouns, adjectives or numerals | `model.lexical_reader` (`agents.LexicalReader`) | a listener understands a combination of words it learned one at a time |
| a thing's parts are named through one mental lexicon | `model.lexical_speaker` (`agents.LexicalSpeaker`) | a speaker says the word it learned for a part wherever it names that part, alone or in a description |
| say as much as the question asks, then stop | a **scaffold** on the production lexicon (`CommNet.turn_so_far`, `go_on`), withdrawn during `mutual` | asked about a whole lot, a speaker is pushed past each word while parts are unnamed and held back once all are named; asked about one field, held back after one word |
| a part once named is passed over | scaffold (`inhibit`), withdrawn with it | describing a whole lot, the next word goes to a part not yet named, instead of the last one again |
| answer the question asked | scaffold (`ask`), withdrawn with it | asked about one field, the lexicon names that field |
| every meaning has a word of its own | `reward.lexicon_exclusive` (`conventions.lexicon_exclusivity`) | 27 meanings get 27 words: no atom serves two meanings, in any field. Which word names what is the speaker's own |
| a learner takes its elders' words | `reward.lexicon_imitate` (`LexicalSpeaker.heard`) | one dialect: a listener remembers the word an elder used where it understood, and its own word moves to it |
| a number word is exact | `curriculum.numeral_near_frac`, `numeral_min_near` | in half the rounds on a quantity or a price the wrong candidates are the nearest values; "about four" loses those |
| an answer names what was asked | `train.answer_class_coef` | a word's class: the first word of the answer to a question about colour is read as a colour word |
| in the naming rungs a word is the lexicon's alone | `curriculum.own_atoms_from_rung` (`LexicalSpeaker.own_atoms`) | the token head decides whether a word goes on, another starts or the turn ends, and not which atom is said, until the market |
| a lot in a barn is a lot | `model.lexical_barn` (`CommNet.row_attention`, `barn_concepts`) | a farmer finds the lot it was asked about through its reader and names that row's parts with the same lexicon |
| what is understood can be used | `model.heard_meaning` (`CommNet.listen`) | each heard word's meaning reaches the listener's own state, not only its report heads |
| numerals sit on a number line | inside the reader, and `model.innate_concepts` | numbers are magnitudes: 3 is near 4 and far from 8, in perception and in word meaning |
| a fruit is an object, colour and quality properties, number magnitudes | `model.innate_concepts` | concepts arrive sorted into the kinds that nouns, adjectives and numerals name |

**Names are words.** The innate lexicon used to treat a whole utterance as a
form. One 2026-09-29 speaker named banana `a16` and red `a16 a16 a16 …`, twelve
times over. Edit distance on the utterances read those as two forms, 96% apart;
they are one word naming two things, which is exactly what mutual exclusivity
forbids, and a colour name like that cannot sit next to a fruit name in a
description at all. Now the name is the first word of the answer to a
one-field question, mutual exclusivity is judged word against word across every
field, and the bonus is divided by the number of words said, so a name is said
once. (Divided only when positive: saying more never softens a wrong word.)

The first local checks of that rule found the same loophole one level down,
inside the word:
- one speaker named red `a19-a19-a19-a19-a19-a19-a19-a19-a19-a19-a19-a19`
  beside `a19` for plum;
- another named pear `a12` and plum `a12-a12-a12`.

Edit distance reads a repeated atom as a different word (`a19` twelve times over
is 92% unlike `a19`), and words averaged 3.7–3.9 atoms. That breaks
composition on plain arithmetic. A turn holds 24 symbols; five words of three
atoms need 29 (15 atoms, 10 hyphens, 4 spaces), and one twelve-atom word fills
the turn alone.

So **a name is short**: its bonus is paid in full up to
`reward.lexicon_name_atoms` (2) atoms and shared out over the atoms beyond. As
before, the division applies only when the bonus is positive, so it shapes
established names and never rewards a short non-name. That keeps it clear of
the collapse length costs cause while words are being invented, where everyone
ends up saying the same short nothing. With it, `a19` twelve times over earns a
sixth of what a distinct short word earns. A two-atom repetition (`a19-a19`
beside `a19`) is still half a new word, a stepping stone rather than a
loophole. Two atoms also leaves room for multi-atom words, which the `duality`
experiment needs.

A stricter rule was tried and dropped: counting a repetition as the *same* word,
so it could never be a second name. It closed the quickest way to coin a new
word from an atom already in use. Colour was at chance at the first `name-color`
checkpoint on both local runs that had it (0.30–0.36 against 0.33), where the
run without it was at 0.43. That is one seed each at a sixteenth of the GPU's
episodes, so it is a hint, not a measurement, and it is why the gentler rule is
the one kept.

**A name is learned from words that have to work.** The speakers' lexicons were
also recording what they said in rounds they opened with a gesture. With a
gesture possible in 83% of rounds and made in 84% of those, most of a lexicon
came from turns where the gesture did the naming and the words went unchecked.
On 2026-09-29 those were the 23-token "names". In a local check they were worse:
both speakers' colour "names" were all one of their fruit words, what they said
by habit while pointing. The cross-field exclusivity rule then charged that word
against the fruit's real name as well. Now a turn opened with a gesture teaches
no name, to the speaker's lexicon or the community's. The bonus is still paid in
gestured rounds, so a name stays the same with or without pointing, and that
consistency is what makes pointing-and-saying an ostensive lesson.

**Describing a thing means naming its parts.** In a round that asks for a whole
lot — `name-all`'s open rounds, each side's lot in `mutual`, a buyer's request —
the speaker is paid, field by field, for including its own established word for
that field's value, and charged for including its word for a *different* value
of that field: calling a red apple green costs what saying "red" earns.
`reward.compose` × the mean over the five fields, so each field named is a
fifth of it, a field with no word yet scores zero, and words that name nothing
cost nothing here (the length costs price them from `mutual` on). It is the
innate lexicon extended from naming a thing to describing one. Children's first
multi-word utterances are made this way: "two red apple" is three words the
child already says alone. Order is free; the words are the speaker's own.

**Word order is consistent, and which order is learned.** Every pair of fields
a description names is paid for coming in the speaker's usual order — its
recent share of "f before g", minus a half — once the pair has a few
descriptions behind it. The comparison is pair by pair so that adding a field
never breaks the order of the ones already said: a whole-sequence comparison
would tax every new word, which is the trap the utterance-level convention
bonus fell into ([§6](#6-speaker-pressures-and-the-community)). This is the
principles-and-parameters idea in miniature: the principle (phrases have a
fixed internal order) is given, the parameter (number before noun, or after)
is set by use. The report prints each speaker's order, `quantity < colour <
fruit` and so on, which is where universals such as Greenberg's would show.

**The innate reader** (`model.lexical_reader`) is the comprehension half, and
the one that makes composition pay. The other party's turn is segmented where
the medium segments it (a word is a run of atoms joined by hyphens), and each
word is encoded from its own atoms alone, in order, with no context, so a word
means the same thing wherever it is said. Each word is then read as a noun, an
adjective, a numeral or none of these, and within its class as naming one
attribute — a noun the fruit, an adjective the colour or the quality, a numeral
the quantity or the price — and as a value of that attribute. A numeral's value
is a place on a number line with a precision, plus whatever exceptions it
learns, so "about four" is as easy to mean as "four". Finally a description's
attributes are assembled from its words, each from the word that names it: an
attention over the words by how strongly each is of that attribute's class,
with "no word names it" as an option, so a missing attribute stays uncertain
instead of being guessed from the others. The five readings are added to the
five belief heads the transformer already reads (a product of experts), and
from there reach the lineup choice, the report rungs and trading. Muted, the
reader hears no word and adds nothing, so every channel control keeps its
meaning. The straight-through gradient reaches the speaker's atoms through it
too, and since the reader is context-free, that gradient says "make this word
more like the word for red" the same way in every utterance.

`tests/test_language_faculty.py` has the demonstration. A reader trained only
on one-word utterances, one field at a time as the naming rungs teach,
decodes five-word descriptions it has never heard, in any word order, at
**above 0.95 per field**, on three seeds, with the agents' own initialisation.
It leaves a field no word names uncertain rather than guessing (a four-word
description leaves the price below 0.5 confidence). Words learned alone are
understood together.

That test used to build a bare reader with PyTorch's default initialisation,
and passed. Built the way a brain builds it, the reader could not learn some
single words at all: in every one of five seeds some field was read at
0.49–0.85 from one-word utterances. Two failure modes, both fixed:
- **a word filed under the wrong attribute was never re-filed.** Its weight
  for its own attribute went to zero, so its meaning for that attribute was
  never trained, so there was nothing to gain by moving it back. No word's
  chance of naming any attribute now falls below 0.05
  (`LexicalReader.ATTRIBUTION_FLOOR`);
- **a numeral whose place drifted could not come back.** The number line was
  a Gaussian: "seven" read by a word whose place sat at 0 at the narrowest
  width cost about 270 nats, and no gradient could move it. The kernel is now
  heavy-tailed, so near misses are still near and a far value is unlikely,
  never impossible.

With both, the worst field over five seeds is 1.00 (0.63 before).

**The production lexicon** (`model.lexical_speaker`) is the speaker's half, and
the answer to the question the reader left open: the reader understands a
description made of words learned alone, but nothing made a speaker *say* one.
A speaker's token logits get a second term while it describes a lot it can see.
The hidden state chooses which of the lot's five parts it is naming now (an
attention over the parts' concepts, each built from the field's value and kind
alone), and one shared output layer turns that concept into atoms. The layer
never sees the context, so the word for red is the same word whether red is
asked about alone or is the second thing said about a red apple: a word learned
alone is available in company. This is the production twin of the reader's
context-free lookup (in Levelt's terms, lexical access). What stays learned is
which atoms name which value, which part to name first and next, how many
words to say and when to stop, and every structural choice.

It exists because of a measurement. Two founders were each taught a one-word
dialect, a distinct atom for every value of every field, until each could name
any single field at 0.99–1.00. Put into `name-all`, their descriptions grew to
2.5 words and then shrank to one, the fruit and nothing else, within 100
updates, reusing their own words for 11% of fields. Any second word came from
the transformer's token head, for which "the colour word, second, in a
whole-lot round" was a context it had never been trained in. So the second word
was a random atom, usually some other value's name. The composition term charged
it and the listener misread it, so going on was punished and the speakers
learned to stop. Taught the same way, a speaker with the production lexicon
names a correct part of the lot on the first word of a whole-lot round every
time (1.00 on both seeds tested); without it, 0.22 and 0.75. The test is
`TestTheSpeakersLexicon`.

**Say as much as the question asks.** The first reuse experiment with the
production lexicon found the next obstacle. At update 50 of `name-all` every
description was one word long, in every arm. That word was a correct part's
word, and when the speakers were made to go on, their next word named a
different part of the lot 99–100% of the time. But after the first word they
went on 0.02–0.03% of the time. One-field rounds teach "a word, then stop" (the
name bonus is paid in full only when the name is said once), nothing in a
whole-lot round said there was more to say, and a continuation that is never
tried cannot be learned. So the production side gets the pragmatic principle
the question implies, Grice's maxim of quantity: be as informative as asked. The
speaker sees the question (the query slot: one field, or all of it) and monitors
its own turn. A part counts as named once the turn holds that part's word (what
its lexicon says for the part) or a gesture at it. Asked about a whole lot, and
with parts still unnamed, ending loses `go_on` nats (5.0) to starting
a new word. It never pushes past one word per part, and never at all in a round
that asks about one field. (`go_on` was a learnable strength until 2026-10-01;
it never moved, and it is now part of a scaffold that is withdrawn —
[below](#a-vocabulary-fit-to-hand-on).) On the
update-50 snapshot above, a push of 3 raised the chance of going on from 0.03% to
3–9%. That is a speaker drilled on "a word, then stop" by supervised teaching,
harder than any reinforcement run drills it. In the rerun, descriptions were 2.6
words long at update 50 where both controls were at 1.00. It starts at 5: at 3,
one founder went on 99% of the time and the other 10%, so the second one's
listener rarely heard its longer descriptions and it kept to one word for 100
updates; and once both went on, their token heads had learned it so hard that
descriptions ran to 8.5 words. At 5 both said five words, one per part, from
the first checkpoint ([§11](#11-findings-with-the-evidence), item 15).

**A part once named is passed over.** The same rerun showed what the push alone
does: a speaker named a second part and then said that part's word again until
the five-word cap, `a11 a28 a28 a28 a28`, because nothing told its choice of part
what it had already said. The production lexicon now knows (the same record the
push reads): describing a whole lot, when a new word starts, a part already
named in the turn loses `inhibit` (8.0) from its attention
score. That is inhibition of return, the coverage idea from machine translation,
and the plainest reading of "don't say it twice".

**Answer the question asked.** The first GPU run with the two pieces above
(2026-10-01) learned every single word, `name-fruit` in 100 updates as before,
but its words went wrong from `name-color` on, and the pieces were the cause:

- The production lexicon never looked at the question. Asked about a colour, it
  went on naming the fruit it had learned to name in `name-fruit`. One speaker
  had 4 distinct names for 8 meanings, and the colour answers carried the fruit
  (0.66 of its information) better than the colour (0.51).
- Inhibition of return acted everywhere, inside words and on one-field
  questions. A word's second atom was pushed onto another part, so the only way
  to reach the asked part was to walk through the others inside one word: atoms
  per word went 1.0, 3.7, 3.7, then 11.2 at `name-quantity`, with the quantity
  barely in it (0.10) and success at chance.
- A gesture counts as naming its part, so a speaker that pointed at the
  quantity was steered to say anything but the quantity, which is the
  point-and-say lesson the quantity words form on.

Now, asked about one field, the lexicon's attention to that field gets `ask`
(8.0): at birth it attends there 99.9% of the time, so the first
atom of an answer comes from the asked field's own concept. Inhibition of return
applies only when a new word starts in a whole-lot description; inside a word,
and on any one-field question, nothing is passed over.

#### A vocabulary fit to hand on

The first GPU run of the whole faculty (2026-10-01, `gpu_community`) passed
every naming rung at its first possible check — 100 updates each, `name-all` in
200 — and then spent three hours in `mutual`. When it was stopped, 630 updates
in, each side reported the other's whole lot 0.20 and 0.18 of the time against a
0.25 bar, and both did in one round 0.035 of the time against 0.08. Nothing was
wrong with `mutual`. It had been handed a vocabulary that no gate had looked
at, and the snapshots the run left say what was in it:

| what the gates saw | what the snapshots hold |
|---|---|
| every field named alone at 0.89–1.00 | **8 distinct words for 27 meanings** per founder: one atom for a fruit, a colour and a number, told apart by the question — which a description of a whole lot does not ask. The vocabulary *shrank* as the ladder went on: 12 and 11 words after `name-fruit`, 8 and 8 after `name-all` |
| `name-quantity` at 0.89 | **6 words for 9 quantities**. With the wrong candidates drawn at random a perfect listener wins 0.92 of rounds with six words; `mutual` wants the number, and read it at 0.40 |
| each founder read by the other | **0 of 27 words in common**, at the end of every naming rung. The six newcomers learned from both and came out as mixtures (0.59–0.81 of their words shared with a founder; the founders, 0.26 by then): eight speakers, eight dialects, and a reader that cannot know who is speaking |
| `name-all` at 0.82, every field covered | in `mutual`, **8.3 words per description** of five parts, 5.4 of them distinct. Nothing said when to stop |
| — | the three biases that make a speaker answer the question, go on, and not repeat itself were learnable, and were where they were born (`ask` 4.0 → 4.2, `go_on` 5.0 → 5.1, `inhibit` 4.0 → 3.8–3.9). The token head alone told a field's values apart at the base rate (0.14–0.33 against 0.13–0.28). **Every description was the scaffold's** |
| — | a farmer in the market looks at a barn, where the production lexicon did not exist |

Was there too much hand-holding in that run? In one place, yes — the scaffold
never left — and in another there was none at all: nothing was asked of the
vocabulary. The response has both halves. What is innate about *words* got
stronger; what is innate about *what to say* is now withdrawn before anything
is traded.

**One word per meaning.** Each speaker's whole lexicon — its word for every
value of every field, 27 rows — is pulled towards the nearest table in which no
two meanings share a word (`reward.lexicon_exclusive`;
`conventions.lexicon_exclusivity`). "Nearest" is an exact assignment of
meanings to atoms: the one that keeps the most of what the speaker already
says. Where two meanings share a word the one with the weaker claim is given
the free atom it leans to; where none do, the term is silent. This is mutual
exclusivity as children show it — a new word names something that has no word
yet — and it says nothing about *which* word names what. The first version
used the objective the naming signal uses (information plus separation), taken
over the whole table. It sharpened every word within 50 updates and separated
only some — 17, 16 and 18 distinct words of 27 on three fresh lexicons — because
two meanings both certain of one atom have no gradient left to part them. A
target does not saturate. It is summed over the meanings, not averaged: a word
has to be pulled about as hard as a round's reward pulls it, and averaged, the
game won (two meanings still on one atom at update 50).

**One dialect.** A listener that understood what an elder said remembers the
word it heard for that meaning, and its own word moves towards what it
remembers (`reward.lexicon_imitate`; `LexicalSpeaker.heard`,
`conventions.imitation_loss`). An elder is an agent born earlier, or the earlier
of two born together; the eldest keeps its own words, and the rule runs one way
because two speakers adopting each other's words at once swap them. In a round
about one field the word is the answer and understanding is having picked the
right lot. In a description of a whole lot the listener cannot be sure which
word named which field, and does not need to be: it files each field's value
under the word its own reader took to name it, and a wrong guess lands on a
different word every time while the right one is there every time the value is.
The memory peaks on the right word even for a listener guessing at random
(cross-situational learning; `tests/test_vocabulary.py` has it converge from
guesses that are right a fifth of the time). So a newcomer, who never plays a
one-field round, still takes its elders' words. A word heard from an elder also
has first claim on its atom in the matching above: the word a junior made up
for something else moves aside. It is a memory and not the batch's own rounds
because a word pulled only in the updates it was heard in was pulled back in
all the others.

**Numbers are exact.** In half the rounds on a quantity or a price
(`curriculum.numeral_near_frac`) the wrong candidates are the nearest values —
four against three and five — and in a hard whole-lot round the same share of
near misses in a number are one step away. A rung that plays number rounds is
not passed until rounds of nearest neighbours alone are won 0.80 of the time,
each describer on its own (`numeral_min_near`). Read by a perfect listener,
eight words for nine quantities score 0.89 there, seven 0.78, and the
2026-10-01 founders' six 0.67.

**An answer names what was asked.** In a round about one field the listener's
reader is taught that the first word it hears names that field
(`train.answer_class_coef`). The question is in both observations; what the word
says *about* the field is still learned only from whether the guess landed. A
word's class was otherwise learned through the reading it enables, which
saturates: on one reader in eight, taught on one-word utterances alone, two
quality words were filed as a quantity and a fruit by step 100 and stayed
there — each still answered its own question through the floor that keeps every
reading possible, so nothing moved it — and quality was then read at 0.84 from
a five-word description where every other field read 1.00.

**In the naming rungs a word is the lexicon's alone.** The token head sees the
context, so it is a second place a word can live, and the first run with the
pieces above found it there: a junior's lexicon had taken its elder's word for
a fruit (`a8`, at 0.99) and it went on saying its own old one, `a29`, which its
token head had learned to add 6.5 nats to in exactly that context — that was
the word its listener could already read, and the head was the one place the
game's gradient could still put it. Until the market the head now decides
whether a word goes on, another starts or the turn ends, and has no say in
which atom is said (`curriculum.own_atoms_from_rung` = `order`;
`LexicalSpeaker.own_atoms`). From `order` on its atoms are added again, so a
word for something that is not a part of a lot — a yes, a no, a counter-offer —
has somewhere to come from. With fewer atoms than meanings (`duality`) a word
needs more than its lexicon's one atom and the head's are on throughout.

**The scaffold is withdrawn.** "Answer the question asked", "go on until every
part is named", "not the same part twice" and "then stop" are pragmatics put in
from outside. They are now fixed strengths multiplied by one number, which is 1
below `mutual`, held for the first 100 updates of it, taken linearly to 0 over
the next 300, and 0 in every rung above (`curriculum.scaffold_fade_rung`,
`scaffold_hold_updates`, `scaffold_fade_updates`). `mutual` cannot be passed
until it is 0, so every bar that rung has is cleared by the speakers
themselves, and all of trading runs with no scaffold at all. While any of it is
left, each symbol a scaffolded speaker emits is also a lesson for its own policy
(`train.scaffold_distil`): where a word starts, which part to name; after a
word, whether to go on, start another or stop. The lesson runs to the end of
`mutual`, not only until the scaffold reaches 0 — a schedule does not know how
long a habit takes to form — and no rung above has either. Four things had to
be got right for that hand-over, each found by running it:

- *The lesson is the choice, never the atom.* Taught the scaffolded speaker's
  atoms, the token head learned the words itself: a junior's own old word was
  6.4 nats up there. (And 6.5 with the lesson changed, through the game's own
  gradient, which is why the head now has no say in the atom at all.)
- *The lesson is what the scaffold asks for, not what the scaffolded speaker
  did.* `name-fruit` asks about nothing but the fruit. A pupil taught "do what
  you just did with help" learned "the fruit, whatever is asked" there, strongly
  enough to overrule the help when the colour question came: at update 50 it
  answered every question with the fruit's name, 4 distinct words for 27
  meanings, and its teacher — the pupil plus a nudge — agreed.
- *The scaffold stays in charge while it is on.* The speaker's own scores for
  the parts are bounded (±3) and the biases are well above the bound (8), so
  whatever it has learned so far, the asked part wins.
- *A habit is learned where it was practised.* Withdrawn in `name-all`, the
  first design, the hand-over was clean: the speakers' own policy came to
  within 0.01 nats a symbol of what the scaffold asked as it faded, and
  `name-all` passed with none of it left at 0.98, five words to a description
  and every field covered. But every description in a naming rung opens the
  conversation. In `mutual` the first speaker then said its five words and the
  second, in a seat it had never described from, ran to the end of the buffer:
  `a15 a13 a15 a7 a7 a16 a7 a7 a15 a9 a16`. So the scaffold is withdrawn in
  `mutual`, the last rung over bare lots and the one where both seats describe.

The choice of part is also scored against a learned key per part now, not
against the parts' concepts, which are sums of embeddings that start at 0.02:
against those a score could not be more than a few tenths however the speaker
was trained, which is the other reason the scaffold used to do all the
choosing.

**The faculty in the market** (`model.lexical_barn`, `model.heard_meaning`). A
lot in a barn is the same kind of thing as a lot held in the hand. The farmer's
sixteen rows are scored by the reader's reading of what it heard — a row's
log-probability of its own fruit and colour — exactly as the lineup's candidates
are (the factored choice, with the farmer's lots as the candidates), so a buyer
that says "green pear" in words the farmer can read has pointed at a row and
nothing has to be learned in the market for that. The row found is a lot whose
five parts (fruit, colour, quality, stock, floor price) the lexicon names with
the words the naming rungs built. And each heard word's meaning — the concept of
the value the reader takes it to name, looked up out of context — is added to
the listener's input at the slots the word was heard in, in the same embedding
space as the things it sees. Before, the reader's decoding existed only at the
five report heads, and every *decision* was read off a hidden state that had to
learn to read the words again by itself: on the 2026-10-01 snapshot in `mutual`
the hidden state alone reported the other's lot at 0.46 / 0.54 / 0.59 / 0.42 /
0.30 per field where state and reader together reported 0.79 / 0.82 / 0.84 /
0.76 / 0.67.

**Three gates that were missing.** On `name-all`, the last rung before the
community arrives: the speaker with the fewest distinct words has one for at
least 0.95 of the meanings (`min_vocabulary_distinct`), and the two speakers
furthest apart say the same word for at least 0.90 of them
(`min_vocabulary_agreement`). On `mutual`: the scaffold is gone. The first two
are measured on what the agents say when asked (`metrics.vocabulary`: greedy,
word-only, six lots per meaning), not on their weights, and every checkpoint
prints all three. Above `name-all` the vocabulary line reads the speakers'
lexicons instead: nobody is asked about one field there, a speaker's own policy
is practised on whole lots only, and once the scaffold that made it answer the
question has gone, asking measures a skill the ladder has stopped using (four
speakers with identical lexicons read "17 of 27 words, 63% shared" when asked,
in `mutual`, with a third of the scaffold left).

```
vocabulary        : asked about one field, the speaker with the fewest has 27 distinct
                    words for 27 meanings; two speakers say the same word for 100% of
                    meanings on average, 100% for the pair furthest apart; ...
description scaffold: 67% on (withdrawn during `mutual`); the speakers' own policy is
                    0.044 nats a symbol from what it has them say; ...
```

What these did on the CPU, from scratch, is in
[§11](#11-findings-with-the-evidence), item 17.

**Names and word order keep their own clocks.** The speakers' lexicons learn
names only from one-field rounds, which end with the naming rungs, and word
order only from descriptions, which start at `name-all`. On one clock, every
name expired about 750 updates after `name-all`, and composition quietly
stopped paying half-way through `mutual`. Each now decays only on updates that
can teach it. A buyer twin made at the role split is the same agent in the
other seat, and gets its original's lexicon (a new id with an empty lexicon
meant composition never paid a buyer again). A newcomer or a newborn, who never
plays a one-field round, is scored against the community's words. The word-order
term averages over the pairs that have a history, so a field described for the
first time is not taxed.

**Innate concepts** (`model.innate_concepts`). The perceptual side carries the
same distinctions. Every lot field's embedding adds one learned vector for its
kind (object, property, magnitude), so colour and quality share something
fruit does not. Quantities and prices are also embedded through a thermometer
code (v ≥ 1, v ≥ 2, …): neighbouring magnitudes share most of their
representation, the way infants' approximate number sense orders numerosities
before there is any counting word.

**What this does not give.** No atom is assigned to any meaning from outside:
each speaker's words are the one-to-one table nearest its own random one, and
the community's are its eldest speaker's. No word is given a class. No order is
chosen. Until it is withdrawn in `mutual` a scaffold makes a description name
every part once and stop; from `order` on nothing does, and the gates judge
whether it still happens. What the faculty does change is where the vocabulary and
compositionality come from. The vocabulary is no longer negotiated between a
speaker and a listener: that each meaning has a word of its own is innate
(mutual exclusivity), which word is arbitrary, and agreement comes by imitation
— so "how the words formed" is not something these runs can be asked. What they
can be asked is what is done with the words: reading them, combining them, and
trading with them.
The listener's side of productivity is now largely innate (a reader that
composes understands novel combinations of known words by construction). The
speaker's side is given the means, a lexicon that says a part's word wherever
the part is named, and a bias towards using it; whether a speaker goes on to a
second part, and which, is learned. One cost of the production lexicon: its
output layer maps a part to one distribution over atoms, so it favours
one-atom words, and a multi-atom word has to come from the token head. That
bears on duality of patterning, and the `duality` experiment is where it would
show. What stays emergent and measurable is
everything the table above leaves out: the word forms, homonymy, whether
multi-atom words form and reuse atoms (duality of patterning, which needs the
`duality` experiment's scarce atoms to be *necessary*), which fields get said,
the word order, how far the two founders' words converge, and what survives
transmission to newborns. Every mechanism can be switched off
(`reward.compose` / `reward.word_order` = 0, `model.lexical_reader` /
`model.lexical_speaker` / `model.innate_concepts` = false) to measure what it
bought.

### Telling inherited structure from new structure

Some of the vocabulary visible at the end was inherited from the naming game
rather than caused by negotiation pressure. Every word is stamped with the rung
it first appeared in and the rung it settled in, so the report separates
"structure the naming game already produced" from "structure negotiation
specifically added" — and lists the words that first appeared in a negotiation
rung, which is where anything like offer / counter-offer / accept / refuse
vocabulary would show up.

---

## 6. Speaker pressures and the community

Five terms are paid to or charged to the *speaker* only. All are reward terms,
not restrictions: nothing ever stops an agent from saying anything.

| knob | default | what it does |
|---|---|---|
| `reward.symbol_cost` | 0.005 | per emitted symbol — atoms, hyphens and spaces. Small on purpose: it is neutral between `a-b` and `a b` and only dilutes the fused-versus-split ratio the two rows above exist for |
| `reward.atom_cost` | 0.03 | per atom after the first in a word |
| `reward.word_cost` | 0.005 | per word — a sixth of an atom, so sentences are cheap and words are not |
| `reward.rarity_cost` | 0.05 | per word, scaled by how rare the form is in the population's recent usage (`usage_half_life_updates` = 80), centred on the batch so it favours established forms without ever favouring silence |
| `reward.convention` | 0.30 | for matching the population's current form *for this meaning* (a lot, and which field of it was asked about), minus its similarity to the **closest** other meaning's form — so a code that says the same thing for two meanings earns nothing for either |
| `reward.convention_contrast_samples` | 16 | how many other meanings the contrast looks through for the closest. It is the whole cost of the term — host-side edit distances, one per episode for the bonus and this many per distinct utterance for the contrast, and an unsure speaker repeats nothing so nothing caches. At batch 4,096: 0.29 s per update at 4, 0.75 s at 16, against a ~2.7 s update. Below 8 the sample starts missing the near neighbour and a collapsed code starts earning again, so this is not where to buy speed |
| `train.shaping_reinforce` | 0.2 | how strongly these reach the speaker's token choices |

One rule decides when each of them starts: **a pressure to reuse a word is off
while the rung still has to invent one, and on at the first rung that only
reuses them.** A language has to exist before it can be economised, and the
failure is not subtle: with the costs on from the second rung a GPU run
collapsed onto a single one-atom utterance — coherence 1.000, 1.00 atoms per
word, ~1 word per utterance, 17 distinct words among 15 speakers — and colour
never left chance. Before a word for a colour exists, the cheapest way to be
short *and* to agree with everyone is for everyone to say the same short
nothing, and the costs are fully satisfiable that way. Earlier evidence pointed
the same direction: charged from episode 0 even a small cost drives the
describer to silence, and ramping them in with the first rung's success capped
that success at 0.42 against 0.62 with them off.

**The costs — length and rarity — apply to any rung that invents no new word**
(`Phase.invents`), never before `reward.costs_from_rung` as a floor. Every field
of a lot has a naming rung, so that comes out as off for the six naming rungs
and on for `mutual` and everything above it: a lot is described with the same
words whether it is held, asked for or offered. The rule is stated per rung
rather than as a threshold because the earlier ladder had two trading rungs,
`ask-qty` and `quote`, that still had a field to name, and no threshold could
spare them without sparing `mutual`.

`mutual` turned out to need them badly, and the reason is structural. It is the
first rung with **no lineup** — no candidates, no near misses — so nothing in
the task forces a message to decompose, and "reconstruct the lot from the
message" is solved perfectly by a lookup table. A run on the earlier three-field
world settled on exactly that: 48 memorised labels, field coverage 0.84, and
held-out combinations at 0.01 against 0.36 on trained ones — a productivity
ratio of 0.03 against a 0.60 bar, flat over 200 updates while every other number
improved. High coverage with near-zero held-out *is* the signature of a
memorised code, which is why both are measured.

**They also ramp in rather than switching on, in every costed rung.** Turning
them on at `mutual`'s first update was measured, and it throttled the channel
instead of shaping it: the atom cost drove words to exactly 1.00 atoms — the
hyphen went unused entirely — which caps a word at one of the atoms, so at 1.19
words per utterance about 51 possible messages had to carry 48 meanings.
Against the same episode count with the costs off, success was 0.023 where it
had been 0.142 and the lexicon 14 words where it had been 88. `mutual` invents
no new *word*, but it does have to make its messages longer — `name-all` ran at
~2.4 atoms and `mutual` grew that to ~3.6 unaided — and charging per atom while
the message has to grow is the documented failure in another dress.

So within a rung the gate waits for that rung's own rolling success to reach
`reward.costs_ramp_trigger` × its promotion floor, then ramps to full over
`reward.costs_ramp_updates`; the next rung starts a new job near zero and earns
them again. A language has to exist before it can be economised; that rule was
already applied across rungs, and this applies it inside one. The run log says
when it fires, and the snapshot carries where the ramp had got to, so a resume
in the middle of one carries on rather than restarting it:

```
[costs] mutual reached 0.104 (1.0x its 0.10 floor): the speaker starts paying
        for length and novelty, ramped in over 200 updates
```

The costs are the pressure it was missing. Among codes with room for the 48
trained (fruit, colour, quality) combinations of that three-field world, at the
current rates:

| code | capacity | cost |
|---|---|---|
| one 1-atom word | 32 — **too few** | 0.010 |
| two 1-atom words | 1,024 | 0.025 |
| three 1-atom words (one per field) | 32,768 | 0.040 |
| one 2-atom word (fused) | 1,024 | 0.050 |
| one 3-atom word (fused) | 32,768 | 0.090 |

A fused label costs 2–2.25× a multi-word one, because `atom_cost` is six times
`word_cost` — sentences are cheap, long words are not — and the per-symbol
charge, which is neutral between `a-b` and `a b`, is kept small so it does not
dilute that. And the one-atom collapse is cheaper than all of them but holds
only 32 codes (a whole lot has 3,456 values), so the **task** forbids what the
cost would otherwise reward. That is the difference from the convention bonus
above, whose collapse was both cheap *and* well paid: a length cost makes
collapse marginally cheaper, it does not make it profitable.

**The convention bonus comes on at `name-all`** (`reward.convention_from_rung`),
by the same rule one rung earlier than it can apply to the costs: every field
was invented and promoted below it, and `name-all`'s own job is to say five of
them at once. It pays for agreeing rather than for economy, and it cannot punish
a new word, because a form only counts once it has 12 recent uses behind it —
nor can it collapse the language, because it is contrastive, and a form that
fits every meaning scores its similarity to this meaning's convention minus its
similarity to the closest other meaning's, which is zero.

It waited for the community at `mutual` until a run showed what that left: the
naming rungs with nothing paying a speaker for saying the same thing twice, not
to its partner and not to itself. At `name-all` that run had within-role
coherence 0.15–0.17 — two founders with no form in common — and 686 distinct
words over sampled play for a world of 64 things. The second number is not a
large vocabulary. It is a speaker unsure of its own: the count is taken over
sampled play, and a flawless 12-word code emitted at 98% per-symbol accuracy
already reads as ~170 words. The checkpoint line and the report print the
greedy lexicon beside it — what the describers actually say when asked — so the
two cannot be confused.

A convention is a form *for a meaning*, and on a rung that asks different
questions about the same thing, the question is part of the meaning: keyed on
the lot alone, `name-all`'s conventions blended the answers to "what fruit?"
and "what is it?" into one modal form. The key carries what was asked, and a
buyer's request in the market — the whole lot, in the same layout — shares the
convention of the `name-all` describer's whole lot.

**In the naming rungs the convention is about words** (`reward.convention_words`).
Keyed on whole utterances, the bonus paid each `name-all` describer for
repeating the population's form for that exact lot. That form was the
incomplete two-word description of the moment, so the bonus was paying
speakers to stay incomplete. On 2026-09-29 utterances grew to 2.66 words at
update 150 of the rung and fell back to 1.9 as coherence rose from 0.13 to
0.47, with coverage flat throughout. The community's lexicon is now kept like a
speaker's (every speaker's first words, pooled under one pseudo-speaker, from
the first rung on). In a one-field round the bonus pays for the community's
word for that meaning, contrastively and word against word; in a whole-lot
round it pays for the community's words for the lot's parts, the same
composition as `reward.compose` but against the community instead of the
speaker. So agreement is lexical: two founders converge on the words, and a
description in the shared words is a shared description. From `mutual` on the
utterance-level bonus is unchanged.

**The contrast subtracts the closest other form, not the average one.** Against
the average it punished exactly what this project is for. A compositional code's
forms resemble each other — that is what sharing a morpheme means — so the
average reads it as undistinctive and taxes it, while a collapsed code that
names one field and drops the rest looks maximally distinctive. Scored over a
48-meaning space:

| code | contrast = mean | contrast = closest |
|---|---|---|
| compositional (fruit + colour + quality) | 0.133 | 0.062 |
| arbitrary short labels | 0.281 | 0.158 |
| **collapsed: one field, one atom** | **0.204** | **0.000** |
| collapsed: one form for everything | 0.000 | 0.000 |

The collapsed code was paid *more than the compositional code it replaces*, and
a GPU run duly found it. At `mutual` the task signal starts at zero and the
costs are off, so the convention bonus is the only thing shaping what gets said:
600 updates in, the population had gone from 27 words to 7, from 1.70 atoms per
word to 1.01, from 1.39 words per utterance to 1.03, coherence 0.31 → 0.92, and
field coverage from balanced to [0.83, 0.13, 0.05] — fruit named, colour and
quality gone, channel 0.00 of headroom. That is the collapse this section
already warned about, arrived at by a different road.

Against the *closest* other form the question becomes "is this meaning's
convention the one my form is nearest to", which a collapsed code fails by
construction: every meaning's modal form is the same, so the closest other is
identical to its own and the bonus is exactly zero. Choosing between a
compositional code and an arbitrary one is not this term's job —
`min_holdout_ratio` and `min_field_coverage` do that — but paying for the
collapse was.

### Growing the community

Six speakers and six listeners from random weights never got the lineup off
chance in 200k episodes: each farmer kept its own drifting code (coherence
0.04–0.09), and even a strong convention bonus only lifted that to ~0.2. Two and
two invent a code in ~80–140k episodes.

So a community is **founded small** (`population.founders_farmers/_buyers` = 2)
whatever its final size, and the founders take **all six naming rungs alone**
(`population.grow_from_rung` = `mutual`). From there a newcomer joins every 40
updates until the pool is full, born like any newborn — random weights, then the
transmission bottleneck on the community's transcripts — so it learns the
existing language instead of inventing another.

Growing earlier was measurably harmful: a GPU run grew 2 → 15 across the colour
rung and sat at chance throughout, because every newcomer was apprenticed on a
store of fruit-only utterances that was about to be replaced. From `mutual` on,
every rung waits for, and is judged on, the full community.

While the community is still filling up, the rung's clock stands still: the
growth does not come out of its budget. That clock also drives the rung's
temperature, entropy and gesture schedules, so those wait too, and a growing
`mutual` keeps its starting exploration until the last newcomer has arrived
(about 240 updates for 2 → 8). Whether that helps newcomers learn or only makes
the language they learn noisier has not been measured; it is noted here so it is
not mistaken for an accident if it matters.

The report measures what the pressures are for: distinct words, atoms per word,
words per utterance, the share of utterances that are silent, the share at the
buffer end, coherence within each role and across roles, and **cross-role
vocabulary overlap** — the histogram intersection of the farmer's and the buyer's
word use (1.0 = one shared vocabulary, 0.0 = two foreign codes).

---

## 7. Generations and the transmission bottleneck

**Nobody dies until `mutual`** (`population.turnover_from_rung`), the rung
newcomers start arriving in. Turnover exists to force a code a stranger can
learn; while two founders are still inventing it there is no stranger, and a
death costs half the population. Measured on a run that stalled: six
replacements in 3,200 updates, the first at update 205 — before the first code
had formed — and success rose after each newborn settled and sagged in between.
Ages accumulate anyway, so when turnover starts the living cohort is given fresh
staggered lifespans rather than expiring in the same update.

Agents age, die at a randomised lifespan (900–1,600 training updates), and are
replaced by newborns with fresh random weights. Deaths are staggered
(`population.initial_stagger`), so at any moment some agents already know the
language and some must acquire it. A code that only works between two co-adapted
agents fails to transmit and is selected against.

Age is counted in **training updates the agent took part in**, never episodes.
The first GPU run counted lifespans in episodes: a 4,096 batch shared by 2 + 2
founders aged each founder 2,048 episodes per update — 16× the CPU runs — so each
lived ~50 updates, far too short to invent anything, and the run sat at chance
with the founders already at generation 7–8.

A newborn's apprenticeship (the **transmission bottleneck**) is supervised
learning on the parent generation's recent successful transcripts. It sees
**nearly all of them** (`bottleneck.coverage` = 1.0, up to `max_samples` =
40,000), not a few hundred — with one deliberate hole: a quarter of the (fruit,
colour, quality) **combinations** in its sample are withheld from it altogether
(`bottleneck.meaning_holdout` = 0.25), so those it has to put together from parts
it did see.

That sizing is a deliberate departure from the brief's "a few hundred to
low-thousands", and the reason is worth stating. An earlier version drew a small
fixed sample — as few as 43–90 transcripts in practice — and that had the
asymmetry backwards: with a sample that thin, a form used in 2% of trades might
appear a handful of times or not at all, so *common* vocabulary was at risk of
being lost, not just obscure vocabulary. Real transmission does not look like
that. Children reliably acquire essentially everything the adults around them use
with any regularity; loss and drift are marginal phenomena at the rare end.

With near-complete coverage the asymmetry falls out of the statistics instead of
being imposed by a cap: a form used in 1% of trades still appears hundreds of
times in a 40,000-transcript sample and transmits reliably, while one used in
0.01% may genuinely not appear at all. Only the second kind is at real risk.
Sampling stays proportional to how often each meaning actually came up
(`bottleneck.frequency_skew` = 1.0), so the *composition* of a newborn's
experience still mirrors the parent generation's — it is simply no longer
artificially thin.

### The other axis: meanings, not transcripts

`coverage` is about how many *transcripts* a learner sees, and 1.0 is right on
that axis for the reason above. Compositionality comes off a different one. In
Kirby's iterated-learning models a grammar emerges because the learner is shown
a **subset of the meanings** and has to produce forms for the rest — and only a
code with reusable parts can. Shown every meaning, a learner memorises the
lookup table exactly as faithfully as its parents did (token accuracy 0.82
straight out of the apprenticeship on the GPU runs — a near-clone), and the
bottleneck selects for nothing.

That is what a run showed at `mutual`: field coverage 0.84 on a code scoring
0.36 on trained combinations and **0.01 on held-out** ones, a productivity ratio
of 0.03 against the 0.60 bar. Forty-eight memorised labels, transmitted
perfectly. The mechanism this section is built on was present and had nothing
to select.

So **`bottleneck.meaning_holdout` (0.25) withholds a slice of the (fruit,
colour, quality) combinations from each newborn** — its utterances for those are
simply not in the curriculum, and it has to put them together from parts it did
see. The slice is drawn fresh per newborn, so nothing is lost to the
*population*: every learner has a different gap, and common words are never at
risk. A learner that would be starved outright is given everything instead, and
each birth records how much was held back, both in `births.jsonl` and on the
birth line of the log.

Every birth records what vocabulary it was actually shown, and the report gives
retention for common and rare forms **separately** rather than as an aggregate,
so the asymmetry is visible rather than assumed. When a rare meaning's form is
lost and rebuilt out of words that are common elsewhere, that is the shape of an
irregular verb levelling out, and `FormTracker` logs it with before/after
examples. This is why metrics are bucketed into frequent and rare meanings: a
global average hides exactly this effect.

### What a newborn is taught, and from which rungs

Two faults that only combined once a run reached the trading rungs destroyed a
language it had spent 5,600 updates building — coherence 0.625 → 0.346, 44 words
→ 25, positional structure 0.212 → 0.103, and a scrambled channel costing
nothing, the run's own CHANNEL CARRIES NOTHING warning, correct.

**A newborn is taught every seat it will fill.** Below `curriculum.split_roles_at`
one pool fills both seats, so an agent born to replace a farmer also does every
buyer's job — and `turn_over` walks the two seats over what is then one list, so
every replacement is born a farmer. That was harmless while both seats spoke. At
the first rung where the farmer speaks nowhere (`order`: the buyer asks, the
farmer answers with its heads), a newborn taught only the farmer's side came out
of its apprenticeship with no token lesson at all (`token acc n/a over 0 own
tokens`), took the buyer's chair, and had no words for it. Five of eight founders
were replaced that way in 175 updates. `train_newborn` now takes the seats the
agent will actually fill — both while the pool is shared, its own once the roles
have split — and the birth line lists a lesson per rung *and* per seat.

**One rung cannot flush every earlier rung from the store.** The transcript
store was a plain ring buffer, and a hundred updates of the rung after `mutual`
evicted all 22,359 `mutual` transcripts — every example of the naming language a
newborn could still be taught from, since every later rung only *adds* to what
the naming rungs built. The rungs that are not the one now running share
`bottleneck.history_share` (0.4) of the buffer between them; the running rung
gets the rest; only a rung over its share is evicted from, so with one rung in
the store nothing changes. Measured on the sequence that failed — `mutual`, then
twenty batches of the next rung into a 400-slot store — `mutual` keeps its 160
and the newborn's lesson goes from 446 own tokens to 1,582.

---

## 8. Training

### The agent

Every agent is one **pre-norm causal transformer** (`agents.CommNet`), randomly
initialised, reading its private observation and the dialogue so far as a single
sequence: GELU feed-forward, learned positional embeddings, one attention mask
over observation and dialogue, LayerNorm, and a head per decision.

Ten decision heads: accept/reject, variety, quantity, price, five belief heads
(the other party's fruit, colour, quality, quantity and price — one per field of
a lot), and the lineup choice. Beside them sit the head that emits the next
message symbol and a value head.

Separate embedding tables give each observation *position* its own identity, and
each observation slot also carries the *kind* of field it holds, so a quantity in
a barn row and a quantity in a request are both quantities while "row 7's stock"
is a different thing to look at from "row 3's". There are also embeddings for the
speaker ("these were my words"), the role, and which field the round is asking
about. The shared observation layout is as wide as the barn — 16 rows of four
plus the floor price, 65 slots — and every other observation (a lot and its
query slot: 6; a lineup: 16) is padded to it.

Two pieces of structure express the two lookups the game asks for, so that
neither has to be discovered as a relational trick from nothing. The lineup
guesser's **candidate pointer** scores the final hidden state against each
candidate's own embedding: "does this description fit this candidate" is a dot
product. The farmer's **barn lookup** (`model.barn_lookup`) is one
cross-attention step from every hidden state to the barn's rows, keyed on each
row's (fruit, colour) — the same tables the words for fruit and colour are
grounded in everywhere else — and valued on its (quality, stock): a state that
has decoded "green pears" only has to reproduce those two embeddings as its
query to read back how many green pears there are and how good they are. It is
applied only to a barn, so the naming rungs never touch it, and its output
starts small so a farmer arriving at `order` keeps the listening it has.
Measured, supervised, with the answer given ([§11](#11-findings-with-the-evidence)):
without it the stock and quality of the asked-for lot stayed at the base rate
after 800 steps; with it both reached 1.00 by step 500.

**Innate word classes** (`model.factored_choice`). The lineup guesser's choice
is no longer a free pointer over a candidate's summed embedding. It is read
*through the five belief heads*: a candidate scores the sum over fields of the
log-probability the listener's fruit, colour, quality, quantity and price head
gives that candidate's value. The listener therefore parses a description into
"a kind of thing, its properties, a number" before it can pick anything — the
preconception a child brings to a new word, that it names one sort of thing —
and matches candidates attribute by attribute, which is the reading a
compositional code needs and a holistic one cannot use. Nothing about *which
words* name which field is given; only that there are fields to name. Two
things follow. On a round that varies one field, the other four terms are the
same for every candidate, so the choice *is* that field's head restricted to the
three values on offer; and those heads are the ones `mutual`, the report rungs
and `haggle` score, so naming trains reporting from the first rung instead of
handing `mutual` five untrained heads. The old pointer is kept as the control
(`factored_choice = false`).

**The innate reader and innate concepts** (`model.lexical_reader`,
`model.innate_concepts`; [§5](#the-language-faculty-words-word-classes-and-composition)).
Beside the transformer sits a second listener that reads the other party's
words one at a time, out of context — its own atom table, a position-in-word
gain so `a7-a2` is not `a2-a7`, and a small network per word — into a word
class (noun, adjective, numeral, none), an attribute within the class, and a
value, numerals on a number line. It assembles a description's attributes from
its words and adds the result to the five belief heads. On the perceptual side,
every lot field's embedding carries its kind of concept (object, property,
magnitude), and quantities and prices a thermometer-coded place on a number
line. Both are part of the architecture and so are fixed for a run
(`ARCH_KEYS`): a snapshot records whether it had them.

**The faculty in the market** (`model.lexical_barn`, `model.heard_meaning`;
[§5](#a-vocabulary-fit-to-hand-on)). Three connections that were missing
between the faculty and the rest of the agent. A barn row is scored by the
reader's reading of the other party's words — the log-probability of the row's
own fruit and colour — on top of the learned query above, so a request the
farmer can read finds its row with nothing learned in the market
(`CommNet.row_attention`). The production lexicon speaks on a barn too: its
five parts are the fruit, colour, quality and stock of the row attended to and
the floor price (`CommNet.barn_concepts`). And each heard word's meaning, as
the reader reads it, is added to the listener's input at the slots the word was
heard in (`CommNet.listen`), so what was understood reaches the state every
decision is read off — it is looked up word by word, out of context, so a state
sees exactly the words said before it, in generation and in a newborn's lessons
alike. None of the three adds a parameter.

Sizes are a declared scale choice ([§14](#14-one-method-declared-scale)): about
79k parameters per agent at the reference scale (55k before the reader), up to
~900k in `gpu_large`.

### Why Gumbel-softmax

The brief offered REINFORCE or Gumbel-softmax and asked the implementer to
document the choice. **Straight-through Gumbel-softmax on the message symbols is
the only training path**; the pure-REINFORCE path was removed rather than left to
fall out of date (it is in the git history).

Pure REINFORCE was tried first and the ablation showed it failing: after 24k
episodes, destroying every message in flight cost almost nothing, because almost
nothing was getting through. Crediting a multi-symbol discrete utterance with one
scalar at the end of an episode is too high-variance at this scale.

Straight-through Gumbel fixes the *estimator* without softening the *channel*.
The emitted symbol is still an exact one-hot in the forward pass — the partner
receives one discrete symbol, with no extra bandwidth, which is the
infinite-bandwidth cheat the brief warns about. Only the backward pass uses the
relaxation. The trade decision stays discrete and stays on REINFORCE. No babbling
or auto-encoding pretraining was needed.

Temperature anneals 1.5 → 0.5 over 1,000 updates and entropy bonuses over 800,
**counted within each rung** (`train.anneal_per_rung`). They used to count from
the start of the run, which was fine when a run was one rung; with thirteen,
everything sat at its floor from update 1,000 on, so `name-all` — which begins
thousands of updates in and has to find five-word utterances where one used to
do — explored nothing. In the run that stalled there, the only new word-forms
came from newborns. `train.gumbel_mix_reinforce` (0.1) mixes a score-function
term back over the symbols — see [§11](#11-findings-with-the-evidence) for why it
has to exist.

**Six things the training step got wrong until 2026-09-30**, found by a code
review, fixed and tested:

- **Every pooled agent took two Adam steps per update.** Below the role split one
  list fills both seats, and the step loop listed both seats, so each agent was
  clipped and stepped twice on the same gradient: about twice the intended step
  on every rung from `name-fruit` to `judge`, halving at the split. The naming
  rungs were tuned and validated under that step, so `train.lr` went from 3e-4
  to 6e-4 with the fix, to keep it.
- **The per-token terms were normalised per step,** by the number of
  utterances still going at that step. A token of the rare long utterance
  weighed as much as the first token of every utterance put together. They are
  normalised per episode now.
- **The token-entropy mask lined steps up with the first turn's positions,** so
  the second speaker's entropy was dropped in every two-sided rung.
- **The value baseline saw the actions it was scoring.** The gesture choice and
  the "shorter is better" score-function term took their advantage from a value
  read over the whole dialogue, gesture and tokens included, which can cancel
  the very cost they are meant to carry. Both use an action-independent
  baseline (the batch mean) now.
- **The ostensive lesson taught whatever field the speaker pointed at** from its
  name for the field it was *asked* about; a speaker naming the fruit while
  pointing at the colour taught the colour from the fruit word. And gesture
  supervision trained the reader on babble, which is what the ostensive lesson's
  gate exists to prevent; it now trains the transformer's heads only.
- **Switching `reward.lexicon` off silently switched off composition and word
  order** (no agent ids were passed), so the mechanisms could not be ablated one
  at a time.

The trading rungs had their own mismatch: the heads that got a gradient were not
the heads that were scored. The farmer's colour belief was scored and never
trained, the buyer's fruit belief was trained with nothing reading it, and the
buyer was paid for "decoding" a colour that is its own request. Now exactly the
scored heads train, and the buyer is not asked about its own colour.

### Snapshots and resuming

A snapshot holds the whole community — weights, optimiser state, recent usage,
the transcript store, the curriculum record, the cost ramp — and is written at
every checkpoint and every promotion. Four things a resume gets right that it
once got wrong, each found the hard way on the cloud runs
([§11](#11-findings-with-the-evidence)):

- **The file decides the architecture.** The shape-deciding settings
  (`ARCH_KEYS`: model width and depth, the atom inventory, the turns, the
  world's field sizes) have exactly one valid reading, the one the weights were
  trained under, so they are taken from the snapshot and the run says so. Every
  other setting that differs — community size, batch size, logging cadence — is
  listed but not changed: a resume may shrink a run on purpose, never by
  accident.
- **One pool comes back as one pool.** Below `curriculum.split_roles_at` the two
  seats are the same list, and whether they are is decided by the rung being
  resumed into, not by the file. Two copies restored from one pool drifted into
  two languages and then crashed the rollout a rung later.
- **The store stays on the host.** A newborn's lesson is stacked on the host
  before it moves to the device; a store mapped onto the GPU at resume mixed the
  two and the first birth after a mid-rung resume died on it.
- **A rung can be run again.** `--resume-at <rung>` winds the curriculum back to
  a rung whose mechanism has changed, resetting its clocks and keeping the
  weights, the community, the usage record and the store — `after-<rung>.pt`
  holds a curriculum already pointing at the rung after.

A code review on 2026-09-30 found more a resume got wrong, all fixed:

- **The promotion snapshot into `haggle` never split.** It was written before the
  roles split, so it held a pooled population with the curriculum already at
  `haggle`, and resuming it brought the buyers back as copies of the farmers
  (same ids, farmer role embedding). It is written after the split now, and a
  file whose two seats hold the same agents is restored as one pool and split.
  Winding back from a split population to a pooled rung keeps the farmers in
  both seats; winding forward splits the pool instead of clearing the flag.
- **The run's records started again.** `cloud_run.sh` resumes into the same
  folder, but the history behind the plots, the metrics rows behind the
  scorecard, word provenance, the example archive and the stability baseline
  were not in the snapshot. Provenance then stamped every word as first seen in
  the resumed rung, and the report read 0% inherited. They are saved now, and
  so is every random generator: each resumed segment used to replay the same
  stream of rounds and the same newborn lifespans. The market's day, season and
  inventories come back too.
- **A resume kept the snapshot's learning rate.** Loading an optimiser restores
  its rate, so a resume under a new `train.lr` changed only the newborns. The
  run's rate is set after loading, and a state that does not fit is reported.
- **`config.json` said what the command line said.** It was written before the
  device was resolved ("auto") and before the snapshot decided the architecture;
  it is written after both. The rung is found by name, not position, so a
  changed ladder cannot land a resume on the wrong rung; and the reserved
  combinations are the snapshot's, recorded in the file.

`python -m orchard.run --snapshots` lists what there is to resume from and
whether each snapshot's pool is intact (a healthy split snapshot was once
reported as the old resume bug); `--holdout-report <snapshot>` scores one on the
held-out combinations, field by field, on whatever device this machine has.
[CLOUD.md](CLOUD.md) has the commands.

### Everything is counted in training updates

**Everything that means an amount of learning is counted in training updates**
(one update = one batch), never episodes: rung budgets, promotion checks,
checkpoints, the temperature and entropy anneals, the cost ramp, community
growth, lifespans, and how long the population remembers what it has been
saying. An episode count means different amounts of learning at every batch
size, and that difference silently broke a GPU run twice — once through
lifespans, once through the population's usage memory (20,000 episodes was ~80
updates on the CPU runs but ~5 on the GPU, so the coining cost and convention
bonus were chasing a 16× shorter memory). `tests/test_config.py` fails if a
schedule is named in anything but updates.

---

## 9. What is measured

Everything the brief's §5 asks for, plus the addendum's §3, at every checkpoint:

| measure | what it is |
|---|---|
| task success | fraction of rounds ending in a mutually consistent success, always beside its muted-channel baseline |
| channel ablation | intact / scrambled / muted, and the share of the headroom the messages account for |
| per-field reports | on a report rung, for each role and each field it reports: accuracy intact and muted, and the share of the headroom the channel is worth for that field |
| topological similarity | Spearman correlation between pairwise meaning distance and pairwise message distance, against its own **shuffled null** (scipy if present, pure-Python fallback otherwise) |
| positional structure, posdis, bosdis | how strongly each slot maps to a field |
| **field coverage** | how much of *each* of the five fields a one-piece reader recovers from the messages — the best symbol slot, word position or bag of words, as a lookup table fitted on half the probes and scored on the other half, over the headroom above guessing the commonest value. The measure that exposed a variety-only code scoring 1.00 on positional structure; read in pieces because a five-field lot has 3,456 values and a whole-message statistic sits at its ceiling over any affordable probe count. A perfect code reads 1.00 at 100 probes and at 400 |
| vocabulary stats | distinct words **over sampled play**, beside the **greedy lexicon** — what the describers say when asked. The first counts variants as well as words (a flawless 12-word code at 98% per-symbol accuracy reads as ~170), so the pair is what says whether a big number is a big vocabulary or an unsure speaker. Plus word length in atoms, words per utterance, token entropy, silent share, share at the buffer end |
| stability | re-probing the same meaning against the same agent at different times |
| cross-generation intelligibility | a newborn straight out of its apprenticeship, tested against veterans it never played |
| zero-shot generalisation | success on the reserved combinations against success on trained ones — whole-round in the lineup; on a report rung **per field**, each field's held-out accuracy over trained as a share of the headroom above a message-blind guesser (the commonest value's share on the rounds actually played, per side — the shopper mostly wants LOW quality, so a trained round's quality floor is ~0.44 while a reserved one's is 0.25), then the mean of those ratios, never a ratio of means. The whole round is a conjunction of every field on both sides and sits at 0.00 while each field generalises: on the first run to promote out of `mutual` the fields transferred 0.89, 0.41 and 0.41 of their headroom and the whole round read 0.000 — fewer successes than independence would predict, because a Latin-square holdout asks for exactly the quality a correctly-read (fruit, colour) pair never showed. The checkpoint line prints both, per field by name |
| **price named** | on a trading rung, played greedily: the price each side names most often and its share of the rounds with a deal to be had; how often the named price lies inside both limits; and, where the commonest price does not, how often one that does is named. One price whatever the limits reads 0, a price that follows them 1 — the difference between agreeing and bargaining, which success does not show (`metrics.price_convention`) |
| length ↔ frequency | correlation between how often a meaning occurs and how long its message is, in symbols and in words |
| per-bucket metrics | everything above, split into frequent and rare meanings |
| form survival | whether a meaning's form survives, drifts, or is rebuilt compositionally across turnover |
| cross-role overlap | histogram intersection of the two roles' word use; reported as not yet askable while one pool fills both seats, since the two roles are then the same agents |
| gestures | training-time bookkeeping only, never a measurement of the language: the share of rounds the world allowed a gesture in, the share of those turns the speakers used one in, and how many were about each field, per rung. Every other row in this table is word-only |
| language properties | reference, productivity, word classes, intentionality, decontextualised, displaced, interchangeable, generic, perspectives, cultural transmission, duality of patterning — each with how it is measured, its value, and present / partial / absent / untestable / not reached |

Structure measures (topsim, positional structure, coverage) are taken from
speakers whose *own observation* is a lot: every describer in the naming rungs
(both of them, in a swap rung — the buyer describes in the second view, and a
check that asked the first view alone once took it for a barn speaker and left
`name-all` with an n/a bar), and the buyer in the market. A farmer in the market speaks about the lot it was
asked for, and its own observation is a barn, so a structure measure over that
would be noise; its words are judged by what the buyer recovers from them.

Degenerate outcomes are flagged loudly during the run: success stuck at chance,
vocabulary collapse, length-cap babbling, a channel that carries nothing, and the
two distinct word-structure failures (the hyphen never used, or the space never
used).

---

## 10. Output

| file | what is in it |
|---|---|
| `report.md` | rewritten at every checkpoint: summary statistics, an honest assessment, final metrics, the inferred dictionary, the curriculum and every transition, the vocabulary, length↔frequency, frequent vs rare, forms lost and rebuilt, the properties scorecard, example transcripts early/middle/late, the economy, population and transmission, and the method caveats |
| `trades.jsonl` / `.csv` | every logged episode: hidden state, the full symbol transcript, its word segmentation, both decisions, outcome, failure classification, rewards and money |
| `lineups.jsonl` | naming and report rounds, which have no trades to log: what was shown, said, asked for and reported |
| `metrics.jsonl` | every checkpoint's full metric suite |
| `births.jsonl` | every birth: what the newborn was trained on, which combinations it was not shown, how it fared against veterans |
| `promotions.jsonl` | every promotion check, passed or not, with its evidence |
| `transcripts.txt` | sampled rounds, each as an expected / dialogue / outcome block, with a banner at each rung |
| `token_semantics.json`, `history.json` | the post-hoc token analysis and the metric history the report is built from |
| `run.log` | the complete console history |
| `plots/metrics.svg`, `plots/vocabulary.svg` (+ `.png`) | progress over the run |
| `snapshots/` | `latest.pt` each checkpoint and `after-<rung>.pt` at each promotion, for `--resume` |
| `config.json` | the exact configuration used, so the run is reproducible from its own directory |
| `progress.json` | rewritten every batch: episode, update, rate, ETA, headline numbers |

Rendered messages use placeholder labels (`a7-a2 a3`) only. Meanings are never
hand-assigned; the report's "inferred dictionary" is explicitly a post-hoc
analysis of what each form correlated with, not ground truth.

### Reading a report honestly

The verdict is computed from fixed thresholds, not written by hand, so a mediocre
run cannot be talked up. It comes back as `NO EMERGENCE`, `DEGENERATE CODE`,
`NON-COMPOSITIONAL SIGNALLING`, `PARTIALLY COMPOSITIONAL` or
`COMPOSITIONAL LANGUAGE`, and the evidence for it is listed.

Reports are rewritten at every checkpoint, so a long run can be read while it is
still going and an interrupted one is never left with only raw JSONL.

---

## 11. Findings, with the evidence

Everything here was measured, most of it painfully. Each item is either guarded
by a test or written into a config comment where it would otherwise be undone by
accident.

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
score (the best constant guess on quantity goes from 0.14 to 0.28 as α goes
0→0.9), and concentrating demand on small quantities makes `stock ≥ need` nearly
always true, which raises viability and deepens the "always accept" attractor.

The shipped default is `zipf_alpha = 0.3` — a ~1.9× frequency range across
meanings, enough for the length analysis to have something to measure, mild
enough to still train. **The strong-skew regime the addendum envisages did not
train at this scale**, and that is reported as a finding rather than worked
around.

Note the secondary effect in the first table: a longer per-turn cap costs
transmission on its own. The cap has since been raised to a generous buffer (24
symbols per turn) because a small cap does worse damage: at 4 symbols, 100% of
utterances were hitting it once every field had to be named.

### Silence was the shortest word, and it was the control

The first GPU run of the current ladder had speakers silent in **24–55%** of
lineup rounds while the fruit code was forming — with the speaker costs *off*,
so nothing in the reward preferred short messages at all. Silence won anyway
because it is the most reliable message there is: one decision, with nothing
after it to get wrong. A code of `{silence, a3, a7, a12}` names four fruits, and
the silent share *rose* (20% → 42%) as success rose (0.37 → 0.45), which is what
a code using silence as a word looks like, not one giving up.

Penalising it was considered and rejected. A penalty for silence and a bonus
for "any attempt" are the same signal once advantages are centred, and either
would reach the one decision that matters — END as the first symbol — only
through the small score-function term, for the reason given below. It would
need tuning, and silence would still be used wherever it was worth more than
the penalty. Instead an empty turn is simply not a message the medium has.

That also closed a second hole. The muted control *is* silence, so a word made
of silence was unmeasurable; and a newborn apprenticed on stored transcripts
with silent turns would have been taught a target the grammar now masks, at a
cross-entropy of 1e9 — the bottleneck now skips any lesson the current grammar
forbids.

### Straight-through Gumbel cannot feel a length cost on its own

Under ST-Gumbel the symbol policy gets gradient only through the listener's
decision. The episode return — and therefore the per-symbol cost — reaches it
merely as a scalar reweighting of that term, which is far too weak to teach an
agent to stop talking. The result was unmistakable: 84% of utterances ran to the
cap, and the single commonest "word" in the whole language was
`a8-a8-a8-a8-a8-a8`, one atom repeated six times.

`train.gumbel_mix_reinforce` mixes a score-function term back in over the
symbols, restoring the direct "shorter is better" path. It works — raising the
cost with the mix on drove utterances from 3.83 symbols (93% at the cap) down to
1.20 (16%) — but the score-function term is itself high-variance and too much of
it costs transmission, so the default is a small 0.1. Set it to 0 to reproduce
the babbling, or turn it up to watch agents go quiet.

### The naming ladder climbs; what came after it did not

The runs of 2026-09-20 and 2026-09-21 on a 4090 (the `gpu_community` preset,
founded 2 + 2) are the most complete evidence there is, and they are what the
current design answers.

1. **The three-field naming ladder passed end to end.** `name-fruit` at 525
   updates on one run and 1,525 on the other (the first code forms suddenly and
   late, and the budget is now 2,000); `name-color` and `name-quality` in ~775
   each; `name-all` in 2,125, at 0.73–0.80 success against 0.667 required,
   held-out 0.74 against 0.76 on trained combinations, per-field coverage
   ~0.4–0.5, 1.4 words per utterance of 1.7 atoms. Before the partial credit and
   the per-rung anneal, the same rung had stalled at 0.55–0.66 with one-atom
   utterances (`a0` for `PEAR x0 MED`, `APPLE x3 PRIME` and `PEAR x3 HIGH`
   alike).
2. **A resume split the pool in two.** Below the trading rungs one pool fills
   both seats, and the snapshot wrote it twice; the loader restored the two lists
   separately, so the run came back with two populations under the same ids. They
   drifted apart from the first update, and within one checkpoint `mutual` had
   collapsed to fruit-only messages (coverage 0.85 / 0.13 / 0.06, success
   0.002); on another run it surfaced a rung later as an `IndexError` in the
   rollout, the first newcomer having joined one list of the two. Whether the
   pool is shared is now decided by the rung being resumed into, the loader
   restores it as one list, and a snapshot from the old bug — same ids, drifted
   weights — is detected and repaired with a warning (`tests/test_lots.py`,
   `tests/test_rungs.py`).
3. **With the pool intact, `mutual` climbed** — 0.04 → 0.36 in 900 updates,
   per-field coverage 0.96 / 0.75 / 0.77 — **and stalled on the held-out gate at
   0.02.** The listener's report heads had learned the training set's joint: with
   one quality never seen for each (fruit, colour), the quality head simply never
   produced it, however compositional the words were. Measured per field the
   held-out combinations were reported at 0.61 of the headroom of trained ones.
   The gate on report rungs is now per field ([§5](#promotion-is-on-evidence-not-on-a-schedule)).
4. **Quantity never arrived.** The old `ask-qty` rung asked the buyer to invent a
   word for quantity with hindsight, the speaker costs and the convention bonus
   already on: 0.28 against a muted 0.20 after 225 updates, with buyers babbling
   `a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1` to the buffer end. That is the
   from-scratch failure the naming rungs had already shown hindsight causes, now
   in a rung with every other pressure on as well. It is why every field of a lot
   has a naming rung, why the request *is* a lot, and why nothing above the
   naming rungs ever has to invent a word.
5. **The costs on from the transition halve the pace.** `mutual` with the costs
   fully on from its first update reached 0.015–0.023 at update 140; with them
   off, 0.037–0.14. They now wait for the rung to reach its floor and ramp in
   over 200 updates.
6. **Repetition is what the word cost buys.** With the costs on, the cheapest
   way to fill the buffer is one word repeated with spaces; a small flat
   per-symbol charge (0.005) makes twelve repeats cost 0.175 instead of 0.06
   while a five-word request costs 0.07. It is kept that small because it is
   neutral between `a-b` and `a b` and only dilutes the fused-versus-split
   ratio: at 0.01 a fused two-atom word cost 1.6× two short words, at 0.005 it
   costs 2.0×.
7. **The plain transformer cannot find a lot in its barn.** Trained supervised
   on the barn plus a hand-made two-atom request, with the answer given, the
   reference-size network learned the request's fruit and colour to 1.00 within
   100 steps and left the asked-for lot's stock at 0.26–0.34 and its quality at
   the 0.45 base rate after 800 steps. The same network with one cross-attention
   step to the barn rows reached 0.86 / 1.00 at step 300 and 1.00 / 1.00 from
   step 400 on. `offer` would have stalled on the plain network however good
   the words were; the lookup is now part of the agent ([§8](#8-training)),
   and `tests/test_lots.py` repeats the drill.
8. **The whole-round held-out number is not a productivity measurement.** The
   first run to promote out of `mutual` did so on per-field transfer of 0.89,
   0.41 and 0.41 of the headroom while the whole round read 0.000 — fewer
   successes than independent fields would predict (~60 in 2,048), because a
   Latin-square holdout asks for exactly the quality a correctly-read (fruit,
   colour) pair never showed. And the gate that passed it had taken a ratio of
   the two *means*, 0.617 against a 0.60 bar, where each field counted once
   reads 0.568: one strong field was carrying two weak ones. The gate now
   averages per-field ratios over per-pool floors and names each field, and
   `--holdout-report` prints the breakdown from any snapshot.
9. **A newborn was taught one seat and made to sit in two, and one rung flushed
   the store.** Every replacement below the split is born a farmer; on the first
   rung where the farmer speaks nowhere a newborn left its apprenticeship with
   `0 own tokens`, took the buyer's chair, and five of eight founders were
   replaced that way in 175 updates while the rung's own traffic evicted every
   `mutual` transcript a newborn could have learned from. Coherence 0.625 → 0.346,
   44 words → 25, scrambled channel costing nothing. Newborns are taught every
   seat they will fill and earlier rungs keep `bottleneck.history_share` of the
   store ([§7](#7-generations-and-the-transmission-bottleneck)).
10. **The convention contrast paid for a collapse.** Against the *average* other
   form a code that names one field and drops the rest looked maximally
   distinctive, and a GPU run at `mutual` went from 27 words to 7 and coverage
   [0.83, 0.13, 0.05] in 600 updates with everyone in perfect agreement. The
   contrast is against the closest other form ([§6](#6-speaker-pressures-and-the-community)).
11. **Four resume failures, three of them with the answer already in the file**:
   two copies of one pool (above); a store mapped onto the GPU that mixed device
   and host tensors at the first birth after a mid-rung resume; a forgotten
   `--config` printing sixty `size mismatch` lines that named tensors and never
   the setting; and a promotion snapshot that restarts the rung *after* the one
   it is named for. The loader now reads the file to the host, takes the
   architecture from it, lists every other setting that differs, and
   `--resume-at` winds a rung back ([§8](#snapshots-and-resuming)).
12. **Whole-message field coverage read a flawless five-field describer as
   0.00.** With 3,456 lots every message of a compositional code is unique over
   any affordable probe count, so both the plug-in information and its shuffled
   null sit at the ceiling. Coverage is now read off the pieces of a message,
   cross-validated ([§5](#promotion-is-on-evidence-not-on-a-schedule)): 1.00 for
   a perfect code at 100 probes, 0.78 for one right three times in four, 0.04
   for a holistic code.
13. **The five-field ladder stalled on quantity, at chance, with the words for
   the first three fields intact** (2026-09-24, `gpu_community`): `name-fruit`
   1,775 updates, `name-color` 425, `name-quality` 350, then 850 updates in
   `name-quantity` at 0.32–0.34 against 0.333 while fruit, colour and quality
   rounds still scored 0.91 / 0.83 / 0.68. The describer said one arbitrary
   atom per quantity round; live messages carried −0.008 bits of quantity
   beyond what variety already gave. Nine number words had to break symmetry
   through a listener whose reading of the channel was random. The response is
   the gesture channel and the factored listener
   ([§5](#gestures-the-scaffold-a-word-forms-on), [§8](#the-agent)): fingers
   and pointing that a listener can read from the first round, withdrawn as the
   rung goes on, with every gate still word-only. Whether that gets number
   words to form is the next thing to run.
14. **It did, and then the words were never put together** (2026-09-29,
   `gpu_community`, with gestures and the innate lexicon). All five
   single-field rungs passed in 550 updates: `name-fruit` 100 (was 1,775),
   `name-color` 100, `name-quality` 150, `name-quantity` 100 (was stuck at
   chance for 850), `name-price` 100, each judged word-only. Then `name-all`
   rose from 0.51 to 0.65 and stayed there for 1,000 updates against a 0.667
   bar, coverage 0.20–0.29 throughout, 1.9 words per utterance, while the
   rehearsed single fields scored 0.96–1.00. The whole-lot rounds had grown their
   own holistic code, and a speaker's words for colour and quality appeared in 0
   of its 760 late descriptions. The response is the language faculty
   ([§5](#the-language-faculty-words-word-classes-and-composition)).
   The same run found three gate holes and a display bug:
   - the productivity test compared independent candidates with hard near
     misses and let never-held-out quantity and price decide it;
   - coverage was gated as a mean;
   - light-check promotions skipped the forgetting check;
   - the checkpoint line printed the whole names dictionary in place of the
     greedy lexicon count, because one variable held both.

   Two readings of that run's log are worth keeping. First, with gestures
   possible in 75–100% of a naming rung's rounds, *rolling* success counts
   gestured rounds: `name-quality` read 0.96 rolling against 0.68 word-only.
   Second, the speakers' lexicons then mixed the words said after a gesture with
   those said without one; a 23-token "name" is a speaker running to the buffer
   end after its gesture took the first slot. Only the promotion checks and
   checkpoints are word-only.
15. **Given the words, putting them together took three innate pieces**
   (2026-09-30, a local CPU experiment at the reference scale, one seed per
   arm). To isolate the step `name-all` failed at, two founders were each
   taught a one-word dialect by supervised training, a distinct atom for every
   value of every field, until each named any single field at 0.99–1.00. Then
   the real training loop ran `name-all` on them (held there; batch 256).
   Checkpoints at update 50:

   | arm | words per description | coverage (per role) | success | held-out, per field |
   |---|---|---|---|---|
   | faculty off | 1.00 | 0.06 / 0.05 | 0.34 | at the floor |
   | reader, no production lexicon | 1.00 | 0.23 / 0.22 | 0.50 / 0.49 | 0.58 vs 0.57 |
   | + production lexicon, + "say as much as asked" (go_on 3) | 2.60 | 0.35 / 0.35 | 0.55 / 0.56 | 0.62 vs 0.61 |
   | + inhibition of return (go_on 3) | 4.05 | 0.63 / 0.62 | 0.67 / 0.69 | 0.81 vs 0.78 |
   | the same with go_on 5, the default | **4.98** | **1.00 / 0.99** | **0.85 / 0.89** | **0.92 vs 0.91** |

   Without the production lexicon, descriptions stayed at one word through
   update 250 in both controls. With it, the first word was always a part's
   word, but speakers went on after it 0.02–0.03% of the time; the quantity
   scaffold made them go on, and inhibition of return made the next word a new
   part rather than the last one again. At go_on 3 one founder said all five of
   its taught words in a fixed order by update 50 ("a24 a20 a16 a12 a28":
   quality, quantity, colour, fruit, price) while the other still said one: it
   went on 10% of the time, so its listener rarely heard its longer
   descriptions and could not learn to read them. By update 150 both combined
   — success 0.92, coverage 0.98–0.99 on every field, held-out 0.97 against
   0.97 per field, 74% of fields in the speaker's own taught word — but their
   token heads had learned to go on so hard that descriptions ran to 8.5 words
   of repeats. At go_on 5 both founders said exactly five words, one per part,
   from the first checkpoint. Every `name-all` check passed at update 100 at
   go_on 3 and at update 50 at go_on 5, except the experiment's own hold. By
   update 150 at go_on 5 success was 0.95, held-out 0.99 against 0.99 per field,
   and each founder's own taught word for a field was in its description
   98–100% of the time, in an order of its own: one founder said fruit,
   quantity, quality, price, colour ("a9 a21 a15 a2 a22"), the other quality,
   colour, quantity, fruit, price ("a19 a16 a6 a10 a28"). The word order is the
   one thing in those descriptions nothing chose. See
   [§5](#the-language-faculty-words-word-classes-and-composition) for each
   piece.
16. **The whole ladder passed in 700 updates, and handed `mutual` a vocabulary
   it could not use** (2026-10-01, `gpu_community`, the faculty of item 15 with
   the question read by the lexicon). Every naming rung passed at its first
   check: five single-field rungs at 100 updates each, 0.89–1.00 word-only, and
   `name-all` at 200 with 0.82, five-word descriptions and every field covered.
   `mutual` then ran for 630 updates and three hours without passing: the
   other's whole lot reported 0.055 → 0.20 of the time against 0.25, both sides
   in one round 0.001 → 0.035 against 0.08, per field 0.75 / 0.71 / 0.76 /
   0.67 / 0.58. The snapshots hold the reason, and none of it was visible in a
   gate:

   - each founder ended the naming ladder with **8 distinct words for 27
     meanings** (12 and 11 after `name-fruit`; the vocabulary shrank as fields
     were added), and **6 words for 9 quantities**;
   - the two founders had **0 of 27 words in common** at the end of every
     naming rung, and the six newcomers taught from both came out as mixtures;
   - descriptions in `mutual` ran to **8.3 words** for five parts;
   - the scaffold's three strengths were where they were born and the token
     head alone told nothing apart: **nothing the speakers did in a description
     was their own**;
   - and a farmer's production lexicon did not exist on a barn, so `offer`
     could not have used the words either.

   Every one of those is a property the next rung needed and no rung measured.
   The response is in [§5](#a-vocabulary-fit-to-hand-on): one word per meaning
   and one dialect as properties of the lexicon itself, number rounds against
   the nearest values, three new gates, the scaffold withdrawn before the
   market, and the faculty connected to the barn.
17. **With a vocabulary held to that standard the ladder ran from nothing to
   its first trades** (2026-10-01, local CPU at the reference scale, one seed;
   the response to item 16, [§5](#a-vocabulary-fit-to-hand-on)). From random
   weights, batch 256, two founders:

   | rung | updates | word-only, each describer | the vocabulary, as said when asked |
   |---|---|---|---|
   | `name-fruit` | 100 | 0.99 / 1.00 | 27 distinct words each from update 50; the younger founder's four fruit words are its elder's by then (15% of the 27 shared) |
   | `name-color` | 100 | 1.00 / 1.00 | 30% shared |
   | `name-quality` | 100 | 1.00 / 1.00 | 44% |
   | `name-quantity` | 100 | 1.00 / 1.00; nearest neighbours 1.00 / 1.00 | 78% |
   | `name-price` | 100 | 1.00 / 1.00; nearest neighbours 1.00 / 1.00 (quantities 0.99 / 0.99) | **27 words for 27 meanings, 100% shared** |
   | `name-all` | 100 | 0.96 / 0.97; five words a description, held-out 1.00 against 1.00 per field | the same 27, both speakers |

   (In that run the scaffold was withdrawn in `name-all`, the first design, so
   the rung was held to update 400: every other bar was met at update 100, and
   at 400, with none of the scaffold left, it read 0.98 / 0.99, coverage 1.00
   on every field. As the scaffold went from 100% to 0 the speakers' own policy
   went from 0.65 nats a symbol away from what it asks to 0.01.)

   The 2026-10-01 GPU run left the same six rungs with 8 words for 27 meanings,
   6 for 9 quantities and no word in common.

   From that run's `name-all` snapshot, with the scaffold withdrawn in `mutual`
   as shipped, two more runs. One with no newcomers, batch 128 and a short
   schedule (held 30 updates, withdrawn over 120), to reach the market:

   | rung | updates | at promotion | |
   |---|---|---|---|
   | `mutual` | 150 — the update the scaffold reached 0 | both report the other's lot in one round **0.91**; each side's whole lot 0.95 / 0.96; per field 0.98–1.00; held-out 0.99 against 1.00 | 4.99 words a description, the speakers' own. On the GPU run of item 16: 0.035 after 630 updates |
   | `order` | 100 | the farmer, a barn in view, reports the whole request 0.97 | every field 0.99–1.00 |
   | `offer` | 100 | the buyer reports stock, quality and floor together **0.98** (0.99 / 0.99 / 1.00) | see below |
   | `judge` | 100 | whether the deal is worth doing: the farmer calls it right 0.81 of the time (0.62 with the channel muted), the buyer 0.84 (0.52) | every earlier field still 0.98–1.00 |
   | `haggle` | 100 | the first rung that is a trade — both name the same fruit, quantity and price, and both accept: **0.30** of encounters as the gate measures it (sampled play, as in training; 0.29–0.35 over the checkpoint's three samples, 0.01 muted). Played greedily **0.55**, which is 0.81 of the encounters where a deal exists | the deal the farmer names has the fruit that was asked for 0.93 of the time, the quantity 0.93, a price inside both limits 0.82; both judge rightly whether a deal exists 0.77 |

   `offer` is the rung that asks a farmer to find a lot in its barn by the
   words it heard and say what it holds, and the one predicted to stall. It was
   at 0.95 at its first checkpoint. A round from update 40 of it, the farmer
   never having spoken from a barn before:

   ```
   buyer : a15 a1 a31 a22 a4                    (fruit 2, price 5, quantity 3, colour 1, quality 0)
   farmer: a20 a23 a15 a18 a20 a23 a20 a23 ...  (stock 5, floor 2, fruit 2, quality 2 — and on to the buffer's end)
   ```

   Every word is the naming rungs': `a20` is the word for the quantity five,
   `a23` for the third price, `a15` the buyer's own word for the fruit, said
   back. The row was found through the reader and named through the lexicon,
   with nothing learned in the market for either. What the farmer had *not*
   got is when to stop — there is no scaffold here, and it had never described
   from a barn. Sixty updates later: `a12 a23 a15 a10 a7`, five words, and the
   turn ends. That part was learned in the market, from the length cost.

   `haggle` is where a conversation first has to end in a deal: the roles
   split, each side has a limit the other cannot see, and the round counts only
   if both name the same deal and both accept it. One that did, eighty updates
   in:

   ```
   buyer : a29 a16 a21 a4 a22     two; 2.50 at most; banana; any quality; yellow
   farmer: a24 a23 a21 a10 a22    seven in stock; 2.00 at least; banana; prime; yellow
   both  : BANANA x2 at 2.50, accept
   ```

   Ten words, all of them the naming rungs' and both speakers' the same 27; the
   farmer's five are about the one row of fifteen that the buyer's words
   pointed at. The rung read 0.03 at its first checkpoint and 0.30 at its
   second, against a floor of 0.15. Played greedily at promotion (1,024
   rounds), in the rounds with a deal to be had the two named the same fruit
   1.00 of the time, the same quantity 1.00 and the same price 0.99, both
   accepted 0.97, and 0.81 of those deals were struck.

   **And the price is not bargained.** Both sides named 2.50 in 99% of those
   rounds — with the farmer's floor anywhere from 1.00 to 3.00 and the
   buyer's limit from 1.50 to 3.50, both of which had been said, and read at
   0.97–0.99. 2.50 lies inside both limits in 0.83 of the deals that exist,
   and that is nearly all of the gap between 0.81 and 1: one price whatever
   the limits are agrees every time, which is the cheapest way through a rung
   that pays for naming the same deal. Fifty updates into `bargain` it had not
   moved (2.50 in 100%; in the rounds where it does not fit, a price that does
   was named 0.00 of the time by either side), and where no deal exists both
   refused only about half the time (0.47–0.54). Nothing a checkpoint printed
   could tell this from haggling, so one now does: `price named`, on every
   trading rung (`metrics.price_convention`, [§9](#9-what-is-measured)). This
   is a first deal and not yet a market. `bargain` and `market` were not run
   to a verdict (each trains and checkpoints from this state; `bargain` read
   0.32 fifty updates in).

   The other with the community arriving: two newcomers joining the two
   founders in `mutual`, batch 256, the scaffold held for 100 updates and then
   withdrawn. (It was restarted from its own snapshot part-way through to
   shorten the wait, with the withdrawal speeded up from the shipped 300
   updates to 150.) Each newcomer came out of its lessons with the founders'
   27 words — 27 of 27, and 100% shared with the speaker furthest from it, at
   the first checkpoint after it joined — and scored 0.62 and 0.80 against the
   veterans before it had played a round. With all four present `mutual` read
   0.76 at its first checkpoint and 0.90 from update 150, the scaffold on its
   way out. It passed at update 250, the update the scaffold reached 0, at
   **0.81** both in one round (each side's whole lot 0.86 / 0.95, every field
   0.93–0.99, 5.0 words a description, held-out 0.98 against 0.97), the four
   lexicons still identical. The second speaker's descriptions were a little
   less complete than the first's once the support had gone (coverage 0.97
   against 1.00): the newcomers had had the least practice with it.

   **A birth with no scaffold.** Deaths start in `mutual` and the first comes
   at least 900 updates later, by which time the scaffold has gone and its
   lesson with it, so a newborn in the market has only its elders'
   transcripts. One was forced there: in `offer`, one of the two pooled agents
   retired and its replacement put through the ordinary apprenticeship. It
   copied 0.96 of its elders' symbols in its lessons, came out with their 27
   words (27 of 27 shared), and played `offer` against the veteran at **0.88**
   before a single live round, where two veterans score 0.98. The pair was
   still at 0.88 ten updates of play later and at 0.91–0.92 from twenty to
   forty, the two lexicons still identical — and the newborn is in every round
   of a pool of two, so that number is a whole community with a child in it.

   What this does not show: a GPU run; more than one seed; a community larger
   than four; a price that was bargained; `bargain` or `market` passed; or the
   ladder climbed in one piece. It was climbed in three legs joined by
   snapshots, and both `mutual` runs began from speakers whose own way of
   opening a description was already formed (in the first leg's `name-all`).
   The shipped ladder reaches `mutual` with that still 0.39 nats a symbol
   short of what the scaffold asks and gives it 400 updates there; in the
   short run above the second seat, which started from nothing, closed its
   distance in 150.

### The rest of the log

1. **Farmer bottleneck bug.** A static buyer-opens speaking order meant farmer
   newborns had no targets in the naming rung and buyers imitated farmer words.
   Everything that needs to know who spoke now asks the rung.
2. **Six + six from scratch never leaves chance** (4 attempts, 200k episodes
   each): farmer coherence 0.04–0.09, codes drifting 85% per 50k; no
   convention-bonus strength fixed it. Founding at 2 + 2 and growing works: that
   run reached 6 + 6 and passed the lineup (0.62), the swap rung per role, and
   the mutual rung per role with coherence ~1.0 and cross-role overlap 0.52–0.57.
3. **`haggle` plateaued at ~7% success with channel transfer 0.00–0.02** —
   always-accept plus base-rate guessing. Diagnosis: the language carried quality
   (0.88) and some variety (0.52) but **quantity at chance** (0.20 exact). The
   earlier rungs had never required it. `judge` and the per-field report checks
   exist for exactly this.
4. **The first hard-distractor design leaked the target** (42% muted against 25%
   chance) — fixed by the anchor-cluster design, and a test guards it. **The
   second weighted the near-miss field by its number of values**, so fruit
   decided 7% of open rounds once quantity had nine values; the field is now
   drawn uniformly and a test checks no field is favoured.
5. **A variety-only code can look perfectly structured.** A 2 + 2 validation run
   passed the first two rungs while carrying 1.5 bits of variety and 0.01–0.05
   bits of anything else, e.g. `a13-a13-a13-a13`. "Positional structure 1.00" was
   *redundancy*. This is why field coverage exists, and why hindsight feedback
   was added.
6. **Capacity is not the limit.** Trained supervised on a fixed compositional
   code, the 48k-parameter reference brain learns it to 100% on held-out
   combinations. If the architecture could not learn it supervised, emergence
   would be hopeless; it can.
7. **Hindsight feedback stopped the code forming** (2026-09-19) — see
   [§5](#5-the-curriculum). Mechanism, measured: the listener stops reacting to
   the still-random messages within ~25 updates either way, but without hindsight
   it still forms confident arbitrary preferences and REINFORCE eventually breaks
   the symmetry, while with hindsight the supervised loss correctly teaches it the
   messages are uninformative and it goes near-uniform.
8. **Episode-counted schedules broke GPU runs twice** — lifespans and the
   population's usage memory. Rule of thumb: anything counted in episodes must be
   checked against the batch size *and* the number of agents sharing it.
9. **Speaker costs on from the second rung collapsed the language**, and
   **growing the community through the colour rung kept it at chance** — see
   [§6](#6-speaker-pressures-and-the-community).
10. **A report can flatter a run.** One version judged a lineup success of 0.244
    against the *trading* chance (~0) and called it "far above chance". Every
    number is now compared against its own rung's chance rate.
11. **A probe embedded a buyer's request under a farmer's schema** for as long
    as one pool has filled both seats; it fitted by luck until the request became
    a lot and a price landed in the fruit table. Every probe now takes the seat's
    schema from the rung.

### Do not draw conclusions from single runs

This simulation is bimodal: a population either finds a referential convention or
it does not. Four neighbouring conditions at 40k episodes produced 76%, 0%, 92%
and 6% of the channel headroom — a spread that swamps any effect worth measuring.
One seed per arm is a coin flip with a table around it. `sweep.py` runs each arm
across seeds and reports mean, spread **and every individual seed**; see
[CLOUD.md](CLOUD.md).

---

## 12. Status: what is validated and what is not

**Validated.** The environment and its independence property; the tensor path
agreeing exactly with the readable scalar one; the reward loop; the held-out set
and the lineup builder (no reserved combination is ever a training target, every
candidate could be the answer, "pick the most central candidate" scores chance,
and no field is favoured as the deciding one); one configuration on every device
with no device-specific arithmetic; every schedule in updates; gradient
checkpointing changing nothing. Every rung of the ladder plays a real training
step, passes on perfect evidence and fails on empty evidence; every report rung
runs a checkpoint and yields the per-role, per-field evidence its gate reads;
a snapshot of one pool comes back as one pool; the bottleneck withholds the
combinations it says it does; the costs wait and ramp; the barn lookup is
inactive off a barn, small at birth, and learns the lookup when told the answer.
The vocabulary pieces (`tests/test_vocabulary.py`): the matching is exact and
gives every meaning an atom of its own; a fresh lexicon becomes one-to-one
under it, three seeds; a junior with words of its own ends up with its elder's,
from one-field answers and from descriptions it can only guess the words of;
near rounds take the nearest values and a coarse number code loses them; the
scaffold changes nothing at zero, cannot be overruled while it is on, is
withdrawn on schedule and gated, and a snapshot from before it could be
withdrawn is resumed with its strengths put back and a warning; practice with
it becomes the speaker's own choice of part and its own going on and stopping,
without bending a word; a request in words points at a barn row and the farmer
names that row with the naming rungs' words; and generation and the full pass
agree symbol for symbol on a lot, on a barn, and where the other party speaks
again afterwards. And
the price named on a trading rung is measured through the rung's own sampler:
one price whatever the limits reads 0, a price that follows them 1
(`tests/test_rungs.py`).
444 tests, about nine minutes on a CPU.

**Demonstrated in runs.** Founding at 2 + 2 and growing gets a lineup code off
chance where 6 + 6 never does; the code forms suddenly and late (~300–600
updates on the CPU, 525–1,525 on the GPU); alternating describers are necessary;
hindsight feedback must wait; the three-field naming ladder climbs to and through
`name-all`; `mutual` climbs once the pool is kept whole.

**Also demonstrated, on a GPU (2026-09-29).** With gestures and the innate
lexicon, all five single-field naming rungs pass, 100–150 updates each, judged
word-only: number words form where they had sat at chance. The words then did
not combine: `name-all` stalled at 0.65 ([§11](#11-findings-with-the-evidence),
item 14).

**Also demonstrated, on a GPU (2026-10-01).** With the faculty of 2026-09-30
every naming rung passes at its first check, `name-all` included — and hands on
a vocabulary `mutual` cannot use (8 words for 27 meanings, no word shared;
[§11](#11-findings-with-the-evidence), item 16).

**Demonstrated locally, one seed (2026-10-01).** With one word per meaning, one
dialect, exact numbers and the scaffold withdrawn, the ladder has been climbed
from random weights to its first trades at the reference scale on a CPU, in
three legs joined by snapshots: 27 words for 27 meanings shared by both
founders after 500 updates; `mutual` at 0.91 with no scaffold where the GPU run
had 0.035 (0.81 with two newcomers); `offer` — the farmer finding a lot in its
barn by the words it heard — at 0.98 in 100 updates; `judge` in 100; and
`haggle`, the first rung that is a trade, at 0.30 of encounters in 100, half of
those where a deal exists as the gate measures it (sampled play) and 0.81
played greedily — with one price, 2.50, named whatever the limits were. A
newborn with no scaffold, taught from its elders' transcripts alone, came out
with their 27 words and played `offer` at 0.88
([§11](#11-findings-with-the-evidence), item 17).

**Not yet validated — the open questions.**

- **That result on a GPU, at the community's size, on more than one seed, in
  one run.** The local runs had two founders and at most two newcomers, and
  the ladder was climbed in three legs. What to read first is on the second
  checkpoint line: `vocabulary 27/27 words, 100% shared`, and in `mutual`,
  `scaffold` falling to 0% with `words/utterance` staying at 5.
- **Whether the price is ever bargained, and `bargain` and `market`.** At
  `haggle` both sides named one price, 2.50, whatever their limits
  ([§11](#11-findings-with-the-evidence), item 17). That agrees every time and
  fits 0.83 of the deals that exist, so the rung passes and nothing is
  negotiated; fifty updates into `bargain` it had not moved, and neither rung
  above has been run to a verdict. The words are not what is missing — each
  side reads the other's five at 0.97–0.99, limits included. Every checkpoint
  of a trading rung now prints `price named … follows the limits`. If that
  still reads 0.00 after a long stretch at the GPU's batch, the game pays too
  well for a habit, and that is the next thing to change: rounds drawn so that
  no one price fits most deals (what the nearest-neighbour rounds did for the
  number words), or a bar on the rung itself.
- **Whether a word for something that is not part of a lot ever forms.** In the
  naming rungs a word is the lexicon's alone; from `order` on the token head's
  atoms are added back, and a yes, a no or a counter-offer would have to come
  from there. Nothing has needed one yet.
- **The language faculty's other pieces on a GPU**
  ([§5](#the-language-faculty-words-word-classes-and-composition)). What is
  verified (`tests/test_language_faculty.py`):
  - a name is a word, and repeating it does not make it another name;
  - mutual exclusivity holds across fields, and a name is paid in full only
    when said once;
  - a description is paid per field for the speaker's own words and charged for
    a wrong value's, whatever the word order, while the order term works pair by
    pair;
  - the reader segments words as the grammar makes them, reads a word the same
    anywhere, reads silence as nothing, and passes a gradient to the speaker's
    atoms;
  - trained on single words alone, the reader, as a brain builds it, decodes
    unheard five-word descriptions above 0.95 per field on three seeds;
  - taught one-field rounds only, the production lexicon's first word in a
    whole-lot round names a part of the lot; asked about a whole lot, a speaker
    is pushed on while parts are unnamed, never past one word per part and never
    on a one-field question, and a part once named is passed over;
  - names and word order keep their own clocks, travel to a split twin, and a
    speaker without names describes with the community's;
  - the productivity test is decided by the reserved fields alone, and on
    `name-all` it is read per field and per guesser, so reuse of one word cannot
    pass for reuse of three;
  - `name-all` needs every field;
  - an unmeasured rehearsal is unmet.

  Locally, at the reference scale on a CPU (batch 256, one seed per
  configuration), every run passed `name-fruit` word-only at update 100
  (0.82–0.96), with one- and two-atom names. The shipped configuration had
  colour at 0.48–0.54 against 0.33 at `name-color`'s first checkpoint; the
  dropped repetition rule had it at chance twice. Given words for every part,
  `name-all` now combines them in play: five-word descriptions covering every
  field and generalising to unseen combinations by update 50
  ([§11](#11-findings-with-the-evidence), item 15). The 2026-10-01 GPU run
  then showed the whole naming ladder passing with these pieces — words formed
  by the naming rungs rather than taught, both founders combining, `name-all`
  at 0.82 — and what passing it did not show (item 16).
- Whether the farmer could learn the **lookup** in `offer` *by reinforcement*
  alone is no longer asked: the row is found through the reader and named
  through the lexicon (`model.lexical_barn`), and locally `offer` passed in 100
  updates. With that switched off it is the old open question again.
- Whether separate words specialise to separate fields — the adjective question,
  and the point of the whole naming ladder. The report's "word classes" row is
  where it would show.
- (Price coordination — both sides must pick the same bin, `reward.price_tol`
  = 0 — was listed here as the likely next bottleneck. It was not one: the
  first run to get there coordinated by always naming the same price. See the
  second open question above.)
- The `duality` experiment (12 fruits against 8 atoms, so no atom can name a
  whole meaning — the setting where duality of patterning is *necessary*).
- **Whether the speakers drop the gestures.** On 2026-09-29 gestures were used
  in 31–67% of the rounds that allowed them in the single-field rungs and 7% in
  `name-all` (report §3g). But every single-field naming rung restarts the
  gesture share at 100% and passes long before its 600-update withdrawal, so
  no naming rung has yet had to do without them in training.

**Do not relax a promotion criterion to make a run pass.** The thresholds are the
experiment. Where a threshold was recalibrated here it was because the game
changed under it (five fields exactly instead of three), and the change is in the
config comment beside it.

---

## 13. Performance and engineering

Where the time goes, profiled on CPU with 8 + 8 agents on the market rung:
**784 separate agent forward passes per training step** — one per agent per
symbol step (4 turns × 24 symbols × 8 agents) plus the decisions — each
re-encoding the whole conversation so far, then a backward pass through all of
them. The 24-symbol buffer makes generation 6× longer than the old 4-symbol cap;
that is the right trade, but it makes this loop the bottleneck. On a GPU the cost
is dominated by the *number of calls*, not arithmetic, so wall time grows with
the number of agents, not with the batch: the 4090 ran the naming rungs at
2,000–4,000 episodes per second with two founders and `mutual` at ~600 with
eight. The 65-slot observation (the barn as rows) makes every prefix ~30 slots
longer than before; on the GPU that is arithmetic, not calls, and it was not
what the runs were waiting on.

What makes it fast (speed and memory only; the same code runs on a CPU):

- **Tensor world and reward.** Scenarios are sampled and trades scored as whole
  batches of tensors (`batched.py`); per-episode Python used to cap a large batch
  at ~6,700 episodes/sec. `tests/test_batched.py` asserts the tensor versions
  agree exactly with the scalar ones in `world.py` and `env.py`, which remain the
  readable definition of the rules.
- **Fixed-stride pairings**, so each agent's slice of the batch is a constant and
  the rollout never stalls the device to ask who plays what.
- **Prefix-only embedding**, slicing soft tokens before gathering, grouping
  agents once per batch, and one host copy per batch for the bottleneck store.
- **Gradient checkpointing** (`train.grad_checkpoint`, off at the reference
  size). The straight-through path backpropagates through every symbol step, so
  saved activations were ~14 MB per episode in a lineup and ~90 MB in the market
  — 56–365 GB at batch 4,096, which is how a 48 + 48 run met an out-of-memory
  error in its first batch. `CommNet.encode` now checkpoints embedding, layers
  and final norm, keyed on grad mode (not train mode — newborns leave their
  apprenticeship in eval mode) and embeds only the conversation so far.
  `tests/test_config.py` checks it gives the same update.

**The next two engineering wins, in order.**

1. **Batch the agents.** Run every agent of a role in one call: stack their
   parameters (`torch.func.stack_module_state`) and `vmap` a `functional_call`
   over the agent dimension. Pairing is a fixed stride, so every agent has the
   same number of episodes when the batch is a multiple of the agent count, and
   within a rung all agents of a role share one schema and self-mask. Adam over
   stacked tensors is per-agent already; gradient clipping must be done per agent
   slice; births replace one slice and its optimiser state; the bottleneck trains
   a single module and writes it back. This turns ~n_agents calls per symbol step
   into one, and is what would make communities above 16 + 16 practical.
2. **A KV cache for generation** — needs a hand-written causal attention layer
   instead of `nn.TransformerEncoder`, plus an equivalence test against the
   full-sequence forward. Cuts arithmetic, not call count, so do it second.

Until then, communities much larger than the presets are slow: check
`--benchmark` first.

### Diagnostics that proved their worth

- **Bias-corrected information per field in live messages** (plug-in MI minus a
  shuffled null): `lexicon.live_encoding`, and ad hoc from `lineups.jsonl`. This
  is what exposed the variety-only code.
- **A muted-channel baseline for every success number.** Any new lineup generator
  must be checked with "pick the most central candidate": it must score chance.
- **Per-role, per-field numbers** in `promotions.jsonl` and on every checkpoint
  line: `farmer reads fruit 0.95, colour 0.70, quality 0.66, quantity 0.12,
  price 0.30` is a diagnosis; `0.005` is not.
- **Snapshots plus `orchard.analyse`**, to measure without training.
- **The supervised capacity check**, before blaming the architecture.
- **Short rung-only experiments** (a few hundred updates, `--resume` from a
  snapshot) before any full run. Several full runs were lost to problems a
  ten-minute experiment would have caught.

---

## 14. One method, declared scale

**What is simulated lives in one place: the defaults in `orchard/config.py`** —
the world, the ladder, the rewards, the channel and every schedule. A GPU and a
CPU run the same code in the same fp32 arithmetic (`hardware.setup` pins it;
there is no bf16 or TF32 anywhere), with no device-specific path, so a CPU check
tests what a GPU run does.

**How big it runs is a separate, declared choice.** The presets in `configs/`
change the community, the brain, the batch, the run length and the amount of
output — and nothing else:

| preset | community | brain | batch | episodes |
|---|---|---|---|---|
| *(none)* | 2 → 6, then 6 + 6 | d48, 2 layers, 55k params | 256 | 6M |
| `gpu_small` | 2 → 6, then 6 + 6 | d64, 2 layers, 130k | 1,024 | 20M |
| `gpu_community` | 2 → 8, then 8 + 8 | d96, 3 layers, 380k | 4,096 | 100M |
| `gpu_large` | 2 → 16, then 16 + 16 | d128, 4 layers, 850k | 4,096 | 120M |

The communities are smaller than the brief's 8–20 per role at the top end on
purpose: each agent is a separate forward pass per symbol step
([§13](#13-performance-and-engineering)), and eight per role is what a 4090 runs
at a useful pace today. Batching the agents is what lifts that.

The run header prints the two separately — a `scale` line and a `method` line —
so a big run and a small one can be compared, and neither can quietly become a
different experiment. A run may change only `RUN_KEYS` (length, seed, device,
stall policy, output volume); `SCALE_KEYS` are the sizes above; **anything else
is reported as a method change** in the header and in the report's summary
statistics. `tests/test_config.py` fails if a preset touches the method, and
checks that every preset runs the same ladder, the same world and the same
held-out set. `configs/duality.json` is a declared experiment (12 fruits against
8 atoms, and a 32-symbol buffer) and says so in its header.

Old config keys raise an error naming their replacement rather than being
silently ignored. **Snapshots from before the lot layout do not load** — the
observation layout, the query embedding and the atom inventory all changed —
and the loader says so rather than failing inside `load_state_dict`.

---

## 15. Layout

```
orchard/
  config.py      every knob, JSON-serialisable; nothing is hardcoded
  world.py       lots, the barn as rows, the held-out Latin square, the independence property
  economy.py     market days, seasons, lot inventories, replenishment
  env.py         episode mechanics, word parsing, trade resolution, reward
  agents.py      the randomly-initialised transformer policies, the innate reader
                 (words -> word classes -> attributes), the production lexicon and its
                 scaffold, the barn as a lot, and the innate concepts
  batched.py     the tensor world and reward the training loop uses
  rollout.py     batched play (probes and evaluation)
  gumbel.py      training: straight-through Gumbel channel + REINFORCE decisions
  gesture.py     the gesture channel: fingers and pointing, what a seat may show,
                 when the world allows it, what the listener is taught
  curriculum.py  the ladder of rungs, the lineup and report games, and promotion
  conventions.py the population's recent usage: rarity cost, convention bonus;
                 each speaker's own lexicon of words and the community's, the naming
                 objective, one word per meaning as an exact matching, imitation of
                 elders' words, composition and word order
  population.py  ageing, death, birth, generation counting, the role split
  bottleneck.py  iterated learning: frequency-skewed apprenticeship, withheld combinations
  metrics.py     success, topsim, entropy, stability, intelligibility,
                 zero-shot, channel ablation, per-rung and per-field evidence,
                 the vocabulary probe (what each agent says for each meaning), the
                 price named on a trading rung (agreed, or bargained)
  lexicon.py     words, length↔frequency, buckets, form survival
  ledger.py      trades.jsonl / trades.csv / metrics.jsonl / births.jsonl / run.log
  render.py      human-readable transcripts (placeholder names only)
  report.py      the report and its computed verdict
  plots.py       matplotlib figures, with a dependency-free SVG fallback
  properties.py  the language-properties scorecard
  transcripts.py transcripts.txt: expected / dialogue / outcome for every round
  analyse.py     re-measure a snapshot after the fact
  hardware.py    device resolution; pins fp32 everywhere
  run.py         the CLI (also --smoke, --benchmark, --resume, --compare)
configs/         scale presets and named experiments
tests/           444 tests; test_config.py is the one that keeps the method honest,
                 test_lots.py the one that keeps the lot layout and its mechanisms honest,
                 test_language_faculty.py the one that keeps words, word classes and
                 composition honest, test_vocabulary.py the one that keeps the
                 vocabulary one-to-one and shared, the scaffold withdrawn, and the
                 faculty working on a barn
sweep.py         the same arm across seeds, because one run proves nothing
compare_runs.py  two finished runs side by side, from what they recorded
cloud_run.sh     the GPU launcher: checks the device, picks a folder, auto-resumes
```

Run it: **[CLOUD.md](CLOUD.md)**.
