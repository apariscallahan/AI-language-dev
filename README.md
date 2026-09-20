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
  the three-field product (4 × 4 × 4 things, × quantity × price) rather than by
  a long quantity range, and 8 already makes memorisation infeasible.

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
| 2.1 | a length cost per turn | no absurdly long sentences, without a hard ban | `reward.atom_cost`, `reward.word_cost` — see [§6](#6-speaker-pressures-and-the-community) |
| 2.2 | the length cost paid *per episode*, against a skewed meaning distribution | common meanings get short words (Zipf) | `world.zipf_alpha`; the correlation is reported, not assumed |
| 2.3 | a newborn's sample dominated by frequent meanings | no broadly-useless over-specific words; rare forms are at risk | `bottleneck.frequency_skew`, `bottleneck.coverage` |
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

A thing in this world is a **(fruit, colour, quality)** combination:

| field | values |
|---|---|
| fruit | APPLE, BANANA, PEAR, PLUM |
| colour | RED, YELLOW, GREEN, PURPLE |
| quality | LOW, MED, HIGH, PRIME |

4 × 4 × 4 = **64 things to name**. The three fields are separate on purpose:
that is what makes an adjective worth inventing, because a code can only
describe a combination it has never met if it names the parts.

| the Farmer privately knows | the Buyer privately knows |
|---|---|
| how much of **each (fruit, colour) lot** is in the barn | which fruit, in which colour, they want |
| the quality of each lot | the minimum quality they will accept |
| the lowest per-unit price they will take | how many they need, and the most they can pay |

Quantities run 1–8; prices are six bins from 1.00 to 3.50 in steps of 0.50.
Price is never set by the world — it has to be proposed and agreed.

A deal is possible only if the barn has that fruit in that colour, in enough
quantity, at acceptable quality, within budget. **Neither agent can determine
that alone.** Both then independently declare what they think was agreed, and
the trade succeeds only if those declarations match *each other* and describe a
deal that is actually executable. One agent being right is never enough.

The barn is laid out as one lot per (fruit, colour) — 16 cells — rather than one
per fruit. With a single colour per fruit, two thirds of shoppers could not be
served by anybody and refusing every deal beat trading.

### A quarter of the combinations are never trained on

Sixteen of the 64 combinations are reserved (`world.holdout_combo_frac` = 0.25),
and nothing in the project ever trains on them: no lineup describes one, no barn
stocks one, no shopper asks for one. They are chosen as a **Latin square** — one
quality withheld from every (fruit, colour) pair, one colour from every (fruit,
quality), one fruit from every (colour, quality) — which makes the set balanced
in every direction. Two things follow, and both matter:

* every fruit, colour and quality still appears constantly in training, so there
  is always something to generalise *from*; what is withheld is a pairing, never
  a value;
* a lineup that varies one field always has exactly three candidates that could
  be the answer. An unbalanced set leaves lineups containing a combination that
  is never anybody's target, and a guesser can then rule it out **without
  listening** — which is how an earlier version scored 0.42 against a chance rate
  of 0.33 with the channel muted.

The Latin square requires the three fields to be the same size, which is why
there are four of each. Success on the reserved combinations is the productivity
test, and it gates promotion ([§5](#5-the-curriculum)). A code that gives each
thing its own name scores at chance there however well it has drilled the rest;
a code with reusable parts does not.

### The property everything rests on

Every farmer field is drawn independently of every buyer field. No amount of
staring at your own barn tells you what the customer wants.

This was got wrong once and it is worth recording. An earlier sampler forced
roughly half of all encounters to be compatible so that viable deals would be
common enough to learn from. That made the buyer's wanted variety predictable
from the farmer's own stock — the farmer could score 0.67 against a 0.33 base
rate without listening to anything. Worse, when a farm held only one variety, the
farmer's best answer was always "the one I have", so that dimension could never
reward listening even in principle. Farms now carry a multi-variety inventory and
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
defaults; `--smoke` reports 0.686 on its own sample). The remaining third fails
for reasons that overlap — of failed rounds, 72% have quality too low, 68% not
enough stock, 47% the fruit/colour not stocked at all, 13% a price gap — so
walking away stays a real, multi-reason outcome rather than a rare edge case.
Narrowing the stock range instead would have hit the same viability while
pushing the buyer's blind-guess baseline from 0.57 to 0.70.

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

A pair exploiting base rates rather than talking scores identically in all three
conditions. This is what caught the sampler bug above, and what the verdict in
every report leans on hardest.

---

## 3. The channel: an open vocabulary

Agents do not choose from a fixed word list. They emit a **stream of symbols**,
one at a time, from

```
{ a0 … a15 }  ∪  { HYPHEN, SPACE, END }
```

(`channel.atomic_vocab` = 16 atoms; 20 token ids in all once padding is counted.)

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
beside a quality word (adjective-like) — and the report measures exactly that
("word classes"); nothing requires it.

The vocabulary is open — far more possible words than atoms — while the channel
stays discrete. `channel.max_symbols` (24 per turn) is a **buffer, not a limit**
anyone should feel: the report flags any utterance that reaches it, and the
share at the buffer end should be ~0.

### What keeps utterances short is a cost, not a rule

Length is charged **per atom after the first in a word** (`reward.atom_cost`,
0.03), plus a much smaller charge **per word** (`reward.word_cost`, 0.005).
Ending a message is free, because brevity should not be taxed.

The split is deliberate. A fused name for a whole (fruit, colour, quality) is one
long word; naming the parts is two or three short ones. Charging every symbol
equally would tax the compositional utterance for being longer overall — so words
are pressed to be short, while saying several of them costs almost nothing.
Three atoms as one word cost 0.065; the same three atoms as two words cost 0.040.
(`reward.symbol_cost`, the old flat per-symbol charge, is 0 and kept only so old
configs load.)

In the trading rungs this is also the Zipf mechanism: requests follow a Zipf-like
frequency distribution, so a meaning that comes up constantly pays its length
cost constantly, while a rare one barely pays it at all. Nothing rewards "short
words for common things" directly; it is a prediction, and `report.md` reports
the correlation rather than eyeballing it. (In the naming rungs things are drawn
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
situation to be** — the farmer about the buyer's shopping list, the buyer about
what is actually in the barn for the line it came for — and that statement is
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

---

## 5. The curriculum

Dropped straight into the full trading task from random weights, agents have to
solve five things at once before any of them pays off even once — emit a stable
signal, put true private information in it, have the other side decode it, close
the loop so decoding changes a decision, and get the trade arithmetic right as
well. A run at that setting produced success 0.000 at *every* checkpoint,
comprehension 0.000 throughout, and a channel whose scrambling cost nothing.

So the task is built up over **thirteen rungs**, and a rung is only left behind
once it has demonstrably worked. One rule shapes the whole ladder: **each rung
adds exactly one thing and keeps everything below it in play.**

### The naming rungs

Four of the thirteen are about naming, and nothing is traded until they are
done.
Each is a lineup: the describer sees one thing and which field it is being asked
about, the guesser sees three candidates and picks. The describer alternates
batch by batch, so every agent does both jobs — a single fixed describer produces
a one-way code (in the run that motivated it, the farmer's utterances had
positional structure 0.03 while the buyer's had 0.39, and every farmer newborn's
token accuracy was 0.000).

Until `haggle` **both seats are filled from one pool of agents**, so there is
one language rather than two that have to be reconciled afterwards.

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
| `name-fruit` | a lineup whose candidates share colour and quality and differ only in fruit: only the fruit needs saying | all fruit | 1/3 |
| `name-color` | colour rounds — same fruit, same quality, different colours. A word for a colour and nothing else. | 60% colour, 40% fruit | 1/3 |
| `name-quality` | quality rounds. Every round still asks one field, but which field changes, so a word has to mean the same thing wherever it appears. | 50% quality, 25% fruit, 25% colour | 1/3 |
| `name-all` | rounds where the candidates differ in any field, mostly one-field near misses, so the whole (fruit, colour, quality) is named at once | 70% all fields, 10% each single field | 1/3 |

**The single-field rungs come first because they are learnable from nothing.** A
code has to exist before it can be made compositional: `name-fruit` needs one
word per fruit and nothing else, and the rungs that follow reuse those words
rather than starting again. The point of the first rung is that 1/3 is a gradient
RL can climb, where the full task's success probability from random weights is
about 1e-3 (`--smoke` measures it as 0.0000).

**Hard rounds.** `curriculum.hard_distractor_frac` (0.75) of all-field rounds are
built as an anchor plus one-field near misses, so every field has to be named.
The first version of this built the near misses *around the target*, which made
the target the most central candidate — 42% success with the channel muted
against 25% chance. The cluster is now shuffled and the target drawn uniformly
from it, and a test checks that "pick the most central candidate" scores chance.

### The request rungs

Between naming and trading sit four rungs that are one event with the fields
turned up one at a time: **one side says facts only it holds, and the other has
to put them in its decision heads.** They are the naming ladder's method carried
into the trade format, and they exist because the two hardest fields in the
world — quantity (8 values) and price (6 bins) — have no naming rung at all.
Nothing else ever teaches them before a deal depends on both.

| rung | what is added | who reports | turns |
|---|---|---|---|
| `ask-qty` | **quantity**: the buyer says how many it needs and the farmer has to fill that number | farmer | 1 |
| `order` | the rest of the order — fruit and colour alongside the quantity | farmer | 1 |
| `quote` | **price**: the order now carries what the buyer will pay | farmer | 1 |
| `offer` | **the other direction**: the buyer asks about a lot, the farmer answers with what it holds — how much, what quality, what it wants for it — and the buyer reports what it was told | buyer | 2 |
| `judge` | **the decision**: the same dialogue, and now the buyer has to say whether the deal is worth doing at all, weighing what it was told against what it needs | buyer | 2 |

Like a naming rung, each is **judged on the field it introduced** and has to
show it **still carries** the ones below it, field by field against a muted
channel. A conjunction of four fields would hide which one is at chance, and
that is exactly what the old ladder did: `haggle` reported 0.07 success, and the
diagnosis — quality 0.88, variety 0.52, **quantity 0.20** — had to be dug out by
hand afterwards.

`offer` is the half of the market dialogue that nothing else trains. In every
rung below it the buyer talks and the farmer acts; in `haggle` the farmer has to
describe its own barn, and without `offer` that skill would have to appear at
the same moment as the price agreement and the accept/reject decision.

`judge` exists because of the way `haggle` failed: **always accept**, plus
base-rate guessing, for 7% success and a channel carrying 0.00–0.02. Roughly 68%
of rounds are worth doing, so accepting everything scores 0.68 and looks like
competence. `judge` scores nothing but that decision, and it is judged on the
**gain over silence** rather than on the raw rate — a pair that accepts
everything scores exactly what a mute pair scores, which is zero of the
headroom, and cannot pass. ("Twice the chance rate", the bar everywhere else, is
not a reachable number when silence already scores 0.68.)

### The trading rungs

| rung | what is added | turns | chance |
|---|---|---|---|
| `mutual` | both hold a private thing and each must report the other's; still no price, no accept/reject | 2 | measured (muted channel) |
| `haggle` | the pool splits into farmers and buyers, and the deal starts paying: both sides must name the same one, and it only counts if it is actually executable | 2 | ~0 |
| `bargain` | several turns, so counter-offers become possible | 4 | ~0 |
| `market` | the full economy: persistent stock, restocking, viability | 4 | ~0 |

**The role split moved to `haggle`.** Everything below it is one language in two
seats: the request rungs run in both directions — `quote` has the buyer saying
prices, `offer` has the farmer saying them — and one pool learns both from the
same words. The split exists so the two sides can diverge in *strategy*, which
only starts to matter where selling and buying pay differently.

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
- on mixed rungs, every rehearsed kind of round still clear of chance;
- on `name-all` and `mutual`, **structure** — positional structure ≥ 0.15 and
  **field coverage** ≥ 0.30 — and **success on the reserved combinations**, at
  least 60% of the rate on trained ones (`min_holdout_ratio`).

In the lineup rungs and `mutual` every one of these is checked **per role**,
never pooled: each role's own utterances must show topsim over null and
positional structure, and each role must decode in the view where it is the one
decoding, or report the other's thing (`mutual_min_report` = 0.30). A pooled
average would let a fluent partner carry a role that never learned to speak.

Field coverage is the check that matters most, because positional structure is
fooled by redundancy: a variety-only code like `a13-a13-a13-a13` scores 1.00 on
it by naming the variety in every slot. Coverage asks how much of *each* field
the messages carry, corrected for chance.

Success alone is never enough, because a pair can score on base rates without
saying anything. Every check, passed or not, is written to `promotions.jsonl`.

**Every rung has a budget** (`curriculum.rung_budget_updates`, in training
updates): 80–1,500 for the single-field naming rungs, 80–2,500 for `name-all`
and `offer`, 80–2,000 for `ask-qty`, `order` and `quote`, 80–3,500 for `mutual`,
`haggle` and `bargain`, open for `market`.
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
the lineup target, the partner's actual meaning, the order that was placed, the
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
updates); without it, it formed at ~550 updates. So every rung where a code has
to form from nothing runs without it, and it joins where it was meant to help:
drawing quantity and quality out of a code that already carries variety.

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

Four terms are paid to or charged to the *speaker* only. All are reward terms,
not restrictions: nothing ever stops an agent from saying anything.

| knob | default | what it does |
|---|---|---|
| `reward.atom_cost` | 0.03 | per atom after the first in a word |
| `reward.word_cost` | 0.005 | per word — a sixth of an atom, so sentences are cheap and words are not |
| `reward.rarity_cost` | 0.05 | per word, scaled by how rare the form is in the population's recent usage (`usage_half_life_updates` = 80), centred on the batch so it favours established forms without ever favouring silence |
| `reward.convention` | 0.30 | for matching the population's current form *for this meaning*, minus the similarity to other meanings' forms, so one form for everything earns nothing |
| `train.shaping_reinforce` | 0.2 | how strongly these reach the speaker's token choices |

**All of it is off until `offer`** (`reward.costs_from_rung`) — off through every rung that still has to invent a word, on at the first rung that only reuses them. A language has to
exist before it can be economised, and the failure is not subtle: with the costs
on from the second rung a GPU run collapsed onto a single one-atom utterance —
coherence 1.000, 1.00 atoms per word, ~1 word per utterance, 17 distinct words
among 15 speakers — and colour never left chance. Before a word for a colour
exists, the cheapest way to be short *and* to agree with everyone is for everyone
to say the same short nothing, and the costs are fully satisfiable that way.
Earlier evidence pointed the same direction: charged from episode 0 even a small
cost drives the describer to silence, and ramping them in with the first rung's
success capped that success at 0.42 against 0.62 with them off.

### Growing the community

Six speakers and six listeners from random weights never got the lineup off
chance in 200k episodes: each farmer kept its own drifting code (coherence
0.04–0.09), and even a strong convention bonus only lifted that to ~0.2. Two and
two invent a code in ~80–140k episodes.

So a community is **founded small** (`population.founders_farmers/_buyers` = 2)
whatever its final size, and the founders take **all four naming rungs alone**
(`population.grow_from_rung` = `mutual`). From there a newcomer joins every 40
updates until the pool is full, born like any newborn — random weights, then the
transmission bottleneck on the community's transcripts — so it learns the
existing language instead of inventing another.

Growing earlier was measurably harmful: a GPU run grew 2 → 15 across the colour
rung and sat at chance throughout, because every newcomer was apprenticed on a
store of fruit-only utterances that was about to be replaced. From `mutual` on,
every rung waits for, and is judged on, the full community.

The report measures what the pressures are for: distinct words, atoms per word,
words per utterance, the share of utterances that are silent, the share at the
buffer end, coherence within each role and across roles, and **cross-role
vocabulary overlap** — the histogram intersection of the farmer's and the buyer's
word use (1.0 = one shared vocabulary, 0.0 = two foreign codes).

---

## 7. Generations and the transmission bottleneck

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
40,000), not a few hundred.

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

Every birth records what vocabulary it was actually shown, and the report gives
retention for common and rare forms **separately** rather than as an aggregate,
so the asymmetry is visible rather than assumed. When a rare meaning's form is
lost and rebuilt out of words that are common elsewhere, that is the shape of an
irregular verb levelling out, and `FormTracker` logs it with before/after
examples. This is why metrics are bucketed into frequent and rare meanings: a
global average hides exactly this effect.

---

## 8. Training

### The agent

Every agent is one **pre-norm causal transformer** (`agents.CommNet`), randomly
initialised, reading its private observation and the dialogue so far as a single
sequence: GELU feed-forward, learned positional embeddings, one attention mask
over observation and dialogue, LayerNorm, and a head per decision.

Ten decision heads: accept/reject, variety, quantity, price, four belief heads
(the other party's fruit, quantity, quality and price), the lineup choice, and a
belief about the other party's colour. Beside them sit the head that emits the
next message symbol and a value head.

Separate embedding tables give each observation *position* its own identity, so
"stock of GREEN APPLE" is a different thing to look at from "stock of GOLD PEAR"
even though both are quantities; there are also embeddings for the speaker
("these were my words"), the role, and which field the round is asking about.

Sizes are a declared scale choice ([§14](#14-one-method-declared-scale)): 55k
parameters per agent at the reference scale, up to 849k in `gpu_large`.

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

Temperature anneals 1.5 → 0.5 over 1,000 updates; entropy bonuses anneal over
800. `train.gumbel_mix_reinforce` (0.1) mixes a score-function term back over the
symbols — see [§11](#11-findings-with-the-evidence) for why it has to exist.

### Everything is counted in training updates

**Everything that means an amount of learning is counted in training updates**
(one update = one batch), never episodes: rung budgets, promotion checks,
checkpoints, the temperature and entropy anneals, community growth, lifespans,
and how long the population remembers what it has been saying. An episode count
means different amounts of learning at every batch size, and that difference
silently broke a GPU run twice — once through lifespans, once through the
population's usage memory (20,000 episodes was ~80 updates on the CPU runs but
~5 on the GPU, so the coining cost and convention bonus were chasing a 16×
shorter memory). `tests/test_config.py` fails if a schedule is named in anything
but updates.

---

## 9. What is measured

Everything the brief's §5 asks for, plus the addendum's §3, at every checkpoint:

| measure | what it is |
|---|---|
| task success | fraction of rounds ending in a mutually consistent success, always beside its muted-channel baseline |
| channel ablation | intact / scrambled / muted, and the share of the headroom the messages account for |
| topological similarity | Spearman correlation between pairwise meaning distance and pairwise message distance, against its own **shuffled null** (scipy if present, pure-Python fallback otherwise) |
| positional structure, posdis, bosdis | how strongly each slot maps to a field |
| **field coverage** | bias-corrected information about *each* field in live messages — the measure that exposed a variety-only code scoring 1.00 on positional structure |
| vocabulary stats | distinct words, word length in atoms, words per utterance, token entropy, silent share, share at the buffer end |
| stability | re-probing the same meaning against the same agent at different times |
| cross-generation intelligibility | a newborn straight out of its apprenticeship, tested against veterans it never played |
| zero-shot generalisation | success on the reserved combinations against success on trained ones |
| length ↔ frequency | correlation between how often a meaning occurs and how long its message is, in symbols and in words |
| per-bucket metrics | everything above, split into frequent and rare meanings |
| form survival | whether a meaning's form survives, drifts, or is rebuilt compositionally across turnover |
| cross-role overlap | histogram intersection of the two roles' word use |
| language properties | reference, productivity, word classes, intentionality, decontextualised, displaced, interchangeable, generic, perspectives, cultural transmission, duality of patterning — each with how it is measured, its value, and present / partial / absent / untestable / not reached |

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
| `lineups.jsonl` | naming rounds, which have no trades to log |
| `metrics.jsonl` | every checkpoint's full metric suite |
| `births.jsonl` | every birth: what the newborn was trained on, which meanings it never saw, how it fared against veterans |
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
   earlier rungs had never required it. `order` was added for exactly this.
4. **The first hard-distractor design leaked the target** (42% muted against 25%
   chance) — fixed by the anchor-cluster design, and a test guards it.
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
candidate could be the answer, and "pick the most central candidate" scores
chance); one configuration on every device with no device-specific arithmetic;
every schedule in updates; gradient checkpointing changing nothing. Every rung
of the ladder plays a real training step, passes on perfect evidence and fails
on empty evidence — the check that would have caught the trading rungs going
unexercised for as long as they did. 178 tests, about two minutes.

**Demonstrated in runs.** Founding at 2 + 2 and growing gets a lineup code off
chance where 6 + 6 never does; the code forms suddenly and late (~300–600
updates); alternating describers are necessary; hindsight feedback must wait.

**Not yet validated — the open questions.**

- The full ladder has never been climbed end to end. `name-fruit` and the swap
  and mutual rungs have been passed by a 2 + 2 → 6 + 6 population on an older,
  narrower world; the current four-field naming ladder has not.
- **Whether colour is learnable at all** is the live question. The cumulative
  mixture, the deferred speaker costs and the deferred community growth are all
  aimed at it, and none of them has been shown to work yet — they are diagnosis
  plus a fix, not a result.
- Whether separate words specialise to separate fields — the adjective question,
  and the point of the whole naming ladder. The report's "word classes" row is
  where it would show.
- **The request rungs have never been run for real.** `ask-qty`, `order`,
  `quote`, `offer` and `judge` are new, and what is verified is that they play,
  score and judge correctly — not that a population learns them.
- `haggle` and above. Price coordination (both sides must pick the same bin,
  `reward.price_tol` = 0) is the likely next bottleneck; if it stalls there, that
  is a candidate for a further rung rather than for quietly loosening the test.
- The `duality` experiment (12 fruits against 8 atoms, so no atom can name a
  whole meaning — the setting where duality of patterning is *necessary*).

**Do not relax a promotion criterion to make a run pass.** The thresholds are the
experiment.

---

## 13. Performance and engineering

Where the time goes, profiled on CPU with 8 + 8 agents on the market rung:
**784 separate agent forward passes per training step** — one per agent per
symbol step (4 turns × 24 symbols × 8 agents) plus the decisions — each
re-encoding the whole conversation so far, then a backward pass through all of
them. The 24-symbol buffer makes generation 6× longer than the old 4-symbol cap;
that is the right trade, but it makes this loop the bottleneck. On a GPU the cost
is dominated by the *number of calls*, not arithmetic, so wall time grows with
the number of agents, not with the batch.

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
  apprenticeship in eval mode) and embeds only the conversation so far: 0.17 /
  0.71 / 2.4 MB per episode for lineup / mutual / market, so batch 4,096 needs
  ~10 GB in the market rung. `tests/test_config.py` checks it gives the same
  update.

**The next two engineering wins, in order.**

1. **Batch the agents.** Run every agent of a role in one call: stack their
   parameters (`torch.func.stack_module_state`) and `vmap` a `functional_call`
   over the agent dimension. Pairing is a fixed stride, so every agent has the
   same number of episodes when the batch is a multiple of the agent count, and
   within a rung all agents of a role share one schema and self-mask. Adam over
   stacked tensors is per-agent already; gradient clipping must be done per agent
   slice; births replace one slice and its optimiser state; the bottleneck trains
   a single module and writes it back. This turns ~n_agents calls per symbol step
   into one, and is what would make 128 + 128 practical.
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
- **Per-role, per-field numbers** in `promotions.jsonl`.
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
| `gpu_small` | 2 → 12, then 12 + 12 | d64, 2 layers, 124k | 1,024 | 20M |
| `gpu_community` | 2 → 32, then 32 + 32 | d96, 3 layers, 374k | 4,096 | 100M |
| `gpu_large` | 2 → 64, then 64 + 64 | d128, 4 layers, 849k | 4,096 | 120M |

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
silently ignored, and old snapshots still load (`Config.from_dict(allow_legacy=True)`).

---

## 15. Layout

```
orchard/
  config.py      every knob, JSON-serialisable; nothing is hardcoded
  world.py       private state, the held-out Latin square, the independence property
  economy.py     market days, seasons, multi-lot inventories, replenishment
  env.py         episode mechanics, word parsing, trade resolution, reward
  agents.py      the randomly-initialised transformer policies
  batched.py     the tensor world and reward the training loop uses
  rollout.py     batched play (probes and evaluation)
  gumbel.py      training: straight-through Gumbel channel + REINFORCE decisions
  curriculum.py  the ladder of rungs, their worlds, and promotion
  conventions.py the population's recent usage: rarity cost, convention bonus
  population.py  ageing, death, birth, generation counting, the role split
  bottleneck.py  iterated learning, frequency-skewed apprenticeship
  metrics.py     success, topsim, entropy, stability, intelligibility,
                 zero-shot, channel ablation, per-rung evidence
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
tests/           170 tests; test_config.py is the one that keeps the method honest
sweep.py         the same arm across seeds, because one run proves nothing
compare_runs.py  two finished runs side by side, from what they recorded
cloud_run.sh     the GPU launcher: checks the device, picks a folder, auto-resumes
```

Run it: **[CLOUD.md](CLOUD.md)**.
