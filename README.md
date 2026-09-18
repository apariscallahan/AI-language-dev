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

### The control panel

```bash
python orchard_gui.py
```

Or double-click `Orchard.bat`. A small Windows panel with presets, the settings
worth changing, a progress bar and a live view of the last checkpoint. It is a
front end for the command below and nothing more -- it writes a config, launches
the trainer, and watches the `progress.json` the trainer drops beside its output,
so the panel can never disagree with the CLI about what a setting means.

One setting there is not what it looks like. **Generations cannot be set
directly**: an agent ages by the episodes it personally plays and dies at its
lifespan, so how many times a lineage turns over falls out of run length,
population size and lifespan together. The panel asks for generations because
that is what you actually want to choose, converts it to an episode budget, and
always shows you the number it arrived at.

### The command line

```bash
python -m orchard.run --smoke
```

Runs the environment with scripted agents and no learning at all — the spec's
build-order step 1. It prints the chance-level success rate (essentially zero),
confirms an oracle pair can convert every viable scenario, and shows one rendered
episode.

```bash
python -m orchard.run --config configs/open.json --out runs/main
```

A full run: trains, checkpoints the whole metric suite, writes a trade ledger of
every episode, plots progress, and produces a final report with an honest verdict.

```bash
python -m unittest discover -s tests
```

---

## What makes language necessary here

Language is only needed when one party holds something the other cannot see and
cannot guess. That is built in explicitly and enforced in code:

| Farmer privately knows | Buyer privately knows |
|---|---|
| how much of **each variety** is in the barn | which single variety they want |
| the quality of each variety | the minimum quality they will accept |
| the lowest price they will take | the most they can pay |

A deal is possible only if the barn has the wanted variety, in enough quantity, at
acceptable quality, within budget. **Neither agent can determine that alone.**
Both then independently declare what they think was agreed, and the trade succeeds
only if those declarations match *each other* and describe a deal that is actually
executable. One agent being right is never enough.

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

- a **word** is a run of atoms joined by `HYPHEN` — `a7-a22-a3` is one word;
- a **sentence** (one turn) is words separated by `SPACE`;
- `HYPHEN` and `SPACE` are structural marks and mean nothing themselves, exactly
  as no atom means anything at the start.

The vocabulary is therefore open — there are far more possible words than atoms —
while the channel stays discrete and narrow. Nothing constrains where the marks go:
messages are parsed leniently, so any symbol sequence is legal and stray marks just
cost the speaker a symbol. **Whether word-like structure appears at all is
measured, never enforced** (`orchard/lexicon.py`).

### What keeps utterances short is a cost, not a rule

Every emitted symbol — atoms, hyphens and spaces alike — is charged for. Ending a
message is free, because brevity should not be taxed.

That single cost is also the whole Zipf mechanism. Requests follow a Zipf-like
frequency distribution, so a meaning that comes up constantly pays its length cost
constantly, while a rare one barely pays it at all. Nothing rewards "short words
for common things" directly; it is a prediction, and `report.md` reports the
correlation rather than eyeballing it.

---

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
transmission on its own, which is why `max_symbols` starts small, exactly as the
addendum recommends.

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

## Generations, and what gets lost

Agents age, die at a randomised lifespan, and are replaced by newborns with fresh
random weights. Deaths are staggered, so at any moment some agents already know the
language and some must acquire it. A code that only works between two co-adapted
agents fails to transmit and is selected against.

A newborn's apprenticeship (the **transmission bottleneck**) is supervised learning
on a *deliberately small* sample of recent successful trades — a few hundred, never
the full history. The squeeze is the mechanism: a lookup table cannot survive it, a
systematic code can be rebuilt from fragments.

The sample is also **skewed toward what was common** (`bottleneck.frequency_skew`).
A learner sees hundreds of ordinary trades and may see a given unusual one never.
So it reliably generalises the pattern for common cases and often simply cannot
reproduce whatever narrow form a rare case picked up — which is where vocabulary
loss comes from, with no separate forgetting mechanism. When a rare meaning's form
is lost and rebuilt out of words that are common elsewhere, that is the shape of an
irregular verb levelling out, and `FormTracker` logs it with before/after examples.

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
  rollout.py     batched play + REINFORCE with a learned baseline
  gumbel.py      straight-through Gumbel channel (the default; see below)
  population.py  ageing, death, birth, generation counting
  bottleneck.py  iterated learning, frequency-skewed curriculum
  metrics.py     spec section 5: success, topsim, entropy, stability,
                 cross-generation intelligibility, zero-shot, channel ablation
  lexicon.py     addendum section 3: words, length↔frequency, buckets, form survival
  ledger.py      trades.jsonl / trades.csv / metrics.jsonl / births.jsonl / run.log
  render.py      human-readable transcripts (placeholder names only)
  report.py      the final report and its verdict
  plots.py       matplotlib figures, with a dependency-free SVG fallback
  run.py         CLI
configs/         open.json, open_full.json, and the earlier fixed-token configs
tests/           41 tests, including the ones that would expose a rigged experiment
```

### Why Gumbel-softmax is the default

Spec 2.3 offers REINFORCE or Gumbel-softmax and asks the implementer to document
the choice. Both are implemented and either can be selected with `--algo`.

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
python -m orchard.run --config configs/open.json --out runs/main
python -m orchard.run --config configs/open.json --out runs/nobottleneck --bottleneck off
python -m orchard.run --config configs/open.json --out runs/noturnover  --turnover off
python -m orchard.run --compare runs/main runs/nobottleneck runs/noturnover
```

Any field is overridable: `--set world.zipf_alpha=0 --set bottleneck.frequency_skew=2`.

### Output

| file | what is in it |
|---|---|
| `trades.jsonl` / `.csv` | every episode: hidden state, the full symbol transcript, its word segmentation, both decisions, outcome, failure classification, rewards and money |
| `metrics.jsonl` | every checkpoint's full metric suite |
| `births.jsonl` | every birth: what the newborn was trained on, which meanings it never saw, how it fared against veterans |
| `run.log` | the complete console history |
| `plots/metrics.svg`, `plots/vocabulary.svg` (+ `.png`) | progress over the run |
| `report.md` | final metrics, the inferred dictionary, example transcripts early/middle/late, economic totals, and a threshold-computed verdict |

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
