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

### On a cloud GPU

**[CLOUD.md](CLOUD.md) is the guide for this** — presets, how to change any
setting from the CLI, memory sizing, and the two traps that will waste money.

```bash
bash cloud_run.sh
```

Checks a CUDA device is actually visible, then runs `configs/gpu.json` — two
million episodes, batch 2048, sixteen agents — writing to a timestamped run
directory. Everything after the script name is passed through, so
`bash cloud_run.sh --bottleneck off --episodes 500000` works.

`--device auto` is the default everywhere: CUDA when a GPU is visible, CPU
otherwise, so the same command works on a laptop and on a cloud box. Asking for
`--device cuda` on a CPU-only torch build fails immediately with the reason
rather than silently running on the CPU for six hours.

**What makes it worth a GPU.** The simulation used to do one Python call and one
dataclass construction *per episode* to sample the world and score the trade.
That is invisible on a CPU, where the transformer forwards dominate, and fatal on
a GPU, where they do not. Measured on a 4096-episode batch:

| host-side work per batch | before | after |
|---|---|---|
| world sampling | 105 ms | 5.3 ms |
| trade scoring | 502 ms | 5.2 ms |
| **total** | **607 ms** | **11 ms** |

That was a hard ceiling of about 6,700 episodes/sec no matter how fast the GPU
was; it is now around 390,000, so the GPU is the limit instead of the
interpreter. `orchard/batched.py` holds the tensor world and tensor reward, and
`tests/test_batched.py` asserts they agree with the scalar versions *exactly* —
same rewards, same success flags — on random batches. The scalar code in
`world.py` and `env.py` remains the readable definition of the rules; if the two
ever disagree, it is right and the tensor one is wrong.

Two smaller things also mattered: the rollout no longer reads a tensor on the
host every symbol step to check whether anyone is still talking (that
synchronises the device), and pairings are now fixed strides rather than random
draws, which makes each agent's slice of the batch a constant instead of
something to be derived from a tensor mid-loop. Because scenarios are i.i.d. the
two pairings are the same marketplace, but the fixed one also gives every agent
exactly its share of the batch instead of a multinomial count, which lowers
gradient variance.

Other GPU switches: `--amp on` for bfloat16 autocast (bf16 rather than fp16, so
there is no loss scaling to get wrong), `--compile on` for `torch.compile`, and
TF32 matmuls on by default.

### Comparing anything: use several seeds

```bash
python sweep.py --config configs/gpu.json --out runs/ablation --seeds 5 \
    --arm "bottleneck_on:" --arm "bottleneck_off:--bottleneck off"
```

**Do not draw conclusions from single runs of this simulation.** It is bimodal: a
population either finds a referential convention or it does not. Four
neighbouring conditions at 40k episodes produced 76%, 0%, 92% and 6% of the
channel headroom — a spread that swamps any effect worth measuring. `sweep.py`
runs each arm across seeds and reports mean, spread and the per-seed values, so
the bimodality is visible rather than averaged into a misleading single number.
Use `--parallel 1` on a single GPU and a higher number on a CPU box with cores to
spare.

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

Every emitted symbol — atoms, hyphens and spaces alike — is charged for. Ending a
message is free, because brevity should not be taxed.

That single cost is also the whole Zipf mechanism. Requests follow a Zipf-like
frequency distribution, so a meaning that comes up constantly pays its length cost
constantly, while a rare one barely pays it at all. Nothing rewards "short words
for common things" directly; it is a prediction, and `report.md` reports the
correlation rather than eyeballing it.

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

| rung | what is added | chance rate |
|---|---|---|
| `refer` | a lineup game: the farmer describes one meaning, the buyer picks it out of K candidates. No price, no budget, no negotiation, no market. | 1/K |
| `refer-swap` | the same game with the roles alternating batch by batch, so every agent must both describe and decode | 1/K |
| `refer-mutual` | both hold a private (variety, quantity, quality) tuple and each must report the other's; still no price, no accept/reject | measured (muted channel) |
| `order` | the buyer asks; the farmer must fill the order exactly with its *deal* decision — the first rung that uses the deal heads | measured (muted channel) |
| `haggle` | price and budget, so accept/reject has a payoff — still one message each | ~0 |
| `bargain` | several turns, so counter-offers become possible | ~0 |
| `market` | the full economy: persistent stock, restocking, viability | ~0 |

Each rung adds one thing. `refer-swap` exists because a single fixed describer
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

In `refer-swap` and `refer-mutual` every one of these is checked **per role**,
never pooled: each role's own utterances must show topsim over null *and*
positional structure (`curriculum.min_positional_structure`), and each role must
decode in the view where it is the one decoding (`refer-swap`) or report the
other's tuple (`refer-mutual`, `curriculum.mutual_min_report`). A pooled
average would let a fluent partner carry a role that never learned to speak.

Success alone is not enough, because a pair can score on base rates without
saying anything. Every check, passed or not, is written to `promotions.jsonl`.

**Every rung has a budget** (`curriculum.rung_budgets`, min and max episodes).
Promotion is checked every `curriculum.check_every` episodes with a light probe,
so a rung that works is left promptly. A rung that reaches its maximum without
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
| `reward.rarity_cost` (0.05) | per word, scaled by how rare the form is in the population's recent usage (`usage_half_life`), centred on the batch so it favours established forms without ever favouring silence |
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
first rung: a newcomer of each role joins every `population.grow_every`
episodes, born like any newborn -- random weights, then the transmission
bottleneck on the community's transcripts -- so it learns the existing language
instead of inventing another. Every rung after the first waits for, and is
judged on, the full community.

The report measures what they are for: distinct words, atoms per word, share of
utterances at the length cap, coherence within each role and across roles, and
**cross-role vocabulary overlap**: the histogram intersection of the farmer's and
the buyer's word use (1.0 = one shared vocabulary, 0.0 = two foreign codes).

## Generations, and what gets lost

Agents age, die at a randomised lifespan, and are replaced by newborns with fresh
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
| `progress.json` | rewritten every batch; what the GUI's progress bar reads |

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
