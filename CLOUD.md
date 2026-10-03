# Running Orchard

Everything about *what* Orchard is and why it works the way it does is in
**[README.md](README.md)**. This document is how to run it: the commands, what
the output means, and what to do when something looks wrong.

Orchard is meant for a GPU box. Start a run over SSH, close the laptop, come back
to a report. The same code runs on a CPU, slower, and that is how it is tested.

---

## 1. Install and first output

```bash
git clone https://github.com/apariscallahan/AI-language-dev.git && cd AI-language-dev
```

Install the CUDA build of torch that matches the driver first, then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r requirements.txt
```

`scipy` and `matplotlib` are optional — there is a pure-Python Spearman and a
dependency-free SVG plot writer without them.

Three commands, in this order:

```bash
python -m orchard.run --smoke
```

No learning at all: scripted agents, the chance baseline (0.0000), an oracle
pair's upper bound (~0.68), and one rendered episode. If this looks wrong,
nothing above it will work.

```bash
python -m orchard.run --benchmark
```

**This machine's** episodes per second for each rung at full community size, peak
GPU memory, and what the full run will therefore cost in hours. Run it before
committing to a big preset.

```bash
bash cloud_run.sh
```

The run. `cloud_run.sh` checks a GPU is visible, writes to
`runs/<UTC start time>_orchard/`, and auto-resumes from that folder if it is run
again.

**Start a fresh run.** Snapshots written before the lot layout (anything from
September 2026 or earlier that has `ask-qty` or `quote` in its ladder) cannot be
resumed: the observation layout, the query embedding and the atom inventory all
changed, and the loader stops with a message saying so. Snapshots written before
the language faculty (2026-09-30: the innate reader, the speaker's production
lexicon and innate concepts) *do* resume, but without it: the file decides the
architecture, and a brain trained without a reader is resumed without one (the
`[resume]` line says so). To run the faculty, start fresh. The naming rungs took
about 45 minutes on a 4090.

**Start fresh for the 2026-09-30 fixes too.** Until then every agent below the
role split took two optimiser steps per update; `train.lr` was doubled (3e-4 to
6e-4) when that was fixed, so the step is what the naming rungs were validated
at, but a run resumed from an older snapshot changes its dynamics mid-rung. The
config check now also refuses settings that used to fail silently: a
`world.holdout_combo_frac` that reserves nothing, `curriculum.n_candidates`
wider than a field's unreserved values, and unequal farmer and buyer counts.

**And for the 2026-10-01 vocabulary.** A snapshot from before it holds a
vocabulary no later rung can use (several meanings on one word, each founder
its own), speakers that have never described a lot without the scaffold, and
scaffold strengths that were learned then and are constants now. It loads —
the strengths are put back and a `[resume]` line says so — but resumed in
`mutual` or above, the scaffold is going or gone from the first update (its
clock counts the updates the rung has already played) and nothing has taken
its place; the run prints a `[resume] WARNING` saying so. Do not carry the
2026-10-01 run on. `cloud_run.sh` without `RUN=` starts a new folder, which is
what is wanted; the naming rungs are a few hundred updates.

---

## 2. Choosing a scale

One method, four sizes. A preset changes the community, the brain, the batch, the
run length and how much output there is — and nothing else
([README §14](README.md#14-one-method-declared-scale)).

| preset | community | brain | batch | episodes | what it is for |
|---|---|---|---|---|---|
| *(none)* | 2 → 6, then 6 + 6 | d48, 2 layers, 55k | 256 | 6M | the reference scale, and the one a CPU can check |
| `gpu_small` | 2 → 6, then 6 + 6 | d64, 2 layers, 130k | 1,024 | 20M | a cheap GPU run to see a change through the naming rungs |
| `gpu_community` | 2 → 8, then 8 + 8 | d96, 3 layers, 380k | 4,096 | 100M | the main run: a community big enough that a code has to work for strangers, small enough that a 4090 runs it at a useful pace |
| `gpu_large` | 2 → 16, then 16 + 16 | d128, 4 layers, 850k | 4,096 | 120M | the big one; benchmark before committing to it |

```bash
CONFIG=configs/gpu_community.json bash cloud_run.sh
```

```bash
CONFIG=configs/duality.json bash cloud_run.sh
```

`duality` is a declared *experiment*, not a scale: 12 fruits against 8 atoms, so
no single atom can name a whole meaning. Unvalidated — treat it as the
experiment, not the baseline.

The header prints scale and method apart, and the method line should read that
nothing simulated was changed:

```
scale              : pool of 2 growing to 8, then 8 + 8 once trading starts; d=96 x 3 layers, batch 4,096, 100,000,000 episodes
method             : the one configuration -- size aside, nothing simulated was changed
```

Every community is **founded by 2 agents** whatever the preset, and the founders
take all six naming rungs alone; newcomers start arriving at `mutual`, one every
40 updates. A rung that is still filling up does not spend its budget and cannot
pass, so growth never eats a rung's time. The pool stays one pool until
`haggle`, where each agent is copied into a farmer and a buyer.

Why the communities are the size they are: every agent is a separate forward
pass per symbol step, so wall time grows with the number of agents. The 4090 ran
the naming rungs at 2,000–4,000 episodes per second with two founders and
`mutual` at ~600 with eight. Larger communities wait on batching the agents
([README §13](README.md#13-performance-and-engineering)).

---

## 3. What you see while it runs

`cloud_run.sh` keeps the terminal quiet except for:

- **one status line a minute**: time (UTC), episodes done / total, the update
  count, the rung and how much of its maximum it has used, episodes per second,
  ETA, rolling success, community size, births, peak GPU memory;
- **two lines at every checkpoint** (every 100 updates):

```
[checkpoint 8,601,600] rung name-all | success 0.521 (muted 0.335) / 0.523 (muted 0.339) | channel 0.28 of headroom | held-out 0.50 vs trained 0.52 | buyer coverage 0.227 [0.22 0.14 0.32 0.20 0.18]; farmer coverage 0.256 [0.22 0.19 0.36 0.21 0.19]
    coherence farmer 0.460 buyer 0.470 across 0.52 | overlap n/a | 42 words sampled, 31 said, 1.03 atoms/word, 3.47 words/utterance, 0% silent, 0% at buffer end
```

  The two success numbers are the two views of a swap rung (each role decoding),
  each against its own muted baseline. Coverage is per field, in lot order:
  **fruit, colour, quality, quantity, price**. `silent` is always 0% — every
  turn has to open with a word, because silence is what the muted control sounds
  like — and `at buffer end` should be near 0; if it climbs, agents are babbling
  into the cap.

  The two word counts are different questions. **`words sampled`** is every
  form the speakers' policies emitted in evaluation play, so it counts variants
  as well as words — a flawless 12-word code emitted at 98% per-symbol accuracy
  already reads as ~170. **`said`** is the greedy lexicon: what the describers
  actually say when asked. A wide gap between them is a speaker unsure of its
  own words, not a large vocabulary. `overlap` reads `n/a` until the roles
  split, because below that one pool fills both seats and there are not two
  codes to compare; `across` is cross-role coherence with those self-pairs
  excluded, so on a shared pool it is simply the honest between-agent number.

  On a report rung (`mutual`, `order`, `offer`, `judge`) the same line names each
  field each role reports and how often it arrived, which is the number to watch
  there, and the held-out figure is given per field beside the whole round —
  the whole round is a conjunction of every field on both sides and reads 0.00
  long after each field generalises:

```
[checkpoint 22,118,400] rung mutual | success 0.364 (muted 0.000) | channel 0.51 of headroom | held-out 0.03 vs trained 0.32 (per field 0.58 vs 0.80 = 0.61 of the headroom) [fruit 0.89/0.97, colour 0.43/0.70, quality 0.45/0.73] | farmer reads fruit 0.96, colour 0.75, quality 0.76, quantity 0.71, price 0.68; buyer reads fruit 0.97, colour 0.75, quality 0.76, quantity 0.70, price 0.69 | buyer coverage 0.827 [0.97 0.75 0.76 0.72 0.70]
```
- **rung transitions**, with every criterion, passed or not;
- **`[costs]`**, once per costed rung, when the rung reaches its floor and the
  speaker costs start ramping in;
- **budget stops**, naming exactly what was unmet;
- on the second checkpoint line, **`names 4/4, 4/4`** (distinct names over
  meanings named, per speaker) and **`naming signal S, name used Z%`**: how
  far each speaker's own lexicon has settled. On `name-fruit` expect the names
  to reach 4/4 within ~100 updates and `name used` to climb towards 100%; on a
  later naming rung the count grows by that rung's values. Distinct below
  named means two meanings share a name;
- on the second checkpoint line, **`vocabulary 27/27 words, 100% shared`**: what
  the speakers say when asked about one field, word-only and greedy. The first
  number is the speaker with the *fewest* distinct words for the 27 meanings;
  the second is the pair of speakers furthest apart, as the share of meanings
  they name with the same word. Both should climb field by field through the
  naming rungs — 15% shared after `name-fruit` (4 of 27), 100% after
  `name-price` — and `name-all` is not passed below 0.95 and 0.90. The
  2026-10-01 run would have read `8/27 words, 0% shared`. The full checkpoint
  block in `run.log` prints every speaker's word for every meaning. From
  `mutual` on the same numbers are read off the speakers' lexicons rather than
  by asking (nobody is asked about one field any more, and `run.log` says
  which); they should stay at 27/27 and near 100% as newcomers arrive;
- on the second checkpoint line, **`scaffold 100%`**: how much of the
  description scaffold is on. 100% below `mutual`; in `mutual` it is held for
  100 updates, falls to 0% over the next 300, and stays there for the rest of
  the run. `run.log` adds how far the speakers' own policy is from what the
  scaffold asks of them, in nats per symbol: that number should fall towards
  0.01 *before* the scaffold reaches 0, and `words/utterance` should stay at
  5.00 as it does;
- on the second checkpoint line of a trading rung (`haggle` and above),
  **`price 2.50 in 99% of deals, follows the limits 0.00 / 0.00`**: the price
  the farmer names most often, its share of the rounds with a deal to be had,
  and — in the rounds where that price lies outside the two limits — how often
  the farmer and the buyer name one inside them instead. Played greedily. A
  trade needs both to name the same price, and always naming the same one is
  the cheapest way: it agrees every time and passes the rung without anything
  being bargained. That line is what the first local run to reach `haggle`
  read. `follows the limits` climbing towards 1 is a price worked out from the
  floor and the limit the two have just told each other; 0.00 for hundreds of
  updates is a habit;
- on the second checkpoint line, **`gestures possible X%, used Y%`**: the share
  of rounds the world allowed a gesture in, and the share of those turns the
  speakers used one in. Training-time numbers only; everything else on the line
  is word-only. `possible` should fall to 0 over the first 600 updates of each
  naming rung; `used` should be high while the rung's word is forming and fall
  once it works. **While gestures are possible, the status line's *rolling*
  success counts gestured rounds** and runs well above the word-only number
  (`name-quality` read 0.96 rolling against 0.68 word-only on 2026-09-29): read
  the language off the checkpoint and promotion lines;
- from `name-all` on, **`descriptions reuse X% of fields, Y% in usual order`**:
  when a describer is asked for a whole lot, the share of the lot's fields it
  names with its own established word, and the share of named field pairs it
  puts in its usual order. This is the number that showed `name-all` failing on
  2026-09-29 — a speaker's own colour word appeared in 0 of 760 descriptions —
  and it should climb towards 100% before success does. The full checkpoint
  block in `run.log` lists each speaker's words field by field and its usual
  order (`quantity < colour < fruit`, say). With the speaker's production
  lexicon the first word of a description should be one of the lot's words from
  the start; what has to be learned is going on to a second part, so
  **words per utterance** is the number that shows it;
- the **final verdict** and the report path.

### Reading the rungs

| rung | what to look for |
|---|---|
| `name-fruit` | does it leave chance (0.333) at all, and when? This is the one rung that invents a code from nothing. Hindsight and the speaker costs are both off here. If it sits at chance past ~1,500 updates, nothing above it will work. `vocabulary … shared` should reach 15% here (the four fruit words) within about 50 updates: the younger founder takes the elder's word for each fruit as soon as it understands it. If it stays at 0–4%, imitation is not working — read the two speakers' words in `run.log`. The rolling success on the status line and the checkpoint's success should agree roughly — both are fruit rounds between two different agents — so a wide gap means training and measurement are asking different questions. (From `name-color` on, the rolling number also counts the easier rehearsal rounds and runs higher; the checkpoint measures only the new field.) |
| `name-color`, `name-quality`, `name-quantity`, `name-price` | these start from a population that already has words, so they should be *faster* than `name-fruit`. Each also prints a `still names fruit` / `still names colour` / … check: a rung whose own kind climbs while a rehearsed one falls back to chance is forgetting, not learning. Quantity has nine values (0 is "none of that") and price six; a colour round and a quantity round both have three candidates, so chance is 0.333 throughout. `name-quantity` is where the 2026-09-24 run stalled at chance for 850 updates; the gesture channel exists for it. Watch `gestures possible` fall over the rung's first 600 updates and the word-only success rise as it does; if success is still at chance when `possible` reaches 0, the scaffold did not transfer to the words. On the two number rungs the promotion block also prints **`tells neighbouring quantities apart`** / **`prices`**: rounds whose wrong candidates are the nearest values, which need 0.80 from each describer. A rung whose ordinary success is 0.9 with that line at 0.6–0.7 has words for "about four" — how the 2026-10-01 run left `name-quantity` with six words for nine quantities. |
| `name-all` | the hard one: five fields in one utterance. Watch **descriptions reuse** climb first — the describers saying the words they already have — then **words per utterance** toward 5 and **coverage** on every field: the rung needs 0.30 on average *and* 0.25 on each field (`names each field` in the promotion block). A run that sticks at ~2 words with coverage ~0.25 and reuse near zero has grown a second code for whole lots instead of combining its words: that is how the 2026-09-29 run stalled at 0.65 for 1,000 updates. `held-out vs trained` is measured on rounds only fruit, colour and quality can decide (every candidate shares one quantity and price), and the gate reads it **per field and per guesser** (`per field ... [farmer fruit 0.97/0.98, ...]`): how often the guesser's reading of each field prefers the target's value to a candidate's, over the 0.5 any message-blind reading scores. The whole round cannot tell reuse of all three words from reuse of one — a code that carries only the fruit still picks a reserved lot 0.81 of the time — so it is printed but no longer judged. Two more lines in the promotion block: **`one word per meaning`** and **`one dialect`** (see `vocabulary` above) — the vocabulary this rung hands to the community. `coherence` can sit near 0.6 with both at 100%: it compares whole utterances, and two speakers with the same words may still put them in different orders. |
| `mutual` | both report the other's lot, all five fields, with the five belief heads `haggle` will use. Newcomers, deaths and hindsight feedback all switch on here. With one word per meaning and one dialect coming in, success should be well above its 0.08 floor within the first 50 updates (it took the 2026-10-01 run 630 updates to reach 0.035). What the rung is then *for* is the scaffold: it is withdrawn here, so the rung lasts at least 400 updates, and its last bar is **`describes without the scaffold`**. Watch `scaffold` fall to 0% with `words/utterance` staying at 5.00 and success not dropping; if utterances grow towards the buffer as it falls, the speakers have not taken over (README §5). `vocabulary … shared` should stay high as newcomers arrive: they take their elders' words from the descriptions they understand. The speaker costs come on partway through, once the rung reaches its floor (`[costs]` in the log), and are ramped in over 200 updates: `atoms/word` and `words/utterance` should settle without success dropping. The held-out gate here is per field: `held-out 0.58 vs trained 0.80` is the mean per-field accuracy on reserved combinations against trained ones. |
| `order` | the buyer's request is a lot in the naming layout, so the buyer says exactly what it said in `name-all`; what is new is the farmer reporting it while looking at a barn of sixteen rows. Every field is `still carries`; if one falls to chance the farmer is not finding it among the rows. |
| `offer` | the farmer answers about the lot that was asked for — `stock`, `lot-quality`, `reservation` — and the buyer reports that. This is the first rung where a farmer has to **find a lot in its barn** by the words it heard; `stock arrives` is the number to watch, and stock 0 ("none of that") is a value it has to be able to say. The row is found through the farmer's reader and named with its lexicon (`model.lexical_barn`), so the numbers should be said in the naming rungs' words from the first update; what is learned here is *which* parts of the row to say, with no scaffold. Locally the farmer first ran on to the end of the buffer, in the right words, and had learned to stop at five within about 60 updates: `words/utterance` above 5 early in this rung is that, not a fault. |
| `judge` | both decide whether the deal is worth doing. Judged on the gain over silence, not the raw rate: ~68% of rounds are worth doing, so accepting everything scores 0.68 and still fails — which is exactly how `haggle` used to fail. Locally it passed at its first check: the farmer right 0.81 of the time (0.62 muted), the buyer 0.84 (0.52). |
| `haggle` | the pool splits into farmers and buyers (the log says so). Channel transfer well above zero, not just success from base rates. It starts near zero and lifts off late, because a trade needs both to name the same fruit, quantity and price *and* both to accept: locally 0.03 at its first checkpoint and 0.30 at its second (floor 0.15; 0.01 muted). Do not judge it at its first checkpoint. The success on that line is sampled play, exploration included; played greedily the same agents struck 0.81 of the deals that exist. Then read **`price`** on the second line. Locally it said `2.50 in 99% of deals, follows the limits 0.00 / 0.00`: exact price-bin agreement, the bottleneck this row used to predict, was met by both sides always naming the same price, whatever limits they had just told each other. Each side read the other's five words at 0.97–0.99 by then, so what is missing is not the language. |
| `bargain`, `market` | not yet run to a verdict with this faculty. Locally `bargain` opened at the level `haggle` left and read 0.32 fifty updates in, the price still 2.50 every time and the extra turns not yet used for anything (a price word said twice). Agents start dying of old age 900–1,600 updates after `mutual` begins, so the first births fall about here: a newborn has no scaffold and learns from its elders' transcripts alone, which locally gave it their 27 words and 0.88 at `offer` before its first live round (veterans 0.98). A `[birth]` line's `token acc` near 0.95, and `straight out of the bottleneck vs veterans` not far below the veterans' own score, is that working. |
| throughout | the report's **word classes** row (do separate words specialise to separate fields? — the adjective question), cross-role overlap (should stay high: one pool, one language), and the share of utterances at the buffer end (~0). |

**Expect chance for a while.** The first code forms suddenly and late: the runs
that worked sat at chance until ~300–600 updates on the CPU and 525–1,525 on the
GPU, then climbed past 0.4 within about 50. Chance at update 500 is normal;
chance at update 2,000 is not.

Run folders are named for their start time in UTC and the preset:
`runs/2026-09-18_14-03-12UTC_gpu_community`.

---

## 4. Interruptions, snapshots and resuming

A snapshot of the whole community — weights, optimiser state, recent usage, the
transcript store, the curriculum record, the cost ramp — is written to
`<run>/snapshots/latest.pt` at every checkpoint and to `after-<rung>.pt` at every
promotion. Since 2026-09-30 it also carries the run's own records (the history
behind the plots, the metrics rows behind the scorecard, word provenance, the
example archive), every random generator's state, the market's day and stock,
and the reserved combinations, so a resumed run's report covers the whole run
and a pre-empted run does not replay the same rounds. `cloud_run.sh` resumes
automatically when `latest.pt` exists, so on a spot or pre-emptible instance,
rerun it pointing `RUN` at the same folder:

```bash
RUN=runs/2026-09-18_14-03-12UTC_orchard bash cloud_run.sh
```

To see what there is to resume from before choosing -- which rung each snapshot
stopped on, how far in, how big the community was, and whether its two seats are
still one pool:

```bash
python -m orchard.run --snapshots                     # everything under runs/
python -m orchard.run --snapshots runs/<run folder>   # or one run
```

A resumed run picks up whatever code it is started with, so this is also how to
move a running experiment onto newer code: stop it just after a checkpoint,
update, resume. Older snapshots of this layout load too (one with `ask-qty` or
`quote` in its ladder cannot, see §1). What carries over is the population and
its history; the *rules* are whatever the new code and config say, so a resumed
run re-measures the rung it is on under the new measurements and applies
whatever speaker pressures the new schedule turns on. Two things are taken from
the file rather than the command line. The **architecture** (`ARCH_KEYS` in
`config.py`: model width and depth, the atom inventory, the turns, the world's
field sizes) has exactly one valid reading — the one the weights were trained
under — so a forgotten `--config` no longer ends in sixty `size mismatch` lines
that name tensors; the `[resume]` line says what it took from the file. And every
*other* setting that differs from the snapshot's is listed under `[resume]` too,
without being changed, so a resume that shrinks the community or changes the
batch size does so on purpose and never by accident. One thing does not carry
over: the recent-usage record is keyed by meaning, and when the key changed to
carry *what was asked* the old keys stopped matching. They are dropped on load
and the resume line says how many; they rebuild within one update, and the word
counts (keyed by the word) are untouched.

Below `curriculum.split_roles_at` the two seats are **one list**, and a resume
has to put it back that way. Restoring the two saved lists separately made two
copies of every founder — same id, same weights, then their own gradients from
the next update on — so a resumed run below the split trained twice the
population it reported and had two languages where the rung exists to build one.
It surfaced a rung later as an `IndexError` in the rollout: the first newcomer
appended to the farmer list alone, and pairing, which uses the farmer count for
both seats when the pool is shared, handed out a buyer index the buyer list did
not have. Whether the pool is shared is now decided by the rung being resumed
into; a snapshot written by an affected run is detected on load (the two lists
hold the same ids but drifted weights), the farmer copies are kept and the
resume line says so. The transcript store is read to the host whatever device
the run trains on, because a newborn's apprenticeship stacks what it sampled on
the host first — mapping the whole file onto the GPU left the store half on each
side and the first birth after a mid-rung resume died on it.

To run a rung *again* — because a mechanism it depends on has changed — pass
`--resume-at <rung>` with `--resume`. `after-<rung>.pt` is written after the
curriculum has advanced, so resuming it alone restarts the rung *after*;
`--resume-at` puts the curriculum back on the rung named, resets that rung's
clocks (time in the rung, the rolling success the cost ramp keys off, the ramp)
and keeps the weights, the community, the usage counts and the store:

```bash
python -m orchard.run --config configs/gpu_community.json \
    --resume runs/<run>/snapshots/after-mutual.pt --resume-at mutual
```

Branch an experiment off any rung — the header will list what you changed:

```bash
python -m orchard.run --out runs/branch --resume runs/<run>/snapshots/after-name-all.pt --set curriculum.order_min_success=0.3
```

Iterating on one rung with `--resume` and a few hundred updates is the cheapest
way to test a change. Several full runs have been lost to problems a ten-minute
experiment would have caught.

---

## 5. Measuring a saved community

```bash
python -m orchard.analyse --snapshot runs/<run>/snapshots/after-name-all.pt
```

Runs the full metric suite on the rung the snapshot closed and prints the
**language-properties scorecard** (reference, productivity, word classes,
intentionality, decontextualised, displaced, interchangeable, generic,
perspectives, cultural transmission, duality of patterning), each with how it is
measured, its value, and present / partial / absent / untestable. No training
happens, so this runs on a laptop against a snapshot copied down from the box.

To ask one question of a snapshot — which field generalises to combinations
nobody trained on — without the whole suite:

```bash
python -m orchard.run --holdout-report runs/<run>/snapshots/after-mutual.pt
```

It scores the snapshot under the config stored in it and against the reserved
combinations stored in it, on this machine's device, and prints, per field (and
per role where both are measured), held-out against trained
accuracy and how much of the headroom over a message-blind guesser transfers;
then the one-sided conjunction beside what independent fields would predict, and
the whole round. On the run that first promoted out of `mutual` the three
transferred 0.89, 0.41 and 0.41 of their headroom while the whole round read
0.000: independence would have predicted about sixty successes in 2,048, and
there was under one, because a Latin-square holdout asks for exactly the quality
that a correctly-read (fruit, colour) pair never showed in training. A snapshot
from a rung that does not measure held-out combinations is scored on the last
one that does.

---

## 6. Changing settings

**Run settings** — nothing simulated changes:

```bash
bash cloud_run.sh --episodes 2000000 --seed 3 --device cuda:1 --ledger-stride 100
```

**Anything else** is an experiment and is reported as one. Named flags cover the
common ones, and `--set section.key=value` reaches any field:

```bash
bash cloud_run.sh --bottleneck off
```

```bash
bash cloud_run.sh --set world.zipf_alpha=0.0 --set reward.understood=0.6
```

Every run writes the exact configuration it used to `<out>/config.json`, so a run
is always reproducible from its own directory:

```bash
python -m orchard.run --config runs/<run>/config.json --out runs/rerun --seed 9
```

### The settings worth knowing

| setting | what it does |
|---|---|
| `--curriculum on\|off` | the naming-then-trading ladder. Off means the full task from random weights, which has never been made to work. |
| `--on-stall hold\|stop` | what to do when a rung never converges. Keep `stop` (the default): holding just burns money on a rung that is not working, and stopping leaves a report naming the unmet criteria. |
| `--bottleneck on\|off` | the transmission bottleneck. Turning it off is the headline ablation. |
| `--turnover on\|off` | births and deaths. Off means one fixed cohort forever. |
| `--n-farmers`, `--n-buyers` | community size per role (a scale key, like the presets). |
| `population.founders_farmers/_buyers` | how many agents found the community. 0 = start at full size, which does not work above 2 + 2. |
| `population.grow_from_rung` | the rung from which newcomers start arriving (`mutual`). Earlier, every newborn apprentices on a code that is about to be replaced. |
| `population.turnover_from_rung` | the rung agents start dying of old age in (`mutual`, the same one newcomers arrive in). Earlier, a death costs half of a two-agent pool. |
| `reward.costs_from_rung` | a *floor* on where the speaker starts paying for length and for new words (`mutual`). The rung-by-rung rule is `Phase.invents`: costs are off on every rung that still has a word to invent — the six naming rungs — and on from `mutual` up, where every rung only reuses them. Charged while a word is still being invented, the cheapest way to be short is to say the same short nothing. |
| `reward.costs_ramp_trigger`, `reward.costs_ramp_updates` | within each costed rung the costs wait until the rung's own rolling success reaches this multiple of its promotion floor (1.0), then ramp in over this many updates (200); a later rung re-earns them, since it starts a new job near zero. Switching them on at a rung's first update throttled the channel: the hyphen went unused, ~51 messages had to carry 48 meanings, and success was 0.023 against 0.142 with them off. Trigger 0 restores the old step gate. |
| `bottleneck.meaning_holdout` | share of the (fruit, colour, quality) combinations withheld from each newborn's apprenticeship (0.25), drawn fresh per newborn so nothing is lost to the population. This is the half of iterated learning that `coverage` cannot give: a learner shown every meaning memorises the table as faithfully as its parents. 0 restores the old behaviour. |
| `bottleneck.history_share` | the share of the transcript store that the rungs *not* now running keep between them (0.4). At 0 one rung's traffic flushes every earlier rung: a hundred updates of the rung after `mutual` erased all 22,359 `mutual` transcripts and left a newborn nothing to learn the naming language from. |
| `reward.convention_from_rung` | the rung from which the speaker is paid for using the community's word (`name-all`, the first rung that invents no word of its own — it only has to say five that already exist). It cannot punish a new word — a form only counts once it has 12 recent uses — and it cannot collapse the language, because it is contrastive. Waiting for `mutual` left the naming rungs with nothing paying a speaker for repeating itself: two founders with no form in common (coherence 0.15–0.17) and 686 distinct words over sampled play for a 64-thing world. |
| `reward.convention_contrast_samples` | how many other meanings' forms each utterance is contrasted with (16), taking the *closest* rather than the average: a collapsed code earns exactly nothing at 16 and starts earning again below 8, so this is not where to buy speed. It is the whole cost of the term — host-side edit distances while the device waits, 0.75 s per update at batch 4,096 against a ~2.7 s update. |
| `train.hindsight_from_rung` | the first rung with hindsight feedback (`mutual`). Earlier, it stops the first code forming. |
| `gesture.enabled` | the gesture channel (on): a speaker may open a turn with fingers for a quantity or price, or by pointing at a fruit, colour or quality. Iconic, truthful, never a word, never measured — every gate and probe is word-only. Off is the ablation: the 2026-09-24 run, which stalled at chance on `name-quantity` for 850 updates. |
| `gesture.share_start`, `gesture.share_end`, `gesture.anneal_updates` | in what share of rounds a gesture is possible over a rung that is still inventing a word: every round at its start, withdrawn to none over 600 updates, so the words have to take over before the (word-only) gate can pass. |
| `gesture.ostensive_coef` | the ostensive lesson (1.0): where a speaker gestured *and* said its established name, the listener is taught the gestured value from the words alone. Needs the innate lexicon to say which utterances are names. |
| `reward.lexicon_mi` | positive signalling (2.0): information plus separation between the asked-about values' first-symbol distributions, within the asked-about field, in the speaker's own policy. The term that gets four fruit names to form in ~90 updates instead of ~1,700. At 0.5 the names stayed shared; taken across fields instead of within, speakers named the *field* and no value. |
| `reward.lexicon`, `reward.lexicon_min_support`, `reward.lexicon_top_forms` | each speaker's own lexicon (0.30): paid for an utterance closer to this meaning's recent forms than to any other meaning's, per speaker, from the first round. A form is a name after 3 recent uses. Compared against each meaning's top 4 recent forms, not its modal form alone (the modal form could not pull collapsed names apart). |
| `reward.lexicon_exclusive` | one word per meaning (1.0): each speaker's lexicon — its word for every value of every field — is pulled to the nearest table in which no two meanings share a word, summed over the meanings. Silent where the lexicon is already one-to-one. 0 is the 2026-10-01 run: 8 words for 27 meanings. |
| `reward.lexicon_imitate`, `reward.lexicon_imitate_half_life_updates` | one dialect (2.0; 20 updates): a listener that understood an elder remembers the word it heard for that meaning, and its own word moves towards what it remembers. From elder to younger only. 0 leaves every speaker its own dialect. |
| `curriculum.numeral_near_frac`, `curriculum.numeral_min_near` | exact numbers: the share of quantity and price rounds whose wrong candidates are the nearest values (0.5), and what a rung has to score on those alone (0.80). |
| `curriculum.min_vocabulary_distinct`, `curriculum.min_vocabulary_agreement` | what `name-all` has to hand on: the speaker with the fewest words has one for 0.95 of the meanings, and the two furthest apart agree on 0.90 of them. |
| `curriculum.scaffold_fade_rung`, `scaffold_hold_updates`, `scaffold_fade_updates` | where and how fast the description scaffold is withdrawn (`mutual`; held 100 updates, withdrawn over 300). It has to be a rung over bare lots: the market runs without it. |
| `train.scaffold_distil` | while the scaffold is on, the speakers' own policy is trained towards what it asks of them (1.0). 0 withdraws the scaffold from speakers who never practised without it. |
| `train.answer_class_coef` | in a one-field round the listener's reader is taught that the answer's first word names the field asked about (1.0): what keeps a word from being filed under the wrong class for good. |
| `curriculum.own_atoms_from_rung` | the rung from which a speaker's token head may add atoms of its own to its lexicon's word (`order`). Below it a word is the lexicon's alone — otherwise the head keeps a private copy of a speaker's old words and no dialect is ever given up. |
| `model.lexical_barn`, `model.heard_meaning` | the faculty in the market (both true): a farmer finds the lot it was asked about through its reader and names that row's parts with its lexicon; and what a listener understood of each word is added to its own state, not only to its report heads. |
| `gesture.share_reuse` | the standing share from `mutual` on (0.1). Whether speakers still use it once words work is what §3g of the report shows. |
| `gesture.cost` | what the speaker pays per gesture (0.02): small enough to be worth it while the word fails, enough to drop once the word works. |
| `gesture.supervise_coef` | the listener's head for the gestured field is taught the gestured value (0.5). Not hindsight: the answer is in the message, so it cannot teach the listener to ignore the message. |
| `model.factored_choice` | the innate word classes (true): the lineup choice is read through the five belief heads, attribute by attribute. False restores the plain candidate pointer. |
| `model.barn_lookup` | one cross-attention step from the farmer's hidden state to its barn rows, keyed on (fruit, colour), valued on (quality, stock); only active on a barn (true). Off, the plain transformer never learned to find the asked-for lot even supervised. |
| `curriculum.split_roles_at` | the rung where the one pool becomes farmers and buyers (`haggle`). Everything below it is one language in two seats, the report rungs included -- they run in both directions. |
| `curriculum.hard_distractor_frac` | share of open lineup rounds built as one-field near misses (0.9), the field drawn uniformly, so every field has to be named. |
| `world.holdout_combo_frac` | share of (fruit, colour, quality) combinations reserved and never trained on (0.25, a Latin square). |
| `curriculum.min_holdout_ratio` | how well a rung must do on those, as a share of how well it does on trained ones over the headroom a message-blind reader leaves (0.60): per field and per role everywhere, and the weaker role is judged. The productivity gate. |
| `curriculum.min_field_transfer`, `curriculum.min_field_coverage` | every field is checked for every role; coverage is what catches a code that names one field in every slot. |
| `curriculum.mutual_min_report`, `curriculum.order_min_success` | the floor for reporting the other's whole lot exactly (0.25 for five fields) and for the fields a report rung introduced arriving together (0.25). |
| `bottleneck.meaning_holdout` | the share of the (fruit, colour, quality) combinations a newborn is not shown at all (0.25): the bottleneck proper. 0 makes a newborn a near-clone. |
| `bottleneck.coverage` | how much of the rest of the parent generation a newborn sees (1.0 — essentially all of it; lowering it puts *common* forms back at risk). |
| `bottleneck.frequency_skew` | how strongly a newborn's lessons favour common trades. |
| `reward.belief_qty_tol` | how exactly a reported quantity has to match in the trading rungs (1). Report rungs are exact. |
| `reward.symbol_cost`, `reward.atom_cost`, `reward.word_cost` | per symbol (0.005), per atom after the first in a word (0.03), per word (0.005): short words, not short sentences, and no repeating a word to the buffer end. The symbol charge is kept small because it is neutral between `a-b` and `a b` and only dilutes the fused-versus-split ratio the other two exist for. |
| `reward.refer_partial` | what a lineup guess is paid per field it shares with the target (0.45). The staircase from naming one field to naming all five; 0 restores all-or-nothing. |
| `train.anneal_per_rung` | whether the temperature and entropy anneals count updates in the current rung rather than since the run started (true). Off means no exploration past update 1,000. |
| `channel.atomic_vocab` | how many meaningless atoms words are built from (32: enough for every one of the 27 field values to have an atom, so whether atoms are reused or combined is up to the agents). |
| `channel.max_symbols`, `channel.n_turns` | the per-turn buffer (24 — a buffer, not a pressure; a five-word request is nine symbols) and the number of turns (4). |
| `channel.enforce_word_grammar` | atoms and marks alternate, so `a3-a7 a1` is a two-atom word and a one-atom word, exactly as emitted. |
| `channel.allow_silence` | whether a turn may be empty (false). Silence is what the muted control sounds like, so a speaker may not say it. |
| `world.zipf_alpha` | how skewed demand is. **Read §9 before raising it.** |
| `reward.decode`, `reward.understood` | the two halves of the communication loop. |
| `--episodes` | run length (a run setting). Generations fall out of it — see §7. |

---

## 7. Generations are derived, not set

An agent ages by the training updates it takes part in — nearly every update —
and dies at its lifespan (900–1,600 updates), so turnover falls out of run length
and lifespan together:

```
generations  ~  (episodes / batch_size) / mean_lifespan_in_updates
```

The header and `--benchmark` print the resulting number. Lifespans want to be
long enough that an agent can learn the language before it dies — the first code
takes ~300–600 updates to form — and short enough that the population turns over
often.

Everything that means an amount of learning is counted in **training updates**,
never episodes: rung budgets, promotion checks (every 25), checkpoints (every
100), the temperature and entropy anneals (1,000 and 800), the cost ramp (200),
growth (every 40), lifespans, and how long the population remembers what it has
said (80). An early GPU run counted lifespans in episodes and every founder died
after ~50 updates.

---

## 8. Never conclude anything from one run

This simulation is bimodal. A population either finds a referential convention or
it does not. Four neighbouring conditions at 40k episodes gave **76%, 0%, 92% and
6%** of the channel headroom — a spread far larger than any effect worth
measuring. One seed per arm is a coin flip with a table around it.

```bash
python sweep.py --out runs/ablation --seeds 5 --arm "bottleneck_on:" --arm "bottleneck_off:--bottleneck off"
```

Reports mean, spread **and every individual seed**, so bimodality shows up
instead of being averaged into a number that means nothing. Use `--parallel 1` on
a single GPU.

Comparing two finished runs directly:

```bash
python compare_runs.py runs/a runs/b
```

It reads each run's own `metrics.jsonl`, `token_semantics.json` and ledger, so it
reports what the run recorded rather than what a report was written to say.
`python -m orchard.run --compare runs/a runs/b` overlays them on one set of plots.

---

## 9. Two traps

**Skewing demand suppresses language.** `world.zipf_alpha` exists because the
length/frequency prediction needs some meanings to be commoner than others. But
skew also makes "guess the common case" pay, and that is a local optimum agents
do not leave. Applying the skew to *which fruit is wanted* took the channel from
64% of headroom to **0%**. It is therefore split: `zipf_alpha` (0.3) applies to
quantity, and `zipf_alpha_variety` defaults to 0. Raising the latter reproduces
the failure.

**Raising the viable-deal rate can also suppress it.** More viable rounds means
more practice closing deals, but also a stronger "just accept" attractor. Between
55% and 71% viable the results were non-monotonic and dominated by seed noise. If
you change `world.p_stocked` or `world.need_max_frac`, re-measure with a sweep
rather than a single run.

---

## 10. Memory

At the reference scale a run needs well under 1 GB of GPU memory. Parameters are
never the limit: the straight-through Gumbel channel builds one autograd graph
spanning every symbol step of an episode, so activation memory grows as

```
batch  x  sequence length  x  d_model  x  layers  x  (symbols per turn x turns)
```

If an experiment makes that too big:

```bash
bash cloud_run.sh --set train.grad_checkpoint=true
```

Recomputes activations in the backward pass instead of keeping them. Memory only
— `tests/test_config.py` checks the update is identical — so it is a run setting.
It is what lets a 4,096 batch fit on a 24 GB card, and the GPU presets set it
already.

---

## 11. Output, and what to bring home

| file | keep it? |
|---|---|
| `report.md` | **yes** — rewritten every checkpoint, so it is readable mid-run |
| `metrics.jsonl` | **yes** — every checkpoint's full metric suite, small |
| `promotions.jsonl` | **yes** — every promotion check, passed or not, with its evidence |
| `births.jsonl` | **yes** — what each newborn was taught, what it was not shown, and how it fared |
| `config.json` | **yes** — exactly reproduces the run |
| `plots/*.svg`, `*.png` | yes, small |
| `transcripts.txt` | yes — sampled rounds as expected / dialogue / outcome |
| `run.log` | probably — the full checkpoint blocks |
| `trades.jsonl`, `trades.csv`, `lineups.jsonl` | the big ones — see below |

The trade ledger subsamples episodes (`log.ledger_stride`, 500) and still keeps
thousands of fully reconstructable trades. Set `--ledger-stride 1` only if you
genuinely want all of them, and check your disk first.

```bash
tar czf results.tgz runs/<run> --exclude='trades.*' --exclude='lineups.jsonl'
```

Reports and plots are written at every checkpoint, so a run killed early still
leaves a readable `report.md`, full metrics and a ledger. Interrupting is safe.

---

## 12. Keeping a run alive when you disconnect

A run started straight from a terminal is a child of that terminal's shell, so
it dies when the terminal goes away -- closing an SSH session, or a browser
terminal (the RunPod web terminal, Jupyter's) disconnecting because the laptop
slept. Start it inside `tmux`, which lives on the machine, not in your browser:

```bash
tmux new -s orchard
```

Then, inside it, start or resume the run as usual:

```bash
CONFIG=configs/gpu_community.json bash cloud_run.sh
```

Detach with **Ctrl+B, then D** -- the run keeps going -- and close the tab or
put the laptop to sleep. Come back to it from any new terminal:

```bash
tmux attach -t orchard
```

If `tmux` is missing (`command not found`):

```bash
apt-get update && apt-get install -y tmux
```

Without `tmux`, `nohup` survives a disconnect too, with the output in a file
rather than on screen:

```bash
CONFIG=configs/gpu_community.json nohup bash cloud_run.sh > run.out 2>&1 &
```

```bash
tail -f run.out
```

Either way, only the *terminal* is safe to lose. Stopping the pod stops the run;
the run folder is on the pod's volume, so it resumes from its last snapshot (§4)
once the pod is back.

Which rung is it on:

```bash
grep -E "rung|PHASE" runs/<run>/run.log | tail -20
```

---

## 13. Before a long run

```bash
python -m unittest discover -s tests
```

396 tests, about ten minutes on a CPU. Worth doing on the GPU box, not just locally:
`tests/test_batched.py` asserts the fast tensor path agrees **exactly** with the
readable scalar one, `tests/test_config.py` that there is one configuration and
no device-specific arithmetic, and `tests/test_lots.py` that every report rung's
checkpoint yields the evidence its gate reads.

---

## 14. If something looks wrong

| symptom | first thing to check |
|---|---|
| `$'\r': command not found` from `cloud_run.sh` | the file was checked out with CRLF. `.gitattributes` pins `*.sh` to LF; re-clone or `dos2unix cloud_run.sh`. |
| "No CUDA device visible" | `cloud_run.sh` only runs on a GPU. Use `python -m orchard.run` for a CPU run. |
| "was written by a version with a different observation layout" on resume | a snapshot from before the lot layout. Start a fresh run; nothing in it can be carried over. |
| success at chance past ~1,500 updates in `name-fruit` | a real failure, not slowness. Check the header: speaker costs and hindsight should be off, the pool should be 2 + 2. |
| `silent` above 0% | it cannot be: every turn has to open with a word (`channel.allow_silence`). Check the header's method line for a change to it. |
| ~1 word per utterance on a rung that is still inventing words | the speaker costs came on too early — check `reward.costs_from_rung` in the header. |
| a big "words sampled" count next to a small "said" count | not a large vocabulary: the first is over sampled play and counts every variant the policy emits, the second is the greedy lexicon. A wide gap is a speaker unsure of its own words — check that the convention bonus is on (the header's speaker-pressures line). |
| `name-all` flat at ~0.8 with everything else passing | it was the structure bars, twice over: the probes followed the rung's 70/30 mixture of questions while the metrics ignore which was asked, and field coverage was a whole-message statistic that sits at its ceiling once every lot has its own message — a perfect describer scored 0.33-0.49 on the three-field world and 0.00 at 100 probes on the five-field one, against a 0.30 bar. Both are fixed (the probes ask the rung's own kind; coverage is read off the pieces of a message, cross-validated); `tests/test_rungs.py::TestAPerfectSpeakerPasses` holds them. |
| `coherence across` or `cross-role overlap` looking healthy in a naming rung | below `curriculum.split_roles_at` one pool fills both seats, so those compare agents with themselves. Cross-role coherence now skips self-pairs and overlap reads `n/a`; the number to read is the per-role coherence. |
| `name-all` waiting on `one word per meaning` or `one dialect` | read the `vocabulary` block in `run.log`: it lists every speaker's word for every meaning. Two meanings on one word in one speaker is exclusivity not holding (`reward.lexicon_exclusive`); two speakers with different words for a meaning is imitation not holding (`reward.lexicon_imitate`) — check the header's `one vocabulary` line. |
| a number rung waiting on `tells neighbouring … apart` | the number words are approximate. Not a reason to lower `numeral_min_near`: `mutual` needs the number itself. |
| `mutual` passes everything but `describes without the scaffold` | by design, for its first 400 updates: the scaffold is withdrawn there and the rung cannot be left until it is gone. |
| `price … follows the limits 0.00` for hundreds of updates of `haggle` or `bargain` | the price is a habit: one price whatever the limits, which agrees every time and fits most deals (0.83 locally). Nothing is broken and nothing is bargained. First see whether the GPU's batch moves it by itself — the rounds where the habit fails are about a ninth of all rounds, and each side is paid for naming a price inside the limits there (`reward.correct_per_dim`). If it does not: the game pays too well for the habit. The remedies are in the world, not the channel — draw rounds so that no one price fits most deals, the way the nearest-neighbour rounds made number words exact — and have not been built. |
| `words/utterance` climbing past 5 as `scaffold` falls | the speakers' own policy has not taken over. `run.log` prints how far it is from what the scaffold asks (nats per symbol); if that is not falling while the scaffold is at 100%, check `train.scaffold_distil` in the header. |
| a rehearsed kind falling to chance | forgetting. The mixture weights (`Phase.mix` in `curriculum.py`) are the dial. |
| a report rung passes every field but "reports combinations it never trained on" | the listener's heads have learned the training set's joint. The gate is per field and at 0.60 of the headroom; see README §11 before touching it. |
| `stock arrives` stuck at chance in `offer` while the request fields all carry | the farmer is not finding the asked-for lot among its barn rows. Check `model.barn_lookup` is on in the header; with it the lookup learns in a few hundred supervised steps, so what is missing is the reinforcement signal — look at whether hindsight is on and the buyer's `stock` report is being scored. |
| a rung stops the run | read the criteria it names in the log and in `promotions.jsonl`. Do not relax them to make it pass — they are the experiment. |
| high field coverage but held-out near zero | a memorised code, not a compositional one -- the two numbers exist to be read together. Check that the length costs are on for that rung (`Phase.invents`): with no lineup and no cost, a lookup table solves the rung and fails the productivity gate. |
| the hyphen unused (`1.00 atoms/word`) and the lexicon static | the length costs came on before the messages had grown. Check the run log for the `[costs]` line -- if it fired early, raise `reward.costs_ramp_trigger`. |
| `community N+M` with N != M below `curriculum.split_roles_at` | the pool should be one list, so the two numbers cannot differ. An older resume broke the aliasing; the next birth then crashes the rollout with `IndexError`. Resume on current code, which restores it and says whether the snapshot came from an affected run. |
| out of memory in the first batch | `--set train.grad_checkpoint=true`, then a smaller batch. See §10. |
| every seed disagrees | expected. See §8. |
