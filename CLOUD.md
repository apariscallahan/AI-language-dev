# Running Orchard on a cloud GPU

Orchard runs on a GPU. Start a run over SSH, close the laptop, and come back to
a report.

---

## 1. Ninety seconds to first output

```bash
git clone <your remote> orchard && cd orchard
pip install torch numpy scipy matplotlib        # matplotlib is optional; see §8

python -m orchard.run --smoke                   # no GPU, no learning: checks the world
python -m orchard.run --config configs/gpu_smoke.json --benchmark
```

`--benchmark` is the important one. It reports **this machine's** episodes per
second for **these exact settings**, what the configured run will therefore cost
in hours, and peak GPU memory. Run it before anything long. Every timing in this
document is a shape, not a promise: the numbers were taken on a 4-core CPU, and
your box is not that.

```bash
bash cloud_run.sh                               # the full pipeline, timestamped output
```

---

## 2. The presets

**The method is the code defaults in `orchard/config.py`; a preset only changes
scale.** Every preset runs the seven-rung ladder (`refer`, `refer-swap`,
`refer-mutual`, `order`, `haggle`, `bargain`, `market`), per-role promotion, hard
lineup rounds, held-out combinations, hindsight feedback, the word grammar,
speaker pressures, and a community **founded by 2 farmers and 2 buyers that
grows** to full size once the first rung is passed. A preset may set only the
keys in `PRESET_KEYS` (community size, brain size, batch, run length, hardware
switches, output) and `tests/test_config.py` fails otherwise. `gpu_community.json`
is exactly the defaults. The run header's `method` line, and the same row in the
report's summary statistics, say "the code defaults" or list every setting that
differs -- `gpu_smoke` (short lives, a plumbing check) and `gpu_duality` (a
different world, the experiment) are the only presets that list any.

| preset | community | brain | params / agent | episodes | batch | ~updates | ~generations |
|---|---|---|---|---|---|---|---|
| `gpu_smoke` | 2+2 -> 8+8 | d=64, 2 layers | 87k | 0.5M | 1,024 | 490 | 3 |
| `gpu_small` | 2+2 -> 16+16 | d=64, 2 layers | 87k | 8M | 2,048 | 3,900 | 3 |
| `gpu_community` | 2+2 -> 48+48 | d=96, 3 layers | 294k | 60M | 4,096 | 14,600 | 12 |
| `gpu_full` | 2+2 -> 128+128 | d=96, 3 layers | 294k | 80M | 4,096 | 19,500 | 16 |
| `gpu_duality` | 2+2 -> 48+48 | d=96, 3 layers | 301k | 40M | 2,048 | 19,500 | 16 |

(Updates at the base batch; the lineup rungs run at twice the batch, so they
take half as many. A run usually ends earlier, at a rung's budget.)

**Memory (24 GB card).** Training backpropagates through every symbol step,
and each step re-encodes the conversation so far, so activation memory is the
constraint -- not parameters. All GPU presets use gradient checkpointing, which
keeps only the tokens fed into each step. Measured memory held for the backward
pass, per 1,000 episodes: lineup rung ~0.2 GB, mutual ~0.7 GB, full market
~2.4 GB (duality world: 0.7 / 1.7 / 5.0 GB). Batch sizes are set so the heaviest
rung stays under ~10 GB on a 24 GB card; on a 40-80 GB card they can be doubled
(`--batch-size`). Without checkpointing the same batch needed 56-365 GB.

**Everything that means an amount of learning is counted in training updates**
(one update = one batch): rung budgets (`curriculum.rung_budget_updates`),
promotion checks (`check_every_updates`, 25), checkpoints
(`log.checkpoint_every_updates`, 100), the temperature and entropy anneals
(`train.tau_anneal_updates` 1,000, `entropy_anneal_updates` 800), growth
(`population.grow_every_updates`, 20), lifespans (900-1,600 updates), and how
long the population remembers what it has been saying
(`reward.usage_half_life_updates`, 80). An episode count is a different amount
of learning at every batch size. The first GPU run counted lifespans in episodes:
a 4,096-episode batch shared by the 2 + 2 founders aged each founder 2,048
episodes per update, 16x the CPU runs, so founders lived ~50 updates and the
lineup never left chance (the CPU runs needed ~550 updates). The usage memory
had the same problem: 20,000 episodes was ~80 updates on the CPU but ~5 on the
GPU.

Brains are deliberately small and communities large. A supervised check showed
the 48k-parameter CPU brain already learns a full compositional code for every
meaning (100% on combinations it never saw), so capacity is not what limits
these runs; agent count is what makes "a community" mean something.

```bash
CONFIG=configs/gpu_community.json bash cloud_run.sh
```

**Why founded small.** Six farmers and six buyers starting from random weights
never got the lineup game off chance in 200k episodes: each farmer kept its own
drifting code (coherence 0.04-0.09), so no buyer could learn to read any of them.
Two and two invent a code in ~80-140k episodes. So every preset founds the
community at 2 + 2, and after the first rung a newcomer of each role joins every
20 updates (`population.grow_every_updates`) -- random weights, then the
transmission bottleneck on the community's transcripts -- until it reaches full
size. Every rung after the first waits for, and is judged on, the full
community, and its budget only starts counting once the community is full
(48 + 48 takes ~900 updates to grow, 128 + 128 ~2,500).

**`gpu_smoke`** -- minutes. Proves the box works end to end and writes every
artefact. It will usually stop at the first rung's budget: that is the machinery
working, not a result.

**`gpu_small`** -- the smallest preset worth reading. Use it to check a change
before paying for `gpu_community`.

**`gpu_community`** -- the default. 96 agents (48 + 48) is where "a community"
means something rather than a handful of co-adapted pairs.

**`gpu_full`** -- 256 agents (128 + 128). Memory is modest; the cost is
the per-agent loop (each agent runs its own forward pass per symbol step), so
wall time grows with agent count. Benchmark first.

**`gpu_duality`** -- the community preset in a world with **more things to name
than atoms to name them with**: 12 apple varieties (plus 8 quantities and 3
qualities) against 8 atoms, with a 32-symbol buffer per utterance. In the default world 16
atoms cover 14 field values, so every value can simply get its own atom and
nothing pushes towards *duality of patterning* -- meaningless units combining
into meaningful words. This preset makes that pressure real. It is harder and
has not been validated on a CPU; treat it as the experiment, not the baseline.

**Anneal schedules are in updates, not run fractions.** An earlier version
annealed over a fraction of the whole run, so giving a run more episodes silently
slowed its learning -- a 2.4M-episode run was still at temperature 1.36 after
200k episodes and never left chance.

## 2a. What you see while it runs

`cloud_run.sh` keeps the terminal quiet except for:

- one **status line a minute**: time (UTC), episodes done / total, the update
  count, rung and how many of its maximum updates it has used, episodes per
  second, ETA, rolling success, community size, births, peak GPU memory;
- a **two-line headline at every checkpoint**: success against the muted
  channel, share of headroom the channel carries, each role's field coverage
  (variety / quantity / quality), coherence, cross-role overlap, word counts;
- **rung transitions** and **budget stops** with every criterion;
- the **final verdict** and the report path.

Everything else goes to the run folder:

| file | what it is |
|---|---|
| `run.log` | the complete console history, including the long checkpoint blocks |
| `transcripts.txt` | every `log.transcript_stride`-th round as *expected / dialogue / outcome* lines, with a banner at each rung |
| `report.md` | rewritten at every checkpoint; **summary statistics** at the top |
| `promotions.jsonl` | every promotion check, passed or not, with its evidence |
| `progress.json` | a one-line status, for scripts |

Run folders are named for their start time in UTC and the preset:
`runs/2026-09-18_14-03-12UTC_gpu_community`.

## 2b. Interruptions, snapshots and resuming

A snapshot of the whole community -- weights, optimiser state, recent usage, the
transcript store, the curriculum record -- is written to
`<run>/snapshots/latest.pt` at every checkpoint and to `after-<rung>.pt` at every
promotion. `cloud_run.sh` resumes automatically when `latest.pt` exists, so on a
spot or pre-emptible instance, rerun it pointing `RUN` at the run's folder:

```bash
CONFIG=configs/gpu_community.json bash cloud_run.sh        # starts runs/<UTC time>_gpu_community
RUN=runs/2026-09-18_14-03-12UTC_gpu_community CONFIG=configs/gpu_community.json bash cloud_run.sh   # resumes it
```

A resumed run picks up whatever code and config it is started with, so this is
also how to move a running experiment onto newer code: stop it just after a
checkpoint (the snapshot is written then), update, and resume.

Resume by hand, or branch a new experiment off any rung, under any config:

```bash
python -m orchard.run --config configs/gpu_community.json --out runs/branch \
    --resume runs/community/snapshots/after-refer-mutual.pt \
    --set curriculum.order_min_success=0.6
```

## 2c. Measuring a saved community

```bash
python -m orchard.analyse --snapshot runs/community/snapshots/after-refer-swap.pt
```

Runs the full metric suite on the rung the snapshot closed and prints the
**language-properties scorecard** (reference, productivity, intentionality,
decontextualised, displaced, interchangeable, generic, perspectives, cultural
transmission, duality of patterning), each with how it is measured, the value,
and present / partial / absent / not testable. No training happens; it runs
fine on a laptop CPU against a snapshot copied down from the box.

## 3. Changing settings from the command line

Three ways, in increasing order of bluntness.

**Named flags** for the things you change most:

```bash
python -m orchard.run --config configs/gpu_community.json --out runs/x \
    --episodes 2000000 --batch-size 8192 --n-farmers 32 --n-buyers 32 \
    --seed 3 --bottleneck off --turnover on --device cuda:1
```

**`--set section.key=value`** reaches any field in the config at all:

```bash
--set model.d_model=256 --set model.n_layers=8 --set model.d_ff=1024 \
--set world.zipf_alpha=0.0 --set reward.understood=0.6 \
--set train.grad_checkpoint=true --set log.ledger_stride=500
```

**Edit a config file.** The presets are plain JSON holding only scale; copy one
and change it. Anything you set beyond scale is printed as a method change in
the run header and the report. Every run also writes the exact config it used to
`<out>/config.json`, so a run is always reproducible from its own directory:

```bash
python -m orchard.run --config runs/community/config.json --out runs/community_rerun --seed 9
```

### The settings worth knowing

| setting | what it does |
|---|---|
| `--curriculum on\|off` | the referential-then-trading ladder. Off means the full task from random weights, which has not been made to work. |
| `population.founders_farmers/_buyers`, `population.grow_every_updates` | found the community small and grow it after the first rung. 0 founders = start at full size. |
| `curriculum.hard_distractor_frac` | share of lineup rounds built as one-field near misses, so every field (quantity included) has to be named. |
| `curriculum.holdout_tuple_frac` | share of (variety, quantity, quality) combinations never trained on: the productivity test. |
| `curriculum.min_field_transfer`, `curriculum.mutual_qty_tol` | the mutual rung checks every field for every role, quantity exactly. |
| `--on-stall hold\|stop` | what to do if a phase never converges. |
| `bottleneck.coverage` | how much of the parent generation a newborn sees. 1.0 means essentially all of it; lowering it puts common forms back at risk. |
| `--n-farmers`, `--n-buyers` | population size per role. More agents means a code that has to work for strangers, not a private pair. |
| `--episodes` | run length. Generations fall out of this — see §4. |
| `--batch-size` | episodes per optimiser step. The main speed/memory dial. |
| `--bottleneck on\|off` | the transmission bottleneck. Turning it off is the headline ablation. |
| `--turnover on\|off` | births and deaths. Off means one fixed cohort forever. |
| `model.d_model`, `model.n_layers`, `model.d_ff` | brain size. |
| `channel.atomic_vocab` | how many meaningless atoms words are built from. |
| `channel.max_symbols`, `channel.n_turns` | the per-turn buffer (24 by default: a buffer, not a pressure -- the symbol cost sets length) and the number of turns. Both multiply memory. |
| `channel.enforce_word_grammar` | atoms and marks alternate: `a3-a7 a1` is a two-atom word and a one-atom word, exactly as emitted. |
| `world.zipf_alpha` | how skewed demand is. **Read §7 before raising it.** |
| `reward.decode`, `reward.understood` | the two halves of the communication loop. |
| `bottleneck.frequency_skew` | how strongly a newborn's lessons favour common trades. |

---

## 3b. The curriculum

Runs start on a lineup game and work up to the full market. Nothing is
reinitialised between phases; the same population carries its weights forward.

The rungs are `refer`, `refer-swap`, `refer-mutual`, `order`, `haggle`,
`bargain`, `market`. Each has its own (min, max) budget in training updates --
80 to 2,500 for the lineup rungs and `order`, 80 to 3,500 for `refer-mutual`,
`haggle` and `bargain`, open for `market`:

```bash
--curriculum off                          # straight to the full trading task
--set curriculum.n_candidates=6           # a harder lineup (chance 1/6)
--set 'curriculum.rung_budget_updates={"refer":[80,4000],"refer-swap":[80,2500],"refer-mutual":[80,3500],"order":[80,2500],"haggle":[80,3500],"bargain":[80,3500],"market":[80,1000000000]}'
--set curriculum.check_every_updates=25   # how often promotion is probed
--on-stall stop                           # end the run on a blown budget (the default)
```

(`--set` replaces the whole dict, so name every rung; ones left out fall back to
`curriculum.default_rung_updates`.)

Promotion needs success clear of chance, topsim clear of its shuffled null, *and*
the muted-channel control showing a real drop. In `refer-swap` and
`refer-mutual` each of those is checked separately for the farmer and the buyer.
Thresholds:

```bash
--set curriculum.refer_min_success=0.45   # lineup rungs
--set curriculum.trade_min_success=0.15   # trading rungs
--set curriculum.min_topsim_over_null=0.10
--set curriculum.min_channel_transfer=0.25
--set curriculum.min_positional_structure=0.15   # per role, swap and mutual
--set curriculum.mutual_min_report=0.30          # per role, mutual
--set curriculum.mutual_min_success=0.10         # both at once, mutual
```

Speaker pressures (see the README's section on them):

```bash
--symbol-cost 0.03
--set reward.rarity_cost=0.05 --set reward.convention=0.3
--set train.shaping_reinforce=0.2
```

**On a rented box use `--on-stall stop`.** If a phase runs past its budget without
converging, holding just burns money on a phase that is not working; stopping
leaves you a report saying exactly which criteria were unmet.

Watch the phase in the log or the metrics:

```bash
grep -E "PHASE|phase " runs/x/run.log
python -c "import json;[print(r['episode'],r['phase'],round(r['eval_success'],3)) for r in map(json.loads,open('runs/x/metrics.jsonl'))]"
```

Phase 1 rounds go to `lineups.jsonl` rather than the trade ledger — there are no
trades in it.

## 4. Generations are derived, not set

You cannot ask for N generations directly. An agent ages by the training updates
it takes part in -- nearly every update -- and dies at its lifespan, so turnover
falls out of run length and lifespan together:

```
generations  ~  (episodes / batch_size) / mean_lifespan_in_updates
```

To get more turnover, either run longer or shorten `population.lifespan_min` and
`lifespan_max`. Both presets and the banner print the resulting number, so you can
check before committing:

```bash
python -m orchard.run --config configs/gpu_full.json --benchmark | grep generations
```

Lifespans want to stay long enough that an agent can actually learn the language
before it dies -- the CPU runs needed ~550 updates to invent the lineup code --
and short enough that the population turns over often. The method uses 900-1,600
updates.

---

## 5. Memory, and why the big preset checkpoints

Parameters are **not** what limits this. The straight-through Gumbel channel
builds one autograd graph spanning every symbol step of an episode, so activation
memory grows as

```
batch  x  sequence length  x  d_model  x  layers  x  (symbols per turn x turns)
```

At `gpu_full`'s shape that is tens of gigabytes before a single parameter is
counted. `train.grad_checkpoint=true` recomputes encoder activations in the
backward pass instead of storing them, which cuts that by roughly an order of
magnitude and costs about 2× compute (measured on CPU; less on a GPU, where the
recompute is cheaper relative to memory traffic). It is on by default and
`tests/test_config.py` checks it gives the same update as without it.

If you hit an out-of-memory error, in this order:

1. halve `--batch-size` (a scale setting: the method is unchanged, because
   every schedule counts updates)
2. only then shrink `model.d_model`

(`channel.n_turns` and `channel.max_symbols` also multiply memory, but they are
part of the method: changing them is reported as a method change.)

---

## 6. Never conclude anything from one run

This simulation is bimodal. A population either finds a referential convention or
it does not. Four neighbouring conditions at 40k episodes gave **76%, 0%, 92% and
6%** of the channel headroom — a spread far larger than any effect worth
measuring. One seed per arm is a coin flip with a table around it.

```bash
python sweep.py --config configs/gpu_community.json --out runs/ablation --seeds 5 \
    --arm "bottleneck_on:" \
    --arm "bottleneck_off:--bottleneck off"
```

Reports mean, spread **and every individual seed**, so bimodality shows up instead
of being averaged into a number that means nothing. Use `--parallel 1` on one GPU;
raise it on a CPU box with spare cores.

Comparing two finished runs directly:

```bash
python compare_runs.py runs/a runs/b
```

---

## 7. Two traps that will waste your money

**Skewing demand suppresses language.** `world.zipf_alpha` exists because the
length/frequency prediction needs some meanings to be commoner than others. But
skew also makes "guess the common case" pay, and that is a local optimum agents
do not leave. Applying the skew to *which variety is wanted* took the channel from
64% of headroom to **0%**. It is therefore split: `zipf_alpha` applies to
quantity, and `zipf_alpha_variety` defaults to 0. Raising the latter reproduces
the failure.

**Raising the viable-deal rate can also suppress it.** More viable rounds means
more practice closing deals, but also a stronger "just accept" attractor. Between
55% and 71% viable the results were non-monotonic and dominated by seed noise. If
you change `world.p_stocked` or `world.need_max_frac`, re-measure with a sweep
rather than a single run.

---

## 8. Output, and what to bring home

Per run, in `--out`:

| file | keep it? |
|---|---|
| `report.md` | **yes** — rewritten every checkpoint, so it is readable mid-run |
| `metrics.jsonl` | **yes** — every checkpoint's full metric suite, small |
| `births.jsonl` | **yes** — what each newborn was taught and how it fared |
| `config.json` | **yes** — exactly reproduces the run |
| `plots/*.svg`, `*.png` | yes, small |
| `run.log` | probably |
| `trades.jsonl`, `trades.csv` | the big ones — see below |

The trade ledger is one row per episode. At 30M episodes that is enormous, so the
presets subsample it (`log.ledger_stride` of 200–1000) and you still get tens of
thousands of fully reconstructable trades. Set `--ledger-stride 1` only if you
genuinely want all of them, and check your disk first.

```bash
tar czf results.tgz runs/community --exclude='trades.*'    # a few MB
```

Matplotlib is optional: plots are always written as SVG by a dependency-free
writer, and the PNG versions appear as well if matplotlib imports.

---

## 9. Long runs over SSH

```bash
nohup bash cloud_run.sh > run.out 2>&1 &
tail -f runs/gpu_*/run.log
```

Or watch the machine-readable progress file, which is rewritten every batch:

```bash
watch -n 10 'python -c "import json;p=json.load(open(\"runs/x/progress.json\"));\
print(f\"{p[\"fraction\"]:.1%} {p[\"episodes_per_second\"]}/s\")"'
```

Interrupting is safe. Reports and plots are written at every checkpoint, so a run
you kill early still leaves a readable `report.md`, full metrics and a ledger.

---

## 10. Sanity checks before a long run

```bash
python -m unittest discover -s tests        # 152 tests, one to two minutes
```

Worth doing on the cloud box, not just locally — `tests/test_batched.py` asserts
the fast tensor path agrees **exactly** with the readable scalar one, and that is
the property you are trusting when you run anything large.
