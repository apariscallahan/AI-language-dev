# Running Orchard on a cloud GPU

Everything here is the CLI. The GUI (`Orchard.bat`) is a convenience for a laptop
and is not needed — and not wanted — on a rented box, where you want to start a
run over SSH, close the laptop, and come back to a report.

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

Every preset uses the same method, validated on a CPU at 2 + 2 agents before
being scaled: the seven-rung ladder (`refer`, `refer-swap`, `refer-mutual`,
`order`, `haggle`, `bargain`, `market`), per-role promotion, hard lineup rounds,
held-out combinations, speaker pressures, and a community **founded by 2 farmers
and 2 buyers that grows** to full size once the first rung is passed. Only the
scale differs.

| preset | community | brain | params / agent | episodes | batch | ~generations |
|---|---|---|---|---|---|---|
| `gpu_smoke` | 2+2 -> 8+8 | d=64, 2 layers | 87k | 0.5M | 1,024 | 2 |
| `gpu_small` | 2+2 -> 16+16 | d=64, 2 layers | 87k | 8M | 2,048 | 6 |
| `gpu_community` | 2+2 -> 48+48 | d=96, 3 layers | 294k | 30M | 4,096 | 6 |
| `gpu_full` | 2+2 -> 128+128 | d=96, 3 layers | 294k | 80M | 4,096 | 6 |
| `gpu_duality` | 2+2 -> 48+48 | d=96, 3 layers | 301k | 40M | 2,048 | 6 |

**Memory (24 GB card).** Training backpropagates through every symbol step,
and each step re-encodes the conversation so far, so activation memory is the
constraint -- not parameters. All GPU presets use gradient checkpointing, which
keeps only the tokens fed into each step. Measured memory held for the backward
pass, per 1,000 episodes: lineup rung ~0.2 GB, mutual ~0.7 GB, full market
~2.4 GB (duality world: 0.7 / 1.7 / 5.0 GB). Batch sizes are set so the heaviest
rung stays under ~10 GB on a 24 GB card; on a 40-80 GB card they can be doubled
(`--batch-size`). Without checkpointing the same batch needed 56-365 GB.

Brains are deliberately small and communities large. A supervised check showed
the 48k-parameter CPU brain already learns a full compositional code for every
meaning (100% on combinations it never saw), so capacity is not what limits
these runs; agent count is what makes "a community" mean something. Rung
budgets, checkpoint cadence and annealing are all set in optimiser *updates*
(e.g. a lineup rung may take up to 2,500 updates) and converted to episodes by
the preset's batch size, so they mean the same thing at every scale.

```bash
CONFIG=configs/gpu_community.json bash cloud_run.sh
```

**Why founded small.** Six farmers and six buyers starting from random weights
never got the lineup game off chance in 200k episodes: each farmer kept its own
drifting code (coherence 0.04-0.09), so no buyer could learn to read any of them.
Two and two invent a code in ~80-140k episodes. So every preset founds the
community at 2 + 2, and after the first rung a newcomer of each role joins every
`population.grow_every` episodes -- random weights, then the transmission
bottleneck on the community's transcripts -- until it reaches full size. Every
rung after the first waits for, and is judged on, the full community.

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

**Anneal schedules are in updates, not run fractions.** The Gumbel temperature
and the entropy bonus anneal over ~1,000 and ~800 optimiser updates, as in the
CPU runs that worked. An earlier version annealed over a fraction of the whole
run, so giving a run more episodes silently slowed its learning -- a 2.4M-episode
run was still at temperature 1.36 after 200k episodes and never left chance.

## 2b. Interruptions, snapshots and resuming

A snapshot of the whole community -- weights, optimiser state, recent usage, the
transcript store, the curriculum record -- is written to
`<run>/snapshots/latest.pt` at every checkpoint and to `after-<rung>.pt` at every
promotion. `cloud_run.sh` resumes automatically when `latest.pt` exists, so on a
spot or pre-emptible instance just rerun the same command with the same `RUN`:

```bash
RUN=runs/community CONFIG=configs/gpu_community.json bash cloud_run.sh   # starts
RUN=runs/community CONFIG=configs/gpu_community.json bash cloud_run.sh   # resumes
```

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

**Edit a config file.** The presets are plain JSON; copy one and change it. Every
run also writes the exact config it used to `<out>/config.json`, so a run is
always reproducible from its own directory:

```bash
python -m orchard.run --config runs/community/config.json --out runs/community_rerun --seed 9
```

### The settings worth knowing

| setting | what it does |
|---|---|
| `--curriculum on\|off` | the referential-then-trading ladder. Off means the full task from random weights, which has not been made to work. |
| `population.founders_farmers/_buyers`, `population.grow_every` | found the community small and grow it after the first rung. 0 founders = start at full size. |
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

The rungs are `refer`, `refer-swap`, `refer-mutual`, `haggle`, `bargain`,
`market`. Each has its own (min, max) episode budget:

```bash
--curriculum off                          # straight to the full trading task
--set curriculum.n_candidates=6           # a harder lineup (chance 1/6)
--set 'curriculum.rung_budgets={"refer":[20000,600000],"refer-swap":[20000,600000],"refer-mutual":[20000,800000],"haggle":[20000,600000],"bargain":[20000,600000],"market":[20000,1000000000000]}'
--set curriculum.check_every=5000         # how often promotion is probed
--on-stall stop                           # end the run on a blown budget (the default)
```

Promotion needs success clear of chance, topsim clear of its shuffled null, *and*
the muted-channel control showing a real drop. In `refer-swap` and
`refer-mutual` each of those is checked separately for the farmer and the buyer.
Thresholds:

```bash
--set curriculum.refer_min_success=0.55   # lineup rungs
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
--set reward.rarity_cost=0.05 --set reward.convention=0.15
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

You cannot ask for N generations directly. An agent ages by the episodes **it
personally plays**, and dies at its lifespan, so turnover falls out of three
things together:

```
generations  =  (episodes / n_farmers) / mean_lifespan
```

To get more turnover, either run longer or shorten `population.lifespan_min` and
`lifespan_max`. Both presets and the banner print the resulting number, so you can
check before committing:

```bash
python -m orchard.run --config configs/gpu_full.json --benchmark | grep generations
```

Lifespans want to stay long enough that an agent can actually learn the language
before it dies — somewhere north of 20,000 episodes of its own experience — and
short enough that the population turns over often. The presets sit at 20k–30k.

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
recompute is cheaper relative to memory traffic). It is on for `gpu_full` and
`gpu_community`.

If you hit an out-of-memory error, in this order:

1. turn on `--set train.grad_checkpoint=true`
2. halve `--batch-size`
3. reduce `channel.n_turns` or `channel.max_symbols` — these multiply memory
   *and* are the hardest thing for the agents to learn over, so shortening them
   often helps twice
4. only then shrink `model.d_model`

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
python -m unittest discover -s tests        # 71 tests, about 25 seconds
```

Worth doing on the cloud box, not just locally — `tests/test_batched.py` asserts
the fast tensor path agrees **exactly** with the readable scalar one, and that is
the property you are trusting when you run anything large.
