# Running Orchard on a GPU

Orchard runs on a GPU. Start a run over SSH, close the laptop, and come back to
a report.

---

## 1. Ninety seconds to first output

```bash
git clone <your remote> orchard && cd orchard
pip install torch numpy scipy matplotlib        # matplotlib is optional; see §8

python -m orchard.run --smoke                   # no learning: checks the world
python -m orchard.run --benchmark               # this machine's speed, per rung
bash cloud_run.sh                               # the run
```

`--benchmark` reports **this machine's** episodes per second for the
configuration, what the full run will therefore cost in hours, and peak GPU
memory.

---

## 2. The configuration -- the only one

**There is one configuration: the defaults in `orchard/config.py`.** A GPU and
a CPU run exactly the same thing -- the same agents, brains, batch, schedule and
arithmetic (fp32 on every device; there is no GPU-only precision mode). The GPU
is simply faster: about 100 training updates a minute on an RTX 4090. That is
the point: a check on a CPU tests exactly what a GPU run does, so nothing can
work on one and fail on the other.

| | |
|---|---|
| ladder | `name-fruit` -> `name-color` -> `name-quality` -> `name-all` -> `describe-one` -> `mutual` -> `order` -> `haggle` -> `bargain` -> `market` |
| things to name | 4 fruits x 4 colours x 4 qualities = 64 combinations, of which 16 are reserved and never trained on |
| population | one pool until `order`, where each agent is copied into a farmer and a buyer, both fluent in the language the pool learned |
| community | founded by 2 farmers + 2 buyers; a newcomer of each role joins every 40 updates after the first rung, up to 6 + 6 |
| brain | 2-layer transformer, width 48, ~52k parameters, randomly initialised |
| batch | 256 episodes per training update, every rung |
| lifespan | 900-1,600 training updates |
| channel | atoms + hyphen / space / end; words are hyphen-joined atoms, an utterance is space-separated words; a 24-symbol buffer, not a cap |
| run ceiling | 6M episodes (~23k updates); each rung has its own budget, and a rung that exhausts it stops the run with a report |

These are the sizes the CPU runs that worked were made at. The configuration has
to be one a CPU can test, and a larger size on the GPU would be a second version
again.

**A run may change only how long it runs, its seed, its device and its output**
(`RUN_KEYS` in `config.py`). Anything else -- sizes included -- is printed as a
**method change** in the run header and in the report's summary statistics, so
a run that changed what is simulated cannot be mistaken for one that did not.
`configs/` holds only named experiments (`duality.json`: 12 varieties and 8
atoms, so no atom can name a whole meaning -- unvalidated; treat it as the
experiment, not the baseline). `tests/test_config.py` enforces all of this.

**Everything that means an amount of learning is counted in training updates**
(one update = one batch): rung budgets, promotion checks (every 25), checkpoints
(every 100), the temperature and entropy anneals (1,000 and 800), growth,
lifespans, and how long the population remembers what it has been saying (80).
The first GPU run counted lifespans in episodes, and every founder died after
~50 updates.

**Why founded small.** Six farmers and six buyers starting from random weights
never got the lineup game off chance: each farmer kept its own drifting code
(coherence 0.04-0.09), so no buyer could learn to read any of them. Two and two
invent a code; newcomers then learn it through the transmission bottleneck. Every
rung after the first waits for, and is judged on, the full community, and its
budget only starts counting once the community is full.

**Hindsight feedback starts at `mutual`.** With it on from the first rung,
a listener told the answer learned that the still-random messages carried
nothing, went near-uniform, and the speaker's gradient died with it: no code
ever formed. The rungs where a code has to form from nothing run without it.

## 2a. What you see while it runs

`cloud_run.sh` keeps the terminal quiet except for:

- one **status line a minute**: time (UTC), episodes done / total, the update
  count, rung and how many of its maximum updates it has used, episodes per
  second, ETA, rolling success, community size, births, peak GPU memory;
- a **two-line headline at every checkpoint**: success against the muted
  channel, share of headroom the channel carries, each role's field coverage
  (fruit / colour / quality), coherence, cross-role overlap, word counts;
- **rung transitions** and **budget stops** with every criterion;
- the **final verdict** and the report path.

The header's `method` line should read `the one configuration (nothing simulated
was changed)`.

**Expect chance for a while.** The lineup code forms suddenly, and late: the
runs that worked sat at 0.25 (chance) until ~300-600 updates, then climbed past
0.4 within about 50 updates. Chance at update 200 is normal; chance at update
1,500 is not.

Everything else goes to the run folder:

| file | what it is |
|---|---|
| `run.log` | the complete console history, including the long checkpoint blocks |
| `transcripts.txt` | every `log.transcript_stride`-th round as *expected / dialogue / outcome* lines, with a banner at each rung |
| `report.md` | rewritten at every checkpoint; **summary statistics** at the top |
| `promotions.jsonl` | every promotion check, passed or not, with its evidence |
| `progress.json` | a one-line status, for scripts |

Run folders are named for their start time in UTC and the run:
`runs/2026-09-18_14-03-12UTC_orchard`.

## 2b. Interruptions, snapshots and resuming

A snapshot of the whole community -- weights, optimiser state, recent usage, the
transcript store, the curriculum record -- is written to
`<run>/snapshots/latest.pt` at every checkpoint and to `after-<rung>.pt` at every
promotion. `cloud_run.sh` resumes automatically when `latest.pt` exists, so on a
spot or pre-emptible instance, rerun it pointing `RUN` at the run's folder:

```bash
bash cloud_run.sh                                               # starts runs/<UTC time>_orchard
RUN=runs/2026-09-18_14-03-12UTC_orchard bash cloud_run.sh       # resumes it
```

A resumed run picks up whatever code it is started with, so this is also how to
move a running experiment onto newer code: stop it just after a checkpoint (the
snapshot is written then), update, and resume. Older snapshots load too.

Branch an experiment off any rung (the header will list what you changed):

```bash
python -m orchard.run --out runs/branch \
    --resume runs/<run>/snapshots/after-name-all.pt \
    --set curriculum.order_min_success=0.6
```

## 2c. Measuring a saved community

```bash
python -m orchard.analyse --snapshot runs/<run>/snapshots/after-name-all.pt
```

Runs the full metric suite on the rung the snapshot closed and prints the
**language-properties scorecard** (reference, productivity, intentionality,
decontextualised, displaced, interchangeable, generic, perspectives, cultural
transmission, duality of patterning), each with how it is measured, the value,
and present / partial / absent / not testable. No training happens; it runs
on a laptop against a snapshot copied down from the box.

## 3. Changing settings

**Run settings** -- nothing simulated changes:

```bash
bash cloud_run.sh --episodes 2000000 --seed 3 --device cuda:1 --ledger-stride 100
```

**Anything else** is an experiment and is reported as one. Named flags cover the
common ones, and `--set section.key=value` reaches any field:

```bash
bash cloud_run.sh --bottleneck off                       # the headline ablation
bash cloud_run.sh --n-farmers 16 --n-buyers 16           # a bigger community
bash cloud_run.sh --set world.zipf_alpha=0.0 --set reward.understood=0.6
```

Every run writes the exact configuration it used to `<out>/config.json`, so a
run is always reproducible from its own directory:

```bash
python -m orchard.run --config runs/<run>/config.json --out runs/rerun --seed 9
```

### The settings worth knowing

| setting | what it does |
|---|---|
| `--curriculum on\|off` | the referential-then-trading ladder. Off means the full task from random weights, which has not been made to work. |
| `population.founders_farmers/_buyers`, `population.grow_every_updates` | found the community small and grow it after the first rung. 0 founders = start at full size. |
| `curriculum.hard_distractor_frac` | share of lineup rounds built as one-field near misses, so every field (quantity included) has to be named. |
| `curriculum.holdout_tuple_frac` | share of (variety, quantity, quality) combinations never trained on: the productivity test. |
| `curriculum.min_field_transfer`, `curriculum.mutual_qty_tol` | the mutual rung checks every field for every role, quantity exactly. |
| `train.hindsight_from_rung` | the first rung with hindsight feedback (`mutual`). |
| `curriculum.min_holdout_ratio` | how well a rung must do on combinations it never trained on, as a share of how well it does on trained ones (0.60). The productivity gate. |
| `curriculum.split_roles_at` | the rung where the one pool becomes farmers and buyers (`order`). |
| `world.holdout_combo_frac` | share of (fruit, colour, quality) combinations reserved (0.25, a Latin square). |
| `reward.atom_cost`, `reward.word_cost` | length is charged per atom after the first in a word, and much less per word: short words, not short sentences. |
| `--on-stall hold\|stop` | what to do if a rung never converges (a run setting). |
| `bottleneck.coverage` | how much of the parent generation a newborn sees. 1.0 means essentially all of it; lowering it puts common forms back at risk. |
| `--n-farmers`, `--n-buyers` | community size per role. |
| `--episodes` | run length (a run setting). Generations fall out of this -- see §4. |
| `--bottleneck on\|off` | the transmission bottleneck. Turning it off is the headline ablation. |
| `--turnover on\|off` | births and deaths. Off means one fixed cohort forever. |
| `channel.atomic_vocab` | how many meaningless atoms words are built from. |
| `channel.max_symbols`, `channel.n_turns` | the per-turn buffer (24: a buffer, not a pressure -- the symbol cost sets length) and the number of turns. |
| `channel.enforce_word_grammar` | atoms and marks alternate: `a3-a7 a1` is a two-atom word and a one-atom word, exactly as emitted. |
| `world.zipf_alpha` | how skewed demand is. **Read §7 before raising it.** |
| `reward.decode`, `reward.understood` | the two halves of the communication loop. |
| `bottleneck.frequency_skew` | how strongly a newborn's lessons favour common trades. |

---

## 3b. The curriculum

Runs start on a lineup game and work up to the full market. Nothing is
reinitialised between rungs; the same population carries its weights forward.

Each rung has its own (min, max) budget in training updates -- 80 to 2,500 for
the single-field naming rungs, 80 to 2,500 for `name-all`, `describe-one` and
`order`, 80 to 3,500 for `mutual`, `haggle` and `bargain`, open for `market`.
Promotion needs success clear of chance and the muted-channel control showing a
real drop; in every lineup rung and `mutual` that is checked separately for each
seat. `name-all` and `mutual` additionally need structure (topsim clear of its
shuffled null, coverage of every field) and **success on the reserved
combinations** -- at least 60% of the rate on trained ones. That last one is
what a code of whole-thing names cannot pass.

**Keep `--on-stall stop`** (the default). If a rung runs past its budget
without converging, holding just burns money on a rung that is not working;
stopping leaves a report saying exactly which criteria were unmet.

Watch the rung in the log or the metrics:

```bash
grep -E "PHASE|phase " runs/<run>/run.log
```

Lineup rounds go to `lineups.jsonl` rather than the trade ledger -- there are no
trades in them.

## 4. Generations are derived, not set

An agent ages by the training updates it takes part in -- nearly every update --
and dies at its lifespan, so turnover falls out of run length and lifespan
together:

```
generations  ~  (episodes / batch_size) / mean_lifespan_in_updates
```

The header and `--benchmark` print the resulting number. Lifespans want to stay
long enough that an agent can learn the language before it dies -- the lineup
code takes ~300-600 updates to form -- and short enough that the population
turns over often.

---

## 5. Memory

At this size a run needs well under 1 GB of GPU memory. Parameters are never the
limit here: the straight-through Gumbel channel builds one autograd graph
spanning every symbol step of an episode, so activation memory grows as

```
batch  x  sequence length  x  d_model  x  layers  x  (symbols per turn x turns)
```

If an experiment makes that too big, `--set train.grad_checkpoint=true`
recomputes activations in the backward pass instead of keeping them. It changes
memory only -- `tests/test_config.py` checks the update is the same -- so it is
a run setting. It is what let a 4,096 batch fit on a 24 GB card.

---

## 6. Never conclude anything from one run

This simulation is bimodal. A population either finds a referential convention or
it does not. Four neighbouring conditions at 40k episodes gave **76%, 0%, 92% and
6%** of the channel headroom -- a spread far larger than any effect worth
measuring. One seed per arm is a coin flip with a table around it.

```bash
python sweep.py --out runs/ablation --seeds 5 --arm "bottleneck_on:" --arm "bottleneck_off:--bottleneck off"
```

Reports mean, spread **and every individual seed**, so bimodality shows up instead
of being averaged into a number that means nothing. Use `--parallel 1` on one GPU.

Comparing two finished runs directly:

```bash
python compare_runs.py runs/a runs/b
```

---

## 7. Two traps

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
| `report.md` | **yes** -- rewritten every checkpoint, so it is readable mid-run |
| `metrics.jsonl` | **yes** -- every checkpoint's full metric suite, small |
| `births.jsonl` | **yes** -- what each newborn was taught and how it fared |
| `config.json` | **yes** -- exactly reproduces the run |
| `plots/*.svg`, `*.png` | yes, small |
| `run.log` | probably |
| `trades.jsonl`, `trades.csv` | the big ones -- see below |

The trade ledger subsamples episodes (`log.ledger_stride`, 500) and still keeps
thousands of fully reconstructable trades. Set `--ledger-stride 1` only if you
genuinely want all of them, and check your disk first.

```bash
tar czf results.tgz runs/<run> --exclude='trades.*'    # a few MB
```

Matplotlib is optional: plots are always written as SVG by a dependency-free
writer, and the PNG versions appear as well if matplotlib imports.

---

## 9. Long runs over SSH

```bash
nohup bash cloud_run.sh > run.out 2>&1 &
```

```bash
tail -f run.out
```

Interrupting is safe. Reports and plots are written at every checkpoint, so a run
you kill early still leaves a readable `report.md`, full metrics and a ledger.

---

## 10. Sanity checks before a long run

```bash
python -m unittest discover -s tests
```

154 tests, about a minute. Worth doing on the GPU box, not just locally --
`tests/test_batched.py` asserts the fast tensor path agrees **exactly** with the
readable scalar one, and `tests/test_config.py` that there is one configuration
and no device-specific arithmetic.
