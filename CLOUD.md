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

| preset | agents | brain | episodes | generations | weights + Adam | activations |
|---|---|---|---|---|---|---|
| `gpu_smoke` | 4 + 4 | d=64, 2 layers, 77k params | 20,000 | 2 | 0.01 GB | 0.4 GB |
| `gpu_small` | 8 + 8 | d=96, 3 layers, 353k params | 1,000,000 | 6.2 | 0.07 GB | 5.7 GB |
| `gpu_community` | 24 + 24 | d=160, 4 layers, 1.3M params | 6,000,000 | 10.0 | 0.73 GB | 3.2 GB |
| `gpu_full` | 64 + 64 | d=320, 6 layers, 7.5M params | 30,000,000 | 11.5 GB | 13.3 GB |

```bash
python -m orchard.run --config configs/gpu_community.json --out runs/community
```

**`gpu_smoke`** — four minutes, proves the box works end to end and writes every
artefact. Nothing will have been learned; that is not what it is for.

**`gpu_small`** — the smallest preset that can actually produce a language. Use it
to check a hypothesis before spending money on `gpu_community`.

**`gpu_community`** — 48 agents is where "a population" starts to mean something
rather than a handful of co-adapted pairs, and ten generations gives the
transmission bottleneck real work to do. This is the preset to reach for by
default.

**`gpu_full`** — 128 agents, 7.5M parameters each (960M in total), 30M episodes,
15 generations. Needs roughly **25 GB of VRAM** and will run for a long time;
benchmark it first. Gradient checkpointing is on, which is what makes it fit (see
§5).

---

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
| `--n-farmers`, `--n-buyers` | population size per role. More agents means a code that has to work for strangers, not a private pair. |
| `--episodes` | run length. Generations fall out of this — see §4. |
| `--batch-size` | episodes per optimiser step. The main speed/memory dial. |
| `--bottleneck on\|off` | the transmission bottleneck. Turning it off is the headline ablation. |
| `--turnover on\|off` | births and deaths. Off means one fixed cohort forever. |
| `model.d_model`, `model.n_layers`, `model.d_ff` | brain size. |
| `channel.atomic_vocab` | how many meaningless atoms words are built from. |
| `channel.max_symbols`, `channel.n_turns` | how long an utterance and a negotiation may be. Both multiply memory. |
| `world.zipf_alpha` | how skewed demand is. **Read §7 before raising it.** |
| `reward.decode`, `reward.understood` | the two halves of the communication loop. |
| `bottleneck.frequency_skew` | how strongly a newborn's lessons favour common trades. |

---

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
