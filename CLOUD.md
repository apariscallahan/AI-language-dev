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
changed, and the loader stops with a message saying so.

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
    coherence farmer 0.460 buyer 0.470 across 0.52 | overlap 0.96 | 42 words, 1.03 atoms/word, 3.47 words/utterance, 0% silent, 0% at buffer end
```

  The two success numbers are the two views of a swap rung (each role decoding),
  each against its own muted baseline. Coverage is per field, in lot order:
  **fruit, colour, quality, quantity, price**. `silent` is always 0% — every
  turn has to open with a word, because silence is what the muted control sounds
  like — and `at buffer end` should be near 0; if it climbs, agents are babbling
  into the cap.

  On a report rung (`mutual`, `order`, `offer`, `judge`) the same line names each
  field each role reports and how often it arrived, which is the number to watch
  there:

```
[checkpoint 22,118,400] rung mutual | success 0.364 (muted 0.000) | channel 0.51 of headroom | held-out 0.58 vs trained 0.80 | farmer reads fruit 0.96, colour 0.75, quality 0.76, quantity 0.71, price 0.68; buyer reads fruit 0.97, colour 0.75, quality 0.76, quantity 0.70, price 0.69 | buyer coverage 0.827 [0.97 0.75 0.76 0.72 0.70]
```
- **rung transitions**, with every criterion, passed or not;
- **`[costs]`**, once, when the first costed rung reaches its floor and the
  speaker costs start ramping in;
- **budget stops**, naming exactly what was unmet;
- the **final verdict** and the report path.

### Reading the rungs

| rung | what to look for |
|---|---|
| `name-fruit` | does it leave chance (0.333) at all, and when? This is the one rung that invents a code from nothing. Hindsight and the speaker costs are both off here. If it sits at chance past ~1,500 updates, nothing above it will work. `coherence 0.500` through the single-field rungs is expected: the two founders each keep a dialect the other can read, and converge from `name-all` on. The rolling success on the status line and the checkpoint's success should agree roughly — both are fruit rounds between two different agents — so a wide gap means training and measurement are asking different questions. (From `name-color` on, the rolling number also counts the easier rehearsal rounds and runs higher; the checkpoint measures only the new field.) |
| `name-color`, `name-quality`, `name-quantity`, `name-price` | these start from a population that already has words, so they should be *faster* than `name-fruit`. Each also prints a `still names fruit` / `still names colour` / … check: a rung whose own kind climbs while a rehearsed one falls back to chance is forgetting, not learning. Quantity has nine values (0 is "none of that") and price six; a colour round and a quantity round both have three candidates, so chance is 0.333 throughout. |
| `name-all` | the hard one: five fields in one utterance. Watch **words per utterance** climb toward 5 and **coverage** toward 0.30 on every field — a run that sticks at ~1.5 words and coverage ~0.15 is naming one field and guessing the rest. The checkpoint line prints `held-out vs trained`; they should stay close (a wide gap is memorisation). The convention bonus comes on here, so `coherence` should start to rise from 0.5. |
| `mutual` | both report the other's lot, all five fields, with the five belief heads `haggle` will use. Newcomers, deaths and hindsight feedback all switch on here, and the founders' dialects should merge — **coherence** is the number to watch. The speaker costs come on partway through, once the rung reaches its floor (`[costs]` in the log), and are ramped in over 200 updates: `atoms/word` and `words/utterance` should settle without success dropping. The held-out gate here is per field: `held-out 0.58 vs trained 0.80` is the mean per-field accuracy on reserved combinations against trained ones. |
| `order` | the buyer's request is a lot in the naming layout, so the buyer says exactly what it said in `name-all`; what is new is the farmer reporting it while looking at a barn of sixteen rows. Every field is `still carries`; if one falls to chance the farmer is not finding it among the rows. |
| `offer` | the farmer answers about the lot that was asked for — `stock`, `lot-quality`, `reservation` — and the buyer reports that. This is the first rung where a farmer has to **find a lot in its barn** by the words it heard; `stock arrives` is the number to watch, and stock 0 ("none of that") is a value it has to be able to say. |
| `judge` | both decide whether the deal is worth doing. Judged on the gain over silence, not the raw rate: ~68% of rounds are worth doing, so accepting everything scores 0.68 and still fails — which is exactly how `haggle` used to fail. |
| `haggle` | the pool splits into farmers and buyers (the log says so). Channel transfer well above zero, not just success from base rates. Exact price-bin agreement is the likely bottleneck. |
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
promotion. `cloud_run.sh` resumes automatically when `latest.pt` exists, so on a
spot or pre-emptible instance, rerun it pointing `RUN` at the same folder:

```bash
RUN=runs/2026-09-18_14-03-12UTC_orchard bash cloud_run.sh
```

A resumed run picks up whatever code it is started with, so this is also how to
move a running experiment onto newer code: stop it just after a checkpoint,
update, resume. Below the trading rungs the snapshot holds one pool written
twice; the loader restores it as one pool (an earlier version restored two
copies, which drifted apart and collapsed `mutual` within a checkpoint — a
snapshot from that version is repaired on load with a warning).

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
| `reward.costs_from_rung` | the rung from which the speaker pays for length and for new words (`mutual`, the first rung that invents no new word). Earlier, the cheapest way to be short is to say the same short nothing. |
| `reward.costs_ramp_trigger`, `reward.costs_ramp_updates` | within that rung the costs wait until its rolling success reaches this multiple of its promotion floor (1.0), then ramp in over this many updates (200). Fully on at the transition, `mutual` climbed at half the pace. |
| `reward.convention_from_rung` | the rung from which the speaker is paid for using the community's word (`name-all`, when every word exists). It cannot punish a new word -- a form only counts once it has 12 recent uses. |
| `reward.convention_contrast_samples` | how many other meanings' conventions a form is contrasted against (16), so one form for everything earns nothing. |
| `train.hindsight_from_rung` | the first rung with hindsight feedback (`mutual`). Earlier, it stops the first code forming. |
| `model.barn_lookup` | one cross-attention step from the farmer's hidden state to its barn rows, keyed on (fruit, colour), valued on (quality, stock); only active on a barn (true). Off, the plain transformer never learned to find the asked-for lot even supervised. |
| `curriculum.split_roles_at` | the rung where the one pool becomes farmers and buyers (`haggle`). Everything below it is one language in two seats, the report rungs included -- they run in both directions. |
| `curriculum.hard_distractor_frac` | share of open lineup rounds built as one-field near misses (0.9), the field drawn uniformly, so every field has to be named. |
| `world.holdout_combo_frac` | share of (fruit, colour, quality) combinations reserved and never trained on (0.25, a Latin square). |
| `curriculum.min_holdout_ratio` | how well a rung must do on those, as a share of how well it does on trained ones (0.60): whole-round in the lineup, per field on a report rung. The productivity gate. |
| `curriculum.min_field_transfer`, `curriculum.min_field_coverage` | every field is checked for every role; coverage is what catches a code that names one field in every slot. |
| `curriculum.mutual_min_report`, `curriculum.order_min_success` | the floor for reporting the other's whole lot exactly (0.25 for five fields) and for the fields a report rung introduced arriving together (0.25). |
| `bottleneck.meaning_holdout` | the share of the (fruit, colour, quality) combinations a newborn is not shown at all (0.25): the bottleneck proper. 0 makes a newborn a near-clone. |
| `bottleneck.coverage` | how much of the rest of the parent generation a newborn sees (1.0 — essentially all of it; lowering it puts *common* forms back at risk). |
| `bottleneck.frequency_skew` | how strongly a newborn's lessons favour common trades. |
| `reward.belief_qty_tol` | how exactly a reported quantity has to match in the trading rungs (1). Report rungs are exact. |
| `reward.symbol_cost`, `reward.atom_cost`, `reward.word_cost` | per symbol (0.01), per atom after the first in a word (0.03), per word (0.005): short words, not short sentences, and no repeating a word to the buffer end. |
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

216 tests, about four minutes. Worth doing on the GPU box, not just locally:
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
| a rehearsed kind falling to chance | forgetting. The mixture weights (`Phase.mix` in `curriculum.py`) are the dial. |
| a report rung passes every field but "reports combinations it never trained on" | the listener's heads have learned the training set's joint. The gate is per field and at 0.60 of the headroom; see README §11 before touching it. |
| `stock arrives` stuck at chance in `offer` while the request fields all carry | the farmer is not finding the asked-for lot among its barn rows. Check `model.barn_lookup` is on in the header; with it the lookup learns in a few hundred supervised steps, so what is missing is the reinforcement signal — look at whether hindsight is on and the buyer's `stock` report is being scored. |
| a rung stops the run | read the criteria it names in the log and in `promotions.jsonl`. Do not relax them to make it pass — they are the experiment. |
| out of memory in the first batch | `--set train.grad_checkpoint=true`, then a smaller batch. See §10. |
| every seed disagrees | expected. See §8. |
