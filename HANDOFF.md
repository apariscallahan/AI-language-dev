# Handoff: picking Orchard up on a GPU machine

This is written for the next Claude Code session, which runs on a Linux box with
a GPU. Read it top to bottom before running anything. `README.md` explains the
design; `CLOUD.md` is the command reference; this file is **where things stand,
what is proven, what is not, and what to do next**.

---

## 0. First, on the new machine

```bash
git clone https://github.com/apariscallahan/AI-language-dev.git && cd AI-language-dev
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match the driver
pip install -r requirements.txt
python -m unittest discover -s tests -q        # expect: all tests OK (154 at handoff)
python -m orchard.run --benchmark              # this machine's speed per rung
bash cloud_run.sh                              # the run
```

The user runs **GPU only**, and wants the CPU and GPU versions to be **exactly
the same**. So there is **one configuration**: the defaults in
`orchard/config.py`, sizes included (2 + 2 founders growing to 6 + 6, d48 L2,
batch 256 -- the scale of the CPU runs that worked, and one a CPU can test). A
GPU runs it faster; that is the only difference. There is no device-specific
arithmetic (bf16 and TF32 were removed; `hardware.setup` pins fp32). A run may
change only `RUN_KEYS` (length, seed, device, output); anything else, sizes
included, is printed as a method change in the header and the report.
`configs/` holds only named experiments (`duality.json`). **Test on the CPU with
exactly the configuration the GPU runs** (`python -m orchard.run`), never with a
smaller "CPU version"; `tests/test_config.py` enforces the rest. Every schedule
is counted in training updates (section 4 item 10).

---

## 1. The project in five lines

Randomly-initialised Farmer and Buyer transformer agents must invent a language
to trade apples (`orchard_language_emergence_spec.md`,
`additional-improvements-1.md`). **No pretrained models, embeddings or text
corpora anywhere, ever.** Agents emit atoms plus HYPHEN / SPACE / END; a word is
hyphen-joined atoms, an utterance is space-separated words. They learn through a
seven-rung curriculum (`orchard/curriculum.py`), with population turnover and a
transmission bottleneck (newborns learn from transcripts, never weights).

## 2. What the user wants (standing preferences)

- **Channel shape**: `a3-a7 a1` = a two-atom word and a one-atom word. Atoms and
  marks alternate (`channel.enforce_word_grammar`), and the rendered text is
  exactly what was emitted. **No small length cap**: `max_symbols` is a 24-symbol
  buffer (32 in the duality experiment); length is set by a cost, never a binding
  cap. Ideally separate words become noun-like (variety), adjective-like
  (quality), numeral-like (quantity); if it doesn't happen, report it honestly.
- **Don't impose the language.** Shape pressures in the world and *measure* the
  properties of language (reference, productivity, intentionality,
  decontextualised, displaced, interchangeable, generic, perspectives, cultural
  transmission, duality of patterning). See `orchard/properties.py` and the
  report's section 3f.
- **Honest, per-role reporting.** Two-way rungs are judged per role, never
  pooled. A rung that blows its budget stops the run and says exactly which
  criteria were unmet. Never talk a weak result up.
- Smaller brains, **more agents** (the latest instruction). Presets reflect it.
- The user asks for specific numbers after runs: furthest rung and the criteria
  at each transition, per-role structure scores, coherence, distinct words, mean
  word length, cross-role overlap, newborn token accuracy by role. The report and
  `promotions.jsonl` contain all of them.

## 3. The method as it stands (all in the code now)

| piece | where | why |
|---|---|---|
| rungs: `name-fruit`, `name-color`, `name-quality`, `name-all`, `describe-one`, `mutual`, `order`, `haggle`, `bargain`, `market` | `curriculum.py` | naming is taught one field at a time before anything is traded; `order` (farmer fills the buyer's order with its deal heads) was added because `haggle` needed deal heads nothing had trained |
| a thing is **(fruit, colour, quality)**, 4 x 4 x 4, and a quarter of the combinations (a Latin square) are never trained on | `world.ComboHoldout` | separate fields are what make an adjective worth inventing; the reserved combinations are the productivity test, and the Latin square keeps every lineup free of candidates that could not be the answer |
| **one pool of agents** until `order`, then each is copied into a farmer and a buyer | `Population.split_roles` | one language rather than two to reconcile |
| length charged **per atom after the first in a word**, much less per word | `env.length_cost` | a fused name is one long word; naming the parts is several short ones, and must not be taxed for it |
| per-role promotion in swap/mutual, incl. **field coverage** and **per-field decode** checks | `curriculum.evaluate_rung` | a pooled average let a thin code pass |
| **hard lineup rounds**: an anchor plus one-field near misses, target uniform | `ReferentialWorld._cluster` | random distractors let variety+quality pick the target 69% of the time |
| **held-out tuple combinations** in lineups | `ReferentialWorld.holdout` | the productivity (zero-shot) test |
| **hindsight feedback**: scored heads trained towards the outcome; gradient reaches the speaker through the straight-through channel. **From `mutual` up only** (`train.hindsight_from_rung`) | `curriculum.hindsight_targets`, `hindsight_applies`, `gumbel.py` | without it the code locked into variety only (see section 4); with it from the start, no code formed at all (item 11) |
| **one configuration, fp32 on every device** (no bf16/TF32 at all) | `config.py`, `hardware.py` | a CPU check tests exactly what a GPU run does; the code forms from ~0.02-logit signals |
| speaker costs (symbol, coining) **off for the first rung**, on after it | `Trainer.update_cost_gate` | costs during rung 1 capped it |
| contrastive, symbol-level **convention bonus** | `conventions.py` | coherence across the community |
| community **founded at 2 + 2**, newcomers join after rung 1 | `Population.add_newcomer`, `Trainer.maybe_grow` | 6 + 6 from scratch never left chance |
| phase-aware speaker order everywhere (costs, bottleneck targets, "me" embedding, probes) | `Phase.own_positions` / `self_mask` | the old buyer-opens order broke farmer newborns (token accuracy 0.000) |
| **snapshots** at every promotion and checkpoint; `--resume`; auto-resume in `cloud_run.sh` | `Trainer.save_snapshot` | iterate on a rung without replaying the ones below; survive pre-emption |
| `python -m orchard.analyse --snapshot ...` | `analyse.py` | full metric suite + properties scorecard on any snapshot (old snapshots load: `Config.from_dict(allow_legacy=True)`) |
| one training path: straight-through Gumbel on symbols + REINFORCE on decisions, tensor world | `gumbel.py`, `batched.py` | the pure-REINFORCE and scalar-world training paths were removed (git history has them) |
| every schedule in **training updates**; rung budget counts from full community size | `config.py`, `Trainer.run` | episode counts meant different learning at every batch size |

## 4. What happened, with the evidence (so you do not repeat it)

1. **Farmer bottleneck bug** (fixed): static buyer-opens order meant farmer
   newborns had no targets in `refer` and buyers imitated farmer words.
2. **Six + six from scratch never leaves chance** (4 attempts, 200k episodes
   each): farmer coherence 0.04-0.09, codes drifting 85% per 50k; no
   convention-bonus strength fixed it. **Founding at 2 + 2 and growing works**:
   run `ladder3` grew to 6 + 6 and passed `refer` (0.62), `refer-swap` (per role),
   `refer-mutual` (per role, borderline) with coherence ~1.0 and cross-role
   overlap 0.52-0.57.
3. **`haggle` then plateaued at ~7% success with channel transfer 0.00-0.02**:
   always-accept plus base-rate guessing. Diagnosis: the language carried quality
   (0.88) and some variety (0.52) but **quantity at chance** (0.20 exact) -- the
   earlier rungs never required it, and `haggle` needs exact quantity.
4. **First hard-distractor design leaked the target** (near misses built around
   the target made it the centre of the lineup: 42% success with the channel
   muted, chance 25%). Fixed with the anchor-cluster design; a test guards it.
5. **Ramping speaker costs in during `refer` capped it at ~0.42** once every
   field had to be named; costs are now off for the whole first rung.
6. **Validation (2 + 2, old 4-symbol channel) passed `refer` (0.535) and
   `refer-swap` (per role) but the code was variety-only**: live messages carried
   1.5 bits of variety and 0.01-0.05 bits of quantity or quality, e.g.
   `a13-a13-a13-a13`. "Positional structure 1.00" was *redundancy* (every slot
   naming the variety) -- use **field coverage** instead. `refer-mutual` sat with
   quantity and quality at their muted baselines for 110k episodes. This is why
   **hindsight feedback** was added. **Its effect has not been measured yet.**
7. **Capacity is not the limit**: supervised, the 48k-parameter CPU brain learns a
   full compositional code at 100% on held-out combinations
   (the capacity check is described in section 6).
8. **The last CPU run** (word grammar + 24-symbol buffer + hindsight) was stopped
   at 50k episodes in `refer`, at chance, as expected that early. Utterances
   already looked right: `a6-a4 a3`, `a4-a4 a11 a0 a13`. CPU throughput with the
   longer buffer was ~4k episodes/min, which is why the work moved to a GPU.

9. **First GPU run (`gpu_community`, RTX 4090): stuck at chance in `refer`
   after 1.6M episodes, founders already at generation 7-8.** Lifespans were
   counted in episodes; a 4,096 batch shared by 2 + 2 founders aged each founder
   2,048 episodes per update (16x the CPU runs), so each lived ~50 updates --
   far too short to invent a code (CPU needed ~550 updates). Fixed: lifespans
   are counted in training updates in every GPU preset (900-1,600). Also fixed:
   the report judged a lineup success of 0.244 against the *trading* chance
   (~0) and called it "far above chance"; it now uses the rung's own chance.
   **Rule of thumb: anything counted in episodes must be checked against the
   batch size and the number of agents sharing it.**

10. **Consolidation to one GPU version** (after the user saw CPU/GPU
    inconsistency). Every schedule now counts training updates:
    `curriculum.rung_budget_updates` (refer/swap/order 80-2,500,
    mutual/haggle/bargain 80-3,500), `check_every_updates` 25,
    `log.checkpoint_every_updates` 100, `train.tau_anneal_updates` 1,000,
    `entropy_anneal_updates` 800, `population.grow_every_updates` 20, lifespans
    900-1,600, `reward.usage_half_life_updates` 80. The GPU presets already
    had equivalent update-based values except two things found in the audit:
    the population-usage half-life (20,000 *episodes* = ~80 updates on the CPU
    runs but **~5 updates on the GPU**, so the coining cost and convention bonus
    tracked a 16x shorter memory), and community growth eating rung budgets
    (128 + 128 takes ~2,500 updates to grow, as long as the swap rung's whole
    budget) -- a rung's budget now counts only once the community is full. The
    Windows GUI (its own presets: 4-symbol cap, no founders, old costs), the
    legacy CPU configs, the REINFORCE and scalar training paths and the unused
    `compile` switch were removed. Old config keys raise an error naming their
    replacement; `--checkpoint-every` is now `--checkpoint-every-updates`.
    Snapshots written before this still resume (episode counts are converted
    through that run's batch size).

11. **Hindsight feedback stopped the lineup code forming** (found 2026-09-19).
    A CPU-scale run on the GPU (6 + 6, d48, batch 256) sat at chance for all
    2,500 updates of `refer`. Diagnostics, all at batch 256, 2 + 2:
    | run | result |
    |---|---|
    | GPU, old 4-symbol channel, no hindsight, fp32 (arm A) | verdict "beat chance" by 1,000 updates |
    | GPU, current channel, no hindsight, fp32 (arm C) | verdict "beat chance" by 1,000 updates |
    | GPU, current channel, hindsight, bf16 (main run) | chance at 2,500 updates |
    | CPU core loop, old channel, no hindsight | 0.25 until ~500, **0.39 at 600** and rising |
    | CPU core loop, old channel, hindsight | 0.25 at 600, speaker entropy flat at maximum |
    Mechanism (measured): in both cases the listener stops reacting to the
    still-random messages within ~25 updates (message sensitivity 0.15 ->
    0.02 in logits). Without hindsight it still forms confident, arbitrary
    preferences (choice-logit spread 0.5 -> 0.9), and REINFORCE's feedback
    through it eventually breaks the symmetry. With hindsight the supervised
    loss correctly teaches it that the messages are uninformative, so it goes
    near-uniform (spread 0.5 -> 0.17) and the speaker's gradient dies with it.
    Fix: hindsight only from `refer-mutual` up. Also found by the same runs:
    a `NameError` in `detect_degenerate` (a leftover of the updates rename) that
    crashed two arms at their first settled checkpoint -- fixed and now tested.
    Also: the GPU now runs in full precision (bf16/TF32 removed), so CPU checks
    and GPU runs compute the same thing.

12. **One configuration** (2026-09-19, at the user's request: "I want the two to
    be exactly the same"). The size presets were removed and the defaults set to
    the scale the working CPU runs used: 2 + 2 founders growing to 6 + 6 (one
    newcomer per 40 updates, as in `ladder3`), d48 L2 ff96, batch 256 on every
    rung, bottleneck batch 256 (the GPU value 1024 gave newborns 4x fewer
    training steps). Checked end to end on the CPU with `python -m orchard.run`
    -- the exact command the GPU runs. **Lift-off takes ~550 updates: do not read
    "chance at update 200" as failure.**

## 5. What is NOT validated yet -- your first job

In order of importance. The first run to do is **the configuration**, all the
way up the ladder:

```bash
bash cloud_run.sh                                    # folder: runs/<UTC start>_orchard
```

The terminal gets a status line a minute, a two-line headline per checkpoint
(success vs muted, channel headroom, per-role field coverage, coherence, words),
rung transitions and the verdict. `run.log` has the full checkpoint blocks,
`transcripts.txt` has sampled rounds as expected / dialogue / outcome lines, and
`report.md` opens with a summary-statistics table.

What to check, rung by rung (`promotions.jsonl` in the run folder has every check):

- **`name-fruit`**: does it leave chance (0.333) at all, and when? This is the
  one rung that has to invent a code from nothing, and it is far easier than the
  old `refer` was -- four values, one field, everything else held fixed. If it
  sits at chance past ~1,000 updates, nothing further up will work either.
  Hindsight is off here (section 4 item 11), and so are the speaker costs.
- **`name-color` / `name-quality`**: these start from a population that already
  has words. They should be *much* faster than `name-fruit`; if they are not,
  the words are not being reused and something is resetting the code.
- **`name-all`**: the first rung gated on **held-out combinations**
  (`describes combinations it never trained on` in the checks). Watch the two
  numbers in the checkpoint line: success on trained combinations against
  success on reserved ones. A code of whole-thing names shows a wide gap and
  will stall here -- that is the gate doing its job, not a bug. Also watch each
  seat's **field coverage** (fruit / colour / quality).
- **`describe-one`**: one word has to mean a colour whichever round it is asked
  in. If `name-all` passed and this stalls, the code is positional rather than
  lexical (the report's word-classes row will show it).
- **`mutual`**: both report the other's thing; gated on held-out too. This is
  where hindsight feedback switches on -- the first rung whose behaviour it can
  explain.
- **`order`**: the pool splits into farmers and buyers here (the log says so).
  The farmer must fill fruit, colour and quantity exactly (bar 0.50).
- **`haggle`**: channel transfer well above zero, not just success from base
  rates. Price coordination (both sides must pick the same price bin) is the
  likely next bottleneck.
- throughout: the report's **word-classes** row (do separate words specialise to
  separate fields? that is the adjective question), cross-role overlap (should
  stay high -- one pool, one language), and the share of utterances at the
  buffer end (should be ~0).

Then, if the user wants it: a bigger community (`--n-farmers/--n-buyers`, a
method change the header reports -- and the change belongs in the one
configuration, tested the same way, if it is kept), or the `duality` experiment
(12 varieties vs 8 atoms: where duality of patterning is *necessary*;
unvalidated).

## 5b. Performance

`python -m orchard.run --benchmark` times three rungs (one-turn lineup, two-turn
mutual, full market) at **full community size** and estimates hours, on
whatever device it finds.

Where the time goes (profiled on CPU, 8 + 8 agents, 64 episodes, market rung):
**784 separate agent forward passes per training step** -- one per agent per
symbol step (4 turns x 24 symbols x 8 agents) plus the decisions -- each
re-encoding the whole conversation so far, then a backward pass through all of
them (6 s forward, 8 s backward on CPU). The 24-symbol buffer the user asked for
makes generation 6x longer than the old 4-symbol cap; that is the right
trade, but it makes this loop the bottleneck. On a GPU the cost is dominated by
the *number of calls* (kernel launches), not arithmetic, so it grows linearly
with agents per role.

**Memory (fixed; matters again only for much bigger settings).** A 48 + 48,
batch 4,096 run on an RTX 4090 (24 GB) ran out of memory in the first training
batch: every symbol step re-encodes the conversation and the Gumbel path
backpropagates through all of them, so saved activations were ~14 MB per
episode in the lineup and ~90 MB in the market (56-365 GB at batch 4,096). Now
`CommNet.encode` checkpoints embedding + layers + final norm, keyed on grad
mode (not train mode -- newborns leave their apprenticeship in eval mode), and
embeds only the conversation so far. Measured: 0.17 / 0.71 / 2.4 MB per episode
(lineup / mutual / market), so batch 4,096 needs ~10 GB in the market rung.
It is a switch now (`train.grad_checkpoint`, a run key: same update, tested),
off at the configuration's size, where a run needs under 1 GB.

**Engineering priority 1 -- batch the agents.** Run every agent of a role in one
call: stack their parameters (`torch.func.stack_module_state`) and `vmap` a
`functional_call` over the agent dimension. Pairing is a fixed stride
(`Population.pair`), so every agent has the same number of episodes when the
batch is a multiple of the agent count, and within a rung all agents of a role
share one schema and self-mask. Adam over stacked tensors is per-agent already;
gradient clipping must be done per agent slice; births replace one slice (and
its optimiser state); the bottleneck trains a single module and writes it back;
metrics can keep using per-agent views. This turns ~n_agents calls per symbol
step into one, and is what would make 128 + 128 practical.

**Priority 2 -- a KV cache for generation** (needs a hand-written causal
attention layer instead of `nn.TransformerEncoder`, plus an equivalence test
against the full-sequence forward). Cuts arithmetic, not call count, so do it
after batching.

Until then, communities much larger than the configuration's are slow on a
GPU: check `--benchmark` first.

## 6. Diagnostics that proved their worth

- **Bias-corrected information per field in live messages** (plug-in MI minus a
  shuffled null): `lexicon.live_encoding`, and ad hoc from `lineups.jsonl`.
  This is what exposed the variety-only code.
- **Muted-channel baseline** for every success number. A new lineup generator must
  be checked with "pick the most central candidate": it must score chance.
- **Per-role, per-field numbers** in `promotions.jsonl` (`farmer_fields_intact`,
  `buyer_fields_muted`, `speakers.*.field_coverage`, ...).
- **Snapshots + `orchard.analyse`** to measure without training.
- **Capacity check**: train one speaker/listener pair *supervised* on a fixed
  compositional code (variety atom, quantity atom, quality atom) and test on
  held-out tuples. If the architecture cannot learn it supervised, emergence is
  hopeless; it could (100%).
- Short **rung-only experiments** (a few hundred updates, `--resume` from a
  snapshot) before any full run. Several full runs were lost to problems a
  10-minute experiment would have caught.

## 7. Known risks and ideas, if things stall

- **Code stays variety-only despite hindsight** (it now starts at
  `refer-mutual`): raise `train.hindsight_coef`;
  check the listener's candidate embedding (`CommNet.choice_logits`) actually
  gets gradient for quantity/quality; consider more hard rounds
  (`curriculum.hard_distractor_frac`).
- **Speed with many agents**: see section 5b.
- **Cost of the convention bookkeeping**: ~100-200 ms per 8,192-episode batch on
  CPU Python (`conventions.py`); fine unless the GPU step gets very fast.
- **`haggle` price agreement**: exact price-bin agreement is required
  (`reward.price_tol = 0`). If `haggle` stalls on price mismatches, that is a
  candidate for a further rung rather than loosening the success test silently
  -- ask the user.
- `min_success_over_chance`, `refer_min_success` etc. are unchanged from the
  user-approved values; do not relax criteria to make a run pass.

## 8. Files added or heavily changed this round

`orchard/curriculum.py` (rungs, promotion, hard rounds, holdout, hindsight,
order), `orchard/conventions.py` (coining cost, convention), `orchard/properties.py`
(scorecard, disentanglement, field coverage, duality), `orchard/analyse.py`,
`orchard/train.py` (growth, snapshots, cost gate, evidence), `orchard/gumbel.py`
(hindsight, grammar, grouping), `orchard/env.py` (grammar), `orchard/bottleneck.py`
(phase-aware apprenticeship), `orchard/metrics.py` (phase-aware probes, evidence),
`configs/duality.json`, `cloud_run.sh`, `tests/test_rungs.py`,
`tests/test_config.py` (one configuration; no device-specific arithmetic;
schedules in updates; checkpointing changes nothing; hindsight off while codes
form).

The CPU validation configs (`validate`, `ladder3`) were removed with the rest of
the CPU presets; they are in git history. Their method is the code defaults at
2 + 2 / 6 + 6 with batch 256. Old run folders stay on the Windows machine
(`runs/` is git-ignored); the numbers above are the parts that matter.
