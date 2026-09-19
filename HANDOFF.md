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
python -m unittest discover -s tests -q        # expect: all tests OK (141 at handoff)
python -m orchard.run --config configs/gpu_smoke.json --benchmark      # episodes/sec on CUDA
CONFIG=configs/gpu_smoke.json bash cloud_run.sh                        # minutes; proves the pipeline
```

The GPU code path (bf16 autocast on the transformer layers, TF32, the batched
rollout) has **never been executed on a CUDA device** -- the previous session only
had a 4-core Windows CPU. Every preset was dry-run on CPU on every rung (a
training batch, a newborn apprenticeship, promotion evidence, a snapshot
round-trip), but the first real CUDA run is the first real test of the GPU path.
If anything device-related breaks, that is the likely place.

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
  buffer (32 in the duality preset); length is set by a cost, never a binding
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
| rungs: `refer`, `refer-swap`, `refer-mutual`, `order`, `haggle`, `bargain`, `market` | `curriculum.py` | each adds one thing; `order` (farmer fills the buyer's order with its deal heads) was added because `haggle` needed deal heads nothing had trained |
| per-role promotion in swap/mutual, incl. **field coverage** and **per-field decode** checks | `curriculum.evaluate_rung` | a pooled average let a thin code pass |
| **hard lineup rounds**: an anchor plus one-field near misses, target uniform | `ReferentialWorld._cluster` | random distractors let variety+quality pick the target 69% of the time |
| **held-out tuple combinations** in lineups | `ReferentialWorld.holdout` | the productivity (zero-shot) test |
| **hindsight feedback**: scored heads trained towards the outcome; gradient reaches the speaker through the straight-through channel | `curriculum.hindsight_targets`, `gumbel.py` | without it the code locked into variety only (see section 4) |
| speaker costs (symbol, coining) **off for the first rung**, on after it | `Trainer.update_cost_gate` | costs during rung 1 capped it |
| contrastive, symbol-level **convention bonus** | `conventions.py` | coherence across the community |
| community **founded at 2 + 2**, newcomers join after rung 1 | `Population.add_newcomer`, `Trainer.maybe_grow` | 6 + 6 from scratch never left chance |
| phase-aware speaker order everywhere (costs, bottleneck targets, "me" embedding, probes) | `Phase.own_positions` / `self_mask` | the old buyer-opens order broke farmer newborns (token accuracy 0.000) |
| **snapshots** at every promotion and checkpoint; `--resume`; auto-resume in `cloud_run.sh` | `Trainer.save_snapshot` | iterate on a rung without replaying the ones below; survive pre-emption |
| `python -m orchard.analyse --snapshot ...` | `analyse.py` | full metric suite + properties scorecard on any snapshot |

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

## 5. What is NOT validated yet -- your first job

In order of importance. The first run to do is **`gpu_small`** (16 + 16,
~8M episodes): it is the validation the CPU could not finish.

```bash
CONFIG=configs/gpu_small.json bash cloud_run.sh      # folder: runs/<UTC start>_gpu_small
```

The terminal gets a status line a minute, a two-line headline per checkpoint
(success vs muted, channel headroom, per-role field coverage, coherence, words),
rung transitions and the verdict. `run.log` has the full checkpoint blocks,
`transcripts.txt` has sampled rounds as expected / dialogue / outcome lines, and
`report.md` opens with a summary-statistics table.

What to check, rung by rung (`promotions.jsonl` in the run folder has every check):

- **`refer`**: lift-off in *updates* (the CPU runs took ~550 updates of 256
  episodes; with batch 2048 it may take a different number of episodes). Farmer
  **field coverage per field** must show quantity and quality, not only variety.
  If coverage stays variety-only even with hindsight, that is the next problem to
  solve (see section 7).
- **`refer-swap`**: both roles describe *and* decode; field coverage per role.
- **`refer-mutual`**: per-field decode for quantity and quality must rise above
  the muted baseline. **This is the direct test of hindsight feedback.**
- **community growth**: newcomers join after `refer`; later rungs wait for full size.
- **`order`**: the farmer fills the order exactly (bar 0.50).
- **`haggle`**: channel transfer well above zero, not just success from base rates.
  Price coordination (both sides must pick the same price bin) is the likely next
  bottleneck.
- the word grammar: the report's word-classes row (do separate words specialise
  to separate fields?); share of utterances at the buffer end (should be ~0).

Then `gpu_community` (48 + 48), then `gpu_full` (128 + 128) or `gpu_duality`
(12 varieties vs 8 atoms: the only preset where duality of patterning is
*necessary*; unvalidated).

## 5b. Performance: read this before the big presets

`python -m orchard.run --config configs/<preset>.json --benchmark` now times three
rungs (one-turn lineup, two-turn mutual, full market) at **full community
size** and estimates hours. Run it on the GPU for `gpu_small` and
`gpu_community` before committing to either.

Where the time goes (profiled on CPU, 8 + 8 agents, 64 episodes, market rung):
**784 separate agent forward passes per training step** -- one per agent per
symbol step (4 turns x 24 symbols x 8 agents) plus the decisions -- each
re-encoding the whole conversation so far, then a backward pass through all of
them (6 s forward, 8 s backward on CPU). The 24-symbol buffer the user asked for
makes generation 6x longer than the old 4-symbol cap; that is the right
trade, but it makes this loop the bottleneck. On a GPU the cost is dominated by
the *number of calls* (kernel launches), not arithmetic, so it grows linearly
with agents per role.

**Memory (already fixed, keep it that way).** The first real GPU run
(`gpu_community`, RTX 4090, 24 GB) ran out of memory in the first training
batch: every symbol step re-encodes the conversation and the Gumbel path
backpropagates through all of them, so saved activations were ~14 MB per
episode in the lineup and ~90 MB in the market (56-365 GB at batch 4,096). Now
`CommNet.encode` checkpoints embedding + layers + final norm, keyed on grad
mode (not train mode -- newborns leave their apprenticeship in eval mode), and
embeds only the conversation so far. Measured: 0.17 / 0.71 / 2.4 MB per episode
(lineup / mutual / market), so batch 4,096 needs ~10 GB in the market rung. If
you change the model or buffer, re-measure before raising batch sizes (count
saved-tensor bytes with `torch.autograd.graph.saved_tensors_hooks` over one
training step).

**Engineering priority 1 -- batch the agents.** Run every agent of a role in one
call: stack their parameters (`torch.func.stack_module_state`) and `vmap` a
`functional_call` over the agent dimension. Pairing is a fixed stride
(`Population.pair`), so every agent has the same number of episodes when the
batch is a multiple of the agent count, and within a rung all agents of a role
share one schema and self-mask. Adam over stacked tensors is per-agent already;
gradient clipping must be done per agent slice; births replace one slice (and
its optimiser state); the bottleneck trains a single module and writes it back;
metrics can keep using per-agent views. This turns ~n_agents calls per symbol
step into one, and is what makes `gpu_full` (128 + 128) practical.

**Priority 2 -- a KV cache for generation** (needs a hand-written causal
attention layer instead of `nn.TransformerEncoder`, plus an equivalence test
against the full-sequence forward). Cuts arithmetic, not call count, so do it
after batching.

Until then: `gpu_small` (16 + 16) should be comfortable; check the benchmark
estimate for `gpu_community`; treat `gpu_full` as blocked on priority 1.

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

- **Code stays variety-only despite hindsight**: raise `train.hindsight_coef`;
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
`configs/gpu_*.json`, `cloud_run.sh`, `tests/test_rungs.py`.

`configs/validate.json` is the 2 + 2 CPU validation config; `configs/ladder3.json`
is the growing-community CPU run. Old run folders stay on the Windows machine
(`runs/` is git-ignored); the numbers above are the parts that matter.
