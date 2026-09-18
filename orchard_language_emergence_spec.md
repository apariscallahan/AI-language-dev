# Project: Emergent Language in an Apple-Trading World

## Overview

Build a Python simulation in which two populations of small neural agents — **Farmers** (who grow and sell apples) and **Buyers** (who purchase apples for a household) — must communicate to trade. Neither population starts with any language. All agents begin with randomly initialized, untrained neural network policies and a discrete, meaningless token vocabulary. Communication has to be invented from scratch through repeated interaction, reinforcement learning, population turnover across generations, and a transmission bottleneck between generations. The research goal is to observe whether a compositional, stable "language" (recognizable structure mapping tokens to goods, quantities, qualities, and trade-offers) emerges under these constraints, and to produce hard output logging every trade and every utterance so this can be inspected afterward.

This is a research/simulation project, not a product. Prioritize correctness, inspectability, and good logging over performance or polish. Build it incrementally and get each layer working and *tested* before adding the next.

**Critical constraint: do not use any pretrained language model, pretrained embeddings, or any text corpus anywhere in this project.** All agent "brains" are small transformer or RNN policies with randomly initialized weights, trained only via reinforcement learning on interactions inside the simulated world. If at any point a component seems to need real-world language data, stop and flag it — that indicates a design problem, not a shortcut to take.

---

## 1. The World

### 1.1 Setting

A simple discrete-time, discrete-event market simulation (no spatial grid needed — this isn't a physics/movement problem, it's a negotiation problem). Each "day" (timestep), a subset of Farmers and Buyers are paired up or placed in a shared marketplace and must communicate to complete (or fail to complete) trades.

### 1.2 Goods and private information (the reason language is necessary)

Language is only necessary if there is information one agent has that another needs, and it can't be inferred from observation alone. Build this in explicitly:

- **Apple varieties**: e.g. 3-5 discrete types (`RED`, `GREEN`, `GOLD`, ...). Farmers know which varieties they have in stock and in what quantity; Buyers do not observe this directly and must ask/discover it through communication.
- **Quality/ripeness**: a hidden scalar or small discrete range (e.g. `LOW/MED/HIGH`) known only to the Farmer for their current stock. This cannot be observed by the Buyer at all — it can only be communicated (truthfully or not).
- **Quantity available**: a discrete count known only to the Farmer.
- **Buyer's need**: each Buyer has a private target (which variety, how many, budget ceiling, quality preference) known only to them. The Farmer does not observe this.
- **Price**: not fixed by the world. It must be proposed and agreed via the communication channel. This is what forces negotiation (offer/counter-offer/accept/reject), not just labeling.

This asymmetry is the core design requirement: **every trade requires an exchange of information neither party can get by observing the world directly.** Do not let agents "cheat" by observing hidden state directly — enforce this in code via strict separation of each agent's observation dict.

### 1.3 Trade resolution mechanics

- Each trading episode: Farmer and Buyer are paired, each gets their private observation, they exchange a bounded number of message turns (e.g. up to 6-10 alternating turns), then each simultaneously outputs a trade decision (accept/reject, and if accept, what quantity/price they believe was agreed).
- A trade **succeeds** only if both agents' final understanding of quantity and price actually match (or fall within tolerance) and the Farmer actually has that stock and the Buyer can actually afford it. This is important: success must depend on *mutual understanding*, not on one agent guessing right. If it's checkable by the code without any true information transfer, the task doesn't require language, and agents will find a shortcut around communicating.
- Track and reward both parties: Farmers want successful profitable trades (sell inventory at a good price), Buyers want successful trades that meet their needs within budget. Both are penalized for wasted turns / failed trades / miscommunication (e.g., a trade that resolves inconsistently — one thinks they agreed on 5 apples, the other thinks 3).

### 1.4 Economy loop

- Farmers replenish stock each "season" (some number of days) with fresh random quantity/quality/variety.
- Buyers get a fresh private need each time they enter a new trade (or each day).
- This is what makes it a repeated game across many episodes rather than one-shot — agents need a language that generalizes across many different quantities/varieties/prices, not memorized responses to fixed situations. **This generalization requirement is what will actually force compositionality** — a lookup-table code that maps specific whole scenarios to specific messages will fail to generalize to new quantity/price/variety combinations it hasn't seen, while a compositional code (separate tokens for "variety," "quantity," "price," "offer," "accept") will generalize. Make sure the space of possible quantities/prices/varieties is large enough that memorization is actually infeasible (e.g., quantities 1-20, prices as continuous-ish values discretized into a reasonably fine grid, several varieties) — this pressure is essential, not optional.

---

## 2. Agents

### 2.1 Architecture

- Each agent (Farmer or Buyer) is a small transformer (or GRU/LSTM if transformer proves fiddly to get training — start with whichever the implementer is more confident training quickly, but transformer is preferred if time allows since it's the more "LLM-like" architecture requested) with:
  - An input embedding for: private observation features (goods/quantities/quality/need/budget as embedded discrete/continuous features), plus embedded incoming message tokens, plus a role embedding (Farmer vs Buyer), plus an age/generation embedding is NOT needed in the input (age affects training/lifecycle, not perception).
  - An output head producing a probability distribution over the next message token to emit (from a small fixed vocabulary, see 2.2), AND a separate output head for the final trade decision (accept/reject + quantity + price, discretized into bins for tractable RL).
- Randomly initialized at birth. No pretraining, no weight sharing with a "foundation model." Each agent's weights are its entire linguistic and strategic knowledge, learned only from RL on lived interactions plus (for newly created agents) the imitation-learning bottleneck described in Section 4.

### 2.2 Communication channel (the bottleneck)

This is one of the most important design constraints — get this right:

- **Discrete vocabulary**, small (e.g., 20-40 tokens total, plus an end-of-message token). Do NOT allow continuous/real-valued message vectors — that lets agents cheat with an infinite-bandwidth channel and produces no language-like compression pressure.
- **Bounded message length** per turn (e.g., max 4-8 tokens per turn).
- **Bounded number of turns** per trade negotiation (e.g., 6-10 total alternating turns).
- Tokens have **no pre-assigned meaning**. They are just integer ids into an embedding table at the start. Any meaning is emergent from training, not assigned by the implementer. Do not hand-design a "price token" or "accept token" — that would defeat the purpose.

### 2.3 Training algorithm

- Use policy-gradient RL since message-token selection and the final trade decision are discrete, non-differentiable actions. Recommended: **REINFORCE with a baseline** (simplest to implement and debug) or PPO if the implementer wants more stability and has an RL library available.
- Reward each agent based on trade outcome (Section 1.3) at the end of each episode; credit all message-emitting steps in that episode via the return (standard episodic RL — no need for step-wise reward shaping unless training is unstable, in which case a small per-turn cost for using more turns than necessary is reasonable, to discourage degenerate long babbling).
- Use a library: prefer **CleanRL**-style single-file, readable implementations over heavyweight RL frameworks, since this code will need to be read, modified, and debugged a lot. PyTorch is the expected framework.
- If using Gumbel-softmax relaxation instead of REINFORCE for the message tokens, that's acceptable and may train faster/more stably — implementer's choice, but document which was used and why in code comments.

---

## 3. Population, Generations, Birth and Death

This is a required feature, not optional polish — it is one of the two mechanisms (with Section 4) most responsible for producing systematic, generalizable language rather than an idiosyncratic private code between two fixed agents.

- Maintain a population of, e.g., 8-20 Farmers and 8-20 Buyers at any time (tune based on compute budget).
- Each agent has an **age** (in episodes or in days) and a **max lifespan** (randomized within a range, so deaths are staggered, not synchronized).
- When an agent reaches its lifespan, it **dies** (is removed from the population) and is replaced by a **newly born agent** with freshly randomly initialized weights.
- Track **generation number** per agent (increment when an agent's "lineage slot" in the population turns over — doesn't need literal parent-child weight inheritance, since weights are not inherited; generation here means population-turnover count, not genetic descent).
- Stagger deaths/births across the population (don't reset everyone at once) so that at any time there is a mix of "veteran" agents who already know the current language and "newborn" agents who must learn it — this generational overlap is what creates the pressure toward a learnable, teachable code (an idiosyncratic code that only works between two specific co-adapted agents will fail to transmit to newcomers, so it should be selected against over time as veterans who can't communicate with newcomers "lose" trades and thus get less RL reward, an implicit selection pressure toward transmissibility).

---

## 4. The Transmission Bottleneck (Iterated Learning)

This is the second required mechanism and, per the design discussion, likely the single most important one for producing compressible, systematic (i.e., grammar-like) structure rather than noise. Implement this explicitly, don't skip it:

- When a new agent is born, before (or interleaved with) letting it loose to freely interact and RL-train against the live population, give it a **supervised pretraining phase** on a *sampled, limited* set of transcripts (observation -> message sequences, and message sequences -> trade decisions) drawn from recent successful trades among the current living population.
- Critically, this sample should be **limited**, not the full history — e.g., a few hundred to low-thousands of (input, message) pairs, not every trade ever conducted. The limitation is what forces the newborn to generalize/compress rather than memorize a huge lookup table, which is the mechanism believed to produce systematic structure in iterated-learning models of language evolution (Kirby et al.).
- Implement this as straightforward supervised learning (cross-entropy loss against the sampled transcripts) on the newborn's message-output head and decision head, for some fixed number of steps/epochs, before switching the newborn into the standard RL loop with the live population.
- Log, per newborn, what it was trained on (which generation's transcripts, how many samples) for later analysis.

---

## 5. Metrics and Analysis (build this alongside the simulation, not after)

Without these, you cannot tell "coherent emergent grammar" from "noise that happens to correlate with outcomes." Implement and log all of the following over time (per generation / per N episodes):

1. **Task success rate**: fraction of trade episodes that end in a mutually-consistent successful trade. Should rise over training if anything sensible is happening.
2. **Compositionality (topological similarity)**: for a sample of (meaning, message) pairs at a point in time — where "meaning" is the structured tuple (variety, quantity, quality, price-offer, etc.) — compute the correlation between pairwise distance in meaning-space (e.g. edit distance over the tuple, or hamming distance over discretized fields) and pairwise distance in message-space (e.g. edit distance over token sequences). High positive correlation = compositional structure (similar meanings get similar messages). This is the standard metric from Brighton & Kirby; implement it directly, it's not complicated (just two distance matrices and a Spearman correlation).
3. **Vocabulary usage stats**: token frequency distribution, entropy of message distribution, average message length actually used (vs. max allowed) — degenerate codes often show very low entropy or maximal-length babbling; watch for these failure modes explicitly.
4. **Stability over time**: does the mapping from a given meaning to a message stay consistent across a generation's lifetime, or drift chaotically episode to episode? Track by re-sampling the same synthetic "meaning" against the same agent at different times and measuring message consistency.
5. **Cross-generation intelligibility**: pick a newborn agent right after its bottleneck training and test it against transcripts/live veteran agents from before it was born — can it "understand" (achieve successful trades) messages from agents it never directly trained against live? This tests whether the language is actually a shared code or a pairwise-idiosyncratic one.
6. **Zero-shot generalization**: test trained agents on quantity/price/variety combinations that were rare or absent in training, and see if trade success holds up — this is the direct test of whether compositional generalization (vs. memorization) occurred.

Compute and log all of these at regular checkpoints (e.g., every N episodes and at every generational turnover), not just at the end.

---

## 6. Required Output / Logging

The user specifically wants visibility into what's happening at each generation and a final language summary. Implement:

### 6.1 Per-episode trade ledger (the core requested output)
A structured log (recommend: append to a CSV or JSONL file, one row per completed trade episode) recording at minimum:
- timestamp/episode number, day/season number
- Farmer id, age, generation; Buyer id, age, generation
- true hidden state: variety, quantity available, quality, buyer's true need/budget
- the full sequence of exchanged messages (raw token ids AND a human-readable rendering — see 6.3)
- final decisions each party made (accept/reject, believed quantity, believed price)
- whether the trade succeeded, and if not, why (mismatch in quantity understanding, price understanding, insufficient stock, over budget, etc. — classify the failure mode)
- resulting profit/reward for each party

This is the ledger of "all the apples bought and sold in the world and what the agents spoke while doing it," as requested — make sure it's complete enough to reconstruct any trade after the fact.

### 6.2 Per-generation / per-checkpoint summary output
At regular intervals (e.g. every generation turnover, and every fixed number of episodes), print/log a human-readable summary to console and to a log file:
- current population composition (ages, generations, counts)
- success rate trend
- current compositionality score and how it's changed
- vocabulary entropy / usage stats
- a handful of example transcripts from that period, printed in readable form (see 6.3), so a human watching the run can see the "language" evolving in real time
- flag explicitly if a degenerate outcome is detected (e.g., success rate stuck at chance, entropy near zero, no compositional structure after substantial training) — don't let a failing run silently look fine

### 6.3 Human-readable message rendering
Raw token ids (`[3, 17, 2, 8]`) are hard to read. Implement a rendering helper that:
- Assigns each token id a placeholder label like `tok3`, `tok17`, etc. for now (since meanings are emergent, not designed) — do NOT hand-assign semantic labels like "price" or "accept," since that would be imposing meaning rather than discovering it. Labels should be updated later via analysis, not asserted in advance.
- Optionally, once compositionality analysis has run (Section 5.2) and found which tokens/positions correlate with which meaning-dimensions, generate a *post-hoc* "best guess" annotated rendering (e.g., `tok17(likely:variety=RED) tok3(likely:qty~5)`) for the final report — but clearly label this as an inferred interpretation from the analysis, not ground truth, and keep the raw token log as the authoritative record.

### 6.4 Final "language" report
At the end of a full run, produce a final summary document/file containing:
- final compositionality/stability/generalization metrics
- the inferred "dictionary" — most frequent tokens and, per Section 6.3's post-hoc analysis, what meaning dimension each seems to correlate with and how strongly
- a curated set of example transcripts across the run's history (early/random-looking, middle, late/converged) so the emergence process is visible, not just the endpoint
- total trade volume/economic stats (total apples sold, by whom, total value) as requested
- an honest assessment section: does this run show evidence of compositional emergent language, or a degenerate/non-compositional code? State this plainly based on the metrics, don't oversell an ambiguous result.

---

## 7. Suggested Build Order (get each step working before the next)

1. **Environment core**: implement the trading episode mechanics (private observations, message exchange loop, trade resolution/success check) with two hand-scripted dummy agents (random or fixed-rule) just to verify the environment logic and logging work, before any learning is involved.
2. **Single fixed pair + RL**: one Farmer, one Buyer, train with RL on repeated episodes, verify success rate rises above chance and *some* consistent signaling emerges (even if trivial/degenerate at this stage — that's expected and fine).
3. **Compositionality metric**: implement and sanity-check the topological similarity metric on this simple case (it should be low/near-random at this stage — this confirms the metric works, since two co-adapted agents with no population pressure are exactly the case expected to produce a non-compositional idiosyncratic code).
4. **Small static population**: expand to multiple Farmers/Buyers, randomly paired each episode, no births/deaths yet. Re-check metrics.
5. **Add birth/death/aging**, no bottleneck training yet (newborns start from scratch and must learn purely via live RL against the population). Observe what happens — this alone may or may not help; log it either way.
6. **Add the transmission bottleneck** (Section 4) for newborns. Compare metrics before/after adding this — this comparison IS the experiment, so make sure both configurations are easy to run and log separately for comparison.
7. **Scale up** (more agents, more varieties, longer runs) only once the pipeline and metrics are validated at small scale.

Implement configuration (population size, vocab size, message length, lifespan ranges, bottleneck sample size, number of varieties, etc.) via a config file or command-line args, not hardcoded — the comparison in step 6 requires running the same codebase with different settings.

---

## 8. Tech Stack Recommendations

- **Python 3.10+, PyTorch** for the agent networks and RL.
- Simple custom environment (no need for Gym/PettingZoo formality, but structuring it with a similar step/reset API is fine and may ease debugging).
- **CSV or JSONL** for the trade ledger (simplest to inspect and to load into pandas for analysis).
- **Matplotlib** for plotting success rate, compositionality score, entropy, etc. over time — save these as PNG files at checkpoints so progress is visible without needing to re-run analysis.
- Keep the whole thing in a single reasonably organized repo/directory with clear module separation (environment / agents / training loop / population manager / metrics / logging) rather than one giant script — this will need to be modified and re-run many times.

---

## 9. What "done" looks like

A run that produces:
- A full trade ledger (CSV/JSONL) of every trade attempted across the run, with full message transcripts.
- Console/log output showing per-generation summaries as the run progresses (not just a silent run producing one file at the end).
- Plots of success rate, compositionality, and vocabulary entropy over the run.
- A final written report (Section 6.4) with an honest, metric-backed assessment of whether language-like structure emerged, including example transcripts a human can read.
- A clean way to re-run with the transmission-bottleneck mechanism toggled on/off and population-turnover toggled on/off, to compare their effects — since that comparison is the actual scientific point of the project.
