"""The final language report (spec 6.4).

Writes ``report.md``: final metrics, the *inferred* dictionary, curated
transcripts from early/middle/late in the run, the economic totals, and an
honest verdict.

The verdict is computed from thresholds, not written by hand, so a mediocre run
cannot be talked up.  :func:`assess` returns one of four judgements and the
evidence for it.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

from .config import Config
from .metrics import TokenSemantics
from .render import render_tokens


# --------------------------------------------------------------------------
def assess(cfg: Config, final: dict[str, Any], chance: float) -> dict[str, Any]:
    """Grade the run against fixed thresholds.  No hedging, no cherry-picking."""
    succ = final.get("eval_success", float("nan"))
    comp = final.get("compositionality", {})
    topsim = comp.get("mean", float("nan"))
    null = comp.get("buyer", {}).get("null_mean", 0.0) if isinstance(comp.get("buyer"), dict) else 0.0
    vocab = final.get("vocab", {})
    stab = final.get("stability", {})
    zs = final.get("zero_shot", {})
    intel = final.get("intelligibility", {})

    ev: list[str] = []
    checks: dict[str, bool] = {}

    def nn(x, d=float("nan")):
        return x if isinstance(x, (int, float)) and x == x else d

    succ = nn(succ)
    topsim = nn(topsim)
    coh = (nn(stab.get("coherence_buyer"), 0.0) + nn(stab.get("coherence_farmer"), 0.0)) / 2
    retention = nn(zs.get("retention"))
    transmission = nn(intel.get("transmission_ratio"))
    ent_norm = nn(vocab.get("token_entropy_norm"), 0.0)

    abl = final.get("channel_ablation", {}) or {}
    abl_drop = nn(abl.get("comprehension_drop"))
    abl_rel = nn(abl.get("relative_comprehension_loss"))
    transfer = nn(abl.get("information_transfer"))
    words = final.get("words", {}) or {}
    lenfreq = final.get("length_frequency", {}) or {}
    rho_len = nn(lenfreq.get("rho_symbols"))
    multi = nn(words.get("multi_atom_word_share"), 0.0)
    n_words = int(words.get("distinct_words", 0) or 0)

    var_transfer = nn(abl.get("variety_transfer"))
    checks["named_things"] = var_transfer == var_transfer and var_transfer > 0.2
    checks["learned_to_trade"] = succ == succ and succ > max(3 * chance, chance + 0.05)
    # Measured against silence, and only meaningful once there is some comprehension
    # to lose: the relative figure is 0/0 when nothing is understood either way.
    checks["channel_carries_information"] = (
        transfer == transfer and transfer > 0.15
        and nn(abl.get("intact_comprehension"), 0.0) > 0.01)
    checks["well_above_chance"] = succ == succ and succ > max(6 * chance, 0.15)
    checks["channel_alive"] = ent_norm > 0.15 and int(vocab.get("tokens_used", 0)) >= 3
    checks["compositional"] = topsim == topsim and topsim > 0.20 and topsim > null + 0.10
    checks["weakly_compositional"] = topsim == topsim and topsim > 0.08 and topsim > null + 0.05
    checks["stable"] = nn(stab.get("identical_frac"), 0.0) > 0.5
    checks["shared_code"] = coh > 0.5
    checks["generalises"] = retention == retention and retention > 0.8
    checks["transmits"] = transmission == transmission and transmission > 0.7
    # addendum checks
    checks["open_vocabulary_used"] = n_words >= 8
    checks["multi_atom_words_formed"] = multi > 0.05
    checks["zipfian_length"] = rho_len == rho_len and rho_len < -0.15
    checks["not_babbling"] = nn(words.get("at_length_cap_frac"), 1.0) < 0.5

    if chance > 1e-6 and succ == succ:
        ev.append("final success rate %.3f against a chance baseline of %.4f (%.0fx)"
                  % (succ, chance, succ / chance))
    else:
        ev.append("final success rate %.3f; the chance baseline is indistinguishable "
                  "from zero (%d random-play successes measured)"
                  % (succ, int(round(chance * 4000))))
    ev.append("topological similarity %.3f against a shuffled null of %.3f" % (topsim, null))
    ev.append("%d of %d tokens in use, normalised entropy %.2f"
              % (int(vocab.get("tokens_used", 0)), int(vocab.get("vocab_size", 0)), ent_norm))
    ev.append("population coherence %.3f (1.0 = every agent says the same thing "
              "for the same meaning)" % coh)
    if retention == retention:
        ev.append("zero-shot retention %.2f on held-out (variety, quantity) combinations"
                  % retention)
    if transmission == transmission:
        ev.append("cross-generation transmission ratio %.2f" % transmission)

    intact_comp = nn(abl.get("intact_comprehension"))
    if intact_comp == intact_comp and intact_comp < 0.01:
        ev.append("the channel ablation is uninformative here: comprehension is %.3f "
                  "even with the channel intact, so there is nothing for muting it to "
                  "take away" % intact_comp)
    elif abl_drop == abl_drop:
        ev.append("muting the channel costs %.3f of comprehension, %.0f%% of the "
                  "headroom above silence -- this is the causal test that the messages "
                  "carry information"
                  % (abl_drop, 100 * (transfer if transfer == transfer else 0.0)))
    if var_transfer == var_transfer:
        content = nn(abl.get("variety_transfer_content"), 0.0)
        ev.append("the farmer names the buyer's variety at %.3f, and at %.3f when the "
                  "other party is muted: the channel accounts for %.0f%% of the "
                  "headroom above silence, and %.0f%% of it still disappears if the "
                  "symbols are scrambled but the utterance length is kept"
                  % (nn(abl.get("intact_variety_acc"), 0.0),
                     nn(abl.get("muted_variety_acc"), 0.0),
                     100 * var_transfer, 100 * content))
    if n_words:
        ev.append("%d distinct words in use, %.0f%% of them multi-atom compounds, "
                  "%.2f words per utterance"
                  % (n_words, 100 * multi, nn(words.get("mean_words_per_message"), 0.0)))
    if rho_len == rho_len:
        ev.append("length/frequency correlation %.3f (negative means commoner "
                  "meanings got shorter forms, the human pattern)" % rho_len)

    if not checks["learned_to_trade"] and not checks["named_things"]:
        verdict = "NO EMERGENCE"
        summary = ("Trade success never rose meaningfully above the random-play baseline. "
                   "Whatever the agents are emitting, it is not carrying information that "
                   "improves outcomes, so there is no language here to analyse.")
    elif not checks["channel_alive"] or (transfer == transfer and transfer < 0.05
                                        and not checks["named_things"]):
        verdict = "DEGENERATE CODE"
        summary = ("Agents beat chance, but scrambling the channel barely hurts them "
                   "and/or the vocabulary has collapsed to a near-constant signal. "
                   "The performance is coming from base rates and reward shaping, not "
                   "from communication.")
    elif checks["compositional"] and checks["well_above_chance"] and checks["shared_code"]:
        verdict = "COMPOSITIONAL LANGUAGE"
        summary = ("The run shows the signature of a compositional, shared code: similar "
                   "meanings get similar messages well above the shuffled null, different "
                   "agents converge on the same forms, and trade success is far above "
                   "chance.")
    elif checks["weakly_compositional"] and checks["well_above_chance"]:
        verdict = "PARTIALLY COMPOSITIONAL"
        summary = ("A working communication system emerged and there is measurable but "
                   "modest structure in it. The mapping is more systematic than chance "
                   "but well short of a cleanly compositional grammar; read the "
                   "per-position analysis before claiming syntax.")
    else:
        verdict = "NON-COMPOSITIONAL SIGNALLING"
        summary = ("Agents learned to communicate enough to beat chance, but the "
                   "meaning-to-message mapping shows no more structure than a shuffled "
                   "baseline. This is holistic signalling -- effectively a private code "
                   "with no reusable parts -- not a grammar.")

    return {"verdict": verdict, "summary": summary, "checks": checks, "evidence": ev}


# --------------------------------------------------------------------------
def _dictionary_lines(cfg: Config, sem: TokenSemantics, vocab: dict[str, Any]) -> list[str]:
    counts = vocab.get("token_counts", {}) or {}
    total = sum(int(v) for v in counts.values()) or 1
    rows = sorted(counts.items(), key=lambda kv: -int(kv[1]))
    lines = ["| atom | share of atom use | most associated field | typical value | strength (normalised MI) |",
             "|---|---|---|---|---|"]
    for tok, n in rows[:24]:
        t = int(tok)
        rec = sem.per_token.get(t)
        if rec:
            lines.append("| `a%d` | %.1f%% | %s | %s | %.3f |"
                         % (t, 100 * int(n) / total, rec["dimension"], rec["typical"],
                            rec["score"]))
        else:
            lines.append("| `a%d` | %.1f%% | (no clear association) | - | - |"
                         % (t, 100 * int(n) / total))
    return lines


def _position_lines(sem: TokenSemantics) -> list[str]:
    lines = ["| speaker | message slot | dimension most predicted by this slot | strength | distinct tokens seen |",
             "|---|---|---|---|---|"]
    for role, rows in sem.per_position.items():
        for r in rows:
            lines.append("| %s | %d | %s | %.3f | %d |"
                         % (role, r["position"], r["dimension"], r["score"],
                            r["distinct_tokens"]))
    return lines


def write_report(cfg: Config, out_dir: str, *, final: dict[str, Any],
                 chance: float, sem: TokenSemantics, archive: Sequence[dict[str, Any]],
                 totals: dict[str, Any], history: dict[str, Any],
                 newborn_reports: Sequence[dict[str, Any]],
                 ledger_path: str, wall_minutes: float) -> str:
    verdict = assess(cfg, final, chance)
    vocab = final.get("vocab", {})
    # vocab in the metrics row drops token_counts; recover from the semantics pass if absent
    if "token_counts" not in vocab:
        vocab = dict(vocab)
        vocab["token_counts"] = final.get("token_counts", {})

    L: list[str] = []
    A = L.append
    A("# Orchard: did a language emerge?")
    A("")
    A("Run `%s` -- %d episodes, %.1f wall-clock minutes."
      % (cfg.name, final.get("episode", 0), wall_minutes))
    A("")
    A("> **Verdict: %s**" % verdict["verdict"])
    A(">")
    A("> " + verdict["summary"])
    A("")

    A("## 1. Honest assessment")
    A("")
    A("Evidence the verdict rests on:")
    A("")
    for e in verdict["evidence"]:
        A("- " + e)
    A("")
    A("Threshold checks:")
    A("")
    A("| check | passed |")
    A("|---|---|")
    for k, v in verdict["checks"].items():
        A("| %s | %s |" % (k.replace("_", " "), "yes" if v else "**no**"))
    A("")
    flags = final.get("degenerate_flags") or []
    if flags:
        A("**Degenerate-outcome warnings raised at the final checkpoint:**")
        A("")
        for f in flags:
            A("- " + f)
        A("")
    else:
        A("No degenerate-outcome warnings were raised at the final checkpoint.")
        A("")

    A("## 2. Final metrics")
    A("")
    comp = final.get("compositionality", {})
    stab = final.get("stability", {})
    zs = final.get("zero_shot", {})
    intel = final.get("intelligibility", {})

    def g(d, k, default=float("nan")):
        v = d.get(k, default) if isinstance(d, dict) else default
        return v if isinstance(v, (int, float)) else default

    A("| metric | value | reading |")
    A("|---|---|---|")
    A("| task success rate (5.1) | %.3f | chance is %.4f |"
      % (g(final, "eval_success"), chance))
    A("| farmer names the right variety | %.3f | a fact only the buyer holds; "
      "chance %.3f |" % (g(final, "farmer_variety_acc"), 1.0 / cfg.world.n_varieties))
    A("| farmer names the right quantity | %.3f | likewise; chance %.3f |"
      % (g(final, "farmer_qty_acc"), 1.0 / cfg.world.max_qty))
    A("| success on viable deals | %.3f | deals that were actually possible |"
      % g(final, "eval_success_on_viable"))
    A("| topological similarity (5.2) | %.3f | shuffled null %.3f |"
      % (g(comp, "mean"), g(comp.get("buyer", {}), "null_mean", 0.0)))
    A("| tokens in use (5.3) | %d / %d | entropy %.2f bits |"
      % (int(g(vocab, "tokens_used", 0)), int(g(vocab, "vocab_size", 0)),
         g(vocab, "token_entropy_bits", 0.0)))
    A("| mean message length | %.2f / %d | shorter than the cap means real compression |"
      % (g(vocab, "mean_msg_len", 0.0), int(g(vocab, "max_msg_len", 0))))
    A("| message stability (5.4) | %.0f%% unchanged | drift %.3f between checkpoints |"
      % (100 * g(stab, "identical_frac", 0.0), g(stab, "drift")))
    A("| population coherence | %.3f | do different agents share the code? |"
      % ((g(stab, "coherence_buyer", 0.0) + g(stab, "coherence_farmer", 0.0)) / 2))
    A("| cross-generation (5.5) | %.2f | newcomer success / veteran success |"
      % g(intel, "transmission_ratio"))
    A("| zero-shot (5.6) | %.2f | held-out success / seen success |"
      % g(zs, "retention"))
    abl = final.get("channel_ablation", {}) or {}
    if abl.get("n"):
        A("| comprehension, channel intact | %.3f | the causal control below |"
          % g(abl, "intact_comprehension"))
        A("| comprehension, channel scrambled | %.3f | same message shape, random "
          "content |" % g(abl, "scrambled_comprehension"))
        A("| comprehension, channel muted | %.3f | the honest baseline: the other "
          "party heard as silent |" % g(abl, "muted_comprehension"))
        A("| information transfer | %.0f%% | share of the headroom above silence that "
          "the channel accounts for |" % (100 * g(abl, "information_transfer", 0.0)))
    A("")

    A("## 3. The inferred dictionary")
    A("")
    A("These associations are **inferred after the fact** by correlating emitted tokens "
      "with the speaker's private state (normalised mutual information). They are not "
      "ground truth and were never given to the agents: no token was designed to mean "
      "anything. The raw token ids in the ledger remain the authoritative record.")
    A("")
    L.extend(_dictionary_lines(cfg, sem, vocab))
    A("")
    A("### Positional structure")
    A("")
    A("If a language is compositional, different slots of an utterance should carry "
      "different fields. This table asks, for each symbol slot, which field the "
      "symbol in that slot predicts best.")
    A("")
    L.extend(_position_lines(sem))
    A("")

    A("## 3b. The vocabulary that emerged")
    A("")
    words = final.get("words", {}) or {}
    if words:
        A("| property | value | what it means |")
        A("|---|---|---|")
        A("| distinct words | %d | out of %d possible one-atom words alone |"
          % (int(g(words, "distinct_words", 0)), cfg.channel.atomic_vocab))
        A("| word entropy | %.2f bits | low means a couple of forms dominate |"
          % g(words, "word_entropy_bits", 0.0))
        A("| atoms per word | %.2f (longest %d) | above 1 means the hyphen is doing work |"
          % (g(words, "mean_word_len_atoms", 0.0), int(g(words, "max_word_len_atoms", 0))))
        A("| multi-atom words | %.0f%% of word tokens | compounds, not bare atoms |"
          % (100 * g(words, "multi_atom_word_share", 0.0)))
        A("| words per utterance | %.2f | above 1 means the space is doing work |"
          % g(words, "mean_words_per_message", 0.0))
        A("| symbols per utterance | %.2f of %d allowed | the cost, not the cap, should "
          "be what limits this |"
          % (g(words, "mean_symbols_per_message", 0.0),
             int(g(words, "max_symbols_allowed", 0))))
        A("| utterances at the cap | %.0f%% | high means length-cap babbling |"
          % (100 * g(words, "at_length_cap_frac", 0.0)))
        A("")
        top = words.get("top_words") or []
        if top:
            A("Commonest words (placeholder names; `a7-a3` is atom 7 hyphenated to atom 3):")
            A("")
            A("| word | share of word tokens | atoms |")
            A("|---|---|---|")
            for w in top[:12]:
                A("| `%s` | %.1f%% | %d |" % (w["word"], 100 * w["share"], len(w["atoms"])))
            A("")
        wsem = getattr(sem, "per_word", {}) or {}
        if wsem:
            A("Inferred word associations (post-hoc, hedged, never given to the agents):")
            A("")
            A("| word | most associated field | typical value | strength | used in |")
            A("|---|---|---|---|---|")
            for k, rec in sorted(wsem.items(), key=lambda kv: -kv[1]["score"])[:12]:
                A("| `%s` | %s | %s | %.3f | %.0f%% of messages |"
                  % (k, rec["dimension"], rec["typical"], rec["score"],
                     100 * rec.get("usage", 0.0)))
            A("")

    A("## 3c. Do common meanings get short words?")
    A("")
    A("The addendum predicts the human pattern: a meaning that comes up constantly "
      "should end up with a short form, because its length cost is paid over and "
      "over. Nothing in the code rewards this directly -- it is a prediction about "
      "what falls out of a per-symbol cost in a world where some requests are far "
      "commoner than others.")
    A("")
    lf = final.get("length_frequency", {}) or {}
    if lf.get("n"):
        A("| measure | value |")
        A("|---|---|")
        A("| Spearman rho, log-frequency vs symbols | %.3f |" % g(lf, "rho_symbols"))
        A("| Spearman rho, log-frequency vs word count | %.3f |" % g(lf, "rho_words"))
        A("| mean symbols, commonest third of meanings | %.2f |"
          % g(lf, "mean_symbols_frequent"))
        A("| mean symbols, rarest third of meanings | %.2f |" % g(lf, "mean_symbols_rare"))
        A("")
        rows = final.get("length_frequency_rows") or []
        if rows:
            A("| meaning (variety, quantity) | how often | form used | symbols | words |")
            A("|---|---|---|---|---|")
            for r in rows[:14]:
                A("| %s | %.4f | `%s` | %d | %d |"
                  % (r["meaning"], r["prob"], r.get("form") or "<silence>",
                     r["symbols"], r["words"]))
            A("")
    else:
        A("Not enough distinct meanings to measure.")
        A("")

    A("## 3d. Frequent versus rare meanings")
    A("")
    A("A global average hides the effect worth looking for, so these are split. "
      "The prediction is that frequent meanings settle into short, stable, possibly "
      "irregular forms while rare ones stay longer, more volatile and more "
      "compositional.")
    A("")
    bk = final.get("buckets", {}) or {}
    if bk.get("frequent") and bk.get("rare"):
        A("| bucket | meanings | topsim | coherence | symbols | words |")
        A("|---|---|---|---|---|---|")
        for label in ("frequent", "rare"):
            b = bk.get(label, {})
            A("| %s | %d | %.3f | %.3f | %.2f | %.2f |"
              % (label, int(g(b, "n", 0)), g(b, "topsim"), g(b, "coherence"),
                 g(b, "mean_symbols"), g(b, "mean_words")))
        A("")

    A("## 3e. Forms lost and rebuilt across generations")
    A("")
    A("A newborn's apprenticeship is dominated by trades that were common, so a "
      "form that only ever attached to a rare meaning may simply never be shown to "
      "the next generation. When that happens the meaning has to be rebuilt out of "
      "whatever parts *are* well attested -- the same shape as an irregular verb "
      "levelling out to the regular pattern. Replacements are flagged as "
      "*regularised* when the new form is built from words that are common "
      "elsewhere in the language and the old one was not.")
    A("")
    fs = final.get("form_survival", {}) or {}
    if fs:
        A("| measure | frequent meanings | rare meanings |")
        A("|---|---|---|")
        A("| drift between checkpoints | %.3f | %.3f |"
          % (g(fs, "drift_frequent"), g(fs, "drift_rare")))
        A("| share of forms replaced | %.2f | %.2f |"
          % (g(fs, "changed_frequent"), g(fs, "changed_rare")))
        A("")
        A("Replacements observed: %d, of which %d rebuilt from commoner parts."
          % (int(g(fs, "n_events", 0)), int(g(fs, "n_regularised", 0))))
        A("")
    events = final.get("form_events") or []
    if events:
        A("| episode | meaning | bucket | how often | old form | new form | rebuilt from common parts? |")
        A("|---|---|---|---|---|---|---|")
        for e in events[:12]:
            A("| %d | %s | %s | %.4f | `%s` | `%s` | %s |"
              % (e["episode"], e["meaning"], e["bucket"], e["prob"],
                 e["old_form"], e["new_form"], "yes" if e["regularised"] else "no"))
        A("")
    timeline = final.get("form_timeline") or []
    if timeline:
        A("How a few specific meanings were named over the run:")
        A("")
        for t in timeline:
            trail = " -> ".join("`%s` (ep %d)" % (x["form"], x["episode"])
                                for x in t["trail"][:8])
            A("- **%s** (%s, p=%.4f): %s" % (t["meaning"], t["bucket"], t["prob"], trail))
        A("")

    A("## 4. Example transcripts across the run")
    A("")
    A("Raw token ids only -- placeholder labels, no imposed semantics.")
    A("")
    if archive:
        n = len(archive)
        picks = [("Early (episode %d)" % archive[0]["episode"], archive[0])]
        if n > 2:
            mid = archive[n // 2]
            picks.append(("Middle (episode %d)" % mid["episode"], mid))
        picks.append(("Late (episode %d)" % archive[-1]["episode"], archive[-1]))
        for title, rec in picks:
            A("### " + title)
            A("")
            A("```")
            A(rec["rendered"])
            A("```")
            A("")

    A("## 5. Economy")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A("| episodes played | %d |" % totals.get("episodes", 0))
    A("| completed trades | %d |" % totals.get("trades", 0))
    A("| apples sold | %d |" % totals.get("apples_sold", 0))
    A("| total trade value | %.1f |" % totals.get("value", 0.0))
    A("| total farmer profit | %.1f |" % totals.get("profit", 0.0))
    A("| restocks | %d |" % final.get("economy", {}).get("restocks", 0))
    A("| farms sold out | %d |" % final.get("economy", {}).get("soldouts", 0))
    A("")
    fm = final.get("failure_modes", {})
    if fm:
        tot = sum(fm.values()) or 1
        A("Failure modes across the whole run:")
        A("")
        A("| outcome | share |")
        A("|---|---|")
        for k, v in fm.items():
            A("| %s | %.1f%% |" % (k, 100 * v / tot))
        A("")

    A("## 6. Population and transmission")
    A("")
    popd = final.get("population", {})
    A("- births: %d, deaths: %d" % (popd.get("total_births", 0), popd.get("total_deaths", 0)))
    A("- final farmer generations: %s" % popd.get("farmers", {}).get("generations"))
    A("- final buyer generations: %s" % popd.get("buyers", {}).get("generations"))
    A("")
    if newborn_reports:
        A("Newborns tested against veterans immediately after their bottleneck training "
          "(before any live episode):")
        A("")
        A("| episode | newborn | generation | bottleneck samples | token acc | success vs veterans |")
        A("|---|---|---|---|---|---|")
        for r in newborn_reports[-15:]:
            bn = r.get("bottleneck", {}) or {}
            pr = r.get("at_birth_vs_veterans", {}) or {}
            sr = pr.get("success_rate")
            A("| %d | %s slot %d | %d | %s | %s | %s |"
              % (r.get("episode", 0), r.get("role", "?"), r.get("slot", -1),
                 r.get("generation", 0), bn.get("n_samples", "-"),
                 ("%.3f" % bn["token_accuracy"]) if bn.get("token_accuracy") is not None else "-",
                 ("%.3f" % sr) if isinstance(sr, float) and sr == sr else "-"))
        A("")

    A("## 7. Where everything is")
    A("")
    A("- `%s` -- every episode: hidden state, full token transcript, both decisions, "
      "outcome, failure classification, rewards and money." % os.path.basename(ledger_path))
    A("- `trades.csv` -- the same rows, flattened.")
    A("- `metrics.jsonl` -- every checkpoint's full metric suite.")
    A("- `births.jsonl` -- every birth, what it was trained on, how it fared with veterans.")
    A("- `run.log` -- the complete console history of the run.")
    A("- `plots/metrics.png` -- success, compositionality, vocabulary, stability, "
      "transmission and generalisation over training.")
    A("- `config.json` -- the exact configuration this run used.")
    A("")
    A("## 8. Method caveats")
    A("")
    A("- Reward is shaped with partial credit for *mutual agreement* and for decisions "
      "that match the joint ground truth. Fully sparse success is around 1e-3 under "
      "random play, which REINFORCE cannot bootstrap from. Every shaped term still "
      "requires information neither agent holds alone, so the shaping does not "
      "remove the need to communicate -- but it does mean success rate alone is not "
      "proof of language, which is why topsim and coherence are reported beside it.")
    A("- Topological similarity is measured on greedily-decoded first utterances, so "
      "the meaning-to-message mapping is a deterministic function. Live play samples "
      "from the policy and is noisier.")
    A("- Farmer utterances are probed against one fixed buyer opening; a farmer's reply "
      "genuinely depends on what was said to it, and holding that constant isolates the "
      "farmer's own contribution.")
    A("- No pretrained model, embedding or text corpus is used anywhere in this "
      "project. Every weight starts random and is trained only on interaction inside "
      "this simulation.")
    A("")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path
