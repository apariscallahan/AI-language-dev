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
def rung_chance(final: dict[str, Any], trade_chance: float) -> float:
    """Chance on the rung the run ended on, not the trading task's.

    A lineup is 1/K by chance (0.25 with four candidates); judging a lineup
    success of 0.244 against the trading task's ~0 called it "far above chance"
    when it was exactly at chance. Rungs with no analytic chance (mutual, order)
    use their muted-channel success instead.
    """
    c = final.get("chance_for_phase")
    if isinstance(c, (int, float)) and c == c:
        return float(c)
    views = (final.get("rung_evidence") or {}).get("views") or []
    muted = [v.get("muted_success") for v in views
             if isinstance(v.get("muted_success"), (int, float))]
    if muted:
        return float(sum(muted) / len(muted))
    return trade_chance


def assess(cfg: Config, final: dict[str, Any], chance: float) -> dict[str, Any]:
    """Grade the run against fixed thresholds.  No hedging, no cherry-picking."""
    chance = rung_chance(final, chance)
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
    coh = nn(stab.get("coherence"),
             (nn(stab.get("coherence_buyer"), 0.0) + nn(stab.get("coherence_farmer"), 0.0)) / 2)
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
    _fr, _br = nn(abl.get("farmer_reads_transfer")), nn(abl.get("buyer_reads_transfer"))
    # Unknown (an older run without these numbers) must not read as a failure.
    checks["loop_closes_both_ways"] = (
        (_fr != _fr and _br != _br) or (_fr > 0.15 and _br > 0.15))
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
    elif zs.get("suppressed"):
        ev.append("zero-shot retention not reported (%s)" % zs["suppressed"])
    ov = final.get("cross_role_overlap") or {}
    if nn(ov.get("weighted_overlap")) == nn(ov.get("weighted_overlap")):
        ev.append("cross-role vocabulary overlap %.2f (histogram intersection of the "
                  "two roles' word use; 1.0 = one shared vocabulary)"
                  % ov["weighted_overlap"])
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
    fr_t = nn(abl.get("farmer_reads_transfer"))
    br_t = nn(abl.get("buyer_reads_transfer"))
    if fr_t == fr_t and br_t == br_t:
        ev.append("the loop closes in both directions: the farmer recovers %.0f%% of "
                  "the headroom on the buyer's private fields, the buyer %.0f%% on the "
                  "farmer's (both measured against being muted)"
                  % (100 * fr_t, 100 * br_t))
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
    elif not checks["loop_closes_both_ways"] and checks["named_things"]:
        verdict = "ONE-WAY SIGNALLING"
        summary = ("Information is crossing the channel, but only in one direction: "
                   "one side is being read and the other is not. Check the two "
                   "'reads' rows below -- a language needs both halves, and a "
                   "speaker with no one listening has no reason to stay informative.")
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
def _bottleneck_sentence(bc: dict[str, Any]) -> str:
    """Describe what newborns were actually shown -- from this run's own numbers.

    The previous report asserted "a newborn now sees essentially the whole parent
    generation" as fixed prose, beside a births table showing 400 transcripts per
    newborn (that run's config forced ``n_samples=400``, which overrides
    coverage) and coverage figures of 13% / 0%. The claim is now computed.
    """
    if not bc:
        return "No newborn was trained during this run, so nothing was transmitted."
    shown = float(bc.get("mean_transcripts_shown", 0.0))
    store = float(bc.get("mean_store_size", 0.0))
    share = float(bc.get("mean_share_of_store", 0.0))
    com = float(bc.get("common_form_coverage", float("nan")))
    rare = float(bc.get("rare_form_coverage", float("nan")))
    cap = int(bc.get("fixed_cap", 0) or 0)
    if cap > 0:
        regime = ("This run used a fixed cap of %d transcripts per newborn "
                  "(``bottleneck.n_samples``, which overrides coverage), i.e. %.0f%% of "
                  "a store averaging %d." % (cap, 100 * share, store))
    else:
        regime = ("Newborns were shown on average %d of %d stored transcripts (%.0f%%; "
                  "coverage setting %.0f%%, ceiling %d)."
                  % (shown, store, 100 * share, 100 * float(bc.get("coverage_setting", 1.0)),
                     int(bc.get("max_samples", 0))))
    if com == com and com >= 0.9:
        verdict = ("Common forms were reliably shown (%.0f%% of them at least three "
                   "times) and rare ones much less often (%.0f%%): the intended "
                   "asymmetry, where only the rare end is at risk." % (100 * com, 100 * rare))
    elif com == com:
        verdict = ("Only %.0f%% of common forms (and %.0f%% of rare ones) appeared "
                   "often enough to learn, so this run did **not** achieve near-complete "
                   "transmission: common vocabulary was itself at risk. With the store "
                   "this small relative to the vocabulary in circulation (%d forms), "
                   "even a full pass shows most forms too rarely."
                   % (100 * com, 100 * rare, int(bc.get("vocabulary_in_population", 0))))
    else:
        verdict = ""
    return regime + " " + verdict


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
    from .metrics import positional_structure
    lines = ["| speaker | positional-structure score | slots in use |", "|---|---|---|"]
    for role, rows in sem.per_position.items():
        live = sum(1 for r in rows if r.get("used", 1.0) >= 0.2)
        lines.append("| %s | %.3f | %d of %d |"
                     % (role, positional_structure(rows), live, len(rows)))
    lines += ["", "| speaker | message slot | dimension most predicted by this slot | "
              "strength | slot in use | distinct tokens seen |",
              "|---|---|---|---|---|---|"]
    for role, rows in sem.per_position.items():
        for r in rows:
            lines.append("| %s | %d | %s | %.3f | %.0f%% | %d |"
                         % (role, r["position"], r["dimension"], r["score"],
                            100 * r.get("used", 1.0), r["distinct_tokens"]))
    return lines


# --------------------------------------------------------------------------
def _summary_lines(cfg: Config, final: dict[str, Any], wall_minutes: float) -> list[str]:
    """Headline numbers, one table, at the top of the report."""
    def f(x, fmt="%.3f"):
        return fmt % x if isinstance(x, (int, float)) and x == x else "n/a"

    cur = final.get("curriculum") or {}
    ev = final.get("rung_evidence") or {}
    st = final.get("stability") or {}
    w = final.get("words") or {}
    ov = final.get("cross_role_overlap") or {}
    zs = final.get("zero_shot") or {}
    pop = final.get("population") or {}
    prs = final.get("per_role_structure") or {}
    nta = cur.get("newborn_token_accuracy") or {}
    props = final.get("language_properties") or []
    rows: list[tuple[str, str]] = []

    rows.append(("started", str(final.get("started_utc", "n/a"))))
    from .config import method_changes
    changes = method_changes(cfg)
    rows.append(("method", "the one configuration (nothing simulated was changed)" if not changes
                 else "**changed**: " + ", ".join(
                     "`%s` %s -> %s" % (k, json.dumps(a), json.dumps(b))
                     for k, (a, b) in sorted(changes.items()))))
    rows.append(("episodes / wall time", "%s in %.1f h"
                 % ("{:,}".format(int(final.get("episode", 0))), wall_minutes / 60.0)))
    phases = cur.get("phases") or []
    reached = cur.get("reached", final.get("phase", "?"))
    stop = cur.get("stop_report") or {}
    status = ("stopped: budget exceeded" if stop else
              "ran to the episode budget" if final.get("final") else "in progress")
    rows.append(("furthest rung", "**%s** (%s of %d) -- %s"
                 % (reached, (phases.index(reached) + 1) if reached in phases else "?",
                    len(phases), status)))
    passed = ["%s (%s updates)" % (t.get("from"),
                                   "{:,}".format(int(t.get("updates_in_previous_phase") or 0)))
              for t in cur.get("transitions") or []]
    rows.append(("rungs passed", ", ".join(passed) or "none"))
    growth = cur.get("community_growth") or []
    rows.append(("community", "%d farmers + %d buyers%s; %d births"
                 % ((pop.get("farmers") or {}).get("n", 0), (pop.get("buyers") or {}).get("n", 0),
                    (" (founded %d + %d)" % (cfg.population.founders_farmers,
                                             cfg.population.founders_buyers)
                     if cfg.population.founders_farmers else ""),
                    int(pop.get("total_births", 0)))))
    views = ev.get("views") or []
    if views:
        rows.append(("success on the current rung", "; ".join(
            "%s%s (muted %s)" % ("%s describes: " % v["informer"] if v.get("informer") else "",
                                 f(v.get("success")), f(v.get("muted_success")))
            for v in views)))
    rows.append(("channel carries", "%s of the headroom over a muted channel"
                 % f(ev.get("transfer"), "%.2f")))
    for lbl in ("farmer", "buyer"):
        sp = prs.get(lbl)
        if sp:
            cov = sp.get("per_field_coverage") or []
            rows.append(("%s messages" % lbl,
                         "topsim %s (null %s); field coverage %s [variety %s, quantity %s, quality %s]"
                         % (f(sp.get("topsim")), f(sp.get("null")), f(sp.get("field_coverage")),
                            *[f(x, "%.2f") for x in (cov + [float("nan")] * 3)[:3]])))
    rows.append(("coherence", "farmer %s, buyer %s, across roles %s"
                 % (f(st.get("coherence_farmer")), f(st.get("coherence_buyer")),
                    f(st.get("coherence_cross")))))
    rows.append(("cross-role vocabulary overlap", f(ov.get("weighted_overlap"))))
    rows.append(("vocabulary", "%s distinct words, %s atoms per word, %s words and %s symbols "
                 "per utterance, %s of utterances at the buffer end"
                 % (w.get("distinct_words", "n/a"), f(w.get("mean_word_len_atoms"), "%.2f"),
                    f(w.get("mean_words_per_message"), "%.2f"),
                    f(w.get("mean_symbols_per_message"), "%.2f"),
                    f(100 * w.get("at_length_cap_frac", float("nan")), "%.0f%%"))))
    fm = (nta.get("farmer") or {}).get("mean")
    bm = (nta.get("buyer") or {}).get("mean")
    rows.append(("newborn token accuracy", "farmer %s, buyer %s" % (f(fm), f(bm))))
    ret = zs.get("retention")
    rows.append(("zero-shot (held-out combinations)",
                 f(ret, "%.2f") + (" retention (%s)" % zs.get("context", "") if ret == ret else
                                   " -- %s" % (zs.get("suppressed") or "not measured"))))
    if props:
        counts: dict[str, list[str]] = {}
        for q in props:
            counts.setdefault(q["verdict"], []).append(q["property"])
        rows.append(("properties of language", "; ".join(
            "%s: %s" % (k, ", ".join(v)) for k, v in counts.items())))

    out = ["## Summary statistics", "", "| | |", "|---|---|"]
    out += ["| %s | %s |" % (k, v.replace("|", "/")) for k, v in rows]
    out.append("")
    return out


def write_report(cfg: Config, out_dir: str, *, final: dict[str, Any],
                 chance: float, sem: TokenSemantics, archive: Sequence[dict[str, Any]],
                 totals: dict[str, Any], history: dict[str, Any],
                 newborn_reports: Sequence[dict[str, Any]],
                 ledger_path: str, wall_minutes: float) -> str:
    verdict = assess(cfg, final, chance)
    chance = rung_chance(final, chance)
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
    L.extend(_summary_lines(cfg, final, wall_minutes))
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
    A("| farmer reads the buyer | %.3f | share of the buyer's private fields it "
      "recovered |" % g(final, "farmer_reads_buyer"))
    A("| buyer reads the farmer | %.3f | share of the farmer's private fields it "
      "recovered |" % g(final, "buyer_reads_farmer"))
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
      "symbol in that slot predicts best. Each role is probed only in a phase where "
      "it actually speaks, on the meanings it actually describes there; the score is "
      "the mean strength over the slots in use (a symbol there in at least 20% of "
      "utterances).")
    A("")
    L.extend(_position_lines(sem))
    A("")

    A("## 3a. The curriculum")
    A("")
    cur = final.get("curriculum", {}) or {}
    if not cur.get("enabled", False):
        A("The curriculum was off: agents faced the full trading task from random "
          "weights.")
        A("")
    else:
        phases = cur.get("phases", [])
        reached = cur.get("reached", "?")
        A("Agents work up a ladder, and only leave a rung once it has demonstrably "
          "worked -- success clear of chance, topological similarity clear of its "
          "shuffled null, and muting the channel actually costing something. "
          "Weights carry across every transition; nothing is reinitialised.")
        A("")
        budgets = cur.get("budgets") or {}
        spent = {t.get("from"): (t.get("updates_in_previous_phase"),
                                 t.get("episodes_in_previous_phase"))
                 for t in (cur.get("transitions") or [])}
        A("| rung | phase | reached | budget (min - max updates) | updates spent (episodes) |")
        A("|---|---|---|---|---|")
        for i, name in enumerate(phases):
            mark = ("**yes**" if i <= cur.get("reached_index", 0) else "no")
            b = budgets.get(name) or ["?", "?"]
            hi = ("open" if isinstance(b[1], (int, float)) and b[1] >= 10**8
                  else "{:,}".format(b[1]) if isinstance(b[1], (int, float)) else b[1])
            lo = "{:,}".format(b[0]) if isinstance(b[0], (int, float)) else b[0]
            if name in spent:
                used = "{:,} ({:,})".format(int(spent[name][0] or 0), int(spent[name][1] or 0))
            elif i == cur.get("reached_index", -1):
                used = "{:,} ({:,}) (current)".format(
                    int(g(cur, "updates_in_current_phase", 0)),
                    int(g(cur, "episodes_in_current_phase", 0)))
            else:
                used = "-"
            A("| %d | `%s` | %s | %s - %s | %s |" % (i + 1, name, mark, lo, hi, used))
        A("")
        growth = cur.get("community_growth") or []
        if growth:
            A("The community was founded by %d farmers and %d buyers and grew by "
              "newcomers -- random weights, then the transmission bottleneck on the "
              "community's transcripts -- to %d + %d, reached at update %s (episode %s) "
              "during `%s`. Every rung after the first was judged on the full community."
              % (cfg.population.founders_farmers, cfg.population.founders_buyers,
                 growth[-1]["farmers"], growth[-1]["buyers"],
                 "{:,}".format(growth[-1].get("update", 0)),
                 "{:,}".format(growth[-1]["episode"]), growth[-1]["phase"]))
            A("")
        A("`refer-swap` and `refer-mutual` are judged per role: each role has to clear "
          "every bar on its own, describing and decoding, rather than on a pooled "
          "average that a fluent partner could carry.")
        A("")
        A("Furthest rung reached: **%s** (%s updates, %s episodes in it at the end)."
          % (reached, "{:,}".format(int(g(cur, "updates_in_current_phase", 0))),
             "{:,}".format(int(g(cur, "episodes_in_current_phase", 0)))))
        A("")
        trans = cur.get("transitions") or []
        if trans:
            A("### Transitions, and why each one happened")
            A("")
            for t in trans:
                A("**%s -> %s** at update %s (episode %s), after %s updates in `%s`:"
                  % (t.get("from"), t.get("to"), "{:,}".format(t.get("update", 0)),
                     "{:,}".format(t.get("episode", 0)),
                     "{:,}".format(t.get("updates_in_previous_phase") or 0),
                     t.get("from")))
                A("")
                for k, c in (t.get("criteria") or {}).items():
                    A("- %s: %s" % (k, c.get("detail")))
                A("")
                ev_t = t.get("evidence") or {}
                sp = ev_t.get("speakers") or {}
                if sp:
                    A("  Per role at that check: " + "; ".join(
                        "%s topsim %.3f (null %.3f), positional %.3f"
                        % (k, g(v, "topsim"), g(v, "null"), g(v, "positional"))
                        for k, v in sp.items()))
                    A("")
        else:
            A("No transition happened during this run.")
            A("")
        if cur.get("stalled"):
            sr = cur.get("stop_report") or {}
            last = cur.get("last_promotion_check") or {}
            A("> **Rung `%s` exceeded its budget.** %s updates in it (maximum %s) "
              "without meeting its criteria, so the run %s rather than advance -- "
              "building the next rung on top of one that never converged would only "
              "reproduce the failure a rung higher."
              % (sr.get("phase", cur.get("reached")),
                 "{:,}".format(int(sr.get("updates_in_phase", 0) or 0)),
                 "{:,}".format(int(sr.get("max_updates", 0) or 0)),
                 "stopped" if sr.get("action") == "stop" else "held"))
            A(">")
            A("> Unmet:")
            for k, d in (sr.get("unmet") or {}).items():
                A("> - %s: %s" % (k, d))
            if sr.get("met"):
                A(">")
                A("> Met:")
                for k, d in sr["met"].items():
                    A("> - %s: %s" % (k, d))
            A("")
        elif last_check := (cur.get("last_promotion_check") or {}):
            if not last_check.get("passed") and last_check.get("checks"):
                A("Last promotion check on `%s` (episode %s):"
                  % (last_check.get("phase"), "{:,}".format(int(last_check.get("episode", 0)))))
                A("")
                for k, c in last_check["checks"].items():
                    A("- %s %s: %s" % ("met" if c.get("met") else "**unmet**", k,
                                       c.get("detail")))
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
        stab = final.get("stability", {}) or {}
        A("| coherence | %.3f | agents of a role saying the same thing for the same "
          "meaning (farmer %.3f, buyer %.3f) |"
          % (g(stab, "coherence"), g(stab, "coherence_farmer"), g(stab, "coherence_buyer")))
        A("| coherence across roles | %.3f | a farmer and a buyer describing the same "
          "meaning the same way (lineup rungs only) |" % g(stab, "coherence_cross"))
        ov = final.get("cross_role_overlap") or {}
        A("| cross-role overlap | %.3f | histogram intersection of the two roles' word "
          "use: 1.0 is one shared vocabulary, 0.0 two foreign codes |"
          % g(ov, "weighted_overlap"))
        A("| shared-form share | farmer %.0f%% / buyer %.0f%% | of each role's word "
          "tokens, the share that are forms the other role also uses |"
          % (100 * g(ov, "farmer_share_shared"), 100 * g(ov, "buyer_share_shared")))
        A("| shared forms (types) | %s of %s farmer / %s buyer (Jaccard %.3f) | the "
          "one-off tail drags this down |"
          % (ov.get("shared_types", "-"), ov.get("farmer_types", "-"),
             ov.get("buyer_types", "-"), g(ov, "jaccard_types")))
        A("")
        if ov.get("farmer_only_top") or ov.get("buyer_only_top"):
            A("Role-specific forms (commonest first): farmer-only %s; buyer-only %s. "
              "Shared core: %s."
              % (", ".join("`%s`" % w for w in ov.get("farmer_only_top", [])) or "none",
                 ", ".join("`%s`" % w for w in ov.get("buyer_only_top", [])) or "none",
                 ", ".join("`%s`" % w for w in ov.get("shared_top", [])) or "none"))
            A("")
        q = final.get("quantity_encoding_live") or {}
        if q.get("n", 0) >= 50:
            A("Quantity in live messages: %.3f bits beyond what variety already tells "
              "you (plug-in %.3f, shuffled null %.3f, %d sampled messages). Near zero "
              "means messages do not carry quantity at all, whatever a greedy form "
              "table suggests." % (q["excess_bits"], q["mi_bits"], q["null_bits"], q["n"]))
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

    A("### Where the vocabulary came from")
    A("")
    A("Some of the structure visible at the end was inherited from the lineup "
      "game rather than caused by negotiation. Claiming otherwise without "
      "checking would be crediting a pressure that was not responsible, so every "
      "word is stamped with the phase it first appeared in.")
    A("")
    prov = (cur.get("provenance") or {}) if cur else {}
    if prov.get("n_words"):
        A("| phase | words first seen here | words that settled here | word tokens |")
        A("|---|---|---|---|")
        for name in prov.get("phases", []):
            d = prov.get(name) or {}
            A("| `%s` | %d | %d | %s |"
              % (name, int(d.get("first_appeared", 0)),
                 int(d.get("settled_here", 0)),
                 "{:,}".format(int(d.get("word_tokens", 0)))))
        A("")
        inh = cur.get("inherited") or {}
        if inh:
            A("| later phase | words in use | inherited from `refer` | new here | inherited share |")
            A("|---|---|---|---|---|")
            for name, d in inh.items():
                A("| `%s` | %d | %d | %d | %.0f%% |"
                  % (name, int(d.get("words_in_use", 0)),
                     int(d.get("inherited_from_refer", 0)), int(d.get("new_here", 0)),
                     100 * float(d.get("inherited_share", 0.0))))
            A("")
        new_words = cur.get("new_words") or {}
        for name, rows in new_words.items():
            if not rows:
                continue
            A("Words that first appeared in `%s` -- this is where to look for "
              "anything the lineup game had no reason to invent, such as offer, "
              "counter-offer, accept or refuse:" % name)
            A("")
            A("| word | uses in this phase | settled here | first seen |")
            A("|---|---|---|---|")
            for r in rows[:10]:
                A("| `%s` | %d | %s | episode %s |"
                  % (r["word"], r["count"], "yes" if r["settled"] else "no",
                     "{:,}".format(r["first_episode"])))
            A("")
    else:
        A("No vocabulary was recorded.")
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
        A("All three rows are accumulated over the whole run, across %d "
          "meaning-to-meaning comparisons -- not the last checkpoint alone."
          % int(g(fs, "n_comparisons", 0)))
        A("")
        A("| measure | frequent meanings | rare meanings |")
        A("|---|---|---|")
        A("| mean drift per checkpoint | %.3f | %.3f |"
          % (g(fs, "drift_frequent"), g(fs, "drift_rare")))
        A("| share of checkpoints where the form changed | %.2f | %.2f |"
          % (g(fs, "changed_frequent"), g(fs, "changed_rare")))
        A("| outright replacements logged | %d | %d |"
          % (int(g(fs, "n_events_frequent", 0)), int(g(fs, "n_events_rare", 0))))
        A("")
        A("| retention | frequent meanings | rare meanings |")
        A("|---|---|---|")
        A("| form kept between checkpoints | %.0f%% | %.0f%% |"
          % (100 * (1 - g(fs, "changed_frequent", 0.0)),
             100 * (1 - g(fs, "changed_rare", 0.0))))
        bc = (cur.get("bottleneck_coverage") or {}) if cur else {}
        if bc:
            A("| shown to newborns often enough to learn | %.0f%% | %.0f%% |"
              % (100 * float(bc.get("common_form_coverage", 0.0)),
                 100 * float(bc.get("rare_form_coverage", 0.0))))
        A("")
        A(_bottleneck_sentence(bc))
        A("")
        A("The two retention rows measure different things. \"Form kept between "
          "checkpoints\" is whether the population's greedy form for a meaning "
          "survived from one checkpoint to the next -- generational turnover, "
          "learning drift and phase changes all move it. \"Shown to newborns\" is "
          "whether a word appeared at least three times in what a newborn was "
          "trained on; a word can be shown and still not survive, and vice versa.")
        A("")
        A("Replacements observed: %d, of which %d rebuilt from commoner parts. "
          "Drift at the final checkpoint alone was %.3f (frequent) and %.3f (rare)."
          % (int(g(fs, "n_events", 0)), int(g(fs, "n_regularised", 0)),
             g(fs, "drift_frequent_interval"), g(fs, "drift_rare_interval")))
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

    props = final.get("language_properties") or []
    if props:
        A("## 3f. Properties of language")
        A("")
        A("Nothing tells the agents what kind of language to build: they start from "
          "random weights and are shaped only by what the world rewards. So each "
          "property is *measured*, from numbers recorded during the run, and marked "
          "as present, partial, absent -- or not testable, where the world as "
          "configured gives the property no work to do.")
        A("")
        A("| property | how it is measured | value | verdict | note |")
        A("|---|---|---|---|---|")
        for p in props:
            v = p.get("value")
            vs = ("%.3f" % v) if isinstance(v, float) and v == v else (
                str(v) if isinstance(v, int) else "-")
            A("| %s | %s | %s | **%s** | %s |" % (p["property"], p["measure"], vs,
                                                p["verdict"], p.get("note", "")))
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
        nta = (final.get("curriculum") or {}).get("newborn_token_accuracy") or {}
        if nta:
            A("| role | births | mean newborn token accuracy | range | births with no "
              "own tokens in their curriculum |")
            A("|---|---|---|---|---|")
            for lbl in ("farmer", "buyer"):
                d = nta.get(lbl) or {}
                A("| %s | %d | %s | %s | %d |"
                  % (lbl, int(d.get("births", 0)),
                     ("%.3f" % d["mean"]) if d.get("mean") is not None else "n/a",
                     ("%.3f - %.3f" % (d["min"], d["max"])) if d.get("min") is not None
                     else "-", int(d.get("births_with_nothing_to_say", 0))))
            A("")
        A("| episode | newborn | generation | phase | transcripts shown / stored | own "
          "tokens | token acc | decision acc | success vs veterans |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in newborn_reports[-20:]:
            bn = r.get("bottleneck", {}) or {}
            pr = r.get("at_birth_vs_veterans", {}) or {}
            sr = pr.get("success_rate")

            def f3(x):
                return ("%.3f" % x) if isinstance(x, (int, float)) and x == x else "n/a"
            A("| %d | %s slot %d | %d | %s | %s / %s | %s | %s | %s | %s |"
              % (r.get("episode", 0), r.get("role", "?"), r.get("slot", -1),
                 r.get("generation", 0), r.get("phase", "-"),
                 bn.get("n_samples", "-"), bn.get("store_size", "-"),
                 bn.get("own_token_targets", "-"), f3(bn.get("token_accuracy")),
                 f3(bn.get("decision_accuracy")), f3(sr)))
        A("")
        A("Token accuracy is over the newborn's *own* message slots under the phase "
          "each transcript was played in; `n/a` means its curriculum contained nothing "
          "that role said (a buyer born during `refer`, where only farmers describe).")
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
