"""Which properties of language did this run's code actually show?

Nothing here tells an agent what kind of language to build. The agents are
randomly initialised and can only be shaped by what the world rewards, so the
honest way to ask for "a language" is to (a) make each property *useful* in the
world and (b) *measure* each one, and report which are present, partial, absent,
or simply impossible in this world as configured. That is what this module does.

The properties follow the standard list (Hockett's design features and later
refinements):

* reference            symbols stand for things, arbitrarily
* productivity         symbols combine to express new meanings
* intentionality       the point is to communicate, not merely to cause behaviour
* decontextualised     a form means the same thing across contexts
* displaced            talk about what is not here and now
* interchangeable      a form heard from another can be used by oneself
* generic              a word applies to every member of its category
* perspectives         the same content expressed from different roles
* cultural transmission the specific language is learned, not innate
* duality of patterning meaningless minimal units combine into meaningful ones

Every verdict is computed from numbers recorded during the run, with the
threshold stated, so a weak result cannot be talked up.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Optional, Sequence

from .config import Config

PRESENT, PARTIAL, ABSENT, UNTESTABLE, NOT_REACHED = (
    "present", "partial", "absent", "not testable in this world", "not reached")


def _num(x) -> float:
    return float(x) if isinstance(x, (int, float)) and x == x else float("nan")


def _entropy(values: Sequence) -> float:
    c = Counter(values)
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def _mi(xs: Sequence, ys: Sequence) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    cx, cy, cxy = Counter(xs), Counter(ys), Counter(zip(xs, ys))
    return max(0.0, sum(c / n * math.log2((c / n) / ((cx[x] / n) * (cy[y] / n)))
                        for (x, y), c in cxy.items()))


# ==========================================================================
# disentanglement: does a symbol name one attribute, whatever the others are?
# ==========================================================================
def disentanglement(meanings: Sequence[Sequence[int]], messages: Sequence[Sequence[int]],
                    fields: Sequence[int], max_len: int) -> dict[str, float]:
    """Positional and bag-of-symbols disentanglement (Chaabouni et al., 2020).

    For each message position (posdis) or each symbol type's count (bosdis):
    the gap between the mutual information with the best-explained attribute and
    with the runner-up, over that position's entropy. 1.0 means each position
    (or symbol) carries exactly one attribute; 0 means none or all mixed.
    This is the measurable core of "generic": RED is named the same way
    whatever the quantity and quality are.
    """
    if len(meanings) < 8 or len(fields) < 2:
        return {"posdis": float("nan"), "bosdis": float("nan")}
    attrs = [[m[f] for m in meanings] for f in fields]

    def gap(sym):
        h = _entropy(sym)
        if h <= 1e-9:
            return None
        mis = sorted((_mi(sym, a) for a in attrs), reverse=True)
        return (mis[0] - mis[1]) / h

    pos = []
    for k in range(max_len):
        g = gap([msg[k] if k < len(msg) else -1 for msg in messages])
        if g is not None:
            pos.append(g)
    vocab = sorted({t for msg in messages for t in msg})
    bos = []
    for v in vocab:
        g = gap([sum(1 for t in msg if t == v) for msg in messages])
        if g is not None:
            bos.append(g)
    return {"posdis": sum(pos) / len(pos) if pos else float("nan"),
            "bosdis": sum(bos) / len(bos) if bos else float("nan")}


def _message_units(messages: Sequence[Sequence[int]], *, space: Optional[int],
                    hyphen: Optional[int], end: Optional[int], max_len: int
                    ) -> list[tuple[str, list]]:
    """The pieces of a message a one-piece reader could read a field off.

    Every symbol slot, every word position (a word keeps its hyphens, so a
    two-atom word is one value), and the bag of words. ``None`` for a slot or
    position the message does not reach.
    """
    msgs = [list(m) for m in messages]
    units: list[tuple[str, list]] = []
    for k in range(max_len):
        units.append(("symbol %d" % k, [m[k] if k < len(m) else None for m in msgs]))
    words: list[list[tuple]] = []
    for m in msgs:
        ws, cur = [], []
        for s in m:
            if s == end:
                break
            if s == space:
                if cur:
                    ws.append(tuple(cur))
                cur = []
            else:
                cur.append(s)
        if cur:
            ws.append(tuple(cur))
        words.append(ws)
    for k in range(max((len(w) for w in words), default=0)):
        units.append(("word %d" % k, [w[k] if k < len(w) else None for w in words]))
    units.append(("bag", [frozenset(w) for w in words]))
    return units


def _lookup_accuracy(xs: list, ys: list, train: list, test: list) -> float:
    """Predict ``ys`` from ``xs`` with a table fitted on ``train``, scored on
    ``test`` as the share of the headroom above always guessing the majority."""
    table: dict = {}
    for i in train:
        x = xs[i]
        if isinstance(x, frozenset):          # a bag: one entry per word in it
            for w in x:
                table.setdefault(w, Counter())[ys[i]] += 1
        else:
            table.setdefault(x, Counter())[ys[i]] += 1
    majority = Counter(ys[i] for i in train).most_common(1)[0][0]
    base = sum(ys[i] == majority for i in test) / len(test)
    hit = 0
    for i in test:
        x = xs[i]
        if isinstance(x, frozenset):
            # A bag of words: read the field off the word that predicts it
            # most purely, whichever position it sits in.
            best, guess = -1.0, majority
            for w in x:
                c = table.get(w)
                if not c:
                    continue
                v, k = c.most_common(1)[0]
                purity = k / sum(c.values())
                if purity > best or (purity == best and k > 0):
                    best, guess = purity, v
        else:
            c = table.get(x)
            guess = c.most_common(1)[0][0] if c else majority
        hit += guess == ys[i]
    acc = hit / len(test)
    return (acc - base) / (1.0 - base) if base < 1.0 else 0.0


def field_coverage(meanings: Sequence[Sequence[int]], messages: Sequence[Sequence[int]],
                   fields: Sequence[int], rng=None, *, space: Optional[int] = None,
                   hyphen: Optional[int] = None, end: Optional[int] = None,
                   max_len: Optional[int] = None) -> dict[str, Any]:
    """How much of each field a one-piece reader recovers from the message.

    Per field: the best, over every symbol slot, every word position and the
    bag of words, of how well that piece alone predicts the field on probes it
    was not fitted on -- a lookup table fitted on half the probes, scored on
    the other half (both ways round, averaged), as the share of the headroom
    above always guessing the commonest value, clipped to [0, 1]. ``coverage``
    is the mean over fields. A code that repeats the fruit in every slot covers
    one field of five; a fused label that names nothing twice covers none.

    It reads the message in pieces because a whole-message statistic has
    nothing to say about a five-field lot at any affordable probe count. A
    whole lot takes 3,456 values and a compositional code gives each its own
    message, so over a few hundred probes nearly every message is unique and
    the plug-in I(message; field) sits at H(field) whatever the message means;
    the shuffled null sits there too, and the difference is noise over a
    vanishing headroom. Measured on a flawless describer, whole-message MI over
    its own null read 0.00 at 100 probes, 0.34 at 200 and 0.84 at 400. The
    per-piece reading is 1.00 at every one of them, a code that gets each field
    right three times in four reads 0.78 at every one of them, a holistic code
    (one arbitrary word per lot) 0.04, and a random message 0.03-0.05. Cross-
    validation is what keeps the max over pieces honest: a piece that only
    fits the probes it was fitted on predicts nothing on the rest.
    """
    import random as _r
    rng = rng or _r.Random(0)
    n = len(messages)
    if n < 4 or not fields:
        return {"coverage": float("nan"), "per_field": [float("nan")] * len(fields),
                "units": [None] * len(fields)}
    L = max_len if max_len is not None else max((len(m) for m in messages), default=0)
    units = _message_units(messages, space=space, hyphen=hyphen, end=end, max_len=L)
    idx = list(range(n))
    rng.shuffle(idx)
    folds = (idx[:n // 2], idx[n // 2:])
    per, best_unit = [], []
    for f in fields:
        ys = [m[f] for m in meanings]
        if len(set(ys)) < 2:
            per.append(0.0)
            best_unit.append(None)
            continue
        best, name = 0.0, None
        for label, xs in units:
            if all(x is None for x in xs):
                continue
            acc = (_lookup_accuracy(xs, ys, folds[0], folds[1])
                   + _lookup_accuracy(xs, ys, folds[1], folds[0])) / 2
            if acc > best:
                best, name = acc, label
        per.append(max(0.0, min(1.0, best)))
        best_unit.append(name)
    return {"coverage": sum(per) / len(per), "per_field": per, "units": best_unit}


# ==========================================================================
# duality of patterning
# ==========================================================================
def duality(cfg: Config, sem) -> dict[str, Any]:
    """Meaningless minimal units combining into meaningful larger ones.

    Read off the post-hoc semantics: which *words* reliably carry a field value,
    how many of those are built from several atoms, and whether the same atom
    turns up inside meaningful words that mean different things -- which is
    what "the atom itself carries no meaning, the combination does" looks like.
    Also reports whether the world makes duality *necessary*: if there are
    fewer field values to name than atoms, every value can have its own atom
    and there is no pressure to combine.
    """
    from .env import parse_words  # noqa: F401  (kept for symmetry with lexicon)
    w, c = cfg.world, cfg.channel
    per_word = getattr(sem, "per_word", {}) or {}
    per_tok = getattr(sem, "per_token", {}) or {}
    meaningful = {k: r for k, r in per_word.items() if r.get("score", 0.0) >= 0.10}
    multi = {k: r for k, r in meaningful.items() if r.get("atoms", 1) >= 2}
    # atoms shared by meaningful words that mean different things
    uses: dict[str, set] = {}
    for k, r in meaningful.items():
        for a in k.split("-"):
            uses.setdefault(a, set()).add((r["dimension"], r["typical"]))
    reused = sorted(a for a, meanings in uses.items() if len(meanings) >= 2)
    atom_scores = [r["score"] for r in per_tok.values()]
    word_scores = [r["score"] for r in multi.values()]
    n_values = w.n_varieties + w.n_colors + w.n_quality + (w.max_qty + 1) + w.n_price_bins
    return {
        "atoms": c.atomic_vocab,
        "field_values_to_name": n_values,
        "necessary": n_values > c.atomic_vocab,
        "meaningful_words": len(meaningful),
        "multi_atom_meaningful_words": len(multi),
        "atoms_reused_across_meanings": len(reused),
        "reused_atoms": ["a%s" % a if not str(a).startswith("a") else a for a in reused][:10],
        "mean_atom_informativeness": (sum(atom_scores) / len(atom_scores)
                                      if atom_scores else float("nan")),
        "mean_compound_informativeness": (sum(word_scores) / len(word_scores)
                                          if word_scores else float("nan")),
    }


# ==========================================================================
# the scorecard
# ==========================================================================
def _latest(rows: Sequence[dict], getter, want=lambda v: v == v):
    """Latest row value for which ``getter`` returns something usable."""
    for r in reversed(rows):
        try:
            v = getter(r)
        except Exception:
            continue
        if v is not None and (not isinstance(v, float) or want(v)):
            return v, r
    return None, None


def scorecard(cfg: Config, rows: Sequence[dict], curriculum: dict,
              sem=None) -> list[dict[str, Any]]:
    """One row per property: how it is measured, the value, a verdict, a note."""
    out: list[dict[str, Any]] = []
    names = [t.get("to") for t in curriculum.get("transitions", [])]
    reached = set([curriculum.get("reached")] + names + [t.get("from") for t in
                                                         curriculum.get("transitions", [])])

    def add(prop, measure, value, verdict, note=""):
        out.append({"property": prop, "measure": measure, "value": value,
                    "verdict": verdict, "note": note})

    # ---- reference -------------------------------------------------------
    best_t = max((_num((r.get("rung_evidence") or {}).get("transfer")) for r in rows),
                 default=float("nan"), key=lambda x: x if x == x else -9)
    add("reference", "share of the headroom over a muted channel that messages "
        "account for (best rung)", best_t,
        PRESENT if best_t >= 0.25 else (PARTIAL if best_t >= 0.10 else ABSENT),
        "symbols are built from atoms with no meaning of their own, so any "
        "form-meaning pairing is arbitrary by construction")

    # ---- productivity ----------------------------------------------------
    zs, zr = _latest(rows, lambda r: _num(r["zero_shot"]["retention"])
                     if (r.get("zero_shot") or {}).get("context") == "lineup tuples" else None)
    ts, _ = _latest(rows, lambda r: _num((r.get("compositionality") or {}).get("mean")))
    if zs is None:
        add("productivity", "success on (fruit, colour, quality) combinations never "
            "seen in training, relative to seen ones", float("nan"), NOT_REACHED,
            "no lineup checkpoint measured held-out combinations")
    else:
        v = PRESENT if zs >= 0.8 else (PARTIAL if zs >= 0.5 else ABSENT)
        add("productivity", "held-out combination success / seen success (lineup)", zs, v,
            "topsim %.3f at the same point" % (ts if ts is not None else float("nan")))

    # ---- word classes ------------------------------------------------------
    wc = (getattr(sem, "word_classes", {}) or {}) if sem is not None else {}
    if wc:
        best = max(wc.values(), key=lambda d: (d.get("n_classes", 0), d.get("multi_class_share", 0)))
        n, share = best.get("n_classes", 0), best.get("multi_class_share", 0.0)
        v = (PRESENT if n >= 2 and share >= 0.25 else PARTIAL if n >= 2 or share >= 0.10
             else ABSENT)
        add("word classes (nouns, adjectives...)", "separate space-separated words that "
            "each name one field (variety noun-like, quality adjective-like, quantity "
            "numeral-like), and the share of messages combining two or more of them",
            float(n), v, "; ".join("%s: %d classes %s, %.0f%% of messages combine classes, "
                                   "%.2f words per message"
                                   % (r, d.get("n_classes", 0),
                                      "/".join(sorted(d.get("classes", {}))),
                                      100 * d.get("multi_class_share", 0.0),
                                      d.get("mean_words_per_message", 0.0))
                                   for r, d in wc.items()))

    # ---- intentionality --------------------------------------------------
    add("intentionality", "reward reaches a speaker only through its partner's "
        "decoding; channel headroom as above", best_t,
        PARTIAL if best_t >= 0.25 else ABSENT,
        "behaviour can show messages are shaped to be decoded; it cannot show "
        "intent in the human sense, so this is never scored higher than partial")

    # ---- decontextualised ------------------------------------------------
    cc, _ = _latest(rows, lambda r: _num((r.get("context_consistency") or {}).get("consistency")))
    if cc is None:
        add("decontextualised", "buyer's form for a lot as a lineup describer vs as a "
            "requester in a trade (same-meaning minus different-meaning similarity)",
            float("nan"), NOT_REACHED, "needs a checkpoint at or after the order rung")
    else:
        add("decontextualised", "same-meaning minus different-meaning form similarity, "
            "lineup vs trade request (buyer)", cc,
            PRESENT if cc >= 0.3 else (PARTIAL if cc >= 0.1 else ABSENT))

    # ---- displaced -------------------------------------------------------
    add("displaced", "-", float("nan"), UNTESTABLE,
        "every utterance is about the speaker's current private state; nothing in "
        "the world asks about absent, past or future things (a rung that refers to "
        "an earlier round's item would test this)")

    # ---- interchangeable -------------------------------------------------
    # Which rungs these are comes from the ladder. Hard-coded rung names once
    # outlived the rungs themselves, so both properties reported "not reached"
    # however far a run got.
    from .curriculum import ladder
    rungs = ladder(cfg)
    swap_name = next((p.name for p in rungs if p.swaps and p.whole), "")
    mutual_name = next((p.name for p in rungs if p.mutual), "")
    swap = next((t for t in curriculum.get("transitions", [])
                 if t.get("from") == swap_name), None)
    ov, _ = _latest(rows, lambda r: _num((r.get("cross_role_overlap") or {}).get("weighted_overlap")))
    cx, _ = _latest(rows, lambda r: _num((r.get("stability") or {}).get("coherence_cross")))
    if swap is None and swap_name not in reached:
        add("interchangeable", "both roles describe and decode; cross-role overlap",
            float("nan"), NOT_REACHED)
    else:
        passed = swap is not None
        v = (PRESENT if passed and (ov or 0) >= 0.5 else
             PARTIAL if passed or (ov or 0) >= 0.3 else ABSENT)
        add("interchangeable", "swap rung passed per role; cross-role vocabulary overlap; "
            "cross-role coherence", ov if ov is not None else float("nan"), v,
            "swap passed: %s; cross-role coherence %.3f"
            % ("yes" if passed else "no", cx if cx is not None else float("nan")))

    # ---- generic ---------------------------------------------------------
    # Disentanglement of a code that carries almost nothing is noise, so only
    # roles whose messages cover the fields at all (coverage >= 0.2) count.
    def informative(v):
        return _num(v.get("field_coverage")) >= 0.2

    pdis, _ = _latest(rows, lambda r: max(
        [_num(v.get("posdis")) for v in (r.get("per_role_structure") or {}).values()
         if informative(v) and _num(v.get("posdis")) == _num(v.get("posdis"))]
        or [float("nan")]))
    bdis, _ = _latest(rows, lambda r: max(
        [_num(v.get("bosdis")) for v in (r.get("per_role_structure") or {}).values()
         if informative(v) and _num(v.get("bosdis")) == _num(v.get("bosdis"))]
        or [float("nan")]))
    g = max([x for x in (pdis, bdis) if x is not None and x == x] or [float("nan")])
    add("generic", "disentanglement: a position or symbol names one attribute value "
        "whatever the others are (posdis / bosdis, best role with field coverage >= 0.2)",
        g, PRESENT if g >= 0.3 else (PARTIAL if g >= 0.1 else ABSENT),
        "posdis %.3f, bosdis %.3f" % (pdis if pdis is not None else float("nan"),
                                       bdis if bdis is not None else float("nan")))

    # ---- perspectives ----------------------------------------------------
    mut = next((t for t in curriculum.get("transitions", [])
                if t.get("from") == mutual_name), None)
    if mut is not None:
        add("perspectives", "each role reports the other's private meaning (mutual rung, "
            "per role)", 1.0, PRESENT, "passed per role at episode %s" % mut.get("episode"))
    elif swap is not None:
        add("perspectives", "each role describes and decodes (swap rung)", 1.0, PARTIAL,
            "the mutual rung was not passed")
    else:
        add("perspectives", "-", float("nan"), NOT_REACHED)

    # ---- cultural transmission ------------------------------------------
    nta = curriculum.get("newborn_token_accuracy") or {}
    fm = (nta.get("farmer") or {}).get("mean")
    bm = (nta.get("buyer") or {}).get("mean")
    vals = [x for x in (fm, bm) if x is not None]
    lo = min(vals) if vals else float("nan")
    add("cultural transmission", "newborns (random weights) learn the community's words "
        "from transcripts: mean token accuracy, worse role", lo,
        PRESENT if lo >= 0.5 else (PARTIAL if lo >= 0.2 else ABSENT),
        "farmer %s, buyer %s" % (("%.3f" % fm) if fm is not None else "n/a",
                                 ("%.3f" % bm) if bm is not None else "n/a"))

    # ---- duality of patterning -------------------------------------------
    if sem is not None:
        d = duality(cfg, sem)
        if not d["necessary"]:
            v = UNTESTABLE
            note = ("%d atoms for %d field values: each value can have an atom of its own, "
                    "so nothing pushes towards combining meaningless units"
                    % (d["atoms"], d["field_values_to_name"]))
        else:
            v = (PRESENT if d["multi_atom_meaningful_words"] >= 3
                 and d["atoms_reused_across_meanings"] >= 2 else
                 PARTIAL if d["multi_atom_meaningful_words"] >= 1 else ABSENT)
            note = ""
        add("duality of patterning", "meaningful multi-atom words, and atoms reused inside "
            "words of different meaning", d["multi_atom_meaningful_words"], v,
            (note + " " if note else "") + "%d meaningful words, %d multi-atom, %d atoms "
            "reused across meanings" % (d["meaningful_words"], d["multi_atom_meaningful_words"],
                                        d["atoms_reused_across_meanings"]))
    return out
