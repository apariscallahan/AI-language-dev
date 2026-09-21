"""Readable transcripts: for every round, what was expected, what was said, what happened.

Every round is written as the same three lines, whatever the rung:

    [ep 1,234,567] refer | farmer F0/g1 describes, buyer B1/g0 guesses
      expected : the farmer must name GOLD x3 HIGH so the buyer can pick it out of
                 1) RED x2 LOW  2) GOLD x3 HIGH  3) GOLD x5 HIGH  4) GREEN x3 MED   (answer: 2)
      dialogue : farmer: a3-a7 a1
      outcome  : the buyer picked 2) GOLD x3 HIGH -- CORRECT

What is "expected" differs by rung (a lineup, a pair of private meanings, an
order, a trade), so each rung has its own wording; the dialogue line is always
the words exactly as emitted (hyphens inside words, spaces between them); the
outcome line always ends in a verdict in capitals.

Written to ``<run>/transcripts.txt`` for every ``log.transcript_stride``-th
episode, and used for the example transcripts in ``run.log`` and the report.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from .config import Config
from .env import BUYER, FARMER, ROLE_NAMES


def _tuple(cfg: Config, t) -> str:
    """A thing, as (fruit, colour, quality).

    The middle field was printed as a quantity -- "PEAR x3 HIGH" -- for as long
    as the world has had colours, which made a colour round read as three
    quantities of the same fruit.
    """
    w = cfg.world
    v, c, u = int(t[0]), int(t[1]), int(t[2])
    vn = w.variety_names[v] if 0 <= v < w.n_varieties else "?"
    cn = w.color_names[c] if 0 <= c < w.n_colors else "?"
    un = w.quality_names[u] if 0 <= u < w.n_quality else "?"
    return "%s %s %s" % (cn, vn, un)


def _field_value(cfg: Config, name: str, v: int) -> str:
    """One field of a request, in words."""
    w = cfg.world
    if name == "fruit":
        return w.variety_names[v] if 0 <= v < w.n_varieties else "?"
    if name == "colour":
        return w.color_names[v] if 0 <= v < w.n_colors else "?"
    if name == "quality":
        return w.quality_names[v] if 0 <= v < w.n_quality else "?"
    if name in ("price", "reservation"):
        return _price(cfg, v)
    if name == "deal":
        return "worth doing" if v else "not worth it"
    return str(v)                      # quantity, stock


def _price(cfg: Config, b: int) -> str:
    pv = cfg.world.price_values
    return "%.2f" % pv[b] if 0 <= b < len(pv) else "?"


def _dialogue(cfg: Config, phase, tokens: Sequence[int]) -> str:
    from .render import render_message
    L = cfg.channel.max_msg_len
    parts = []
    for turn in range(min(phase.n_turns, cfg.channel.n_turns)):
        who = ROLE_NAMES[phase.speaker_of_turn(turn)]
        parts.append("%s: %s" % (who, render_message(cfg, tokens[turn * L:(turn + 1) * L])))
    return "  |  ".join(parts) if parts else "(nothing said)"


def _names(pop, batch, i: int) -> tuple[str, str]:
    try:
        f = pop.farmers[int(batch.f_idx[i])].name
        b = pop.buyers[int(batch.b_idx[i])].name
        return f, b
    except Exception:
        return "farmer", "buyer"


def _viability(cfg: Config, sc) -> str:
    """Why a trade is or is not possible, from the two private situations."""
    w = cfg.world
    f, b = sc.farmer, sc.buyer
    v, col = b.want_variety, b.want_color
    vn = "%s %s" % (w.color_names[col], w.variety_names[v])
    stock = f.stock_of(v, col)
    if stock <= 0:
        return "no deal is possible: the farmer has no %s" % vn
    if stock < b.need_qty:
        return "no deal is possible: the farmer has only %d %s" % (stock, vn)
    if f.quality_of(v, col) < b.min_quality:
        return ("no deal is possible: the farmer's %s is %s, below the buyer's minimum"
                % (vn, w.quality_names[f.quality_of(v, col)]))
    if f.reservation > b.max_price:
        return ("no deal is possible: the farmer's floor %s is above the buyer's ceiling %s"
                % (_price(cfg, f.reservation), _price(cfg, b.max_price)))
    return ("a deal is possible: %s x%d at %s-%s"
            % (vn, b.need_qty, _price(cfg, f.reservation), _price(cfg, b.max_price)))


def format_round(cfg: Config, phase, batch, i: int, *, pop=None,
                 episode: Optional[int] = None) -> list[str]:
    """Header plus the three lines (expected / dialogue / outcome) for round ``i``."""
    from .curriculum import H_CHOICE, H_REPORT, MutualBatch, ReferentialBatch
    w = cfg.world
    sb = batch.sb
    toks = [int(x) for x in batch.tokens[i]]
    fname, bname = _names(pop, batch, i) if pop is not None else ("farmer", "buyer")
    tag = "[ep %s] %s" % ("{:,}".format(episode) if episode is not None else "?", phase.name)
    dialogue = _dialogue(cfg, phase, toks)
    ok = bool(batch.res["success"][i]) if batch.res is not None else None

    if isinstance(sb, ReferentialBatch):
        inf = phase.informer
        speaker, guesser = (fname, bname) if inf == FARMER else (bname, fname)
        cands = sb.meanings[i]
        target = int(sb.target[i])
        lineup = "  ".join("%d) %s" % (k + 1, _tuple(cfg, cands[k]))
                           for k in range(cands.shape[0]))
        guess_row = batch.b_dec if phase.guesser == BUYER else batch.f_dec
        pick = int(guess_row[i, H_CHOICE])
        pick_txt = ("%d) %s" % (pick + 1, _tuple(cfg, cands[pick]))
                    if 0 <= pick < cands.shape[0] else "nothing")
        return [
            "%s | %s %s describes, %s %s guesses"
            % (tag, ROLE_NAMES[inf], speaker, ROLE_NAMES[phase.guesser], guesser),
            "  expected : the %s must name %s so the %s can pick it out of"
            % (ROLE_NAMES[inf], _tuple(cfg, cands[target]), ROLE_NAMES[phase.guesser]),
            "             %s   (answer: %d)" % (lineup, target + 1),
            "  dialogue : %s" % dialogue,
            "  outcome  : the %s picked %s -- %s"
            % (ROLE_NAMES[phase.guesser], pick_txt, "CORRECT" if ok else "WRONG"),
        ]

    if isinstance(sb, MutualBatch):
        rep = list(H_REPORT)
        fm, bm = sb.f_meaning[i], sb.b_meaning[i]
        fr = [int(x) for x in batch.f_dec[i, rep]]
        br = [int(x) for x in batch.b_dec[i, rep]]

        def verdict(fields) -> str:
            names = ("fruit", "colour", "quality")
            wrong = [n for n, good in zip(names, fields) if not bool(good)]
            return "right" if not wrong else "%s wrong" % " and ".join(wrong)
        ffields = batch.res["farmer_fields"][i].tolist() if "farmer_fields" in batch.res else [False] * 3
        bfields = batch.res["buyer_fields"][i].tolist() if "buyer_fields" in batch.res else [False] * 3
        return [
            "%s | farmer %s and buyer %s" % (tag, fname, bname),
            "  expected : the farmer holds %s and the buyer holds %s; each must report the other's"
            % (_tuple(cfg, fm), _tuple(cfg, bm)),
            "  dialogue : %s" % dialogue,
            "  outcome  : farmer reported %s (%s)  |  buyer reported %s (%s) -- %s"
            % (_tuple(cfg, fr), verdict(ffields), _tuple(cfg, br), verdict(bfields),
               "BOTH RIGHT" if ok else "ROUND FAILED"),
        ]

    # an order or a trade: a ScenarioBatch (or a list of Scenarios)
    sc = batch.scenario(i)
    b = sc.buyer
    want = "%s %s x%d" % (w.color_names[b.want_color],
                          w.variety_names[b.want_variety], b.need_qty)
    fd = [int(x) for x in batch.f_dec[i, :4]]
    bd = [int(x) for x in batch.b_dec[i, :4]]
    if getattr(phase, "order", False):
        # Whatever this rung asked for, and whatever came back: the fields differ
        # rung by rung (quantity, then fruit and colour, then price, and in
        # `offer` the farmer's own lot instead).
        from .curriculum import request_truth
        dec = batch.b_dec if phase.reporter == BUYER else batch.f_dec
        said, heard, wrong = [], [], []
        for name, head in phase.ask_heads.items():
            truth = int(request_truth(cfg, batch.sb, name)[i])
            got = int(dec[i, head])
            said.append("%s %s" % (name, _field_value(cfg, name, truth)))
            heard.append("%s %s" % (name, _field_value(cfg, name, got)))
            if got != truth:
                wrong.append(name)
        who = "buyer" if phase.reporter == BUYER else "farmer"
        teller = "farmer" if phase.reporter == BUYER else "buyer"
        return [
            "%s | %s %s tells, %s %s reports" % (
                tag, teller, bname if teller == "buyer" else fname,
                who, bname if who == "buyer" else fname),
            "  expected : %s" % ", ".join(said),
            "  dialogue : %s" % dialogue,
            "  outcome  : reported %s -- %s"
            % (", ".join(heard),
               "CORRECT" if ok else "WRONG (%s)" % " and ".join(wrong)),
        ]

    f = sc.farmer
    from .render import barn_text
    barn = barn_text(w, f)
    expected = ("the buyer wants %s, quality >= %s, pays at most %s; the farmer has %s and "
                "sells at no less than %s -> %s"
                % (want, w.quality_names[b.min_quality], _price(cfg, b.max_price), barn,
                   _price(cfg, f.reservation), _viability(cfg, sc)))

    def deal(d) -> str:
        vn = w.variety_names[d[1]] if 0 <= d[1] < w.n_varieties else "?"
        return "%s %s x%d at %s" % ("ACCEPT" if d[0] else "REJECT", vn, d[2], _price(cfg, d[3]))
    try:
        o = batch.outcome(i)
    except Exception:
        o = None
    if o is not None and o.success:
        result = ("TRADE: %d %s at %s (value %.2f; farmer profit %.2f, buyer saved %.2f)"
                  % (o.traded_qty, w.variety_names[o.traded_variety],
                     _price(cfg, o.traded_price_bin), o.trade_value,
                     o.farmer_profit, o.buyer_savings))
    else:
        result = "NO TRADE (%s)" % (o.failure_mode.replace("_", " ") if o is not None else "?")
    return [
        "%s | buyer %s meets farmer %s" % (tag, bname, fname),
        "  expected : %s" % expected,
        "  dialogue : %s" % dialogue,
        "  outcome  : farmer %s  |  buyer %s -- %s" % (deal(fd), deal(bd), result),
    ]


def pick_examples(batch, n: int) -> list[int]:
    """A few successes and a few failures from one batch, for summaries."""
    if batch.res is None:
        return list(range(min(n, len(batch))))
    succ = batch.res["success"].tolist()
    hits = [i for i, s in enumerate(succ) if s]
    miss = [i for i, s in enumerate(succ) if not s]
    k = max(1, n // 2)
    picks = hits[:k] + miss[:n - min(k, len(hits))]
    return picks[:n]


class TranscriptWriter:
    """Appends every ``stride``-th round of the run to ``transcripts.txt``."""

    def __init__(self, cfg: Config, out_dir: str):
        self.cfg = cfg
        self.stride = max(0, int(cfg.log.transcript_stride))
        self.path = os.path.join(out_dir, "transcripts.txt")
        self.fh = open(self.path, "a", encoding="utf-8") if self.stride else None
        self._rung: Optional[str] = None

    def write_batch(self, batch, phase, episode0: int, pop) -> None:
        if not self.fh:
            return
        first = (-episode0) % self.stride
        idx = range(first, len(batch), self.stride)
        if not len(idx):
            return
        if phase.name != self._rung:
            self._rung = phase.name
            self.fh.write("\n%s\nRUNG %s -- %s  (from episode %s)\n%s\n\n"
                          % ("=" * 78, phase.name, phase.blurb,
                             "{:,}".format(episode0), "=" * 78))
        for i in idx:
            try:
                lines = format_round(self.cfg, phase, batch, i, pop=pop,
                                     episode=episode0 + i)
            except Exception as exc:        # a transcript must never kill a run
                lines = ["[ep %s] (could not format: %s)" % (episode0 + i, exc)]
            self.fh.write("\n".join(lines) + "\n\n")
        self.fh.flush()

    def close(self) -> None:
        if self.fh:
            self.fh.close()
