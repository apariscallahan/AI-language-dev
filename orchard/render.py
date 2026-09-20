"""Human-readable rendering of token sequences and transcripts (spec 6.3).

Two rules, both from the spec:

* Token labels are **placeholders** (``tok7``).  We do not hand-assign meanings
  like "price" or "accept" -- meanings are supposed to be discovered, and naming
  them in advance would be assuming the conclusion.
* An *inferred* annotation is available (:func:`annotate_token`), but only from a
  post-hoc :class:`orchard.metrics.TokenSemantics` analysis, and it is always
  printed with a ``likely:`` hedge so it cannot be mistaken for ground truth.
  The raw ids in the ledger stay the authoritative record.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .config import Config
from .env import FARMER, BUYER, Transcript, parse_words, speaker_of_turn


def barn_text(w, f) -> str:
    """The barn as a human reads it: one entry per coloured lot that has stock."""
    parts = []
    for v in range(w.n_varieties):
        for c in range(w.n_colors):
            if f.stock_of(v, c) > 0:
                parts.append("%s %s x%d (%s)"
                             % (w.color_names[c], w.variety_names[v], f.stock_of(v, c),
                                w.quality_names[f.quality_of(v, c)]))
    return ", ".join(parts) or "nothing"


def token_label(cfg: Config, tok: int) -> str:
    """Placeholder names only.  ``a7`` is atom seven and nothing more."""
    c = cfg.channel
    if tok == c.end_id:
        return "<end>"
    if tok == c.pad_id:
        return "<pad>"
    if tok == c.hyphen_id:
        return "-"
    if tok == c.space_id:
        return "_"
    return "a%d" % tok


def render_message(cfg: Config, symbols: Iterable[int], *,
                   semantics: Optional["object"] = None) -> str:
    """Render a turn the way the agent structured it: words, hyphens and all.

    Words come from the agent's own SPACE marks; the atoms inside a word are
    joined with hyphens as the agent joined them.  Nothing is renamed or
    regrouped -- the raw symbol ids remain the authoritative record (spec 6.3).
    """
    words = parse_words(cfg, list(symbols))
    if not words:
        return "<silence>"
    out = []
    for w in words:
        text = "-".join("a%d" % t for t in w)
        if semantics is not None:
            hint = semantics.word_hint(w) if hasattr(semantics, "word_hint") else ""
            if hint:
                text = "%s(likely:%s)" % (text, hint)
        out.append(text)
    return " ".join(out)


def render_tokens(cfg: Config, tokens: Iterable[int], *, keep_pad: bool = False,
                  semantics: Optional["object"] = None) -> str:
    out = []
    for t in tokens:
        if t == cfg.channel.pad_id and not keep_pad:
            continue
        lbl = token_label(cfg, t)
        if semantics is not None and cfg.channel.is_atom(t):
            hint = semantics.hint(t)
            if hint:
                lbl = "%s(likely:%s)" % (lbl, hint)
        out.append(lbl)
    return " ".join(out) if out else "<silence>"


def render_decision(cfg: Config, d) -> str:
    if d is None:
        return "-"
    w = cfg.world
    verb = "ACCEPT" if d.accept else "REJECT"
    return "%s variety=%s qty=%d price=%.2f" % (
        verb, w.variety_names[d.variety] if d.variety < w.n_varieties else "?",
        d.qty, w.price_values[d.price] if d.price < w.n_price_bins else float("nan"))


def render_transcript(cfg: Config, tr: Transcript, *, semantics=None,
                      header: str = "", indent: str = "  ") -> str:
    """The full readable form of one episode, for console summaries and the report."""
    w = cfg.world
    sc = tr.scenario
    f, b = sc.farmer, sc.buyer
    lines = []
    if header:
        lines.append(header)
    lines.append("%sday %d  |  viable=%s  held_out=%s" % (indent, sc.day, sc.viable, sc.held_out))
    barn = barn_text(w, f)
    lines.append("%sFARMER sees: barn holds %s; will not sell below %.2f"
                 % (indent, barn, w.price_values[f.reservation]))
    lines.append("%sBUYER  sees: wants %s %s x%d, quality >= %s, cannot pay above %.2f" % (
        indent, w.color_names[b.want_color], w.variety_names[b.want_variety], b.need_qty,
        w.quality_names[b.min_quality], w.price_values[b.max_price]))
    lines.append("%s--- channel ---" % indent)
    for turn in range(cfg.channel.n_turns):
        role = speaker_of_turn(turn)
        who = "BUYER " if role == BUYER else "FARMER"
        utt = tr.utterance(cfg, turn)
        lines.append("%s  t%d %s: %s" % (
            indent, turn, who, render_message(cfg, utt, semantics=semantics)))
    lines.append("%s--- decisions ---" % indent)
    lines.append("%s  FARMER: %s" % (indent, render_decision(cfg, tr.farmer_decision)))
    lines.append("%s  BUYER : %s" % (indent, render_decision(cfg, tr.buyer_decision)))
    if tr.outcome is not None:
        o = tr.outcome
        tag = "SUCCESS" if o.success else ("-> " + o.failure_mode)
        extra = ""
        if o.success:
            extra = "  (%d apples of %s at %.2f = %.2f)" % (
                o.traded_qty, w.variety_names[o.traded_variety],
                w.price_values[o.traded_price_bin], o.trade_value)
        lines.append("%s  RESULT: %s%s   reward F=%.3f B=%.3f" % (
            indent, tag, extra, o.farmer_reward, o.buyer_reward))
    return "\n".join(lines)


def compact_transcript_row(cfg: Config, tr: Transcript) -> dict[str, Any]:
    """Message fields for the ledger: raw ids (authoritative) + readable rendering."""
    per_turn_raw = []
    per_turn_txt = []
    for turn in range(cfg.channel.n_turns):
        utt = tr.utterance(cfg, turn)
        per_turn_raw.append(utt)
        per_turn_txt.append("%s:%s" % (
            "B" if speaker_of_turn(turn) == BUYER else "F",
            render_message(cfg, utt)))
    return {
        "msg_symbols": per_turn_raw,
        "msg_words": [[list(w) for w in tr.words(cfg, t)]
                      for t in range(cfg.channel.n_turns)],
        "msg_text": " | ".join(per_turn_txt),
    }
