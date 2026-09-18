"""Dependency-free SVG line charts.

Insurance, not preference: matplotlib is the intended plotting path, but it
loads a compiled extension that this machine's application-control policy has
been observed to block intermittently.  Plots are a required deliverable, so
when matplotlib cannot be imported the same panels are written as hand-rolled
SVG instead.  Pure stdlib, no compiled code, nothing to block.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Sequence

_COLOURS = ["#2f6fdb", "#d1495b", "#2a9d8f", "#e9a13b", "#7d5ba6", "#4c4c4c"]


def _finite(xs, ys):
    ox, oy = [], []
    for x, y in zip(xs, ys):
        if y is None or y != y:
            continue
        ox.append(float(x))
        oy.append(float(y))
    return ox, oy


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _panel_svg(x0: float, y0: float, w: float, h: float, title: str,
               series: Sequence[tuple[str, Sequence[float], Sequence[float]]],
               ylim: Optional[tuple[float, float]] = None,
               hline: Optional[tuple[float, str]] = None) -> list[str]:
    pad_l, pad_b, pad_t, pad_r = 46.0, 30.0, 26.0, 8.0
    px, py = x0 + pad_l, y0 + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b

    xs_all = [v for _, xs, _ in series for v in xs]
    ys_all = [v for _, _, ys in series for v in ys]
    if hline and hline[0] == hline[0]:
        ys_all.append(hline[0])
    out: list[str] = []
    out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#ffffff" '
               'stroke="#d8d8d8"/>' % (px, py, pw, ph))
    out.append('<text x="%.1f" y="%.1f" font-size="11" font-family="sans-serif" '
               'font-weight="600" fill="#222">%s</text>' % (x0 + pad_l, y0 + 16, _esc(title)))
    if not xs_all or not ys_all:
        out.append('<text x="%.1f" y="%.1f" font-size="10" font-family="sans-serif" '
                   'fill="#999">no data yet</text>' % (px + 8, py + 20))
        return out

    xmin, xmax = min(xs_all), max(xs_all)
    ymin, ymax = (ylim if ylim else (min(ys_all), max(ys_all)))
    if xmax - xmin < 1e-12:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-12:
        ymax = ymin + 1.0
    span = ymax - ymin
    if not ylim:
        ymin -= 0.06 * span
        ymax += 0.06 * span

    def sx(v: float) -> float:
        return px + (v - xmin) / (xmax - xmin) * pw

    def sy(v: float) -> float:
        return py + ph - (v - ymin) / (ymax - ymin) * ph

    for i in range(5):
        gv = ymin + (ymax - ymin) * i / 4.0
        gy = sy(gv)
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#eee" '
                   'stroke-width="1"/>' % (px, gy, px + pw, gy))
        out.append('<text x="%.1f" y="%.1f" font-size="8" font-family="sans-serif" '
                   'fill="#777" text-anchor="end">%.2f</text>' % (px - 4, gy + 3, gv))
    for i in range(4):
        gv = xmin + (xmax - xmin) * i / 3.0
        gx = sx(gv)
        out.append('<text x="%.1f" y="%.1f" font-size="8" font-family="sans-serif" '
                   'fill="#777" text-anchor="middle">%s</text>'
                   % (gx, py + ph + 13, _fmt_int(gv)))

    if hline and hline[0] == hline[0] and ymin <= hline[0] <= ymax:
        hy = sy(hline[0])
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#999" '
                   'stroke-width="1" stroke-dasharray="4 3"/>' % (px, hy, px + pw, hy))
        out.append('<text x="%.1f" y="%.1f" font-size="8" font-family="sans-serif" '
                   'fill="#999">%s</text>' % (px + 3, hy - 3, _esc(hline[1])))

    ly = py + 10
    for i, (label, xs, ys) in enumerate(series):
        col = _COLOURS[i % len(_COLOURS)]
        if not xs:
            continue
        pts = " ".join("%.2f,%.2f" % (sx(a), sy(b)) for a, b in zip(xs, ys))
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.7"/>'
                   % (pts, col))
        for a, b in zip(xs, ys):
            out.append('<circle cx="%.2f" cy="%.2f" r="2" fill="%s"/>' % (sx(a), sy(b), col))
        out.append('<rect x="%.1f" y="%.1f" width="9" height="3" fill="%s"/>'
                   % (px + pw - 150, ly - 3, col))
        out.append('<text x="%.1f" y="%.1f" font-size="8" font-family="sans-serif" '
                   'fill="#444">%s</text>' % (px + pw - 138, ly + 1, _esc(label)))
        ly += 11
    return out


def _fmt_int(v: float) -> str:
    n = int(round(v))
    if abs(n) >= 1_000_000:
        return "%.1fM" % (n / 1e6)
    if abs(n) >= 1000:
        return "%dk" % (n // 1000)
    return str(n)


def write_svg_grid(path: str, panels: Sequence[dict[str, Any]], *, cols: int = 3,
                   panel_w: float = 380, panel_h: float = 230,
                   title: str = "") -> str:
    rows = (len(panels) + cols - 1) // cols
    W = cols * panel_w + 16
    H = rows * panel_h + 40
    body: list[str] = []
    body.append('<rect width="%.0f" height="%.0f" fill="#fafafa"/>' % (W, H))
    if title:
        body.append('<text x="12" y="20" font-size="13" font-family="sans-serif" '
                    'font-weight="700" fill="#111">%s</text>' % _esc(title))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        body.extend(_panel_svg(8 + c * panel_w, 30 + r * panel_h, panel_w, panel_h,
                               p["title"], p["series"], p.get("ylim"), p.get("hline")))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" '
           'viewBox="0 0 %.0f %.0f">%s</svg>' % (W, H, W, H, "".join(body)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
    return path


def build_panels(history, chance: Optional[float] = None) -> list[dict[str, Any]]:
    h = history
    x = h.episodes

    def S(label, ys):
        cx, cy = _finite(x, ys)
        return (label, cx, cy)

    return [
        {"title": "Task success rate (5.1)", "ylim": (0, 1),
         "hline": (chance, "chance") if chance is not None else None,
         "series": [S("success (eval)", h.eval_success),
                    S("success on viable", h.eval_success_viable),
                    S("mutual comprehension", getattr(h, "comprehension", [])),
                    S("viability judged", getattr(h, "judgement", []))]},
        {"title": "Compositionality / topsim (5.2)",
         "series": [S("topsim", h.topsim), S("shuffled null", h.topsim_null)]},
        {"title": "Vocabulary usage (5.3)",
         "series": [S("token entropy (bits)", h.entropy),
                    S("mean message length", h.msg_len)]},
        {"title": "Stability of the mapping (5.4)", "ylim": (0, 1),
         "series": [S("drift between checkpoints", h.drift),
                    S("population coherence", h.coherence)]},
        {"title": "Transmission (5.5) & channel ablation", "ylim": (0, 1.4),
         "series": [S("newcomer / veteran success", h.transmission),
                    S("comprehension lost when scrambled",
                      getattr(h, "ablation_drop", []))]},
        {"title": "Zero-shot generalisation (5.6)", "ylim": (0, 1.4),
         "series": [S("held-out / seen success", h.zeroshot)]},
    ]


def build_vocab_panels(history) -> list[dict[str, Any]]:
    """The open-vocabulary series (addendum section 3)."""
    h = history
    x = h.episodes

    def S(label, ys):
        cx, cy = _finite(x, ys)
        return (label, cx, cy)

    return [
        {"title": "Utterance length",
         "series": [S("symbols per utterance", getattr(h, "mean_symbols", [])),
                    S("words per utterance", getattr(h, "mean_words", []))]},
        {"title": "Vocabulary size and word formation",
         "series": [S("distinct words", getattr(h, "distinct_words", [])),
                    S("multi-atom word share", getattr(h, "multi_atom_share", []))]},
        {"title": "Length vs meaning frequency (3.1)",
         "series": [S("Spearman rho (negative = Zipfian)",
                      getattr(h, "rho_length_frequency", []))]},
        {"title": "Compositionality by bucket (3.3)",
         "series": [S("frequent meanings", getattr(h, "topsim_frequent", [])),
                    S("rare meanings", getattr(h, "topsim_rare", []))]},
        {"title": "Form drift by bucket (3.4)", "ylim": (0, 1),
         "series": [S("frequent meanings", getattr(h, "drift_frequent", [])),
                    S("rare meanings", getattr(h, "drift_rare", []))]},
        {"title": "Information actually carried by the channel",
         "ylim": (0, 1),
         "series": [S("full comprehension", getattr(h, "information_transfer", [])),
                    S("variety naming", getattr(h, "variety_transfer", []))]},
        {"title": "Reference accuracy (can the farmer name it?)", "ylim": (0, 1),
         "series": [S("variety", getattr(h, "variety_acc", [])),
                    S("quantity", getattr(h, "qty_acc", []))]},
    ]
