"""Progress plots, written as PNGs at every checkpoint (spec 8/9).

Saved during the run, not at the end, so a long run can be watched from the
filesystem without re-running any analysis.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from .plots_svg import build_panels, build_vocab_panels, write_svg_grid

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:            # compiled extension blocked / not installed
    plt = None
    HAVE_MPL = False


def _clean(xs: Sequence[float], ys: Sequence[float]) -> tuple[list[float], list[float]]:
    ox, oy = [], []
    for x, y in zip(xs, ys):
        if y is None or y != y:
            continue
        ox.append(x)
        oy.append(y)
    return ox, oy


def _panel(ax, x, series, title, ylabel, ylim=None, hlines=()):
    any_drawn = False
    for label, ys, style in series:
        cx, cy = _clean(x, ys)
        if not cx:
            continue
        ax.plot(cx, cy, style, label=label, linewidth=1.6, markersize=3)
        any_drawn = True
    for y, label, style in hlines:
        if y is None or y != y:
            continue
        ax.axhline(y, linestyle="--", linewidth=1.0, color="0.55")
        ax.annotate(label, xy=(0.01, y), xycoords=("axes fraction", "data"),
                    fontsize=7, color="0.35", va="bottom")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("episode", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.5)
    if ylim:
        ax.set_ylim(*ylim)
    if any_drawn:
        ax.legend(fontsize=7, framealpha=0.85)


def write_plots(history, out_dir: str, chance: Optional[float] = None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    h = history
    x = h.episodes
    if not x:
        return []
    # The SVG grid is always written; it needs nothing beyond the standard
    # library, so the plots deliverable survives a matplotlib failure.
    written: list[str] = [
        write_svg_grid(os.path.join(out_dir, "metrics.svg"),
                       build_panels(history, chance),
                       title="Orchard run: emergence metrics over training"),
        write_svg_grid(os.path.join(out_dir, "vocabulary.svg"),
                       build_vocab_panels(history),
                       title="Orchard run: open-vocabulary metrics"),
    ]
    if not HAVE_MPL:
        return written

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _panel(axes[0][0], x,
           [("success (eval)", h.eval_success, "-o"),
            ("success (rolling train)", h.train_success, "-"),
            ("success on viable deals", h.eval_success_viable, "-"),
            ("mutual comprehension", getattr(h, "comprehension", []), "--"),
            ("viability judged", getattr(h, "judgement", []), ":")],
           "Task success rate (spec 5.1)", "fraction of episodes", (0, 1),
           hlines=((chance, "chance", "--"),))
    _panel(axes[0][1], x,
           [("topsim", h.topsim, "-o"), ("shuffled null", h.topsim_null, "-")],
           "Compositionality / topological similarity (5.2)", "Spearman rho")
    _panel(axes[0][2], x,
           [("token entropy (bits)", h.entropy, "-o"),
            ("mean message length", h.msg_len, "-s")],
           "Vocabulary usage (5.3)", "bits / tokens")
    _panel(axes[1][0], x,
           [("drift between checkpoints", h.drift, "-o"),
            ("population coherence", h.coherence, "-s")],
           "Stability of the meaning->message map (5.4)", "normalised edit distance", (0, 1))
    _panel(axes[1][1], x,
           [("newcomer / veteran success", h.transmission, "-o"),
            ("comprehension lost when channel scrambled",
             getattr(h, "ablation_drop", []), "--s")],
           "Transmission (5.5) and channel ablation", "ratio / drop", (0, 1.4))
    _panel(axes[1][2], x,
           [("held-out / seen success", h.zeroshot, "-o")],
           "Zero-shot generalisation (5.6)", "ratio", (0, 1.4))
    fig.suptitle("Orchard run: emergence metrics over training", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out_dir, "metrics.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _panel(ax, x, [("mean episode reward", h.reward, "-o"),
                   ("mean generation", h.generations, "-s")],
           "Reward and population turnover", "value")
    fig.tight_layout()
    p = os.path.join(out_dir, "reward_and_generations.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)
    return written


def write_comparison(runs: dict[str, Any], out_path: str) -> str:
    """Overlay several runs' histories -- this is the bottleneck/turnover experiment."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not HAVE_MPL:
        panels = []
        for key, title, ylim in (("variety_acc", "Farmer names the right variety", (0, 1)),
                                 ("information_transfer",
                                  "Information carried by the channel", (0, 1)),
                                 ("topsim", "Compositionality (topsim)", None)):
            series = []
            for name, hh in runs.items():
                cx, cy = _clean(hh.get("episodes", []), hh.get(key, []))
                series.append((name, cx, cy))
            panels.append({"title": title, "series": series, "ylim": ylim})
        return write_svg_grid(out_path.replace(".png", ".svg"), panels,
                              title="Ablation comparison")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    keys = [("variety_acc", "Farmer names the right variety", (0, 1)),
            ("information_transfer", "Information carried by the channel", (0, 1)),
            ("topsim", "Compositionality (topsim)", None)]
    for ax, (key, title, ylim) in zip(axes, keys):
        for name, h in runs.items():
            xs = h.get("episodes", [])
            ys = h.get(key, [])
            cx, cy = _clean(xs, ys)
            if cx:
                ax.plot(cx, cy, "-o", label=name, linewidth=1.6, markersize=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("episode", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7)
    fig.suptitle("Ablation comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
