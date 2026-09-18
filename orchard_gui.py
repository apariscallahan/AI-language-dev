"""Orchard control panel -- a small Windows GUI for setting up and watching a run.

Everything here is a front end for ``python -m orchard.run``: it writes a config
file, launches the trainer as a separate process, and watches the ``progress.json``
that the trainer drops beside its output.  Nothing about the simulation lives in
this file, so the GUI can never disagree with the CLI about what a setting means.

Run it with::

    python orchard_gui.py

or double-click ``Orchard.bat``.

A note on "generations", since it is the one setting that is not what it looks
like.  You cannot set generations directly: an agent ages by the episodes *it*
personally plays, and dies at its lifespan, so how many times a lineage turns over
falls out of run length, population size and lifespan together.  The panel asks
for generations because that is the thing people actually want to choose, and
converts it to an episode budget for you -- the arithmetic is
:func:`orchard.train.episodes_for_generations`, and the derived number is always
shown so nothing is hidden.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from orchard.config import Config, validate                    # noqa: E402
from orchard.train import episodes_for_generations             # noqa: E402


# ==========================================================================
# presets
# ==========================================================================
# Every preset is a complete set of choices, so switching to one and pressing
# Start is always a sensible thing to do.  "Balanced" is the configuration the
# project's headline result came from.
PRESETS: dict[str, dict] = {
    "Balanced - about half an hour": {
        "blurb": "A sensible first run. Long enough for words to start attaching "
                 "to things, short enough to watch.",
        "generations": 3, "population": 3, "bottleneck": True, "turnover": True,
        "varieties": 3, "max_qty": 8, "atoms": 16, "symbols": 4, "turns": 4,
        "zipf": 0.3, "lifetime": 8000, "batch": 128, "lr": 3e-4,
        "symbol_cost": 0.012, "algo": "gumbel",
    },
    "Quick look - a few minutes": {
        "blurb": "Small and fast, for seeing the machinery work end to end. "
                 "Too short to expect much of a language.",
        "generations": 2, "population": 2, "bottleneck": True, "turnover": True,
        "varieties": 3, "max_qty": 6, "atoms": 12, "symbols": 3, "turns": 4,
        "zipf": 0.3, "lifetime": 2500, "batch": 128, "lr": 3e-4,
        "symbol_cost": 0.012, "algo": "gumbel",
    },
    "Headline - the project's main result": {
        "blurb": "Exactly the settings behind the reported result: four farms, "
                 "four buyers, four generations. Takes a while.",
        "generations": 4, "population": 4, "bottleneck": True, "turnover": True,
        "varieties": 3, "max_qty": 8, "atoms": 16, "symbols": 4, "turns": 4,
        "zipf": 0.3, "lifetime": 15000, "batch": 128, "lr": 3e-4,
        "symbol_cost": 0.012, "algo": "gumbel",
    },
    "Headline, no bottleneck - the ablation": {
        "blurb": "The Headline settings with the transmission bottleneck switched "
                 "off. Run both and compare: that contrast is the experiment.",
        "generations": 4, "population": 4, "bottleneck": False, "turnover": True,
        "varieties": 3, "max_qty": 8, "atoms": 16, "symbols": 4, "turns": 4,
        "zipf": 0.3, "lifetime": 15000, "batch": 128, "lr": 3e-4,
        "symbol_cost": 0.012, "algo": "gumbel",
    },
    "Full scale - the spec's world": {
        "blurb": "Four varieties, orders up to 20, a 36-symbol inventory. Much "
                 "harder: expect many hours and a weaker result.",
        "generations": 4, "population": 8, "bottleneck": True, "turnover": True,
        "varieties": 4, "max_qty": 20, "atoms": 36, "symbols": 6, "turns": 6,
        "zipf": 0.3, "lifetime": 15000, "batch": 192, "lr": 3e-4,
        "symbol_cost": 0.012, "algo": "gumbel",
    },
}
DEFAULT_PRESET = "Balanced - about half an hour"

# Throughput model, fitted to measurements on this machine.  Cost is dominated by
# how many symbol slots there are to generate one at a time, and rises with
# population because each agent in a batch needs its own forward pass.
#
#   slots  pop   measured      model
#      12    2   78.8 eps/s    78.5
#      16    4   50.5          51.0
#      36    8   16.3          17.4
#
# This is only the estimate shown before you press Start.  Once a run is going,
# the remaining time comes from the run's own measured rate.
_EPS_K = 1030.0
_EPS_CROWD = 0.093


def estimate_eps(symbols: int, turns: int, population: int, batch: int) -> float:
    slots = max(1, symbols * turns)
    return max(2.0, _EPS_K / (slots * (1.0 + _EPS_CROWD * max(0, population - 1))))


def checkpoint_every_for(episodes: int) -> int:
    """The cadence build_config will choose -- kept here so the estimate matches."""
    return max(2000, (episodes // 10 // 1000) * 1000 or 2000)


def estimate_seconds(episodes: int, symbols: int, turns: int, population: int,
                     batch: int, n_checkpoints: int | None = None) -> float:
    """Wall-clock estimate: training, plus startup, plus the metric suite.

    The metric suite is not free -- it replays episodes three times for the
    channel ablation and probes every agent for the vocabulary analyses -- so at
    short run lengths it is a real share of the total.
    """
    slots = max(1, symbols * turns)
    if n_checkpoints is None:
        n_checkpoints = max(1, episodes // checkpoint_every_for(episodes)) + 1
    training = episodes / estimate_eps(symbols, turns, population, batch)
    startup = 8.0
    per_checkpoint = 8.0 + 2.0 * population + slots / 3.0
    return training + startup + n_checkpoints * per_checkpoint


def human_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    if seconds < 90:
        return "%d sec" % seconds
    if seconds < 5400:
        return "%d min" % round(seconds / 60)
    return "%.1f hours" % (seconds / 3600.0)


# ==========================================================================
class OrchardGUI(ttk.Frame):
    POLL_MS = 400

    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=10)
        self.master = master
        self.proc: subprocess.Popen | None = None
        self.out_dir: str | None = None
        self.log_pos = 0
        self.started_at = 0.0
        self._loading_preset = False

        master.title("Orchard - emergent language in an apple-trading world")
        master.minsize(860, 730)
        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass

        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self._vars()
        self._build()
        self.apply_preset()

    # ------------------------------------------------------------------
    def _vars(self) -> None:
        self.v_preset = tk.StringVar(value=DEFAULT_PRESET)
        self.v_generations = tk.IntVar(value=4)
        self.v_population = tk.IntVar(value=4)
        self.v_bottleneck = tk.BooleanVar(value=True)
        self.v_turnover = tk.BooleanVar(value=True)
        self.v_varieties = tk.IntVar(value=3)
        self.v_max_qty = tk.IntVar(value=8)
        self.v_atoms = tk.IntVar(value=16)
        self.v_symbols = tk.IntVar(value=4)
        self.v_turns = tk.IntVar(value=4)
        self.v_zipf = tk.DoubleVar(value=0.3)
        self.v_lifetime = tk.IntVar(value=15000)
        self.v_batch = tk.IntVar(value=128)
        self.v_lr = tk.StringVar(value="0.0003")
        self.v_cost = tk.StringVar(value="0.012")
        self.v_seed = tk.IntVar(value=0)
        self.v_algo = tk.StringVar(value="gumbel")
        self.v_name = tk.StringVar(value="my_run")
        self.v_derived = tk.StringVar(value="")
        self.v_status = tk.StringVar(value="Ready.")
        self.v_metrics = tk.StringVar(value="")
        self.v_progress = tk.DoubleVar(value=0.0)
        self.v_pct = tk.StringVar(value="")

        for var in (self.v_generations, self.v_population, self.v_symbols,
                    self.v_turns, self.v_lifetime, self.v_batch, self.v_varieties,
                    self.v_max_qty, self.v_atoms):
            var.trace_add("write", lambda *_: self.refresh_estimate())
        self.v_turnover.trace_add("write", lambda *_: self.refresh_estimate())

    # ------------------------------------------------------------------
    def _build(self) -> None:
        row = 0
        head = ttk.Frame(self)
        head.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        head.columnconfigure(1, weight=1)
        ttk.Label(head, text="Preset", width=10).grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(head, textvariable=self.v_preset, state="readonly",
                             values=list(PRESETS))
        combo.grid(row=0, column=1, sticky="ew")
        combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_preset())
        self.lbl_blurb = ttk.Label(head, text="", foreground="#555", wraplength=800,
                                   justify="left")
        self.lbl_blurb.grid(row=1, column=1, sticky="w", pady=(3, 0))
        row += 1

        # ---- experiment ------------------------------------------------
        exp = ttk.LabelFrame(self, text="Experiment", padding=8)
        exp.grid(row=row, column=0, sticky="nsew", padx=(0, 5))
        exp.columnconfigure(1, weight=1)
        self._spin(exp, 0, "Generations", self.v_generations, 1, 20,
                   "How many times the whole population is replaced.")
        self._spin(exp, 1, "Farms / buyers", self.v_population, 1, 16,
                   "Agents of each kind alive at once.")
        ttk.Checkbutton(exp, text="Transmission bottleneck",
                        variable=self.v_bottleneck).grid(row=2, column=0, columnspan=2,
                                                         sticky="w", pady=(6, 0))
        ttk.Label(exp, text="Newborns learn from a small, frequency-skewed sample of "
                            "recent trades.", foreground="#666",
                  wraplength=360, justify="left").grid(row=3, column=0, columnspan=2,
                                                       sticky="w", padx=(20, 0))
        ttk.Checkbutton(exp, text="Population turnover",
                        variable=self.v_turnover).grid(row=4, column=0, columnspan=2,
                                                       sticky="w", pady=(6, 0))
        ttk.Label(exp, text="Agents age, die and are replaced by newcomers who must "
                            "learn the language.", foreground="#666",
                  wraplength=360, justify="left").grid(row=5, column=0, columnspan=2,
                                                       sticky="w", padx=(20, 0))

        # ---- world -----------------------------------------------------
        wld = ttk.LabelFrame(self, text="World and language", padding=8)
        wld.grid(row=row, column=1, sticky="nsew", padx=(5, 0))
        wld.columnconfigure(1, weight=1)
        self._spin(wld, 0, "Apple varieties", self.v_varieties, 2, 6)
        self._spin(wld, 1, "Largest order", self.v_max_qty, 4, 20)
        self._spin(wld, 2, "Atomic symbols", self.v_atoms, 8, 48,
                   "The closed inventory words are built from.")
        self._spin(wld, 3, "Symbols per turn", self.v_symbols, 2, 12,
                   "Hard cap on utterance length.")
        self._spin(wld, 4, "Turns per haggle", self.v_turns, 2, 10)
        self._scale(wld, 5, "Demand skew", self.v_zipf, 0.0, 1.0,
                    "How much a few orders dominate. Above ~0.5 agents stop "
                    "learning to talk and just guess the common case.")
        row += 1

        # ---- advanced --------------------------------------------------
        adv = ttk.LabelFrame(self, text="Training (leave alone unless you know)",
                             padding=8)
        adv.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for c in (1, 3, 5):
            adv.columnconfigure(c, weight=1)
        self._spin(adv, 0, "Episodes per lifetime", self.v_lifetime, 1000, 60000,
                   col=0, step=1000)
        self._spin(adv, 0, "Batch size", self.v_batch, 16, 512, col=2, step=16)
        self._entry(adv, 0, "Learning rate", self.v_lr, col=4)
        self._entry(adv, 1, "Symbol cost", self.v_cost, col=0)
        self._spin(adv, 1, "Random seed", self.v_seed, 0, 9999, col=2)
        ttk.Label(adv, text="Algorithm").grid(row=1, column=4, sticky="w", padx=(8, 4))
        ttk.Combobox(adv, textvariable=self.v_algo, state="readonly", width=12,
                     values=["gumbel", "reinforce"]).grid(row=1, column=5, sticky="ew")
        row += 1

        # ---- run -------------------------------------------------------
        run = ttk.LabelFrame(self, text="Run", padding=8)
        run.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        run.columnconfigure(1, weight=1)
        self.rowconfigure(row, weight=1)

        ttk.Label(run, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(run, textvariable=self.v_name).grid(row=0, column=1, sticky="ew",
                                                      padx=(4, 8))
        ttk.Label(run, textvariable=self.v_derived, foreground="#333").grid(
            row=0, column=2, sticky="e")

        btns = ttk.Frame(run)
        btns.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.btn_start = ttk.Button(btns, text="Start run", command=self.start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(btns, text="Stop", command=self.stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        self.btn_report = ttk.Button(btns, text="Open report", command=self.open_report,
                                     state="disabled")
        self.btn_report.pack(side="left", padx=(20, 0))
        self.btn_folder = ttk.Button(btns, text="Open folder", command=self.open_folder,
                                     state="disabled")
        self.btn_folder.pack(side="left", padx=6)

        self.bar = ttk.Progressbar(run, variable=self.v_progress, maximum=100.0)
        self.bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        line = ttk.Frame(run)
        line.grid(row=3, column=0, columnspan=3, sticky="ew")
        line.columnconfigure(0, weight=1)
        ttk.Label(line, textvariable=self.v_status).grid(row=0, column=0, sticky="w")
        ttk.Label(line, textvariable=self.v_pct).grid(row=0, column=1, sticky="e")
        ttk.Label(run, textvariable=self.v_metrics, foreground="#1a5",
                  font=("Segoe UI", 9, "bold")).grid(row=4, column=0, columnspan=3,
                                                     sticky="w", pady=(2, 6))

        run.rowconfigure(5, weight=1)
        wrap = ttk.Frame(run)
        wrap.grid(row=5, column=0, columnspan=3, sticky="nsew")
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        self.log = tk.Text(wrap, height=12, wrap="none", font=("Consolas", 9),
                           background="#fbfbfb", relief="solid", borderwidth=1)
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set, state="disabled")

    # ---- small widget helpers -----------------------------------------
    def _spin(self, parent, r, label, var, lo, hi, hint="", col=0, step=1):
        ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=(0, 4),
                                           pady=2)
        sp = ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=9,
                         increment=step, command=self.mark_custom)
        sp.grid(row=r, column=col + 1, sticky="ew", pady=2)
        sp.bind("<KeyRelease>", lambda _e: self.mark_custom())
        if hint:
            ttk.Label(parent, text=hint, foreground="#666", wraplength=340,
                      justify="left").grid(row=r, column=col, columnspan=2,
                                           sticky="w", padx=(20, 0))
            parent.grid_rowconfigure(r, minsize=0)
        return sp

    def _scale(self, parent, r, label, var, lo, hi, hint=""):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=2)
        box = ttk.Frame(parent)
        box.grid(row=r, column=1, sticky="ew")
        box.columnconfigure(0, weight=1)
        val = ttk.Label(box, width=4)
        sc = ttk.Scale(box, from_=lo, to=hi, variable=var, orient="horizontal",
                       command=lambda v: (val.configure(text="%.2f" % float(v)),
                                          self.mark_custom()))
        sc.grid(row=0, column=0, sticky="ew")
        val.grid(row=0, column=1, padx=(6, 0))
        val.configure(text="%.2f" % var.get())
        if hint:
            ttk.Label(parent, text=hint, foreground="#666", wraplength=340,
                      justify="left").grid(row=r + 1, column=0, columnspan=2,
                                           sticky="w", padx=(20, 0))

    def _entry(self, parent, r, label, var, col=0):
        ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=(8, 4))
        e = ttk.Entry(parent, textvariable=var, width=11)
        e.grid(row=r, column=col + 1, sticky="ew")
        e.bind("<KeyRelease>", lambda _e: self.mark_custom())
        return e

    # ------------------------------------------------------------------
    def mark_custom(self) -> None:
        if not self._loading_preset and self.v_preset.get() != "Custom":
            self.v_preset.set("Custom")
            self.lbl_blurb.configure(text="Your own settings.")
        self.refresh_estimate()

    def apply_preset(self) -> None:
        name = self.v_preset.get()
        p = PRESETS.get(name)
        if not p:
            return
        self._loading_preset = True
        try:
            self.v_generations.set(p["generations"])
            self.v_population.set(p["population"])
            self.v_bottleneck.set(p["bottleneck"])
            self.v_turnover.set(p["turnover"])
            self.v_varieties.set(p["varieties"])
            self.v_max_qty.set(p["max_qty"])
            self.v_atoms.set(p["atoms"])
            self.v_symbols.set(p["symbols"])
            self.v_turns.set(p["turns"])
            self.v_zipf.set(p["zipf"])
            self.v_lifetime.set(p["lifetime"])
            self.v_batch.set(p["batch"])
            self.v_lr.set(str(p["lr"]))
            self.v_cost.set(str(p["symbol_cost"]))
            self.v_algo.set(p["algo"])
            self.lbl_blurb.configure(text=p["blurb"])
        finally:
            self._loading_preset = False
        self.refresh_estimate()

    def episodes(self) -> int:
        try:
            gens = max(1, int(self.v_generations.get()))
            pop = max(1, int(self.v_population.get()))
            life = max(500, int(self.v_lifetime.get()))
        except (tk.TclError, ValueError):
            return 0
        return max(int(self.v_batch.get() or 128),
                   episodes_for_generations(gens, pop, life))

    def refresh_estimate(self) -> None:
        eps = self.episodes()
        if not eps:
            self.v_derived.set("")
            return
        try:
            secs = estimate_seconds(eps, int(self.v_symbols.get()),
                                    int(self.v_turns.get()),
                                    int(self.v_population.get()),
                                    int(self.v_batch.get()))
        except (tk.TclError, ValueError):
            return
        note = "" if self.v_turnover.get() else "  (turnover off - no generations)"
        self.v_derived.set("~%s episodes, roughly %s%s"
                           % ("{:,}".format(eps), human_time(secs), note))

    # ------------------------------------------------------------------
    def build_config(self) -> Config:
        cfg = Config()
        cfg.name = self.v_name.get().strip() or "my_run"
        w = cfg.world
        w.n_varieties = int(self.v_varieties.get())
        w.max_qty = int(self.v_max_qty.get())
        w.n_price_bins = 8 if w.max_qty <= 10 else 12
        w.reservation_max_bin = w.n_price_bins - 2
        w.budget_min_bin = 1
        w.zipf_alpha = round(float(self.v_zipf.get()), 3)
        w.zipf_alpha_variety = 0.0

        c = cfg.channel
        c.atomic_vocab = int(self.v_atoms.get())
        c.max_symbols = int(self.v_symbols.get())
        c.n_turns = int(self.v_turns.get())

        cfg.model.d_model = 48 if w.max_qty <= 10 else 64
        cfg.model.d_ff = cfg.model.d_model * 2

        p = cfg.population
        p.n_farmers = p.n_buyers = int(self.v_population.get())
        p.turnover = bool(self.v_turnover.get())
        life = int(self.v_lifetime.get())
        p.lifespan_min = max(500, int(life * 0.7))
        p.lifespan_max = max(p.lifespan_min + 1, int(life * 1.3))

        cfg.bottleneck.enabled = bool(self.v_bottleneck.get())
        cfg.bottleneck.n_samples = 400

        t = cfg.train
        t.algo = self.v_algo.get()
        t.episodes = self.episodes()
        t.batch_size = int(self.v_batch.get())
        t.lr = float(self.v_lr.get())
        t.seed = int(self.v_seed.get())
        t.gumbel_mix_reinforce = 0.1
        t.torch_threads = max(1, (os.cpu_count() or 4))

        cfg.reward.symbol_cost = float(self.v_cost.get())
        if w.max_qty > 10:
            cfg.reward.qty_tol = 1

        cfg.log.checkpoint_every = checkpoint_every_for(t.episodes)
        cfg.log.summary_every = cfg.log.checkpoint_every
        validate(cfg)
        return cfg

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.proc is not None:
            return
        try:
            cfg = self.build_config()
        except Exception as exc:
            messagebox.showerror("Those settings do not work", str(exc))
            return

        out = os.path.join(HERE, "runs", cfg.name)
        if os.path.exists(os.path.join(out, "trades.jsonl")):
            if not messagebox.askyesno(
                    "Name already used",
                    "A run called %r already exists.\n\nIts files will be added to "
                    "rather than replaced, which makes the results hard to read.\n\n"
                    "Carry on anyway?" % cfg.name):
                return
        os.makedirs(out, exist_ok=True)
        cfg_path = os.path.join(out, "gui_config.json")
        cfg.to_json(cfg_path)

        self.out_dir = out
        self.log_pos = 0
        self.started_at = time.time()
        self._clear_log()
        self._append("Starting %s - %s episodes, %d farms, %d buyers, %s generations."
                     % (cfg.name, "{:,}".format(cfg.train.episodes),
                        cfg.population.n_farmers, cfg.population.n_buyers,
                        self.v_generations.get()))
        self._append("Bottleneck %s, turnover %s, %d atoms, %d symbols per turn.\n"
                     % ("on" if cfg.bottleneck.enabled else "OFF",
                        "on" if cfg.population.turnover else "OFF",
                        cfg.channel.atomic_vocab, cfg.channel.max_symbols))

        cmd = [sys.executable, "-m", "orchard.run", "--config", cfg_path,
               "--out", out, "--quiet"]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, creationflags=flags)
        except Exception as exc:
            messagebox.showerror("Could not start the run", str(exc))
            self.proc = None
            return
        threading.Thread(target=self._drain, args=(self.proc,), daemon=True).start()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_folder.configure(state="normal")
        self.btn_report.configure(state="disabled")
        self.v_progress.set(0.0)
        self.v_status.set("Warming up...")
        self.v_metrics.set("")
        self.after(self.POLL_MS, self.poll)

    def _drain(self, proc) -> None:
        """Keep the pipe empty so the child never blocks; surface any crash."""
        tail: list[str] = []
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    tail.append(line)
                    del tail[:-40]
        except Exception:
            pass
        self._crash_tail = "\n".join(tail)

    def stop(self) -> None:
        if self.proc is None:
            return
        if not messagebox.askyesno(
                "Stop the run?",
                "The run will stop where it is.\n\nWhatever it has produced so far "
                "is kept, including a report from the last checkpoint."):
            return
        self.v_status.set("Stopping...")
        try:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def poll(self) -> None:
        if self.out_dir is None:
            return
        self._read_progress()
        self._tail_log()

        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            self.proc = None
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            report = os.path.join(self.out_dir, "report.md")
            if os.path.exists(report):
                self.btn_report.configure(state="normal")
            if code == 0:
                self.v_progress.set(100.0)
                self.v_pct.set("100%")
                self.v_status.set("Finished in %s." % human_time(time.time() - self.started_at))
                self._append("\nDone. The report is in %s" % report)
                self._show_verdict()
            else:
                self.v_status.set("Stopped (exit code %s)." % code)
                tail = getattr(self, "_crash_tail", "")
                if tail and code not in (1, -1):
                    self._append("\n" + tail)
            return
        self.after(self.POLL_MS, self.poll)

    def _read_progress(self) -> None:
        path = os.path.join(self.out_dir or "", "progress.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                p = json.load(fh)
        except Exception:
            return
        frac = float(p.get("fraction", 0.0))
        self.v_progress.set(100.0 * frac)
        self.v_pct.set("%d%%" % round(100 * frac))
        state = p.get("state", "running")
        if state == "analysing":
            self.v_status.set("Episodes done - writing the final analysis...")
        else:
            eta = p.get("eta_seconds")
            self.v_status.set(
                "Episode %s of %s | %s/sec | generation %d | %s left"
                % ("{:,}".format(p.get("episode", 0)),
                   "{:,}".format(p.get("total_episodes", 0)),
                   p.get("episodes_per_second", 0), p.get("generation", 0),
                   human_time(eta if eta is not None else float("nan"))))
        h = p.get("headline") or {}
        if h:
            bits = ["variety naming %.2f" % h["variety_naming"]]
            if h.get("channel_transfer") is not None:
                bits.append("channel %d%%" % round(100 * h["channel_transfer"]))
            bits.append("success %.2f" % h.get("success", 0.0))
            if h.get("topsim") is not None:
                bits.append("topsim %.2f" % h["topsim"])
            bits.append("%d words" % h.get("distinct_words", 0))
            self.v_metrics.set("last checkpoint:  " + " | ".join(bits))

    def _tail_log(self) -> None:
        path = os.path.join(self.out_dir or "", "run.log")
        try:
            size = os.path.getsize(path)
            if size < self.log_pos:
                self.log_pos = 0
            if size == self.log_pos:
                return
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self.log_pos)
                chunk = fh.read()
                self.log_pos = fh.tell()
        except Exception:
            return
        if chunk.strip():
            self._append(chunk.rstrip())

    def _show_verdict(self) -> None:
        path = os.path.join(self.out_dir or "", "report.md")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("> **Verdict"):
                        self.v_metrics.set(line.strip("> *\n"))
                        return
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _open(self, path: str) -> None:
        if not os.path.exists(path):
            messagebox.showinfo("Not there yet", "%s does not exist yet." % path)
            return
        try:
            os.startfile(path)                                   # noqa: S606
        except AttributeError:
            subprocess.Popen(["xdg-open", path])

    def open_folder(self) -> None:
        if self.out_dir:
            self._open(self.out_dir)

    def open_report(self) -> None:
        if self.out_dir:
            self._open(os.path.join(self.out_dir, "report.md"))

    def on_close(self) -> None:
        if self.proc is not None:
            if not messagebox.askyesno("A run is going",
                                       "Closing will stop the run. Close anyway?"):
                return
            self.stop()
        self.master.destroy()


def main() -> int:
    root = tk.Tk()
    app = OrchardGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
