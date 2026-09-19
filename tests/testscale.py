"""The method -- the code defaults -- at a size a CPU test can afford.

Only keys a preset may change (``orchard.config.PRESET_KEYS``) are touched, so
every test exercises the same method a GPU run uses. A test that needs a
different world or channel says so itself, on top of this.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchard.config import Config  # noqa: E402


def method_at_test_scale() -> Config:
    cfg = Config()
    cfg.population.n_farmers = cfg.population.n_buyers = 8
    cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads, cfg.model.d_ff = 64, 2, 4, 128
    cfg.train.batch_size = 64
    cfg.train.episodes = 200_000
    cfg.train.rung_batch_scale = {}
    cfg.train.grad_checkpoint = False     # same numbers, slower on CPU (tested separately)
    cfg.bottleneck.batch_size = 256
    cfg.log.ledger_stride = 1
    cfg.log.intelligibility_episodes = 400
    cfg.log.zeroshot_episodes = 600
    cfg.log.ablation_episodes = 600
    cfg.log.transcript_stride = 50
    return cfg
