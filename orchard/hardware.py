"""Device selection and the handful of global torch switches worth setting.

Kept in one place so that "where does this run" is answered once, and so that the
same command works unchanged on a laptop and on a cloud GPU box.
"""
from __future__ import annotations

import os

import torch

from .config import Config


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "device %r was requested but torch reports no CUDA device.\n"
            "This build is %s. On a cloud box install a CUDA build of torch, or "
            "pass --device cpu." % (spec, torch.__version__))
    return torch.device(spec)


def setup(cfg: Config) -> torch.device:
    """Pick the device and set the global knobs.  Returns the device."""
    dev = resolve_device(cfg.train.device)
    torch.set_num_threads(max(1, cfg.train.torch_threads))
    # Full fp32 matmuls on every device. TF32 (Ampere and later) would round
    # GPU arithmetic to 10 bits of mantissa where a CPU uses 23; pinning it off
    # keeps a CPU check and a GPU run computing the same thing.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dev


def describe(dev: torch.device, cfg: Config) -> str:
    bits = ["device %s" % dev]
    if dev.type == "cuda":
        i = dev.index or 0
        props = torch.cuda.get_device_properties(i)
        bits.append("%s, %.0f GB, capability %d.%d"
                    % (props.name, props.total_memory / 1e9, props.major, props.minor))
    bits.append("%d CPU threads" % cfg.train.torch_threads)
    bits.append("fp32 throughout (the same arithmetic on every device)")
    if cfg.train.grad_checkpoint:
        bits.append("gradient checkpointing")
    return ", ".join(bits)
