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
    if dev.type == "cpu":
        torch.set_num_threads(max(1, cfg.train.torch_threads))
    else:
        # One process owning one GPU: the default allocator behaviour is right,
        # but TF32 is off by default and is free accuracy-for-speed on these
        # tiny matmuls.
        if cfg.train.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return dev


def autocast(cfg: Config, dev: torch.device):
    """bfloat16 autocast when asked for and supported, else a no-op context."""
    if not cfg.train.amp:
        return torch.autocast(device_type=dev.type, enabled=False)
    if dev.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    try:
        if torch.backends.cpu.is_bf16_supported():
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    except Exception:
        pass
    return torch.autocast(device_type=dev.type, enabled=False)


def describe(dev: torch.device, cfg: Config) -> str:
    bits = ["device %s" % dev]
    if dev.type == "cuda":
        i = dev.index or 0
        props = torch.cuda.get_device_properties(i)
        bits.append("%s, %.0f GB, capability %d.%d"
                    % (props.name, props.total_memory / 1e9, props.major, props.minor))
        if cfg.train.tf32:
            bits.append("tf32 on")
    else:
        bits.append("%d threads" % cfg.train.torch_threads)
    if cfg.train.amp:
        bits.append("bf16 autocast on the transformer layers" if dev.type == "cuda"
                    else "amp requested but ignored on CPU (fp32)")
    if cfg.train.grad_checkpoint:
        bits.append("gradient checkpointing")
    return ", ".join(bits)
