#!/usr/bin/env bash
# Orchard on a cloud GPU box.  Usage:  bash cloud_run.sh [extra orchard.run args]
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| device count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible. Install a CUDA build of torch, or pass --device cpu.")
print("gpu:", torch.cuda.get_device_name(0))
PY

RUN="${RUN:-runs/gpu_$(date +%Y%m%d_%H%M%S)}"
echo "writing to $RUN"
exec python -m orchard.run --config configs/gpu.json --out "$RUN" --device auto "$@"
