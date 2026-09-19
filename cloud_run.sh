#!/usr/bin/env bash
# Orchard on a cloud GPU box.
#
#   bash cloud_run.sh                                    # gpu_community preset
#   CONFIG=configs/gpu_full.json bash cloud_run.sh       # another preset
#   RUN=runs/<existing folder> bash cloud_run.sh         # resume that run
#   bash cloud_run.sh --set train.seed=3                 # any orchard.run flags
#
# Resuming is automatic: if $RUN/snapshots/latest.pt exists (the run was
# interrupted -- a pre-empted spot instance, a dropped SSH session), the run
# carries on from it instead of starting over. Snapshots are written at every
# checkpoint and at every rung promotion.
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| device count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible. Install a CUDA build of torch, or pass --device cpu.")
print("gpu:", torch.cuda.get_device_name(0),
      "| bf16:", torch.cuda.is_bf16_supported())
PY

# Fewer fragmentation OOMs with many differently-sized small tensors.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/gpu_community.json}"
# A new run gets a folder named for its start time in UTC and the preset:
#   runs/2026-09-18_14-03-12UTC_gpu_community
# To resume an interrupted run, pass that folder: RUN=runs/<that folder>.
RUN="${RUN:-runs/$(date -u +%Y-%m-%d_%H-%M-%SUTC)_$(basename "$CONFIG" .json)}"
RESUME=()
if [[ -f "$RUN/snapshots/latest.pt" ]]; then
    echo "found $RUN/snapshots/latest.pt -- resuming"
    RESUME=(--resume "$RUN/snapshots/latest.pt")
fi
echo "config $CONFIG -> $RUN"
exec python -m orchard.run --config "$CONFIG" --out "$RUN" --device auto --quiet "${RESUME[@]}" "$@"
