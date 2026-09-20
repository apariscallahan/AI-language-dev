#!/usr/bin/env bash
# Orchard on a GPU box.
#
#   bash cloud_run.sh                                  # the reference scale
#   CONFIG=configs/gpu_community.json bash cloud_run.sh   # a bigger scale (see configs/)
#   CONFIG=configs/duality.json bash cloud_run.sh      # a named experiment
#   RUN=runs/<existing folder> bash cloud_run.sh       # resume that run
#   bash cloud_run.sh --seed 3                         # any orchard.run flags
#
# One method, at whatever scale the preset asks for: the presets in configs/
# change the community, the brain, the batch and the run length, never what is
# simulated. This script only checks the GPU is visible and picks a folder; the
# same run on a CPU (slower, same rules) is
#   python -m orchard.run [--config configs/<preset>.json]
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
    raise SystemExit("No CUDA device visible. Install a CUDA build of torch, or run "
                     "`python -m orchard.run` to use the CPU.")
print("gpu:", torch.cuda.get_device_name(0))
PY

# Fewer fragmentation OOMs with many differently-sized small tensors.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_ARGS=()
NAME=orchard
if [[ -n "${CONFIG:-}" ]]; then
    CONFIG_ARGS=(--config "$CONFIG")
    NAME="$(basename "$CONFIG" .json)"
fi
# A new run gets a folder named for its start time in UTC and what it is:
#   runs/2026-09-18_14-03-12UTC_orchard
# To resume an interrupted run, pass that folder: RUN=runs/<that folder>.
RUN="${RUN:-runs/$(date -u +%Y-%m-%d_%H-%M-%SUTC)_$NAME}"
RESUME=()
if [[ -f "$RUN/snapshots/latest.pt" ]]; then
    echo "found $RUN/snapshots/latest.pt -- resuming"
    RESUME=(--resume "$RUN/snapshots/latest.pt")
fi
echo "run -> $RUN"
exec python -m orchard.run "${CONFIG_ARGS[@]}" --out "$RUN" --device auto --quiet "${RESUME[@]}" "$@"
