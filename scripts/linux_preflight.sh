#!/usr/bin/env bash
# Linux CPU preflight of the exact Kaggle upload ZIP (docs/experiment_workflow_standard.md, pre-upload checklist).
#
# Usage: scripts/linux_preflight.sh OVERLAY_ZIP HANDOFF_JSON BASE_ZIP_OR_DIR OUTPUT_DIR [exp038|exp039]
# Needs Docker (or run scripts/linux_preflight.py directly inside WSL / a Linux machine with the CPU lock).
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 OVERLAY_ZIP HANDOFF_JSON BASE_ZIP_OR_DIR OUTPUT_DIR [exp038|exp039]" >&2
  exit 2
fi
for path in "$1" "$2" "$3"; do
  [ -e "$path" ] || { echo "missing: $path" >&2; exit 2; }
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mkdir -p "$4" && cd "$4" && pwd)"
IMAGE="python:3.12.10-slim-bookworm"
abs() { (cd "$(dirname "$1")" && printf '%s/%s' "$(pwd)" "$(basename "$1")"); }
OVERLAY="$(abs "$1")"; HANDOFF="$(abs "$2")"; BASE="$(abs "$3")"; EXPERIMENT="${5:-exp038}"

docker run --rm --cpus 2 -e OPENBLAS_CORETYPE=Haswell -e PYTHONDONTWRITEBYTECODE=1 \
  -e EXPERIMENT="$EXPERIMENT" -v "$REPO":/repo:ro -v "$OVERLAY":/in/overlay.zip:ro -v "$HANDOFF":/in/handoff.json:ro \
  -v "$BASE":/in/base:ro -v "$OUT":/out "$IMAGE" bash -euo pipefail -c '
    pip install --quiet --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      -r /repo/requirements-exp038-cpu-preflight.txt
    base=/in/base; [ -d /in/base/exp037_inputs ] && base=/in/base/exp037_inputs
    python -u /repo/scripts/linux_preflight.py --experiment "$EXPERIMENT" --overlay-zip /in/overlay.zip --handoff /in/handoff.json \
      --base "$base" --work /tmp/preflight --receipt /out/linux_preflight.json'
echo "Linux preflight receipt: $OUT/linux_preflight.json"
