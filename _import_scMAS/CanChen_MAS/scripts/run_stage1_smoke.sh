#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CANCHEN_MAS_ROOT="${CANCHEN_MAS_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m scmas build-label-maps
python -m scmas build-reference \
  --max-cells-per-source 1000 \
  --max-cells-seaad-reference 5000 \
  --max-cells-seaad-test 2000
python -m scmas prepare-sources --max-cells-per-source 1000
python -m scmas write-scdesign3-configs --target-total 1000 --n-cores 2
python -m scmas run-scdesign3 --dry-run
python -m scmas evaluate --prepare-only --no-new-synthetic --output-dir "${REPO_ROOT}/runs/stage1_smoke_eval"
