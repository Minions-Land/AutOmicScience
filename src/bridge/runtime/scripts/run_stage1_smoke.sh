#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AOS_MAS_ROOT="${AOS_MAS_ROOT:-${REPO_ROOT}}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python -m aos_agent build-label-maps
python -m aos_agent build-reference \
  --max-cells-per-source 1000 \
  --max-cells-seaad-reference 5000 \
  --max-cells-seaad-test 2000
python -m aos_agent prepare-sources --max-cells-per-source 1000
python -m aos_agent write-scdesign3-configs --target-total 1000 --n-cores 2
python -m aos_agent run-scdesign3 --dry-run
python -m aos_agent evaluate --prepare-only --no-new-synthetic --output-dir "${REPO_ROOT}/runs/stage1_smoke_eval"
