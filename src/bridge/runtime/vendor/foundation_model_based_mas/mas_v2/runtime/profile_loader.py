from __future__ import annotations

from pathlib import Path

import yaml

from mas_v2.contracts.schemas import RunProfile


def load_run_profile(profile_path: str | Path) -> RunProfile:
    path = Path(profile_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RunProfile.model_validate(payload)
