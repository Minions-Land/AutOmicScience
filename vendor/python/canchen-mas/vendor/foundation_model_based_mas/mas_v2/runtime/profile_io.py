from __future__ import annotations

from pathlib import Path

import yaml

from mas_v2.contracts import RunProfile


def load_run_profile(path: str | Path) -> RunProfile:
    path = Path(path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return RunProfile.model_validate(payload)


def save_run_profile(profile: RunProfile, path: str | Path) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path

