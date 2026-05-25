from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return str(path)


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return json.loads(resolved.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def read_obs_csv(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).resolve()
    return pd.read_csv(resolved, index_col=0)


def to_abs_str(path: str | Path) -> str:
    return str(Path(path).resolve())

