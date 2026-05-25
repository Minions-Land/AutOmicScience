from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict


StatusLiteral = Literal["success", "failed"]
JudgeStatusLiteral = Literal["pending", "passed", "failed"]
AvailabilityStatusLiteral = Literal["available", "partial_assets", "missing_assets", "unavailable"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

