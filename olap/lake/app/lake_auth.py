"""Require DATA_GOV_LAKE_TOKEN on /api/v1."""

from __future__ import annotations

import hmac
import os
from pathlib import Path


def load_token() -> str | None:
    env = os.getenv("DATA_GOV_LAKE_TOKEN")
    if env:
        return env.strip()
    explicit = os.getenv("DATA_GOV_LAKE_TOKEN_FILE")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    here = Path(__file__).resolve()
    # olap/lake/app → predictor → GitHub/data-gov
    candidates.append(here.parents[3].parent / "data-gov" / "var" / "lake_token")
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip() or None
    return None


def check_bearer(header: str | None, expected: str | None) -> bool:
    if not expected:
        return False
    given = (header or "").strip()
    if not given.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(given[7:].strip(), expected)
