"""Require the lake token on /api/v1. It comes only from DATA_GOV_LAKE_TOKEN
or the file named by DATA_GOV_LAKE_TOKEN_FILE, never from a path into
another checkout."""

from __future__ import annotations

import hmac
import os
from pathlib import Path


def load_token() -> str | None:
    env = os.getenv("DATA_GOV_LAKE_TOKEN")
    if env:
        return env.strip() or None
    explicit = os.getenv("DATA_GOV_LAKE_TOKEN_FILE")
    if explicit and Path(explicit).is_file():
        return Path(explicit).read_text(encoding="utf-8").strip() or None
    return None


def check_bearer(header: str | None, expected: str | None) -> bool:
    if not expected:
        return False
    given = (header or "").strip()
    if not given.lower().startswith("bearer "):
        return False
    return hmac.compare_digest(given[7:].strip(), expected)
