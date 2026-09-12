#!/usr/bin/env python3
"""C104: the v4 per-variable screen's score entry point refuses first.

No external review of design v4 and no license exist. This entry point
stops with EXTERNAL_V4_DESIGN_REVIEW_AND_LICENSE_REQUIRED before opening
labels, models or any numeric library.
"""
from __future__ import annotations

import sys

REFUSAL = "EXTERNAL_V4_DESIGN_REVIEW_AND_LICENSE_REQUIRED"


class ScoreRefusal(SystemExit):
    def __init__(self) -> None:
        super().__init__(f"{REFUSAL}: scoring is closed; no label read, no model built")
        self.code_name = REFUSAL


def score(*a, **k):
    raise ScoreRefusal()


def main(argv=None) -> int:
    raise ScoreRefusal()


if __name__ == "__main__":
    sys.exit(main())
