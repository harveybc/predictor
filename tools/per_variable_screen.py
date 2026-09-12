#!/usr/bin/env python3
"""C85: the per-variable screen's score entry point. It stops first.

No license exists: the design v3 has not been externally reviewed and no
scoring license has been granted. This entry point refuses with
EXTERNAL_DESIGN_REVIEW_AND_LICENSE_REQUIRED before it reads a label,
opens a dataset or imports a learning library. The refusal is the whole
of its behaviour under this order.
"""
from __future__ import annotations

import sys

REFUSAL = "EXTERNAL_DESIGN_REVIEW_AND_LICENSE_REQUIRED"


class ScoreRefusal(SystemExit):
    def __init__(self) -> None:
        super().__init__(
            f"{REFUSAL}: per-variable scoring is closed. The design is a "
            "draft candidate with no external review and no license; no "
            "label was read and no model was built")
        self.code_name = REFUSAL


def score(*args, **kwargs):
    raise ScoreRefusal()


def main(argv=None) -> int:
    raise ScoreRefusal()


if __name__ == "__main__":
    sys.exit(main())
