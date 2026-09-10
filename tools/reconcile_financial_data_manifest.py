#!/usr/bin/env python3
"""CLI wrapper for financial-data manifest reconciliation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap.financial_data_reconciliation import main


if __name__ == "__main__":
    raise SystemExit(main())
