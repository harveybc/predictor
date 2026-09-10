#!/usr/bin/env python3
"""CLI wrapper for the CRISP-DM dataset inventory builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap.crispdm_inventory import main


if __name__ == "__main__":
    raise SystemExit(main())
