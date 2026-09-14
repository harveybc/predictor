#!/usr/bin/env python3
"""Retry durable Flow-v3 terminals and move only reconciled reports to sent/."""

from __future__ import annotations

import argparse
import json
import sys

from governed_run import (
    DEFAULT_OUTBOX,
    GovHttp,
    GovernedRunError,
    TerminalOutbox,
    _send_pending,
    load_api_key,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file")
    parser.add_argument("--outbox-dir", default=DEFAULT_OUTBOX)
    args = parser.parse_args(argv)
    try:
        key = load_api_key(args.api_key_file)
        gov = GovHttp(args.gov_url, key, "terminal-outbox-flush")
        result = _send_pending(gov, TerminalOutbox(args.outbox_dir))
    except (OSError, GovernedRunError) as exc:
        print(f"flush_governed_terminals: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
