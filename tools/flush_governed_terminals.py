#!/usr/bin/env python3
"""Retry durable Flow-v3 terminals, show the outbox health, or adjudicate one.

    flush_governed_terminals.py [--gov-url URL --api-key-file FILE --outbox-dir DIR]
        (default)                       retry every pending envelope once, exit 1 if any remains
        --status                        JSON health: recoverable / awaiting adjudication / unresolved / adjudicated
        --dispose FILE --reason TEXT    close a pending envelope as INVALID_ENVELOPE (moved, never deleted)
        --supersede FILE --terminal T.json --reason TEXT
                                        send T.json as the next generation of the same campaign/unit
                                        (same outcome, same deliveries), then dispose FILE as SUPERSEDED

Retries are idempotent: replaying the same terminal returns the existing receipt;
a different terminal for the same campaign, unit and generation is refused. A
4xx refusal alone never closes an envelope: it waits, visibly, for a disposition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from governed_run import (
    DEFAULT_OUTBOX,
    GovHttp,
    GovernedRunError,
    TerminalOutbox,
    _require_reconciled,
    _send_pending,
    load_api_key,
)


def _sender(gov):
    def send(envelope):
        status, receipt = gov.report_terminal(
            envelope["campaign_sha256"], envelope["unit_id"], envelope["terminal"]
        )
        if status not in (200, 201):
            raise GovernedRunError(f"terminal refused: http {status} {receipt.get('error', '')}".strip())
        _require_reconciled(gov, envelope["campaign_sha256"], envelope["unit_id"], before_run=False)
        return receipt
    return send


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file")
    parser.add_argument("--outbox-dir", default=DEFAULT_OUTBOX)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dispose", metavar="FILE")
    parser.add_argument("--supersede", metavar="FILE")
    parser.add_argument("--terminal", metavar="T.json", help="corrected governed_terminal.v1 for --supersede")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    outbox = TerminalOutbox(args.outbox_dir)
    try:
        if args.status:
            print(json.dumps(outbox.status(), indent=2, sort_keys=True))
            return 0
        if args.dispose:
            record = outbox.dispose(args.dispose, "INVALID_ENVELOPE", args.reason or "")
            print(json.dumps(record, sort_keys=True))
            return 0
        key = load_api_key(args.api_key_file)
        gov = GovHttp(args.gov_url, key, "terminal-outbox-flush")
        if args.supersede:
            if not args.terminal:
                raise GovernedRunError("--supersede needs --terminal")
            with open(Path(args.terminal).expanduser(), encoding="utf-8") as handle:
                corrected = json.load(handle)
            record = outbox.supersede(args.supersede, corrected, _sender(gov), args.reason or "")
            print(json.dumps(record, sort_keys=True))
            return 0
        result = _send_pending(gov, outbox)
    except (OSError, GovernedRunError, ValueError) as exc:
        print(f"flush_governed_terminals: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
