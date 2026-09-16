#!/usr/bin/env python3
"""Drain the OLAP outbox into the DuckDB cube THROUGH the warehouse service. CPU only.

E4 of `docs/handoffs/MUSASHI_DUCKDB_CLOSEOUT_CORRECTIONS_2026_09_16.md`:

    "Implement the df_* writer through the warehouse-owned interface and durable outbox,
     preserving the existing event schemas/identities and excluding unrelated PostgreSQL uses.
     [...] Do not simply restart the obsolete direct-PostgreSQL loader."

The difference from `tools/olap_loader.py` is the destination and nothing else. That loader
opened PostgreSQL directly; this one posts each envelope to the warehouse host, which is the
single process that owns the DuckDB file. The outbox, its states, its adjudications and its
heartbeat are the same durable mechanism, and the envelope documents are unchanged — the event
schemas and identities travel exactly as they were.

A database that is unavailable still costs a retry and never a scientific result: a failed post
leaves the entry pending with its reason attached, and an entry the store REFUSES (400) is a
permanent verdict about those bytes and is marked failed, as before.

  --once      drain what is pending and exit (the default)
  --watch N   drain every N seconds until interrupted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap import outbox as ob  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:5057"


def post_envelope(url: str, token: str, document: dict, *, timeout: float = 60.0):
    """Hand one envelope to the owner. Returns (status, body)."""
    payload = json.dumps({"document": document}).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/api/v2/foundation-envelopes", data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {"error": f"http {exc.code}"}
    except Exception as exc:                      # transport: retryable, never a verdict
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def drain_once(root=None, *, url: str, token: str) -> dict:
    result = {"loaded": 0, "failed": 0, "retryable": 0, "counts": {}}
    for path in ob.pending_entries(root):
        try:
            body = ob.read_entry(path)
        except Exception as exc:                  # noqa: BLE001
            ob.mark(path, ob.FAILED, root=root,
                    reason=f"unreadable entry: {exc.__class__.__name__}")
            result["failed"] += 1
            continue
        if body.get("outbox_kind") != "envelope":
            ob.mark(path, ob.FAILED, root=root, reason="only envelopes are loadable today")
            result["failed"] += 1
            continue
        status, answer = post_envelope(url, token, body.get("document", {}))
        if status == 201:
            ob.mark(path, ob.LOADED, root=root)
            result["loaded"] += 1
            for key, value in (answer or {}).items():
                if isinstance(value, int):
                    result["counts"][key] = result["counts"].get(key, 0) + value
        elif status == 400:
            # the store looked at these bytes and refused them: a PERMANENT verdict
            ob.mark(path, ob.FAILED, root=root,
                    reason=str(answer.get("error", "refused"))[:400])
            result["failed"] += 1
        else:
            # transport, auth or the owner being down: keep the evidence, try again later
            result["retryable"] += 1
    ob.heartbeat(root)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warehouse-url", default=os.environ.get("WAREHOUSE_URL", DEFAULT_URL))
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN",
                        help="environment variable holding the store token; never a literal")
    parser.add_argument("--root", default=None, help="outbox root")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--watch", type=float, metavar="SECONDS")
    args = parser.parse_args(argv)

    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"{args.token_env} is not set: the loader does not embed credentials")
    if args.watch:
        while True:
            print(json.dumps(drain_once(args.root, url=args.warehouse_url, token=token)),
                  flush=True)
            time.sleep(args.watch)
    print(json.dumps(drain_once(args.root, url=args.warehouse_url, token=token), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
