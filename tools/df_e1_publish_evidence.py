#!/usr/bin/env python3
"""Copy a run root's retained artifacts into a repository evidence folder, with the machine's host name redacted.

This repository is public. `AGENTS.md` forbids writing a machine host name into it, and `df_e1_block.run_cell` records
`os.uname().nodename` in every cell record, from where it reaches the block report and the closure table. This tool copies
the named JSON artifacts and replaces that one value — read from the live host, never printed — with the literal
`<worker-host>` wherever it appears as a value, then records what it replaced (the count, not the value) in
`REDACTIONS.json` so a reader knows the files are not byte-identical to the run root and why.

Nothing else is changed: no number, no digest, no rule, no problem list. Digests recorded INSIDE the artifacts therefore
still refer to the run root's bytes, which is the point — the run root keeps the originals.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PLACEHOLDER = "<worker-host>"


def redact(value, host: str, counter: dict):
    if isinstance(value, str):
        if host and host in value:
            counter["strings"] += value.count(host)
            return value.replace(host, PLACEHOLDER)
        return value
    if isinstance(value, list):
        return [redact(v, host, counter) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, host, counter) for k, v in value.items()}
    return value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", nargs="*", default=[], help="JSON artifacts to copy, relative to --root")
    ap.add_argument("--text", nargs="*", default=[], help="text artifacts to copy verbatim (already redacted by hand or host-free)")
    a = ap.parse_args(argv)
    host = os.uname().nodename
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "df_e1_publish_evidence.v1", "run_root": str(a.root), "files": {},
              "rule": f"every occurrence of this host's name replaced by the literal {PLACEHOLDER!r}; nothing else changed",
              "why": "this repository is public and AGENTS.md forbids a machine host name in it"}
    for rel in a.json:
        src, dst = a.root / rel, a.out / Path(rel).name
        counter = {"strings": 0}
        doc = redact(json.loads(src.read_text()), host, counter)
        dst.write_text(json.dumps(doc, indent=1, default=str))
        report["files"][dst.name] = {"from": rel, "host_name_occurrences_replaced": counter["strings"]}
    for rel in a.text:
        src, dst = Path(rel), a.out / Path(rel).name
        text = src.read_text()
        n = text.count(host) if host else 0
        dst.write_text(text.replace(host, PLACEHOLDER) if n else text)
        report["files"][dst.name] = {"from": str(src), "host_name_occurrences_replaced": n}
    (a.out / "REDACTIONS.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"out": str(a.out), "files": {k: v["host_name_occurrences_replaced"] for k, v in report["files"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
