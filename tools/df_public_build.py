#!/usr/bin/env python3
"""C126: build every public panel independently, with a typed receipt.

Each dataset of the custody manifest is built on its own by
`df_public_contract.build`. A refusal (for example timestamps that go
backwards in the source) or a failure is recorded in the build receipt with
its problems and never stops the other datasets. Nothing is sorted,
deduplicated or repaired to make a dataset pass. The receipt is write-once
and binds the custody manifest and the adapter code by digest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("df_public_contract")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _redact(text: str) -> str:
    return str(text).replace(str(Path.home()), "~")


def build_all(raw_root: Path = P.RAW_ROOT, panel_root: Path = P.PANEL_ROOT, datasets=None) -> dict:
    raw_root, panel_root = Path(raw_root), Path(panel_root)
    out = panel_root / "BUILD_RECEIPT.json"
    if out.exists():
        raise SystemExit("REFUSED: BUILD_RECEIPT.json exists; the build is write-once")
    results = []
    for lid in datasets or sorted(P.SOURCES):
        try:
            c = P.build(lid, raw_root, panel_root)
            rec = c["original_fields"]["parse_receipt"]
            results.append({"logical_id": lid, "status": "BUILT", "dataset_id": c["dataset_id"],
                            "contract_sha256": c["contract_sha256"], "rows": rec["rows"],
                            "variables": len(c["variables"]), "units_unknown": len(rec["units_unknown"]),
                            "duplicate_timestamp_labels": rec["duplicate_timestamp_labels"],
                            "missing_values": rec["missing_values"]})
        except P.C.ContractRefusal as exc:
            results.append({"logical_id": lid, "status": "REFUSED", "problems": [_redact(p) for p in exc.problems]})
        except Exception as exc:  # noqa: BLE001 - every outcome is recorded
            results.append({"logical_id": lid, "status": "FAILED",
                            "problems": [_redact(f"{type(exc).__name__}: {exc}")[:500]]})
    receipt = {"schema": "crispdm.data_foundation.public_build_receipt.v1",
               "raw_manifest_sha256": _sha(raw_root / "PUBLIC_RAW_MANIFEST.json"),
               "adapter_code_sha256": _sha(HERE / "df_public_contract.py"),
               "build_code_sha256": _sha(Path(__file__)),
               "results": results,
               "counts": {s: sum(r["status"] == s for r in results) for s in ("BUILT", "REFUSED", "FAILED")},
               "rule": "a dataset that refuses is recorded with its problems; nothing is sorted, deduplicated or repaired"}
    text = json.dumps(receipt, indent=1, sort_keys=True)
    if "/home/" in text:
        raise SystemExit("REFUSED: absolute home path in the build receipt")
    panel_root.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-root", type=Path, default=P.RAW_ROOT)
    ap.add_argument("--panel-root", type=Path, default=P.PANEL_ROOT)
    ap.add_argument("--dataset", action="append", default=None)
    a = ap.parse_args(argv)
    r = build_all(a.raw_root, a.panel_root, a.dataset)
    print(json.dumps({"counts": r["counts"],
                      "results": [{k: v for k, v in x.items() if k != "problems"} | (
                          {"problem": x["problems"][0]} if x.get("problems") else {}) for x in r["results"]]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
