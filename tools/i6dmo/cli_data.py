"""Data-lane CLI commands (additive registration for ``python -m tools.i6dmo``).

``register(subparsers)`` adds ``data-bundle``, ``data-verify`` and
``naive-replay``; the module also runs alone: ``python -m tools.i6dmo.cli_data``.
No command reads a TEST bar, trains, or writes outside ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.i6dmo import data, naive, provenance


def _design(path: str) -> dict:
    return json.loads(Path(path).read_text())


def cmd_bundle(args) -> dict:
    design = _design(args.design)
    if args.long_data_view:
        design = data.long_data_view(design)
    manifest = data.build_frozen_bundle(args.input_dir, design, args.out,
                                        code_commit=args.code_commit)
    return {"status": "BUILT", "bundle_sha256": manifest["bundle_sha256"],
            "feature_manifest_sha256": manifest["feature_manifest_sha256"],
            "availability_contract_sha256": manifest["availability_contract_sha256"],
            "calendar": manifest["calendar"],
            "inputs": {k: v["sha256"] for k, v in manifest["inputs"].items()},
            "splits": {s: {k: v[k] for k in ("n_rows", "row_ids_sha256", "arrays_sha256",
                                              "first_origin", "last_origin", "head_valid")}
                       for s, v in manifest["splits"].items()},
            "coverage_status": manifest["coverage"]["status"],
            "experiment_id": manifest["experiment_id"],
            "data_contract_sha256": manifest["data_contract_sha256"],
            "sampling_rule": manifest["input_sampling"]["rule"],
            "fit_naive_mae": manifest["fit_statistics"]["naive_mae_fit"],
            "history_provenance": manifest["history_provenance"],
            "confirmatory_test_blocked": manifest["confirmatory_test_blocked"]}


def cmd_verify(args) -> dict:
    return data.verify_bundle(args.bundle)


def cmd_replay(args) -> dict:
    arrays = data.load_split(args.bundle, args.split)
    manifest = json.loads((Path(args.bundle) / "BUNDLE.json").read_text())

    class _A:  # minimal view for naive_replay
        row_ids = arrays["row_ids"]
        mask = arrays["mask"]
    return {"split": args.split,
            "naive": naive.naive_replay(args.input_dir, _A, manifest["horizons_hours"])}


def register(subparsers) -> None:
    p = subparsers.add_parser("data-bundle", help="build the frozen I6-DMO data bundle")
    p.add_argument("--design", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--code-commit")
    p.add_argument("--long-data-view", action="store_true",
                   help="derive the long data view (PLAN section 5) from the short design")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_bundle)
    p = subparsers.add_parser("data-verify", help="re-hash a frozen data bundle")
    p.add_argument("--bundle", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify)
    p = subparsers.add_parser("naive-replay", help="independent naive replay on bundle row ids")
    p.add_argument("--bundle", required=True)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--split", choices=data.SPLITS, required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_replay)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.i6dmo.cli_data")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except (data.DataContractError, provenance.ProvenanceError, naive.NaiveReplayMismatch) as exc:
        print(json.dumps({"status": "REFUSED", "code": getattr(exc, "code", type(exc).__name__),
                          "detail": str(exc)}))
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
