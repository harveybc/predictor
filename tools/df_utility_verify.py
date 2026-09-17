#!/usr/bin/env python3
"""Content check of a utility run: the metrics the cube holds must equal the verified files (N4).

For every contrast attempt of a run root: re-hash `contrast.json` against the digest
`result.json` declared (and, when present, the runner-verified digest in `outcome.json`),
then compare each expected metric with the value the cube's `gov_terminal_metric` holds for
that unit at the given generation — value by value, never count against count. Calibration
attempts are checked the same way against `calibration.json`. Writes a write-once
`CONTENT_CHECK.json` and exits non-zero on any inequality.

    python tools/df_utility_verify.py --root RUN_ROOT [--generation 1]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load("df_utility_harness")


def cube_query(url: str, token: str, sql: str) -> list:
    request = urllib.request.Request(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}),
                                     headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        return json.loads(answer.read())["rows"]


def expected_metrics(attempt_dir: Path) -> tuple:
    """(unit, kind, {metric: value}) from the verified file, or a refusal."""
    result_path = attempt_dir / "result.json"
    if not result_path.is_file():
        return None, {"why": "no result.json (not COMPLETED)"}
    result = json.loads(result_path.read_text())
    verified = {"output_sha256": result.get("output_sha256")}
    if (attempt_dir / "outcome.json").is_file():
        verified = json.loads((attempt_dir / "outcome.json").read_text()).get("verified") or verified
    name = result.get("output_file")
    path = attempt_dir / str(name)
    if not name or not path.is_file():
        return None, {"why": f"output {name!r} absent"}
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != verified.get("output_sha256"):
        return None, {"why": "output bytes differ from the declared/verified digest",
                      "declared": result.get("output_sha256"), "verified": verified.get("output_sha256"),
                      "found": digest}
    doc = json.loads(body)
    if name == "contrast.json":
        if not isinstance(doc.get("delta_mean"), (int, float)):
            return {"kind": "contrast", "metrics": {}, "outcome": doc.get("outcome")}, None
        return {"kind": "contrast", "outcome": doc.get("outcome"),
                "metrics": {"utility.delta_mean": doc["delta_mean"],
                            "utility.delta_lower": doc["delta_lower"],
                            "utility.delta_se": doc["delta_se"],
                            "utility.blocks_used": float(doc["blocks_used"])}}, None
    if name == "calibration.json":
        return {"kind": "calibration", "metrics": {
            "calibration.false_advance_rate": doc["false_advance_rate"],
            "calibration.upper_bound": doc["upper_bound"],
            "calibration.scored": float(doc["scored"]),
            "calibration.failed": float(doc["failed"])}}, None
    return {"kind": name, "metrics": {}}, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--out", default="CONTENT_CHECK.json")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"REFUSED: {args.token_env} is not set")
    report = json.loads((args.root / "REPORT.json").read_text())
    campaigns = {"contrast": report["contrasts"]["campaign"]["key"],
                 "calibration": report["calibration"]["campaign"]["key"]}
    held = {}
    for kind, key in campaigns.items():
        rows = cube_query(args.warehouse_url, token,
                          f"SELECT t.unit_id, m.metric, m.value FROM \"main\".\"gov_terminal_metric\" m "
                          f"JOIN \"main\".\"gov_terminal\" t ON m.terminal_sha256 = t.terminal_sha256 "
                          f"WHERE t.campaign_key = '{key}' AND t.generation = {args.generation} LIMIT 1000")
        for r in rows:
            held.setdefault((kind, r["unit_id"]), {})[r["metric"]] = r["value"]
    check = {"schema": "df_utility_content_check.v1", "run_id": report["run_id"],
             "generation": args.generation, "units": {}, "all_equal": True, "refused": []}
    for attempt in sorted((args.root / "attempts").iterdir()):
        expected, refusal = expected_metrics(attempt)
        name = attempt.name
        if refusal:
            check["refused"].append({"attempt": name, **refusal})
            continue
        if expected["kind"] not in ("contrast", "calibration"):
            continue
        unit = name.split("__", 1)[1] if expected["kind"] == "calibration" else name
        got = held.get((expected["kind"], unit), {})
        equal = all(k in got and abs(float(got[k]) - float(v)) < 1e-12
                    for k, v in expected["metrics"].items())
        if not expected["metrics"] and got:
            equal = False                       # the cube holds metrics the file does not
        check["units"][name] = {"kind": expected["kind"], "expected": expected["metrics"],
                                "cube": got, "equal": equal,
                                "outcome": expected.get("outcome")}
        check["all_equal"] &= equal
    out = args.root / args.out
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists; a check is never written over")
    out.write_text(json.dumps(check, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"all_equal": check["all_equal"],
                      "units": {k: (v["equal"], v["outcome"]) for k, v in check["units"].items()},
                      "refused": check["refused"]}, indent=1))
    return 0 if check["all_equal"] and not check["refused"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
