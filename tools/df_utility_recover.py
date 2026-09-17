#!/usr/bin/env python3
"""Recover the utility contrasts a rehearsal measured but reported without metrics (N1).

The children of `utilreh-v4` wrote their results to `contrast.json` and the reporter read the
process summary (`result.json`), so the three COMPLETED terminals reached data-gov and the cube
with `metrics: []`. Nothing was lost: every `contrast.json` is on disk with the digest its
child declared. This tool, without training anything again:

  1. re-hashes each `contrast.json` against the digest `result.json` declared and validates
     the result (schema, contrast identity, protocol identity, finite values);
  2. reads the original generation-1 terminal digests from the cube;
  3. sends one generation-2 terminal per contrast through the durable outbox — same status,
     the real metrics, tags naming the original terminal and this recovery — and reconciles;
  4. compares the metrics the cube now holds for generation 2 with the files, value by value;
  5. emits a corrective DEVELOPMENT envelope with the real deltas (UNAVAILABLE where absent),
     naming the envelope it corrects; the original envelope and terminals stay as history.

Additive, traceable, non-governing. Write-once receipt `RECOVERY.json` under the run root.

    python tools/df_utility_recover.py --root RUN_ROOT --api-key-file KEY
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
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str, where: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load("df_utility_harness")
campaign = _load("df_d3_campaign")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cube_query(url: str, token: str, sql: str) -> list:
    request = urllib.request.Request(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}),
                                     headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        return json.loads(answer.read())["rows"]


def recover_scores(root: Path, report: dict) -> tuple:
    """(unit -> verified contrast result, refusals) from the preserved files. The protocol
    identity every file must carry is the freeze's, never left blank."""
    scores, refusals = {}, []
    frozen = json.loads((root / "FREEZE.json").read_text())
    protocol_sha = frozen["protocol"]["protocol_sha256"]
    for unit_id, out in report["outcomes"].items():
        adir = root / "attempts" / unit_id
        result_path = adir / "result.json"
        if not result_path.is_file():
            refusals.append({"unit_id": unit_id, "why": "no result.json (not COMPLETED)",
                             "outcome": out.get("outcome")})
            continue
        result = json.loads(result_path.read_text())
        # the runner's own re-hash was not persisted by that version; the digest the child
        # declared is checked against the bytes now, and said so in the receipt.
        # utilreh-v4's files predate the schema field; every other check applies, including
        # the protocol identity against the freeze
        score, refusal = H.verified_score(adir, result, {"output_sha256": result.get("output_sha256")},
                                          {"contrast_id": unit_id,
                                           "protocol": {"protocol_sha256": protocol_sha}},
                                          allow_legacy_schema=True)
        if refusal:
            refusals.append({"unit_id": unit_id, **refusal})
            continue
        scores[unit_id] = {"score": score, "output_sha256": result["output_sha256"],
                           "verification": "DIGEST_DECLARED_BY_CHILD_RECHECKED_NOW"}
    return scores, refusals


def metrics_of(score: dict) -> list:
    return [{"metric": name, "split": "development", "horizon": 1, "unit": score["loss_name"],
             "value": float(val), "std_dev": None, "min_value": None, "max_value": None}
            for name, val in (("utility.delta_mean", score["delta_mean"]),
                              ("utility.delta_lower", score["delta_lower"]),
                              ("utility.delta_se", score["delta_se"]),
                              ("utility.blocks_used", float(score["blocks_used"])))]


def main(argv=None) -> int:
    GR = _load("governed_run")
    OB = _load("outbox", REPO / "olap")
    CE = _load("campaign_envelope", REPO / "olap")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=GR.DEFAULT_OUTBOX)
    parser.add_argument("--receipt", default="RECOVERY.json")
    parser.add_argument("--envelope-key-suffix", default="recovery",
                        help="the corrective envelope's campaign key suffix (a key already "
                             "holding another identity in the cube cannot be reused)")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"REFUSED: {args.token_env} is not set")
    root = args.root
    report = json.loads((root / "REPORT.json").read_text())
    frozen = json.loads((root / "FREEZE.json").read_text())
    code_identity = GR.strict_code_identity(REPO)
    run_id = report["run_id"]
    campaign_sha = report["campaign"]["campaign_sha256"]
    campaign_key = report["campaign"]["key"]

    scores, refusals = recover_scores(root, report)
    originals = {r["unit_id"]: r for r in cube_query(
        args.warehouse_url, token,
        f"SELECT unit_id, terminal_sha256, generation, status FROM \"main\".\"gov_terminal\" "
        f"WHERE campaign_key = '{campaign_key}' AND generation = 1 LIMIT 100")}
    receipt = {"schema": "df_utility_recovery.v1", "run_id": run_id, "started_at": now_iso(),
               "source_report": "REPORT.json", "campaign": report["campaign"],
               "recovered": {}, "refused": refusals, "reconciliation": None,
               "content_check": {}, "envelope": None, "note":
               "additive recovery from preserved verified outputs; no new measurement; the "
               "generation-1 terminals and the original envelope stay as history; nothing is "
               "promoted scientifically"}
    if not scores:
        raise SystemExit(f"REFUSED: nothing recovered ({len(refusals)} refusals); no terminal, "
                         "no envelope, no receipt is written for an empty recovery")
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir)).resolve())
    original_cost = {t["unit_id"]: t.get("cost") or {} for t in report["terminals"]}
    for unit_id, rec in scores.items():
        score = rec["score"]
        if "schema" not in score:
            rec["verification"] += ";LEGACY_FILE_WITHOUT_SCHEMA_FIELD"
        original = originals.get(unit_id)
        if original is None:
            refusals.append({"unit_id": unit_id, "why": "no generation-1 terminal in the cube"})
            continue
        cost = original_cost.get(unit_id, {})
        terminal = {"schema": "governed_terminal.v1", "generation": 2, "status": original["status"],
                    "reason": None, "started_at": cost.get("started_at") or now_iso(),
                    "finished_at": cost.get("ended_at") or now_iso(),
                    "costs": {"wall_seconds": max(0.0, float(cost.get("wall_seconds") or 0)),
                              "cpu_seconds": max(0.0, float(cost.get("cpu_seconds") or 0))},
                    "deliveries": [], "artifacts": [], "metrics": metrics_of(score),
                    "tags": {"purpose": "UTILITY_HARNESS_REHEARSAL", "grants": "NONE",
                             "classification": "NON_GOVERNING", "outcome": str(score["outcome"]),
                             "protocol_sha256": score["protocol_sha256"],
                             "freeze_sha256": frozen["freeze_sha256"],
                             "supersedes_terminal_sha256": original["terminal_sha256"],
                             "supersedes_generation": "1", "recovery": "N1 from contrast.json",
                             "output_sha256": rec["output_sha256"],
                             "recovery_verification": rec["verification"]}}
        outbox.put({"campaign_sha256": campaign_sha, "unit_id": unit_id, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        receipt["recovered"][unit_id] = {"delta_mean": score["delta_mean"],
                                         "delta_lower": score["delta_lower"],
                                         "outcome": score["outcome"],
                                         "output_sha256": rec["output_sha256"],
                                         "supersedes_terminal_sha256": original["terminal_sha256"],
                                         "pending_after_flush": flushed["pending"]}
    rstatus, rbody = gov.reconcile_campaign(campaign_sha)
    receipt["reconciliation"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                                 "accounting_only": rbody.get("accounting_only"),
                                 "lake_only": rbody.get("lake_only")}
    # content: what the cube holds for generation 2 must equal the files, value by value
    held = cube_query(args.warehouse_url, token,
                      f"SELECT t.unit_id, m.metric, m.value FROM \"main\".\"gov_terminal_metric\" m "
                      f"JOIN \"main\".\"gov_terminal\" t ON m.terminal_sha256 = t.terminal_sha256 "
                      f"WHERE t.campaign_key = '{campaign_key}' AND t.generation = 2 LIMIT 500")
    by_unit = {}
    for r in held:
        by_unit.setdefault(r["unit_id"], {})[r["metric"]] = r["value"]
    all_equal = True
    for unit_id, rec in scores.items():
        expected = {m["metric"]: m["value"] for m in metrics_of(rec["score"])}
        got = by_unit.get(unit_id, {})
        equal = all(abs(float(got.get(k, float("nan"))) - v) < 1e-12 for k, v in expected.items())
        receipt["content_check"][unit_id] = {"expected": expected, "cube": got, "equal": equal}
        all_equal &= equal
    # the corrective envelope
    units = []
    for unit_id, rec in scores.items():
        score = rec["score"]
        units.append({"candidate_key": unit_id.split("__")[2], "cell_key": unit_id,
                      "metric_name": "utility.delta_mean", "metric_value": float(score["delta_mean"]),
                      "terminal_state": "COMPLETE", "uncertainty_kind": "BLOCK_T_LOWER",
                      "uncertainty_low": float(score["delta_lower"]), "uncertainty_high": "UNAVAILABLE"})
    for entry in refusals:
        units.append({"candidate_key": entry["unit_id"].split("__")[2], "cell_key": entry["unit_id"],
                      "metric_name": "outcome", "metric_value": "UNAVAILABLE",
                      "terminal_state": str(entry.get("outcome") or "UNRECOVERED"),
                      "uncertainty_kind": "NONE", "uncertainty_low": "UNAVAILABLE",
                      "uncertainty_high": "UNAVAILABLE"})
    receipt_sha = hashlib.sha256(json.dumps({k: v for k, v in receipt.items() if k != "envelope"},
                                            sort_keys=True, default=str).encode()).hexdigest()
    envelope = CE.build_envelope(
        campaign_key=f"utility-rehearsal-{run_id}-{args.envelope_key_suffix}", producer="predictor",
        result_class="DEVELOPMENT",
        identity={"run_id": run_id, "code_identity": code_identity["value"],
                  "design_sha256": frozen["protocol"]["protocol_sha256"],
                  "record_sha256": receipt_sha},
        data_consumed={"datasets": [{"id": f"{frozen['data']['generator']}/{frozen['data']['seed']}",
                                     "digest": frozen["data"]["sha256"],
                                     "eligibility_state": "FABRICATED_REHEARSAL"}],
                       "variables": [{"id": "v0", "digest": "UNAVAILABLE",
                                      "eligibility_state": "FABRICATED_REHEARSAL"}],
                       "operators": [{"id": u.split("__")[2], "digest": "UNAVAILABLE",
                                      "eligibility_state": "FABRICATED_REHEARSAL"}
                                     for u in scores]},
        partitions={"exposure": "DEVELOPMENT_REHEARSAL_NO_RESERVE", "splits": "UNAVAILABLE"},
        budget={"device": "cpu", "wall_seconds": 0.0, "cost_units": len(scores)},
        terminal={"state": "COMPLETED", "adjudication": "RECOVERY_NO_ADJUDICATION"},
        artifacts={"verification": "RECOVERED_FROM_VERIFIED_OUTPUT_FILES",
                   "freeze": frozen["freeze_sha256"]},
        units=units)
    emitted = OB.emit(envelope, kind="envelope")
    receipt["envelope"] = {"envelope_sha256": envelope["envelope_sha256"],
                           "corrects_envelope_sha256": report["envelope"]["envelope_sha256"],
                           **emitted}
    receipt["content_all_equal"] = all_equal
    receipt["finished_at"] = now_iso()
    campaign.write_once(root / args.receipt, receipt)
    print(json.dumps({"recovered": receipt["recovered"], "refused": refusals,
                      "reconciliation": receipt["reconciliation"],
                      "content_all_equal": all_equal, "envelope": receipt["envelope"]},
                     indent=1, default=str))
    return 0 if all_equal and not rbody.get("missing_units") else 1


if __name__ == "__main__":
    raise SystemExit(main())
