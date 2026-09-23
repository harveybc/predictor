"""Audit retained identities against the live store; resend only exact pending terminals."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--gov-url", required=True)
    p.add_argument("--warehouse-url", required=True)
    p.add_argument("--actor-key", type=Path, required=True)
    p.add_argument("--warehouse-key", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resend", action="store_true")
    a = p.parse_args()
    path = a.repo / "tools/df_sota_repro.py"
    spec = importlib.util.spec_from_file_location("df_sota_repro", path)
    R = importlib.util.module_from_spec(spec)
    sys.modules["df_sota_repro"] = R
    spec.loader.exec_module(R)
    C = R._module("df_mod_e0_close")
    G = R._module("df_e1_governed")
    design = json.loads((a.root / "DESIGN.json").read_text())
    deliveries = json.loads((a.root / "DELIVERIES.json").read_text())["units"]
    token = a.warehouse_key.read_text().strip().strip('"').strip("'")
    result = {"resends": [], "rows": [], "scope": "no fitting, no benchmark replay, no deletion"}
    for cell in design["cells"]:
        unit = cell["cell_id"]
        folder = a.root / "attempts" / unit
        rec = json.loads((folder / "cell.json").read_text())
        if a.resend and cell["horizon"] == 720:
            pending = sorted((a.root / "outbox/units" / unit / "pending").glob("*.json"))
            assert len(pending) == 1, (unit, "expected one original pending envelope", len(pending))
            envelope = json.loads(pending[0].read_text())
            assert envelope["campaign_sha256"] == deliveries[unit]["campaign_sha256"]
            assert envelope["unit_id"] == unit and envelope["terminal"]["status"] == "COMPLETED"
            artifacts = {r["role"]: r for r in envelope["terminal"]["artifacts"]}
            for role, name in (("predictions", "arrays.npz"), ("record", "cell.json"), ("checkpoint", "checkpoint.pth")):
                assert R.sha_file(folder / name) == artifacts[role]["sha256"], (unit, role)
                assert (folder / name).stat().st_size == artifacts[role]["bytes"]
            outcome = G.report_terminal(a.root, unit, envelope["terminal"], gov_url=a.gov_url,
                api_key_file=a.actor_key, outbox_dir=str(a.root / "outbox"),
                started_at=envelope["terminal"].get("started_at"))
            result["resends"].append({"unit": unit, "original_envelope_sha256": R.sha_obj(envelope),
                "flushed": outcome["flushed"], "reconciliation": outcome["reconciliation"],
                "receipt_persisted": outcome["persisted_receipt"] is not None})
            if outcome["flushed"]["pending"] or outcome["flushed"]["failures"]:
                a.output.write_text(json.dumps(result, indent=2, default=str))
                raise RuntimeError("terminal resend refused; original retained, no replacement constructed")
        receipts = json.loads((a.root / "TERMINAL_RECEIPTS.json").read_text())["units"]
        receipt = receipts.get(unit)
        if not receipt:
            result["rows"].append({"unit": unit, "accepted": False})
            continue
        live = C.warehouse_terminals(a.warehouse_url, token, receipt["campaign_sha256"])["current"].get(unit)
        artifacts = {r["role"]: r for r in (live or {}).get("artifacts", [])}
        row = {"unit": unit, "accepted": bool(live and live["status"] == "COMPLETED" and live["terminal_sha256"] == receipt["terminal_sha256"]),
               "record_digest": R.sha_file(folder / "cell.json"), "checkpoint_digest": R.sha_file(folder / "checkpoint.pth"),
               "arrays_present": (folder / "arrays.npz").is_file(), "vault_present": (folder / "METRICS_VAULT.json").is_file()}
        row["record_bound"] = row["record_digest"] == artifacts.get("record", {}).get("sha256")
        row["checkpoint_bound"] = row["checkpoint_digest"] == artifacts.get("checkpoint", {}).get("sha256")
        row["metric_basis"] = R.metric_of(rec)[1]
        row["record_metric"] = R.metric_of(rec)[0]
        row["warehouse_metrics"] = [{k: m[k] for k in ("metric", "value", "horizon")} for m in (live or {}).get("metrics", [])]
        if row["vault_present"]:
            row["vault_digest"] = R.sha_file(folder / "METRICS_VAULT.json")
            vault = json.loads((folder / "METRICS_VAULT.json").read_text())
            row["vault_record_bound"] = vault["identity"]["record_sha256"] == row["record_digest"]
        result["rows"].append(row)
        a.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"resends": result["resends"], "rows": [{k: r.get(k) for k in ("unit", "accepted", "record_bound", "checkpoint_bound", "arrays_present", "vault_record_bound")} for r in result["rows"]]}, default=str))


if __name__ == "__main__":
    main()
