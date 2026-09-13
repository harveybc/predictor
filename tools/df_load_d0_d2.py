#!/usr/bin/env python3
"""C139, C140: load the D0-D2 outputs into the cube, additively.

Rows loaded (every one validated by load_data_foundation before insert):

* datasets and variables from every sealed contract (public v2 panels, the
  synthetic units, the financial first batch);
* lab operator runs, raw/denoised/residual metrics, delay and cost, and lab
  decisions (C137-C138) as written by the lab run;
* SNR calibration rows (C134), bound to each unit's contract digest, fitted on
  train; ESTIMATED is COMPLETED, NOT_IDENTIFIABLE is INCONCLUSIVE, a fit
  failure is FAILED;
* profile rows (C130-C133), routed by module and grain;
* the coverage ledger (C140), dataset x variable x metric x operator, declared
  from the contracts and the metric names each module emits, with every cell's
  state derived from its rows.

Modes:

* throwaway: create a throwaway database, create the schema, load everything
  twice (the second load must insert nothing), drop the database;
* real: record the historical cube tables, create the schema additively, load
  once, and verify the historical tables did not change.

The running outbox loader is not touched. A write-once receipt records every
table's offered, inserted, already-present and refused rows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import uuid
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


L = _load("load_data_foundation")
COV = _load("df_coverage")
HISTORICAL = ("fact_variable_characterization", "fact_terminal_verification_variable_v2", "fact_campaign_unit",
              "dim_campaign", "dim_campaign_run", "fact_campaign_consumption")
C164_TABLES = ("df_fact_resource_estimate", "df_fact_dataset_terminal", "df_fact_causal_test",
               "df_fact_naming_isolation_decision", "df_fact_host_receipt", "df_fact_incident_attempt")


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonl(p: Path):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------- collection
def contracts_from(public_panels=None, synthetic_bank=None, financial_contracts=None) -> list[dict]:
    out = []
    if public_panels:
        out += [json.loads((d / "CONTRACT.json").read_text()) for d in sorted(Path(public_panels).iterdir())
                if d.is_dir() and (d / "CONTRACT.json").is_file()]
    if synthetic_bank:
        sync = _load("df_synthetic_contract")
        out += [sync.unit_contract(d) for d in sorted(Path(synthetic_bank).iterdir()) if d.is_dir()]
    if financial_contracts:
        out += json.loads(Path(financial_contracts).read_text())["contracts"]
    return out


SNR_STATUS = {"ESTIMATED": "COMPLETED"}


def snr_rows(snr_json: Path, unit_content: dict) -> tuple[str, list[dict]]:
    doc = json.loads(Path(snr_json).read_text())
    rows_path = Path(str(snr_json) + ".olap_rows.jsonl") if not str(snr_json).endswith(".json") else \
        Path(str(snr_json)[:-len(".json")] + ".olap_rows.jsonl")
    if not rows_path.is_file():
        rows_path = next(Path(snr_json).parent.glob("*.olap_rows.jsonl"))
    run_id = "c134_" + _sha_file(rows_path)[:24]
    out = []
    for r in _jsonl(rows_path):
        status = SNR_STATUS.get(r["status"], "INCONCLUSIVE")
        reason = r.get("reason") or ""
        if reason.startswith("fit_failed"):
            status = "FAILED"
        value = r.get("value")
        if status == "COMPLETED" and (value is None or isinstance(value, str)):
            status, reason = "INCONCLUSIVE", reason or "no finite value"
        if status != "COMPLETED":
            value, reason = None, reason or "not identifiable"
        out.append({"run_id": run_id, "unit_id": r["unit_id"], "content_sha256": unit_content[r["unit_id"]],
                    "variable_index": int(r["variable_index"]), "partition": "train", "estimator": r["estimator"],
                    "metric": r["metric"], "value": value, "value_text": None, "status": status,
                    "reason": reason if status != "COMPLETED" else "", "code_sha256": r["code_sha256"]})
    return run_id, out


def profile_rows(profile_root: Path, content_by_dataset: dict) -> tuple[str, dict]:
    receipt = json.loads((Path(profile_root) / "PROFILE_RUN_RECEIPT.json").read_text())
    run_id = "c130_" + hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()[:24]
    tables = {t: [] for t in ("df_fact_variable_profile", "df_fact_information_metric", "df_fact_pair_relation",
                              "df_fact_group_relation", "df_fact_sampling_quality")}
    for d in receipt["datasets"]:
        if d["status"] != "COMPLETED":
            continue
        for item in _jsonl(Path(profile_root) / d["file"]):
            m, r = item["module"], item["row"]
            content = content_by_dataset[r["dataset_id"]]
            if "pair" in r:
                a, b = r["pair"]
                lag = r["estimator"].get("params", {}).get("lag")
                tables["df_fact_pair_relation"].append(L.metric_row(
                    r, run_id=run_id, content_sha256=content, variable_id_a=a, variable_id_b=b,
                    lag=lag if type(lag) is int else None))
            elif "group_id" in r:
                if m == "df_profile_multivariate":
                    members = r["estimator"].get("params", {}).get("members", [])
                    tables["df_fact_group_relation"].append(L.metric_row(
                        r, run_id=run_id, content_sha256=content, group_id=r["group_id"],
                        members=members if isinstance(members, list) else []))
                elif m == "df_profile_information":
                    tables["df_fact_information_metric"].append(L.metric_row(
                        r, run_id=run_id, content_sha256=content, subject_kind="MATRIX", subject_id=r["group_id"]))
                else:
                    tables["df_fact_sampling_quality"].append(L.metric_row(
                        r, run_id=run_id, content_sha256=content, variable_id=f"DATASET:{r['group_id']}"))
            else:
                grain = {"variable_id": r["variable_id"]}
                if m == "df_profile_univariate":
                    tables["df_fact_variable_profile"].append(L.metric_row(r, run_id=run_id, content_sha256=content, **grain))
                elif m == "df_sampling":
                    tables["df_fact_sampling_quality"].append(L.metric_row(r, run_id=run_id, content_sha256=content, **grain))
                else:
                    tables["df_fact_information_metric"].append(L.metric_row(
                        r, run_id=run_id, content_sha256=content, subject_kind="VARIABLE", subject_id=r["variable_id"]))
    return run_id, tables


def runtime_rows(profile_root: Path) -> tuple[list[dict], list[dict]]:
    """C164: every durable terminal of a profile root, and every memory estimate row, typed by stage."""
    root = Path(profile_root)
    terminals = [json.loads(p.read_text()) for p in sorted((root / "terminals").glob("*.json"))]
    estimates = []
    for stage, name in (("PREFLIGHT_METADATA_UPPER_BOUND", "preflight_estimates.jsonl"),
                        ("CHILD_RUNTIME", "resource_estimates.jsonl")):
        for p in sorted((root / "attempts").glob(f"*/attempt-*/{name}")):
            estimates += [dict(r, stage=stage) for r in _jsonl(p)]
    return terminals, estimates


def coverage(contracts: list[dict], profile_tables: dict, lab_runs: list[dict], run_id: str) -> list[dict]:
    variable_metrics = sorted({r["metric"] for t in ("df_fact_variable_profile", "df_fact_sampling_quality")
                               for r in profile_tables.get(t, []) if not r["variable_id"].startswith("DATASET:")}
                              | {r["metric"] for r in profile_tables.get("df_fact_information_metric", [])
                                 if r["subject_kind"] == "VARIABLE"})
    real = [c for c in contracts if c["bank"] != "SYNTHETIC"]
    synth = [c for c in contracts if c["bank"] == "SYNTHETIC"]
    cells = COV.expected_grid(real + synth, variable_metrics) if variable_metrics else []
    ops = sorted({f"{r['operator_kind']}:{json.dumps(r['operator_params'], sort_keys=True)}" for r in lab_runs})
    if synth and ops:
        cells += COV.expected_grid(synth, ["operator_evaluation"], ops)
    rows = [{"dataset_id": r["dataset_id"], "variable_id": r["variable_id"], "metric": r["metric"],
             "operator": "NONE", "status": r["status"]}
            for t in ("df_fact_variable_profile", "df_fact_sampling_quality") for r in profile_tables.get(t, [])
            if not r["variable_id"].startswith("DATASET:")]
    rows += [{"dataset_id": r["dataset_id"], "variable_id": r["subject_id"], "metric": r["metric"],
              "operator": "NONE", "status": r["status"]}
             for r in profile_tables.get("df_fact_information_metric", []) if r["subject_kind"] == "VARIABLE"]
    unit_dataset = {c["original_fields"]["unit_record"]["unit_id"]: c["dataset_id"] for c in synth}
    rows += [{"dataset_id": unit_dataset.get(r["subject_id"], r["subject_id"]), "variable_id": r["variable_id"],
              "metric": "operator_evaluation",
              "operator": f"{r['operator_kind']}:{json.dumps(r['operator_params'], sort_keys=True)}",
              "status": r["status"]} for r in lab_runs]
    matrix = COV.build_matrix(cells, rows)
    return matrix, COV.coverage_rows(matrix, run_id=run_id, code_sha256=_sha_file(HERE / "df_coverage.py"))


def collect(args) -> tuple[dict, dict]:
    contracts = contracts_from(args.public_panels, args.synthetic_bank, args.financial_contracts)
    content = {c["dataset_id"]: c["content_sha256"] for c in contracts}
    unit_content = {c["original_fields"]["unit_record"]["unit_id"]: c["content_sha256"]
                    for c in contracts if c["bank"] == "SYNTHETIC"}
    dims_run = "d0_" + hashlib.sha256("".join(sorted(c["contract_sha256"] for c in contracts)).encode()).hexdigest()[:24]
    tables = {"df_dim_dataset": [], "df_dim_variable": []}
    for c in contracts:
        ds, vs = L.dataset_rows(c, dims_run)
        tables["df_dim_dataset"].append(ds)
        tables["df_dim_variable"] += vs
    runs = [{"run_id": dims_run, "module": "D0 contracts", "code_sha256": _sha_file(HERE / "df_contract.py"),
             "inputs_sha256": dims_run[3:].ljust(64, "0"), "status": "COMPLETED", "cpu_seconds": None,
             "details": {"contracts": len(contracts)}}]
    lab_runs = []
    if args.lab:
        for t in ("df_fact_operator_run", "df_fact_operator_signal_metric", "df_fact_operator_delay_cost",
                  "df_fact_lab_decision"):
            tables[t] = list(_jsonl(Path(args.lab) / f"{t}.jsonl"))
        lab_runs = tables["df_fact_operator_run"]
        summary = json.loads((Path(args.lab) / "LAB_EVALUATION_SUMMARY.json").read_text())
        runs.append({"run_id": summary["run_id"], "module": "C137-C138 lab evaluation",
                     "code_sha256": summary["code_sha256"], "inputs_sha256": summary["bank_manifest_sha256"],
                     "status": "COMPLETED", "cpu_seconds": None,
                     "details": {"decision_counts": summary["decision_counts"], "rule_sha256": summary["rule_sha256"]}})
    if getattr(args, "lab_delay_cost", None):
        # The first lab run's delay/cost table lacked per-frequency group and
        # phase delay. The corrected table replaces it in the load; the first
        # table stays in custody as the record of that gap.
        dc_dir = Path(args.lab_delay_cost)
        tables["df_fact_operator_delay_cost"] = list(_jsonl(dc_dir / "df_fact_operator_delay_cost.jsonl"))
        dcs = json.loads((dc_dir / "DELAY_COST_SUMMARY.json").read_text())
        runs.append({"run_id": dcs["run_id"], "module": "C135/C137 delay and cost (corrected table)",
                     "code_sha256": dcs["code_sha256"], "inputs_sha256": dcs["bank_manifest_sha256"],
                     "status": "COMPLETED", "cpu_seconds": None,
                     "details": {"rows": dcs["table"]["rows"], "supersedes": dcs["supersedes"]}})
    if args.snr:
        snr_run, rows = snr_rows(args.snr, unit_content)
        tables["df_fact_snr_calibration"] = rows
        runs.append({"run_id": snr_run, "module": "C134 SNR calibration", "code_sha256": rows[0]["code_sha256"] if rows else "0" * 64,
                     "inputs_sha256": _sha_file(args.snr), "status": "COMPLETED", "cpu_seconds": None,
                     "details": {"rows": len(rows)}})
    profile_tables = {}
    roots = args.profiles if isinstance(args.profiles, (list, tuple)) else ([args.profiles] if args.profiles else [])
    for root in roots:
        # C162: one sealed root per role; C164: each root's terminals and memory estimates are loaded too
        prof_run, part = profile_rows(root, content)
        for t, rows in part.items():
            profile_tables.setdefault(t, []).extend(rows)
        terms, ests = runtime_rows(root)
        tables.setdefault("df_fact_dataset_terminal", []).extend(terms)
        tables.setdefault("df_fact_resource_estimate", []).extend(ests)
        receipt = json.loads((Path(root) / "PROFILE_RUN_RECEIPT.json").read_text())
        runs.append({"run_id": prof_run, "module": f"C130-C133 profiles ({receipt.get('host_role', 'COORDINATOR')})",
                     "code_sha256": _sha_file(HERE / "df_profile_run.py"),
                     "inputs_sha256": _sha_file(Path(root) / "PROFILE_RUN_RECEIPT.json"),
                     "status": "COMPLETED", "cpu_seconds": None,
                     "details": dict({t: len(v) for t, v in part.items()}, terminals=len(terms),
                                     resource_estimates=len(ests), host_role=receipt.get("host_role"))})
    tables.update(profile_tables)
    for d in getattr(args, "table_dir", None) or []:
        # C164: runtime, causal, naming, host and incident outputs, each a directory of
        # table-named JSONL files. Only the C164 grains are taken from such a directory.
        d = Path(d)
        found = [t for t in C164_TABLES if (d / f"{t}.jsonl").is_file()]
        if not found:
            raise SystemExit(f"REFUSED: {d.name} holds none of the C164 tables")
        digests, counts, first = [], {}, None
        for t in found:
            rows = list(_jsonl(d / f"{t}.jsonl"))
            tables.setdefault(t, []).extend(rows)
            digests.append(_sha_file(d / f"{t}.jsonl"))
            counts[t] = len(rows)
            first = first or (rows[0] if rows else None)
        runs.append({"run_id": first["run_id"] if first else f"c164_{d.name}", "module": f"C164 outputs: {d.name}",
                     "code_sha256": first.get("code_sha256") if first and first.get("code_sha256") else "0" * 64,
                     "inputs_sha256": hashlib.sha256("".join(digests).encode()).hexdigest(),
                     "status": "COMPLETED", "cpu_seconds": None, "details": counts})
    cov_run = "c140_" + dims_run[3:]
    matrix, cov_rows = coverage(contracts, profile_tables, lab_runs, cov_run)
    tables["df_fact_coverage"] = cov_rows
    runs.append({"run_id": cov_run, "module": "C140 coverage", "code_sha256": _sha_file(HERE / "df_coverage.py"),
                 "inputs_sha256": matrix["ledger_sha256"], "status": "COMPLETED", "cpu_seconds": None,
                 "details": {"cells": matrix["cells"], "counts": matrix["counts_derived_from_ledger"],
                             "undeclared_rows": len(matrix["undeclared_rows"])}})
    tables["df_dim_run"] = runs
    return tables, {"cells": matrix["cells"], "counts": matrix["counts_derived_from_ledger"],
                    "ledger_sha256": matrix["ledger_sha256"], "undeclared_rows": len(matrix["undeclared_rows"])}


# ------------------------------------------------------------------- load
def _engine(db=None):
    from sqlalchemy import create_engine
    e = os.environ
    return create_engine(f"postgresql://{e['PGUSER']}:{e['PGPASSWORD']}@{e.get('PGHOST', '127.0.0.1')}:"
                         f"{e.get('PGPORT', '5432')}/{db or e.get('PGDATABASE', 'predictor_olap')}",
                         isolation_level="AUTOCOMMIT" if db == "postgres" else None)


def _counts(engine, names) -> dict:
    from sqlalchemy import text
    out = {}
    with engine.connect() as c:
        for t in names:
            try:
                out[t] = c.execute(text(f"SELECT count(*) FROM public.{t}")).scalar()
            except Exception:  # noqa: BLE001
                c.rollback()
                out[t] = None
    return out


def load_all(engine, tables: dict) -> dict:
    L.ensure_schema(engine)
    receipts = {}
    for t, rows in tables.items():
        run_id = rows[0]["run_id"] if rows else "none"
        receipts[t] = L.load(engine, t, rows, run_id)
        receipts[t].pop("refusals", None) if not receipts[t]["rows_refused"] else None
    return receipts


def main(argv=None) -> int:
    from sqlalchemy import text
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("throwaway", "real"), required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--public-panels", type=Path)
    ap.add_argument("--synthetic-bank", type=Path)
    ap.add_argument("--financial-contracts", type=Path)
    ap.add_argument("--lab", type=Path)
    ap.add_argument("--lab-delay-cost", type=Path,
                    help="corrected delay/cost table; replaces the lab run's own table in the load")
    ap.add_argument("--snr", type=Path)
    ap.add_argument("--profiles", type=Path, action="append",
                    help="a sealed profile root; repeat once per role of the campaign")
    ap.add_argument("--table-dir", type=Path, action="append",
                    help="C164 output directory of table-named JSONL files; repeatable")
    a = ap.parse_args(argv)
    if a.receipt.exists():
        raise SystemExit("REFUSED: the receipt exists; each load is write-once")
    tables, cov = collect(a)
    offered = {t: len(v) for t, v in tables.items()}
    receipt = {"schema": "crispdm.data_foundation.load_receipt.v1", "mode": a.mode, "offered": offered,
               "coverage": cov, "loader_code_sha256": _sha_file(HERE / "load_data_foundation.py")}
    if a.mode == "throwaway":
        name = "c139_d0d2_throwaway_" + uuid.uuid4().hex[:10]
        admin = _engine("postgres")
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
        eng = _engine(name)
        try:
            receipt["first_load"] = load_all(eng, tables)
            receipt["second_load"] = load_all(eng, tables)
            receipt["idempotent"] = all(r["rows_inserted"] == 0 for r in receipt["second_load"].values())
        finally:
            eng.dispose()
            with admin.connect() as c:
                c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            admin.dispose()
        receipt["throwaway_database_dropped"] = True
    else:
        eng = _engine()
        # C164: history is every pre-existing base table outside the data-foundation grains,
        # discovered at load time, not a fixed list.
        with eng.connect() as c:
            history = [r[0] for r in c.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public' "
                "AND table_type='BASE TABLE' AND table_name NOT LIKE 'df\\_%' ORDER BY 1"))]
        before = _counts(eng, history)
        receipt["load"] = load_all(eng, tables)
        after = _counts(eng, history)
        receipt["tables_written_by_this_load"] = sorted(tables)
        receipt["historical_before"], receipt["historical_after"] = before, after
        receipt["historical_unchanged"] = before == after
        eng.dispose()
    text_out = json.dumps(receipt, indent=1, sort_keys=True, default=str).replace(str(Path.home()), "~")
    a.receipt.write_text(text_out + "\n")
    print(json.dumps({k: receipt[k] for k in receipt if k in ("mode", "offered", "coverage", "idempotent",
                                                             "historical_unchanged")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
