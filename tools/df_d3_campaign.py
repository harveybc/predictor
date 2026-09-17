#!/usr/bin/env python3
"""The D3 governed mechanics campaign (J3): freeze, dispatch, collect, report, reconcile.

Every step leaves a receipt in the campaign root, and every step refuses to run twice over a
root it has already sealed. In order:

  freeze      the population (which bank units, which toy resources, which operators) and the
              budgets (per-unit memory, wall and CPU; per-role caps), sealed by digest BEFORE
              any unit runs, with the cost pilot that justifies them;
  toys        register the DATASETS campaign with data-gov, deliver and confirm the seven toy
              resources, materialise them as units;
  shards      symlink shards of bank units plus the toy units, and the jobs file the
              dispatcher consumes (`{role}` placeholders, one process per unit inside);
  sync        fetch this checkout's commit into each worker's clean worktree; the preflight
              then verifies commit, clean tree and code digest before any dispatch;
  dispatch    `df_dispatch` across the three roles, NON_GOVERNING with the stated reason;
  collect     rsync every role's shard outputs back, verifying the per-unit terminals'
              output digests against the rows they name;
  report      one governed terminal per unit through the SYNTHETIC campaign (bank units) or
              the DATASETS campaign (toys), via the durable outbox, reconciled with data-gov;
              then one MECHANICAL envelope per run to the OLAP outbox for the loader.

Runs are NON_GOVERNING mechanical evidence: no utility ranking, no promotion. A failing
operator is recorded and excluded. Infrastructure readiness, battery acceptance and
scientific utility are three separate words in every receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATE = Path.home() / ".local/state/crispdm-data-foundation"
BANK_ROOT = STATE / "synthetic_bank_c128_v1"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("df_d3_contract")
design = _load("df_d3_design")
ops = _load("df_d3_operators")
worker = _load("df_d3_unit_worker")

TOY_RESOURCES = (
    ("synthetic_typical_price_4h_train.csv", "input_file"),
    ("synthetic_typical_price_4h_validation.csv", "input_file"),
    ("synthetic_typical_price_4h_test.csv", "input_file"),
    ("synthetic_features_4h_train.csv", "input_file"),
    ("synthetic_features_4h_validation.csv", "input_file"),
    ("synthetic_features_4h_test.csv", "input_file"),
    ("synthetic_ohlc_1h.csv", "input_file"),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path: Path, doc: dict) -> None:
    if path.exists():
        raise SystemExit(f"REFUSED: {path} already exists; a receipt is never written over")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def redact(text: str) -> str:
    return text.replace(str(Path.home()), "~")


# --- freeze ---------------------------------------------------------------------------------

def select_bank_units(bank_root: Path, *, lengths=(512, 2048), missingness=("none", "mcar", "blocks"),
                      per_family_limit=None) -> list:
    """The bank population, deterministic: every unit whose length and missingness the
    amendment names, sorted by unit id. `per_family_limit` bounds a rehearsal, never the run."""
    chosen, per_family = [], {}
    for unit_dir in sorted(p for p in bank_root.iterdir() if p.is_dir()):
        rec_path = unit_dir / "UNIT.json"
        if not rec_path.is_file():
            continue
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if rec["n_samples"] not in lengths:
            continue
        if (rec.get("missingness") or {}).get("kind") not in missingness:
            continue
        family = rec["family"]
        if per_family_limit is not None and per_family.get(family, 0) >= per_family_limit:
            continue
        per_family[family] = per_family.get(family, 0) + 1
        chosen.append({"unit_id": rec["unit_id"], "family": family, "n_samples": rec["n_samples"],
                       "n_variables": rec["n_variables"],
                       "missingness": (rec.get("missingness") or {}).get("kind"),
                       "path": str(unit_dir)})
    return chosen


def freeze(root: Path, *, bank_root: Path, pilot_units: list, per_unit_wall: float,
           per_unit_cpu: int, task_memory_bytes: int, role_caps: dict, per_family_limit=None,
           toys: bool = True) -> dict:
    """The sealed population and budgets, with the pilot that justifies them."""
    pilot = worker.pilot(pilot_units)
    over = [k for k, v in pilot["operators"].items() if not v["within_declaration"]]
    if over:
        raise SystemExit(f"REFUSED: the cost pilot exceeds the declaration for {over}")
    units = select_bank_units(bank_root, per_family_limit=per_family_limit)
    doc = {"schema": "d3_mechanics_freeze.v1", "frozen_utc": now_iso(),
           "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"],
           "code_sha256s": worker.code_sha256s(),
           "operators": [{"kind": op.KIND, "params": op.params,
                          "spec_sha256": _load("df_d3_contract").spec_sha256(op.describe()),
                          "twin": (op.twin().get("kind") or "NOT_APPLICABLE")}
                         for op in ops.bank()],
           "library_versions": ops.library_versions(),
           "bank": {"root": redact(str(bank_root)), "units": units,
                    "count": len(units), "lengths": list(design.LENGTHS),
                    "missingness": list(design.MISSINGNESS_REGIMES),
                    "per_family_limit": per_family_limit},
           "toys": [{"resource": r, "role": role} for r, role in TOY_RESOURCES] if toys else [],
           "budget": {"basis": "cost pilot below, single thread, on the smallest and widest "
                               "bank units; one unit per process; the full battery is many "
                               "transforms per operator, measured on the smoke unit",
                      "per_unit_wall_seconds": per_unit_wall, "per_unit_cpu_seconds": per_unit_cpu,
                      "task_memory_bytes": task_memory_bytes,
                      "memory_ceiling_bytes": design.D3_AMENDMENT_V1["memory_ceiling_bytes"],
                      "role_caps": role_caps},
           "cost_pilot": pilot,
           "classification": "NON_GOVERNING", "result_class": "MECHANICAL",
           "readiness": {"INFRASTRUCTURE_PRESENT": True,
                         "TEMPORAL_BATTERY_ACCEPTED": "measured per unit by the run",
                         "SCIENTIFIC_UTILITY": "not claimed"},
           "freeze_sha256": ""}
    body = {k: v for k, v in doc.items() if k != "freeze_sha256"}
    doc["freeze_sha256"] = sha_obj(body)
    write_once(root / "FREEZE.json", doc)
    return doc


# --- shards and jobs ------------------------------------------------------------------------

def build_shards(root: Path, frozen: dict, *, toy_units_root, shard_size: int) -> dict:
    shards_root = root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    members = [Path(u["path"]) for u in frozen["bank"]["units"]]
    if toy_units_root is not None and Path(toy_units_root).is_dir():
        members += sorted(p for p in Path(toy_units_root).iterdir() if p.is_dir())
    shards = []
    for index in range(0, len(members), shard_size):
        name = f"shard_{len(shards):02d}"
        sdir = shards_root / name
        sdir.mkdir(exist_ok=True)
        names = []
        for unit_dir in members[index:index + shard_size]:
            link = sdir / unit_dir.name
            if not link.exists():
                link.symlink_to(unit_dir.resolve())
            names.append(unit_dir.name)
        (sdir / "MEMBERS.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        shards.append({"name": name, "members": names, "path": str(sdir)})
    return {"shards": shards, "root": str(shards_root)}


def build_jobs(root: Path, frozen: dict, shards: dict, *, run_id: str, python_rel: str,
               out_rel: str) -> Path:
    """The dispatcher's jobs file. Paths are home-relative on every host (`$HOME` at launch)
    and `{role}` is bound by the dispatcher."""
    budget = frozen["budget"]
    jobs = []
    for shard in shards["shards"]:
        members = shard["members"]
        # Toy units with many variables need proportionally more wall time; the freeze fixes
        # the per-unit figure and the shard's wall is the sum, never a guess per shard.
        wall = int(sum(budget["per_unit_wall_seconds"] * _variables_of(Path(shard["path"]) / m)
                       for m in members)) + 120
        cpu_bytes = int(budget["task_memory_bytes"] * 1.25)
        jobs.append({"job_id": f"{run_id}-{shard['name']}", "cpu_bytes": cpu_bytes,
                     "gpu_bytes": 0, "cpus": 1, "wall": wall,
                     "roles": ["WORKER_A", "WORKER_B", "COORDINATOR"], "gpu_index": None,
                     "split": None,
                     # Every host path is written home-relative with the `~/` the launch
                     # script expands to "$HOME"; a bare relative path resolved against the
                     # worktree and the interpreter was "not found" (exit 127, first dispatch).
                     "argv": ["env", "-u", "PYTHONPATH", _home(python_rel), "-B",
                              "tools/df_d3_unit_worker.py",
                              "--units-root", redact(shard["path"]),
                              "--out", f"{_home(out_rel)}/{{role}}/{shard['name']}",
                              "--run-id", run_id, "--host-role", "{role}",
                              "--task-memory", str(budget["task_memory_bytes"]),
                              "--wall-seconds", str(budget["per_unit_wall_seconds"]),
                              "--cpu-seconds", str(budget["per_unit_cpu_seconds"])]})
    path = root / "JOBS.json"
    write_once(path, {"schema": "d3_mechanics_jobs.v1", "run_id": run_id,
                      "freeze_sha256": frozen["freeze_sha256"], "jobs": jobs})
    return path


def _home(rel: str) -> str:
    """`~/<rel>` for a home-relative path, as the dispatcher's launch script expects."""
    rel = str(rel)
    if rel.startswith("~/") or rel.startswith("/"):
        return rel
    if rel.startswith("./"):
        rel = rel[2:]
    return "~/" + rel


def _variables_of(unit_dir: Path) -> int:
    unit_dir = Path(unit_dir)
    if (unit_dir / "UNIT.json").is_file():
        return int(json.loads((unit_dir / "UNIT.json").read_text())["n_variables"])
    if (unit_dir / "TOY.json").is_file():
        return len(json.loads((unit_dir / "TOY.json").read_text())["variables"])
    return 1


# --- toys: a DATASETS campaign, delivered and materialised ----------------------------------

def toy_unit_id(resource: str) -> str:
    return "toy-" + Path(resource).stem


def deliver_toys(root: Path, *, gov_url: str, api_key_file: str, lake: str, lake_config: Path,
                 metrics_lake: str, project: str, run_id: str, code_identity: dict,
                 config_sha256: str) -> dict:
    """One DATASETS campaign PER toy resource, each declaring that one resource and that one
    unit; deliver and confirm the resource under it and materialise it as a D3 unit.

    data-gov binds a delivery to (campaign, actor, unit) and completes a DATASETS terminal only
    when it covers every dataset the campaign declares: d3mech-v1 declared all seven resources
    under one campaign and no per-resource unit could complete (422); those seven were
    re-reported under per-resource campaigns after the fact. From v2 the shape is right from
    the start."""
    GR = _load("governed_run")
    TOY = _load("df_d3_toy_resources")
    host = json.loads(Path(lake_config).read_text(encoding="utf-8"))
    contracts = host["backend"]["settings"]["resource_contracts"]
    toys_root = root / "toys"
    cache = Path(GR.DEFAULT_CACHE).expanduser() / run_id
    delivered = []
    for resource, role in TOY_RESOURCES:
        unit_id = toy_unit_id(resource)
        key = f"{run_id}-toy-{Path(resource).stem}"
        gov = GR.GovHttp(gov_url, GR.load_api_key(api_key_file), key)
        status, receipt = gov.submit_campaign({
            "schema": "governed_campaign.v1", "campaign_key": key,
            "classification": "NON_GOVERNING", "project": project,
            "code_identity": code_identity, "config_sha256": config_sha256,
            "input_mode": "DATASETS", "synthetic_spec_sha256": None, "units": [unit_id],
            "datasets": [{"lake": lake, "resource": resource, "role": role,
                          "from": None, "to": None}],
            "terminal_lake": metrics_lake})
        if status not in (200, 201):
            raise SystemExit(f"REFUSED: toy campaign {key} refused: http {status} "
                             f"{receipt.get('error', '')}".strip())
        campaign_sha256 = receipt["campaign_sha256"]
        record = TOY.deliver_and_materialise(
            gov, campaign_sha256=campaign_sha256, unit_id=unit_id, lake=lake, resource=resource,
            role=role, cache_dir=cache, resource_contract=contracts[resource],
            out_dir=toys_root / unit_id)
        delivered.append({"unit_id": unit_id, "resource": resource, "role": role,
                          "campaign_key": key, "campaign_sha256": campaign_sha256,
                          "delivery_id": record["delivery"]["delivery_id"],
                          "verification_state": record["delivery"]["verification_state"],
                          "n_samples": record["n_samples"],
                          "variables": len(record["variables"]),
                          "contract_sha256": record["contract_sha256"]})
    doc = {"schema": "d3_toys_delivery.v2", "run_id": run_id, "lake": lake,
           "campaign_shape": "one DATASETS campaign per resource and unit",
           "units": delivered, "delivered_utc": now_iso()}
    write_once(root / "TOYS.json", doc)
    return doc


# --- worker sync ----------------------------------------------------------------------------

def sync_workers(commit: str, checkout_rel: str, roles_map: dict) -> dict:
    """Fetch this commit into each worker's clean worktree and check it out detached."""
    script = f'''
set -e
W="$HOME"/{shlex.quote(checkout_rel)}
git -C "$W" fetch --quiet origin {commit}
git -C "$W" checkout --quiet --detach {commit}
printf "HEAD=%s dirty=%s\\n" "$(git -C "$W" rev-parse HEAD)" "$(git -C "$W" status --porcelain --untracked-files=all | wc -l)"
'''
    out = {}
    for role, entry in roles_map.items():
        alias = entry.get("ssh")
        if role == "COORDINATOR" or not alias:
            continue
        run = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", alias,
                              "bash -s"], input=script, capture_output=True, text=True,
                             timeout=300)
        text = (run.stdout + run.stderr).replace(alias, "<alias>").strip()
        out[role] = {"returncode": run.returncode, "output": redact(text)[-400:],
                     "synced": run.returncode == 0 and f"HEAD={commit}" in text
                     and "dirty=0" in text}
    return out


# --- collect --------------------------------------------------------------------------------

def collect(root: Path, out_rel: str, roles_map: dict, *, run_id: str,
            receipt: str = "COLLECT.json") -> dict:
    """rsync each worker's outputs back and verify every terminal's output digest.

    A unit may carry several attempts (a killed attempt stays on disk; the retry takes the
    next number). The receipt records the HIGHEST attempt of each unit and names it, so the
    rows it points at are the ones it verified. Receipts are write-once; a later collect
    takes its own name.
    """
    collected_root = root / "collected"
    collected_root.mkdir(parents=True, exist_ok=True)
    report = {"roles": {}, "units": [], "verified": 0, "mismatched": 0}
    for role, entry in roles_map.items():
        alias = entry.get("ssh")
        src = f"{alias}:~/{out_rel}/{role}/" if alias and role != "COORDINATOR" \
            else str(Path.home() / out_rel / role) + "/"
        dst = collected_root / role
        dst.mkdir(exist_ok=True)
        run = subprocess.run(["rsync", "-a", "--quiet", src, str(dst) + "/"],
                             capture_output=True, text=True, timeout=1800)
        report["roles"][role] = {"returncode": run.returncode,
                                 "error": redact((run.stderr or "").replace(alias or "", "<alias>")
                                                 [-200:])}
    latest = {}
    for terminal_path in sorted(collected_root.glob("*/*/terminals/*.json")):
        m = re.fullmatch(r"(.+)\.attempt-(\d+)\.json", terminal_path.name)
        if not m:
            continue
        key = (terminal_path.parts[-4], terminal_path.parts[-3], m.group(1))
        if key not in latest or int(m.group(2)) > latest[key][0]:
            latest[key] = (int(m.group(2)), terminal_path)
    for (role, shard, _), (attempt, terminal_path) in sorted(latest.items()):
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        unit = terminal["dataset_id"]
        entry = {"role": role, "shard": shard, "unit": unit, "attempt": attempt,
                 "status": terminal["status"],
                 "rows": terminal["rows_written"], "wall_seconds": terminal["wall_seconds"],
                 "cpu_seconds": terminal["cpu_seconds"]}
        if terminal["status"] == "COMPLETED":
            rows = (terminal_path.parents[1] / "attempts" / unit / f"attempt-{attempt}"
                    / "rows.jsonl")
            ok = rows.is_file() and sha_file(rows) == terminal["output_sha256"] \
                and sum(1 for _ in rows.open("rb")) == terminal["rows_written"]
            entry["output_verified"] = ok
            report["verified" if ok else "mismatched"] += 1
        report["units"].append(entry)
    write_once(root / receipt, {"schema": "d3_mechanics_collect.v1", "run_id": run_id,
                                       **report})
    return report


# --- report: terminals and the envelope ---------------------------------------------------

def unit_terminal(rows: list, *, status: str, reason, wall: float, cpu: float,
                  deliveries: list, bank: str, unit_id: str, run_id: str) -> dict:
    verdicts = [r for r in rows if r["test"] == "verdict"]
    metrics = []
    # One metric identity (metric, split, horizon, unit) per operator: a multivariate unit
    # has a verdict per variable, so the terminal carries the aggregate over variables and
    # the per-variable rows stay in the mechanics fact. The server refused one identity
    # per variable as a duplicate (d3mech-v1, 67 units) and that refusal is kept as evidence.
    by_op = {}
    for v in verdicts:
        by_op.setdefault(v["operator_kind"], []).append(float(v["value"]))
    for kind, values in sorted(by_op.items()):
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        metrics.append({"metric": f"d3.verdict.{kind}", "split": None, "horizon": None,
                        "unit": "review_ready_over_variables", "value": mean,
                        "std_dev": std, "min_value": min(values), "max_value": max(values)})
    if verdicts:
        metrics.append({"metric": "d3.variables", "split": None, "horizon": None,
                        "unit": "count", "value": float(len({r["variable"] for r in verdicts})),
                        "std_dev": None, "min_value": None, "max_value": None})
    by_test = {}
    for r in rows:
        if r["test"] in ("verdict", "battery"):
            continue
        key = (r["test"], r["outcome"])
        by_test[key] = by_test.get(key, 0) + 1
    for (test, outcome), count in sorted(by_test.items()):
        metrics.append({"metric": f"d3.test.{test}.{outcome}", "split": None, "horizon": None,
                        "unit": "rows", "value": float(count), "std_dev": None,
                        "min_value": None, "max_value": None})
    return {"schema": "governed_terminal.v1", "generation": 1, "status": status,
            "reason": reason, "started_at": now_iso(), "finished_at": now_iso(),
            "costs": {"wall_seconds": max(0.0, float(wall)), "cpu_seconds": max(0.0, float(cpu))},
            "deliveries": list(deliveries), "artifacts": [],
            "metrics": metrics if status == "COMPLETED" else [],
            "tags": {"purpose": "D3_MECHANICS", "grants": "NONE", "classification": "NON_GOVERNING",
                     "result_class": "MECHANICAL", "bank": bank, "unit_id": unit_id,
                     "run_id": run_id, "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"],
                     "readiness": "INFRASTRUCTURE_PRESENT;TEMPORAL_BATTERY_MEASURED;"
                                  "SCIENTIFIC_UTILITY_NOT_CLAIMED",
                     "externally_reviewed": "false"}}


#: What a mechanical run says about everything it consumed: evidence, never eligibility.
MECHANICAL_ELIGIBILITY = "MECHANICAL_EVIDENCE_NON_GOVERNING"


def mechanics_consumption(units: list) -> dict:
    """`data_consumed` as the envelope loader stores it: every item is {id, digest,
    eligibility_state}. The first d3mech-v1 envelope carried bare strings and the store
    fell over on `item.get` (answered as 503); that envelope is kept as FAILED evidence."""
    datasets, variables = {}, set()
    for u in units:
        digest = u["rows"][0]["contract_sha256"] if u.get("rows") else "UNAVAILABLE"
        datasets[u["unit_id"]] = digest
        variables.update(row["variable"] for row in u.get("verdict_rows", []))
    item = lambda id_, digest: {"id": id_, "digest": digest,
                                "eligibility_state": MECHANICAL_ELIGIBILITY}
    return {"datasets": [item(k, datasets[k]) for k in sorted(datasets)],
            "variables": [item(v, "UNAVAILABLE") for v in sorted(variables)],
            "operators": [item(op.KIND, contract.spec_sha256(op.describe()))
                          for op in ops.bank()]}


def build_mechanics_envelope(*, campaign_key: str, run_id: str, code_identity: dict,
                             frozen: dict, units: list, wall_seconds: float) -> dict:
    CE = _load_olap_envelope()
    envelope_units = []
    for u in units:
        for row in u["verdict_rows"]:
            envelope_units.append({"cell_key": f"{u['unit_id']}:{row['variable']}",
                                   "candidate_key": row["spec_sha256"],
                                   "metric_name": "review_ready",
                                   "terminal_state": row["outcome"],
                                   "metric_value": float(row["value"] or 0.0),
                                   "uncertainty_kind": "NONE"})
    completed = sum(1 for u in units if u["status"] == "COMPLETED")
    return CE.build_envelope(
        campaign_key=campaign_key, producer="predictor", result_class="MECHANICAL",
        identity={"run_id": run_id, "code_identity": code_identity["value"],
                  "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"],
                  "record_sha256": frozen["freeze_sha256"]},
        data_consumed=mechanics_consumption(units),
        partitions={"exposure": "MECHANICAL_NO_SCIENTIFIC_EXPOSURE",
                    "splits": {"train": "fit prefix only", "evaluated": "whole unit"}},
        budget={"device": "cpu", "wall_seconds": float(wall_seconds), "cost_units": len(units),
                "units_verified": completed, "units_failed": len(units) - completed},
        terminal={"state": "COMPLETED" if completed == len(units) else "PARTIAL",
                  "adjudication": "MECHANICAL_EVIDENCE_NO_ADJUDICATION",
                  "reason": "D3 temporal battery over the frozen population; no utility"},
        artifacts={"verification": "BORN_AT_PRODUCER_TERMINAL",
                   "freeze": frozen["freeze_sha256"]},
        units=envelope_units)


def _load_olap_envelope():
    name = "campaign_envelope"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "olap" / "campaign_envelope.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--bank-root", type=Path, default=BANK_ROOT)
    f.add_argument("--pilot-unit", type=Path, action="append", required=True)
    f.add_argument("--per-unit-wall", type=float, required=True)
    f.add_argument("--per-unit-cpu", type=int, required=True)
    f.add_argument("--task-memory", type=int, default=2 << 30)
    f.add_argument("--role-cap", action="append", default=[])
    f.add_argument("--per-family-limit", type=int)
    f.add_argument("--no-toys", action="store_true")
    s = sub.add_parser("shards")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--toy-units", type=Path)
    s.add_argument("--shard-size", type=int, default=12)
    s.add_argument("--run-id", required=True)
    s.add_argument("--python-rel", default="anaconda3/envs/trading-stack/bin/python")
    s.add_argument("--out-rel", required=True)
    t = sub.add_parser("toys")
    t.add_argument("--root", type=Path, required=True)
    t.add_argument("--run-id", required=True)
    t.add_argument("--gov-url", default="http://127.0.0.1:5055")
    t.add_argument("--api-key-file", required=True)
    t.add_argument("--lake", default="governance_smoke")
    t.add_argument("--lake-config", type=Path, required=True,
                   help="the lake host config that declares the resource contracts")
    t.add_argument("--metrics-lake", default="olap_cube")
    t.add_argument("--project", default="predictor")
    y = sub.add_parser("sync")
    y.add_argument("--commit", required=True)
    y.add_argument("--checkout-rel", default="Documents/GitHub/.worktrees/predictor-c146")
    y.add_argument("--roles", type=Path, default=Path.home() / ".config/crispdm/host_roles.json")
    c = sub.add_parser("collect")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--out-rel", required=True)
    c.add_argument("--run-id", required=True)
    c.add_argument("--roles", type=Path, default=Path.home() / ".config/crispdm/host_roles.json")
    c.add_argument("--receipt", default="COLLECT.json",
                   help="receipt file name under --root; write-once, so a re-collect names its own")
    args = parser.parse_args(argv)

    if args.cmd == "freeze":
        caps = {"COORDINATOR": 2, "WORKER_A": 10, "WORKER_B": 6}
        for item in args.role_cap:
            k, _, v = item.partition("=")
            caps[k] = int(v)
        doc = freeze(args.root, bank_root=args.bank_root, pilot_units=args.pilot_unit,
                     per_unit_wall=args.per_unit_wall, per_unit_cpu=args.per_unit_cpu,
                     task_memory_bytes=args.task_memory, role_caps=caps,
                     per_family_limit=args.per_family_limit, toys=not args.no_toys)
        print(json.dumps({"freeze_sha256": doc["freeze_sha256"], "units": doc["bank"]["count"],
                          "toys": len(doc["toys"])}, indent=1))
        return 0
    if args.cmd == "shards":
        frozen = json.loads((args.root / "FREEZE.json").read_text(encoding="utf-8"))
        shards = build_shards(args.root, frozen, toy_units_root=args.toy_units,
                              shard_size=args.shard_size)
        jobs = build_jobs(args.root, frozen, shards, run_id=args.run_id,
                          python_rel=args.python_rel, out_rel=args.out_rel)
        print(json.dumps({"shards": len(shards["shards"]), "jobs": str(jobs)}, indent=1))
        return 0
    if args.cmd == "toys":
        GR = _load("governed_run")
        frozen = json.loads((args.root / "FREEZE.json").read_text(encoding="utf-8"))
        code_identity = GR.strict_code_identity(REPO)
        config_sha = sha_obj({"schema": "d3_mechanics_execution.v1", "run_id": args.run_id,
                              "freeze_sha256": frozen["freeze_sha256"],
                              "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"]})
        doc = deliver_toys(args.root, gov_url=args.gov_url, api_key_file=args.api_key_file,
                           lake=args.lake, lake_config=args.lake_config,
                           metrics_lake=args.metrics_lake, project=args.project,
                           run_id=args.run_id, code_identity=code_identity,
                           config_sha256=config_sha)
        print(json.dumps({"units": [(u["unit_id"], u["campaign_sha256"][:12],
                                     u["verification_state"]) for u in doc["units"]]},
                         indent=1))
        return 0
    if args.cmd == "sync":
        roles = json.loads(args.roles.read_text(encoding="utf-8"))
        out = sync_workers(args.commit, args.checkout_rel, roles)
        print(json.dumps(out, indent=1))
        return 0 if all(v["synced"] for v in out.values()) else 1
    if args.cmd == "collect":
        roles = json.loads(args.roles.read_text(encoding="utf-8"))
        report = collect(args.root, args.out_rel, roles, run_id=args.run_id, receipt=args.receipt)
        print(json.dumps({"verified": report["verified"], "mismatched": report["mismatched"],
                          "units": len(report["units"])}, indent=1))
        return 0 if not report["mismatched"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
