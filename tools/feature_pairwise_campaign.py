#!/usr/bin/env python3
"""Durable driver for feature-selection phases 2 and 3.

The driver discovers missing work from content-addressed terminals, never from agent
memory.  A plan materialises every pair of the frozen population, splits the pairs into
deterministic hash shards, names every shard's terminal by the digest of population,
shard, method and parameters, and assigns shards to host classes by estimated cost
(the smallest third to the small class).  Workers claim shards exclusively, compute one
bounded block at a time, write an atomic local terminal and continue; a failed shard is
recorded with its exact reason and never stops the others.  The follower adopts valid
terminals, quarantines corrupt ones, submits rows to the warehouse through the DATA
agent's interface, verifies receipts and readback, closes phase 2 from evidence and
chains phase 3 automatically.

Verbs (all paths and identities by argument; nothing host-specific lives here)::

    plan          --manifest FILE --state-root DIR --n-shards N --host ID:CLASS [...]
    run-worker    --plan FILE --state-root DIR --data-root DIR --host-id ID [--max-shards N] [--steal] [--threads 1]
    follow        --plan FILE --state-root DIR --terminals DIR [...] --warehouse PATH --data-root DIR [--every 60] [--once]
    close-phase2  --plan FILE --state-root DIR --warehouse PATH --data-root DIR
    run-phase3    --plan FILE --state-root DIR --warehouse PATH --data-root DIR [--workers N]
    close-phase3  --plan FILE --state-root DIR --warehouse PATH
    status        --plan FILE [...] --terminals DIR [...] --state-root DIR [...] --out STATUS.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import shutil
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import fs_phase23_manifest as man  # noqa: E402
from tools import feature_pairwise_worker as pw  # noqa: E402

PLAN_SCHEMA = "fs_phase23.pairwise_plan.v1"
PHASE2_SCHEMA = "fs_phase23.phase2_complete.v1"
PHASE3_SCHEMA = "fs_phase23.phase3_filter_complete.v1"
CLAIM_STALE_SECONDS = 900
K_GRID = (4, 8, 12, 16, 24, 32)


class CampaignError(RuntimeError):
    """An explicit, reported driver failure."""


# ----------------------------------------------------------------------------- warehouse resolution

def open_warehouse(path: str | Path):
    """The DATA agent's module when present, else the marked-for-replacement adapter."""
    try:
        module = importlib.import_module("tools.fs_phase23_warehouse")
        if hasattr(module, "open_warehouse"):
            # Preserve service URLs. Path("http://...") collapses the second slash and
            # makes the warehouse client try to open the URL as a DuckDB filename.
            return module.open_warehouse(path)
    except ModuleNotFoundError as exc:
        if not exc.name.endswith("fs_phase23_warehouse"):
            raise
    from tools import fs_phase23_warehouse_adapter as adapter
    return adapter.open_warehouse(Path(path))


def rows_digest(rows: list[dict]) -> str:
    from tools.fs_phase23_warehouse_adapter import rows_digest as _rd
    return _rd(rows)


# ----------------------------------------------------------------------------- small helpers

def load_manifest(path: Path) -> dict:
    try:
        return man.load_manifest(path)
    except man.ManifestError as exc:
        raise CampaignError(str(exc)) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(man.canonical_bytes(value)).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _now() -> float:
    return time.time()


def _ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# ----------------------------------------------------------------------------- plan

def _pair_shard(identity: str, left: str, right: str, n_shards: int) -> int:
    h = hashlib.sha256(f"{identity}|{left}|{right}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n_shards


def all_pairs(features: list[str]) -> Iterable[tuple[str, str]]:
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            yield features[i], features[j]


def shard_pairs(plan: dict, shard_index: int) -> list[tuple[str, str]]:
    feats = plan["features"]
    n = plan["n_shards"]
    ident = plan["identity"]
    return [(l, r) for l, r in all_pairs(feats) if _pair_shard(ident, l, r, n) == shard_index]


def unit_id_for(population_sha256: str, shard_index: int, method: str, params_sha: str) -> str:
    return _digest({"population_sha256": population_sha256, "shard_index": shard_index, "method": method, "params_sha256": params_sha})


def plan(*, manifest_path: Path, state_root: Path, n_shards: int, hosts: list[dict], params: dict | None = None,
         small_fraction: float = 1.0 / 3.0) -> dict:
    manifest = load_manifest(manifest_path)
    for key in ("features_file", "targets_file"):
        name = manifest["data"][key].lower()
        if any(tok in name for tok in man.FORBIDDEN_SPLIT_TOKENS):
            raise CampaignError(f"manifest {key} names validation/test data: {manifest['data'][key]}")
    if n_shards < 1:
        raise CampaignError("n_shards must be >= 1")
    if not hosts:
        raise CampaignError("at least one host is required")
    merged = dict(pw.DEFAULT_PARAMS)
    merged.update(params or {})
    psha = pw.params_sha256(merged)
    feats = [f["feature_id"] for f in manifest["features"]]
    cov = {f["feature_id"]: (f.get("train_coverage") if isinstance(f.get("train_coverage"), (int, float)) else 1.0) for f in manifest["features"]}
    n_rows = int(manifest["data"]["train_rows"])
    fold_count = len(manifest["folds"])
    counts = [0] * n_shards
    costs = [0.0] * n_shards
    for l, r in all_pairs(feats):
        s = _pair_shard(manifest["identity"], l, r, n_shards)
        counts[s] += 1
        costs[s] += n_rows * min(cov[l], cov[r]) * (fold_count + 1)
    shards = []
    for s in range(n_shards):
        shards.append({"shard_index": s, "pair_count": counts[s], "estimated_cost": round(costs[s], 3),
                       "unit_id": unit_id_for(manifest["population_sha256"], s, merged["method"], psha), "host_id": None})
    small = [h["host_id"] for h in hosts if h.get("size_class") == "small"]
    large = [h["host_id"] for h in hosts if h.get("size_class") != "small"]
    order = sorted(range(n_shards), key=lambda s: (costs[s], s))
    n_small = int(round(n_shards * small_fraction)) if (small and large) else (n_shards if not large else 0)
    for k, s in enumerate(order):
        if k < n_small:
            shards[s]["host_id"] = small[k % len(small)]
        else:
            pool = large or small
            shards[s]["host_id"] = pool[(k - n_small) % len(pool)]
    doc = {
        "schema": PLAN_SCHEMA, "population_id": manifest["population_id"], "identity": manifest["identity"],
        "manifest_sha256": manifest["manifest_sha256"], "population_sha256": manifest["population_sha256"],
        "manifest_file": Path(manifest_path).name, "features": feats, "fold_ids": [f["fold_id"] for f in manifest["folds"]],
        "targets": [t["target_id"] for t in manifest["targets"]], "train_rows": n_rows,
        "expected_pairs": manifest["expected_pairs"], "params": merged, "params_sha256": psha, "method": merged["method"],
        "n_shards": n_shards, "pair_assignment_rule": "sha256(identity|left|right)[:8] mod n_shards",
        "hosts": [{"host_id": h["host_id"], "size_class": h.get("size_class", "large")} for h in hosts],
        "host_assignment_rule": f"smallest {small_fraction:.3f} of shards by estimated cost to size_class=small, rest to large",
        "shards": shards, "k_grid": list(K_GRID),
        "expected_row_counts_per_shard": {s["unit_id"]: pw.expected_row_counts(s["pair_count"], fold_count, merged) for s in shards},
    }
    doc["plan_sha256"] = _digest(doc)
    state_root = Path(state_root)
    for sub in ("terminals", "claims", "failures", "adopted", "quarantine", "receipts", "phase3"):
        (state_root / sub).mkdir(parents=True, exist_ok=True)
    existing = state_root / "PLAN.json"
    if existing.exists():
        old = _read_json(existing)
        if old.get("plan_sha256") != doc["plan_sha256"]:
            raise CampaignError("a different plan already exists in this state root; use a new state root")
    else:
        _write_json_atomic(existing, doc)
    shutil.copyfile(manifest_path, state_root / "MANIFEST.json")
    return doc


def load_plan(path: Path) -> dict:
    doc = _read_json(path)
    if doc.get("schema") != PLAN_SCHEMA:
        raise CampaignError(f"unexpected plan schema {doc.get('schema')!r}")
    body = {k: v for k, v in doc.items() if k != "plan_sha256"}
    if _digest(body) != doc["plan_sha256"]:
        raise CampaignError("plan digest mismatch")
    return doc


# ----------------------------------------------------------------------------- terminals

def seal_terminal(doc: dict) -> dict:
    doc = dict(doc)
    doc["row_counts"] = {t: len(r) for t, r in doc["rows"].items()}
    doc["rows_sha256"] = _digest(doc["rows"])
    doc["terminal_sha256"] = _digest({k: v for k, v in doc.items() if k != "terminal_sha256"})
    return doc


def terminal_path(root: Path, unit_id: str) -> Path:
    return Path(root) / f"{unit_id}.json.gz"


def write_terminal(path: Path, sealed: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump(sealed, fh, sort_keys=True, separators=(",", ":"))
    os.replace(tmp, path)
    meta = {k: v for k, v in sealed.items() if k != "rows"}
    _write_json_atomic(path.with_name(path.name.replace(".json.gz", ".meta.json")), meta)


def load_terminal(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def validate_terminal(plan_doc: dict, doc: dict) -> dict:
    """Return the plan shard the terminal completes, or raise CampaignError with the exact reason."""
    if not isinstance(doc, dict) or doc.get("schema") != pw.SCHEMA_TERMINAL:
        raise CampaignError("terminal schema mismatch")
    if doc.get("identity") != plan_doc["identity"] or doc.get("population_id") != plan_doc["population_id"]:
        raise CampaignError(f"terminal identity {doc.get('identity')!r} is foreign to plan {plan_doc['identity']!r}")
    units = {s["unit_id"]: s for s in plan_doc["shards"]}
    shard = units.get(doc.get("unit_id"))
    if shard is None:
        raise CampaignError(f"terminal unit {doc.get('unit_id')!r} is not in the plan")
    if doc.get("params_sha256") != plan_doc["params_sha256"] or doc.get("method") != plan_doc["method"]:
        raise CampaignError("terminal method/params differ from the plan")
    if doc.get("shard_index") != shard["shard_index"]:
        raise CampaignError("terminal shard index differs from its unit")
    body = {k: v for k, v in doc.items() if k != "terminal_sha256"}
    if _digest(body) != doc.get("terminal_sha256"):
        raise CampaignError("terminal content does not match terminal_sha256 (corrupt)")
    if _digest(doc["rows"]) != doc.get("rows_sha256"):
        raise CampaignError("terminal rows do not match rows_sha256 (corrupt)")
    counts = {t: len(r) for t, r in doc["rows"].items()}
    if counts != doc.get("row_counts"):
        raise CampaignError("terminal row_counts disagree with its rows")
    expected = plan_doc["expected_row_counts_per_shard"][shard["unit_id"]]
    if counts != expected:
        raise CampaignError(f"terminal row counts {counts} != expected {expected}")
    if doc.get("pair_count") != shard["pair_count"]:
        raise CampaignError("terminal pair count differs from the plan")
    return shard


# ----------------------------------------------------------------------------- claims

def claim_path(state_root: Path, unit_id: str) -> Path:
    return Path(state_root) / "claims" / f"{unit_id}.json"


def write_claim(state_root: Path, unit_id: str, *, host_id: str) -> bool:
    path = claim_path(state_root, unit_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"unit_id": unit_id, "host_id": host_id, "pid": os.getpid(), "claimed_at": _now(), "node": socket.gethostname()[:0] or "worker"}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            old = _read_json(path)
        except Exception:  # noqa: BLE001
            old = {}
        if _now() - float(old.get("claimed_at", 0)) < CLAIM_STALE_SECONDS:
            return False
        path.unlink(missing_ok=True)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh)
    return True


def release_claim(state_root: Path, unit_id: str) -> None:
    claim_path(state_root, unit_id).unlink(missing_ok=True)


# ----------------------------------------------------------------------------- worker

def run_worker(*, plan_path: Path, state_root: Path, data_root: Path, host_id: str, max_shards: int | None = None,
               steal: bool = False, threads: int = 1) -> dict:
    pw.limit_numeric_threads(threads)
    plan_doc = load_plan(plan_path)
    state_root = Path(state_root)
    manifest = load_manifest(state_root / "MANIFEST.json")
    if manifest["manifest_sha256"] != plan_doc["manifest_sha256"]:
        raise CampaignError("state-root manifest differs from the plan's manifest")
    tdir = state_root / "terminals"
    tdir.mkdir(parents=True, exist_ok=True)
    summary = {"host_id": host_id, "computed": 0, "adopted_existing": 0, "skipped_claimed": 0, "failed": 0,
               "quarantined": 0, "units": [], "wall_seconds": 0.0, "peak_rss_bytes": 0, "pairs_computed": 0}
    mine = [s for s in plan_doc["shards"] if s["host_id"] == host_id]
    others = [s for s in plan_doc["shards"] if s["host_id"] != host_id] if steal else []
    data = None
    t0 = _now()
    for shard in [*mine, *others]:
        if max_shards is not None and summary["computed"] >= max_shards:
            break
        uid = shard["unit_id"]
        path = terminal_path(tdir, uid)
        if path.exists():
            try:
                validate_terminal(plan_doc, load_terminal(path))
                summary["adopted_existing"] += 1
                continue
            except (CampaignError, OSError, ValueError, EOFError) as exc:
                qdir = state_root / "quarantine"
                qdir.mkdir(exist_ok=True)
                shutil.move(str(path), qdir / f"{uid}.{_ts()}.json.gz")
                _write_json_atomic(qdir / f"{uid}.{_ts()}.reason.json", {"unit_id": uid, "reason": str(exc), "host_id": host_id})
                summary["quarantined"] += 1
        if not write_claim(state_root, uid, host_id=host_id):
            summary["skipped_claimed"] += 1
            continue
        try:
            if data is None:
                data = pw.PopulationData(manifest, data_root)
            pairs = shard_pairs(plan_doc, shard["shard_index"])
            doc = pw.compute_shard(data, plan_doc, shard["shard_index"], pairs, host_id)
            sealed = seal_terminal(doc)
            validate_terminal(plan_doc, sealed)
            write_terminal(path, sealed)
            summary["computed"] += 1
            summary["pairs_computed"] += len(pairs)
            summary["peak_rss_bytes"] = max(summary["peak_rss_bytes"], sealed["peak_rss_bytes"])
            summary["units"].append({"unit_id": uid, "wall_seconds": sealed["wall_seconds"], "pairs": len(pairs)})
        except Exception as exc:  # noqa: BLE001 - one defective shard never stops the rest
            summary["failed"] += 1
            _write_json_atomic(state_root / "failures" / f"{uid}.{_ts()}.json",
                               {"unit_id": uid, "host_id": host_id, "reason": f"{type(exc).__name__}: {exc}", "at": _now(),
                                "shard_index": shard["shard_index"]})
            if isinstance(exc, (pw.WorkerError, CampaignError)) and data is None:
                release_claim(state_root, uid)
                break      # cannot load the population at all: every later shard would fail the same way
        finally:
            release_claim(state_root, uid)
    summary["wall_seconds"] = _now() - t0
    return summary


# ----------------------------------------------------------------------------- receipts and follower

def accept_receipt(plan_doc: dict, terminal: dict, receipt: dict) -> dict:
    if receipt.get("run_id") != plan_doc["identity"]:
        raise CampaignError(f"receipt identity {receipt.get('run_id')!r} is foreign to {plan_doc['identity']!r}")
    table = receipt.get("table")
    rows = terminal["rows"].get(table)
    if rows is None:
        raise CampaignError(f"receipt names unknown table {table!r}")
    expected = rows_digest(rows)
    if receipt.get("rows_sha256") != expected:
        raise CampaignError(f"receipt digest {str(receipt.get('rows_sha256'))[:16]} != submitted rows digest {expected[:16]}")
    if int(receipt.get("row_count", -1)) != len(rows):
        raise CampaignError("receipt row count differs from submitted rows")
    return {"unit_id": terminal["unit_id"], "table": table, "rows_sha256": expected, "row_count": len(rows),
            "receipt_sha256": receipt.get("receipt_sha256"), "inserted": receipt.get("inserted"),
            "duplicates_ignored": receipt.get("duplicates_ignored"), "backend": receipt.get("backend")}


SUBMIT_BATCH_ROWS = int(os.environ.get("FS23_SUBMIT_BATCH_ROWS", "2000"))


def _readback_matches(wh, run_id: str, table: str, unit_id: str | None, rows: list[dict], expected_digest: str) -> int:
    """Readback of one unit: (count, digest) computed in the store when the handle offers it, else the
    rows themselves.  Returns the stored count; raises CampaignError on any disagreement."""
    summary = getattr(wh, "readback_summary", None)
    if callable(summary):
        got = summary(run_id, table, unit_id)
        if int(got.get("count", -1)) != len(rows) or got.get("rows_sha256") != expected_digest:
            raise CampaignError(f"readback of {table} for unit {str(unit_id)[:12]} does not match the submitted rows "
                                f"(stored {got.get('count')} rows, digest {str(got.get('rows_sha256'))[:12]})")
        return int(got["count"])
    try:
        back = wh.read_run(run_id, table, unit_id)
    except TypeError:
        back = [r for r in wh.read_run(run_id, table) if r.get("unit_id") == unit_id]
    if rows_digest(back) != expected_digest or len(back) != len(rows):
        raise CampaignError(f"readback of {table} for unit {str(unit_id)[:12]} does not match the submitted rows")
    return len(back)


def submit_rows_batched(wh, run_id: str, table: str, rows: list[dict], *, host_role: str | None, batch_rows: int | None = None) -> dict:
    """Submit one table of one unit in bounded batches (incident 2026-10-06: a 26k-row POST drove the
    host over its memory cap).  Every batch receipt must carry the run identity and the digest of
    exactly that batch; the table-level record carries the digest of all rows."""
    batch_rows = batch_rows or SUBMIT_BATCH_ROWS
    receipts = []
    inserted = duplicates = 0
    for start in range(0, len(rows), max(1, batch_rows)):
        batch = rows[start:start + batch_rows]
        receipt = wh.submit_rows(run_id, table, batch, host_role=host_role)
        if receipt.get("run_id") != run_id:
            raise CampaignError(f"receipt identity {receipt.get('run_id')!r} is foreign to {run_id!r}")
        expected = rows_digest(batch)
        if receipt.get("rows_sha256") != expected:
            raise CampaignError(f"receipt digest {str(receipt.get('rows_sha256'))[:16]} != submitted batch digest {expected[:16]}")
        if int(receipt.get("row_count", -1)) != len(batch):
            raise CampaignError("receipt row count differs from submitted batch")
        inserted += int(receipt.get("inserted") or 0)
        duplicates += int(receipt.get("duplicates_ignored") or 0)
        receipts.append({"receipt_sha256": receipt.get("receipt_sha256"), "rows_sha256": expected, "row_count": len(batch),
                         "inserted": receipt.get("inserted"), "duplicates_ignored": receipt.get("duplicates_ignored")})
    return {"table": table, "rows_sha256": rows_digest(rows), "row_count": len(rows), "batches": receipts,
            "batch_rows": batch_rows, "inserted": inserted, "duplicates_ignored": duplicates,
            "backend": getattr(wh, "backend", None)}


def _reconcile(wh, run_id: str, expected: dict) -> dict:
    """reconcile(run_id, expected=...) when the handle accepts expectations (the DATA store), else reconcile(run_id)."""
    try:
        return wh.reconcile(run_id, expected=expected)
    except TypeError:
        return wh.reconcile(run_id)


def _submit_terminal(wh, plan_doc: dict, terminal: dict, receipts_dir: Path, batch_rows: int | None = None) -> dict:
    uid = terminal["unit_id"]
    out = {"unit_id": uid, "tables": {}}
    for table, rows in terminal["rows"].items():
        accepted = submit_rows_batched(wh, plan_doc["identity"], table, rows, host_role=terminal.get("host_id"), batch_rows=batch_rows)
        accepted["unit_id"] = uid
        accepted["readback_count"] = _readback_matches(wh, plan_doc["identity"], table, uid, rows, accepted["rows_sha256"])
        out["tables"][table] = accepted
    out["accepted_at"] = _now()
    _write_json_atomic(receipts_dir / f"{uid}.json", out)
    return out


def _scan_terminals(plan_doc: dict, state_root: Path, terminal_dirs: list[Path]) -> dict:
    adopted_dir = state_root / "adopted"
    qdir = state_root / "quarantine"
    adopted_dir.mkdir(parents=True, exist_ok=True)
    qdir.mkdir(parents=True, exist_ok=True)
    units = {s["unit_id"] for s in plan_doc["shards"]}
    res = {"adopted": 0, "quarantined": 0, "already": 0, "foreign": 0}
    for tdir in terminal_dirs:
        for path in sorted(Path(tdir).glob("*.json.gz")):
            uid = path.name[: -len(".json.gz")]
            if uid not in units:
                res["foreign"] += 1
                continue
            target = adopted_dir / path.name
            if target.exists():
                res["already"] += 1
                continue
            try:
                doc = load_terminal(path)
                validate_terminal(plan_doc, doc)
            except (CampaignError, OSError, ValueError, EOFError, KeyError, TypeError) as exc:
                stamp = _ts()
                shutil.copyfile(path, qdir / path.name)
                _write_json_atomic(qdir / f"{uid}.{stamp}.reason.json", {"unit_id": uid, "source": path.name, "reason": str(exc)})
                res["quarantined"] += 1
                continue
            tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
            shutil.copyfile(path, tmp)
            os.replace(tmp, target)
            meta_src = path.with_name(path.name.replace(".json.gz", ".meta.json"))
            if meta_src.exists():
                shutil.copyfile(meta_src, target.with_name(target.name.replace(".json.gz", ".meta.json")))
            else:
                _write_json_atomic(target.with_name(target.name.replace(".json.gz", ".meta.json")),
                                   {k: v for k, v in doc.items() if k != "rows"})
            res["adopted"] += 1
    return res


def follow_once(*, plan_path: Path, state_root: Path, terminal_dirs: list[Path], warehouse_path: Path, data_root: Path,
                chain_phase3: bool = True, phase3_workers: int = 1, warehouse=None) -> dict:
    plan_doc = load_plan(plan_path)
    state_root = Path(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    if not (state_root / "PLAN.json").exists():
        _write_json_atomic(state_root / "PLAN.json", plan_doc)
        src = Path(plan_path).with_name("MANIFEST.json")
        if src.exists():
            shutil.copyfile(src, state_root / "MANIFEST.json")
    res = _scan_terminals(plan_doc, state_root, [Path(d) for d in terminal_dirs])
    wh = warehouse or open_warehouse(warehouse_path)
    receipts_dir = state_root / "receipts"
    receipts_dir.mkdir(exist_ok=True)
    res["submitted"] = 0
    res["submit_errors"] = []
    failures_dir = state_root / "submit_failures"
    for path in sorted((state_root / "adopted").glob("*.json.gz")):
        if _STOP.is_set():
            break
        uid = path.name[: -len(".json.gz")]
        if (receipts_dir / f"{uid}.json").exists():
            continue
        try:
            _submit_terminal(wh, plan_doc, load_terminal(path), receipts_dir)
            res["submitted"] += 1
            (failures_dir / f"{uid}.json").unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - a store refusal/engine error on one unit never stops the follower
            reason = f"{type(exc).__name__}: {exc}"
            res["submit_errors"].append({"unit_id": uid, "reason": reason})
            _write_json_atomic(failures_dir / f"{uid}.json", {"unit_id": uid, "reason": reason, "at": _now(), "stage": "submit"})
    res["phase2_closed"] = (state_root / "PHASE_2_COMPLETE.json").exists()
    res["closure_error"] = None
    receipted = {p.name[:-5] for p in receipts_dir.glob("*.json")}
    if not res["phase2_closed"] and all(s["unit_id"] in receipted for s in plan_doc["shards"]):
        try:
            close_phase2(plan_path=plan_path, state_root=state_root, warehouse_path=warehouse_path, data_root=data_root, warehouse=wh)
            res["phase2_closed"] = True
        except CampaignError as exc:
            res["closure_error"] = str(exc)
    res["phase3_closed"] = (state_root / "PHASE_3_FILTER_COMPLETE.json").exists()
    res["phase3_error"] = None
    if res["phase2_closed"] and chain_phase3 and not res["phase3_closed"]:
        try:
            run_phase3(plan_path=plan_path, state_root=state_root, data_root=data_root, warehouse_path=warehouse_path,
                       workers=phase3_workers, warehouse=wh)
            close_phase3(plan_path=plan_path, state_root=state_root, warehouse_path=warehouse_path, warehouse=wh)
            res["phase3_closed"] = True
        except CampaignError as exc:
            res["phase3_error"] = str(exc)
    from tools import feature_selection_phase23_status as st
    # the follower's view counts only validated (adopted) terminals as complete evidence
    st.build_status(plan_paths=[plan_path], terminal_dirs=[state_root / "adopted"],
                    state_roots=[state_root], out_path=state_root / "STATUS.json", follower_result=res)
    return res


class _StopFlag:
    def __init__(self) -> None:
        self._set = False

    def set(self, *_: Any) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


_STOP = _StopFlag()


def follow(*, every: float = 60.0, once: bool = False, **kwargs) -> None:
    """Durable loop: a failed pass is reported on stderr and retried next cycle; SIGTERM finishes the
    unit in flight (no transaction is cut mid-write) and then returns."""
    try:
        signal.signal(signal.SIGTERM, _STOP.set)
    except (ValueError, OSError):  # not the main thread
        pass
    while True:
        try:
            res = follow_once(**kwargs)
        except Exception as exc:  # noqa: BLE001 - never crash-loop on one bad pass
            import traceback
            print(json.dumps({"follow_pass_error": f"{type(exc).__name__}: {exc}", "at": _ts()}), file=sys.stderr)
            traceback.print_exc()
            res = {"phase3_closed": False}
        if once or res.get("phase3_closed") or _STOP.is_set():
            return
        deadline = time.time() + max(1.0, every)
        while time.time() < deadline and not _STOP.is_set():
            time.sleep(1.0)
        if _STOP.is_set():
            return


# ----------------------------------------------------------------------------- alias groups and clusters

def _rep_sort_key(meta: dict):
    clock = meta.get("clock") or ""
    cov = meta.get("train_coverage")
    cov = float(cov) if isinstance(cov, (int, float)) else 0.0
    support = meta.get("support_h")
    support = float(support) if isinstance(support, (int, float)) else float("inf")
    cost = meta.get("source_bytes")
    cost = float(cost) if isinstance(cost, (int, float)) else float("inf")
    return (0 if clock == "OBSERVED" else 1, -cov, support, cost, meta["feature_id"])


def alias_groups_from_gate_rows(gate_rows: list[dict], manifest: dict) -> list[dict]:
    meta = {f["feature_id"]: f for f in manifest["features"]}
    parent = {f: f for f in meta}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges: dict[tuple[str, str], str] = {}
    for r in gate_rows:
        if r["gate_state"] in pw.GATE_ALIAS_STATES:
            parent[find(r["left"])] = find(r["right"])
            edges[(r["left"], r["right"])] = r["gate_state"]
    groups: dict[str, list[str]] = {}
    for f in meta:
        groups.setdefault(find(f), []).append(f)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        rep = sorted(members, key=lambda f: _rep_sort_key(meta[f]))[0]
        gid = _digest({"identity": manifest["identity"], "members": members})
        out.append({"run_id": manifest["identity"], "population_id": manifest["population_id"], "alias_group_id": gid,
                    "members": members, "representative": rep, "representative_rule": "clock OBSERVED, coverage desc, support_h asc, source_bytes asc, name",
                    "evidence": [{"left": l, "right": r, "gate_state": s} for (l, r), s in sorted(edges.items()) if l in members and r in members],
                    "disposition": "ALIAS_GROUP", "dropped_columns": [], "row_key": pw.row_key(manifest["identity"], "alias", gid)})
    out.sort(key=lambda g: g["alias_group_id"])
    return out


def redundancy_clusters(features: list[str], spearman: "np.ndarray", admissible: list[bool], manifest: dict, params: dict) -> list[dict]:
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    idx = [i for i, ok in enumerate(admissible) if ok]
    if len(idx) < 2:
        return []
    sub = np.abs(spearman[np.ix_(idx, idx)])
    sub = np.where(np.isfinite(sub), sub, 0.0)
    dist = 1.0 - sub
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    Z = linkage(squareform(dist, checks=False), method=params["redundancy_linkage"])
    labels = fcluster(Z, t=1.0 - float(params["redundancy_abs_spearman"]), criterion="distance")
    meta = {f["feature_id"]: f for f in manifest["features"]}
    clusters: dict[int, list[str]] = {}
    for k, i in zip(labels, idx):
        clusters.setdefault(int(k), []).append(features[i])
    out = []
    for members in clusters.values():
        members = sorted(members)
        rep = sorted(members, key=lambda f: _rep_sort_key(meta[f]))[0]
        cid = _digest({"identity": manifest["identity"], "members": members, "rule": "spearman"})
        out.append({"run_id": manifest["identity"], "population_id": manifest["population_id"], "cluster_id": cid,
                    "members": members, "size": len(members), "representative": rep,
                    "rule": f"average-linkage on 1-|spearman_TRAIN|, cut at |rho|>={params['redundancy_abs_spearman']}",
                    "row_key": pw.row_key(manifest["identity"], "cluster", cid)})
    out.sort(key=lambda c: c["cluster_id"])
    return out


# ----------------------------------------------------------------------------- phase 2 closure

def close_phase2(*, plan_path: Path, state_root: Path, warehouse_path: Path, data_root: Path, warehouse=None) -> dict:
    import numpy as np
    plan_doc = load_plan(plan_path)
    state_root = Path(state_root)
    manifest = load_manifest(state_root / "MANIFEST.json") if (state_root / "MANIFEST.json").exists() else load_manifest(Path(plan_path).with_name("MANIFEST.json"))
    if manifest["manifest_sha256"] != plan_doc["manifest_sha256"]:
        raise CampaignError("manifest/plan mismatch at closure")
    adopted_dir = state_root / "adopted"
    receipts_dir = state_root / "receipts"
    feats = plan_doc["features"]
    p = len(feats)
    index = {f: i for i, f in enumerate(feats)}
    spearman = np.full((p, p), np.nan)
    mi = np.full((p, p), np.nan)
    np.fill_diagonal(spearman, 1.0)
    np.fill_diagonal(mi, np.nan)
    gate_rows: list[dict] = []
    table_digests: dict[str, list[str]] = {"feature_pair_metrics": [], "feature_pair_stability": [], "feature_pair_gate": []}
    table_counts = {t: 0 for t in table_digests}
    missing_units = []
    missing_keys: list[str] = []
    observed_pairs = 0
    unit_evidence = []
    state_counts: dict[str, int] = {}
    for shard in plan_doc["shards"]:
        uid = shard["unit_id"]
        path = terminal_path(adopted_dir, uid)
        if not path.exists():
            missing_units.append(uid)
            continue
        if not (receipts_dir / f"{uid}.json").exists():
            raise CampaignError(f"unit {uid} adopted but has no warehouse receipt")
        doc = load_terminal(path)
        try:
            validate_terminal(plan_doc, doc)
        except CampaignError as exc:
            if "row counts" not in str(exc):
                raise
            # a sealed terminal with fewer rows than expected: name the missing dispositions below
        pairs = shard_pairs(plan_doc, shard["shard_index"])
        expected = pw.expected_row_keys(plan_doc["identity"], pairs, plan_doc["fold_ids"], plan_doc["params"])
        for table, keys in expected.items():
            observed = {r["row_key"] for r in doc["rows"][table]}
            gone = sorted(keys - observed)
            if gone:
                missing_keys.extend(f"{table}:{k}" for k in gone)
            elif len(observed) != len(doc["rows"][table]):
                raise CampaignError(f"unit {uid} table {table} carries duplicate row keys")
            if observed - keys:
                raise CampaignError(f"unit {uid} table {table} carries {len(observed - keys)} unexpected row keys")
        for table, rows in doc["rows"].items():
            table_digests[table].extend(hashlib.sha256(man.canonical_bytes(r)).hexdigest() for r in rows)
            table_counts[table] += len(rows)
        for r in doc["rows"]["feature_pair_metrics"]:
            state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1
            if r["fold_id"] == "TRAIN" and r["lag_hours"] == 0 and r["state"] == "MEASURED":
                i, j = index[r["left"]], index[r["right"]]
                if r["metric"] == "spearman":
                    spearman[i, j] = spearman[j, i] = r["value"]
                elif r["metric"] == "mutual_information":
                    mi[i, j] = mi[j, i] = r["value"]
            elif r["fold_id"] == "TRAIN" and r["lag_hours"] == 0 and r["metric"] in ("spearman", "mutual_information"):
                i, j = index[r["left"]], index[r["right"]]
                if r["metric"] == "spearman":
                    spearman[i, j] = spearman[j, i] = 0.0 if r["state"] != "MEASURED" else r["value"]
                else:
                    mi[i, j] = mi[j, i] = 0.0
        gate_rows.extend(doc["rows"]["feature_pair_gate"])
        observed_pairs += doc["pair_count"]
        unit_evidence.append({"unit_id": uid, "host_id": doc["host_id"], "pairs": doc["pair_count"], "wall_seconds": doc["wall_seconds"],
                              "peak_rss_bytes": doc["peak_rss_bytes"], "rows_sha256": doc["rows_sha256"]})
    if missing_units:
        raise CampaignError(f"phase 2 cannot close: {len(missing_units)} units without adopted terminal, first {missing_units[0]}")
    if missing_keys:
        raise CampaignError(f"phase 2 cannot close: {len(missing_keys)} expected dispositions missing, first {missing_keys[0]}")
    if observed_pairs != plan_doc["expected_pairs"]:
        raise CampaignError(f"observed pairs {observed_pairs} != expected {plan_doc['expected_pairs']}")
    # alias groups and redundancy clusters from evidence
    aliases = alias_groups_from_gate_rows(gate_rows, manifest)
    in_alias_non_rep = {m for g in aliases for m in g["members"] if m != g["representative"]}
    admissible = [f not in in_alias_non_rep for f in feats]
    clusters = redundancy_clusters(feats, spearman, admissible, manifest, plan_doc["params"])
    wh = warehouse or open_warehouse(warehouse_path)
    extra_receipts = {}
    for table, rows in (("feature_alias_groups", aliases), ("feature_redundancy_clusters", clusters)):
        if rows:
            extra_receipts[table] = submit_rows_batched(wh, plan_doc["identity"], table, rows, host_role="coordinator")
            extra_receipts[table]["readback_count"] = _readback_matches(wh, plan_doc["identity"], table, None, rows, extra_receipts[table]["rows_sha256"])
        else:
            extra_receipts[table] = {"rows_sha256": rows_digest(rows), "row_count": 0, "batches": [], "note": "no rows"}
        table_digests[table] = [hashlib.sha256(man.canonical_bytes(r)).hexdigest() for r in rows]
        table_counts[table] = len(rows)
    # readback reconciliation
    recon = _reconcile(wh, plan_doc["identity"], {t: n for t, n in table_counts.items()})
    if recon.get("run_id") != plan_doc["identity"]:
        raise CampaignError("warehouse reconcile answered for a foreign identity")
    mismatches = []
    for table, shas in table_digests.items():
        expected_digest = hashlib.sha256("".join(sorted(shas)).encode()).hexdigest()
        got = recon["tables"].get(table, {})
        if got.get("count") != table_counts[table] or got.get("rows_sha256") != expected_digest:
            mismatches.append({"table": table, "expected_count": table_counts[table], "warehouse_count": got.get("count"),
                               "expected_sha256": expected_digest, "warehouse_sha256": got.get("rows_sha256")})
    if mismatches:
        raise CampaignError(f"warehouse readback does not reconcile: {mismatches[0]}")
    np.savez_compressed(state_root / "PHASE2_MATRICES.npz", features=np.array(feats), spearman=spearman, mi=mi,
                        admissible=np.array(admissible), population_id=plan_doc["population_id"], identity=plan_doc["identity"],
                        alias_representative=np.array([next((g["representative"] for g in aliases if f in g["members"]), f) for f in feats]),
                        params_sha256=plan_doc["params_sha256"])
    closure = {
        "schema": PHASE2_SCHEMA, "state": "PHASE_2_COMPLETE", "generated_from_evidence": True,
        "population_id": plan_doc["population_id"], "identity": plan_doc["identity"], "plan_sha256": plan_doc["plan_sha256"],
        "manifest_sha256": plan_doc["manifest_sha256"], "params_sha256": plan_doc["params_sha256"],
        "expected_pairs": plan_doc["expected_pairs"], "observed_pairs": observed_pairs, "units": len(unit_evidence),
        "fold_ids": plan_doc["fold_ids"], "metric_slots": pw.metric_slots(plan_doc["params"]),
        "row_counts": table_counts, "table_sha256": {t: hashlib.sha256("".join(sorted(s)).encode()).hexdigest() for t, s in table_digests.items()},
        "metric_state_counts": state_counts, "alias_groups": len(aliases), "features_in_alias_groups": sum(len(g["members"]) for g in aliases),
        "admissible_features": int(sum(admissible)), "redundancy_clusters": len(clusters),
        "warehouse_reconcile": recon, "extra_receipts": extra_receipts, "unit_evidence": unit_evidence,
        "closed_at": _now(), "closed_at_utc": _ts(),
        "uses_validation_or_test": False,
    }
    closure["closure_sha256"] = _digest({k: v for k, v in closure.items() if k != "closure_sha256"})
    _write_json_atomic(state_root / "PHASE_2_COMPLETE.json", closure)
    _write_json_atomic(state_root / "ALIAS_GROUPS.json", aliases)
    _write_json_atomic(state_root / "REDUNDANCY_CLUSTERS.json", clusters)
    return closure


# ----------------------------------------------------------------------------- phase 3

def _phase3_unit_id(plan_doc: dict, target_id: str, params_sha: str) -> str:
    return _digest({"population_sha256": plan_doc["population_sha256"], "phase2_params": plan_doc["params_sha256"],
                    "target_id": target_id, "method": "filter_v1", "params_sha256": params_sha})


def _phase3_target(args: tuple) -> dict:
    plan_doc, state_root, data_root, target_id, seed = args
    from tools import feature_filter_selection as sel
    pw.limit_numeric_threads(1)
    manifest = load_manifest(Path(state_root) / "MANIFEST.json")
    mats = sel.load_phase2_matrices(Path(state_root) / "PHASE2_MATRICES.npz")
    feats, X, y = sel.load_train_matrix(manifest, data_root, target_id)
    return sel.run_filter_methods(manifest, mats, feats, X, y, target_id=target_id, seed=seed, k_grid=tuple(plan_doc["k_grid"]))


def _phase3_terminal_valid(path: Path) -> bool:
    try:
        doc = load_terminal(path)
        return _digest({k: v for k, v in doc.items() if k != "terminal_sha256"}) == doc.get("terminal_sha256") \
            and _digest(doc["rows"]) == doc.get("rows_sha256")
    except Exception:  # noqa: BLE001
        return False


def run_phase3(*, plan_path: Path, state_root: Path, data_root: Path, warehouse_path: Path, workers: int = 1,
               seed: int = 20261005, warehouse=None) -> dict:
    from tools import feature_filter_selection as sel
    plan_doc = load_plan(plan_path)
    state_root = Path(state_root)
    if not (state_root / "PHASE_2_COMPLETE.json").exists():
        raise CampaignError("phase 3 requires PHASE_2_COMPLETE.json in the state root")
    p3 = state_root / "phase3"
    (p3 / "terminals").mkdir(parents=True, exist_ok=True)
    (p3 / "receipts").mkdir(parents=True, exist_ok=True)
    params_sha = sel.params_sha256(sel.default_params(seed=seed, k_grid=tuple(plan_doc["k_grid"])))
    wh = warehouse or open_warehouse(warehouse_path)
    todo = []
    summary = {"computed": 0, "existing": 0, "submitted": 0, "failed": [], "units": {}}
    for target_id in plan_doc["targets"]:
        uid = _phase3_unit_id(plan_doc, target_id, params_sha)
        summary["units"][target_id] = uid
        existing = p3 / "terminals" / f"{uid}.json.gz"
        if existing.exists() and _phase3_terminal_valid(existing):
            summary["existing"] += 1
        else:
            if existing.exists():   # sealed under a digest that does not verify: quarantine and recompute (cheap)
                qdir = p3 / "quarantine"
                qdir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(existing), qdir / f"{uid}.{_ts()}.json.gz")
                meta = existing.with_name(existing.name.replace(".json.gz", ".meta.json"))
                if meta.exists():
                    meta.unlink()
                (p3 / "receipts" / f"{uid}.json").unlink(missing_ok=True)
                summary.setdefault("quarantined", 0)
                summary["quarantined"] += 1
            todo.append((plan_doc, str(state_root), str(data_root), target_id, seed))
    results = []
    if todo:
        if workers > 1 and len(todo) > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
                results = pool.map(_phase3_target, todo)
        else:
            results = [_phase3_target(t) for t in todo]
    for res in results:
        uid = _phase3_unit_id(plan_doc, res["target_id"], params_sha)
        res = dict(res, unit_id=uid, schema="fs_phase23.filter_terminal.v1", plan_sha256=plan_doc["plan_sha256"])
        for table in ("feature_filter_rankings", "feature_filter_subsets"):
            for r in res["rows"][table]:
                r["unit_id"] = uid
        sealed = seal_terminal(res)
        write_terminal(p3 / "terminals" / f"{uid}.json.gz", sealed)
        summary["computed"] += 1
    for target_id, uid in summary["units"].items():
        rpath = p3 / "receipts" / f"{uid}.json"
        if rpath.exists():
            continue
        doc = load_terminal(p3 / "terminals" / f"{uid}.json.gz")
        out = {"unit_id": uid, "target_id": target_id, "tables": {}}
        for table, rows in doc["rows"].items():
            accepted = submit_rows_batched(wh, plan_doc["identity"], table, rows, host_role="coordinator")
            accepted["readback_count"] = _readback_matches(wh, plan_doc["identity"], table, uid, rows, accepted["rows_sha256"])
            out["tables"][table] = accepted
        _write_json_atomic(rpath, out)
        summary["submitted"] += 1
    return summary


def close_phase3(*, plan_path: Path, state_root: Path, warehouse_path: Path, warehouse=None) -> dict:
    from tools import feature_filter_selection as sel
    plan_doc = load_plan(plan_path)
    state_root = Path(state_root)
    phase2 = _read_json(state_root / "PHASE_2_COMPLETE.json") if (state_root / "PHASE_2_COMPLETE.json").exists() else None
    if phase2 is None:
        raise CampaignError("phase 3 closure requires PHASE_2_COMPLETE.json")
    p3 = state_root / "phase3"
    params_sha = sel.params_sha256(sel.default_params(seed=20261005, k_grid=tuple(plan_doc["k_grid"])))
    candidates = []
    units = []
    table_digests = {"feature_filter_rankings": [], "feature_filter_subsets": []}
    counts = {t: 0 for t in table_digests}
    admissible_n = None
    for target_id in plan_doc["targets"]:
        uid = _phase3_unit_id(plan_doc, target_id, params_sha)
        path = p3 / "terminals" / f"{uid}.json.gz"
        if not path.exists():
            raise CampaignError(f"phase 3 cannot close: target {target_id} has no terminal")
        if not (p3 / "receipts" / f"{uid}.json").exists():
            raise CampaignError(f"phase 3 cannot close: target {target_id} has no warehouse receipt")
        doc = load_terminal(path)
        if _digest({k: v for k, v in doc.items() if k != "terminal_sha256"}) != doc["terminal_sha256"]:
            raise CampaignError(f"phase 3 terminal for {target_id} is corrupt")
        admissible_n = doc["admissible_count"]
        expected_methods = set(sel.METHOD_NAMES)
        if set(doc["methods"]) != expected_methods:
            raise CampaignError(f"phase 3 terminal for {target_id} lacks methods {sorted(expected_methods - set(doc['methods']))}")
        for r in doc["rows"]["feature_filter_subsets"]:
            candidates.append({"population_id": plan_doc["population_id"], "identity": plan_doc["identity"], "target_id": target_id,
                               "horizon_hours": r["horizon_hours"], "method": r["method"], "k": r["k"], "members": r["members"],
                               "label": r["label"], "is_final_selection": False, "subset_sha256": r["subset_sha256"]})
        for table, rows in doc["rows"].items():
            table_digests[table].extend(hashlib.sha256(man.canonical_bytes(r)).hexdigest() for r in rows)
            counts[table] += len(rows)
        units.append({"target_id": target_id, "unit_id": uid, "wall_seconds": doc.get("wall_seconds")})
    wh = warehouse or open_warehouse(warehouse_path)
    recon = _reconcile(wh, plan_doc["identity"], dict(counts))
    for table, shas in table_digests.items():
        expected = hashlib.sha256("".join(sorted(shas)).encode()).hexdigest()
        got = recon["tables"].get(table, {})
        if got.get("count") != counts[table] or got.get("rows_sha256") != expected:
            raise CampaignError(f"phase 3 warehouse readback does not reconcile for {table}")
    closure = {
        "schema": PHASE3_SCHEMA, "state": "PHASE_3_FILTER_COMPLETE", "generated_from_evidence": True,
        "population_id": plan_doc["population_id"], "identity": plan_doc["identity"], "plan_sha256": plan_doc["plan_sha256"],
        "phase2_closure_sha256": phase2["closure_sha256"], "phase3_params_sha256": params_sha,
        "targets": len(units), "methods": list(sel.METHOD_NAMES), "k_grid": list(plan_doc["k_grid"]),
        "admissible_features": admissible_n, "row_counts": counts,
        "table_sha256": {t: hashlib.sha256("".join(sorted(s)).encode()).hexdigest() for t, s in table_digests.items()},
        "warehouse_reconcile": {t: recon["tables"][t] for t in table_digests}, "units": units,
        "predictive_winner": None, "uses_test_split": False, "uses_validation_split": False, "final_selection": False,
        "closed_at": _now(), "closed_at_utc": _ts(),
    }
    closure["closure_sha256"] = _digest({k: v for k, v in closure.items() if k != "closure_sha256"})
    cand = {"schema": "fs_phase23.candidates_for_validation.v1", "population_id": plan_doc["population_id"],
            "identity": plan_doc["identity"], "phase3_closure_sha256": closure["closure_sha256"], "k_grid": list(plan_doc["k_grid"]),
            "admissible_features": {plan_doc["population_id"]: admissible_n}, "predictive_winner": None, "uses_test_split": False,
            "final_selection": False, "next_step": "wrapper validation under BUSINESS_WEEKLY_WALK_FORWARD (not part of phase 3)",
            "candidates": candidates}
    _write_json_atomic(state_root / "PHASE_3_FILTER_COMPLETE.json", closure)
    _write_json_atomic(state_root / "CANDIDATES_FOR_VALIDATION.json", cand)
    return closure


# ----------------------------------------------------------------------------- CLI

def _host_spec(value: str) -> dict:
    host_id, _, size = value.partition(":")
    if not host_id:
        raise argparse.ArgumentTypeError("host spec must be ID or ID:small|large")
    return {"host_id": host_id, "size_class": size or "large"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--manifest", required=True, type=Path)
    s.add_argument("--state-root", required=True, type=Path)
    s.add_argument("--n-shards", required=True, type=int)
    s.add_argument("--host", action="append", required=True, type=_host_spec, dest="hosts")
    s.add_argument("--small-fraction", type=float, default=1.0 / 3.0)
    s.add_argument("--param", action="append", default=[], help="override KEY=JSON")
    s = sub.add_parser("run-worker")
    s.add_argument("--plan", required=True, type=Path)
    s.add_argument("--state-root", required=True, type=Path)
    s.add_argument("--data-root", required=True, type=Path)
    s.add_argument("--host-id", required=True)
    s.add_argument("--max-shards", type=int, default=None)
    s.add_argument("--steal", action="store_true")
    s.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("follow")
    s.add_argument("--plan", required=True, type=Path)
    s.add_argument("--state-root", required=True, type=Path)
    s.add_argument("--terminals", action="append", required=True, type=Path)
    s.add_argument("--warehouse", required=True)
    s.add_argument("--data-root", required=True, type=Path)
    s.add_argument("--every", type=float, default=60.0)
    s.add_argument("--once", action="store_true")
    s.add_argument("--no-chain", action="store_true")
    s.add_argument("--phase3-workers", type=int, default=1)
    for verb in ("close-phase2", "run-phase3", "close-phase3"):
        s = sub.add_parser(verb)
        s.add_argument("--plan", required=True, type=Path)
        s.add_argument("--state-root", required=True, type=Path)
        s.add_argument("--warehouse", required=True)
        if verb != "close-phase3":
            s.add_argument("--data-root", required=True, type=Path)
        if verb == "run-phase3":
            s.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("status")
    s.add_argument("--plan", action="append", required=True, type=Path)
    s.add_argument("--terminals", action="append", default=[], type=Path)
    s.add_argument("--state-root", action="append", default=[], type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--every", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.verb == "plan":
            params = {}
            for item in args.param:
                k, _, v = item.partition("=")
                params[k] = json.loads(v)
            doc = plan(manifest_path=args.manifest, state_root=args.state_root, n_shards=args.n_shards, hosts=args.hosts,
                       params=params, small_fraction=args.small_fraction)
            print(json.dumps({"plan_sha256": doc["plan_sha256"], "shards": doc["n_shards"], "expected_pairs": doc["expected_pairs"],
                              "by_host": {h["host_id"]: sum(1 for s in doc["shards"] if s["host_id"] == h["host_id"]) for h in doc["hosts"]}}))
        elif args.verb == "run-worker":
            print(json.dumps(run_worker(plan_path=args.plan, state_root=args.state_root, data_root=args.data_root, host_id=args.host_id,
                                        max_shards=args.max_shards, steal=args.steal, threads=args.threads)))
        elif args.verb == "follow":
            follow(every=args.every, once=args.once, plan_path=args.plan, state_root=args.state_root, terminal_dirs=args.terminals,
                   warehouse_path=args.warehouse, data_root=args.data_root, chain_phase3=not args.no_chain, phase3_workers=args.phase3_workers)
        elif args.verb == "close-phase2":
            print(json.dumps({"closure_sha256": close_phase2(plan_path=args.plan, state_root=args.state_root, warehouse_path=args.warehouse,
                                                             data_root=args.data_root)["closure_sha256"]}))
        elif args.verb == "run-phase3":
            print(json.dumps(run_phase3(plan_path=args.plan, state_root=args.state_root, data_root=args.data_root, warehouse_path=args.warehouse,
                                        workers=args.workers)))
        elif args.verb == "close-phase3":
            print(json.dumps({"closure_sha256": close_phase3(plan_path=args.plan, state_root=args.state_root, warehouse_path=args.warehouse)["closure_sha256"]}))
        elif args.verb == "status":
            from tools import feature_selection_phase23_status as st
            while True:
                st.build_status(plan_paths=args.plan, terminal_dirs=args.terminals, state_roots=args.state_root, out_path=args.out)
                if args.every is None:
                    break
                time.sleep(args.every)
        return 0
    except CampaignError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
