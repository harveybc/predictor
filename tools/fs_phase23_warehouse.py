#!/usr/bin/env python3
"""Phase-2/3 feature-selection warehouse: contract, additive migration, submit, readback, snapshot.

Subplan: ``docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md``
(§2 identities, §3.3 closure, §6 acceptance) and the order
``docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md`` (§B, §D, §F, §G).

The storage semantics live in ONE place, ``predictor_olap_store.fs_phase23_store`` (the
packaged backend the warehouse service loads); this module imports them, so the local file
path used by tests and the follower, and the service path behind ``/api/v2/fs-phase23/*``,
are the same code. What this module adds:

* ``open_warehouse(path_or_url)`` — the driver's entry point
  (``tools/feature_pairwise_campaign.py``): a DuckDB file (migration applied on open) or an
  ``http(s)://`` service URL (token from ``WAREHOUSE_TOKEN``, never an argument). Both return
  an object with ``submit_rows(run_id, table, rows)``, ``read_run(run_id, table, unit_id=None)``
  and ``reconcile(run_id)``.
* ``contract`` — derives ``CONTRACT.json`` from the phase-1 closure artifacts and asserts the
  pair denominators n(n-1)/2 = 66,795 (EURUSD) and 3,403 (ETH).
* ``migrate`` (with ``--dry-run``), ``readback`` (counts/digests per asset x method x host
  role x fold, same SQL locally and through the service), ``reconcile``, ``compare-readback``.
* ``snapshot`` / ``verify-snapshot`` — ``.duckdb.zst`` + SHA-256 + release-asset manifest;
  refuses a live store (non-empty write-ahead log beside the file).

This module never opens the configured production cube. Host references are roles only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_SQL = REPO_ROOT / "olap" / "migrations" / "fs_phase23" / "0001_fs_phase23_additive.sql"

try:
    from predictor_olap_store import fs_phase23_store as core
except ImportError:  # a checkout without the package installed: the source tree is the package
    sys.path.insert(0, str(REPO_ROOT / "olap" / "store" / "src"))
    from predictor_olap_store import fs_phase23_store as core

Refusal = core.Refusal
FACT_TABLES = core.FACT_TABLES
ALL_TABLES = core.ALL_TABLES
ALL_RELATIONS = core.ALL_RELATIONS
MIGRATION_ID = core.MIGRATION_ID
STATES = core.STATES
canonical_bytes = core.canonical_bytes
digest = core.digest
rows_digest = core.rows_digest
rows_sha256 = core.rows_sha256
pair_denominator = core.pair_denominator
order_pair = core.order_pair
prepare_row = core.prepare_row

SCHEMA_CONTRACT = "fs_phase23_contract.v1"
SCHEMA_READBACK = "fs_phase23_readback_report.v1"
SCHEMA_SNAPSHOT = "olap_snapshot_manifest.v2"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return core.now()


# --------------------------------------------------------------------------- migration file
def load_ddl(path: Path = MIGRATION_SQL) -> list[str]:
    """Statements of the migration file, comments stripped. Must equal ``core.ddl()``."""
    text = Path(path).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def write_migration_file(path: Path = MIGRATION_SQL) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(core.render_sql(), encoding="utf-8")
    return file_sha256(path)


def migration_file_matches_code(path: Path = MIGRATION_SQL) -> bool:
    norm = lambda s: " ".join(s.split())  # noqa: E731
    return [norm(s) for s in load_ddl(path)] == [norm(s) for s in core.ddl()]


def apply_migration(conn, path: Path = MIGRATION_SQL) -> dict:
    """Apply the additive migration from the code (the file is checked to match it)."""
    if path.exists() and not migration_file_matches_code(path):
        raise Refusal(f"{path.name} does not match fs_phase23_store.ddl(); regenerate it with "
                      "`fs_phase23_warehouse.py render-migration`")
    result = core.apply_migration(conn)
    result["sql_sha256"] = file_sha256(path) if path.exists() else None
    return result


def migration_plan(conn) -> dict:
    present = set(core.list_relations(conn))
    return {"migration_id": MIGRATION_ID,
            "sql_sha256": file_sha256(MIGRATION_SQL) if MIGRATION_SQL.exists() else None,
            "would_create": [t for t in ALL_RELATIONS if t not in present],
            "already_present": [t for t in ALL_RELATIONS if t in present],
            "statements": len(core.ddl()), "destructive_statements": 0}


# --------------------------------------------------------------------------- warehouses
class Warehouse:
    """A DuckDB file. The driver's interface plus the explicit binding and readback helpers."""

    backend = "duckdb"

    def __init__(self, path: Path, *, read_only: bool = False, memory_limit: str = "256MB"):
        import duckdb
        self.path = Path(path).expanduser()
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path), read_only=read_only,
                                   config={"memory_limit": memory_limit, "threads": 1})
        if not read_only:
            self.migration = apply_migration(self.conn)

    def register_run(self, run: dict) -> dict:
        return core.register_run(self.conn, run)

    def submit_rows(self, run_id: str, table: str, rows: list[dict], *, host_role: str | None = None,
                    shard_id: str | None = None) -> dict:
        return core.submit_rows(self.conn, run_id, table, rows, host_role=host_role, shard_id=shard_id,
                                backend=self.backend)

    def read_run(self, run_id: str, table: str, unit_id: str | None = None) -> list[dict]:
        return core.read_run(self.conn, run_id, table, unit_id)

    def readback_summary(self, run_id: str, table: str, unit_id: str | None = None) -> dict:
        if table not in core.FACT_TABLES:
            raise Refusal(f"unknown table {table!r}")
        where = "run_id = ? AND unit_id = ?" if unit_id is not None else "run_id = ? AND unit_id IS NULL"
        params = [run_id, unit_id] if unit_id is not None else [run_id]
        n, d = self.conn.execute(f"SELECT count(*), sha256(string_agg(row_sha256, '' ORDER BY row_sha256)) FROM {table} WHERE {where}", params).fetchone()
        return {"count": int(n), "rows_sha256": (d if n else rows_digest([]))}

    def reconcile(self, run_id: str, receipts: list[dict] | None = None, expected: dict | None = None) -> dict:
        return core.reconcile(self.conn, run_id, receipts, expected, backend=self.backend)

    def verify_receipt(self, receipt: dict) -> dict:
        return core.verify_receipt(self.conn, receipt)

    def query(self, sql: str) -> list[dict]:
        cursor = self.conn.execute(sql)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        self.conn.close()


class ServiceWarehouse:
    """The running warehouse through its host routes. The token comes from the environment."""

    backend = "service"

    # Incident 2026-10-06: the host was OOM-killed on large submissions and the client had a single
    # 600 s socket timeout and no retry, so one dead host meant one lost pass per unit.  Every request
    # now has a bounded timeout and transport failures (connection refused/reset, 5xx, timeouts) are
    # retried with backoff inside a declared budget, after the host answers /healthz again.  A 400/422
    # refusal is a verdict about the bytes and is never retried.
    def __init__(self, url: str, token_env: str = "WAREHOUSE_TOKEN", *, page: int = 5000, timeout: float = 300.0,
                 retries: int = 6, backoff: float = 5.0, max_backoff: float = 60.0):
        self.url = url.rstrip("/")
        self.token = os.environ.get(token_env)
        if not self.token:
            raise Refusal(f"{token_env} is not set: the service token comes from the environment, "
                          "never from an argument or a file in the repository")
        self.page = page
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.max_backoff = float(max_backoff)
        self.transport_log: list[dict] = []

    def healthy(self, timeout: float = 10.0) -> bool:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{self.url}/healthz"), timeout=timeout) as handle:
                return 200 <= handle.status < 300
        except Exception:  # noqa: BLE001
            return False

    def _once(self, method: str, path: str, body: dict | None, params: dict | None) -> dict:
        url = f"{self.url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        data = None if body is None else canonical_bytes(body)
        request = urllib.request.Request(url, data=data, method=method,
                                         headers={"Authorization": f"Bearer {self.token}",
                                                  "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as handle:
            return json.load(handle)

    def _call(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        import socket
        import time
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._once(method, path, body, params)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in (400, 422):
                    raise Refusal(f"service refused ({exc.code}): {detail}") from None
                if exc.code in (401, 403, 404, 405, 413, 414):
                    raise
                failure = f"http {exc.code}: {detail[:120]}"
            except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError, OSError) as exc:
                failure = f"{type(exc).__name__}: {str(exc)[:160]}"
            except Exception as exc:  # noqa: BLE001 - http.client.RemoteDisconnected and friends
                if exc.__class__.__module__.startswith("http"):
                    failure = f"{type(exc).__name__}: {str(exc)[:160]}"
                else:
                    raise
            self.transport_log.append({"method": method, "path": path, "attempt": attempt, "failure": failure})
            if attempt > self.retries:
                raise ConnectionError(f"{method} {path} failed after {attempt} attempts; last: {failure}")
            wait = min(self.backoff * (2 ** (attempt - 1)), self.max_backoff)
            time.sleep(wait)
            deadline = time.time() + self.max_backoff * 2
            while not self.healthy() and time.time() < deadline:
                time.sleep(self.backoff)

    def readback_summary(self, run_id: str, table: str, unit_id: str | None = None) -> dict:
        """Stored (count, rows_sha256) for one unit, computed inside the database: the same digest rule
        as receipts and reconcile (sha256 of the sorted concatenation of row_sha256), without paging
        the payloads back through the host."""
        q = lambda v: v.replace("'", "''")
        where = f"run_id = '{q(run_id)}'" + (f" AND unit_id = '{q(unit_id)}'" if unit_id is not None else " AND unit_id IS NULL")
        rows = self.query(f"SELECT count(*) AS n, sha256(string_agg(row_sha256, '' ORDER BY row_sha256)) AS d "
                          f"FROM {table} WHERE {where} LIMIT 1")
        n = int(rows[0]["n"]) if rows else 0
        return {"count": n, "rows_sha256": (rows[0]["d"] if n else rows_digest([]))}

    def submit_rows(self, run_id: str, table: str, rows: list[dict], *, host_role: str | None = None,
                    shard_id: str | None = None) -> dict:
        document = {"run_id": run_id, "table": table, "rows": rows}
        if host_role:
            document["host_role"] = host_role
        if shard_id:
            document["shard_id"] = shard_id
        return self._call("POST", "/api/v2/fs-phase23/rows", document)

    def read_run(self, run_id: str, table: str, unit_id: str | None = None) -> list[dict]:
        out: list[dict] = []
        after = None
        while True:
            page = self._call("GET", "/api/v2/fs-phase23/rows",
                              params={"run_id": run_id, "table": table, "unit_id": unit_id,
                                      "after": after, "limit": self.page})
            out.extend(page.get("rows") or [])
            after = page.get("next_after")
            if not after:
                return out

    def reconcile(self, run_id: str, receipts: list[dict] | None = None, expected: dict | None = None) -> dict:
        document: dict[str, Any] = {"run_id": run_id}
        if receipts:
            document["receipts"] = receipts
        if expected:
            document["expected"] = expected
        return self._call("POST", "/api/v2/fs-phase23/reconcile", document)

    def query(self, sql: str) -> list[dict]:
        body = self._call("GET", "/api/v1/query", params={"sql": sql})
        return body.get("rows") or []

    def close(self) -> None:
        return None


def open_warehouse(path_or_url, *, token_env: str = "WAREHOUSE_TOKEN"):
    """The driver's entry point: a DuckDB file path or an http(s) service URL."""
    target = str(path_or_url)
    if target.startswith(("http://", "https://")):
        return ServiceWarehouse(target, token_env)
    return Warehouse(Path(target))


# --------------------------------------------------------------------------- readback report
def _readback_sql(table: str, run_id: str | None) -> str:
    where = f"WHERE run_id = '{run_id.replace(chr(39), chr(39) * 2)}'" if run_id else ""
    has_state = "state" in core.VALUES[table]
    state_cols = (" sum(CASE WHEN state = 'MEASURED' THEN 1 ELSE 0 END) AS measured,"
                  " sum(CASE WHEN state = 'INSUFFICIENT_SUPPORT' THEN 1 ELSE 0 END) AS insufficient,"
                  " sum(CASE WHEN state = 'FAILED' THEN 1 ELSE 0 END) AS failed"
                  if has_state else " NULL AS measured, NULL AS insufficient, NULL AS failed")
    method = "method" if "method" in core.IDENTITY[table] else "'-'"
    fold = "fold" if "fold" in core.IDENTITY[table] else "'-'"
    return (f"SELECT population_id, {method} AS method, coalesce(host_role, 'UNDECLARED') AS host_role,"
            f" {fold} AS fold, count(*) AS n,"
            f" sha256(string_agg(row_sha256, '' ORDER BY row_sha256)) AS rows_sha256,{state_cols}"
            f" FROM {table} {where} GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4 LIMIT 100000")


def readback_report(query, run_id: str | None = None, *, source: str = "local") -> dict:
    """Counts and digests per asset x method x host role x fold, per table (closure input)."""
    tables = {}
    for table in FACT_TABLES:
        groups = []
        for row in query(_readback_sql(table, run_id)):
            groups.append({"population_id": row["population_id"], "method": row["method"],
                           "host_role": row["host_role"], "fold": row["fold"], "n": int(row["n"]),
                           "rows_sha256": row["rows_sha256"],
                           "measured": None if row["measured"] is None else int(row["measured"]),
                           "insufficient_support": None if row["insufficient"] is None else int(row["insufficient"]),
                           "failed": None if row["failed"] is None else int(row["failed"])})
        by_asset: dict[str, int] = {}
        by_method: dict[str, int] = {}
        by_host: dict[str, int] = {}
        for g in groups:
            by_asset[g["population_id"]] = by_asset.get(g["population_id"], 0) + g["n"]
            by_method[g["method"]] = by_method.get(g["method"], 0) + g["n"]
            by_host[g["host_role"]] = by_host.get(g["host_role"], 0) + g["n"]
        tables[table] = {"groups": groups, "total": sum(g["n"] for g in groups),
                         "table_rows_sha256": rows_digest(g["rows_sha256"] for g in groups),
                         "by_asset": by_asset, "by_method": by_method, "by_host_role": by_host}
    report = {"schema": SCHEMA_READBACK, "source": source, "run_id": run_id, "tables": tables,
              "generated_at": now()}
    report["report_sha256"] = digest({k: v for k, v in report.items() if k != "generated_at"})
    return report


def compare_readback(local: dict, remote: dict) -> dict:
    differences = []
    for table in FACT_TABLES:
        left = {(g["population_id"], g["method"], g["host_role"], g["fold"]): g
                for g in local["tables"].get(table, {}).get("groups", [])}
        right = {(g["population_id"], g["method"], g["host_role"], g["fold"]): g
                 for g in remote["tables"].get(table, {}).get("groups", [])}
        for key in sorted(set(left) | set(right)):
            a, b = left.get(key), right.get(key)
            if a is None or b is None or a["n"] != b["n"] or a["rows_sha256"] != b["rows_sha256"]:
                differences.append({"table": table, "group": list(key),
                                    "local": None if a is None else {"n": a["n"], "rows_sha256": a["rows_sha256"]},
                                    "remote": None if b is None else {"n": b["n"], "rows_sha256": b["rows_sha256"]}})
    return {"agree": not differences, "differences": differences}


# --------------------------------------------------------------------------- the contract
def _targets_from_envelope(rows: dict) -> list[dict]:
    seen = {}
    for family in ("selection_decisions", "information_metrics", "pair_relations"):
        for row in rows.get(family, []):
            if row.get("target_id") is not None:
                seen[row["target_id"]] = int(row["horizon"])
    return [{"target_id": t, "horizon": h} for t, h in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))]


def _causal_supported(rows: dict) -> list[dict]:
    out = []
    for row in rows.get("selection_decisions", []):
        if row.get("decision") == "SELECTED":
            out.append({"feature_id": row["feature_id"], "target_id": row["target_id"],
                        "horizon": int(row["horizon"]), "method": row.get("method"),
                        "phase1_evidence_sha256": row.get("evidence_sha256")})
    return sorted(out, key=lambda r: (r["feature_id"], r["horizon"], r["target_id"]))


def build_population(population_id: str, run_id: str, plan_path: Path, envelope_path: Path,
                     folds_path: Path, features_parquet: Path, targets_parquet: Path,
                     completion_path: Path, *, expected_features: int, expected_targets: int,
                     expected_pairs: int, train_period: list, fold_names: list[str] | None,
                     causal_note: str | None = None) -> dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    envelope = json.loads(Path(envelope_path).read_text(encoding="utf-8"))
    if "envelope" in envelope and "rows" not in envelope:
        envelope = envelope["envelope"]
    folds_doc = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    completion = json.loads(Path(completion_path).read_text(encoding="utf-8"))
    run = envelope["run"]
    if run["run_id"] != run_id:
        raise Refusal(f"{population_id}: envelope run_id {run['run_id']!r} is not {run_id!r}")
    if plan["population_id"] != population_id:
        raise Refusal(f"plan population {plan['population_id']!r} is not {population_id!r}")
    features = sorted(item["feature_id"] for item in plan["items"])
    if len(features) != expected_features or len(set(features)) != expected_features:
        raise Refusal(f"{population_id}: plan holds {len(features)} features, expected {expected_features}")
    targets = _targets_from_envelope(envelope["rows"])
    if len(targets) != expected_targets:
        raise Refusal(f"{population_id}: envelope names {len(targets)} targets, expected {expected_targets}")
    pairs = pair_denominator(len(features))
    if pairs != expected_pairs:
        raise Refusal(f"{population_id}: n(n-1)/2 = {pairs}, declared {expected_pairs}")
    folds = []
    for index, fold in enumerate(folds_doc["folds"]):
        name = fold.get("name") or (fold_names[index] if fold_names else f"inner_{index + 1}")
        folds.append({"name": name, "train_rows": list(fold["train_rows"]),
                      "validation_rows": list(fold.get("validation_rows") or fold.get("val_rows")),
                      "train_time": fold.get("train_time"), "validation_time": fold.get("val_time"),
                      "label_purge_rows": fold.get("label_purge_h")})
    row_population_ids = sorted({r.get("population_id") for fam in envelope["rows"].values()
                                 for r in fam if r.get("population_id")})
    causal = _causal_supported(envelope["rows"])
    distinct_causal = sorted({c["feature_id"] for c in causal})
    return {
        "population_id": population_id,
        "phase1_run_id": run_id,
        "phase1_identity": {
            "run_id": run_id,
            "campaign_sha256": run["campaign_sha256"], "code_sha256": run["code_sha256"],
            "input_sha256": run["input_sha256"], "inventory_sha256": run["inventory_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "final_envelope_sha256": envelope.get("envelope_sha256"),
            "row_population_id_values": row_population_ids,
            "phase1_completion": completion,
        },
        "feature_count": len(features), "features": features,
        "target_count": len(targets), "targets": targets,
        "target_pack": plan.get("target_pack"),
        "pair_denominator": pairs, "pair_denominator_rule": "n*(n-1)/2 over the frozen feature list",
        "train_period": train_period,
        "fold_count": len(folds), "folds": folds,
        "fold_source_schema": folds_doc.get("schema"),
        "sources": [
            {"role": "phase1_plan", "sha256": file_sha256(plan_path)},
            {"role": "phase1_final_envelope", "sha256": file_sha256(envelope_path)},
            {"role": "phase1_folds", "sha256": file_sha256(folds_path)},
            {"role": "train_features_parquet", "sha256": file_sha256(features_parquet)},
            {"role": "train_targets_parquet", "sha256": file_sha256(targets_parquet)},
            {"role": "phase1_completion", "sha256": file_sha256(completion_path)},
        ],
        "causal_supported": {
            "label": "CAUSAL_SUPPORTED",
            "meaning": "phase-1 causal-ladder support for a feature-target pair; a control variant "
                       "of the phase-3 ranking, NOT a final selection and NOT a predictive claim",
            "pair_count": len(causal), "distinct_feature_count": len(distinct_causal),
            "distinct_features": distinct_causal, "pairs": causal,
            "note": causal_note,
        },
    }


def build_contract(args) -> dict:
    eurusd = build_population(
        "EURUSD", "phase1-eurusd-final:94d20c038d55e152", args.eurusd_plan, args.eurusd_envelope,
        args.eurusd_folds, args.eurusd_features, args.eurusd_targets, args.eurusd_completion,
        expected_features=366, expected_targets=14, expected_pairs=66_795,
        train_period=["2012-05-01T00:00:00Z", "2024-01-01T00:00:00Z"], fold_names=None)
    eth = build_population(
        "ETH", "phase1-final:ETH:29d2f745f5d9e87c", args.eth_plan, args.eth_envelope,
        args.eth_folds, args.eth_features, args.eth_targets, args.eth_completion,
        expected_features=83, expected_targets=6, expected_pairs=3_403,
        train_period=[None, "2024-01-01T00:00:00Z"],
        fold_names=["inner_1", "inner_2", "inner_3"],
        causal_note="ETH phase-1 is PROVISIONAL_DEVELOPMENT under its point-in-time caveat; "
                    "no point-in-time admissibility, production selection or EURUSD comparability "
                    "claim is carried by this contract")
    if eurusd["causal_supported"]["distinct_feature_count"] != 12 or eurusd["causal_supported"]["pair_count"] != 34:
        raise Refusal("EURUSD causal support is not the 34-pair / 12-feature phase-1 record")
    if eth["causal_supported"]["distinct_feature_count"] != 3:
        raise Refusal("ETH causal support is not the three-feature phase-1 record")
    contract = {
        "schema": SCHEMA_CONTRACT,
        "frozen_on": args.frozen_on,
        "authority": [
            "docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_PHASE2_PHASE3_WORK_PLAN_2026_10_05.md",
            "docs/handoffs/MUSASHI_TO_SATOSHI_FS_PHASE2_PHASE3_AUTOMATED_2026_10_05.md",
        ],
        "populations": {"EURUSD": eurusd, "ETH": eth},
        "rules": {
            "pair_identity": "left < right (lexicographic); a pair is stored once; lagged relations "
                             "carry lag_hours >= 0 under the same ordered identity",
            "alias_evidence": "an alias group contributes ONE independent observation; aliases never "
                              "count as independent evidence in any pair denominator, cluster, ranking "
                              "or stability statistic; the disposition and representative are stored, "
                              "no column is deleted",
            "splits": "only TRAIN rows exist (fold_id = TRAIN or an inner fold); validation and test "
                      "remain closed; fitting, discretisation, scaling and estimators resolve inside "
                      "each TRAIN fold",
            "causal_supported": "the 12 EURUSD and 3 ETH phase-1 features are CAUSAL_SUPPORTED candidates "
                                "feeding a declared ranking variant and the CAUSAL_SUPPORTED control; "
                                "they are not the final selection; NOT_IDENTIFIED is zero causal "
                                "evidence, not rejection",
            "row_identity": "UNIQUE (run_id, row_key) AND UNIQUE over the typed identity columns per "
                            "table (predictor_olap_store.fs_phase23_store.IDENTITY); "
                            "row_sha256 = sha256(canonical JSON of the submitted row); "
                            "rows_sha256 = sha256(sorted row_sha256 concatenated)",
            "lags_hours": [0, 1, 2, 6, 24, 48, 168],
            "k_path": [4, 8, 12, 16, 24, 32],
            "states": list(STATES),
            "host_references": "roles only: coordinator, worker_a, worker_b",
        },
        "warehouse": {
            "migration_id": MIGRATION_ID,
            "migration_sql": str(MIGRATION_SQL.relative_to(REPO_ROOT)),
            "migration_sql_sha256": file_sha256(MIGRATION_SQL) if MIGRATION_SQL.exists() else None,
            "tables": list(ALL_TABLES), "views": list(core.VIEWS),
            "identity_columns": {t: list(c) for t, c in core.IDENTITY.items()},
        },
    }
    contract["contract_sha256"] = digest({k: v for k, v in contract.items() if k != "contract_sha256"})
    return contract


def verify_contract(contract: dict) -> dict:
    body = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if digest(body) != contract.get("contract_sha256"):
        raise Refusal("contract digest does not match its body")
    out = {}
    for pid, pop in contract["populations"].items():
        n = len(pop["features"])
        if n != pop["feature_count"] or len(set(pop["features"])) != n:
            raise Refusal(f"{pid}: feature list is not {pop['feature_count']} unique names")
        if pair_denominator(n) != pop["pair_denominator"]:
            raise Refusal(f"{pid}: denominator mismatch")
        if len(pop["targets"]) != pop["target_count"]:
            raise Refusal(f"{pid}: target count mismatch")
        if pop["causal_supported"]["label"] != "CAUSAL_SUPPORTED":
            raise Refusal(f"{pid}: causal features must be labelled CAUSAL_SUPPORTED")
        unknown = set(pop["causal_supported"]["distinct_features"]) - set(pop["features"])
        if unknown:
            raise Refusal(f"{pid}: causal features outside the population: {sorted(unknown)}")
        out[pid] = {"features": n, "pairs": pop["pair_denominator"], "targets": len(pop["targets"]),
                    "folds": pop["fold_count"], "causal_supported_features": pop["causal_supported"]["distinct_feature_count"]}
    return out


# --------------------------------------------------------------------------- snapshot
def _zstd_compress(source: Path, target: Path, zstd_bin: str | None, level: int) -> str:
    try:
        import zstandard  # type: ignore
    except ImportError:
        zstandard = None
    if zstandard is not None:
        compressor = zstandard.ZstdCompressor(level=level)
        with source.open("rb") as src, target.open("wb") as dst:
            compressor.copy_stream(src, dst)
        return f"python-zstandard {zstandard.__version__}"
    binary = zstd_bin or shutil.which("zstd")
    if not binary:
        raise Refusal("neither the zstandard module nor a zstd binary is available")
    subprocess.run([binary, "-q", f"-{level}", "-T1", "--force", "-o", str(target), str(source)], check=True)
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False).stdout.strip()
    return version or "zstd"


def snapshot(source: Path, out_dir: Path, *, tag: str, repo: str, phase: str, zstd_bin: str | None,
             level: int = 19, warehouse_runs: dict | None = None,
             expected_tables: Iterable[str] = ALL_TABLES) -> dict:
    """Compress a cube COPY to .duckdb.zst, digest both forms, write the release-asset manifest."""
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise Refusal(f"snapshot source does not exist: {source.name}")
    wal = source.with_name(source.name + ".wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise Refusal("the source has a non-empty write-ahead log beside it: that is a live store, "
                      "not a snapshot copy. Take the copy with tools/olap_duckdb_migrate.py snapshot "
                      "(it CHECKPOINTs the copy) and point this command at the copy")
    import duckdb
    conn = duckdb.connect(str(source), read_only=True)
    try:
        present = set(core.list_relations(conn))
        counts = {}
        for table in expected_tables:
            if table in present:
                counts[table] = {"rows": int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])}
                if table in FACT_TABLES:
                    n, sha = conn.execute(
                        f"SELECT count(*), coalesce(sha256(string_agg(row_sha256, '' ORDER BY row_sha256)), "
                        f"'{core.EMPTY_DIGEST}') FROM {table}").fetchone()
                    counts[table]["rows_sha256"] = sha
            else:
                counts[table] = {"rows": None, "missing": True}
    finally:
        conn.close()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    asset = out_dir / f"{source.stem}.duckdb.zst"
    tool = _zstd_compress(source, asset, zstd_bin, level)
    manifest = {
        "schema": SCHEMA_SNAPSHOT, "phase": phase, "created_on": datetime.now(timezone.utc).date().isoformat(),
        "database": "DuckDB",
        "snapshot_file": source.name, "snapshot_size_bytes": source.stat().st_size,
        "snapshot_sha256": file_sha256(source),
        "compressed_asset": asset.name, "compressed_size_bytes": asset.stat().st_size,
        "compressed_sha256": file_sha256(asset), "compressor": tool, "zstd_level": level,
        "release_tag": tag, "release_url": f"https://github.com/{repo}/releases/tag/{tag}",
        "asset_url": f"https://github.com/{repo}/releases/download/{tag}/{asset.name}",
        "relations": counts, "warehouse_runs": warehouse_runs or {},
        "missing_relations": [t for t, c in counts.items() if c.get("missing")],
        "custody_policy": "The repository retains this manifest, the schema and the SHA-256; the "
                          "compressed binary is a release asset. Restore requires SHA-256 "
                          "verification of the asset and of the decompressed file before opening.",
        "publish_commands": [
            f"gh release create {tag} --repo {repo} --title '{phase} OLAP snapshot' --notes-file RELEASE_NOTES.md",
            f"gh release upload {tag} {asset.name} {asset.name}.sha256 --repo {repo} --clobber",
        ],
    }
    manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    (out_dir / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    (out_dir / f"{asset.name}.sha256").write_text(f"{manifest['compressed_sha256']}  {asset.name}\n",
                                                  encoding="utf-8")
    return manifest


def verify_snapshot_asset(asset: Path, manifest: dict, *, zstd_bin: str | None, work_dir: Path) -> dict:
    asset = Path(asset)
    if file_sha256(asset) != manifest["compressed_sha256"]:
        raise Refusal("asset SHA-256 does not match the manifest; refusing to open")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    restored = work_dir / manifest["snapshot_file"]
    try:
        import zstandard  # type: ignore
        with asset.open("rb") as src, restored.open("wb") as dst:
            zstandard.ZstdDecompressor().copy_stream(src, dst)
    except ImportError:
        binary = zstd_bin or shutil.which("zstd")
        if not binary:
            raise Refusal("no zstd available to decompress")
        subprocess.run([binary, "-q", "-d", "--force", "-o", str(restored), str(asset)], check=True)
    if file_sha256(restored) != manifest["snapshot_sha256"]:
        raise Refusal("decompressed file SHA-256 does not match the manifest")
    import duckdb
    conn = duckdb.connect(str(restored), read_only=True)
    try:
        observed = {}
        for table, expected in manifest["relations"].items():
            if expected.get("missing"):
                continue
            observed[table] = int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            if observed[table] != expected["rows"]:
                raise Refusal(f"{table}: {observed[table]} rows, manifest says {expected['rows']}")
    finally:
        conn.close()
    return {"verified": True, "restored": restored.name, "relations": observed}


# --------------------------------------------------------------------------- CLI
def _dump(document: dict, out: str | None) -> None:
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8")
    sys.stdout.write(text if not out else f"wrote {out}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="derive CONTRACT.json from phase-1 artifacts")
    for pop in ("eurusd", "eth"):
        for role in ("plan", "envelope", "folds", "features", "targets", "completion"):
            contract.add_argument(f"--{pop}-{role}", required=True, type=Path)
    contract.add_argument("--frozen-on", default=datetime.now(timezone.utc).date().isoformat())
    contract.add_argument("--out", required=True)

    verify = sub.add_parser("verify-contract", help="re-check a contract's arithmetic and labels")
    verify.add_argument("contract", type=Path)

    sub.add_parser("render-migration", help="write the migration .sql from fs_phase23_store.ddl()")

    migrate = sub.add_parser("migrate", help="apply (or plan) the additive migration on a DuckDB file")
    migrate.add_argument("--duckdb", required=True)
    migrate.add_argument("--dry-run", action="store_true", help="report the plan; write nothing")
    migrate.add_argument("--out")

    rb = sub.add_parser("readback", help="counts/digests per asset x method x host role x fold")
    rb.add_argument("--duckdb", help="a local file (opened read-only)")
    rb.add_argument("--service-url", help="the live store's routes (token from the environment)")
    rb.add_argument("--token-env", default="WAREHOUSE_TOKEN")
    rb.add_argument("--run-id")
    rb.add_argument("--out")

    rc = sub.add_parser("reconcile", help="stored counts/digests for one run vs expectations and receipts")
    rc.add_argument("--duckdb")
    rc.add_argument("--service-url")
    rc.add_argument("--token-env", default="WAREHOUSE_TOKEN")
    rc.add_argument("--run-id", required=True)
    rc.add_argument("--receipts", nargs="*", type=Path, default=[])
    rc.add_argument("--out")

    cmp_ = sub.add_parser("compare-readback", help="do two readback reports agree?")
    cmp_.add_argument("local", type=Path)
    cmp_.add_argument("remote", type=Path)

    snap = sub.add_parser("snapshot", help="compress a cube COPY to .duckdb.zst with manifest")
    snap.add_argument("--source", required=True, help="a checkpointed copy, never the live cube")
    snap.add_argument("--out-dir", required=True)
    snap.add_argument("--tag", required=True)
    snap.add_argument("--repo", default="harveybc/predictor")
    snap.add_argument("--phase", required=True)
    snap.add_argument("--zstd-bin")
    snap.add_argument("--level", type=int, default=19)
    snap.add_argument("--publish", action="store_true", help="run the gh commands after writing")

    vs = sub.add_parser("verify-snapshot", help="verify an asset against its manifest")
    vs.add_argument("--asset", required=True, type=Path)
    vs.add_argument("--manifest", required=True, type=Path)
    vs.add_argument("--work-dir", required=True, type=Path)
    vs.add_argument("--zstd-bin")

    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            document = build_contract(args)
            verify_contract(document)
            _dump(document, args.out)
        elif args.command == "verify-contract":
            _dump(verify_contract(json.loads(args.contract.read_text(encoding="utf-8"))), None)
        elif args.command == "render-migration":
            _dump({"migration_sql": str(MIGRATION_SQL.relative_to(REPO_ROOT)),
                   "sql_sha256": write_migration_file()}, None)
        elif args.command == "migrate":
            target = Path(args.duckdb).expanduser()
            if args.dry_run and not target.exists():
                _dump({"migration_id": MIGRATION_ID, "sql_sha256": file_sha256(MIGRATION_SQL),
                       "would_create": list(ALL_RELATIONS), "already_present": [],
                       "statements": len(core.ddl()), "destructive_statements": 0,
                       "note": "target file does not exist; a dry run creates nothing"}, args.out)
            elif args.dry_run:
                wh = Warehouse(target, read_only=True)
                try:
                    _dump(migration_plan(wh.conn), args.out)
                finally:
                    wh.close()
            else:
                wh = Warehouse(target)
                try:
                    _dump(wh.migration, args.out)
                finally:
                    wh.close()
        elif args.command in ("readback", "reconcile"):
            if bool(args.duckdb) == bool(args.service_url):
                raise Refusal("give exactly one of --duckdb or --service-url")
            wh = Warehouse(Path(args.duckdb), read_only=True) if args.duckdb \
                else ServiceWarehouse(args.service_url, args.token_env)
            try:
                if args.command == "readback":
                    _dump(readback_report(wh.query, args.run_id, source=wh.backend), args.out)
                else:
                    receipts = [json.loads(p.read_text(encoding="utf-8")) for p in args.receipts]
                    _dump(wh.reconcile(args.run_id, receipts or None), args.out)
            finally:
                wh.close()
        elif args.command == "compare-readback":
            result = compare_readback(json.loads(args.local.read_text()), json.loads(args.remote.read_text()))
            _dump(result, None)
            return 0 if result["agree"] else 3
        elif args.command == "snapshot":
            manifest = snapshot(Path(args.source), Path(args.out_dir), tag=args.tag, repo=args.repo,
                                phase=args.phase, zstd_bin=args.zstd_bin, level=args.level)
            _dump(manifest, None)
            if args.publish:
                for command in manifest["publish_commands"]:
                    subprocess.run(command, shell=True, check=True, cwd=args.out_dir)
        elif args.command == "verify-snapshot":
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            _dump(verify_snapshot_asset(args.asset, manifest, zstd_bin=args.zstd_bin, work_dir=args.work_dir), None)
    except Refusal as exc:
        sys.stderr.write(f"REFUSED: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
