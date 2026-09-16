"""Cube access for data-gov: SELECT-only `query`, append-only `write_metrics`
on the gov_* tables (docs/04_FLOW_V2.md §3). Never truncates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

_WRITE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|copy|grant|revoke|create|vacuum|call|do)\b",
    re.I,
)
_BANNED = re.compile(
    r"\b(pg_sleep|pg_read_file|pg_ls_dir|lo_import|dblink|file_fdw)\b",
    re.I,
)
_LIMIT = re.compile(r"\blimit\s+(\d+)\s*;?\s*$", re.I)
MAX_LIMIT = 5000

_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_LINEAGE = ("VERIFIED", "UNVERIFIED")
METRIC_FIELDS = ("metric", "value", "split", "horizon", "std_dev", "min_value", "max_value", "unit")
DATASET_FIELDS = ("lake", "resource", "sha256", "role")
DATASET_LINEAGE_FIELDS = (
    "lineage", "reason", "event_id", "source_sha256",
    "range_from", "range_to", "delivery", "time_column",
)

GOV_TABLES = (
    "gov_report", "gov_metric", "gov_dataset", "gov_terminal",
    "gov_terminal_metric", "gov_terminal_dataset", "gov_terminal_artifact",
    "gov_availability_contract",
)
GOV_VIEW = "gov_metric_current"
#: S2: a delivery carries a contract DIGEST; without somewhere to resolve it, the cube cannot
#: say that a delivery was a retrospective archive without asking the producer, and a stopped
#: producer makes that unanswerable. This view resolves it, or says UNRESOLVED.
GOV_AVAILABILITY_VIEW = "gov_delivery_availability"
#: The exact canonicalization the digest is taken over, recorded per row so a future change of
#: canonicalization cannot be mistaken for a corrupted contract.
CANONICALIZATION = "json.sort_keys.separators-comma-colon.ascii.v1"
DIGEST_ALGORITHM = "sha256"
#: What this store can VERIFY. A contract written under anything else may be perfectly valid
#: elsewhere; it is simply not something this reader is able to check, and it says so rather
#: than displaying semantics it cannot stand behind.
SUPPORTED_CANONICALIZATIONS = (CANONICALIZATION,)
SUPPORTED_DIGEST_ALGORITHMS = (DIGEST_ALGORITHM,)
#: The availability classes this store can interpret, matching the lake provider's own set.
SUPPORTED_USE_CLASSES = ("OFFLINE_DAY_GRANULAR", "LIVE_EQUIVALENT", "ARCHIVE_RETROSPECTIVE")

#: Outcomes of resolving a delivery to its contract. Exactly one of them carries availability
#: semantics; every other one is a refusal to make a claim, and they are distinct so that
#: "nobody stored it" is never confused with "what is stored does not verify".
RESOLUTION_VERIFIED = "VERIFIED"
RESOLUTION_NO_DELIVERY = "NO_SUCH_DELIVERY"
RESOLUTION_ABSENT = "UNRESOLVED_REFERENCE_ABSENT"
RESOLUTION_AMBIGUOUS = "UNRESOLVED_AMBIGUOUS_DELIVERY"
RESOLUTION_DIGEST_MISMATCH = "UNRESOLVED_DIGEST_MISMATCH"
RESOLUTION_UNSUPPORTED = "UNRESOLVED_UNSUPPORTED_FORMAT"
RESOLUTION_MALFORMED = "UNRESOLVED_MALFORMED_CONTRACT"
RESOLUTION_DISAGREEMENT = "UNRESOLVED_STORED_SEMANTICS_DISAGREE"

TERMINAL_STATES = {"COMPLETED", "FAILED", "INCONCLUSIVE", "REFUSED", "QUARANTINED"}
TERMINAL_KEYS = {
    "schema", "campaign_sha256", "campaign_key", "classification", "project",
    "actor", "unit_id", "generation", "status", "reason", "started_at",
    "finished_at", "costs", "deliveries", "artifacts", "metrics", "tags",
    "terminal_lake", "config_sha256", "code_identity", "synthetic_spec_sha256",
    "terminal_sha256", "verified_datasets",
}
TERMINAL_DATASET_KEYS = {
    "delivery_id", "lake_id", "resource_id", "role", "sha256", "bytes",
    "source_sha256", "range_from", "range_to", "delivery_kind", "time_column",
    "availability_contract_sha256", "state",
}


def _opt_str(obj, name):
    value = obj.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _req_str(obj, name):
    value = obj.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    return value


def _opt_float(obj, name):
    value = obj.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or null")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite value")
    return value


def _opt_int(obj, name):
    value = obj.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an integer or null")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer or null")
    return int(value)


def normalise_report(report) -> dict:
    """Validate a metrics report (§3) and normalise it: numbers coerced,
    duplicate datasets collapsed, metrics and datasets sorted. ValueError on
    anything invalid (the HTTP layer maps it to 400)."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    key = _req_str(report, "experiment_key")
    if not _KEY.match(key):
        raise ValueError("invalid experiment_key")
    set_key = _opt_str(report, "experiment_set_key")
    if set_key is not None and not _KEY.match(set_key):
        raise ValueError("invalid experiment_set_key")
    tags = report.get("tags")
    if tags is None:
        tags = {}
    if not isinstance(tags, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in tags.items()
    ):
        raise ValueError("tags must be a map of strings")

    metrics_in = report.get("metrics")
    if not isinstance(metrics_in, list) or not metrics_in:
        raise ValueError("metrics must be a non-empty list")
    metrics = []
    for item in metrics_in:
        if not isinstance(item, dict):
            raise ValueError("metric entries must be objects")
        metrics.append({
            "metric": _req_str(item, "metric"),
            "value": _opt_float(item, "value"),
            "split": _opt_str(item, "split"),
            "horizon": _opt_int(item, "horizon"),
            "std_dev": _opt_float(item, "std_dev"),
            "min_value": _opt_float(item, "min_value"),
            "max_value": _opt_float(item, "max_value"),
            "unit": _opt_str(item, "unit"),
        })
    metrics.sort(key=lambda m: (
        m["metric"], m["split"] or "", m["horizon"] if m["horizon"] is not None else -1
    ))

    datasets_in = report.get("datasets")
    if datasets_in is None:
        datasets_in = []
    if not isinstance(datasets_in, list):
        raise ValueError("datasets must be a list")
    seen = {}
    for item in datasets_in:
        if not isinstance(item, dict):
            raise ValueError("dataset entries must be objects")
        entry = {
            "lake": _req_str(item, "lake"),
            "resource": _req_str(item, "resource"),
            "sha256": _req_str(item, "sha256"),
            "role": _opt_str(item, "role"),
        }
        if not _HEX64.match(entry["sha256"]):
            raise ValueError("dataset sha256 must be 64 hex characters")
        ident = tuple(entry[k] for k in DATASET_FIELDS)
        if ident in seen:
            continue
        lineage = _opt_str(item, "lineage") or "UNVERIFIED"
        if lineage not in _LINEAGE:
            raise ValueError("dataset lineage must be VERIFIED or UNVERIFIED")
        # data-gov's accounting detail names the range "from"/"to"; the
        # gov_dataset columns are range_from/range_to. Both are accepted.
        entry.update({
            "lineage": lineage,
            "reason": _opt_str(item, "reason"),
            "event_id": _opt_int(item, "event_id"),
            "source_sha256": _opt_str(item, "source_sha256"),
            "range_from": _opt_str(item, "range_from") if "range_from" in item else _opt_str(item, "from"),
            "range_to": _opt_str(item, "range_to") if "range_to" in item else _opt_str(item, "to"),
            "delivery": _opt_str(item, "delivery"),
            "time_column": _opt_str(item, "time_column"),
        })
        seen[ident] = entry
    datasets = sorted(
        seen.values(), key=lambda d: (d["lake"], d["resource"], d["sha256"], d["role"] or "")
    )

    lineage = _opt_str(report, "lineage")
    if lineage is None:
        lineage = "VERIFIED" if datasets and all(
            d["lineage"] == "VERIFIED" for d in datasets
        ) else "UNVERIFIED"
    elif lineage not in _LINEAGE:
        raise ValueError("lineage must be VERIFIED or UNVERIFIED")

    return {
        "experiment_key": key,
        "experiment_set_key": set_key,
        "actor": _req_str(report, "actor"),
        "lake": _req_str(report, "lake"),
        "config_sha256": _opt_str(report, "config_sha256"),
        "code_commit": _opt_str(report, "code_commit"),
        "project": _opt_str(report, "project"),
        "phase": _opt_str(report, "phase"),
        "tags": tags,
        "datasets": datasets,
        "metrics": metrics,
        "lineage": lineage,
        "received_at": _opt_str(report, "received_at"),
    }


def canonical_body(normalised: dict) -> dict:
    """The hashed body of §3 'Report identity': lineage, receipt time and
    the per-dataset lineage fields are left out."""
    return {
        "experiment_key": normalised["experiment_key"],
        "experiment_set_key": normalised["experiment_set_key"],
        "actor": normalised["actor"],
        "lake": normalised["lake"],
        "config_sha256": normalised["config_sha256"],
        "code_commit": normalised["code_commit"],
        "project": normalised["project"],
        "phase": normalised["phase"],
        "tags": normalised["tags"],
        "datasets": [{k: d[k] for k in DATASET_FIELDS} for d in normalised["datasets"]],
        "metrics": [{k: m[k] for k in METRIC_FIELDS} for m in normalised["metrics"]],
    }


def canonical_text(body: dict) -> str:
    return json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def report_sha256(report) -> str:
    """sha256 of the canonical body of a report (raw or normalised)."""
    normalised = normalise_report(report)
    return hashlib.sha256(canonical_text(canonical_body(normalised)).encode()).hexdigest()


def _strict_terminal(terminal):
    if not isinstance(terminal, dict) or set(terminal) != TERMINAL_KEYS:
        raise ValueError("invalid terminal schema")
    if terminal["schema"] != "governed_terminal.v1":
        raise ValueError("invalid terminal schema")
    for name in ("campaign_sha256", "config_sha256", "terminal_sha256"):
        if not isinstance(terminal[name], str) or not _HEX64.fullmatch(terminal[name]):
            raise ValueError(f"invalid {name}")
    for name in ("campaign_key", "project", "actor", "unit_id", "terminal_lake"):
        if not isinstance(terminal[name], str) or not _KEY.fullmatch(terminal[name]):
            raise ValueError(f"invalid {name}")
    if terminal["classification"] not in {"GOVERNING", "NON_GOVERNING"}:
        raise ValueError("invalid classification")
    if terminal["status"] not in TERMINAL_STATES:
        raise ValueError("invalid status")
    if isinstance(terminal["generation"], bool) or not isinstance(terminal["generation"], int):
        raise ValueError("invalid generation")
    if terminal["generation"] < 1:
        raise ValueError("invalid generation")
    for name in ("costs", "tags", "code_identity"):
        if not isinstance(terminal[name], dict):
            raise ValueError(f"invalid {name}")
    for name in ("deliveries", "artifacts", "metrics", "verified_datasets"):
        if not isinstance(terminal[name], list):
            raise ValueError(f"invalid {name}")
    datasets = terminal["verified_datasets"]
    for item in datasets:
        if not isinstance(item, dict) or set(item) != TERMINAL_DATASET_KEYS:
            raise ValueError("invalid verified dataset schema")
        if item["state"] not in {"VERIFIED_TRANSFER", "VERIFIED_CACHE"}:
            raise ValueError("invalid verified dataset state")
        if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or item["bytes"] < 0:
            raise ValueError("invalid verified dataset bytes")
        for name in ("delivery_id", "lake_id", "resource_id", "role", "sha256"):
            if not isinstance(item[name], str) or not item[name]:
                raise ValueError(f"invalid verified dataset {name}")
        if not re.fullmatch(r"[0-9a-f]{32}", item["delivery_id"]):
            raise ValueError("invalid verified dataset delivery_id")
        if not _HEX64.fullmatch(item["sha256"]):
            raise ValueError("invalid verified dataset sha256")
        source = item["source_sha256"]
        if source is not None and (not isinstance(source, str) or not _HEX64.fullmatch(source)):
            raise ValueError("invalid verified dataset source_sha256")
        contract = item["availability_contract_sha256"]
        if not isinstance(contract, str) or not _HEX64.fullmatch(contract):
            raise ValueError("invalid verified dataset availability_contract_sha256")
    body = {
        key: value for key, value in terminal.items()
        if key != "terminal_sha256"
    }
    digest = hashlib.sha256(canonical_text(body).encode("ascii")).hexdigest()
    if digest != terminal["terminal_sha256"]:
        raise ValueError("terminal_sha256 mismatch")
    return terminal


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Plugin:
    plugin_params = {
        "sqlite_path": None,
        "schema": "public",
        "holdout_start": "2025-01-01",
        "time_column": None,
        "lake_id": "olap_cube",
        "title": "predictor OLAP",
        "description": "",
        "kind": "sql_olap",
    }

    def __init__(self):
        self.params = dict(self.plugin_params)
        self._engine = None
        self._write_engine = None
        self._schema_error = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        self._engine = None
        self._write_engine = None
        self._schema_error = None

    def _pg_url(self, user, password):
        host = os.getenv("PGHOST", "127.0.0.1")
        port = os.getenv("PGPORT", "5432")
        db = os.getenv("PGDATABASE", "predictor_olap")
        if not password:
            raise RuntimeError("PGPASSWORD is not set")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    def engine(self):
        if self._engine is not None:
            return self._engine
        sqlite_path = self.params.get("sqlite_path")
        if sqlite_path:
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"timeout": 30})
        else:
            user = os.getenv("PGUSER") or "metabase"
            engine = create_engine(
                self._pg_url(user, os.getenv("PGPASSWORD")), pool_pre_ping=True
            )
        self._engine = engine
        # The gov_* DDL runs once per engine, in autocommit, under the read
        # role (the write role may hold INSERT only). A failure here must
        # not take the SELECT path down, so it is kept for write_metrics.
        self._schema_error = None
        try:
            self._ensure_schema(engine)
        except Exception as exc:
            self._schema_error = exc
            print(f"gov_* schema not ready: {exc}", file=sys.stderr)
        return engine

    def write_engine(self):
        """Engine for write_metrics: PGUSER_WRITE / PGPASSWORD_WRITE when set,
        else the read engine (same credentials, same SQLite file)."""
        self.engine()
        if self._write_engine is not None:
            return self._write_engine
        user = os.getenv("PGUSER_WRITE")
        if self.params.get("sqlite_path") or not user:
            self._write_engine = self._engine
        else:
            password = os.getenv("PGPASSWORD_WRITE") or os.getenv("PGPASSWORD")
            self._write_engine = create_engine(self._pg_url(user, password), pool_pre_ping=True)
        return self._write_engine

    def _qualified(self, name: str) -> str:
        if self.params.get("sqlite_path"):
            return f'"{name}"'
        schema = self.params.get("schema") or "public"
        return f'"{schema}"."{name}"'

    def _ddl(self, dialect: str):
        t = self._qualified
        view_select = (
            "SELECT m.report_sha256, m.experiment_key, r.lake_id, r.received_at, m.metric,"
            " m.value, m.split, m.horizon, m.std_dev, m.min_value, m.max_value, m.unit"
            f" FROM {t('gov_metric')} m JOIN {t('gov_report')} r"
            " ON r.report_sha256 = m.report_sha256"
            " WHERE r.received_at = (SELECT MAX(r2.received_at)"
            f" FROM {t('gov_report')} r2 WHERE r2.experiment_key = r.experiment_key"
            " AND r2.lake_id = r.lake_id)"
        )
        create_view = (
            "CREATE VIEW IF NOT EXISTS" if dialect == "sqlite" else "CREATE OR REPLACE VIEW"
        )
        return [
            f"CREATE TABLE IF NOT EXISTS {t('gov_report')} ("
            " report_sha256 TEXT PRIMARY KEY, experiment_key TEXT NOT NULL,"
            " experiment_set_key TEXT, actor TEXT NOT NULL, lake_id TEXT NOT NULL,"
            " received_at TEXT NOT NULL, lineage TEXT NOT NULL,"
            " config_sha256 TEXT, code_commit TEXT, project TEXT, phase TEXT, tags_json TEXT,"
            " n_metrics INTEGER NOT NULL, n_datasets INTEGER NOT NULL)",
            f"CREATE TABLE IF NOT EXISTS {t('gov_metric')} ("
            " report_sha256 TEXT NOT NULL, experiment_key TEXT NOT NULL, metric TEXT NOT NULL,"
            " value DOUBLE PRECISION, split TEXT, horizon INTEGER, std_dev DOUBLE PRECISION,"
            " min_value DOUBLE PRECISION, max_value DOUBLE PRECISION, unit TEXT)",
            f"CREATE TABLE IF NOT EXISTS {t('gov_dataset')} ("
            " report_sha256 TEXT NOT NULL, experiment_key TEXT NOT NULL, lake_id TEXT NOT NULL,"
            " resource_id TEXT NOT NULL, sha256 TEXT NOT NULL, role TEXT, lineage TEXT NOT NULL,"
            " reason TEXT, event_id INTEGER, source_sha256 TEXT, range_from TEXT, range_to TEXT,"
            " delivery TEXT, time_column TEXT)",
            f"CREATE TABLE IF NOT EXISTS {t('gov_terminal')} ("
            " terminal_sha256 TEXT PRIMARY KEY, campaign_sha256 TEXT NOT NULL,"
            " campaign_key TEXT NOT NULL, unit_id TEXT NOT NULL, generation INTEGER NOT NULL,"
            " actor TEXT NOT NULL, project TEXT NOT NULL, classification TEXT NOT NULL,"
            " status TEXT NOT NULL, reason TEXT, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,"
            " terminal_lake TEXT NOT NULL, config_sha256 TEXT NOT NULL, code_identity_json TEXT NOT NULL,"
            " costs_json TEXT NOT NULL, tags_json TEXT NOT NULL, synthetic_spec_sha256 TEXT,"
            " received_at TEXT NOT NULL, UNIQUE(campaign_sha256, unit_id, generation))",
            f"CREATE TABLE IF NOT EXISTS {t('gov_terminal_metric')} ("
            " terminal_sha256 TEXT NOT NULL, metric TEXT NOT NULL, split TEXT, horizon INTEGER,"
            " unit TEXT, value DOUBLE PRECISION, std_dev DOUBLE PRECISION,"
            " min_value DOUBLE PRECISION, max_value DOUBLE PRECISION)",
            f"CREATE TABLE IF NOT EXISTS {t('gov_terminal_dataset')} ("
            " terminal_sha256 TEXT NOT NULL, delivery_id TEXT NOT NULL, lake_id TEXT NOT NULL,"
            " resource_id TEXT NOT NULL, role TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,"
            " source_sha256 TEXT, range_from TEXT, range_to TEXT, delivery_kind TEXT,"
            " time_column TEXT, availability_contract_sha256 TEXT NOT NULL,"
            " verification_state TEXT NOT NULL)",
            f"CREATE TABLE IF NOT EXISTS {t('gov_terminal_artifact')} ("
            " terminal_sha256 TEXT NOT NULL, role TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL)",
            # S2: the contract dimension. Keyed by its own digest, immutable by construction:
            # the bytes ARE the key, so a row can be inserted or left alone, never updated.
            # `completion_lag_max` is TEXT on purpose — an archive's lag is the string
            # 'UNKNOWN', and a numeric column would have to turn that into a zero or a null.
            f"CREATE TABLE IF NOT EXISTS {t('gov_availability_contract')} ("
            " contract_sha256 TEXT NOT NULL PRIMARY KEY, canonical_bytes TEXT NOT NULL,"
            " digest_algorithm TEXT NOT NULL, canonicalization TEXT NOT NULL,"
            " use_class TEXT NOT NULL, completion_lag_max TEXT NOT NULL,"
            " availability_label TEXT, timezone_evidence TEXT, first_seen TEXT NOT NULL)",
            f"CREATE INDEX IF NOT EXISTS gov_metric_report_idx ON {t('gov_metric')} (report_sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_dataset_report_idx ON {t('gov_dataset')} (report_sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_dataset_sha256_idx ON {t('gov_dataset')} (sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_report_experiment_idx ON {t('gov_report')}"
            " (experiment_key, received_at)",
            f"CREATE INDEX IF NOT EXISTS gov_terminal_campaign_idx ON {t('gov_terminal')}"
            " (campaign_sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_terminal_metric_sha_idx ON {t('gov_terminal_metric')}"
            " (terminal_sha256)",
            f"CREATE INDEX IF NOT EXISTS gov_terminal_dataset_sha_idx ON {t('gov_terminal_dataset')}"
            " (sha256)",
            f"{create_view} {t(GOV_VIEW)} AS {view_select}",
            # What is STORED, and only that. A join proves a key matched; it does not hash
            # anything, so every semantic column here is prefixed `stored_` and the reference
            # column says STORED/ABSENT rather than RESOLVED/UNRESOLVED. Calling a row
            # "RESOLVED" because a key exists is exactly how tampered bytes were displayed as
            # a valid availability claim. Verification lives in resolve_delivery_availability().
            f"{create_view} {t(GOV_AVAILABILITY_VIEW)} AS"
            " SELECT d.terminal_sha256, d.delivery_id, d.lake_id, d.resource_id, d.role,"
            " d.availability_contract_sha256,"
            " CASE WHEN c.contract_sha256 IS NULL THEN 'ABSENT' ELSE 'STORED' END"
            " AS contract_reference,"
            " c.use_class AS stored_use_class,"
            " c.completion_lag_max AS stored_completion_lag_max,"
            " c.availability_label AS stored_availability_label,"
            " c.timezone_evidence AS stored_timezone_evidence,"
            " c.digest_algorithm AS stored_digest_algorithm,"
            " c.canonicalization AS stored_canonicalization, c.canonical_bytes"
            f" FROM {t('gov_terminal_dataset')} d LEFT JOIN {t('gov_availability_contract')} c"
            " ON c.contract_sha256 = d.availability_contract_sha256",
        ]

    def _ensure_schema(self, engine):
        dialect = engine.dialect.name
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            if dialect == "sqlite":
                conn.execute(text("PRAGMA journal_mode=WAL"))
            for statement in self._ddl(dialect):
                conn.execute(text(statement))
            schema = None if dialect == "sqlite" else self.params.get("schema") or "public"
            columns = {
                column["name"] for column in inspect(engine).get_columns(
                    "gov_terminal_dataset", schema=schema
                )
            }
            if "availability_contract_sha256" not in columns:
                conn.execute(text(
                    f"ALTER TABLE {self._qualified('gov_terminal_dataset')} "
                    "ADD COLUMN availability_contract_sha256 TEXT"
                ))

    def write_metrics(self, report):
        """Store one report (§3 'Lake side'). Returns
        {"stored", "already_stored", "lineage"}; ValueError on an invalid
        report or a report_sha256 that does not match the canonical body."""
        normalised = normalise_report(report)
        given = report.get("report_sha256")
        if not isinstance(given, str) or not _HEX64.match(given):
            raise ValueError("report_sha256 is required")
        digest = hashlib.sha256(
            canonical_text(canonical_body(normalised)).encode()
        ).hexdigest()
        if given.lower() != digest:
            raise ValueError("report_sha256 mismatch")
        if self._engine is None:
            self.engine()
        if self._schema_error is not None:
            raise RuntimeError(f"gov_* schema not ready: {self._schema_error}")
        t = self._qualified
        key = normalised["experiment_key"]
        report_row = {
            "report_sha256": digest,
            "experiment_key": key,
            "experiment_set_key": normalised["experiment_set_key"],
            "actor": normalised["actor"],
            "lake_id": normalised["lake"],
            "received_at": normalised["received_at"] or _now_utc(),
            "lineage": normalised["lineage"],
            "config_sha256": normalised["config_sha256"],
            "code_commit": normalised["code_commit"],
            "project": normalised["project"],
            "phase": normalised["phase"],
            "tags_json": canonical_text(normalised["tags"]),
            "n_metrics": len(normalised["metrics"]),
            "n_datasets": len(normalised["datasets"]),
        }
        metric_rows = [
            {"report_sha256": digest, "experiment_key": key, **m} for m in normalised["metrics"]
        ]
        dataset_rows = [
            {
                "report_sha256": digest,
                "experiment_key": key,
                "lake_id": d["lake"],
                "resource_id": d["resource"],
                "sha256": d["sha256"],
                "role": d["role"],
                **{k: d[k] for k in DATASET_LINEAGE_FIELDS},
            }
            for d in normalised["datasets"]
        ]
        with self.write_engine().begin() as conn:
            inserted = conn.execute(
                text(
                    f"INSERT INTO {t('gov_report')} (report_sha256, experiment_key,"
                    " experiment_set_key, actor, lake_id, received_at, lineage, config_sha256,"
                    " code_commit, project, phase, tags_json, n_metrics, n_datasets)"
                    " VALUES (:report_sha256, :experiment_key, :experiment_set_key, :actor,"
                    " :lake_id, :received_at, :lineage, :config_sha256, :code_commit, :project,"
                    " :phase, :tags_json, :n_metrics, :n_datasets)"
                    " ON CONFLICT (report_sha256) DO NOTHING"
                ),
                report_row,
            ).rowcount
            if inserted:
                conn.execute(
                    text(
                        f"INSERT INTO {t('gov_metric')} (report_sha256, experiment_key, metric,"
                        " value, split, horizon, std_dev, min_value, max_value, unit)"
                        " VALUES (:report_sha256, :experiment_key, :metric, :value, :split,"
                        " :horizon, :std_dev, :min_value, :max_value, :unit)"
                    ),
                    metric_rows,
                )
                if dataset_rows:
                    conn.execute(
                        text(
                            f"INSERT INTO {t('gov_dataset')} (report_sha256, experiment_key,"
                            " lake_id, resource_id, sha256, role, lineage, reason, event_id,"
                            " source_sha256, range_from, range_to, delivery, time_column)"
                            " VALUES (:report_sha256, :experiment_key, :lake_id, :resource_id,"
                            " :sha256, :role, :lineage, :reason, :event_id, :source_sha256,"
                            " :range_from, :range_to, :delivery, :time_column)"
                        ),
                        dataset_rows,
                    )
        if inserted:
            return {"stored": True, "already_stored": False, "lineage": normalised["lineage"]}
        # Read through the read engine: the write role may hold INSERT only.
        with self.engine().connect() as conn:
            stored = conn.execute(
                text(f"SELECT lineage FROM {t('gov_report')} WHERE report_sha256 = :h"),
                {"h": digest},
            ).scalar()
        return {"stored": False, "already_stored": True, "lineage": stored}

    def write_terminal(self, terminal):
        terminal = _strict_terminal(terminal)
        if self._engine is None:
            self.engine()
        if self._schema_error is not None:
            raise RuntimeError(f"gov_* schema not ready: {self._schema_error}")
        t = self._qualified
        digest = terminal["terminal_sha256"]
        slot = {
            "campaign": terminal["campaign_sha256"],
            "unit": terminal["unit_id"],
            "generation": terminal["generation"],
        }
        with self.engine().connect() as conn:
            existing = conn.execute(text(
                f"SELECT terminal_sha256 FROM {t('gov_terminal')}"
                " WHERE campaign_sha256 = :campaign AND unit_id = :unit"
                " AND generation = :generation"
            ), slot).scalar()
        if existing:
            if existing != digest:
                raise ValueError("terminal generation conflict")
            return {"stored": False, "already_stored": True, "terminal_sha256": digest}
        row = {
            "terminal_sha256": digest,
            "campaign_sha256": terminal["campaign_sha256"],
            "campaign_key": terminal["campaign_key"],
            "unit_id": terminal["unit_id"],
            "generation": terminal["generation"],
            "actor": terminal["actor"],
            "project": terminal["project"],
            "classification": terminal["classification"],
            "status": terminal["status"],
            "reason": terminal["reason"],
            "started_at": terminal["started_at"],
            "finished_at": terminal["finished_at"],
            "terminal_lake": terminal["terminal_lake"],
            "config_sha256": terminal["config_sha256"],
            "code_identity_json": canonical_text(terminal["code_identity"]),
            "costs_json": canonical_text(terminal["costs"]),
            "tags_json": canonical_text(terminal["tags"]),
            "synthetic_spec_sha256": terminal["synthetic_spec_sha256"],
            "received_at": _now_utc(),
        }
        with self.write_engine().begin() as conn:
            inserted = conn.execute(text(
                f"INSERT INTO {t('gov_terminal')} (terminal_sha256, campaign_sha256,"
                " campaign_key, unit_id, generation, actor, project, classification, status,"
                " reason, started_at, finished_at, terminal_lake, config_sha256, code_identity_json,"
                " costs_json, tags_json, synthetic_spec_sha256, received_at) VALUES"
                " (:terminal_sha256, :campaign_sha256, :campaign_key, :unit_id, :generation,"
                " :actor, :project, :classification, :status, :reason, :started_at, :finished_at,"
                " :terminal_lake, :config_sha256, :code_identity_json, :costs_json, :tags_json,"
                " :synthetic_spec_sha256, :received_at) ON CONFLICT (terminal_sha256) DO NOTHING"
            ), row).rowcount
            if not inserted:
                return {"stored": False, "already_stored": True, "terminal_sha256": digest}
            metrics = [{"terminal_sha256": digest, **item} for item in terminal["metrics"]]
            if metrics:
                conn.execute(text(
                    f"INSERT INTO {t('gov_terminal_metric')} (terminal_sha256, metric, split,"
                    " horizon, unit, value, std_dev, min_value, max_value) VALUES"
                    " (:terminal_sha256, :metric, :split, :horizon, :unit, :value, :std_dev,"
                    " :min_value, :max_value)"
                ), metrics)
            datasets = [{
                "terminal_sha256": digest,
                "verification_state": item["state"],
                **{key: item[key] for key in TERMINAL_DATASET_KEYS if key != "state"},
            } for item in terminal["verified_datasets"]]
            if datasets:
                conn.execute(text(
                    f"INSERT INTO {t('gov_terminal_dataset')} (terminal_sha256, delivery_id,"
                    " lake_id, resource_id, role, sha256, bytes, source_sha256, range_from,"
                    " range_to, delivery_kind, time_column, availability_contract_sha256,"
                    " verification_state) VALUES"
                    " (:terminal_sha256, :delivery_id, :lake_id, :resource_id, :role, :sha256,"
                    " :bytes, :source_sha256, :range_from, :range_to, :delivery_kind,"
                    " :time_column, :availability_contract_sha256, :verification_state)"
                ), datasets)
            artifacts = [{"terminal_sha256": digest, **item} for item in terminal["artifacts"]]
            if artifacts:
                conn.execute(text(
                    f"INSERT INTO {t('gov_terminal_artifact')}"
                    " (terminal_sha256, role, sha256, bytes) VALUES"
                    " (:terminal_sha256, :role, :sha256, :bytes)"
                ), artifacts)
        return {"stored": True, "already_stored": False, "terminal_sha256": digest}

    # ---- S2: the availability contract dimension ---------------------------------
    def write_availability_contracts(self, contracts):
        """Retain canonical contracts keyed by their own digest. Additive and idempotent.

        Three rules, each from what would otherwise be guessed:

        * the digest is **recomputed** over the submitted bytes and a mismatch is refused, so
          a caller cannot file arbitrary text under a digest a delivery already trusts;
        * `use_class` and `completion_lag_max` are READ OUT of the canonical bytes, never
          supplied alongside them, so the retained semantics and the retained bytes cannot
          disagree;
        * an archive's lag is the string it declares. `UNKNOWN` is stored as `UNKNOWN`; there
          is no path in this method that turns an absent or unknown lag into a zero.

        A contract already present is left exactly as it is: the bytes are the key, so a
        second submission of the same digest is the same contract.
        """
        if not isinstance(contracts, list):
            raise ValueError("contracts must be a list")
        rows, seen = [], set()
        for item in contracts:
            if not isinstance(item, dict):
                raise ValueError("each contract must be an object")
            given = item.get("contract_sha256")
            canonical = item.get("canonical_bytes")
            if not isinstance(given, str) or not _HEX64.match(given):
                raise ValueError("contract_sha256 is required")
            if not isinstance(canonical, str) or not canonical:
                raise ValueError(f"canonical_bytes is required for {given}")
            actual = hashlib.sha256(canonical.encode("ascii", "strict")).hexdigest()
            if actual != given.lower():
                raise ValueError(
                    f"contract_sha256 mismatch: declared {given.lower()}, bytes hash to {actual}")
            algorithm = item.get("digest_algorithm") or DIGEST_ALGORITHM
            canonicalization = item.get("canonicalization") or CANONICALIZATION
            if algorithm not in SUPPORTED_DIGEST_ALGORITHMS:
                raise ValueError(f"unsupported digest algorithm {algorithm!r}")
            if canonicalization not in SUPPORTED_CANONICALIZATIONS:
                # Retaining a contract whose canonicalization this store cannot reproduce
                # would file bytes it can never verify again: refused on the way in rather
                # than discovered as an unreadable row later.
                raise ValueError(
                    f"unsupported canonicalization {canonicalization!r}; this store can verify "
                    f"{list(SUPPORTED_CANONICALIZATIONS)}")
            try:
                body = json.loads(canonical)
            except ValueError as exc:
                raise ValueError(f"canonical_bytes for {actual} is not JSON: {exc}") from exc
            if not isinstance(body, dict):
                raise ValueError(f"canonical_bytes for {actual} is not an object")
            scope = body.get("availability")
            if not isinstance(scope, dict):
                raise ValueError(
                    f"contract {actual} declares no availability block: its semantics cannot "
                    "be retained, and inventing them is what this table exists to prevent")
            lag = scope.get("completion_lag_max")
            use_class = scope.get("use_class")
            if lag is None or not isinstance(use_class, str) or not use_class:
                raise ValueError(
                    f"contract {actual} must declare use_class and completion_lag_max")
            # A lag is a duration or the word UNKNOWN. `str()` of a dict or a list produces a
            # plausible-looking string that means nothing and can never be compared again, so
            # the shape is checked instead of coerced.
            if isinstance(lag, bool) or not isinstance(lag, (str, int, float)):
                raise ValueError(
                    f"contract {actual} declares completion_lag_max as {type(lag).__name__}; "
                    "it must be a string or a number, and is not stringified here")
            if use_class not in SUPPORTED_USE_CLASSES:
                raise ValueError(
                    f"contract {actual} declares use_class {use_class!r}, which this store "
                    f"cannot interpret; supported: {list(SUPPORTED_USE_CLASSES)}")
            if actual in seen:
                continue
            seen.add(actual)
            rows.append({
                "contract_sha256": actual,
                "canonical_bytes": canonical,
                "digest_algorithm": algorithm,
                "canonicalization": canonicalization,
                "use_class": use_class,
                # the DECLARED lag, as a string: 'UNKNOWN' survives as itself
                "completion_lag_max": str(lag),
                "availability_label": _opt_str(scope, "label"),
                "timezone_evidence": _opt_str(scope, "timezone_evidence"),
                "first_seen": _now_utc(),
            })
        if self._engine is None:
            self.engine()
        if self._schema_error is not None:
            raise RuntimeError(f"gov_* schema not ready: {self._schema_error}")
        t = self._qualified
        stored = 0
        with self.write_engine().begin() as conn:
            for row in rows:
                stored += conn.execute(text(
                    f"INSERT INTO {t('gov_availability_contract')} (contract_sha256,"
                    " canonical_bytes, digest_algorithm, canonicalization, use_class,"
                    " completion_lag_max, availability_label, timezone_evidence, first_seen)"
                    " VALUES (:contract_sha256, :canonical_bytes, :digest_algorithm,"
                    " :canonicalization, :use_class, :completion_lag_max, :availability_label,"
                    " :timezone_evidence, :first_seen)"
                    " ON CONFLICT (contract_sha256) DO NOTHING"
                ), row).rowcount
        return {"stored": stored, "already_stored": len(rows) - stored,
                "contracts": [row["contract_sha256"] for row in rows]}

    def resolve_delivery_availability(self, delivery_id):
        """Verify a delivery's contract from the retained bytes, or refuse to claim anything.

        The defect this replaces was reproducible: the reader returned the view's row, and the
        view calls a contract RESOLVED because a key matched. Changing the stored bytes while
        leaving the key and the cached columns alone therefore produced `RESOLVED` with
        `completion_lag_max: UNKNOWN` from bytes that said `0s` and did not hash to the key.
        A test that hashes the returned text separately does not make the READER check it.

        So this does the checking, in order, and each failure has its own outcome:

          the delivery is not recorded at all                  NO_SUCH_DELIVERY
          it is recorded more than once, disagreeing           UNRESOLVED_AMBIGUOUS_DELIVERY
          no contract was ever retained for the reference      UNRESOLVED_REFERENCE_ABSENT
          the format is not one this store can verify          UNRESOLVED_UNSUPPORTED_FORMAT
          the bytes do not hash to the key they are filed under UNRESOLVED_DIGEST_MISMATCH
          the bytes are not a contract                         UNRESOLVED_MALFORMED_CONTRACT
          the cached columns disagree with the bytes           UNRESOLVED_STORED_SEMANTICS_DISAGREE
          everything checks                                    VERIFIED

        Only `VERIFIED` carries `use_class` and `completion_lag_max`, and both are derived from
        the verified bytes rather than read from the cached columns. `stored_*` fields are
        always returned so an operator can see what the database holds even when it does not
        verify — visible, but never mistaken for a claim.
        """
        if not isinstance(delivery_id, str) or not _KEY.match(delivery_id):
            raise ValueError("invalid delivery_id")
        t = self._qualified
        with self.engine().connect() as conn:
            rows = [dict(row._mapping) for row in conn.execute(text(
                f"SELECT * FROM {t(GOV_AVAILABILITY_VIEW)} WHERE delivery_id = :delivery"
            ), {"delivery": delivery_id})]

        def refusal(resolution, row=None, **extra):
            body = {"delivery_id": delivery_id, "contract_resolution": resolution,
                    "use_class": None, "completion_lag_max": None,
                    "availability_label": None, "timezone_evidence": None}
            if row is not None:
                body.update({key: row.get(key) for key in (
                    "terminal_sha256", "lake_id", "resource_id", "role",
                    "availability_contract_sha256", "contract_reference",
                    "stored_use_class", "stored_completion_lag_max",
                    "stored_availability_label", "stored_timezone_evidence",
                    "stored_digest_algorithm", "stored_canonicalization")})
            body.update(extra)
            return body

        if not rows:
            return refusal(RESOLUTION_NO_DELIVERY)
        references = {row.get("availability_contract_sha256") for row in rows}
        if len(references) > 1:
            # The same delivery id recorded against different contracts: which one is true is
            # not this reader's guess to make.
            return refusal(RESOLUTION_AMBIGUOUS, rows[0],
                           references=sorted(reference for reference in references if reference),
                           rows=len(rows))
        row = rows[0]
        if row.get("contract_reference") != "STORED" or not row.get("canonical_bytes"):
            return refusal(RESOLUTION_ABSENT, row)

        algorithm = row.get("stored_digest_algorithm")
        canonicalization = row.get("stored_canonicalization")
        if (algorithm not in SUPPORTED_DIGEST_ALGORITHMS
                or canonicalization not in SUPPORTED_CANONICALIZATIONS):
            return refusal(RESOLUTION_UNSUPPORTED, row,
                           reason=f"digest_algorithm={algorithm!r} "
                                  f"canonicalization={canonicalization!r} is not one this "
                                  "store can verify")

        canonical = row["canonical_bytes"]
        try:
            recomputed = hashlib.sha256(canonical.encode("ascii", "strict")).hexdigest()
        except UnicodeEncodeError:
            return refusal(RESOLUTION_MALFORMED, row,
                           reason="retained bytes are not ASCII under the declared "
                                  "canonicalization")
        declared = str(row.get("availability_contract_sha256") or "").lower()
        if recomputed != declared:
            return refusal(RESOLUTION_DIGEST_MISMATCH, row, recomputed_sha256=recomputed,
                           reason="the retained bytes do not hash to the digest the delivery "
                                  "references; they have drifted or been replaced")

        try:
            body = json.loads(canonical)
            scope = body["availability"]
            use_class = scope["use_class"]
            lag = scope["completion_lag_max"]
        except (ValueError, KeyError, TypeError) as exc:
            return refusal(RESOLUTION_MALFORMED, row,
                           reason=f"retained bytes are not a contract: {exc}")
        if use_class not in SUPPORTED_USE_CLASSES:
            return refusal(RESOLUTION_UNSUPPORTED, row,
                           reason=f"use_class {use_class!r} is not one this store interprets")

        label = scope.get("label")
        evidence = scope.get("timezone_evidence")
        # The cached columns exist for querying, not for answering. If they disagree with the
        # verified bytes the row is internally inconsistent and no claim is made from either.
        disagreements = {
            name: (row.get(f"stored_{name}"), value)
            for name, value in (("use_class", use_class), ("completion_lag_max", str(lag)),
                                ("availability_label", label),
                                ("timezone_evidence", evidence))
            if row.get(f"stored_{name}") != (None if value is None else str(value))
        }
        if disagreements:
            return refusal(RESOLUTION_DISAGREEMENT, row, disagreements={
                name: {"stored": stored, "bytes_say": derived}
                for name, (stored, derived) in disagreements.items()})

        verified = refusal(RESOLUTION_VERIFIED, row)
        verified.update(use_class=use_class, completion_lag_max=str(lag),
                        availability_label=label, timezone_evidence=evidence,
                        canonical_bytes=canonical, verified_sha256=recomputed)
        return verified

    def terminal_digests(self, campaign_sha256):
        if not isinstance(campaign_sha256, str) or not _HEX64.fullmatch(campaign_sha256):
            raise ValueError("invalid campaign_sha256")
        t = self._qualified
        with self.engine().connect() as conn:
            rows = conn.execute(text(
                f"SELECT terminal_sha256, unit_id, generation FROM {t('gov_terminal')}"
                " WHERE campaign_sha256 = :campaign ORDER BY unit_id, generation"
            ), {"campaign": campaign_sha256})
            return [dict(row._mapping) for row in rows]

    def discover(self):
        insp = inspect(self.engine())
        dialect = self.engine().dialect.name
        schema = None if dialect == "sqlite" else (self.params.get("schema") or "public")
        items = []
        for kind, names in [('table', insp.get_table_names(schema=schema)), ('view', insp.get_view_names(schema=schema))]:
            for name in sorted(names):
                items.append({"resource_id": name, "kind": kind, "schema": schema, "rows": None,
                              "row_count_status": "NOT_SCANNED"})
        return sorted(items, key=lambda item: item['resource_id'])

    def resource_schema(self, resource_id):
        inventory = {item['resource_id']: item for item in self.discover()}
        if resource_id not in inventory:
            raise ValueError('resource not in inventory')
        resource = inventory[resource_id]
        schema = resource['schema']
        inspector = inspect(self.engine())
        columns = inspector.get_columns(resource_id, schema=schema)
        return {
            **resource,
            'columns': [
                {'name': column['name'], 'type': str(column['type']),
                 'nullable': column.get('nullable'), 'default': column.get('default')}
                for column in columns
            ],
            'primary_key': inspector.get_pk_constraint(resource_id, schema=schema).get('constrained_columns', []),
            'foreign_keys': inspector.get_foreign_keys(resource_id, schema=schema),
            'indexes': inspector.get_indexes(resource_id, schema=schema),
        }

    def list_resources(self):
        return self.discover()

    def _guard(self, sql: str) -> str:
        text_sql = (sql or "").strip().rstrip(";").strip()
        if ";" in text_sql:
            raise ValueError("multiple statements are not allowed")
        head = text_sql.split(None, 1)[0].lower() if text_sql else ""
        if head not in {"select", "with"}:
            raise ValueError("only SELECT/WITH is allowed")
        if _WRITE.search(text_sql):
            raise ValueError("write SQL is not allowed")
        if _BANNED.search(text_sql):
            raise ValueError("function is not allowed")
        match = _LIMIT.search(text_sql)
        if not match:
            raise ValueError("LIMIT n is required")
        if int(match.group(1)) > MAX_LIMIT:
            raise ValueError("LIMIT too large")
        return text_sql

    def query(self, sql: str):
        guarded = self._guard(sql)
        with self.engine().connect() as conn:
            result = conn.execute(text(guarded))
            rows = [dict(row._mapping) for row in result]
        holdout = self.params.get("holdout_start")
        if holdout:
            limit = date.fromisoformat(str(holdout)[:10])
            time_cols = [self.params.get("time_column")] if self.params.get("time_column") else []
            if rows:
                for key in rows[0]:
                    lk = str(key).lower()
                    if lk in {"ts", "time", "date", "datetime", "timestamp"} or "time" in lk or "date" in lk:
                        if key not in time_cols:
                            time_cols.append(key)
            for row in rows:
                for col in time_cols:
                    if not col or col not in row or not row[col]:
                        continue
                    try:
                        day = date.fromisoformat(str(row[col])[:10])
                    except ValueError:
                        continue
                    if day >= limit:
                        raise PermissionError("holdout")
        canonical = json.dumps(rows, default=str, sort_keys=True, separators=(",", ":"))
        return {
            "rows": rows,
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "bytes": len(canonical.encode()),
        }

    def storage(self):
        import shutil

        sqlite_path = self.params.get("sqlite_path")
        if sqlite_path:
            path = Path(sqlite_path)
            usage = shutil.disk_usage(path.parent if path.parent.exists() else Path("."))
            size = path.stat().st_size if path.exists() else 0
            root = str(path)
        else:
            usage = shutil.disk_usage(".")
            size = 0
            root = "postgresql (PGHOST)"
        return {
            "root": root,
            "host_total": usage.total,
            "host_used": usage.used,
            "host_free": usage.free,
            "lake_bytes": size,
        }

    def describe(self):
        try:
            n_resources = len(self.discover())
        except Exception:
            n_resources = None
        return {
            "lake_id": self.params.get("lake_id"),
            "title": self.params.get("title"),
            "description": self.params.get("description"),
            "kind": self.params.get("kind"),
            "root_path": self.storage()["root"],
            "holdout_start": self.params.get("holdout_start"),
            "n_resources": n_resources,
        }
