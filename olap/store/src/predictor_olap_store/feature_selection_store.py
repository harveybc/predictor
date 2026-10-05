"""Normalized, immutable storage for phase-1 feature-selection envelopes."""

from __future__ import annotations

import json

from sqlalchemy import text

from .feature_selection import ROW_FAMILIES, canonical_json, digest, row_identity


TABLES = (
    "df_dim_feature_selection_run",
    "df_fact_sampling_quality",
    "df_fact_variable_profile",
    "df_fact_information_metric",
    "df_fact_pair_relation",
    "df_fact_feature_causal_evidence",
    "df_fact_feature_selection_decision",
    "df_fact_feature_selection_load_receipt",
)
VIEWS = (
    "df_feature_profile_current",
    "df_feature_causal_ladder_current",
    "df_feature_selection_current",
    "df_feature_selection_coverage",
    "df_feature_selection_failures",
    "df_feature_selection_dashboard",
)

FAMILY_TABLES = {
    "sampling_quality": "df_fact_sampling_quality",
    "variable_profiles": "df_fact_variable_profile",
    "information_metrics": "df_fact_information_metric",
    "pair_relations": "df_fact_pair_relation",
    "causal_evidence": "df_fact_feature_causal_evidence",
    "selection_decisions": "df_fact_feature_selection_decision",
}


def ddl(qualified, dialect: str) -> list[str]:
    """Return additive schema DDL for all normalized facts and read-only views."""
    t = qualified
    create_view = "CREATE VIEW IF NOT EXISTS" if dialect == "sqlite" else "CREATE OR REPLACE VIEW"
    common = (
        " row_identity_sha256 TEXT PRIMARY KEY, row_sha256 TEXT NOT NULL,"
        f" run_id TEXT NOT NULL REFERENCES {t('df_dim_feature_selection_run')}(run_id),"
        " feature_id TEXT NOT NULL, split TEXT NOT NULL, metric_name TEXT NOT NULL,"
        " metric_value DOUBLE PRECISION, state TEXT NOT NULL, unit TEXT,"
        " population_id TEXT, fold TEXT"
    )
    return [
        f"CREATE TABLE IF NOT EXISTS {t('df_dim_feature_selection_run')} ("
        " run_id TEXT PRIMARY KEY, run_sha256 TEXT NOT NULL UNIQUE,"
        " campaign_sha256 TEXT NOT NULL, code_sha256 TEXT NOT NULL,"
        " input_sha256 TEXT NOT NULL, inventory_sha256 TEXT NOT NULL, created_at TEXT NOT NULL)",
        f"CREATE TABLE IF NOT EXISTS {t('df_fact_sampling_quality')} ({common})",
        f"CREATE TABLE IF NOT EXISTS {t('df_fact_variable_profile')} ({common})",
        f"CREATE TABLE IF NOT EXISTS {t('df_fact_information_metric')} ("
        " row_identity_sha256 TEXT PRIMARY KEY, row_sha256 TEXT NOT NULL,"
        f" run_id TEXT NOT NULL REFERENCES {t('df_dim_feature_selection_run')}(run_id),"
        " feature_id TEXT NOT NULL, target_id TEXT NOT NULL, horizon INTEGER NOT NULL,"
        " split TEXT NOT NULL, metric_name TEXT NOT NULL, metric_value DOUBLE PRECISION,"
        " state TEXT NOT NULL, population_id TEXT, fold TEXT)",
        f"CREATE TABLE IF NOT EXISTS {t('df_fact_pair_relation')} ("
        " row_identity_sha256 TEXT PRIMARY KEY, row_sha256 TEXT NOT NULL,"
        f" run_id TEXT NOT NULL REFERENCES {t('df_dim_feature_selection_run')}(run_id),"
        " feature_id TEXT NOT NULL, target_id TEXT NOT NULL, horizon INTEGER NOT NULL,"
        " split TEXT NOT NULL, lag INTEGER NOT NULL, metric_name TEXT NOT NULL,"
        " metric_value DOUBLE PRECISION, state TEXT NOT NULL, population_id TEXT, fold TEXT)",
        f"CREATE TABLE IF NOT EXISTS {t('df_fact_feature_causal_evidence')} ("
        " row_identity_sha256 TEXT PRIMARY KEY, row_sha256 TEXT NOT NULL,"
        f" run_id TEXT NOT NULL REFERENCES {t('df_dim_feature_selection_run')}(run_id),"
        " feature_id TEXT NOT NULL, target_id TEXT NOT NULL, horizon INTEGER NOT NULL,"
        " split TEXT NOT NULL, rung INTEGER NOT NULL CHECK (rung BETWEEN 1 AND 3),"
        " estimand TEXT NOT NULL, estimator TEXT NOT NULL, state TEXT NOT NULL,"
        " effect DOUBLE PRECISION, lower_bound DOUBLE PRECISION, upper_bound DOUBLE PRECISION,"
        " support_n INTEGER NOT NULL, assumptions_json TEXT NOT NULL,"
        " adjustment_set_json TEXT NOT NULL, evidence_sha256 TEXT NOT NULL,"
        " population_id TEXT, fold TEXT)",
        f"CREATE TABLE IF NOT EXISTS {t('df_fact_feature_selection_decision')} ("
        " row_identity_sha256 TEXT PRIMARY KEY, row_sha256 TEXT NOT NULL,"
        f" run_id TEXT NOT NULL REFERENCES {t('df_dim_feature_selection_run')}(run_id),"
        " feature_id TEXT NOT NULL, target_id TEXT NOT NULL, horizon INTEGER NOT NULL,"
        " method TEXT NOT NULL, score DOUBLE PRECISION, rank INTEGER, decision TEXT NOT NULL,"
        " rule TEXT NOT NULL, evidence_sha256 TEXT NOT NULL, population_id TEXT,"
        " CHECK (decision IN ('SELECTED','REJECTED','NEUTRAL','UNAVAILABLE')))",
        f"CREATE TABLE IF NOT EXISTS {t('df_fact_feature_selection_load_receipt')} ("
        " envelope_sha256 TEXT PRIMARY KEY,"
        f" run_id TEXT NOT NULL REFERENCES {t('df_dim_feature_selection_run')}(run_id),"
        " schema_version TEXT NOT NULL, row_count INTEGER NOT NULL, body_sha256 TEXT NOT NULL,"
        " feature_ids_json TEXT,"
        " stored_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
        f"{create_view} {t('df_feature_profile_current')} AS "
        "SELECT run_id, feature_id, split, metric_name, metric_value, state,"
        f" 'sampling_quality' AS family, row_sha256 FROM {t('df_fact_sampling_quality')} "
        "UNION ALL SELECT run_id, feature_id, split, metric_name, metric_value, state,"
        f" 'variable_profiles', row_sha256 FROM {t('df_fact_variable_profile')} "
        "UNION ALL SELECT run_id, feature_id, split, metric_name, metric_value, state,"
        f" 'information_metrics', row_sha256 FROM {t('df_fact_information_metric')} "
        "UNION ALL SELECT run_id, feature_id, split, metric_name, metric_value, state,"
        f" 'pair_relations', row_sha256 FROM {t('df_fact_pair_relation')}",
        f"{create_view} {t('df_feature_causal_ladder_current')} AS "
        f"SELECT * FROM {t('df_fact_feature_causal_evidence')}",
        f"{create_view} {t('df_feature_selection_current')} AS "
        f"SELECT * FROM {t('df_fact_feature_selection_decision')}",
        f"{create_view} {t('df_feature_selection_coverage')} AS WITH profile AS ("
        f" SELECT run_id, feature_id, count(*) AS profile_rows FROM {t('df_feature_profile_current')}"
        " GROUP BY run_id, feature_id), causal AS ("
        f" SELECT run_id, feature_id, count(*) AS causal_rows FROM {t('df_fact_feature_causal_evidence')}"
        " GROUP BY run_id, feature_id), decisions AS ("
        f" SELECT run_id, feature_id, count(*) AS decision_rows FROM {t('df_fact_feature_selection_decision')}"
        " GROUP BY run_id, feature_id), features AS ("
        " SELECT run_id, feature_id FROM profile UNION SELECT run_id, feature_id FROM causal"
        " UNION SELECT run_id, feature_id FROM decisions)"
        " SELECT f.run_id, f.feature_id, coalesce(p.profile_rows, 0) AS profile_rows,"
        " coalesce(c.causal_rows, 0) AS causal_rows, coalesce(d.decision_rows, 0) AS decision_rows"
        " FROM features f LEFT JOIN profile p USING (run_id, feature_id)"
        " LEFT JOIN causal c USING (run_id, feature_id)"
        " LEFT JOIN decisions d USING (run_id, feature_id)",
        f"{create_view} {t('df_feature_selection_failures')} AS "
        f"SELECT run_id, 'sampling_quality' AS family, feature_id, state, row_sha256 FROM {t('df_fact_sampling_quality')} WHERE state IN ('FAILED','ERROR','INVALID') "
        f"UNION ALL SELECT run_id, 'variable_profiles', feature_id, state, row_sha256 FROM {t('df_fact_variable_profile')} WHERE state IN ('FAILED','ERROR','INVALID') "
        f"UNION ALL SELECT run_id, 'information_metrics', feature_id, state, row_sha256 FROM {t('df_fact_information_metric')} WHERE state IN ('FAILED','ERROR','INVALID') "
        f"UNION ALL SELECT run_id, 'pair_relations', feature_id, state, row_sha256 FROM {t('df_fact_pair_relation')} WHERE state IN ('FAILED','ERROR','INVALID') "
        f"UNION ALL SELECT run_id, 'causal_evidence', feature_id, state, row_sha256 FROM {t('df_fact_feature_causal_evidence')} WHERE state IN ('FAILED','ERROR','INVALID')",
        f"{create_view} {t('df_feature_selection_dashboard')} AS SELECT r.run_id,"
        f" (SELECT count(DISTINCT feature_id) FROM {t('df_feature_selection_coverage')} c WHERE c.run_id = r.run_id) AS total_features,"
        f" (SELECT count(DISTINCT feature_id) FROM {t('df_fact_feature_selection_decision')} d WHERE d.run_id = r.run_id AND d.decision = 'SELECTED') AS selected_features,"
        f" (SELECT count(*) FROM {t('df_feature_selection_failures')} f WHERE f.run_id = r.run_id) AS failed_rows"
        f" FROM {t('df_dim_feature_selection_run')} r",
    ]


def _existing_digest(connection, qualified, table: str, identity_column: str,
                     identity: str) -> str | None:
    return connection.execute(text(
        f"SELECT row_sha256 FROM {qualified(table)} WHERE {identity_column} = :identity"
    ), {"identity": identity}).scalar()


def _insert_run(connection, qualified, run: dict) -> None:
    run_sha256 = digest(run)
    existing = connection.execute(text(
        f"SELECT run_sha256 FROM {qualified('df_dim_feature_selection_run')} WHERE run_id = :run_id"
    ), {"run_id": run["run_id"]}).scalar()
    if existing is not None:
        if existing != run_sha256:
            raise ValueError(f"run_id {run['run_id']!r} contradicts its stored identity")
        return
    connection.execute(text(
        f"INSERT INTO {qualified('df_dim_feature_selection_run')} (run_id, run_sha256,"
        " campaign_sha256, code_sha256, input_sha256, inventory_sha256, created_at)"
        " VALUES (:run_id, :run_sha256, :campaign_sha256, :code_sha256, :input_sha256,"
        " :inventory_sha256, :created_at)"
    ), {**run, "run_sha256": run_sha256})


def _insert_row(connection, qualified, run_id: str, family: str, row: dict) -> None:
    table = FAMILY_TABLES[family]
    identity = row_identity(run_id, family, row)
    existing = _existing_digest(connection, qualified, table, "row_identity_sha256", identity)
    if existing is not None:
        if existing != row["row_sha256"]:
            raise ValueError(f"{family} identity {identity} contradicts its stored row")
        return

    values = {**row, "run_id": run_id, "row_identity_sha256": identity}
    if family in {"sampling_quality", "variable_profiles"}:
        columns = ("row_identity_sha256", "row_sha256", "run_id", "feature_id", "split",
                   "metric_name", "metric_value", "state", "unit", "population_id", "fold")
    elif family == "information_metrics":
        columns = ("row_identity_sha256", "row_sha256", "run_id", "feature_id", "target_id",
                   "horizon", "split", "metric_name", "metric_value", "state",
                   "population_id", "fold")
    elif family == "pair_relations":
        columns = ("row_identity_sha256", "row_sha256", "run_id", "feature_id", "target_id",
                   "horizon", "split", "lag", "metric_name", "metric_value", "state",
                   "population_id", "fold")
    elif family == "causal_evidence":
        values.update(
            lower_bound=row.get("lower"), upper_bound=row.get("upper"),
            assumptions_json=canonical_json(row["assumptions"]),
            adjustment_set_json=canonical_json(row["adjustment_set"]),
        )
        columns = ("row_identity_sha256", "row_sha256", "run_id", "feature_id", "target_id",
                   "horizon", "split", "rung", "estimand", "estimator", "state", "effect",
                   "lower_bound", "upper_bound", "support_n", "assumptions_json",
                   "adjustment_set_json", "evidence_sha256", "population_id", "fold")
    else:
        columns = ("row_identity_sha256", "row_sha256", "run_id", "feature_id", "target_id",
                   "horizon", "method", "score", "rank", "decision", "rule",
                   "evidence_sha256", "population_id")
    names = ", ".join(columns)
    parameters = ", ".join(f":{name}" for name in columns)
    connection.execute(text(f"INSERT INTO {qualified(table)} ({names}) VALUES ({parameters})"),
                       {name: values.get(name) for name in columns})


def write(connection, qualified, document: dict) -> dict:
    """Commit one already-validated envelope using the caller's single transaction."""
    envelope_sha256 = document["envelope_sha256"]
    row_count = sum(len(document["rows"][name]) for name in ROW_FAMILIES)
    feature_ids = sorted({
        row["feature_id"]
        for family in ROW_FAMILIES
        for row in document["rows"][family]
    })
    existing = connection.execute(text(
        f"SELECT body_sha256 FROM {qualified('df_fact_feature_selection_load_receipt')}"
        " WHERE envelope_sha256 = :envelope_sha256"
    ), {"envelope_sha256": envelope_sha256}).scalar()
    if existing is not None:
        if existing != digest(document):
            raise ValueError(f"envelope {envelope_sha256} contradicts its stored receipt")
        return {"stored": False, "already_stored": True,
                "envelope_sha256": envelope_sha256, "row_count": row_count}

    run = document["run"]
    _insert_run(connection, qualified, run)
    for family in ROW_FAMILIES:
        for row in document["rows"][family]:
            _insert_row(connection, qualified, run["run_id"], family, row)
    connection.execute(text(
        f"INSERT INTO {qualified('df_fact_feature_selection_load_receipt')}"
        " (envelope_sha256, run_id, schema_version, row_count, body_sha256, feature_ids_json)"
        " VALUES (:envelope_sha256, :run_id, :schema_version, :row_count, :body_sha256,"
        " :feature_ids_json)"
    ), {
        "envelope_sha256": envelope_sha256,
        "run_id": run["run_id"],
        "schema_version": document["schema_version"],
        "row_count": row_count,
        "body_sha256": digest(document),
        "feature_ids_json": canonical_json(feature_ids),
    })
    return {"stored": True, "already_stored": False,
            "envelope_sha256": envelope_sha256, "row_count": row_count}


def _retained_features(connection, qualified, run_id: str) -> set[str]:
    selects = [
        f"SELECT feature_id FROM {qualified(table)} WHERE run_id = :run_id"
        for table in FAMILY_TABLES.values()
    ]
    rows = connection.execute(text(" UNION ".join(selects)), {"run_id": run_id})
    return {row[0] for row in rows}


def _authenticate_run(connection, qualified, run_id: str) -> None:
    row = connection.execute(text(
        f"SELECT run_sha256, campaign_sha256, code_sha256, input_sha256, inventory_sha256,"
        f" created_at FROM {qualified('df_dim_feature_selection_run')} WHERE run_id = :run_id"
    ), {"run_id": run_id}).mappings().one_or_none()
    if row is None:
        raise ValueError(f"warehouse population contradiction; retained run {run_id!r} is absent")
    run = {
        "run_id": run_id,
        "campaign_sha256": row["campaign_sha256"],
        "code_sha256": row["code_sha256"],
        "input_sha256": row["input_sha256"],
        "inventory_sha256": row["inventory_sha256"],
        "created_at": row["created_at"],
    }
    if digest(run) != row["run_sha256"]:
        raise ValueError(
            f"warehouse population contradiction; retained run {run_id!r} changed identity"
        )


def reconcile(connection, qualified, request: dict) -> dict:
    """Verify a canonical request against retained receipts, runs and fact rows."""
    from .feature_selection_reconciliation import reconciliation_response

    observed = []
    missing = []
    contradictions = []
    for identity in request["identities"]:
        if identity["terminal_state"] == "UNAVAILABLE":
            observed.append(identity)
            continue
        receipt = connection.execute(text(
            f"SELECT run_id, feature_ids_json FROM "
            f"{qualified('df_fact_feature_selection_load_receipt')}"
            " WHERE envelope_sha256 = :envelope_sha256"
        ), {"envelope_sha256": identity["envelope_sha256"]}).mappings().one_or_none()
        if receipt is None:
            missing.append(identity["feature_id"])
            continue
        _authenticate_run(connection, qualified, receipt["run_id"])
        retained_features = _retained_features(connection, qualified, receipt["run_id"])
        if identity["feature_id"] not in retained_features:
            raise ValueError(
                "warehouse population contradiction; claimed feature has no retained rows: "
                f"{identity['feature_id']!r}"
            )
        encoded = receipt["feature_ids_json"]
        if encoded is None:
            raise ValueError(
                "warehouse population contradiction; receipt feature identity is absent"
            )
        try:
            receipt_features = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "warehouse population contradiction; receipt feature identity is invalid"
            ) from exc
        if (
            not isinstance(receipt_features, list)
            or not all(isinstance(item, str) and item for item in receipt_features)
            or receipt_features != sorted(set(receipt_features))
            or not set(receipt_features).issubset(retained_features)
            or identity["feature_id"] not in receipt_features
        ):
            contradictions.append(identity["feature_id"])
            continue
        observed.append(identity)
    if missing:
        raise ValueError(
            f"incomplete warehouse population; missing envelopes for {sorted(missing)}"
        )
    if contradictions:
        raise ValueError(
            "warehouse population contradiction; claimed feature is absent from its envelope: "
            f"{sorted(contradictions)}"
        )
    if len(observed) != request["expected_count"]:
        raise ValueError("incomplete warehouse population after reconciliation")
    return reconciliation_response(request, observed)
