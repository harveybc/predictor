#!/usr/bin/env python3
"""Readiness ledger for the retained 366 candidates.

The accepted PS4 profile covers ten emitted transforms. It does not close PS4
for the population and it does not select a feature. Reconstruction is not
selection utility. Preparation of PS5 inputs is not permission to train.
"""

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

BASE = Path("docs/audits/evidence/canonical_20261003")
DENOMINATOR = 366
JOIN_COUNT = 279
OUTSIDE_COUNT = 87
PROFILE_SCHEMA = "ps4_transform_profile.v1"
PROFILE_UNITS = 50
PROFILE_ROWS = 1050
PRIMARY_K = 24
SEALED_K = (8, 16, 24, 32, 48)
SEED = 0
INNER_FOLDS = ("inner_2019", "inner_2020", "inner_2021", "inner_2022", "inner_2023")
MEASURED_FEATURES = (
    "tv.wav_d1",
    "tv.wav_d2",
    "tv.wav_d3",
    "tv.wav_d4",
    "tv.wav_d5",
    "tv.mt_band_6_48h",
    "tv.hilbert_amp",
    "tv.stl_dev",
    "tv.stl_seasonal",
    "tv.kalman_dev",
)
PS4_METRICS = (
    "compressed_bits_per_sample_lzma6_raw_float64",
    "compressed_bits_per_sample_lzma6_symbols",
    "compressed_bits_per_sample_zlib9_raw_float64",
    "compressed_bits_per_sample_zlib9_symbols",
    "compression_gain_vs_uncompressed_lzma6_raw_float64",
    "compression_gain_vs_uncompressed_lzma6_symbols",
    "compression_gain_vs_uncompressed_zlib9_raw_float64",
    "compression_gain_vs_uncompressed_zlib9_symbols",
    "conditional_redundancy_bits_lag1",
    "conditional_surprisal_bits_lag1",
    "discrete_entropy_bits",
    "permutation_entropy_order_3",
    "permutation_entropy_order_4",
    "permutation_entropy_order_5",
    "spectral_entropy_iqr",
    "spectral_entropy_median",
    "spectral_entropy_window_count",
    "temporal_structure_gain_lzma6_raw_float64",
    "temporal_structure_gain_lzma6_symbols",
    "temporal_structure_gain_zlib9_raw_float64",
    "temporal_structure_gain_zlib9_symbols",
)
VIX_FEATURE = "fred.stress.vixcls.logret_5d"
VIX_RESULTS_SHA256 = "ec16a107f0ecc2e5c86002e15a2397e214e2d7c0a9df1145048f678e14546f8a"
DGS30_FEATURE = "fred.rates.dgs30.level"
DGS30_RESULTS_SHA256 = "f7763f142b7f02c72ae50e6b93562d6614f0563eb1694c26c21dc0a57e9416d5"
DPRIME_FEATURE = "fred.rates.dprime.logret_5d"
DPRIME_RESULTS_SHA256 = "a7b85bd93e2a22e76a85ce90d9d16dc75adc516c8109d11068c1326e0d8f4bb3"
AUD_EWMA_FEATURE = "fx.audusd.ewma_vol_24"
AUD_EWMA_RESULTS_SHA256 = "e582170a89d19b244682f44a8b60e722226e97e22ba31e788745bf2e4c4a44e3"
AUD_LOGRET1H_FEATURE = "fx.audusd.logret_1h"
AUD_LOGRET1H_RESULTS_SHA256 = "c59cf69c6a211ac5d66c033f6aeb0d3777dc57fd65bcccd6182b8e5ee9de9839"
AUD_LOGRET24H_FEATURE = "fx.audusd.logret_24h"
AUD_LOGRET24H_RESULTS_SHA256 = "80a3a6e5686b2bc78f30be5c0126e09c83d7d5aa7dc904e5fba68bbaf8c77590"
EURGBP_EWMA_FEATURE = "fx.eurgbp.ewma_vol_24"
EURGBP_EWMA_RESULTS_SHA256 = "2df6a5faffb2e0222aa913de3b11fcb1f380e73b2ffc40aded9fae89fd2018cc"
EURGBP_LOGRET1H_FEATURE = "fx.eurgbp.logret_1h"
EURGBP_LOGRET1H_RESULTS_SHA256 = "7195ba2cc42627cfff2bccbf4af9672334a31c00c91e60ecc13195b9cbe6013e"
EURJPY_EWMA_FEATURE = "fx.eurjpy.ewma_vol_24"
EURJPY_EWMA_RESULTS_SHA256 = "b9fc80552d48ec655b17a63d626d6b41cbc26debaa3f81926641f1c23d298c0f"
EURJPY_LOGRET1H_FEATURE = "fx.eurjpy.logret_1h"
EURJPY_LOGRET1H_RESULTS_SHA256 = "9660aa42fd31d4539d7940f6d7bab3eb2dd410fcf963632e2d12d0c497c0505b"
SIAMESE_FEATURE = "px.logret_6h"
SIAMESE_FAMILY = "past_to_current_siamese"
TRAIN_EXCLUSIVE_END = datetime.fromisoformat("2024-01-01T00:00:00+00:00")
ZERO_OR_REJECTED = {"", "0", "0.0", "ZERO", "REJECTED"}
TERMINAL_STATUSES = {
    "ADMITTED_PS1_MEASURED",
    "ADMITTED_PS1_MEASURED_WITH_NOT_APPLICABLE",
    "ADMITTED_PS1_CELL_FAILED",
    "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS",
    "MIXED_IDENTIFIED_AND_NOT_IDENTIFIED",
    "LANE_E_MEASURED_NOT_SELECTION",
    "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
    "MEASURED_SUBPOPULATION",
}
READINESS_FIELDS = (
    "row_index",
    "feature_id",
    "denominator",
    "ps0_ps1_status",
    "ps2_status",
    "ps3c_status",
    "ps3c_producer_revision",
    "ps3c_join_sha256",
    "ps3r_status",
    "ps4_status",
    "ps5_status",
    "evidence_locator",
    "evidence_digest",
    "evidence_kind",
    "missing_next_action",
)
SCHEDULE_FIELDS = (
    "feature_id",
    "fold",
    "metric",
    "state",
    "reason",
    "product_bound",
    "selection_decision",
)
INTEGRATION_FIELDS = (
    "feature_id",
    "prior_status",
    "profile_status",
    "schema",
    "feature_fold_units",
    "metric_rows",
    "profile_rows_sha256",
    "report_sha256",
    "input_digests_sha256",
    "profiler_code_digest",
)
LOCAL_PROFILE_INPUTS = {
    "READY": BASE / "laneA" / "batch_003" / "READY",
    "admissible_features.json": BASE / "laneA" / "batch_003" / "admissible_features.json",
    "digests.json": BASE / "laneA" / "batch_003" / "digests.json",
    "folds.json": BASE / "laneA" / "batch_001" / "folds.json",
    "transform_feature_join.csv": BASE / "ps4_transform_join" / "transform_feature_join.csv",
}
ARMS = (
    "predictive_baseline",
    "plus_causal",
    "plus_extractibility",
    "random_k",
    "all_admissible",
)


class ReadinessError(Exception):
    """A readiness invariant failed. The population is not shortened."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def digest_path(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def require_fresh_digest(path, claimed):
    actual = digest_path(path)
    if claimed != actual:
        raise ReadinessError("STALE_PS4_DIGEST", f"{Path(path).name} {actual} != {claimed}")
    return actual


def require_population(rows, expected_ids):
    got = [row["feature_id"] for row in rows]
    counts = Counter(got)
    duplicated = sorted(feature for feature, count in counts.items() if count > 1)
    if duplicated:
        raise ReadinessError("DUPLICATE_CANDIDATE", duplicated[0])
    missing = [feature for feature in expected_ids if feature not in counts]
    if missing:
        raise ReadinessError("MISSING_CANDIDATE", missing[0])
    if len(expected_ids) != DENOMINATOR or len(got) != DENOMINATOR:
        raise ReadinessError("DENOMINATOR", f"{len(got)} != {DENOMINATOR}")
    if len(set(expected_ids)) != DENOMINATOR:
        raise ReadinessError("DUPLICATE_CANDIDATE", "expected population")
    return got


def refuse_missing_cast(status):
    if status in ZERO_OR_REJECTED or status == 0:
        raise ReadinessError("MISSING_CAST_TO_ZERO_OR_REJECTED", str(status))


def refuse_false_causal_rejection(status, rung2):
    neutral = int(rung2.get("NOT_IDENTIFIED", 0)) + int(rung2.get("NOT_EVALUATED", 0))
    if status in ZERO_OR_REJECTED or status == 0:
        raise ReadinessError("FALSE_CAUSAL_REJECTION", str(status))
    if neutral and "REJECT" in status:
        raise ReadinessError("FALSE_CAUSAL_REJECTION", status)


def load_causal_joins(root):
    """Authoritative PS3-C source: the three 48ae17c joins, not coverage counts."""

    index = {}
    hashes = {}
    base = Path(root) / BASE / "laneC" / "reanalysis_48ae17c"
    for batch in ("batch_001", "batch_002", "batch_003"):
        path = base / batch / "ps3c_join.json"
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        payload = json.loads(blob)
        if payload.get("producer_revision") != "48ae17c":
            raise ReadinessError("PS3C_SOURCE", batch)
        if payload.get("survivors"):
            raise ReadinessError("PS3C_SOURCE", "survivors")
        hashes[batch] = digest
        for row in payload["join"]:
            feature_id = row["feature_id"]
            if feature_id in index:
                raise ReadinessError("PS3C_SOURCE", feature_id)
            index[feature_id] = {
                "producer_revision": "48ae17c",
                "join_sha256": digest,
                "batch": batch,
            }
    if len(index) != JOIN_COUNT:
        raise ReadinessError("PS3C_SOURCE", str(len(index)))
    return index, hashes


def coverage_causal_counts(features):
    """Superseded counter. Retained so the old 65/13 claim stays reproducible."""

    counts = Counter(ps3c_status(feature) for feature in features)
    return {
        "identified": counts["IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS"],
        "mixed": counts["MIXED_IDENTIFIED_AND_NOT_IDENTIFIED"],
        "not_identified": counts["NOT_IDENTIFIED"],
        "not_applicable": counts["NOT_APPLICABLE"],
    }


def ps3c_status(feature):
    if feature.get("coverage_status") == "JOINED_CALENDAR_CONDITIONING_NOT_EXTRACTED":
        status = "NOT_APPLICABLE"
    else:
        rung2 = feature["rung2_state_counts"]
        rung3 = feature["rung3_state_counts"]
        identified = int(rung2.get("IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS", 0))
        identified += int(rung3.get("COUNTERFACTUAL_UNDER_DECLARED_SCM", 0))
        neutral = int(rung2.get("NOT_IDENTIFIED", 0)) + int(rung2.get("NOT_EVALUATED", 0))
        neutral += int(rung3.get("NOT_IDENTIFIED", 0)) + int(rung3.get("NOT_EVALUATED", 0))
        if identified and neutral:
            status = "MIXED_IDENTIFIED_AND_NOT_IDENTIFIED"
        elif identified:
            status = "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS"
        elif neutral:
            status = "NOT_IDENTIFIED"
        else:
            raise ReadinessError("PS3C_UNCLASSIFIED", feature.get("feature_id", ""))
    refuse_false_causal_rejection(status, feature.get("rung2_state_counts") or {})
    refuse_missing_cast(status)
    return status


def require_reconstruction_not_selection(cell):
    decision = cell.get("selection_decision", "NOT_ISSUED")
    feature = cell.get("feature_id", "")
    if decision in {"SELECTED", "REJECTED"}:
        raise ReadinessError("RECONSTRUCTION_PRESENTED_AS_SELECTION", feature or decision)
    if cell.get("reconstruction_selects") or cell.get("utility") == "reconstruction":
        raise ReadinessError("RECONSTRUCTION_PRESENTED_AS_SELECTION", feature or "reconstruction")
    return "NOT_SELECTION"


def require_siamese_not_promoted(cell):
    """An overwritten siamese results file is not a new selection result."""
    if cell.get("feature_id") != SIAMESE_FEATURE or cell.get("family") != SIAMESE_FAMILY:
        return "NOT_THIS_CELL"
    promoted = bool(cell.get("promote")) or cell.get("selection_decision") in {"SELECTED", "REJECTED"}
    if promoted or cell.get("utility") == "reconstruction":
        raise ReadinessError(
            "SIAMESE_RESULTS_OVERWRITTEN",
            "manifest digest is not a new selection result",
        )
    if cell.get("results_overwritten"):
        return "NOT_PROMOTED_RESULTS_OVERWRITTEN"
    return "NOT_PROMOTED"


def ps2_status(counts):
    parts = []
    for name in ("PROVISIONAL_SURVIVOR", "PROVISIONAL_LOW_PRIORITY", "EXPLORATION"):
        parts.append(f"{name}={int(counts.get(name, 0))}")
    extra = sorted(name for name in counts if name not in {
        "PROVISIONAL_SURVIVOR", "PROVISIONAL_LOW_PRIORITY", "EXPLORATION",
    })
    for name in extra:
        parts.append(f"{name}={int(counts[name])}")
    status = ";".join(parts)
    refuse_missing_cast(status)
    return status


def ps1_status(states):
    if "FAILED" in states:
        return "ADMITTED_PS1_CELL_FAILED"
    if "NOT_APPLICABLE" in states:
        return "ADMITTED_PS1_MEASURED_WITH_NOT_APPLICABLE"
    if states == {"MEASURED"}:
        return "ADMITTED_PS1_MEASURED"
    raise ReadinessError("PS1_UNCLASSIFIED", ",".join(sorted(states)))


def prioritized_for_ps4(counts):
    return int(counts.get("PROVISIONAL_SURVIVOR", 0)) > 0 or int(counts.get("EXPLORATION", 0)) > 0


def schedule_ps4(completed, ps2_counts, already_measured):
    """One bounded metric-fold row. No target, horizon, or extractor product."""
    rows = []
    bound = f"{len(PS4_METRICS)}_metrics_x_{len(INNER_FOLDS)}_inner_folds"
    for feature in sorted(completed):
        if feature in already_measured:
            continue
        counts = ps2_counts.get(feature) or {}
        if not prioritized_for_ps4(counts):
            continue
        reason = "PS3R_TERMINAL_PRIORITIZED_SURVIVOR_OR_EXPLORATION"
        for fold in INNER_FOLDS:
            for metric in PS4_METRICS:
                rows.append({
                    "feature_id": feature,
                    "fold": fold,
                    "metric": metric,
                    "state": "SCHEDULED_NOT_RUN",
                    "reason": reason,
                    "product_bound": bound,
                    "selection_decision": "NOT_ISSUED",
                })
    return rows


def _aware(text):
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ReadinessError("FOLD_TIMEZONE", text)
    return value


def weeks_inside(fold_name, train_start, train_end, val_start, val_end):
    """ISO weeks contained in one inner TRAIN prefix. Validation is not included."""
    start = _aware(train_start)
    last = _aware(train_end)
    exclusive_end = last + timedelta(hours=1)
    val0 = _aware(val_start)
    val1 = _aware(val_end) + timedelta(hours=1)
    monday = (start - timedelta(days=start.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    if monday < start:
        monday += timedelta(days=7)
    rows = []
    while monday + timedelta(days=7) <= exclusive_end:
        week_end = monday + timedelta(days=7)
        if week_end > val0 and monday < val1:
            raise ReadinessError("WEEK_INTERSECTS_VALIDATION", fold_name)
        if monday >= TRAIN_EXCLUSIVE_END or week_end > TRAIN_EXCLUSIVE_END:
            raise ReadinessError("WEEK_TOUCHES_EXTERNAL_HOLDOUT", fold_name)
        iso = monday.isocalendar()
        rows.append({
            "fold": fold_name,
            "week_id": f"{iso.year}-W{iso.week:02d}",
            "start": monday.isoformat(),
            "end": week_end.isoformat(),
        })
        monday = week_end
    if not rows:
        raise ReadinessError("WEEK_GRID_EMPTY", fold_name)
    return rows


def inner_train_weeks(folds_payload):
    rows = []
    names = []
    for fold in folds_payload["folds"]:
        names.append(fold["name"])
        rows.extend(weeks_inside(
            fold["name"],
            fold["train_time"][0],
            fold["train_time"][1],
            fold["val_time"][0],
            fold["val_time"][1],
        ))
    if tuple(names) != INNER_FOLDS:
        raise ReadinessError("FOLD_IDENTITY", ",".join(names))
    return rows


def ps5_inputs(population_sha, folds_sha, weeks):
    grid_sha = digest_bytes(canonical_json(weeks))
    body = {
        "arms": [
            {"arm_id": "predictive_baseline", "evidence": ["ps2_predictive", "redundancy"], "k": PRIMARY_K},
            {"arm_id": "plus_causal", "evidence": ["predictive_baseline", "ps3c_identified_only_not_identified_stays_neutral"], "k": PRIMARY_K},
            {"arm_id": "plus_extractibility", "evidence": ["plus_causal", "ps3r_utility_not_reconstruction"], "k": PRIMARY_K},
            {"arm_id": "random_k", "evidence": ["sealed_tape_same_k"], "k": PRIMARY_K, "tapes": 1},
            {"arm_id": "all_admissible", "evidence": ["full_retained_population"], "k_exempt": True},
        ],
        "denominator": DENOMINATOR,
        "evaluation_mode": "INNER_TRAIN_WEEKLY_FOLDS_ONLY",
        "fold_grid_sha256": grid_sha,
        "folds_json_sha256": folds_sha,
        "forbidden": [
            "train",
            "read_sealed_external_test",
            "read_holdout",
            "h_core",
            "neat",
            "rl",
            "strategy_evaluation",
        ],
        "manifesto_final": "NO_EMITIDO",
        "permission_to_train": False,
        "population_sha256": population_sha,
        "primary_k": PRIMARY_K,
        "schema": "m2_ps5_comparison_inputs.v1",
        "sealed_sensitivity_k": list(SEALED_K),
        "seed": SEED,
        "weeks": weeks,
    }
    if [item["arm_id"] for item in body["arms"]] != list(ARMS):
        raise ReadinessError("PS5_ARM", "arm identity changed")
    if body["permission_to_train"] is not False:
        raise ReadinessError("PS5_TRAIN_NOT_AUTHORIZED", "preparation is not training")
    body["preparation_identity"] = digest_bytes(canonical_json(body))
    return body


def _statuses_of(row):
    return [
        row["ps0_ps1_status"],
        row["ps2_status"],
        row["ps3c_status"],
        row["ps3r_status"],
        row["ps4_status"],
        row["ps5_status"],
    ]


def require_terminal_evidence(rows):
    for row in rows:
        for status in _statuses_of(row):
            refuse_missing_cast(status)
        terminal = [status for status in _statuses_of(row) if status in TERMINAL_STATUSES]
        if terminal and (len(row.get("evidence_digest") or "") != 64 or not row.get("evidence_locator")):
            raise ReadinessError("TERMINAL_WITHOUT_EVIDENCE", row["feature_id"])
        if row["ps3c_status"] == "NOT_IDENTIFIED" and row["ps3c_status"] in ZERO_OR_REJECTED:
            raise ReadinessError("FALSE_CAUSAL_REJECTION", row["feature_id"])


def _load_ps2(root):
    counts = {}
    for batch in ("batch_001", "batch_002", "batch_003"):
        for row in read_csv(root / BASE / "laneB" / batch / "ps2_status.csv"):
            bucket = counts.setdefault(row["feature"], Counter())
            bucket[row["status"]] += 1
    return counts


def _load_ps1(root):
    states = {}
    batches = {}
    digests = {}
    for batch in ("batch_001", "batch_002", "batch_003"):
        path = root / BASE / "laneA" / batch / "profile_cells.csv"
        digests[batch] = digest_path(path)
        for row in read_csv(path):
            states.setdefault(row["feature_id"], set()).add(row["state"])
            batches.setdefault(row["feature_id"], batch)
    return states, batches, digests


def _profile_acceptance(root):
    profile_dir = root / BASE / "ps4_transform_profile"
    paths = {
        "profile_rows.jsonl": profile_dir / "profile_rows.jsonl",
        "REPORT.json": profile_dir / "REPORT.json",
        "input_digests.json": profile_dir / "input_digests.json",
    }
    digests = {name: digest_path(path) for name, path in paths.items()}
    report = read_json(paths["REPORT.json"])
    claimed_inputs = read_json(paths["input_digests.json"])
    if report.get("schema") != PROFILE_SCHEMA:
        raise ReadinessError("PROFILE_ACCEPTANCE", report.get("schema", ""))
    if report.get("denominator_feature_fold_units") != PROFILE_UNITS or report.get("metric_rows") != PROFILE_ROWS:
        raise ReadinessError("PROFILE_ACCEPTANCE", "units or metric rows")
    if tuple(report.get("features") or []) != MEASURED_FEATURES:
        raise ReadinessError("PROFILE_ACCEPTANCE", "feature list")
    if tuple(report.get("folds") or []) != INNER_FOLDS:
        raise ReadinessError("PROFILE_ACCEPTANCE", "folds")
    rows = []
    per_feature = {feature: [] for feature in MEASURED_FEATURES}
    profiler = set()
    with paths["profile_rows.jsonl"].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(item)
            feature = item.get("feature_id")
            if feature not in per_feature:
                raise ReadinessError("PROFILE_ACCEPTANCE", f"unexpected {feature}")
            per_feature[feature].append(line)
            profiler.add(item.get("profiler_code_digest"))
            if item.get("source_digests") != claimed_inputs.get("sha256"):
                raise ReadinessError("STALE_PS4_DIGEST", "source_digests")
    if len(rows) != PROFILE_ROWS:
        raise ReadinessError("PROFILE_ACCEPTANCE", str(len(rows)))
    units = {(item["feature_id"], item["fold"]) for item in rows}
    if len(units) != PROFILE_UNITS:
        raise ReadinessError("PROFILE_ACCEPTANCE", str(len(units)))
    metrics = {item["metric"] for item in rows}
    if tuple(sorted(metrics)) != PS4_METRICS:
        raise ReadinessError("PROFILE_ACCEPTANCE", "metric set")
    if any(item.get("status") != "COMPLETED" for item in rows):
        raise ReadinessError("PROFILE_ACCEPTANCE", "status")
    code_digest = digest_path(root / "tools" / "df_profile_information.py")
    if profiler != {code_digest}:
        raise ReadinessError("STALE_PS4_DIGEST", "profiler_code_digest")
    recomputed = {}
    for name, relative in LOCAL_PROFILE_INPUTS.items():
        recomputed[name] = require_fresh_digest(root / relative, claimed_inputs["sha256"][name])
    parquet = claimed_inputs["sha256"].get("features_train.parquet")
    if not parquet or len(parquet) != 64:
        raise ReadinessError("STALE_PS4_DIGEST", "features_train.parquet")
    feature_digests = {
        feature: digest_bytes("".join(per_feature[feature]).encode("utf-8"))
        for feature in MEASURED_FEATURES
    }
    return {
        "file_digests": digests,
        "inputs_recomputed": recomputed,
        "parquet_retained_not_in_worktree": parquet,
        "profiler_code_digest": code_digest,
        "feature_digests": feature_digests,
        "claimed_inputs": claimed_inputs["sha256"],
    }


def _lane_e(root):
    matrix_path = root / BASE / "laneE" / "extractibility_matrix.csv"
    cost_path = root / BASE / "laneE" / "feature_cost.csv"
    grouped = {}
    with matrix_path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if "feature" not in header:
            raise ReadinessError("LANE_E_EVIDENCE", "matrix header")
        for line in handle:
            feature = line.split(",", 2)[1]
            grouped.setdefault(feature, []).append(line)
    cost = {row["feature"]: row for row in read_csv(cost_path)}
    done = sorted(feature for feature, row in cost.items() if row.get("status") == "DONE")
    digests = {
        feature: digest_bytes("".join(grouped.get(feature, [])).encode("utf-8"))
        for feature in cost
    }
    return {
        "cost": cost,
        "done": done,
        "digests": digests,
        "matrix_sha256": digest_path(matrix_path),
        "cost_sha256": digest_path(cost_path),
        "matrix_locator": str(BASE / "laneE" / "extractibility_matrix.csv"),
        "cost_locator": str(BASE / "laneE" / "feature_cost.csv"),
    }


def _next_action(feature, ps3c, ps3r, ps4, ps1):
    if ps3r == "ACCEPTED_PS3R_CELL_MIXED_UTILITY":
        return "Conservar la medición PS3-R de utilidad mixta. La reconstrucción no selecciona. El perfil PS4 queda solo agendado."
    if ps3r == "ACCEPTED_PS3R_TERMINAL_NOT_SELECTION":
        return "Conservar la medición PS3-R terminal. La reconstrucción no selecciona."
    if feature == SIAMESE_FEATURE:
        return (
            "Conservar la medición lane E de identity, random, ae y dae. "
            "No promover past_to_current_siamese: su results.jsonl fue sobrescrito por un duplicado posterior."
        )
    if ps4 == "MEASURED_SUBPOPULATION":
        return "Perfil PS4 medido solo para esta transformada emitida. No es selección ni cierra PS4 de las 366."
    if ps4 == "SCHEDULED_NOT_MEASURED":
        return "Ejecutar solo las 21 métricas de información y compresión en los cinco pliegues inner TRAIN. No entrenar ni seleccionar."
    if ps1 == "ADMITTED_PS1_CELL_FAILED":
        return "Hay celdas PS1 fallidas. No eliminar la candidata ni tratar el fallo como rechazo."
    if ps3c == "NOT_IDENTIFIED":
        return "NOT_IDENTIFIED en PS3-C es neutral. No convertir la ausencia en cero ni en rechazo."
    if ps3c == "NOT_APPLICABLE":
        return "El condicionamiento de calendario no es un rechazo causal ni cobertura cero."
    if ps3r == "NOT_SCHEDULED":
        return "PS3-R no está programado para esta candidata. La fila permanece en el denominador 366."
    if ps3r == "NOT_TERMINAL":
        return "PS3-R no es terminal. No fabricar utilidad ni agendar PS4."
    return "Falta la medición de la etapa abierta. No emitir un manifiesto de selección."


def _evidence(feature, ps4, ps3r, batch, profile, lane, ps1_digests, adoptions):
    if ps4 == "MEASURED_SUBPOPULATION":
        return (
            str(BASE / "ps4_transform_profile" / "profile_rows.jsonl") + "#" + feature,
            profile["feature_digests"][feature],
            "recomputed_profile_rows",
        )
    if ps3r in {"ACCEPTED_PS3R_CELL_MIXED_UTILITY", "ACCEPTED_PS3R_TERMINAL_NOT_SELECTION"}:
        adopted = adoptions[feature]
        return adopted["locator"], adopted["results_sha256"], "recomputed_ps3r_results_sha256"
    if ps3r == "LANE_E_MEASURED_NOT_SELECTION":
        return (
            lane["matrix_locator"] + "#" + feature,
            lane["digests"][feature],
            "recomputed_lane_e_matrix_rows",
        )
    if ps3r == "RUNNING_NOT_TERMINAL":
        return (
            lane["cost_locator"] + "#" + feature,
            lane["digests"][feature],
            "non_terminal_cost_row",
        )
    locator = str(BASE / "laneA" / batch / "profile_cells.csv") + "#" + feature
    return locator, ps1_digests[batch], "recomputed_profile_cells"


def build(root):
    root = Path(root)
    reconciliation = read_json(root / BASE / "coverage_reconciliation" / "coverage_reconciliation.json")
    features = reconciliation["features"]
    expected_ids = [row["feature_id"] for row in features]
    joined = sum(1 for row in features if row["in_ps3c_join"])
    outside = sum(1 for row in features if not row["in_ps3c_join"])
    if len(expected_ids) != DENOMINATOR or joined != JOIN_COUNT or outside != OUTSIDE_COUNT:
        raise ReadinessError("DENOMINATOR", f"{len(expected_ids)}={joined}+{outside}")
    audit = (root / BASE / "RETSU_PS4_PS3R_AUDIT_2026_10_03.md").read_text(encoding="utf-8")
    causal_index, causal_hashes = load_causal_joins(root)
    flagged = {row["feature_id"] for row in features if row.get("in_ps3c_join")}
    if flagged != set(causal_index):
        raise ReadinessError("PS3C_SOURCE", "join membership")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ps3r_manifest_ingestor as ingestor
    discovered = ingestor.discover(root, Path(__file__).resolve().with_name("ps3r_ingest_config.json"))
    if any(item["reason"] == "CONTRADICTORY_TERMINAL" for item in discovered):
        raise ReadinessError("PS3R_INGEST", "CONTRADICTORY_TERMINAL")
    baseline = {
        item["feature_id"]: item
        for item in discovered
        if item["role"] == "baseline" and item["disposition"] == "ADOPTED"
    }
    alternative = [
        item for item in discovered
        if item["role"] == "alternative" and item["disposition"] == "ADOPTED"
    ]
    if VIX_FEATURE not in baseline or baseline[VIX_FEATURE]["results_sha256"] not in audit:
        raise ReadinessError("VIX_EVIDENCE", "accepted results digest absent from the audit")
    require_reconstruction_not_selection({
        "feature_id": VIX_FEATURE,
        "selection_decision": "NOT_ISSUED",
        "reconstruction_selects": False,
        "utility": "mixed",
    })
    require_siamese_not_promoted({
        "feature_id": SIAMESE_FEATURE,
        "family": SIAMESE_FAMILY,
        "results_overwritten": True,
        "promote": False,
        "selection_decision": "NOT_ISSUED",
    })
    profile = _profile_acceptance(root)
    ps2 = _load_ps2(root)
    ps1_states, ps1_batch, ps1_digests = _load_ps1(root)
    lane = _lane_e(root)
    if VIX_FEATURE in lane["done"]:
        raise ReadinessError("VIX_EVIDENCE", "lane E matrix is not the accepted results file")
    completed = set(lane["done"]) | set(baseline)
    schedule = schedule_ps4(completed, ps2, set(MEASURED_FEATURES))
    scheduled_features = {row["feature_id"] for row in schedule}
    rows = []
    for feature in features:
        feature_id = feature["feature_id"]
        if feature_id not in ps1_states or feature_id not in ps2:
            raise ReadinessError("MISSING_CANDIDATE", feature_id)
        ps1 = ps1_status(ps1_states[feature_id])
        if feature_id in causal_index:
            causal = "NOT_IDENTIFIED"
            producer_revision = causal_index[feature_id]["producer_revision"]
            join_sha = causal_index[feature_id]["join_sha256"]
        else:
            causal = "OUTSIDE_JOIN_PENDING"
            producer_revision = ""
            join_sha = ""
        if feature_id in baseline:
            ps3r = baseline[feature_id]["ps3r_status"]
        elif feature_id in lane["done"]:
            ps3r = "LANE_E_MEASURED_NOT_SELECTION"
        elif feature.get("in_extractibility_queue"):
            ps3r = "NOT_TERMINAL"
        else:
            ps3r = "NOT_SCHEDULED"
        if feature_id in MEASURED_FEATURES:
            ps4 = "MEASURED_SUBPOPULATION"
        elif feature_id in scheduled_features:
            ps4 = "SCHEDULED_NOT_MEASURED"
        else:
            ps4 = "PENDING_PROFILE"
        locator, evidence_digest, kind = _evidence(
            feature_id, ps4, ps3r, ps1_batch[feature_id], profile, lane, ps1_digests, baseline,
        )
        if ps4 == "MEASURED_SUBPOPULATION" and evidence_digest != profile["feature_digests"][feature_id]:
            raise ReadinessError("STALE_PS4_DIGEST", feature_id)
        row = {
            "feature_id": feature_id,
            "denominator": str(DENOMINATOR),
            "ps0_ps1_status": ps1,
            "ps2_status": ps2_status(ps2[feature_id]),
            "ps3c_status": causal,
            "ps3c_producer_revision": producer_revision,
            "ps3c_join_sha256": join_sha,
            "ps3r_status": ps3r,
            "ps4_status": ps4,
            "ps5_status": (
                "EVIDENCE_PRESENT_NOT_TRAINED"
                if ps3r in {
                    "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
                    "ACCEPTED_PS3R_TERMINAL_NOT_SELECTION",
                }
                and ps4 == "MEASURED_SUBPOPULATION"
                else "NOT_READY_EVIDENCE_INCOMPLETE"
            ),
            "evidence_locator": locator,
            "evidence_digest": evidence_digest,
            "evidence_kind": kind,
            "missing_next_action": _next_action(feature_id, causal, ps3r, ps4, ps1),
        }
        rows.append(row)
    require_population(rows, expected_ids)
    for index, row in enumerate(rows):
        row["row_index"] = str(index)
    require_terminal_evidence(rows)
    if sum(row["ps4_status"] == "MEASURED_SUBPOPULATION" for row in rows) != len(MEASURED_FEATURES):
        raise ReadinessError("PROFILE_ACCEPTANCE", "measured count")
    historical = read_csv(root / BASE / "source_transform_coverage" / "transform_coverage.csv")
    if len(historical) != 9 or any(row["ps1_ps4_profile_state"] != "PENDING_PROFILE" for row in historical):
        raise ReadinessError("HISTORICAL_LEDGER", "variant rows are not the ten emitted features")
    integration = []
    for feature in MEASURED_FEATURES:
        integration.append({
            "feature_id": feature,
            "prior_status": "PENDING_PROFILE",
            "profile_status": "MEASURED",
            "schema": PROFILE_SCHEMA,
            "feature_fold_units": "5",
            "metric_rows": "105",
            "profile_rows_sha256": profile["file_digests"]["profile_rows.jsonl"],
            "report_sha256": profile["file_digests"]["REPORT.json"],
            "input_digests_sha256": profile["file_digests"]["input_digests.json"],
            "profiler_code_digest": profile["profiler_code_digest"],
        })
    folds_path = root / LOCAL_PROFILE_INPUTS["folds.json"]
    folds_payload = read_json(folds_path)
    weeks = inner_train_weeks(folds_payload)
    population_sha = digest_bytes(("\n".join(expected_ids) + "\n").encode("utf-8"))
    comparison = ps5_inputs(population_sha, digest_path(folds_path), weeks)
    by_ps4 = dict(Counter(row["ps4_status"] for row in rows))
    by_ps3r = dict(Counter(row["ps3r_status"] for row in rows))
    by_ps3c = dict(Counter(row["ps3c_status"] for row in rows))
    report = {
        "schema": "m2_readiness_ledger.v1",
        "cuerpo": (
            "El denominador es 366, la partición retenida 279+87, no el inventario de 388 fuentes. "
            "Solo diez transformadas emitidas pasan de PENDING_PROFILE a MEASURED. "
            "Eso no cierra PS4 ni selecciona ninguna candidata. "
            "NOT_IDENTIFIED en PS3-C permanece neutral. "
            "La preparación PS5 no autoriza entrenamiento, lectura de test externo ni evaluación de estrategia."
        ),
        "denominator": DENOMINATOR,
        "partition": {"ps3c_join": JOIN_COUNT, "low_priority_outside_join": OUTSIDE_COUNT},
        "selection_manifest": "NOT_ISSUED",
        "ps4_complete_for_366": False,
        "measured_transforms": list(MEASURED_FEATURES),
        "measured_profile": {
            "schema": PROFILE_SCHEMA,
            "feature_fold_units": PROFILE_UNITS,
            "metric_rows": PROFILE_ROWS,
            "file_digests": profile["file_digests"],
            "inputs_recomputed": profile["inputs_recomputed"],
            "parquet_retained_not_in_worktree": profile["parquet_retained_not_in_worktree"],
            "profiler_code_digest": profile["profiler_code_digest"],
            "feature_row_digests": profile["feature_digests"],
            "not_selection_over_366": True,
        },
        "ps3r": {
            "lane_e_measured_not_selection": lane["done"],
            "lane_e_matrix_sha256": lane["matrix_sha256"],
            "lane_e_cost_sha256": lane["cost_sha256"],
            "vix_feature": VIX_FEATURE,
            "vix_results_sha256": VIX_RESULTS_SHA256,
            "vix_utility": "mixed",
            "reconstruction_is_selection": False,
            "dgs30_feature": DGS30_FEATURE,
            "dgs30_results_sha256": DGS30_RESULTS_SHA256,
            "dgs30_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "dgs30_utility": "mixed",
            "dprime_feature": DPRIME_FEATURE,
            "dprime_results_sha256": DPRIME_RESULTS_SHA256,
            "dprime_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "aud_ewma_feature": AUD_EWMA_FEATURE,
            "aud_ewma_results_sha256": AUD_EWMA_RESULTS_SHA256,
            "aud_ewma_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "aud_ewma_utility": "mixed",
            "aud_logret_1h_feature": AUD_LOGRET1H_FEATURE,
            "aud_logret_1h_results_sha256": AUD_LOGRET1H_RESULTS_SHA256,
            "aud_logret_1h_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "aud_logret_1h_utility": "mixed",
            "aud_logret_24h_feature": AUD_LOGRET24H_FEATURE,
            "aud_logret_24h_results_sha256": AUD_LOGRET24H_RESULTS_SHA256,
            "aud_logret_24h_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "aud_logret_24h_utility": "mixed",
            "eurgbp_ewma_vol_24_feature": EURGBP_EWMA_FEATURE,
            "eurgbp_ewma_vol_24_results_sha256": EURGBP_EWMA_RESULTS_SHA256,
            "eurgbp_ewma_vol_24_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "eurgbp_ewma_vol_24_utility": "mixed",
            "eurgbp_logret_1h_feature": EURGBP_LOGRET1H_FEATURE,
            "eurgbp_logret_1h_results_sha256": EURGBP_LOGRET1H_RESULTS_SHA256,
            "eurgbp_logret_1h_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "eurgbp_logret_1h_utility": "mixed",
            "eurjpy_ewma_vol_24_feature": EURJPY_EWMA_FEATURE,
            "eurjpy_ewma_vol_24_results_sha256": EURJPY_EWMA_RESULTS_SHA256,
            "eurjpy_ewma_vol_24_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "eurjpy_ewma_vol_24_utility": "mixed",
            "eurjpy_logret_1h_feature": EURJPY_LOGRET1H_FEATURE,
            "eurjpy_logret_1h_results_sha256": EURJPY_LOGRET1H_RESULTS_SHA256,
            "eurjpy_logret_1h_status": "ACCEPTED_PS3R_CELL_MIXED_UTILITY",
            "eurjpy_logret_1h_utility": "mixed",
            "dprime_utility": "mixed",
            "siamese_feature": SIAMESE_FEATURE,
            "siamese_family": SIAMESE_FAMILY,
            "siamese_promotion": "NOT_PROMOTED_RESULTS_OVERWRITTEN",
        },
        "ps4_schedule": {
            "features": sorted(scheduled_features),
            "rows": len(schedule),
            "metrics": list(PS4_METRICS),
            "folds": list(INNER_FOLDS),
            "product_bound": f"{len(PS4_METRICS)}_metrics_x_{len(INNER_FOLDS)}_inner_folds",
        },
        "ps5_preparation_identity": comparison["preparation_identity"],
        "status_counts": {"ps3c": by_ps3c, "ps3r": by_ps3r, "ps4": by_ps4},
        "ps3c_source": {
            "producer_revision": "48ae17c",
            "join_sha256": causal_hashes,
            "identified": by_ps3c.get("IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS", 0),
            "not_identified": by_ps3c.get("NOT_IDENTIFIED", 0),
            "outside_join_pending": by_ps3c.get("OUTSIDE_JOIN_PENDING", 0),
            "not_identified_is_not_rejected": True,
        },
        "ps5_design_status": "PREPARED_NOT_TRAINED",
        "alternative_family_evidence": [
            {
                "feature_id": item["feature_id"],
                "results_sha256": item["results_sha256"],
                "utility": item["utility"],
                "replaces_baseline": False,
            }
            for item in alternative
        ],
        "audit_sha256": digest_path(root / BASE / "RETSU_PS4_PS3R_AUDIT_2026_10_03.md"),
        "population_sha256": population_sha,
        "huecos": [
            "PS4 medido cubre 10 de 366. El resto sigue PENDING_PROFILE o solo agendado.",
            "PS3-C vigente: 0 identificadas de 279. NOT_IDENTIFIED no es rechazo. 87 quedan fuera del join.",
            "Las celdas baseline adoptadas por manifiesto tienen utilidad medida y no están seleccionadas.",
            "px.logret_6h past_to_current_siamese no se promueve.",
            "features_train.parquet no está en este árbol; se conserva el digest retenido sin recomputarlo.",
            "El diseño global PS5 está preparado y no entrenado. Cada candidata sin PS3-R y PS4 requeridos queda NOT_READY_EVIDENCE_INCOMPLETE.",
            "Las nueve filas históricas de variantes siguen en PENDING_PROFILE.",
        ],
    }
    return {
        "rows": rows,
        "schedule": schedule,
        "integration": integration,
        "comparison": comparison,
        "report": report,
    }


def render_csv(rows, fields):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue().encode("utf-8")


def render(built):
    report = dict(built["report"])
    report["readiness_rows_sha256"] = digest_bytes(render_csv(built["rows"], READINESS_FIELDS))
    report["ps4_schedule_sha256"] = digest_bytes(render_csv(built["schedule"], SCHEDULE_FIELDS))
    report["ps4_integration_sha256"] = digest_bytes(render_csv(built["integration"], INTEGRATION_FIELDS))
    comparison = json.dumps(built["comparison"], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report["ps5_inputs_sha256"] = digest_bytes(comparison.encode("utf-8"))
    return {
        "readiness_rows.csv": render_csv(built["rows"], READINESS_FIELDS),
        "ps4_schedule.csv": render_csv(built["schedule"], SCHEDULE_FIELDS),
        "ps5_comparison_inputs.json": comparison.encode("utf-8"),
        "REPORT.json": (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "ps4_emitted_profile_status.csv": render_csv(built["integration"], INTEGRATION_FIELDS),
    }


def emit(root):
    root = Path(root)
    built = build(root)
    rendered = render(built)
    out = root / BASE / "m2_readiness"
    out.mkdir(parents=True, exist_ok=True)
    coverage = root / BASE / "source_transform_coverage"
    for name, payload in rendered.items():
        target = coverage / name if name == "ps4_emitted_profile_status.csv" else out / name
        target.write_bytes(payload)
    return built["report"]


def main():
    report = emit(Path(__file__).resolve().parents[5])
    print(json.dumps({
        "denominator": report["denominator"],
        "ps5_preparation_identity": report["ps5_preparation_identity"],
        "measured_transforms": report["measured_transforms"],
        "ps4_complete_for_366": report["ps4_complete_for_366"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
