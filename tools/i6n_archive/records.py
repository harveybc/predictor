"""Archive documents for I6-N candidates and their projection into governed reports.

The round payload (``i6n.archive.round.v1``) is what goes into the DOIN external body: ONE
candidate (every candidate, winners or not, failed ones with ``fitness=None``) plus A's
per-fold ``candidate_receipt`` documents unchanged. The projection is computed from the
verified payload only, so the warehouse never receives a number that is not in the bytes
the chain committed to.

Identity definitions are lane E's (INTERFACE_E E.1):
``effective_config_sha256 = sha256(json.dumps(config, sort_keys=True, separators=(',',':'),
allow_nan=False).encode('utf-8'))`` and ``search_candidate_sha256 = sha256(canonical
{plan_sha256, horizon_hours, genotype_canonical_sha256})``. Receipt rows follow lane A's
``i6n.candidate_receipt.v1`` (INTERFACE_A A.2: role = C1 MANAGEMENT|ENTRY, additive c2_role,
support_sha256, outer-week VALIDATION rows) and its row identity keys.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

CANDIDATE_SCHEMA = "i6n.archive.candidate.v1"
DECISION_SCHEMA = "i6n.archive.decision.v1"
ISLAND_CANDIDATE_SCHEMA = "i6n.archive.island_candidate.v1"
OUTCOMES = ("WINNER", "FINALIST", "CONTROL", "NOT_SELECTED")
ROUND_SCHEMA = "i6n.archive.round.v1"
RECEIPT_SCHEMA = "i6n.candidate_receipt.v1"
DOIN_METADATA_SCHEMA = "i6n.doin_projection.v1"
SEED = 6102026
DISPOSITIONS = ("COMPLETE", "ADOPTED_CACHE", "FAILED", "INFEASIBLE", "INTERRUPTED", "NOT_ADMITTED")
SCORED_DISPOSITIONS = ("COMPLETE", "ADOPTED_CACHE")
SPLITS = ("TRAIN", "INNER", "VALIDATION")
ROLES = ("MANAGEMENT", "ENTRY")  # INTERFACE_A A.2: C1 canonical; the C2 name is rows.c2_role
ROW_IDENTITY = ("experiment", "candidate_identity_sha256", "fold", "horizon_hours", "head_offset_hours",
                "role", "population", "split", "metric", "unit", "schema_version")
_SUMS = ("sum_abs_error", "sum_sq_error", "naive_sum_abs_error", "naive_sum_sq_error")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,96}$")
MAX_ATTEMPTS_PER_FOLD = 3
_ISLAND_SPLITS = ("train", "inner")


class ArchiveRecordError(ValueError):
    """A typed refusal of a candidate or receipt document; nothing is written."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def e_config_sha256(config: Any) -> str:
    """Lane E's effective_config_sha256 (canonical JSON, UTF-8)."""
    return hashlib.sha256(canonical(config)).hexdigest()


def search_candidate_sha256(plan_sha256: str, horizon_hours: int, genotype_canonical_sha256: str) -> str:
    return hashlib.sha256(canonical({"plan_sha256": plan_sha256, "horizon_hours": horizon_hours,
                                     "genotype_canonical_sha256": genotype_canonical_sha256})).hexdigest()


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _island_split_mapping(value: Any, name: str, *, kind: str) -> dict:
    if not isinstance(value, Mapping) or set(value) != set(_ISLAND_SPLITS):
        _fail(f"ISLAND_{name.upper()} must contain exactly train and inner")
    result = dict(value)
    for split in _ISLAND_SPLITS:
        item = result[split]
        if kind == "count":
            if isinstance(item, bool) or not isinstance(item, int) or item < 1:
                _fail(f"ISLAND_{name.upper()}.{split} must be an integer >= 1")
        elif kind == "positive":
            _finite(item, f"{name}.{split}", nullable=False)
            if item <= 0:
                _fail(f"ISLAND_{name.upper()}.{split} must be positive")
        elif kind == "nonnegative":
            _finite(item, f"{name}.{split}", nullable=False)
            if item < 0:
                _fail(f"ISLAND_{name.upper()}.{split} must be nonnegative")
        elif kind == "digest":
            if not isinstance(item, str) or not _HEX64.fullmatch(item):
                _fail(f"ISLAND_{name.upper()}.{split} must be a lowercase sha256")
        else:  # pragma: no cover - internal caller contract
            raise AssertionError(f"unsupported island evidence kind: {kind}")
    return result


def _island_origins(metrics: Mapping, folds: Mapping) -> tuple[str, dict[str, dict] | None]:
    binding = metrics.get("origins_binding")
    configured = {}
    for fold, fold_config in folds.items():
        if not isinstance(fold_config, Mapping):
            _fail("ISLAND_FOLD_BINDING")
        configured[fold] = fold_config.get("i6n_origins_sha256")
    if any(value is not None for value in configured.values()):
        if any(value is None for value in configured.values()):
            _fail("ISLAND_ORIGINS_CONFIG_PARTIAL")
        if binding != "VERIFIED":
            _fail("ISLAND_ORIGINS_BINDING_NOT_VERIFIED")
        supplied = metrics.get("origins_sha256")
        if not isinstance(supplied, Mapping) or set(supplied) != set(folds):
            _fail("ISLAND_ORIGINS_FOLD_BINDING")
        result = {}
        for fold in folds:
            expected = _island_split_mapping(
                configured[fold], f"configured_origins.{fold}", kind="digest"
            )
            observed = _island_split_mapping(
                supplied[fold], f"origins_sha256.{fold}", kind="digest"
            )
            if observed != expected:
                _fail("ISLAND_ORIGINS_CONFIG_MISMATCH")
            result[fold] = observed
        return binding, result

    if binding != "NOT_CHECKED" or metrics.get("origins_sha256") is not None:
        _fail("ISLAND_ORIGINS_UNDECLARED")
    return binding, None


def _island_fold_evidence(
    *, fold: str, ratio: Any, receipt: Any, binding: Mapping, genotype_sha: str,
    horizon: int, origins_binding: str, origins_sha256: Mapping | None,
) -> dict:
    if not isinstance(binding, Mapping):
        _fail("ISLAND_FOLD_BINDING")
    support = binding.get("i6n_support")
    context = binding.get("i6n_evaluation_context")
    if (not isinstance(support, Mapping) or set(support) != {
            "counts", "naive_sums", "support_digest"} or not isinstance(context, Mapping)):
        _fail("ISLAND_FOLD_PROVENANCE")
    expected_counts = _island_split_mapping(
        support["counts"], f"configured_counts.{fold}", kind="count"
    )
    expected_naive = _island_split_mapping(
        support["naive_sums"], f"configured_naive_sums.{fold}", kind="positive"
    )
    expected_support = _island_split_mapping(
        support["support_digest"], f"configured_support_digest.{fold}", kind="digest"
    )

    required = {"status", "counts", "error_sums", "naive_sums", "support_digest", "metadata"}
    if (not isinstance(receipt, Mapping) or not required <= set(receipt)
            or set(receipt) - required - {"fitness"}):
        _fail("ISLAND_RECEIPT_SCHEMA")
    if receipt.get("status") != "ok":
        _fail("ISLAND_RECEIPT_STATUS")
    counts = _island_split_mapping(receipt.get("counts"), f"counts.{fold}", kind="count")
    error_sums = _island_split_mapping(
        receipt.get("error_sums"), f"error_sums.{fold}", kind="nonnegative"
    )
    naive_sums = _island_split_mapping(
        receipt.get("naive_sums"), f"naive_sums.{fold}", kind="positive"
    )
    support_digest = _island_split_mapping(
        receipt.get("support_digest"), f"support_digest.{fold}", kind="digest"
    )
    if (counts != expected_counts or naive_sums != expected_naive
            or support_digest != expected_support):
        _fail("ISLAND_RECEIPT_SUPPORT_MISMATCH")

    metadata = receipt.get("metadata")
    candidate_receipt = metadata.get("candidate_receipt") if isinstance(metadata, Mapping) else None
    if not isinstance(candidate_receipt, Mapping):
        _fail("ISLAND_RECEIPT_BINDING_MISSING")
    if candidate_receipt.get("genotype_canonical_sha256") != genotype_sha:
        _fail("ISLAND_RECEIPT_CANDIDATE_MISMATCH")
    if candidate_receipt.get("fold") != fold:
        _fail("ISLAND_RECEIPT_FOLD_MISMATCH")
    if candidate_receipt.get("horizon_hours") != horizon:
        _fail("ISLAND_RECEIPT_HORIZON_MISMATCH")

    origin_digests = dict(origins_sha256) if origins_sha256 is not None else None
    if origins_binding == "VERIFIED":
        rows = candidate_receipt.get("rows")
        if not isinstance(rows, list):
            _fail("ISLAND_RECEIPT_ENDPOINT_ROW_MISSING")
        for split in _ISLAND_SPLITS:
            endpoint = [
                row for row in rows
                if isinstance(row, Mapping)
                and row.get("split") == split.upper()
                and row.get("head_offset_hours") == horizon
                and row.get("head_role") == "ENDPOINT"
                and row.get("comparison") is True
            ]
            if len(endpoint) != 1:
                _fail("ISLAND_RECEIPT_ENDPOINT_ROW_MISSING")
            if endpoint[0].get("support_sha256") != origin_digests[split]:
                _fail("ISLAND_RECEIPT_ORIGINS_MISMATCH")

    _finite(ratio, f"per_fold.{fold}", nullable=False)
    recomputed = math.fsum(
        error_sums[split] / naive_sums[split] for split in _ISLAND_SPLITS
    ) / len(_ISLAND_SPLITS)
    if not math.isclose(float(ratio), recomputed, rel_tol=1e-12, abs_tol=1e-12):
        _fail("ISLAND_FOLD_RATIO_EVIDENCE")
    if "fitness" in receipt:
        _finite(receipt["fitness"], f"receipt_fitness.{fold}", nullable=False)
        if not math.isclose(float(receipt["fitness"]), recomputed,
                            rel_tol=1e-12, abs_tol=1e-12):
            _fail("ISLAND_RECEIPT_FITNESS_EVIDENCE")

    retained_receipt = json.loads(canonical(dict(receipt)))
    receipt_sha = _sha256(retained_receipt)
    receipt_identity_sha = _sha256(candidate_receipt)
    origin_identity = {
        "origins_binding": origins_binding,
        "origins_sha256": origin_digests,
    }
    support_sha = _sha256(support)
    context_sha = _sha256(context)
    origin_identity_sha = _sha256(origin_identity)
    fold_identity = _sha256({
        "fold": fold,
        "support_sha256": support_sha,
        "evaluation_context_sha256": context_sha,
        "receipt_sha256": receipt_sha,
        "origin_identity_sha256": origin_identity_sha,
    })
    return {
        "fold": fold,
        "ratio_to_naive": float(recomputed),
        "relative_error_vs_naive": float(recomputed) - 1.0,
        "counts": counts,
        "error_sums": error_sums,
        "naive_sums": naive_sums,
        "support_digest": support_digest,
        "receipt": retained_receipt,
        "receipt_sha256": receipt_sha,
        "receipt_identity_sha256": receipt_identity_sha,
        "origins_binding": origins_binding,
        "origins_sha256": origin_digests,
        "origin_identity_sha256": origin_identity_sha,
        "support_sha256": support_sha,
        "evaluation_context_sha256": context_sha,
        "fold_identity_sha256": fold_identity,
    }


def island_candidate_payload(
    domain_id: str,
    candidate_info: Mapping,
    optimization_config: Mapping,
) -> dict:
    """Build the external-only record emitted by the I6N island optimizer.

    The callback exposes the effective config plus per-fold receipts and
    ratios. This adapter verifies and retains those receipts, binds them to
    the sealed support/origins, and derives every archived score from them.
    """
    if not isinstance(domain_id, str) or not _KEY.match(domain_id):
        _fail("ISLAND_DOMAIN_ID")
    if not isinstance(candidate_info, Mapping) or not isinstance(optimization_config, Mapping):
        _fail("ISLAND_CANDIDATE_INPUT")
    if optimization_config.get("i6n_island_mode") is not True:
        _fail("ISLAND_MODE_REQUIRED")

    horizon = optimization_config.get("i6n_horizon")
    if horizon not in (4, 72):
        _fail("ISLAND_HORIZON must be 4 or 72")
    plan = optimization_config.get("i6n_plan")
    if not isinstance(plan, Mapping):
        _fail("ISLAND_PLAN_REQUIRED")
    plan_sha = plan.get("plan_sha256")
    if not isinstance(plan_sha, str) or not _HEX64.match(plan_sha):
        _fail("ISLAND_PLAN_SHA256")

    effective = candidate_info.get("parameters")
    if not isinstance(effective, Mapping) or effective.get("horizon") != horizon:
        _fail("ISLAND_EFFECTIVE_CONFIG")
    effective = dict(effective)
    effective_sha = e_config_sha256(effective)
    genotype_sha = _sha256({"schema": "i6n-search-v1", "config": effective})
    search_sha = search_candidate_sha256(plan_sha, horizon, genotype_sha)
    contract_sha = candidate_info.get("contract_sha256")
    checkpoint_sha = candidate_info.get("checkpoint_identity_sha256")
    callback_identities = {
        "genotype_canonical_sha256": genotype_sha,
        "effective_config_sha256": effective_sha,
        "plan_sha256": plan_sha,
    }
    for name, expected_identity in callback_identities.items():
        if candidate_info.get(name) != expected_identity:
            _fail(f"ISLAND_{name.upper()}_MISMATCH")
    for name, value in (("contract_sha256", contract_sha),
                        ("checkpoint_identity_sha256", checkpoint_sha)):
        if not isinstance(value, str) or not _HEX64.match(value):
            _fail(f"ISLAND_{name.upper()}")

    seed = optimization_config.get("seed")
    training_seed = candidate_info.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or training_seed != seed:
        _fail("ISLAND_SEED_MISMATCH")
    island_id = candidate_info.get("island_id")
    if not isinstance(island_id, str) or not _KEY.match(island_id):
        _fail("ISLAND_ID")
    generation = candidate_info.get("generation")
    population_index = candidate_info.get("candidate_in_gen")
    total_evaluations = candidate_info.get("total_eval")
    for name, value in (("generation", generation), ("population_index", population_index),
                        ("total_evaluations", total_evaluations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"ISLAND_{name.upper()}")

    metrics = candidate_info.get("metrics")
    folds = optimization_config.get("i6n_folds")
    if not isinstance(metrics, Mapping) or not isinstance(folds, Mapping) or len(folds) != 4:
        _fail("ISLAND_FOUR_FOLDS_REQUIRED")
    ratios = metrics.get("per_fold")
    receipts = metrics.get("receipts")
    if (not isinstance(ratios, Mapping) or set(ratios) != set(folds)
            or not isinstance(receipts, Mapping) or set(receipts) != set(folds)):
        _fail("ISLAND_FOLD_RATIO_BINDING")
    origins_binding, origins = _island_origins(metrics, folds)
    metric_identities = dict(callback_identities, contract_sha256=contract_sha,
                             checkpoint_identity_sha256=checkpoint_sha)
    for name, expected_identity in metric_identities.items():
        if metrics.get(name) != expected_identity:
            _fail(f"ISLAND_METRIC_{name.upper()}_MISMATCH")

    per_fold = []
    for fold in sorted(folds):
        per_fold.append(_island_fold_evidence(
            fold=fold,
            ratio=ratios[fold],
            receipt=receipts[fold],
            binding=folds[fold],
            genotype_sha=genotype_sha,
            horizon=horizon,
            origins_binding=origins_binding,
            origins_sha256=None if origins is None else origins[fold],
        ))

    fitness = candidate_info.get("fitness")
    relative = metrics.get("relative_error_vs_naive")
    _finite(fitness, "fitness", nullable=False)
    _finite(relative, "relative_error_vs_naive", nullable=False)
    expected = sum(row["ratio_to_naive"] for row in per_fold) / 4.0 - 1.0
    if not math.isclose(float(fitness), expected, rel_tol=0.0, abs_tol=1e-12):
        _fail("ISLAND_FITNESS_NOT_FOUR_FOLD_MEAN")
    if not math.isclose(float(relative), float(fitness), rel_tol=0.0, abs_tol=1e-12):
        _fail("ISLAND_RELATIVE_ERROR_MISMATCH")

    aggregate = {}
    for split, prefix, source, naive_source in (
        ("train", "train", "train_mae", "train_naive_mae"),
        ("inner", "inner", "val_mae", "val_naive_mae"),
    ):
        count = sum(row["counts"][split] for row in per_fold)
        expected_mae = math.fsum(row["error_sums"][split] for row in per_fold) / count
        expected_naive = math.fsum(row["naive_sums"][split] for row in per_fold) / count
        for field, observed, expected_value in (
            (source, candidate_info.get(source), expected_mae),
            (naive_source, candidate_info.get(naive_source), expected_naive),
        ):
            _finite(observed, field, nullable=False)
            if not math.isclose(float(observed), expected_value, rel_tol=1e-12, abs_tol=1e-12):
                _fail(f"ISLAND_AGGREGATE_{field.upper()}")
        aggregate[f"{prefix}_mae"] = expected_mae
        aggregate[f"{prefix}_naive_mae"] = expected_naive
    aggregate["relative_error_vs_naive"] = float(fitness)

    event_identity = {
        "domain_id": domain_id, "plan_sha256": plan_sha, "horizon_hours": horizon,
        "seed": seed, "island_id": island_id, "generation": generation,
        "population_index": population_index, "total_evaluations": total_evaluations,
        "search_candidate_sha256": search_sha,
    }
    payload = {
        "schema": ISLAND_CANDIDATE_SCHEMA,
        "experiment": domain_id,
        "plan_sha256": plan_sha,
        "contract_sha256": contract_sha,
        "checkpoint_identity_sha256": checkpoint_sha,
        "horizon_hours": horizon,
        "seed": seed,
        "island_id": island_id,
        "generation": generation,
        "population_index": population_index,
        "total_evaluations": total_evaluations,
        "is_champion": bool(candidate_info.get("is_champion")),
        "genotype": {"schema": "i6n-search-v1-effective-config", "effective": effective},
        "genotype_canonical_sha256": genotype_sha,
        "effective_config": effective,
        "effective_config_sha256": effective_sha,
        "search_candidate_sha256": search_sha,
        "archive_event_sha256": _sha256(event_identity),
        "fitness": float(fitness),
        "aggregate_metrics": aggregate,
        "per_fold": per_fold,
        "metric_schema": metrics.get("metric_schema"),
        "aggregation": metrics.get("aggregation"),
    }
    return validate_island_candidate_payload(payload)


def validate_island_candidate_payload(payload: Any) -> dict:
    """Fail closed if an island candidate body or any identity was altered."""
    if not isinstance(payload, Mapping) or payload.get("schema") != ISLAND_CANDIDATE_SCHEMA:
        _fail(f"payload schema must be {ISLAND_CANDIDATE_SCHEMA}")
    p = dict(payload)
    required = {
        "schema", "experiment", "plan_sha256", "contract_sha256",
        "checkpoint_identity_sha256", "horizon_hours", "seed", "island_id",
        "generation", "population_index", "total_evaluations", "is_champion", "genotype",
        "genotype_canonical_sha256", "effective_config", "effective_config_sha256",
        "search_candidate_sha256", "archive_event_sha256", "fitness", "aggregate_metrics",
        "per_fold", "metric_schema", "aggregation",
    }
    if set(p) != required:
        _fail("ISLAND_PAYLOAD_KEYS")
    if not isinstance(p["experiment"], str) or not _KEY.match(p["experiment"]):
        _fail("ISLAND_EXPERIMENT")
    for name in ("plan_sha256", "contract_sha256", "checkpoint_identity_sha256",
                 "genotype_canonical_sha256", "effective_config_sha256",
                 "search_candidate_sha256", "archive_event_sha256"):
        _hex(p, name)
    if p["horizon_hours"] not in (4, 72):
        _fail("ISLAND_HORIZON must be 4 or 72")
    if not isinstance(p["effective_config"], Mapping) or p["effective_config"].get("horizon") != p["horizon_hours"]:
        _fail("ISLAND_EFFECTIVE_CONFIG")
    effective = dict(p["effective_config"])
    if e_config_sha256(effective) != p["effective_config_sha256"]:
        _fail("ISLAND_EFFECTIVE_CONFIG_DIGEST")
    expected_genotype = _sha256({"schema": "i6n-search-v1", "config": effective})
    if expected_genotype != p["genotype_canonical_sha256"]:
        _fail("ISLAND_GENOTYPE_DIGEST")
    if p["genotype"] != {"schema": "i6n-search-v1-effective-config", "effective": effective}:
        _fail("ISLAND_GENOTYPE")
    if search_candidate_sha256(p["plan_sha256"], p["horizon_hours"], expected_genotype) \
            != p["search_candidate_sha256"]:
        _fail("ISLAND_SEARCH_CANDIDATE_DIGEST")
    for name in ("seed", "generation", "population_index", "total_evaluations"):
        _int(p, name)
    if not isinstance(p["island_id"], str) or not _KEY.match(p["island_id"]):
        _fail("ISLAND_ID")
    if not isinstance(p["is_champion"], bool):
        _fail("ISLAND_CHAMPION_FLAG")
    _finite(p["fitness"], "fitness", nullable=False)
    if not isinstance(p["aggregate_metrics"], Mapping):
        _fail("ISLAND_AGGREGATE_METRICS")
    for name, value in p["aggregate_metrics"].items():
        _finite(value, f"aggregate_metrics.{name}", nullable=False)
    if not isinstance(p["per_fold"], list) or len(p["per_fold"]) != 4:
        _fail("ISLAND_FOUR_FOLDS_REQUIRED")
    fold_names = []
    for row in p["per_fold"]:
        if not isinstance(row, Mapping) or set(row) != {
            "fold", "ratio_to_naive", "relative_error_vs_naive", "counts",
            "error_sums", "naive_sums", "support_digest", "receipt",
            "receipt_sha256", "receipt_identity_sha256", "origins_binding",
            "origins_sha256", "origin_identity_sha256", "support_sha256",
            "evaluation_context_sha256", "fold_identity_sha256",
        }:
            _fail("ISLAND_FOLD_ROW")
        fold_names.append(row["fold"])
        for name in (
            "receipt_sha256", "receipt_identity_sha256", "origin_identity_sha256",
            "support_sha256", "evaluation_context_sha256", "fold_identity_sha256",
        ):
            _hex(row, name)
        counts = _island_split_mapping(row["counts"], "counts", kind="count")
        error_sums = _island_split_mapping(
            row["error_sums"], "error_sums", kind="nonnegative"
        )
        naive_sums = _island_split_mapping(
            row["naive_sums"], "naive_sums", kind="positive"
        )
        support_digest = _island_split_mapping(
            row["support_digest"], "support_digest", kind="digest"
        )
        receipt = row["receipt"]
        if not isinstance(receipt, Mapping) or _sha256(receipt) != row["receipt_sha256"]:
            _fail("ISLAND_RECEIPT_DIGEST")
        if (receipt.get("status") != "ok" or receipt.get("counts") != counts
                or receipt.get("error_sums") != error_sums
                or receipt.get("naive_sums") != naive_sums
                or receipt.get("support_digest") != support_digest):
            _fail("ISLAND_RECEIPT_EVIDENCE_MISMATCH")
        metadata = receipt.get("metadata")
        candidate_receipt = metadata.get("candidate_receipt") if isinstance(metadata, Mapping) else None
        if not isinstance(candidate_receipt, Mapping):
            _fail("ISLAND_RECEIPT_BINDING_MISSING")
        if _sha256(candidate_receipt) != row["receipt_identity_sha256"]:
            _fail("ISLAND_RECEIPT_IDENTITY")
        if candidate_receipt.get("genotype_canonical_sha256") != p["genotype_canonical_sha256"]:
            _fail("ISLAND_RECEIPT_CANDIDATE_MISMATCH")
        if candidate_receipt.get("fold") != row["fold"]:
            _fail("ISLAND_RECEIPT_FOLD_MISMATCH")
        if candidate_receipt.get("horizon_hours") != p["horizon_hours"]:
            _fail("ISLAND_RECEIPT_HORIZON_MISMATCH")

        origins_binding = row["origins_binding"]
        origins = row["origins_sha256"]
        if origins_binding == "VERIFIED":
            origins = _island_split_mapping(origins, "origins_sha256", kind="digest")
            rows = candidate_receipt.get("rows")
            if not isinstance(rows, list):
                _fail("ISLAND_RECEIPT_ENDPOINT_ROW_MISSING")
            for split in _ISLAND_SPLITS:
                endpoint = [
                    item for item in rows
                    if isinstance(item, Mapping)
                    and item.get("split") == split.upper()
                    and item.get("head_offset_hours") == p["horizon_hours"]
                    and item.get("head_role") == "ENDPOINT"
                    and item.get("comparison") is True
                ]
                if len(endpoint) != 1:
                    _fail("ISLAND_RECEIPT_ENDPOINT_ROW_MISSING")
                if endpoint[0].get("support_sha256") != origins[split]:
                    _fail("ISLAND_RECEIPT_ORIGINS_MISMATCH")
        elif origins_binding == "NOT_CHECKED" and origins is None:
            origins = None
        else:
            _fail("ISLAND_ORIGINS_BINDING")
        expected_origin_identity = _sha256({
            "origins_binding": origins_binding,
            "origins_sha256": origins,
        })
        if expected_origin_identity != row["origin_identity_sha256"]:
            _fail("ISLAND_ORIGIN_IDENTITY")

        expected_support_sha = _sha256({
            "counts": counts,
            "naive_sums": naive_sums,
            "support_digest": support_digest,
        })
        if expected_support_sha != row["support_sha256"]:
            _fail("ISLAND_SUPPORT_IDENTITY")
        _finite(row["ratio_to_naive"], "ratio_to_naive", nullable=False)
        _finite(row["relative_error_vs_naive"], "fold_relative_error", nullable=False)
        recomputed_ratio = math.fsum(
            error_sums[split] / naive_sums[split] for split in _ISLAND_SPLITS
        ) / len(_ISLAND_SPLITS)
        if not math.isclose(row["ratio_to_naive"], recomputed_ratio,
                            rel_tol=1e-12, abs_tol=1e-12):
            _fail("ISLAND_FOLD_RATIO_EVIDENCE")
        if not math.isclose(row["relative_error_vs_naive"], row["ratio_to_naive"] - 1.0,
                            rel_tol=0.0, abs_tol=1e-12):
            _fail("ISLAND_FOLD_RELATIVE_ERROR")
        expected_fold = _sha256({
            "fold": row["fold"],
            "support_sha256": row["support_sha256"],
            "evaluation_context_sha256": row["evaluation_context_sha256"],
            "receipt_sha256": row["receipt_sha256"],
            "origin_identity_sha256": row["origin_identity_sha256"],
        })
        if expected_fold != row["fold_identity_sha256"]:
            _fail("ISLAND_FOLD_IDENTITY")
    if fold_names != sorted(set(fold_names)):
        _fail("ISLAND_FOLD_ORDER_OR_DUPLICATE")
    expected_fitness = sum(row["ratio_to_naive"] for row in p["per_fold"]) / 4.0 - 1.0
    if not math.isclose(p["fitness"], expected_fitness, rel_tol=0.0, abs_tol=1e-12):
        _fail("ISLAND_FITNESS_NOT_FOUR_FOLD_MEAN")
    expected_aggregate = {"relative_error_vs_naive": expected_fitness}
    for split, prefix in (("train", "train"), ("inner", "inner")):
        count = sum(row["counts"][split] for row in p["per_fold"])
        expected_aggregate[f"{prefix}_mae"] = math.fsum(
            row["error_sums"][split] for row in p["per_fold"]
        ) / count
        expected_aggregate[f"{prefix}_naive_mae"] = math.fsum(
            row["naive_sums"][split] for row in p["per_fold"]
        ) / count
    if set(p["aggregate_metrics"]) != set(expected_aggregate):
        _fail("ISLAND_AGGREGATE_METRIC_KEYS")
    for name, expected_value in expected_aggregate.items():
        if not math.isclose(p["aggregate_metrics"][name], expected_value,
                            rel_tol=1e-12, abs_tol=1e-12):
            _fail(f"ISLAND_AGGREGATE_{name.upper()}")
    event_identity = {name: p[name] for name in (
        "plan_sha256", "horizon_hours", "seed", "island_id", "generation",
        "population_index", "total_evaluations", "search_candidate_sha256",
    )}
    event_identity["domain_id"] = p["experiment"]
    if _sha256(event_identity) != p["archive_event_sha256"]:
        _fail("ISLAND_EVENT_IDENTITY")
    canonical(p)
    return p


def _fail(message: str):
    raise ArchiveRecordError(message)


def _hex(doc: Mapping, name: str, nullable: bool = False) -> None:
    value = doc.get(name)
    if value is None and nullable:
        return
    if not isinstance(value, str) or not _HEX64.match(value):
        _fail(f"{name} must be a lowercase sha256")


def _int(doc: Mapping, name: str, minimum: int = 0) -> int:
    value = doc.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str, nullable: bool = True) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{name} must be a finite number")


def validate_candidate(candidate: Any) -> dict:
    if not isinstance(candidate, Mapping):
        _fail("candidate must be an object")
    c = dict(candidate)
    if "won" in c:
        _fail("WON_IS_A_DECISION_EVENT: the search outcome is recorded with record_decision, "
              "not as a field of the candidate record")
    if c.get("schema") != CANDIDATE_SCHEMA:
        _fail(f"candidate schema must be {CANDIDATE_SCHEMA}")
    required = ("experiment", "search_candidate_sha256", "horizon_hours", "plan_sha256", "plan_metadata",
                "genotype", "genotype_canonical_sha256", "effective_config", "effective_config_sha256",
                "disposition", "fitness", "seed", "lineage", "attempts", "costs")
    missing = [k for k in required if k not in c]
    if missing:
        _fail("candidate fields missing: " + ",".join(missing))
    extra = set(c) - set(required) - {"schema", "disposition_reason", "provenance"}
    if extra:
        _fail("candidate fields not in the schema: " + ",".join(sorted(extra)))
    if not isinstance(c["experiment"], str) or not _KEY.match(c["experiment"]):
        _fail("experiment must match [A-Za-z0-9._:-]{1,96}")
    for name in ("search_candidate_sha256", "plan_sha256", "genotype_canonical_sha256", "effective_config_sha256"):
        _hex(c, name)
    _int(c, "horizon_hours", 1)
    for name in ("plan_metadata", "genotype", "effective_config", "costs", "lineage"):
        if not isinstance(c[name], Mapping):
            _fail(f"{name} must be an object")
    if e_config_sha256(c["effective_config"]) != c["effective_config_sha256"]:
        _fail("effective_config_sha256 is not the canonical sha256 of effective_config")
    if search_candidate_sha256(c["plan_sha256"], c["horizon_hours"], c["genotype_canonical_sha256"]) \
            != c["search_candidate_sha256"]:
        _fail("search_candidate_sha256 does not match (plan_sha256, horizon_hours, genotype_canonical_sha256)")
    if c["disposition"] not in DISPOSITIONS:
        _fail("disposition must be one of " + ",".join(DISPOSITIONS))
    if c["disposition"] in SCORED_DISPOSITIONS:
        _finite(c["fitness"], "fitness", nullable=False)
    elif c["fitness"] is not None:
        _fail(f"fitness must be null for disposition {c['disposition']} (no invented score)")
    if c["seed"] != SEED:
        _fail(f"seed must be {SEED}")
    lineage = c["lineage"]
    _int(lineage, "generation")
    _int(lineage, "population_index")
    if not isinstance(lineage.get("operator"), str) or not isinstance(lineage.get("parents"), list):
        _fail("lineage needs operator (str) and parents (list)")
    for parent in lineage["parents"]:
        if not isinstance(parent, str) or not _HEX64.match(parent):
            _fail("lineage parents must be sha256 identities")
    if not isinstance(c["attempts"], list):
        _fail("attempts must be a list")
    per_fold: dict[str, int] = {}
    for attempt in c["attempts"]:
        if not isinstance(attempt, Mapping) or not isinstance(attempt.get("fold"), str):
            _fail("attempt needs a fold")
        index = _int(attempt, "index", 1)
        if index > MAX_ATTEMPTS_PER_FOLD:
            _fail("more than three physical attempts per identity/fold")
        per_fold[attempt["fold"]] = per_fold.get(attempt["fold"], 0) + 1
        if not isinstance(attempt.get("state"), str):
            _fail("attempt needs a state")
        _hex(attempt, "parent_status_sha256", nullable=True)
    if any(n > MAX_ATTEMPTS_PER_FOLD for n in per_fold.values()):
        _fail("more than three physical attempts per identity/fold")
    for key, value in c["costs"].items():
        _finite(value, f"costs.{key}")
    canonical(c)  # refuses non-finite numbers anywhere
    return c


def validate_receipt(receipt: Any, candidate: Mapping) -> dict:
    if not isinstance(receipt, Mapping):
        _fail("fold receipt must be an object")
    r = dict(receipt)
    if r.get("schema_version") != RECEIPT_SCHEMA:
        _fail(f"fold receipt schema_version must be {RECEIPT_SCHEMA}")
    for name in ("experiment", "horizon_hours", "genotype_canonical_sha256", "effective_config_sha256"):
        if r.get(name) != candidate[name]:
            _fail(f"fold receipt {name} differs from the candidate")
    if not isinstance(r.get("fold"), str) or not r["fold"]:
        _fail("fold receipt needs a fold")
    if "2025" in r["fold"] or "TEST" in r["fold"].upper():
        _fail("TEST folds are never archived")
    _hex(r, "candidate_identity_sha256")
    rows = r.get("rows")
    if not isinstance(rows, list):
        _fail("fold receipt rows must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            _fail("receipt row must be an object")
        missing = [k for k in ROW_IDENTITY if k not in row]
        if missing:
            _fail("receipt row identity missing: " + ",".join(missing))
        if row["split"] not in SPLITS:
            _fail(f"receipt row split {row['split']!r} refused (TEST is never opened)")
        if row["population"] not in ("S", "U"):
            _fail("population must be S or U")
        if row["split"] == "VALIDATION" and not r["fold"].startswith("W"):
            _fail("VALIDATION_ON_INNER_FOLD: outer-week VALIDATION rows belong to WEEKLY_REFIT units "
                  "(fold W<week>), never to an inner fold")
        if row["role"] not in ROLES:
            _fail("role must be the C1 canonical MANAGEMENT or ENTRY (INTERFACE_A A.2)")
        if row.get("support_sha256") is not None:
            _hex(row, "support_sha256")
        if row["fold"] != r["fold"] or row["experiment"] != r["experiment"] \
                or row["candidate_identity_sha256"] != r["candidate_identity_sha256"] \
                or row["schema_version"] != RECEIPT_SCHEMA or row["horizon_hours"] != r["horizon_hours"]:
            _fail("receipt row identity differs from its receipt")
        n = _int(row, "n")
        _int(row, "head_offset_hours")
        for name in _SUMS:
            _finite(row.get(name), name)
            if n > 0 and row.get("status") == "SCORED" and row.get(name) is None:
                _fail(f"{name} is required for a scored row")
            if row.get(name) is not None and row[name] < 0:
                _fail(f"{name} must be non-negative")
        key = tuple(row[k] for k in ROW_IDENTITY)
        if key in seen:
            _fail("duplicate receipt row identity")
        seen.add(key)
    canonical(r)
    return r


def _attempt(receipt: Mapping) -> dict:
    attempt = receipt.get("attempt") or {"index": 1, "parent_status_sha256": None}
    if not isinstance(attempt, Mapping):
        _fail("ATTEMPT_LINEAGE: attempt must be an object")
    return dict(attempt)


def round_payload(candidate: Mapping, fold_receipts: list) -> dict:
    """One candidate and ALL its receipts. Physical attempts of one fold are kept (max 3);
    the canonical scientific measurement of a fold is its lowest attempt index."""
    c = validate_candidate(candidate)
    receipts = [validate_receipt(r, c) for r in fold_receipts]
    by_fold: dict[str, list] = {}
    for r in receipts:
        attempt = _attempt(r)
        index = attempt.get("index", 1)
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            _fail("ATTEMPT_LINEAGE: attempt.index must be an integer >= 1")
        if index > 1:
            parent = attempt.get("parent_status_sha256")
            if not isinstance(parent, str) or not _HEX64.match(parent):
                _fail("ATTEMPT_LINEAGE: a recovery attempt needs the previous terminal's parent_status_sha256")
        by_fold.setdefault(r["fold"], []).append((index, r))
    for fold, attempts in by_fold.items():
        indexes = [i for i, _ in attempts]
        if len(indexes) != len(set(indexes)):
            _fail(f"ATTEMPT_LINEAGE: two receipts for fold {fold} with the same attempt index")
        if len(attempts) > MAX_ATTEMPTS_PER_FOLD or max(indexes) > MAX_ATTEMPTS_PER_FOLD:
            _fail(f"ATTEMPTS_EXCEEDED: more than {MAX_ATTEMPTS_PER_FOLD} physical attempts for fold {fold}")
        if len({r["candidate_identity_sha256"] for _, r in attempts}) != 1:
            _fail(f"ATTEMPT_IDENTITY_MISMATCH: attempts of fold {fold} carry different identities")
    receipts.sort(key=lambda r: (r["fold"], _attempt(r).get("index", 1)))
    return {"schema": ROUND_SCHEMA, "experiment": c["experiment"],
            "search_candidate_sha256": c["search_candidate_sha256"], "candidate": c,
            "fold_receipts": receipts}


def validate_round_payload(payload: Any) -> dict:
    if not isinstance(payload, Mapping) or payload.get("schema") != ROUND_SCHEMA:
        _fail(f"payload schema must be {ROUND_SCHEMA}")
    if set(payload) != {"schema", "experiment", "search_candidate_sha256", "candidate", "fold_receipts"}:
        _fail("payload keys differ from the schema")
    rebuilt = round_payload(payload["candidate"], list(payload["fold_receipts"]))
    if canonical(rebuilt) != canonical(dict(payload)):
        _fail("payload is not in canonical order or differs from its candidate")
    return rebuilt


# --------------------------------------------------------------------------- projection
def row_key_sha256(identity: Mapping) -> str:
    return hashlib.sha256(json.dumps({k: identity[k] for k in ROW_IDENTITY}, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode("ascii")).hexdigest()


def _derived(row: Mapping) -> list[dict]:
    unit = row["unit"]
    n = row["n"]
    out = [("n", float(n), "count")]
    if n > 0 and row.get("status") == "SCORED" and all(row.get(s) is not None for s in _SUMS):
        sa, ss, na, ns = (float(row[s]) for s in _SUMS)
        out += [("sum_abs_error", sa, unit), ("sum_sq_error", ss, unit + "_SQUARED"),
                ("naive_sum_abs_error", na, unit), ("naive_sum_sq_error", ns, unit + "_SQUARED"),
                ("mae", sa / n, unit), ("naive_mae", na / n, unit),
                ("mse", ss / n, unit + "_SQUARED"), ("naive_mse", ns / n, unit + "_SQUARED")]
        if na > 0:
            out.append(("skill_mae", (sa - na) / na, "RATIO"))
    return [{"metric": m, "value": v, "unit": u, "split": row["split"],
             "horizon": row["head_offset_hours"]} for m, v, u in out]


def _base_report(candidate: Mapping, key: str, phase: str, tags: dict, metrics: list) -> dict:
    from predictor_olap_store.query import report_sha256  # olap/store/src on sys.path (package init)

    provenance = candidate.get("provenance") or {}
    report = {"experiment_key": "i6n-doin:" + key,
              "experiment_set_key": "i6n-doin-cand:" + candidate["search_candidate_sha256"],
              "actor": "predictor", "lake": "doin_archive", "lineage": "UNVERIFIED",
              "config_sha256": candidate["effective_config_sha256"],
              "code_commit": provenance.get("code_commit"), "project": "predictor", "phase": phase,
              "tags": {k: ("" if v is None else str(v)) for k, v in tags.items()}, "metrics": metrics}
    report["report_sha256"] = report_sha256(report)
    return report


def projection_units(payload: Mapping) -> list[dict]:
    """Receipt rows and one candidate-level row, each as {identity, extra, report}."""
    p = validate_round_payload(payload)
    c = p["candidate"]
    common = {"search_candidate_sha256": c["search_candidate_sha256"], "plan_sha256": c["plan_sha256"],
              "genotype_canonical_sha256": c["genotype_canonical_sha256"], "disposition": c["disposition"],
              "doin_metadata_schema": DOIN_METADATA_SCHEMA, "test_read": "false"}
    units = []
    history = []
    canonical_folds: set[str] = set()
    for receipt in p["fold_receipts"]:  # sorted by (fold, attempt index)
        attempt = _attempt(receipt).get("index", 1)
        is_canonical = receipt["fold"] not in canonical_folds
        history.append({"fold": receipt["fold"], "index": attempt, "canonical": is_canonical,
                        "receipt_sha256": hashlib.sha256(canonical(receipt)).hexdigest(),
                        "parent_status_sha256": _attempt(receipt).get("parent_status_sha256")})
        if not is_canonical:
            continue  # kept in the body and in attempt_history_json, never projected as metrics
        canonical_folds.add(receipt["fold"])
        for row in receipt["rows"]:
            identity = {k: row[k] for k in ROW_IDENTITY}
            key = row_key_sha256(identity)
            tags = dict(common, row_key_sha256=key, **{k: row[k] for k in ROW_IDENTITY},
                        head_role=row.get("head_role"), comparison=str(bool(row.get("comparison"))).lower(),
                        scale=row.get("scale"), status=row.get("status"),
                        c2_receipt_sha256=row.get("c2_receipt_sha256"), c2_role=row.get("c2_role"),
                        support_sha256=row.get("support_sha256"), attempt_index=attempt)
            units.append({"identity": identity,
                          "extra": {"search_candidate_sha256": c["search_candidate_sha256"],
                                    "head_role": row.get("head_role"), "comparison": bool(row.get("comparison")),
                                    "status": row.get("status")},
                          "report": _base_report(c, key, "I6N_SWARM_DOIN", tags, _derived(row))})
    identity = {"experiment": c["experiment"], "candidate_identity_sha256": c["search_candidate_sha256"],
                "fold": "ALL_FOLDS", "horizon_hours": c["horizon_hours"], "head_offset_hours": c["horizon_hours"],
                "role": "CANDIDATE", "population": "S", "split": "SEARCH", "metric": "CANDIDATE_DISPOSITION",
                "unit": "NONE", "schema_version": CANDIDATE_SCHEMA}
    key = row_key_sha256(identity)
    metrics = [{"metric": "attempts_total", "value": float(len(c["attempts"])), "unit": "count",
                "split": "SEARCH", "horizon": c["horizon_hours"]}]
    if c["fitness"] is not None:
        metrics.append({"metric": "fitness", "value": float(c["fitness"]), "unit": "RATIO", "split": "SEARCH",
                        "horizon": c["horizon_hours"]})
    for name, value in sorted(c["costs"].items()):
        if value is not None:
            unit = "SECONDS" if name.endswith("_seconds") else "BYTES" if name.endswith("_bytes") else "VALUE"
            metrics.append({"metric": "cost." + name, "value": float(value), "unit": unit, "split": "SEARCH",
                            "horizon": c["horizon_hours"]})
    lineage = c["lineage"]
    tags = dict(common, row_key_sha256=key, **identity,
                attempt_history_json=json.dumps(history, sort_keys=True, separators=(",", ":")),
                disposition_reason=c.get("disposition_reason"), generation=lineage["generation"],
                population_index=lineage["population_index"], operator=lineage["operator"],
                parents_json=json.dumps(lineage["parents"], separators=(",", ":")),
                genotype_json=json.dumps(c["genotype"], sort_keys=True, separators=(",", ":")),
                effective_config_json=json.dumps(c["effective_config"], sort_keys=True, separators=(",", ":")),
                effective_config_sha256=c["effective_config_sha256"],
                plan_metadata_json=json.dumps(c["plan_metadata"], sort_keys=True, separators=(",", ":")),
                attempts_json=json.dumps(c["attempts"], sort_keys=True, separators=(",", ":")),
                seed=c["seed"])
    units.append({"identity": identity,
                  "extra": {"search_candidate_sha256": c["search_candidate_sha256"], "head_role": None,
                            "comparison": None, "status": c["disposition"]},
                  "report": _base_report(c, key, "I6N_SWARM_DOIN_CANDIDATE", tags, metrics)})
    return units


# --------------------------------------------------------------------------- search decisions
def validate_decision(decision: Any) -> dict:
    """An append-only search outcome event (winner/finalist/control). A later decision for the
    same (experiment, plan, horizon) has a higher sequence; nothing earlier is rewritten."""
    if not isinstance(decision, Mapping) or decision.get("schema") != DECISION_SCHEMA:
        _fail(f"DECISION_SCHEMA: schema must be {DECISION_SCHEMA}")
    d = dict(decision)
    keys = {"schema", "experiment", "plan_sha256", "horizon_hours", "sequence", "supersedes_decision_sha256",
            "basis", "outcomes"}
    if set(d) != keys:
        _fail(f"DECISION_SCHEMA: keys must be {sorted(keys)}")
    if not isinstance(d["experiment"], str) or not _KEY.match(d["experiment"]):
        _fail("DECISION_SCHEMA: experiment")
    _hex(d, "plan_sha256")
    _int(d, "horizon_hours", 1)
    if _int(d, "sequence", 1) > 999999:
        _fail("DECISION_SCHEMA: sequence above 999999")
    _hex(d, "supersedes_decision_sha256", nullable=True)
    if not isinstance(d["basis"], Mapping):
        _fail("DECISION_SCHEMA: basis must be an object")
    outcomes = d["outcomes"]
    if not isinstance(outcomes, list) or not outcomes:
        _fail("DECISION_SCHEMA: outcomes must be a non-empty list")
    seen = set()
    for item in outcomes:
        if not isinstance(item, Mapping) or set(item) != {"search_candidate_sha256", "outcome"}:
            _fail("DECISION_SCHEMA: outcome entries are {search_candidate_sha256, outcome}")
        _hex(item, "search_candidate_sha256")
        if item["outcome"] not in OUTCOMES:
            _fail("DECISION_OUTCOME: outcome must be one of " + ",".join(OUTCOMES))
        if item["search_candidate_sha256"] in seen:
            _fail("DECISION_DUPLICATE_CANDIDATE")
        seen.add(item["search_candidate_sha256"])
    if sum(1 for item in outcomes if item["outcome"] == "WINNER") > 1:
        _fail("DECISION_TWO_WINNERS: at most one WINNER per decision")
    d["outcomes"] = sorted((dict(o) for o in outcomes), key=lambda o: o["search_candidate_sha256"])
    canonical(d)
    return d


def decision_units(payload: Mapping) -> list[dict]:
    d = validate_decision(payload)
    if canonical(d) != canonical(dict(payload)):
        _fail("DECISION_SCHEMA: payload is not in canonical order")
    fold = f"DECISION-{d['sequence']:06d}"
    units = []
    for item in d["outcomes"]:
        identity = {"experiment": d["experiment"], "candidate_identity_sha256": item["search_candidate_sha256"],
                    "fold": fold, "horizon_hours": d["horizon_hours"], "head_offset_hours": d["horizon_hours"],
                    "role": "DECISION", "population": "S", "split": "SEARCH", "metric": "SEARCH_OUTCOME",
                    "unit": "NONE", "schema_version": DECISION_SCHEMA}
        key = row_key_sha256(identity)
        tags = dict({k: str(v) for k, v in identity.items()}, row_key_sha256=key, outcome=item["outcome"],
                    sequence=str(d["sequence"]), plan_sha256=d["plan_sha256"],
                    supersedes_decision_sha256=d["supersedes_decision_sha256"] or "",
                    basis_json=json.dumps(d["basis"], sort_keys=True, separators=(",", ":")),
                    doin_metadata_schema=DOIN_METADATA_SCHEMA, test_read="false")
        metrics = [{"metric": "won", "value": 1.0 if item["outcome"] == "WINNER" else 0.0, "unit": "FLAG",
                    "split": "SEARCH", "horizon": d["horizon_hours"]},
                   {"metric": "selected", "value": 0.0 if item["outcome"] == "NOT_SELECTED" else 1.0,
                    "unit": "FLAG", "split": "SEARCH", "horizon": d["horizon_hours"]}]
        pseudo = {"search_candidate_sha256": item["search_candidate_sha256"],
                  "effective_config_sha256": None, "provenance": {}}
        units.append({"identity": identity,
                      "extra": {"search_candidate_sha256": item["search_candidate_sha256"], "head_role": None,
                                "comparison": None, "status": item["outcome"]},
                      "report": _base_report(pseudo, key, "I6N_SWARM_DOIN_DECISION", tags, metrics)})
    return units


def island_candidate_units(payload: Mapping) -> list[dict]:
    """Project one island callback into four fold rows and one candidate row."""
    p = validate_island_candidate_payload(payload)
    common = {
        "search_candidate_sha256": p["search_candidate_sha256"],
        "archive_event_sha256": p["archive_event_sha256"],
        "plan_sha256": p["plan_sha256"],
        "contract_sha256": p["contract_sha256"],
        "checkpoint_identity_sha256": p["checkpoint_identity_sha256"],
        "genotype_canonical_sha256": p["genotype_canonical_sha256"],
        "effective_config_sha256": p["effective_config_sha256"],
        "island_id": p["island_id"],
        "generation": p["generation"],
        "population_index": p["population_index"],
        "total_evaluations": p["total_evaluations"],
        "seed": p["seed"],
        "doin_metadata_schema": DOIN_METADATA_SCHEMA,
        "test_read": "false",
    }
    pseudo = {
        "search_candidate_sha256": p["search_candidate_sha256"],
        "effective_config_sha256": p["effective_config_sha256"],
        "provenance": {},
    }
    units = []
    for fold in p["per_fold"]:
        identity = {
            "experiment": p["experiment"],
            "candidate_identity_sha256": p["archive_event_sha256"],
            "fold": fold["fold"],
            "horizon_hours": p["horizon_hours"],
            "head_offset_hours": p["horizon_hours"],
            "role": "CANDIDATE_FOLD",
            "population": "S",
            "split": "SEARCH",
            "metric": "FOUR_FOLD_FITNESS",
            "unit": "RATIO",
            "schema_version": ISLAND_CANDIDATE_SCHEMA,
        }
        key = row_key_sha256(identity)
        tags = dict(common, row_key_sha256=key, **identity,
                    support_sha256=fold["support_sha256"],
                    evaluation_context_sha256=fold["evaluation_context_sha256"],
                    fold_identity_sha256=fold["fold_identity_sha256"],
                    receipt_sha256=fold["receipt_sha256"],
                    receipt_identity_sha256=fold["receipt_identity_sha256"],
                    origins_binding=fold["origins_binding"],
                    origins_sha256_json=json.dumps(
                        fold["origins_sha256"], sort_keys=True, separators=(",", ":")
                    ),
                    origin_identity_sha256=fold["origin_identity_sha256"])
        metrics = [
            {"metric": "fold_ratio_to_naive", "value": fold["ratio_to_naive"], "unit": "RATIO",
             "split": "SEARCH", "horizon": p["horizon_hours"]},
            {"metric": "fold_relative_error_vs_naive", "value": fold["relative_error_vs_naive"],
             "unit": "RATIO", "split": "SEARCH", "horizon": p["horizon_hours"]},
        ]
        for split in _ISLAND_SPLITS:
            split_name = split.upper()
            count = fold["counts"][split]
            error_sum = fold["error_sums"][split]
            naive_sum = fold["naive_sums"][split]
            metrics.extend([
                {"metric": "n", "value": float(count), "unit": "count",
                 "split": split_name, "horizon": p["horizon_hours"]},
                {"metric": "sum_abs_error", "value": error_sum, "unit": "PRICE",
                 "split": split_name, "horizon": p["horizon_hours"]},
                {"metric": "naive_sum_abs_error", "value": naive_sum, "unit": "PRICE",
                 "split": split_name, "horizon": p["horizon_hours"]},
                {"metric": "mae", "value": error_sum / count, "unit": "PRICE",
                 "split": split_name, "horizon": p["horizon_hours"]},
                {"metric": "naive_mae", "value": naive_sum / count, "unit": "PRICE",
                 "split": split_name, "horizon": p["horizon_hours"]},
            ])
        units.append({"identity": identity,
                      "extra": {"search_candidate_sha256": p["search_candidate_sha256"],
                                "head_role": None, "comparison": True, "status": "COMPLETE"},
                      "report": _base_report(pseudo, key, "I6N_ISLAND_FOLD", tags, metrics)})

    identity = {
        "experiment": p["experiment"],
        "candidate_identity_sha256": p["archive_event_sha256"],
        "fold": "ALL_FOLDS",
        "horizon_hours": p["horizon_hours"],
        "head_offset_hours": p["horizon_hours"],
        "role": "CANDIDATE",
        "population": "S",
        "split": "SEARCH",
        "metric": "FOUR_FOLD_FITNESS",
        "unit": "RATIO",
        "schema_version": ISLAND_CANDIDATE_SCHEMA,
    }
    key = row_key_sha256(identity)
    tags = dict(common, row_key_sha256=key, **identity,
                genotype_json=json.dumps(p["genotype"], sort_keys=True, separators=(",", ":")),
                effective_config_json=json.dumps(p["effective_config"], sort_keys=True, separators=(",", ":")),
                is_champion=str(p["is_champion"]).lower(), aggregation=p["aggregation"],
                metric_schema=p["metric_schema"])
    metrics = [{"metric": "fitness", "value": p["fitness"], "unit": "RATIO",
                "split": "SEARCH", "horizon": p["horizon_hours"]}]
    metrics.extend({"metric": name, "value": value,
                    "unit": "RATIO" if "relative" in name else "PRICE",
                    "split": "SEARCH", "horizon": p["horizon_hours"]}
                   for name, value in sorted(p["aggregate_metrics"].items()))
    units.append({"identity": identity,
                  "extra": {"search_candidate_sha256": p["search_candidate_sha256"],
                            "head_role": None, "comparison": True, "status": "COMPLETE"},
                  "report": _base_report(pseudo, key, "I6N_ISLAND_CANDIDATE", tags, metrics)})
    return units


def units_for(payload_schema: str, payload: Mapping) -> list[dict]:
    if payload_schema == ROUND_SCHEMA:
        return projection_units(payload)
    if payload_schema == DECISION_SCHEMA:
        return decision_units(payload)
    if payload_schema == ISLAND_CANDIDATE_SCHEMA:
        return island_candidate_units(payload)
    _fail(f"PAYLOAD_SCHEMA: {payload_schema!r} is not an archive payload")
