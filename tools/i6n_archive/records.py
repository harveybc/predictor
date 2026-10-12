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
    if c.get("schema") != CANDIDATE_SCHEMA:
        _fail(f"candidate schema must be {CANDIDATE_SCHEMA}")
    required = ("experiment", "search_candidate_sha256", "horizon_hours", "plan_sha256", "plan_metadata",
                "genotype", "genotype_canonical_sha256", "effective_config", "effective_config_sha256",
                "disposition", "fitness", "won", "seed", "lineage", "attempts", "costs")
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
    if not isinstance(c["won"], bool):
        _fail("won must be a boolean")
    if c["won"] and c["disposition"] not in SCORED_DISPOSITIONS:
        _fail("an unscored candidate cannot be a winner")
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


def round_payload(candidate: Mapping, fold_receipts: list) -> dict:
    c = validate_candidate(candidate)
    receipts = [validate_receipt(r, c) for r in fold_receipts]
    folds = [(r["fold"], (r.get("attempt") or {}).get("index", 1)) for r in receipts]
    if len(set(folds)) != len(folds):
        _fail("two receipts for the same fold and attempt")
    receipts.sort(key=lambda r: (r["fold"], (r.get("attempt") or {}).get("index", 1)))
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
    for receipt in p["fold_receipts"]:
        attempt = (receipt.get("attempt") or {}).get("index", 1)
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
    metrics = [{"metric": "won", "value": 1.0 if c["won"] else 0.0, "unit": "FLAG", "split": "SEARCH",
                "horizon": c["horizon_hours"]},
               {"metric": "attempts_total", "value": float(len(c["attempts"])), "unit": "count",
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
    tags = dict(common, row_key_sha256=key, **identity, won=str(c["won"]).lower(),
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
