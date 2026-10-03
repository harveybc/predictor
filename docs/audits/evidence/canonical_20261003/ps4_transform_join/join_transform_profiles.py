#!/usr/bin/env python3
"""Join nine transform variants to their emitted features and retained PS1 cells.

PS1 cells already on disk are counted. They are not recomputed. PS4 expanded
profiles stay pending. Nothing here selects a feature.
"""

import csv
import hashlib
import json
from collections import Counter
from io import StringIO
from pathlib import Path

BASE = Path("docs/audits/evidence/canonical_20261003")
METRICS = (
    "missingness", "constant", "scale_tails", "volatility", "acf", "trend",
    "adf", "kpss", "seasonality", "spectrum", "cost",
)
METRICS_VERSION = "laneA_ps1_metrics.v1"
PREFIX_CROSSCHECK = (
    ("tv.wav_d", "tv.wavelet_modwt_haar_causal"),
    ("tv.mt_", "tv.multitaper_trailing"),
    ("tv.hilbert", "tv.hilbert_trailing_lastsample"),
    ("tv.stl", "tv.stl_trailing_lastsample"),
    ("tv.kalman", "tv.kalman_local_level_filter"),
)
FIELDS = (
    "variant_id", "emitted_feature_id", "metric", "variant_state",
    "feature_admissibility", "ps1_metric_state", "ps1_metric_version",
    "ps2_status_by_target_horizon", "ps4_expanded_state", "input_digests",
)
PS1_STATES = ("MEASURED", "FAILED", "PENDING", "NOT_APPLICABLE")


class JoinRefusal(Exception):
    """A named refusal. The denominator is not shortened to hide it."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prefix_variant(feature_id):
    for prefix, variant_id in PREFIX_CROSSCHECK:
        if feature_id.startswith(prefix):
            return variant_id
    return None


def transform_variant_id(transform, feature_id):
    token = (transform or "").split(" ", 1)[0]
    marker = "output "
    if marker not in (transform or "") or not token.startswith("tv."):
        raise JoinRefusal("CONTRADICTORY_TRANSFORM_FIELD", feature_id)
    output = transform.split(marker, 1)[1].strip()
    if output != feature_id:
        raise JoinRefusal("CONTRADICTORY_TRANSFORM_FIELD", f"{feature_id} != {output}")
    return token


def expected_variant_state(probe):
    violation = probe.get("status") == "PREFIX_VIOLATION_MEASURED"
    declared = str(probe.get("admissible_as_feature")) == "True"
    if violation or not declared:
        return "NOT_ADMISSIBLE"
    if probe.get("status") == "PREFIX_INVARIANT_MEASURED" and declared:
        return "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY"
    raise JoinRefusal("CONTRADICTORY_VARIANT_STATE", probe.get("variant_id", ""))


def ps2_key(rows):
    parts = []
    for row in sorted(rows, key=lambda item: (item["target"], int(item["horizon"]))):
        parts.append(f"{row['target']}@{row['horizon']}={row['status']}")
    return ";".join(parts)


def load(root):
    root = Path(root)
    lane_a = root / BASE / "laneA" / "batch_003"
    lane_b = root / BASE / "laneB" / "batch_003"
    coverage = root / BASE / "source_transform_coverage" / "transform_coverage.csv"
    probes_path = root / BASE / "laneA" / "batch_001" / "transform_variants.csv"
    interpreted = {
        "transform_coverage.csv": coverage,
        "transform_variants.csv": probes_path,
        "admissible_features.json": lane_a / "admissible_features.json",
        "profile_cells.csv": lane_a / "profile_cells.csv",
        "ps2_status.csv": lane_b / "ps2_status.csv",
        "digests.json": lane_a / "digests.json",
        "ps2_manifest.json": lane_b / "ps2_manifest.json",
    }
    hashes = {name: digest(path) for name, path in interpreted.items()}
    features = read_json(interpreted["admissible_features.json"])["features"]
    digests = read_json(interpreted["digests.json"])
    manifest = read_json(interpreted["ps2_manifest.json"])
    return {
        "variants": read_csv(coverage),
        "probes": read_csv(probes_path),
        "features": features,
        "cells": read_csv(interpreted["profile_cells.csv"]),
        "ps2_rows": read_csv(interpreted["ps2_status.csv"]),
        "digests": digests,
        "ps2_manifest": manifest,
        "hashes": hashes,
        "lane_a": lane_a,
        "lane_b": lane_b,
    }


def check_digests(loaded):
    checks = []
    artifacts = loaded["digests"].get("artifacts_sha256", {})
    for name, expected in artifacts.items():
        path = loaded["lane_a"] / name
        if not path.is_file():
            checks.append({"name": name, "state": "INCOMPLETE_EVIDENCE", "reason": "ABSENT_FILE"})
            continue
        actual = digest(path)
        if actual != expected:
            raise JoinRefusal("DIGEST_MISMATCH", name)
        checks.append({"name": name, "state": "VERIFIED", "sha256": actual})
    for name, expected in loaded["digests"].get("inputs_sha256", {}).items():
        checks.append({
            "name": name,
            "state": "INCOMPLETE_EVIDENCE",
            "reason": "RETAINED_DIGEST_FILE_NOT_IN_WORKTREE",
            "retained_sha256": expected,
        })
    outputs = loaded["ps2_manifest"].get("output_sha256", {})
    for name, expected in outputs.items():
        path = loaded["lane_b"] / name
        if not path.is_file():
            checks.append({"name": f"ps2:{name}", "state": "INCOMPLETE_EVIDENCE", "reason": "ABSENT_FILE"})
            continue
        actual = digest(path)
        if actual != expected:
            raise JoinRefusal("DIGEST_MISMATCH", f"ps2:{name}")
        checks.append({"name": f"ps2:{name}", "state": "VERIFIED", "sha256": actual})
    if loaded["hashes"]["profile_cells.csv"] != artifacts.get("profile_cells.csv"):
        raise JoinRefusal("DIGEST_MISMATCH", "profile_cells.csv")
    if loaded["hashes"]["admissible_features.json"] != artifacts.get("admissible_features.json"):
        raise JoinRefusal("DIGEST_MISMATCH", "admissible_features.json")
    return checks


def join_loaded(loaded):
    digest_checks = check_digests(loaded)
    probes = {row["variant_id"]: row for row in loaded["probes"]}
    if len(probes) != len(loaded["probes"]):
        raise JoinRefusal("DUPLICATE_FEATURE_ID", "duplicate probe variant_id")
    variants = []
    for row in loaded["variants"]:
        variant_id = row["variant_id"]
        if variant_id not in probes:
            raise JoinRefusal("UNRECOGNIZED_VARIANT", variant_id)
        expected = expected_variant_state(probes[variant_id])
        if row.get("admissibility") != expected:
            raise JoinRefusal("CONTRADICTORY_VARIANT_STATE", variant_id)
        variants.append(row)
    if len({row["variant_id"] for row in variants}) != len(variants):
        raise JoinRefusal("UNRECOGNIZED_VARIANT", "duplicate variant row")
    probe_ids = set(probes)
    ledger_ids = {row["variant_id"] for row in variants}
    missing = sorted(probe_ids - ledger_ids)
    if missing:
        raise JoinRefusal("INCOMPLETE_EVIDENCE", "probe variant absent from ledger: " + ",".join(missing))

    emitted = [row for row in loaded["features"] if row.get("family") == "transform_variant"]
    by_id = {}
    for feature in emitted:
        feature_id = feature["feature_id"]
        if feature_id in by_id:
            raise JoinRefusal("DUPLICATE_FEATURE_ID", feature_id)
        variant_id = transform_variant_id(feature.get("transform", ""), feature_id)
        if variant_id not in ledger_ids:
            raise JoinRefusal("UNRECOGNIZED_VARIANT", variant_id)
        state = next(row["admissibility"] for row in variants if row["variant_id"] == variant_id)
        if state != "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY":
            raise JoinRefusal("REJECTED_VARIANT_EMITTED", f"{variant_id} -> {feature_id}")
        abbreviated = prefix_variant(feature_id)
        if abbreviated is not None and abbreviated != variant_id:
            raise JoinRefusal(
                "CONTRADICTORY_TRANSFORM_FIELD",
                f"{feature_id} abbreviation {abbreviated} contradicts transform {variant_id}",
            )
        if feature.get("admissibility") != "ADMISSIBLE":
            raise JoinRefusal("INCOMPLETE_EVIDENCE", f"{feature_id} admissibility {feature.get('admissibility')}")
        by_id[feature_id] = (feature, variant_id)

    cells = {}
    for cell in loaded["cells"]:
        if cell["feature_id"] not in by_id:
            continue
        key = (cell["feature_id"], cell["metric"])
        if key in cells:
            raise JoinRefusal("DUPLICATE_FEATURE_ID", f"{key[0]}:{key[1]}")
        cells[key] = cell

    ps2_by_feature = {}
    for row in loaded["ps2_rows"]:
        if row["feature"] in by_id:
            ps2_by_feature.setdefault(row["feature"], []).append(row)
    grid = set()
    for rows in ps2_by_feature.values():
        for row in rows:
            grid.add((row["target"], row["horizon"]))
    if not grid and by_id:
        raise JoinRefusal("ABSENT_PS2_STATUS", "no PS2 rows for emitted features")

    input_digests = ";".join(
        f"{name}={loaded['hashes'][name]}"
        for name in (
            "admissible_features.json",
            "profile_cells.csv",
            "ps2_status.csv",
            "transform_variants.csv",
            "transform_coverage.csv",
        )
    )
    rows = []
    measured_states = Counter()
    emitted_ids = []
    for variant in variants:
        variant_id = variant["variant_id"]
        owned = [feature_id for feature_id, (_feature, owner) in by_id.items() if owner == variant_id]
        # Preserve admissible-file order.
        owned = [feature["feature_id"] for feature in emitted if feature["feature_id"] in owned]
        if variant["admissibility"] != "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY":
            if owned:
                raise JoinRefusal("REJECTED_VARIANT_EMITTED", variant_id)
            rows.append({
                "variant_id": variant_id,
                "emitted_feature_id": "",
                "metric": "",
                "variant_state": "NOT_ADMISSIBLE",
                "feature_admissibility": "NOT_EMITTED",
                "ps1_metric_state": "NO_EMITTED_FEATURE",
                "ps1_metric_version": "",
                "ps2_status_by_target_horizon": "NO_EMITTED_FEATURE",
                "ps4_expanded_state": "PENDING",
                "input_digests": input_digests,
            })
            continue
        if not owned:
            raise JoinRefusal("INCOMPLETE_EVIDENCE", f"{variant_id} emitted no feature")
        for feature_id in owned:
            emitted_ids.append(feature_id)
            ps2_rows = ps2_by_feature.get(feature_id, [])
            present = {(row["target"], row["horizon"]) for row in ps2_rows}
            if present != grid:
                missing_pairs = sorted(grid - present)
                raise JoinRefusal(
                    "ABSENT_PS2_STATUS",
                    f"{feature_id} missing {missing_pairs[0][0]}@{missing_pairs[0][1]}",
                )
            status_text = ps2_key(ps2_rows)
            for metric in METRICS:
                cell = cells.get((feature_id, metric))
                if cell is None:
                    raise JoinRefusal("MISSING_METRIC_CELL", f"{feature_id} {metric}")
                state = cell.get("state", "")
                if state not in PS1_STATES:
                    raise JoinRefusal("INCOMPLETE_EVIDENCE", f"{feature_id} {metric} state {state}")
                if cell.get("metrics_version") != METRICS_VERSION:
                    raise JoinRefusal("INCOMPLETE_EVIDENCE", f"{feature_id} {metric} version")
                measured_states[state] += 1
                rows.append({
                    "variant_id": variant_id,
                    "emitted_feature_id": feature_id,
                    "metric": metric,
                    "variant_state": "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY",
                    "feature_admissibility": "ADMISSIBLE",
                    "ps1_metric_state": state,
                    "ps1_metric_version": cell["metrics_version"],
                    "ps2_status_by_target_horizon": status_text,
                    "ps4_expanded_state": "PENDING",
                    "input_digests": input_digests,
                })
    if len(rows) != len(emitted_ids) * len(METRICS) + sum(
        1 for row in variants if row["admissibility"] != "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY"
    ):
        raise JoinRefusal("INCOMPLETE_EVIDENCE", "row denominator changed during the join")
    for state in PS1_STATES:
        measured_states.setdefault(state, 0)
    admissible = sum(row["admissibility"] == "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY" for row in variants)
    report = {
        "schema": "canonical_20261003.ps4_transform_join.v1",
        "measurement_flag": "NO_NEW_MODEL_MEASUREMENT",
        "selection_status": "PENDING",
        "ps4_expanded_state": "PENDING",
        "ps1_profile_coverage": "COMPLETE" if measured_states["MEASURED"] == len(emitted_ids) * len(METRICS) and not any(
            measured_states[state] for state in ("FAILED", "PENDING", "NOT_APPLICABLE")
        ) else "CELLS_PRESENT_WITH_NON_MEASURED_STATES",
        "denominators": {
            "variants": len(variants),
            "admissible_causal_computability": admissible,
            "not_admissible_variants": len(variants) - admissible,
            "emitted_features": len(emitted_ids),
            "ps1_metric_cells": len(emitted_ids) * len(METRICS),
            "zero_output_rows": len(variants) - admissible,
            "join_rows": len(rows),
        },
        "ps1_cell_states": {state: measured_states[state] for state in PS1_STATES},
        "digest_checks": digest_checks,
        "inputs_sha256": dict(loaded["hashes"]),
        "findings": [
            "The variant ledger marks all nine rows PENDING_PROFILE. The ten emitted features already have 110 retained PS1 cells. That ledger token is not rewritten.",
            "PS4 expanded profiles are not among the retained inputs, so ps4_expanded_state stays PENDING.",
            "PS2 statuses are provisional target/horizon results, not a final selection.",
            "Byte inputs named by batch 003 digests.json are absent from this worktree and stay INCOMPLETE_EVIDENCE. They are not counted as zero features.",
        ],
        "metric_keys": list(METRICS),
        "metric_key_source": "feature-eng@1f1abdf tools/eurusd_ps/profile.py METRICS",
        "variant_feature_source": "feature-eng@1f1abdf tools/eurusd_ps/run_batch3.py variant_features",
    }
    if measured_states["MEASURED"] != len(emitted_ids) * len(METRICS):
        report["ps1_profile_coverage"] = "INCOMPLETE_EVIDENCE"
    return report, rows


def render_csv(rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in FIELDS})
    return buffer.getvalue().encode("utf-8")


def render_json(report):
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render(report, rows):
    return {
        "transform_feature_join.csv": render_csv(rows),
        "REPORT.json": render_json(report),
    }


def build(root):
    return join_loaded(load(root))


def emit(root):
    root = Path(root)
    report, rows = build(root)
    out = root / BASE / "ps4_transform_join"
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in render(report, rows).items():
        (out / name).write_bytes(payload)
    return report


def main():
    emit(Path(__file__).resolve().parents[5])


if __name__ == "__main__":
    main()
