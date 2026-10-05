#!/usr/bin/env python3
"""Ledger of PS0/PS1 source admission and transform prefix causality.

A prefix probe is causal computability. It is not predictive utility,
extractibility, or selection. A registry listing is not ingested bytes.
"""

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE = Path("docs/audits/evidence/canonical_20261003")
BATCHES = ("batch_001", "batch_002", "batch_003")
STATUSES = (
    "MEASURED",
    "DECLARED_ONLY",
    "NOT_AVAILABLE_FOR_TRAIN",
    "NOT_INGESTED",
    "NOT_APPLICABLE",
    "UNKNOWN",
    "PENDING_PROFILE",
)
SOURCE_FIELDS = (
    "row_index",
    "provider",
    "resource_identity",
    "frequency",
    "event_clock",
    "availability_clock",
    "timezone",
    "train_intersection",
    "contract_byte_status",
    "disposition",
    "reason",
    "next_action",
    "inventory_state",
    "duplicate_identity",
)
TRANSFORM_FIELDS = (
    "row_index",
    "variant_id",
    "family",
    "variant_class",
    "prefix_causality_result",
    "admissibility",
    "ps1_ps4_profile_state",
    "denominator",
    "pending_action",
    "causal_by_construction",
    "agrees_with_declaration",
    "note",
)
TRAILING_OR_FILTER = {
    "tv.wavelet_modwt_haar_causal",
    "tv.multitaper_trailing",
    "tv.hilbert_trailing_lastsample",
    "tv.stl_trailing_lastsample",
    "tv.kalman_local_level_filter",
}
GLOBAL_OR_SMOOTHER = {
    "tv.wavelet_dwt_db4_global",
    "tv.hilbert_global",
    "tv.stl_global",
    "tv.kalman_local_level_smoother",
}
COLUMN_ALIASES = {
    "features/trading_asset_data/eurusd": "lake_eurusd_5m",
    "economic_calendar/release_actuals/fxmacrodata": "fxmacrodata_announcements",
    "economic_calendar/scheduled_events/fxmacrodata": "fxmacrodata_release_calendar",
    "feature-eng:tests/data/economic_calendar_2011_2021.csv": "calendar_archive_2011_2021",
}
SPAN_RE = re.compile(r"(\d{4}-\d{2}(?:-\d{2})?(?:[ T]\d{2}:\d{2}:\d{2}(?:\+\d{2}:\d{2})?)?)")
MISSING_CLOCKS = {"", "UNDEFINED", "UNDEFINED_UNTIL_EVIDENCED", "per file"}


class AuditError(ValueError):
    """An input, a fixture, or a ledger invariant failed."""


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


def parse_dt(text):
    token = text.strip().replace("T", " ")
    if token.endswith("Z"):
        token = token[:-1] + "+00:00"
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
    ):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            continue
    raise AuditError(f"unparsed timestamp: {text}")


def to_comparable(value):
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.fromisoformat("2000-01-01T00:00:00+00:00").tzinfo)
    return value


def parse_span_blob(blob):
    found = SPAN_RE.findall(blob or "")
    if len(found) < 2:
        return None
    start, end = parse_dt(found[0]), parse_dt(found[1])
    return to_comparable(start), to_comparable(end)


def overlaps_train(span, train):
    if span is None:
        return None
    start, end = span
    train_start, train_end = train
    return start < train_end and end >= train_start


def clock_kind(availability, state):
    avail = (availability or "").strip()
    lowered = avail.lower()
    state_l = (state or "").lower()
    if avail in MISSING_CLOCKS:
        return "MISSING"
    if "ASSUMED" in avail.upper():
        return "ASSUMED"
    revised = any(token in state_l for token in (
        "revised macro", "no publication", "no_publication", "without vintage",
    ))
    if "undefined" in lowered and revised:
        return "MISSING"
    return "DECLARED"


def timezone_of(event, availability):
    blob = f"{event or ''} {availability or ''}"
    if "America/New_York" in blob or "NY local" in blob:
        return "America/New_York"
    if "UTC" in blob or "utc" in blob:
        return "UTC"
    return "UNKNOWN"


def input_paths():
    paths = [BASE / "laneA" / "batch_001" / name for name in (
        "inventory_sources.csv", "transform_variants.csv", "inventory_columns.csv",
        "coverage_matrix.csv", "batch_report.json", "digests.json", "contract.json",
    )]
    for batch in ("batch_002", "batch_003"):
        paths += [BASE / "laneA" / batch / name for name in (
            "inventory_columns.csv", "coverage_matrix.csv", "batch_report.json", "digests.json",
        )]
    paths.append(BASE / "coverage_reconciliation" / "coverage_reconciliation.json")
    return paths


def load_columns(root):
    grouped = {}
    for batch in BATCHES:
        for row in read_csv(root / BASE / "laneA" / batch / "inventory_columns.csv"):
            source = row.get("source", "")
            key = source[5:] if source.startswith("lake:") else source
            for suffix in ("/daily.parquet", "/observations.parquet"):
                if key.endswith(suffix):
                    key = key[: -len(suffix)]
            grouped.setdefault(key, []).append(row)
    return grouped


def column_evidence(path, grouped):
    key = COLUMN_ALIASES.get(path, path)
    rows = grouped.get(key) or []
    spans = []
    rows_in_train = []
    absent_columns = []
    for row in rows:
        span = parse_span_blob(row.get("span") or row.get("span_read") or "")
        if span is not None:
            spans.append(span)
        if row.get("rows_in_train", "") != "":
            rows_in_train.append(int(row["rows_in_train"]))
            non_null = row.get("non_null_in_train", "")
            if non_null != "" and int(non_null) == 0:
                absent_columns.append(row.get("column", ""))
    unique_spans = {(a.isoformat(), b.isoformat()) for a, b in spans}
    return {
        "matched": bool(rows),
        "span_conflict": len(unique_spans) > 1 or len(set(rows_in_train)) > 1,
        "span": spans[0] if spans else None,
        "rows_in_train": rows_in_train[0] if rows_in_train else None,
        "absent_columns": sorted(set(absent_columns)),
    }


def duplicate_notes(reports):
    notes = {}
    for report in reports:
        for item in report.get("skipped") or []:
            path = item.get("path", "")
            reason = item.get("reason", "")
            if path:
                notes[path] = reason
                stem = path
                for suffix in ("/daily.parquet", "/observations.parquet"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                notes.setdefault(stem, reason)
    return notes


def train_bounds(contract):
    start, end = contract["periods"]["train"]
    return to_comparable(parse_dt(start)), to_comparable(parse_dt(end))


def intersection_label(span, train, rows_in_train, zero_train):
    if zero_train:
        return "ZERO_TRAIN_ROWS"
    overlap = overlaps_train(span, train)
    if overlap is True:
        return "OBSERVATION_SPAN_OVERLAPS_TRAIN_NOT_AVAILABILITY"
    if overlap is False:
        return "DISJOINT_FROM_TRAIN"
    if rows_in_train == 0:
        return "ZERO_TRAIN_ROWS"
    return "NOT_ESTABLISHED"


def classify_source(row, evidence, train, same_bytes_reason):
    state = row.get("state", "")
    availability = row.get("availability_time", "")
    event = row.get("event_time", "")
    kind = clock_kind(availability, state)
    state_span = parse_span_blob(state)
    span = evidence["span"] or state_span
    zero_train = "0 TRAIN" in state
    label = intersection_label(span, train, evidence["rows_in_train"], zero_train)
    if kind == "ASSUMED" and evidence["rows_in_train"]:
        label = f"COLUMN_ROWS_IN_TRAIN_{evidence['rows_in_train']}_UNDER_ASSUMED_CLOCK"
    reason = state
    next_action = "Keep the inventory reason. Do not impute coverage."
    disposition = "PENDING"
    status = "UNKNOWN"

    if evidence["span_conflict"]:
        status = "UNKNOWN"
        label = "NOT_ESTABLISHED"
        disposition = "COLUMN_SPAN_CONFLICT"
        next_action = "Do not collapse conflicting column spans into one coverage figure."
    elif "NOT_APPLICABLE_TO_EURUSD" in state:
        status = "NOT_APPLICABLE"
        disposition = "OUTSIDE_EURUSD_DENOMINATOR"
        next_action = "Keep this resource out of the EURUSD denominator."
    elif state.startswith("NOT_INGESTED") or "NOT_INGESTED" in state:
        status = "NOT_INGESTED"
        disposition = "NO_BYTES"
        label = "NOT_ESTABLISHED"
        next_action = "A subscription or a resource listing is not historical bytes. Do not search for credentials."
    elif state.startswith("SUPERSEDED_BY"):
        status = "NOT_APPLICABLE"
        disposition = "SUPERSEDED"
        next_action = "Use the retained successor path. This row stays a separate inventory record."
    elif state.startswith("RECONCILED_ONLY"):
        status = "NOT_APPLICABLE"
        disposition = "RECONCILED_NOT_A_FEATURE_SOURCE"
        next_action = "Keep the clock-conversion check. Do not admit this path as a feature source."
    elif zero_train or (overlaps_train(state_span, train) is False):
        status = "NOT_AVAILABLE_FOR_TRAIN"
        disposition = "OUTSIDE_THIS_TRAIN"
        next_action = "Do not admit this resource to the retained TRAIN interval."
    elif kind == "MISSING":
        status = "UNKNOWN"
        disposition = "CLOCK_NOT_MEASURED"
        next_action = "Obtain a publication or availability clock. Do not convert the unknown into zero coverage or into a measured admission."
    elif kind == "ASSUMED":
        status = "DECLARED_ONLY"
        disposition = "ASSUMED_CLOCK"
        next_action = "Replace the assumed publication rule with a measured clock before calling the TRAIN rows usable."
    elif (
        "measured" in event.lower()
        and kind == "DECLARED"
        and evidence["rows_in_train"] not in (None, 0)
        and "IN_BATCH" in state
    ):
        status = "MEASURED"
        disposition = "IN_BATCH_MEASURED_BYTES"
        label = f"MEASURED_ROWS_IN_TRAIN_{evidence['rows_in_train']}"
        next_action = "The measured bytes stay inside the batch that already used them. This is not a new model measurement."
        if evidence["absent_columns"]:
            reason = state + " Absent non-null TRAIN columns: " + ",".join(evidence["absent_columns"]) + "."
            next_action = "OHLC rows are measured. Volume and spread stay absent and are not a cost source."
    else:
        status = "DECLARED_ONLY"
        disposition = "DECLARED_CLOCK_NOT_ADMITTED"
        next_action = "The clock rule is declared. Measure point-in-time availability against TRAIN before admission."

    if same_bytes_reason and status != "MEASURED":
        reason = state + " Retained duplicate note: " + same_bytes_reason
        if status == "DECLARED_ONLY":
            next_action = "Keep this path as its own row. The retained note names the profiled sibling and does not make this row measured."
    if status == "UNKNOWN" and label.startswith("OBSERVATION_SPAN_OVERLAPS"):
        next_action = "An observation span that meets TRAIN is not admission while the availability clock is missing."
    return {
        "provider": row.get("provider", ""),
        "resource_identity": row.get("path", ""),
        "frequency": row.get("frequency", ""),
        "event_clock": event,
        "availability_clock": availability,
        "timezone": timezone_of(event, availability),
        "train_intersection": label,
        "contract_byte_status": status,
        "disposition": disposition,
        "reason": reason,
        "next_action": next_action,
        "inventory_state": state,
        "duplicate_identity": "",
    }


def classify_transform(row, denominator):
    variant_id = row.get("variant_id", "")
    if variant_id in TRAILING_OR_FILTER:
        variant_class = "TRAILING_OR_FILTER"
    elif variant_id in GLOBAL_OR_SMOOTHER:
        variant_class = "GLOBAL_OR_SMOOTHER"
    else:
        variant_class = "UNKNOWN"
    probe = row.get("status", "")
    declared_admissible = str(row.get("admissible_as_feature", "")).strip() == "True"
    if probe == "PREFIX_VIOLATION_MEASURED" or not declared_admissible:
        admissibility = "NOT_ADMISSIBLE"
    elif probe == "PREFIX_INVARIANT_MEASURED" and declared_admissible:
        admissibility = "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY"
    else:
        admissibility = "UNKNOWN"
    profile = row.get("ps1_profile", "")
    profile_state = "PENDING_PROFILE" if "PENDING" in profile else "UNKNOWN"
    if admissibility == "NOT_ADMISSIBLE" and probe == "PREFIX_VIOLATION_MEASURED":
        pending = "Do not admit. The prefix probe measured future use. PENDING_PROFILE remains a profile state, not a rejection of the variant record."
    elif profile_state == "PENDING_PROFILE":
        pending = "PS1/PS4 profile remains pending. The prefix result is causal computability only, not predictive utility or selection."
    else:
        pending = "Profile state is not a retained pending marker. Do not invent a profile."
    return {
        "variant_id": variant_id,
        "family": row.get("family", ""),
        "variant_class": variant_class,
        "prefix_causality_result": probe,
        "admissibility": admissibility,
        "ps1_ps4_profile_state": profile_state,
        "denominator": str(denominator),
        "pending_action": pending,
        "causal_by_construction": row.get("causal_by_construction", ""),
        "agrees_with_declaration": row.get("agrees_with_declaration", ""),
        "note": row.get("note", ""),
    }


def mark_duplicate_identities(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["resource_identity"], []).append(row)
    for ident, group in grouped.items():
        if len(group) < 2:
            continue
        states = {item["inventory_state"] for item in group}
        flag = "CONFLICT" if len(states) > 1 else "REPEATED_PATH"
        for item in group:
            item["duplicate_identity"] = flag
    return rows


def require_source_completeness(inputs, outputs):
    if len(inputs) != len(outputs):
        raise AuditError(f"omitted source: inputs {len(inputs)} outputs {len(outputs)}")
    for index, (src, out) in enumerate(zip(inputs, outputs)):
        if src.get("path") != out["resource_identity"] or src.get("provider") != out["provider"]:
            raise AuditError(f"omitted or reordered source at row {index}")
        if out["contract_byte_status"] not in STATUSES:
            raise AuditError(f"status outside the ledger vocabulary: {out['contract_byte_status']}")


def require_measured_clocks(outputs):
    for row in outputs:
        if row["contract_byte_status"] != "MEASURED":
            continue
        kind = clock_kind(row["availability_clock"], row["inventory_state"])
        if kind != "DECLARED":
            raise AuditError("missing or assumed availability clock presented as measured")
        if not str(row["train_intersection"]).startswith("MEASURED_ROWS_IN_TRAIN_"):
            raise AuditError("measured status without a measured TRAIN row count")


def require_train_labels(outputs, train):
    for row in outputs:
        label = row["train_intersection"]
        if "OVERLAPS" in label and "NOT_AVAILABILITY" not in label and not label.startswith("MEASURED_ROWS"):
            raise AuditError(f"train overlap label is not tied to evidence: {label}")
        state_span = parse_span_blob(row["inventory_state"])
        if overlaps_train(state_span, train) is False and "OVERLAPS" in label:
            raise AuditError("fake TRAIN overlap on a disjoint inventory span")
        if "ZERO_TRAIN_ROWS" in row["inventory_state"] or "0 TRAIN" in row["inventory_state"]:
            if row["contract_byte_status"] == "MEASURED" or "OVERLAPS" in label:
                raise AuditError("zero TRAIN rows presented as overlap or measured")


def require_transform_rules(inputs, outputs):
    if len(inputs) != len(outputs):
        raise AuditError("transform denominator does not match input rows")
    for src, out in zip(inputs, outputs):
        if out["denominator"] != str(len(inputs)):
            raise AuditError("transform denominator is not the input row count")
        if src.get("status") == "PREFIX_VIOLATION_MEASURED" and out["admissibility"] != "NOT_ADMISSIBLE":
            raise AuditError("prefix violation presented as admissible")
        if "PENDING" in (src.get("ps1_profile") or "") and out["ps1_ps4_profile_state"] != "PENDING_PROFILE":
            raise AuditError("pending profile converted to a rejection")


def registry_path(root):
    try:
        common = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (Path(root) / common_path).resolve()
    candidate = common_path.parent.parent / "data-gov" / "docs" / "08_RESOURCE_REGISTRY.md"
    return candidate if candidate.is_file() else None


def registry_facts(root):
    registry = registry_path(root)
    if registry is None:
        return {"status": "NOT_AVAILABLE", "interpretation": "No sibling registry file was read."}
    text = registry.read_text(encoding="utf-8")
    declared = []
    for line in text.splitlines():
        if "fxmacrodata" not in line.lower():
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if len(cells) >= 4:
            declared.append({
                "resource": cells[0],
                "span": cells[2] if len(cells) > 2 else "",
                "clock": cells[3] if len(cells) > 3 else "",
            })
    commit = "UNKNOWN"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(registry.parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        commit = "UNKNOWN"
    return {
        "status": "LISTING_ONLY",
        "repository_commit": commit,
        "relative_path": "data-gov/docs/08_RESOURCE_REGISTRY.md",
        "sha256": digest(registry),
        "alpaca_mentions": text.lower().count("alpaca"),
        "yahoo_mentions": text.lower().count("yahoo"),
        "fxmacro_declared_rows": declared,
        "interpretation": "A registry row or an API subscription is not ingested historical bytes.",
    }


def build(root):
    root = Path(root)
    sources = read_csv(root / BASE / "laneA" / "batch_001" / "inventory_sources.csv")
    transforms = read_csv(root / BASE / "laneA" / "batch_001" / "transform_variants.csv")
    contract = read_json(root / BASE / "laneA" / "batch_001" / "contract.json")
    train = train_bounds(contract)
    columns = load_columns(root)
    reports = [read_json(root / BASE / "laneA" / batch / "batch_report.json") for batch in BATCHES]
    notes = duplicate_notes(reports)
    classified = []
    for index, row in enumerate(sources):
        evidence = column_evidence(row.get("path", ""), columns)
        item = classify_source(row, evidence, train, notes.get(row.get("path", ""), ""))
        item["row_index"] = str(index)
        classified.append(item)
    mark_duplicate_identities(classified)
    transform_rows = []
    for index, row in enumerate(transforms):
        item = classify_transform(row, len(transforms))
        item["row_index"] = str(index)
        transform_rows.append(item)
    require_source_completeness(sources, classified)
    require_measured_clocks(classified)
    require_train_labels(classified, train)
    require_transform_rules(transforms, transform_rows)
    reconciliation = read_json(root / BASE / "coverage_reconciliation" / "coverage_reconciliation.json")
    summary = reconciliation["summary"]
    matrix_counts = {}
    for batch in BATCHES:
        matrix_counts[batch] = len(read_csv(root / BASE / "laneA" / batch / "coverage_matrix.csv"))
    joined = set()
    for row in sources:
        evidence = column_evidence(row.get("path", ""), columns)
        if evidence["matched"]:
            joined.add(COLUMN_ALIASES.get(row.get("path", ""), row.get("path", "")))
    unjoined = sorted(set(columns) - joined)
    by_status = Counter(row["contract_byte_status"] for row in classified)
    by_provider = {}
    for row in classified:
        bucket = by_provider.setdefault(row["provider"], Counter())
        bucket[row["contract_byte_status"]] += 1
    report = {
        "schema": "canonical_20261003.source_transform_coverage.v1",
        "measurement_flag": "NO_NEW_MODEL_MEASUREMENT",
        "train_interval": {
            "start": contract["periods"]["train"][0],
            "end": contract["periods"]["train"][1],
            "end_rule": "exclusive",
            "source": "laneA/batch_001/contract.json periods.train",
        },
        "denominators": {
            "inventory_source_rows": len(sources),
            "inventory_source_rows_are_not_selected_features": True,
            "feature_population_admissible": summary.get("admissible_features"),
            "feature_population_ps3c_join": summary.get("ps3c_join_features"),
            "feature_population_low_priority_outside_join": summary.get("low_priority_outside_join"),
            "transform_variant_rows": len(transforms),
            "coverage_matrix_rows_by_batch": matrix_counts,
        },
        "source_status_counts": dict(by_status),
        "source_status_by_provider": {key: dict(value) for key, value in sorted(by_provider.items())},
        "transform_admissibility_counts": dict(Counter(row["admissibility"] for row in transform_rows)),
        "transform_profile_counts": dict(Counter(row["ps1_ps4_profile_state"] for row in transform_rows)),
        "transform_class_counts": dict(Counter(row["variant_class"] for row in transform_rows)),
        "inputs_sha256": {str(path): digest(root / path) for path in input_paths()},
        "resource_contracts": registry_facts(root),
        "unjoined_column_sources": unjoined,
        "unresolved_blockers": [
            "Alpaca remains NOT_INGESTED. No credential search was done.",
            "Both FXMacroData resources have zero TRAIN rows on this contract interval.",
            "Yahoo Finance clocks stay declared conservative rules, not measured publication instants.",
            "Revised macro with an undefined publication clock stays UNKNOWN, even where an observation span meets TRAIN.",
            "PS1/PS4 profiles of every transform variant stay PENDING_PROFILE.",
            "Prefix-invariant variants are causal-computability results, not selected features.",
            "Column sources without a unique inventory path are listed in unjoined_column_sources and are not given a TRAIN admission.",
        ],
        "feature_population_note": "366 = 279 + 87 is the retained feature partition. It is not the source denominator and not proof that subscriptions or transform variants were admitted.",
    }
    return report, classified, transform_rows


def render_csv(rows, fields):
    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue().encode("utf-8")


def render_json(report):
    return (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")


def render(report, sources, transforms):
    return {
        "source_coverage.csv": render_csv(sources, SOURCE_FIELDS),
        "transform_coverage.csv": render_csv(transforms, TRANSFORM_FIELDS),
        "REPORT.json": render_json(report),
    }


def emit(root):
    root = Path(root)
    report, sources, transforms = build(root)
    out = root / BASE / "source_transform_coverage"
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in render(report, sources, transforms).items():
        (out / name).write_bytes(payload)
    return report


def main():
    emit(Path(__file__).resolve().parents[5])


if __name__ == "__main__":
    main()
