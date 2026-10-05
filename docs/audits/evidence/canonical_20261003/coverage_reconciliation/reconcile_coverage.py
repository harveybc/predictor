#!/usr/bin/env python3
"""Reconcile PS2 admissibility, PS3-C dossiers, and Lane E queue coverage."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

BATCHES = ("batch_001", "batch_002", "batch_003")
BASE = Path("docs/audits/evidence/canonical_20261003")
TIERS = ("tier_1", "tier_2", "tier_3", "exploration")
CODE_STATUS = {"S": "PROVISIONAL_SURVIVOR", "L": "PROVISIONAL_LOW_PRIORITY",
               "X": "EXPLORATION", "T": "TECHNICAL_REJECT"}
EXPECTED = {"admissible_features": 366, "ps3c_join_features": 279,
            "low_priority_outside_join": 87, "extractibility_queue_features": 137,
            "join_features_outside_extractibility_queue": 142,
            "outside_queue_tier3_only": 132,
            "outside_queue_calendar_conditioning": 10, "unaccounted_features": 0}
RUNGS = ("rung1", "rung2", "rung3")


class AuditError(ValueError):
    """An input or reconciliation invariant failed."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_paths():
    paths = []
    for batch in BATCHES:
        paths += [BASE / "laneB" / batch / name for name in (
            "ps2_manifest.json", "ps2_status.csv", "ps2_status_matrix.csv",
            "ps2_counts.csv", "ps2_candidates_lane_c.json",
            "ps2_extractor_priority.json", "ps2_exploration.json")]
        paths += [BASE / "laneC" / batch / name for name in (
            "batch_report.json", "dossiers.jsonl.gz", "digests.json", "READY")]
        paths.append(BASE / "laneC" / f"join_{batch}.json")
    paths += [BASE / "laneE" / name for name in ("queue.json", "extractibility_matrix.csv")]
    return paths


def normalized_state(rung, payload):
    state = payload["state"]
    estimate = {
        "rung1": "ASSOCIATION_REPORTED",
        "rung2": "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS",
        "rung3": "COUNTERFACTUAL_UNDER_DECLARED_SCM",
    }
    if state == estimate[rung]:
        return "ESTIMATED"
    if state == "NOT_EVALUATED":
        if "KNOWN_IN_ADVANCE_CALENDAR_NOT_AN_OBSERVED_INTERVENTION" in payload.get("reasons", []):
            return "NOT_APPLICABLE"
        return "NOT_IDENTIFIED"
    return state


def load_ps2(root, batch):
    folder = root / BASE / "laneB" / batch
    manifest = read_json(folder / "ps2_manifest.json")
    priority = read_json(folder / "ps2_extractor_priority.json")
    exploration = read_json(folder / "ps2_exploration.json")
    payload = read_json(folder / "ps2_candidates_lane_c.json")
    admissible_list = manifest["control_all_admissible"]
    admissible = set(admissible_list)
    if len(admissible) != manifest["n_admissible"] or len(admissible) != len(admissible_list):
        raise AuditError(f"{batch}: admissible count or uniqueness mismatch")

    matrix = read_csv(folder / "ps2_status_matrix.csv")
    cells = [key for key in matrix[0] if key.startswith("Y_")]
    if len(cells) != 14:
        raise AuditError(f"{batch}: expected 14 PS2 cells, got {len(cells)}")
    matrix_by_id = {row["feature"]: row for row in matrix}
    if len(matrix_by_id) != len(matrix) or not admissible <= set(matrix_by_id):
        raise AuditError(f"{batch}: duplicate/missing feature in PS2 matrix")
    rejects = set(manifest.get("technical_rejects", {}))
    if set(matrix_by_id) - admissible != rejects:
        raise AuditError(f"{batch}: non-admissible matrix rows differ from technical rejects")

    status_cells = {}
    for feature, row in matrix_by_id.items():
        for cell in cells:
            code = row[cell]
            if code not in CODE_STATUS:
                raise AuditError(f"{batch}: unknown PS2 status {code!r}")
            status_cells[(feature, cell)] = CODE_STATUS[code]
        if feature in rejects and any(row[cell] != "T" for cell in cells):
            raise AuditError(f"{batch}: technical reject not marked T in every cell")

    long_rows = read_csv(folder / "ps2_status.csv")
    long_status = {}
    for row in long_rows:
        key = (row["feature"], f"{row['target']}_h{row['horizon']}")
        if key in long_status:
            raise AuditError(f"{batch}: duplicate PS2 status row {key}")
        long_status[key] = row["status"]
    if long_status != status_cells:
        raise AuditError(f"{batch}: PS2 long and matrix status disagree")

    cell_counts = Counter()
    for (feature, cell), status in status_cells.items():
        target, horizon = cell.rsplit("_h", 1)
        cell_counts[(target, horizon, status)] += 1
    count_rows = read_csv(folder / "ps2_counts.csv")
    if len(count_rows) != 14:
        raise AuditError(f"{batch}: PS2 counts must cover 14 cells")
    for row in count_rows:
        for status in ("PROVISIONAL_SURVIVOR", "PROVISIONAL_LOW_PRIORITY", "EXPLORATION", "TECHNICAL_REJECT"):
            if int(row[status]) != cell_counts[(row["target"], row["horizon"], status)]:
                raise AuditError(f"{batch}: PS2 cell counts disagree for {row['target']}/{row['horizon']}")

    tiers = {tier: set(priority[tier]) for tier in TIERS}
    sampled = set(exploration["sample"])
    if sampled != set(manifest.get("exploration_sample", [])) or sampled != tiers["exploration"]:
        raise AuditError(f"{batch}: exploration identities disagree")
    candidate_ids = {item["feature_id"] for item in payload["candidates"]}
    union = set.union(*tiers.values())
    if candidate_ids != union or set(manifest["extractor_worklist"]) != union:
        raise AuditError(f"{batch}: candidate, tier union, and extractor worklist disagree")
    if not candidate_ids <= admissible:
        raise AuditError(f"{batch}: candidate outside admissible set")

    by_feature_status = {
        feature: Counter(status_cells[(feature, cell)] for cell in cells)
        for feature in admissible
    }
    return {"manifest": manifest, "admissible": admissible, "rejects": rejects,
            "candidate_ids": candidate_ids, "tiers": tiers,
            "status_counts": by_feature_status}


def load_dossiers(root, batch):
    path = root / BASE / "laneC" / batch / "dossiers.jsonl.gz"
    features = defaultdict(lambda: {r: Counter() for r in RUNGS})
    normalized_features = defaultdict(lambda: {r: Counter() for r in RUNGS})
    events, normalized_events = Counter(), Counter()
    seen, event_count = set(), 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            dossier = json.loads(line)
            subject = dossier["subject"]
            event_type = str(subject.get("event_type", ""))
            if event_type.startswith("feature:"):
                feature = event_type.removeprefix("feature:")
                key = (feature, str(subject["target"]), str(subject["horizon_minutes"]))
                if key in seen:
                    raise AuditError(f"{batch}: duplicate dossier cell {key}")
                seen.add(key)
                for rung in RUNGS:
                    features[feature][rung][dossier[rung]["state"]] += 1
                    normalized_features[feature][rung][normalized_state(rung, dossier[rung])] += 1
            else:
                event_count += 1
                for rung in RUNGS:
                    events[(rung, dossier[rung]["state"])] += 1
                    normalized_events[(rung, normalized_state(rung, dossier[rung]))] += 1
    return features, normalized_features, events, normalized_events, event_count


def counts(counter):
    return dict(sorted(counter.items()))


def build_report(repo_root):
    root = Path(repo_root).resolve()
    inputs = source_paths()
    missing = [str(p) for p in inputs if not (root / p).is_file()]
    if missing:
        raise AuditError(f"missing inputs: {missing}")
    input_hashes = {str(p): digest(root / p) for p in inputs}

    queue_data = read_json(root / BASE / "laneE" / "queue.json")["queue"]
    queue = {}
    for item in queue_data:
        key = (item["batch"], item["feature"])
        if key in queue or item["stage"] not in {"tier_1", "tier_2", "exploration"}:
            raise AuditError(f"duplicate or invalid Lane E queue row: {key}")
        queue[key] = item["stage"]
    matrix_rows = read_csv(root / BASE / "laneE" / "extractibility_matrix.csv")
    matrix_keys = {(row["batch"], row["feature"]) for row in matrix_rows}
    if matrix_keys != set(queue):
        raise AuditError("Lane E queue and matrix feature identities disagree")

    rows, batch_summaries = [], {}
    all_admissible, all_joined, all_queued = set(), set(), set()
    feature_ladder = {r: Counter() for r in RUNGS}
    joined_ladder = {r: Counter() for r in RUNGS}
    outside_ladder = {r: Counter() for r in RUNGS}
    event_ladder = {r: Counter() for r in RUNGS}
    batch_report_normalized = {}
    event_total = 0

    for batch in BATCHES:
        ps2 = load_ps2(root, batch)
        lane_c = root / BASE / "laneC"
        join = read_json(lane_c / f"join_{batch}.json")
        ready = read_json(lane_c / batch / "READY")
        digests_path = lane_c / batch / "digests.json"
        if ready["digests_sha256"] != digest(digests_path):
            raise AuditError(f"{batch}: READY digest does not authenticate digests.json")
        if join["ps3c_ready_digests_sha256"] != ready["digests_sha256"]:
            raise AuditError(f"{batch}: join does not reference its READY digest")
        joined = {row["feature_id"] for row in join["join"]}
        if joined != ps2["candidate_ids"] or join["candidates"] != len(joined) or join["prioritized"] != len(joined):
            raise AuditError(f"{batch}: PS3-C join and PS2 candidate identities/count disagree")
        if join["candidates_without_any_dossier"]:
            raise AuditError(f"{batch}: join has candidate dossiers missing")
        if join["tier_counts"] != {t: len(ps2["tiers"][t]) for t in TIERS}:
            raise AuditError(f"{batch}: join tier memberships disagree with PS2")

        dossiers, norm_dossiers, events, norm_events, event_count = load_dossiers(root, batch)
        if set(dossiers) != ps2["admissible"]:
            raise AuditError(f"{batch}: dossier feature universe differs from admissible set")
        if any(sum(states[r].values()) != 14 for states in dossiers.values() for r in RUNGS):
            raise AuditError(f"{batch}: feature dossier does not cover 14 cells at every rung")

        report = read_json(lane_c / batch / "batch_report.json")
        den = report["denominators"]
        expected_events = den["cells"] - den["features_role_feature_evaluated"] * den["targets"]
        if event_count != expected_events:
            raise AuditError(f"{batch}: event dossier denominator mismatch")
        event_total += event_count

        raw_all, norm_all = {r: Counter() for r in RUNGS}, {r: Counter() for r in RUNGS}
        for feature in dossiers:
            for rung in RUNGS:
                raw_all[rung].update(dossiers[feature][rung])
                norm_all[rung].update(norm_dossiers[feature][rung])
        for rung in RUNGS:
            raw_all[rung].update({s: n for (rr, s), n in events.items() if rr == rung})
            norm_all[rung].update({s: n for (rr, s), n in norm_events.items() if rr == rung})
            if counts(raw_all[rung]) != report["per_rung_raw_counts"][rung]:
                raise AuditError(f"{batch}: raw dossier states disagree with batch report ({rung})")
            if counts(norm_all[rung]) != report["per_rung_counts"][rung]:
                raise AuditError(f"{batch}: reason-based normalized states disagree with report ({rung})")

        for rung in RUNGS:
            event_ladder[rung].update({s: n for (rr, s), n in events.items() if rr == rung})
        for feature in sorted(ps2["admissible"]):
            key = (batch, feature)
            all_admissible.add(key)
            in_join = feature in joined
            stage = queue.get(key, "")
            in_queue = bool(stage)
            if in_queue and not in_join:
                raise AuditError(f"{batch}: Lane E queue contains non-join feature {feature}")
            status = ps2["status_counts"][feature]
            tiers = [t for t in TIERS if feature in ps2["tiers"][t]]
            if not in_join:
                if status != Counter({"PROVISIONAL_LOW_PRIORITY": 14}):
                    raise AuditError(f"{batch}: outside-join feature is not low priority in all cells: {feature}")
                coverage = "LOW_PRIORITY_OUTSIDE_PS3C_JOIN"
                join_reason = "PROVISIONAL_LOW_PRIORITY_IN_ALL_14_CELLS"
                extraction_reason = "NOT_APPLICABLE_OUTSIDE_PS3C_JOIN"
            elif in_queue:
                coverage, join_reason, extraction_reason = "IN_EXTRACTIBILITY_QUEUE", "IN_PS3C_JOIN", "IN_LANE_E_QUEUE"
            elif feature.startswith("cal."):
                coverage = "JOINED_CALENDAR_CONDITIONING_NOT_EXTRACTED"
                join_reason, extraction_reason = "IN_PS3C_JOIN", "CALENDAR_CONDITIONING_NOT_EXTRACTOR_SERIES"
            elif feature in ps2["tiers"]["tier_3"] and not any(
                feature in ps2["tiers"][t] for t in ("tier_1", "tier_2", "exploration")
            ):
                coverage = "JOINED_TIER3_ONLY_NOT_IN_EXTRACTIBILITY_QUEUE"
                join_reason, extraction_reason = "IN_PS3C_JOIN", "TIER_3_ONLY_DEFERRED_BY_QUEUE_RULE"
            else:
                raise AuditError(f"{batch}: unexplained Lane E omission: {feature}")

            rung_counts = dossiers[feature]
            for rung in RUNGS:
                feature_ladder[rung].update(rung_counts[rung])
                (joined_ladder if in_join else outside_ladder)[rung].update(rung_counts[rung])
            rows.append({
                "batch": batch, "feature_id": feature, "admissible": True,
                "ps2_cell_count": 14, "ps2_status_counts": counts(status),
                "in_ps3c_join": in_join, "outside_join_reason": join_reason,
                "ps2_priority_memberships": tiers,
                "in_extractibility_queue": in_queue, "extractibility_queue_stage": stage,
                "extractibility_omission_reason": extraction_reason,
                "coverage_status": coverage, "dossier_count": 14,
                **{f"{rung}_state_counts": counts(rung_counts[rung]) for rung in RUNGS},
            })
            if in_join:
                all_joined.add(key)
            if in_queue:
                all_queued.add(key)

        tier_sets = ps2["tiers"]
        union = set.union(*tier_sets.values())
        batch_summaries[batch] = {
            "admissible_features": len(ps2["admissible"]),
            "ps3c_join_features": len(joined),
            "low_priority_outside_join": len(ps2["admissible"] - joined),
            "tier_membership_counts": {t: len(tier_sets[t]) for t in TIERS},
            "tier_union": len(union),
            "tier_membership_overlap_count": sum(map(len, tier_sets.values())) - len(union),
            "extractibility_queue_features": sum(k[0] == batch for k in queue),
            "feature_dossier_count": len(dossiers) * 14,
            "event_dossier_count": event_count,
            "raw_feature_rung_states": {r: counts(sum((dossiers[f][r] for f in dossiers), Counter())) for r in RUNGS},
            "batch_report_normalized_rung_states": {r: counts(norm_all[r]) for r in RUNGS},
        }

    joined_not_queued = all_joined - all_queued
    calendar_out = {(r["batch"], r["feature_id"]) for r in rows
                    if r["extractibility_omission_reason"] == "CALENDAR_CONDITIONING_NOT_EXTRACTOR_SERIES"}
    tier3_out = {(r["batch"], r["feature_id"]) for r in rows
                 if r["extractibility_omission_reason"] == "TIER_3_ONLY_DEFERRED_BY_QUEUE_RULE"}
    overlaps = {}
    for batch in BATCHES:
        p = read_json(root / BASE / "laneB" / batch / "ps2_extractor_priority.json")
        groups = {t: set(p[t]) for t in TIERS}
        patterns = Counter("|".join(sorted(t for t in TIERS if f in groups[t])) for f in set.union(*groups.values()))
        overlaps[batch] = dict(sorted(patterns.items()))

    summary = {
        "admissible_features": len(all_admissible), "ps3c_join_features": len(all_joined),
        "low_priority_outside_join": len(all_admissible - all_joined),
        "extractibility_queue_features": len(all_queued),
        "join_features_outside_extractibility_queue": len(joined_not_queued),
        "outside_queue_tier3_only": len(tier3_out),
        "outside_queue_calendar_conditioning": len(calendar_out),
        "unaccounted_features": len(joined_not_queued - tier3_out - calendar_out),
        "not_in_extractibility_queue_total": len(all_admissible - all_queued),
        "technical_reject_features_outside_admissible_denominator": sum(
            len(load_ps2(root, b)["rejects"]) for b in BATCHES),
        "separate_batch001_selector_episode_sources": read_json(root / BASE / "laneC/batch_001/batch_report.json")["selector_episode_sources_not_feature_candidates"]["n"],
        "separate_ps3c_event_dossiers": event_total,
        "feature_ladder_states": {r: counts(c) for r, c in feature_ladder.items()},
        "joined_feature_ladder_states": {r: counts(c) for r, c in joined_ladder.items()},
        "outside_join_feature_ladder_states": {r: counts(c) for r, c in outside_ladder.items()},
        "separate_event_ladder_states": {r: counts(c) for r, c in event_ladder.items()},
        "tier_overlap_membership_patterns": overlaps,
        "batches": batch_summaries,
    }
    report = {
        "schema": "canonical_20261003.coverage_reconciliation.v1",
        "scope": "Three PS2/PS3-C batches and Lane E queue; not a global inventory claim.",
        "interpretation": {
            "decision_status": "Eligibility, joins, evidence states, and queue membership only; no final feature inclusion decision.",
            "rung1": "ASSOCIATION_REPORTED is associational, not a causal effect.",
            "rung2": "IDENTIFIED_CONDITIONAL_ON_DECLARED_ASSUMPTIONS is conditional on declared assumptions and support.",
            "rung3": "COUNTERFACTUAL_UNDER_DECLARED_SCM is conditional on the declared structural causal model.",
            "normalization": "Report ESTIMATED maps to the dossier's rung-specific estimated state; calendar NOT_EVALUATED maps to NOT_APPLICABLE; other NOT_EVALUATED maps to NOT_IDENTIFIED. Table rows retain raw states.",
        },
        "input_sha256": {str(p): digest(root / p) for p in inputs},
        "summary": summary,
        "features": rows,
    }
    validate_report(report)
    return report


def validate_report(report):
    rows = report["features"]
    keys = {(r["batch"], r["feature_id"]) for r in rows}
    if len(keys) != len(rows) or len(rows) != EXPECTED["admissible_features"]:
        raise AuditError("feature table has duplicate keys or wrong denominator")
    if any(not r["admissible"] or r["ps2_cell_count"] != 14 or r["dossier_count"] != 14 for r in rows):
        raise AuditError("feature row is inadmissible or lacks 14 cells")
    joined = [r for r in rows if r["in_ps3c_join"]]
    outside = [r for r in rows if not r["in_ps3c_join"]]
    queue = [r for r in rows if r["in_extractibility_queue"]]
    join_not_queue = [r for r in rows if r["in_ps3c_join"] and not r["in_extractibility_queue"]]
    actual = {"admissible_features": len(rows), "ps3c_join_features": len(joined),
              "low_priority_outside_join": len(outside), "extractibility_queue_features": len(queue),
              "join_features_outside_extractibility_queue": len(join_not_queue),
              "outside_queue_tier3_only": sum(r["extractibility_omission_reason"] == "TIER_3_ONLY_DEFERRED_BY_QUEUE_RULE" for r in join_not_queue),
              "outside_queue_calendar_conditioning": sum(r["extractibility_omission_reason"] == "CALENDAR_CONDITIONING_NOT_EXTRACTOR_SERIES" for r in join_not_queue),
              "unaccounted_features": sum(r["extractibility_omission_reason"] not in {"TIER_3_ONLY_DEFERRED_BY_QUEUE_RULE", "CALENDAR_CONDITIONING_NOT_EXTRACTOR_SERIES"} for r in join_not_queue)}
    if actual != EXPECTED:
        raise AuditError(f"coverage counts differ: {actual}")
    for row in outside:
        if row["coverage_status"] != "LOW_PRIORITY_OUTSIDE_PS3C_JOIN" or row["outside_join_reason"] != "PROVISIONAL_LOW_PRIORITY_IN_ALL_14_CELLS" or row["ps2_status_counts"] != {"PROVISIONAL_LOW_PRIORITY": 14}:
            raise AuditError(f"unexplained join omission: {row['feature_id']}")
    if any(not r["extractibility_queue_stage"] or r["coverage_status"] != "IN_EXTRACTIBILITY_QUEUE" for r in queue):
        raise AuditError("queue feature lacks a stage/status")
    if report["summary"]["not_in_extractibility_queue_total"] != 229:
        raise AuditError("outside-queue total does not reconcile")


CSV_FIELDS = ("batch", "feature_id", "admissible", "ps2_cell_count", "ps2_status_counts_json",
              "in_ps3c_join", "outside_join_reason", "ps2_priority_memberships_json",
              "in_extractibility_queue", "extractibility_queue_stage", "extractibility_omission_reason",
              "coverage_status", "dossier_count", "rung1_state_counts_json", "rung2_state_counts_json",
              "rung3_state_counts_json")


def write_outputs(report, output_dir):
    validate_report(report)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coverage_reconciliation.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "coverage_by_batch_feature.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in report["features"]:
            out = dict(row)
            for key in ("ps2_status_counts", "ps2_priority_memberships", *(f"{r}_state_counts" for r in RUNGS)):
                out[key + "_json"] = json.dumps(out.pop(key), sort_keys=True, separators=(",", ":"))
            writer.writerow(out)


def main():
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=here.parents[5])
    parser.add_argument("--output-dir", type=Path, default=here.parent)
    args = parser.parse_args()
    report = build_report(args.repo_root)
    write_outputs(report, args.output_dir)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
