"""Independent retained-array audit: temporal support, business masks and metrics.

This is CPU-only evidence, not a claim that upstream vendor vintages or feature
producers are verified. No fitting, TensorFlow import, TEST or campaign writes.
"""
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from tools.i6dmo import common_population as CP, provenance, worker
from tools.i6dmo.prediction_diagnostics import load_fit, describe
from tools.i6n_swarm.common import file_sha256


def epoch(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


def check(condition, code):
    if not condition:
        raise ValueError(code)


def audit_arrays(raw, arrays, calendar, split, horizons, spacing):
    """Re-derive windows/labels/support without the production window or C1 code."""
    ts, x, price = raw["ts"], raw["x"], raw["price"]
    rid, slots, mask = arrays["row_id"], arrays["slot_instants"], arrays["head_mask"]
    check(np.array_equal(slots[:, -1], rid), "ORIGIN_NOT_INCLUDED")
    check(bool(np.all(slots <= rid[:, None])), "FUTURE_INPUT")
    ix = np.searchsorted(ts, slots)
    check(bool(np.all(ix < len(ts))) and np.array_equal(ts[ix], slots), "SYNTHETIC_SLOT")
    check(np.array_equal(x[ix], arrays["x"]), "INPUT_VALUES_DIFFER_FROM_SOURCE")
    origins = np.searchsorted(ts, rid)
    check(np.array_equal(price[origins], arrays["origin_price"]), "ORIGIN_PRICE_MISMATCH")
    offsets = (slots - rid[:, None]) / 3600
    check(np.array_equal(offsets, arrays["offsets_hours"]), "OFFSETS_MISMATCH")
    adjacent = np.concatenate((np.zeros((len(rid), 1)), np.diff(slots, axis=1) / 3600), axis=1)
    check(np.array_equal(adjacent, arrays["adjacent_delta_hours"]), "ADJACENT_MISMATCH")
    expected_closures = (adjacent > spacing).astype(int)
    check(np.array_equal(expected_closures, arrays["closure_crossed"]), "CLOSURE_FLAGS_MISMATCH")
    weeks = calendar["market_calendar"]["weeks"]
    opens = np.array([epoch(w["market_open"]) for w in weeks])
    closes = np.array([epoch(w["market_close"]) for w in weeks])
    cuts = np.array([epoch(w["entry_cutoff"]) for w in weeks])
    # The expected bar-END grid excludes market-open instants and includes market-close.
    grid = np.concatenate([np.arange(o + 3600, c + 1, 3600, dtype=np.int64) for o, c in zip(opens, closes)])
    ranks = np.searchsorted(grid, slots)
    check(bool(np.all(ranks < len(grid))) and np.array_equal(grid[ranks], slots), "CLOSED_HOUR_INPUT")
    check(bool(np.all(np.diff(ranks, axis=1) == spacing)), "WRONG_OPEN_MARKET_SPACING")
    wi = np.searchsorted(opens, rid, side="right") - 1
    check(bool(np.all(wi >= 0)), "OUTSIDE_CALENDAR")
    a = opens[wi] + calendar["entry_delay_after_open_hours"] * 3600
    b = closes[wi] - calendar["force_flat_before_close_hours"] * 3600
    q = rid[:, None] + np.array(horizons)[None, :] * 3600
    qi = np.searchsorted(ts, q)
    observed = (qi < len(ts)) & (ts[np.minimum(qi, len(ts) - 1)] == q)
    expected = ((rid >= a) & (rid < b))[:, None] & (q < b[:, None]) & observed
    expected &= (q >= epoch(split["start"])) & (q < epoch(split["end"]))
    expected &= q <= epoch(split["label_available_by"])
    check(np.array_equal(expected, mask), "BUSINESS_HEAD_MASK_MISMATCH")
    check(np.array_equal(mask, arrays["management_mask"]), "MANAGEMENT_MASK_MISMATCH")
    check(np.array_equal(mask & (rid < cuts[wi])[:, None], arrays["entry_mask"]), "ENTRY_MASK_MISMATCH")
    check(np.array_equal(price[qi[mask]], arrays["target_price"][mask]), "TARGET_NOT_EXACT_T_PLUS_H")
    return dict(origins=len(rid), supported_labels=int(mask.sum()), input_slots=int(slots.size),
                origin_included=True, future_input_slots=0, synthetic_slots=0, closed_hour_slots=0,
                business_mask_mismatches=0, target_price_mismatches=0)


def audit(sequence_root, input_dir, storage_root):
    """Audit every FIT/INNER/DEVELOPMENT row in both selected input recipes."""
    sequence_root, input_dir, storage_root = map(Path, (sequence_root, input_dir, storage_root))
    with np.load(input_dir / "INPUT.npz", allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    check(bool(np.all(np.diff(raw["ts"]) > 0)), "SOURCE_TIME_ORDER")
    check(bool(np.all(raw["ts"] < 1735689600)), "TEST_SOURCE_FORBIDDEN")
    calendar = json.loads((input_dir / "CALENDAR.json").read_text())
    output = dict(schema="i6dmo.prediction_causal_audit.v1", fits_started=0, test_open=False,
                  source_sha256=file_sha256(input_dir / "INPUT.npz"),
                  calendar_sha256=file_sha256(input_dir / "CALENDAR.json"),
                  offsets={k: calendar[k] for k in ("entry_delay_after_open_hours", "force_flat_before_close_hours")},
                  families={}, qualification="ARRAY_AND_METRIC_CHECKS_ONLY_NOT_END_TO_END_CAUSAL_CERTIFICATION")
    for family in ("short", "long"):
        root = sequence_root / family / "campaign"
        e = json.loads((root / "EFFECTIVE_CONFIG.json").read_text())
        pin = e["data_pins"]["common_population"]
        folder = storage_root / pin["relpath"]
        check(file_sha256(folder / "POPULATION.json") == pin["sha256"], "POPULATION_FILE_PIN")
        pop = CP.load_common_population(folder, expected_digest=pin["population_sha256"], input_dir=input_dir)
        body = json.loads((folder / "POPULATION.json").read_text())
        arm = e["adapter_effective"]["matched_population"]["arm"]
        scope = e["adapter_effective"]["model"]["market_time"]
        family_result = dict(population_sha256=pop.digest, splits={}, metrics={}, scaler={})
        output["families"][family] = family_result
        selected_arrays = {}
        for split in CP.SPLIT_NAMES:
            arrays = CP._load_npz(folder / body["splits"][split]["arms"][arm]["path"])
            selected_arrays[split] = arrays
            family_result["splits"][split] = audit_arrays(raw, arrays, calendar, body["identities"]["splits"][split],
                                                         pop.horizons, scope["spacing_open_market_hours"])
        # Recompute scaler rows and statistics independently from raw source values.
        rows = np.unique(np.concatenate([CP._load_npz(folder / info["path"])["slot_instants"].ravel()
                                         for info in body["splits"]["FIT"]["arms"].values()]))
        with np.load(folder / "SCALER_ROWS.npz", allow_pickle=False) as z:
            check(np.array_equal(rows, z["fit_source_bar_ts"]), "SCALER_ROW_UNION")
        check(bool(np.all(rows < epoch(body["identities"]["splits"]["FIT"]["end"]))), "SCALER_READS_INNER_OR_DEVELOPMENT")
        values = raw["x"][np.searchsorted(raw["ts"], rows)]
        check(np.allclose(values.mean(axis=0), pop.scaler.mean, rtol=1e-12, atol=1e-14), "SCALER_MEAN")
        check(np.allclose(values.std(axis=0), pop.scaler.std, rtol=1e-12, atol=1e-14), "SCALER_STD")
        family_result["scaler"] = dict(rows=len(rows), source="FIT_INPUT_ROWS_ONLY", matched_raw_statistics=True)
        plan = json.loads((root / "PLAN.json").read_text())
        for regime in ("r0", "r2"):
            predictions, _ = load_fit(root, regime)
            unit = next(u for u in plan["units"] if u["unit_id"] == f"{regime}.report")
            terminal_folder = root / "terminals" / unit["identity_sha256"]
            terminal = worker._verify_stored(terminal_folder, unit)["terminal"]
            metrics = json.loads((terminal_folder / "out" / terminal["artifacts"]["metrics"]["path"]).read_text())
            replays = {}
            for source_split, prediction_split in (("FIT", "fit"), ("INNER", "inner"), ("DEVELOPMENT", "validation")):
                arrays, p = selected_arrays[source_split], predictions[prediction_split]
                for name, expected in (("row_id", arrays["row_id"]), ("mask", arrays["head_mask"]),
                                       ("origin_price", arrays["origin_price"]), ("target_price", arrays["target_price"])):
                    check(np.array_equal(p[name], expected, equal_nan=True), "PREDICTION_POPULATION_" + name)
                replays[prediction_split] = {}
                for j, h in enumerate(pop.horizons):
                    replay = describe(p, h)
                    replays[prediction_split][str(h)] = replay
                    if source_split == "FIT":
                        check(abs(replay["naive_mae"] - pop.naive_loss_scale[h]) < 1e-14, "FIT_LOSS_SCALE")
                    if source_split == "DEVELOPMENT":
                        stored = metrics["per_horizon"][str(h)]
                        check(replay["n"] == stored["n"], "METRIC_ROW_COUNT")
                        for k in ("mae", "naive_mae", "relative_error"):
                            check(abs(replay[k] - stored[k]) < 1e-12, "METRIC_REPLAY_" + k)
            family_result["metrics"][regime] = replays
    gate = provenance.availability_gate(provenance.MANIFEST)
    output["upstream_limitations"] = dict(
        unverified_features=gate["unverified_features"], producer_unresolved=gate["producer_unresolved"],
        confirmatory_test_blocked=gate["confirmatory_test_blocked"],
        revision_policies={k: v["revision_policy"] for k, v in gate["features"].items()},
        calendar_status=calendar["market_calendar"]["calendar_id"],
        selection="RETROSPECTIVE_DEVELOPMENT_2024_NOT_INDEPENDENT_HOLDOUT",
        source_clock="Declared retained bar-END clock; vendor/publication/vintage availability not established by array timestamps")
    return output
