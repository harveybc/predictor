"""The PS3-R paired-contrast tool refuses what would make the table meaningless (stdlib + numpy only)."""
import copy
import json

import pytest

from tools import ps3r_pilot_contrast as pc


def probe(target="Y_s", horizon=4, trained=0.9, random=1.0, raw=0.95, naive=1.1, sha="a" * 64):
    return {"target": target, "horizon": horizon, "fold": "inner_1", "seed": 2021, "loss_trained": trained,
            "loss_random": random, "loss_raw": raw, "naive": naive, "delta_probe": random - trained,
            "preservation": raw - trained, "eval_rows_sha256": sha, "eval_rows": 100, "loss_name": "mse"}


def record(arm, trained, **kw):
    return {"schema": "ps3r.pilot.record.v1", "status": "PILOT_ENGINEERING", "input": "rsi_21", "fold": "inner_1",
            "seed": 2021, "arm": arm, "architecture": {"latent_shape": [24, 16]},
            "objective": {"name": "x", "sha256": "b" * 64},
            "declared_deviations": ["d"] if arm == "contrastive" else [],
            "ranges": {"probe_fit_origins": [23, 7473], "probe_eval_origins": [7534, 9588]},
            "fit": {"observed_updates": 10, "stop_reason": "patience", "fit_seconds": 1.0,
                    "tracing_seconds_estimate": 0.5 if arm == "contrastive" else None},
            "cost": {"seconds_total": 2.0}, "probes": [probe(trained=trained, **kw)]}


def pair():
    return [record("ae", 0.92), record("contrastive", 0.90)]


def test_contrast_rows_are_paired_and_signed():
    table = pc.build_table(pair())
    row = table["rows"][0]
    assert row["contrast_ae_minus_contrastive"] == pytest.approx(0.02)
    assert row["delta_probe_ae"] == pytest.approx(0.08) and row["delta_probe_contrastive"] == pytest.approx(0.10)
    assert row["naive"] == 1.1 and table["status"] == "PAIRED_CONTRAST_GENERATED"


def test_refuses_overlapping_fit_and_eval_rows():
    recs = pair()
    recs[1]["ranges"]["probe_eval_origins"] = [7000, 9588]
    with pytest.raises(ValueError, match="OVERLAP"):
        pc.build_table(recs)


def test_refuses_unpaired_eval_rows():
    recs = pair()
    recs[1]["probes"][0]["eval_rows_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="UNPAIRED_ROWS"):
        pc.build_table(recs)


def test_refuses_self_forecast_targets():
    recs = pair()
    for r in recs:
        r["probes"][0]["target"] = "rsi_21"
    with pytest.raises(ValueError, match="SELF_FORECAST_REFUSED"):
        pc.build_table(recs)


def test_refuses_pooled_latents():
    recs = pair()
    recs[0]["architecture"]["latent_shape"] = [1, 16]
    with pytest.raises(ValueError, match="POOLED"):
        pc.build_table(recs)


def test_refuses_unpaired_random_or_raw_arm():
    recs = pair()
    recs[1]["probes"][0]["loss_random"] = 1.5
    with pytest.raises(ValueError, match="RANDOM_ARM_NOT_PAIRED"):
        pc.build_table(recs)


def test_failed_fits_are_reported_not_dropped():
    recs = pair() + [{"schema": "ps3r.pilot.record.v1", "status": "FIT_FAILED", "input": "mfi_14",
                      "fold": "inner_2", "seed": 2022, "arm": "contrastive", "reason": "no epoch improved"}]
    table = pc.build_table(recs)
    assert table["failed_fits"] == [{"input": "mfi_14", "fold": "inner_2", "seed": 2022, "arm": "contrastive",
                                     "reason": "no epoch improved"}]


def test_summary_aggregates_over_folds_and_seeds():
    recs = []
    for seed, (ae, cl) in zip((2021, 2022), ((0.92, 0.90), (0.91, 0.93))):
        for arm, value in (("ae", ae), ("contrastive", cl)):
            r = record(arm, value)
            r["seed"] = seed
            r["probes"][0]["seed"] = seed
            recs.append(r)
    summary = pc.build_table(recs)["summary"]
    s = summary[0]
    assert s["n"] == 2 and s["contrast_mean"] == pytest.approx(0.0)
    assert s["contrastive_beats_naive"] == 2 and json.dumps(summary)
