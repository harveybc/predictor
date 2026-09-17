"""The D3 mechanics pipeline pieces that need no service: units, rows, freeze, shards, jobs,
terminals and the MECHANICAL envelope (J3).

Everything here runs on a throwaway bank under `tmp_path`, built from real generator code, and
on toy CSV bytes shaped like the lake's. Nothing touches production or a running service.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, directory: Path = REPO / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pytest.importorskip("scipy")
pytest.importorskip("pywt")
pytest.importorskip("pandas")

design = _load("df_d3_design")
worker = _load("df_d3_unit_worker")
campaign = _load("df_d3_campaign")
toys = _load("df_d3_toy_resources")
envelope = _load("campaign_envelope", REPO / "olap")


def synthetic_unit(root: Path, name: str, *, n: int = 300, v: int = 1, missing: bool = False) -> Path:
    """A unit shaped exactly like the C128 bank's, small enough to run in a test."""
    unit = root / name
    unit.mkdir(parents=True)
    rng = np.random.default_rng(3)
    clean = np.sin(np.arange(n) / 9.0)[None, :].repeat(v, axis=0)
    noise = rng.normal(0, 0.2, size=(v, n))
    observed = clean + noise
    mask = np.zeros((v, n), dtype=bool)
    if missing:
        mask[:, 120:130] = True
    np.save(unit / "observed_signal.npy", observed, allow_pickle=False)
    np.save(unit / "missing_mask.npy", mask, allow_pickle=False)
    np.save(unit / "clean_signal.npy", clean, allow_pickle=False)
    np.save(unit / "additive_noise.npy", noise, allow_pickle=False)
    np.save(unit / "metric_support.npy", ~mask, allow_pickle=False)
    (unit / "events.json").write_text("[]")
    train_end = int(n * 0.6)
    (unit / "UNIT.json").write_text(json.dumps({
        "unit_id": name, "family": "sinusoid", "n_samples": n, "n_variables": v,
        "missingness": {"kind": "blocks" if missing else "none"},
        "partitions": {"train": [0, train_end], "calibration": [train_end, int(n * 0.8)],
                       "confirmation": [int(n * 0.8), n]},
        "generator": {"version": "test.v1", "code_sha256": "0" * 64},
        "digests": {"observed_signal": "1" * 64}}))
    return unit


def toy_csv(n: int = 120, cols: int = 2, freq_hours: int = 4) -> bytes:
    lines = ["DATE_TIME," + ",".join(f"c{i}" for i in range(cols))]
    import datetime as dt

    start = dt.datetime(2024, 1, 1)
    for i in range(n):
        stamp = (start + dt.timedelta(hours=freq_hours * i)).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(stamp + "," + ",".join(f"{np.sin(i / 7.0 + j):.6f}" for j in range(cols)))
    return ("\n".join(lines) + "\n").encode()


TOY_CONTRACT = {"event_time_column": "DATE_TIME", "timezone": "NAIVE_WALL_CLOCK",
                "frequency": "4h", "availability": {"label": "WINDOW_START",
                                                    "completion_lag_max": "4h",
                                                    "timezone_evidence": "PRODUCER_STATEMENT",
                                                    "use_class": "OFFLINE_DAY_GRANULAR"}}


# --- units ----------------------------------------------------------------------------------

def test_a_bank_unit_loads_under_sample_index_semantics(tmp_path):
    unit = worker.load_unit(synthetic_unit(tmp_path, "u1", missing=True))
    assert unit["bank"] == "SYNTHETIC" and unit["timestamp_meaning"] == "SAMPLE_INDEX"
    x = unit["inputs"][0]
    assert x["period_seconds"] == 1 and x["available_at"] == x["timestamps"]
    assert all(np.isnan(x["values"][120:130]))
    assert unit["resource_contract"]["availability"]["completion_lag_max"] == "0s"


def test_a_toy_unit_carries_the_lakes_availability_and_frequency(tmp_path):
    record = toys.materialise(toy_csv(), resource="synthetic_typical_price_4h_train.csv",
                              resource_contract=TOY_CONTRACT, out_dir=tmp_path / "toy",
                              delivery={"delivery_id": "d1", "sha256": "x"}, unit_id="toy-1")
    unit = worker.load_unit(tmp_path / "toy")
    x = unit["inputs"][0]
    assert unit["bank"] == "TOY" and unit["family"] == "toy_price"
    assert x["period_seconds"] == 4 * 3600 and x["available_at"][0] - x["timestamps"][0] == 4 * 3600
    assert record["contract_sha256"] and len(record["files"]) == 2


def test_backwards_timestamps_refuse_a_toy_resource(tmp_path):
    lines = toy_csv().decode().splitlines()
    lines[2], lines[3] = lines[3], lines[2]
    with pytest.raises(SystemExit, match="go backwards"):
        toys.materialise(("\n".join(lines) + "\n").encode(), resource="synthetic_ohlc_1h.csv",
                         resource_contract=TOY_CONTRACT, out_dir=tmp_path / "bad",
                         delivery={}, unit_id="bad")


def test_a_resource_outside_the_three_toys_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not one of the three"):
        toys.materialise(toy_csv(), resource="panel.csv", resource_contract=TOY_CONTRACT,
                         out_dir=tmp_path / "p", delivery={}, unit_id="p")


# --- rows -----------------------------------------------------------------------------------

def test_rows_carry_the_identities_a_row_needs_to_be_traced(tmp_path):
    unit = worker.load_unit(synthetic_unit(tmp_path, "u2", n=200))
    job = {"run_id": "r1", "host_role": "COORDINATOR", "code_sha256": "c" * 64}
    rows = worker.rows_for(job, unit)
    verdicts = [r for r in rows if r["test"] == "verdict"]
    assert len(verdicts) == 9
    for r in rows:
        assert r["design_sha256"] == design.D3_AMENDMENT_V1["design_sha256"]
        assert len(r["spec_sha256"]) == 64 and r["result_class"] == "MECHANICAL"
        assert r["classification"] == "NON_GOVERNING"
    tests = {r["test"] for r in rows} - {"verdict"}
    assert set(design.REQUIRED_TESTS) <= tests


# --- freeze, shards, jobs -------------------------------------------------------------------

def test_the_population_selection_is_deterministic_and_bounded(tmp_path):
    root = tmp_path / "bank"
    for i in range(3):
        synthetic_unit(root, f"sinusoid__white__n300__seed{i}", n=300)
    synthetic_unit(root, "long", n=8192)
    chosen = campaign.select_bank_units(root, lengths=(300,), missingness=("none",))
    assert [u["unit_id"] for u in chosen] == sorted(u["unit_id"] for u in chosen)
    assert len(chosen) == 3
    assert len(campaign.select_bank_units(root, lengths=(300,), missingness=("none",),
                                          per_family_limit=2)) == 2


def test_freeze_seals_population_budget_and_pilot(tmp_path):
    root = tmp_path / "bank"
    u = synthetic_unit(root, "sinusoid__white__n300__seed0", n=300)
    doc = campaign.freeze(tmp_path / "run", bank_root=root, pilot_units=[u],
                          per_unit_wall=120.0, per_unit_cpu=120, task_memory_bytes=2 << 30,
                          role_caps={"COORDINATOR": 1, "WORKER_A": 1, "WORKER_B": 1},
                          toys=False)
    assert doc["freeze_sha256"] and doc["cost_pilot"]["operators"]
    assert doc["readiness"]["SCIENTIFIC_UTILITY"] == "not claimed"
    with pytest.raises(SystemExit, match="never written over"):
        campaign.freeze(tmp_path / "run", bank_root=root, pilot_units=[u], per_unit_wall=1,
                        per_unit_cpu=1, task_memory_bytes=2 << 30, role_caps={}, toys=False)


def test_shards_and_jobs_follow_the_dispatch_precedent(tmp_path):
    root = tmp_path / "bank"
    for i in range(5):
        synthetic_unit(root, f"sinusoid__white__n300__seed{i}", n=300)
    frozen = campaign.freeze(tmp_path / "run", bank_root=root,
                             pilot_units=[root / "sinusoid__white__n300__seed0"],
                             per_unit_wall=100.0, per_unit_cpu=100, task_memory_bytes=2 << 30,
                             role_caps={"COORDINATOR": 2, "WORKER_A": 10, "WORKER_B": 6},
                             toys=False)
    frozen["bank"]["units"] = campaign.select_bank_units(root, lengths=(300,),
                                                         missingness=("none",))
    shards = campaign.build_shards(tmp_path / "run", frozen, toy_units_root=None, shard_size=2)
    assert len(shards["shards"]) == 3
    jobs_path = campaign.build_jobs(tmp_path / "run", frozen, shards, run_id="d3test",
                                    python_rel="anaconda3/envs/trading-stack/bin/python",
                                    out_rel=".local/state/x")
    jobs = json.loads(jobs_path.read_text())["jobs"]
    assert len(jobs) == 3 and jobs[0]["cpus"] == 1 and jobs[0]["gpu_bytes"] == 0
    assert "{role}" in " ".join(jobs[0]["argv"])
    assert jobs[0]["wall"] == int(100.0 * 2) + 120
    assert "/home/" not in json.dumps(jobs)


# --- terminals and the envelope -------------------------------------------------------------

def test_a_unit_terminal_names_its_verdicts_and_grants_nothing(tmp_path):
    unit = worker.load_unit(synthetic_unit(tmp_path, "u3", n=200))
    rows = worker.rows_for({"run_id": "r", "host_role": "COORDINATOR", "code_sha256": "c" * 64},
                           unit)
    terminal = campaign.unit_terminal(rows, status="COMPLETED", reason=None, wall=1.0, cpu=1.0,
                                      deliveries=[], bank="SYNTHETIC", unit_id="u3", run_id="r")
    names = {m["metric"] for m in terminal["metrics"]}
    assert any(n.startswith("d3.verdict.") for n in names)
    ids = [(m["metric"], m["split"], m["horizon"], m["unit"]) for m in terminal["metrics"]]
    assert len(ids) == len(set(ids))
    assert terminal["tags"]["grants"] == "NONE" and terminal["tags"]["classification"] == "NON_GOVERNING"
    assert all(isinstance(v, str) for v in terminal["tags"].values())


def test_the_mechanics_envelope_validates_as_mechanical(tmp_path):
    unit = worker.load_unit(synthetic_unit(tmp_path, "u4", n=200))
    rows = worker.rows_for({"run_id": "r", "host_role": "COORDINATOR", "code_sha256": "c" * 64},
                           unit)
    frozen = {"freeze_sha256": "f" * 64}
    doc = campaign.build_mechanics_envelope(
        campaign_key="d3-mechanics-test", run_id="r", code_identity={"value": "a" * 40},
        frozen=frozen, units=[{"unit_id": "u4", "status": "COMPLETED",
                               "verdict_rows": [r for r in rows if r["test"] == "verdict"]}],
        wall_seconds=3.0)
    assert doc["result_class"] == "MECHANICAL" and len(doc["units"]) == 9
    assert envelope.validate_envelope(doc)
    assert doc["terminal"]["adjudication"] == "MECHANICAL_EVIDENCE_NO_ADJUDICATION"
