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
    # exit 127 on the first real dispatch: the launch script expands "~/" to "$HOME" and
    # nothing else, so every host path must carry it
    argv = jobs[0]["argv"]
    assert argv[3].startswith("~/anaconda3/")
    assert argv[argv.index("--out") + 1].startswith("~/.local/state/")
    units_root = argv[argv.index("--units-root") + 1]
    assert units_root.startswith("~/") or units_root.startswith("/")


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


# --- the budget scales with the unit's variables; a killed attempt is never overwritten -----

def test_the_per_unit_budget_scales_with_the_variable_count(tmp_path):
    """Three 26-variable toy units died at WALL_TIME_LIMIT on the first run because every unit
    got the flat per-variable figure. The scale is the unit's own variable count."""
    assert worker.variables_of(synthetic_unit(tmp_path, "one", n=120, v=1)) == 1
    assert worker.variables_of(synthetic_unit(tmp_path, "three", n=120, v=3)) == 3
    toys.materialise(toy_csv(cols=5), resource="synthetic_ohlc_1h.csv",
                     resource_contract=TOY_CONTRACT, out_dir=tmp_path / "toy",
                     delivery={"delivery_id": "d", "sha256": "x"}, unit_id="toy-x")
    assert worker.variables_of(tmp_path / "toy") == 5


def test_a_retry_takes_the_next_attempt_directory(tmp_path, monkeypatch):
    unit = synthetic_unit(tmp_path / "units", "u", n=120)
    out = tmp_path / "out"
    killed = out / "attempts" / "u" / "attempt-1"
    killed.mkdir(parents=True)
    (killed / "child.log").write_text("killed before result.json\n")
    seen = {}

    class FakeTask:
        def __init__(self, **kw):
            seen["attempt_dir"] = kw["attempt_dir"]
            seen["wall"] = kw["wall_seconds"]
            self.outcome = {"returncode": 0, "timed_out": False, "scope_result": "success",
                            "polled_oom_kill": 0, "cgroup_memory_peak": 0,
                            "child_maxrss_bytes": 1, "cpu_seconds": 0.1, "wall_seconds": 0.1,
                            "result": None, "started_at": "t", "ended_at": "t"}

        def start(self):
            pass

        def wait(self):
            pass

    IR = _load("df_isolated_runner")
    monkeypatch.setattr(IR, "Task", FakeTask)
    monkeypatch.setattr(IR, "classify", lambda *a, **k: ("UNCERTAIN", "fake", {"output_sha256": None, "rows_written": 0}))
    monkeypatch.setattr(IR, "detect_mechanism", lambda *a, **k: "PRLIMIT_AS")
    worker.run_shard(tmp_path / "units", out, run_id="r", host_role="COORDINATOR",
                     task_memory_bytes=2 << 30, wall_seconds=100.0, cpu_seconds=100)
    assert seen["attempt_dir"].name == "attempt-2"
    assert (killed / "child.log").read_text().startswith("killed")
    assert seen["wall"] == 100.0


def test_collect_records_the_highest_attempt_of_each_unit(tmp_path, monkeypatch):
    """attempt-1 was killed (RESOURCE_EXCEEDED, no rows); attempt-2 completed. The receipt
    names attempt 2 and verifies ITS rows; a receipt name is write-once."""
    import hashlib
    campaign = _load("df_d3_campaign")
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    out = home / "out" / "COORDINATOR" / "shard_00"
    (out / "terminals").mkdir(parents=True)
    a2 = out / "attempts" / "u" / "attempt-2"
    a2.mkdir(parents=True)
    (out / "attempts" / "u" / "attempt-1").mkdir()
    (a2 / "rows.jsonl").write_text('{"test": "verdict"}\n')
    sha = hashlib.sha256((a2 / "rows.jsonl").read_bytes()).hexdigest()
    base = {"dataset_id": "u", "wall_seconds": 1.0, "cpu_seconds": 1.0}
    (out / "terminals" / "u.attempt-1.json").write_text(json.dumps(
        dict(base, status="RESOURCE_EXCEEDED", rows_written=0, output_sha256=None)))
    (out / "terminals" / "u.attempt-2.json").write_text(json.dumps(
        dict(base, status="COMPLETED", rows_written=1, output_sha256=sha)))
    root = tmp_path / "root"
    root.mkdir()
    rep = campaign.collect(root, "out", {"COORDINATOR": {}}, run_id="r", receipt="C.json")
    assert [(u["unit"], u["attempt"], u["status"], u.get("output_verified"))
            for u in rep["units"]] == [("u", 2, "COMPLETED", True)]
    assert rep["verified"] == 1 and rep["mismatched"] == 0
    with pytest.raises(SystemExit):
        campaign.collect(root, "out", {"COORDINATOR": {}}, run_id="r", receipt="C.json")


def test_a_multivariate_unit_terminal_has_one_metric_identity_per_operator():
    """The server refused d3mech-v1's 67 multivariate terminals: one d3.verdict.<op> per
    variable is a duplicate identity. The terminal aggregates over variables."""
    campaign = _load("df_d3_campaign")
    rows = [{"test": "verdict", "operator_kind": "op", "variable": v, "value": val}
            for v, val in (("v0", 1.0), ("v1", 0.0), ("v2", 1.0))]
    rows += [{"test": "prefix_all_available", "operator_kind": "op", "variable": v,
              "outcome": "PASSED", "value": None} for v in ("v0", "v1", "v2")]
    t = campaign.unit_terminal(rows, status="COMPLETED", reason=None, wall=1.0, cpu=1.0,
                               deliveries=[], bank="SYNTHETIC", unit_id="u", run_id="r")
    ids = [(m["metric"], m["split"], m["horizon"], m["unit"]) for m in t["metrics"]]
    assert len(ids) == len(set(ids))
    verdict = next(m for m in t["metrics"] if m["metric"] == "d3.verdict.op")
    assert verdict["value"] == pytest.approx(2 / 3) and verdict["min_value"] == 0.0
    assert next(m for m in t["metrics"] if m["metric"] == "d3.variables")["value"] == 3.0


def test_the_envelope_consumption_items_are_what_the_loader_stores():
    """The first d3mech-v1 envelope carried bare strings under data_consumed; the store's
    loader calls item.get(...) and fell over. Every item is {id, digest, eligibility_state}."""
    campaign = _load("df_d3_campaign")
    units = [{"unit_id": "u1", "status": "COMPLETED",
              "rows": [{"contract_sha256": "c" * 64, "variable": "v0", "test": "verdict"}],
              "verdict_rows": [{"variable": "v0"}, {"variable": "v1"}]}]
    consumed = campaign.mechanics_consumption(units)
    assert set(consumed) == {"datasets", "variables", "operators"}
    for kind, items in consumed.items():
        assert items and all(set(i) == {"id", "digest", "eligibility_state"} for i in items)
    assert consumed["datasets"] == [{"id": "u1", "digest": "c" * 64,
                                     "eligibility_state": campaign.MECHANICAL_ELIGIBILITY}]
    assert [i["id"] for i in consumed["variables"]] == ["v0", "v1"]
    assert len(consumed["operators"]) == 9
    assert all(len(i["digest"]) == 64 for i in consumed["operators"])
