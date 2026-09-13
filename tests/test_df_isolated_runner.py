"""C150-C151: one bounded process per dataset, durable terminals, stop file,
resume by contract and code identity, sibling isolation, and the worst-case
smoke battery on development-scale copies (never the real 13M-row file).

The only test that exceeds a memory limit is the preflight-bypass mutation:
its cap is 300M, inside the batch slice, and it runs once."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _spec_load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _spec_load("df_profile_run")
IR = _spec_load("df_isolated_runner")
SLICE = "crispdm-batch.slice"
MUTATION_CAP = 300 * 1000 * 1000

pytestmark = pytest.mark.skipif(IR.detect_mechanism(SLICE) != "SYSTEMD_USER_SCOPE",
                                reason="systemd user scopes unavailable on this host")


def financial_fixture(root: Path, specs):
    """specs: [(name, T, kind)] -> (contracts_file, root, jobs, dataset_ids). kind: walk | noise."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root.mkdir(parents=True, exist_ok=True)
    contracts, ids = [], []
    for i, (name, T, kind) in enumerate(specs):
        rng = np.random.default_rng(1000 + i)
        v = np.cumsum(rng.integers(-3, 4, T)) if kind == "walk" else rng.integers(0, 1000, T)
        ts = np.arange(T, dtype=np.int64) * 60_000_000_000 + 1_600_000_000_000_000_000
        path = root / f"{name}.parquet"
        pq.write_table(pa.table({"timestamp": pa.array(ts, type=pa.timestamp("ns", tz="UTC")),
                                 "value": pa.array(v.astype(np.int64))}), path, row_group_size=max(1, T // 4))
        b0, b1 = int(T * 0.6), int(T * 0.8)
        ds = f"financial_data.dev_copy.{name}"
        ids.append(ds)
        contracts.append({"dataset_id": ds, "contract_sha256": hashlib.sha256(ds.encode()).hexdigest(),
                          "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}],
                          "variables": [{"name": "timestamp", "role": "TIMESTAMP", "variable_id": ds + ".timestamp"},
                                        {"name": "value", "role": "UNKNOWN", "variable_id": ds + ".value"}],
                          "partitions": {"boundaries": {"train": [0, b0], "calibration": [b0, b1],
                                                        "confirmation": [b1, T]}},
                          "time": {"timestamp_meaning": "UNKNOWN", "frequency_nominal_seconds": 60}})
    cf = root / "CONTRACTS.json"
    cf.write_text(json.dumps({"contracts": contracts}))
    return cf, root, R.financial_jobs(cf, root), ids


def terminal_of(root: Path, entry: dict) -> dict:
    t = json.loads((root / entry["terminal"]).read_text())
    assert IR.validate_terminal(t) == []
    return t


def test_c151a_dev_scale_heavy_copy_stays_under_its_estimate(tmp_path):
    cf, froot, jobs, ids = financial_fixture(tmp_path / "fin", [("heavy_dev_copy", 1_000_000, "walk")])
    out = tmp_path / "root"
    receipt = R.run(out, jobs, workers=1, task_memory_bytes=3 << 30, host_budget_bytes=3 << 30,
                    heartbeat_seconds=1.0, slice_=SLICE, wall_seconds=1800, cpu_seconds=1800)
    (entry,) = receipt["datasets"]
    t = terminal_of(out, entry)
    assert t["status"] == "COMPLETED", t["reason"]
    assert f"slice={SLICE}" in t["limit_mechanism"]
    assert t["observed_peak_rss_bytes"] is not None
    assert t["observed_peak_rss_bytes"] <= t["planned_peak_bytes"] <= t["memory_limit_bytes"]
    adir = out / Path(t["output_file"]).parent
    est = [json.loads(line) for line in (adir / "resource_estimates.jsonl").read_text().splitlines()]
    admitted = [e for e in est if e["decision"] != "NOT_RUN_RESOURCE_BOUND"]
    assert t["observed_peak_rss_bytes"] <= max(e["estimated_peak_bytes"] for e in admitted)
    assert all(e["estimated_peak_bytes"] <= e["budget_bytes"] < t["memory_limit_bytes"] for e in admitted)
    adf = [e for e in est if e["metric"] == "unit_root_adf" and e["partition"] == "train"]
    assert len(adf) == 3 and {e["decision"] for e in adf} == {"RUN_BOUNDED"}
    rows = [json.loads(line)["row"] for line in (out / t["output_file"]).read_text().splitlines()]
    blocks = [r for r in rows if r["metric"].startswith("adf_statistic_block_") and r["partition"] == "train"]
    assert {r["metric"] for r in blocks} >= {"adf_statistic_block_start", "adf_statistic_block_middle",
                                             "adf_statistic_block_end"}
    assert t["rows_written"] == len(rows) and hashlib.sha256((out / t["output_file"]).read_bytes()).hexdigest() \
        == t["output_sha256"]
    assert (out / "PROFILE_RUN_RECEIPT.json").exists() and "/home/" not in (out / "PROFILE_RUN_RECEIPT.json").read_text()


def test_c151b_mutation_bypassing_preflight_is_killed_inside_its_cgroup(tmp_path):
    # train = 204,000 rows: exact ADF needs about 5.04 * n * (lag + 2) * 8 = 0.67 GB, while every group
    # before it (counts, quantiles, ACF on 2^19 FFT) stays far below the 300M cap
    cf, froot, jobs, ids = financial_fixture(tmp_path / "fin", [("adf_bomb_dev_copy", 340_000, "walk")])
    out = tmp_path / "root"
    assigned = int(MUTATION_CAP / IR.LIMIT_RATIO)
    before = time.time()
    receipt = R.run(out, jobs, workers=1, task_memory_bytes=assigned, host_budget_bytes=assigned,
                    heartbeat_seconds=0.5, slice_=SLICE, wall_seconds=900, cpu_seconds=900,
                    mutation_bypass_preflight=True)
    (entry,) = receipt["datasets"]
    t = terminal_of(out, entry)
    assert t["memory_limit_bytes"] <= MUTATION_CAP
    assert t["status"] == "RESOURCE_EXCEEDED", t["reason"]
    assert "CGROUP_OOM_KILL" in t["reason"] and "MUTATION_BYPASS_PREFLIGHT" in t["reason"]
    assert "unit_root_adf" in t["reason"], t["reason"]          # the last boundary before the kill was the ADF gate
    assert t["output_sha256"] is None and t["output_file"] is None
    # the parent (this test process) survived, wrote its durable receipt and is far below the task cap
    import resource
    assert time.time() - before < 900
    assert json.loads((out / "PROFILE_RUN_RECEIPT.json").read_text())["counts"] == {"RESOURCE_EXCEEDED": 1}
    mem_available = int(next(ln.split()[1] for ln in open("/proc/meminfo") if ln.startswith("MemAvailable"))) * 1024
    assert mem_available > 1 << 30
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 4 << 30


def test_c150_stop_file_and_resume_by_contract_and_code_identity(tmp_path):
    specs = [("tiny_a", 3000, "noise"), ("tiny_b", 3000, "walk"), ("medium_c", 400_000, "walk"), ("tiny_d", 3000, "noise")]
    cf, froot, jobs, ids = financial_fixture(tmp_path / "fin", specs)
    out, stop = tmp_path / "root", tmp_path / "STOP"

    # 1. a stop file present before launch: nothing starts, nothing is hidden
    stop.write_text("stop")
    r0 = R.run(out, jobs, workers=1, task_memory_bytes=2 << 30, host_budget_bytes=2 << 30, stop_file=stop,
               slice_=SLICE)
    assert r0["final"] is False and r0["counts"] == {"NOT_STARTED": 4}
    assert not list((out / "terminals").iterdir())
    stop.unlink()

    # 2. stop while medium_c runs: tiny_a and tiny_b complete, medium_c stops gracefully, tiny_d never starts
    sname_c = R.safe_name(ids[2])
    box = {}
    th = threading.Thread(target=lambda: box.update(r=R.run(out, jobs, workers=1, task_memory_bytes=2 << 30,
                                                            host_budget_bytes=2 << 30, stop_file=stop, resume=True,
                                                            heartbeat_seconds=0.2, slice_=SLICE)))
    th.start()
    hb = out / "attempts" / sname_c / "attempt-1" / "heartbeat.json"
    deadline = time.time() + 300
    while not hb.exists() and time.time() < deadline:
        time.sleep(0.05)
    stop.write_text("stop")
    th.join(600)
    r1 = box["r"]
    st = [e["status"] for e in r1["datasets"]]          # job order; a dataset never opened has no id yet
    assert st == ["COMPLETED", "COMPLETED", "INCONCLUSIVE", "NOT_STARTED"]
    assert "dataset_id" not in r1["datasets"][3]
    tc = terminal_of(out, r1["datasets"][2])
    assert "STOP_FILE_REQUESTED" in tc["reason"] and tc["output_sha256"] is None
    assert r1["final"] is False
    stop.unlink()

    # 3. resume: tiny_a skipped (same contract and code); tiny_b's contract digest changed -> rerun;
    #    medium_c (not COMPLETED) -> new attempt; tiny_d -> first attempt
    doc = json.loads(cf.read_text())
    doc["contracts"][1]["contract_sha256"] = "f" * 64
    cf.write_text(json.dumps(doc))
    r2 = R.run(out, jobs, workers=1, task_memory_bytes=2 << 30, host_budget_bytes=2 << 30, resume=True, slice_=SLICE)
    by = {e["dataset_id"]: e for e in r2["datasets"]}
    assert r2["final"] is True and r2["counts"] == {"COMPLETED": 4}
    assert by[ids[0]].get("resumed_skip") is True
    assert by[ids[1]]["terminal"].endswith("attempt-2.json") and not by[ids[1]].get("resumed_skip")
    assert by[ids[2]]["terminal"].endswith("attempt-2.json")
    assert by[ids[3]]["terminal"].endswith("attempt-1.json")
    # the earlier attempts' terminals are still there, unchanged (write-once)
    assert (out / "terminals" / f"{sname_c}.attempt-1.json").exists()
    with pytest.raises(SystemExit, match="sealed"):
        R.run(out, jobs, workers=1, resume=True, slice_=SLICE)


def test_c150_failing_siblings_never_remove_others_evidence_and_admission_is_by_memory(tmp_path):
    specs = [("ok_first", 3000, "noise"), ("bytes_changed", 3000, "noise"), ("bad_partitions", 3000, "noise"),
             ("ok_last", 3000, "walk")]
    cf, froot, jobs, ids = financial_fixture(tmp_path / "fin", specs)
    doc = json.loads(cf.read_text())
    doc["contracts"][1]["files"][0]["sha256"] = "0" * 64                      # bytes refusal in the parent
    doc["contracts"][2]["partitions"]["boundaries"]["confirmation"] = [2400, 2999]   # modules refuse in the child
    cf.write_text(json.dumps(doc))
    out = tmp_path / "root"
    # host budget 1.5 GiB with 1 GiB tasks: two tasks never run at the same time although workers=2
    r = R.run(out, jobs, workers=2, task_memory_bytes=1 << 30, host_budget_bytes=(3 << 30) // 2, slice_=SLICE)
    by = {e["dataset_id"]: e for e in r["datasets"]}
    assert by[ids[0]]["status"] == by[ids[3]]["status"] == "COMPLETED"
    assert by[ids[1]]["status"] == "REFUSED" and "bytes differ" in by[ids[1]]["reason"]
    assert by[ids[2]]["status"] == "FAILED" and "partition boundaries" in by[ids[2]]["reason"]
    spans = []
    for d in (ids[0], ids[2], ids[3]):
        t = terminal_of(out, by[d])
        spans.append((t["started_at"], t["ended_at"]))
        if t["status"] == "COMPLETED":
            p = out / t["output_file"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == t["output_sha256"]
    spans.sort()
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
