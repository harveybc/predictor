"""The D3 matrix aggregator counts what the battery said, per operator, and nothing else."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


matrix = _load("df_d3_matrix")


def _row(unit, op, test, outcome, value=None, detail=""):
    return {"unit_id": unit, "variable": "v0", "operator_kind": op, "operator_group": "g",
            "bank": "SYNTHETIC", "family": "f", "test": test, "outcome": outcome,
            "value": value, "detail": detail}


def _write(root, role, shard, unit, rows, verified=True):
    d = root / "collected" / role / shard / "attempts" / unit / "attempt-1"
    d.mkdir(parents=True)
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"unit": unit, "role": role, "shard": shard, "output_verified": verified,
            "status": "COMPLETED", "rows": len(rows)}


def test_only_verified_units_count_and_refusals_are_kept(tmp_path):
    ok = _write(tmp_path, "WORKER_A", "shard_00", "u1", [
        _row("u1", "op", "prefix_all_available", "PASSED"),
        _row("u1", "op", "cost_pilot", "PASSED", 0.01),
        _row("u1", "op", "response_probe", "PASSED", 0.0),
        _row("u1", "op", "verdict", "MECHANICALLY_ACCEPTED", 1.0)])
    refused = _write(tmp_path, "WORKER_A", "shard_00", "u2", [
        _row("u2", "op", "response_probe", "FAILED", 1.0, "moved 1"),
        _row("u2", "op", "cost_pilot", "PASSED", 0.03),
        _row("u2", "op", "verdict", "MECHANICALLY_REFUSED", 0.0, "probe")])
    unverified = _write(tmp_path, "WORKER_B", "shard_01", "u3", [
        _row("u3", "op", "verdict", "MECHANICALLY_ACCEPTED", 1.0)], verified=False)
    (tmp_path / "COLLECT.json").write_text(json.dumps(
        {"run_id": "r", "mismatched": 1, "units": [ok, refused, unverified]}))
    m = matrix.aggregate(tmp_path)
    assert m["units_verified"] == 2 and m["units_collected"] == 3
    op = m["operators"]["op"]
    assert op["verdicts"] == {"MECHANICALLY_ACCEPTED": 1, "MECHANICALLY_REFUSED": 1}
    assert op["tests"]["response_probe"] == {"PASSED": 1, "FAILED": 1}
    assert op["cost_cpu_s_per_1000"] == {"n": 2, "median": 0.02, "max": 0.03}
    assert op["response_probe_lags"] == {"0.0": 1, "1.0": 1}
    md = matrix.markdown(m)
    assert "MECHANICALLY_REFUSED 1" in md and "**2** units verified of 3" in md
