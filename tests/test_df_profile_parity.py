"""C149, C169: the columnar, incremental path writes the same module rows as the
frozen whole-table path of predictor@ad30cf3, on synthetic units and on a
small public-panel-shaped parquet fixture.

Equality is exact for every row and every field, except:

* code digests, cpu seconds and the C148 unit-root policy fields (declared);
* C169: the linear-algebra descriptors (PCA explained-variance ratios,
  effective rank, PC1 loadings and shares) are NOT claimed equal. LAPACK on a
  differently laid-out block, or on another numpy/BLAS stack, moves them by
  about 1e-16..1e-14. Their raw values must agree within the tolerance
  declared in tools/df_linalg_parity.py (pinned below by digest, before any
  POST), and their canonical representation (`value_canonical`) must be
  identical. Every other field of those rows is still compared exactly, and
  the integer descriptors of the same decompositions stay exact. The new path
  records numpy version and backends in `params.linalg`, which the frozen path
  did not have."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = "ad30cf3"
OLD_FILES = ("df_profile_run", "df_profile_univariate", "df_profile_information", "df_profile_multivariate",
             "df_sampling")
POLICY_PARAM_KEYS = {"policy", "policy_sha256", "exact_max_n", "temporal_universe", "run_universe", "run_length",
                     "maxlag"}
NEW_ASSUMPTION = "a descriptor only: never a causality gate and never an eligibility gate"
# Declared before any C169 POST; changing a tolerance changes this digest and fails the test.
DECLARED_TOLERANCES_SHA256 = "a55dc46f05dce5803ee49254378a6e6e5e24ea57e4a1e5621f1fff4ed57b8cf6"


def _spec_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _spec_load("df_profile_run", ROOT / "tools/df_profile_run.py")
LP = _spec_load("df_linalg_parity", ROOT / "tools/df_linalg_parity.py")


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    d = tmp_path_factory.mktemp("frozen_ad30cf3")
    for name in OLD_FILES:
        try:
            src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:tools/{name}.py"], check=True,
                                 capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip(f"git cannot show {BASE}:tools/{name}.py")
        (d / f"{name}.py").write_bytes(src)
    mods = {n: _spec_load(f"frozen_{n}", d / f"{n}.py") for n in OLD_FILES}
    run = mods["df_profile_run"]

    def _load(name):                           # old profile modules, current contract helpers
        return mods[name] if name in mods else R._load(name)
    run._load = _load
    return run


def normalize(module, row):
    """-> (row without admitted fields, canonical value or None)."""
    r = {k: v for k, v in row.items() if k not in ("code_sha256", "cpu_seconds", "value_canonical")}
    canon = row.get("value_canonical")
    e = r["estimator"]
    if "linalg" in e["params"]:
        e = r["estimator"] = {"name": e["name"], "params": {k: v for k, v in e["params"].items() if k != "linalg"},
                              "assumptions": e["assumptions"]}
    if module == "df_profile_univariate" and r["metric"] in ("adf_statistic", "adf_pvalue", "kpss_statistic",
                                                             "kpss_pvalue"):
        r["estimator"] = {"name": e["name"],
                          "params": {k: v for k, v in e["params"].items() if k not in POLICY_PARAM_KEYS},
                          "assumptions": [a for a in e["assumptions"] if a != NEW_ASSUMPTION]}
    return r, canon


def old_rows(old, job):
    res = old.run_job(job)
    assert res["status"] == "COMPLETED", res.get("error")
    return [(it["module"], *normalize(it["module"], it["row"])) for it in res["rows"]]


def new_rows(job, tmp_path):
    adir = tmp_path / "attempt"
    adir.mkdir()
    spec = {"job": job, "run_id": "parity", "attempt_dir": str(adir), "budget_bytes": 64 << 30,
            "heartbeat_seconds": 60, "stop_file": None, "code_sha256": R.code_sha256(), "dataset_id": "parity"}
    (adir / "job.json").write_text(json.dumps(spec))
    assert R.worker_main(adir / "job.json") == 0, (adir / "result.json").read_text()
    result = json.loads((adir / "result.json").read_text())
    out = adir / "profile.jsonl"
    assert result["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert not (adir / "profile.jsonl.partial").exists()
    est = [json.loads(line) for line in (adir / "resource_estimates.jsonl").read_text().splitlines()]
    assert est and all(e["decision"] == "RUN_EXACT" for e in est)
    items = [json.loads(line) for line in out.read_text().splitlines()]
    for it in items:                               # provenance and canonical form are present where declared
        r = it["row"]
        if LP.tolerance_for(it["module"], r["metric"]) and r["status"] == "COMPLETED":
            lin = r["estimator"]["params"]["linalg"]
            assert lin["tolerance_sha256"] == LP.TOLERANCES_SHA256 and lin["numpy_version"] == np.__version__
            assert r["value_canonical"] == LP.canonical(r["value"])
    return [(it["module"], *normalize(it["module"], it["row"])) for it in items]


def compare(a, b):
    """Exact equality, except raw values of tolerance-declared metrics (tolerance + canonical equality)."""
    assert len(a) == len(b)
    tolerated = 0
    for i, ((ma, x, _), (mb, y, yc)) in enumerate(zip(a, b)):
        assert ma == mb, i
        if not (LP.tolerance_for(ma, x["metric"]) and x["status"] == y["status"] == "COMPLETED"):
            assert x == y, (i, x, y)
            continue
        tolerated += 1
        assert {k: v for k, v in x.items() if k != "value"} == {k: v for k, v in y.items() if k != "value"}, (i, x, y)
        assert LP.agree(ma, x["metric"], x["value"], y["value"]), (i, x["metric"], x["value"], y["value"])
        assert yc == LP.canonical(y["value"]) == LP.canonical(x["value"]), (i, x["metric"], yc, x["value"])
    return tolerated


def test_tolerances_are_the_declared_ones():
    assert LP.TOLERANCES_SHA256 == DECLARED_TOLERANCES_SHA256
    for module, metrics in LP.TOLERANCES.items():
        for name, t in metrics.items():
            assert 0 < t["abs"] <= 1e-9 and 0 < t["rel"] <= 1e-9, (module, name)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    out = tmp_path_factory.mktemp("bank") / "bank"
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/df_synthetic_bank.py"), "--out", str(out), "--limit", "3"],
                   check=True, capture_output=True)
    return out


def test_synthetic_units_rows_identical_or_within_declared_tolerance(old, bank, tmp_path):
    for i, job in enumerate(R.synthetic_jobs(bank)):
        d = tmp_path / f"u{i}"
        d.mkdir()
        compare(old_rows(old, job), new_rows(job, d))


def public_fixture(root: Path, T=3000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(149)
    d = root / "uci_321_fixture_panel"
    d.mkdir(parents=True)
    stamps = np.datetime64("2012-01-01T00:00:00") + np.arange(T) * np.timedelta64(15, "m")
    labels = [str(s).replace("T", " ") for s in stamps]
    labels[1000] = labels[999]                                   # a duplicate timestamp
    x0 = rng.standard_normal(T)
    x0[rng.integers(0, T, 60)] = np.nan
    x1 = np.cumsum(rng.standard_normal(T))
    ints = pa.array([None if i % 97 == 0 else int(v) for i, v in enumerate(rng.integers(0, 50, T))], type=pa.int64())
    table = pa.table({"timestamp_label": pa.array(labels), "load_a": pa.array(x0), "load_b": pa.array(x1),
                      "count_c": ints, "load_d": pa.array(np.roll(x1, 3) + rng.standard_normal(T)),
                      "site_name": pa.array(["s"] * T)})
    pq.write_table(table, d / "panel.parquet", row_group_size=700)
    digest = hashlib.sha256((d / "panel.parquet").read_bytes()).hexdigest()
    variables = [{"name": n, "variable_id": f"public.fixture.{n}", "role": "TIMESTAMP" if n == "timestamp_label"
                  else "INPUT_CANDIDATE"} for n in table.column_names]
    b0, b1 = int(T * 0.6), int(T * 0.8)
    contract = {"dataset_id": "public.fixture.uci_321_shaped", "contract_sha256": "a" * 64, "content_sha256": "b" * 64,
                "files": [{"name": "panel.parquet", "role": "DERIVED_CANONICAL_PANEL", "sha256": digest}],
                "original_fields": {"parse_receipt": {"parser": "uci_321"}},
                "variables": variables,
                "partitions": {"boundaries": {"train": [0, b0], "calibration": [b0, b1], "confirmation": [b1, T]}},
                "time": {"timestamp_meaning": "PERIOD_START", "frequency_nominal_seconds": 900}}
    (d / "CONTRACT.json").write_text(json.dumps(contract))
    return {"bank": "PUBLIC", "dir": str(d)}


def test_public_panel_shaped_fixture_rows_identical_or_within_declared_tolerance(old, tmp_path):
    job = public_fixture(tmp_path / "panels")
    a = old_rows(old, job)
    b = new_rows(job, tmp_path)
    assert any(m == "df_profile_multivariate" for m, _, _ in a)
    tolerated = compare(a, b)
    # the PCA, effective-rank and loading rows went through the declared tolerance, not through equality:
    # 4 numeric variables -> 4 PCA ratios, 2 effective ranks (multivariate, information), 3 loading rows each
    assert tolerated == 4 + 2 + 3 * 4


def test_columnar_reader_matches_whole_table_read(tmp_path):
    job = public_fixture(tmp_path / "panels")
    src = R.Source(job)
    import pyarrow.parquet as pq
    import pandas as pd
    table = pq.read_table(Path(job["dir"]) / "panel.parquet")
    ts_old = pd.to_datetime(pd.Series(table.column("timestamp_label").to_pylist()),
                            format="%Y-%m-%d %H:%M:%S").to_numpy("datetime64[ns]").astype("int64")
    assert np.array_equal(src.timestamps(), ts_old)
    assert [s["name"] for s in src.skipped] == ["site_name"]
    for j, v in enumerate(src.numeric):
        whole = np.asarray(table.column(v["name"]).to_numpy(zero_copy_only=False), dtype="float64")
        assert np.array_equal(src.column(j), whole, equal_nan=True)
        assert np.array_equal(src.column(j, 650, 2100), whole[650:2100], equal_nan=True)


def test_timestamp_column_int64_matches_the_old_pylist_parse(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    ns = (np.arange(5000, dtype=np.int64) * 3_600_000_000_000) + 1_230_000_000_000_000_000
    pq.write_table(pa.table({"timestamp": pa.array(ns, type=pa.timestamp("ns", tz="UTC"))}), tmp_path / "f.parquet",
                   row_group_size=1234)
    pf = pq.ParquetFile(tmp_path / "f.parquet")
    old = pd.to_datetime(pd.Series(pq.read_table(tmp_path / "f.parquet").column("timestamp").to_pylist()),
                         utc=True).to_numpy("datetime64[ns]").astype("int64")
    assert np.array_equal(R.read_timestamp_column(pf, "timestamp", 5000), old)


def test_the_whole_table_pattern_is_gone_from_the_runner():
    src = (ROOT / "tools/df_profile_run.py").read_text()
    assert "column_stack" not in src and "read_table(" not in src
