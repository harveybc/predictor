"""Fold-local stores for the nested walk-forward folds: decisions and constants come from the fold's TRAIN rows only
(the witness fails when a statistic peeks), std-zero columns are refused, near-duplicates keep the first column, the
elapsed-hours spike gets its declared robust transform."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from tools import nested_fold_stores as NF  # noqa: E402


def make_pool(tmp_path, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    close = 1.1 + np.cumsum(rng.normal(0, 1e-3, n))
    elapsed = np.ones(n)
    elapsed[::120] = 48.0
    pool = pd.DataFrame({"timestamp": ts, "HIGH": close + np.abs(rng.normal(0, 5e-4, n)), "CLOSE": close,
                         "RAW_CLOSE_LOG_RETURN": np.r_[np.nan, np.diff(np.log(close))],
                         "RAW_ELAPSED_HOURS": elapsed, "RSI": 50 + 20 * np.sin(np.arange(n) / 9),
                         "CONST": np.ones(n)})
    pool["CLOSE_TWIN"] = pool["CLOSE"] * 1.0000001        # exact near-duplicate of CLOSE
    f = tmp_path / "pool" / "full_pool_eurusd_h1.parquet"
    f.parent.mkdir()
    pool.to_parquet(f, index=False)
    man0 = {"source_sha256": "src", "file": f.name, "file_sha256": NF.sha(f),
            "decisions": {"HIGH": "DIFF", "CLOSE": "DIFF", "CLOSE_TWIN": "DIFF", "RAW_CLOSE_LOG_RETURN": "NONE", "RAW_ELAPSED_HOURS": "NONE", "RSI": "NONE", "CONST": "NONE"}}
    (f.parent / "MANIFEST.json").write_text(json.dumps(man0))
    return f, pool, man0


SLOTS = ["HIGH@diff", "RAW_CLOSE_LOG_RETURN", "RAW_ELAPSED_HOURS", "RSI", "CLOSE@diff", "CLOSE_TWIN@diff"]
WIN = ("2020-01-01", "2020-03-01")


def test_build_records_decisions_hygiene_and_passes_the_witness(tmp_path):
    f, pool, man0 = make_pool(tmp_path)
    man = NF.build_fold(pool, man0, SLOTS, "f1", WIN, tmp_path / "out" / "f1", ["2020-04-01", "2020-07-01"], ["2020-03-25", "2020-04-01"])
    assert man["learned_producers"] == [] and man["test_read"] is False and man["train_rows"] < len(pool)
    assert man["decisions"]["RAW_ELAPSED_HOURS"]["scope"] == "LOG1P_SHIFT" and man["decisions"]["RAW_ELAPSED_HOURS"]["shift"] == 1.0
    # CLOSE@diff ~ RAW_CLOSE_LOG_RETURN (R^2 > 0.99, as in the real pool): the first in list order is kept
    assert [d["dropped"] for d in man["hygiene"]["dedup_dropped"]] == ["CLOSE@diff", "CLOSE_TWIN@diff"] and "RAW_CLOSE_LOG_RETURN" in man["features"]
    assert "RSI" in man["features"] and man["decisions"]["RSI"]["scope"] == "NONE"
    built = pd.read_parquet(tmp_path / "out" / "f1" / man["file"])
    assert list(built.columns) == ["timestamp"] + man["features"] and len(built) == len(pool)
    assert np.allclose(built["RAW_ELAPSED_HOURS"].to_numpy()[::120][1:], np.log1p(47.0))
    doc = NF.check_fold(pool, man0, tmp_path / "out" / "f1")
    assert doc["status"] == "PASS" and doc["perturbed_rows"] > 0


def test_std_zero_column_is_refused(tmp_path):
    f, pool, man0 = make_pool(tmp_path)
    with pytest.raises(SystemExit, match="STD_ZERO_COLUMN:CONST"):
        NF.build_fold(pool, man0, SLOTS + ["CONST"], "f1", WIN, tmp_path / "out" / "f1")


def test_witness_fails_when_a_statistic_peeks_past_train(tmp_path, monkeypatch):
    f, pool, man0 = make_pool(tmp_path)
    NF.build_fold(pool, man0, SLOTS, "f1", WIN, tmp_path / "out" / "f1")
    real = NF.fold_columns

    def peeking(pool_, slots, global_dec, train):          # mutation: decides on EVERY row, not the TRAIN rows
        return real(pool_, slots, global_dec, np.ones(len(pool_), bool))
    monkeypatch.setattr(NF, "fold_columns", peeking)
    man = json.loads((tmp_path / "out" / "f1" / "MANIFEST.json").read_text())
    # the peeking build changes the LOG1P shift constant only if post-TRAIN rows move the minimum; force it
    pool2 = pool.copy()
    pool2.loc[pool2.index[-10:], "RAW_ELAPSED_HOURS"] = 0.5
    pool2.to_parquet(f, index=False)
    man0["file_sha256"] = NF.sha(f)
    (f.parent / "MANIFEST.json").write_text(json.dumps(man0))
    doc = NF.check_fold(pool2, man0, tmp_path / "out" / "f1")
    assert doc["status"] == "FAIL"


def test_dedup_keeps_first_in_order():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "b": a * 2 + 1e-9, "c": rng.normal(size=500)})
    keep, dropped = NF.dedup_pairs(X, ["a", "b", "c"])
    assert keep == ["a", "c"] and dropped[0]["dropped"] == "b" and dropped[0]["kept"] == "a"


def test_cli_build_and_check(tmp_path):
    f, pool, man0 = make_pool(tmp_path)
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"folds": {"f1": {"producer_fit_window": list(WIN)}, "f2": {"producer_fit_window": ["2020-01-01", "2020-04-01"]}}}))
    feats = tmp_path / "FEATURE_LIST.json"
    feats.write_text(json.dumps({"features": SLOTS}))
    NF.main(["build", "--pool", str(f), "--folds", str(folds), "--features", str(feats), "--out", str(tmp_path / "out")])
    NF.main(["check", "--pool", str(f), "--folds", str(folds), "--features", str(feats), "--out", str(tmp_path / "out")])
    for name in ("f1", "f2"):
        assert json.loads((tmp_path / "out" / name / "WITNESS.json").read_text())["status"] == "PASS"
    m1, m2 = (json.loads((tmp_path / "out" / n / "MANIFEST.json").read_text()) for n in ("f1", "f2"))
    assert m1["fit_window"] != m2["fit_window"] and m1["train_rows"] < m2["train_rows"]
    # identical decisions and constants give byte-identical columns: the fold identity lives in the MANIFEST (window, rows)
    assert NF.sha(tmp_path / "out" / "f1" / "MANIFEST.json") != NF.sha(tmp_path / "out" / "f2" / "MANIFEST.json")
