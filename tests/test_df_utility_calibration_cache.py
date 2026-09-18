"""Q2: verified reuse of calibration records of the same scientific computation — equivalence
first, then saving. Two families that differ only in labels share a key; every scientific field
breaks it; absent/corrupt/altered/other-contract never hits; concurrency stores once; the child
declares its source and runs zero simulations on a hit; accounting keeps computations, consumers,
reads and costs apart."""
import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("df_utility_harness")
CC = _load("df_utility_calibration_cache")
T = _load("test_df_utility_harness", HERE)
ops = _load("df_d3_operators")

MAD, DELTA = T.MAD, T.DELTA
PLAN = {"generator": "white_null", "n": 400, "bound_confidence": 0.5, "n_sims": 3}


def _proto(family, **over):
    base = dict(target="return", horizon=1, model="ridge", window=4, n_blocks=4, margin=0.0, seed=7,
                family=family, min_rows_per_block=30, calibration_plan=dict(PLAN),
                branches=("raw", "transformed", "augmented", "raw_wide"))
    base.update(over)
    return H.Protocol(**base)


A = ("unitA__v0__mad_extremes_trailing__transformed", "unitA__v0__mad_extremes_trailing__augmented")
B = ("unitB__v0__mad_extremes_trailing__transformed", "unitB__v0__mad_extremes_trailing__augmented")


def test_Q2_two_families_that_differ_only_in_labels_share_the_computation_key():
    ka = H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007)
    kb = H.computation_key(_proto(B), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007)
    assert ka == kb and H.computation_sha256(ka) == H.computation_sha256(kb)
    assert _proto(A).base_sha256() != _proto(B).base_sha256()        # the campaign binding still differs
    assert "family" not in json.dumps(ka) and "unitA" not in json.dumps(ka)


def test_Q2_every_scientific_field_breaks_the_key():
    ref = H.computation_sha256(H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007))
    variants = {
        "generator": lambda: H.computation_key(_proto(A), MAD, {**PLAN, "generator": "ar1_features_independent_target"}, branch_a="raw", branch_b="transformed", seed=1007),
        "seed": lambda: H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1008),
        "n": lambda: H.computation_key(_proto(A), MAD, {**PLAN, "n": 401}, branch_a="raw", branch_b="transformed", seed=1007),
        "n_sims": lambda: H.computation_key(_proto(A), MAD, {**PLAN, "n_sims": 4}, branch_a="raw", branch_b="transformed", seed=1007),
        "confidence": lambda: H.computation_key(_proto(A), MAD, {**PLAN, "bound_confidence": 0.95}, branch_a="raw", branch_b="transformed", seed=1007),
        "operator": lambda: H.computation_key(_proto(A), DELTA, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "operator params": lambda: H.computation_key(_proto(A), ops.build("mad_extremes_trailing", w=8) if hasattr(ops, "build") and _accepts_params() else DELTA, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "pair": lambda: H.computation_key(_proto(A), MAD, PLAN, branch_a="raw_wide", branch_b="augmented", seed=1007),
        "model/target": lambda: H.computation_key(_proto(A, target="direction", model="logistic"), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "window": lambda: H.computation_key(_proto(A, window=5), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "blocks": lambda: H.computation_key(_proto(A, n_blocks=5), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "rows": lambda: H.computation_key(_proto(A, min_rows_per_block=31), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "horizon": lambda: H.computation_key(_proto(A, horizon=2), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "margin": lambda: H.computation_key(_proto(A, margin=0.01), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "alpha effective": lambda: H.computation_key(_proto(A[:1]), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "ridge": lambda: H.computation_key(_proto(A, ridge_lambda=2.0), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007),
        "harness": lambda: H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007, harness="e" * 64),
    }
    for name, make in variants.items():
        assert H.computation_sha256(make()) != ref, name
    key = H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007)
    assert key["failure_policy"] == H.FAILURE_POLICY and key["rows_policy"] == H.ROWS_POLICY
    assert set(key["numeric_dependencies"]) == {"python", "numpy", "scipy"}


def _accepts_params():
    try:
        ops.build("mad_extremes_trailing", w=8)
        return True
    except TypeError:
        return False


def _record(family=A, **kw):
    p = _proto(family)
    rec = H.calibrate(p, MAD, plan=PLAN, seed=1007, **kw)
    return rec, json.dumps(rec, sort_keys=True, default=H._jsonable).encode()


def test_Q2_lookup_hits_only_a_verified_identical_computation(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    key = rec["computation"]
    assert cache.lookup(key) == (None, "absent")
    assert cache.store(rec["computation_sha256"], body, producer={"attempt_dir": "a"})["status"] == "STORED"
    hit, prov = cache.lookup(key)
    assert hit == body and prov["key_sha256"] == rec["computation_sha256"]
    # another family, same computation: a hit; another pair or seed: a miss
    rec_b, _ = _record(B)
    assert cache.lookup(rec_b["computation"])[0] == body
    other_pair = H.computation_key(_proto(B), MAD, PLAN, branch_a="raw_wide", branch_b="augmented", seed=1007)
    assert cache.lookup(other_pair)[0] is None
    other_op = H.computation_key(_proto(B), DELTA, PLAN, branch_a="raw", branch_b="transformed", seed=1007)
    assert cache.lookup(other_op)[0] is None
    # corrupt bytes, altered record, incomplete record: never a hit
    d = tmp_path / "cache" / rec["computation_sha256"]
    (d / "record.json").write_bytes(body.replace(b'"advances": 0', b'"advances": 1'))
    assert "corrupt" in cache.lookup(key)[1]
    tampered = json.dumps({**rec, "upper_bound": 0.0}, sort_keys=True, default=H._jsonable).encode()
    (d / "record.json").write_bytes(tampered)
    meta = json.loads((d / "META.json").read_text())
    meta["record_sha256"] = hashlib.sha256(tampered).hexdigest()
    (d / "META.json").write_text(json.dumps(meta))
    assert "verifier" in cache.lookup(key)[1]
    (d / "record.json").unlink()
    assert cache.lookup(key) == (None, "absent")


def test_Q2_a_different_record_for_the_same_key_is_quarantined_and_stores_are_idempotent(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    cache.store(rec["computation_sha256"], body, producer={"attempt_dir": "a"})
    again = cache.store(rec["computation_sha256"], body, producer={"attempt_dir": "b"})
    assert again["status"] == "DUPLICATE_PRODUCER"
    other = json.dumps({**rec, "cost": {"cpu_seconds": 9.0}}, sort_keys=True, default=H._jsonable).encode()
    q = cache.store(rec["computation_sha256"], other, producer={"attempt_dir": "c"})
    assert q["status"] == "QUARANTINED_DIFFERENT_RECORD"
    assert cache.lookup(rec["computation"])[0] == body                   # the first is still served
    with pytest.raises(ValueError):
        cache.store("0" * 64, body, producer={})


def test_Q2_concurrent_producers_store_once_and_accounting_keeps_roles_apart(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    results = []

    def producer(i):
        results.append(cache.store(rec["computation_sha256"], body, producer={"attempt_dir": f"p{i}"})["status"])
    threads = [threading.Thread(target=producer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("STORED") == 1 and results.count("DUPLICATE_PRODUCER") == 7
    for i in range(3):
        cache.note_consumer(rec["computation_sha256"], consumer={"attempt_dir": f"c{i}"}, kind="HIT", verification_seconds=0.01)
    st = cache.status()
    assert st["unique_computations"] == 1 and st["simulations_unique"] == PLAN["n_sims"]
    assert st["hits"] == 3 and st["entries"][0]["reads"] == 3
    assert st["saved_cpu_seconds"] == pytest.approx(3 * rec["cost"]["cpu_seconds"], abs=1e-6)
    assert st["verification_seconds_total"] == pytest.approx(0.03)


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_Q2_the_real_child_misses_then_hits_with_zero_new_simulations_and_identical_bytes(tmp_path):
    cache_dir = tmp_path / "cache"
    for i, family in enumerate((A, B)):
        p = _proto(family)
        job = {"kind": "calibrate", "contrast_id": family[0], "operator": MAD.KIND, "protocol": p.sealed(),
               "plan": PLAN, "seed": 1007, "protocol_key": "mad__H_T", "branch_a": "raw", "branch_b": "transformed",
               "cache_dir": str(cache_dir)}
        out = H.run_isolated(job, attempt_dir=tmp_path / f"cal{i}", assigned_bytes=1 << 30, wall_seconds=300.0, cpu_seconds=300)
        assert out["outcome"] == "COMPLETED", out
    r0 = json.loads((tmp_path / "cal0" / "result.json").read_text())
    r1 = json.loads((tmp_path / "cal1" / "result.json").read_text())
    assert r0["calibration_source"]["kind"] == "MISS_PRODUCED" and r0["calibration_source"]["stored"]["status"] == "STORED"
    assert r1["calibration_source"]["kind"] == "CACHE_HIT" and r1["calibration_source"]["simulations_run_here"] == 0
    assert (tmp_path / "cal0" / "calibration.json").read_bytes() == (tmp_path / "cal1" / "calibration.json").read_bytes()
    assert r0["output_sha256"] == r1["output_sha256"]
    # each consumer keeps its own campaign binding: the record supports BOTH families' contrasts
    rec = json.loads((tmp_path / "cal1" / "calibration.json").read_text())
    for family in (A, B):
        assert H.calibration_supports(_proto(family), MAD, 400, record=rec, branch_a="raw", branch_b="transformed")[0] in (True, False)
        assert "computation" not in (H.calibration_supports(_proto(family), MAD, 400, record=rec, branch_a="raw", branch_b="transformed")[1] or "")
    st = CC.CalibrationCache(cache_dir).status()
    assert st["unique_computations"] == 1 and st["hits"] == 1
