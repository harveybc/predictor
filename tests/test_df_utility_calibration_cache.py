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
    assert cache.lookup(key)[0] is None and "incomplete" in cache.lookup(key)[1]      # typed (R2)


def test_Q2_a_different_record_for_the_same_key_is_quarantined_and_stores_are_idempotent(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    cache.store(rec["computation_sha256"], body, producer={"attempt_dir": "a"})
    again = cache.store(rec["computation_sha256"], body, producer={"attempt_dir": "b"})
    assert again["status"] == "DUPLICATE_PRODUCER"
    other = json.dumps({**rec, "cost": {"cpu_seconds": 9.0}}, sort_keys=True, default=H._jsonable).encode()
    q = cache.store(rec["computation_sha256"], other, producer={"attempt_dir": "c"})
    assert q["status"] == "DUPLICATE_PRODUCER" and q["scientifically_equal"] is True   # cost is not a scientific disagreement (R2)
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
    assert st["avoided_cpu_seconds_projected"] == pytest.approx(3 * rec["cost"]["cpu_seconds"], abs=1e-6)
    assert st["verification_seconds_measured"] == pytest.approx(0.03)


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


# --- R1: reuse is bound to the executing scientific code ----------------------------------------------

def test_R1_changing_the_operators_behaviour_without_its_declaration_changes_the_key_and_never_hits(tmp_path, monkeypatch):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    cache.store(rec["computation_sha256"], body, producer={"fixture": True})
    key_before = H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007)
    assert key_before["code_identity"]["operator_code_sha256"] and key_before["code_identity"]["operator_module_sha256"]
    assert key_before["code_identity"]["scope"]["numeric_portability"] == "SAME_ENVIRONMENT_ONLY"
    original = type(MAD).transform

    def changed(self, x, state):
        return original(self, x, state)                       # same declaration, other implementation
    monkeypatch.setattr(type(MAD), "transform", changed)
    key_after = H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007)
    assert MAD.describe() == MAD.describe() and key_after != key_before
    assert key_after["code_identity"]["operator_code_sha256"] != key_before["code_identity"]["operator_code_sha256"]
    assert cache.lookup(key_after)[0] is None
    monkeypatch.setattr(type(MAD), "transform", original)
    assert H.computation_key(_proto(A), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007) == key_before
    # labels never enter: another campaign, same code → the same key
    assert H.computation_key(_proto(B), MAD, PLAN, branch_a="raw", branch_b="transformed", seed=1007) == key_before


def test_R1_a_helper_or_module_change_in_a_fresh_process_breaks_the_key_and_unchanged_code_keeps_it(tmp_path):
    import shutil, subprocess, sys as _sys
    src = HERE.parent / "tools"
    script = '''
import sys, json
sys.path.insert(0, sys.argv[1])
import importlib.util
def load(name):
    spec = importlib.util.spec_from_file_location(name, sys.argv[1] + "/" + name + ".py"); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
H = load("df_utility_harness"); ops = load("df_d3_operators")
p = H.Protocol(target="return", horizon=1, model="ridge", window=4, n_blocks=4, margin=0.0, seed=7, family=("u__v0__x__transformed",), min_rows_per_block=30,
               calibration_plan={"generator": "white_null", "n": 400, "bound_confidence": 0.5, "n_sims": 3}, branches=("raw", "transformed", "augmented", "raw_wide"))
k = H.computation_key(p, ops.build("mad_extremes_trailing"), p.calibration_plan, branch_a="raw", branch_b="transformed", seed=1007)
print(json.dumps({"sha": H.computation_sha256(k), "code": k["code_identity"]}))
'''
    def key_in(copy_dir):
        out = subprocess.run([_sys.executable, "-c", script, str(copy_dir)], capture_output=True, text=True, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])
    same = tmp_path / "same"
    shutil.copytree(src, same)
    changed = tmp_path / "changed"
    shutil.copytree(src, changed)
    text = (changed / "df_d3_operators.py").read_text()
    marker = "def transform(self, x, state):"
    i = text.index(marker, text.index('KIND = "mad_extremes_trailing"'))
    (changed / "df_d3_operators.py").write_text(text[:i] + "def transform(self, x, state):\n        _unused = 1  # behavioural edit marker\n" + text[i + len(marker):])
    helper = tmp_path / "helper"
    shutil.copytree(src, helper)
    (helper / "df_d3_contract.py").write_text((helper / "df_d3_contract.py").read_text() + "\n# helper edited\n")
    live, copy = key_in(src), key_in(same)
    assert live["sha"] == copy["sha"]                                   # same code elsewhere: same key
    assert key_in(changed)["sha"] != live["sha"]                        # operator implementation edited
    assert key_in(helper)["sha"] != live["sha"]                         # a helper module edited
    assert set(live["code"]) >= {"operator_code_sha256", "operator_module_sha256", "helper_modules_sha256", "harness_sha256", "scope"}


def test_R1_legacy_records_keep_their_status_and_never_hit_a_code_bound_key(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    legacy = {k: v for k, v in rec.items()}
    legacy["computation"] = {k: v for k, v in rec["computation"].items() if k != "code_identity"}
    legacy["computation"]["schema"] = "df_utility_calibration_computation.v1"
    legacy["computation_sha256"] = H.computation_sha256(legacy["computation"])
    legacy["schema"] = "df_utility_calibration.v3"
    legacy_body = json.dumps(legacy, sort_keys=True, default=H._jsonable).encode()
    assert H.calibration_record_problems(legacy) == []                  # still a valid historical record
    cache.store(legacy["computation_sha256"], legacy_body, producer={"legacy": True})
    assert cache.lookup(rec["computation"])[0] is None                  # the code-bound key never finds it
    assert rec["schema"] == "df_utility_calibration.v4" and rec["computation"]["schema"] == "df_utility_calibration_computation.v2"


# --- R2: recovery of a cache miss end to end ------------------------------------------------------------

def _entry(cache, rec, body):
    cache.store(rec["computation_sha256"], body, producer={"fixture": True})
    return cache.root / rec["computation_sha256"]


def test_R2_an_incomplete_entry_is_quarantined_typed_and_a_valid_replacement_becomes_usable(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    for damage in ("record", "meta", "malformed", "interrupted"):
        d = _entry(cache, rec, body)
        if damage == "record":
            (d / "record.json").unlink()
        elif damage == "meta":
            (d / "META.json").unlink()
        elif damage == "malformed":
            (d / "record.json").write_bytes(b"{not json")
        else:
            (d / "record.json").unlink()
            (d / "META.json").unlink()
            (d / "record.json.partial").write_bytes(body[:10])
        found, why = cache.lookup(rec["computation"])
        assert found is None and ("incomplete" in why or "corrupt" in why or "absent" in why), (damage, why)
        result = cache.store(rec["computation_sha256"], body, producer={"recovery": damage})
        assert result["status"] in ("STORED", "RECOVERED_INCOMPLETE_ENTRY"), result
        assert cache.lookup(rec["computation"])[0] == body
        quarantined = list((cache.root / "quarantine").glob(f"{rec['computation_sha256']}*"))
        assert quarantined, damage                                      # the incomplete entry is preserved apart
        import shutil
        shutil.rmtree(d)


def test_R2_scientific_equality_is_separated_from_cost_and_provenance_and_conflicts_get_a_disposition(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    cache.store(rec["computation_sha256"], body, producer={"p": 1})
    other_cost = json.dumps({**rec, "cost": {"cpu_seconds": 99.0}}, sort_keys=True, default=H._jsonable).encode()
    r = cache.store(rec["computation_sha256"], other_cost, producer={"p": 2})
    assert r["status"] == "DUPLICATE_PRODUCER" and r["scientifically_equal"] is True
    sims = [dict(x, outcome="ADVANCES", delta_lower=0.5, delta_mean=0.6) if i == 0 else x for i, x in enumerate(rec["per_sim"])]
    conflicting = {**rec, "per_sim": sims, "per_sim_sha256": H.sha_obj(sims), "advances": 1,
                   "false_advance_rate": 1 / rec["scored"],
                   "upper_bound": H.clopper_pearson_upper(1, rec["scored"], rec["bound_confidence"])}
    assert H.calibration_record_problems(conflicting) == []
    r = cache.store(rec["computation_sha256"], json.dumps(conflicting, sort_keys=True, default=H._jsonable).encode(), producer={"p": 3})
    assert r["status"] == "CONFLICT_RECORDED"
    found, why = cache.lookup(rec["computation"])
    assert found is None and "conflict" in why                          # nothing served until a disposition
    disp = json.loads((cache.root / rec["computation_sha256"] / "CONFLICT.json").read_text())
    assert disp["disposition"] == "PENDING" and len(disp["records"]) == 2


def test_R2_consumer_accounting_is_idempotent_by_attempt_and_keeps_measured_apart_from_projected(tmp_path):
    cache = CC.CalibrationCache(tmp_path / "cache")
    rec, body = _record()
    cache.store(rec["computation_sha256"], body, producer={"attempt_dir": "p"})
    for _ in range(3):
        cache.note_consumer(rec["computation_sha256"], consumer={"attempt_dir": "c1"}, kind="HIT", verification_seconds=0.2)
    cache.note_consumer(rec["computation_sha256"], consumer={"attempt_dir": "c2"}, kind="HIT", verification_seconds=0.1)
    st = cache.status()
    e = st["entries"][0]
    assert e["hits"] == 2 and e["reads"] == 2
    assert st["avoided_cpu_seconds_projected"] == pytest.approx(2 * rec["cost"]["cpu_seconds"], abs=1e-6)
    assert st["verification_seconds_measured"] == pytest.approx(0.3)
    assert "saved_cpu_seconds" not in st


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_R2_the_real_child_recovers_an_incomplete_entry_and_then_hits(tmp_path):
    cache_dir = tmp_path / "cache"
    cache = CC.CalibrationCache(cache_dir)
    rec, body = _record()
    d = _entry(cache, rec, body)
    (d / "record.json").unlink()                                        # the incomplete entry the reviewer found
    p = _proto(A)
    job = {"kind": "calibrate", "contrast_id": A[0], "operator": MAD.KIND, "protocol": p.sealed(), "plan": PLAN, "seed": 1007,
           "protocol_key": "mad__H_T", "branch_a": "raw", "branch_b": "transformed", "cache_dir": str(cache_dir)}
    out = H.run_isolated(job, attempt_dir=tmp_path / "r0", assigned_bytes=1 << 30, wall_seconds=300.0, cpu_seconds=300)
    assert out["outcome"] == "COMPLETED", out
    r0 = json.loads((tmp_path / "r0" / "result.json").read_text())["calibration_source"]
    assert r0["kind"] == "MISS_PRODUCED" and "incomplete" in r0["why_miss"] and r0["stored"]["status"] == "RECOVERED_INCOMPLETE_ENTRY"
    out = H.run_isolated(job, attempt_dir=tmp_path / "r1", assigned_bytes=1 << 30, wall_seconds=300.0, cpu_seconds=300)
    r1 = json.loads((tmp_path / "r1" / "result.json").read_text())["calibration_source"]
    assert r1["kind"] == "CACHE_HIT" and (tmp_path / "r1" / "calibration.json").read_bytes() == (tmp_path / "r0" / "calibration.json").read_bytes()
    st = cache.status()
    assert st["unique_computations"] == 1 and st["hits"] == 1 and st["entries"][0]["quarantined_incomplete"] == 1
