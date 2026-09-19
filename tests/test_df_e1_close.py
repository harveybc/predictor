"""RP34: the counterexamples of the dictum (F2), run against the FULL closing entry point on copies of
the preserved pilot — the originals are never written. Each case forges exactly one thing and the
closure must refuse THAT unit for THAT reason; the control case (an untouched copy) must verify.

The five cases the previous closure accepted: an MAE forged to 999, a MASE recorded as NaN, a foreign
design/data/cell_id, origins shifted by 123 rows, and a unit with neither weights nor job. Plus the
two that mattered most: a DESIGN with a zeroed digest and another horizon (which used to publish that
horizon with the same means), and a design whose digest is recomputed after the forgery.
"""
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
RUN = Path.home() / ".local/state/crispdm-data-foundation/e1_household_dev_pilot_v1"


def _load(name, where=HERE.parent / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("df_e1_close")
E = _load("df_mod_e0")

pytestmark = pytest.mark.skipif(not (RUN / "DESIGN.json").is_file(),
                                reason=f"the preserved pilot run is not present at {RUN}")


def _copy(tmp_path) -> Path:
    root = tmp_path / "run"
    shutil.copytree(RUN, root, ignore=shutil.ignore_patterns("closure_replays", "replays"))
    return root


def _rewrite_cell(root: Path, cell_id: str, mutate) -> None:
    """Rewrite a unit's record AND its digests, so the forgery is internally consistent: the closure
    must refuse it on the science, not on a digest the forger would obviously fix."""
    attempt = root / "attempts" / cell_id
    rec = json.loads((attempt / "cell.json").read_text())
    mutate(rec)
    body = json.dumps(rec, indent=1, sort_keys=True, default=float)
    (attempt / "cell.json").write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    result = json.loads((attempt / "result.json").read_text())
    result.update(output_sha256=digest, rows_written=body.count("\n") + (0 if body.endswith("\n") else 1))
    (attempt / "result.json").write_text(json.dumps(result))
    outcome = json.loads((attempt / "outcome.json").read_text())
    outcome["verified"]["output_sha256"] = digest
    (attempt / "outcome.json").write_text(json.dumps(outcome))


def _close(root: Path, **kw) -> dict:
    return C.close(root, do_replay=kw.pop("replay", False))


@pytest.fixture(scope="module")
def originals():
    return {str(p.relative_to(RUN)): C.sha_file(p) for p in RUN.rglob("*") if p.is_file()}


def test_RP34_an_untouched_copy_verifies_and_is_declared_historical_ungoverned(tmp_path):
    doc = _close(_copy(tmp_path))
    assert doc["verdict"] == "ALL_METRICS_VERIFIED_NO_REPLAY" and doc["counts"]["declared"] == 15
    assert doc["counts"]["metrics_verified"] == 15 and not doc["population"]["absent_ids"]
    assert doc["governance"]["units_historical_ungoverned"] == 15 and doc["governance"]["units_governed"] == 0
    assert all(u["scope"] in ("METRICS_VERIFIED_INFERENCE_NOT_REPLAYED", "SCIENTIFICALLY_VERIFIED_HISTORICAL_UNGOVERNED")
               for u in doc["units"].values())


@pytest.mark.parametrize("case", ["mae_999", "mase_nan", "mae_bool", "empty_arrays", "foreign_ids", "shifted_origins",
                                  "no_weights_no_job", "foreign_denominator", "labels_from_another_horizon",
                                  "updates_past_the_ceiling", "regime_relabelled"])
def test_RP34_the_dictum_counterexamples_are_refused_at_the_full_entry_point(tmp_path, case):
    root = _copy(tmp_path)
    cid = "R0_s1"
    attempt = root / "attempts" / cid
    if case == "mae_999":
        _rewrite_cell(root, cid, lambda r: r["scores"]["validation"]["model"].update(mae_mean=999.0))
        expect = "mae_mean"
    elif case == "mase_nan":
        _rewrite_cell(root, cid, lambda r: r["scores"]["validation"]["model"].update(mase_mean=float("nan")))
        expect = "non-finite"
    elif case == "mae_bool":
        _rewrite_cell(root, cid, lambda r: r["scores"]["validation"]["model"].update(mae_mean=True))
        expect = "not a number"
    elif case == "empty_arrays":
        z = dict(np.load(attempt / "arrays.npz"))
        z["validation_pred"] = z["validation_pred"][:0]
        np.savez(attempt / "arrays.npz", **z)
        _rewrite_cell(root, cid, lambda r: r.update(arrays_sha256=C.sha_file(attempt / "arrays.npz")))
        expect = "empty"
    elif case == "foreign_ids":
        _rewrite_cell(root, cid, lambda r: r.update(cell_id="R0_s9", design_sha256="0" * 64, data_sha256="0" * 64))
        expect = "another design"
    elif case == "shifted_origins":
        z = dict(np.load(attempt / "arrays.npz"))
        z["eval_origins"] = z["eval_origins"] + 123
        np.savez(attempt / "arrays.npz", **z)
        _rewrite_cell(root, cid, lambda r: r.update(arrays_sha256=C.sha_file(attempt / "arrays.npz")))
        expect = "evaluation origins are not the prepared ones"
    elif case == "no_weights_no_job":
        (attempt / "job.json").unlink()
        (attempt / "weights.weights.h5").unlink()
        expect = "records no job"
    elif case == "foreign_denominator":
        z = dict(np.load(attempt / "arrays.npz"))
        z["denominator"] = z["denominator"] * 2.0
        np.savez(attempt / "arrays.npz", **z)
        _rewrite_cell(root, cid, lambda r: r.update(arrays_sha256=C.sha_file(attempt / "arrays.npz")))
        expect = "denominator is not the prepared one"
    elif case == "labels_from_another_horizon":
        d = dict(np.load(root / "DATA.npz"))
        z = dict(np.load(attempt / "arrays.npz"))
        z["validation_y"] = d["Y"][np.asarray(z["eval_origins"], dtype=int) + 30][:, None]
        np.savez(attempt / "arrays.npz", **z)
        _rewrite_cell(root, cid, lambda r: r.update(arrays_sha256=C.sha_file(attempt / "arrays.npz")))
        expect = "labels are not the prepared targets"
    elif case == "updates_past_the_ceiling":
        _rewrite_cell(root, cid, lambda r: r["training"].update(updates=99999))
        expect = "past the sealed update ceiling"
    else:
        _rewrite_cell(root, cid, lambda r: r.update(regime="R2"))
        expect = "is not the sealed"
    doc = _close(root)
    unit = doc["units"][cid]
    assert not unit["verified"], f"{case} was accepted"
    assert any(expect in p for p in unit["problems"]), (case, unit["problems"])
    assert doc["verdict"] == "PARTIAL" and doc["counts"]["refused"] >= 1
    others = [c for c, u in doc["units"].items() if c != cid]
    assert all(doc["units"][c]["verified"] for c in others), "one forgery refused another unit"


def test_RP34_a_forged_design_is_refused_whether_its_digest_is_old_or_recomputed(tmp_path):
    for mode in ("old_digest", "recomputed_digest"):
        root = _copy(tmp_path / mode)
        design = json.loads((root / "DESIGN.json").read_text())
        design["task"]["horizon_steps"] = 120
        design["task"]["horizon_physical_seconds"] = 7200
        design["task"]["id"] = "W60_h120"
        if mode == "old_digest":
            design["design_sha256"] = "0" * 64
        else:
            body = {k: v for k, v in design.items() if k != "design_sha256"}
            design["design_sha256"] = E.sha_obj(body)                    # a forger who fixes the self-digest
        (root / "DESIGN.json").write_text(json.dumps(design, indent=1))
        doc = _close(root)
        assert doc["verdict"] != "ALL_VERIFIED", mode
        if mode == "old_digest":
            assert any("self-digest" in p for p in doc["register_problems"]), doc["register_problems"]
        else:
            # the digest recomputes, so the forgery is caught where it matters: DATA and every record
            # were produced under the real design, and the task each unit ran is not the published one
            assert any("another design" in p for u in doc["units"].values() for p in u["problems"])
            assert any("is not the sealed" in p for u in doc["units"].values() for p in u["problems"])
        assert all(not u["verified"] for u in doc["units"].values())


def test_RP34_a_transplanted_unit_and_a_missing_one_are_named_by_id(tmp_path):
    root = _copy(tmp_path)
    shutil.copytree(root / "attempts" / "R0_s2", root / "attempts" / "R0_s3", dirs_exist_ok=True)   # s2's attempt under s3's name
    shutil.rmtree(root / "attempts" / "R1_s3")
    doc = _close(root)
    assert "R1_s3" in doc["population"]["absent_ids"] and not doc["units"]["R1_s3"]["verified"]
    assert not doc["units"]["R0_s3"]["verified"]
    assert any("calls itself" in p or "differs from the design" in p for p in doc["units"]["R0_s3"]["problems"])
    assert doc["counts"]["declared"] == 15 and doc["counts"]["verified"] == 13


def test_RP34_a_stranger_attempt_on_disk_is_reported_not_absorbed(tmp_path):
    root = _copy(tmp_path)
    shutil.copytree(root / "attempts" / "R0_s1", root / "attempts" / "R9_s9")
    doc = _close(root)
    assert doc["population"]["strangers_on_disk"] == ["R9_s9"] and doc["counts"]["declared"] == 15


def test_RP34_the_replay_is_what_verifies_inference_and_a_gradient_summary_is_not(tmp_path):
    """With the replay off, inference and regime are NOT_ATTEMPTED even though the record carries a
    gradient report; with it on, altered weights are caught."""
    root = _copy(tmp_path)
    doc = C.close(root, do_replay=False)
    u = doc["units"]["R0_s1"]
    assert u["inference"] == "NOT_ATTEMPTED" and u["regime"] == "NOT_ATTEMPTED"
    assert u["scope"] == "METRICS_VERIFIED_INFERENCE_NOT_REPLAYED"           # never called verified-and-reproduced
    assert (root / "attempts" / "R0_s1" / "cell.json").read_text().count("gradient_proof") == 1
    # with the replay on, altered weights are caught: the stored predictions no longer follow from them
    os.environ["DF_E1_REPLAY_WINDOWS"] = "16"
    weights = root / "attempts" / "R0_s2" / "weights.weights.h5"
    weights.write_bytes((root / "attempts" / "R0_s3" / "weights.weights.h5").read_bytes())
    doc2 = C.close(root, do_replay=True, replay_units={"R0_s2"})
    u2 = doc2["units"]["R0_s2"]
    assert u2["inference"] == "REFUSED" and not u2["verified"]
    assert any("do not reproduce the stored predictions" in p for p in u2["problems"]), u2["problems"]


def test_RP34_the_originals_were_never_written(originals):
    now = {str(p.relative_to(RUN)): C.sha_file(p) for p in RUN.rglob("*") if p.is_file()}
    changed = [k for k in originals if now.get(k) != originals[k]]
    assert not changed, changed
