"""RP10 / RP11: the MOD-E0 closure is bound to the population the sealed design implies and
verifies the task, the weights and the metrics from the files, through the real CLI/API on
COPIES of a small real campaign (real generator, real training with a tiny update allowance,
real weights, real fresh-process replays). Every counterexample of the review dictum is a case
here and each one now refuses; the intact root closes TOTAL with exit 0."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name, where=TOOLS):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("df_mod_e0")
D = _load("df_mod_e0_design")
CLOSE = _load("df_mod_e0_close")
RUN_ID = "close-test"
PILOT_UPDATES = 4


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _design():
    doc = D.build(max_updates=6, successor_of="a" * 64, reason="test stage")
    doc.update(levels=[3], replicates=[1], random_assignments=1)
    doc["cells"] = D.cells(doc)
    doc["cells_total"] = len(doc["cells"])
    doc["design_sha256"] = E.sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return doc


def _outcome(attempt, cost=1.0):
    digest = _sha(attempt / "cell.json")
    doc = {"status": "COMPLETED", "verified": {"output_sha256": digest, "rows_written": 1},
           "summary": {"outcome": "COMPLETED", "reason": "", "cost": {"cpu_seconds": cost, "wall_seconds": cost, "peak_rss_bytes": 1, "cgroup_memory_peak": None,
                                                                     "started_at": "2026-09-18T12:00:00Z", "ended_at": "2026-09-18T12:00:01Z"},
                       "output_sha256": digest}}
    (attempt / "outcome.json").write_text(json.dumps(doc))
    return digest


def build_root(root: Path) -> dict:
    """A real, small campaign the way the runner lays it out: pilots then cells, each attempt
    with job.json (the runner's job), cell.json + result.json (the worker's), outcome.json
    (the runner's receipt); DESIGN.json, CAMPAIGNS.json, REPORT.json."""
    design = _design()
    root.mkdir(parents=True)
    (root / "DESIGN.json").write_text(json.dumps(design, indent=1, sort_keys=True))
    pop = CLOSE.population(design)
    regs = {f"{RUN_ID}{s}": {"campaign_sha256": hashlib.sha256(s.encode()).hexdigest(), "http": 201, "at": "2026-09-18T12:00:00Z"} for s in pop["campaigns"]}
    (root / "CAMPAIGNS.json").write_text(json.dumps(regs))
    report = {"schema": "df_mod_e0_report.v1", "run_id": RUN_ID, "design_sha256": design["design_sha256"], "code_identity": {"kind": "git_commit", "value": "0" * 40},
              "cells": {}, "terminals": [], "cost_pilot": {}, "stopped": None}
    for unit in pop["pilots"] + pop["cells"]:
        job = CLOSE.expected_job(design, RUN_ID, unit, root, PILOT_UPDATES)
        if unit.get("depends_on"):
            job["depends_on"] = unit["depends_on"]
        attempt = root / "attempts" / unit["cell_id"]
        attempt.mkdir(parents=True)
        job_file = attempt / "job.json"
        job_file.write_text(json.dumps({**job, "attempt_dir": str(attempt)}))
        E.worker_main(job_file)
        _outcome(attempt)
        rec = json.loads((attempt / "cell.json").read_bytes())
        report["terminals"].append({"unit_id": unit["cell_id"], "status": "COMPLETED"})
        if job["role"] == "CELL":
            report["cells"][unit["cell_id"]] = {"outcome": "COMPLETED", "mase_validation": rec["scores"]["validation"]["model"]["mase_mean"],
                                                 "updates": rec["training"]["updates"]}
        else:
            report["cost_pilot"][unit["cell_id"]] = {"cpu_seconds": 1.0, "updates": rec["training"]["updates"]}
    (root / "REPORT.json").write_text(json.dumps(report, indent=1))
    return design


@pytest.fixture(scope="session")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("mod_e0_close") / "run"
    design = build_root(root)
    out = tmp_path_factory.mktemp("closure_intact")
    local = CLOSE.local_closure(root, out, replays=True, workers=4)
    return {"root": root, "design": design, "local": local, "replays": out / "replays"}


def _copy(campaign, tmp_path, name="copy"):
    """A disposable copy of the root; prior replays are reused only for attempts whose bytes are unchanged."""
    root = tmp_path / name
    shutil.copytree(campaign["root"], root)
    for attempt in (root / "attempts").iterdir():
        job_file = attempt / "job.json"
        job = json.loads(job_file.read_text())
        for k in ("extractor_weights", "attempt_dir"):
            if job.get(k):
                job[k] = job[k].replace(str(campaign["root"]), str(root))
        job_file.write_text(json.dumps(job))
    out = tmp_path / f"{name}_closure"
    shutil.copytree(campaign["replays"], out / "replays")
    return root, out


def _reseal(attempt: Path, rec: dict, arr: dict | None = None):
    """Re-sign the copy's receipts so that only the CONTENT is wrong (as in the review's counterexamples)."""
    if arr is not None:
        np.savez(attempt / "arrays.npz", **arr)
        rec["arrays_sha256"] = _sha(attempt / "arrays.npz")
    if (attempt / "weights.weights.h5").is_file():
        rec["weights_sha256"] = _sha(attempt / "weights.weights.h5")
    (attempt / "cell.json").write_bytes(json.dumps(rec, sort_keys=True, default=float).encode())
    digest = _sha(attempt / "cell.json")
    for name in ("result.json", "outcome.json"):
        doc = json.loads((attempt / name).read_text())
        dst = doc if name == "result.json" else doc["verified"]
        dst["output_sha256"] = digest
        if name == "outcome.json":
            doc["summary"]["output_sha256"] = digest
        (attempt / name).write_text(json.dumps(doc))


def _rescore(rec, arr):
    for part in [q for q in ("train", "validation", "test") if f"{q}_y" in arr]:
        for model, src in CLOSE.BASELINES:
            rec["scores"][part][model] = E.mase(arr[f"{part}_{src}"], arr[f"{part}_y"], arr["denominator"].tolist())
            rec["scores"][part]["rows"] = int(arr[f"{part}_y"].shape[0])


def _load_arrays(attempt):
    with np.load(attempt / "arrays.npz") as z:
        return {k: z[k].copy() for k in z.files}


def _cli(root, out, *extra):
    proc = subprocess.run([sys.executable, "-B", str(TOOLS / "df_mod_e0_close.py"), "--root", str(root), "--out-dir", str(out), "--no-live", *extra],
                          capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=1800)
    return proc.returncode, proc.stdout, proc.stderr


# --- the intact campaign ------------------------------------------------------------------------------------

def test_RP10_intact_campaign_closes_TOTAL_over_the_population_the_design_implies(campaign):
    local = campaign["local"]
    assert local["closure"] == CLOSE.TOTAL and local["all_verified"] and local["parent_equal"]
    assert local["population"]["cells"] == campaign["design"]["cells_total"] == 8 and local["population"]["pilots"] == 3
    assert local["counts"] == {CLOSE.VERIFIED: 11} and local["not_verified"] == []
    assert local["effects_population"]["complete"] and "H2" in local["effects"] and "H3" in local["effects"]
    for u in local["units"].values():
        assert all(u["checks"].values()), u
        assert u["measured"]["replay"]["prediction_max_abs_diff"]["validation"] <= local["tolerance"]["prediction_atol"]
    arm = local["units"]["H3__r1__s1__sequence"]["measured"]["replay"]
    assert arm["extractor_weights_unequal_layers"] == [] and max(arm["adapter_activation_max_abs_diff"].values()) == 0.0
    assert local["replays"]["cpu_seconds_children"] > 0
    # RP11: every score of every baseline and variable was recomputed; the states are explicit
    rc = local["units"]["H2__h3__s1__profiles"]["recomputed"]
    assert set(rc) == {"train", "validation", "test"} and set(rc["validation"]) == {"model", "naive", "oracle", "linear_window"}
    assert all(v["status"] == "MEDIDO" for v in rc["validation"].values())


def test_RP10_the_real_cli_returns_zero_only_for_a_total_closure(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path)
    code, stdout, stderr = _cli(root, out)
    assert code == 0, stderr[-2000:]
    doc = json.loads((out / "CLOSE.json").read_text())
    assert doc["local"]["closure"] == CLOSE.TOTAL and doc["live"] is None and doc["verdict"]["live_all_equal"] is None
    # never written over
    code2, _, _ = _cli(root, out)
    assert code2 != 0


# --- RP10 population counterexamples (each through the real API; the CLI for the exit codes) ---------------

def test_RP10_an_empty_root_is_refused_and_a_root_without_attempts_is_PARTIAL_with_every_member_missing(campaign, tmp_path):
    empty = tmp_path / "empty"
    (empty / "attempts").mkdir(parents=True)
    with pytest.raises(CLOSE.ClosureRefusal):
        CLOSE.local_closure(empty, tmp_path / "o1", replays=False)
    code, _, _ = _cli(empty, tmp_path / "o1b")
    assert code == 2
    root, out = _copy(campaign, tmp_path)
    shutil.rmtree(root / "attempts")
    (root / "attempts").mkdir()
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["closure"] == CLOSE.PARTIAL and local["all_verified"] is False and local["parent_equal"] is False
    assert local["counts"] == {CLOSE.MISSING: 11} and local["effects"] == {} and local["effects_population"]["verified_cells"] == 0
    assert local["denominator"]["cells"] == 8


def test_RP10_six_of_the_population_present_is_PARTIAL_with_the_denominator_fixed_not_a_closure(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path)
    keep = {"H3__r1__s1__extractor", "H3__r1__s1__sequence", "H3__r1__s1__summary"}
    for attempt in list((root / "attempts").iterdir()):
        if attempt.name not in keep:
            shutil.rmtree(attempt)
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["closure"] == CLOSE.PARTIAL and not local["all_verified"]
    assert local["counts"][CLOSE.MISSING] == 8 and local["counts"][CLOSE.VERIFIED] == 3
    assert local["effects_population"] == {"verified_cells": 3, "of": 8, "scope": "VERIFIED_MEMBERS_ONLY", "complete": False}
    assert "H2" not in local["effects"] and local["effects"]["H3"]["gamma"] is None
    code, _, _ = _cli(root, tmp_path / "cli_out")
    assert code == 1


def test_RP10_omitted_member_duplicate_terminal_and_stranger_are_typed(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path, "omit")
    shutil.rmtree(root / "attempts" / "H2__h3__s1__random_0")
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["units"]["H2__h3__s1__random_0"]["status"] == CLOSE.MISSING and local["closure"] == CLOSE.PARTIAL
    assert local["not_verified"] == ["H2__h3__s1__random_0"] and not local["effects"]["H2"]["e"]   # no random arm -> no e(h)
    root, out = _copy(campaign, tmp_path, "dup")
    report = json.loads((root / "REPORT.json").read_text())
    report["terminals"].append(dict(report["terminals"][0]))
    (root / "REPORT.json").write_text(json.dumps(report))
    with pytest.raises(CLOSE.ClosureRefusal, match="duplicated terminal"):
        CLOSE.local_closure(root, out, replays=False)
    root, out = _copy(campaign, tmp_path, "stranger")
    shutil.copytree(root / "attempts" / "H2__h3__s1__profiles", root / "attempts" / "H2__h9__s1__profiles")
    with pytest.raises(CLOSE.ClosureRefusal, match="not members"):
        CLOSE.local_closure(root, out, replays=False)
    root, out = _copy(campaign, tmp_path, "stranger_report")
    report = json.loads((root / "REPORT.json").read_text())
    report["cells"]["H2__h9__s1__profiles"] = {"outcome": "COMPLETED", "mase_validation": 0.1}
    (root / "REPORT.json").write_text(json.dumps(report))
    with pytest.raises(CLOSE.ClosureRefusal, match="not members"):
        CLOSE.local_closure(root, out, replays=False)


def test_RP10_an_altered_design_is_refused_and_a_resealed_design_unbinds_every_job(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path, "altered")
    design = json.loads((root / "DESIGN.json").read_text())
    design["training"]["max_updates"] = 999
    (root / "DESIGN.json").write_text(json.dumps(design))
    with pytest.raises(CLOSE.ClosureRefusal, match="seal"):
        CLOSE.local_closure(root, out, replays=False)
    # a consistently re-sealed design: the report names another design -> refused; without a report the jobs disagree
    design["design_sha256"] = E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"})
    (root / "DESIGN.json").write_text(json.dumps(design))
    with pytest.raises(CLOSE.ClosureRefusal, match="another design"):
        CLOSE.local_closure(root, out, replays=False)
    (root / "REPORT.json").unlink()
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["closure"] == CLOSE.PARTIAL and local["counts"].get(CLOSE.PROBLEMS, 0) + local["counts"].get(CLOSE.NO_DONOR, 0) == 11
    assert any("job.design_sha256" in q for q in local["units"]["H2__h3__s1__profiles"]["problems"])
    # the design's cells must be its own enumeration
    root, out = _copy(campaign, tmp_path, "cells")
    design = json.loads((root / "DESIGN.json").read_text())
    design["cells"] = design["cells"][:-1]
    design["cells_total"] = len(design["cells"])
    design["design_sha256"] = E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"})
    (root / "DESIGN.json").write_text(json.dumps(design))
    with pytest.raises(CLOSE.ClosureRefusal, match="own enumeration"):
        CLOSE.local_closure(root, out, replays=False)


def test_RP10_a_parent_disagreement_never_leaves_all_verified_true(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path)
    report = json.loads((root / "REPORT.json").read_text())
    report["cells"]["H3__r1__s1__sequence"]["mase_validation"] = 999999.0
    (root / "REPORT.json").write_text(json.dumps(report))
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["parent_equal"] is False and local["all_verified"] is False and local["closure"] == CLOSE.PARTIAL
    assert local["parent"]["H3__r1__s1__sequence"]["equal"] is False
    assert local["units"]["H3__r1__s1__sequence"]["status"] == CLOSE.VERIFIED       # the file itself is fine: the parent is not
    assert "parent" in " ".join(local["units"]["H3__r1__s1__sequence"]["problems"])
    assert local["all_verified"] == (not local["not_verified"] and local["parent_equal"])


def test_RP10_a_missing_replicate_or_arm_of_the_design_is_MISSING_and_a_failed_attempt_is_FAILED(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path, "rep")
    design = json.loads((root / "DESIGN.json").read_text())
    design["replicates"] = [1, 2]
    design["cells"] = D.cells(design)
    design["cells_total"] = len(design["cells"])
    design["design_sha256"] = E.sha_obj({k: v for k, v in design.items() if k != "design_sha256"})
    (root / "DESIGN.json").write_text(json.dumps(design))
    report = json.loads((root / "REPORT.json").read_text())
    report["design_sha256"] = design["design_sha256"]
    (root / "REPORT.json").write_text(json.dumps(report))
    for attempt in (root / "attempts").iterdir():           # the jobs of seed 1 must carry the new design identity to stay bound
        job = json.loads((attempt / "job.json").read_text())
        job["design_sha256"] = design["design_sha256"]
        (attempt / "job.json").write_text(json.dumps(job))
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["denominator"]["cells"] == 16 and local["counts"][CLOSE.MISSING] == 8 and local["closure"] == CLOSE.PARTIAL
    assert local["units"]["H3__r1__s2__summary"]["status"] == CLOSE.MISSING
    root, out = _copy(campaign, tmp_path, "failed")
    attempt = root / "attempts" / "H3__r0__s1__extractor"
    (attempt / "outcome.json").write_text(json.dumps({"status": "RESOURCE_EXCEEDED", "verified": None,
                                                      "summary": {"outcome": "RESOURCE_EXCEEDED", "reason": "WALL", "cost": {"cpu_seconds": 1.0}}}))
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["units"]["H3__r0__s1__extractor"]["status"] == CLOSE.FAILED
    assert local["units"]["H3__r0__s1__sequence"]["status"] == CLOSE.NO_DONOR and local["units"]["H3__r0__s1__summary"]["status"] == CLOSE.NO_DONOR
    assert local["closure"] == CLOSE.PARTIAL and local["effects"]["H3"]["gamma"] is None and 1 in local["effects"]["H3"]["d"]


def test_RP10_an_H3_receiver_without_a_verifiable_donor_is_not_verified(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path)
    donor = root / "attempts" / "H3__r1__s1__extractor"
    (donor / "weights.weights.h5").write_bytes(b"not model weights")
    local = CLOSE.local_closure(root, out, replays=True)
    assert local["units"]["H3__r1__s1__extractor"]["status"] == CLOSE.PROBLEMS
    for arm in ("H3__r1__s1__sequence", "H3__r1__s1__summary"):
        assert local["units"][arm]["status"] in (CLOSE.PROBLEMS, CLOSE.NO_DONOR)      # its own digest check sees the donor first
        assert any("donor" in q for q in local["units"][arm]["problems"])
    # a donor that is merely unverified (not corrupted) still denies its arms
    root2, out2 = _copy(campaign, tmp_path, "unverified_donor")
    donor2 = root2 / "attempts" / "H3__r1__s1__extractor"
    rec = json.loads((donor2 / "cell.json").read_bytes())
    rec["scores"]["validation"]["model"]["mae_mean"] = 999999.0                  # the donor's own record lies; its weights are intact
    _reseal(donor2, rec)
    local2 = CLOSE.local_closure(root2, out2, replays=True)
    assert local2["units"]["H3__r1__s1__extractor"]["status"] == CLOSE.PROBLEMS
    assert local2["units"]["H3__r1__s1__sequence"]["status"] == CLOSE.NO_DONOR
    assert local["closure"] == CLOSE.PARTIAL


# --- RP11 content counterexamples: each altered copy keeps consistent receipts and is rejected --------------

def _altered(campaign, tmp_path, name, mutate):
    root, out = _copy(campaign, tmp_path, name)
    attempt = root / "attempts" / "H3__r1__s1__sequence"
    rec = json.loads((attempt / "cell.json").read_bytes())
    arr = _load_arrays(attempt)
    new_arr = mutate(attempt, rec, arr)
    _reseal(attempt, rec, new_arr)
    local = CLOSE.local_closure(root, out, replays=True)
    return local, local["units"]["H3__r1__s1__sequence"]


def test_RP11_labels_equal_to_predictions_are_rejected(campaign, tmp_path):
    def mutate(attempt, rec, arr):
        for part in ("train", "validation", "test"):
            arr[f"{part}_y"] = arr[f"{part}_pred"].copy()
        _rescore(rec, arr)
        return arr
    local, u = _altered(campaign, tmp_path, "labels", mutate)
    assert u["status"] == CLOSE.PROBLEMS and any("validation.y: not the generator's" in q for q in u["problems"])
    assert local["closure"] == CLOSE.PARTIAL


def test_RP11_a_scaled_denominator_is_rejected(campaign, tmp_path):
    def mutate(attempt, rec, arr):
        arr["denominator"] = arr["denominator"] * 100
        rec["mase_denominator"] = arr["denominator"].tolist()
        _rescore(rec, arr)
        return arr
    local, u = _altered(campaign, tmp_path, "denominator", mutate)
    assert u["status"] == CLOSE.PROBLEMS and any("denominators are not the train seasonal-naive" in q for q in u["problems"])


def test_RP11_shifted_row_ids_are_rejected(campaign, tmp_path):
    def mutate(attempt, rec, arr):
        for part in ("train", "validation", "test"):
            arr[f"{part}_rows"] = arr[f"{part}_rows"] + 100000
        return arr
    local, u = _altered(campaign, tmp_path, "rows", mutate)
    assert u["status"] == CLOSE.PROBLEMS and any("rows: not the generator's" in q for q in u["problems"])


def test_RP11_an_altered_baseline_or_MAE_in_the_record_is_rejected(campaign, tmp_path):
    def mutate(attempt, rec, arr):
        rec["scores"]["validation"]["model"]["mae_mean"] = 999999.0
        rec["scores"]["validation"]["linear_window"]["mase_mean"] = 999999.0
        return None
    local, u = _altered(campaign, tmp_path, "mae", mutate)
    assert any("validation.model.mae_mean" in q for q in u["problems"]) and any("validation.linear_window.mase_mean" in q for q in u["problems"])
    # per-variable alterations are seen too
    def mutate2(attempt, rec, arr):
        rec["scores"]["validation"]["naive"]["per_variable"]["3"]["mae"] = 0.0
        return None
    local, u = _altered(campaign, tmp_path, "var", mutate2)
    assert any("validation.naive.var3.mae" in q for q in u["problems"])


def test_RP11_unreadable_or_changed_weights_are_rejected_by_the_fresh_process_replay(campaign, tmp_path):
    def unreadable(attempt, rec, arr):
        (attempt / "weights.weights.h5").write_bytes(b"not model weights")
        return None
    local, u = _altered(campaign, tmp_path, "unreadable", unreadable)
    assert u["status"] == CLOSE.PROBLEMS and any("weights unreadable" in q for q in u["problems"])
    def changed(attempt, rec, arr):
        other = attempt.parent / "H3__r0__s1__sequence" / "weights.weights.h5"     # valid weights of another cell
        shutil.copy2(other, attempt / "weights.weights.h5")
        return None
    local, u = _altered(campaign, tmp_path, "changed", changed)
    assert u["status"] == CLOSE.PROBLEMS
    assert any("predictions from the reloaded weights differ" in q for q in u["problems"])
    assert u["measured"]["replay"]["prediction_max_abs_diff"]["validation"] > local["tolerance"]["prediction_atol"]


def test_RP11_the_parity_flag_alone_is_not_evidence_the_replayed_arrays_are(campaign, tmp_path):
    def mutate(attempt, rec, arr):
        assert rec["prediction_parity_after_reload"] is True
        arr["validation_pred"] = arr["validation_naive"].copy()          # the flag stays True; the arrays are not the weights'
        _rescore(rec, arr)
        return arr
    local, u = _altered(campaign, tmp_path, "parity", mutate)
    assert u["status"] == CLOSE.PROBLEMS and any("validation predictions from the reloaded weights differ" in q for q in u["problems"])


def test_RP11_an_unfrozen_or_foreign_extractor_is_rejected(campaign, tmp_path):
    def foreign(attempt, rec, arr):
        job = json.loads((attempt / "job.json").read_text())
        job["extractor_weights"] = str(attempt.parent / "H3__r0__s1__extractor" / "weights.weights.h5")
        (attempt / "job.json").write_text(json.dumps(job))
        rec["frozen"]["extractor_sha256"] = _sha(attempt.parent / "H3__r0__s1__extractor" / "weights.weights.h5")
        return None
    local, u = _altered(campaign, tmp_path, "foreign", foreign)
    assert u["status"] in (CLOSE.PROBLEMS, CLOSE.NO_DONOR)
    assert any("job.extractor_weights" in q for q in u["problems"])
    assert any("extractor weights differ from the donor" in q for q in u["problems"]) and u["measured"]["replay"]["extractor_weights_unequal_layers"]
    def unfrozen(attempt, rec, arr):
        rec["extractor_weight_change"] = 0.5
        return None
    local, u = _altered(campaign, tmp_path, "unfrozen", unfrozen)
    assert any("does not show a frozen extractor" in q for q in u["problems"])


def test_RP11_non_finite_or_empty_arrays_are_never_MEDIDO(campaign, tmp_path):
    def nan_claimed(attempt, rec, arr):
        arr["validation_pred"][3, 2] = np.nan                                 # the record keeps claiming MEDIDO
        return arr
    local, u = _altered(campaign, tmp_path, "nan", nan_claimed)
    assert u["status"] == CLOSE.PROBLEMS and any("non-finite or empty arrays recorded as measured" in q for q in u["problems"])
    assert u["recomputed"]["validation"]["model"]["status"] == E.NO_MEDIDO
    def inf_honest(attempt, rec, arr):
        arr["validation_pred"][0, 0] = np.inf
        _rescore(rec, arr)                                                     # honest: the score becomes NO_MEDIDO with None means
        assert rec["scores"]["validation"]["model"]["status"] == E.NO_MEDIDO and rec["scores"]["validation"]["model"]["mase_mean"] is None
        return arr
    local, u = _altered(campaign, tmp_path, "inf", inf_honest)
    assert u["status"] == CLOSE.PROBLEMS            # an honest NO_MEDIDO cell is not a verified score either: its predictions are not the weights'
    assert not any("recorded as measured" in q for q in u["problems"])
    # the metric itself
    out = E.mase(np.array([[np.nan, 1.0]]), np.array([[1.0, 1.0]]), [1.0, 1.0])
    assert out["status"] == E.NO_MEDIDO and out["mase_mean"] is None and out["mae_mean"] is None and out["per_variable"][0]["reason"] == "NON_FINITE"
    assert out["per_variable"][1]["status"] == E.MEDIDO
    assert E.mase(np.zeros((0, 2)), np.zeros((0, 2)), [1.0, 1.0])["status"] == E.NO_MEDIDO
    with pytest.raises(ValueError):
        E.mase(np.zeros((3, 2)), np.zeros((3, 3)), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        E.mase(np.zeros((3, 2)), np.zeros((3, 2)), [1.0])


def test_RP11_a_random_arm_that_is_not_the_predefined_redistribution_is_rejected(campaign, tmp_path):
    root, out = _copy(campaign, tmp_path)
    attempt = root / "attempts" / "H2__h3__s1__random_0"
    rec = json.loads((attempt / "cell.json").read_bytes())
    rec["assignment"] = rec["assignment"][::-1]
    _reseal(attempt, rec)
    local = CLOSE.local_closure(root, out, replays=True)
    u = local["units"]["H2__h3__s1__random_0"]
    assert u["status"] == CLOSE.PROBLEMS and any("predefined redistribution" in q for q in u["problems"])


def test_RP11_the_update_allowance_is_enforced_inside_an_epoch_and_the_best_checkpoint_is_restored_under_it():
    g = E.generate(2, 1, 6)
    periods = [g["params"]["groups"][gg]["period"] for gg in g["params"]["latent_groups"]]
    prep = E.prepare(g["x"], g["oracle"], periods)
    P, s = prep["parts"], prep["scale"]
    Xtr, ytr = E._sx(P["train"]["X"], s), E._sy(P["train"]["y"], s)
    Xva, yva = E._sx(P["validation"]["X"], s), E._sy(P["validation"]["y"], s)
    model = E.build_modular([0] * 4 + [1] * 4, E.WINDOW, 8, fusion="sequence", seed=1)
    steps = -(-Xtr.shape[0] // E.TRAINING["batch"])
    rec = E.fit(model, Xtr, ytr, Xva, yva, training={**E.TRAINING, "max_updates": 5}, descriptors=False)
    assert rec["updates"] == 5 < steps and rec["stop_reason"] == "UPDATE_BUDGET" and rec["restore_verified"]
    rec = E.fit(model, Xtr, ytr, Xva, yva, training={**E.TRAINING, "max_updates": steps + 3}, descriptors=False)
    assert rec["updates"] == steps + 3 and rec["epochs"] == 2 and rec["stop_reason"] == "UPDATE_BUDGET" and rec["max_epochs_allowed"] == 2
    restored = float(model.evaluate(Xva, yva, verbose=0))
    assert abs(restored - min(rec["curve"]["validation"])) <= 1e-5 * max(1.0, abs(min(rec["curve"]["validation"])))   # best restored, not last
    assert rec["restore_verified"] and rec["restored_checkpoint_epoch"] == int(np.argmin(rec["curve"]["validation"])) + 1
