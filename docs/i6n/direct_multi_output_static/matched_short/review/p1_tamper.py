"""P1 (SC01/SC06): tamper COPIES of the real frozen population; every tamper must be refused at
load (common_population.load_common_population), at the trainer entry (the real
``python -m tools.i6dmo.train`` CLI with the matched flags) and at the report entry
(``reporting.run_unit`` -> ``run_matched_unit``). Nothing is fitted: a tamper that were accepted
would reach a 1-epoch override (scientific=false) and is reported as a defect.

Run (from the review worktree):
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p1_tamper.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

from tools.i6dmo import common_population as CP, provenance  # noqa: E402

WORK = C.SCRATCH / "p1"


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _save(path, arrays):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _pj(root):
    return json.loads((root / "POPULATION.json").read_text())


def _wpj(root, body):
    (root / "POPULATION.json").write_text(json.dumps(body, indent=1, sort_keys=True, allow_nan=False))


def _reseal_body(root, body):
    """A sophisticated tamper: recompute the self-declared body digest + file digests."""
    body = dict(body)
    body.pop("digest", None)
    files = body.pop("files", None) or {}
    digest = CP._population_digest(body)
    body["digest"] = digest
    for name in files:
        files[name] = provenance.file_sha256(root / name)
    body["files"] = files
    _wpj(root, body)
    files["POPULATION.json"] = None
    return digest


def t_row(root):
    a = _npz(root / "POPULATION_FIT.npz")
    a["origin_row_ids"][-1] += 3600
    _save(root / "POPULATION_FIT.npz", a)


def t_row_resealed(root):
    a = _npz(root / "POPULATION_FIT.npz")
    a["origin_row_ids"][-1] += 3600
    _save(root / "POPULATION_FIT.npz", a)
    body = _pj(root)
    info = body["splits"]["FIT"]
    counts = {int(k): v for k, v in info["counts"].items()}
    info["row_ids_sha256"] = provenance.array_sha256({"origin_row_ids": a["origin_row_ids"]})
    info["arrays_sha256"] = provenance.array_sha256(a)
    info["digest"] = CP._split_digest("FIT", a, counts)
    _reseal_body(root, body)


def t_head(root):
    a = _npz(root / "POPULATION_DEVELOPMENT.npz")
    row = int(np.flatnonzero(a["head_mask"].sum(axis=1) >= 2)[0])
    col = int(np.flatnonzero(a["head_mask"][row])[0])
    a["head_mask"][row, col] = False
    a["entry_mask"][row, col] = False
    a["management_mask"][row, col] = False
    _save(root / "POPULATION_DEVELOPMENT.npz", a)


def t_calendar_identity(root):
    body = _pj(root)
    body["identities"]["calendar"]["version"] = body["identities"]["calendar"]["version"] + "-x"
    _wpj(root, body)


def t_scaler_mean(root):
    body = _pj(root)
    body["scaler"]["mean"][0] = body["scaler"]["mean"][0] * (1 + 1e-7)
    _wpj(root, body)


def t_scaler_mean_resealed(root):
    body = _pj(root)
    body["scaler"]["mean"][0] = body["scaler"]["mean"][0] * (1 + 1e-7)
    _reseal_body(root, body)


def t_scaler_rows(root):
    a = _npz(root / "SCALER_ROWS.npz")
    a["fit_source_bar_ts"] = a["fit_source_bar_ts"][:-1]
    _save(root / "SCALER_ROWS.npz", a)


def t_naive_scale(root):
    body = _pj(root)
    body["naive_loss_scale"]["2"] = body["naive_loss_scale"]["2"] * 1.01
    _wpj(root, body)


def t_arm_x(root):
    a = _npz(root / "ARM_mg1_FIT.npz")
    a["x"][0, 0, 0] += 1e-6
    _save(root / "ARM_mg1_FIT.npz", a)


def t_arm_target(root):
    a = _npz(root / "ARM_mg2m_DEVELOPMENT.npz")
    r, c = np.argwhere(a["head_mask"])[0]
    a["target_price"][r, c] += 1e-5
    _save(root / "ARM_mg2m_DEVELOPMENT.npz", a)


def t_none(root):
    pass


CASES = [  # name, tamper, arm for trainer/report entries, input-dir tamper, expected digest override
    ("row", t_row, "mg1", False, None),
    ("row_resealed_body", t_row_resealed, "mg2m", False, None),
    ("head_mask", t_head, "mg1", False, None),
    ("calendar_identity", t_calendar_identity, "mg2m", False, None),
    ("calendar_input_file", t_none, "mg1", True, None),
    ("scaler_mean", t_scaler_mean, "mg2m", False, None),
    ("scaler_mean_resealed_body", t_scaler_mean_resealed, "mg1", False, None),
    ("scaler_rows", t_scaler_rows, "mg2m", False, None),
    ("naive_loss_scale", t_naive_scale, "mg1", False, None),
    ("arm_tensor_x_mg1_fit", t_arm_x, "mg1", False, None),
    ("arm_tensor_target_mg2m_dev", t_arm_target, "mg2m", False, None),
    ("wrong_expected_digest", t_none, "mg1", False, "0" * 64),
]


def _tampered_input():
    dst = WORK / "input_calendar_tampered"
    if not dst.exists():
        shutil.copytree(C.INPUT, dst)
        cal = json.loads((dst / "CALENDAR.json").read_text())
        cal["retsu_tamper"] = True
        (dst / "CALENDAR.json").write_text(json.dumps(cal))
    return dst


def load_entry(pop_dir, digest, input_dir):
    try:
        CP.load_common_population(pop_dir, expected_digest=digest, input_dir=input_dir)
        return {"refused": False, "code": "ACCEPTED"}
    except Exception as exc:  # noqa: BLE001
        return {"refused": True, "code": getattr(exc, "code", type(exc).__name__)}


def trainer_entry(pop_dir, digest, input_dir, arm, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               I6DMO_TEST_TRAINING_OVERRIDES=json.dumps({"max_epochs": 1, "patience": 1}))
    argv = [sys.executable, "-m", "tools.i6dmo.train", "--design", str(C.DESIGNS[arm]),
            "--common-population", str(pop_dir), "--common-population-digest", digest,
            "--input-dir", str(input_dir), "--out", str(out), "--device", "cpu"]
    t0 = time.time()
    proc = subprocess.run(argv, cwd=C.REPO, env=env, capture_output=True, text=True, timeout=900)
    receipt = json.loads((out / "RECEIPT.json").read_text()) if (out / "RECEIPT.json").exists() else {}
    return {"exit": proc.returncode, "status": receipt.get("status"), "code": receipt.get("code"),
            "fit_ran": bool((receipt.get("training") or {}).get("epochs_completed")),
            "refused": proc.returncode != 0 and receipt.get("status") != "COMPLETED",
            "seconds": round(time.time() - t0, 1), "argv_flags": [a for a in argv if a.startswith("--")]}


def report_entry(pop_dir, digest, input_dir, arm, out):
    from tools.i6dmo import reporting
    design = json.loads(C.DESIGNS[arm].read_text())
    spec = {"kind": "report", "family": "short", "unit_id": f"{arm}.report", "campaign_id": "retsu-probe",
            "identity_sha256": "0" * 64, "effective_config": {"design": design},
            "matched": {"arm": arm, "fit_unit": f"{arm}.fit",
                        "input": {"path": str(input_dir), "sha256": C.sha(Path(input_dir) / "INPUT.npz")}},
            "common_population": {"path": str(pop_dir), "sha256": digest}, "upstream": {}, "pins": {}}
    out.mkdir(parents=True, exist_ok=True)
    term = reporting.run_unit(spec, out, out / "PROGRESS.json")
    return {"status": term.get("status"), "code": term.get("disposition"),
            "refused": term.get("status") != "COMPLETE"}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    results = {"population_digest_expected": C.DIGEST, "cases": {}}
    # positive control: the untampered REAL population verifies at load
    results["control_untampered_load"] = load_entry(C.POPULATION, C.DIGEST, C.INPUT)
    for name, tamper, arm, bad_input, digest_override in CASES:
        dst = WORK / f"pop_{name}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(C.POPULATION, dst)
        tamper(dst)
        input_dir = _tampered_input() if bad_input else C.INPUT
        digest = digest_override or C.DIGEST
        tout = WORK / f"train_{name}"
        if tout.exists():
            shutil.rmtree(tout)
        rout = WORK / f"report_{name}"
        if rout.exists():
            shutil.rmtree(rout)
        row = {"arm": arm, "load": load_entry(dst, digest, input_dir),
               "trainer": trainer_entry(dst, digest, input_dir, arm, tout),
               "report": report_entry(dst, digest, input_dir, arm, rout)}
        row["all_refused"] = row["load"]["refused"] and row["trainer"]["refused"] and row["report"]["refused"] \
            and not row["trainer"]["fit_ran"]
        results["cases"][name] = row
        print(name, json.dumps({k: (v.get("code") if isinstance(v, dict) else v) for k, v in row.items()}))
        shutil.rmtree(dst)
    results["verdict"] = "ALL_REFUSED" if all(r["all_refused"] for r in results["cases"].values()) and \
        not results["control_untampered_load"]["refused"] else "DEFECT"
    C.write("p1_tamper.json", results)
    return 0 if results["verdict"] == "ALL_REFUSED" else 1


if __name__ == "__main__":
    sys.exit(main())
