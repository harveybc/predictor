"""P8 (item 2 engineering_cpu_fixture + item 3 seam + load/accessor TOCTOU). No fit runs: every path
below is refused before training, or stops at a typed disposition; scratch roots only.

  a  init_matched_contrast(REAL designs, engineering_cpu_fixture=True)
  b  a FORGED design (calendar_id SYNTH_*, resealed): init accepts a CPU campaign; the trainer on the
     REAL population refuses it (design identity), and building a population from it on the real
     input refuses it (calendar identity)
  c  unit identity_sha256 with fit_device cpu vs gpu (same forged designs): does device enter identity?
  d  direct EFFECTIVE_CONFIG edit (fit_device -> cpu, resealed) after a GPU init: queue devices
  e  the REAL seam: unit_runner.run_fit -> real `train` CLI flags on the real pins with a GPU spec
     and CUDA hidden -> NO_GPU_VISIBLE (flags parse; no CPU fallback; nothing loaded or fitted)
  f  worker admission with the pinned GPU missing from the facts -> never CPU
  g  `train --device cpu` on the REAL design + REAL population (wrong digest so it stops at load):
     what does the receipt stamp for `scientific` and `device`?
  h  TOCTOU: after a verified load, the arm accessor re-reads ARM_*.npz from disk and checks it
     against the digest STRING stored in POPULATION.json, not against the verified load

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review-p8 -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p8_cpu_fixture_and_seam.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

import numpy as np  # noqa: E402

from tests.i6dmo.ops_fakes import facts, private_spec  # noqa: E402
from tools.i6dmo import campaign, common_population as CP, design as D, provenance, unit_runner, worker as W  # noqa: E402

WORK = C.SCRATCH / "p8"


def spec_file(base):
    base.mkdir(parents=True, exist_ok=True)
    spec = private_spec(base)
    for role in spec["roles"].values():
        role.update(data_root=str(C.INPUT.parent.parent), experiment_storage_root=str(C.storage_root()))
    path = base / "private_spec.json"
    path.write_text(json.dumps(spec))
    return path


def pins():
    rel = C.SEAL["private_location"].split("$EXPERIMENT_STORAGE_ROOT/", 1)[1]
    return dict(common_population=dict(base="experiment_storage", relpath=rel,
                                       sha256=C.sha(C.POPULATION / "POPULATION.json"), population_sha256=C.DIGEST),
                matched_input=dict(relpath=str(C.INPUT.relative_to(C.INPUT.parent.parent)),
                                   sha256=C.sha(C.INPUT / "INPUT.npz")))


def forged_designs(base):
    out = {}
    for arm in ("mg1", "mg2m"):
        doc = json.loads(C.DESIGNS[arm].read_text())
        doc["sealed_identities"]["calendar"]["calendar_id"] = "SYNTH_FORGED_BY_RETSU"
        doc["seal_sha256"] = D._sha(D._seal_body(doc))
        D.validate_design(doc)
        path = base / f"FORGED.{arm}.json"
        path.write_text(json.dumps(doc))
        out[arm] = path
    return out


def plan_units(root):
    return {u["unit_id"]: (u["device"], u["identity_sha256"]) for u in json.loads((root / "PLAN.json").read_text())["units"]}


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    res = {}
    # a
    try:
        campaign.init_matched_contrast(C.DESIGNS, spec_file(WORK / "a"), WORK / "a" / "root", code_commit="f" * 40,
                                       data_pins=pins(), engineering_cpu_fixture=True)
        res["a_real_designs_cpu_fixture"] = "ACCEPTED"
    except campaign.Refused as exc:
        res["a_real_designs_cpu_fixture"] = f"REFUSED:{exc}"
    # b
    forged = forged_designs(WORK)
    try:
        out = campaign.init_matched_contrast(forged, spec_file(WORK / "b"), WORK / "b" / "root", code_commit="f" * 40,
                                             data_pins=pins(), engineering_cpu_fixture=True)
        res["b_forged_calendar_init_cpu_fixture"] = "ACCEPTED"
        res["b_forged_units"] = plan_units(WORK / "b" / "root")
    except campaign.Refused as exc:
        res["b_forged_calendar_init_cpu_fixture"] = f"REFUSED:{exc}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
               I6DMO_TEST_TRAINING_OVERRIDES=json.dumps({"max_epochs": 1, "patience": 1}))
    tout = WORK / "b_train"
    proc = subprocess.run([sys.executable, "-m", "tools.i6dmo.train", "--design", str(forged["mg1"]),
                           "--common-population", str(C.POPULATION), "--common-population-digest", C.DIGEST,
                           "--input-dir", str(C.INPUT), "--out", str(tout), "--device", "cpu"],
                          cwd=C.REPO, env=env, capture_output=True, text=True, timeout=900)
    rec = json.loads((tout / "RECEIPT.json").read_text()) if (tout / "RECEIPT.json").exists() else {}
    res["b_trainer_forged_design_real_population"] = {"exit": proc.returncode, "status": rec.get("status"),
                                                      "code": rec.get("code"),
                                                      "fit_ran": bool((rec.get("training") or {}).get("epochs_completed"))}
    exps = {a: D.load_design(forged[a]) for a in ("mg1", "mg2m")}
    dds = {a: json.loads((C.MATCHED / f"DESIGN.{a}.json").read_text()) for a in ("mg1", "mg2m")}
    try:
        CP.build_common_population(C.INPUT, exps, dds)
        res["b_build_population_from_forged_on_real_input"] = "ACCEPTED"
    except CP.CommonPopulationError as exc:
        res["b_build_population_from_forged_on_real_input"] = f"REFUSED:{exc.code}"
    # c
    try:
        campaign.init_matched_contrast(forged, spec_file(WORK / "c"), WORK / "c" / "root", code_commit="f" * 40,
                                       data_pins=pins(), engineering_cpu_fixture=False)
        cpu, gpu = plan_units(WORK / "b" / "root"), plan_units(WORK / "c" / "root")
        res["c_unit_identity_equal_cpu_vs_gpu_campaign"] = {u: cpu[u][1] == gpu[u][1] for u in cpu}
        res["c_unit_device_cpu_vs_gpu_campaign"] = {u: [cpu[u][0], gpu[u][0]] for u in cpu}
        res["c_campaign_effective_sha_differs"] = json.loads((WORK / "b" / "root" / "PLAN.json").read_text())[
            "effective_config_sha256"] != json.loads((WORK / "c" / "root" / "PLAN.json").read_text())["effective_config_sha256"]
    except campaign.Refused as exc:
        res["c"] = f"REFUSED:{exc}"
    # d
    droot = WORK / "d" / "root"
    campaign.init_matched_contrast(C.DESIGNS, spec_file(WORK / "d"), droot, code_commit="f" * 40, data_pins=pins())
    eff = json.loads((droot / "EFFECTIVE_CONFIG.json").read_text())
    eff["fit_device"], eff["engineering_cpu_fixture"] = "cpu", True
    eff = campaign._seal({k: v for k, v in eff.items() if k != "seal_sha256"})
    (droot / "EFFECTIVE_CONFIG.json").write_text(json.dumps(eff))
    campaign.tick(droot, systemctl=lambda *a: dict(rc=0, stdout=""))
    q = campaign.open_queue(droot)
    try:
        res["d_effective_edit_queue_devices"] = {u["unit_id"]: u["device"] for u in q.units()}
    finally:
        q.close()
    res["d_effective_edit_load_effective"] = "LOADS" if campaign.load_effective(droot) else "REFUSED"
    # e  real seam with a GPU spec and CUDA hidden
    groot = WORK / "e" / "root"
    campaign.init_matched_contrast(C.DESIGNS, spec_file(WORK / "e"), groot, code_commit="f" * 40, data_pins=pins())
    eff = campaign.load_effective(groot)
    plan = campaign.load_plan(groot)
    unit = next(u for u in plan["units"] if u["unit_id"] == "mg1.fit")
    spec = json.loads((WORK / "e" / "private_spec.json").read_text())
    q = campaign.open_queue(groot)
    try:
        uspec = W.build_unit_spec(groot, q, plan, eff, dict(unit), spec["roles"]["preferred"],
                                  device_class="rtx5090", gpu={"uuid": "GPU-fake-preferred-0"}, cap=4 << 30,
                                  bundle=None, attempt=dict(index=1, kind="INITIAL"))
    finally:
        q.close()
    folder = WORK / "e" / "attempt"
    (folder / "out").mkdir(parents=True)
    (folder / "UNIT_SPEC.json").write_text(json.dumps(uspec))
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rc = unit_runner.run_fit(uspec, folder / "out", folder / "progress.json", poll=0.2)
    term = json.loads((folder / "out" / "TERMINAL.json").read_text()) if (folder / "out" / "TERMINAL.json").exists() else {}
    receipt = json.loads((folder / "out" / "train" / "RECEIPT.json").read_text()) \
        if (folder / "out" / "train" / "RECEIPT.json").exists() else {}
    res["e_seam_unit_runner_to_train_gpu_spec_no_gpu"] = {
        "rc": rc, "terminal_status": term.get("status"), "terminal_disposition": term.get("disposition"),
        "receipt_code": receipt.get("code"), "receipt_device": receipt.get("device"),
        "matched_argv_flags_passed_by_unit_runner": [a for a in (unit_runner._matched_data_argv(uspec)[0] or [])
                                                     if a.startswith("--")],
        "data_loaded": bool(receipt.get("matched"))}
    # f  admission without the pinned GPU in the facts
    def no_gpu(role):
        doc = facts()
        doc["gpu_free_bytes"] = {}
        doc["runtime"] = {"tensorflow": "2.18.0", "keras": "3.9.0"}
        return doc
    campaign.tick(groot, systemctl=lambda *a: dict(rc=0, stdout=""))
    r1 = W.worker_once(groot, "preferred", facts_provider=no_gpu, executor=W.LocalExecutor(), poll_interval=0.05)
    r2 = W.worker_once(groot, "coordinator", facts_provider=no_gpu, executor=W.LocalExecutor(), poll_interval=0.05)
    res["f_missing_gpu_at_claim"] = {"preferred": r1, "coordinator": r2,
                                     "attempts": len(list((groot / "attempts").glob("*"))) if (groot / "attempts").exists() else 0}
    if old is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = old
    # g
    gout = WORK / "g_train"
    env_no_overrides = {k: v for k, v in os.environ.items() if k != "I6DMO_TEST_TRAINING_OVERRIDES"}
    env_no_overrides["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([sys.executable, "-m", "tools.i6dmo.train", "--design", str(C.DESIGNS["mg1"]),
                           "--common-population", str(C.POPULATION), "--common-population-digest", "0" * 64,
                           "--input-dir", str(C.INPUT), "--out", str(gout), "--device", "cpu"],
                          cwd=C.REPO, env=env_no_overrides, capture_output=True, text=True, timeout=900)
    rec = json.loads((gout / "RECEIPT.json").read_text()) if (gout / "RECEIPT.json").exists() else {}
    res["g_cpu_device_real_design_receipt_stamp"] = {"exit": proc.returncode, "status": rec.get("status"),
                                                     "code": rec.get("code"), "scientific": rec.get("scientific"),
                                                     "device": rec.get("device")}
    # h  TOCTOU on a COPY
    hdir = WORK / "h_pop"
    shutil.copytree(C.POPULATION, hdir)
    pop = CP.load_common_population(hdir, expected_digest=C.DIGEST, input_dir=C.INPUT)
    with np.load(hdir / "ARM_mg1_FIT.npz", allow_pickle=False) as z:
        arrs = {k: z[k].copy() for k in z.files}
    arrs["x"][0, 0, 0] += 1.0
    np.savez(hdir / "ARM_mg1_FIT.npz", **arrs)
    body = json.loads((hdir / "POPULATION.json").read_text())
    body["splits"]["FIT"]["arms"]["mg1"]["arrays_sha256"] = provenance.array_sha256(arrs)
    (hdir / "POPULATION.json").write_text(json.dumps(body, indent=1, sort_keys=True))
    design = json.loads(C.DESIGNS["mg1"].read_text())
    try:
        t = CP.arm_tensors(pop, design, "FIT")
        res["h_toctou_arm_tensors_after_verified_load"] = {"accepted": True,
                                                           "x_differs_from_sealed": bool(t.x[0, 0, 0] != arrs["x"][0, 0, 0] - 1.0)}
    except CP.CommonPopulationError as exc:
        res["h_toctou_arm_tensors_after_verified_load"] = {"accepted": False, "code": exc.code}
    try:
        CP.load_common_population(hdir, expected_digest=C.DIGEST, input_dir=C.INPUT)
        res["h_fresh_verified_load_of_tampered_copy"] = "ACCEPTED"
    except CP.CommonPopulationError as exc:
        res["h_fresh_verified_load_of_tampered_copy"] = f"REFUSED:{exc.code}"
    shutil.rmtree(hdir)
    C.write("p8_cpu_fixture_and_seam.json", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
