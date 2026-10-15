"""Bounded re-check at 796ec8f4 of review defects D1/D2/D3/D5/D6 (scratch roots only; no fit runs).

Every worker round below either adopts, refuses admission before launch, or launches a child that
refuses BEFORE loading data (CUDA hidden + a fake GPU UUID => NO_GPU_VISIBLE). systemctl injected.

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 2400s -n i6dmo-review-recheck -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/r_recheck_796ec8f4.py
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import p6_orchestration as P  # noqa: E402

import numpy as np  # noqa: E402

from tests.i6dmo.ops_fakes import facts  # noqa: E402
from tools.i6dmo import (campaign, common_population as CP, design as D, matched_runner as MR,  # noqa: E402
                         provenance, reporting, status, worker as W)

WORK = C.SCRATCH / "recheck"
PIN_RT = dict(D.MATCHED_RUNTIME)
FAIL_CHILD = C.REVIEW / "r_fail_child.py"
NOOP = lambda *a: dict(rc=0, stdout="")  # noqa: E731


def provider(runtime):
    def probe(role):
        doc = facts()
        doc["runtime"] = dict(runtime)
        return doc
    return probe


def wk(root, role, *, runtime=PIN_RT, env=None):
    return W.worker_once(root, role, facts_provider=provider(runtime), executor=W.LocalExecutor(),
                         poll_interval=0.05, extra_env=dict(env or {}))


def states(root):
    return P.states(root)


def events(root, kind):
    q = campaign.open_queue(root)
    try:
        return [dict(unit=e["unit_id"], detail=e["detail"]) for e in q.events(100000) if e["event"] == kind]
    finally:
        q.close()


def sealed():
    return {a: {"seal_sha256": json.loads(C.DESIGNS[a].read_text())["seal_sha256"]} for a in ("mg1", "mg2m")}


def reseal_effective(root, **changes):
    eff = json.loads((root / "EFFECTIVE_CONFIG.json").read_text())
    eff.update(changes)
    eff = campaign._seal({k: v for k, v in eff.items() if k != "seal_sha256"})
    (root / "EFFECTIVE_CONFIG.json").write_text(json.dumps(eff))


# ------------------------------------------------------------------------- D1
def d1(base):
    res = {}
    counter = base / "counter.txt"
    a_root, _ = P.init(base, "A_fake", entries=P.fake_entries())
    for _ in range(12):  # drive A with the CURRENT sealed runtime in the facts (fake children only)
        campaign.tick(a_root, systemctl=NOOP)
        for role in ("preferred", "secondary", "coordinator"):
            wk(a_root, role, env={"I6DMO_FAKE_COUNTER": str(counter)})
    campaign.tick(a_root, systemctl=NOOP)
    res["A_states"] = {u: s for u, (s, _) in states(a_root).items()}
    donors = {"fake_entrypoints": a_root}
    # forged real-procedure-shaped donors whose terminals are A's (same identities), without receipts
    for name, change in (("no_receipt", {}), ("cpu_donor", {"fit_device": "cpu"}),
                         ("engineering_donor", {"engineering_cpu_fixture": True}),
                         ("override_receipt", {}), ("forged_scientific_receipt_not_in_artifacts", {})):
        root, _ = P.init(base, f"D_{name}", entries=None)
        shutil.copytree(a_root / "terminals", root / "terminals", dirs_exist_ok=True)
        if change:
            reseal_effective(root, **change)
        if name in ("override_receipt", "forged_scientific_receipt_not_in_artifacts"):
            for t in (root / "terminals").glob("*/TERMINAL.json"):
                if json.loads(t.read_text())["unit_id"].endswith(".fit"):
                    (t.parent / "out" / "train").mkdir(parents=True, exist_ok=True)
                    rec = ({"scientific": False, "device": "gpu", "overrides": {"max_epochs": 2}}
                           if name == "override_receipt" else {"scientific": True, "device": "gpu", "overrides": {}})
                    (t.parent / "out" / "train" / "RECEIPT.json").write_text(json.dumps(rec))
        donors[name] = root
    res["donors"] = {}
    for name, droot in donors.items():
        row = {}
        try:
            MR._check_donor(droot, sealed(), P.real_pins())
            row["prepare_check_donor"] = "ACCEPTED"
        except MR.PrepareRefused as exc:
            row["prepare_check_donor"] = f"REFUSED:{str(exc).split(' ', 1)[-1][:160]}"
        b_root, _ = P.init(base, f"B_prior_{name}", entries=None, prior=[droot])
        campaign.tick(b_root, systemctl=NOOP)
        # runtime mismatch in the facts => a refused adoption can never launch anything
        log = [wk(b_root, r, runtime={"tensorflow": "0", "keras": "0"}).get("result")
               for r in ("preferred", "secondary", "coordinator")]
        row["worker_results"] = log
        row["adopted"] = sorted(u for u, (s, c) in states(b_root).items() if c == "ADOPTED")
        row["prior_rejected_reasons"] = sorted({e["detail"].get("reason", "")[:90] for e in events(b_root, "PRIOR_REJECTED")})
        row["attempt_dirs"] = len(list((b_root / "attempts").glob("*"))) if (b_root / "attempts").exists() else 0
        res["donors"][name] = row
    return res


# ------------------------------------------------------------------------- D2
def d2(base):
    res = {}
    out = base / "g_train"
    env = {k: v for k, v in os.environ.items() if not k.startswith("I6DMO_TEST_")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([sys.executable, "-m", "tools.i6dmo.train", "--design", str(C.DESIGNS["mg1"]),
                           "--common-population", str(C.POPULATION), "--common-population-digest", "0" * 64,
                           "--input-dir", str(C.INPUT), "--out", str(out), "--device", "cpu"],
                          cwd=C.REPO, env=env, capture_output=True, text=True, timeout=900)
    rec = json.loads((out / "RECEIPT.json").read_text()) if (out / "RECEIPT.json").exists() else {}
    res["p8g_cpu_receipt"] = {"exit": proc.returncode, "code": rec.get("code"), "scientific": rec.get("scientific"),
                              "device": rec.get("device")}
    # I6DMO_TEST_* in role environment / injected env / process env => admission refused, nothing launched
    for where in ("role_environment", "extra_env", "process_env"):
        root, _ = P.init(base, f"override_{where}", entries=None)
        if where == "role_environment":
            spec = json.loads((root / "private" / "deployment.json").read_text())
            spec["roles"]["preferred"].setdefault("environment", {})["I6DMO_TEST_TRAINING_OVERRIDES"] = '{"max_epochs": 2}'
            (root / "private" / "deployment.json").write_text(json.dumps(spec))
        campaign.tick(root, systemctl=NOOP)
        extra = {"I6DMO_TEST_TRAINING_OVERRIDES": '{"max_epochs": 2}'} if where == "extra_env" else None
        if where == "process_env":
            os.environ["I6DMO_TEST_MAX_ROWS"] = "10"
        try:
            r = wk(root, "preferred", env=extra)
        finally:
            os.environ.pop("I6DMO_TEST_MAX_ROWS", None)
        res[f"override_in_{where}"] = {"result": r.get("result"), "cause": r.get("cause"),
                                       "attempt_dirs": len(list((root / "attempts").glob("*")))
                                       if (root / "attempts").exists() else 0}
    # report + decision: campaign path strict; direct calls default lenient (ruling item)
    import p4_report_real as R4
    R4.WORK = base / "p4"
    R4.WORK.mkdir(parents=True, exist_ok=True)
    good = {}
    for arm in ("mg1", "mg2m"):
        up = R4.write_upstream(R4.WORK / f"up_{arm}", arm,
                               receipt_patch=lambda r: r.update(scientific=True, device="gpu"))
        term, good[arm] = R4.report(arm, up)
        res[f"report_scientific_gpu_{arm}"] = term["status"]
    bad_up = R4.write_upstream(R4.WORK / "up_bad", "mg1",
                               receipt_patch=lambda r: r.update(scientific=False, device="cpu"))

    def report_with(require):
        design = json.loads(C.DESIGNS["mg1"].read_text())
        out = R4.WORK / f"report_bad_require_{require}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        spec = {"kind": "report", "family": "short", "unit_id": "mg1.report", "campaign_id": "retsu",
                "identity_sha256": "0" * 64, "effective_config": {"design": design, "code_commit": "probe"},
                "matched": {"arm": "mg1", "fit_unit": "mg1.fit", "require_scientific_fit": require,
                            "input": {"path": str(C.INPUT), "sha256": C.sha(C.INPUT / "INPUT.npz")}},
                "common_population": {"path": str(C.POPULATION), "sha256": C.DIGEST},
                "upstream": {"mg1.fit": bad_up}, "pins": {}}
        term = reporting.run_unit(spec, out, out / "P.json")
        met = json.loads((out / "METRICS.json").read_text()) if (out / "METRICS.json").exists() else None
        return term, met
    t_strict, _ = report_with(True)
    t_lenient, met_bad = report_with(False)
    res["report_nonscientific_receipt_require_true"] = t_strict["disposition"]
    res["report_nonscientific_receipt_require_false_default_direct"] = t_lenient["disposition"]
    for require in (True, False):
        try:
            v = reporting.decide_matched(met_bad, good["mg2m"], require_scientific=require)
            res[f"decide_nonscientific_require_{require}"] = v["decision"]
        except reporting.ReportFailure as exc:
            res[f"decide_nonscientific_require_{require}"] = f"REFUSED:{exc.code}"
    try:
        v = reporting.decide_matched(good["mg1"], good["mg2m"], require_scientific=True)
        res["decide_scientific_require_True"] = v["decision"]
    except reporting.ReportFailure as exc:
        res["decide_scientific_require_True"] = f"REFUSED:{exc.code}"
    return res


# ------------------------------------------------------------------------- D3
def d3(base):
    hdir = base / "h_pop"
    shutil.copytree(C.POPULATION, hdir)
    pop = CP.load_common_population(hdir, expected_digest=C.DIGEST, input_dir=C.INPUT)
    with np.load(hdir / "ARM_mg1_FIT.npz", allow_pickle=False) as z:
        arrs = {k: z[k].copy() for k in z.files}
    arrs["x"][0, 0, 0] += 1.0
    np.savez(hdir / "ARM_mg1_FIT.npz", **arrs)
    body = json.loads((hdir / "POPULATION.json").read_text())
    body["splits"]["FIT"]["arms"]["mg1"]["arrays_sha256"] = provenance.array_sha256(arrs)
    (hdir / "POPULATION.json").write_text(json.dumps(body, indent=1, sort_keys=True))
    out = {}
    try:
        CP.arm_tensors(pop, json.loads(C.DESIGNS["mg1"].read_text()), "FIT")
        out["toctou_after_verified_load"] = "ACCEPTED"
    except CP.CommonPopulationError as exc:
        out["toctou_after_verified_load"] = f"REFUSED:{exc.code}"
    untampered = CP.arm_tensors(pop, json.loads(C.DESIGNS["mg2m"].read_text()), "FIT")
    out["untampered_arm_still_served"] = int(len(untampered.row_id))
    shutil.rmtree(hdir)
    return out


# ------------------------------------------------------------------------- D5
def d5(base):
    res = {}
    # (i) remote pre-fit refusals: preferred holds a WRONG population replica; secondary has CUDA hidden
    root, _ = P.init(base, "infra", entries=None)
    bad_store = base / "bad_store"
    rel = C.SEAL["private_location"].split("$EXPERIMENT_STORAGE_ROOT/", 1)[1]
    (bad_store / rel).mkdir(parents=True)
    (bad_store / rel / "POPULATION.json").write_text("{}")
    spec = json.loads((root / "private" / "deployment.json").read_text())
    spec["roles"]["preferred"].update(ssh_alias="retsu-fake-alias", experiment_storage_root=str(bad_store))
    spec["roles"]["secondary"].update(ssh_alias="retsu-fake-alias-2")
    (root / "private" / "deployment.json").write_text(json.dumps(spec))
    campaign.tick(root, systemctl=NOOP)
    log = []
    for _ in range(3):
        for role in ("preferred", "secondary", "coordinator"):
            r = wk(root, role)
            log.append([role, r.get("result"), r.get("unit_id"), (r.get("cause") or "")[:80]])
        campaign.tick(root, systemctl=NOOP)
    res["infra_worker_log"] = log
    res["infra_states"] = states(root)
    q = campaign.open_queue(root)
    try:
        res["infra_attempts"] = {u: [dict(idx=a["idx"], outcome=a["outcome"], cause=(a["cause"] or "")[:70])
                                     for a in q.attempts(u)] for u in ("mg1.fit", "mg2m.fit")}
        res["comparison_terminal"] = q.terminal("comparison") is not None
    finally:
        q.close()
    res["closed"] = (root / "CLOSE.json").exists()
    late = status.compute(root, level="campaign", now=time.time() + 1801)
    res["status_after_1801s"] = {"state": late.get("state"),
                                 "failures": sorted({f["kind"] for f in late.get("failures") or []})}
    # (ii) post-training failures stay terminal: a child that trained then failed
    for name, code, fit_s in (("nonfinite_after_training", "NONFINITE_LOSS", 12.5),
                              ("infra_code_but_training_started", "NO_GPU_VISIBLE", 12.5),
                              ("infra_code_before_training", "NO_GPU_VISIBLE", None)):
        entries = dict(campaign.MATCHED_ENTRYPOINTS, fit=[sys.executable, str(FAIL_CHILD)],
                       report=[sys.executable, str(P.FAKE), "--kind", "report"])
        r2, _ = P.init(base, f"post_{name}", entries=entries)
        campaign.tick(r2, systemctl=NOOP)
        out = wk(r2, "preferred", env={"RETSU_FAIL_CODE": code, "RETSU_FIT_SECONDS": "" if fit_s is None else str(fit_s)})
        res[f"child_{name}"] = {"result": out.get("result"), "cause": (out.get("cause") or "")[:80],
                                "unit_state": states(r2).get(out.get("unit_id"), [None])[0]}
    return res


# ------------------------------------------------------------------------- D6
def d6():
    out = {}
    for argv in (["status", "--help"], ["matched-short", "status", "--help"]):
        p = subprocess.run([sys.executable, "-m", "tools.i6dmo", *argv], cwd=C.REPO, capture_output=True, text=True)
        lines = [" ".join(ln.split()) for ln in p.stdout.splitlines() if "mg1" in ln]
        out[" ".join(argv)] = {"exit": p.returncode, "arm_help_lines": lines}
    return out


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    res = {"pin": "796ec8f4", "population_digest": C.DIGEST,
           "D1": d1(WORK / "d1"), "D2": d2(WORK / "d2"), "D3": d3(WORK / "d3"), "D5": d5(WORK / "d5"), "D6": d6()}
    res["D1"]["A_fake_children_invoked"] = P.counter_lines(WORK / "d1" / "counter.txt")
    C.write("recheck_796ec8f4.json", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
