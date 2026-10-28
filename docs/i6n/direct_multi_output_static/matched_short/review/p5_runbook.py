"""P5 (item 5 / SC10): run every SAFE command documented in RUN_MATCHED_SHORT.md, verbatim in shape,
against the REAL population and the REAL retained input, with private files in a scratch dir
outside git. Records exact argv shapes (paths redacted), exit codes and the typed results.

Not run: `deploy --no-start` (calls systemctl daemon-reload) and plain `deploy` (enables timers).
`init` and `deploy --render-to` run ONLY into a NEW scratch root under /var/tmp; the scratch spec
copy drops the `storage` estimate so the storage gate does not write its transient probe file
into the real experiment storage root (that gate is therefore NOT exercised here).

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 3600s -n i6dmo-review-p5 -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p5_runbook.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

WORK = C.SCRATCH / "p5"
PRIV = WORK / "private"
PY = sys.executable
RESULTS = []


def run(name, argv, expect_rc, env=None, timeout=1800):
    proc = subprocess.run(argv, cwd=C.REPO, capture_output=True, text=True, timeout=timeout,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", **(env or {})))
    last = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    try:
        parsed = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else None
    except ValueError:
        parsed = None
    row = {"name": name, "argv": [a if not a.startswith("/") else "<path>" for a in argv[1:]],
           "exit": proc.returncode, "expected_exit": expect_rc, "ok": proc.returncode == expect_rc,
           "result": parsed if parsed is not None else (last[-1][:400] if last else proc.stderr.strip()[-400:])}
    RESULTS.append(row)
    print(name, proc.returncode, "OK" if row["ok"] else "UNEXPECTED")
    return row


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    PRIV.mkdir(parents=True)
    os.chmod(PRIV, 0o700)
    spec = json.loads(C._SPEC.read_text())
    spec["roles"]["coordinator"]["experiment_storage_root"] = str(C.storage_root())
    spec.pop("storage", None)
    dep = PRIV / "deployment.json"
    dep.write_text(json.dumps(spec))
    os.chmod(dep, 0o600)
    inp = str(C.INPUT)
    m = [PY, "-m", "tools.i6dmo"]
    # --help of every documented command
    for sub in (["matched-short"], ["matched-short", "prepare"], ["matched-short", "validate"],
                ["matched-short", "preflight"], ["matched-short", "init"], ["matched-short", "status"],
                ["deploy"], ["status"]):
        run("help:" + " ".join(sub), m + sub + ["--help"], 0)
    run("help:common_population verify", [PY, "-m", "tools.i6dmo.common_population", "verify", "--help"], 0)
    # 1. verify the frozen population
    run("verify", [PY, "-m", "tools.i6dmo.common_population", "verify", "--population-dir", str(C.POPULATION),
                   "--expected-digest", C.DIGEST, "--input-dir", inp], 0)
    # 2. prepare (real population + real input)
    out = PRIV / "matched-short.prepared.json"
    run("prepare", m + ["matched-short", "prepare", "--deployment", str(dep), "--input-dir", inp, "--out", str(out)], 0)
    # refusals
    run("prepare --test", m + ["matched-short", "prepare", "--deployment", str(dep), "--input-dir", inp,
                               "--out", str(PRIV / "t.json"), "--test"], 2)
    seal = json.loads((C.MATCHED / "POPULATION_SEAL.json").read_text())
    seal["population_digest"] = "0" * 64
    (PRIV / "seal_wrong_digest.json").write_text(json.dumps(seal))
    run("prepare wrong digest", m + ["matched-short", "prepare", "--deployment", str(dep), "--input-dir", inp,
                                     "--out", str(PRIV / "w.json"), "--population-seal",
                                     str(PRIV / "seal_wrong_digest.json")], 2)
    run("prepare missing population mount", m + ["matched-short", "prepare", "--deployment", str(dep),
                                                 "--input-dir", inp, "--out", str(PRIV / "x.json"),
                                                 "--population-dir", "/nonexistent-retsu-mount/population"], 2)
    bad = json.loads(dep.read_text())
    bad["roles"]["coordinator"]["experiment_storage_root"] = "/nonexistent-retsu-mount"
    (PRIV / "deployment_bad_storage.json").write_text(json.dumps(bad))
    run("prepare missing storage root", m + ["matched-short", "prepare", "--deployment",
                                             str(PRIV / "deployment_bad_storage.json"), "--input-dir", inp,
                                             "--out", str(PRIV / "y.json")], 2)
    bad2 = json.loads(dep.read_text())
    bad2["roles"]["coordinator"]["data_root"] = "/nonexistent-retsu-mount/data"
    (PRIV / "deployment_bad_data.json").write_text(json.dumps(bad2))
    run("prepare missing data root", m + ["matched-short", "prepare", "--deployment",
                                          str(PRIV / "deployment_bad_data.json"), "--input-dir", inp,
                                          "--out", str(PRIV / "z.json")], 2)
    run("prepare out inside git", m + ["matched-short", "prepare", "--deployment", str(dep), "--input-dir", inp,
                                       "--out", str(C.REVIEW / "out" / "MUST_NOT_EXIST.json")], 2)
    run("prepare missing input", m + ["matched-short", "prepare", "--deployment", str(dep), "--input-dir",
                                      "/nonexistent-retsu-mount/input", "--out", str(PRIV / "i.json")], 2)
    # 3. validate
    run("validate", m + ["matched-short", "validate", "--prepared", str(out)], 0)
    # 4. preflight (read-only per-role probes; remote roles over the existing ssh trust)
    pre = run("preflight", m + ["matched-short", "preflight", "--prepared", str(out), "--out-dir",
                                str(PRIV / "precheck")], 3)
    precheck = json.loads((PRIV / "precheck" / "PRECHECK.matched.json").read_text()) \
        if (PRIV / "precheck" / "PRECHECK.matched.json").exists() else None
    # 5. init into a NEW scratch root, render units only, status at every level
    root = WORK / "root-scratch"
    run("init (scratch root)", m + ["matched-short", "init", "--prepared", str(out), "--root", str(root)], 0)
    run("init refuses existing root", m + ["matched-short", "init", "--prepared", str(out), "--root", str(root)], 2,
        timeout=600)
    run("deploy --render-to", m + ["deploy", "--root", str(root), "--deployment", str(dep), "--render-to",
                                   str(PRIV / "units-preview")], 0)
    for lvl in (["--level", "campaign"], ["--level", "arm"], ["--level", "arm", "--arm", "mg1"],
                ["--level", "unit", "--unit", "mg2m.fit"]):
        run("matched-short status " + " ".join(lvl), m + ["matched-short", "status", "--root", str(root)] + lvl, 0)
    run("status --arm mg2 (as documented in --help)", m + ["status", "--root", str(root), "--level", "arm",
                                                           "--arm", "mg2", "--json"], 0)
    run("status campaign --json", m + ["status", "--root", str(root), "--level", "campaign", "--json"], 0)
    # SC10 in the real init + render: no TEST unit anywhere
    plan = json.loads((root / "PLAN.json").read_text())
    rendered = {p.name: p.read_text() for p in (PRIV / "units-preview").glob("*")} \
        if (PRIV / "units-preview").exists() else {}
    eff = json.loads((root / "EFFECTIVE_CONFIG.json").read_text())
    sc10 = {"plan_units": [u["unit_id"] for u in plan["units"]],
            "any_unit_test_open": any(u.get("test_open") for u in plan["units"]),
            "any_unit_kind_final_test": any(u["kind"] in ("final_test", "final_refit") for u in plan["units"]),
            "test_units_enumerated": plan.get("test_units_enumerated"),
            "rendered_units": sorted(rendered),
            "rendered_mentions_test": sorted(n for n, t in rendered.items() if "final_test" in t or "--test" in t
                                             or "final-test" in t),
            "effective_test_open": eff.get("test_open"), "fit_device": eff.get("fit_device"),
            "engineering_cpu_fixture": eff.get("engineering_cpu_fixture"),
            "unit_devices": {u["unit_id"]: u["device"] for u in plan["units"]}}
    try:
        from tools.i6dmo import campaign
        campaign.enumerate_test_units(root)
        sc10["enumerate_test_units"] = "ALLOWED"
    except Exception as exc:  # noqa: BLE001
        sc10["enumerate_test_units"] = f"REFUSED:{str(exc)[:120]}"
    doc = {"commands": RESULTS, "precheck": precheck, "sc10": sc10,
           "all_expected": all(r["ok"] for r in RESULTS),
           "not_run": ["deploy --no-start (systemctl daemon-reload)", "deploy (enables timers)",
                       "storage gate (scratch spec drops the storage estimate; it writes a probe file into "
                       "the real storage root)"]}
    C.write("p5_runbook.json", doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
