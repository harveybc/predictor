"""POST for C166-C184 D2 REATTESTATION (order 2026-09-13), from the final tip and the physical roots.

Every check runs through the current public code, in new processes where the order asks for it:

* the eight PRE counterexamples, now expected CORRECTED;
* the ten acceptance tests of C181;
* physical counts of the design, tape, reserve, historical reanalysis and fresh confirmation roots on the roles;
* the C137 -> reanalysis comparison re-derived for the two families with flips;
* the unseen-reserve proof (a fresh prior-seed scan against the tape);
* fresh adjudications re-derived for two families and two units re-run from their arrays and contracts;
* the OLAP load receipts, the cube counts and the loader's uptime;
* health by role (C180) and the identities the PRE froze.

Nothing is written outside POST_DIR (a new directory under ~/.cache, because two evidence CLIs refuse ~/.local)
and nothing in the cube changes: every SQL here is a SELECT. Hosts appear by role only; home paths are redacted.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

sys.dont_write_bytecode = True

HOME = Path.home()
PRED = HOME / "Documents/GitHub/predictor"
TOOLS = PRED / "tools"
EVID = PRED / "docs/audits/evidence"
HELPERS = EVID / "repro_runs/c166_c184_tools"
PRE_OUT = EVID / "repro_runs/c166_c184_pre_2026_09_13.out"
STATE = HOME / ".local/state/crispdm-data-foundation"
ROLES_FILE = HOME / ".config/crispdm/host_roles.json"
PY = HOME / "anaconda3/envs/trading-stack/bin/python"
BASE_PY = HOME / "anaconda3/bin/python"
BASE = "e009cb14b570"
# v2: the first run (c166_c184_post, output committed unchanged) found two harness errors and one measured difference
POST_DIR = HOME / ".cache/crispdm-post/c166_c184_post_v2"
WORKER_CHECKOUT = "Documents/GitHub/.worktrees/predictor-c146"
ROLES = ("COORDINATOR", "WORKER_A", "WORKER_B")

DESIGN = STATE / "d2_design_c171_v2/D2_DESIGN_V2.json"
TAPE = STATE / "d2_fresh_tape_c173_v1/SEED_TAPE.json"
RESERVE = STATE / "d2_fresh_reserve_c173_v1"
REANALYSIS = STATE / "d2_reanalysis_c172_v2"
COMPARISON = STATE / "d2_reanalysis_c172_v2_comparison"
FRESH_SHARDS = "d2_fresh_c174_v1"
FRESH_ROOT = STATE / "d2_fresh_c174_v1_collected"
SUBMISSION = STATE / "d2_submission_c179_v1/D2_REVIEW_SUBMISSION.json"
RECEIPTS = STATE / "load_receipts_c178_v1"
FLIP_FAMILIES = ("multiband", "sinusoid")
FORBIDDEN_PREFIXES = ("predictor_plugins/", "pipeline_plugins/", "optimizer_plugins/", "preprocessor_plugins/",
                      "target_plugins/", "app/", "examples/config/", "examples/data")
DATA_GOV_OWNER_ORDERED = ("olap/lake/", "tools/governed_run.py", "tests/test_governed_run.py", "docs/GOVERNED_RUN.md",
                          "AGENTS.md")
RESULTS = []


def redact(x) -> str:
    return str(x).replace(str(HOME), "~")


def item(n, title, holds, detail):
    RESULTS.append((n, bool(holds)))
    print(f"[{n}] {'CORRECTED' if holds else 'NOT CORRECTED'}  {title}")
    print(f"     {redact(json.dumps(detail, default=str, sort_keys=True) if not isinstance(detail, str) else detail)[:2500]}")


def guarded(n, title, fn):
    try:
        holds, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a check that crashes is reported, never skipped
        holds, detail = False, f"CHECK CRASHED {type(exc).__name__}: {exc}"
    item(n, title, holds, detail)


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        h.update(b"D" if p.is_dir() else sha(p).encode())
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(PRED), *args], capture_output=True, text=True, check=True).stdout.strip()


def load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def capped(name: str, mem: str, wall: str, argv: list, cwd=PRED, timeout=7200, env_extra=None):
    """A new process under crispdm-run; returns (rc, stdout, stderr)."""
    cmd = ["crispdm-run", "-m", mem, "-t", wall, "-n", f"post-{name}", "--", "env", "-u", "PYTHONPATH",
           "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", *(env_extra or []), *map(str, argv)]
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def on(role: str, script: str, timeout=300) -> str:
    alias = json.loads(ROLES_FILE.read_text())[role]["ssh"]
    argv = ["bash", "-c", script] if alias is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                                         alias, script]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout


def pg():
    from sqlalchemy import create_engine
    e = os.environ
    return create_engine(f"postgresql://{e.get('PGUSER', 'metabase')}:{e.get('PGPASSWORD', 'metabase_pass')}@"
                         f"{e.get('PGHOST', '127.0.0.1')}:{e.get('PGPORT', '5432')}/{e.get('PGDATABASE', 'predictor_olap')}")


def select(sql: str):
    from sqlalchemy import text
    with pg().connect() as c:
        return [tuple(r) for r in c.execute(text(sql))]


# ------------------------------------------------------------------ base
def sec_base():
    head = git("rev-parse", "HEAD")
    tracked_dirty = [ln for ln in git("status", "--porcelain").splitlines() if ln and not ln.startswith("??")]
    item("BASE.tip", "the POST runs from the final tip with no tracked edits",
         not tracked_dirty, {"head": head[:12], "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                             "tracked_dirty": tracked_dirty, "order_base": BASE})


# ---------------------------------------------- C166.1-4 + acceptance 1-3
def _code_tokens(path: Path) -> list[str]:
    """Names and strings of a module, comments and docstrings excluded."""
    src = path.read_text()
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            doc_lines.update(range(node.body[0].lineno, node.body[0].end_lineno + 1))
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.NAME, tokenize.STRING) and tok.start[0] not in doc_lines:
            out.append(tok.string)
    return out


def sec_switches():
    def t1():
        # A switch needs a name to be assigned or read: identifiers containing "guard" must be 0. String literals
        # are listed, not counted: the first POST counted the probe module's file name and an evidence key as switches.
        scan = {}
        for m in ("df_snapshot", "df_operators", "df_causal_battery"):
            toks = _code_tokens(TOOLS / f"{m}.py")
            names = [t for t in toks if t.isidentifier()]
            scan[m] = {"guard_identifiers": sum(bool(re.search("guard", t, re.I)) for t in names),
                       "guard_string_literals": sorted({t for t in toks if not t.isidentifier() and re.search("guard", t, re.I)}),
                       "environment_reads": sum(t in ("environ", "getenv") for t in names),
                       "global_statements": sum(t == "global" for t in names)}
        rc, out, err = capped("switch-tests", "3G", "20m", [PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                                            "tests/test_df_operators_causality.py::test_no_switch_in_production_modules",
                                                            "tests/test_df_operators_causality.py::test_battery_has_no_switch_and_emits_no_guard_mutation",
                                                            "tests/test_df_operators_causality.py::test_no_attribute_table_or_environment_switches_a_check_off",
                                                            "tests/test_df_operators_causality.py::test_oracle_mode_does_not_lift_a_causal_check"])
        tail = (out.strip().splitlines() or [err[-200:]])[-1]
        clean = all(v["guard_identifiers"] == 0 and v["environment_reads"] == 0 and v["global_statements"] == 0
                    for v in scan.values())
        return clean and rc == 0 and "failed" not in tail, {"ast_scan": scan, "behaviour_tests": tail}
    guarded("AT1.C166.1-4", "no public switch can skip a causal check: no guard table, environment read or global flag "
            "in the productive modules; assigning tables, setting variables or passing oracle_mode lifts nothing", t1)

    def t2():
        out_dir = POST_DIR / "structural_mutations"
        rc, out, err = capped("mutations", "2G", "20m", [PY, "-B", "tools/df_structural_mutation.py", "--out", out_dir,
                                                         "--python", PY, "--child-prefix",
                                                         "crispdm-run -m 1G -t 5m -n postmut-{name} --"])
        s = json.loads((out_dir / "STRUCTURAL_MUTATION_SUMMARY.json").read_text())
        rows = [json.loads(l) for l in open(out_dir / "structural_mutations.jsonl")]
        return (s["declared_mutations"] == s["mutations"] == s["detected"] == 17 and s["all_detected"]
                and not s["evidence_process_imported_tools_modules"]
                and all(r["children_loaded_the_right_bytes"] for r in rows)), {
            "rc": rc, "detected": s["detected"], "declared": s["declared_mutations"], "not_detected": s["not_detected"],
            "guards": sorted(r["guard"] for r in rows), "wall_seconds": s["wall_seconds"]}
    guarded("AT2.C168", "each of the 17 structural mutants bites on its own, in its own process, against verified bytes", t2)

    def t3():
        full, sample = POST_DIR / "battery_full", POST_DIR / "battery_sample"
        rc_f, _, _ = capped("battery-full", "3G", "30m", [PY, "-B", "tools/df_causal_battery.py", "--out", full])
        rc_s, _, _ = capped("battery-sample", "3G", "20m", [PY, "-B", "tools/df_causal_battery.py", "--out", sample,
                                                            "--mechanics-sample", "3"])
        f = json.loads((full / "CAUSAL_BATTERY_SUMMARY.json").read_text())
        s = json.loads((sample / "CAUSAL_BATTERY_SUMMARY.json").read_text())
        rc_w, out_w, _ = capped("leak-invalidates", "3G", "20m", [PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                                                   "tests/test_df_d2_lab.py::test_wavelet_audit_invalidates_the_root_on_a_leaking_stand_in"])
        neg = f["counts_per_test_class_and_outcome"].get("NEGATIVE_CONTROL", {})
        return (rc_f == 0 and f["all_pass"] and f["battery_scope"] == "FULL" and not f["failures"]
                and neg.get("DETECTED", 0) > 0 and not neg.get("NOT_DETECTED")
                and rc_s != 0 and s["all_pass"] is False and s["battery_scope"] == "MECHANICS_SAMPLE"
                and rc_w == 0), {
            "full": {"rc": rc_f, "counts": f["counts_per_test_class_and_outcome"], "wall_seconds": f["wall_seconds"]},
            "mechanics_sample": {"rc": rc_s, "all_pass": s["all_pass"], "scope": s["battery_scope"]},
            "leaking_stand_in_invalidates_the_root": (out_w.strip().splitlines() or ["?"])[-1]}
    guarded("AT3", "an operator with a one-sample leak cannot publish evidence: the full battery detects every negative "
            "control, a sampled battery never reports PASS, and a leaking stand-in invalidates the whole D2 root", t3)


# ------------------------------------------------- C166.5-6, acceptance 4-5
def sec_identity_coverage():
    def t4():
        U, P = load("df_profile_univariate"), load("df_memory_plan")
        import numpy as np
        rows = []
        planner = P.Planner(run_id="post", bank="SYNTHETIC", dataset_id="post_ds", budget_bytes=8 << 30,
                            code_sha256="0" * 64, context={"T": 20000, "n_train": 20000, "has_ts": False,
                                                           "rg_rows": 20000, "k_pairs": 0, "V_matrix": 1},
                            sink=rows.append)
        x = np.cumsum(np.random.default_rng(3).standard_normal(20000))
        policy = dict(U.UNIT_ROOT_POLICY, exact_max_n=5000)
        U.unit_root_rows("post_ds", {"variable_id": "v"}, "train", x, gate=planner.for_module("df_profile_univariate"),
                         policy=policy)
        ests = {m: [r for r in rows if r["metric"] == m] for m in ("unit_root_adf", "unit_root_kpss")}
        ident = {m: len({json.dumps(r, sort_keys=True) for r in rs}) for m, rs in ests.items()}
        bound = all("block_identity" in (r.get("params") or {}) for rs in ests.values() for r in rs)
        cube = select("SELECT count(*) FROM public.df_fact_resource_estimate_v2")[0][0]
        return (all(len(rs) == 3 for rs in ests.values()) and all(v == 3 for v in ident.values()) and bound
                and cube == 257884), {"estimate_rows": {m: len(rs) for m, rs in ests.items()},
                                      "distinct_identities": ident, "block_identity_bound": bound,
                                      "cube_v2_rows_distinct_by_primary_key": cube}
    guarded("AT4.C166.5", "every ADF/KPSS block has a distinct identity (planner rows and the cube's v2 estimates)", t4)

    def t5():
        COV = load("df_coverage")
        code = {"REFUSED": COV.cell_state_v2(["REFUSED"]), "FAILED": COV.cell_state_v2(["FAILED"]),
                "NOT_RUN(empty)": COV.cell_state_v2([]), "row REFUSED": COV.row_state_v2("REFUSED"),
                "inapplicable": COV.applicability("mean", "TIMESTAMP", "train", {})[0]}
        states = dict(select("SELECT state, count(*) FROM public.df_fact_coverage_v2 GROUP BY state"))
        v1_refused_as_failed = select("SELECT count(*) FROM public.df_fact_coverage_v1_v2_map "
                                      "WHERE v1_state = 'FAILED' AND v2_state = 'REFUSED'")[0][0]
        ok = (code["REFUSED"] == "REFUSED" and code["FAILED"] == "FAILED" and code["NOT_RUN(empty)"] == "NOT_RUN"
              and code["row REFUSED"] == "REFUSED" and code["inapplicable"] == "NOT_APPLICABLE"
              and len({"REFUSED", "FAILED", "NOT_APPLICABLE", "NOT_RUN"} & set(COV.STATES_V2)) == 4
              and states.get("NOT_APPLICABLE", 0) > 0)
        return ok, {"code": code, "cube_v2_states": states, "v1_failed_now_refused_cells": v1_refused_as_failed}
    guarded("AT5.C166.6", "REFUSED != FAILED != NOT_APPLICABLE != NOT_RUN in code and in the cube", t5)


# ------------------------------------------------------ C166.8, acceptance 6
def sec_parity():
    def t6():
        reports = sorted((STATE / "c169_parity_reports").glob("*_*.json"))
        new = []
        for label, py in (("post_base", BASE_PY), ("post_ts", PY)):
            out = POST_DIR / f"parity_COORDINATOR_{label}.json"
            capped(f"parity-{label}", "2G", "15m", [PY, "-B", "tools/df_parity_fixture.py", "--interpreter", py,
                                                     "--role", "COORDINATOR", "--label", f"COORDINATOR:{label}",
                                                     "--out", out])
            new.append(out)
        rc, out, err = capped("parity-compare", "1G", "5m", [PY, "-B", "tools/df_parity_fixture.py", "--compare",
                                                              *[r for r in reports if r.name != "COMPARISON.json"], *new])
        cmp = json.loads(out)
        custody = json.loads((STATE / "c169_parity_reports/COMPARISON.json").read_text())
        return (cmp["parity"] and not cmp["problems"] and cmp["portable_digests"] == custody["portable_digests"]), {
            "reports": cmp["reports"], "parity": cmp["parity"], "portable_digests": cmp["portable_digests"],
            "custody_portable_digests": custody["portable_digests"], "max_abs_difference_by_metric": cmp.get("max_abs_difference_by_metric")}
    guarded("AT6.C166.8", "PCA parity holds under the predeclared tolerance and canonical representation, on the "
            "custody reports of three roles plus two new reports from both interpreters", t6)


# ------------------------------------------------------ C166.7, acceptance 7
def sec_gate():
    def t7():
        G, OPS, DZ = load("df_consumption_gate"), load("df_operators"), load("df_d2_design")
        d = next(json.loads(l) for l in open(STATE / "lab_evaluation_c137_v1/df_fact_lab_decision.jsonl")
                 if '"LAB_CALIBRATED"' in l and '"wavelet_haar_atrous"' in l)
        subject = f"{d['operator_kind']}:{json.dumps(d['operator_params'], sort_keys=True)}"
        v1 = {"schema": G.RECORD_SCHEMA, "stage": "D2", "reviewer": "external reviewer (POST fixture, in memory)",
              "reviewed_at_date": "2026-09-13", "subject_kind": "OPERATOR", "states": {subject: "LAB_CALIBRATED"},
              "regimes": {}, "grants_public_eligibility": False, "record_sha256": ""}
        v1["record_sha256"] = G.record_digest(v1)
        binding = {k: None for k in G.D2_BINDING_KEYS}
        binding.update(operator_kind=d["operator_kind"], fit_mode="EXPANDING_PREFIX", operator_code_sha256=d["code_sha256"],
                       lab_code_sha256=d["code_sha256"], root_mode="HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY",
                       spec_sha256="0" * 64, fresh_root_tape_sha256="0" * 64, design_sha256="0" * 64)
        v2 = dict(v1, schema=G.RECORD_SCHEMA_D2, d2_binding=binding, record_sha256="")
        v2["record_sha256"] = G.record_digest(v2)
        out_v1, out_v2 = G.decide(subject, "OPERATOR", [v1]), G.decide(subject, "OPERATOR", [v2])
        fresh = [json.loads(l) for l in open(FRESH_ROOT / "DECISIONS.jsonl")] if (FRESH_ROOT / "DECISIONS.jsonl").is_file() else []
        passed = [r for r in fresh if r["decision"] in ("LAB_CALIBRATED", "REGIME_LIMITED")][:3]
        rc, out, _ = capped("gate-cli", "1G", "5m", [PY, "-B", "tools/df_consumption_gate.py",
                                                     *sum([["--subject", f"OPERATOR:{r['subject']}:{json.dumps(r['operator_params'], sort_keys=True)}"]
                                                           for r in passed], [])])
        cli = json.loads(out) if out.strip() else {}
        cli_refuses = all(not x["may_be_considered_by_i5_under_review"] and "D2" in x["missing_stages"]
                          for x in cli.get("decisions", []))
        return ("D2" in out_v1["missing_stages"] and "D2" in out_v2["missing_stages"] and out_v2["states_not_consumable"]
                and d["operator_kind"] not in OPS.KINDS and cli_refuses), {
            "historical_subject": subject, "v1_record": out_v1["missing_stages"], "v2_stale_binding": out_v2["states_not_consumable"],
            "fresh_subjects_asked_of_the_gate": len(cli.get("decisions", [])), "gate_refuses_all_without_external_record": cli_refuses,
            "records_valid_in_authority": cli.get("records_valid")}
    guarded("AT7.C166.7", "no historical C137 decision passes the D2 gate, and the gate refuses fresh decisions without "
            "an external record", t7)


# ------------------------------------------- physical counts, unseen reserve
def sec_counts():
    def counts():
        DZ, ST = load("df_d2_design"), load("df_seed_tape")
        design = json.loads(DESIGN.read_text())
        tape = json.loads(TAPE.read_text())
        per_role = {}
        for role in ROLES:
            out = on(role, (f'b=$HOME/.local/state/crispdm-data-foundation/{FRESH_SHARDS}/{role}; '
                            'echo "shards=$(ls -d $b/shard_* 2>/dev/null | wc -l)"; '
                            'echo "terminals=$(ls $b/shard_*/terminals 2>/dev/null | grep -c json)"; '
                            'echo "completed=$(cat $b/shard_*/terminals/*.json 2>/dev/null | grep -c \'"status": "COMPLETED"\')"; '
                            'echo "markers=$(ls $b/shard_*/ROOT_INVALIDATED__* 2>/dev/null | wc -l)"; '
                            'echo "reserve_manifest=$(sha256sum $HOME/.local/state/crispdm-data-foundation/d2_fresh_reserve_c173_v1/ROOT_MANIFEST.json | cut -c1-64)"; '
                            # the workers hold the tape inside the reserve, which is the copy the unit worker reads;
                            # the first POST looked for the separate tape root, which only the COORDINATOR has
                            'echo "tape=$(sha256sum $HOME/.local/state/crispdm-data-foundation/d2_fresh_reserve_c173_v1/SEED_TAPE.json | cut -c1-64)"; '
                            'echo "design=$(sha256sum $HOME/.local/state/crispdm-data-foundation/d2_design_c171_v2/D2_DESIGN_V2.json | cut -c1-64)"'))
            per_role[role] = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
        fresh_units = sum(int(v["terminals"]) for v in per_role.values())
        manifest = json.loads((FRESH_ROOT / "RUN_MANIFEST.json").read_text()) if FRESH_ROOT.is_dir() else {}
        reserve_units = len(json.loads((RESERVE / "ROOT_MANIFEST.json").read_text())["units"])
        planned = sum(p["n_seeds"] for p in design["seeds_per_regime"].values())
        hist = [json.loads(p.read_text()) for p in REANALYSIS.glob("*/shard_*/terminals/*.json")]
        same = {k: len({v[k] for v in per_role.values()}) == 1 for k in ("reserve_manifest", "tape", "design")}
        ok = (reserve_units == planned == fresh_units == manifest.get("terminals") and
              manifest.get("terminals_by_status") == {"COMPLETED": planned} and not manifest.get("invalidation_markers")
              and all(v["markers"] == "0" for v in per_role.values()) and all(same.values())
              and len(hist) == 513 and all(t["status"] == "COMPLETED" for t in hist)
              and DZ.validate_design(design, require_current_code=True) == [] and ST.verify_tape(tape, design) == [])
        return ok, {"design_regimes": len(design["seeds_per_regime"]), "planned_fresh_units": planned,
                    "reserve_units": reserve_units, "fresh_terminals_by_role": {r: v["terminals"] for r, v in per_role.items()},
                    "fresh_completed_by_role": {r: v["completed"] for r, v in per_role.items()},
                    "collected_root": {k: manifest.get(k) for k in ("run_id", "terminals", "terminals_by_status", "invalidation_markers")},
                    "same_digests_on_three_roles": same, "historical_terminals": len(hist)}
    guarded("COUNTS", "physical counts: design, tape and reserve identical on the three roles; every planned fresh unit "
            "has one COMPLETED terminal; 513 historical terminals; no invalidation marker; the design and the tape verify "
            "against the current code", counts)

    def unseen():
        ST = load("df_seed_tape")
        tape = json.loads(TAPE.read_text())
        prior = ST.scan_prior_seeds(STATE)
        prior_set = set(prior["seeds"])
        tape_seeds = set()
        for reg in tape["regimes"]:
            for s in reg["seeds"]:
                tape_seeds.add(s["seed"])
                tape_seeds.update(int(v) for v in s["derived_seeds"].values())
        digest_equal = prior["report"]["prior_seeds_sha256"] == tape["prior_scan"]["prior_seeds_sha256"]
        return (bool(prior_set) and not (prior_set & tape_seeds) and digest_equal), {
            "prior_seeds_rescanned": len(prior_set), "tape_seeds_and_derived": len(tape_seeds),
            "intersection": len(prior_set & tape_seeds), "scan_digest_equals_tape": digest_equal}
    guarded("UNSEEN", "the fresh reserve is unseen: a new scan of C128-C163 seeds shares no seed or derived seed with "
            "the tape, and matches the scan the tape sealed", unseen)


# ------------------------------------------------ C137 -> reanalysis comparison
def sec_comparison():
    def cmp():
        rep = json.loads((COMPARISON / "C172_COMPARISON.json").read_text())
        work = POST_DIR / "c172_compare_work"
        work.mkdir(parents=True, exist_ok=False)
        shutil.copy2(COMPARISON / "work/CENSUS.json", work / "CENSUS.json")
        found = {}
        for fam in FLIP_FAMILIES:
            for kind in ("hdec", "hrun", "hmet", "re"):
                src = COMPARISON / f"work/{kind}__{fam}.jsonl"
                if src.is_file():
                    os.link(src, work / src.name)
            rc, out, err = capped(f"c172-{fam}", "4G", "30m", [PY, "-B", HELPERS / "c172_compare.py", "compare", "--work",
                                                              work, "--family", fam, "--design", DESIGN])
            new = json.loads((work / f"report__{fam}.json").read_text())
            old_rows = sorted((r for r in rep["rows"] if r["regime"]["family"] == fam), key=lambda r: (r["spec_sha256"], json.dumps(r["regime"], sort_keys=True)))
            new_rows = sorted(new["rows"], key=lambda r: (r["spec_sha256"], json.dumps(r["regime"], sort_keys=True)))
            found[fam] = {"rc": rc, "rows_equal": old_rows == new_rows, "flips": new["flips"],
                          "units_equal": new["units_equal"], "truth_content_equal": new["truth_content_equal"]}
        return (all(v["rows_equal"] and v["units_equal"] and v["truth_content_equal"] for v in found.values())
                and rep["flips"] == sum(r["flipped"] for r in rep["rows"])), {
            "report_sha256": sha(COMPARISON / "C172_COMPARISON.json"), "flips": rep["flips"],
            "flips_by_primary_cause": rep["flips_by_primary_cause"], "decision_counts": rep["decision_counts"],
            "status_counts_kept_apart": rep["status_counts"], "rederived_families": found}
    guarded("C172", "the C137 -> reanalysis comparison re-derives for the families with flips, units and truth equal; "
            "REFUSED, FAILED and INCONCLUSIVE counts kept apart", cmp)


# ------------------------------------------------- acceptance 8-9
def sec_fresh():
    def t8():
        A = load("df_d2_adjudicate")
        design = json.loads(DESIGN.read_text())
        fam_file = next(iter(sorted((FRESH_ROOT / "post_split").glob("df_fact_d2_unit_snr__*.jsonl")))) \
            if (FRESH_ROOT / "post_split").is_dir() else None
        refusals = {}
        sample = None
        work = FRESH_ROOT.parent / (FRESH_ROOT.name + "_work")
        snr_files = sorted(work.glob("df_fact_d2_unit_snr__*.jsonl"))
        sample = json.loads(open(snr_files[0]).readline())
        for label, mut in (("calibration row", dict(sample, partition="calibration")),
                           ("historical row", dict(sample, mode="HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY")),
                           ("time-grained row", dict(sample, t=17)),
                           ("row without tape", dict(sample, tape_sha256=None))):
            try:
                A.check_rows([mut], "snr")
                refusals[label] = "ACCEPTED"
            except A.AdjudicationRefusal as exc:
                refusals[label] = str(exc)[:120]
        try:
            A.check_rows([sample, dict(sample)], "snr")
            refusals["duplicated grain"] = "ACCEPTED"
        except A.AdjudicationRefusal as exc:
            refusals["duplicated grain"] = str(exc)[:120]
        rederived = {}
        decisions = [json.loads(l) for l in open(FRESH_ROOT / "DECISIONS.jsonl")]
        post_work = POST_DIR / "fresh_decide_work"
        post_work.mkdir(parents=True, exist_ok=False)
        shutil.copy2(work / "CENSUS.json", post_work / "CENSUS.json")
        for fam in FLIP_FAMILIES:
            for f in work.glob(f"*__{fam}.jsonl"):
                if not f.name.startswith("decisions__"):
                    os.link(f, post_work / f.name)
            rc, out, err = capped(f"decide-{fam}", "4G", "30m", [PY, "-B", HELPERS / "d2_fresh_adjudicate.py", "decide",
                                                                "--root", FRESH_ROOT, "--work", post_work, "--family", fam,
                                                                "--design", DESIGN])
            new = sorted(json.dumps(json.loads(l), sort_keys=True) for l in open(post_work / f"decisions__{fam}.jsonl"))
            old = sorted(json.dumps(r, sort_keys=True) for r in decisions if r["regime"]["family"] == fam)
            rederived[fam] = {"rc": rc, "rows": len(new), "equal": new == old}
        seed_level = all(isinstance(r["evidence"].get("per_seed_mean_abs_error_db"), dict)
                         for r in decisions if r["subject_kind"] == "SNR_ESTIMATOR" and r["decision"] != "SNR_NOT_IDENTIFIABLE")
        return (all(v != "ACCEPTED" for v in refusals.values()) and all(v["equal"] for v in rederived.values())
                and seed_level), {"refusals": refusals, "rederived_decisions": rederived,
                                  "snr_decisions_carry_per_seed_evidence": seed_level}
    guarded("AT8.C175-C176", "no fresh decision derives from calibration, historical, time-grained or duplicated rows; "
            "decisions re-derive byte-equal in new processes for two families", t8)

    def t9():
        BANK = load("df_synthetic_bank")
        SYNC = load("df_synthetic_contract")
        picked = []
        for role in ("WORKER_A", "WORKER_B"):
            t = sorted(glob.glob(str(FRESH_ROOT / "terminals" / "*.json")))
            for p in t:
                term = json.loads(Path(p).read_text())
                if term["host_role"] == role and term["status"] == "COMPLETED":
                    picked.append((role, term))
                    break
        units_root = POST_DIR / "rederive_units"
        units_root.mkdir(parents=True, exist_ok=False)
        checks = {}
        for role, term in picked:
            # dataset_id is synthetic.df_synthetic_bank.<generator version>.<unit_id>; unit ids hold no dot
            ud = RESERVE / term["dataset_id"].rsplit(".", 1)[-1]
            v = BANK.verify_unit(ud)
            (units_root / ud.name).symlink_to(ud)
            checks[ud.name] = {"role": role, "regenerates_from_seed": v["ok"],
                               "contract_matches_terminal": SYNC.unit_contract(ud)["contract_sha256"] == term["contract_sha256"],
                               "stored_output": term["output_file"]}
        out_root = POST_DIR / "rederive_root"
        rc, out, err = capped("rederive", "3G", "30m", [PY, "-B", "tools/df_d2_unit_worker.py", "--out", out_root,
                                                        "--design", DESIGN, "--units-root", units_root, "--mode",
                                                        "FRESH_CONFIRMATION", "--host-role", "COORDINATOR",
                                                        "--task-memory", str(2 << 30)])
        # A unit computed on a worker and re-run here may differ in the last ulps of floating point (C169 measured
        # up to 1.0e-13 across roles). Numbers compare under the tolerance declared in C169, never by false byte
        # equality; every other field compares exactly. Measured COST values (CPU and wall time) are not data results.
        ABS_TOL, REL_TOL = 1e-12, 1e-10

        def rows_of(path):
            out_rows = {}
            for line in open(path):
                obj = json.loads(line)
                r = dict(obj["row"])
                r.pop("run_id", None)
                cost = obj["table"] == "df_fact_d2_unit_denoising" and r.get("branch") == "COST"
                nums = {k: v for k, v in r.items() if isinstance(v, float) and not (cost and k == "value")}
                rest = {k: v for k, v in r.items() if k not in nums and not (cost and k == "value")}
                key = json.dumps({"table": obj["table"], "row": rest}, sort_keys=True)
                out_rows.setdefault(key, []).append(nums)
            return out_rows

        def compare(a, b):
            if set(a) != set(b) or any(len(a[k]) != len(b[k]) for k in a):
                return {"same_row_identities": False, "only_stored": len(set(a) - set(b)),
                        "only_rerun": len(set(b) - set(a))}
            worst, exact, over = 0.0, 0, 0
            for k in a:
                for x, y in zip(sorted(a[k], key=lambda d: json.dumps(d, sort_keys=True)),
                                sorted(b[k], key=lambda d: json.dumps(d, sort_keys=True))):
                    if set(x) != set(y):
                        over += 1
                        continue
                    for f in x:
                        d = abs(x[f] - y[f])
                        worst = max(worst, d)
                        exact += d == 0.0
                        over += d > max(ABS_TOL, REL_TOL * max(abs(x[f]), abs(y[f])))
            return {"same_row_identities": True, "numbers_exactly_equal": exact, "numbers_over_tolerance": over,
                    "max_abs_difference": worst, "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL}}

        new_terms = {}
        for p in (out_root / "terminals").glob("*.json"):
            t = json.loads(p.read_text())
            new_terms[t["contract_sha256"]] = t
        for role, term in picked:
            nt = new_terms.get(term["contract_sha256"])
            name = next(k for k, v in checks.items() if v["role"] == role)
            if nt is None or nt["status"] != "COMPLETED":
                checks[name]["rerun"] = nt["status"] if nt else "MISSING"
                continue
            checks[name]["rerun_vs_stored"] = compare(rows_of(FRESH_ROOT / term["output_file"]),
                                                      rows_of(out_root / nt["output_file"]))
        return (rc == 0 and len(picked) == 2 and all(
            c.get("regenerates_from_seed") and c.get("contract_matches_terminal")
            and c.get("rerun_vs_stored", {}).get("same_row_identities")
            and c["rerun_vs_stored"].get("numbers_over_tolerance") == 0 for c in checks.values())), checks
    guarded("AT9", "results re-derive from arrays and bound contracts: two fresh units, one from each worker, regenerate "
            "from their seeds and re-run in a new process to the same rows", t9)

    def t9b():
        # The first POST found one cross-host difference (a WORKER_B unit re-run on the COORDINATOR differs in the
        # local_level_kalman SNR estimate). This check re-runs each picked unit on the role that produced it, in a
        # new process under crispdm-run on that host, and requires every number to be exactly equal.
        probe = r'''
import json, glob, sys
S, W, unit, role = sys.argv[1:5]
stored = glob.glob(f"{S}/d2_fresh_c174_v1/{role}/shard_*/attempts/*__{unit}/attempt-1/d2_unit_rows.jsonl")[0]
rerun = glob.glob(f"{W}/root/attempts/*__{unit}/attempt-*/d2_unit_rows.jsonl")[0]
def load(p):
    out = {}
    for line in open(p):
        o = json.loads(line); r = dict(o["row"]); r.pop("run_id", None)
        cost = o["table"] == "df_fact_d2_unit_denoising" and r.get("branch") == "COST"
        nums = {k: v for k, v in r.items() if isinstance(v, float) and not (cost and k == "value")}
        rest = {k: v for k, v in r.items() if k not in nums and not (cost and k == "value")}
        out.setdefault(json.dumps({"table": o["table"], "row": rest}, sort_keys=True), []).append(nums)
    return out
a, b = load(stored), load(rerun)
exact = diff = 0
for k in a:
    for x, y in zip(a[k], b.get(k, [])):
        for f in x:
            if x[f] == y[f]: exact += 1
            else: diff += 1
print(json.dumps({"same_row_identities": set(a) == set(b), "numbers_exactly_equal": exact, "numbers_different": diff}))
'''
        out = {}
        for role in ("WORKER_A", "WORKER_B"):
            term = next(json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(FRESH_ROOT / "terminals" / "*.json")))
                        if json.loads(Path(p).read_text())["host_role"] == role
                        and json.loads(Path(p).read_text())["status"] == "COMPLETED")
            unit = term["dataset_id"].rsplit(".", 1)[-1]
            w = "$HOME/.cache/crispdm-post/c166_c184_post_v2_same_role"
            s = "$HOME/.local/state/crispdm-data-foundation"
            script = (f'set -e; W={w}/{role}; rm -rf $W; mkdir -p $W/units; ln -s {s}/d2_fresh_reserve_c173_v1/{unit} $W/units/{unit}; '
                      f'cd $HOME/{WORKER_CHECKOUT}; $HOME/.local/bin/crispdm-run -m 3G -t 20m -n post-same-role -- env -u PYTHONPATH '
                      f'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= '
                      f'$HOME/anaconda3/envs/trading-stack/bin/python -B tools/df_d2_unit_worker.py --out $W/root '
                      f'--design {s}/d2_design_c171_v2/D2_DESIGN_V2.json --units-root $W/units --mode FRESH_CONFIRMATION '
                      f'--host-role {role} --task-memory 2147483648 >/dev/null; '
                      f'echo "code=$(git log --oneline -1 | cut -c1-12) cpu=$(lscpu | grep \'Model name\' | sed \'s/.*: *//\')"; '
                      f'python3 - {s} $W {unit} {role} <<\'PY\'\n{probe}\nPY')
            text = on(role, script, timeout=1500)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            res = json.loads(lines[-1]) if lines and lines[-1].startswith("{") else {"error": text[-400:]}
            res["host"] = next((ln for ln in lines if ln.startswith("code=")), None)
            out[f"{role}:{unit}"] = res
        return all(v.get("same_row_identities") and v.get("numbers_different") == 0 for v in out.values()), out
    guarded("AT9b", "each re-derived unit re-runs on the role that produced it, in a new process, to exactly equal numbers",
            t9b)


# ------------------------------------------------------------- OLAP
def sec_olap():
    def olap():
        # two loads: D0-D2 with the v2 estimates, coverage v2 and causal evidence (df_load_d0_d2, one process), and
        # the four D2 grains streamed in chunks (c178_d2_load.py), each rehearsed on a throwaway database first
        thr = json.loads((RECEIPTS / "throwaway_d0d2_v2grains_c178_v1.json").read_text())
        real = json.loads((RECEIPTS / "real_d0d2_v2grains_c178_v1.json").read_text())
        thr2 = json.loads((RECEIPTS / "throwaway_d2_c178_v1.json").read_text())
        real2 = json.loads((RECEIPTS / "real_d2_c178_v1.json").read_text())
        d2 = ("df_fact_d2_unit_denoising", "df_fact_d2_unit_snr", "df_fact_d2_decision", "df_fact_d2_historical_reanalysis")
        v2 = ("df_fact_resource_estimate_v2", "df_fact_coverage_v2", "df_fact_coverage_v1_v2_map")
        cube = {t: select(f"SELECT count(*) FROM public.{t}")[0][0] for t in d2 + v2}
        by_mode = dict(select("SELECT mode, count(*) FROM public.df_fact_d2_unit_denoising GROUP BY mode"))
        refused = {t: v["rows_refused"] for t, v in real["load"].items() if v["rows_refused"]}
        silent = {t: v for t, v in real["load"].items() if v["rows_offered"] != v["rows_inserted"] + v["rows_already_present"] + v["rows_refused"]}
        loader = on("COORDINATOR", "systemctl --user show crispdm-olap-loader.service -p ActiveState -p NRestarts -p ActiveEnterTimestamp")
        kv = dict(ln.split("=", 1) for ln in loader.splitlines() if "=" in ln)
        d2_distinct = real2["distinct_offered"]
        ok = (thr.get("idempotent") and thr.get("throwaway_database_dropped")
              and all(v["rows_refused"] == 0 for v in thr["first_load"].values())
              and real.get("historical_unchanged") and not refused and not silent
              and thr2.get("idempotent") and thr2.get("throwaway_database_dropped") and thr2["no_silent_dedup"]
              and real2.get("historical_unchanged") and real2["no_silent_dedup"] and not any(real2["rows_refused"].values())
              and all(cube[t] == d2_distinct.get(t, 0) for t in d2)
              and set(by_mode) == {"FRESH_CONFIRMATION", "HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY"}
              and kv.get("ActiveState") == "active" and kv.get("NRestarts") == "0")
        return ok, {
            "d0d2_throwaway": {"idempotent": thr.get("idempotent"), "dropped": thr.get("throwaway_database_dropped")},
            "d0d2_real": {"historical_unchanged": real.get("historical_unchanged"), "refused": refused,
                          "offered_not_accounted": silent},
            "d2_throwaway": {k: thr2.get(k) for k in ("idempotent", "throwaway_database_dropped", "no_silent_dedup")},
            "d2_real": {k: real2.get(k) for k in ("historical_unchanged", "no_silent_dedup", "rows_refused",
                                                  "changed_tables_outside_d2")},
            "cube_counts": cube, "denoising_rows_by_mode": by_mode, "loader": kv}
    guarded("C178", "throwaway loads twice (second inserts nothing, database dropped), real loads additive with history "
            "unchanged and zero rows refused or silently deduplicated; historical reanalysis and fresh confirmation "
            "separate by mode and run; the loader never restarted", olap)

    def submission():
        s = json.loads(SUBMISSION.read_text())
        G = load("df_consumption_gate")
        good, bad = G.load_records()
        return (s["kind"] == "SUBMISSION_FOR_EXTERNAL_REVIEW" and s["authority"].startswith("NONE")
                and not any("D2" == r.get("stage") for r in good)), {
            "submission_sha256": sha(SUBMISSION), "design_sha256": s["design"]["design_sha256"],
            "tape_sha256": s["seed_tape"]["tape_sha256"], "fresh_run_id": s["fresh_confirmation"]["run_id"],
            "decisions_sha256": s["fresh_confirmation"]["decisions_sha256"], "external_d2_records": 0}
    guarded("C179", "the submission binds design, code, tape, roots, decisions and load, grants nothing, and no D2 "
            "record exists in the reviewer authority", submission)


# ------------------------------------------------ acceptance 10, C180
def sec_scope_health():
    def t10():
        changed = git("diff", "--name-only", f"{BASE}..HEAD").splitlines()
        forbidden = [f for f in changed if f.startswith(FORBIDDEN_PREFIXES)]
        data_gov = [f for f in changed if f.startswith(DATA_GOV_OWNER_ORDERED)]
        d345 = [f for f in changed if re.search(r"D[345]_.*DESIGN", f)]
        launches = []
        for root in ("d2_dispatch_c172_v2", "d2_dispatch_c174_coordinator", "d2_dispatch_c174_workers", "d2_dispatch_c174_workers_v2"):
            for p in (STATE / root / "receipts").glob("*.launch.json"):
                launches.append(json.loads(p.read_text())["command"])
        cuda_blank = all("CUDA_VISIBLE_DEVICES=" in c and "CUDA_VISIBLE_DEVICES=GPU" not in c for c in launches)
        gpu = {r: on(r, "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do "
                        "grep -l crispdm /proc/$p/cgroup 2>/dev/null; done | wc -l").strip() for r in ROLES}
        return (not forbidden and not d345 and cuda_blank and all(v == "0" for v in gpu.values())), {
            "files_changed_since_order": len(changed), "forbidden_paths": forbidden, "d3_d5_design_files": d345,
            "owner_ordered_data_gov_files_declared_separately": data_gov,
            "dispatch_launches": len(launches), "every_launch_hides_the_gpus": cuda_blank,
            "gpu_compute_processes_in_crispdm_cgroups": gpu}
    guarded("AT10", "no GPU, D3, selector, model, RL, DOIN or live path touched by the D2 work", t10)

    def health():
        probe = ("echo memguard=$(systemctl --user is-active crispdm-memguard.service); "
                 "echo slice_max=$(systemctl --user show crispdm-batch.slice -p MemoryMax --value); "
                 "echo mem_available_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); "
                 "echo failed_units=$(systemctl --user list-units --state=failed --no-legend --plain | wc -l); "
                 "echo crispdm_units_left=$(systemctl --user list-units --no-legend --plain --all 'crispdm-dispatch-*' | wc -l); "
                 "echo gpu_errors=$(nvidia-smi -L 2>&1 | grep -ci 'unable\\|error'); "
                 "echo gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); "
                 "echo kernel_gpu_errors_24h=$(journalctl -k --since '-24h' --no-pager 2>/dev/null | grep -ci 'NVRM.*Xid\\|GPU has fallen off\\|RmInitAdapter failed')")
        out = {r: dict(ln.split("=", 1) for ln in on(r, probe).splitlines() if "=" in ln) for r in ROLES}
        quarantined = out["WORKER_B"].get("gpu_errors", "0") != "0"
        return (all(v.get("memguard") == "active" for v in out.values()) and quarantined), {
            "roles": out, "WORKER_B_GPU1": "QUARANTINED_NOT_SCHEDULABLE" if quarantined else "RESPONDING",
            "logrotate": "owner task, not blocking D2"}
    guarded("C180", "health by role: memory guard active, batch slice capped, dispatch units released, WORKER_B GPU1 "
            "quarantined while it has no handle", health)


# ---------------------------------------------------------- preservation
def sec_preservation():
    frozen = {}
    for ln in PRE_OUT.read_text().splitlines():
        m = re.match(r"\s+(state|evidence|order):(\S+): ([0-9a-f]{16}) unchanged=True", ln)
        if m:
            frozen[f"{m.group(1)}:{m.group(2)}"] = m.group(3)
    now = {}
    for k in frozen:
        kind, name = k.split(":", 1)
        if kind == "state":
            now[k] = tree_digest(STATE / name)[:16] if (STATE / name).exists() else None
        elif kind == "evidence":
            now[k] = sha(EVID / name)[:16]
        else:
            p = next(iter(list((PRED / "docs/audits").glob(name)) + list((PRED / "docs/handoffs").glob(name))), None)
            now[k] = sha(p)[:16] if p else None
    changed = {k: (frozen[k], now[k]) for k in frozen if frozen[k] != now[k]}
    item("PRESERVED", "every root and document the PRE froze is byte-identical", not changed,
         {"frozen": len(frozen), "changed": changed})


def main() -> int:
    if POST_DIR.exists():
        raise SystemExit(f"REFUSED: {redact(POST_DIR)} exists; the POST writes into a new directory")
    POST_DIR.mkdir(parents=True)
    print(f"POST C166-C184 python={sys.version.split()[0]}")
    sec_base()
    sec_switches()
    sec_identity_coverage()
    sec_parity()
    sec_gate()
    sec_counts()
    sec_comparison()
    sec_fresh()
    sec_olap()
    sec_scope_health()
    sec_preservation()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n=== POST SUMMARY: {len(RESULTS)} checks, {len(RESULTS) - len(bad)} corrected, not corrected {bad} ===")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
