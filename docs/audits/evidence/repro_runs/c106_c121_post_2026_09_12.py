"""POST for C106-C121 (order 2026-09-12), at the final tips.

Every PRE defect runs again through the same API or CLI and must now
refuse, diverge or be answered correctly by its exact cause. The bank
scenarios are written by the PRE's own builder and their input digests are
compared with the ones the frozen PRE printed. Every preserved identity the
PRE captured is compared with the PRE's value. The real population is
recomputed from its bound inputs and compared with the published ledger.
Missing artifacts are NOT CORRECTED. The cube is read, never written.
Private paths are redacted to `~`.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

HOME = Path.home()
GH = HOME / "Documents/GitHub"
PRED, FD = GH / "predictor", GH / "financial-data"
B4, T2 = GH / ".worktrees/am-data-first", GH / ".worktrees/am-t0t1"
HERE = Path(__file__).resolve().parent
PRE_OUT = HERE / "c106_c121_pre_2026_09_12.out"
EVID = PRED / "docs/audits/evidence"
FC = FD / "features/census"
SUCC_ID = "financial_data.project3.ethusdt_4h_tech_stat.model_ready.successor_stage22_rerun.v1"
SUCC_SHA = "427a754ba3774e381e4f1353690b997d4e0ca89bf4ce6439ff2f8d651fd16d7c"
CHAR_ROOT = HOME / ".local/state/crispdm-successors/eth_h4_stage22_rerun_v1_characterization_v1"
B4_KEY = "b4::screen_b_v7_campaign"
RESULTS = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PRE = load(HERE / "c106_c121_pre_2026_09_12.py", "c106_c121_pre")
PRE.T2_PRE = T2  # identities and evidence are read at the final T2 tip
V5 = load(PRED / "tools/per_variable_design_v5.py", "design_v5_post")
SCOPE = load(PRED / "tools/terminal_evidence_scope.py", "scope_post")
sys.path.insert(0, str(PRED / "tests"))
import population_join_fixture as FX  # noqa: E402
sys.path.remove(str(PRED / "tests"))
sha, git, redact = PRE.sha, PRE.git, PRE.redact


def item(n, title, ok, detail):
    RESULTS.append((n, bool(ok)))
    print(f"[{n}] {'CORRECTED' if ok else 'NOT CORRECTED'}  {title}")
    print(f"     {redact(detail)}")


def pytest_line(repo, *tests):
    r = subprocess.run((sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"),
                       cwd=str(repo), capture_output=True, text=True, env=pg_env_merged(), timeout=3600)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return r.returncode == 0, (lines[-1] if lines else "no output")


def pg_env() -> dict:
    env = {}
    for line in (HOME / ".config/crispdm/olap-loader.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def pg_env_merged() -> dict:
    env = PRE.clean_env()
    env.update({k: v for k, v in pg_env().items() if k.startswith("PG")})
    return env


# ------------------------------------------------------------------- tips
def sec_tips():
    for repo in (PRED, FD, B4, T2):
        dirty = [ln for ln in git(repo, "status", "--porcelain").splitlines()
                 if not ln.endswith("docs/gobernanza_datalakes/") and "c106_c121_post" not in ln]
        print(f"  tip {repo.name}: {git(repo, 'rev-parse', 'HEAD')[:12]} "
              f"branch={git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')} tracked_dirty={dirty}")


# -------------------------------------------------------------------- T2
def sec_t2():
    E = T2 / "docs/audits/evidence"
    head = git(T2, "rev-parse", "HEAD")
    files = git(T2, "diff", "--name-only", PRE.BASES[PRE.T2_PRE] if False else
                "4bf38b6f4ba2a2e3e97f331f52619757d33be538", head).splitlines()
    item("C108.commit_b_is_evidence_only", "the T2 publication carries evidence and no code",
         bool(files) and all(f.startswith("docs/audits/evidence/") for f in files)
         and git(T2, "status", "--porcelain") == "", f"{head[:12]} {files}")
    rec = PRE.AUTH / "MUSASHI_T2_HARDENED_READJUDICATION_REVIEW_RECORD.json"
    mode = oct(rec.stat().st_mode & 0o777)
    item("C107.record_consumed_in_place", "the external record's bytes and mode are unchanged",
         sha(rec) == PRE.RECORD_SHA and mode == "0o600", f"{sha(rec)[:16]} mode={mode}")
    code = ("import sys, json; sys.path.insert(0, 'tools'); import t2_hardened_readjudicate as e; "
            "from pathlib import Path; print(json.dumps(e.preserved_inventory(Path(sys.argv[1]))))")
    r = subprocess.run([sys.executable, "-B", "-c", code, str(PRE.T2_ROOT)], cwd=T2,
                       env=PRE.clean_env(), capture_output=True, text=True, timeout=300)
    inv = json.loads(r.stdout) if r.returncode == 0 else {}
    item("C108.root_unchanged", "the preserved root keeps the inventory the record binds",
         inv.get("inventory_sha256") == "71c8ae7e87300f1e3b5491ce22ea602a96d004a4052cbaca9b85d7a4fe401193",
         {k: str(v)[:16] for k, v in inv.items()} or r.stderr[-200:])
    sub_p = E / "T2_READJUDICATION_SUBMISSION_V4_2026_09_12.json"
    clo_p = E / "T2_C108_HARDENED_READJUDICATION_CLOSURE_2026_09_12.json"
    add_p = E / "T2_READJUDICATION_SUBMISSION_V4_HISTORY_ADDENDUM_2026_09_12.json"
    if not all(p.is_file() for p in (sub_p, clo_p, add_p)):
        item("C109.published", "submission v4, closure and addendum published", False, "absent")
        return
    sub, clo, add = (json.loads(p.read_text()) for p in (sub_p, clo_p, add_p))
    hist = json.loads((T2 / PRE.EVID).read_text())["screen_adjudication"]
    h, cs = sub["hardened_readjudication"], clo["screen_adjudication"]
    item("C109.same_result", "242/0, DOES_NOT_ADVANCE, the same estimand and six equal effects",
         h["final_adjudication_counts"] == {"COMPLETED_VERIFIED": 242, "TERMINAL_FAILED": 0}
         and h["verdict"] == "DOES_NOT_ADVANCE"
         and h["primary_estimand"] == hist["primary_estimand_unweighted_mean_of_panel_effects"] == -0.001048443391358884
         and len(h["panel_effects_equal_to_historical"]) == 6 and all(h["panel_effects_equal_to_historical"].values()),
         f"{h['verdict']} {h['primary_estimand']}")
    item("C109.digest_equals_record", "scientific digest equals the record's corrected expectation",
         h["scientific_adjudication_sha256"] == PRE.CORRECTED == h["candidate_adjudication_sha256"]
         and h["equal_to_candidate"] is True and h["review_record_sha256"] == PRE.RECORD_SHA
         and sub["review_record_kind"] == "EXTERNAL_REVIEW_RECORD", h["scientific_adjudication_sha256"][:16])
    item("C106.flawed_template_not_used", "the published candidate is the record's, not the round-7 template",
         h["template_sha256"] == "UNDECLARED" and h["candidate_adjudication_sha256"] != PRE.ROUND7_CANDIDATE,
         {"template_sha256": h["template_sha256"]})
    item("C109.sign_test", "signs_positive 3 and sign_test_exact_p_two_sided 1.0 (replay closure)",
         cs["signs_positive"] == 3 and cs["sign_test_exact_p_two_sided"] == 1.0
         and clo["submission_written"]["sha256"] == sub["submission_sha256"],
         {"signs_positive": cs["signs_positive"], "p": cs["sign_test_exact_p_two_sided"]})
    sth = add["sign_test_history"]
    item("C109.history_bound", "1.3125 visible as invalid, superseded and not edited (addendum bound to the files)",
         sth["historical_value"] == 1.3125 and sth["historical_value_valid"] is False
         and sth["state"] == "INVALID_SUPERSEDED_NOT_EDITED" and hist["sign_test_exact_p_two_sided"] == 1.3125
         and add["binds"]["submission_file_sha256"] == sha(sub_p)
         and add["binds"]["historical_evidence_file_sha256"] == sha(T2 / PRE.EVID)
         and add["binds"]["closure_output_file_sha256"] == sha(clo_p), sth["state"])
    sup = h["sign_test_supersession"]
    item("C109.submission_itself_shows_1_3125",
         "the submission's own sign-test block shows 1.3125 as an invalid superseded value",
         sup.get("published_value") == 1.3125 and sup.get("published_value_valid") is False,
         f"published_value={sup.get('published_value')} valid={sup.get('published_value_valid')}: "
         "the builder superseded the RECOMPUTED screen; the history is only in the addendum")
    item("C109.grants_nothing", "no promotion, retraining, downloads or model execution",
         sub["grants_promotion"] is False and bool(sub["grants_nothing"])
         and not any(h[k] for k in ("retraining", "downloads", "model_execution")), sub["requires"])


# -------------------------------------------------------------------- B4
def sec_b4():
    head = git(B4, "rev-parse", "HEAD")
    base = "ff52ca7d9b771978dfe60bfd15e351ffbe3d1d22"
    doc = B4 / "docs/audits/work_plan/B4_V4_CLOSURE_DISPOSITION_2026_09_12.md"
    text = doc.read_text() if doc.is_file() else ""
    item("C110.disposition", "decision, identities, counts and verdict recorded",
         all(s in text for s in ("B4_V4_ACCEPTED_AS_NON_AUTHORIZING_FINAL_CLOSURE",
                                 "63ea37901c8756cd9669d18548176d337efeb08f1bb457a3e395eb90c7be7317",
                                 "9e7047ea077a3dfebc24999654d57c87a623a7345557ad308bfb8202e23a109b",
                                 "4c842dd10da9f4956eea4ffc9435492fcd36243e", base,
                                 "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT")), head[:12])
    numstat = git(B4, "diff", "--numstat", base, head, "--", "docs/audits/work_plan/04_OPEN_FINDINGS_REGISTER.md")
    deleted = sum(int(ln.split()[1]) for ln in numstat.splitlines()) if numstat else -1
    item("C110.register_append_only", "the register gained lines and lost none", deleted == 0, numstat)
    changed = git(B4, "diff", "--name-only", base, head).splitlines()
    tools = [f for f in changed if f.startswith("tools/")]
    item("C110.no_runner_or_consumer", "the only new tool is the closure envelope; no runner",
         tools == ["tools/b4_closure_envelope.py"], tools)

    from sqlalchemy import create_engine, text as sql
    e = pg_env()
    eng = create_engine(f"postgresql://{e['PGUSER']}:{e['PGPASSWORD']}@{e.get('PGHOST', '127.0.0.1')}:"
                        f"{e.get('PGPORT', '5432')}/{e.get('PGDATABASE', 'predictor_olap')}")
    with eng.connect() as c:
        runs = [dict(r) for r in c.execute(sql(
            "SELECT result_class, terminal_state, adjudication, run_id FROM public.dim_campaign_run "
            "WHERE campaign_key = :k"), {"k": B4_KEY}).mappings()]
        units = [dict(r) for r in c.execute(sql(
            "SELECT metric_name, metric_value, terminal_state FROM public.fact_campaign_unit "
            "WHERE campaign_key = :k ORDER BY metric_name"), {"k": B4_KEY}).mappings()]
        hist = {t: c.execute(sql(f"SELECT count(*) FROM public.{t}")).scalar()
                for t in ("fact_variable_characterization", "fact_terminal_verification_variable_v2")}
        stub = [dict(r) for r in c.execute(sql(
            "SELECT result_class, run_id FROM public.dim_campaign WHERE campaign_key = "
            "'b4_campaign_generation_v7_20260908'")).mappings()]
    eng.dispose()
    metrics = {u["metric_name"]: u["metric_value"] for u in units}
    item("C111.cube", "one closed B4 run with five campaign units: counts and declared costs, no failure",
         runs == [{"result_class": "DEVELOPMENT", "terminal_state": "CLOSED",
                   "adjudication": "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT",
                   "run_id": "b4_campaign_generation_v7_20260907"}]
         and metrics == {"cells_completed_verified": 2.0, "cells_not_started": 9.0,
                         "cells_quarantined_partial": 1.0,
                         "declared_wall_seconds_completed_cells": 35682.4,
                         "declared_wall_seconds_lower_bound_quarantined_partial": 124993.6}
         and not any("FAIL" in str(u["terminal_state"]).upper() for u in units), {"runs": runs, "metrics": metrics})
    item("C111.additive", "historical cube rows unchanged; the earlier B4 stub left as it was",
         hist == {"fact_variable_characterization": 40749, "fact_terminal_verification_variable_v2": 1965}
         and len(stub) == 1, {"historical": hist, "stub": stub})
    sys.path.insert(0, str(PRED))
    try:
        from olap import outbox as ob
    finally:
        sys.path.remove(str(PRED))
    h = ob.health(None)
    st = subprocess.run(["systemctl", "--user", "show", "crispdm-olap-loader.service",
                         "-p", "ActiveState", "-p", "NRestarts"], capture_output=True, text=True).stdout.split()
    item("C111.loader", "loader active with zero restarts; outbox healthy, nothing pending, dead letters adjudicated",
         "ActiveState=active" in st and "NRestarts=0" in st and h["healthy"]
         and h["backlog_pending"] == 0 and h["dead_letters_unadjudicated"] == 0,
         {"systemd": st, "counts": ob.counts(None)})


# ------------------------------------------------------------------ C112
def sec_c112():
    p = EVID / "TERMINALS_V4_DISPOSITION.v1.json"
    if not p.is_file():
        item("C112.disposition", "terminals v4 disposition published", False, "absent")
        return
    d = json.loads(p.read_text())
    sc = {k: v["count"] for k, v in d["scopes"].items()}
    item("C112.disposition", "the four scopes are kept apart, with counts",
         d["decision"] == "TERMINALS_V4_ACCEPTED_WITH_PHYSICAL_AND_STATISTICAL_SCOPE"
         and sc == {"NUMERIC_DESCRIPTORS_RECOMPUTED": 1501, "PHYSICAL_TYPE_KNOWN": 1505,
                    "SEMANTIC_DECLARATIONS_KNOWN": 0, "PRODUCER_AUTHORITY_ONLY": 460,
                    "SEMANTICALLY_UNRESOLVED": 4}, sc)
    item("C112.language", "'verified variables' phrasing is refused and absent from the disposition",
         bool(SCOPE.language_problems("1,501 verified variables")) and not SCOPE.language_problems(p.read_text()),
         SCOPE.describe("NUMERIC_DESCRIPTORS_RECOMPUTED", 1501))


# ----------------------------------------------------------- C113 / C117
EXPECTED = {
    "C117.terminal_of_other_variable": ("x_0", "NO_TERMINAL_FOR_KEY"),
    "C117.census_of_other_dataset": ("x_0", "NO_CENSUS_ROW_FOR_KEY"),
    "C117.source_digest_distinct": ("x_0", "NO_TERMINAL_FOR_KEY"),
    "C117.license_absent": ("x_0", "UNDECLARED_LICENSE"),
    "C117.role_absent": ("x_0", "ROLE_ABSENT"),
    "C117.temporal_contract_other_digest": ("x_0", "NO_TEMPORAL_CONTRACT_FOR_DATASET_AND_DIGEST"),
    "C117.mask_other_dataset": ("x_0", "MASK_NOT_BOUND_TO_SAME_DATASET"),
    "C117.member_only_in_aggregates": ("x_4", "NO_TERMINAL_FOR_KEY"),
    "C113.recomputed_role_unknown": ("x_0", "ROLE_UNKNOWN"),
    "C113.recomputed_license_unknown": ("x_0", "UNDECLARED_LICENSE"),
    "C113.semantically_unresolved_terminal": ("x_0", "TERMINAL_LAYER_SEMANTICALLY_UNRESOLVED"),
    "C113.date_stored_as_integer": ("x_0", "TIMESTAMP_EXCLUDED_BY_RULE"),
}
REFUSED = {"C117.defect_zero_terminals_zero_semantics": "KEY_INCOMPLETE",
           "C117.duplicate_column": "DUPLICATE"}


def sec_bank():
    pre_text = PRE_OUT.read_text()
    panel0 = ("panel_0", FX._dsha("panel_0"))
    for name in FX.SCENARIOS:
        with tempfile.TemporaryDirectory() as td:
            kw, digest = FX.write_bank(Path(td), name)
            m = re.search(r"\[" + re.escape(name) + r"\][^\n]*\n\s+input=([0-9a-f]{16})", pre_text)
            same = bool(m) and digest[:16] == m.group(1)
            try:
                pop = V5.derive_population(terminals_dir=kw["terminals_v4"], dag=kw["dag_v4"],
                                           census=kw["census"], temporal_contracts=kw["temporal_contracts"])
                refused = None
            except V5.PopulationRefusal as exc:
                pop, refused = {}, str(exc)
        head = f"input={digest[:16]} same_as_pre={same}"
        if name == "control_complete_bank":
            got = (pop.get("verdict"), pop.get("panel_count"), pop.get("members"))
            item(name, "a complete bank stays sufficient",
                 same and got == ("BANK_SUFFICIENT_FOR_REVIEW", 6, 30), f"{head} {got}")
        elif name in REFUSED:
            item(name, "refused by the join", same and refused is not None and REFUSED[name] in refused,
                 f"{head} refused={refused}")
        else:
            col, reason = EXPECTED[name]
            row = next((r for r in pop.get("ledger", [])
                        if (r["dataset_id"], r["dataset_sha256"], r["column"]) == (*panel0, col)), {})
            item(name, "not a sufficient bank, excluded by its exact reason",
                 same and pop.get("verdict") == "BANK_INSUFFICIENT" and pop.get("members") == 0
                 and row.get("member") is False and reason in row.get("reasons", []),
                 f"{head} verdict={pop.get('verdict')} panels={pop.get('panel_count')} reasons={row.get('reasons')}")


# ------------------------------------------------------------------ C118
def sec_c118():
    p = EVID / "PER_VARIABLE_DESIGN_V5_POPULATION.v1.json"
    if not p.is_file():
        item("C118.real_population", "real successor population published", False, "absent")
        return
    pub = json.loads(p.read_text())
    again = V5.derive_population(
        terminals_dir=CHAR_ROOT / "terminals", dag=FC / "FEATURE_DAG.v4.json",
        binding_manifest=FC / "PRODUCER_BINDING_MANIFEST.v2.json",
        census=FC / "ETH_H4_SUCCESSOR_SEMANTIC_CENSUS.v1.json",
        temporal_contracts=[FC / "ETH_H4_SUCCESSOR_TEMPORAL_CONTRACT.v3.json"])
    item("C118.recomputed_equals_published", "the ledger recomputed from its bound inputs equals the published one",
         all(again[k] == pub[k] for k in ("inputs", "population_sha256", "ledger_sha256", "verdict")),
         {k: str(again[k])[:16] for k in ("population_sha256", "ledger_sha256", "verdict")})
    item("C118.real_result", "89 candidates, 0 eligible, BANK_INSUFFICIENT, 6 panels missing; each exclusion names its cause",
         pub["candidates"] == 89 and pub["eligible_variables"] == 0 and pub["verdict"] == "BANK_INSUFFICIENT"
         and pub["deficit"]["panels_missing"] == 6
         and all(r["reasons"] and not r["member"] for r in pub["ledger"]),
         pub["exclusion_reasons"])


# ------------------------------------------------------------------ C119
def _setp(path, value):
    def mut(d):
        node = d
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return mut


def sec_c119():
    base = V5.build_design()

    def resealed(mut):
        d = copy.deepcopy(base)
        mut(d)
        d["design_sha256"] = V5.sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
        return d

    cases = {
        "bool_as_margin": (_setp(("hypotheses", "H1", "margin"), True), True),
        "nan_margin": (_setp(("hypotheses", "H1", "margin"), float("nan")), True),
        "inf_margin": (_setp(("hypotheses", "H2", "margins"), [0.01, float("inf")]), True),
        "margin_out_of_domain": (_setp(("hypotheses", "H1", "margin"), -5.0), True),
        "float_as_integer_minimum": (_setp(("panels", "minimum_independent_panels"), 6.0), True),
        "duplicate_seed": (_setp(("evaluation", "seeds"), [101, 101, 303]), True),
        "duplicate_operator": (_setp(("operators",), base["operators"] + ["O0_IDENTITY"]), True),
        "duplicate_contrast_in_family": (_setp(("inference", "family"), ["H1", "H1", "H2_vs_A3"]), True),
        "lopo_declared_not_derived": (_setp(("inference", "sensitivity"), "LOPO: PASSED"), False),
    }
    for name, (mut, typed) in cases.items():
        d = resealed(mut)
        problems = V5.validate(d)
        item(f"C119.{name}", "refused by the v5 validator",
             bool(problems) and (not typed or bool(V5.typed_problems(d))), problems[:2])
    for name, pv in {"p_above_one": {"H1": 1.5, "H2_vs_A0": 0.01, "H2_vs_A3": 0.02},
                     "p_nan": {"H1": float("nan"), "H2_vs_A0": 0.01, "H2_vs_A3": 0.02},
                     "incomplete_family": {"H1": 0.01}}.items():
        try:
            V5.holm_adjust_family(pv)
            item(f"C119.holm_{name}", "Holm refuses", False, "accepted")
        except V5.InferenceRefusal as exc:
            item(f"C119.holm_{name}", "Holm refuses", True, str(exc))
    rows = [{"panel": f"p{i}", "contrast": "H1", "sample_sha256": "a" * 64, "value": v}
            for i, v in enumerate((True, False, True, True, False, True))]
    try:
        V5.panel_contrast_rows(rows, contrast="H1", sample_sha256="a" * 64, level=0.95)
        item("C119.bool_panel_values", "booleans are refused as panel values", False, "accepted")
    except V5.InferenceRefusal as exc:
        item("C119.bool_panel_values", "booleans are refused as panel values", True, str(exc))
    try:
        V5.panel_contrast_rows([dict(r, value=0.01) for r in rows], contrast="H1", sample_sha256="", level=0.95)
        item("C119.sample_bound", "a contrast without a frozen sample digest is refused", False, "accepted")
    except V5.InferenceRefusal as exc:
        item("C119.sample_bound", "a contrast without a frozen sample digest is refused", True, str(exc))
    with tempfile.TemporaryDirectory() as td:
        text = json.dumps(base, sort_keys=True)
        dup, nan = Path(td) / "dup.json", Path(td) / "nan.json"
        dup.write_text('{"schema": "foreign.schema", ' + text[1:])
        nan.write_text(json.dumps(resealed(cases["nan_margin"][0]), sort_keys=True))
        for label, path in (("duplicate_key", dup), ("nan_constant", nan)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = V5.main(["--validate", str(path)])
            item(f"C119.parser_{label}", "the --validate CLI parses strictly and refuses",
                 rc == 1 and "STRICT_JSON" in buf.getvalue(), buf.getvalue().strip()[:160])


# ---------------------------------------------------------- C114 - C116
def sec_successor():
    c = json.loads((FC / "ETH_H4_SUCCESSOR_TEMPORAL_CONTRACT.v3.json").read_text())
    bs, cmp_ = c["bar_structure"], c["comparison_with_v2"]
    mask = c["mask_artifact"]
    item("C114.identity", "the contract and its mask are the successor's, not the historical dataset's",
         c["dataset"]["dataset_id"] == SUCC_ID == c["identity"]["dataset_id"] == mask["dataset_id"]
         and c["dataset"]["dataset_sha256"] == SUCC_SHA == mask["dataset_sha256"]
         and mask["sha256"] == sha(FC / "ETH_H4_SUCCESSOR_SAMPLE_ELIGIBILITY_MASK.v1.parquet"),
         {"contract": c["contract_id"], "mask": mask["sha256"][:16]})
    item("C114.counts", "18,085 rows, 20 truncated inside and 1 outside, 8 gaps, no filling",
         c["dataset"]["rows"] == 18085 and bs["truncated_bars_inside_dataset"] == 20
         and bs["truncated_bars_outside_dataset_count"] == 1 and bs["gaps_inside_dataset"] == 8
         and bs["gap_filling"].startswith("NONE"), {k: v for k, v in bs.items() if isinstance(v, int)})
    item("C114.geometry_not_identity", "equal geometry to v2 on every aspect, identity different",
         cmp_["geometry_all_equal"] is True and cmp_["identity_equal"] is False
         and cmp_["kind"] == "GEOMETRY_EQUALITY_NOT_IDENTITY_EQUALITY" and not cmp_["geometry_differences"],
         {"aspects": len(cmp_["geometry"]), "sample_summary": {k: c["sample_summary"][k] for k in ("eligible", "ineligible")}})
    b = c["bindings"]
    item("C114.bindings", "FEATURE_DAG.v4 and binding manifest v2 bound and verified",
         b["feature_dag_v4"]["dag_sha256_verified"] is True and b["binding_manifest_v2"]["manifest_sha256_verified"] is True
         and b["feature_dag_v4"]["file_sha256"] == sha(FC / "FEATURE_DAG.v4.json")
         and b["binding_manifest_v2"]["file_sha256"] == sha(FC / "PRODUCER_BINDING_MANIFEST.v2.json"),
         {"dag": b["feature_dag_v4"]["dag_sha256"][:16], "manifest": b["binding_manifest_v2"]["manifest_sha256"][:16]})
    cen = json.loads((FC / "ETH_H4_SUCCESSOR_SEMANTIC_CENSUS.v1.json").read_text())
    rows = V5.census_rows(cen)
    keys = {(r["dataset_id"], r["dataset_sha256"], r["column"]) for r in rows}
    item("C115.census", "89 rows with exact keys, one per active column, UNKNOWN kept as UNKNOWN",
         len(rows) == 89 and len(keys) == 89 and all(k[:2] == (SUCC_ID, SUCC_SHA) for k in keys)
         and len({r["variable_id"] for r in rows}) == 89,
         {"license": cen["counts"]["by_license"], "role": cen["counts"]["by_role"],
          "unit_unknown": sum(r["unit"] == "UNKNOWN" for r in rows)})
    ch = json.loads((FC / "ETH_H4_SUCCESSOR_CHARACTERIZATION.v1.json").read_text())
    terms = sorted((CHAR_ROOT / "terminals").glob("*.json"))
    item("C116.characterization", "89 own terminals from the bound bytes, pre-ledger first and unchanged",
         ch["columns"] == 89 == len(terms) and ch["counts_by_layer"] == {"INDEPENDENTLY_RECOMPUTED": 89}
         and ch["pre_ledger"]["sha256"] == sha(CHAR_ROOT / "PRE_LEDGER.json")
         and all(json.loads(t.read_text())["dataset_sha256"] == SUCC_SHA for t in terms),
         {"layers": ch["counts_by_layer"], "semantic": ch["counts_by_semantic_state"]})


# ------------------------------------------------------------ C120, C121
def sec_c120_c121():
    p = EVID / "PANEL_INVENTORY.v1.json"
    if not p.is_file():
        item("C120.inventory", "panel inventory published", False, "absent")
    else:
        inv = json.loads(p.read_text())
        pop = json.loads((EVID / "PER_VARIABLE_DESIGN_V5_POPULATION.v1.json").read_text())
        bank = inv["bank"]
        item("C120.inventory", "every candidate inventoried without labels or scores; deficit exact and consistent with the join",
             inv["panel_count"] == len(inv["panels"]) and inv["counts_toward_six_total"] == 0
             and bank["qualifying_panels"] == pop["panel_count"] == 0
             and bank["deficit"]["panels_missing"] == 6 and bank["verdict"] == "BANK_INSUFFICIENT"
             and "/home/" not in p.read_text(),
             {"candidates": inv["panel_count"], "deficit": bank["deficit"],
              "downloads": bank["download_authorization"]["used"]})
    d = V5.strict_json_file(EVID / "PER_VARIABLE_PREPROCESSING_DESIGN.v5.json")
    v4, v4_sha = V5.v4_reference()
    item("C121.design_v5", "v5 validates, supersedes v4 by digest with scientific change NONE, scoring closed",
         V5.validate(d) == [] and d["supersedes"]["design_sha256"] == v4["design_sha256"]
         and d["supersedes"]["file_sha256"] == v4_sha and d["supersedes"]["scientific_change"] == "NONE"
         and d["license"]["scoring"] == "NOT_GRANTED", d["design_sha256"][:16])
    try:
        V5.score()
        item("C121.score_refuses", "scoring refuses", False, "scored")
    except V5.ScoringRefusal as exc:
        item("C121.score_refuses", "scoring refuses", True, str(exc))


def sec_batteries():
    for label, repo, tests in (
            ("predictor", PRED, ("tests/test_per_variable_design_v5.py", "tests/test_terminal_evidence_scope.py",
                                 "tests/test_per_variable_design_v4.py", "tests/test_panel_inventory.py")),
            ("B4", B4, ("tests/test_b4_closure_envelope.py", "tests/test_b4_campaign_closure.py")),
            ("T2", T2, ("tests/test_t2_hardened_readjudicate.py", "tests/test_t2_campaign_closure.py")),
            ("financial-data", FD, ("tests/test_successor_temporal_quality.py", "tests/test_successor_semantic_census.py",
                                    "tests/test_successor_characterization.py", "tests/test_temporal_quality.py"))):
        present = [t for t in tests if (repo / t).is_file()]
        ok, line = pytest_line(repo, *present)
        item(f"{label}.battery", f"focal battery at the final tip ({len(present)} files)", ok, line)


def zero_line():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    apps = [ln for ln in r.stdout.splitlines() if ln.strip()] if r.returncode == 0 else ["nvidia-smi unavailable"]
    print(f"\n=== zero line ===\n  GPU compute processes now: {apps or 'none'}; "
          "training 0, scores 0, confirmation 0, promotion 0, live 0, venue 0, account actions 0")


def main() -> int:
    print(f"POST C106-C121 python={sys.version.split()[0]}")
    sec_tips()
    before = PRE.identities()
    sec_t2()
    sec_b4()
    sec_c112()
    sec_bank()
    sec_c118()
    sec_c119()
    sec_successor()
    sec_c120_c121()
    sec_batteries()
    after = PRE.identities()
    pre_ids = dict(re.findall(r"^  (\S[^\n]*?): ([0-9a-f]{16}|ABSENT) unchanged=", PRE_OUT.read_text(), re.M))
    print("\n=== identities (PRE value == now, and unchanged during the POST) ===")
    changed = []
    for k, v in pre_ids.items():
        now = after.get(k, "MISSING")[:16]
        ok = now == v and before.get(k) == after.get(k)
        changed += [] if ok else [k]
        print(f"  {k}: pre={v} now={now} equal={ok}")
    new = sorted(set(after) - set(pre_ids))
    print(f"  new since PRE (not preserved items): {new}")
    zero_line()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n=== POST SUMMARY: {len(RESULTS)} checks; {len(bad)} NOT corrected {bad}; "
          f"preserved identities changed {changed} ===")
    return 0 if not (bad or changed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
