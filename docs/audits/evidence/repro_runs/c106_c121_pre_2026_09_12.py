"""PRE for C106-C121 (order 2026-09-12), frozen before any edit.

Runs at the ordered bases: predictor 6a0dce44, financial-data 7d9862a3,
B4 ff52ca7d, and T2 4bf38b6f from a clean detached checkout reserved for
this PRE (the C108 replay uses a different one). Every defect runs through
the real API or CLI; facts that are only captured say so. A defect
REPRODUCED is the expected PRE outcome. Preserved roots and historical
artifacts are read only. Private paths are redacted to `~`.

The full hardened replay is NOT re-run here: C108 authorizes a single
attempt. Its round-7 output is the committed divergence report at
4bf38b6f, which C106.2 reads and recomputes through the closure API.

The bank scenarios (C113, C117) are written by `write_bank`. Each prints
its input digest, so the POST can prove it attacks the same bytes.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

HOME = Path.home()
GH = HOME / "Documents/GitHub"
PRED, FD, AM = GH / "predictor", GH / "financial-data", GH / "agent-multi"
B4 = GH / ".worktrees/am-data-first"
T2 = GH / ".worktrees/am-t0t1"
T2_PRE = GH / ".runtime/am-t2-c106-pre-4bf38b6f"
REPRO = GH / ".runtime/am-t2-reproducer"
LAKE = HOME / ".local/state/crispdm/lake_characterization"
T2_ROOT = HOME / ".local/share/agent-multi/t2_confirmatory_results_resource_successor_v1_20260909"
AUTH = HOME / ".config/agent-multi/reviewer_authority"
SUCCESSOR = HOME / ".local/state/crispdm-successors/eth_h4_stage22_rerun_v1"
EVID = "docs/audits/evidence/T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_ADJUDICATION_2026_09_10.json"
DIVERGENCE = "docs/audits/evidence/T2_HARDENED_READJUDICATION_DIVERGENCE_2026_09_12.json"
MUSASHI_REF = "origin/musashi/round7-c87-c105-audit-20260912"
MUSASHI_REPRO = "docs/audits/evidence/repro_runs/musashi_round7_additional_pre_2026_09_12"

BASES = {PRED: "6a0dce448036a77d9e2b3c1c2037f5457b63f8d7",
         FD: "7d9862a331b98f3d07d1fd606c6963e1e5575dbe",
         B4: "ff52ca7d9b771978dfe60bfd15e351ffbe3d1d22",
         T2_PRE: "4bf38b6f4ba2a2e3e97f331f52619757d33be538"}
ROUND7_CANDIDATE = "a989d02d3673c04c8eaa7976050ddc0b7b3e7dec3e0e6a175f04a7ef02b004ca"
CORRECTED = "1be80a0ab6d091794f7ce3ec97c2dbf88920911037c0415c383043bd95ca3a6d"
RECORD_SHA = "c75d8d6ef4f1a1af67768d8c53507b231c316d9373b9d8dc52c22a0bdfdff1ff"
FORMULA_COMMIT = "5fb2849eb7dc2723d00db1e4f365fbba19a35672"
REPLAY_COMMIT = "4d0b315952838b93ec207e2d1faa1655a47183b5"
DESIGN_COMMIT = "86ca8ca88b8d1f43c37c8a262e39182ee1a4d772"
SUCCESSOR_ID = "financial_data.project3.ethusdt_4h_tech_stat.model_ready.successor_stage22_rerun.v1"

RESULTS = []


def redact(x) -> str:
    return str(x).replace(str(HOME), "~")


def item(n, title, reproduced, detail, kind="DEFECT"):
    """kind DEFECT: reproduced is the expected PRE state.
    kind FACT: a captured baseline; `reproduced` means the fact holds."""
    RESULTS.append((n, kind, reproduced))
    label = {("DEFECT", True): "REPRODUCED", ("DEFECT", False): "NOT REPRODUCED",
             ("FACT", True): "FACT HOLDS", ("FACT", False): "FACT DIFFERS"}[(kind, bool(reproduced))]
    print(f"[{n}] {label}  {title}")
    print(f"     {redact(detail)}")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def tree_digest(root):
    e = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            e.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns, st.st_ino))
    return hashlib.sha256(json.dumps(e).encode()).hexdigest()


def content_digest(d):
    h = hashlib.sha256()
    for p in sorted(Path(d).glob("*.json")):
        h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def clean_env():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    env.pop("PYTHONPATH", None)
    return env


# ---------------------------------------------------------------- bases
def sec_bases():
    for repo, want in BASES.items():
        head = git(repo, "rev-parse", "HEAD")
        dirty = [l for l in git(repo, "status", "--porcelain").splitlines()
                 if not l.endswith("docs/gobernanza_datalakes/")
                 and "c106_c121_pre_2026_09_12" not in l]
        item(f"BASE.{repo.name}", "checkout at the ordered base, no tracked edits",
             head == want and not dirty, f"{head[:12]} dirty={dirty}", kind="FACT")


# ----------------------------------------------------------- identities
def identities():
    ids = {"b4_v7_real": tree_digest(HOME / ".local/share/agent-multi/b4_campaign_results_v7_20260907"),
           "t2_original_real": tree_digest(T2_ROOT),
           "b4_v7_private_copy": tree_digest(HOME / ".local/state/crispdm-pre-copies/b4_v7"),
           "t2_successor_private_copy": tree_digest(HOME / ".local/state/crispdm-pre-copies/t2_successor"),
           "external_t2_record": sha(AUTH / "MUSASHI_T2_HARDENED_READJUDICATION_REVIEW_RECORD.json"),
           "t2_historical_evidence_1_3125": sha(T2_PRE / EVID),
           "t2_round7_divergence_report": sha(T2_PRE / DIVERGENCE)}
    for v in ("terminals", "terminals_v2", "terminals_v3", "terminals_v4"):
        d = LAKE / v
        ids[f"lake_{v}"] = content_digest(d) if d.is_dir() else "ABSENT"
    for f in ("FEATURE_DAG.v1.json", "FEATURE_DAG.v2.json", "FEATURE_DAG.v3.json",
              "FEATURE_DAG.v4.json", "PRODUCER_BINDING_MANIFEST.v1.json",
              "PRODUCER_BINDING_MANIFEST.v2.json", "ETH_H4_TEMPORAL_CONTRACT.v1.json",
              "ETH_H4_TEMPORAL_CONTRACT.v2.json", "ETH_H4_SAMPLE_ELIGIBILITY_MASK.v1.parquet"):
        ids[f] = sha(FD / "features/census" / f)
    for p in sorted((B4 / "docs/audits/evidence").glob("B4_*SUBMISSION*.json")):
        ids[f"b4:{p.name}"] = sha(p)
    for base in (T2_PRE, REPRO):
        for p in sorted((base / "docs/audits/evidence").glob("T2_*SUBMISSION*.json")) if base.is_dir() else []:
            ids[f"t2:{p.name}"] = sha(p)
    for p in sorted((PRED / "docs/audits/evidence").glob("PER_VARIABLE*DESIGN*.json")):
        ids[f"design:{p.name}"] = sha(p)
    return ids


# ------------------------------------------------------------------ C106
API = r'''
import copy, json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, "tools")
import t2_campaign_closure as C
ev = json.load(open(sys.argv[1]))
screen = ev["screen_adjudication"]
sup = C.supersede_sign_test(screen)
template_body = {"final_adjudication_counts": ev["final_adjudication_counts"],
                 "screen_adjudication": screen, "sign_test_supersession": sup}
fixed = copy.deepcopy(screen)
fixed["sign_test_exact_p_two_sided"] = sup["corrected_value"]
print(json.dumps({
    "template_digest": C.scientific_adjudication_digest(template_body),
    "template_field": screen["sign_test_exact_p_two_sided"],
    "correction_beside_field": sup["corrected_value"],
    "correction_state": sup["state"],
    "published_value_valid": sup["published_value_valid"],
    "signs_positive": sup["signs_positive"],
    "table": sup["corrected_table_0_to_6"],
    "corrected_digest": C.scientific_adjudication_digest(dict(template_body, screen_adjudication=fixed)),
}))
'''


def committed_function(repo, commit, path, name):
    """Execute a function exactly as committed, isolated from its module."""
    src = git(repo, "show", f"{commit}:{path}")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"math": math, "comb": math.comb}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), f"{commit[:8]}:{path}", "exec"), ns)
    return ns[name], hashlib.sha256(ast.get_source_segment(src, fn).encode()).hexdigest()


def sec_c106():
    env = clean_env()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "TEMPLATE.json"
        r = subprocess.run([sys.executable, "-B", "tools/t2_hardened_readjudicate.py",
                            "--checkout", str(T2_PRE), "--root", str(T2_ROOT),
                            "--template-out", str(out), "--candidate-evidence", EVID],
                           cwd=T2_PRE, env=env, capture_output=True, text=True, timeout=600)
        cli = json.loads(r.stdout) if r.returncode == 0 else {}
        written = json.loads(out.read_text()) if out.is_file() else {}
    p_hist = json.loads((T2_PRE / EVID).read_text())["screen_adjudication"]["sign_test_exact_p_two_sided"]
    item("C106.1", "round-7 candidate: template CLI yields a989d02d from evidence holding 1.3125",
         cli.get("candidate_adjudication_sha256") == ROUND7_CANDIDATE
         and written.get("candidate_adjudication_sha256") == ROUND7_CANDIDATE and p_hist == 1.3125,
         f"rc={r.returncode} cli={cli.get('candidate_adjudication_sha256', r.stderr[-200:])} evidence_p={p_hist}")

    a = subprocess.run([sys.executable, "-B", "-c", API, EVID], cwd=T2_PRE, env=env,
                       capture_output=True, text=True, timeout=600)
    api = json.loads(a.stdout) if a.returncode == 0 else {}
    div = json.loads((T2_PRE / DIVERGENCE).read_text())
    diffs = div["differences"]
    item("C106.2", "hardened replay produced 1.0 and 1be80a0a (committed report + closure API)",
         div["recomputed_adjudication_sha256"] == CORRECTED and len(diffs) == 1
         and diffs[0] == {"path": "screen_adjudication.sign_test_exact_p_two_sided",
                          "candidate": 1.3125, "recomputed": 1.0}
         and api.get("corrected_digest") == CORRECTED,
         f"report={div['recomputed_adjudication_sha256'][:12]} diffs={diffs} "
         f"api_corrected={api.get('corrected_digest', a.stderr[-200:])[:12]} "
         "(replay not re-run: C108 is single-attempt)")

    fn_f, src_f = committed_function(AM, FORMULA_COMMIT, "tools/t2_confirmatory.py", "_two_sided_binomial_p")
    fn_r, src_r = committed_function(AM, REPLAY_COMMIT, "tools/t2_confirmatory.py", "_two_sided_binomial_p")
    table = [fn_f(k, 6) for k in range(7)]
    anc = subprocess.run(["git", "-C", str(AM), "merge-base", "--is-ancestor",
                          FORMULA_COMMIT, REPLAY_COMMIT]).returncode == 0
    t_f = git(AM, "log", "-1", "--format=%cI", FORMULA_COMMIT)
    t_r = git(AM, "log", "-1", "--format=%cI", REPLAY_COMMIT)
    t_d = git(PRED, "log", "-1", "--format=%cI", DESIGN_COMMIT)
    item("C106.3", "the two-sided formula was committed in 5fb2849e before 4d0b3159 (executed as committed)",
         anc and t_f < t_d < t_r and fn_f(3, 6) == 1.0
         and table == [0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875, 0.03125]
         and src_f == src_r,
         f"ancestor={anc} formula={t_f} design={t_d} replay={t_r} p(3,6)={fn_f(3, 6)} "
         f"table={table} same_function_at_replay={src_f == src_r}")

    item("C106.4", "template flaw: the correction sits beside the field and is not applied to it",
         api.get("template_field") == 1.3125 and api.get("correction_beside_field") == 1.0
         and api.get("correction_state") == "SUPERSEDED" and api.get("published_value_valid") is False
         and api.get("template_digest") == ROUND7_CANDIDATE and api.get("corrected_digest") == CORRECTED,
         {k: api.get(k) for k in ("template_field", "correction_beside_field", "correction_state",
                                  "published_value_valid", "signs_positive")})
    item("C107.record", "external record installed with the ordered digest", sha(
        AUTH / "MUSASHI_T2_HARDENED_READJUDICATION_REVIEW_RECORD.json") == RECORD_SHA,
         "bytes read only; never copied, regenerated or chmod-ed", kind="FACT")
    clean = git(T2_PRE, "status", "--porcelain") == ""
    item("C106.clean", "PRE checkout still clean after CLI and API runs", clean, git(T2_PRE, "rev-parse", "HEAD"), kind="FACT")


# ---------------------------------------------------------- C113 / C117
PANELS, COLS = 6, 5


def _dsha(tag: str) -> str:
    return hashlib.sha256(f"fixture-bytes:{tag}".encode()).hexdigest()


def bank_rows() -> dict:
    """A bank in which every member satisfies all eight C118 conditions."""
    dag, terms, census, temporal = [], [], [], []
    for p in range(PANELS):
        ds = f"panel_{p}"
        dsha = _dsha(ds)
        for c in range(COLS):
            col = f"x_{c}"
            vid = f"{ds}:{col}"
            dag.append({"dataset_id": ds, "dataset_sha256": dsha, "column": col,
                        "class": "CAUSAL_ACTIVE", "binding": {"complete": True}})
            terms.append({"dataset_id": ds, "dataset_sha256": dsha, "column": col,
                          "variable_id": vid, "layer": "INDEPENDENTLY_RECOMPUTED",
                          "semantic_state": "NUMERIC_MEASURABLE", "physical_type": "float64",
                          "observations": 4096, "missing_fraction": 0.0})
            census.append({"variable_id": vid, "dataset_id": ds, "dataset_sha256": dsha,
                           "column": col, "physical_type": "float64",
                           "semantic_type": "continuous_measurement",
                           "semantics": "continuous_measurement", "role": "input_feature",
                           "unit": "1", "license": "CC-BY-4.0", "license_source": "fixture",
                           "missing_policy": "EXCLUDE_ROW_NO_IMPUTATION",
                           "sentinel_policy": "NO_SENTINELS_DECLARED", "producer": "fixture",
                           "symbol": "fixture", "lookback_bars": 1,
                           "evidence_sha256": _dsha("evidence:" + vid)})
        temporal.append({"dataset": {"dataset_id": ds, "dataset_sha256": dsha},
                         "contract_sha256": _dsha("contract:" + ds),
                         "mask_artifact": {"dataset_id": ds, "dataset_sha256": dsha,
                                           "sha256": _dsha("mask:" + ds)}})
    return {"dag": dag, "terminals": terms, "census": census, "temporal": temporal,
            "census_extra": {}}


def _find(rows, ds="panel_0", col="x_0"):
    return next(r for r in rows if r.get("dataset_id") == ds and r.get("column") == col)


def _mut_defect(b):
    b["terminals"] = []
    b["census"] = [{"variable_id": "not_a_member", "semantics": "UNKNOWN", "role": "UNKNOWN",
                    "license": "UNKNOWN", "unit": "UNKNOWN"}]


def _mut_other_variable(b):
    t = _find(b["terminals"]); t["column"] = "x_9"; t["variable_id"] = "panel_0:x_9"


def _mut_census_other_dataset(b):
    c = _find(b["census"]); c["dataset_id"] = "panel_other"; c["dataset_sha256"] = _dsha("panel_other")


def _mut_source_digest(b):
    _find(b["terminals"])["dataset_sha256"] = _dsha("different-bytes")


def _mut_duplicate_column(b):
    b["census"].remove(_find(b["census"], col="x_4"))
    b["census"].append(copy.deepcopy(_find(b["census"])))


def _mut_license_absent(b):
    del _find(b["census"])["license"]


def _mut_role_absent(b):
    del _find(b["census"])["role"]


def _mut_temporal_other_digest(b):
    b["temporal"][0]["dataset"]["dataset_sha256"] = _dsha("other-contract-dataset")


def _mut_mask_other_dataset(b):
    b["temporal"][0]["mask_artifact"]["dataset_id"] = "panel_5"
    b["temporal"][0]["mask_artifact"]["dataset_sha256"] = _dsha("panel_5")


def _mut_only_in_aggregates(b):
    # panel_0:x_4 keeps its DAG node and an aggregate claim, but has no
    # terminal and no census row; cardinalities are kept with rows for a
    # column that is in no DAG.
    for key in ("terminals", "census"):
        row = _find(b[key], col="x_4")
        row.update(dataset_id="panel_5", dataset_sha256=_dsha("panel_5"), column="x_9",
                   variable_id="panel_5:x_9")
    b["census_extra"] = {"aggregates": {"panel_0": {"members": 5}}}


def _mut_role_unknown(b):
    _find(b["census"])["role"] = "UNKNOWN"


def _mut_license_unknown(b):
    _find(b["census"])["license"] = "UNKNOWN"


def _mut_semantically_unresolved(b):
    t = _find(b["terminals"]); t["layer"] = "SEMANTICALLY_UNRESOLVED"
    t["semantic_state"] = "SEMANTIC_TYPE_UNRESOLVED"


def _mut_date_as_integer(b):
    t = _find(b["terminals"]); t["physical_type"] = "int64"
    t["semantic_state"] = "SEMANTIC_TYPE_UNRESOLVED"
    c = _find(b["census"]); c["physical_type"] = "int64"; c["semantic_type"] = "datetime"


SCENARIOS = {
    "control_complete_bank": None,
    "C117.defect_zero_terminals_zero_semantics": _mut_defect,
    "C117.terminal_of_other_variable": _mut_other_variable,
    "C117.census_of_other_dataset": _mut_census_other_dataset,
    "C117.source_digest_distinct": _mut_source_digest,
    "C117.duplicate_column": _mut_duplicate_column,
    "C117.license_absent": _mut_license_absent,
    "C117.role_absent": _mut_role_absent,
    "C117.temporal_contract_other_digest": _mut_temporal_other_digest,
    "C117.mask_other_dataset": _mut_mask_other_dataset,
    "C117.member_only_in_aggregates": _mut_only_in_aggregates,
    "C113.recomputed_role_unknown": _mut_role_unknown,
    "C113.recomputed_license_unknown": _mut_license_unknown,
    "C113.semantically_unresolved_terminal": _mut_semantically_unresolved,
    "C113.date_stored_as_integer": _mut_date_as_integer,
}


def write_bank(root: Path, scenario: str) -> tuple[dict, str]:
    b = bank_rows()
    if SCENARIOS[scenario]:
        SCENARIOS[scenario](b)
    (root / "terminals").mkdir()
    for i, t in enumerate(b["terminals"]):
        (root / "terminals" / f"{i:03d}.json").write_text(json.dumps(t, sort_keys=True))
    (root / "dag.json").write_text(json.dumps({"nodes": b["dag"]}, sort_keys=True))
    (root / "census.json").write_text(json.dumps(dict(b["census_extra"], variables=b["census"]), sort_keys=True))
    contracts = []
    for i, c in enumerate(b["temporal"]):
        p = root / f"temporal_{i}.json"
        p.write_text(json.dumps(c, sort_keys=True))
        contracts.append(p)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.json")):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return {"terminals_v4": root / "terminals", "dag_v4": root / "dag.json",
            "census": root / "census.json", "temporal_contracts": contracts}, h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sec_bank():
    design = load_module(PRED / "tools/per_variable_design_v4.py", "design_v4_pre")
    for name in SCENARIOS:
        with tempfile.TemporaryDirectory() as td:
            kwargs, digest = write_bank(Path(td), name)
            pop = design.derive_population(**kwargs)
        got = (pop.get("verdict"), pop.get("panel_count"), pop.get("members"))
        detail = f"input={digest[:16]} verdict={got[0]} panels={got[1]} members={got[2]}"
        if name == "control_complete_bank":
            item(name, "a complete bank is sufficient (control, must stay so)",
                 got == ("BANK_SUFFICIENT_FOR_REVIEW", 6, 30), detail, kind="FACT")
        else:
            item(name, "the current derivation admits it as a sufficient bank",
                 got == ("BANK_SUFFICIENT_FOR_REVIEW", 6, 30), detail)

    with tempfile.TemporaryDirectory() as td:
        # The reproducer's argparse default evaluates parents[4] of its own
        # path even when --predictor-root is given; it must sit at the same
        # depth it has in predictor (docs/audits/evidence/repro_runs/).
        script = Path(td) / "predictor/docs/audits/evidence/repro_runs/musashi_repro.py"
        script.parent.mkdir(parents=True)
        script.write_text(git(PRED, "show", f"{MUSASHI_REF}:{MUSASHI_REPRO}.py") + "\n")
        r = subprocess.run([sys.executable, "-B", str(script), "--t2-root", str(T2_PRE),
                            "--predictor-root", str(PRED)], env=clean_env(),
                           capture_output=True, text=True, timeout=600)
    want = git(PRED, "show", f"{MUSASHI_REF}:{MUSASHI_REPRO}.out")
    item("C117.musashi_reproducer", "the auditor's independent reproducer gives its captured output here",
         r.returncode == 0 and r.stdout.strip() == want, r.stdout.strip().splitlines()[-1:] or r.stderr[-300:])


# ------------------------------------------------------------ C110, C114
def sec_facts():
    sub = B4 / "docs/audits/evidence/B4_READJUDICATION_SUBMISSION_V4_2026_09_12.json"
    d = json.loads(sub.read_text())
    item("C110.identity", "B4 v4 file and self digest are the ones the decision names",
         sha(sub) == "63ea37901c8756cd9669d18548176d337efeb08f1bb457a3e395eb90c7be7317"
         and d["submission_sha256"] == "9e7047ea077a3dfebc24999654d57c87a623a7345557ad308bfb8202e23a109b",
         {"file": sha(sub)[:12], "self": d["submission_sha256"][:12]}, kind="FACT")

    v2 = json.loads((FD / "features/census/ETH_H4_TEMPORAL_CONTRACT.v2.json").read_text())
    summary = json.loads((FD / "features/census/ETH_H4_STAGE22_RERUN_SUMMARY.v1.json").read_text())
    succ = {p.name: sha(p) for p in sorted((SUCCESSOR / "successor").iterdir())}
    item("C114.pre", "no successor temporal contract: v2 names the historical dataset",
         v2["dataset"]["dataset_id"] != SUCCESSOR_ID
         and not (FD / "features/census/ETH_H4_TEMPORAL_CONTRACT.v3.json").exists(),
         {"v2_dataset": v2["dataset"]["dataset_id"], "successor_declared_sha": summary["successor_dataset_sha256"][:12],
          "successor_files": {k: v[:12] for k, v in succ.items()}})
    item("C115-C116.pre", "no successor semantic census and no successor characterization",
         not list((FD / "features/census").glob("*SUCCESSOR*SEMANTIC*"))
         and not (LAKE / "successor_terminals").exists(),
         "absent before this order")


# ------------------------------------------------------------------ C111
def sec_c111():
    sys.path.insert(0, str(PRED))
    try:
        from olap import outbox as ob
    finally:
        sys.path.remove(str(PRED))
    root = ob.outbox_root(None)
    keys = {}
    for state in ob.STATE_DIRS:
        for p in sorted((root / state).glob("envelope-*.json")):
            if p.name.endswith(ob.SIDECAR_SUFFIX):
                continue
            key = json.loads(p.read_text()).get("document", {}).get("campaign_key")
            keys.setdefault(state, []).append(key)
    b4 = [k for ks in keys.values() for k in ks if str(k).lower().startswith("b4")]
    item("C111.pre", "no B4 campaign-closure envelope exists in the outbox (outbox API)",
         not b4, {"counts": ob.counts(root), "b4_keys": b4,
                  "campaign_keys": sorted({str(k) for ks in keys.values() for k in ks})}, kind="FACT")


# ------------------------------------------------------------------ C119
def sec_c119():
    import contextlib
    import inspect
    import io
    design = load_module(PRED / "tools/per_variable_design_v4.py", "design_v4_c119")
    base = design.build_design()
    item("C119.control", "the committed v4 design validates", design.validate(base) == [],
         design.validate(base), kind="FACT")

    def resealed(mut):
        d = copy.deepcopy(base)
        mut(d)
        d["design_sha256"] = design.sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
        return d

    def setp(path, value):
        def mut(d):
            node = d
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
        return mut

    cases = {
        "bool_as_margin": setp(("hypotheses", "H1", "margin"), True),
        "nan_margin": setp(("hypotheses", "H1", "margin"), float("nan")),
        "inf_margin": setp(("hypotheses", "H2", "margins"), [0.01, float("inf")]),
        "margin_out_of_domain": setp(("hypotheses", "H1", "margin"), -5.0),
        "float_as_integer_minimum": setp(("panels", "minimum_independent_panels"), 6.0),
        "duplicate_seed": setp(("evaluation", "seeds"), [101, 101, 303]),
        "duplicate_operator": setp(("operators",), base["operators"] + ["O0_IDENTITY"]),
        "duplicate_contrast_in_family": setp(("inference", "family"), ["H1", "H1", "H2_vs_A3"]),
        "lopo_declared_not_derived": setp(("inference", "sensitivity"), "LOPO: PASSED"),
    }
    for name, mut in cases.items():
        problems = design.validate(resealed(mut))
        item(f"C119.{name}", "the design validator accepts it", problems == [], problems)

    holm = {
        "p_above_one": {"H1": 1.5, "H2_vs_A0": 0.01, "H2_vs_A3": 0.02},
        "p_nan": {"H1": float("nan"), "H2_vs_A0": 0.01, "H2_vs_A3": 0.02},
        "incomplete_family": {"H1": 0.01},
    }
    for name, pv in holm.items():
        try:
            out = design.holm_adjust(pv)
            item(f"C119.holm_{name}", "Holm adjusts it without refusing", True,
                 {k: v["adjusted_p"] for k, v in out.items()})
        except Exception as exc:  # noqa: BLE001
            item(f"C119.holm_{name}", "Holm adjusts it without refusing", False, repr(exc))
    pc = design.panel_contrast([True, False, True, True, False, True], 0.95)
    item("C119.bool_panel_values", "a panel contrast accepts booleans as panel values",
         pc.get("state") == "INFERENTIAL", pc)
    params = inspect.signature(design.panel_contrast).parameters
    item("C119.sample_unbound", "a contrast is not bound to a frozen common-sample digest",
         not any("sample" in p for p in params), list(params))

    with tempfile.TemporaryDirectory() as td:
        text = json.dumps(base, sort_keys=True)
        dup = Path(td) / "dup.json"
        dup.write_text('{"schema": "foreign.schema", ' + text[1:])
        nan = Path(td) / "nan.json"
        nan.write_text(json.dumps(resealed(cases["nan_margin"]), sort_keys=True))
        for label, path in (("duplicate_key", dup), ("nan_constant", nan)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                design.main(["--validate", str(path)])
            problems = json.loads(buf.getvalue())["problems"]
            item(f"C119.parser_{label}", "the --validate CLI parses it leniently and accepts it",
                 problems == [], problems)


def main() -> int:
    print(f"PRE C106-C121 python={sys.version.split()[0]}")
    sec_bases()
    before = identities()
    sec_c106()
    sec_bank()
    sec_facts()
    sec_c111()
    sec_c119()
    after = identities()
    print("\n=== identities (before == after) ===")
    for k in before:
        print(f"  {k}: {before[k][:16]} unchanged={before[k] == after[k]}")
    defects = [n for n, kind, ok in RESULTS if kind == "DEFECT"]
    missing = [n for n, kind, ok in RESULTS if kind == "DEFECT" and not ok]
    facts_differ = [n for n, kind, ok in RESULTS if kind == "FACT" and not ok]
    changed = [k for k in before if before[k] != after[k]]
    print(f"\n=== PRE SUMMARY: {len(defects)} defects, {len(defects) - len(missing)} reproduced, "
          f"not reproduced {missing}; facts differing {facts_differ}; identities changed {changed} ===")
    return 0 if not (missing or facts_differ or changed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
