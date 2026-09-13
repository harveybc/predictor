"""PRE for C122-C145 DATA FOUNDATION (order 2026-09-12), frozen before any
implementation.

C125 asks for an executable status matrix for STEP 01-13 derived from
artifacts, not prose. For each step this script checks:

* protocol present   - the protocol document exists on disk;
* implementation     - step-specific symbols DEFINED in code (found by parsing
                       the AST, not by grepping prose), and which operator kinds
                       the common causal-operator contract DECLARES for it;
* experiment         - executed output artifacts exist;
* evidence scope     - derived from those artifacts;
* eligibility        - any PUBLICLY_ELIGIBLE grant, in files or in the cube.

It must reproduce, at minimum: zero v5-eligible variables; zero publicly
eligible denoising operators; no executed matrix for STEP 04-13; that the
seven v5 operators do not represent the 13 steps; and that a numeric
statistic grants neither semantics nor license. It also captures the D0-D2
baseline (what does not exist yet), so the POST can show what was built.

Read only. The cube is queried, never written. Private paths are redacted
to `~`.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

HOME = Path.home()
GH = HOME / "Documents/GitHub"
PRED, FD, AM = GH / "predictor", GH / "financial-data", GH / "agent-multi"
T2, B4 = GH / ".worktrees/am-t0t1", GH / ".worktrees/am-data-first"
PREP = GH / ".worktrees/prep-t0t1"
DOCS = PRED / "docs/tres_temas_entrevista"
EVID = PRED / "docs/audits/evidence"
FC = FD / "features/census"
CONTRACT_MODULE = PREP / "app/causal_operators.py"

BASES = {PRED: "f6bf437def84d99e575c5c64893b49644e7546d5",
         FD: "fba552195", T2: "a18ca3b2", B4: "432599be", PREP: "e6c3cdc"}
RESULTS = []


def redact(x) -> str:
    return str(x).replace(str(HOME), "~")


def item(n, title, holds, detail, kind="ABSENCE"):
    """ABSENCE/DEFECT: `holds` means the gap is reproduced (expected PRE).
    FACT: `holds` means the baseline fact is true."""
    RESULTS.append((n, kind, bool(holds)))
    label = {"ABSENCE": ("REPRODUCED", "NOT REPRODUCED"),
             "FACT": ("FACT HOLDS", "FACT DIFFERS")}[kind if kind == "FACT" else "ABSENCE"][0 if holds else 1]
    print(f"[{n}] {label}  {title}")
    print(f"     {redact(detail)}")


def git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ bases
def sec_bases():
    for repo, want in BASES.items():
        head = git(repo, "rev-parse", "HEAD")
        dirty = [ln for ln in git(repo, "status", "--porcelain").splitlines()
                 if not ln.startswith("??")]
        item(f"BASE.{repo.name}", "checkout at the ordered base with no tracked edits",
             head.startswith(want) and not dirty, f"{head[:12]} dirty={dirty}", kind="FACT")


# ---------------------------------------------------------- code index
CODE_ROOTS = [
    (PRED, ("tools", "olap", "app", "preprocessor_plugins", "pipeline_plugins", "predictor_plugins",
            "target_plugins", "optimizer_plugins")),
    (FD, ("_scripts",)),
    (PREP, ("app",)),
    (T2, ("tools", "agent_plugins", "feature_branch_plugins")),
    (GH / "preprocessor", (".",)),
    (GH / "feature-eng", (".",)),
    (GH / "feature-extractor", (".",)),
    (GH / "synthetic-datagen", ("sdg_plugins",)),
]


def code_index() -> list[tuple[Path, str]]:
    """(file, defined function/class name) for implementation code; tests,
    virtualenvs and repro scripts are not implementations."""
    out, seen = [], set()
    for repo, subdirs in CODE_ROOTS:
        for sub in subdirs:
            base = repo / sub
            if not base.is_dir():
                continue
            for p in base.rglob("*.py"):
                parts = set(p.parts)
                if parts & {".git", "tests", "test", ".venv", "venv", "site-packages", "repro_runs",
                            "node_modules", "__pycache__"} or p.name.startswith("test_"):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                try:
                    tree = ast.parse(p.read_text(errors="replace"))
                except (SyntaxError, ValueError):
                    continue
                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        out.append((p, n.name))
    return out


# The common causal-operator contract declares its operators as a literal
# tuple, OPERATOR_KINDS, dispatched internally; no function is named after a
# kind. The first run of this PRE searched function names for them and
# found none: a harness defect, corrected by reading the declared literal.
CONTRACT_KIND_STEP = {"ewma": "03", "trailing_mean": "03", "trailing_median": "03",
                      "local_level_kalman": "03", "identity": "CONTROL",
                      "centered_mean_oracle": "NON_CAUSAL_ORACLE"}


def contract_kinds() -> tuple[str, ...]:
    """The operator kinds the common contract DECLARES, read from the
    OPERATOR_KINDS literal by AST; the module is not imported."""
    tree = ast.parse(CONTRACT_MODULE.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "OPERATOR_KINDS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    return ()


# One regex per step over DEFINED symbol names (never over prose).
STEP_SYMBOLS = {
    "01": r"nyquist|alias|sampling_(rate|interval|seconds)|jitter|downsampl",
    "02": r"snr|noise_(estimate|std|level|floor)|estimate_noise|signal_to_noise",
    "03": r"denois|ewma|kalman|trailing_median|trailing_mean|smooth",
    "04": r"quantiz|quantis|compand|mu_law|a_law|bin_edges",
    "05": r"entropy|mdl|compress|lempel|kolmogorov|source_cod",
    "06": r"wavelet|stft|spectrogram|hilbert|multitaper|emd|psd|time_freq",
    "07": r"matched_filter|detector|detect_(pattern|event|motif)|cusum|motif",
    "08": r"equaliz|equalis|canonicaliz|domain_align",
    "09": r"crosstalk|echo_cancel|common_private|interference|redundancy_cancel",
    "10": r"lead_lag|timing_recover|synchroni|align_(causal|series)|lag_estimat",
    "11": r"controlled_redundan|channel_cod|repetition_cod",
    "12": r"adaptive_rout|link_adapt|quality_rout|router",
    "13": r"multi_?branch|mimo|budget_alloc|multiplex",
}
STEP_PROTOCOL = {
    "01": ("MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v2.md", r"STEP 01"),
    "02": ("STEP_03_NOISE_SNR_DENOISING_PROTOCOL_FINAL.md", r"SNR"),
}


def protocol_for(step: str):
    if step in STEP_PROTOCOL:
        name, pat = STEP_PROTOCOL[step]
        p = DOCS / name
        return (p.name if p.is_file() and re.search(pat, p.read_text(errors="replace")) else None)
    hits = sorted(DOCS.glob(f"STEP_{step}_*.md"))
    return hits[0].name if hits else None


EXECUTED = {
    "01": [FC / "ETH_H4_SUCCESSOR_TEMPORAL_CONTRACT.v3.json"],
    "02": [HOME / ".local/share/agent-multi/t1_bank_v3_20260906/BANK_INVENTORY.json"],
    "03": [T2 / "docs/audits/evidence/t1_lab_v4_20260906/T1_ADJUDICATION_V4.json",
           T2 / "docs/audits/evidence/T2_C108_HARDENED_READJUDICATION_CLOSURE_2026_09_12.json"],
}
SCOPE = {
    "01": "ONE_PILOT_DATASET (ETH successor temporal contract: gaps and truncations, no Nyquist/aliasing run)",
    "02": "SYNTHETIC_GENERATOR_ONLY (declared SNR of generated noise; no estimator from observed data)",
    "03": "LIMITED_LAB (T1, 5 operators) PLUS ONE NEGATIVE PUBLIC RESULT (T2 DOES_NOT_ADVANCE)",
}


def executed_for(step: str) -> list[str]:
    found = [p for p in EXECUTED.get(step, []) if p.is_file()]
    # A step-tagged result anywhere in the evidence directories would also count.
    for d in (EVID, T2 / "docs/audits/evidence", AM / "docs/audits/evidence", FC):
        if d.is_dir():
            found += [p for p in d.rglob(f"*STEP_{step}*") if p.suffix in (".json", ".parquet", ".csv")]
            found += [p for p in d.rglob(f"*STEP{step}*") if p.suffix in (".json", ".parquet", ".csv")]
    return sorted({redact(p) for p in found})


def eligibility_grants_in_files() -> list[str]:
    """Places where a variable or operator is GRANTED PUBLICLY_ELIGIBLE: a JSON
    value equal to it, or a Python assignment/return of that literal outside
    enumerations and refusals. Guards, enums and disclaimers are not grants."""
    grants = []
    for d in (EVID, T2 / "docs/audits/evidence", AM / "docs/audits/evidence", FC,
              HOME / ".local/share/agent-multi"):
        if not d.is_dir():
            continue
        for p in d.glob("*.json") if d.name == "agent-multi" else d.rglob("*.json"):
            try:
                if p.stat().st_size > 50_000_000:
                    continue
                doc = json.loads(p.read_text(errors="replace"))
            except (ValueError, OSError):
                continue
            stack = [doc]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    stack.extend(node.values())
                elif isinstance(node, list):
                    stack.extend(node)
                elif node in ("PUBLICLY_ELIGIBLE", "LIVE_ELIGIBLE"):
                    grants.append(redact(p))
                    break
    return sorted(set(grants))


def cube_grants() -> dict:
    env = {}
    for line in (HOME / ".config/crispdm/olap-loader.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    from sqlalchemy import create_engine, text
    eng = create_engine(f"postgresql://{env['PGUSER']}:{env['PGPASSWORD']}@{env.get('PGHOST', '127.0.0.1')}:"
                        f"{env.get('PGPORT', '5432')}/{env.get('PGDATABASE', 'predictor_olap')}")
    hits, n, tables = {}, 0, []
    with eng.connect() as c:
        cols = c.execute(text("""SELECT table_schema, table_name, column_name FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog','information_schema')
              AND data_type IN ('text','character varying')
              AND (column_name ILIKE '%state%' OR column_name ILIKE '%eligib%' OR column_name ILIKE '%status%'
                   OR column_name ILIKE '%decision%' OR column_name ILIKE '%class%'
                   OR column_name ILIKE '%adjudication%')""")).all()
        for s, t, col in cols:
            n += 1
            k = c.execute(text(f'SELECT count(*) FROM "{s}"."{t}" WHERE "{col}" IN '
                               "('PUBLICLY_ELIGIBLE', 'LIVE_ELIGIBLE')")).scalar()
            if k:
                hits[f"{s}.{t}.{col}"] = k
        tables = [r[0] for r in c.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1"))]
    eng.dispose()
    return {"columns_scanned": n, "grants": hits, "public_tables": tables}


# ------------------------------------------------------------- matrix
def sec_matrix(index, grants_files, cube):
    kinds = contract_kinds()
    unmapped = [k for k in kinds if k not in CONTRACT_KIND_STEP]
    matrix = {}
    for step, pat in STEP_SYMBOLS.items():
        rx = re.compile(pat, re.IGNORECASE)
        defined = sorted({(redact(p), name) for p, name in index if rx.search(name)})
        executed = executed_for(step)
        matrix[step] = {
            "protocol_present": protocol_for(step),
            "implementation_symbols": len(defined),
            "implementation_files": sorted({f for f, _ in defined})[:8],
            "under_common_operator_contract": sorted(k for k in kinds if CONTRACT_KIND_STEP.get(k) == step),
            "experiment_executed": executed,
            "evidence_scope": SCOPE.get(step, "NONE") if executed else "NONE",
            "eligibility": "PUBLICLY_ELIGIBLE" if (grants_files or cube["grants"]) else "NONE_GRANTED",
        }
    print("\n=== STEP 01-13 status matrix (derived from artifacts) ===")
    print(f"  {'step':<4} {'protocol':<9} {'symbols':>7} {'contract':<9} {'executed':<9} eligibility")
    for s, m in matrix.items():
        print(f"  {s:<4} {'yes' if m['protocol_present'] else 'NO':<9} {m['implementation_symbols']:>7} "
              f"{'yes' if m['under_common_operator_contract'] else 'no':<9} "
              f"{'yes' if m['experiment_executed'] else 'NO':<9} {m['eligibility']}")
    for s in ("01", "02", "03"):
        print(f"  STEP {s} executed: {matrix[s]['experiment_executed']} scope={matrix[s]['evidence_scope']}")
    item("C125.matrix_protocols", "every STEP 01-13 has a protocol document on disk",
         all(m["protocol_present"] for m in matrix.values()),
         {s: m["protocol_present"] for s, m in matrix.items()}, kind="FACT")
    no_exec = [s for s in matrix if int(s) >= 4 and not matrix[s]["experiment_executed"]]
    item("C125.no_executed_matrix_04_13", "no executed experiment exists for STEP 04-13",
         no_exec == [f"{i:02d}" for i in range(4, 14)], {"steps_without_execution": no_exec})
    contract = {s: m["under_common_operator_contract"] for s, m in matrix.items() if m["under_common_operator_contract"]}
    item("C125.common_contract_only_step_03", "only STEP 03 has operators declared by the common contract",
         bool(kinds) and not unmapped and set(contract) == {"03"},
         {"declared_kinds": kinds, "by_step": contract, "unmapped_kinds": unmapped})
    return matrix


# ------------------------------------------------------- required PREs
V5_OPERATOR_STEP = {  # the step each v5 operator's transformation belongs to, as declared by its name
    "O0_IDENTITY": None, "O1_FIRST_DIFFERENCE": None, "O2_LOG_RETURN": None,
    "O3_TRAILING_ZSCORE_64": "04", "O4_TRAILING_RANK_PERCENTILE_256": "04",
    "O5_CAUSAL_EWMA_0.3": "03", "O6_TRAILING_WINSORIZE_512": "04",
}
V5_OPERATOR_CODE = {  # a defined function computing each operator would match these
    "O1_FIRST_DIFFERENCE": r"first_diff|difference", "O2_LOG_RETURN": r"log_return",
    "O3_TRAILING_ZSCORE_64": r"zscore|z_score", "O4_TRAILING_RANK_PERCENTILE_256": r"rank_percentile|percentile_rank",
    "O6_TRAILING_WINSORIZE_512": r"winsor",
}


def sec_required(index, grants_files, cube):
    pop = json.loads((EVID / "PER_VARIABLE_DESIGN_V5_POPULATION.v1.json").read_text())
    item("C125.v5_zero_eligible", "v5 has zero eligible variables",
         pop["eligible_variables"] == 0 and pop["verdict"] == "BANK_INSUFFICIENT"
         and not any(r["member"] for r in pop["ledger"]),
         {"candidates": pop["candidates"], "eligible": pop["eligible_variables"], "verdict": pop["verdict"]})

    t2 = json.loads((T2 / "docs/audits/evidence/T2_READJUDICATION_SUBMISSION_V4_2026_09_12.json").read_text())
    t1 = json.loads((T2 / "docs/audits/evidence/t1_lab_v4_20260906/T1_ADJUDICATION_V4.json").read_text())
    review = AM / "docs/audits/evidence/MUSASHI_T1_V4_EXTERNAL_REVIEW_2026_09_06.json"
    review_doc = json.loads(review.read_text()) if review.is_file() else {}
    item("C125.zero_publicly_eligible_denoisers", "no denoising operator is publicly eligible",
         not grants_files and not cube["grants"]
         and t2["hardened_readjudication"]["verdict"] == "DOES_NOT_ADVANCE" and t2["grants_promotion"] is False,
         {"grants_in_files": grants_files, "grants_in_cube": cube["grants"],
          "cube_columns_scanned": cube["columns_scanned"],
          "t2_verdict": t2["hardened_readjudication"]["verdict"],
          "t1_lab_verdict_counts": t1.get("verdict_counts"),
          "t1_review": review_doc.get("decision") or review_doc.get("disposition") or "see file"})

    steps_with_operator = {s for s in V5_OPERATOR_STEP.values() if s}
    code_hits = {}
    for op, pat in V5_OPERATOR_CODE.items():
        rx = re.compile(pat, re.IGNORECASE)
        code_hits[op] = sorted({f"{redact(p)}:{n}" for p, n in index if rx.search(n)})[:4]
    design = json.loads((EVID / "PER_VARIABLE_PREPROCESSING_DESIGN.v5.json").read_text())
    item("C125.v5_operators_do_not_represent_13_steps",
         "the seven v5 operators touch at most two of the thirteen steps",
         design["operators"] == list(V5_OPERATOR_STEP) and len(steps_with_operator) <= 2,
         {"steps_with_a_v5_operator": sorted(steps_with_operator),
          "steps_without": [f"{i:02d}" for i in range(1, 14) if f"{i:02d}" not in steps_with_operator],
          "code_defining_operator_like_functions": code_hits})

    disp = json.loads((EVID / "TERMINALS_V4_DISPOSITION.v1.json").read_text())
    sc = {k: v["count"] for k, v in disp["scopes"].items()}
    v5 = load(PRED / "tools/per_variable_design_v5.py", "v5_pre_c125")
    sys.path.insert(0, str(PRED / "tests"))
    import population_join_fixture as FX
    sys.path.remove(str(PRED / "tests"))
    with tempfile.TemporaryDirectory() as td:
        kw, _ = FX.write_bank(Path(td), "C113.recomputed_license_unknown")
        popx = v5.derive_population(terminals_dir=kw["terminals_v4"], dag=kw["dag_v4"], census=kw["census"],
                                    temporal_contracts=kw["temporal_contracts"])
    row = next(r for r in popx["ledger"] if r["dataset_id"] == "panel_0" and r["column"] == "x_0")
    item("C125.statistic_grants_no_semantics_or_license",
         "a recomputed numeric statistic grants neither semantics nor license",
         sc["NUMERIC_DESCRIPTORS_RECOMPUTED"] == 1501 and sc["SEMANTIC_DECLARATIONS_KNOWN"] == 0
         and row["conditions"]["terminal_independently_recomputed"] and not row["member"],
         {"scopes": sc, "fixture_row_reasons": row["reasons"]})


# ------------------------------------------------------------ D0-D2 gaps
def sec_baseline(index, cube):
    bank = load(T2 / "tools/t1_known_truth_bank.py", "t1_bank_pre")

    def absent(n, title, pattern, detail_extra=None):
        rx = re.compile(pattern, re.IGNORECASE)
        found = sorted({f"{redact(p)}:{s}" for p, s in index if rx.search(s)})[:6]
        item(n, title, not found, {"defined": found, **(detail_extra or {})})

    absent("C129.no_common_dataset_contract", "no common dataset/variable contract module",
           r"^(DatasetContract|VariableContract|dataset_contract|variable_contract)$")
    item("C128.synthetic_bank_gaps", "the known-truth bank lacks trend, seasonality, steps, motifs, missing blocks and null controls",
         not ({"trend", "seasonal", "step", "motif", "null_signal", "null_noise"} & set(bank.FAMILIES))
         and "missing_block" not in bank.PERTURBATIONS,
         {"families": bank.FAMILIES, "perturbations": bank.PERTURBATIONS})
    absent("C133.no_nyquist_or_aliasing_code", "no Nyquist or aliasing analysis exists", r"nyquist|alias")
    absent("C134.no_snr_estimator_from_data", "no SNR/noise estimator from observed data exists",
           r"estimate_(snr|noise)|snr_estimat|noise_estimat|mad_sigma|wavelet_noise")
    # The first run wrote `families & tokens | kinds`, which Python reads as
    # `(families & tokens) | kinds`: never empty, so the true absence printed
    # NOT REPRODUCED. A harness defect, corrected with explicit sets.
    kinds = contract_kinds()
    tokens = set(kinds) | {tok for k in kinds for tok in k.split("_")}
    families = {"fir", "iir", "butterworth", "savgol", "wavelet", "stl", "trend_seasonal", "hampel"}
    item("C136.operator_families_missing", "the operator contract has no FIR/IIR, wavelet, decomposition or robust families",
         bool(kinds) and not (families & tokens),
         {"declared_kinds": kinds, "families_found": sorted(families & tokens)})
    absent("C135.no_delay_measurement", "no group-delay / algorithmic-delay measurement exists",
           r"group_delay|phase_delay|algorithmic_delay|measure_delay|impulse_response")
    absent("C131.no_permutation_or_spectral_entropy", "no permutation or spectral entropy estimator exists",
           r"permutation_entropy|spectral_entropy")
    wanted = {"fact_sampling_quality", "fact_variable_profile_partition", "fact_information_metric",
              "fact_pair_relation", "fact_operator_run", "fact_operator_signal_metric", "fact_lab_decision"}
    item("C139.no_data_foundation_tables", "the cube has none of the data-foundation grains",
         not (wanted & set(cube["public_tables"])), {"public_tables": len(cube["public_tables"])})
    d3 = [p.name for p in EVID.glob("*D3*DESIGN*")] + [p.name for p in EVID.glob("*D4*DESIGN*")] \
        + [p.name for p in EVID.glob("*D5*DESIGN*")]
    item("C141_C143.no_successor_designs", "no D3, D4 or D5 design document exists", not d3, d3)
    census = next(FC.glob("artifacts/census-49a8813d*.json"))
    cen = json.loads(census.read_text())
    v0 = cen["variables"][0]
    item("C127.census_has_no_producer_family_consumer", "the 1,965-variable census has no producer, family or consumer field",
         len(cen["variables"]) == 1965 and not ({"producer", "feature_family", "consumer"} & set(v0)),
         {"variable_keys": sorted(v0)})


def identities() -> dict:
    ids = {}
    groups = {
        T2 / "docs/audits/evidence": ("T2_*SUBMISSION*.json", "T2_C108_*.json", "T2_*DIVERGENCE*.json",
                                      "T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_ADJUDICATION_2026_09_10.json"),
        B4 / "docs/audits/evidence": ("B4_*SUBMISSION*.json",),
        B4 / "docs/audits/work_plan": ("B4_V4_CLOSURE_DISPOSITION_2026_09_12.md",),
        EVID: ("PER_VARIABLE*DESIGN*.json", "TERMINALS_V4_DISPOSITION.v1.json", "PANEL_INVENTORY.v1.json",
               "LAKE_TERMINAL_VERIFICATION.v*.json", "TERMINAL_SUPERSESSION.v*.json"),
        FC: ("FEATURE_DAG.v*.json", "PRODUCER_BINDING_MANIFEST.v*.json", "*TEMPORAL_CONTRACT*.json",
             "*ELIGIBILITY_MASK*.parquet", "ETH_H4_SUCCESSOR_*.json"),
    }
    for base, pats in groups.items():
        for pat in pats:
            for p in sorted(base.glob(pat)):
                ids[f"{base.name}:{p.name}"] = sha(p)
    lake = HOME / ".local/state/crispdm/lake_characterization"
    for v in ("terminals", "terminals_v2", "terminals_v3", "terminals_v4"):
        h = hashlib.sha256()
        for p in sorted((lake / v).glob("*.json")):
            h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
        ids[f"lake:{v}"] = h.hexdigest()
    ch = HOME / ".local/state/crispdm-successors/eth_h4_stage22_rerun_v1_characterization_v1/terminals"
    h = hashlib.sha256()
    for p in sorted(ch.glob("*.json")):
        h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    ids["successor:characterization_terminals"] = h.hexdigest()
    return ids


def main() -> int:
    print(f"PRE C122-C145 python={sys.version.split()[0]}")
    sec_bases()
    before = identities()
    index = code_index()
    print(f"  code index: {len(index)} defined symbols")
    grants_files = eligibility_grants_in_files()
    cube = cube_grants()
    sec_matrix(index, grants_files, cube)
    sec_required(index, grants_files, cube)
    sec_baseline(index, cube)
    after = identities()
    print("\n=== identities (before == after) ===")
    for k in before:
        print(f"  {k}: {before[k][:16]} unchanged={before[k] == after.get(k)}")
    absences = [n for n, kind, ok in RESULTS if kind == "ABSENCE"]
    missing = [n for n, kind, ok in RESULTS if kind == "ABSENCE" and not ok]
    facts_differ = [n for n, kind, ok in RESULTS if kind == "FACT" and not ok]
    changed = [k for k in before if before[k] != after.get(k)]
    print(f"\n=== PRE SUMMARY: {len(absences)} absences/defects, {len(absences) - len(missing)} reproduced, "
          f"not reproduced {missing}; facts differing {facts_differ}; identities changed {changed} ===")
    return 0 if not (missing or facts_differ or changed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
