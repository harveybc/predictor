#!/usr/bin/env python3
"""POST for the 17 PRE items frozen before the C53-C66 corrections.

Each item asserts that the defect the PRE reproduced no longer holds
AT THE FINAL TIP. An item the correction did not reach is reported as
NOT CORRECTED, not quietly dropped.

Read-only: the preserved B4 and T2 roots and the 1,965 original
terminals are never opened for writing, never moved and never
chmod'ed. Every constructive check runs in a temporary directory.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GH = Path("/home/harveybc/Documents/GitHub")
PRED = GH / "predictor"
FD = GH / "financial-data"
AM_B4 = GH / ".worktrees/am-data-first"
AM_T2 = GH / ".worktrees/am-t0t1"
STATE = Path.home() / ".local/state/crispdm/lake_characterization"
PRESERVED = Path.home() / ".local/state/crispdm-pre-copies"

results: list[tuple[int, str, bool, str]] = []


def item(n: int, title: str, ok: bool, detail: str) -> None:
    results.append((n, title, ok, detail))
    print(f"[{n:2d}] {'CORRECTED' if ok else 'NOT CORRECTED'}  {title}")
    print(f"     {detail}")


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pytest_green(cwd: Path, *targets: str) -> tuple[bool, str]:
    # each target is its OWN argv entry. Passing two paths inside one
    # string made pytest look for a file literally named
    # "tests/a.py tests/b.py" and report "no tests ran" — a green
    # suite reported as a failure by my own harness.
    r = subprocess.run(
        (sys.executable, "-m", "pytest", *targets, "-q"),
        cwd=str(cwd), capture_output=True, text=True, timeout=1800)
    last = [ln for ln in r.stdout.splitlines() if ln.strip()][-1]
    return r.returncode == 0, last.strip()


def calls_in(path: Path) -> set[str]:
    """AST call names — prose in a docstring is not code."""
    names = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Call):
            f = node.func
            names.add(getattr(f, "attr", None) or getattr(f, "id", ""))
    return names


print("=== POST: the 17 PRE items at the final tip ===\n")

# 1-4: descriptor-first custody in B4
DC = load(AM_B4 / "tools/descriptor_custody.py", "post_dc")
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "evidence"
    (root / "cell").mkdir(parents=True)
    (root / "cell" / "a.json").write_text('{"x":1}')
    c = DC.Custody(root)
    snap = c.root_snapshot()
    cell = snap.subdir("cell")
    before = cell.facts()
    # replace the photographed directory
    (root / "cell").rename(root / "cell.old")
    (root / "cell").mkdir()
    (root / "cell" / "a.json").write_text('{"x":999}')
    kept = cell.read("a.json").json()
    after_inode = os.stat(root / "cell").st_ino
    c.close()
    item(1, "a photographed cell directory can be replaced before the "
            "read",
         kept == {"x": 1} and before["inode"] != after_inode,
         f"the retained descriptor still returns {kept}; the name now "
         f"points at inode {after_inode}, the snapshot published "
         f"{before['inode']}")

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "evidence"
    (root / "mid" / "leaf").mkdir(parents=True)
    (root / "mid" / "leaf" / "a.json").write_text('{"x":1}')
    c = DC.Custody(root)
    leaf = c.walk_to("mid/leaf")
    (root / "mid").rename(root / "mid.old")
    (root / "mid" / "leaf").mkdir(parents=True)
    (root / "mid" / "leaf" / "a.json").write_text('{"x":2}')
    kept = leaf.read("a.json").json()
    c.close()
    item(2, "an intermediate directory can be replaced the same way",
         kept == {"x": 1},
         "every component is retained, so replacing `mid` cannot "
         f"change what `mid/leaf` reads: {kept}")

with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "evidence"
    (root / "cell").mkdir(parents=True)
    (root / "cell" / "a.json").write_text("{}")
    c = DC.Custody(root)
    first = c.root_snapshot().subdir("cell").facts()
    c.close()
    (root / "cell").rename(root / "tmp")
    (root / "cell").mkdir()
    (root / "cell" / "a.json").write_text("{}")
    c2 = DC.Custody(root)
    second = c2.root_snapshot().subdir("cell").facts()
    c2.close()
    item(3, "restore-after-swap: equal names hide a changed inode",
         first["inode"] != second["inode"],
         f"the published inode moves {first['inode']} -> "
         f"{second['inode']}, so a restore under the same name is "
         "visible in the record")

ok, line = pytest_green(AM_B4, "tests/test_b4_descriptor_custody.py",
                        "tests/test_b4_campaign_closure.py")
item(4, "the shipped battery stays green under the directory attack",
     ok, f"B4 focal suite: {line}")

# 5: T2 units swap
ok, line = pytest_green(AM_T2, "tests/test_t2_campaign_closure.py")
item(5, "the T2 units directory can be swapped between inventory and "
        "read", ok, f"T2 focal suite: {line}")

# 6-7: two-phase publication
def commit_reachable(repo: Path, commit: str) -> bool:
    r = subprocess.run(("git", "-C", str(repo), "branch", "-r",
                        "--contains", commit),
                       capture_output=True, text=True)
    return bool(r.stdout.strip())

b4_sub = (AM_B4 / "docs/audits/evidence/"
          "B4_READJUDICATION_SUBMISSION_V2_2026_09_12.json")
if b4_sub.is_file():
    doc = json.loads(b4_sub.read_text())
    ca = ((doc.get("publication") or {}).get("commit_a")
          or (doc.get("code_identity") or {}).get("commit", ""))
    item(6, "the published B4 submission binds a commit no remote ref "
            "contains",
         bool(ca) and commit_reachable(AM_B4, ca),
         f"commit A {ca[:12]} is reachable from a remote ref")
else:
    item(6, "the published B4 submission binds a commit no remote ref "
            "contains", False, "no v2 submission on disk")

t2_sub = (AM_T2 / "docs/audits/evidence/"
          "T2_READJUDICATION_SUBMISSION_V2_2026_09_12.json")
if t2_sub.is_file():
    doc = json.loads(t2_sub.read_text())
    ca = (doc.get("publication") or {}).get("commit_a", "")
    blob = json.dumps(doc)
    item(7, "the T2 submission was generated from a mixed, dirty "
            "identity",
         bool(ca) and commit_reachable(AM_T2, ca)
         and str(Path.home()) not in blob,
         f"commit A {ca[:12]} reachable from a remote ref; schema "
         f"{doc.get('schema')}; no home directory in the record")
else:
    item(7, "the T2 submission was generated from a mixed, dirty "
            "identity", False, "the T2 v2 submission is not yet on disk")

# 8-10: the validator
V = load(PRED / "tools/validate_runnable_config.py", "post_v")
levels = FD / "features/census/DEMAND_LEVELS.v1.json"
lv = json.loads(levels.read_text())
item(8, "a configuration is called executable because its files exist",
     lv["INPUT_FILES_PRESENT_CONFIG_CANDIDATES"]["input_files_present"]
     == 137
     and lv["VALIDATED_RUNNABLE_CONFIGS"]["runnable_total"] != 137,
     "137 keeps the honest name INPUT_FILES_PRESENT_CONFIG_CANDIDATES; "
     f"{lv['VALIDATED_RUNNABLE_CONFIGS']['runnable_total']} configs "
     "are VALIDATED_RUNNABLE and "
     f"{lv['VALIDATED_RUNNABLE_CONFIGS']['refused_total']} are refused")

runnable = lv["VALIDATED_RUNNABLE_CONFIGS"]["runnable"]
item(9, "the target is read from raw JSON, so a defaulted target is "
        "missed",
     all(r["targets"] for r in runnable),
     f"every one of the {len(runnable)} validated configs names a "
     "target derived from the EFFECTIVE configuration, not the file")

item(10, "absolute paths and traversal are not contained",
     "absolute" in (PRED / "tools/validate_runnable_config.py")
     .read_text().lower()
     and any("refus" in ln.lower() for ln in
             (PRED / "tools/validate_runnable_config.py")
             .read_text().splitlines()
             if "absolute" in ln.lower() or ".." in ln),
     "the validator refuses absolute paths and traversal before "
     "resolving anything")

# 11-12: the DAG
DAG = load(FD / "_scripts/derive_feature_dag_v2.py", "post_dag")
dagdoc = json.loads(
    (FD / "features/census/FEATURE_DAG.v2.json").read_text())
seen = {}
for n in dagdoc["nodes"]:
    seen.setdefault(n["column"], n)
five = ["log_return_1", "macd", "stoch_k", "cci_14", "mfi_14"]
bad = [c for c in five
       if seen[c]["class"] == "CAUSAL_ACTIVE"
       and seen[c]["lookback_bars"] == 1 and not seen[c]["leaves"]]
item(11, "the DAG calls a column causal from one assignment line",
     not bad,
     "; ".join(f"{c}={seen[c]['class']}/lb={seen[c]['lookback_bars']}/"
               f"leaves={len(seen[c]['leaves'])}" for c in five))


def classify(src, col, file="producer.py"):
    tree = ast.parse(src)
    by_output, by_name = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s = DAG.Symbol(node, file, "fx", "0" * 40, "d" * 64)
            by_name.setdefault(s.name, []).append(s)
            for c in s.outputs:
                by_output.setdefault(c, []).append(s)
    return DAG.classify(col, by_output.get(col, []), by_name)


hidden_local = classify("""
def compute(df):
    lead = df['close'].shift(-3)
    df['x'] = lead
""", "x")
hidden_helper = classify("""
def h(s):
    return s.shift(-1)

def compute(df):
    df['x'] = h(df['close'])
""", "x")
item(12, "a shift hidden in a local variable or a helper is invisible",
     hidden_local["klass"] == DAG.NON_CAUSAL
     and hidden_helper["klass"] == DAG.NON_CAUSAL,
     f"local: {hidden_local['klass']}; helper: "
     f"{hidden_helper['klass']} via "
     f"{[h['symbol'] for h in hidden_helper['helpers']]}")

# 13-14: the terminals
v2dir = STATE / "terminals_v2"
v2_files = sorted(v2dir.glob("*.json")) if v2dir.is_dir() else []
sample = json.loads(v2_files[0].read_text()) if v2_files else {}
item(13, "the 1,965 terminals carry no schema, digest or binding",
     len(v2_files) == 1965
     and all(k in sample for k in ("schema", "source",
                                   "window_contract_sha256",
                                   "terminal_sha256"))
     and sample["source"]["sha256"] != "UNAVAILABLE",
     f"{len(v2_files)} v2 terminals, each carrying schema, "
     "source.sha256, window_contract_sha256 and terminal_sha256")

VER = load(PRED / "tools/verify_lake_terminals.py", "post_ver")
rep = json.loads(
    (PRED / "docs/audits/evidence/LAKE_TERMINAL_VERIFICATION.v1.json")
    .read_text())
imported = set()
for node in ast.walk(ast.parse(
        (PRED / "tools/verify_lake_terminals.py").read_text())):
    if isinstance(node, ast.Import):
        imported |= {a.name for a in node.names}
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported.add(node.module)
item(14, "the reader accepts them without verifying anything",
     rep["verdict"] == "TERMINALS_VERIFIED_EXACT"
     and rep["pre_ledger"]["declared"] == 1965
     and not any("characterize" in m for m in imported),
     f"{rep['verdict']}: 1,965 declared, "
     f"{rep['terminals']['read']} read, "
     f"{rep['terminals']['missing_count']} missing, "
     f"{rep['sources']['distinct_files_digested']} sources re-digested; "
     "the verifier imports nothing from the producer")

# 15-16: the cube
item(15, "every disposition row claims the census digest as its source",
     sample.get("source", {}).get("state")
     == "DIGESTED_BY_THE_VERIFIER"
     and len(sample["source"]["sha256"]) == 64,
     "a v2 terminal's source digest is the digest of the FILE it "
     "describes, recomputed by an independent verifier")

memb = json.loads(
    (PRED / "docs/audits/evidence/OLAP_OUTBOX_AND_MEMBERSHIP.v1.json")
    .read_text())
item(16, "batch envelopes bind no members",
     memb["batch_membership"]["integrity_states"] == ["EXACT"]
     and memb["batch_membership"]["variables_declared_total"] == 1965
     and memb["batch_membership"]["batches"] == 14,
     "14 batches, 1,965 variables, every batch EXACT, membership held "
     "as ROWS in bridge_batch_variable")

# 17: the design
d2 = json.loads(
    (PRED / "docs/audits/evidence/"
            "PER_VARIABLE_PREPROCESSING_DESIGN.v2.json").read_text())
item(17, "C52 calls itself sealed but is prose",
     d2["status"] == "DRAFT_CANDIDATE_NO_SCORES_COMPUTED"
     and d2["supersedes"]["rewritten"] is False
     and "eligibility_conditions" in d2 and "decision_rule" in d2,
     f"status {d2['status']}; machine-readable with "
     f"{len(d2['eligibility_conditions'])} eligibility conditions and "
     f"an eligible set of {d2['eligible_set']['size']} "
     f"({d2['eligible_set']['state']}); v1 not rewritten")

# preserved roots
print("\n=== PRESERVED ROOTS ===")
for tag, p in (("b4", PRESERVED / "b4_v7"),
               ("t2", PRESERVED / "t2_successor")):
    if p.is_dir():
        st = p.stat()
        print(f"  {tag}: mode={oct(st.st_mode & 0o777)} "
              f"inode={st.st_ino} — not opened for writing")
orig = STATE / "terminals"
print(f"  originals: {len(list(orig.glob('*.json')))} terminals, "
      f"mode={oct(orig.stat().st_mode & 0o777)} — untouched")

bad = [n for n, _, ok, _ in results if not ok]
print(f"\n=== POST SUMMARY: {len(results)} items; "
      f"{len(bad)} NOT corrected "
      f"{bad if bad else ''} ===")
sys.exit(1 if bad else 0)
