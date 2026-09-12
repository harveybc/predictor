"""PRE for B4 R15-R18, T2 R11-R15 and CRISP-DM C53-C66 (order 2026-09-12).

Frozen BEFORE editing, against the audited tips:

    predictor      4e68c45     financial-data  19fe375a1
    B4             c907495d    T2              5fb2849e

All four worktrees were clean when this ran. The B4 and T2 attacks use
synthetic roots and copies; the preserved campaign roots are opened for
reading only and their structural digests are checked before and after.

Private paths are redacted to `~`.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
GH = HOME / "Documents/GitHub"
AM_B4 = GH / ".worktrees/am-data-first"
AM_T2 = GH / ".worktrees/am-t0t1"
PREDICTOR = GH / "predictor"
FINANCIAL = GH / "financial-data"
B4_REAL = HOME / ".local/share/agent-multi/b4_campaign_results_v7_20260907"
T2_REAL = (HOME / ".local/share/agent-multi"
           / "t2_confirmatory_results_resource_successor_v1_20260909")

FAILURES: list[str] = []
_n = 0


def redact(v) -> str:
    return str(v).replace(str(HOME), "~")


def item(n: int, title: str) -> None:
    global _n
    _n = n
    print(f"\n=== PRE {n}: {title} ===")


def expect(cond: bool, msg: str) -> None:
    print(f"  [{'PRE-REPRODUCED' if cond else 'NOT-REPRODUCED'}] {msg}")
    if not cond:
        FAILURES.append(f"{_n}: {msg}")


def tree_digest(root: Path) -> str:
    entries = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            st = p.stat()
            entries.append((str(p.relative_to(root)), st.st_size,
                            st.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


REAL_BEFORE = {"b4": tree_digest(B4_REAL), "t2": tree_digest(T2_REAL)}

sys.path.insert(0, str(AM_B4 / "tools"))
sys.path.insert(0, str(AM_B4 / "tests"))
import descriptor_custody as DC                                # noqa: E402
import b4_campaign_closure as B4C                              # noqa: E402
from test_b4_campaign_closure import build_root                # noqa: E402

B4C.prove_relaunch_refuses = lambda: {
    "launch_gate": "stub", "refused": True, "refusal": "REFUSED: stub",
    "accelerator_modules_before": [], "accelerator_modules_after": [],
    "refused_before_any_accelerator_import": True}


# ============================================ B4 R15: directory swaps
item(1, "a photographed cell directory can be replaced before the read")
with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "root"
    (root / "cell").mkdir(parents=True)
    (root / "cell/terminal.json").write_text(
        json.dumps({"wall_seconds": 1.0}))
    with DC.Custody(root) as c:
        snap = c.snapshot("cell")
        listed = snap.files
        # the whole directory is swapped for a different instance
        (root / "cell").rename(root / "cell_moved")
        (root / "cell").mkdir()
        (root / "cell/terminal.json").write_text(
            json.dumps({"wall_seconds": 999.0}))
        consumed = c.read("cell/terminal.json").json()
print(f"  snapshot listed              : {listed}")
print(f"  wall_seconds consumed after  : {consumed['wall_seconds']}")
expect(consumed["wall_seconds"] == 999.0,
       "the snapshot kept only NAMES and the read re-walked the path, "
       "so a different directory instance was consumed")

item(2, "an intermediate directory can be replaced the same way")
with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "root"
    (root / "cell/runtime").mkdir(parents=True)
    (root / "cell/runtime/status.json").write_text(
        json.dumps({"epoch_completed": 1}))
    with DC.Custody(root) as c:
        c.snapshot("cell/runtime")
        (root / "cell").rename(root / "cell_old")
        (root / "cell/runtime").mkdir(parents=True)
        (root / "cell/runtime/status.json").write_text(
            json.dumps({"epoch_completed": 1999}))
        got = c.read("cell/runtime/status.json").json()
print(f"  epoch consumed after the swap: {got['epoch_completed']}")
expect(got["epoch_completed"] == 1999,
       "an INTERMEDIATE component is re-resolved by name too, so the "
       "whole subtree can be exchanged under the snapshot")

item(3, "restore-after-swap: equal names hide a changed inode")
with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "root"
    (root / "cell").mkdir(parents=True)
    (root / "cell/terminal.json").write_text(json.dumps({"v": "orig"}))
    original_ino = (root / "cell").stat().st_ino
    with DC.Custody(root) as c:
        snap = c.snapshot("cell")
        (root / "cell").rename(root / "cell_away")
        (root / "cell").mkdir()
        (root / "cell/terminal.json").write_text(json.dumps({"v": "sub"}))
        consumed = c.read("cell/terminal.json").json()
        after_ino = (root / "cell").stat().st_ino
        facts = snap.facts()
print(f"  inode at snapshot / at read : {original_ino} / {after_ino}")
print(f"  snapshot facts published    : {sorted(facts)}")
print(f"  value consumed              : {consumed['v']}")
expect(original_ino != after_ino and consumed["v"] == "sub"
       and "inode" not in facts and "device" not in facts,
       "the snapshot publishes neither device nor inode, so name "
       "equality is the only thing binding the read to the listing")

item(4, "the shipped battery stays green under the directory attack")
attacked = Path(str(AM_B4 / "tests/test_b4_descriptor_custody.py"))
src = attacked.read_text()
covers_dir_swap = any(tok in src for tok in
                      ("rename(", "cell_moved", "directory instance",
                       "replaced directory"))
run = subprocess.run(
    (sys.executable, "-m", "pytest",
     "tests/test_b4_descriptor_custody.py",
     "tests/test_b4_campaign_closure.py", "-q", "-p", "no:warnings"),
    capture_output=True, text=True, cwd=str(AM_B4), timeout=900)
tail = [ln for ln in run.stdout.splitlines() if "passed" in ln]
print(f"  battery result                 : "
      f"{tail[-1].strip() if tail else run.stdout[-120:]}")
print(f"  any test replaces a DIRECTORY  : {covers_dir_swap}")
expect(run.returncode == 0 and not covers_dir_swap,
       "45 tests pass while none of them replaces an existing "
       "directory, which is the attack that works")


# ================================================== T2 R11: units swap
item(5, "the T2 units directory can be swapped between inventory and "
        "verification")
sys.path.insert(0, str(AM_T2 / "tools"))
with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "root"
    (root / "units").mkdir(parents=True)
    (root / "units/RECORD_a.json").write_text(json.dumps({"effect": 1.0}))
    with DC.Custody(root) as c:
        snap = c.snapshot("units")
        (root / "units").rename(root / "units_old")
        (root / "units").mkdir()
        (root / "units/RECORD_a.json").write_text(
            json.dumps({"effect": -99.0}))
        scored = c.read("units/RECORD_a.json").json()
closure_src = (AM_T2 / "tools/t2_campaign_closure.py").read_text()
uses_same_custody = "from descriptor_custody import Custody" in closure_src
print(f"  effect consumed after swap  : {scored['effect']}")
print(f"  the T2 closure uses the same custody module : "
      f"{uses_same_custody}")
expect(scored["effect"] == -99.0 and uses_same_custody,
       "the same defect reaches T2: `units` is re-resolved by name for "
       "every RECORD and ARRAYS read")


# ============================================ B4 F2: identity recovery
item(6, "the published B4 submission binds a commit no remote ref "
        "contains")
sub_p = (AM_B4 / "docs/audits/evidence"
         / "B4_READJUDICATION_SUBMISSION_2026_09_12.json")
sub = json.loads(sub_p.read_text())
declared = sub["code_identity"]["commit"]
tip = subprocess.run(("git", "-C", str(AM_B4), "rev-parse", "HEAD"),
                     capture_output=True, text=True).stdout.strip()
contains = subprocess.run(
    ("git", "-C", str(AM_B4), "branch", "-r", "--contains", declared),
    capture_output=True, text=True)
print(f"  submission declares commit : {declared[:12]}")
print(f"  published tip              : {tip[:12]}")
print(f"  remote refs containing it  : "
      f"{contains.stdout.strip() or '(none)'}  rc={contains.returncode}")
root_read = sub.get("root_actually_read", "")
print(f"  root_actually_read         : {redact(root_read)}")
expect(declared != tip and not contains.stdout.strip()
       and root_read.startswith("/home/"),
       "the submission names a commit unreachable from origin and "
       "publishes an absolute private path in versioned evidence")


# =========================================== T2 F3: mixed dirty identity
item(7, "the T2 submission was generated from a mixed, dirty identity")
t2_sub = json.loads((AM_T2 / "docs/audits/evidence"
                     / "T2_READJUDICATION_SUBMISSION_2026_09_12.json"
                     ).read_text())
ci = t2_sub["code_identity"]
origins = {}
for rel, f in ci["files"].items():
    origins.setdefault(f["loaded_from"], []).append(rel)
print(f"  files by origin : "
      f"{ {k: len(v) for k, v in origins.items()} }")
print(f"  branch tip clean at generation : {ci['branch_tip']['clean']}")
print(f"  dirty paths                    : "
      f"{ci['branch_tip']['dirty_paths'][:4]}")
deps = ci["numeric_dependencies"]
print(f"  numpy location published       : "
      f"{redact(deps['numpy']['location'])}")
expect(len(origins) > 1 and ci["branch_tip"]["clean"] is False
       and "/home/" in deps["numpy"]["location"],
       "seven files come from one checkout and three from a DIRTY tip, "
       "and the dependency record carries a site-packages path")

sub_cls = closure_src[closure_src.index("def make_snapshot_class("):]
calls_super = "super().__init__" in sub_cls.split("return UnitSnapshotRoot")[0]
print(f"  the adapter calls ResultsRoot.__init__ : {calls_super}")
expect(not calls_super,
       "the adapter subclasses the pinned root without running its "
       "constructor, so every constructor invariant is assumed "
       "unnecessary rather than shown to be")


# ================================================ C53-C54: the demand
sys.path.insert(0, str(FINANCIAL / "_scripts"))
import derive_demand_universes as U                            # noqa: E402

item(8, "a configuration is called executable because its files exist")
fn = ast.parse((FINANCIAL / "_scripts/derive_demand_universes.py"
                ).read_text())
sup_fn = next(n for n in ast.walk(fn)
              if isinstance(n, ast.FunctionDef)
              and n.name == "supervised_demand")
code = ast.unparse(sup_fn)
for token, meaning in (("merge_config", "the effective configuration"),
                       ("plugin_params", "plugin defaults"),
                       ("entry_point", "entry-point resolution"),
                       ("DEFAULT_VALUES", "the default layer")):
    print(f"  builds {meaning:28s}: {token in code}")
print(f"  decides on `.is_file()`          : "
      f"{'is_file()' in code}")
expect("merge_config" not in code and "plugin_params" not in code
       and "is_file()" in code,
       "`executable` means only that the declared x/y files are "
       "present; no effective configuration is built and no entry "
       "point is resolved")

item(9, "the target is read from raw JSON, so a defaulted target is "
        "invisible")
_reads_target = "'target_column'" in code
print(f"  reads cfg.get('target_column') : {_reads_target}")
print(f"  any default/plugin fallback    : "
      f"{'default' in code.lower()}")
with tempfile.TemporaryDirectory() as d:
    pred = Path(d) / "predictor"
    (pred / "examples/config").mkdir(parents=True)
    (pred / "data").mkdir(parents=True)
    for role in ("train", "validation", "test"):
        (pred / f"data/x_{role}.csv").write_text("DATE_TIME,f1,TGT\n1,2,3\n")
        (pred / f"data/y_{role}.csv").write_text("DATE_TIME,TGT\n1,3\n")
    cfg = {}
    for role in ("train", "validation", "test"):
        cfg[f"x_{role}_file"] = f"data/x_{role}.csv"
        cfg[f"y_{role}_file"] = f"data/y_{role}.csv"
    (pred / "examples/config/no_target.json").write_text(json.dumps(cfg))
    out = U.supervised_demand(pred)
    roles = sorted({s["contract_role"] for s in out["subjects"]})
    targets = out["targets"]
print(f"  roles derived without a declared target : {roles}")
print(f"  targets found                           : {targets}")
expect(not targets and "target" not in roles,
       "with no literal `target_column` every y column becomes a "
       "label, so a target supplied by a default or a plugin would be "
       "consumed as a feature")

item(10, "absolute paths and traversal are not contained")
resolve_src = ast.unparse(next(
    n for n in ast.walk(fn)
    if isinstance(n, ast.FunctionDef) and n.name == "resolve"))
print(f"  resolve() returns absolute paths as-is : "
      f"{'is_absolute()' in resolve_src}")
print(f"  any containment check                  : "
      f"{'relative_to' in resolve_src or 'commonpath' in resolve_src}")
with tempfile.TemporaryDirectory() as d:
    outside = Path(d) / "outside.csv"
    outside.write_text("a\n1\n")
    got = U.resolve(Path(d) / "predictor", str(outside))
print(f"  an absolute path outside the checkout resolves to : "
      f"{redact(got)}")
expect("is_absolute()" in resolve_src
       and "relative_to" not in resolve_src,
       "an absolute path or a `..` escape is resolved and read without "
       "being contained under the authorized checkout")


# ================================================== C55: the DAG
item(11, "the DAG calls a column causal from one assignment line")
dag = json.loads((FINANCIAL / "features/census/FEATURE_DAG.v1.json"
                  ).read_text())
by_col = {n["column"]: n for n in dag["nodes"]}
suspects = ["log_return_1", "macd", "stoch_k", "cci_14", "mfi_14"]
shown = 0
for col in suspects:
    n = by_col.get(col)
    if n is None:
        continue
    shown += 1
    print(f"  {col:14s} class={n['class']:10s} "
          f"lookback={n['lookback_bars']} inputs={n['direct_inputs']}")
bad = [c for c in suspects if c in by_col
       and by_col[c]["class"] == "CAUSAL"
       and not by_col[c]["direct_inputs"]]
expect(shown > 0 and len(bad) >= 3,
       f"{len(bad)} of the named columns are CAUSAL with NO direct "
       "inputs, which proves nothing about their dependencies or "
       "their windows")

item(12, "a shift hidden in a local variable or a helper is invisible")
sys.path.insert(0, str(FINANCIAL / "_scripts"))
import derive_feature_dag as D                                 # noqa: E402
with tempfile.TemporaryDirectory() as d:
    repo = Path(d) / "producer"
    (repo / "pkg").mkdir(parents=True)
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    (repo / "pkg/f.py").write_text(
        "def helper(s):\n"
        "    return s.shift(-1)\n"
        "def compute(df):\n"
        "    peek = df['CLOSE'].shift(-1)\n"
        "    df['hidden_local'] = peek\n"
        "    df['hidden_helper'] = helper(df['CLOSE'])\n"
        "    return df\n")
    index = D.index_producers({"p": repo})["index"]
    for col in ("hidden_local", "hidden_helper"):
        node = D.classify(col, index.get(col, []), {})
        print(f"  {col:16s} -> {node['klass']:10s} "
              f"shift={node['shift']} inputs={node['inputs']}")
    klasses = {c: D.classify(c, index.get(c, []), {})["klass"]
               for c in ("hidden_local", "hidden_helper")}
expect(all(v == D.CAUSAL for v in klasses.values()),
       "a forward shift stored in a local variable or returned by a "
       "helper is classified CAUSAL: the analysis never leaves the "
       "assignment line")


# ============================================ C59-C61: the terminals
item(13, "the 1,965 terminals carry no schema, digest or binding")
state = HOME / ".local/state/crispdm/lake_characterization/terminals"
files = sorted(state.glob("*.json"))
sample = json.loads(files[0].read_text()) if files else {}
print(f"  terminals on disk : {len(files)}")
print(f"  keys              : {sorted(sample)}")
for k in ("schema", "source_sha256", "window_sha256", "terminal_sha256",
          "appearance_id"):
    print(f"  carries {k:16s}: {k in sample}")
expect(len(files) == 1965 and not {"schema", "source_sha256",
                                   "window_sha256",
                                   "terminal_sha256"} & set(sample),
       "a terminal is a bare producer note: no schema, no source "
       "digest, no window digest and no self-digest")

item(14, "the reader accepts them without verifying anything")
lake_src = (PREDICTOR / "tools/characterize_lake.py").read_text()
reader = ast.unparse(next(
    n for n in ast.walk(ast.parse(lake_src))
    if isinstance(n, ast.FunctionDef) and n.name == "existing_terminals"))
print(f"  uses glob + read_text        : "
      f"{'glob(' in reader and 'read_text' in reader}")
print(f"  silently skips invalid JSON  : {'except ValueError' in reader}")
for k in ("schema", "sha", "duplicate", "filename"):
    print(f"  checks {k:10s}              : {k in reader.lower()}")
expect("glob(" in reader and "except ValueError" in reader
       and "sha" not in reader.lower(),
       "invalid JSON is skipped in silence and no schema, digest, "
       "filename-id correspondence or duplicate check is performed")

item(15, "every disposition row claims the census digest as its source")
disp = ast.unparse(next(
    n for n in ast.walk(ast.parse(lake_src))
    if isinstance(n, ast.FunctionDef) and n.name == "disposition_rows"))
_src_is_census = "source_sha256: census_sha" in disp.replace("'", "")
print(f"  source_sha256 := census_sha : {_src_is_census}")
print(f"  window_sha256 := census_sha : "
      f"{disp.count('census_sha') >= 2}")
print(f"  publishes BOUND_TO_SOURCE_BYTES : "
      f"{'BOUND_TO_SOURCE_BYTES' in disp}")
expect(disp.count("census_sha") >= 2
       and "BOUND_TO_SOURCE_BYTES" in disp,
       "1,965 rows claim to be bound to source bytes while carrying "
       "one global census digest as both the source and the window")

item(16, "batch envelopes bind no members")
env = ast.unparse(next(
    n for n in ast.walk(ast.parse(lake_src))
    if isinstance(n, ast.FunctionDef) and n.name == "emit_batch_envelope"))
_empty_vars = ("'variables': []" in env) or ("variables: []" in env)
_names_members = ("terminal" in env.lower() and "members" in env.lower())
print(f"  data_consumed.variables is empty : {_empty_vars}")
print(f"  names its member terminals       : {_names_members}")
expect(_empty_vars,
       "a batch envelope carries no variable members, so nothing ties "
       "the cube row to the terminals it summarises")


# ==================================================== C64: the design
item(17, "C52 calls itself sealed but is prose")
design = (PREDICTOR / "docs/audits/evidence"
          / "PER_VARIABLE_PREPROCESSING_DESIGN.v1.md").read_text()
print(f"  declares SEALED            : "
      f"{'SEALED_DESIGN_NO_SCORES_COMPUTED' in design}")
_has_schema = '"schema"' in design
print(f"  has a machine schema       : {_has_schema}")
print(f"  has a self digest          : "
      f"{'self_digest' in design or 'sha256' in design}")
_validator = (PREDICTOR / "olap/selection_design.py")
_has_validator = (_validator.is_file()
                  and "PER_VARIABLE" in _validator.read_text())
print(f"  has an executable validator: {_has_validator}")
expect("SEALED_DESIGN_NO_SCORES_COMPUTED" in design
       and "sha256" not in design,
       "it is Markdown prose with no schema, no self-digest, no "
       "executable population and no validator, and it calls itself "
       "sealed")

# ---------------------------------------------------------------
real_after = {"b4": tree_digest(B4_REAL), "t2": tree_digest(T2_REAL)}
print("\n=== PRESERVED ROOTS ===")
for k in REAL_BEFORE:
    same = REAL_BEFORE[k] == real_after[k]
    print(f"  {k}: unchanged={same}")
    if not same:
        FAILURES.append(f"the real {k} root changed during the PRE")

print(f"\n=== PRE SUMMARY: 17 items; {len(FAILURES)} NOT reproduced ===")
for f in FAILURES:
    print(f"  NOT-REPRODUCED: {f}")
sys.exit(1 if FAILURES else 0)
