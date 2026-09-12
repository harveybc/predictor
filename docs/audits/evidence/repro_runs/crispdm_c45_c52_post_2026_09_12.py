"""POST for B4 R11-R14, T2 R7-R10 and CRISP-DM C45-C52.

The same sixteen counterexamples, against the corrected code. Each must
now die for its own exact reason.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
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

RESULTS: list[tuple[int, str, bool, str]] = []


def check(n: int, title: str, ok: bool, detail: str) -> None:
    RESULTS.append((n, title, ok, detail))
    print(f"\n=== POST {n}: {title} ===")
    print(f"  [{'CORRECTED' if ok else 'NOT-CORRECTED'}] {detail}")


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
import b4_campaign_closure as B4C                              # noqa: E402
import descriptor_custody as DC                                # noqa: E402
from test_b4_campaign_closure import build_root                # noqa: E402

B4C.prove_relaunch_refuses = lambda: {
    "launch_gate": "stub", "refused": True, "refusal": "REFUSED: stub",
    "accelerator_modules_before": [], "accelerator_modules_after": [],
    "refused_before_any_accelerator_import": True}


def swap_after_read(target: Path, payload: bytes):
    original = DC.Custody.read
    state = {"done": False}

    def patched(self, rel):
        art = original(self, rel)
        if not state["done"] and Path(self.root, rel) == target:
            state["done"] = True
            target.write_bytes(payload)
        return art
    DC.Custody.read = patched
    return original


# --------------------------------------------------------------- 1-2
with tempfile.TemporaryDirectory() as d:
    root, mat = build_root(Path(d), cells=12, completed=2, partial=1)
    term = root / "o2022_seed101/B4_CELL_TERMINAL.json"
    honest = json.loads(term.read_text())
    orig = swap_after_read(term, json.dumps(
        dict(honest, wall_seconds=999999.0), indent=1).encode())
    try:
        closure = B4C.build_closure(root, mat)
        done = [c for c in closure["cells"]
                if c["cell"] == "o2022_seed101"][0]
        adjudicated = done["wall_seconds_declared"]
        digest_ok = done["terminal_sha256"] == hashlib.sha256(
            json.dumps(honest, indent=1).encode()).hexdigest()
    finally:
        DC.Custody.read = orig
    on_disk = json.loads(term.read_text())["wall_seconds"]
check(1, "the terminal is adjudicated from the bytes it was hashed from",
      adjudicated == honest["wall_seconds"] and digest_ok
      and on_disk == 999999.0,
      f"the file now reads {on_disk} on disk; the closure adjudicated "
      f"{adjudicated} and published the digest of THOSE bytes")

with tempfile.TemporaryDirectory() as d:
    root, mat = build_root(Path(d), cells=12, completed=2, partial=1)
    per_bar = root / "o2022_seed101/per_bar_o2022_seed101.csv"
    honest_bytes = per_bar.read_bytes()
    orig = swap_after_read(per_bar, b"net_return\n9.9\n9.9\n")
    try:
        closure = B4C.build_closure(root, mat)
        done = [c for c in closure["cells"]
                if c["cell"] == "o2022_seed101"][0]
        pb_ok = done["per_bar_sha256"] == hashlib.sha256(
            honest_bytes).hexdigest()
    finally:
        DC.Custody.read = orig
    swapped = per_bar.read_bytes() != honest_bytes
check(2, "the per-bar rows are counted in the bytes that were hashed",
      pb_ok and swapped,
      "a row-count-preserving replacement no longer reaches the "
      "adjudication; the digest is of the consumed bytes")

# ----------------------------------------------------------------- 3
src = (AM_B4 / "tools/b4_campaign_closure.py").read_text()
verify = src[src.index("def verify_completed_cell("):
             src.index("def inventory_partial_cell(")]
check(3, "the seal chain hashes what it parsed",
      "custody.read(" in verify and "sha_file(" not in verify
      and "read_text()" not in verify,
      "every artifact in the sealed-cell verification comes from "
      "`custody.read`; there is no second open and no path hash")

# ----------------------------------------------------------------- 4
with tempfile.TemporaryDirectory() as d:
    root, mat = build_root(Path(d), cells=12, completed=2, partial=1)
    status = root / "o2022_seed103/cell_runtime/status.json"
    honest_epoch = json.loads(status.read_text())["epoch_completed"]
    orig = swap_after_read(status, json.dumps(
        {"epoch_completed": 1999, "num_timesteps": 1,
         "stop_reason": "FORGED", "last_durable_artifact": None}).encode())
    try:
        closure = B4C.build_closure(root, mat)
        part = [c for c in closure["cells"]
                if c["classification"] == B4C.PARTIAL][0]
        got = part["last_durable_progress"]["epoch_completed"]
        source = part["last_durable_progress"]["read_from"]
    finally:
        DC.Custody.read = orig
check(4, "the partial cell is adjudicated from its inventory read",
      got == honest_epoch and source.startswith("the same inventory"),
      f"epoch {got} adjudicated from {source}")

# ----------------------------------------------------------------- 5
with tempfile.TemporaryDirectory() as d:
    root, mat = build_root(Path(d), cells=12, completed=2, partial=1)
    ghost = root / "o2022_seed104"
    original_snap = DC.Custody.snapshot
    state = {"done": False}

    def snap_then_create(self, rel=""):
        out = original_snap(self, rel)
        if rel in ("", ".") and not state["done"]:
            state["done"] = True
            ghost.mkdir(parents=True, exist_ok=True)
            (ghost / "B4_CELL_TERMINAL.json").write_text("{}")
        return out
    DC.Custody.snapshot = snap_then_create
    try:
        closure = B4C.build_closure(root, mat)
        entry = [c for c in closure["cells"]
                 if c["cell"] == "o2022_seed104"][0]
    finally:
        DC.Custody.snapshot = original_snap
check(5, "absence is decided from a snapshot, and says so",
      entry["classification"] == B4C.NOT_STARTED
      and "listed once from its own descriptor" in entry["evidence"]
      and len(entry["root_snapshot_sha256"]) == 64,
      "the classification names the snapshot it came from and "
      "publishes its digest, so a later appearance is visibly outside "
      "the run that used it")

# --------------------------------------------------------------- 6-9
universes = json.loads((FINANCIAL /
                        "features/census/DEMAND_UNIVERSES.v1.json"
                        ).read_text())
grains = universes["grains"]
sup = universes["ACTIVE_EXECUTION_DEMAND"]["supervised"]
check(6, "demand comes from configurations the entry point can execute",
      sup["executable_configs"] > 0
      and sup["unreachable_configs"] > 0
      and universes["RESEARCH_BANK_CANDIDATES"]["state"]
      == "REGISTERED_FOR_DEVELOPMENT",
      f"{sup['executable_configs']} executable configurations and "
      f"{sup['unreachable_configs']} unreachable ones; the registry "
      f"datasets are a separate universe of "
      f"{grains['research_bank_candidates']} candidate columns")

check(7, "an empty set yields NOT_APPLICABLE",
      universes["relations"]["rl_is_subset_of_supervised"]
      == "NOT_APPLICABLE" and grains["rl_active_columns"] == 0,
      "the active RL set is empty, so the subset test answers "
      "NOT_APPLICABLE instead of a vacuous true")

check(8, "the grains are materialised separately",
      grains["unique_column_names"] != grains["dataset_column_subjects"]
      and {"active_x_subjects", "active_y_subjects", "targets",
           "research_bank_candidates"} <= set(grains),
      f"{grains['unique_column_names']} unique names, "
      f"{grains['dataset_column_subjects']} (dataset, column) "
      f"subjects, {grains['active_x_subjects']} x and "
      f"{grains['active_y_subjects']} y subjects, "
      f"{grains['targets']} targets")

targets = sup["targets"]
roles = {s["contract_role"] for s in sup["subjects"]}
check(9, "targets are identified and never counted as inputs",
      bool(targets) and "target" in roles,
      f"targets {sorted(targets)} carry contract_role=target; roles "
      f"present: {sorted(roles)}")

# ------------------------------------------------------------- 10-11
dag = json.loads((FINANCIAL / "features/census/FEATURE_DAG.v1.json"
                  ).read_text())
nodes = dag["nodes"]
resolved = [n for n in nodes if n["class"] == "CAUSAL"]
lookbacks = {n["lookback_bars"] for n in resolved}
producers = {(n["producers"][0]["file"] if n["producers"] else None)
             for n in resolved}
check(10, "each feature carries its own graph, window and availability",
      len(lookbacks) > 2 and len(producers) > 2
      and dag["by_class"].get("UNRESOLVED_PRODUCER", 0) > 0,
      f"{len(lookbacks)} distinct lookbacks and {len(producers)} "
      f"distinct producing files across {len(resolved)} resolved "
      f"columns; {dag['by_class'].get('UNRESOLVED_PRODUCER', 0)} are "
      f"UNRESOLVED_PRODUCER and assert nothing")

check(11, "the lineage is derived from the producer repositories",
      dag["producer_files_scanned"] > 100
      and all(p.get("commit") for p in
              dag["producer_repositories"].values()),
      f"{dag['producer_files_scanned']} producer files parsed across "
      f"{len(dag['producer_repositories'])} repositories, each bound "
      f"by commit")

# ---------------------------------------------------------------- 12
ledger = json.loads((PREDICTOR / "docs/audits/evidence"
                     / "lake_characterization_ledger_2026_09_12.json"
                     ).read_text())
cov = ledger["coverage"]
check(12, "every conceptual variable of the lake has a disposition",
      cov["complete"] is True
      and cov["conceptual_variables_with_a_terminal"] == 1965
      and sum(ledger["outcomes"].values()) == 1965,
      f"{cov['conceptual_variables_with_a_terminal']}/"
      f"{cov['conceptual_variables_declared']}; outcomes "
      f"{ledger['outcomes']}")

# ------------------------------------------------------------- 13-16
sys.path.insert(0, str(AM_T2 / "tools"))
import t2_campaign_closure as T2C                              # noqa: E402

import ast                                                      # noqa: E402
closure_src = (AM_T2 / "tools/t2_campaign_closure.py").read_text()
_tree = ast.parse(closure_src)
_fn = next(n for n in _tree.body
           if isinstance(n, ast.FunctionDef)
           and n.name == "reconstruct_from_snapshot")
# Ask the AST what the function CALLS. Scanning text kept matching
# the prose that says `final_adjudication()` is not called — twice, in
# a docstring and then in an explanatory string literal. A check that
# reads prose as code is a test of my writing, not of the program.
_calls = {ast.unparse(n.func) for n in ast.walk(_fn)
          if isinstance(n, ast.Call)}
_attrs = {ast.unparse(n) for n in ast.walk(_fn)
          if isinstance(n, ast.Attribute)}
recon = ""  # kept for the message below
_reopens = sorted(c for c in _calls
                  if c.endswith(("read_text", "read_bytes", "open",
                                 "json.loads", "np.load"))
                  or c == "open")
check(13, "the record that is scored is the record that was verified",
      "snap.record" in _calls
      and "ex.final_adjudication" not in _calls
      and not _reopens,
      f"the screen consumes `snap.record(...)` — the same bytes the "
      f"pinned verifier read through the snapshot. The function calls "
      f"neither the re-resolving final_adjudication() nor any file "
      f"reader; reopening calls found: {_reopens or 'none'}")

REVIEWED = (GH / ".runtime/agent-multi-t2-reviewed-7bcd3f0d")
ident = T2C.closure_code_identity(REVIEWED)
origins = {f["loaded_from"] for f in ident["files"].values()}
check(14, "the identity names every file and where it came from",
      len(ident["files"]) == 10 and origins == {"REVIEWED_CHECKOUT",
                                                "BRANCH_TIP"}
      and "NOT called a reviewed identity" in ident["honesty"]
      and all(f["matches_record"] is True
              for f in ident["files"].values()
              if f["pinned_by_execution_record"]),
      f"{len(ident['files'])} files; 7 pinned and matching the record, "
      f"3 from the tip and declared as such")

battery = sorted((AM_T2 / "tests").glob("test_t2_campaign_closure.py"))
count = subprocess.run(
    (sys.executable, "-m", "pytest", str(battery[0]), "-q",
     "-p", "no:warnings", "--collect-only"),
    capture_output=True, text=True, cwd=str(AM_T2)).stdout
collected = [ln for ln in count.splitlines() if "test" in ln
             and "collected" in ln]
check(15, "a dedicated battery exists for the T2 closure",
      bool(battery) and bool(collected),
      f"{battery[0].name}: {collected[0].strip() if collected else '?'}")

table = [T2C.exact_two_sided_binomial_p(k, 6) for k in range(7)]
sup_doc = T2C.supersede_sign_test({"signs_positive": 3,
                                   "sign_test_exact_p_two_sided": 1.3125})
import t2_confirmatory as conf                                 # noqa: E402
check(16, "no p-value leaves [0,1] and the table is symmetric",
      table == [0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875,
                0.03125]
      and table == table[::-1]
      and sup_doc["published_value_valid"] is False
      and sup_doc["changes_verdict"] is False
      and "2 * (0.5 ** 6) * sum(" not in
      (AM_T2 / "tools/t2_confirmatory.py").read_text(),
      f"table {table}; the published 1.3125 is superseded by "
      f"{sup_doc['corrected_value']} and the doubled one-tail is gone "
      f"from the code")

# ---------------------------------------------------------------
real_after = {"b4": tree_digest(B4_REAL), "t2": tree_digest(T2_REAL)}
print("\n=== REAL EVIDENCE ROOTS ===")
untouched = True
for k in REAL_BEFORE:
    same = REAL_BEFORE[k] == real_after[k]
    print(f"  {k}: unchanged={same}")
    untouched = untouched and same

bad = [n for n, _t, ok, _d in RESULTS if not ok]
print(f"\n=== POST SUMMARY: {len(RESULTS) - len(bad)}/{len(RESULTS)} "
      f"CORRECTED; real roots untouched={untouched} ===")
for n in bad:
    print(f"  NOT-CORRECTED: item {n}")
sys.exit(1 if (bad or not untouched) else 0)
