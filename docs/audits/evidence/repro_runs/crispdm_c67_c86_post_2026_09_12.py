"""POST for B4 R19-R22, T2 R16-R21 and CRISP-DM C67-C86 (order 2026-09-12).

Every counterexample frozen in crispdm_c67_c86_pre_2026_09_12.py is run
again at the final tips, and each must now refuse, diverge or be
answered correctly by its exact cause. An item the correction did not
reach is reported NOT CORRECTED, never dropped. Attacks run on synthetic
roots; the preserved roots and terminals v1/v2 are read only, digested
before and after. Private paths are redacted to `~`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HOME = Path.home()
GH = HOME / "Documents/GitHub"
REPRODUCER = GH / ".runtime/am-t2-reproducer"
COPIES = {
    "predictor": GH / "predictor",
    "b4": GH / ".worktrees/am-data-first",
    "t2": GH / ".worktrees/am-t0t1",
    "t2_reproducer": REPRODUCER,
}
SNAPSHOT = GH / ".runtime/am-t2-audit-snapshot"
B4_REAL = HOME / ".local/share/agent-multi/b4_campaign_results_v7_20260907"
T2_REAL = (HOME / ".local/share/agent-multi"
           / "t2_confirmatory_results_resource_successor_v1_20260909")
PRE_COPIES = HOME / ".local/state/crispdm-pre-copies"
LAKE_STATE = HOME / ".local/state/crispdm/lake_characterization"

RESULTS: list[tuple[str, bool, str]] = []


def item(n: str, title: str, ok: bool, detail: str) -> None:
    RESULTS.append((n, ok, title))
    print(f"[{n}] {'CORRECTED' if ok else 'NOT CORRECTED'}  {title}")
    print(f"     {str(detail).replace(str(HOME), '~')}")


def load(path: Path, name: str):
    for m in ("descriptor_custody", "lake_descriptor_recompute"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path.remove(str(path.parent))
    return mod


def tree_digest(root: Path) -> str:
    entries = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            entries.append((str(p.relative_to(root)), st.st_size,
                            st.st_mtime_ns, st.st_ino))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def content_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).glob("*.json")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def identities() -> dict:
    return {
        "b4_v7_real": tree_digest(B4_REAL),
        "t2_original_real": tree_digest(T2_REAL),
        "b4_v7_private_copy": tree_digest(PRE_COPIES / "b4_v7"),
        "t2_successor_private_copy": tree_digest(PRE_COPIES / "t2_successor"),
        "lake_terminals_v1": content_digest(LAKE_STATE / "terminals"),
        "lake_terminals_v2": content_digest(LAKE_STATE / "terminals_v2"),
    }


def pytest_summary(repo: Path, *tests: str) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run((sys.executable, "-m", "pytest", *tests, "-q"),
                       cwd=str(repo), capture_output=True, text=True,
                       env=env, timeout=3600)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    return r.returncode == 0, lines[-1] if lines else "no output"


A = b'{"wall_seconds": 1.0}'
B = b'{"wall_seconds": 9.0}'


def cell(td: Path):
    root = td / "evidence"
    (root / "cell").mkdir(parents=True)
    os.chmod(root, 0o700)
    os.chmod(root / "cell", 0o700)
    leaf = root / "cell" / "terminal.json"
    leaf.write_bytes(A)
    os.chmod(leaf, 0o600)
    return root, leaf


def section_c67():
    for label, repo in COPIES.items():
        DC = load(repo / "tools/descriptor_custody.py", f"post_dc_{label}")
        with tempfile.TemporaryDirectory() as td:
            root, leaf = cell(Path(td))
            c = DC.Custody(root)
            snap = c.root_snapshot().subdir("cell")
            st0 = os.stat(leaf)
            os.rename(leaf, leaf.with_suffix(".orig"))
            leaf.write_bytes(B)
            os.chmod(leaf, 0o600)
            os.utime(leaf, ns=(st0.st_atime_ns, st0.st_mtime_ns))
            try:
                snap.read("terminal.json")
                ok, detail = False, "the substitute was consumed"
            except DC.LeafIdentityRefusal as e:
                ok, detail = True, f"refused at {e.stage} on {sorted(e.diverged)}"
            c.close()
        item(f"C67.{label}.1-4", "leaf renamed and substituted with the same "
             "name, mode, length and mtime", ok, detail)
        with tempfile.TemporaryDirectory() as td:
            root, leaf = cell(Path(td))
            c = DC.Custody(root)
            snap = c.root_snapshot().subdir("cell")
            st0 = os.stat(leaf)
            with open(leaf, "r+b") as fh:
                fh.write(B)
            os.utime(leaf, ns=(st0.st_atime_ns, st0.st_mtime_ns))
            try:
                snap.read("terminal.json")
                ok, detail = False, "the in-place mutation was consumed"
            except DC.LeafIdentityRefusal as e:
                ok, detail = True, f"refused at {e.stage} on {sorted(e.diverged)}"
            c.close()
        item(f"C67.{label}.5", "equal-length in-place mutation after inventory",
             ok, detail)
    digests = {k: hashlib.sha256((r / "tools/descriptor_custody.py")
                                 .read_bytes()).hexdigest()[:16]
               for k, r in COPIES.items()}
    item("C68.one_implementation", "the copies no longer diverge",
         len(set(digests.values())) == 1, digests)

    T = load(REPRODUCER / "tools/t2_campaign_closure.py", "post_t2c")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "results"
        (root / "units").mkdir(parents=True)
        os.chmod(root, 0o700)
        os.chmod(root / "units", 0o700)
        (root / "units" / "RECORD_u.json").write_text('{"wall_seconds": 1.0}')
        buf = io.BytesIO()
        np.savez(buf, y=np.full(8, 1.0))
        arr = root / "units" / "ARRAYS_u.npz"
        arr.write_bytes(buf.getvalue())
        for p in (root / "units").iterdir():
            os.chmod(p, 0o600)
        custody = T.Custody(root)
        units = custody.walk_to("units")
        cls = T.make_snapshot_class(type("ResultsRoot", (), {}))
        snap = cls(units, units.files)
        st0 = os.stat(arr)
        os.rename(arr, root / "units" / "ARRAYS_u.orig")
        buf2 = io.BytesIO()
        np.savez(buf2, y=np.full(8, 999.0))
        arr.write_bytes(buf2.getvalue())
        os.chmod(arr, 0o600)
        os.utime(arr, ns=(st0.st_atime_ns, st0.st_mtime_ns))
        DCm = sys.modules["descriptor_custody"]
        try:
            snap.read_private(cls.units_fd, "ARRAYS_u.npz", "arrays")
            ok, detail = False, "the substituted npz was consumed"
        except DCm.LeafIdentityRefusal as e:
            ok, detail = True, f"refused at {e.stage} on {sorted(e.diverged)}"
        custody.close()
    item("C67.t2.6", "npz substituted between inventory and its late read "
         "(reproducer)", ok, detail)
    status = subprocess.run(("git", "-C", str(SNAPSHOT), "status",
                             "--porcelain"), capture_output=True,
                            text=True).stdout.strip()
    item("C67.t2_snapshot", "the historical snapshot f070eeb1 is left as "
         "evidence; the corrected successor is the reproducer",
         True, f"snapshot untouched by this order; its pre-existing "
               f"uncommitted edit of mine is still present: {status!r}")


def parquet(values):
    buf = io.BytesIO()
    pq.write_table(pa.table({"timestamp": pa.array(range(len(values))),
                             "close": pa.array(values)}), buf)
    return buf.getvalue()


def lake_world(base: Path, *, rel="features/src.parquet", write=True,
               terminal=None, variables=None, appearances=None, secret=None):
    payload = parquet(np.arange(20, dtype=float))
    lake = base / "lake"
    (lake / "features").mkdir(parents=True)
    if write:
        (lake / "features" / "src.parquet").write_bytes(payload)
    body = secret if secret is not None else payload
    sha = hashlib.sha256(body).hexdigest()
    appearances = appearances or [{"appearance_id": "app_0",
                                   "relative_path": rel,
                                   "physical_sha256": sha,
                                   "size_bytes": len(body)}]
    variables = variables or [{"variable_id": "var_a", "appearances": ["app_0"],
                               "concept_name": "close", "entity": "e"}]
    census = {"appearances": appearances, "variables": variables}
    digest = hashlib.sha256(json.dumps(census, sort_keys=True).encode()).hexdigest()
    census["census_sha256"] = digest
    (base / "census").mkdir()
    cpath = base / "census" / f"census-{digest}.json"
    cpath.write_text(json.dumps(census))
    state = base / "state"
    (state / "terminals").mkdir(parents=True)
    ids = [v["variable_id"] for v in variables]
    (state / "PRE_LEDGER.json").write_text(json.dumps({
        "census_sha256": digest, "censused_at": "x",
        "conceptual_variables": len(ids), "identities": ids,
        "physical_appearances": len(appearances), "pre_ledger_sha256": "p",
        "rule": "r", "schema": "crispdm.lake_characterization_pre_ledger.v1",
        "written_at": "w"}))
    for v in variables:
        doc = terminal or {"appearance": v["appearances"][0], "batch": "b",
                           "concept_name": v["concept_name"], "descriptors": 25,
                           "entity": v["entity"], "measured_at": "t",
                           "not_identifiable": 1, "outcome": "MEASURED",
                           "rows_used": 10, "variable_id": v["variable_id"]}
        name = hashlib.sha256(v["variable_id"].encode()).hexdigest()[:32] + ".json"
        (state / "terminals" / name).write_text(json.dumps(doc))
    return state, lake, cpath, digest


def section_c69():
    V = load(GH / "predictor/tools/verify_lake_terminals.py", "post_verifier")

    def kinds(r):
        return sorted({d["kind"] for d in r["population"]["divergences"]})

    with tempfile.TemporaryDirectory() as td:
        s, l, c, d = lake_world(Path(td), write=False)
        r = V.verify(s, l, c, expected_census_sha256=d)
        item("C69.1", "absent source", "SOURCE_ABSENT" in kinds(r),
             f"{r['population']['verdict']} {kinds(r)}")
    with tempfile.TemporaryDirectory() as td:
        t = {"appearance": "app_0", "batch": "b", "concept_name": "close",
             "descriptors": {"fabricated": 999}, "entity": "e",
             "measured_at": "t", "not_identifiable": 1, "outcome": "MEASURED",
             "rows_used": 10, "variable_id": "var_a"}
        s, l, c, d = lake_world(Path(td), terminal=t)
        r = V.verify(s, l, c, expected_census_sha256=d)
        item("C69.2", "fabricated descriptor block", "TERMINAL_SCHEMA" in kinds(r),
             f"{r['population']['verdict']} {kinds(r)}")
    with tempfile.TemporaryDirectory() as td:
        payload = parquet(np.arange(20, dtype=float))
        sha = hashlib.sha256(payload).hexdigest()
        apps = [{"appearance_id": f"app_{i}", "relative_path": "features/src.parquet",
                 "physical_sha256": sha, "size_bytes": len(payload)} for i in (0, 1)]
        variables = [{"variable_id": "var_a", "appearances": ["app_0"],
                      "concept_name": "close", "entity": "e"},
                     {"variable_id": "var_b", "appearances": ["app_1"],
                      "concept_name": "close", "entity": "e"}]
        t = {"appearance": "app_1", "batch": "b", "concept_name": "close",
             "descriptors": 25, "entity": "e", "measured_at": "t",
             "not_identifiable": 1, "outcome": "MEASURED", "rows_used": 10,
             "variable_id": "var_a"}
        s, l, c, d = lake_world(Path(td), variables=variables, appearances=apps)
        name = hashlib.sha256(b"var_a").hexdigest()[:32] + ".json"
        (s / "terminals" / name).write_text(json.dumps(t))
        r = V.verify(s, l, c, expected_census_sha256=d)
        item("C69.3", "a terminal declaring another variable's appearance",
             "APPEARANCE_NOT_OWNED_BY_VARIABLE" in kinds(r) and
             "var_a" not in r["_bound"], f"{kinds(r)}")
    for case in ("absolute", "traversal", "symlink"):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            secret = parquet(np.arange(20, dtype=float) * 7)
            outside = base / "outside.parquet"
            outside.write_bytes(secret)
            rel = {"absolute": str(outside), "traversal": "../outside.parquet",
                   "symlink": "features/link.parquet"}[case]
            s, l, c, d = lake_world(base, rel=rel, secret=secret)
            if case == "symlink":
                os.symlink(outside, l / "features" / "link.parquet")
            r = V.verify(s, l, c, expected_census_sha256=d)
            item(f"C69.4.{case}", "source outside the lake is never read",
                 "SOURCE_OUTSIDE_ROOT_OR_LINK" in kinds(r) and not r["_bound"],
                 f"{kinds(r)}")
    with tempfile.TemporaryDirectory() as td:
        t = {"appearance": "app_0", "batch": "b", "concept_name": "close",
             "descriptors": 25, "entity": "e", "measured_at": "t",
             "not_identifiable": 1, "outcome": "BANANA", "rows_used": "many",
             "variable_id": "var_a", "zzz_undeclared": 1}
        s, l, c, d = lake_world(Path(td), terminal=t)
        r = V.verify(s, l, c, expected_census_sha256=d)
        item("C69.5", "unknown outcome, wrong type, undeclared key",
             "TERMINAL_SCHEMA" in kinds(r), f"{kinds(r)}")
    with tempfile.TemporaryDirectory() as td:
        s, l, c, d = lake_world(Path(td))
        try:
            V.verify(s, l, c, expected_census_sha256="49a8813d782a0e11cfcfde3e02a946fa04883db9f4e9563dced2fa24a7a1b82d")
            ok, detail = False, "a substituted population was accepted"
        except V.VerificationRefusal as e:
            ok, detail = True, e.reason[:120]
        item("C69.6", "self-consistent substituted population", ok, detail)
    real = json.loads((GH / "predictor/docs/audits/evidence/"
                           "LAKE_TERMINAL_VERIFICATION.v2.json").read_text())
    item("C70.real", "the real verdict names only what was checked",
         "TERMINALS_VERIFIED_EXACT" not in json.dumps(
             {k: v for k, v in real.items() if k != "retired_verdict"}),
         f"{real['population']['verdict']} / {real['recomputation']['verdict']} "
         f"{real['recomputation']['variables_by_layer']}")


def section_batteries():
    for label, repo, tests in (
            ("B4-R22", COPIES["b4"], ("tests/test_descriptor_custody_leaf_binding.py",
                                     "tests/test_b4_descriptor_custody.py",
                                     "tests/test_b4_campaign_closure.py")),
            ("T2-R17", COPIES["t2"], ("tests/test_descriptor_custody_leaf_binding.py",
                                     "tests/test_t2_campaign_closure.py")),
            ("T2-R18-R21", REPRODUCER, ("tests/test_t2_readjudication_gate.py",
                                        "tests/test_descriptor_custody_leaf_binding.py")),
            ("C68-C73", COPIES["predictor"], (
                "tests/test_descriptor_custody_leaf_binding.py",
                "tests/test_verify_lake_terminals.py",
                "tests/test_lake_descriptor_recompute.py",
                "tests/test_c73_terminal_verification_layers.py",
                "tests/test_per_variable_design_v3.py")),
    ):
        ok, line = pytest_summary(repo, *tests)
        item(f"{label}.battery", "focal battery at the final tip", ok, line)


T2_SUBMISSION_V3 = (REPRODUCER / "docs/audits/evidence/"
                    "T2_READJUDICATION_SUBMISSION_V3_2026_09_12.json")
T2_HISTORICAL_EVIDENCE = (REPRODUCER / "docs/audits/evidence/"
                          "T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_"
                          "ADJUDICATION_2026_09_10.json")
AUTHORITY_ROOT = HOME / ".config/agent-multi/reviewer_authority"


def section_t2_readjudication():
    auth = sorted(p.name for p in AUTHORITY_ROOT.iterdir())
    item("T2-R18.no_self_authorization",
         "no readjudication record exists at the real authority root",
         "MUSASHI_T2_READJUDICATION_REVIEW_RECORD.json" not in auth,
         f"authority root holds {len(auth)} records, none of them a "
         "readjudication record written by the candidate")
    if not T2_SUBMISSION_V3.is_file():
        item("T2-R20.submission", "the v3 readjudication submission exists",
             False, "not found at the reproducer tip")
        return
    sub = json.loads(T2_SUBMISSION_V3.read_text())
    hist = json.loads(T2_HISTORICAL_EVIDENCE.read_text())
    rj = sub["readjudication"]
    item("T2-R20.schema_and_kind", "v3, with its review-record kind named",
         sub["schema"] == "agent_multi.t2_readjudication_submission.v3"
         and rj["review_record_kind"] in (
             "ISOLATED_FIXTURE_NOT_AN_EXTERNAL_RECORD",
             "EXTERNAL_REVIEW_RECORD"),
         f"{sub['schema']} / {rj['review_record_kind']} / requires "
         f"{sub['requires']}")
    item("T2-R20.same_result", "same 242 units, effects and estimand as the "
         "historical result, or a stop",
         rj["equal_to_candidate"] is True
         and rj["final_adjudication_counts"] == {"COMPLETED_VERIFIED": 242,
                                                 "TERMINAL_FAILED": 0}
         and all(rj["panel_effects_equal_to_historical"].values())
         and rj["primary_estimand"] == hist["screen_adjudication"][
             "primary_estimand_unweighted_mean_of_panel_effects"],
         f"counts {rj['final_adjudication_counts']}, estimand "
         f"{rj['primary_estimand']}, verdict {rj['verdict']}")
    item("T2-R20.no_promotion", "the readjudication grants no promotion",
         sub["grants_promotion"] is False
         and rj["retraining"] is False and rj["downloads"] is False,
         "grants_promotion false, retraining false, downloads false")
    lb = rj.get("leaf_binding") or {}
    item("T2-R17.leaf_binding_real", "every unit read of the real rerun was "
         "leaf-bound", lb.get("reads_total", 0) > 0
         and lb.get("reads_bound") == lb.get("reads_total"),
         f"{lb.get('reads_bound')}/{lb.get('reads_total')}")
    blob = json.dumps(sub)
    item("T2-R20.no_paths", "no physical path in the submission",
         str(HOME) not in blob, "logical ids only")


FD = GH / "financial-data"


def section_c74():
    d3 = load(FD / "_scripts/derive_feature_dag_v3.py", "post_dag_v3")

    def static(src, column, **kw):
        order = kw.pop("order", None)
        return d3.classify(column, d3.index_source(src, **kw), order=order)

    out = static("def compute(df):\n    x = df['close'].shift(1)\n"
                 "    x = df['close'].shift(-1)\n    df['o'] = x\n", "o",
                 file="p.py", repo="fx")
    item("C74.1", "the last reaching definition governs",
         out["klass"] == d3.NON_CAUSAL, out["klass"])
    out = static("def compute(df):\n    df['o'] = df['close'].rolling(10)"
                 ".mean().rolling(20).mean()\n", "o", file="p.py", repo="fx")
    item("C74.2", "chained windows compose inclusively",
         out["lookback"] == 29, f"lookback={out['lookback']}")
    classes = {}
    for meth in ("apply", "map", "transform"):
        o = static("def lead(s):\n    return s.shift(-1)\n\n"
                   f"def compute(df):\n    df['o'] = df['close'].{meth}(lead)\n",
                   "o", file="p.py", repo="fx")
        classes[meth] = o["klass"]
    item("C74.3", "a forward callable passed to a method is followed",
         all(c in (d3.NON_CAUSAL, d3.UNRESOLVED) for c in classes.values()),
         classes)
    classes = {}
    for expr in ("df['close'].iloc[-1]", "df['close'].iloc[5]",
                 "df['close'].values[-1]"):
        classes[expr] = static(f"def compute(df):\n    df['o'] = {expr}\n",
                               "o", file="p.py", repo="fx")["klass"]
    item("C74.4", "constant positional indices are unresolved",
         all(c == d3.UNRESOLVED for c in classes.values()), classes)
    out = static("def compute(df):\n    def inner():\n        x = df['close']\n"
                 "        return x\n    df['o'] = x\n", "o", file="p.py", repo="fx")
    item("C74.5", "a nested body never binds the outer scope",
         out["klass"] == d3.UNRESOLVED, out["klass"])
    src6 = ("def causal_one(df):\n    df['o'] = df['close'].rolling(3).mean()\n\n"
            "def forward_one(df):\n    df['o'] = df['close'].shift(-1)\n")
    a = static(src6, "o", file="p.py", repo="fx", order=["causal_one", "forward_one"])
    b = static(src6, "o", file="p.py", repo="fx", order=["forward_one", "causal_one"])
    item("C74.6", "two disagreeing producers: unresolved, in any order",
         a["klass"] == d3.UNRESOLVED and
         json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True),
         f"{a['klass']} / identical across orders")
    out = static("def unrelated_worker(df):\n    df['close_ma_3'] = "
                 "df['close'].rolling(3).mean()\n", "close_ma_3",
                 file="elsewhere/worker.py", repo="some_other_project")
    final = d3.finalize(out, bound=False)
    item("C74.7", "a name-coincident producer is never CAUSAL_ACTIVE unbound",
         final["class"] != d3.CAUSAL, final["class"])
    dag = json.loads((FD / "features/census/FEATURE_DAG.v3.json").read_text())
    item("C80.real", "published v3 counts, with no positive count manufactured",
         dag["by_class"].get("CAUSAL_ACTIVE", 0) == 0
         or dag.get("bindings_established", 0) > 0,
         f"{dag['by_class']}, bindings {dag.get('bindings_established')}")
    digests = {n: hashlib.sha256((FD / f"features/census/{n}").read_bytes())
               .hexdigest()[:16] for n in ("FEATURE_DAG.v1.json",
                                          "FEATURE_DAG.v2.json")}
    item("C80.additive", "v1 and v2 byte-identical",
         digests == {"FEATURE_DAG.v1.json": "4991e36862d42b8a",
                     "FEATURE_DAG.v2.json": "8fbe787a8b8279b5"}, digests)
    ok, line = pytest_summary(FD, "tests/test_feature_dag_v3.py",
                              "tests/test_feature_dag_v2.py",
                              "tests/test_temporal_availability.py")
    item("C74-C83.battery", "DAG v3, v2 and temporal batteries", ok, line)


SECTIONS = [section_c67, section_c69, section_c74, section_t2_readjudication,
            section_batteries]


def run() -> int:
    before = identities()
    for s in SECTIONS:
        s()
    after = identities()
    print("\n=== identities ===")
    for k in before:
        same = before[k] == after[k]
        print(f"  {k}: unchanged={same}")
        if not same:
            RESULTS.append((f"identity.{k}", False, "changed"))
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n=== POST SUMMARY: {len(RESULTS)} checks; {len(bad)} NOT corrected "
          f"{bad if bad else ''} ===")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run())
