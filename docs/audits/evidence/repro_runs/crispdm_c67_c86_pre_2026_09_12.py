"""PRE for B4 R19-R22, T2 R16-R21 and CRISP-DM C67-C86 (order 2026-09-12).

Frozen BEFORE any edit, against the audited bases:

    B4 0eab2705   T2 26b4214b   predictor 9bb90fa   financial-data d53f24b6a
    T2 recoverable snapshot f070eeb1 (its working tree carries an
    uncommitted edit of mine; it is read, never modified)

Every attack runs on synthetic roots in a temporary directory. The real
B4 v7 root, the T2 original root, the private copies and the lake
terminals v1/v2 are opened for reading only, and their digests are taken
before and after. Private paths are redacted to `~`.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HOME = Path.home()
GH = HOME / "Documents/GitHub"
COPIES = {
    "predictor": GH / "predictor/tools/descriptor_custody.py",
    "b4": GH / ".worktrees/am-data-first/tools/descriptor_custody.py",
    "t2": GH / ".worktrees/am-t0t1/tools/descriptor_custody.py",
    "t2_snapshot": GH / ".runtime/am-t2-audit-snapshot/tools/descriptor_custody.py",
}
T2_TOOLS = GH / ".worktrees/am-t0t1/tools"
VERIFIER = GH / "predictor/tools/verify_lake_terminals.py"
DAG = GH / "financial-data/_scripts/derive_feature_dag_v2.py"
B4_REAL = HOME / ".local/share/agent-multi/b4_campaign_results_v7_20260907"
T2_REAL = (HOME / ".local/share/agent-multi"
           / "t2_confirmatory_results_resource_successor_v1_20260909")
PRE_COPIES = HOME / ".local/state/crispdm-pre-copies"
LAKE_STATE = HOME / ".local/state/crispdm/lake_characterization"
DURABLE_CENSUS = ("49a8813d782a0e11cfcfde3e02a946fa04883db9f4e9563dced2fa24a7a1b82d")

FAILURES: list[str] = []
_n = ""


def redact(v) -> str:
    return str(v).replace(str(HOME), "~")


def item(n: str, title: str) -> None:
    global _n
    _n = n
    print(f"\n=== PRE {n}: {title} ===")


def expect(cond: bool, msg: str) -> None:
    print(f"  [{'PRE-REPRODUCED' if cond else 'NOT-REPRODUCED'}] {msg}")
    if not cond:
        FAILURES.append(f"{_n}: {msg}")


def facts(p: Path) -> dict:
    st = os.stat(p)
    return {"ino": st.st_ino, "size": st.st_size, "mode": oct(st.st_mode & 0o7777),
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def tree_digest(root: Path) -> str:
    entries = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            entries.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns,
                            st.st_ino))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def content_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).glob("*.json")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def load(path: Path, name: str):
    sys.modules.pop("descriptor_custody", None)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def identities() -> dict:
    return {
        "b4_v7_real": tree_digest(B4_REAL),
        "t2_original_real": tree_digest(T2_REAL),
        "b4_v7_private_copy": tree_digest(PRE_COPIES / "b4_v7"),
        "t2_successor_private_copy": tree_digest(PRE_COPIES / "t2_successor"),
        "lake_terminals_v1": content_digest(LAKE_STATE / "terminals"),
        "lake_terminals_v2": content_digest(LAKE_STATE / "terminals_v2"),
    }


BEFORE = identities()
print("=== identities BEFORE ===")
for k, v in BEFORE.items():
    print(f"  {k}: {v[:16]}")
print("  implementation digests: " + ", ".join(
    f"{k}={hashlib.sha256(p.read_bytes()).hexdigest()[:12]}" for k, p in COPIES.items()))

A = b'{"wall_seconds": 1.0}'
B = b'{"wall_seconds": 9.0}'
assert len(A) == len(B)


def make_cell(td: Path) -> Path:
    root = td / "evidence"
    (root / "cell").mkdir(parents=True)
    os.chmod(root, 0o700)
    os.chmod(root / "cell", 0o700)
    leaf = root / "cell" / "terminal.json"
    leaf.write_bytes(A)
    os.chmod(leaf, 0o600)
    return root


# ==================================================== C67 leaf identity
for label, path in COPIES.items():
    DC = load(path, f"pre_dc_{label}")

    item(f"C67.{label}.1-4", "leaf renamed inside the RETAINED directory, "
         "substitute with same name, mode, length and mtime")
    with tempfile.TemporaryDirectory() as td:
        root = make_cell(Path(td))
        leaf = root / "cell" / "terminal.json"
        c = DC.Custody(root)
        cell = c.root_snapshot().subdir("cell")
        f0 = facts(leaf)
        os.rename(leaf, root / "cell" / "terminal.orig")
        leaf.write_bytes(B)
        os.chmod(leaf, 0o600)
        os.utime(leaf, ns=(f0["mtime_ns"], f0["mtime_ns"]))
        f1 = facts(leaf)
        consumed = cell.read("terminal.json").json()["wall_seconds"]
        c.close()
        print(f"  inventoried {f0}\n  substituted {f1}")
        expect(consumed == 9.0 and f0["ino"] != f1["ino"]
               and (f0["size"], f0["mode"], f0["mtime_ns"])
               == (f1["size"], f1["mode"], f1["mtime_ns"]),
               f"photographed=1.0 consumed={consumed}: the reader re-opens "
               "the NAME and compares no inventoried leaf facts")

    item(f"C67.{label}.5", "in-place mutation of equal length after inventory")
    with tempfile.TemporaryDirectory() as td:
        root = make_cell(Path(td))
        leaf = root / "cell" / "terminal.json"
        c = DC.Custody(root)
        cell = c.root_snapshot().subdir("cell")
        f0 = facts(leaf)
        with open(leaf, "r+b") as fh:
            fh.write(B)
            fh.flush()
            os.fsync(fh.fileno())
        os.utime(leaf, ns=(f0["mtime_ns"], f0["mtime_ns"]))
        f1 = facts(leaf)
        consumed = cell.read("terminal.json").json()["wall_seconds"]
        c.close()
        print(f"  inventoried {f0}\n  mutated     {f1}")
        expect(consumed == 9.0 and f0["ino"] == f1["ino"]
               and f0["size"] == f1["size"] and f0["mtime_ns"] == f1["mtime_ns"],
               f"same inode, size and mtime; only ctime moved "
               f"({f0['ctime_ns'] != f1['ctime_ns']}); consumed={consumed}")

# ============================================ C67 T2: late NPZ read
item("C67.t2.6", "ARRAYS_<uid>.npz substituted between inventory and its "
     "late read")
sys.modules.pop("descriptor_custody", None)
T2C = load(T2_TOOLS / "t2_campaign_closure.py", "pre_t2_closure")


def npz_bytes(v: float) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, y=np.full(8, v, dtype=np.float64))
    return buf.getvalue()


with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "results"
    (root / "units").mkdir(parents=True)
    os.chmod(root, 0o700)
    os.chmod(root / "units", 0o700)
    (root / "units" / "RECORD_u.json").write_text('{"wall_seconds": 1.0}')
    arr = root / "units" / "ARRAYS_u.npz"
    arr.write_bytes(npz_bytes(1.0))
    for p in (root / "units").iterdir():
        os.chmod(p, 0o600)
    custody = T2C.Custody(root)
    units = custody.walk_to("units")
    cls = T2C.make_snapshot_class(type("ResultsRoot", (), {}))
    snap = cls(units, units.files)
    f0 = facts(arr)
    os.rename(arr, root / "units" / "ARRAYS_u.orig")
    sub = npz_bytes(999.0)
    arr.write_bytes(sub)
    os.chmod(arr, 0o600)
    os.utime(arr, ns=(f0["mtime_ns"], f0["mtime_ns"]))
    f1 = facts(arr)
    got = np.load(io.BytesIO(snap.read_private(cls.units_fd, "ARRAYS_u.npz",
                                               "arrays")))["y"][0]
    custody.close()
    print(f"  inventoried {f0}\n  substituted {f1}")
    expect(float(got) == 999.0 and f0["size"] == f1["size"],
           f"the adapter photographed 1.0 and the verifier would consume {got}")

# =============================================== C69 terminal verifier
V = load(VERIFIER, "pre_verifier")


def tname(vid: str) -> str:
    return hashlib.sha256(vid.encode()).hexdigest()[:32] + ".json"


def build(base: Path, *, variables, appearances, terminals, files,
          ledger_ids=None) -> tuple[Path, Path, Path]:
    lake = base / "lake"
    lake.mkdir(parents=True)
    for rel, data in files.items():
        (lake / rel).write_bytes(data)
    census = {"census_sha256": "", "appearances": appearances,
              "variables": variables}
    digest = hashlib.sha256(json.dumps(census, sort_keys=True).encode()).hexdigest()
    census["census_sha256"] = digest
    cdir = base / "census"
    cdir.mkdir()
    cpath = cdir / f"census-{digest}.json"
    cpath.write_text(json.dumps(census, sort_keys=True))
    state = base / "state"
    (state / "terminals").mkdir(parents=True)
    ids = ledger_ids or [v["variable_id"] for v in variables]
    (state / "PRE_LEDGER.json").write_text(json.dumps({
        "identities": ids, "conceptual_variables": len(ids),
        "census_sha256": digest, "pre_ledger_sha256": "producer-declared"}))
    for vid, doc in terminals.items():
        (state / "terminals" / tname(vid)).write_text(json.dumps(doc))
    return state, lake, cpath


def simple(base, *, terminal=None, rel="src.csv", files=None, extra_files=None):
    files = files if files is not None else {"src.csv": b"a,b\n1,2\n"}
    return build(base,
                 variables=[{"variable_id": "var_a", "appearances": ["app_0"]}],
                 appearances=[{"appearance_id": "app_0", "relative_path": rel}],
                 terminals={"var_a": terminal or {"variable_id": "var_a",
                                                  "outcome": "MEASURED",
                                                  "appearance": "app_0"}},
                 files=files)


item("C69.1", "an absent source file")
with tempfile.TemporaryDirectory() as td:
    s, l, c = simple(Path(td))
    (l / "src.csv").unlink()
    r = V.verify(s, l, c)
    expect(r["verdict"] == "TERMINALS_VERIFIED_EXACT"
           and r["sources"]["absent_count"] == 1,
           f"verdict={r['verdict']} with absent_count={r['sources']['absent_count']}")

item("C69.2", "a fabricated descriptor block")
with tempfile.TemporaryDirectory() as td:
    s, l, c = simple(Path(td), terminal={
        "variable_id": "var_a", "outcome": "MEASURED", "appearance": "app_0",
        "descriptors": {"fabricated": 999}})
    r = V.verify(s, l, c)
    expect(r["verdict"] == "TERMINALS_VERIFIED_EXACT",
           f"verdict={r['verdict']}: no terminal schema, no recomputation")

item("C69.3", "a terminal declaring ANOTHER variable's appearance and file")
with tempfile.TemporaryDirectory() as td:
    s, l, c = build(
        Path(td),
        variables=[{"variable_id": "var_a", "appearances": ["app_0"]},
                   {"variable_id": "var_b", "appearances": ["app_1"]}],
        appearances=[{"appearance_id": "app_0", "relative_path": "a.csv"},
                     {"appearance_id": "app_1", "relative_path": "b.csv"}],
        terminals={"var_a": {"variable_id": "var_a", "outcome": "MEASURED",
                             "appearance": "app_1"},
                   "var_b": {"variable_id": "var_b", "outcome": "MEASURED",
                             "appearance": "app_1"}},
        files={"a.csv": b"x\n1\n", "b.csv": b"y\n2\n"})
    r = V.verify(s, l, c)
    bound = r["_declared"]["var_a"]["relative_path"]
    expect(r["verdict"] == "TERMINALS_VERIFIED_EXACT" and bound == "a.csv",
           f"verdict={r['verdict']}; var_a declares app_1 (b.csv) and the "
           f"verifier silently bound it to the census's FIRST appearance ({bound})")

item("C69.4", "absolute path, traversal and symlink sources reach the read")
SECRET = b"OUTSIDE-THE-LAKE\n"
for case in ("absolute", "traversal", "symlink"):
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        outside = base / "outside.csv"
        outside.write_bytes(SECRET)
        rel = {"absolute": str(outside), "traversal": "../outside.csv",
               "symlink": "link.csv"}[case]
        s, l, c = simple(base, rel=rel, files={})
        if case == "symlink":
            os.symlink(outside, l / "link.csv")
        r = V.verify(s, l, c)
        read = any(d["sha256"] == hashlib.sha256(SECRET).hexdigest()
                   for d in r["sources"]["digests"].values())
        expect(read and r["verdict"] == "TERMINALS_VERIFIED_EXACT",
               f"{case}: bytes outside the lake root were read and digested "
               f"(verdict={r['verdict']})")

item("C69.5", "extra key, wrong type and unknown outcome in a terminal")
with tempfile.TemporaryDirectory() as td:
    s, l, c = simple(Path(td), terminal={
        "variable_id": "var_a", "outcome": "BANANA", "rows_used": "many",
        "zzz_undeclared": 1, "appearance": "app_0"})
    r = V.verify(s, l, c)
    expect(r["verdict"] == "TERMINALS_VERIFIED_EXACT",
           f"verdict={r['verdict']}; outcomes={r['outcomes']}")

item("C69.6", "a self-consistent ledger and census over a SUBSTITUTED population")
with tempfile.TemporaryDirectory() as td:
    s, l, c = build(
        Path(td),
        variables=[{"variable_id": v, "appearances": ["app_0"]}
                   for v in ("var_forged_x", "var_forged_y")],
        appearances=[{"appearance_id": "app_0", "relative_path": "src.csv"}],
        terminals={v: {"variable_id": v, "outcome": "MEASURED",
                       "appearance": "app_0"}
                   for v in ("var_forged_x", "var_forged_y")},
        files={"src.csv": b"a\n1\n"})
    r = V.verify(s, l, c)
    params = list(inspect.signature(V.verify).parameters)
    expect(r["verdict"] == "TERMINALS_VERIFIED_EXACT"
           and r["pre_ledger"]["census_sha256"] != DURABLE_CENSUS
           and not any("expected" in p for p in params),
           f"verdict={r['verdict']} for census "
           f"{r['pre_ledger']['census_sha256'][:12]} != durable "
           f"{DURABLE_CENSUS[:12]}; verify() takes no expected authority: {params}")

# ======================================================== C74 DAG v2
D = load(DAG, "pre_dag")


def classify(src: str, column: str, *, repo="fixture",
             file="producer.py", order=None):
    tree = ast.parse(src)
    by_output, by_name = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym = D.Symbol(node, file, repo, "0" * 40, "d" * 64)
            by_name.setdefault(sym.name, []).append(sym)
            for col in sym.outputs:
                by_output.setdefault(col, []).append(sym)
    syms = by_output.get(column, [])
    if order:
        syms = sorted(syms, key=lambda s: order.index(s.name))
    return D.classify(column, syms, by_name)


item("C74.1", "local reassignment whose LAST definition leaks")
out = classify("""
def compute(df):
    x = df['close'].shift(1)
    x = df['close'].shift(-1)
    df['o'] = x
""", "o")
expect(out["klass"] == D.CAUSAL, f"class={out['klass']} lookback={out['lookback']}")

item("C74.2", "chained windows 10 then 20")
out = classify("""
def compute(df):
    df['o'] = df['close'].rolling(10).mean().rolling(20).mean()
""", "o")
expect(out["klass"] == D.CAUSAL and out["lookback"] == 20,
       f"lookback={out['lookback']}; inclusive composition is 29")

item("C74.3", "a forward callable passed to apply/map/transform")
for meth in ("apply", "map", "transform"):
    out = classify(f"""
def lead(s):
    return s.shift(-1)

def compute(df):
    df['o'] = df['close'].{meth}(lead)
""", "o")
    expect(out["klass"] == D.CAUSAL, f"{meth}(lead): class={out['klass']}")

item("C74.4", "constant positional indices")
for expr in ("df['close'].iloc[-1]", "df['close'].iloc[5]", "df['close'].values[-1]"):
    out = classify(f"def compute(df):\n    df['o'] = {expr}\n", "o")
    expect(out["klass"] == D.CAUSAL, f"{expr}: class={out['klass']}")

item("C74.5", "a nested function body contaminating the outer scope")
out = classify("""
def compute(df):
    def inner():
        x = df['close']
        return x
    df['o'] = x
""", "o")
expect(out["klass"] == D.CAUSAL,
       f"class={out['klass']}: `x` is unbound in the outer scope, yet the "
       "nested definition was harvested as an outer local")

item("C74.6", "two live producers of one column; the order decides")
SRC6 = """
def causal_one(df):
    df['o'] = df['close'].rolling(3).mean()

def forward_one(df):
    df['o'] = df['close'].shift(-1)
"""
a = classify(SRC6, "o", order=["causal_one", "forward_one"])
b = classify(SRC6, "o", order=["forward_one", "causal_one"])
expect(a["klass"] != b["klass"],
       f"order A -> {a['klass']}, order B -> {b['klass']}")

item("C74.7", "a name-coincident causal producer that never generated the dataset")
out = classify("""
def unrelated_worker(df):
    df['close_ma_3'] = df['close'].rolling(3).mean()
""", "close_ma_3", repo="some_other_project", file="elsewhere/worker.py")
dag = json.loads((GH / "financial-data/features/census/FEATURE_DAG.v2.json").read_text())
keys = set()
for n in dag["nodes"]:
    if n.get("producer"):
        keys |= set(n["producer"])
bound = {"physical_sha256", "dataset_id", "dataset_sha256"} & keys
expect(out["klass"] == D.CAUSAL and not bound,
       f"class={out['klass']} from repo {out['producer']['repository']}; "
       f"published producer keys {sorted(keys)} carry no dataset/physical binding")

# ====================================================== identities after
AFTER = identities()
print("\n=== identities AFTER ===")
for k in BEFORE:
    same = BEFORE[k] == AFTER[k]
    print(f"  {k}: unchanged={same}")
    if not same:
        FAILURES.append(f"identity {k} changed during the PRE")

total = sum(1 for _ in range(1))
print(f"\n=== PRE SUMMARY: {len(FAILURES)} NOT reproduced ===")
for f in FAILURES:
    print(f"  {redact(f)}")
sys.exit(1 if FAILURES else 0)
