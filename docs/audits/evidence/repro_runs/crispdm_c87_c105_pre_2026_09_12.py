"""PRE for B4 R23-R26, T2 R22-R27 and CRISP-DM C87-C105 (order 2026-09-12).

Frozen BEFORE any edit, against the audited bases:
    predictor fc62a073  financial-data b24b2271  B4 2fb3e2b9
    T2 closure 53ada070 (descends from hardened 26b4214b)
    T2 historical reproducer 6fe6c1ea (preserved, never edited)

All attacks run on temporary fixtures. Preserved roots, submissions,
terminals v1-v3, FEATURE_DAG v1-v3 and the temporal contract v1 are read
only and digested before and after. Private paths are redacted to `~`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

HOME = Path.home()
GH = HOME / "Documents/GitHub"
PRED = GH / "predictor"
FD = GH / "financial-data"
B4 = GH / ".worktrees/am-data-first"
T2 = GH / ".worktrees/am-t0t1"
REPRO = GH / ".runtime/am-t2-reproducer"
COPIES = {"predictor": PRED, "b4": B4, "t2": T2, "t2_reproducer": REPRO}
LAKE = HOME / ".local/state/crispdm/lake_characterization"
FAIL: list[str] = []
_n = ""


def item(n, title):
    global _n
    _n = n
    print(f"\n=== PRE {n}: {title} ===")


def expect(cond, msg):
    print(f"  [{'PRE-REPRODUCED' if cond else 'NOT-REPRODUCED'}] {msg}")
    if not cond:
        FAIL.append(f"{_n}: {msg}")


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


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    e = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            st = p.stat()
            e.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns, st.st_ino))
    return hashlib.sha256(json.dumps(e).encode()).hexdigest()


def content_digest(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).glob("*.json")):
        h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def identities() -> dict:
    ids = {
        "b4_v7_real": tree_digest(HOME / ".local/share/agent-multi/b4_campaign_results_v7_20260907"),
        "t2_original_real": tree_digest(HOME / ".local/share/agent-multi/t2_confirmatory_results_resource_successor_v1_20260909"),
        "b4_v7_private_copy": tree_digest(HOME / ".local/state/crispdm-pre-copies/b4_v7"),
        "t2_successor_private_copy": tree_digest(HOME / ".local/state/crispdm-pre-copies/t2_successor"),
    }
    for v in ("terminals", "terminals_v2", "terminals_v3"):
        ids[f"lake_{v}"] = content_digest(LAKE / v)
    for f in ("FEATURE_DAG.v1.json", "FEATURE_DAG.v2.json", "FEATURE_DAG.v3.json",
              "PRODUCER_BINDING_MANIFEST.v1.json", "ETH_H4_TEMPORAL_CONTRACT.v1.json"):
        ids[f] = sha(FD / "features/census" / f)
    for label, p in (("B4_SUBMISSION_V2", B4 / "docs/audits/evidence/B4_READJUDICATION_SUBMISSION_V2_2026_09_12.json"),
                     ("B4_SUBMISSION_V3", B4 / "docs/audits/evidence/B4_READJUDICATION_SUBMISSION_V3_2026_09_12.json"),
                     ("T2_SUBMISSION_V2", T2 / "docs/audits/evidence/T2_READJUDICATION_SUBMISSION_V2_2026_09_12.json"),
                     ("T2_SUBMISSION_V3", REPRO / "docs/audits/evidence/T2_READJUDICATION_SUBMISSION_V3_2026_09_12.json")):
        ids[label] = sha(p)
    return ids


BEFORE = identities()
print("=== identities BEFORE ===")
for k, v in BEFORE.items():
    print(f"  {k}: {v[:16]}")

# ====================================================== C87 directories
for label, repo in COPIES.items():
    DC = load(repo / "tools/descriptor_custody.py", f"pre87_{label}")
    print(f"\n--- copy {label}: {sha(repo / 'tools/descriptor_custody.py')[:12]}")

    item(f"C87.{label}.1-4", "child directory replaced after the root photograph")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cell").mkdir()
        (root / "cell" / "terminal.json").write_text('{"wall":1}')
        c = DC.Custody(root, require_owner=False)
        try:
            os.rename(root / "cell", root / "cell.original")
            (root / "cell").mkdir()
            (root / "cell" / "terminal.json").write_text('{"wall":9}')
            value = c.walk_to("cell").read("terminal.json").json()
            if value == {"wall": 9}:
                print("  ACCEPTED_REPLACED_CHILD_DIR", value)
            expect(value == {"wall": 9}, f"walk_to('cell') consumed {value}")
        finally:
            c.close()

    item(f"C87.{label}.5", "child directory substituted and restored by name")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cell").mkdir()
        (root / "cell" / "terminal.json").write_text('{"wall":1}')
        st0 = os.stat(root / "cell")
        c = DC.Custody(root, require_owner=False)
        try:
            os.rename(root / "cell", root / "cell.original")
            (root / "cell").mkdir()
            (root / "cell" / "terminal.json").write_text('{"wall":9}')
            os.rename(root / "cell", root / "cell.substitute")
            os.rename(root / "cell.original", root / "cell")
            st1 = os.stat(root / "cell")
            value = c.walk_to("cell").read("terminal.json").json()
            expect(value == {"wall": 1} and st0.st_ctime_ns != st1.st_ctime_ns,
                   f"accepted after substitute-and-restore; directory ctime moved "
                   f"({st0.st_ctime_ns != st1.st_ctime_ns}) and nothing compared it")
        finally:
            c.close()

    item(f"C87.{label}.6", "entries changed while a directory is enumerated")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cell").mkdir()
        (root / "cell" / "a.json").write_text("{}")
        c = DC.Custody(root, require_owner=False)
        real = os.listdir
        cell_path = root / "cell"
        before = os.stat(cell_path)
        def listing(target):
            names = real(target)
            (cell_path / "late.json").write_text("{}")
            return names
        try:
            os.listdir = listing
            snap = c.walk_to("cell")
        finally:
            os.listdir = real
        after = os.stat(cell_path)
        expect("late.json" not in snap.files and before.st_mtime_ns != after.st_mtime_ns,
               f"snapshot accepted with files {list(snap.files)}; directory mtime moved "
               "during enumeration and no before/after fstat was compared")
        c.close()

    item(f"C87.{label}.7", "duplicate JSON keys and non-finite constants")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "dup.json").write_text('{"rows_used": 999, "rows_used": 11}')
        (root / "nan.json").write_text('{"value": NaN, "other": Infinity}')
        c = DC.Custody(root, require_owner=False)
        try:
            dup = c.root_snapshot().read("dup.json").json()
            nan = c.root_snapshot().read("nan.json").json()
            expect(dup == {"rows_used": 11} and nan["value"] != nan["value"],
                   f"duplicate accepted as {dup}; NaN and Infinity accepted {nan}")
        finally:
            c.close()

# ============================================================ C89 census
V = load(PRED / "tools/verify_lake_terminals.py", "pre89_verify")
R = sys.modules["lake_descriptor_recompute"]


def parquet_bytes(values, column="close"):
    buf = io.BytesIO()
    pq.write_table(pa.table({"timestamp": pa.array(range(len(values))),
                             column: pa.array(values)}), buf)
    return buf.getvalue()


def fixture(base: Path, *, appearance_extra=None, variable_extra=None):
    payload = parquet_bytes(np.arange(1, 17, dtype=np.float64))
    lake = base / "lake"
    (lake / "features").mkdir(parents=True)
    (lake / "features/source.parquet").write_bytes(payload)
    app = {"appearance_id": "app_0", "relative_path": "features/source.parquet",
           "physical_sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    app.update(appearance_extra or {})
    var = {"variable_id": "var_a", "appearances": ["app_0"], "concept_name": "close",
           "entity": "entity_a"}
    var.update(variable_extra or {})
    body = {"appearances": [app], "variables": [var]}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["census_sha256"] = digest
    (base / "census").mkdir()
    cpath = base / "census" / f"census-{digest}.json"
    cpath.write_text(json.dumps(body))
    state = base / "state"
    (state / "terminals").mkdir(parents=True)
    (state / "PRE_LEDGER.json").write_text(json.dumps({
        "census_sha256": digest, "censused_at": "f", "conceptual_variables": 1,
        "identities": ["var_a"], "physical_appearances": 1, "pre_ledger_sha256": "f",
        "rule": "f", "schema": V.LEDGER_SCHEMA, "written_at": "f"}))
    t = {"appearance": "app_0", "batch": "f", "concept_name": "close", "descriptors": 25,
         "entity": "entity_a", "measured_at": "f", "not_identifiable": 1,
         "outcome": "MEASURED", "rows_used": 11, "variable_id": "var_a"}
    tpath = state / "terminals" / (hashlib.sha256(b"var_a").hexdigest()[:32] + ".json")
    tpath.write_text(json.dumps(t))
    return state, lake, cpath, digest, tpath


item("C89.1", "a mutated appearance keeps its declared census digest")
with tempfile.TemporaryDirectory() as td:
    s, l, c, d, _ = fixture(Path(td))
    evil = parquet_bytes(np.arange(101, 117, dtype=np.float64))
    (l / "features/evil.parquet").write_bytes(evil)
    census = json.loads(c.read_text())
    a = census["appearances"][0]
    a.update(relative_path="features/evil.parquet",
             physical_sha256=hashlib.sha256(evil).hexdigest(), size_bytes=len(evil))
    c.write_text(json.dumps(census))
    r = V.verify(s, l, c, expected_census_sha256=d)
    expect(r["population"]["verdict"] == V.POPULATION_VERIFIED
           and r["_bound"]["var_a"]["logical_id"] == "features/evil.parquet",
           f"MUTATED_CENSUS {r['population']['verdict']} {r['_bound']['var_a']['logical_id']}")

item("C89.2", "duplicate JSON keys in a terminal and in the census")
with tempfile.TemporaryDirectory() as td:
    s, l, c, d, tpath = fixture(Path(td))
    tpath.write_text(tpath.read_text()[:-1] + ',"rows_used":999,"rows_used":11}')
    r = V.verify(s, l, c, expected_census_sha256=d)
    expect(r["population"]["verdict"] == V.POPULATION_VERIFIED and r["_docs"]["var_a"]["rows_used"] == 11,
           f"terminal duplicate: {r['population']['verdict']} rows_used={r['_docs']['var_a']['rows_used']}")
with tempfile.TemporaryDirectory() as td:
    s, l, c, d, _ = fixture(Path(td))
    raw = c.read_text().replace('"appearance_id": "app_0"', '"appearance_id": "app_x", "appearance_id": "app_0"', 1)
    c.write_text(raw)
    r = V.verify(s, l, c, expected_census_sha256=d)
    expect(r["population"]["verdict"] == V.POPULATION_VERIFIED,
           f"census duplicate key: {r['population']['verdict']}")

item("C89.3", "extra keys in appearance and variable")
with tempfile.TemporaryDirectory() as td:
    s, l, c, d, _ = fixture(Path(td), appearance_extra={"zzz_undeclared": 1},
                            variable_extra={"yyy_undeclared": "x"})
    r = V.verify(s, l, c, expected_census_sha256=d)
    expect(r["population"]["verdict"] == V.POPULATION_VERIFIED,
           f"extra keys accepted: {r['population']['verdict']}")

item("C89.4", "a non-finite JSON constant")
with tempfile.TemporaryDirectory() as td:
    s, l, c, d, _ = fixture(Path(td), appearance_extra={"quality": float("nan")})
    assert "NaN" in c.read_text()
    r = V.verify(s, l, c, expected_census_sha256=d)
    expect(r["population"]["verdict"] == V.POPULATION_VERIFIED,
           f"NaN in census accepted: {r['population']['verdict']}")

item("C89.5", "an int64 datetime sentinel measured as a finite number")
payload = parquet_bytes(np.full(32, np.iinfo(np.int64).min, dtype=np.int64),
                        "announcement_datetime_local_utc")
window, rows, why = V.column_window(payload, "announcement_datetime_local_utc")
rec = R.recompute(window)
expect(why is None and rec["missing_count"][0] == 0 and rec["mean"][0] < -9e18,
       f"rows={rows} why={why} missing={rec['missing_count'][0]} mean={rec['mean'][0]:.3e} "
       f"entropy={rec['discrete_entropy_bits'][0]}")

# ================================================================= C100
design = json.loads((PRED / "docs/audits/evidence/PER_VARIABLE_PREPROCESSING_DESIGN.v3.json").read_text())
item("C100.1", "A3 (a dimension control) sits in the family with A1, which adds no dimension")
expect("A1_VS_A3" in design["decision_rule"]["family"],
       f"family {design['decision_rule']['family']}")
item("C100.2", "a t interval with only three panels")
expect(design["panels"]["minimum_panels"] == 3 and "t" in design["decision_rule"]["interval"],
       f"minimum_panels={design['panels']['minimum_panels']} interval={design['decision_rule']['interval']!r}")
item("C100.3", "population admits variables without semantic type, unit, role, license or null policy")
text = json.dumps(design["population"]).lower()
missing = [w for w in ("semantic", "unit", "role", "license", "sentinel") if w not in text]
expect(len(missing) >= 4, f"population conditions never mention {missing}")
item("C100.4", "no exclusion of samples crossing gaps or truncated bars")
t = json.dumps(design).lower()
expect("truncat" not in t and "gap" not in t, "neither truncated bars nor gaps appear in the design")
item("C100.5", "Holm inversion for the bounds is not executable")
dr = design["decision_rule"]
expect(isinstance(dr["multiplicity"], str) and "algorithm" not in json.dumps(dr).lower(),
       f"multiplicity is prose: {dr['multiplicity']!r}")

AFTER = identities()
print("\n=== identities AFTER ===")
for k in BEFORE:
    same = BEFORE[k] == AFTER[k]
    print(f"  {k}: unchanged={same}")
    if not same:
        FAIL.append(f"identity {k} changed")
print(f"\n=== PRE SUMMARY: {len(FAIL)} NOT reproduced ===")
for f in FAIL:
    print("  " + f.replace(str(HOME), "~"))
sys.exit(1 if FAIL else 0)
