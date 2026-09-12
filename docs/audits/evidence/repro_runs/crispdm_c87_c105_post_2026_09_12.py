"""POST for B4 R23-R26, T2 R22-R27 and CRISP-DM C87-C105 (order 2026-09-12).

Every PRE counterexample runs again at the final tips and must refuse,
diverge or be answered correctly by its exact cause; missing artifacts
are reported NOT CORRECTED. Attacks use temporary fixtures; preserved
roots and historical artifacts are read only and digested before and
after. Private paths are redacted to `~`.
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
PRED, FD = GH / "predictor", GH / "financial-data"
B4, T2, REPRO = GH / ".worktrees/am-data-first", GH / ".worktrees/am-t0t1", GH / ".runtime/am-t2-reproducer"
LIVE = {"predictor": PRED, "b4": B4, "t2_hardened": T2}
LAKE = HOME / ".local/state/crispdm/lake_characterization"
RESULTS = []


def item(n, title, ok, detail):
    RESULTS.append((n, ok))
    print(f"[{n}] {'CORRECTED' if ok else 'NOT CORRECTED'}  {title}")
    print(f"     {str(detail).replace(str(HOME), '~')}")


def load(path, name):
    for m in ("descriptor_custody", "lake_descriptor_recompute", "lake_semantic_contract",
              "verify_lake_terminals"):
        sys.modules.pop(m, None)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path.remove(str(path.parent))
    return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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


def identities():
    ids = {"b4_v7_real": tree_digest(HOME / ".local/share/agent-multi/b4_campaign_results_v7_20260907"),
           "t2_original_real": tree_digest(HOME / ".local/share/agent-multi/t2_confirmatory_results_resource_successor_v1_20260909"),
           "b4_v7_private_copy": tree_digest(HOME / ".local/state/crispdm-pre-copies/b4_v7"),
           "t2_successor_private_copy": tree_digest(HOME / ".local/state/crispdm-pre-copies/t2_successor")}
    for v in ("terminals", "terminals_v2", "terminals_v3"):
        ids[f"lake_{v}"] = content_digest(LAKE / v)
    for f in ("FEATURE_DAG.v1.json", "FEATURE_DAG.v2.json", "FEATURE_DAG.v3.json",
              "PRODUCER_BINDING_MANIFEST.v1.json", "ETH_H4_TEMPORAL_CONTRACT.v1.json"):
        ids[f] = sha(FD / "features/census" / f)
    for label, p in (("B4_V2", B4 / "docs/audits/evidence/B4_READJUDICATION_SUBMISSION_V2_2026_09_12.json"),
                     ("B4_V3", B4 / "docs/audits/evidence/B4_READJUDICATION_SUBMISSION_V3_2026_09_12.json"),
                     ("T2_V2", T2 / "docs/audits/evidence/T2_READJUDICATION_SUBMISSION_V2_2026_09_12.json"),
                     ("T2_V3", REPRO / "docs/audits/evidence/T2_READJUDICATION_SUBMISSION_V3_2026_09_12.json")):
        ids[label] = sha(p)
    return ids


def pytest_line(repo, *tests):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    r = subprocess.run((sys.executable, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"),
                       cwd=str(repo), capture_output=True, text=True, env=env, timeout=3600)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    return r.returncode == 0, (lines[-1] if lines else "no output")


def sec_c87():
    digests = {}
    for label, repo in LIVE.items():
        DC = load(repo / "tools/descriptor_custody.py", f"post87_{label}")
        digests[label] = sha(repo / "tools/descriptor_custody.py")[:16]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "cell").mkdir(); (root / "cell/terminal.json").write_text('{"wall":1}')
            c = DC.Custody(root, require_owner=False)
            os.rename(root / "cell", root / "cell.original"); (root / "cell").mkdir()
            (root / "cell/terminal.json").write_text('{"wall":9}')
            try:
                c.walk_to("cell").read("terminal.json").json(); ok, d = False, "ACCEPTED_REPLACED_CHILD_DIR"
            except DC.DirectoryIdentityRefusal as e:
                ok, d = True, f"refused {e.stage} {sorted(e.diverged)}"
            c.close()
        item(f"C87.{label}.1-4", "replaced child directory", ok, d)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "cell").mkdir(); (root / "cell/terminal.json").write_text('{"wall":1}')
            c = DC.Custody(root, require_owner=False)
            os.rename(root / "cell", root / "o"); (root / "cell").mkdir()
            os.rename(root / "cell", root / "s"); os.rename(root / "o", root / "cell")
            try:
                c.walk_to("cell"); ok, d = False, "accepted"
            except DC.DirectoryIdentityRefusal as e:
                ok, d = True, f"refused {sorted(e.diverged)}"
            c.close()
        item(f"C87.{label}.5", "substitute and restore by name", ok, d)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "cell").mkdir(); (root / "cell/a.json").write_text("{}")
            c = DC.Custody(root, require_owner=False)
            real = os.listdir
            def listing(t):
                n = real(t); (root / "cell/late.json").write_text("{}"); return n
            os.listdir = listing
            try:
                c.walk_to("cell"); ok, d = False, "accepted"
            except DC.DirectoryIdentityRefusal as e:
                ok, d = True, f"refused {e.stage}"
            finally:
                os.listdir = real
            c.close()
        item(f"C87.{label}.6", "listing mutated during enumeration", ok, d)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "d.json").write_text('{"k":1,"k":2}'); (root / "n.json").write_text('{"v":NaN}')
            c = DC.Custody(root, require_owner=False)
            refused = 0
            for n in ("d.json", "n.json"):
                try:
                    c.root_snapshot().read(n).json()
                except DC.StrictJsonRefusal:
                    refused += 1
            c.close()
        item(f"C87.{label}.7", "duplicate keys and NaN", refused == 2, f"{refused}/2 refused")
    item("C88.one_implementation", "live copies identical", len(set(digests.values())) == 1,
         f"{digests}; historical reproducer 6fe6c1ea preserved at its round-6 custody by design")


def sec_c89():
    V = load(PRED / "tools/verify_lake_terminals_v3.py", "post89")
    tests = PRED / "tests/test_verify_lake_terminals_v3.py"
    ok, line = pytest_line(PRED, str(tests))
    item("C89.1-5", "stale census digest, duplicate keys, extra keys, NaN, datetime sentinel", ok, line)
    real = json.loads((PRED / "docs/audits/evidence/LAKE_TERMINAL_VERIFICATION.v3.json").read_text())
    ci = real["census_identity"]
    item("C90.real", "canonical census recomputed on the real lake",
         ci["recomputed_canonical"] == ci["reviewer_supplied_expectation"] == ci["declared_in_document"],
         f"{real['population']['verdict']} layers {real['layers']}")
    item("C91.real", "the datetime sentinel is no longer measured",
         real["layers"]["SEMANTICALLY_UNRESOLVED"] >= 1 and real["layers"]["DIVERGES"] == 0,
         real["semantic_sweep"]["by_state"])


def sec_t2():
    ok, line = pytest_line(T2, "tests/test_t2_hardened_readjudicate.py", "tests/test_t2_campaign_closure.py")
    item("T2-R25.battery", "gate before imports, environment guards, identity views", ok, line)
    sub = T2 / "docs/audits/evidence/T2_READJUDICATION_SUBMISSION_V4_2026_09_12.json"
    div = T2 / "docs/audits/evidence/T2_HARDENED_READJUDICATION_DIVERGENCE_2026_09_12.json"
    if not sub.is_file():
        if div.is_file():
            # The order: any difference is an audit result and stops
            # publication. A published divergence report is that result;
            # the item stays failed because nothing was readjudicated.
            r = json.loads(div.read_text())
            item("T2-R26.divergence", "hardened replay stopped: candidate diverges",
                 False, f"{len(r['differences'])} fields: "
                 + ", ".join(x["path"] for x in r["differences"][:6]))
        item("T2-R26-R27.submission", "hardened readjudication submission v4", False, "absent")
        return
    d = json.loads(sub.read_text()); h = d["hardened_readjudication"]
    hist = json.loads((T2 / "docs/audits/evidence/T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_ADJUDICATION_2026_09_10.json").read_text())
    item("T2-R26.same_result", "242/0, DOES_NOT_ADVANCE, same estimand and six effects",
         h["final_adjudication_counts"] == {"COMPLETED_VERIFIED": 242, "TERMINAL_FAILED": 0}
         and h["primary_estimand"] == hist["screen_adjudication"]["primary_estimand_unweighted_mean_of_panel_effects"]
         and len(h["panel_effects_equal_to_historical"]) == 6 and all(h["panel_effects_equal_to_historical"].values())
         and h["equal_to_candidate"], f"{h['verdict']} {h['primary_estimand']}")
    item("T2-R27.kind", "fixture marked, no promotion, no path",
         d["review_record_kind"] == "ISOLATED_FIXTURE_NOT_EXTERNAL_REVIEW" and d["grants_promotion"] is False
         and str(HOME) not in json.dumps(d), d["requires"])


def sec_c93_c97():
    dag = FD / "features/census/FEATURE_DAG.v4.json"
    bind = FD / "features/census/PRODUCER_BINDING_MANIFEST.v2.json"
    disp = FD / "features/census/FEATURE_DAG_V3_DISPOSITION.v1.json"
    item("C93", "v3 disposition recorded", disp.is_file(), "present" if disp.is_file() else "absent")
    if not (dag.is_file() and bind.is_file()):
        item("C96-C97", "binding v2 and DAG v4 for the successor", False, "absent")
        return
    d = json.loads(dag.read_text())
    item("C96-C97", "binding v2 and DAG v4 published with real counts", True,
         {k: d[k] for k in d if k in ("by_class", "historical_dataset_causal_active", "bindings_established")})


def sec_c100():
    ok, line = pytest_line(PRED, "tests/test_per_variable_design_v4.py")
    item("C100-C104.battery", "v4 design validator, Holm, population, score refusal", ok, line)


def sec_batteries():
    for label, repo, tests in (
            ("B4", B4, ("tests/test_descriptor_custody_component_binding.py", "tests/test_descriptor_custody_leaf_binding.py",
                        "tests/test_b4_descriptor_custody.py", "tests/test_b4_campaign_closure.py")),
            ("predictor", PRED, ("tests/test_descriptor_custody_component_binding.py", "tests/test_verify_lake_terminals_v3.py",
                                 "tests/test_c92_terminal_verification_layers_v2.py", "tests/test_per_variable_design_v4.py")),
            ("financial-data", FD, ("tests/test_temporal_quality.py", "tests/test_temporal_availability.py",
                                    "tests/test_prospective_producer_rerun.py", "tests/test_feature_dag_v4.py",
                                    "tests/test_feature_dag_v3.py", "tests/test_feature_dag_v2.py"))):
        ok, line = pytest_line(repo, *tests)
        item(f"{label}.battery", "focal battery at final tip", ok, line)


def run():
    before = identities()
    for s in (sec_c87, sec_c89, sec_t2, sec_c93_c97, sec_c100, sec_batteries):
        s()
    after = identities()
    print("\n=== identities ===")
    for k in before:
        print(f"  {k}: unchanged={before[k] == after[k]}")
        if before[k] != after[k]:
            RESULTS.append((f"identity.{k}", False))
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n=== POST SUMMARY: {len(RESULTS)} checks; {len(bad)} NOT corrected {bad if bad else ''} ===")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run())
