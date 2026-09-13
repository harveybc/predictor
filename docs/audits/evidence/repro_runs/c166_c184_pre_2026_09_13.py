"""PRE for C166-C184 D2 REATTESTATION (order 2026-09-13), frozen before any edit.

C166 asks, through the real public APIs and before editing, to reproduce:

1. switching off df_snapshot.GUARDS["source_rederive"] lets a mutated, re-digested matrix that does not come
   from the contract bytes be consumed;
2. switching off "availability" lets a row available after its decision instant be transformed;
3. switching off "fit_mode_enforcement" lets trailing_haar_threshold fit in EXPANDING mode, and later training
   rows then change earlier outputs;
4. switching off BATTERY_GUARDS["exhaustive_cuts"] lets a one-sample leak pass;
5. the three ADF/KPSS blocks produce resource estimates with identical identity;
6. df_coverage turns REFUSED into FAILED, and an inapplicable metric into NOT_RUN;
7. a historical C137 decision can look consumable although its code, name and temporal mode are not current;
8. the public historical/columnar parity differs in PCA, effective rank and loadings, while the test demands
   exact equality.

It also freezes the identities of the roots the order preserves byte for byte (C137, C156, C162, C163, C164
and the historical roots), and the live CPU RAM / GPU VRAM inventory of the three roles, including the
quarantined GPU. Every guard switched off here is restored in a finally block, and nothing is written.
Memory: this runs under crispdm-run; item 8 runs each interpreter in its own capped child. Hosts appear by
role only; private paths are redacted to `~`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HOME = Path.home()
GH = HOME / "Documents/GitHub"
PRED = GH / "predictor"
TOOLS, TESTS, EVID = PRED / "tools", PRED / "tests", PRED / "docs/audits/evidence"
STATE = HOME / ".local/state/crispdm-data-foundation"
ROLES_FILE = HOME / ".config/crispdm/host_roles.json"
BASE = "e009cb14b570"
INTERPRETERS = {"base_anaconda": HOME / "anaconda3/bin/python",
                "trading_stack": HOME / "anaconda3/envs/trading-stack/bin/python"}
RESULTS = []


def redact(x) -> str:
    return str(x).replace(str(HOME), "~")


def item(n, title, holds, detail, kind="ABSENCE"):
    RESULTS.append((n, kind, bool(holds)))
    label = ("FACT HOLDS", "FACT DIFFERS") if kind == "FACT" else ("REPRODUCED", "NOT REPRODUCED")
    print(f"[{n}] {label[0 if holds else 1]}  {title}")
    print(f"     {redact(detail)[:1500]}")


def guarded(n, title, fn, kind="ABSENCE"):
    try:
        holds, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a check that crashes is reported, never skipped
        holds, detail = False, f"CHECK CRASHED {type(exc).__name__}: {exc}"
    item(n, title, holds, detail, kind)


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        h.update(b"D" if p.is_dir() else sha(p).encode())
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(PRED), *args], capture_output=True, text=True, check=True).stdout.strip()


def load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def outcome(fn):
    try:
        fn()
        return "ACCEPTED"
    except Exception as exc:  # noqa: BLE001
        return f"REFUSED {type(exc).__name__}: {str(exc)[:110]}"


class switched_off:
    def __init__(self, table: dict, flag: str):
        self.table, self.flag = table, flag

    def __enter__(self):
        self.old = self.table[self.flag]
        self.table[self.flag] = False

    def __exit__(self, *exc):
        self.table[self.flag] = self.old


# ------------------------------------------------------------------ base
def sec_base():
    head = git("rev-parse", "HEAD")
    dirty = [ln for ln in subprocess.run(["git", "-C", str(PRED), "status", "--porcelain"], capture_output=True,
                                         text=True).stdout.splitlines() if ln and not ln.startswith("??")]
    item("BASE.predictor", "checkout at the ordered base (the audit commit), no tracked edits",
         head.startswith(BASE) and not dirty, {"head": head[:12], "tracked_dirty": dirty}, kind="FACT")


# ------------------------------------------------------ causal bypasses
def sec_bypasses():
    SNAP, OPS, BAT = load("df_snapshot"), load("df_operators"), load("df_causal_battery")

    def by_probe(table, flag, probe):
        on = outcome(probe())
        with switched_off(table, flag):
            off = outcome(probe())
        return on, off

    def c1():
        on, off = by_probe(SNAP.GUARDS, "source_rederive", BAT._probe_source_rederive)
        return on.startswith("REFUSED") and off == "ACCEPTED", {"guard_on": on, "guard_off": off,
                                                                 "switch": "df_snapshot.GUARDS['source_rederive']"}
    guarded("C166.1.source_rederive", "a public dict switch lets a mutated, re-digested matrix be consumed", c1)

    def c2():
        on, off = by_probe(SNAP.GUARDS, "availability", BAT._probe_availability)
        return on.startswith("REFUSED") and off == "ACCEPTED", {"guard_on": on, "guard_off": off}
    guarded("C166.2.availability", "a public dict switch lets a row available after its decision be transformed", c2)

    def c3():
        on, off = by_probe(OPS.GUARDS, "fit_mode_enforcement", BAT._probe_fit_mode)
        SYNC = load("df_synthetic_contract")
        # The first dry run added a small sine to rows 31-59, which barely moves the per-level detail MAD, so
        # the thresholds and the early outputs stayed equal: a harness defect. Rescaling those training rows
        # changes the MAD the bypassed fit estimates from them.
        X = BAT.base_series(100, 2, 50)
        Y = X.copy()
        Y[31:60] = Y[31:60] * 5.0 + 3.0
        ca, La = SYNC.in_memory_contract(X, name="pre_fitmode_a")
        cb, Lb = SYNC.in_memory_contract(Y, name="pre_fitmode_b")
        moved = None
        with switched_off(OPS.GUARDS, "fit_mode_enforcement"):
            fa = OPS.fit(BAT.HAAR, SNAP.FitSnapshot.from_contract(ca, "TRAIN", La), OPS.EXPANDING_PREFIX)
            fb = OPS.fit(BAT.HAAR, SNAP.FitSnapshot.from_contract(cb, "TRAIN", Lb), OPS.EXPANDING_PREFIX)
            ya, _, _ = OPS.transform_batch(fa, SNAP.TransformSnapshot.from_contract(ca, La, start=0, end=31))
            yb, _, _ = OPS.transform_batch(fb, SNAP.TransformSnapshot.from_contract(cb, Lb, start=0, end=31))
            moved = not np.array_equal(ya, yb, equal_nan=True)
        return (on.startswith("REFUSED") and off == "ACCEPTED" and moved), {
            "guard_on": on, "guard_off": off, "rows_changed": "31-59 (training)",
            "outputs_at_t_le_30_changed_with_guard_off": moved}
    guarded("C166.3.fit_mode", "a public dict switch lets trailing_haar_threshold fit EXPANDING; later training "
            "rows then move outputs at t<=30", c3)

    def c4():
        on = outcome(BAT._probe_exhaustive_battery())
        with switched_off(BAT.BATTERY_GUARDS, "exhaustive_cuts"):
            off = outcome(BAT._probe_exhaustive_battery())
        return on.startswith("REFUSED") and off == "ACCEPTED", {"guard_on": on, "guard_off": off,
                                                                 "switch": "BATTERY_GUARDS['exhaustive_cuts']"}
    guarded("C166.4.exhaustive_cuts", "a public dict switch reduces the battery to seven cuts and a one-sample "
            "leak passes", c4)

    def switches():
        found = {m: sorted(k for k, v in vars(load(m)).items() if k.isupper() and "GUARD" in k and isinstance(v, dict))
                 for m in ("df_snapshot", "df_operators", "df_causal_battery")}
        return all(found.values()), found
    guarded("C167.switch_tables_exist", "production modules expose mutable guard tables", switches)


# ------------------------------------------------------ identity, coverage
def sec_identity_coverage():
    def c5():
        U, P = load("df_profile_univariate"), load("df_memory_plan")
        rows = []
        planner = P.Planner(run_id="pre", bank="SYNTHETIC", dataset_id="pre_ds", budget_bytes=8 << 30,
                            code_sha256="0" * 64, context={"T": 20000, "n_train": 20000, "has_ts": False,
                                                           "rg_rows": 20000, "k_pairs": 0, "V_matrix": 1},
                            sink=rows.append)
        x = np.cumsum(np.random.default_rng(3).standard_normal(20000))
        policy = dict(U.UNIT_ROOT_POLICY, exact_max_n=5000)
        out = U.unit_root_rows("pre_ds", {"variable_id": "v"}, "train", x,
                               gate=planner.for_module("df_profile_univariate"), policy=policy)
        blocks = [r["estimator"]["params"].get("block_offset") for r in out if r["metric"] == "adf_statistic_block_start"
                  or r["metric"] == "adf_statistic_block_middle" or r["metric"] == "adf_statistic_block_end"]
        adf = [r for r in rows if r["metric"] == "unit_root_adf"]
        ident = {hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in adf}
        return (len(adf) == 3 and len(ident) == 1 and len(set(blocks)) == 3), {
            "adf_estimate_rows": len(adf), "distinct_row_identities": len(ident), "result_block_offsets": blocks,
            "row_keys": sorted(adf[0]) if adf else None}
    guarded("C166.5.block_identity", "three distinct ADF blocks produce resource estimates of identical identity", c5)

    def c6():
        COV = load("df_coverage")
        refused = COV._cell_state(["REFUSED"])
        cells = COV.expected_grid([{"dataset_id": "d", "variables": [{"variable_id": "timestamp"}]}],
                                  ["adf_statistic_block_start"])
        m = COV.build_matrix(cells, [])
        return (refused == "FAILED" and m["ledger"][0]["state"] == "NOT_RUN"
                and "NOT_APPLICABLE" not in COV.STATES and "REFUSED" not in COV.STATES), {
            "REFUSED_becomes": refused, "inapplicable_cell_becomes": m["ledger"][0]["state"], "states": COV.STATES}
    guarded("C166.6.coverage_states", "coverage maps REFUSED to FAILED and an inapplicable cell to NOT_RUN", c6)

    def c7():
        G = load("df_consumption_gate")
        OPS = load("df_operators")
        decisions = [json.loads(l) for l in open(STATE / "lab_evaluation_c137_v1/df_fact_lab_decision.jsonl")]
        d = next(x for x in decisions if x["decision"] == "LAB_CALIBRATED" and x["operator_kind"] == "wavelet_haar_atrous")
        subject = f"{d['operator_kind']}:{json.dumps(d['operator_params'], sort_keys=True)}"
        rec = {"schema": G.RECORD_SCHEMA, "stage": "D2", "reviewer": "external reviewer (PRE fixture, in memory)",
               "reviewed_at_date": "2026-09-13", "subject_kind": "OPERATOR", "states": {subject: "LAB_CALIBRATED"},
               "regimes": {}, "grants_public_eligibility": False, "record_sha256": ""}
        rec["record_sha256"] = G.record_digest(rec)
        out = G.decide(subject, "OPERATOR", [rec])
        checks_code = any(w in open(TOOLS / "df_consumption_gate.py").read() for w in ("code_sha256", "fit_mode", "KINDS"))
        return ("D2" not in out["missing_stages"] and not G.record_problems(rec) and not checks_code
                and d["operator_kind"] not in OPS.KINDS), {
            "historical_subject": subject, "c137_code_sha256": d["code_sha256"][:16],
            "current_kinds_contain_it": d["operator_kind"] in OPS.KINDS, "d2_counted_as_present": "D2" not in out["missing_stages"],
            "gate_checks_code_name_or_mode": checks_code}
    guarded("C166.7.historical_decision", "a C137 decision with a retired name and old code passes the gate's D2 "
            "stage unchallenged", c7)


# ------------------------------------------------------------ PCA parity
def sec_parity():
    def c8():
        test = "tests/test_df_profile_parity.py::test_public_panel_shaped_fixture_rows_identical"
        out = {}
        for label, py in INTERPRETERS.items():
            info = subprocess.run(["crispdm-run", "-m", "1G", "-t", "3m", "-n", f"pre-np-{label}", "--", str(py), "-c",
                                   "import numpy,threadpoolctl,json;print(json.dumps({'numpy':numpy.__version__,"
                                   "'blas':[(d.get('internal_api'),d.get('version')) for d in threadpoolctl.threadpool_info()]}))"],
                                  capture_output=True, text=True, timeout=300)
            r = subprocess.run(["crispdm-run", "-m", "3G", "-t", "20m", "-n", f"pre-parity-{label}", "--", "env", "-u",
                                "PYTHONPATH", "CUDA_VISIBLE_DEVICES=", str(py), "-B", "-m", "pytest", "-q", "-p",
                                "no:cacheprovider", test], cwd=str(PRED), capture_output=True, text=True, timeout=1500)
            lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
            diff = next((ln for ln in r.stdout.splitlines() if re.search(r"(assert|!=|==).*\d\.\d{10,}", ln)), None)
            out[label] = {"stack": (info.stdout.strip().splitlines() or ["?"])[-1], "summary": lines[-1] if lines else r.stderr[-200:],
                          "first_numeric_diff_line": diff[:200] if diff else None}
        base_failed = "failed" in out["base_anaconda"]["summary"]
        return base_failed, out
    guarded("C166.8.pca_parity", "the exact-equality parity test fails under a different linear-algebra stack "
            "(PCA, effective rank, loadings)", c8)


# ---------------------------------------------------------- preservation
PRESERVED_ROOTS = ("lab_evaluation_c137_v1", "lab_delay_cost_c137_v2", "causal_battery_c156_v1",
                   "campaign_plan_c162_v1", "profiles_c162_v1_coordinator", "profiles_c162_v1_worker_a",
                   "profiles_c162_v1_worker_b", "snr_calibration_c163_v1", "load_receipts",
                   "c151_worst_case_smoke_v1", "c151_worst_case_smoke_v2", "host_preflight_c161_v1",
                   "host_preflight_c161_v2", "c146_incident_v1", "synthetic_bank_c128_v1", "public_panels_c126_v2",
                   "profiles_c130_v1", "snr_calibration_c134_v1")


def identities() -> dict:
    ids = {f"state:{r}": tree_digest(STATE / r) for r in PRESERVED_ROOTS if (STATE / r).exists()}
    for p in sorted(EVID.glob("C146_*.json")) + sorted(EVID.glob("C122_C124_*.json")) + \
            sorted(EVID.glob("D[345]_*DESIGN.v1.json")):
        ids[f"evidence:{p.name}"] = sha(p)
    for p in sorted((PRED / "docs/audits").glob("MUSASHI_AUDIT_C146_C165_*.md")) + \
            sorted((PRED / "docs/handoffs").glob("MUSASHI_TO_GENERAL_SATOSHI_C166_C184_*.md")):
        ids[f"order:{p.name}"] = sha(p)
    return ids


def sec_preservation(before: dict):
    missing = [r for r in PRESERVED_ROOTS if not (STATE / r).exists()]
    writable = [r for r in PRESERVED_ROOTS if (STATE / r).exists()
                and any(p.stat().st_mode & 0o222 for p in [STATE / r, *(STATE / r).rglob("*")])]
    item("C166.preserved_roots", "every root the order preserves exists and is read-only",
         not missing and not writable, {"roots": len(PRESERVED_ROOTS), "missing": missing, "writable": writable},
         kind="FACT")


# --------------------------------------------------------------- hosts
def sec_hosts():
    roles = json.loads(ROLES_FILE.read_text())
    probe = ("echo cpus=$(nproc --all); echo mem_total_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo); "
             "echo mem_available_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); "
             "echo batch_slice_max=$(systemctl --user show crispdm-batch.slice -p MemoryMax --value); "
             "echo memguard=$(systemctl --user is-active crispdm-memguard.service); "
             "nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>&1 | sed 's/^/gpu=/'")
    inv = {}
    for role, cfg in roles.items():
        argv = ["bash", "-c", probe] if cfg["ssh"] is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                                                cfg["ssh"], probe]
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout
        kv, gpus, errors = {}, [], []
        for ln in out.splitlines():
            if ln.startswith("gpu="):
                g = ln[4:]
                if "Unable to determine" in g or "Unknown Error" in g:
                    errors.append(g.strip())
                else:
                    idx, name, total, free = [x.strip() for x in g.split(",")]
                    gpus.append({"index": int(idx), "name": name, "vram_total_mib": int(total), "vram_free_mib": int(free)})
            elif "=" in ln:
                k, v = ln.split("=", 1)
                kv[k] = v
        inv[role] = {"cpus": int(kv["cpus"]), "cpu_ram_total_gib": round(int(kv["mem_total_kib"]) / 2**20, 2),
                     "cpu_ram_available_gib": round(int(kv["mem_available_kib"]) / 2**20, 2),
                     "batch_slice_max_gib": round(int(kv["batch_slice_max"]) / 2**30, 2), "memguard": kv["memguard"],
                     "gpus_responding": gpus, "gpu_handle_errors": errors}
    quarantined = bool(inv["WORKER_B"]["gpu_handle_errors"])
    item("C180.host_inventory", "live CPU RAM and GPU VRAM per role; WORKER_B GPU1 has no handle (quarantined, not "
         "schedulable); this order uses CPU only", quarantined and all(v["memguard"] == "active" for v in inv.values()),
         inv, kind="FACT")


def main() -> int:
    print(f"PRE C166-C184 python={sys.version.split()[0]} numpy={np.__version__}")
    sec_base()
    before = identities()
    sec_bypasses()
    sec_identity_coverage()
    sec_parity()
    sec_preservation(before)
    sec_hosts()
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
