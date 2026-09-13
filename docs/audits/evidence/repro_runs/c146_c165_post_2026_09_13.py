"""POST for C146-C165 MEMORY AND CAUSALITY (order 2026-09-13), at the final tips.

Reads the final tips and the real write-once roots, never objects built in this
process as evidence. It re-runs every PRE reproduction against the public API
(each defect must now refuse or be absent), reads the exhaustive causal battery
and its mutants from their root, the worst-case memory smoke and the preflight-
bypass mutant, the physical counts of the three campaign roots, the SNR root,
the OLAP receipts and the cube, and the final health of the three roles. The
focal batteries run once, under the memory-capped launcher's scope this script
runs in. Missing evidence is NOT CORRECTED. Hosts appear by role only; private
paths are redacted to `~`.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PRE = _load(HERE / "c146_c165_pre_2026_09_13.py", "c146_c165_pre")
HOME, PRED, FD, TOOLS, TESTS, EVID = PRE.HOME, PRE.PRED, PRE.FD, PRE.TOOLS, PRE.TESTS, PRE.EVID
STATE, SYN, NOISY_UNIT = PRE.STATE, PRE.SYN, PRE.NOISY_UNIT
sha, git, redact, tool, tree_digest = PRE.sha, PRE.git, PRE.redact, PRE.tool, PRE.tree_digest
ROLES = json.loads(PRE.ROLES_FILE.read_text())
CAMPAIGN_COMMIT = "cc1c1f7c2e76"
CAMPAIGN_START = "2026-09-13 03:29:00"
ROOTS = {"COORDINATOR": STATE / "profiles_c162_v1_coordinator", "WORKER_A": STATE / "profiles_c162_v1_worker_a",
         "WORKER_B": STATE / "profiles_c162_v1_worker_b"}
PLAN = STATE / "campaign_plan_c162_v1"
SNR_ROOT = STATE / "snr_calibration_c163_v1"
BATTERY = STATE / "causal_battery_c156_v1"
SMOKE_V1, SMOKE_V2 = STATE / "c151_worst_case_smoke_v1", STATE / "c151_worst_case_smoke_v2"
PREFLIGHT = STATE / "host_preflight_c161_v2"
LOAD_THROWAWAY = STATE / "load_receipts/throwaway_d0d2_c164_v1.json"
LOAD_REAL = STATE / "load_receipts/real_d0d2_c164_v1.json"
DECLARED_POLICY_SHA256 = "dcea5b87627e8e768f460f8eb0db3ea2e2fbe54769158ab960916b32bfa6beec"
RESULTS = []


def item(n, title, ok, detail):
    RESULTS.append((n, bool(ok)))
    print(f"[{n}] {'CORRECTED' if ok else 'NOT CORRECTED'}  {title}")
    print(f"     {redact(detail)[:1400]}")


def guarded(n, title, fn):
    try:
        ok, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a check that crashes is reported, never skipped
        ok, detail = False, f"CHECK CRASHED {type(exc).__name__}: {exc}"
    item(n, title, ok, detail)


def jsonl(p: Path):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def refuses(fn, *exc_types, match=None):
    try:
        fn()
    except exc_types as exc:
        return match is None or match in str(exc), f"{type(exc).__name__}: {str(exc)[:160]}"
    return False, "no refusal"


def on_role(role: str, cmd: str, timeout=60) -> str:
    alias = ROLES[role]["ssh"]
    argv = ["bash", "-c", cmd] if alias is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                                       alias, cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout


# --------------------------------------------------------------------- tips
def sec_tips():
    for repo in (PRED, FD):
        porcelain = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                                   text=True).stdout
        dirty = [ln[3:] for ln in porcelain.splitlines() if ln and not ln.startswith("??")]
        print(f"  tip {repo.name}: {git(repo, 'rev-parse', 'HEAD')[:12]} branch={git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')} "
              f"tracked_dirty={dirty}")
    for role in ("WORKER_A", "WORKER_B"):
        head = on_role(role, "git -C ~/Documents/GitHub/.worktrees/predictor-c146 rev-parse HEAD").strip()
        print(f"  {role} campaign worktree: {head[:12]}")


# ------------------------------------------------------------------- C146
def sec_incident():
    def record():
        doc = json.loads((EVID / "C146_OOM_INCIDENT_RECORD.v1.json").read_text())
        body = {k: v for k, v in doc.items() if k != "record_sha256"}
        rederives = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == doc["record_sha256"]
        bound = (doc["bound_pre"]["script_sha256"] == sha(HERE / "c146_c165_pre_2026_09_13.py")
                 and doc["bound_pre"]["output_sha256"] == sha(HERE / "c146_c165_pre_2026_09_13.out"))
        roots = {m: (r["listing_sha256"] == tree_digest(STATE / r["root_name"]),
                     not any(p.stat().st_mode & 0o222 for p in [STATE / r["root_name"], *(STATE / r["root_name"]).rglob("*")]),
                     r["missing_final_receipts"]) for m, r in doc["roots_left"].items()}
        ok = rederives and bound and len(doc["attempts"]) == 4 and all(a and b and c for a, b, c in roots.values())
        return ok, {"record_sha256": doc["record_sha256"][:16], "rederives": rederives, "bound_to_pre": bound,
                    "attempts": [a["attempt_id"] for a in doc["attempts"]],
                    "roots_unchanged_read_only_unpromoted": roots}
    guarded("C146.record", "both OOM terminations recorded as non-governing attempts; incomplete roots kept, "
            "unchanged and unpromoted", record)


# -------------------------------------------------------------- C147-C151
def sec_memory():
    def smoke():
        term = [json.loads(p.read_text()) for p in (SMOKE_V2 / "terminals").glob("*.json")]
        t = term[0]
        out = SMOKE_V2 / t["output_file"]
        rows = [r.get("row", r) for r in jsonl(out)]
        v1_failed = sum(1 for p in (SMOKE_V1 / "attempts").glob("*/attempt-*/profile.jsonl")
                        for r in jsonl(p) if r.get("row", r)["status"] == "FAILED")
        ok = (len(term) == 1 and t["status"] == "COMPLETED" and sha(out) == t["output_sha256"]
              and len(rows) == t["rows_written"] and t["observed_peak_rss_bytes"] <= t["planned_peak_bytes"]
              <= t["memory_limit_bytes"] and not [r for r in rows if r["status"] == "FAILED"] and v1_failed == 20)
        return ok, {"dataset": t["dataset_id"][-40:], "status": t["status"], "rows": t["rows_written"],
                    "observed_cgroup_peak_gib": round(t["observed_peak_rss_bytes"] / 2**30, 3),
                    "planned_gib": round(t["planned_peak_bytes"] / 2**30, 3),
                    "limit_gib": round(t["memory_limit_bytes"] / 2**30, 3), "wall_seconds": t["wall_seconds"],
                    "limit_mechanism": t["limit_mechanism"], "v1_record_failed_rows": v1_failed}
    guarded("C151.worst_case_smoke", "the worst dataset completes below its planned peak and hard limit, on a "
            "development copy; the first smoke's constant-block defect stays recorded", smoke)

    def policy():
        U = tool("df_profile_univariate")
        block = U.UNIT_ROOT_EXACT_MAX_N
        lag = U.schwert_lag(block)
        projected = block * (lag + 2) * 8 * 5.04
        host_min = min(int(on_role(r, "grep MemTotal /proc/meminfo | tr -s ' ' | cut -d' ' -f2").strip()) * 1024
                       for r in ROLES)
        return (U.UNIT_ROOT_POLICY_SHA256 == DECLARED_POLICY_SHA256 and projected < host_min / 10), {
            "policy_sha256": U.UNIT_ROOT_POLICY_SHA256[:16], "block_n": block, "lag": lag,
            "largest_adf_projection_gib": round(projected / 2**30, 3),
            "pre_worst_single_task_gib": 60.655, "smallest_host_gib": round(host_min / 2**30, 2)}
    guarded("C148.finite_adf", "ADF/KPSS read at most a declared block; the largest ADF now projects to a small "
            "fraction of the smallest host", policy)

    def isolation():
        src = (TOOLS / "df_profile_run.py").read_text()
        tree = ast.parse(src)
        pool = any(isinstance(n, ast.Name) and n.id == "ProcessPoolExecutor" for n in ast.walk(tree))
        ir = (TOOLS / "df_isolated_runner.py").read_text()
        words = all(w in ir for w in ("systemd-run", "MemoryMax", "MemorySwapMax=0", "RESOURCE_EXCEEDED", "UNCERTAIN"))
        words_run = all(w in src for w in ("heartbeat", "stop_file", "resume"))
        return (not pool and words and words_run), {"process_pool": pool, "hard_limits_and_terminals": words,
                                                   "heartbeat_stop_resume": words_run}
    guarded("C150.isolation", "one process per dataset under a hard cgroup limit with heartbeat, stop file, resume "
            "and six durable terminals; no shared process pool", isolation)


# -------------------------------------------------------------- C152-C160
def sec_causal():
    ops, snap, lab, snr, meas = tool("df_operators"), tool("df_snapshot"), tool("df_lab_evaluation"), \
        tool("df_snr"), tool("df_operator_measure")
    u = lab.load_unit(SYN / NOISY_UNIT)
    tr = u["rec"]["partitions"]["train"]
    s = lab.train_fit_slice(u["observed"], tr)
    hampel = {"kind": "trailing_hampel", "params": {"window": 9, "k": 3.0}}
    Refusal = (ops.OperatorRefusal, snap.SnapshotRefusal)

    def c152():
        cs, ce = u["rec"]["partitions"]["confirmation"]
        bare = refuses(lambda: ops.fit({"kind": "ewma", "params": {"alpha": 0.3}}, u["observed"][cs:ce], "train"),
                       *Refusal, match="FitSnapshot")
        conf = refuses(lambda: ops.fit(hampel, snap.FitSnapshot.from_contract(u["contract"], "TRAIN", u["loader"],
                                                                                start=cs, end=ce),
                                       ops.FROZEN_PREVIOUS_PARTITION), *Refusal, match="not inside the TRAIN")
        return bare[0] and conf[0], {"bare_confirmation_array_as_train": bare[1], "confirmation_rows_as_TRAIN": conf[1]}
    guarded("C152.fit_snapshot", "PRE defect: fit accepted confirmation bytes labelled train -> now refused", c152)

    def c153():
        f = ops.fit(hampel, snap.FitSnapshot.from_contract(u["contract"], "TRAIN", u["loader"], start=s.start,
                                                           end=s.stop), ops.FROZEN_PREVIOUS_PARTITION)
        perm = np.random.default_rng(3).permutation(u["observed"].shape[0])
        bare = refuses(lambda: ops.transform_batch(f, u["observed"][perm]), *Refusal, match="TransformSnapshot")
        insample = refuses(lambda: ops.transform_batch(f, snap.TransformSnapshot.from_contract(
            u["contract"], u["loader"], start=s.start, end=s.stop)), *Refusal, match="not licensed")
        battery = json.loads((BATTERY / "CAUSAL_BATTERY_SUMMARY.json").read_text())
        reorder = [r for r in jsonl(BATTERY / "df_fact_causal_test.jsonl")
                   if r["test_class"] == "NEGATIVE_CONTROL" and "reorder" in r["case_id"]]
        ok = bare[0] and insample[0] and reorder and all(r["outcome"] == "DETECTED" for r in reorder)
        return ok, {"unordered_bare_rows": bare[1], "frozen_fit_on_its_own_rows": insample[1],
                    "reorder_negative_control": [r["outcome"] for r in reorder], "battery_run": battery["run_id"]}
    guarded("C153.transform_snapshot", "PRE defect: unordered rows transformed without refusal -> refused; reorder "
            "control detected", c153)

    def c154():
        rows = [r for r in jsonl(BATTERY / "df_fact_causal_test.jsonl") if r["test_class"] == "FIT_MODE"]
        decomposition = [r for r in rows if r["operator_kind"] == "causal_decomposition"]
        modes = {"frozen": ops.FROZEN_PREVIOUS_PARTITION, "expanding": ops.EXPANDING_PREFIX,
                 "offline": ops.OFFLINE_ANALYSIS_ONLY_NON_CAUSAL}
        ok = rows and decomposition and all(r["outcome"] == "PASS" for r in rows)
        return ok, {"fit_mode_rows": len(rows), "decomposition_rows": len(decomposition),
                    "outcomes": sorted({r["outcome"] for r in rows}), "modes": modes}
    guarded("C154.fit_modes", "PRE defect: in-sample seasonal output moved with later train rows -> fit modes "
            "separated and exercised", c154)

    def c155():
        named = "wavelet_haar_atrous" not in ops.KINDS and "trailing_haar_threshold" in ops.KINDS
        old = refuses(lambda: ops.validate_spec({"kind": "wavelet_haar_atrous", "params": {"levels": 2, "threshold_k": 3.0}}),
                      *Refusal)
        decisions = list(jsonl(BATTERY / "df_fact_naming_isolation_decision.jsonl"))
        t06 = [d for d in decisions if d["subject"] == "T06_CAUSAL_SWT"]
        return (named and old[0] and t06 and t06[0]["decision"] == "DESIGN_ONLY_NOT_IMPLEMENTED"), {
            "kinds_named": named, "old_name": old[1], "decisions": [(d["subject"], d["decision"]) for d in decisions]}
    guarded("C155.name", "PRE defect: a trous claimed without proof -> renamed; SWT stays design only", c155)

    def c156_c159():
        b = json.loads((BATTERY / "CAUSAL_BATTERY_SUMMARY.json").read_text())
        c = b["counts_per_test_class_and_outcome"]
        cuts = [r for r in jsonl(BATTERY / "df_fact_causal_test.jsonl")
                if r["test_class"] == "GUARD_MUTATION" and "exhaustive_cuts" in r["case_id"]]
        live_removed = not hasattr(meas, "future_access_invariant")
        ok = (b["all_pass"] and not b["failures"] and c.get("NEGATIVE_CONTROL", {}) == {"DETECTED": 10}
              and c.get("GUARD_MUTATION", {}) == {"DETECTED": 17} and cuts and cuts[0]["outcome"] == "DETECTED"
              and all(set(v) <= {"PASS", "DETECTED"} for v in c.values()) and live_removed)
        return ok, {"counts": c, "exhaustive_cuts_mutant": [r["outcome"] for r in cuts],
                    "seven_cut_helper_removed": live_removed, "wall_seconds": b["wall_seconds"],
                    "peak_rss_bytes": b["peak_rss_bytes"]}
    guarded("C156-C159.battery", "PRE defect: a leak at t=100 passed seven cuts -> every-t prefix, suffixes, "
            "reference, 10 controls and 17 guard mutants all detected", c156_c159)

    def c160():
        x = np.asarray(u["observed"][0 if u["observed"].ndim == 1 else slice(None)], dtype=float)
        x = x if x.ndim == 1 else x[:, 0] if x.shape[0] > x.shape[1] else x[0]
        per_t = refuses(lambda: snr.estimate(x[:512], (0, 512), "wavelet_mad"), Exception, match="OFFLINE_TRAIN_DIAGNOSTIC")
        state = snr.ESTIMATORS["wavelet_mad"]["contract_state"]
        no_kernel = all("kernel" not in e for e in snr.ESTIMATORS.values()) and not hasattr(snr, "_nv_wavelet_mad")
        return per_t[0] and state == "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL" and no_kernel, {
            "bare_array_call": per_t[1], "contract_state": state, "no_public_kernel": no_kernel}
    guarded("C160.wavelet_mad", "PRE defect: offline wavelet MAD reachable per timestamp -> refused outside its "
            "one-figure train path", c160)


# ------------------------------------------------------------------ C161
def sec_hosts():
    def preflight():
        doc = json.loads((PREFLIGHT / "HOST_PREFLIGHT.json").read_text())
        rows = list(jsonl(PREFLIGHT / "df_fact_host_receipt.jsonl"))
        commits = {r["host_role"]: r["observed"][:12] for r in rows if r["check_name"] == "commit"}
        leaks = bool(re.search(r"omega|dragon|gamma", (PREFLIGHT / "HOST_PREFLIGHT.json").read_text()
                               + (PREFLIGHT / "df_fact_host_receipt.jsonl").read_text()))
        ok = all(v["dispatchable"] for v in doc["roles"].values()) and set(commits.values()) == {CAMPAIGN_COMMIT} \
            and not leaks
        return ok, {"roles": doc["roles"], "commit_per_role": commits, "host_names_in_evidence": leaks}
    guarded("C161.preflight", "every role verified commit, code, data bytes, memory cgroup and free GPUs before "
            "dispatch; evidence by role only", preflight)


# ------------------------------------------------------------------ C162
def sec_campaign():
    def counts():
        plan = json.loads((PLAN / "CAMPAIGN_PLAN.json").read_text())
        planned = {r: set(l.strip() for l in (PLAN / f"JOBS_{r}.txt").read_text().splitlines() if l.strip())
                   for r in ROLES}
        per_role, problems, code = {}, [], set()
        datasets = {}
        for role, root in ROOTS.items():
            rec = json.loads((root / "PROFILE_RUN_RECEIPT.json").read_text())
            terms = {}
            for p in (root / "terminals").glob("*.json"):
                t = json.loads(p.read_text())
                terms.setdefault(t["dataset_id"], []).append(t)
            stat = {}
            for ds, ts in terms.items():
                last = sorted(ts, key=lambda t: t["started_at"])[-1]
                stat[last["status"]] = stat.get(last["status"], 0) + 1
                datasets.setdefault(ds, []).append(role)
                code.add(last["code_sha256"])
                if "slice=crispdm-batch.slice" not in last["limit_mechanism"]:
                    problems.append(f"{role}:{ds[-20:]}:no batch slice")
                if last["observed_peak_rss_bytes"] and last["observed_peak_rss_bytes"] > last["memory_limit_bytes"] > 0:
                    problems.append(f"{role}:{ds[-20:]}:peak above limit")
                if last["status"] == "COMPLETED":
                    f = root / last["output_file"]
                    if not f.is_file() or sha(f) != last["output_sha256"]:
                        problems.append(f"{role}:{ds[-20:]}:output does not re-hash")
                    else:
                        with open(f, "rb") as fh:
                            if sum(1 for _ in fh) != last["rows_written"]:
                                problems.append(f"{role}:{ds[-20:]}:row count differs")
            per_role[role] = {"datasets": len(terms), "planned_jobs": len(planned[role]), "terminal_status": stat,
                              "receipt_counts": rec.get("counts"), "rows_total": rec.get("rows_total"),
                              "sealed": (root / "PROFILE_RUN_RECEIPT.json").is_file()}
        dup = {d: r for d, r in datasets.items() if len(r) > 1}
        total = sum(v["datasets"] for v in per_role.values())
        ok = (total == len(plan["datasets"]) == 715 and not dup and not problems and len(code) == 1
              and all(v["sealed"] and v["datasets"] == v["planned_jobs"] for v in per_role.values()))
        return ok, {"per_role": per_role, "datasets_total": total, "in_more_than_one_role": len(dup),
                    "code_sha256_distinct": len(code), "problems": problems[:10]}
    guarded("C162.physical_counts", "every planned dataset has exactly one durable terminal in one sealed root; "
            "outputs re-hash; peaks under limits; one code digest", counts)

    def by_bank():
        out = {}
        for role, root in ROOTS.items():
            for p in (root / "terminals").glob("*.json"):
                t = json.loads(p.read_text())
                b = out.setdefault(t["bank"], {})
                b[t["status"]] = b.get(t["status"], 0) + 1
        bad = {b: {k: v for k, v in s.items() if k in ("RESOURCE_EXCEEDED", "UNCERTAIN")} for b, s in out.items()}
        return not any(bad.values()), {"terminals_by_bank_and_status": out, "resource_exceeded_or_uncertain": bad}
    guarded("C162.by_bank", "per bank terminal states published; no RESOURCE_EXCEEDED or UNCERTAIN hidden", by_bank)


# ------------------------------------------------------------------ C163
def sec_snr():
    def check():
        snr = tool("df_snr")
        doc = json.loads((SNR_ROOT / "SNR_CALIBRATION.v1.json").read_text())
        rows_path = next(SNR_ROOT.glob("*.olap_rows.jsonl"))
        ok = (doc["schema"] == snr.CALIBRATION_SCHEMA and doc["unit_count"] == 513 and doc["record_count"] > 0
              and sha(rows_path) == doc["olap_rows"]["sha256"]
              and doc["estimators"]["wavelet_mad"]["contract_state"] == "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL"
              and snr.REAL_LABEL == "MODEL_CONDITIONAL_SNR_ESTIMATE"
              and not any(p.stat().st_mode & 0o222 for p in [SNR_ROOT, *SNR_ROOT.rglob("*")]))
        return ok, {"units": doc["unit_count"], "records": doc["record_count"], "rows": doc["olap_rows"]["count"],
                    "rows_rehash": sha(rows_path) == doc["olap_rows"]["sha256"], "real_label": snr.REAL_LABEL,
                    "least_biased_label": doc["least_biased_per_perturbation"]["label"]}
    guarded("C163.snr", "SNR calibrated against separate clean and noise from a fresh write-once root; real data "
            "stays model-conditional", check)


# ------------------------------------------------------------------ C164
def sec_olap():
    def load():
        t = json.loads(LOAD_THROWAWAY.read_text())
        r = json.loads(LOAD_REAL.read_text())
        refused = {k: v["rows_refused"] for k, v in r["load"].items() if v["rows_refused"]}
        st = subprocess.run(["systemctl", "--user", "show", "crispdm-olap-loader.service", "-p", "ActiveState",
                             "-p", "NRestarts"], capture_output=True, text=True).stdout.split()
        c164 = {k: r["offered"].get(k, 0) for k in ("df_fact_dataset_terminal", "df_fact_resource_estimate",
                                                    "df_fact_causal_test", "df_fact_naming_isolation_decision",
                                                    "df_fact_host_receipt", "df_fact_incident_attempt")}
        ok = (t["idempotent"] is True and t["throwaway_database_dropped"] is True and r["historical_unchanged"] is True
              and not refused and all(c164.values()) and r["coverage"]["undeclared_rows"] == 0
              and "NRestarts=0" in st and "ActiveState=active" in st)
        return ok, {"c164_rows_offered": c164, "refused": refused, "coverage": r["coverage"],
                    "history_tables_compared": len(r["historical_before"]), "loader": st}
    guarded("C164.olap", "throwaway load idempotent and dropped; real load additive with every C146-C163 outcome, "
            "history intact, zero refused", load)


# --------------------------------------------------------------- batteries
def sec_batteries():
    env = dict(PRE.__dict__.get("clean_env", lambda: {})() or {})
    import os
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for line in (HOME / ".config/crispdm/olap-loader.env").read_text().splitlines():
        if "=" in line and line.startswith("PG"):
            k, v = line.split("=", 1)
            env[k] = v
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    tests = sorted(str(p.relative_to(PRED)) for p in TESTS.glob("test_df_*.py")) + [
        "tests/test_c139_data_foundation_olap.py", "tests/test_c164_runtime_grains.py"]
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=str(PRED),
                       env=env, capture_output=True, text=True, timeout=5400)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    item("batteries.focal", f"focal battery at the final tip, including the C151(b) preflight-bypass mutant "
         f"({len(tests)} files)", r.returncode == 0, lines[-1] if lines else r.stderr[-300:])


# ------------------------------------------------------------------ health
def sec_health():
    probe = (f"echo avail_gib=$(( $(awk '/MemAvailable/ {{print $2}}' /proc/meminfo) / 1048576 )); "
             f"echo failed_units=$(( $(systemctl --failed --no-legend | wc -l) + $(systemctl --user --failed --no-legend | wc -l) )); "
             f"echo memguard=$(systemctl --user is-active crispdm-memguard.service); "
             f"echo campaign=$(systemctl --user is-active crispdm-c162-profiles.service)/$(systemctl --user show crispdm-c162-profiles.service -p Result --value); "
             f"echo gpu_compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ); "
             # first run counted every kernel kill under this label; the one it found was the C151(b) mutant
             # inside crispdm-batch.slice. Kills are now split by the cgroup the kernel names for the victim.
             f"echo kernel_oom_kills_total=$(journalctl -k --since '{CAMPAIGN_START}' --no-pager | grep -c 'Killed process' ); "
             f"echo kernel_oom_kills_outside_batch=$(journalctl -k --since '{CAMPAIGN_START}' --no-pager | grep 'task_memcg=' | grep -vc 'crispdm-batch.slice' ); "
             f"echo unit_oom_failures_outside_batch=$(journalctl --since '{CAMPAIGN_START}' --no-pager | grep \"Failed with result 'oom-kill'\" | grep -vc crispdm ); "
             f"echo failed_unit_names=$( (systemctl --failed --no-legend --plain; systemctl --user --failed --no-legend --plain) | awk '{{print $1}}' | tr '\\n' ',' ); "
             # only this order's batch work: the watchdog and the pre-existing OLAP outbox loader are expected to run
             f"echo running_batch_units=$(systemctl --user list-units --no-legend --plain 'crispdm-*' | grep -v -E 'memguard|olap-loader|batch.slice|c165-post' | grep -c running)")
    report, ok = {}, True
    for role in ROLES:
        kv = dict(ln.split("=", 1) for ln in on_role(role, probe).splitlines() if "=" in ln)
        report[role] = kv
        ok = ok and kv.get("memguard") == "active" and kv.get("gpu_compute") == "0" \
            and kv.get("unit_oom_failures_outside_batch") == "0" and kv.get("kernel_oom_kills_outside_batch") == "0" \
            and kv.get("running_batch_units") == "0" and kv.get("campaign", "").startswith("inactive/success")
    item("health.roles", "final health of the three roles: memory, failed units, watchdog, campaign service, GPUs, "
         "OOM kills since the campaign started, units still running", ok, report)
    return report


def main() -> int:
    print(f"POST C146-C165 python={sys.version.split()[0]}")
    sec_tips()
    before = PRE.identities()
    sec_incident()
    sec_memory()
    sec_causal()
    sec_hosts()
    sec_campaign()
    sec_snr()
    sec_olap()
    sec_batteries()
    health = sec_health()
    after = PRE.identities()
    # first run crashed here reading PRE.HERE, which the PRE module does not define: a harness defect
    pre_ids = dict(re.findall(r"^  (\S[^\n]*?): ([0-9a-f]{16}) unchanged=", (HERE / "c146_c165_pre_2026_09_13.out")
                              .read_text(), re.M))
    print("\n=== preserved identities (PRE value == now, unchanged during the POST) ===")
    changed = []
    for k, v in pre_ids.items():
        now = after.get(k, "MISSING")[:16]
        same = now == v and before.get(k) == after.get(k)
        changed += [] if same else [k]
        print(f"  {k}: pre={v} now={now} equal={same}")
    gpu = {r: h.get("gpu_compute") for r, h in health.items()}
    print(f"\n=== zero line ===\n  GPU compute processes per role: {gpu}; training 0, feature selection 0, models 0, "
          "D3-D5 execution 0, RL 0, DOIN 0, live 0, venue 0, eligibility grants 0")
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n=== POST SUMMARY: {len(RESULTS)} checks; {len(bad)} NOT corrected {bad}; "
          f"preserved identities changed {changed} ===")
    return 0 if not (bad or changed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
