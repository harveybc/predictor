"""PRE for C146-C165 MEMORY AND CAUSALITY (order 2026-09-13), frozen before any
edit of code.

C146 asks, before editing:

* the two OOM terminations recorded as non-governing attempts, from the
  kernel journal, not from memory;
* the incomplete roots preserved without promotion, and the missing final
  receipts derived from disk;
* each process, dataset, metric, n, lag count and projected memory;
* the 22.71 GiB arithmetic for ADF on 13,253,761 samples frozen;
* proof that one task can exhaust a host even with workers=1.

It must also reproduce: fit() accepting confirmation bytes under the label
"train"; a transform of unordered rows without refusal; a future mutation the
seven current cuts do not exercise; the offline wavelet MAD reachable as an
ordinary numeric function; and the in-sample seasonal decomposition
initialized from the end of training.

Memory is never probed by OOM: the ADF peak factor is measured on small n in
child processes under a hard cgroup cap, and every larger figure is a
projection. Read only: roots, cube and worker hosts are read, never written.
Hosts appear by role only; the role map lives outside Git. Private paths are
redacted to `~`.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HOME = Path.home()
GH = HOME / "Documents/GitHub"
PRED, FD = GH / "predictor", GH / "financial-data"
TOOLS, TESTS, EVID = PRED / "tools", PRED / "tests", PRED / "docs/audits/evidence"
STATE = HOME / ".local/state/crispdm-data-foundation"
RAW = HOME / ".local/share/crispdm-data-foundation/public_raw_c126_20260912"
SYN, PANELS = STATE / "synthetic_bank_c128_v1", STATE / "public_panels_c126_v2"
PROFILES_V1, SNR_V1 = STATE / "profiles_c130_v1", STATE / "snr_calibration_c134_v1"
FIN_CONTRACTS = FD / "features/census/FINANCIAL_FIRST_BATCH_CONTRACTS.v1.json"
ROLES_FILE = HOME / ".config/crispdm/host_roles.json"
NOISY_UNIT = "sinusoid__white__snr0__none__n2048__v1__seed11"

BASES = {PRED: "f0e3f622bdc3", FD: "fa82af800904"}
# The owner committed an additive lake service on top of both bases after this
# order was issued (predictor olap/lake/, financial-data lake/ and README.md).
# Those commits are recorded and never touched; nothing else may differ.
FOREIGN_PATHS = {PRED: ("olap/lake/",), FD: ("lake/", "README.md")}
WORKER_BASES = {"predictor": "14a1077f9c79", "financial-data": "307342627033"}
INCIDENT_WINDOW = ("2026-09-12 22:00:00", "2026-09-12 22:45:00")
AUDIT_N = 13_253_761
PY = sys.executable
RESULTS = []


def redact(x) -> str:
    return str(x).replace(str(HOME), "~")


def item(n, title, holds, detail, kind="ABSENCE"):
    """ABSENCE/DEFECT: `holds` means the gap is reproduced (expected PRE).
    FACT: `holds` means the baseline fact is true."""
    RESULTS.append((n, kind, bool(holds)))
    label = ("FACT HOLDS", "FACT DIFFERS") if kind == "FACT" else ("REPRODUCED", "NOT REPRODUCED")
    print(f"[{n}] {label[0 if holds else 1]}  {title}")
    print(f"     {redact(detail)}")


def guarded(n, title, fn, kind="ABSENCE"):
    try:
        holds, detail = fn()
    except Exception as exc:  # noqa: BLE001 - a check that crashes is reported, never skipped
        holds, detail = False, f"CHECK CRASHED {type(exc).__name__}: {exc}"
    item(n, title, holds, detail, kind)


def git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tool(name: str):
    return load(TOOLS / f"{name}.py", name)


def gib(b) -> float:
    return round(b / 2**30, 3)


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        h.update(rel.encode())
        h.update(b"D" if p.is_dir() else sha(p).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ bases
def sec_bases():
    # The first dry run compared HEAD with the base literally and parsed a
    # stripped porcelain line; both were harness defects, corrected here.
    for repo, want in BASES.items():
        head = git(repo, "rev-parse", "HEAD")
        porcelain = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                                   text=True, check=True).stdout
        dirty = sorted(ln[3:] for ln in porcelain.splitlines() if ln and not ln.startswith("??"))
        descends = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", want, "HEAD"]).returncode == 0
        changed = git(repo, "diff", "--name-only", want, "HEAD").splitlines()
        foreign = [p for p in changed if p.startswith(FOREIGN_PATHS[repo])]
        other = [p for p in changed if p not in foreign]
        commits = git(repo, "log", "--format=%h %s", f"{want}..HEAD").splitlines()
        item(f"BASE.{repo.name}", "checkout descends from the ordered base; after it only the owner's lake commits, "
             "no tracked edits", descends and not other and not dirty,
             {"head": head[:12], "base": want, "foreign_commits_not_touched": commits,
              "foreign_files": len(foreign), "other_changed_files": other, "tracked_dirty": dirty}, kind="FACT")


# ------------------------------------------------------- C146: incident
KILL = re.compile(r"^(\S+) \S+ kernel: Out of memory: Killed process (\d+) \((\S+)\) total-vm:(\d+)kB, "
                  r"anon-rss:(\d+)kB")
INVOKED = re.compile(r"^(\S+) \S+ kernel: (\S+(?: \S+)?) invoked oom-killer")
MEMCG = re.compile(r"task_memcg=(\S+),task=(\S+),pid=(\d+),uid=(\d+)")
TASK = re.compile(r"kernel: \[\s*(\d+)\]\s+(\d+)\s+\d+\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?\d+)\s+(\S+)$")
PAGE = 4096


def scope_kind(memcg: str) -> str:
    leaf = memcg.rstrip("/").split("/")[-1]
    for pat, kind in (("snap.code", "EDITOR_SESSION_SCOPE"), ("snap.firefox", "BROWSER_SCOPE"),
                      ("docker-", "CONTAINER_SCOPE"), ("session-", "LOGIN_SESSION_SCOPE")):
        if pat in leaf:
            return kind
    return "OTHER_SCOPE"


def kernel_incident() -> dict:
    out = subprocess.run(["journalctl", "-k", "--since", INCIDENT_WINDOW[0], "--until", INCIDENT_WINDOW[1],
                          "-o", "short-iso", "--no-pager"], capture_output=True, text=True).stdout.splitlines()
    events, cur = [], None
    for ln in out:
        m = INVOKED.search(ln)
        if m:
            cur = {"invoked_at": m.group(1), "tasks_uid1000_rss_over_64mib": [], "victim": None}
            events.append(cur)
            continue
        if cur is None:
            continue
        t = TASK.search(ln)
        if t and t.group(2) == "1000" and int(t.group(4)) * PAGE > 64 * 2**20:
            cur["tasks_uid1000_rss_over_64mib"].append(
                {"pid": int(t.group(1)), "name": t.group(6), "rss_gib": gib(int(t.group(4)) * PAGE)})
        m = MEMCG.search(ln)
        if m:
            cur["victim_scope"] = scope_kind(m.group(1))
        k = KILL.search(ln)
        if k:
            cur["victim"] = {"at": k.group(1), "pid": int(k.group(2)), "name": k.group(3),
                             "anon_rss_gib": gib(int(k.group(5)) * 1024)}
    user = subprocess.run(["journalctl", "--since", INCIDENT_WINDOW[0], "--until", INCIDENT_WINDOW[1],
                           "-o", "short-iso", "--no-pager"], capture_output=True, text=True).stdout.splitlines()
    scopes = []
    for ln in user:
        m = re.search(r"systemd\[\d+\]: (\S+): Failed with result 'oom-kill'", ln)
        if m:
            scopes.append({"at": ln.split()[0], "unit": scope_kind(m.group(1))})
    return {"kernel_oom_events": [e for e in events if e["victim"]], "units_failed_with_oom_kill": scopes}


def sec_incident():
    inc = kernel_incident()

    def kills():
        ev = inc["kernel_oom_events"]
        ok = (len(ev) == 2 and all(e["victim"]["name"] == "python" and e.get("victim_scope") == "EDITOR_SESSION_SCOPE"
                                   for e in ev))
        return ok, inc
    guarded("C146.oom_terminations", "two kernel OOM kills of runner python processes in the editor session scope, "
            "and the units they took down", kills, kind="FACT")

    def roots():
        found = {}
        for root, receipts in ((PROFILES_V1, ("PROFILE_RUN_RECEIPT.json",)),
                               (SNR_V1, ("SNR_CALIBRATION.v1.json", "SNR_CALIBRATION.v1.olap_rows.jsonl"))):
            files = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            found[root.name] = {"exists": root.is_dir(), "files": files,
                                "sizes": {f: (root / f).stat().st_size for f in files},
                                "missing_final_receipts": [r for r in receipts if not (root / r).exists()],
                                "writable": any((p.stat().st_mode & 0o222) for p in [root, *root.rglob("*")]),
                                "listing_sha256": tree_digest(root)[:16],
                                "created_after_first_kill": root.stat().st_mtime > _first_kill_epoch(inc)}
        ok = all(v["exists"] and v["missing_final_receipts"] and not v["writable"] for v in found.values())
        return ok, found
    guarded("C146.receipts_missing", "profile and SNR roots exist, read-only and unpromoted, without their final "
            "receipts", roots)
    return inc


def _first_kill_epoch(inc) -> float:
    from datetime import datetime
    ev = inc["kernel_oom_events"]
    return datetime.fromisoformat(ev[0]["victim"]["at"]).timestamp() if ev else math.inf


def adf_rule(n: int) -> int:
    lag = int(math.floor(12 * (n / 100.0) ** 0.25))
    return max(0, min(lag, n // 2 - 2))


PROBE = r"""
import json, math, resource, sys, numpy as np
from statsmodels.tsa.stattools import adfuller
n = int(sys.argv[1])
rng = np.random.default_rng(0)
x = np.cumsum(rng.normal(size=n)) * 0.01 + rng.normal(size=n)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
lag = max(0, min(int(math.floor(12 * (n / 100.0) ** 0.25)), n // 2 - 2))
adfuller(x, maxlag=lag, regression="c", autolag=None)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"n": n, "lag": lag, "delta_bytes": (peak - base) * 1024, "design_bytes": n * (lag + 2) * 8}))
"""


def adf_peak_factor() -> dict:
    rows = []
    for n in (100_000, 200_000, 400_000):
        r = subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=6G", "-p", "MemorySwapMax=0",
                            "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                            PY, "-B", "-c", PROBE, str(n)], capture_output=True, text=True, timeout=900)
        d = json.loads(r.stdout.strip().splitlines()[-1])
        d["factor"] = round(d["delta_bytes"] / d["design_bytes"], 3)
        rows.append(d)
    return {"probes_under_6GiB_cgroup_cap": rows, "factor_used": max(d["factor"] for d in rows)}


def dataset_series():
    """Yields (bank, job_index, dataset_id, variable, partitions, array) one numeric variable at a time,
    in the runner's job order, never materializing a dataset matrix."""
    import pyarrow.parquet as pq
    run = tool("df_profile_run")
    sync = tool("df_synthetic_contract")
    jobs = run.public_jobs(PANELS) + run.synthetic_jobs(SYN) + run.financial_jobs(FIN_CONTRACTS, FD)
    fin_doc = json.loads(FIN_CONTRACTS.read_text())
    for idx, job in enumerate(jobs):
        if job["bank"] == "SYNTHETIC":
            d = Path(job["dir"])
            contract = sync.unit_contract(d)
            obs = np.load(d / "observed_signal.npy", allow_pickle=False)
            rec = contract["original_fields"]["unit_record"]
            if obs.shape == (rec["n_variables"], rec["n_samples"]):
                obs = obs.T
            cols = {v["name"]: obs[:, i] for i, v in enumerate(contract["variables"])}
            reader = cols.__getitem__
        else:
            if job["bank"] == "PUBLIC":
                d = Path(job["dir"])
                contract = json.loads((d / "CONTRACT.json").read_text())
                path = d / "panel.parquet"
            else:
                contract = fin_doc["contracts"][job["index"]]
                path = FD / contract["files"][0]["name"]
            pf = pq.ParquetFile(path)
            reader = lambda name, pf=pf: pf.read(columns=[name]).column(name).to_numpy(zero_copy_only=False)  # noqa: E731
        b = contract["partitions"]["boundaries"]
        for v in contract["variables"]:
            if v["name"] == "timestamp" or v["role"] == "TIMESTAMP":
                continue
            try:
                arr = np.asarray(reader(v["name"]))
            except Exception:  # noqa: BLE001 - the runner skips unreadable columns too
                continue
            if arr.dtype.kind not in "fiu":
                continue
            yield job["bank"], idx, contract["dataset_id"], v["name"], b, np.asarray(arr, dtype="float64")


def sec_projection(inc, host_mem):
    U = tool("df_profile_univariate")
    factor = adf_peak_factor()

    def arithmetic():
        lag = adf_rule(AUDIT_N)
        design = AUDIT_N * (lag + 2) * 8
        return (lag == 228 and round(design / 2**30, 2) == 22.71), \
            {"n": AUDIT_N, "lag": lag, "formula": "n * (lag + 2) * 8 bytes", "gib": round(design / 2**30, 4)}
    guarded("C146.arithmetic_22_71", "the audit's ADF arithmetic for n=13,253,761 reproduces", arithmetic, kind="FACT")

    per_job, cells = {}, []
    for bank, idx, ds, var, b, arr in dataset_series():
        for pname in U.PARTITIONS:
            s, e = int(b[pname][0]), int(b[pname][1])
            runv = U.longest_finite_run(arr[s:e])
            n = int(runv.size)
            if n < U.UNIT_ROOT_MIN_N or np.all(runv == runv[0]):
                continue
            lag = adf_rule(n)
            design = n * (lag + 2) * 8
            cell = {"bank": bank, "job_index": idx, "dataset_id": ds, "variable": var, "partition": pname,
                    "metric": "adf_statistic/adf_pvalue", "n": n, "lags": lag, "design_gib": gib(design),
                    "projected_peak_gib": gib(design * factor["factor_used"])}
            cells.append(cell)
            j = per_job.setdefault(idx, {"bank": bank, "dataset_id": ds, "max_projected_peak_gib": 0.0})
            j["max_projected_peak_gib"] = max(j["max_projected_peak_gib"], cell["projected_peak_gib"])
    cells.sort(key=lambda c: -c["projected_peak_gib"])
    worst = cells[0]

    def actual_n():
        full_n_reached = any(c["n"] == AUDIT_N for c in cells)
        return (not full_n_reached and worst["n"] < AUDIT_N), {
            "adf_peak_factor": factor, "worst_cell": worst, "full_file_n_ever_passed_to_adf": full_n_reached,
            "top_cells": cells[:8]}
    guarded("C146.actual_adf_n", "the runner passes ADF one partition's longest finite run, never the whole file; "
            "the worst cell is smaller than the audit's n but still projects past any host", actual_n, kind="FACT")

    def one_task():
        over = {role: sorted({c["job_index"] for c in cells if c["projected_peak_gib"] > mem})
                for role, mem in host_mem.items() if mem}
        ok = all(worst["projected_peak_gib"] > mem for mem in host_mem.values() if mem)
        return ok, {"host_mem_total_gib": host_mem, "worst_single_task_projected_gib": worst["projected_peak_gib"],
                    "jobs_whose_single_adf_exceeds_host_ram": {r: len(v) for r, v in over.items()},
                    "those_jobs": sorted({(per_job[i]["bank"], i, per_job[i]["dataset_id"][-40:])
                                          for v in over.values() for i in v})}
    guarded("C146.one_task_exhausts_host", "a single ADF task projects past every host's total RAM, so workers=1 "
            "does not bound memory", one_task)

    def mapping():
        victims = [e["victim"] for e in inc["kernel_oom_events"]]
        candidates = {v["pid"]: sorted({(per_job[i]["bank"], i, per_job[i]["dataset_id"][-40:])
                                        for i in per_job if per_job[i]["max_projected_peak_gib"] >= v["anon_rss_gib"]})
                      for v in victims}
        src = (TOOLS / "df_profile_run.py").read_text()
        logs_job_start = bool(re.search(r"(print|log|write)\w*\(.*(job|dataset).*start", src, re.I))
        return (not logs_job_start), {"victims": victims, "candidate_jobs_by_projection": candidates,
                                      "mapping": "UNCERTAIN: the runner records no per-job start or heartbeat, "
                                                 "so a killed worker pid cannot be tied to its dataset from disk"}
    guarded("C146.pid_to_dataset_unrecorded", "a killed worker cannot be tied to its dataset from disk; only "
            "projection-consistent candidates can be named", mapping)
    return cells


# ------------------------------------------------------- C147-C150 code
def _calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            kws = sorted(k.arg for k in n.keywords if k.arg)
            out.append(f"{name}({','.join(kws)})")
    return out


def sec_runtime_code():
    run_calls = _calls(TOOLS / "df_profile_run.py")
    uni_src = (TOOLS / "df_profile_univariate.py").read_text()

    def planner():
        text = "".join((TOOLS / f).read_text() for f in ("df_profile_run.py", "df_profile_univariate.py",
                                                         "df_profile_information.py", "df_profile_multivariate.py",
                                                         "df_sampling.py", "df_snr.py"))
        hits = [w for w in ("estimated_peak_bytes", "NOT_RUN_RESOURCE_BOUND", "MemoryMax", "RLIMIT_AS", "setrlimit")
                if w in text]
        return not hits, {"memory_words_found": hits}
    guarded("C147.no_memory_preflight", "no metric has a memory estimate or a resource-bound refusal", planner)

    def adf():
        m = re.search(r"adfuller\(run, maxlag=lag", uni_src)
        return bool(m) and "max_observations" not in uni_src, {"full_run_to_adfuller": bool(m)}
    guarded("C148.adf_unbounded", "ADF is called on the full run with the Schwert lag, no observation cap", adf)

    def columnar():
        # first dry run tested list membership of the prefix "read_table(": a harness defect
        reads = [c for c in run_calls if c.startswith("read_table(")]
        full_read = bool(reads) and not any("columns" in c for c in reads)
        return (full_read and "column_stack()" in run_calls and "append()" in run_calls), \
            {"read_table_without_columns": full_read, "column_stack": "column_stack()" in run_calls}
    guarded("C149.whole_table_and_rows_in_ram", "whole tables are read, stacked into one matrix, and every result "
            "row is held in RAM until the dataset ends", columnar)

    def isolation():
        src = (TOOLS / "df_profile_run.py").read_text()
        words = [w for w in ("ProcessPoolExecutor", "systemd-run", "heartbeat", "stop_file", "RESOURCE_EXCEEDED",
                             "resume") if w in src]
        return words == ["ProcessPoolExecutor"], {"found": words}
    guarded("C150.no_isolation_or_limits", "datasets share a process pool with no hard limit, heartbeat, stop file, "
            "resource terminal or resume", isolation)


# ------------------------------------------------------- causal boundary
def sec_causal():
    ops, meas, snr = tool("df_operators"), tool("df_operator_measure"), tool("df_snr")
    T = load(TESTS / "test_df_operators.py", "pre_test_df_operators")
    unit = json.loads((SYN / NOISY_UNIT / "UNIT.json").read_text())
    obs = np.load(SYN / NOISY_UNIT / "observed_signal.npy")
    obs = obs.T if obs.shape[0] < obs.shape[1] else obs
    parts = unit["partitions"]

    def fit_label():
        cs, ce = parts["confirmation"]
        f_conf = ops.fit({"kind": "ewma", "params": {"alpha": 0.3}}, obs[cs:ce], "train")
        f_all = ops.fit({"kind": "wavelet_haar_atrous", "params": {"levels": 2, "threshold_k": 3.0}}, obs, "train")
        binding = {k for k in f_conf if k in ("dataset", "dataset_id", "range", "partition", "input_sha256",
                                              "bytes_sha256", "timestamps")}
        return (f_conf["status"] == "FITTED" and f_all["status"] == "FITTED" and not binding), \
            {"confirmation_rows_fitted_as_train": [cs, ce], "whole_series_fitted_as_train": list(obs.shape),
             "artifact_keys": sorted(f_conf), "input_binding_keys": sorted(binding)}
    guarded("C152.fit_trusts_label", "fit() accepts confirmation rows, or the whole series, under the label 'train'",
            fit_label)

    def unordered():
        f = ops.fit({"kind": "trailing_mean", "params": {"window": 5}}, T.TRAIN, "train")
        perm = np.random.default_rng(3).permutation(T.TEST.shape[0])
        y, _, _ = ops.transform_batch(f, T.TEST[perm])
        dup = np.vstack([T.TEST[:50], T.TEST[:50]])
        ops.transform_batch(f, dup)
        params = list(inspect.signature(ops.transform_batch).parameters)
        return ("timestamps" not in params and y.shape == T.TEST.shape), \
            {"permuted_rows_transformed": True, "duplicated_rows_transformed": True, "signature": params}
    guarded("C153.transform_accepts_disorder", "transform_batch() takes unordered or duplicated rows and has no "
            "timestamp, availability or partition input", unordered)

    def cuts_miss():
        f = ops.fit({"kind": "ewma", "params": {"alpha": 0.3}}, T.TRAIN, "train")
        leak_t = 100
        honest = ops.transform_batch

        def leaking(fitted, X, oracle_mode=False):
            y, a, r = honest(fitted, X, oracle_mode=oracle_mode)
            y = y.copy()
            y[leak_t] = np.asarray(X, dtype=float)[leak_t + 1]
            return y, a, r
        ops.transform_batch = leaking
        try:
            passes_cuts = meas.future_access_invariant(f, T.TEST, T.CUTS, seed=5)
            every_t = meas.future_access_invariant(f, T.TEST, range(T.TEST.shape[0] - 1), seed=5)
        finally:
            ops.transform_batch = honest
        return (passes_cuts and not every_t), {"cuts": list(T.CUTS), "leak_at": leak_t,
                                               "undetected_by_cuts": passes_cuts, "detected_by_every_t": not every_t}
    guarded("C156.cuts_miss_a_leak", "a one-sample future leak at t=100 passes the seven-cut check and fails an "
            "every-t check", cuts_miss)

    def wavelet_mad():
        kernel = snr.ESTIMATORS["wavelet_mad"]["kernel"]
        params = dict(snr.ESTIMATORS["wavelet_mad"]["parameters"])
        x = obs[:512, 0].copy()
        per_t = [kernel(x[max(0, t - 63):t + 1], params) for t in range(63, 512)]
        import pywt
        _, d1 = pywt.dwt(x, params["wavelet"], mode=params["mode"])
        x2 = x.copy()
        x2[-1] += 100.0
        _, d2 = pywt.dwt(x2, params["wavelet"], mode=params["mode"])
        early_changed = bool(np.any(d1[:4] != d2[:4]))
        state = snr.ESTIMATORS["wavelet_mad"].get("contract_state")
        return (len(per_t) == 449 and all(np.isfinite(per_t)) and early_changed and state is None), \
            {"per_timestamp_values_produced": len(per_t), "last_sample_changes_first_detail_coefficients": early_changed,
             "contract_state": state}
    guarded("C160.wavelet_mad_reachable", "the offline wavelet MAD is an ordinary function: it yields a value per "
            "timestamp, and periodization lets the last sample move the first coefficients", wavelet_mad)

    def seasonal():
        spec = {"kind": "causal_decomposition", "params": {"period": 24, "season_alpha": 0.1, "trend_alpha": 0.1}}
        t0 = 100
        f1 = ops.fit(spec, T.TRAIN, "train")
        y1, _, _ = ops.transform_batch(f1, T.TRAIN[:t0 + 1])
        later = T.TRAIN.copy()
        later[t0 + 1:] += 10.0 * np.sin(np.arange(later.shape[0] - t0 - 1) / 3.0)[:, None]
        f2 = ops.fit(spec, later, "train")
        y2, _, _ = ops.transform_batch(f2, later[:t0 + 1])
        moved = not np.array_equal(y1, y2, equal_nan=True)
        return moved, {"rows_compared": t0 + 1, "outputs_at_or_before_t_moved_by_later_train_rows": moved,
                       "phase_origin": f1["fitted"]["phase_origin"]}
    guarded("C154.seasonal_in_sample", "in-sample seasonal output at t<=100 changes when training rows after t "
            "change", seasonal)

    def naming():
        tests = "".join(p.read_text() for p in TESTS.glob("test_df_*.py"))
        d3 = (EVID / "D3_QUANTIZATION_COMPRESSION_TIMEFREQ_DETECTORS_DESIGN.v1.json").read_text()
        equivalence = bool(re.search(r"pywt\.swt|a_trous_reference|swt_equivalence", tests))
        return ("wavelet_haar_atrous" in ops.KINDS and not equivalence and "T06_CAUSAL_SWT" in d3), \
            {"kind_name": "wavelet_haar_atrous", "equivalence_test": equivalence, "d3_names_T06_CAUSAL_SWT": True}
    guarded("C155.name_unproven", "the operator is named a trous with no equivalence proof, and D3 names a separate "
            "SWT arm", naming)

    def battery():
        # The first dry run matched words ("reference", "phase") instead of behaviour: the existing
        # `_reference_path` fixture is a batch parity check against an older six-kind module, not a
        # prefix-only reference, and "phase" matched delay assertions. Corrected to exact patterns.
        src = "".join(p.read_text() for p in TESTS.glob("test_df_oper*.py"))
        prefix_reference = bool(re.search(r"for t in range\([^)]*\):[^\n]*\n[^\n]*\[:\s*t\s*\+\s*1\]", src))
        classes = {"filtfilt": r"filtfilt\(", "shift_minus_k": r"\.shift\(\s*-", "same_convolution":
                   r"mode\s*=\s*[\"']same[\"']", "right_padding": r"np\.pad\(.*(symmetric|reflect|wrap)",
                   "full_series_dwt_swt_cwt": r"pywt\.(wavedec|swt|cwt|dwt)\(", "full_series_fft_feature":
                   r"np\.fft\.(i?rfft|i?fft)\(", "phase_compensation_shift_back": r"np\.roll\(.*-",
                   "fit_on_calibration_as_control": r"fit\([^)]*(calibration|confirmation)[^)]*\"train\"",
                   "reorder_after_materialization": r"(argsort|sort_values)\(.*timestamp",
                   "state_reuse_across_series": r"init_state\([^)]*\).*other"}
        present = {k: bool(re.search(v, src)) for k, v in classes.items()}
        return (not prefix_reference and not any(present.values())), {
            "prefix_only_reference": prefix_reference, "negative_control_classes_present": present,
            "only_control": "centered_mean_oracle", "existing_parity_fixture": "_reference_path (batch parity "
            "with an older six-kind module; not prefix-only)"}
    guarded("C158_C159.no_reference_or_controls", "no independent slow reference and no negative control beyond the "
            "centered oracle", battery)


# --------------------------------------------------------------- hosts
def ssh(alias, cmd, timeout=40):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", alias, cmd],
                          capture_output=True, text=True, timeout=timeout)


def sec_hosts() -> dict:
    roles = json.loads(ROLES_FILE.read_text())
    mem = {}
    for role, cfg in roles.items():
        probe = ("grep MemTotal /proc/meminfo | tr -s ' ' | cut -d' ' -f2; "
                 "for r in predictor financial-data; do git -C ~/Documents/GitHub/$r rev-parse --short=12 HEAD; "
                 "git -C ~/Documents/GitHub/$r status --porcelain | grep -vc '^??'; done; "
                 "test -d ~/.local/state/crispdm-data-foundation && echo BANKS_PRESENT || echo BANKS_ABSENT; "
                 "cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.subtree_control; "
                 "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; nproc")
        r = subprocess.run(["bash", "-c", probe], capture_output=True, text=True) if cfg["ssh"] is None \
            else ssh(cfg["ssh"], probe)
        lines = r.stdout.split("\n")
        mem[role] = round(int(lines[0]) / 2**20, 2)
        facts = {"mem_total_gib": mem[role], "predictor": lines[1], "predictor_tracked_dirty": int(lines[2]),
                 "financial_data": lines[3], "financial_data_tracked_dirty": int(lines[4]), "d0_banks": lines[5],
                 "user_cgroup_controllers": lines[6], "gpu_compute_processes": int(lines[7]), "cpus": int(lines[8])}
        if role == "COORDINATOR":
            ok = "memory" in facts["user_cgroup_controllers"] and facts["d0_banks"] == "BANKS_PRESENT"
            title = "coordinator holds the D0 banks and can cap memory per process"
        else:
            ok = (facts["predictor"] == WORKER_BASES["predictor"] and facts["financial_data"] ==
                  WORKER_BASES["financial-data"] and facts["d0_banks"] == "BANKS_ABSENT"
                  and "memory" in facts["user_cgroup_controllers"] and facts["gpu_compute_processes"] == 0)
            title = "worker on earlier checkouts, without D0 banks, GPUs free, memory cgroup available"
        item(f"C161.host.{role}", title, ok, facts, kind="FACT")
    return mem


# ---------------------------------------------------------------- cube
def sec_cube():
    env = {}
    for line in (HOME / ".config/crispdm/olap-loader.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    from sqlalchemy import create_engine, text
    eng = create_engine(f"postgresql://{env['PGUSER']}:{env['PGPASSWORD']}@{env.get('PGHOST', '127.0.0.1')}:"
                        f"{env.get('PGPORT', '5432')}/{env.get('PGDATABASE', 'predictor_olap')}")
    with eng.connect() as c:
        tables = [r[0] for r in c.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public' "
            "AND table_type='BASE TABLE' ORDER BY 1"))]
        counts = {t: c.execute(text(f'SELECT count(*) FROM public."{t}"')).scalar() for t in tables}
    eng.dispose()
    st = subprocess.run(["systemctl", "--user", "show", "crispdm-olap-loader.service", "-p", "ActiveState",
                         "-p", "NRestarts"], capture_output=True, text=True).stdout.split()
    df = [t for t in tables if t.startswith("df_")]
    item("C164.cube_baseline", "cube history captured; no data-foundation table loaded yet; loader active, "
         "never restarted", not df and "ActiveState=active" in st and "NRestarts=0" in st,
         {"base_tables": len(tables), "df_tables": df, "loader": st,
          "row_counts_sha256": hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()[:16],
          "row_counts": counts}, kind="FACT")


# ----------------------------------------------------------- identities
def identities() -> dict:
    ids = {}
    for root in ("public_panels_c126_v1", "public_panels_c126_v2", "lab_evaluation_c137_v1",
                 "lab_delay_cost_c137_v2", "profiles_c130_v1", "snr_calibration_c134_v1", "run_logs"):
        ids[f"state:{root}"] = tree_digest(STATE / root)
    ids["state:synthetic_bank_c128_v1:manifest"] = sha(SYN / "BANK_MANIFEST.json")
    ids["raw:PUBLIC_RAW_MANIFEST.json"] = sha(RAW / "PUBLIC_RAW_MANIFEST.json")
    for p in sorted(EVID.glob("C122_C124_*.json")) + sorted(EVID.glob("D[345]_*DESIGN.v1.json")):
        ids[f"evidence:{p.name}"] = sha(p)
    for p in (FIN_CONTRACTS, FD / "features/census/FINANCIAL_PROFILE_QUEUE.v1.json"):
        ids[f"census:{p.name}"] = sha(p)
    for p in sorted((PRED / "docs/audits").glob("MUSASHI_AUDIT_C122_C145_*.md")) + sorted(
            (PRED / "docs/handoffs").glob("MUSASHI_TO_GENERAL_SATOSHI_C146_C165_*.md")):
        ids[f"order:{p.name}"] = sha(p)
    return ids


def main() -> int:
    print(f"PRE C146-C165 python={sys.version.split()[0]}")
    sec_bases()
    before = identities()
    host_mem = sec_hosts()
    inc = sec_incident()
    sec_projection(inc, host_mem)
    sec_runtime_code()
    sec_causal()
    sec_cube()
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
