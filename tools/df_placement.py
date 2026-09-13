#!/usr/bin/env python3
"""C174 (order 2026-09-13): pure, deterministic job placement on a live capacity inventory.

No I/O. `place(job, inventory, running, policy)` returns one decision:

* PLACED       - a role (and a GPU for a GPU job) where the job fits NOW;
* WAIT         - it fits some reachable role's ceiling, but not the capacity free now
                 (memory in use, concurrency cap, CPUs busy); retry after a fresh inventory;
* SPLIT        - it fits no role's ceiling, and its declared chunk plan gives the fewest
                 chunks that each fit some role's ceiling; the chunks replace the job;
* UNPLACEABLE  - refused, with the reason for every role.

A job declares (dict):
    job_id, cpu_bytes (peak RAM estimate), gpu_bytes (0 = CPU job), cpus,
    optional roles (affinity), gpu_index (GPU affinity), and split: either
    {"max_chunks": K, "fixed_cpu_bytes": F, "fixed_gpu_bytes": G} (chunk peak =
    fixed + ceil((total - fixed) / n); argv elements may use {chunk_index} and
    {n_chunks}) or a callable split_fn(job, n) -> list of chunk jobs.

Margins (declared constants, overridable through policy):

    request_bytes  = cpu_bytes * (1 + RAM_MARGIN_RATIO), rounded up to a MiB.
                     This is the MemoryMax crispdm-run enforces. With 0.25 the estimate
                     is 80% of MemoryMax, below crispdm-run's MemoryHigh (90%).
    host headroom  = MemAvailable - HOST_RESERVE_BYTES - outstanding reservations.
                     5 GiB = crispdm-memguard's 4 GiB soft stop + 1 GiB; stricter than
                     crispdm-run's own 3 GiB refusal.
    slice headroom = slice MemoryMax - slice memory.current - outstanding reservations.
    RAM fits       : request_bytes <= min(host headroom, slice headroom).
    VRAM need      = gpu_bytes + max(VRAM_MARGIN_MIN_BYTES, gpu_bytes * VRAM_MARGIN_RATIO),
                     fits under free VRAM minus outstanding GPU reservations, on a
                     SCHEDULABLE GPU only; with exclusive_gpu (default) the GPU must also
                     have no compute process and no reservation.
    CPU slots      = cpus - max(reserved cpus, ceil(load1)) >= job cpus.
    outstanding    = sum over this dispatcher's running jobs on the role of
                     max(0, reserved - observed current) (the live reading already
                     counts what they use).
    ceiling (static, for WAIT versus SPLIT/UNPLACEABLE): request_bytes <=
                     min(slice MemoryMax, MemTotal - HOST_RESERVE_BYTES), cpus <= cpu count,
                     VRAM need <= total VRAM of a SCHEDULABLE GPU.

GPU jobs are refused unless policy allow_gpu is True (default False: the D2 order
forbids GPU use). A QUARANTINED_NOT_SCHEDULABLE GPU is never chosen. A role is only
eligible when reachable, crispdm-run is installed, crispdm-memguard is active and
the batch slice has a finite MemoryMax.

Tie-breaking (deterministic): GPU jobs by the most free VRAM after placement, then
CPU jobs and GPU ties by the most RAM headroom after placement, then the most free
CPU slots after placement, then role order COORDINATOR, WORKER_A, WORKER_B, then
GPU index.
"""
from __future__ import annotations

import math

ROLES = ("COORDINATOR", "WORKER_A", "WORKER_B")
MIB = 1 << 20
GIB = 1 << 30
HOST_RESERVE_BYTES = 5 * GIB
RAM_MARGIN_RATIO = 0.25
VRAM_MARGIN_RATIO = 0.10
VRAM_MARGIN_MIN_BYTES = 512 * MIB
DEFAULT_ROLE_CAPS = {"COORDINATOR": 2, "WORKER_A": 3, "WORKER_B": 2}
SCHEDULABLE = "SCHEDULABLE"
DECISIONS = ("PLACED", "WAIT", "SPLIT", "UNPLACEABLE")


def default_policy(**over) -> dict:
    p = {"allow_gpu": False, "exclusive_gpu": True, "role_caps": dict(DEFAULT_ROLE_CAPS),
         "host_reserve_bytes": HOST_RESERVE_BYTES, "ram_margin_ratio": RAM_MARGIN_RATIO,
         "vram_margin_ratio": VRAM_MARGIN_RATIO, "vram_margin_min_bytes": VRAM_MARGIN_MIN_BYTES}
    p.update(over)
    return p


def request_bytes(cpu_bytes: int, policy: dict) -> int:
    raw = math.ceil(int(cpu_bytes) * (1 + policy["ram_margin_ratio"]))
    return int(math.ceil(raw / MIB) * MIB)


def vram_need(gpu_bytes: int, policy: dict) -> int:
    if not gpu_bytes:
        return 0
    return int(gpu_bytes) + max(int(policy["vram_margin_min_bytes"]),
                                int(math.ceil(int(gpu_bytes) * policy["vram_margin_ratio"])))


def validate_job(job: dict) -> list:
    p = []
    if not isinstance(job.get("job_id"), str) or not job["job_id"]:
        p.append("job_id: non-empty string required")
    for k in ("cpu_bytes",):
        if type(job.get(k)) is not int or job[k] <= 0:
            p.append(f"{k}: positive integer required")
    if type(job.get("gpu_bytes", 0)) is not int or job.get("gpu_bytes", 0) < 0:
        p.append("gpu_bytes: non-negative integer required")
    if type(job.get("cpus", 1)) is not int or job.get("cpus", 1) < 1:
        p.append("cpus: integer >= 1 required")
    roles = job.get("roles")
    if roles is not None and (not isinstance(roles, list) or any(r not in ROLES for r in roles)):
        p.append(f"roles: a list drawn from {list(ROLES)}")
    return p


def _outstanding(running: list, role: str) -> tuple:
    ram = cpus = 0
    gpu = {}
    n = 0
    for r in running:
        if r["role"] != role:
            continue
        n += 1
        ram += max(0, int(r["request_bytes"]) - int(r.get("observed_current_bytes") or 0))
        cpus += int(r.get("cpus", 1))
        if r.get("gpu_uuid"):
            gpu[r["gpu_uuid"]] = gpu.get(r["gpu_uuid"], 0) + max(
                0, int(r.get("gpu_reserved_bytes", 0)) - int(r.get("observed_gpu_bytes") or 0))
    return n, ram, cpus, gpu


def _role_blocker(inv: dict):
    if not inv or not inv.get("reachable"):
        return "UNREACHABLE" + (f": {inv.get('error')}" if inv and inv.get("error") else "")
    if not inv.get("crispdm_run_present"):
        return "CRISPDM_RUN_MISSING"
    if not inv.get("memguard_active"):
        return "MEMGUARD_NOT_ACTIVE"
    sl = inv.get("batch_slice") or {}
    if not sl.get("memory_max_bytes"):
        return "BATCH_SLICE_WITHOUT_FINITE_MEMORYMAX"
    if inv.get("mem_available_bytes") is None or inv.get("mem_total_bytes") is None or not inv.get("cpus"):
        return "INVENTORY_INCOMPLETE"
    return None


def _gpu_candidates(job, inv, policy):
    want = job.get("gpu_index")
    return [g for g in inv.get("gpus", []) if g.get("status") == SCHEDULABLE and g.get("uuid")
            and (want is None or g.get("index") == want)]


def evaluate_role(job: dict, role: str, inv: dict, running: list, policy: dict) -> dict:
    """-> {"fits_now", "fits_ceiling", "reason", "score", "gpu"} for one role."""
    out = {"role": role, "fits_now": False, "fits_ceiling": False, "reason": None, "score": None, "gpu": None}
    blocker = _role_blocker(inv)
    if blocker:
        out["reason"] = blocker
        return out
    req = request_bytes(job["cpu_bytes"], policy)
    cpus_need = int(job.get("cpus", 1))
    need_v = vram_need(job.get("gpu_bytes", 0), policy)
    sl = inv["batch_slice"]
    reserve = int(policy["host_reserve_bytes"])
    ceiling_ram = min(int(sl["memory_max_bytes"]), int(inv["mem_total_bytes"]) - reserve)
    gpus = _gpu_candidates(job, inv, policy) if need_v else []
    ceiling_reasons = []
    if req > ceiling_ram:
        ceiling_reasons.append(f"RAM_REQUEST_{req}_OVER_CEILING_{ceiling_ram}")
    if cpus_need > int(inv["cpus"]):
        ceiling_reasons.append(f"CPUS_{cpus_need}_OVER_{inv['cpus']}")
    if need_v:
        if not gpus:
            ceiling_reasons.append("NO_SCHEDULABLE_GPU" + ("" if job.get("gpu_index") is None
                                                           else f"_AT_INDEX_{job['gpu_index']}"))
        elif not any(need_v <= int(g["vram_total_bytes"]) for g in gpus):
            ceiling_reasons.append(f"VRAM_NEED_{need_v}_OVER_EVERY_GPU_TOTAL")
    if ceiling_reasons:
        out["reason"] = "; ".join(ceiling_reasons)
        return out
    out["fits_ceiling"] = True

    n_run, ram_out, cpus_out, gpu_out = _outstanding(running, role)
    cap = int(policy["role_caps"].get(role, 0))
    host_head = int(inv["mem_available_bytes"]) - reserve - ram_out
    slice_head = int(sl["memory_max_bytes"]) - int(sl.get("memory_current_bytes") or 0) - ram_out
    ram_head = min(host_head, slice_head)
    load = inv.get("load1")
    busy = max(cpus_out, int(math.ceil(load)) if load is not None else 0)
    cpu_free = int(inv["cpus"]) - busy
    now_reasons = []
    if n_run >= cap:
        now_reasons.append(f"ROLE_AT_CONCURRENCY_CAP_{cap}")
    if req > ram_head:
        now_reasons.append(f"RAM_REQUEST_{req}_OVER_HEADROOM_{ram_head}"
                           f"(host_{host_head},slice_{slice_head})")
    if cpus_need > cpu_free:
        now_reasons.append(f"CPUS_{cpus_need}_OVER_FREE_{cpu_free}")
    best_gpu = None
    if need_v:
        fitting = []
        for g in gpus:
            if policy.get("exclusive_gpu", True) and ((g.get("compute_processes") or 0) > 0 or g["uuid"] in gpu_out):
                continue
            free_after = int(g["vram_free_bytes"]) - gpu_out.get(g["uuid"], 0) - need_v
            if free_after >= 0:
                fitting.append((free_after, g))
        if not fitting:
            now_reasons.append(f"VRAM_NEED_{need_v}_OVER_FREE_ON_EVERY_SCHEDULABLE_GPU")
        else:
            fitting.sort(key=lambda t: (-t[0], t[1]["index"]))
            best_gpu = fitting[0]
    if now_reasons:
        out["reason"] = "; ".join(now_reasons)
        return out
    out["fits_now"] = True
    out["reason"] = f"FITS ram_headroom_after={ram_head - req} cpu_free_after={cpu_free - cpus_need}" + (
        f" vram_free_after={best_gpu[0]}" if best_gpu else "")
    out["score"] = {"vram_free_after": best_gpu[0] if best_gpu else 0, "ram_headroom_after": ram_head - req,
                    "cpu_free_after": cpu_free - cpus_need}
    if best_gpu:
        g = best_gpu[1]
        out["gpu"] = {"index": g["index"], "uuid": g["uuid"], "reserved_bytes": need_v}
    return out


def split_job(job: dict, n: int) -> list:
    """Chunk jobs of a splittable job, or [] when it is not splittable into n."""
    fn = job.get("split_fn")
    if callable(fn):
        return list(fn(job, n) or [])
    sp = job.get("split")
    if not isinstance(sp, dict) or n > int(sp.get("max_chunks", 1)) or n < 2:
        return []
    fixed_c, fixed_g = int(sp.get("fixed_cpu_bytes", 0)), int(sp.get("fixed_gpu_bytes", 0))
    per_c = fixed_c + int(math.ceil(max(0, job["cpu_bytes"] - fixed_c) / n))
    g = int(job.get("gpu_bytes", 0))
    per_g = (fixed_g + int(math.ceil(max(0, g - fixed_g) / n))) if g else 0
    chunks = []
    for i in range(n):
        c = {k: v for k, v in job.items() if k not in ("split", "split_fn")}
        c.update(job_id=f"{job['job_id']}.chunk-{i + 1}-of-{n}", cpu_bytes=per_c, gpu_bytes=per_g,
                 parent_job_id=job["job_id"], chunk_index=i, n_chunks=n)
        if isinstance(job.get("argv"), list):
            c["argv"] = [a.replace("{chunk_index}", str(i)).replace("{n_chunks}", str(n)) for a in job["argv"]]
        chunks.append(c)
    return chunks


def _max_chunks(job) -> int:
    if callable(job.get("split_fn")):
        return int(job.get("max_chunks", 64))
    sp = job.get("split")
    return int(sp.get("max_chunks", 1)) if isinstance(sp, dict) else 1


def place(job: dict, inventory: dict, running: list | None = None, policy: dict | None = None) -> dict:
    policy = policy or default_policy()
    running = list(running or [])
    problems = validate_job(job)
    base = {"job_id": job.get("job_id"), "decision": None, "role": None, "gpu": None, "request_bytes": None,
            "reason": None, "per_role": {}, "chunks": None,
            "inventory_digest": inventory.get("digest_sha256")}
    if problems:
        return dict(base, decision="UNPLACEABLE", reason="INVALID_JOB: " + "; ".join(problems))
    base["request_bytes"] = request_bytes(job["cpu_bytes"], policy)
    if job.get("gpu_bytes", 0) > 0 and not policy.get("allow_gpu"):
        return dict(base, decision="UNPLACEABLE", reason="GPU_JOB_REFUSED_ALLOW_GPU_IS_FALSE",
                    per_role={r: "GPU_NOT_ALLOWED" for r in ROLES})
    roles_inv = inventory.get("roles", {})
    allowed = job.get("roles") or list(ROLES)
    evals = {}
    for role in ROLES:
        if role not in allowed:
            base["per_role"][role] = "EXCLUDED_BY_AFFINITY"
            continue
        e = evaluate_role(job, role, roles_inv.get(role), running, policy)
        evals[role] = e
        base["per_role"][role] = e["reason"]
    now = [e for e in evals.values() if e["fits_now"]]
    if now:
        best = sorted(now, key=lambda e: (-e["score"]["vram_free_after"], -e["score"]["ram_headroom_after"],
                                          -e["score"]["cpu_free_after"], ROLES.index(e["role"]),
                                          e["gpu"]["index"] if e["gpu"] else -1))[0]
        return dict(base, decision="PLACED", role=best["role"], gpu=best["gpu"],
                    reason=f"{best['role']}: {best['reason']}")
    if any(e["fits_ceiling"] for e in evals.values()):
        return dict(base, decision="WAIT", reason="FITS_A_CEILING_BUT_NOT_THE_CAPACITY_FREE_NOW")
    kmax = _max_chunks(job)
    for n in range(2, kmax + 1):
        chunks = split_job(job, n)
        if not chunks:
            continue
        if all(any(evaluate_role(c, r, roles_inv.get(r), [], policy)["fits_ceiling"]
                   for r in (c.get("roles") or ROLES)) for c in chunks):
            return dict(base, decision="SPLIT", chunks=chunks, n_chunks=n,
                        reason=f"FITS_NO_ROLE_CEILING; SPLIT_INTO_{n}_CHUNKS_EACH_FITTING_A_CEILING")
    unreachable = [r for r, e in evals.items() if str(e["reason"]).startswith("UNREACHABLE")]
    if unreachable:
        return dict(base, decision="WAIT", reason="FITS_NO_REACHABLE_ROLE; UNREACHABLE_ROLES_MAY_FIT: "
                                                  + ",".join(unreachable))
    tail = f"; NOT_SPLITTABLE_WITHIN_{kmax}_CHUNKS" if kmax > 1 else "; NOT_SPLITTABLE"
    return dict(base, decision="UNPLACEABLE", reason="FITS_NO_ROLE_CEILING" + tail)


def plan(jobs: list, inventory: dict, policy: dict | None = None) -> list:
    """Dry-run: place the jobs in order on one inventory, each PLACED job reserving its request
    (as if launched and not yet using memory). Nothing is launched."""
    policy = policy or default_policy()
    running, rows = [], []
    queue = list(jobs)
    while queue:
        job = queue.pop(0)
        d = place(job, inventory, running, policy)
        row = {k: d[k] for k in ("job_id", "decision", "role", "gpu", "request_bytes", "reason", "per_role")}
        rows.append(row)
        if d["decision"] == "PLACED":
            running.append({"role": d["role"], "request_bytes": d["request_bytes"], "cpus": job.get("cpus", 1),
                            "gpu_uuid": (d["gpu"] or {}).get("uuid"),
                            "gpu_reserved_bytes": (d["gpu"] or {}).get("reserved_bytes", 0)})
        elif d["decision"] == "SPLIT":
            row["n_chunks"] = d["n_chunks"]
            queue[0:0] = d["chunks"]
    return rows
