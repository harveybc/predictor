"""C174 test helpers: fake role inventories shaped exactly like df_host_capacity output."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import df_host_capacity as HC  # noqa: E402

GIB = 1 << 30


def gpu(index, total_gib, free_gib, status=HC.SCHEDULABLE, procs=0, uuid=None):
    return {"index": index, "name": f"GPU{index}", "uuid": uuid or f"GPU-fake-{index}-{total_gib}",
            "pci_bus_id": None, "vram_total_bytes": int(total_gib * GIB), "vram_free_bytes": int(free_gib * GIB),
            "vram_used_bytes": int((total_gib - free_gib) * GIB), "compute_processes": procs,
            "compute_used_bytes": 0, "handle_error": None, "status": status, "quarantine_reason": None}


def role(name, *, cpus, total_gib, avail_gib, slice_gib, slice_cur_gib=0.0, load1=0.0, gpus=(), memguard=True,
         reachable=True, scopes=()):
    if not reachable:
        return {"role": name, "reachable": False, "error": "PROBE_FAILED exit=255: <host> unreachable"}
    return {"role": name, "reachable": True, "error": None, "cpus": cpus, "load1": load1, "load5": load1,
            "load15": load1, "mem_total_bytes": int(total_gib * GIB), "mem_available_bytes": int(avail_gib * GIB),
            "swap_total_bytes": 0, "swap_free_bytes": 0,
            "batch_slice": {"present": True, "active_state": "active", "memory_max_bytes": int(slice_gib * GIB),
                            "memory_high_bytes": None, "memory_current_bytes": int(slice_cur_gib * GIB)},
            "batch_scopes": list(scopes), "memguard_active": memguard, "crispdm_run_present": True,
            "gpus": list(gpus)}


def inventory(coord=None, a=None, b=None, stamp="t0"):
    roles = {
        "COORDINATOR": coord if coord is not None else role("COORDINATOR", cpus=16, total_gib=30.5, avail_gib=22,
                                                            slice_gib=14, gpus=[gpu(0, 8, 7)]),
        "WORKER_A": a if a is not None else role("WORKER_A", cpus=32, total_gib=30.6, avail_gib=20, slice_gib=14,
                                                 gpus=[gpu(0, 16, 15.5)]),
        "WORKER_B": b if b is not None else role("WORKER_B", cpus=32, total_gib=14.3, avail_gib=11, slice_gib=8,
                                                 load1=3.7,
                                                 gpus=[gpu(0, 12, 11.4),
                                                       dict(gpu(1, 0, 0, status=HC.QUARANTINED), uuid=None,
                                                            vram_total_bytes=None, vram_free_bytes=None,
                                                            handle_error="Unable to determine the device handle")]),
    }
    doc = {"schema": HC.SCHEMA, "read_at": stamp, "quarantine": list(HC.DEFAULT_QUARANTINE), "roles": roles}
    doc["digest_sha256"] = HC.inventory_digest(doc)
    return doc
