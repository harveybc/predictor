"""C174: the capacity inventory parses captured probe text, quarantines a GPU without a
handle or named by a declared quarantine, and never records an alias or address."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import df_host_capacity as HC  # noqa: E402

MIB = 1 << 20

GPUS_TWO_WITH_HANDLE_ERROR = """0, NVIDIA GeForce RTX 5070 Ti Laptop GPU, GPU-00000000-1111-2222-3333-444444444444, 00000000:01:00.0, 12227, 11759, 14
Unable to determine the device handle for GPU1: 0000:0A:00.0: Unknown Error
rc=0"""

APPS_WITH_HANDLE_ERROR = """Unable to determine the device handle for GPU1: 0000:0A:00.0: Unknown Error
GPU-00000000-1111-2222-3333-444444444444, 4242, 512"""


def probe_text(meminfo_avail_kib=11416308, gpus=GPUS_TWO_WITH_HANDLE_ERROR, apps=APPS_WITH_HANDLE_ERROR,
               memguard="active", end=True):
    t = f"""@@cpus
32
@@loadavg
3.66 3.68 3.73 3/1017 376381
@@meminfo
MemTotal:       14981196 kB
MemAvailable:   {meminfo_avail_kib} kB
SwapTotal:       4194300 kB
SwapFree:        4194300 kB
@@slice
MemoryMax=8589934592
MemoryHigh=infinity
ControlGroup=/user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice
ActiveState=active
@@slice_cgroup
memory.current 337547264
memory.max 8589934592
@@scopes
crispdm-jobA-abcdef12-1789325173-1047683.scope 1048576 2097152
@@memguard
{memguard}
@@crispdm_run
PRESENT
@@gpus
{gpus}
@@apps
{apps}
"""
    return t + ("@@end\n" if end else "")


def test_meminfo_and_loadavg_parse_to_bytes():
    m = HC.parse_meminfo("MemTotal:       32010112 kB\nMemAvailable:   23595368 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB")
    assert m["mem_total_bytes"] == 32010112 * 1024 and m["mem_available_bytes"] == 23595368 * 1024
    assert HC.parse_loadavg("0.63 0.57 0.35 1/2113 1047019")["load1"] == 0.63


def test_nvidia_smi_handle_error_is_a_device_not_a_crash():
    gpus, errors = HC.parse_nvidia_gpus(GPUS_TWO_WITH_HANDLE_ERROR)
    assert len(gpus) == 1 and gpus[0]["vram_total_bytes"] == 12227 * MIB and gpus[0]["vram_free_bytes"] == 11759 * MIB
    assert len(errors) == 1 and errors[0]["gpu_index"] == 1 and errors[0]["pci_bus_id"] == "0000:0A:00.0"
    assert HC.parse_nvidia_gpus("NO_NVIDIA_SMI") == ([], [])


def test_role_inventory_quarantines_the_gpu_without_handle_and_counts_processes():
    inv = HC.build_role_inventory("WORKER_B", probe_text())
    assert inv["reachable"] and inv["cpus"] == 32 and inv["memguard_active"] and inv["crispdm_run_present"]
    assert inv["batch_slice"]["memory_max_bytes"] == 8 << 30
    assert inv["batch_slice"]["memory_current_bytes"] == 337547264
    assert inv["batch_slice"]["memory_high_bytes"] is None
    assert inv["batch_scopes"][0]["memory_peak_bytes"] == 2097152
    g0, g1 = inv["gpus"]
    assert g0["status"] == HC.SCHEDULABLE and g0["compute_processes"] == 1 and g0["compute_used_bytes"] == 512 * MIB
    assert g1["index"] == 1 and g1["status"] == HC.QUARANTINED and "NO_DEVICE_HANDLE" in g1["quarantine_reason"]
    assert "auditor" in g1["quarantine_reason"]          # the declared default quarantine also names it


def test_declared_quarantine_of_a_healthy_gpu_and_of_a_missing_one():
    healthy = "0, GPU X, GPU-aaaaaaaa-0000-0000-0000-000000000000, 00000000:01:00.0, 16376, 15915, 30\nrc=0"
    q = [{"role": "WORKER_A", "pci_bus_id": "00000000:01:00.0", "reason": "test quarantine"}]
    inv = HC.build_role_inventory("WORKER_A", probe_text(gpus=healthy, apps=""), q)
    assert [g["status"] for g in inv["gpus"]] == [HC.QUARANTINED]
    # WORKER_B's declared GPU 1 is reported even when nvidia-smi does not enumerate it at all
    one = "0, GPU Y, GPU-bbbbbbbb-0000-0000-0000-000000000000, 00000000:01:00.0, 12227, 11759, 14\nrc=0"
    inv_b = HC.build_role_inventory("WORKER_B", probe_text(gpus=one, apps=""))
    assert [(g["index"], g["status"]) for g in inv_b["gpus"]] == [(0, HC.SCHEDULABLE), (1, HC.QUARANTINED)]


def test_incomplete_probe_output_is_unreachable_and_memguard_state_is_read():
    assert HC.build_role_inventory("WORKER_A", probe_text(end=False))["reachable"] is False
    assert HC.build_role_inventory("WORKER_A", probe_text(memguard="inactive"))["memguard_active"] is False


def test_read_inventory_redacts_an_unreachable_worker_and_probes_coordinator_locally():
    calls = []

    def runner(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "bash":
            return subprocess.CompletedProcess(cmd, 0, probe_text(), "")
        if "secret-alias-b" in cmd:
            return subprocess.CompletedProcess(cmd, 255, "", "ssh: connect to host secret-alias-b (10.1.2.3) port 22: "
                                                              "No route to host")
        return subprocess.CompletedProcess(cmd, 0, probe_text(), "")

    roles = {"COORDINATOR": {"ssh": None}, "WORKER_A": {"ssh": "secret-alias-a"}, "WORKER_B": {"ssh": "secret-alias-b"}}
    doc = HC.read_inventory(roles, HC.DEFAULT_QUARANTINE, runner=runner)
    assert calls[0][0] == "bash" and all("ssh" == c[0] for c in calls[1:])
    assert doc["roles"]["WORKER_B"]["reachable"] is False
    text = json.dumps(doc)
    assert "secret-alias" not in text and "10.1.2.3" not in text
    assert len(doc["digest_sha256"]) == 64 and doc["read_at"]
    assert doc["digest_sha256"] == HC.inventory_digest({k: v for k, v in doc.items() if k != "digest_sha256"})
