"""DMO17: no new scientific GPU child on the preferred host while another heavy child holds its host lease."""
from tests.i6dmo.ops_fakes import GIB, MIB, facts_provider, fit_count, make_campaign, run_worker


def test_dmo17_heavy_host_lease_blocks_sibling_gpu_on_same_host(tmp_path):
    from tools.i6dmo import queue
    q = queue.Queue(tmp_path / "q.sqlite3", clock=lambda: 100.0)
    for unit in ("a", "b"):
        q.ensure_unit(unit, dict(identity_sha256=unit * 64, kind="fit", family="short", device="gpu",
                                 depends_on=[], stage="S"), state="QUEUED")
    held = q.claim("a", owner="w1", role="preferred", host_key="host-preferred", boot_id="b",
                   device_class="rtx5090", heavy=True, ttl=600)
    assert held["result"] == "CLAIMED"
    sibling = q.claim("b", owner="w2", role="preferred", host_key="host-preferred", boot_id="b",
                      device_class="rtx5070ti", heavy=True, ttl=600)
    assert sibling == dict(result="WAIT_RESOURCE", cause="HOST_HEAVY_LEASE_HELD")
    other = q.claim("b", owner="w3", role="secondary", host_key="host-secondary", boot_id="c",
                    device_class="rtx4090", heavy=True, ttl=600)
    assert other["result"] == "CLAIMED"


def test_dmo17_foreign_heavy_gpu_process_refuses_admission(tmp_path):
    counter = tmp_path / "calls.txt"
    root = make_campaign(tmp_path)
    out = run_worker(root, "preferred", counter, provider=facts_provider(foreign_heavy=1))
    assert out["result"] == "WAIT_RESOURCE" and out["cause"] == "FOREIGN_HEAVY_GPU_PROCESS"
    assert fit_count(counter) == 0


def test_dmo17_5090_first_and_never_the_5070ti(tmp_path):
    import json
    counter = tmp_path / "calls.txt"
    root = make_campaign(tmp_path)
    assert run_worker(root, "secondary", counter)["result"] == "IDLE"  # preferred has not refused it
    out = run_worker(root, "preferred", counter)
    spec = json.loads((root / out["attempt_dir"] / "UNIT_SPEC.json").read_text())
    assert spec["device"] == dict(cls="rtx5090", gpu_uuid="GPU-fake-preferred-0")
    assert spec["cap_bytes"] == 7936 * MIB


def test_dmo17_refused_cap_is_never_shrunk_and_routes_to_secondary(tmp_path):
    from tools.i6dmo import queue, worker
    counter = tmp_path / "calls.txt"
    root = make_campaign(tmp_path)
    low = facts_provider(per_role={"preferred": dict(mem_available=6 * GIB)})
    refused = run_worker(root, "preferred", counter, provider=low)
    assert refused["result"] == "WAIT_RESOURCE" and refused["cause"] == "HOST_MEMORY_BELOW_GATE"
    q = queue.Queue(root / "state" / "queue.sqlite3")
    assert q.min_cap("short.fit") == 7936 * MIB
    assert worker.cap_for(q, "short.fit", "rtx4090", initial_cap=7936 * MIB, measured_peak=2 * GIB) == 7936 * MIB
    assert worker.cap_for(q, "long.fit", "rtx5090", initial_cap=7936 * MIB, measured_peak=8 * GIB) == 10 * GIB
    q.close()
    routed = run_worker(root, "secondary", counter, provider=low)
    assert routed["result"] == "COMPLETE" and routed["unit_id"] == "short.fit"
    spec = json.loads((root / routed["attempt_dir"] / "UNIT_SPEC.json").read_text())
    assert spec["device"]["cls"] == "rtx4090" and spec["cap_bytes"] >= 7936 * MIB


import json  # noqa: E402
