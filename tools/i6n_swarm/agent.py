"""Worker-host agent (run by the campaign's agent timer and as the node's ExecStartPre).

One invocation, then exit (no resident polling loop):
1. boot recovery: this host's RUNNING attempts from a previous boot -> INTERRUPTED;
2. healthcheck file for admission (boot id, slab, NV_ERR_NO_MEMORY count, GPU probe);
3. fixed units claimable on this host: A's ``freeze`` (data preparation, no fit,
   CPU) and the annual weekly / multi-output fits (admitted GPU child);
4. DOIN node-config sync: render the domains whose swarm context is frozen and
   start the campaign node unit, or restart it only when this host is idle.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tools import i6n_search as search
from tools.i6n_swarm import bridge as B, common, ledger as L, placement, plan as P, runner as R

V3 = "i6n.historical_evaluator.v3"
ACTIVE_SEARCH = ("QUEUED", "RUNNING", "WAIT_RESOURCE", "INTERRUPTED")


def recover(node, led, boot_id=None):
    return led.recover(host=node["host_key"], boot_id=boot_id or common.boot_id())


def healthcheck(node, now=None, run=subprocess.run):
    """Admission facts only; never launches a fit. Missing tools are recorded, not guessed."""
    facts = placement.host_facts(now or time.time(), common.boot_id())
    nv_err = None
    try:
        out = run(["journalctl", "-k", "-b", "--no-pager", "-q", "--grep", "NV_ERR_NO_MEMORY"],
                  capture_output=True, text=True, timeout=30)
        nv_err = len([line for line in out.stdout.splitlines() if line.strip()])
    except (OSError, subprocess.TimeoutExpired):
        pass
    probe = None
    visible = node["device"].get("cuda_visible_devices")
    if visible:
        try:
            out = run(["nvidia-smi", "-i", visible, "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=20)
            probe = out.returncode == 0
            if probe:
                facts["gpu_free_bytes"] = int(out.stdout.strip().split(",")[-1]) * 1024 * 1024
        except (OSError, subprocess.TimeoutExpired, ValueError):
            probe = False
    health = dict(boot_id=facts["boot_id"], checked_at=facts["now"], nv_err_no_memory=nv_err,
                  sunreclaim_bytes=facts["sunreclaim_bytes"], gpu_probe_ok=probe)
    common.atomic_json(Path(node["state_dir"]) / "HEALTH.json", health, mode=0o600)
    return health


def make_admit(node, clock=time.time):
    """Placement decision for one unit from local facts, the healthcheck file and admitted caps."""
    def admit(spec):
        if node.get("fixture"):
            return dict(admit=True, cause=None, exclusive_host=node.get("exclusive_host", True))
        health_path = Path(node["state_dir"]) / "HEALTH.json"
        health = json.loads(health_path.read_text()) if health_path.exists() else None
        facts = placement.host_facts(clock(), common.boot_id(), healthcheck=health)
        if health and health.get("gpu_probe_ok") is not None:
            facts.setdefault("gpu_free_bytes", 0)
        caps = node.get("class_caps", {}).get(spec.get("size_class") or "")
        if caps is None:
            return dict(admit=False, cause=f"CLASS_NOT_ADMITTED: {spec.get('size_class')}", exclusive_host=True)
        class_spec = dict(caps, class_id=spec.get("size_class"))
        sibling = node.get("sibling_class")
        return placement.admit(node["device"]["cls"], facts, class_spec, node.get("measured", {}),
                               sibling_class=sibling)
    return admit


def _freeze(node, led, unit, boot_id):
    spec = unit["spec"]
    claim = led.claim(unit["fit_id"], node["owner"] + "-cpu", node["host_key"], boot_id, "cpu",
                      float(node["lease"]["ttl_seconds"]), exclusive_host=False)
    if claim["result"] != "CLAIMED":
        return claim["result"]
    token, attempt = claim["token"], claim["attempt"]
    cell = spec["cell_config"]
    data_root = node["shared_data_root"]
    try:
        common.read_pinned(cell, "cell config", root=data_root)
        cell_path = common.resolve_pin(cell, data_root)
    except ValueError as exc:
        led.report(unit["fit_id"], token, "FAILED", f"CELL_CONFIG: {exc}")
        return "FAILED"
    safe = unit["dag_node"].replace(":", "_")
    out = Path(node["shared_data_root"]) / "bindings" / node["campaign_id"] / safe / unit["fold"]
    if attempt["index"] > 1:
        out = out.with_name(f"{out.name}.attempt{attempt['index']}")
    # A.4: always a v3 host-relative context (--data-root); a family unit seals a direct multi-output binding.
    child = ([*node["freeze_argv"], "--config", str(cell_path), "--output", str(out), "--experiment",
              unit["experiment"], "--fold", unit["fold"], "--data-root", str(data_root)]
             + (["--score-outer"] if spec.get("score_outer") else [])
             + (["--family", spec["family"]] if spec.get("family") else []))
    caps = node.get("freeze_caps") or {}
    argv = child if node.get("launcher") is None else R.build_argv(
        dict(node["launcher"], memory_bytes=caps["child_memory_bytes"], wall_seconds=caps["child_wall_seconds"]),
        "freeze-" + unit["fit_id"][:20], child)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", I6N_DATA_ROOT=str(data_root))
    result = R.run_attempt(argv, env, "", wall_seconds=caps.get("child_wall_seconds", 0),
                           renew=lambda p=None: led.renew(unit["fit_id"], token, ttl=float(node["lease"]["ttl_seconds"])),
                           renew_interval=float(node["lease"]["renew_seconds"]), ttl=float(node["lease"]["ttl_seconds"]),
                           cwd=node.get("workdir"), log_dir=Path(node["logs_dir"]) / "freeze" / unit["fit_id"][:16],
                           guard=True)
    if result["kind"] != "OK":
        led.report(unit["fit_id"], token, result["kind"], result["cause"])
        return result["kind"]
    path = out / "BINDING.json"
    try:
        binding = json.loads(path.read_text())
        if binding.get("i6n_horizon") != unit["horizon"]:
            raise ValueError("binding horizon mismatch")
        if not node.get("fixture") and binding.get("i6n_evaluation_context", {}).get("schema") != V3:
            raise ValueError("BINDING_NOT_HOST_RELATIVE_V3: production bindings must be frozen with --data-root")
        if spec.get("family") and binding.get("i6n_family") != spec["family"]:
            raise ValueError("family binding does not name this family")
        # Root-relative: another host resolves it against its own shared data root.
        pin = common.relative_pin(dict(path=str(path), sha256=common.file_sha256(path)), node["shared_data_root"])
    except (OSError, ValueError) as exc:
        led.report(unit["fit_id"], token, "FAILED", f"BINDING_INVALID: {exc}")
        return "FAILED"
    # Every evaluating host needs these exact bytes: publish them content-addressed before completing.
    for blob in [path, *[common.resolve_pin(p, data_root) for p in B.context_pins(binding["i6n_evaluation_context"])]]:
        data = Path(blob).read_bytes()
        import hashlib
        led.put_blob(hashlib.sha256(data).hexdigest(), data)
    from tools.i6n_swarm.coordinator import freeze_receipt_document
    led.complete(unit["fit_id"], token, freeze_receipt_document(unit, binding, pin))
    return "COMPLETE"


def unit_receipt_document(unit, receipt, check_genotype):
    spec = unit["spec"]
    search._score(receipt, search._support(spec["request"]["support"]))
    cr = receipt.get("metadata", {}).get("candidate_receipt")
    if not isinstance(cr, dict):
        raise L.ReceiptMismatch("metadata.candidate_receipt missing")
    expected = dict(schema_version="i6n.candidate_receipt.v1", experiment=unit["experiment"], fold=unit["fold"],
                    effective_config_sha256=unit["effective_config_sha256"], seed=common.SEED)
    if check_genotype:
        expected["genotype_canonical_sha256"] = unit["genotype"]
    for key, value in expected.items():
        if cr.get(key) != value:
            raise L.ReceiptMismatch(f"candidate_receipt.{key} does not belong to this unit")
    return dict(fit_id=unit["fit_id"], fold=unit["fold"], experiment=unit["experiment"],
                effective_config_sha256=unit["effective_config_sha256"], genotype_canonical_sha256=unit["genotype"],
                shared_receipt=receipt, candidate_receipt=cr)


def _fit(node, led, unit, boot_id, admit):
    spec = unit["spec"]
    request = dict(config=spec["effective_config"], generation=0, context=spec["request"]["context"],
                   support=spec["request"]["support"])
    bridge = B.Bridge(dict(node, lease=dict(node["lease"], max_wait_seconds=0)), led, boot_id=boot_id, admit=admit)
    state, _ = bridge.run_fold(unit["fit_id"], spec, request,
                               lambda r: unit_receipt_document(unit, r, unit["stage"] == "annual"))
    return state


def run_fixed_units(node, led, stages=("freeze", "annual", "multi_output"), boot_id=None, admit=None, max_passes=5):
    boot_id = boot_id or common.boot_id()
    admit = admit or make_admit(node)
    done = dict(completed=0, failed=0, other=0)
    for _ in range(max_passes):
        progressed = False
        for unit in led.next_claimable(list(stages)):
            if unit["stage"] == "freeze":
                result = _freeze(node, led, unit, boot_id)
            else:
                result = _fit(node, led, unit, boot_id, admit)
            if result == "COMPLETE":
                done["completed"] += 1
                progressed = True
            elif result == "FAILED":
                done["failed"] += 1
                progressed = True
            else:
                done["other"] += 1
        if not progressed:
            break
    return done


def sync_node(node, led, systemctl=None):
    from tools.i6n_swarm.coordinator import default_systemctl
    systemctl = systemctl or default_systemctl
    contexts = {}
    for item in led.dag_nodes():
        summary = item.get("summary") or {}
        if item["stage"] == "search" and item["state"] in ACTIVE_SEARCH and summary.get("swarm_context"):
            contexts[str(item["spec"]["horizon"])] = summary["swarm_context"]
    unit = f"i6n-swarm-{node['campaign_id']}-{node['host_key']}-node.service"
    if not contexts:
        return dict(action="no_ready_domains", unit=unit)
    config = P.render_doin_config(node["doin_template"], contexts)
    path = Path(node["doin_node_config"])
    current = json.loads(path.read_text()) if path.exists() else None
    changed = current != config
    active = systemctl("is-active", unit) == 0
    if active and changed and led.running_on_host(node["host_key"]):
        return dict(action="deferred_restart_host_busy", unit=unit)
    if changed:
        common.atomic_json(path, config, mode=0o600)
    if not active:
        systemctl("start", unit)
        return dict(action="started", unit=unit, domains=sorted(contexts, key=int))
    if changed:
        systemctl("restart", unit)
        return dict(action="restarted", unit=unit, domains=sorted(contexts, key=int))
    return dict(action="unchanged", unit=unit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-config", required=True)
    parser.add_argument("--recover-only", action="store_true")
    args = parser.parse_args(argv)
    node = json.loads(Path(args.node_config).read_text())
    led = B.open_ledger(node)
    report = dict(recovered=recover(node, led))
    if not args.recover_only:
        report["health"] = healthcheck(node)
        report["fixed_units"] = run_fixed_units(node, led)
        report["node"] = sync_node(node, led)
    common.atomic_json(Path(node["state_dir"]) / "AGENT.json", dict(report, at=time.time()), mode=0o600)
    print(json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
