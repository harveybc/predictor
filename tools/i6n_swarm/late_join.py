"""Prepare auditable late GPU workers without changing the sealed scientific plan."""
from copy import deepcopy
import json
from pathlib import Path
import time

from tools.i6n_swarm import common, ledger as L, plan as P

SCHEMA = "i6n.swarm.worker_amendment.v1"
ALLOWED_STAGES = {"annual", "multi_output"}


def _contexts(root):
    led = L.Ledger.open_readonly(Path(root) / "state" / "ledger.sqlite3")
    result = {}
    for item in led.dag_nodes():
        summary = item.get("summary") or {}
        if item["stage"] == "search" and summary.get("swarm_context"):
            result[str(item["spec"]["horizon"])] = summary["swarm_context"]
    return result


def _validate(payload, root, spec):
    if spec.get("schema") != SCHEMA:
        raise ValueError("unsupported worker amendment schema")
    if spec.get("campaign_id") != payload["campaign_id"]:
        raise ValueError("campaign_id mismatch")
    if spec.get("plan_file_sha256") != common.file_sha256(Path(root) / "PLAN.json"):
        raise ValueError("plan_file_sha256 mismatch")
    if spec.get("e_plan_sha256") != payload["e_plan_sha256"]:
        raise ValueError("e_plan_sha256 mismatch")
    workers = spec.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("workers must be a nonempty list")
    roles = [w.get("role") for w in workers]
    if any(not isinstance(role, str) or not role for role in roles) or len(set(roles)) != len(roles):
        raise ValueError("worker roles must be unique nonempty strings")
    for worker in workers:
        device = worker.get("device") or {}
        if not str(device.get("cuda_visible_devices", "")).startswith("GPU-"):
            raise ValueError(f"{worker['role']}: explicit GPU UUID required")
        stages = worker.get("fixed_stages")
        if not isinstance(stages, list) or not stages or not set(stages) <= ALLOWED_STAGES:
            raise ValueError(f"{worker['role']}: late workers may run only annual/multi_output fixed stages")
        host = worker.get("host") or {}
        for key in ("python", "workdir", "data_dir", "doin_port", "environment", "shared_data_root"):
            if key not in host:
                raise ValueError(f"{worker['role']}: host.{key} missing")
        if host["environment"].get("CUDA_VISIBLE_DEVICES") != device["cuda_visible_devices"]:
            raise ValueError(f"{worker['role']}: host/device CUDA identity mismatch")


def prepare(root, spec_path):
    root = Path(root)
    plan_bytes = (root / "PLAN.json").read_bytes()
    envelope = P.load(root)
    payload = envelope["payload"]
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    _validate(payload, root, spec)
    amendment_sha = common.sha256_json(spec)
    receipt_dir = root / "status" / "WORKER_AMENDMENTS"
    receipt = receipt_dir / f"{amendment_sha}.json"
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        if existing.get("amendment_sha256") != amendment_sha:
            raise ValueError("existing amendment receipt does not match")
        return dict(state="ALREADY_PREPARED", amendment_sha256=amendment_sha,
                    amendment_receipt=str(receipt), roles=existing["roles"])

    private_path = root / "private" / "campaign.json"
    private_before = common.file_sha256(private_path)
    private = P.private(root)
    for worker in spec["workers"]:
        role = worker["role"]
        if role in private["hosts"]:
            raise ValueError(f"role already exists: {role}")
        host = deepcopy(worker["host"])
        host.update(role=role, ledger_host=False, devices=[deepcopy(worker["device"])],
                    fixed_stages=list(worker["fixed_stages"]),
                    physical_host_key=worker["physical_host_key"])
        if worker.get("desktop_workload"):
            host["desktop_workload"] = True
            host["vram_bytes"] = int(worker.get("vram_bytes", 0))
        sibling = worker.get("sibling_class_id")
        if sibling:
            if sibling not in payload["class_caps"] or payload["class_caps"][sibling]["state"] != "ADMITTED":
                raise ValueError(f"{role}: sibling class is not admitted")
            host["sibling_class"] = {"class_id": sibling}
        private["hosts"][role] = host
        private["doin"].setdefault("peers", {})[role] = worker["doin_address"]

    contexts = _contexts(root) if (root / "state" / "ledger.sqlite3").exists() else payload.get("swarm_contexts", {})
    prepared = {}
    for worker in spec["workers"]:
        role = worker["role"]
        prepared[role] = {
            "node": P.bridge_node_config(payload, private, role, private["hosts"][role]["devices"][0]),
            "doin": P.doin_node_config(payload, private, role, contexts),
        }

    # Do not alter the live operational envelope until every derived config has
    # been generated successfully. Each subsequent write is itself atomic.
    common.atomic_json(private_path, private, mode=0o600)
    files = {}
    for worker in spec["workers"]:
        role = worker["role"]
        directory = root / "private" / "hosts" / role
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        node_path = directory / "node.json"
        doin_path = directory / "doin_node.json"
        common.atomic_json(node_path, prepared[role]["node"], mode=0o600)
        common.atomic_json(doin_path, prepared[role]["doin"], mode=0o600)
        files[role] = {"node_sha256": common.file_sha256(node_path),
                       "doin_sha256": common.file_sha256(doin_path)}

    if (root / "PLAN.json").read_bytes() != plan_bytes:
        raise RuntimeError("sealed PLAN.json changed while preparing late workers")
    document = dict(schema="i6n.swarm.worker_amendment_receipt.v1", campaign_id=payload["campaign_id"],
                    amendment_sha256=amendment_sha, plan_file_sha256=spec["plan_file_sha256"],
                    e_plan_sha256=payload["e_plan_sha256"], private_before_sha256=private_before,
                    private_after_sha256=common.file_sha256(private_path), roles=[w["role"] for w in spec["workers"]],
                    files=files, prepared_at=time.time(), scientific_plan_unchanged=True,
                    no_fit_started=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_json(receipt, document)
    return dict(state="PREPARED", amendment_sha256=amendment_sha,
                amendment_receipt=str(receipt), roles=document["roles"], files=files)
