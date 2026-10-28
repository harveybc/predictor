"""Late GPU workers extend capacity without changing the sealed scientific plan."""
import json
from pathlib import Path

import pytest

from tests import i6n_swarm_fixtures as F
from tools.i6n_swarm import common, late_join, plan as P


def _spec(root, plan_sha256, e_plan_sha256):
    return {
        "schema": late_join.SCHEMA,
        "campaign_id": "fx1",
        "plan_file_sha256": plan_sha256,
        "e_plan_sha256": e_plan_sha256,
        "workers": [
            {
                "role": "desktop_gpu",
                "physical_host_key": "coordinator",
                "doin_address": "127.0.0.1:18472",
                "host": {
                    "python": "/opt/i6n/bin/python",
                    "workdir": "/opt/i6n/predictor",
                    "data_dir": str(root / "hostdata" / "desktop_gpu"),
                    "doin_port": 18472,
                    "environment": {"CUDA_VISIBLE_DEVICES": "GPU-C"},
                    "shared_data_root": str(root / "shared"),
                    "node_command": ["/opt/i6n/bin/python", "-m", "doin_node.cli"],
                },
                "device": {"cls": "rtx4070", "cuda_visible_devices": "GPU-C"},
                "fixed_stages": ["annual", "multi_output"],
                "desktop_workload": True,
                "vram_bytes": 2 * 1024**3,
            },
            {
                "role": "preferred_aux",
                "physical_host_key": "gamma",
                "doin_address": "127.0.0.1:18473",
                "host": {
                    "python": "/opt/i6n/bin/python",
                    "workdir": "/opt/i6n/predictor",
                    "data_dir": str(root / "hostdata" / "preferred_aux"),
                    "doin_port": 18473,
                    "environment": {"CUDA_VISIBLE_DEVICES": "GPU-D"},
                    "shared_data_root": str(root / "shared"),
                    "node_command": ["/opt/i6n/bin/python", "-m", "doin_node.cli"],
                },
                "device": {"cls": "rtx5070ti", "cuda_visible_devices": "GPU-D"},
                "fixed_stages": ["annual", "multi_output"],
                "sibling_class_id": "S1",
            },
        ],
    }


def test_prepare_is_idempotent_and_preserves_plan_bytes(tmp_path):
    cfg = F.private_config(tmp_path)
    root = tmp_path / "campaign"
    payload = P.build(cfg, root)
    before = (root / "PLAN.json").read_bytes()
    spec = _spec(root, common.file_sha256(root / "PLAN.json"), payload["e_plan_sha256"])
    spec_path = tmp_path / "amendment.json"
    spec_path.write_text(json.dumps(spec))

    first = late_join.prepare(root, spec_path)
    second = late_join.prepare(root, spec_path)

    assert first["state"] == "PREPARED" and second["state"] == "ALREADY_PREPARED"
    assert (root / "PLAN.json").read_bytes() == before
    private = P.private(root)
    assert {"desktop_gpu", "preferred_aux"} <= set(private["hosts"])
    desktop = json.loads((root / "private/hosts/desktop_gpu/node.json").read_text())
    assert desktop["fixed_stages"] == ["annual", "multi_output"]
    assert all(v["desktop_workload"] and v["vram_bytes"] == 2 * 1024**3
               for v in desktop["class_caps"].values())
    aux = json.loads((root / "private/hosts/preferred_aux/node.json").read_text())
    assert aux["sibling_class"]["class_id"] == "S1"
    assert aux["exclusive_host"] is False
    assert all("desktop_workload" not in v for v in aux["class_caps"].values())
    assert Path(first["amendment_receipt"]).exists()


def test_prepare_rejects_wrong_plan_identity(tmp_path):
    cfg = F.private_config(tmp_path)
    root = tmp_path / "campaign"
    payload = P.build(cfg, root)
    spec = _spec(root, "0" * 64, payload["e_plan_sha256"])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="plan_file_sha256"):
        late_join.prepare(root, path)


def test_prepare_does_not_mutate_private_config_when_derivation_fails(tmp_path, monkeypatch):
    cfg = F.private_config(tmp_path)
    root = tmp_path / "campaign"
    payload = P.build(cfg, root)
    before = (root / "private/campaign.json").read_bytes()
    spec = _spec(root, common.file_sha256(root / "PLAN.json"), payload["e_plan_sha256"])
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(spec))

    def fail(*_args, **_kwargs):
        raise RuntimeError("derived config failed")

    monkeypatch.setattr(P, "bridge_node_config", fail)
    with pytest.raises(RuntimeError, match="derived config failed"):
        late_join.prepare(root, path)

    assert (root / "private/campaign.json").read_bytes() == before
    assert not (root / "status/WORKER_AMENDMENTS").exists()
