"""Ops lane engineering tests: units/bootstrap rendering, preflight typing, unit_runner adapter, mirrors."""
import json
import os
import stat
from pathlib import Path

import pytest

from tests.i6dmo.ops_fakes import GIB, FakeWarehouse, ROOT, facts, make_campaign, run_worker, write_design


def _systemctl(log):
    def call(*args):
        log.append(args)
        return dict(rc=0, stdout="enabled" if args[0] == "is-enabled" else "active")
    return call


def test_deploy_is_idempotent_and_refuses_foreign_or_closed(tmp_path):
    from tools.i6dmo import campaign, deploy
    root = make_campaign(tmp_path)
    units, log = tmp_path / "units", []
    first = deploy.deploy(root, unit_dir=units, systemctl=_systemctl(log))
    cid = campaign.load_plan(root)["campaign_id"]
    assert first["written"] and all(n.startswith(cid + "-") for n in first["units"])
    assert any(a[:2] == ("enable", "--now") for a in log)
    publish = (units / f"{cid}-publish.service").read_text()
    assert "EnvironmentFile=" in publish and "TOKEN" not in publish
    worker = (units / f"{cid}-preferred-worker.service").read_text()
    assert "--once" in worker and "KillMode=process" in worker
    second = deploy.deploy(root, unit_dir=units, systemctl=_systemctl([]))
    assert second["written"] == [] and second["unchanged"] == first["units"]
    (units / f"{cid}-tick.service").write_text("[Unit]\nDescription=foreign\n")
    with pytest.raises(campaign.Refused, match="FOREIGN_UNIT_FILE"):
        deploy.deploy(root, unit_dir=units, systemctl=_systemctl([]))
    (root / "CLOSE.json").write_text("{}")
    with pytest.raises(campaign.Refused, match="CAMPAIGN_CLOSED_OR_RETIRED"):
        deploy.deploy(root, unit_dir=units, systemctl=_systemctl([]))


def test_bootstrap_template_has_placeholders_only_and_renders_privately(tmp_path):
    from tools.i6dmo import campaign, deploy
    text = deploy.BOOTSTRAP_TEMPLATE.read_text()
    assert "@PRIVATE_SPEC@" in text and "@ROOT@" in text and "/home/" not in text and "/var/tmp" not in text
    with pytest.raises(campaign.Refused, match="BOOTSTRAP_INSIDE_GIT"):
        deploy.render_bootstrap(tmp_path / "spec.json", tmp_path / "root", ROOT / "tools" / "bootstrap.sh")
    out = deploy.render_bootstrap(tmp_path / "spec.json", tmp_path / "root", tmp_path / "private" / "boot.sh")
    path = Path(out["out"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o700 and "@" not in path.read_text()


def test_preflight_types_blockers(tmp_path):
    from tests.i6dmo.ops_fakes import fake_adapter, private_spec
    from tools.i6dmo import preflight
    spec = private_spec(tmp_path)
    spec["roles"]["preferred"]["gpus"][1]["parallel_slot"] = True
    (tmp_path / "spec.json").write_text(json.dumps(spec))

    def probe(role, files):
        f = facts(mem_available=6 * GIB if role == "preferred" else 40 * GIB)
        f["legacy_active_units"] = ["i6n-old.timer"] if role == "secondary" else []
        f["strategy_pin"] = dict(state="FAIL", code="STRATEGY_PIN_UNRESOLVED")
        return f
    doc = preflight.preflight(write_design(tmp_path), tmp_path / "spec.json", adapter=fake_adapter(), probe=probe,
                              data_pins={"short_bundle": {"relpath": "b/BUNDLE.json", "sha256": "0" * 64}},
                              out=tmp_path / "PRECHECK.json")
    assert doc["state"] == "FAIL"
    blockers = set(doc["launch_blockers"])
    assert {"admission_initial_cap:preferred", "legacy_units_stopped:secondary",
            "non_parallel_slot:preferred:rtx5070ti", "data_pin:preferred:short_bundle"} <= blockers
    strategy = next(c for c in doc["checks"] if c["check"] == "strategy_pin:coordinator")
    assert strategy["state"] == "FAIL" and strategy["blocks"] == "PAIR"
    assert "strategy_pin:coordinator" not in blockers  # pair blocker, not a launch blocker


def _runner_spec(tmp_path, bundle_sha):
    design = json.loads(write_design(tmp_path).read_text())
    folder = tmp_path / "attempt"
    folder.mkdir()
    spec = dict(unit_id="short.fit", kind="fit", family="short", identity_sha256="a" * 64, campaign_id="i6dmo-x",
                cadence="NONE", test_open=False, device=dict(cls="rtx5090", gpu_uuid="GPU-fake"),
                effective_config=dict(design=design, code_commit="c" * 40),
                bundle=dict(key="short_bundle", path=str(tmp_path / "bundle"), sha256=bundle_sha))
    return spec, folder


@pytest.mark.parametrize("rc,status", [(0, "COMPLETE"), (3, "FAILED"), (2, "FAILED")])
def test_unit_runner_converts_model_receipt_to_terminal(tmp_path, monkeypatch, rc, status):
    from tools.i6dmo import unit_runner
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "BUNDLE.json").write_text("{}")
    spec, folder = _runner_spec(tmp_path, unit_runner._sha(tmp_path / "bundle" / "BUNDLE.json"))
    monkeypatch.setattr(unit_runner, "TRAINER_MODULE", "tests.i6dmo.ops_fake_train")
    monkeypatch.setenv("I6DMO_FAKE_TRAIN_RC", str(rc))
    monkeypatch.chdir(ROOT)
    assert unit_runner.run_fit(spec, folder / "out", folder / "progress.json", poll=0.01) == 0
    doc = json.loads((folder / "out" / "TERMINAL.json").read_text())
    assert doc["status"] == status and doc["identity_sha256"] == "a" * 64
    if rc == 0:
        assert doc["costs"]["fit_seconds"] == 1.5 and doc["costs"]["cgroup_peak_bytes"] == 123456
        assert doc["artifacts"]["model/MODEL.json"]["path"] == "train/model/MODEL.json"
        assert json.loads((folder / "progress.json").read_text())["total"] == 200
        assert doc["reports"] and doc["reports"][0]["tags"]["test_read"] == "false"
    else:
        assert doc["disposition"] in ("ZERO_NAIVE_SCALE", "CONTRACT_ERROR") and not doc["reports"]


def test_unit_runner_killed_trainer_leaves_no_terminal_and_bad_pin_is_typed(tmp_path, monkeypatch):
    from tools.i6dmo import unit_runner
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "BUNDLE.json").write_text("{}")
    spec, folder = _runner_spec(tmp_path, unit_runner._sha(tmp_path / "bundle" / "BUNDLE.json"))
    monkeypatch.setattr(unit_runner, "TRAINER_MODULE", "tests.i6dmo.ops_fake_train")
    monkeypatch.setenv("I6DMO_FAKE_TRAIN_RC", "137")
    monkeypatch.chdir(ROOT)
    assert unit_runner.run_fit(spec, folder / "out", folder / "progress.json", poll=0.01) == 137
    assert not (folder / "out" / "TERMINAL.json").exists()
    spec["bundle"]["sha256"] = "0" * 64
    other = tmp_path / "other"
    other.mkdir()
    assert unit_runner.run_fit(spec, other / "out", other / "progress.json", poll=0.01) == 0
    assert json.loads((other / "out" / "TERMINAL.json").read_text())["disposition"] == "BUNDLE_PIN_MISMATCH"


def test_missing_bundle_pin_or_hook_waits_for_artifact_without_an_attempt(tmp_path):
    import sys
    from tools.i6dmo import campaign, status
    entries = {k: [sys.executable, "-m", "tools.i6dmo.unit_runner", "--kind", k]
               for k in ("fit", "report", "pair", "final_refit", "final_test")}
    design = write_design(tmp_path)
    spec_path = tmp_path / "spec.json"
    from tests.i6dmo.ops_fakes import fake_adapter, private_spec
    spec_path.write_text(json.dumps(private_spec(tmp_path)))
    root = tmp_path / "root"
    campaign.init_campaign(design, spec_path, root, adapter=fake_adapter(), code_commit="c" * 40,
                           data_pins={"other": {"relpath": "x", "sha256": "d" * 64}}, entrypoints=entries)
    out = run_worker(root, "preferred", tmp_path / "c.txt")
    assert out["result"] == "WAIT_ARTIFACT"
    assert "ENTRY_UNAVAILABLE" in out["cause"] or "BUNDLE_PIN_MISSING:short_bundle" in out["cause"]
    unit = status.compute(root, level="unit", unit="short.fit")
    assert unit["state"] == "WAIT_ARTIFACT" and unit["attempts"] == []


def test_mirrors_are_verified_byte_copies(tmp_path):
    from tools.i6dmo import campaign, mirror
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")
    out = campaign.publish(root, warehouse=FakeWarehouse())
    assert out["mirrors"]["state"] == "VERIFIED"
    entries = mirror.manifest(root)
    assert any(k.startswith("terminals/") for k in entries) and "backup/queue.sqlite3" in entries
    for target in ("mirror-local", "mirror-second"):
        assert (tmp_path / target / "EFFECTIVE_CONFIG.json").read_bytes() == (root / "EFFECTIVE_CONFIG.json").read_bytes()
        assert not (tmp_path / target / "private").exists()
    victim = next((tmp_path / "mirror-second" / "terminals").rglob("CHILD_TERMINAL.json"))
    victim.write_text("tampered")
    assert campaign.publish(root, warehouse=FakeWarehouse())["mirrors"]["state"] == "VERIFIED"  # repaired
    assert os.path.exists(victim) and victim.read_text() != "tampered"
