"""Finite context bridge integration boundaries, without scientific fits."""
import pytest

from tools.i6dmo import matched_runner as MR, campaign as C


def test_profile_is_distinct_and_tie_retains_reference():
    p = MR.contrast_profile(("mg2l", "mg12c"))
    assert p["arms"] == ("mg12c", "mg2l")
    assert p["horizons"] == (18, 24, 36, 48, 60, 72)
    assert p["tie_keeps"] == "mg12c"
    assert p["context_bridge"] is True


@pytest.mark.parametrize("arms", [("mg12c", "mg12m"), ("mg2l",), ("mg2m", "mg12c")])
def test_no_mixed_or_partial_bridge(arms):
    with pytest.raises(ValueError):
        MR.contrast_profile(arms)


@pytest.mark.parametrize("decision,arm,expected", [("LOWER_SCORE", "mg2l", "SHORT_CONTEXT_SELECTED"),
    ("LOWER_SCORE", "mg12c", "LONG_CONTEXT_RETAINED"),
    ("TIE_RETAIN_CONTROL", "mg12c", "LONG_CONTEXT_RETAINED_WITHIN_TOLERANCE")])
def test_bridge_results_are_automatically_closeable(decision, arm, expected):
    disposition = MR.disposition_of(dict(decision=decision, selected=arm))
    assert disposition == expected
    assert disposition in C.AUTO_CLOSE_DISPOSITIONS


def test_launcher_rejects_output_in_source_and_wrong_config():
    from tools.i6dmo.context_bridge_campaign import validate_config
    with pytest.raises(ValueError):
        validate_config(dict(schema="i6dmo.context_bridge_launch.v1"))
    with pytest.raises(ValueError):
        validate_config(dict(schema="i6dmo.context_bridge_launch.v1", work_root="/tmp/bridge",
            input_dir="/tmp/bridge/input", source_deployment="/tmp/deployment.json"))


def test_timer_shutdown_is_bound_and_does_not_touch_foreign_units(tmp_path, monkeypatch):
    import json
    from tools.i6dmo import context_bridge_campaign as B, deploy
    config = dict(schema="i6dmo.context_bridge_launch.v1", work_root=str(tmp_path),
                  input_dir="/tmp/source", source_deployment="/tmp/deploy.json")
    name = "i6dmo-context-bridge-" + C.sha256_json(config)[:12]
    receipt = tmp_path / "LAUNCHER.json"
    receipt.write_text(json.dumps(dict(name="foreign.timer")))
    calls = []
    monkeypatch.setattr(deploy, "default_systemctl", lambda *args: calls.append(args) or dict(rc=0))
    with pytest.raises(ValueError, match="FOREIGN_LAUNCH"):
        B.stop_launcher(config)
    assert not calls
    receipt.write_text(json.dumps(dict(name=name)))
    B.stop_launcher(config)
    assert calls == [("disable", "--now", name + ".timer")]


def test_effective_deployment_tampering_refused_before_worker(tmp_path):
    import json
    (tmp_path / "private").mkdir()
    spec = dict(roles={"preferred": {"gpu": "fixture"}})
    effective = C._seal(dict(experiment_id="I6-DMO-LONG-SHORT-INPUT-CONTEXT-BRIDGE-R0",
        deployment_sha256=C.sha256_json(spec), design=dict(splits=dict(test=dict(open=False))), test_open=False))
    (tmp_path / "EFFECTIVE_CONFIG.json").write_text(json.dumps(effective))
    path = tmp_path / "private/deployment.json"
    path.write_text(json.dumps(spec))
    assert C.load_deployment(tmp_path) == spec
    spec["roles"]["preferred"]["gpu"] = "tampered"
    path.write_text(json.dumps(spec))
    with pytest.raises(C.Refused, match="DEPLOYMENT_DIGEST"):
        C.load_deployment(tmp_path)


def test_failed_timer_activation_never_records_deployment(tmp_path, monkeypatch):
    from tools.i6dmo import context_bridge_campaign as B, deploy
    def fake_deploy(root, *, systemctl):
        systemctl("enable", "--now", "owned.timer")
        (root / "DEPLOY.json").write_text("{}")
    monkeypatch.setattr(deploy, "deploy", fake_deploy)
    monkeypatch.setattr(deploy, "default_systemctl", lambda *args: dict(rc=1, stdout="denied"))
    with pytest.raises(RuntimeError, match="ACTIVATION_FAILED"):
        B.ensure_deployed(tmp_path)
    assert not (tmp_path / "DEPLOY.json").exists()


def test_atomic_initialization_recovers_partial_without_live_work(tmp_path, monkeypatch):
    import json
    from tools.i6dmo import context_bridge_campaign as B
    private = tmp_path / "deployment.json"
    private.write_text("{}")
    source = tmp_path / "design.json"
    source.write_text("{}")
    root = tmp_path / "campaign"
    partial = tmp_path / "campaign.building"
    partial.mkdir()
    (partial / "partial").write_text("evidence")
    created = []
    def init(paths, private, staging, *, code_commit, data_pins):
        created.append(staging)
        staging.mkdir()
        (staging / "PLAN.json").write_text("{}")
    monkeypatch.setattr(C, "init_matched_contrast", init)
    monkeypatch.setattr(C, "load_effective", lambda p: dict(code_commit="a"*40, data_pins={}, deployment_sha256=C.sha256_json({})))
    monkeypatch.setattr(C, "load_deployment", lambda p: {})
    monkeypatch.setattr(C, "write_status", lambda p: None)
    B.initialize_atomic({"mg2l": str(source)}, private, root, "a"*40, {})
    assert (root / "PLAN.json").exists() and len(created) == 1
    assert len(list(tmp_path.glob("campaign.incomplete-*/partial"))) == 1


def test_missing_storage_refused_before_remote_or_population_write(tmp_path, monkeypatch):
    from tools.i6dmo import context_bridge_campaign as B, storage
    monkeypatch.setattr(storage, "probe_mount", lambda p: dict(fstype="ext4", writable=True))
    monkeypatch.setattr(storage, "run_preflight", lambda *a: dict(returncode=1))
    monkeypatch.setattr(B.L, "_remote", lambda *a: pytest.fail("remote before local storage gate"))
    spec = dict(roles=dict(coordinator=dict(experiment_storage_root="/missing")),
                storage=dict(estimate_bytes=1, reserve_bytes=1))
    with pytest.raises(ValueError, match="STORAGE_NOT_READY"):
        B.storage_preflight(spec, tmp_path)
