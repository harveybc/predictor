"""Deployment owner: admission, effective config, remote byte verification and join gate.

No ssh, no systemd, no fit: remote hosts are directories under tmp_path and the
campaign CLI is a recorded fake. Synthetic plans only.
"""

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from tools import i6n_swarm_deploy as dep

GIB = 1024 ** 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(uid, kind, rows):
    support = {"MANAGEMENT": {str(h): {s: {"n": n} for s, n in rows[h].items()} for h in rows}}
    return {"unit_id": uid, "kind": kind, "data": {"state": "READY"}, "support": support}


H = (2, 12, 36)


def c_plan():
    fold = {2: {"TRAIN": 19000, "INNER": 300}, 12: {"TRAIN": 17000, "INNER": 300},
            36: {"TRAIN": 11400, "INNER": 240}}
    week = {2: {"TRAIN": 18300, "INNER": 350, "VALIDATION": 94}, 12: {"TRAIN": 16100, "INNER": 310, "VALIDATION": 84},
            36: {"TRAIN": 11450, "INNER": 230, "VALIDATION": 60}}
    return {"plan_sha256": "p" * 64, "horizons_hours": list(H),
            "units": [unit("2023Q1_DCA", "INNER_FOLD", fold), unit("W2024-01-07", "WEEKLY_REFIT", week)]}


CLASSES = [{"class_id": "S1", "max_samples": 25, "max_width": 32},
           {"class_id": "S2", "max_samples": 55, "max_width": 64},
           {"class_id": "S3", "max_samples": 433, "max_width": 128}]
GUARDS = {"max_samples": 433, "max_materialized_bytes": 2 * GIB, "max_batch_bytes": 64 * 1024 ** 2}


def receipt(tmp_path, name, peak, state="COMPLETED"):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"state": state, "observed_peak_bytes": peak}))
    return {"path": str(p), "sha256": sha(p)}


def pilots(tmp_path):
    return {"S1": {"receipt": receipt(tmp_path, "s1", 2410315776), "horizon": 12, "fold": "2023Q1_DCA",
                   "samples": 25, "width": 32},
            "S2": {"receipt": receipt(tmp_path, "s2", 2390552576), "horizon": 36, "fold": "2023Q1_DCA",
                   "samples": 55, "width": 64}}


def test_class_admission_keeps_the_margin_over_the_declared_worst_case(tmp_path):
    a = dep.class_admission(c_plan(), CLASSES, pilots(tmp_path), GUARDS, child_wall_seconds=3720)
    s1, s2, s3 = a["S1"], a["S2"], a["S3"]
    assert s3 == {"admitted": False, "cause": "NO_COMPLETED_PILOT_RECEIPT"}
    pilot_bytes = (17000 + 300) * dep.row_bytes(25)
    worst = (19000 + 300) * dep.row_bytes(25)  # the h2 fold unit has the most rows
    assert s1["coverage_adjustment_bytes"] == worst - pilot_bytes > 0
    assert s1["child_memory_bytes"] == -(-5 * (2410315776 + worst - pilot_bytes) // 4)
    assert s1["child_memory_bytes"] >= 1.25 * 2410315776
    assert s2["coverage_derivation"]["worst"]["horizon"] == 12  # 55 samples first reachable above h2
    assert dep.max_samples_at(2, 55) == 25 and dep.max_samples_at(12, 55) == 55


def test_class_admission_refuses_off_corner_pilots_guards_and_incomplete_receipts(tmp_path):
    p = pilots(tmp_path)
    p["S1"]["samples"] = 13
    p["S2"]["receipt"] = receipt(tmp_path, "s2f", 1, state="FAILED")
    a = dep.class_admission(c_plan(), CLASSES, p, GUARDS, child_wall_seconds=3720)
    assert a["S1"]["cause"] == "PILOT_NOT_AT_CLASS_CORNER" and a["S2"]["cause"] == "PILOT_NOT_COMPLETED"
    small = dict(GUARDS, max_batch_bytes=1024)
    a = dep.class_admission(c_plan(), CLASSES, pilots(tmp_path), small, child_wall_seconds=3720)
    assert a["S1"]["cause"].startswith("CELL_GUARD_EXCEEDED")
    tampered = pilots(tmp_path)
    Path(tampered["S1"]["receipt"]["path"]).write_text("{}")
    with pytest.raises(dep.DeployError) as e:
        dep.class_admission(c_plan(), CLASSES, tampered, GUARDS, child_wall_seconds=3720)
    assert e.value.code == "PILOT_RECEIPT_DIGEST"


def pinned(tmp_path, name, body):
    p = tmp_path / "data" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body))
    return {"path": str(p), "sha256": sha(p)}


def spec(tmp_path):
    inv = [{"role": "preferred", "host": "pref.invalid", "fit_python": "/opt/py/bin/python",
            "admission": "/opt/bin/crispdm-run", "environment": {"CUDA_VISIBLE_DEVICES": "GPU-A"}},
           {"role": "secondary", "host": "sec.invalid", "fit_python": "/opt/py/bin/python",
            "admission": "/opt/bin/crispdm-run", "environment": {"CUDA_VISIBLE_DEVICES": "GPU-B"}}]
    (tmp_path / "WORKERS.json").write_text(json.dumps(inv))
    s = {"schema": dep.SPEC_SCHEMA, "campaign_id": "prod1", "experiment": "exp", "pin": "a" * 40,
         "repos": {"predictor": {"path": str(tmp_path), "pin": "a" * 40}}, "code_dir": str(tmp_path / "code"),
         "campaign_root": str(tmp_path / "root"), "data_root": str(tmp_path / "data"),
         "inventory": str(tmp_path / "WORKERS.json"),
         "c_plan": pinned(tmp_path, "c_plan.json", {"x": 1}), "c_views": pinned(tmp_path, "views.json", {"x": 2}),
         "multi_output": pinned(tmp_path, "mo.json", {"x": 3}), "search_plan": pinned(tmp_path, "e.json", {"x": 4}),
         "pilots": {}, "ports": {"ledger": 18790, "doin": {"preferred": 18470, "secondary": 18471}},
         "workers": {"secondary": {"start": "immediate"},
                     "preferred": {"start": "after_pilot", "pilot": {"unit": "pilot.service", "code_dir": "/opt/c",
                                                                    "python": "/opt/py/bin/python",
                                                                    "config": "/opt/cfg.json"}}},
         "deploy_dir": str(tmp_path / "deploy")}
    return s, {r["role"]: r for r in inv}


def admission_ok(tmp_path):
    return {"S1": {"admitted": True, "receipt": receipt(tmp_path, "r1", 10), "child_wall_seconds": 3720,
                   "small_pilot": False, "coverage_adjustment_bytes": 5, "coverage_derivation_sha256": "d" * 64},
            "S3": {"admitted": False, "cause": "NO_COMPLETED_PILOT_RECEIPT"}}


def test_effective_config_resolves_every_argument_and_validates(tmp_path):
    s, inv = spec(tmp_path)
    addresses = {"coordinator": "10.0.0.1", "preferred": "10.0.0.2", "secondary": "10.0.0.3"}
    cfg = dep.effective_campaign_config(s, inv, admission_ok(tmp_path), addresses, str(tmp_path / "token"))
    dep.validate_campaign_config(cfg)
    roles = {h["role"]: h for h in cfg["hosts"]}
    assert set(roles) == {"coordinator", "secondary", "preferred"} and roles["coordinator"]["devices"] == []
    assert roles["coordinator"]["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert roles["preferred"]["devices"] == [{"cls": "rtx5090", "cuda_visible_devices": "GPU-A"}]
    assert roles["secondary"]["environment"]["I6N_DATA_ROOT"] == s["data_root"]
    assert cfg["class_admission"].keys() == {"S1"} and cfg["seed"] == 6102026 and cfg["mode"] == "production"
    assert cfg["ledger"]["url"] == "http://10.0.0.1:18790" and cfg["doin"]["shared_min_peers"] == 0
    assert cfg["launcher"]["crispdm_run"] == "/opt/bin/crispdm-run"
    bad = json.loads(json.dumps(cfg))
    bad["hosts"][1]["workdir"] = "<pinned predictor checkout>"
    with pytest.raises(dep.DeployError) as e:
        dep.validate_campaign_config(bad)
    assert e.value.code == "PLACEHOLDER_LEFT"
    bad = json.loads(json.dumps(cfg))
    bad["hosts"][1]["doin_port"] = 18790
    with pytest.raises(dep.DeployError):
        dep.validate_campaign_config(bad)
    Path(s["c_plan"]["path"]).write_text("changed")
    with pytest.raises(dep.DeployError) as e:
        dep.validate_campaign_config(cfg)
    assert e.value.code == "INPUT_PIN_MISMATCH"
    with pytest.raises(dep.DeployError):
        dep.effective_campaign_config(s, inv, {"S3": admission_ok(tmp_path)["S3"]}, addresses, "t")


class FakeHost(dep.Remote):
    """A remote host whose filesystem is a directory; understands the few commands the deployer runs."""

    def __init__(self, base, pilot=None, corrupt=()):
        self.base, self.pilot, self.corrupt, self.commands = Path(base), pilot or {}, set(corrupt), []
        self.host = "fake.invalid"

    def map(self, path):
        return self.base / str(path).lstrip("/")

    def put(self, local, remote):
        dst = self.map(remote)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = Path(local).read_bytes()
        if remote in self.corrupt:
            data = data + b"x"
        dst.write_bytes(data)

    def put_many(self, paths):
        for path in paths:
            self.put(Path(path), path)

    def run(self, command, check=True):
        self.commands.append(command)
        out, rc = "", 0
        if command.startswith("sha256sum -c --quiet "):
            manifest = self.map(shlex.split(command)[3]).read_text().splitlines()
            bad = []
            for line in manifest:
                digest, path = line.split("  ", 1)
                f = self.map(path)
                if not f.exists() or sha(f) != digest:
                    bad.append(f"{path}: FAILED")
            out, rc = "\n".join(bad), (1 if bad else 0)
        elif command.startswith("systemctl --user is-active"):
            out, rc = self.pilot.get("active", "inactive"), 0
        elif "cost_pilot" in command:
            out = json.dumps(self.pilot["status"])
        elif command.startswith("sha256sum "):
            f = self.map(shlex.split(command)[1])
            out = (sha(f) if f.exists() else "0" * 64) + "  x"
        result = subprocess.CompletedProcess(command, rc, out, "")
        if check and rc:
            raise dep.DeployError("REMOTE_COMMAND_FAILED", command[:40])
        return result


def deployer(tmp_path, hosts):
    s, inv = spec(tmp_path)
    s["spec_path"] = str(tmp_path / "spec.json")
    (tmp_path / "data" / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "inputs" / "INPUT.npz").write_bytes(b"npz-bytes")
    root = Path(s["campaign_root"])
    (root / "private").mkdir(parents=True)
    (root / "PLAN.json").write_text("{}")
    (root / "private" / "campaign.json").write_text("{}")
    Path(s["deploy_dir"]).mkdir(parents=True)
    (Path(s["deploy_dir"]) / "ledger.token").write_text("t")
    code = Path(s["code_dir"]) / "predictor"
    code.mkdir(parents=True)
    (code / "mod.py").write_text("x = 1\n")
    d = dep.Deployer.__new__(dep.Deployer)
    d.spec, d.inv, d.remotes, d.local, d.state, d.secrets = s, inv, hosts, None, {"steps": {}}, []
    d.code_manifest = dep.manifest_lines({str(code / "mod.py"): code / "mod.py"})
    return d


def test_worker_copies_verifies_and_starts_only_immediate_roles(tmp_path):
    sec, pref = FakeHost(tmp_path / "sec"), FakeHost(tmp_path / "pref")
    d = deployer(tmp_path, {"secondary": sec, "preferred": pref})
    d.worker("secondary")
    d.worker("preferred")
    assert (sec.map(tmp_path / "data" / "inputs" / "INPUT.npz")).read_bytes() == b"npz-bytes"
    assert any(" start --root " in c for c in sec.commands)
    assert not any(" start --root " in c for c in pref.commands)
    assert any(" install --root " in c for c in pref.commands)
    assert d.state["steps"]["worker:secondary"]["verified"] is True
    sec.commands.clear()
    d.worker("secondary")  # idempotent: nothing differs, nothing copied
    assert sum(c.startswith("sha256sum -c") for c in sec.commands) == 2


def test_worker_refuses_when_remote_bytes_still_differ(tmp_path):
    bad = FakeHost(tmp_path / "sec", corrupt={str(tmp_path / "data" / "inputs" / "INPUT.npz")})
    d = deployer(tmp_path, {"secondary": bad, "preferred": FakeHost(tmp_path / "pref")})
    with pytest.raises(dep.DeployError) as e:
        d.worker("secondary")
    assert e.value.code == "REMOTE_BYTES_DIFFER"
    assert not any(" start --root " in c or " install --root " in c for c in bad.commands)


def test_join_gate_waits_for_the_pilot_then_starts_preferred(tmp_path):
    result = tmp_path / "pref" / "opt" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}")
    running = FakeHost(tmp_path / "pref", pilot={"active": "activating", "status": {"state": "RUNNING"}})
    d = deployer(tmp_path, {"preferred": running, "secondary": FakeHost(tmp_path / "sec")})
    v = d.join("preferred")
    assert v["state"] == "WAIT_DEPENDENCY" and v["cause"].startswith("PILOT_LEASE_HELD")
    assert not any(" start --root " in c for c in running.commands)
    done = FakeHost(tmp_path / "pref", pilot={"active": "inactive", "status": {
        "state": "COMPLETED", "observed_peak_bytes": 7, "result": {"path": "/opt/result.json", "sha256": sha(result)}}})
    d.remotes["preferred"] = done
    v = d.join("preferred")
    assert v["state"] == "JOINED" and v["pilot"]["state"] == "COMPLETED"
    assert any(" start --root " in c and "--role preferred" in c for c in done.commands)
    done.commands.clear()
    assert d.join("preferred") == {"role": "preferred", "state": "JOINED"} and done.commands == []


def test_join_gate_refuses_a_mismatched_result_and_joins_after_a_failed_pilot(tmp_path):
    forged = FakeHost(tmp_path / "pref", pilot={"active": "inactive", "status": {
        "state": "COMPLETED", "result": {"path": "/opt/missing.json", "sha256": "1" * 64}}})
    d = deployer(tmp_path, {"preferred": forged, "secondary": FakeHost(tmp_path / "sec")})
    v = d.join("preferred")
    assert v["state"] == "FAILED" and v["cause"] == "PILOT_RESULT_DIGEST_MISMATCH"
    assert not any(" start --root " in c for c in forged.commands)
    failed = FakeHost(tmp_path / "pref", pilot={"active": "failed", "status": {"state": "FAILED", "reason": "OOM"}})
    d.remotes["preferred"] = failed
    v = d.join("preferred")
    assert v["state"] == "JOINED" and v["pilot"] == {"state": "FAILED", "cause": "OOM"}
    assert d.state["steps"]["joined:preferred"]["pilot"]["cause"] == "OOM"


def test_spec_must_be_private_and_complete(tmp_path):
    s, _ = spec(tmp_path)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(s))
    os.chmod(path, 0o644)
    with pytest.raises(dep.DeployError) as e:
        dep.load_spec(path)
    assert e.value.code == "SPEC_NOT_PRIVATE"
    os.chmod(path, 0o600)
    assert dep.load_spec(path)["campaign_id"] == "prod1"
    path.write_text(json.dumps({k: v for k, v in s.items() if k != "pilots"}))
    with pytest.raises(dep.DeployError) as e:
        dep.load_spec(path)
    assert e.value.code == "SPEC_INCOMPLETE"


def test_wrapper_pins_pythonpath_and_data_root(tmp_path):
    s, _ = spec(tmp_path)
    s["repos"]["doin-node"] = {"path": "x", "pin": "b" * 40, "pythonpath": ["src"]}
    text = dep.wrapper_text(s, "/opt/py/bin/python", "/data/root")
    assert f"{s['code_dir']}/doin-node/src" in text and f"{s['code_dir']}/predictor:" in text
    assert "export I6N_DATA_ROOT=/data/root" in text and text.rstrip().endswith('exec /opt/py/bin/python "$@"')


def test_quoted_sha256sum_names_with_spaces_are_mapped(tmp_path):
    class Quoting(FakeHost):
        def run(self, command, check=True):
            if command.startswith("sha256sum -c --quiet "):
                return subprocess.CompletedProcess(command, 1, "'/x/data copy.py': FAILED open or read\n", "")
            return super().run(command, check)
    host = Quoting(tmp_path / "h")
    assert dep.remote_mismatches(host, "0  /x/data copy.py\n", "/m.sha256") == ["/x/data copy.py"]
