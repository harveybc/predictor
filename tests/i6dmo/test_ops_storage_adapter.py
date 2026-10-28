"""Versioned storage adapter: archive large immutable terminal artifacts, never live state.

Policy: `docs/EXPERIMENT_STORAGE_POLICY.md`. The campaign root, every `state/*.sqlite3` and all
lock/claim files stay on the coordinator's ext4; only authenticated terminal artifacts are archived
to `$EXPERIMENT_STORAGE_ROOT/i6dmo/<campaign-id>/<unit-id>/...`. The mount probe is injectable so
read-only and missing-mount failures are exercised without touching the real mount.
"""
import json

import pytest

from tests.i6dmo.ops_fakes import make_campaign, run_worker

GIB = 1024 ** 3


def fake_probe(root, *, fstype="fuseblk", writable=True, read_only=False, exists=True,
               available=700 * GIB):
    """Seam-level injection: no real mount is created, remounted or touched."""
    def probe(path):
        return dict(exists=exists, fstype=fstype, options="rw" if not read_only else "ro",
                    read_only=read_only, writable=writable and not read_only,
                    available_bytes=available, target=str(root))
    return probe


def _archive(tmp_path, root, unit_id="short.fit", **kw):
    from tools.i6dmo import storage
    store = tmp_path / "store"
    (store / "i6dmo").mkdir(parents=True, exist_ok=True)
    kw.setdefault("probe", fake_probe(store))
    kw.setdefault("min_bytes", 0)   # the fake trainer's artifacts are small; the threshold is policy
    return storage.archive_unit(root, unit_id, storage_root=store, **kw)


def test_archives_terminal_artifacts_and_verifies_the_digest_round_trip(tmp_path):
    from tools.i6dmo import storage
    root = make_campaign(tmp_path)
    assert run_worker(root, "preferred", tmp_path / "c.txt")["result"] == "COMPLETE"
    out = _archive(tmp_path, root)
    assert out["state"] == "ARCHIVED" and out["schema"] == storage.SCHEMA
    assert out["artifacts"], "the completed fit must have archived at least one artifact"
    for art in out["artifacts"]:
        dest = tmp_path / "store" / art["archive_path"]
        assert dest.is_file()
        assert storage.file_sha256(dest) == art["sha256"] == art["verified_sha256"]
        # identity is relative and unchanged: it never mentions the mount
        assert not art["path"].startswith("/") and "store" not in art["path"]
    stored = json.loads((root / "terminals" / out["identity_sha256"] / "TERMINAL.json").read_text())
    assert stored["identity_sha256"] == out["identity_sha256"]          # identity untouched
    archive = stored["archive"]
    assert archive["schema"] == storage.SCHEMA and archive["campaign_id"] == out["campaign_id"]
    assert {a["path"] for a in archive["artifacts"]} == {a["path"] for a in out["artifacts"]}


def test_re_archiving_adopts_instead_of_copying_and_a_partial_copy_is_never_published(tmp_path):
    from tools.i6dmo import storage
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")
    first = _archive(tmp_path, root)
    again = _archive(tmp_path, root)
    assert again["state"] == "ARCHIVED"
    assert all(a["action"] == "ADOPTED" for a in again["artifacts"])     # restart re-adopts
    assert [a["sha256"] for a in again["artifacts"]] == [a["sha256"] for a in first["artifacts"]]
    # a truncated destination is a partial copy: it is replaced, never adopted
    victim = tmp_path / "store" / first["artifacts"][0]["archive_path"]
    victim.write_bytes(b"partial")
    repaired = _archive(tmp_path, root)
    assert repaired["artifacts"][0]["action"] == "REPLACED_PARTIAL"
    assert storage.file_sha256(victim) == first["artifacts"][0]["sha256"]
    assert not list((tmp_path / "store").rglob("*.tmp-*"))               # no half-published temp left


def test_a_digest_mismatch_after_copy_is_a_typed_refusal(tmp_path, monkeypatch):
    from tools.i6dmo import storage
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")

    real = storage.file_sha256
    calls = {"n": 0}

    def corrupt(path):
        calls["n"] += 1
        return "f" * 64 if calls["n"] > 1 else real(path)     # the post-copy read disagrees
    monkeypatch.setattr(storage, "file_sha256", corrupt)
    with pytest.raises(storage.StorageRefusal) as exc:
        _archive(tmp_path, root)
    assert exc.value.code == "ARCHIVE_DIGEST_MISMATCH"
    stored = json.loads(next((root / "terminals").rglob("TERMINAL.json")).read_text())
    assert "archive" not in stored                            # nothing is published on mismatch


@pytest.mark.parametrize("kind,code", [
    ("missing", "STORAGE_MOUNT_MISSING"),
    ("read_only", "STORAGE_MOUNT_READ_ONLY"),
    ("wrong_fs", "STORAGE_WRONG_FILESYSTEM"),
    ("not_writable", "STORAGE_NOT_WRITABLE"),
    ("full", "STORAGE_INSUFFICIENT_SPACE"),
])
def test_every_storage_failure_is_a_typed_refusal_with_no_fallback(tmp_path, kind, code):
    from tools.i6dmo import storage
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")
    store = tmp_path / "store"
    probes = dict(
        missing=fake_probe(store, exists=False),
        read_only=fake_probe(store, read_only=True),
        wrong_fs=fake_probe(store, fstype="ext4"),
        not_writable=fake_probe(store, writable=False),
        full=fake_probe(store, available=1024),
    )
    with pytest.raises(storage.StorageRefusal) as exc:
        _archive(tmp_path, root, probe=probes[kind], required_bytes=10 * GIB)
    assert exc.value.code == code
    assert not list((tmp_path / "store").rglob("*.npz"))      # never silently written elsewhere
    assert not list(root.rglob("*fallback*"))


def test_the_approved_tree_is_enforced(tmp_path):
    from tools.i6dmo import storage
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")
    with pytest.raises(storage.StorageRefusal) as exc:
        _archive(tmp_path, root, subdir="../../etc")
    assert exc.value.code == "PATH_OUTSIDE_EXPERIMENT_STORAGE"


def test_live_state_is_never_archived_and_stays_on_the_local_filesystem(tmp_path):
    from tools.i6dmo import storage
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")
    _archive(tmp_path, root)
    archived = {p.name for p in (tmp_path / "store").rglob("*") if p.is_file()}
    assert not {n for n in archived if n.endswith((".sqlite3", ".lock", ".json-journal", "-wal", "-shm"))}
    assert not {n for n in archived if "deployment" in n or "private" in n}
    live = storage.live_state_paths(root)
    assert any(p.name == "queue.sqlite3" for p in live)
    for path in live:
        assert str(path).startswith(str(root))               # all live state stays under the ext4 root


def test_archival_paths_are_published_to_the_outbox_facts(tmp_path):
    from tools.i6dmo import storage
    root = make_campaign(tmp_path)
    run_worker(root, "preferred", tmp_path / "c.txt")
    out = _archive(tmp_path, root)
    report = storage.archive_report(root, out)
    tags = report["tags"]
    assert tags["archive_root_kind"] == "EXPERIMENT_STORAGE" and tags["unit"] == "short.fit"
    assert tags["archive_sha256"] == out["archive_sha256"]
    names = {m["metric"] for m in report["metrics"]}
    assert {"Archived_Artifacts", "Archived_Bytes"} <= names
    from tools.i6dmo import outbox
    box = outbox.Outbox(tmp_path / "ob.sqlite3")
    try:
        box.enqueue(report, source_unit="short.fit")         # a valid governed report
    finally:
        box.close()


def test_preflight_reports_local_and_worker_disk_separately(tmp_path):
    from tools.i6dmo import storage
    def worker_disk(role):
        if role == "secondary":
            raise RuntimeError("unreachable")
        return dict(disk_free_bytes=400 * GIB, run_root="<role scratch>")
    doc = storage.disk_report(roles=["preferred", "secondary"], worker_disk=worker_disk,
                              probe=fake_probe(tmp_path / "store"), storage_root=tmp_path / "store")
    assert doc["local"]["experiment_storage"]["available_bytes"] == 700 * GIB
    assert doc["workers"]["preferred"]["disk_free_bytes"] == 400 * GIB
    assert doc["workers"]["secondary"]["state"] == "UNAVAILABLE"
    assert doc["note"].startswith("local and worker filesystems are reported separately")
    # the coordinator's experiment storage is never presented as a worker's free space
    assert "experiment_storage" not in doc["workers"]["preferred"]


def test_the_storage_gate_refuses_init_deploy_and_claim_without_a_fallback(tmp_path, monkeypatch):
    """(a) before init, (b) before deploy, (c) when a job claims an artifact-writing unit."""
    import json as _json
    from tests.i6dmo.ops_fakes import private_spec
    from tools.i6dmo import campaign, storage

    refusing = dict(returncode=5, stderr="INSUFFICIENT_SPACE available=1 required=2 reserve=3",
                    stdout="", state="REFUSED", schema="i6dmo.storage_preflight.v1", command="<preflight>")
    monkeypatch.setattr(storage, "run_preflight", lambda *a, **k: refusing)
    monkeypatch.setenv(storage.BASE_ENV, str(tmp_path / "store"))
    spec = private_spec(tmp_path)
    spec["storage"] = {"estimate_bytes": 60 * GIB, "reserve_bytes": 5 * GIB}
    (tmp_path / "private_spec.json").write_text(_json.dumps(spec))
    # (a) init
    with pytest.raises(campaign.Refused, match="STORAGE_GATE_REFUSED: stage=init"):
        make_campaign(tmp_path, name="gated")
    # with storage healthy the campaign builds, then (b) deploy and (c) claim are gated too
    monkeypatch.setattr(storage, "run_preflight",
                        lambda *a, **k: dict(refusing, returncode=0, state="STORAGE_READY"))
    root = make_campaign(tmp_path, name="ok")
    monkeypatch.setattr(storage, "run_preflight", lambda *a, **k: refusing)
    from tools.i6dmo import deploy as D
    with pytest.raises(campaign.Refused, match="STORAGE_GATE_REFUSED: stage=deploy"):
        D.deploy(root, unit_dir=tmp_path / "units", systemctl=lambda *a: dict(rc=0, stdout="x"),
                 render_only=False, start=False)
    out = run_worker(root, "preferred", tmp_path / "c.txt")
    assert out["result"] == "WAIT_RESOURCE" and "STORAGE_GATE_REFUSED" in out["cause"]
    assert "claim:short.fit" in out["cause"]
    assert not list(root.rglob("*fallback*"))
