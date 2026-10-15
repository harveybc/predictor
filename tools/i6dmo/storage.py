"""Versioned storage adapter: archive large immutable terminal artifacts to experiment storage.

Policy: ``docs/EXPERIMENT_STORAGE_POLICY.md``. The coordinator's ext4 keeps everything mutable —
the campaign root, every ``state/*.sqlite3`` (durable queue and outbox), POSIX lock/claim/lease
files, private deployment specs and anything secret-bearing. **No SQLite, PostgreSQL or DuckDB
live database is ever placed on the experiment-storage volume.**

Only artifacts of a terminal that already returned to the coordinator and was authenticated are
archived, to ``$EXPERIMENT_STORAGE_ROOT/i6dmo/<campaign-id>/<unit-id>/<relative path>``:

- the digest is computed before the copy and re-verified after the copy and ``fsync``; a mismatch
  is the typed refusal ``ARCHIVE_DIGEST_MISMATCH`` and nothing is published;
- the stored terminal keeps its **relative-path identity unchanged** and merely gains an
  ``archive`` block, so a unit identity never depends on the mount;
- restart is idempotent: an already-archived artifact whose digest matches is ADOPTED, never
  re-copied, and a partial destination is REPLACED (copies land on a temporary name and are
  renamed atomically, so nothing is ever half-published);
- a missing, read-only, wrong-filesystem, unwritable or too-full target is a hard typed refusal —
  **never a silent fallback to the home filesystem**;
- the archival path and digest are published as a governed report for the outbox and the mirrors.

The mount probe is injectable, so failure modes are exercised at the seam without creating,
remounting or touching a real mount.
"""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

SCHEMA = "i6dmo.storage_adapter.v1"
BASE_ENV = "EXPERIMENT_STORAGE_ROOT"
APP_SUBDIR = "i6dmo"
PREFLIGHT = Path(__file__).resolve().parents[1] / "experiment_storage_preflight.sh"
ALLOWED_FSTYPES = ("fuseblk", "ntfs3", "ntfs")
DEFAULT_RESERVE_BYTES = 5 * 1024 ** 3
#: Only large immutable artifacts are archived; small receipts stay with the campaign evidence.
MIN_ARCHIVE_BYTES = 1024
#: Never archived, whatever a terminal claims: live databases, journals and lock/claim files.
LIVE_STATE_SUFFIXES = (".sqlite3", ".sqlite", ".db", ".duckdb", "-wal", "-shm", ".lock",
                       ".json-journal")
LIVE_STATE_DIRS = ("state", "private")


class StorageRefusal(RuntimeError):
    """A typed refusal. ``code`` names the exact condition; there is never a fallback path."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def storage_root(env=None):
    env = os.environ if env is None else env
    base = env.get(BASE_ENV)
    if not base:
        raise StorageRefusal("STORAGE_ROOT_NOT_CONFIGURED", f"{BASE_ENV} is not set")
    return Path(base)


# ------------------------------------------------------------------ mount probe
def probe_mount(path, runner=subprocess.run):
    """Read-only facts about the filesystem holding ``path`` (findmnt + statvfs)."""
    path = Path(path)
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not path.exists() and not probe.exists():
        return dict(exists=False, fstype=None, options="", read_only=True, writable=False,
                    available_bytes=0, target=None)
    out = runner(["findmnt", "-rn", "-T", str(probe), "-o", "TARGET,FSTYPE,OPTIONS"],
                 capture_output=True, text=True, check=False)
    target = fstype = options = None
    if out.returncode == 0 and out.stdout.strip():
        fields = out.stdout.strip().split(" ")
        target, fstype, options = (fields + [None, None, None])[:3]
    try:
        usage = shutil.disk_usage(probe)
        available = usage.free
    except OSError:
        available = 0
    read_only = bool(options and ",ro," in f",{options},")
    return dict(exists=path.exists(), fstype=fstype, options=options or "", read_only=read_only,
                writable=os.access(probe, os.W_OK) and not read_only, available_bytes=available,
                target=target)


def _check_target(target, facts, required_bytes, reserve_bytes):
    if not facts.get("exists"):
        raise StorageRefusal("STORAGE_MOUNT_MISSING", "the experiment storage path is absent")
    if facts.get("read_only"):
        raise StorageRefusal("STORAGE_MOUNT_READ_ONLY", "the experiment storage mount is read-only")
    if facts.get("fstype") and facts["fstype"] not in ALLOWED_FSTYPES:
        raise StorageRefusal("STORAGE_WRONG_FILESYSTEM",
                             f"{facts['fstype']} is not an approved experiment-storage filesystem")
    if not facts.get("writable"):
        raise StorageRefusal("STORAGE_NOT_WRITABLE", "the experiment storage target is not writable")
    needed = int(required_bytes) + int(reserve_bytes)
    if int(facts.get("available_bytes", 0)) < needed:
        raise StorageRefusal("STORAGE_INSUFFICIENT_SPACE",
                             f"available={facts.get('available_bytes')} needed={needed}")
    return facts


def resolve_target(storage_root_path, campaign_id, unit_id, subdir=APP_SUBDIR):
    """The approved destination directory; anything escaping the tree is refused."""
    base = Path(storage_root_path).resolve()
    target = Path(os.path.normpath(base / subdir / str(campaign_id) / str(unit_id)))
    if base not in target.parents and target != base:
        raise StorageRefusal("PATH_OUTSIDE_EXPERIMENT_STORAGE", str(target))
    return target


def run_preflight(path, required_bytes, reserve_bytes=DEFAULT_RESERVE_BYTES, runner=subprocess.run,
                  script=PREFLIGHT):
    """The versioned shell preflight; its exit code and output are returned verbatim."""
    out = runner([str(script), str(path), str(int(required_bytes)), str(int(reserve_bytes))],
                 capture_output=True, text=True, check=False)
    doc = dict(schema="i6dmo.storage_preflight.v1", command=f"{script} <path> {int(required_bytes)} "
               f"{int(reserve_bytes)}", returncode=out.returncode,
               stdout=out.stdout.strip(), stderr=out.stderr.strip(),
               state="STORAGE_READY" if out.returncode == 0 else "REFUSED")
    return doc


# ---------------------------------------------------------------- live state
def live_state_paths(root):
    """Every live database, journal and lock file of a campaign: all must stay on local ext4."""
    root = Path(root)
    found = []
    for pattern in ("state/*", "private/*"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                found.append(path)
    return found


def _is_live_state(path):
    path = Path(path)
    if any(str(path).endswith(suffix) for suffix in LIVE_STATE_SUFFIXES):
        return True
    return any(part in LIVE_STATE_DIRS for part in path.parts)


# ------------------------------------------------------------------ archiving
def _artifact_sources(stored, terminal_dir, min_bytes=MIN_ARCHIVE_BYTES):
    """Large immutable artifacts of an authenticated terminal, as (relative path, source file)."""
    out = []
    for name, art in sorted((stored.get("terminal", {}).get("artifacts") or {}).items()):
        rel = art.get("path")
        if not rel or _is_live_state(rel):
            continue
        source = terminal_dir / "out" / rel
        if not source.is_file():
            raise StorageRefusal("ARCHIVE_SOURCE_MISSING", f"{name}: {rel}")
        if source.stat().st_size < min_bytes:
            continue
        out.append((name, rel, source, art.get("sha256")))
    return out


def _copy_verified(source, dest, expected):
    """Copy to a temporary name, fsync, verify, then rename: never half-published."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        verified = file_sha256(tmp)
        if verified != expected:
            raise StorageRefusal("ARCHIVE_DIGEST_MISMATCH",
                                 f"{dest.name}: expected {expected[:12]} got {verified[:12]}")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return verified


def archive_unit(root, unit_id, *, storage_root=None, subdir=APP_SUBDIR, probe=probe_mount,
                 required_bytes=None, reserve_bytes=DEFAULT_RESERVE_BYTES, env=None, clock=time.time,
                 min_bytes=MIN_ARCHIVE_BYTES):
    """Archive one authenticated terminal's large artifacts. Idempotent; typed refusals only."""
    from tools.i6dmo import campaign as C
    root = Path(root)
    base = Path(storage_root) if storage_root is not None else _default_root(env)
    plan = C.load_plan(root)
    q = C.open_queue(root)
    try:
        term = q.terminal(unit_id)
    finally:
        q.close()
    if term is None:
        raise StorageRefusal("TERMINAL_NOT_AUTHENTICATED", f"{unit_id} has no accepted terminal")
    identity = term["identity_sha256"]
    terminal_dir = root / "terminals" / identity
    stored_path = terminal_dir / "TERMINAL.json"
    stored = json.loads(stored_path.read_text())
    sources = _artifact_sources(stored, terminal_dir, min_bytes)
    total = sum(s.stat().st_size for _, _, s, _ in sources)
    target = resolve_target(base, plan["campaign_id"], unit_id, subdir=subdir)
    # The mount facts come from the APPROVED TREE, not the per-unit leaf, which this call creates.
    approved = Path(base) / subdir if subdir else Path(base)
    facts = _check_target(approved, probe(approved), total if required_bytes is None else required_bytes,
                          reserve_bytes)
    target.mkdir(parents=True, exist_ok=True)
    archived = []
    for name, rel, source, declared in sources:
        expected = declared or file_sha256(source)
        if file_sha256(source) != expected:
            raise StorageRefusal("ARCHIVE_SOURCE_DIGEST_MISMATCH", f"{name}: {rel}")
        dest = target / rel
        action = "COPIED"
        if dest.is_file():
            action = "ADOPTED" if file_sha256(dest) == expected else "REPLACED_PARTIAL"
        verified = expected if action == "ADOPTED" else _copy_verified(source, dest, expected)
        archived.append(dict(name=name, path=rel, sha256=expected, verified_sha256=verified,
                             archive_path=str(dest.relative_to(base)), bytes=source.stat().st_size,
                             action=action))
    doc = dict(schema=SCHEMA, state="ARCHIVED", campaign_id=plan["campaign_id"], unit_id=unit_id,
               identity_sha256=identity, storage_kind="EXPERIMENT_STORAGE",
               filesystem=facts.get("fstype"), artifacts=archived,
               bytes=sum(a["bytes"] for a in archived), archived_at=clock(),
               note=("identity stays relative and local; only large immutable artifacts are copied, "
                     "and no live database, lock or private spec is ever archived"))
    doc["archive_sha256"] = hashlib.sha256(
        json.dumps([dict(path=a["path"], sha256=a["sha256"]) for a in archived],
                   sort_keys=True).encode()).hexdigest()
    stored["archive"] = {k: v for k, v in doc.items() if k != "state"}
    tmp = stored_path.with_name(f".{stored_path.name}.tmp")
    tmp.write_text(json.dumps(stored, sort_keys=True, indent=1) + "\n")
    os.replace(tmp, stored_path)
    return doc


def _default_root(env=None):
    return storage_root(env) if (env is not None or os.environ.get(BASE_ENV)) else None


def archive_report(root, archived):
    """The governed report that publishes the archival path and digest to the outbox and mirrors."""
    from tools.i6dmo import campaign as C
    effective = C.load_effective(root)
    return dict(experiment_key=f"i6dmo:{archived['campaign_id']}:{archived['unit_id']}:archive"[:128],
                experiment_set_key=f"i6dmo:{archived['campaign_id']}", actor="predictor",
                lake="LOCAL_RETAINED", lineage="UNVERIFIED",
                config_sha256=archived["identity_sha256"], code_commit=effective["code_commit"],
                project="predictor", phase="I6DMO",
                tags={"unit": archived["unit_id"], "archive_root_kind": archived["storage_kind"],
                      "archive_sha256": archived["archive_sha256"], "test_read": "false",
                      "filesystem": str(archived.get("filesystem")),
                      "identity_is_relative": "true"},
                metrics=[dict(metric="Archived_Artifacts", value=len(archived["artifacts"]),
                              split=None, horizon=None, unit="count"),
                         dict(metric="Archived_Bytes", value=archived["bytes"], split=None,
                              horizon=None, unit="bytes")])


# ---------------------------------------------------------------- disk report
def disk_report(*, roles=(), worker_disk=None, probe=probe_mount, storage_root=None, env=None,
                local_root=None):
    """Local and per-worker disk, reported SEPARATELY; a silent worker is UNAVAILABLE.

    The coordinator's experiment-storage volume is never presented as another host's free space.
    """
    base = Path(storage_root) if storage_root is not None else _default_root(env)
    local = dict(schema="i6dmo.disk_report.v1")
    if base is not None:
        facts = probe(base)
        local["experiment_storage"] = dict(available_bytes=facts.get("available_bytes"),
                                           filesystem=facts.get("fstype"),
                                           read_only=facts.get("read_only"),
                                           exists=facts.get("exists"))
    if local_root is not None:
        facts = probe_mount(local_root)
        local["campaign_root"] = dict(available_bytes=facts.get("available_bytes"),
                                      filesystem=facts.get("fstype"))
    workers = {}
    for role in roles:
        if worker_disk is None:
            workers[role] = dict(state="UNAVAILABLE", cause="no worker disk probe supplied")
            continue
        try:
            value = worker_disk(role)
        except Exception as exc:
            workers[role] = dict(state="UNAVAILABLE", cause=str(exc)[:160])
            continue
        workers[role] = dict(state="REPORTED", **(value or {}))
    return dict(schema="i6dmo.disk_report.v1", local=local, workers=workers,
                note=("local and worker filesystems are reported separately; the coordinator's "
                      "experiment storage is not mounted on the workers and is never shown as "
                      "their free space"))
