"""Canonical JSON, digests, atomic files and boot identity (stdlib only)."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

SEED = 6102026
DIRECTION = "minimize"


def canonical(value):
    """Same formula as tools.i6n_contracts.eligibility.canonical_digest (A, C1)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_id(*, experiment, fold, effective_config, context, support):
    """INTERFACE_B fit_id: one physical fit, plan- and directory-independent."""
    return sha256_json(dict(experiment=experiment, fold=fold,
                            effective_config_sha256=sha256_json(effective_config),
                            context_sha256=sha256_json(context), support_sha256=sha256_json(support)))


def search_candidate_sha256(plan_sha256, horizon_hours, genotype_canonical_sha256):
    return sha256_json(dict(plan_sha256=plan_sha256, horizon_hours=horizon_hours,
                            genotype_canonical_sha256=genotype_canonical_sha256))


def atomic_write(path, text, mode=0o644):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_json(path, value, mode=0o644):
    atomic_write(path, json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n", mode)


def resolve_pin(pin, root=None):
    """Absolute path of a pin. ``{relpath, sha256}`` is relative to THIS host's root (roots may differ)."""
    if isinstance(pin, dict) and set(pin) == {"relpath", "sha256"}:
        rel = Path(pin["relpath"])
        if root is None or rel.is_absolute() or ".." in rel.parts:
            raise ValueError("relative pin needs this host's root and a path inside it")
        return Path(root) / rel
    if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
        raise ValueError("exact {path, sha256} or {relpath, sha256} pin required")
    path = Path(pin["path"])
    if not path.is_absolute():
        raise ValueError("absolute path required")
    return path


def relative_pin(pin, root):
    """Turn an absolute pin under ``root`` into a host-independent root-relative pin."""
    path, root = Path(pin["path"]).resolve(), Path(root).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError:
        raise ValueError("pinned file is not under the shared data root") from None
    return dict(relpath=rel.as_posix(), sha256=pin["sha256"])


def read_pinned(pin, what="input", root=None):
    """Read a pinned JSON file; refuse a digest mismatch."""
    try:
        path = resolve_pin(pin, root)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None
    if file_sha256(path) != pin["sha256"]:
        raise ValueError(f"{what}: digest mismatch")
    return json.loads(path.read_text())


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return "unknown-boot"
