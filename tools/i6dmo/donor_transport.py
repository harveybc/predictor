"""Copy authenticated donors into an attempt and reverify after host relocation.

Transport never changes donor terminal/weight bytes or original provenance.
The consuming process resolves attempt-relative paths before importing TF.
"""
import copy
import json
from pathlib import Path
import shutil

from tools.i6n_swarm.common import file_sha256


class TransportError(ValueError):
    """A typed pre-fit transport or custody failure."""


def _file(path, root=None):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise TransportError("R2_TRANSPORT_PATH: missing or symlinked file")
    if root is not None and not path.resolve().is_relative_to(Path(root).resolve()):
        raise TransportError("R2_TRANSPORT_PATH: outside attempt")
    return path


def _check(terminal, digest, feature, root=None):
    terminal = _file(terminal, root)
    if file_sha256(terminal) != digest:
        raise TransportError("R2_TRANSPORT_TERMINAL_DIGEST")
    doc = json.loads(terminal.read_text())
    if doc.get("feature") != feature:
        raise TransportError("R2_TRANSPORT_FEATURE")
    weights = _file(terminal.parent / "donor.npz", root)
    expected = (doc.get("artifacts") or {}).get("donor.npz")
    if not isinstance(expected, str) or file_sha256(weights) != expected:
        raise TransportError("R2_TRANSPORT_WEIGHTS_DIGEST")
    return terminal, weights, expected


def stage(spec, folder):
    """Return a spec whose donor paths move with the verified attempt files."""
    spec = copy.deepcopy(spec)
    binding = spec.get("r2_donors")
    if binding is None:
        return spec
    if spec.get("regime") == "R0":
        raise TransportError("R0_HAS_DONORS")
    donors = binding.get("donors")
    if not isinstance(donors, dict) or not donors:
        raise TransportError("R2_TRANSPORT_DONORS_MISSING")
    root = Path(folder).resolve()
    rows = {}
    for index, (feature, donor) in enumerate(donors.items()):
        ref = donor["terminal"]
        terminal, weights, weights_sha = _check(ref["path"], ref["sha256"], feature)
        relative = Path("staged_donors") / f"{index:02d}"
        target = root / relative
        target.mkdir(parents=True, exist_ok=True)
        for source in (terminal, weights):
            destination = target / source.name
            if destination.exists() or destination.is_symlink():
                if file_sha256(_file(destination, root)) != file_sha256(source):
                    raise TransportError("R2_TRANSPORT_EXISTING_CONFLICT")
            else:
                shutil.copyfile(source, destination)
        _check(target / "TERMINAL.json", ref["sha256"], feature, root)
        donor["terminal"] = dict(path=str(relative / "TERMINAL.json"), sha256=ref["sha256"])
        rows[feature] = dict(path=str(relative / "TERMINAL.json"),
                             terminal_sha256=ref["sha256"], weights_sha256=weights_sha)
    spec["r2_staging"] = dict(schema="i6dmo.r2_attempt_donors.v1", donors=rows)
    return spec


def resolve(spec, folder):
    """Authenticate attempt-local bytes before any model or objective is loaded."""
    spec = copy.deepcopy(spec)
    staging = spec.get("r2_staging")
    if staging is None:
        if spec.get("r2_donors") is not None:
            raise TransportError("R2_TRANSPORT_MANIFEST_MISSING")
        return spec
    binding = spec.get("r2_donors") or {}
    donors, rows = binding.get("donors") or {}, staging.get("donors") or {}
    if staging.get("schema") != "i6dmo.r2_attempt_donors.v1" or not donors or set(donors) != set(rows):
        raise TransportError("R2_TRANSPORT_MANIFEST")
    root = Path(folder).resolve()
    for feature, donor in donors.items():
        ref, row = donor["terminal"], rows[feature]
        rel = Path(ref["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.parts[:1] != ("staged_donors",) \
                or str(rel) != row["path"] or ref["sha256"] != row["terminal_sha256"]:
            raise TransportError("R2_TRANSPORT_PATH")
        terminal, _, weights_sha = _check(root / rel, ref["sha256"], feature, root)
        if weights_sha != row["weights_sha256"]:
            raise TransportError("R2_TRANSPORT_WEIGHTS_DIGEST")
        donor["terminal"] = dict(path=str(terminal), sha256=ref["sha256"])
    return spec
