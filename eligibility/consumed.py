"""Derive the subjects a run will actually consume.

The gate used to ask a question the caller answered about itself:
a config listed some subject ids, and the gate checked those. A
list can name variables the run never reads and omit variables it
does, so the answer proved nothing about the bytes.

This module derives the set instead. It opens the files the run's
own contract names, reads their headers and schema — without
building a window, fitting a scaler or loading a row of data —
and returns the exact columns that will reach the model, each
with a stable id, together with the digests of the bytes, the
partition layout and the code at the last point of use.

`eligibility_subjects` survives only as an ASSERTION: if a config
declares it, it must equal what was derived, or the run refuses.
An empty list grants nothing, because it is no longer the source
of the answer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from eligibility.strict import StrictParseRefusal

# The partition roles a supervised run declares, and the config
# keys that name their files.
PARTITION_KEYS = {
    "train": ("x_train_file", "y_train_file"),
    "validation": ("x_validation_file", "y_validation_file"),
    "test": ("x_test_file", "y_test_file"),
}

# Columns that are never model inputs: they identify a row rather
# than describe it. Named explicitly so the exclusion is visible.
NON_FEATURE_COLUMNS = ("DATE_TIME", "date_time", "datetime",
                       "timestamp", "TIMESTAMP", "date", "DATE")


# The repository that holds the CODE. Data roots vary per run;
# the consuming code does not, and conflating the two would make
# a code identity depend on where the data happens to live.
CODE_ROOT = Path(__file__).resolve().parents[1]


class ConsumedSubjectsRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def subject_id(dataset_id: str, column: str, *,
               side: str = "x", contract_role: str = "input") -> str:
    """Stable logical id over (dataset, side, contract role, column).

    C33 (order 2026-09-11): it used to be `dataset::column`, and the
    y-side builder skipped any column whose NAME already appeared on
    x. A file with a column `SAME` on both sides therefore produced
    ONE subject, labelled x — the y-side subject simply did not exist,
    so a review of "every consumed subject" covered an input and
    silently omitted the label that shared its name.

    Side and role are part of the identity because they are part of
    what the subject IS: the same column consumed as an input and as a
    target are two different things to review. The physical PATH is
    still never part of it, so relocating a file does not rename a
    variable.
    """
    return f"{dataset_id}::{side}::{contract_role}::{column}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path: Path) -> list[str]:
    """The schema only: one line, no rows, no fitting."""
    with open(path, newline="", encoding="utf-8",
              errors="strict") as fh:
        try:
            header = next(csv.reader(fh))
        except StopIteration:
            raise ConsumedSubjectsRefusal(
                f"{path.name} is empty — a partition with no "
                "header declares no contract")
    header = [c.strip() for c in header]
    if not header or not any(header):
        raise ConsumedSubjectsRefusal(
            f"{path.name} has no column names")
    return header


def _dataset_id(config: dict, path: Path) -> str:
    """Logical dataset identity, declared or derived.

    A declared `dataset_id` wins. Otherwise the id is derived from
    the file's LOGICAL location inside the repository — its
    directory and stem — never its absolute path, so two
    checkouts agree.
    """
    declared = config.get("dataset_id")
    if declared:
        return str(declared)
    parts = path.parts
    if "data" in parts:
        i = parts.index("data")
        return "/".join(parts[i:])
    if "data_downsampled" in parts:
        i = parts.index("data_downsampled")
        return "/".join(parts[i:])
    return path.name


def code_identity(repo_root: Path, files: list[str],
                  plugins: dict | None = None) -> tuple:
    """Digest of the code that will consume the data.

    Returns (digest, inventory). The inventory names every file
    and plugin that entered the digest, so a reviewer can see
    WHAT was bound rather than trusting a hash. Paths are
    repo-relative wherever possible, so the identity reproduces
    in another clean checkout of the same commit.
    """
    repo_root = Path(repo_root)
    inventory = []
    for rel in sorted(files):
        p = repo_root / rel
        if not p.is_file():
            raise CodeIdentityRefusal(
                f"the consuming module {rel!r} is absent — an "
                "identity that skips a missing file is not an "
                "identity")
        inventory.append({"kind": "module", "id": rel,
                          "sha256": sha256_file(p)})
    for role, info in sorted((plugins or {}).items()):
        origin = Path(info["origin"])
        try:
            ident = str(origin.relative_to(repo_root))
        except ValueError:
            ident = f"{info['module']}@external"
        if not origin.is_file():
            raise CodeIdentityRefusal(
                f"{role}: the resolved module file {origin.name} "
                "does not exist")
        inventory.append({
            "kind": "plugin", "role": role, "id": ident,
            "entry_point": f"{info['entry_point_group']}:"
                           f"{info['entry_point_name']}",
            "entry_point_value": info["entry_point_value"],
            "resolution_source": info.get("resolution_source",
                                          "UNAVAILABLE"),
            "sha256": sha256_file(origin)})
    for ext in external_distributions():
        inventory.append({"kind": "external", **ext})
    digest = hashlib.sha256(json.dumps(
        inventory, sort_keys=True).encode()).hexdigest()
    return digest, inventory


# C32 (order 2026-09-11): the code identity covers the COMPLETE local
# surface, not a list of twelve files.
#
# The audit's counterexample was exact: mutating
# `predictor_plugins/common/base.py` changed training and did not
# change the digest, because only the entry point's own file was
# hashed and nothing it imports was. Chasing the import graph is one
# answer; a COMPLETE, FINITE local surface is the other, and it is the
# one that cannot be fooled by a conditional or dynamic import.
#
# So every `.py` under the packages a run can execute is bound, plus
# the packaging metadata that decides which entry point resolves where.
# 128 files at the time of writing; hashing them costs milliseconds.
LOCAL_PACKAGES = (
    "app",
    "eligibility",
    "olap",
    "predictor_plugins",
    "pipeline_plugins",
    "preprocessor_plugins",
    "target_plugins",
    "optimizer_plugins",
)

#: files outside a package that still decide what runs.
LOCAL_ROOT_FILES = ("setup.py",)

#: kept for readers of the pre-C32 identity: the twelve modules the
#: old digest covered. The surface below is a strict superset.
CONSUMING_CODE = (
    "app/main.py",
    "app/config.py",
    "app/config_merger.py",
    "app/config_handler.py",
    "app/plugin_loader.py",
    "app/plugin_resolver.py",
    "app/data_handler.py",
    "app/data_processor.py",
    "eligibility/gate.py",
    "eligibility/consumed.py",
    "eligibility/integration.py",
    "eligibility/review.py",
    "eligibility/strict.py",
)

#: external distributions whose behaviour a run depends on. Hashing one
#: wrapper file would not cover TensorFlow, so these are recorded by
#: distribution name, version and logical location instead — an honest
#: weaker binding, named as such, rather than a strong-looking one that
#: covers nothing.
EXTERNAL_DISTRIBUTIONS = (
    "tensorflow", "keras", "numpy", "pandas", "scikit-learn",
    "scipy", "deap", "neat-python", "sqlalchemy", "psycopg2-binary",
)


def local_code_surface(repo_root: Path) -> list[str]:
    """Every local `.py` a run could execute, repo-relative, sorted."""
    repo_root = Path(repo_root)
    out: set[str] = set()
    for pkg in LOCAL_PACKAGES:
        base = repo_root / pkg
        if not base.is_dir():
            continue
        for f in base.rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            out.add(str(f.relative_to(repo_root)))
    for rel in LOCAL_ROOT_FILES:
        if (repo_root / rel).is_file():
            out.add(rel)
    if not out:
        raise CodeIdentityRefusal(
            f"no local code surface found under {repo_root} — an "
            "identity over nothing is not an identity")
    return sorted(out)


def external_distributions() -> list[dict]:
    """Name, version and logical location of the external libraries.

    A digest is not claimed: hashing a package's whole tree on every
    run would be dishonest about what was verified. What IS claimed is
    exactly what is recorded — which distribution, which version, and
    where it was resolved from.
    """
    from importlib import metadata
    out = []
    for name in sorted(EXTERNAL_DISTRIBUTIONS):
        try:
            dist = metadata.distribution(name)
            version = dist.version
            location = str(dist.locate_file(""))
        except Exception:                       # noqa: BLE001
            version, location = "NOT_INSTALLED", "UNAVAILABLE"
        out.append({"distribution": name, "version": version,
                    "location": location,
                    "binding": "NAME_VERSION_LOCATION_ONLY"})
    return out


class CodeIdentityRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def resolve_plugin_modules(config: dict) -> dict:
    """Resolve each plugin role to the FILE that will be loaded.

    C31: this delegates to the ONE resolver the executor also uses, so
    the identity can no longer describe a different plugin from the one
    that runs. It used to look the predictor up under the legacy key
    `plugin` while `app/main.py` read `predictor_plugin`.
    """
    from app.plugin_resolver import resolve_all
    return resolve_all(config, code_root=CODE_ROOT)


def resolve_consumed_subjects(config: dict, *,
                              repo_root: Path) -> dict:
    """The single derivation. Returns the exact consumed set."""
    repo_root = Path(repo_root)
    partitions, headers, y_headers = {}, {}, {}
    files_in_order = []
    for role, (xkey, ykey) in PARTITION_KEYS.items():
        xpath = config.get(xkey)
        if not xpath:
            continue
        p = (repo_root / xpath) if not Path(xpath).is_absolute() \
            else Path(xpath)
        if not p.is_file():
            raise ConsumedSubjectsRefusal(
                f"{role}: the contract names {xkey}={xpath!r} "
                "but the file does not exist — a run cannot be "
                "gated on data it cannot open")
        headers[role] = read_header(p)
        x_sha = sha256_file(p)
        entry = {"role": role, "x_key": xkey,
                 "x_file": str(Path(xpath)),
                 "x_sha256": x_sha,
                 "x_bytes": p.stat().st_size,
                 "x_columns": list(headers[role])}
        files_in_order.append((role, "x", xkey, str(Path(xpath)),
                               x_sha))
        # C18: the y side is part of the consumed set. It was
        # opened and hashed before but never reached a subject or
        # a digest, so a mutated y was invisible to every derived
        # identity.
        ypath = config.get(ykey)
        if ypath:
            yp = (repo_root / ypath) \
                if not Path(ypath).is_absolute() else Path(ypath)
            if not yp.is_file():
                raise ConsumedSubjectsRefusal(
                    f"{role}: the contract names {ykey}="
                    f"{ypath!r} but the file does not exist")
            y_sha = sha256_file(yp)
            y_headers[role] = read_header(yp)
            entry["y_key"] = ykey
            entry["y_file"] = str(Path(ypath))
            entry["y_sha256"] = y_sha
            entry["y_bytes"] = yp.stat().st_size
            entry["y_columns"] = list(y_headers[role])
            files_in_order.append((role, "y", ykey,
                                   str(Path(ypath)), y_sha))
        partitions[role] = entry

    declares_inputs = any(config.get(k) for k, _ in
                          PARTITION_KEYS.values())
    if not partitions:
        if declares_inputs:
            raise ConsumedSubjectsRefusal(
                "the contract declares inputs but no partition "
                "could be resolved")
        raise ConsumedSubjectsRefusal(
            "this run declares no input partitions — there is "
            "nothing to gate, and an ungated run is not gated "
            "evidence")

    # every declared partition must present the SAME schema
    roles = sorted(headers)
    base_role = roles[0]
    base = headers[base_role]
    for role in roles[1:]:
        if headers[role] != base:
            only_base = sorted(set(base) - set(headers[role]))
            only_other = sorted(set(headers[role]) - set(base))
            raise ConsumedSubjectsRefusal(
                f"partition schemas differ: {base_role} vs "
                f"{role} (only in {base_role}: {only_base}; "
                f"only in {role}: {only_other}) — a variable "
                "mapping that is not identical across "
                "partitions is not one mapping")

    dupes = sorted({c for c in base if base.count(c) > 1})
    if dupes:
        raise ConsumedSubjectsRefusal(
            f"duplicate columns in the consumed schema: {dupes}")

    x_path = repo_root / partitions[base_role]["x_file"]
    dataset_id = _dataset_id(config, Path(
        partitions[base_role]["x_file"]))
    target = config.get("target_column")
    subjects, excluded = [], []

    # C33: the schema of EACH side must be constant across the
    # partitions. Train, validation and test are the same contract
    # observed three times; a column that exists in one and not
    # another is not a partition, it is two datasets sharing a name.
    for side, per_role in (("x", headers), ("y", y_headers)):
        shapes = {r: tuple(per_role[r]) for r in sorted(per_role)}
        distinct = sorted(set(shapes.values()))
        if len(distinct) > 1:
            differing = {r: list(v) for r, v in shapes.items()}
            raise ConsumedSubjectsRefusal(
                f"the {side} schema differs across partitions — "
                f"{ {r: len(v) for r, v in differing.items()} } columns "
                "per role. Train, validation and test must observe the "
                "SAME contract, or a model is fitted and scored on "
                "different variables")

    dataset_id_y = _dataset_id(
        config, Path(partitions[base_role].get("y_file")
                     or partitions[base_role]["x_file"]))

    # C33: both sides build subjects INDEPENDENTLY. The old builder
    # skipped a y column whose name already appeared on x, so a column
    # present on both sides produced only its x subject.
    y_columns: list[str] = []
    for role in sorted(y_headers):
        for column in y_headers[role]:
            if column not in y_columns:
                y_columns.append(column)

    for side, columns, ds in (("x", list(base), dataset_id),
                              ("y", y_columns, dataset_id_y)):
        for column in columns:
            if not column.strip():
                raise ConsumedSubjectsRefusal(
                    "the consumed schema contains an unnamed column "
                    "— a column without an identity cannot be "
                    "reviewed")
            if column in NON_FEATURE_COLUMNS:
                excluded.append({"column": column, "side": side,
                                 "reason": "ROW_IDENTIFIER"})
                continue
            if column == target:
                role = "target"
            else:
                role = "input" if side == "x" else "label"
            subjects.append({
                "subject_id": subject_id(ds, column, side=side,
                                         contract_role=role),
                "column": column,
                "dataset_id": ds,
                "contract_role": role,
                "side": side,
            })

    # C33: a declared target must exist exactly where the contract
    # says it does. A target nobody can find is not a contract.
    if target:
        found = sorted({s["side"] for s in subjects
                        if s["contract_role"] == "target"})
        if not found:
            raise ConsumedSubjectsRefusal(
                f"the declared target {target!r} appears on neither "
                "the x nor the y side of any partition — a target that "
                "is not in the data is not a contract")
        declared_side = config.get("target_side")
        if declared_side and declared_side not in found:
            raise ConsumedSubjectsRefusal(
                f"the contract declares the target on side "
                f"{declared_side!r} but {target!r} was found on "
                f"{found}")

    # C33: zero subject_id collisions. Two subjects sharing an id
    # would make one of them invisible to review.
    seen: dict[str, dict] = {}
    for sub in subjects:
        clash = seen.get(sub["subject_id"])
        if clash is not None:
            raise ConsumedSubjectsRefusal(
                f"subject id collision on {sub['subject_id']!r}: "
                f"{clash} and {sub} — one of them would be invisible "
                "to a review")
        seen[sub["subject_id"]] = sub

    if not subjects:
        raise ConsumedSubjectsRefusal(
            "the consumed schema resolved to zero reviewable "
            "subjects — every column was excluded as a row "
            "identifier, so the run consumes nothing a review "
            "could cover")

    # C18: canonical digests over the COMPLETE ordered file list,
    # both sides, with role and side named. A single role's x
    # digest can no longer stand for the whole consumed set.
    schema_digest = hashlib.sha256(json.dumps(
        {"x": {r: headers[r] for r in sorted(headers)},
         "y": {r: y_headers[r] for r in sorted(y_headers)},
         "target_column": config.get("target_column")},
        sort_keys=True).encode()).hexdigest()
    ordered = sorted(files_in_order)
    data_digest = hashlib.sha256("|".join(
        f"{role}:{side}:{key}:{sha}"
        for role, side, key, _path, sha in ordered
    ).encode()).hexdigest()
    # The PARTITION contract says which file plays which role, so
    # the path belongs here — renaming or swapping a partition
    # changes the contract even when the bytes happen to match.
    # A SUBJECT id still never depends on a path.
    partitions_digest = hashlib.sha256(json.dumps(
        {"files": [{"role": role, "side": side, "key": key,
                    "path": path, "sha256": sha}
                   for role, side, key, path, sha in ordered],
         "roles": sorted(partitions),
         "temporal_contract": {
             "row_identifier_columns": [
                 c for c in base if c in NON_FEATURE_COLUMNS],
             "target_column": config.get("target_column"),
             "target_side": ("y" if any(
                 config.get("target_column") in y_headers[r]
                 for r in y_headers) else "x")}},
        sort_keys=True).encode()).hexdigest()

    plugins = resolve_plugin_modules(config)
    code_digest, code_inventory = code_identity(
        CODE_ROOT, local_code_surface(CODE_ROOT), plugins)

    resolved = {
        "dataset_id": dataset_id,
        "code_inventory": code_inventory,
        "plugins_resolved": plugins,
        "subjects": sorted(subjects,
                           key=lambda s: s["subject_id"]),
        "subject_ids": sorted(s["subject_id"]
                              for s in subjects),
        "excluded_columns": excluded,
        "partitions": partitions,
        "schema": base,
        "digests": {
            "data": data_digest,
            "schema": schema_digest,
            "partitions": partitions_digest,
            "code": code_digest,
        },
        "derivation": "columns read from the partition headers "
                      "named by this run's own contract; no "
                      "window built, no scaler fitted, no row "
                      "loaded",
    }

    # the config's list is an ASSERTION, never the source
    declared = config.get("eligibility_subjects")
    if declared is not None:
        declared_set = set(declared)
        derived_set = set(resolved["subject_ids"])
        if declared_set != derived_set:
            missing = sorted(derived_set - declared_set)
            extra = sorted(declared_set - derived_set)
            raise ConsumedSubjectsRefusal(
                "eligibility_subjects does not match the "
                f"consumed set (not declared: {missing}; "
                f"declared but not consumed: {extra}) — the "
                "list is an assertion about the run, and this "
                "assertion is false")
    return resolved
