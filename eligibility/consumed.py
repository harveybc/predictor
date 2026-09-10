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


class ConsumedSubjectsRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def subject_id(dataset_id: str, column: str) -> str:
    """Stable logical id. The physical path is never part of it,
    so relocating a file does not rename a variable."""
    return (f"{dataset_id}::{column}")


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


def code_identity(repo_root: Path, files: list[str]) -> str:
    """One digest over the code that will consume the data."""
    h = hashlib.sha256()
    for rel in sorted(files):
        p = Path(repo_root) / rel
        h.update(rel.encode())
        h.update(sha256_file(p).encode() if p.is_file()
                 else b"ABSENT")
    return h.hexdigest()


CONSUMING_CODE = (
    "app/main.py",
    "eligibility/gate.py",
    "eligibility/consumed.py",
    "eligibility/integration.py",
)


def resolve_consumed_subjects(config: dict, *,
                              repo_root: Path) -> dict:
    """The single derivation. Returns the exact consumed set."""
    repo_root = Path(repo_root)
    partitions, headers = {}, {}
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
        entry = {"role": role, "x_file": str(Path(xpath)),
                 "x_sha256": sha256_file(p),
                 "x_bytes": p.stat().st_size}
        ypath = config.get(ykey)
        if ypath:
            yp = (repo_root / ypath) \
                if not Path(ypath).is_absolute() else Path(ypath)
            if not yp.is_file():
                raise ConsumedSubjectsRefusal(
                    f"{role}: the contract names {ykey}="
                    f"{ypath!r} but the file does not exist")
            entry["y_file"] = str(Path(ypath))
            entry["y_sha256"] = sha256_file(yp)
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
    for column in base:
        if not column.strip():
            raise ConsumedSubjectsRefusal(
                "the consumed schema contains an unnamed column "
                "— a column without an identity cannot be "
                "reviewed")
        if column in NON_FEATURE_COLUMNS:
            excluded.append({"column": column,
                             "reason": "ROW_IDENTIFIER"})
            continue
        role = "target" if column == target else "input"
        subjects.append({
            "subject_id": subject_id(dataset_id, column),
            "column": column,
            "dataset_id": dataset_id,
            "contract_role": role,
        })

    if not subjects:
        raise ConsumedSubjectsRefusal(
            "the consumed schema resolved to zero reviewable "
            "subjects — every column was excluded as a row "
            "identifier, so the run consumes nothing a review "
            "could cover")

    schema_digest = hashlib.sha256(
        "\n".join(base).encode()).hexdigest()
    partitions_digest = hashlib.sha256("|".join(
        f"{r}:{partitions[r]['x_sha256']}"
        for r in sorted(partitions)).encode()).hexdigest()
    data_digest = partitions[base_role]["x_sha256"]

    resolved = {
        "dataset_id": dataset_id,
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
            "code": code_identity(repo_root,
                                  list(CONSUMING_CODE)),
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
