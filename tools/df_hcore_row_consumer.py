#!/usr/bin/env python3
"""Consumidor por filas de una representación ya materializada.

Lee un almacén que ya existe. No regenera el prefijo, no reentrena y no
abre la reserva. La igualdad de filas y de valores es exacta: no hay
tolerancia. Los digest salen de los helpers de DR05, no de un segundo
lector.

No hay en el repositorio una oración que defina «alcance». El estado es
NOT_DEFINED_IN_REPO. No se inventa un protocolo a partir del reach medido.
Los casos de fila son los cinco de abajo, y con bytes reales se añaden los
rechazos de mutación, digest y versión.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ALCANCE = "NOT_DEFINED_IN_REPO"
RECORD_SCHEMA = "df_dr05_prefix_output.v1"
STORE_SCHEMA = "df_dr05_prefix_store.v1"
DIGEST_KEYS = ("data_sha256", "design_sha256", "panel_sha256", "donor_weights_sha256")
ROW_CASES = (
    "EXACT_MATCH",
    "VALUE_MISMATCH",
    "ROW_COUNT_MISMATCH",
    "WRONG_IDENTITY",
    "MISSING_ARTIFACT",
)
REJECTIONS = ROW_CASES + ("MUTATED_ROW", "MUTATED_VALUE", "WRONG_DIGEST", "WRONG_VERSION")

_HELPERS = None


class RowConsumerRefusal(ValueError):
    def __init__(self, code: str, detail: str = ""):
        if code not in REJECTIONS:
            raise ValueError(f"unknown refusal {code}")
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


def _helpers():
    global _HELPERS
    if _HELPERS is None:
        path = Path(__file__).resolve().parent / "df_dr05_prefix_output.py"
        spec = importlib.util.spec_from_file_location("_hcore_dr05_helpers", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _HELPERS = mod
    return _HELPERS


def _version_string(obj: dict):
    version = obj.get("version")
    if isinstance(version, dict):
        return version.get("version")
    return version


def _derived(obj: dict) -> dict:
    version = obj.get("version")
    if isinstance(version, dict):
        return version.get("derived_from") or {}
    return {}


def _window_proof(record: dict, split: str):
    proofs = record.get("proofs") or {}
    fresh = proofs.get("fresh_process_reload") or {}
    report = fresh.get("report") or {}
    return (report.get("per_split") or {}).get(split)


def open_store(store: Path) -> dict:
    """Mide los bytes que hay en el almacén. No los compara todavía con un registro."""
    store = Path(store)
    manifest_path = store / "MANIFEST.json"
    if not store.is_dir() or not manifest_path.is_file():
        raise RowConsumerRefusal("MISSING_ARTIFACT", "manifest")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        raise RowConsumerRefusal("WRONG_IDENTITY", "manifest is not json") from None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("arrays"), dict):
        raise RowConsumerRefusal("WRONG_IDENTITY", "manifest arrays")
    helpers = _helpers()
    splits = {}
    for name, spec in manifest["arrays"].items():
        array_path = store / spec["array"]
        origins_path = store / spec["origins"]
        if not array_path.is_file() or not origins_path.is_file():
            raise RowConsumerRefusal("MISSING_ARTIFACT", name)
        values = np.load(array_path, mmap_mode="r")
        origins = np.load(origins_path)
        if origins.ndim != 1:
            raise RowConsumerRefusal("WRONG_IDENTITY", f"{name} origins")
        if values.shape[0] == 0 or origins.size == 0:
            raise RowConsumerRefusal("ROW_COUNT_MISMATCH", name)
        first = np.asarray(values[0])
        first_again = np.asarray(values[0])
        last = np.asarray(values[-1])
        splits[name] = {
            "sha256": helpers.sha_file(array_path),
            "origins_sha256": helpers.sha_file(origins_path),
            "manifest_sha256": spec.get("sha256"),
            "manifest_origins_sha256": spec.get("origins_sha256"),
            "shape": [int(x) for x in values.shape],
            "dtype": str(values.dtype),
            "nbytes": array_path.stat().st_size,
            "n_rows": int(values.shape[0]),
            "n_origins": int(origins.shape[0]),
            "origin_min": int(origins.min()) if origins.size else None,
            "origin_max": int(origins.max()) if origins.size else None,
            "first_window_digest": helpers.sha_array(first),
            "last_window_digest": helpers.sha_array(last),
            "endpoint_elements": int(first.size + last.size),
            "second_read_byte_equal": bool(
                first.tobytes() == first_again.tobytes() and np.array_equal(first, first_again)),
        }
    code = manifest.get("code_identity") if isinstance(manifest.get("code_identity"), dict) else None
    return {
        "alcance": ALCANCE,
        "schema": manifest.get("schema"),
        "version": _version_string(manifest),
        "derived_from": _derived(manifest),
        "donor_cell": manifest.get("donor_cell"),
        "prefix_ends_at": manifest.get("prefix_ends_at"),
        "code_identity": code,
        "manifest_sha256": helpers.sha_file(manifest_path),
        "splits": splits,
    }


def require_identity(measured: dict, record: dict) -> None:
    """Rechaza una identidad, un digest, una versión o un conteo que no es el del registro."""
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        raise RowConsumerRefusal("WRONG_IDENTITY", "record schema")
    if measured.get("schema") != STORE_SCHEMA:
        raise RowConsumerRefusal("WRONG_IDENTITY", "store schema")
    if _version_string(record) != measured.get("version"):
        raise RowConsumerRefusal("WRONG_VERSION", "version string")
    record_from = _derived(record)
    got_from = measured.get("derived_from") or {}
    for key in DIGEST_KEYS:
        if record_from.get(key) != got_from.get(key):
            raise RowConsumerRefusal("WRONG_IDENTITY", key)
    for key in ("donor_cell", "prefix_ends_at"):
        if record.get(key) != measured.get(key) or record_from.get(key) != got_from.get(key):
            raise RowConsumerRefusal("WRONG_IDENTITY", key)
    if "code_identity" in record:
        got = measured.get("code_identity") or {}
        claimed = record["code_identity"]
        if (claimed.get("revision") != got.get("revision")
                or claimed.get("module_sha256") != got.get("module_sha256")
                or claimed.get("worktree_has_uncommitted_changes")
                != got.get("worktree_has_uncommitted_changes")):
            raise RowConsumerRefusal("WRONG_IDENTITY", "code identity")
    if record.get("manifest_sha256") and record["manifest_sha256"] != measured.get("manifest_sha256"):
        raise RowConsumerRefusal("WRONG_DIGEST", "manifest")
    arrays = record.get("arrays") or {}
    if not arrays:
        raise RowConsumerRefusal("MISSING_ARTIFACT", "record names no array")
    for name, spec in arrays.items():
        got = measured["splits"].get(name)
        if got is None:
            raise RowConsumerRefusal("MISSING_ARTIFACT", name)
        if ("bytes" in spec and int(spec["bytes"]) != got["nbytes"]) or (
                got["sha256"] != spec.get("sha256")
                or got["sha256"] != got["manifest_sha256"]
                or got["origins_sha256"] != spec.get("origins_sha256")
                or got["origins_sha256"] != got["manifest_origins_sha256"]):
            raise RowConsumerRefusal("WRONG_DIGEST", name)
        if got["shape"] != [int(x) for x in spec["shape"]] or got["dtype"] != spec.get("dtype"):
            raise RowConsumerRefusal("WRONG_IDENTITY", f"{name} shape or dtype")
        declared = int(record["splits"][name]["n_origins"])
        if got["n_rows"] != declared or got["n_origins"] != declared or got["n_rows"] != got["n_origins"]:
            raise RowConsumerRefusal("ROW_COUNT_MISMATCH", name)
        span = record["splits"][name].get("origin_span")
        if span is not None and [got["origin_min"], got["origin_max"]] != [int(span[0]), int(span[1])]:
            raise RowConsumerRefusal("WRONG_IDENTITY", f"{name} origin span")
        proof = _window_proof(record, name)
        if proof is not None and (
                got["first_window_digest"] != proof.get("first_window_digest")
                or got["last_window_digest"] != proof.get("last_window_digest")
                or not got["second_read_byte_equal"]):
            raise RowConsumerRefusal("VALUE_MISMATCH", f"{name} window")


def equality_report(measured: dict, record: dict) -> dict:
    require_identity(measured, record)
    splits = {}
    for name in record["arrays"]:
        got = measured["splits"][name]
        proof = _window_proof(record, name)
        splits[name] = {
            "rows": got["n_rows"],
            "row_count_equal": True,
            "origin_span_equal": True,
            "sha256": got["sha256"],
            "origins_sha256": got["origins_sha256"],
            "file_sha256_matches_record": True,
            "endpoint_rows": 2,
            "endpoint_elements": got["endpoint_elements"],
            "endpoint_digests": "MATCH" if proof is not None else "NOT_IN_RECORD",
            "second_read_byte_equal": got["second_read_byte_equal"],
        }
    return {
        "alcance": ALCANCE,
        "bytes": "MATCH",
        "parity": "ROW_AND_VALUE_EQUAL",
        "version": measured["version"],
        "splits": splits,
    }


def judge_rows(reference, presented, reference_origins=None, presented_origins=None) -> str:
    """Eje 0 es la fila direccionable. Una ventana suelta se envuelve antes de entrar."""
    ref = np.asarray(reference)
    pre = np.asarray(presented)
    if ref.ndim < 1 or pre.ndim < 1:
        return "VALUE_MISMATCH"
    if reference_origins is not None or presented_origins is not None:
        if reference_origins is None or presented_origins is None:
            return "WRONG_IDENTITY"
        left = np.asarray(reference_origins)
        right = np.asarray(presented_origins)
        if left.ndim != 1 or right.ndim != 1:
            return "WRONG_IDENTITY"
        if left.shape[0] != right.shape[0]:
            return "ROW_COUNT_MISMATCH"
        if not np.array_equal(left, right):
            return "WRONG_IDENTITY"
        if left.shape[0] != ref.shape[0] or right.shape[0] != pre.shape[0]:
            return "ROW_COUNT_MISMATCH"
    if ref.shape[0] != pre.shape[0]:
        return "ROW_COUNT_MISMATCH"
    if ref.shape != pre.shape:
        return "VALUE_MISMATCH"
    if np.array_equal(ref, pre):
        return "EXACT_MATCH"
    unequal = [i for i in range(ref.shape[0]) if not np.array_equal(ref[i], pre[i])]
    if len(unequal) == 1:
        differed = int(np.count_nonzero(ref[unequal[0]] != pre[unequal[0]]))
        if differed == 1:
            return "MUTATED_VALUE"
        return "MUTATED_ROW"
    return "VALUE_MISMATCH"


def judge_window(stored_window, presented_window) -> str:
    stored = np.asarray(stored_window)
    presented = np.asarray(presented_window)
    return judge_rows(stored[None, ...], presented[None, ...])


def accept_rows(reference, presented, reference_origins=None, presented_origins=None) -> dict:
    code = judge_rows(reference, presented, reference_origins, presented_origins)
    if code != "EXACT_MATCH":
        raise RowConsumerRefusal(code)
    ref = np.asarray(reference)
    return {"code": code, "row_equal": True, "value_equal": True, "rows": int(ref.shape[0]),
            "elements": int(ref.size), "alcance": ALCANCE}


def accept_window(stored_window, presented_window) -> dict:
    code = judge_window(stored_window, presented_window)
    if code != "EXACT_MATCH":
        raise RowConsumerRefusal(code)
    stored = np.asarray(stored_window)
    return {"code": code, "row_equal": True, "value_equal": True, "rows": 1,
            "elements": int(stored.size), "alcance": ALCANCE}


def _array_path(store: Path, split: str, key: str) -> Path:
    manifest_path = store / "MANIFEST.json"
    if not manifest_path.is_file():
        raise RowConsumerRefusal("MISSING_ARTIFACT", "manifest")
    manifest = json.loads(manifest_path.read_text())
    spec = (manifest.get("arrays") or {}).get(split)
    if spec is None:
        raise RowConsumerRefusal("MISSING_ARTIFACT", split)
    path = store / spec[key]
    if not path.is_file():
        raise RowConsumerRefusal("MISSING_ARTIFACT", split)
    return path


def read_row_index(store: Path, split: str, index: int) -> np.ndarray:
    values = np.load(_array_path(Path(store), split, "array"), mmap_mode="r")
    if index < 0 or index >= values.shape[0]:
        raise RowConsumerRefusal("ROW_COUNT_MISMATCH", "index outside the materialized rows")
    return np.array(values[index], copy=True)


def read_origin(store: Path, split: str, origin: int) -> np.ndarray:
    origins = np.load(_array_path(Path(store), split, "origins"))
    hits = np.flatnonzero(origins == origin)
    if hits.size != 1:
        raise RowConsumerRefusal("WRONG_IDENTITY", "origin is not one row")
    return read_row_index(store, split, int(hits[0]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read an already materialized representation by row")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args(argv)
    record = json.loads(args.record.read_text())
    try:
        report = equality_report(open_store(args.store), record)
    except RowConsumerRefusal as refusal:
        # La ausencia se nombra antes de cualquier paridad.
        print(json.dumps({
            "bytes": "ABSENT" if refusal.code == "MISSING_ARTIFACT" else "PRESENT",
            "parity": "NOT_CLAIMED",
            "alcance": ALCANCE,
            "refused": refusal.code,
            "detail": refusal.detail,
        }, indent=1))
        return 1
    print(json.dumps(report, indent=1))
    return 0 if report["parity"] == "ROW_AND_VALUE_EQUAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
