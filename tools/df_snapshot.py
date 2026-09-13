#!/usr/bin/env python3
"""C152, C153 (order 2026-09-13): the data a fit or a transform may consume,
materialized from a sealed dataset contract and re-derived at the last point
of use.

A bare array, a self-declared role string or a hand-built object never grants
a fit or a transform. A snapshot is built only by ``FitSnapshot.from_contract``
or ``TransformSnapshot.from_contract``, which read the contract's OBSERVED file
bytes through a loader, check them against the contract's digests, decode and
slice them. Every consumer calls ``verify_fit_snapshot`` or
``verify_transform_snapshot``, which re-derive, in this order:

``contract_digest``        the contract validates and its sha256 re-derives;
``matrix_digest``          sha256 of the float64 C-order matrix;
``source_rederive``        the loader's bytes still match the contract, and
                           decoding and slicing them gives the same matrix,
                           timestamps and availability;
``snapshot_digest``        the snapshot's own digest over all its facts;
``monotonic_timestamps``   timestamps strictly increase (no duplicates);
``range_in_partition``     (fit) ``[start, end)`` lies inside the partition of
                           the declared role;
``role_allowed``           (fit) the role is one the design allows;
``later_partition_exclusion`` (fit) every later partition is listed as
                           excluded and none of its rows is included;
``availability``           (transform) no row is available after its decision
                           instant.

Each check can be switched off only through ``GUARDS`` for mutation testing
(C159); production never touches it.

Time: a SAMPLE_INDEX contract has timestamps ``0..n-1``; otherwise the
contract's unique ``TIMESTAMPS`` file (int64, one per row) is used.
Availability follows the contract rule: SAMPLE_INDEX, INSTANT, PERIOD_END and
BAR_CLOSE make a row available at its timestamp plus the declared numeric delay
(zero when not numeric); INSTANT_PLUS_DECLARED_DELAY requires a numeric delay;
PUBLICATION_TIME and UNKNOWN refuse. A transform's decision instant for a row
is its timestamp plus a declared ``decision_lag``.

Python cannot seal an object absolutely. The constructor token keeps
accidental construction out, and the re-derivation above makes a forged
snapshot refuse unless every fact it carries is true of the contract's bytes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SNAPSHOT_SCHEMA = "df_snapshot.v1"
PARTITION_ORDER = ("TRAIN", "CALIBRATION", "CONFIRMATION")
DESIGN_FIT_ROLES = ("TRAIN", "CALIBRATION")
_AVAILABILITY_AT_TIMESTAMP = ("SAMPLE_INDEX", "INSTANT", "PERIOD_END", "BAR_CLOSE")

GUARDS = {name: True for name in (
    "contract_digest", "matrix_digest", "source_rederive", "snapshot_digest", "monotonic_timestamps",
    "range_in_partition", "role_allowed", "later_partition_exclusion", "availability")}


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("df_contract")


class SnapshotRefusal(Exception):
    """Typed refusal; every message starts with REFUSED."""

    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)}, sort_keys=True).encode())
    h.update(b"\n")
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True, order="C")
    a.flags.writeable = False
    return a


_TOKEN = object()


# ------------------------------------------------------------------ source
def _read(loader, name: str) -> bytes:
    if callable(loader):
        blob = loader(name)
    elif isinstance(loader, dict):
        blob = loader.get(name)
    else:
        raise SnapshotRefusal("loader must be a callable or a mapping name -> bytes")
    if not isinstance(blob, (bytes, bytearray)):
        raise SnapshotRefusal(f"loader returned no bytes for {name!r}")
    return bytes(blob)


def _file_entry(contract: dict, role: str) -> dict:
    hits = [f for f in contract["files"] if f["role"] == role]
    if len(hits) != 1:
        raise SnapshotRefusal(f"contract must bind exactly one {role} file, found {len(hits)}")
    return hits[0]


def _checked_bytes(loader, entry: dict) -> bytes:
    blob = _read(loader, entry["name"])
    if len(blob) != entry["bytes"] or hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise SnapshotRefusal(f"source bytes of {entry['name']!r} do not match the contract digest")
    return blob


def _npy(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _materialize(contract: dict, loader) -> dict:
    """Whole-dataset arrays and the file entries they came from."""
    problems = C.validate(contract)
    if problems:
        raise SnapshotRefusal(f"contract does not validate: {problems[:3]}")
    obs_entry = _file_entry(contract, "OBSERVED")
    raw = _npy(_checked_bytes(loader, obs_entry))
    V = int(contract["panel"]["n_series"])
    time = contract["time"]
    files = [obs_entry]
    if time["timestamp_meaning"] == "SAMPLE_INDEX":
        T = int(time["range_end"]) - int(time["range_start"]) + 1
        ts = None
    else:
        ts_entry = _file_entry(contract, "TIMESTAMPS")
        ts = _npy(_checked_bytes(loader, ts_entry))
        files.append(ts_entry)
        if ts.ndim != 1 or ts.dtype.kind not in "iu":
            raise SnapshotRefusal("TIMESTAMPS file must be a 1-D integer array")
        T = int(ts.shape[0])
    if raw.ndim == 1 and V == 1:
        raw = raw[:, None]
    if raw.ndim != 2:
        raise SnapshotRefusal(f"OBSERVED array must be 2-D, got shape {raw.shape}")
    if raw.shape == (V, T) and V != T:
        raw = raw.T
    if raw.shape != (T, V):
        raise SnapshotRefusal(f"OBSERVED array shape {raw.shape} is not (T={T}, V={V}) or (V, T)")
    if raw.dtype.kind not in "fiu":
        raise SnapshotRefusal(f"OBSERVED dtype {raw.dtype} is not real numeric")
    if ts is None:
        ts = np.arange(T, dtype=np.int64)
    ts = ts.astype(np.int64)
    rule, delay = time["availability_rule"], time["availability_delay_seconds"]
    numeric = type(delay) in (int, float)
    if rule in _AVAILABILITY_AT_TIMESTAMP:
        avail = ts + (np.int64(delay) if numeric else np.int64(0))
    elif rule == "INSTANT_PLUS_DECLARED_DELAY":
        if not numeric:
            raise SnapshotRefusal("INSTANT_PLUS_DECLARED_DELAY needs a numeric availability delay")
        avail = ts + np.int64(delay)
    else:
        raise SnapshotRefusal(f"availability rule {rule} gives no per-row available-at")
    return {"matrix": np.asarray(raw, dtype=np.float64), "timestamps": ts, "available_at": avail,
            "files": tuple((f["name"], int(f["bytes"]), f["sha256"]) for f in files), "n_rows": T}


def _columns(contract: dict, names) -> tuple:
    vs = [v for v in contract["variables"] if v["role"] == "INPUT_CANDIDATE"]
    by_name = {v["name"]: (i, v) for i, v in enumerate(vs)}
    if names is None:
        names = [v["name"] for v in vs]
    idx, ids = [], []
    for n in names:
        if n not in by_name:
            raise SnapshotRefusal(f"column {n!r} is not an INPUT_CANDIDATE variable of the contract")
        pos, v = by_name[n]
        idx.append(int(v["original_fields"].get("variable_index", pos)))
        ids.append(v["variable_id"])
    if len(set(ids)) != len(ids):
        raise SnapshotRefusal("a column is selected twice")
    return tuple(idx), tuple(names), tuple(ids)


def _partition_ranges(contract: dict) -> dict:
    b = contract["partitions"]["boundaries"]
    return {p: (int(b[p.lower()][0]), int(b[p.lower()][1])) for p in PARTITION_ORDER}


def _covered(ranges: dict, start: int, end: int) -> tuple:
    return tuple(p for p in PARTITION_ORDER if ranges[p][0] < end and start < ranges[p][1])


# --------------------------------------------------------------- snapshots
@dataclass(frozen=True, eq=False)
class _Snapshot:
    token: object = field(repr=False, compare=False)
    kind: str
    schema: str
    dataset_id: str
    contract_sha256: str
    contract_json: bytes = field(repr=False)
    source_files: tuple
    column_index: tuple
    column_names: tuple
    column_ids: tuple
    timestamp_meaning: str
    timestamps: np.ndarray = field(repr=False)
    timestamps_sha256: str
    timestamps_strictly_increasing: bool
    availability_rule: str
    available_at: np.ndarray = field(repr=False)
    availability_sha256: str
    start: int
    end: int
    matrix: np.ndarray = field(repr=False)
    matrix_sha256: str
    loader: object = field(repr=False, compare=False)
    snapshot_sha256: str

    def __post_init__(self):
        if self.token is not _TOKEN:
            raise SnapshotRefusal("a snapshot is built only by from_contract, never by hand")

    @classmethod
    def _build(cls, contract, loader, m, s, e, columns, **extra):
        idx, names, ids = _columns(contract, columns)
        mat = _readonly(m["matrix"][s:e][:, list(idx)])
        ts = _readonly(m["timestamps"][s:e])
        av = _readonly(m["available_at"][s:e])
        body = dict(token=_TOKEN, kind=cls.__name__, schema=SNAPSHOT_SCHEMA, dataset_id=contract["dataset_id"],
                    contract_sha256=contract["contract_sha256"], contract_json=C.canonical(contract),
                    source_files=m["files"], column_index=idx, column_names=names, column_ids=ids,
                    timestamp_meaning=contract["time"]["timestamp_meaning"], timestamps=ts,
                    timestamps_sha256=array_sha256(ts),
                    timestamps_strictly_increasing=bool(np.all(np.diff(ts) > 0)),
                    availability_rule=contract["time"]["availability_rule"], available_at=av,
                    availability_sha256=array_sha256(av), start=s, end=e, matrix=mat,
                    matrix_sha256=array_sha256(mat), loader=loader, snapshot_sha256="", **extra)
        draft = cls(**body)
        body["snapshot_sha256"] = C.sha_obj(draft.facts())
        return cls(**body)

    @property
    def contract(self) -> dict:
        return json.loads(self.contract_json)

    def facts(self) -> dict:
        skip = {"token", "contract_json", "timestamps", "available_at", "matrix", "loader", "snapshot_sha256"}
        out = {}
        for k in self.__dataclass_fields__:
            if k in skip:
                continue
            v = getattr(self, k)
            out[k] = list(v) if isinstance(v, tuple) else v
        out["source_files"] = [list(f) for f in self.source_files]
        return out


@dataclass(frozen=True, eq=False)
class FitSnapshot(_Snapshot):
    role: str = ""
    allowed_roles: tuple = ()
    excluded_partitions: tuple = ()

    @classmethod
    def from_contract(cls, contract: dict, role: str, loader, *, start: int | None = None,
                      end: int | None = None, columns=None) -> "FitSnapshot":
        if role not in PARTITION_ORDER:
            raise SnapshotRefusal(f"unknown role {role!r}")
        m = _materialize(contract, loader)
        ranges = _partition_ranges(contract)
        s = ranges[role][0] if start is None else int(start)
        e = ranges[role][1] if end is None else int(end)
        if not 0 <= s < e <= m["n_rows"]:
            raise SnapshotRefusal(f"range [{s}, {e}) is not a non-empty range of the dataset")
        later = PARTITION_ORDER[PARTITION_ORDER.index(role) + 1:]
        return cls._build(contract, loader, m, s, e, columns, role=role, allowed_roles=DESIGN_FIT_ROLES,
                          excluded_partitions=tuple((p, *ranges[p]) for p in later))

@dataclass(frozen=True, eq=False)
class TransformSnapshot(_Snapshot):
    partitions: tuple = ()
    decision_lag: int = 0
    decision_at: np.ndarray = field(default=None, repr=False)
    decision_sha256: str = ""

    @classmethod
    def from_contract(cls, contract: dict, loader, *, start: int, end: int, columns=None,
                      decision_lag: int = 0) -> "TransformSnapshot":
        if type(decision_lag) is not int or decision_lag < 0:
            raise SnapshotRefusal("decision_lag must be a non-negative integer")
        m = _materialize(contract, loader)
        s, e = int(start), int(end)
        if not 0 <= s < e <= m["n_rows"]:
            raise SnapshotRefusal(f"range [{s}, {e}) is not a non-empty range of the dataset")
        dec = _readonly(m["timestamps"][s:e] + np.int64(decision_lag))
        return cls._build(contract, loader, m, s, e, columns,
                          partitions=_covered(_partition_ranges(contract), s, e),
                          decision_lag=decision_lag, decision_at=dec, decision_sha256=array_sha256(dec))

    def facts(self) -> dict:
        out = super().facts()
        out.pop("decision_at", None)
        return out

    def row(self, i: int) -> "TransformRow":
        verify_transform_snapshot(self)
        return TransformRow._make(self, i)


@dataclass(frozen=True, eq=False)
class TransformRow:
    """One row of a verified TransformSnapshot, for the incremental path."""
    token: object = field(repr=False, compare=False)
    dataset_id: str
    contract_sha256: str
    column_ids: tuple
    index: int
    timestamp: int
    available_at: int
    decision_at: int
    values: np.ndarray = field(repr=False)
    row_sha256: str

    def __post_init__(self):
        if self.token is not _TOKEN:
            raise SnapshotRefusal("a row is taken from a verified TransformSnapshot, never built by hand")

    def facts(self) -> dict:
        return {"dataset_id": self.dataset_id, "contract_sha256": self.contract_sha256,
                "column_ids": list(self.column_ids), "index": self.index, "timestamp": self.timestamp,
                "available_at": self.available_at, "decision_at": self.decision_at,
                "values_sha256": array_sha256(self.values)}

    @classmethod
    def _make(cls, snap: TransformSnapshot, i: int) -> "TransformRow":
        if type(i) is not int or not 0 <= i < snap.end - snap.start:
            raise SnapshotRefusal(f"row {i!r} is outside the snapshot")
        body = dict(token=_TOKEN, dataset_id=snap.dataset_id, contract_sha256=snap.contract_sha256,
                    column_ids=snap.column_ids, index=snap.start + i, timestamp=int(snap.timestamps[i]),
                    available_at=int(snap.available_at[i]), decision_at=int(snap.decision_at[i]),
                    values=_readonly(snap.matrix[i]), row_sha256="")
        body["row_sha256"] = C.sha_obj(cls(**body).facts())
        return cls(**body)


# ------------------------------------------------------------ verification
def _refuse_if(guard: str, bad: bool, msg: str) -> None:
    if GUARDS[guard] and bad:
        raise SnapshotRefusal(msg)


def _verify_common(s: _Snapshot, cls) -> dict:
    if not isinstance(s, cls) or s.token is not _TOKEN or s.schema != SNAPSHOT_SCHEMA or s.kind != cls.__name__:
        raise SnapshotRefusal(f"a {cls.__name__} built by from_contract is required, got {type(s).__name__}")
    contract = s.contract
    _refuse_if("contract_digest", bool(C.validate(contract)) or contract.get("contract_sha256") != s.contract_sha256
               or contract.get("dataset_id") != s.dataset_id, "contract digest does not re-derive")
    _refuse_if("matrix_digest", s.matrix.dtype != np.float64 or not s.matrix.flags.c_contiguous
               or array_sha256(s.matrix) != s.matrix_sha256
               or array_sha256(s.timestamps) != s.timestamps_sha256
               or array_sha256(s.available_at) != s.availability_sha256,
               "matrix digest does not re-derive (bytes changed after materialization)")
    if GUARDS["source_rederive"]:
        try:
            m = _materialize(contract, s.loader)
            idx, _, ids = _columns(contract, list(s.column_names))
        except SnapshotRefusal as exc:
            raise SnapshotRefusal(f"source bytes do not re-derive: {exc}") from exc
        again = np.ascontiguousarray(m["matrix"][s.start:s.end][:, list(idx)])
        if (m["files"] != s.source_files or idx != s.column_index or ids != s.column_ids
                or again.shape != s.matrix.shape or again.tobytes() != s.matrix.tobytes()
                or not np.array_equal(m["timestamps"][s.start:s.end], s.timestamps)
                or not np.array_equal(m["available_at"][s.start:s.end], s.available_at)):
            raise SnapshotRefusal("source bytes do not re-derive the snapshot matrix, timestamps or availability")
    _refuse_if("snapshot_digest", C.sha_obj(s.facts()) != s.snapshot_sha256, "snapshot digest does not re-derive")
    _refuse_if("monotonic_timestamps", not bool(np.all(np.diff(s.timestamps) > 0)),
               "timestamps are not strictly increasing (unordered or duplicated rows; no duplicate policy "
               "is declared)")
    if s.matrix.shape[0] != s.end - s.start or s.timestamps.shape[0] != s.end - s.start:
        raise SnapshotRefusal("rows do not match the declared range")
    return contract


def verify_fit_snapshot(s) -> FitSnapshot:
    contract = _verify_common(s, FitSnapshot)
    ranges = _partition_ranges(contract)
    lo, hi = ranges.get(s.role, (0, -1))
    _refuse_if("range_in_partition", not (lo <= s.start < s.end <= hi),
               f"fit range [{s.start}, {s.end}) is not inside the {s.role} partition [{lo}, {hi})")
    _refuse_if("role_allowed", s.role not in DESIGN_FIT_ROLES or tuple(s.allowed_roles) != DESIGN_FIT_ROLES,
               f"fit role {s.role!r} is not allowed by the design {list(DESIGN_FIT_ROLES)}")
    if GUARDS["later_partition_exclusion"]:
        later = PARTITION_ORDER[PARTITION_ORDER.index(s.role) + 1:] if s.role in PARTITION_ORDER else ()
        want = tuple((p, *ranges[p]) for p in later)
        overlap = [p for p in later if ranges[p][0] < s.end and s.start < ranges[p][1]]
        if tuple(tuple(x) for x in s.excluded_partitions) != want or overlap:
            raise SnapshotRefusal(f"later partitions are not excluded from the fit (overlap {overlap})")
    return s


def verify_transform_snapshot(s) -> TransformSnapshot:
    contract = _verify_common(s, TransformSnapshot)
    ranges = _partition_ranges(contract)
    dec = s.timestamps + np.int64(s.decision_lag)
    if (s.decision_at is None or array_sha256(s.decision_at) != s.decision_sha256
            or not np.array_equal(dec, s.decision_at) or s.partitions != _covered(ranges, s.start, s.end)):
        raise SnapshotRefusal("decision instants or covered partitions do not re-derive")
    _refuse_if("availability", bool(np.any(s.available_at > s.decision_at)),
               "a row is available after its decision instant")
    return s


def verify_row(r) -> TransformRow:
    if not isinstance(r, TransformRow) or r.token is not _TOKEN:
        raise SnapshotRefusal(f"a TransformRow from a verified TransformSnapshot is required, got {type(r).__name__}")
    if C.sha_obj(r.facts()) != r.row_sha256:
        raise SnapshotRefusal("row digest does not re-derive")
    return r
