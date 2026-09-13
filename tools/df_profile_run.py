#!/usr/bin/env python3
"""C130-C133 run, rebuilt under C147-C151: raw profiles, without targets, one bounded process per dataset.

For every dataset contract:

* public: each canonical panel of the C126 v2 root (panel.parquet with its
  sealed CONTRACT.json), timestamps parsed from the panel's labels with the
  parser's declared format;
* synthetic: each unit of the C128 bank, profiled on OBSERVED only, on the
  sample index. The clean and noise truth arrays are read only to verify the
  unit's recorded digests; they are never passed to a profile module;
* financial: each file of the C127 first batch, read from the financial-data
  checkout after its bytes are re-hashed against the contract, timestamps from
  its `timestamp` column.

The four modules run on the dataset's numeric variables: univariate
(df_profile_univariate), information (df_profile_information), multivariate
(df_profile_multivariate, train only, with its declared pair cap) and sampling
(df_sampling). Text variables are not profiled numerically and are listed as
such.

C149 columnar path. No table is read whole and no matrix of every column is
stacked: numeric columns are read one at a time, row group by row group, into
one preallocated float64 array (timestamps into one int64 array); univariate,
information and sampling rows are produced variable by variable; the
multivariate and effective-rank blocks hold only the capped columns and the
train rows the planner admits. Rows are validated (load_data_foundation
validate_row for their OLAP table) and appended to `profile.jsonl.partial` as
they are produced, flushed and fsynced periodically, renamed atomically at the
end, and bound by sha256. Module rows keep their existing format:
{"module": m, "row": r}.

C147 planner. Every metric group asks df_memory_plan before allocating, both in
a metadata-only preflight in the scheduler (upper-bound sizes) and inside the
task on actual sizes; both are written as df_fact_resource_estimate rows.

C150 isolation. Every dataset runs in its own process through
df_isolated_runner under a hard memory limit inside a systemd user slice, with
wall and CPU limits, a heartbeat, a stop file and a durable parent-written
terminal. The scheduler admits datasets by planned peak memory against a host
budget (not by count) with a concurrency cap; it skips a dataset whose durable
terminal is COMPLETED or REFUSED with the same contract and code digests, and
runs anything else as a new attempt in the write-once root.

C151 smoke: `--smoke-dataset BANK:selector --memory-limit BYTES` runs one
dataset through the same path (the lead runs the real worst case, not tests).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ("df_profile_univariate", "df_profile_information", "df_profile_multivariate", "df_sampling")
RUNNERS = {"df_profile_univariate": "run_univariate", "df_profile_information": "run_information",
           "df_profile_multivariate": "run_multivariate", "df_sampling": "run_sampling"}
CODE_FILES = MODULES + ("df_profile_run", "df_memory_plan", "df_isolated_runner")
RECEIPT_SCHEMA = "crispdm.data_foundation.profile_run_receipt.v2"
MANIFEST_SCHEMA = "crispdm.data_foundation.profile_run_manifest.v1"
RESUME_SKIP_STATUSES = ("COMPLETED", "REFUSED")
LOAD_JOB_MAX_BYTES = 256 * (1 << 20)
MIN_TASK_BYTES = 1 << 30
FSYNC_EVERY_ROWS = 2000
FSYNC_EVERY_SECONDS = 5.0
MUTATION_ENV = "DF_MUTATION_BYPASS_PREFLIGHT"
TABLE_OF_MODULE = {"df_profile_univariate": "df_fact_variable_profile", "df_sampling": "df_fact_sampling_quality",
                   "df_profile_information": "df_fact_information_metric"}


class DatasetRefusal(ValueError):
    pass


class StopRequested(Exception):
    pass


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256() -> str:
    doc = {m: _sha_file(HERE / f"{m}.py") for m in CODE_FILES}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def safe_name(dataset_id: str) -> str:
    return hashlib.sha256(dataset_id.encode()).hexdigest()[:20] + "__" + "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_id)[-80:]


# ----------------------------------------------------------------- inputs
def public_jobs(panel_root: Path) -> list[dict]:
    return [{"bank": "PUBLIC", "dir": str(d)} for d in sorted(Path(panel_root).iterdir())
            if d.is_dir() and (d / "CONTRACT.json").is_file()]


def synthetic_jobs(bank_root: Path) -> list[dict]:
    return [{"bank": "SYNTHETIC", "dir": str(d)} for d in sorted(Path(bank_root).iterdir()) if d.is_dir()]


def financial_jobs(contracts_file: Path, financial_root: Path) -> list[dict]:
    doc = json.loads(Path(contracts_file).read_text())
    return [{"bank": "FINANCIAL", "index": i, "contracts_file": str(contracts_file), "root": str(financial_root)}
            for i in range(len(doc["contracts"]))]


def read_numeric_column(pf, name: str, T: int, start: int = 0, end: int | None = None):
    """Rows [start, end) of one numeric parquet column into a new float64 array,
    reading only the overlapping row groups (nulls become NaN)."""
    import numpy as np
    end = T if end is None else end
    out = np.empty(end - start, dtype="float64")
    pos = 0
    for i in range(pf.metadata.num_row_groups):
        n_rg = pf.metadata.row_group(i).num_rows
        lo, hi = pos, pos + n_rg
        pos = hi
        if hi <= start or lo >= end:
            continue
        col = pf.read_row_group(i, columns=[name]).column(0)
        off = lo
        for chunk in col.chunks:
            a = chunk.to_numpy(zero_copy_only=False)
            s, e = max(start, off), min(end, off + a.size)
            if e > s:
                out[s - start:e - start] = a[s - off:e - off]
            off += a.size
            del a
        del col
    if pos != T:
        raise DatasetRefusal("parquet row groups do not add up to the file's row count")
    return out


def read_timestamp_column(pf, name: str, T: int):
    """int64 nanoseconds since the epoch in UTC, row group by row group; nulls as NaT."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    field = pf.schema_arrow.field(name)
    if not pa.types.is_timestamp(field.type):
        return _timestamps_via_pandas(pf, name, T)
    out = np.empty(T, dtype="int64")
    pos = 0
    nat = np.iinfo(np.int64).min
    for i in range(pf.metadata.num_row_groups):
        col = pf.read_row_group(i, columns=[name]).column(0)
        for chunk in col.chunks:
            c = chunk.cast(pa.timestamp("ns", tz=field.type.tz)).cast(pa.int64())
            c = pc.fill_null(c, nat)
            a = c.to_numpy(zero_copy_only=False)
            out[pos:pos + a.size] = a
            pos += a.size
            del a, c
        del col
    return out


def _timestamps_via_pandas(pf, name, T):
    """Fallback for a non-timestamp column, identical to the original whole-column parse."""
    import pandas as pd
    s = pd.Series(pf.read(columns=[name]).column(name).to_pylist())
    return pd.to_datetime(s, utc=True).to_numpy("datetime64[ns]").astype("int64")


def read_label_timestamps(pf, name: str, fmt: str, T: int):
    """Text labels parsed with an explicit format, row group by row group."""
    import numpy as np
    import pandas as pd
    out = np.empty(T, dtype="int64")
    pos = 0
    for i in range(pf.metadata.num_row_groups):
        labels = pf.read_row_group(i, columns=[name]).column(0).to_pylist()
        a = pd.to_datetime(pd.Series(labels), format=fmt).to_numpy("datetime64[ns]").astype("int64")
        out[pos:pos + a.size] = a
        pos += a.size
        del labels, a
    return out


class Source:
    """A dataset opened for columnar reading. Nothing but metadata is read on open."""

    def __init__(self, job: dict, verify_bytes: bool = True):
        self.job = job
        bank = job["bank"]
        self.kind = "parquet"
        self.text_timestamp = False
        if bank == "PUBLIC":
            d = Path(job["dir"])
            self.contract = json.loads((d / "CONTRACT.json").read_text())
            panel = next(f for f in self.contract["files"] if f["role"] == "DERIVED_CANONICAL_PANEL")
            self.path = d / "panel.parquet"
            if verify_bytes and _sha_file(self.path) != panel["sha256"]:
                raise DatasetRefusal("panel bytes differ from the contract")
            parser = self.contract["original_fields"]["parse_receipt"]["parser"]
            self.ts_format = _load("df_public_contract").TIMESTAMP_FORMATS[parser]
            self.ts_name, self.text_timestamp = "timestamp_label", True
        elif bank == "SYNTHETIC":
            import numpy as np
            d = Path(job["dir"])
            self.contract = _load("df_synthetic_contract").unit_contract(d)
            self.kind = "npy"
            obs = np.load(d / "observed_signal.npy", allow_pickle=False)
            rec = self.contract["original_fields"]["unit_record"]
            if obs.shape == (rec["n_variables"], rec["n_samples"]):
                obs = obs.T
            self.obs = obs
            self.ts_name = None
        else:
            doc = json.loads(Path(job["contracts_file"]).read_text())
            self.contract = doc["contracts"][job["index"]]
            f = self.contract["files"][0]
            self.path = Path(job["root"]) / f["name"]
            if verify_bytes and _sha_file(self.path) != f["sha256"]:
                raise DatasetRefusal("financial file bytes differ from the contract")
            self.ts_name = "timestamp"
        self._classify()

    def _classify(self):
        numeric, skipped = [], []
        if self.kind == "npy":
            self.T = int(self.obs.shape[0])
            names = {v["name"]: i for i, v in enumerate(self.contract["variables"])}
            self.rg_rows = self.T
            self.has_ts = False
            for v in self.contract["variables"]:
                if v["name"] == "timestamp" or v["role"] == "TIMESTAMP":
                    continue
                if self.obs.dtype.kind not in "fiu":
                    skipped.append({"variable_id": v["variable_id"], "name": v["name"],
                                    "reason": "NON_NUMERIC_NOT_PROFILED"})
                    continue
                numeric.append(v)
            self._col_index = names
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq
            self.pf = pq.ParquetFile(self.path)
            md = self.pf.metadata
            self.T = int(md.num_rows)
            self.rg_rows = max((md.row_group(i).num_rows for i in range(md.num_row_groups)), default=0)
            schema = self.pf.schema_arrow
            self.has_ts = self.ts_name in schema.names
            for v in self.contract["variables"]:
                if v["name"] == "timestamp" or v["role"] == "TIMESTAMP":
                    continue
                if v["name"] not in schema.names:
                    skipped.append({"variable_id": v["variable_id"], "name": v["name"], "reason": "COLUMN_NOT_READABLE"})
                    continue
                t = schema.field(v["name"]).type
                if not (pa.types.is_integer(t) or pa.types.is_floating(t)):
                    skipped.append({"variable_id": v["variable_id"], "name": v["name"],
                                    "reason": "NON_NUMERIC_NOT_PROFILED"})
                    continue
                numeric.append(v)
        self.numeric, self.skipped = numeric, skipped
        self.sub = dict(self.contract, variables=numeric)

    def column(self, j: int, start: int = 0, end: int | None = None):
        import numpy as np
        v = self.numeric[j]
        end = self.T if end is None else end
        if self.kind == "npy":
            return np.ascontiguousarray(self.obs[start:end, self._col_index[v["name"]]], dtype="float64")
        return read_numeric_column(self.pf, v["name"], self.T, start, end)

    def block(self, cols, start: int, end: int):
        import numpy as np
        out = np.empty((end - start, len(cols)), dtype="float64")
        for i, j in enumerate(cols):
            out[:, i] = self.column(j, start, end)
        return out

    def timestamps(self):
        if not self.has_ts or self.kind == "npy":
            return None
        if self.text_timestamp:
            return read_label_timestamps(self.pf, self.ts_name, self.ts_format, self.T)
        return read_timestamp_column(self.pf, self.ts_name, self.T)

    def meta(self) -> dict:
        b = self.contract["partitions"]["boundaries"]
        return {"T": self.T, "variables": [v["variable_id"] for v in self.numeric],
                "partitions": {p: [int(b[p][0]), int(b[p][1])] for p in ("train", "calibration", "confirmation")},
                "has_ts": bool(self.has_ts), "rg_rows": int(self.rg_rows), "text_timestamp": bool(self.text_timestamp)}


def load_job(job: dict):
    """-> (contract, X (T, V) float with NaN, timestamps int64 ns or None, skipped text variables).
    Small datasets only (tests, parity): refused above LOAD_JOB_MAX_BYTES; the runner never uses it."""
    import numpy as np
    src = Source(job)
    V = len(src.numeric)
    if 8 * src.T * V > LOAD_JOB_MAX_BYTES:
        raise DatasetRefusal(f"load_job refuses a {8 * src.T * V}-byte matrix; use the columnar runner")
    X = src.block(list(range(V)), 0, src.T) if V else np.empty((0, 0))
    return src.sub, X, src.timestamps(), src.skipped


# ------------------------------------------------------------------ rows
def route(module: str, row: dict, run_id: str, content_sha256: str):
    """(table, table row) exactly as df_load_d0_d2.profile_rows routes a module row."""
    L = _load("load_data_foundation")
    if "pair" in row:
        a, b = row["pair"]
        lag = row["estimator"].get("params", {}).get("lag")
        return "df_fact_pair_relation", L.metric_row(row, run_id=run_id, content_sha256=content_sha256,
                                                     variable_id_a=a, variable_id_b=b,
                                                     lag=lag if type(lag) is int else None)
    if "group_id" in row:
        if module == "df_profile_multivariate":
            members = row["estimator"].get("params", {}).get("members", [])
            return "df_fact_group_relation", L.metric_row(row, run_id=run_id, content_sha256=content_sha256,
                                                          group_id=row["group_id"],
                                                          members=members if isinstance(members, list) else [])
        if module == "df_profile_information":
            return "df_fact_information_metric", L.metric_row(row, run_id=run_id, content_sha256=content_sha256,
                                                              subject_kind="MATRIX", subject_id=row["group_id"])
        return "df_fact_sampling_quality", L.metric_row(row, run_id=run_id, content_sha256=content_sha256,
                                                        variable_id=f"DATASET:{row['group_id']}")
    if module in ("df_profile_univariate", "df_sampling"):
        return TABLE_OF_MODULE[module], L.metric_row(row, run_id=run_id, content_sha256=content_sha256,
                                                     variable_id=row["variable_id"])
    # information and multivariate per-variable rows, as df_load_d0_d2 routes them
    return "df_fact_information_metric", L.metric_row(row, run_id=run_id, content_sha256=content_sha256,
                                                      subject_kind="VARIABLE", subject_id=row["variable_id"])


class JsonlWriter:
    """Append-only JSONL into `<name>.partial`, flushed and fsynced every
    FSYNC_EVERY_ROWS rows or FSYNC_EVERY_SECONDS; close() renames atomically."""

    def __init__(self, path: Path):
        self.final = Path(path)
        self.partial = self.final.with_name(self.final.name + ".partial")
        if self.final.exists() or self.partial.exists():
            raise DatasetRefusal(f"{self.final.name} exists; outputs are write-once")
        self.f = open(self.partial, "w")
        self.rows = 0
        self._since, self._t = 0, time.time()

    def write(self, obj) -> None:
        self.f.write(json.dumps(obj, sort_keys=True, allow_nan=False) + "\n")
        self.rows += 1
        self._since += 1
        if self._since >= FSYNC_EVERY_ROWS or time.time() - self._t > FSYNC_EVERY_SECONDS:
            self.sync()

    def sync(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self._since, self._t = 0, time.time()

    def close(self) -> str:
        self.sync()
        self.f.close()
        os.rename(self.partial, self.final)
        dfd = os.open(self.final.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return _sha_file(self.final)

    def abandon(self):
        try:
            self.sync()
            self.f.close()
        except (OSError, ValueError):
            pass


# ----------------------------------------------------------------- worker
def _rss_bytes() -> int:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return 0


def _self_cgroup_peak():
    try:
        line = Path("/proc/self/cgroup").read_text().strip().splitlines()[0]
        return int((Path("/sys/fs/cgroup" + line.split("::", 1)[1]) / "memory.peak").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


class Heartbeat:
    def __init__(self, path: Path, dataset_id: str, every: float):
        self.path, self.every = Path(path), float(every)
        self.state = {"dataset_id": dataset_id, "module": None, "metric": None, "variable": None, "rows_written": 0}
        self._stop = threading.Event()
        self.beat()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def beat(self, **kw):
        self.state.update(kw)
        doc = dict(self.state, rss_bytes=_rss_bytes(), at=time.time())
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(doc, sort_keys=True))
            os.replace(tmp, self.path)
        except OSError:
            pass

    def _loop(self):
        while not self._stop.wait(self.every):
            self.beat()

    def stop(self):
        self._stop.set()
        self.beat()


def worker_main(job_file: Path) -> int:
    """The child: profile one dataset, write rows and estimates incrementally, then result.json."""
    spec = json.loads(Path(job_file).read_text())
    adir = Path(spec["attempt_dir"])
    mutation = os.environ.get(MUTATION_ENV) == "1"
    stop_file = Path(spec["stop_file"]) if spec.get("stop_file") else None
    hb = Heartbeat(adir / "heartbeat.json", spec.get("dataset_id") or "UNRESOLVED", spec["heartbeat_seconds"])
    result = {"status": "FAILED", "reason": "", "rows_written": 0, "variables_profiled": 0, "metrics_completed": 0,
              "metrics_missing": 0, "output_file": None, "output_sha256": None, "estimates_file": None,
              "mutation_bypass_preflight": mutation}
    writer = est_writer = None
    exit_code = 2
    try:
        src = Source(spec["job"])
        contract = src.sub
        ds = contract["dataset_id"]
        hb.beat(dataset_id=ds)
        content = contract.get("content_sha256") or ""
        content = content if re.fullmatch(r"[0-9a-f]{64}", content) else "0" * 64
        run_id = spec["run_id"]
        writer = JsonlWriter(adir / "profile.jsonl")
        est_writer = JsonlWriter(adir / "resource_estimates.jsonl")
        P = _load("df_memory_plan")
        U, I, M, S = (_load(m) for m in MODULES)
        L = _load("load_data_foundation")
        meta = src.meta()
        parts = [(p, *meta["partitions"][p]) for p in ("train", "calibration", "confirmation")]
        tr_s, tr_e = parts[0][1], parts[0][2]
        k, vm = P.pair_counts(len(src.numeric))
        ctx = {"T": src.T, "n_train": tr_e - tr_s, "has_ts": meta["has_ts"], "rg_rows": meta["rg_rows"],
               "k_pairs": k, "V_matrix": vm}
        planner = P.Planner(run_id=run_id, bank=spec["job"]["bank"], dataset_id=ds, budget_bytes=spec["budget_bytes"],
                            code_sha256=spec["code_sha256"], context=ctx, sink=est_writer.write)
        counts = {"completed": 0, "missing": 0, "invalid": 0}

        def gate_for(module):
            base = planner.for_module(module)

            def gate(group, key=None, partition=None, **sizes):
                if stop_file is not None and stop_file.exists():
                    raise StopRequested()
                hb.beat(module=module, metric=group, variable=key if isinstance(key, str) else json.dumps(key),
                        rows_written=writer.rows)
                if mutation:
                    return {"decision": "RUN_EXACT", "window": None}
                if group == "pair_coherence":
                    planner.ctx["pair_rows"] = planner.ctx.get("pair_rows", sizes.get("L"))
                d = base(group, key, partition, **sizes)
                if group == "pair_block" and d["window"]:
                    planner.ctx["pair_rows"] = d["window"][1] - d["window"][0]
                return d
            return gate

        policy = dict(U.UNIT_ROOT_POLICY, exact_max_n=U.MUTATION_EXACT_MAX_N) if mutation else None

        def emit(module, rows):
            for r in rows:
                table, trow = route(module, r, run_id, content)
                problems = L.validate_row(table, trow)
                if problems:
                    counts["invalid"] += 1
                    continue
                if r["status"] == "COMPLETED":
                    counts["completed"] += 1
                elif r["status"] in ("NOT_RUN", "FAILED"):
                    counts["missing"] += 1
                writer.write({"module": module, "row": r})

        V = len(src.numeric)
        vids = [v["variable_id"] for v in src.numeric]
        meaning = contract["time"]["timestamp_meaning"]
        if V:
            # the reader itself is planned before the first column is read
            if not mutation:
                d = planner.gate("df_profile_run", "reader_column", None, None, T=src.T, rg_rows=meta["rg_rows"],
                                 columns_held=1 + int(meta["has_ts"]), text_timestamp=int(meta["text_timestamp"]))
                if d["decision"] == "NOT_RUN_RESOURCE_BOUND":
                    raise DatasetRefusal("READER_EXCEEDS_BUDGET")
            ts = src.timestamps() if meta["has_ts"] else None
            U.layout(contract, src.T, V, ts)      # the same contract checks the whole-matrix path made
            g = gate_for("df_profile_univariate")
            emit("df_profile_univariate", U.timestamp_rows(ds, parts, ts, meaning, g))
            for j, vid in enumerate(vids):
                col = src.column(j)
                emit("df_profile_univariate", U.variable_rows(ds, vid, col, parts, g, policy))
                del col
            del ts
            g = gate_for("df_profile_information")
            for j, vid in enumerate(vids):
                col = src.column(j)
                emit("df_profile_information", I.variable_rows(ds, vid, col, parts, g))
                del col
            emit("df_profile_information", I.matrix_rows(ds, V, lambda s, e: src.block(list(range(V)), tr_s + s, tr_s + e),
                                                         tr_e - tr_s, g))
            g = gate_for("df_profile_multivariate")
            emit("df_profile_multivariate", M.multivariate_rows(ds, vids, lambda cols, s, e: src.block(cols, tr_s + s, tr_s + e),
                                                                tr_e - tr_s, g))
            g = gate_for("df_sampling")
            ts = src.timestamps() if meta["has_ts"] and meaning != "SAMPLE_INDEX" else None
            emit("df_sampling", S.timestamp_rows(contract, parts, ts, g))
            for pname, s, e in parts:
                reg, fs = S.partition_regularity(contract, s, e, ts)
                for j, vid in enumerate(vids):
                    x = src.column(j, s, e)
                    emit("df_sampling", S.variable_partition_rows(contract, vid, x, pname, s, e, reg, fs, ts is None,
                                                                  None, src.T, g))
                    del x
                del reg
            del ts
        result.update(rows_written=writer.rows, variables_profiled=V, metrics_completed=counts["completed"],
                      metrics_missing=counts["missing"], skipped_variables=src.skipped, dataset_id=ds,
                      contract_sha256=contract.get("contract_sha256"), planner_decisions=dict(planner.decisions),
                      invalid_rows=counts["invalid"])
        result["output_sha256"] = writer.close()
        result["output_file"] = writer.final.name
        result["estimates_sha256"] = est_writer.close()
        result["estimates_file"] = est_writer.final.name
        if counts["invalid"]:
            result.update(status="FAILED", reason=f"ROWS_FAILED_VALIDATION {counts['invalid']}")
        else:
            result.update(status="COMPLETED", reason="")
            exit_code = 0
    except StopRequested:
        result.update(status="INCONCLUSIVE", reason="STOP_FILE_REQUESTED; partial output kept unpromoted",
                      rows_written=writer.rows if writer else 0)
        exit_code = 0
    except (DatasetRefusal,) as exc:
        result.update(status="REFUSED", reason=f"{type(exc).__name__}: {exc}"[:500])
        exit_code = 3
    except _contract_refusal_types() as exc:
        result.update(status="REFUSED", reason=f"{type(exc).__name__}: {exc}"[:500])
        exit_code = 3
    except MemoryError as exc:
        result.update(status="FAILED", reason=f"MemoryError: {exc}"[:500], exception_type="MemoryError")
        exit_code = 4
    except Exception as exc:  # noqa: BLE001 - every outcome is recorded
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}"[:500], exception_type=type(exc).__name__)
        exit_code = 2
    finally:
        for w in (writer, est_writer):
            if w is not None and not w.f.closed:
                w.abandon()
        hb.stop()
    import resource
    result["child_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    result["cgroup_memory_peak_bytes"] = _self_cgroup_peak()
    text = json.dumps(result, sort_keys=True, default=str).replace(str(Path.home()), "~")
    tmp = adir / "result.json.tmp"
    with open(tmp, "w") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, adir / "result.json")
    return exit_code


def _contract_refusal_types():
    return (_load("df_contract").ContractRefusal,)


# -------------------------------------------------------------- scheduler
def _job_label(job: dict) -> str:
    return f"{job['bank']}:{job.get('dir') or job.get('index')}"


def _terminals(root: Path, sname: str):
    out = []
    for p in sorted((root / "terminals").glob(f"{sname}.attempt-*.json")):
        try:
            out.append((int(p.name.rsplit("attempt-", 1)[1].split(".")[0]), json.loads(p.read_text()), p))
        except (ValueError, OSError):
            continue
    return sorted(out)


def contract_identity(job: dict):
    """(dataset_id, contract_sha256) from the contract text alone, without reading data bytes;
    (None, None) when the contract itself cannot be read (e.g. a missing synthetic unit)."""
    try:
        if job["bank"] == "PUBLIC":
            c = json.loads((Path(job["dir"]) / "CONTRACT.json").read_text())
        elif job["bank"] == "FINANCIAL":
            c = json.loads(Path(job["contracts_file"]).read_text())["contracts"][job["index"]]
        else:
            return None, None
        csha = c.get("contract_sha256")
        return c["dataset_id"], (csha if isinstance(csha, str) and re.fullmatch(r"[0-9a-f]{64}", csha) else None)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None, None


def _redact(text: str) -> str:
    return text.replace(str(Path.home()), "~")


def run(out_dir: Path, jobs: list[dict], workers: int = 4, *, host_budget_bytes: int = 8 << 30,
        task_memory_bytes: int = 4 << 30, wall_seconds: float = 6 * 3600, cpu_seconds: int = 6 * 3600,
        heartbeat_seconds: float = 10.0, stop_file: Path | None = None, resume: bool = False,
        host_role: str = "COORDINATOR", slice_: str | None = None, mechanism: str | None = None,
        extra_env: dict | None = None, mutation_bypass_preflight: bool = False) -> dict:
    """Profile `jobs` into the write-once root `out_dir`. Returns the receipt (final or progress)."""
    IR = _load("df_isolated_runner")
    P = _load("df_memory_plan")
    out_dir = Path(out_dir)
    slice_ = slice_ or IR.DEFAULT_SLICE
    code = code_sha256()
    if host_role not in IR.HOST_ROLES:
        raise SystemExit(f"REFUSED: host_role must be one of {IR.HOST_ROLES}")
    if out_dir.exists():
        if not resume:
            raise SystemExit(f"REFUSED: {out_dir.name} exists; profile outputs are write-once")
        if (out_dir / "PROFILE_RUN_RECEIPT.json").exists():
            raise SystemExit(f"REFUSED: {out_dir.name} is sealed by its final receipt; resume is over")
        manifest = json.loads((out_dir / "RUN_MANIFEST.json").read_text())
    else:
        out_dir.mkdir(parents=True)
        (out_dir / "terminals").mkdir()
        (out_dir / "attempts").mkdir()
        created = IR.now_iso()
        manifest = {"schema": MANIFEST_SCHEMA, "created_at": created, "host_role": host_role,
                    "run_id": "df_profile_" + hashlib.sha256(f"{out_dir.name}|{code}|{created}".encode()).hexdigest()[:24],
                    "code_sha256_at_creation": code, "limit_ratio": IR.LIMIT_RATIO, "budget_ratio": IR.BUDGET_RATIO}
        IR.atomic_write_once(out_dir / "RUN_MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    run_id = manifest["run_id"]
    mechanism = mechanism or IR.detect_mechanism(slice_)
    label = IR.mechanism_label(mechanism, slice_)
    task_lim = IR.limits_for(task_memory_bytes)
    started = IR.now_iso()

    entries = {}
    pending = []
    for idx, job in enumerate(jobs):
        entries[idx] = {"bank": job["bank"], "job": _redact(_job_label(job)), "status": "NOT_STARTED"}
        pending.append(idx)

    def terminal(idx, sname, attempt, **fields):
        row = {k: None for k in IR.TERMINAL_KEYS}
        row.update(run_id=run_id, host_role=host_role, bank=jobs[idx]["bank"], code_sha256=code, rows_written=0,
                   variables_profiled=0, metrics_completed=0, metrics_missing=0, planned_peak_bytes=0,
                   memory_limit_bytes=0, limit_mechanism=label, wall_seconds=0.0, cpu_seconds=0.0,
                   started_at=IR.now_iso(), ended_at=IR.now_iso(), reason="")
        row.update(fields)
        row["reason"] = _redact(row["reason"] or "")
        path = out_dir / "terminals" / f"{sname}.attempt-{attempt}.json"
        IR.write_terminal(path, row)
        e = entries[idx]
        e.update(status=row["status"], dataset_id=row["dataset_id"], contract_sha256=row["contract_sha256"],
                 terminal=str(path.relative_to(out_dir)), reason=row["reason"], wall_seconds=row["wall_seconds"],
                 row_count=row["rows_written"], observed_peak_rss_bytes=row["observed_peak_rss_bytes"],
                 planned_peak_bytes=row["planned_peak_bytes"], memory_limit_bytes=row["memory_limit_bytes"])
        return row

    def prepare(idx):
        """Parent preflight: contract, bytes, metadata plan. -> task dict or None (terminal written)."""
        job = jobs[idx]
        try:
            src = Source(job)
        except Exception as exc:  # noqa: BLE001
            refused = isinstance(exc, (DatasetRefusal,) + _contract_refusal_types())
            ds_known, csha_known = contract_identity(job)
            ds_label = ds_known or "UNRESOLVED:" + _redact(_job_label(job))
            sname = safe_name(ds_label)
            attempt = len(_terminals(out_dir, sname)) + 1
            terminal(idx, sname, attempt, dataset_id=ds_label, contract_sha256=csha_known,
                     status="REFUSED" if refused else "FAILED", reason=f"{type(exc).__name__}: {exc}"[:500],
                     memory_limit_bytes=task_lim["memory_limit_bytes"])
            return None
        c = src.contract
        ds, csha = c["dataset_id"], c.get("contract_sha256")
        csha = csha if isinstance(csha, str) and re.fullmatch(r"[0-9a-f]{64}", csha) else None
        sname = safe_name(ds)
        prior = _terminals(out_dir, sname)
        if prior:
            _, last, lpath = prior[-1]
            if last["status"] in RESUME_SKIP_STATUSES and last["contract_sha256"] == csha and last["code_sha256"] == code:
                entries[idx].update(status=last["status"], dataset_id=ds, contract_sha256=csha,
                                    terminal=str(lpath.relative_to(out_dir)), reason=last["reason"],
                                    row_count=last["rows_written"], resumed_skip=True,
                                    file=(str(Path("attempts") / sname / f"attempt-{prior[-1][0]}" / "profile.jsonl")
                                          if last["status"] == "COMPLETED" else None),
                                    sha256=last["output_sha256"], numeric_variables=last["variables_profiled"])
                return None
        existing = [int(p.name.split("-", 1)[1]) for p in (out_dir / "attempts" / sname).glob("attempt-*")
                    if p.name.split("-", 1)[1].isdigit()]
        attempt = max([len(prior)] + existing) + 1
        adir = out_dir / "attempts" / sname / f"attempt-{attempt}"
        adir.mkdir(parents=True)
        meta = src.meta()
        k, vm = P.pair_counts(len(src.numeric))
        pre_rows = []
        planner = P.Planner(run_id=run_id, bank=job["bank"], dataset_id=ds, budget_bytes=task_lim["budget_bytes"],
                            code_sha256=code, context={"T": src.T, "n_train": meta["partitions"]["train"][1],
                                                       "has_ts": meta["has_ts"], "rg_rows": meta["rg_rows"],
                                                       "k_pairs": k, "V_matrix": vm},
                            sink=pre_rows.append, stage="PREFLIGHT_METADATA_UPPER_BOUND")
        plan = P.preflight(meta, planner) if meta["variables"] else {"planned_peak_bytes": P.BASE_PROCESS_BYTES,
                                                                     "reader_decision": "RUN_EXACT"}
        with open(adir / "preflight_estimates.jsonl", "w") as f:
            for r in pre_rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        entries[idx].update(numeric_variables=len(src.numeric), skipped_variables=src.skipped, dataset_id=ds)
        planned = int(plan["planned_peak_bytes"])
        if plan["reader_decision"] == "NOT_RUN_RESOURCE_BOUND" and not mutation_bypass_preflight:
            terminal(idx, sname, attempt, dataset_id=ds, contract_sha256=csha, status="REFUSED",
                     reason="READER_EXCEEDS_TASK_BUDGET", planned_peak_bytes=planned,
                     memory_limit_bytes=task_lim["memory_limit_bytes"])
            return None
        need = int(math.ceil(planned / (IR.LIMIT_RATIO * IR.BUDGET_RATIO))) + (1 << 20)
        assigned = task_memory_bytes if mutation_bypass_preflight else min(task_memory_bytes, max(MIN_TASK_BYTES, need))
        # integer flooring in the ratios must never leave the task budget below the planned peak
        while not mutation_bypass_preflight and IR.limits_for(assigned)["budget_bytes"] < planned \
                and assigned < task_memory_bytes:
            assigned = min(task_memory_bytes, assigned + (1 << 20))
        if assigned > host_budget_bytes:
            terminal(idx, sname, attempt, dataset_id=ds, contract_sha256=csha, status="REFUSED",
                     reason="TASK_MEMORY_EXCEEDS_HOST_BUDGET", planned_peak_bytes=planned,
                     memory_limit_bytes=IR.limits_for(assigned)["memory_limit_bytes"])
            return None
        lim = IR.limits_for(assigned)
        spec = {"job": job, "run_id": run_id, "attempt_dir": str(adir), "budget_bytes": lim["budget_bytes"],
                "heartbeat_seconds": heartbeat_seconds, "stop_file": str(stop_file) if stop_file else None,
                "code_sha256": code, "dataset_id": ds}
        (adir / "job.json").write_text(json.dumps(spec, sort_keys=True))
        env = dict(extra_env or {})
        if mutation_bypass_preflight:
            env[MUTATION_ENV] = "1"
        del src
        task = IR.Task(argv=["env", "-u", "PYTHONPATH", sys.executable, "-B", str(HERE / "df_profile_run.py"),
                             "--worker", str(adir / "job.json")],
                       name=sname[:40], attempt_dir=adir, assigned_bytes=assigned, wall_seconds=wall_seconds,
                       cpu_seconds=cpu_seconds, slice_=slice_, mechanism=mechanism, extra_env=env)
        return {"idx": idx, "task": task, "sname": sname, "attempt": attempt, "adir": adir, "ds": ds, "csha": csha,
                "planned": planned, "assigned": assigned}

    def finish(t):
        o = t["task"].outcome
        status, reason, ver = IR.classify(o, t["adir"], mutation_bypass_preflight)
        res = o["result"] or {}
        # The cgroup's own memory.peak is what MemoryMax enforces; the child's maxrss also counts shared
        # library pages and can exceed the limit, so it is only a fallback when no cgroup peak was read.
        observed = (res.get("cgroup_memory_peak_bytes") or o["cgroup_memory_peak"]
                    or res.get("child_maxrss_bytes") or o["child_maxrss_bytes"] or None)
        row = terminal(t["idx"], t["sname"], t["attempt"], dataset_id=t["ds"], contract_sha256=t["csha"],
                       status=status, reason=reason, rows_written=int(res.get("rows_written") or 0),
                       variables_profiled=int(res.get("variables_profiled") or 0),
                       metrics_completed=int(res.get("metrics_completed") or 0),
                       metrics_missing=int(res.get("metrics_missing") or 0), planned_peak_bytes=t["planned"],
                       observed_peak_rss_bytes=int(observed) if observed else None,
                       memory_limit_bytes=t["task"].lim["memory_limit_bytes"], wall_seconds=o["wall_seconds"],
                       cpu_seconds=o["cpu_seconds"],
                       output_file=(str((t["adir"] / "profile.jsonl").relative_to(out_dir))
                                    if status == "COMPLETED" else None),
                       output_sha256=ver["output_sha256"], started_at=o["started_at"], ended_at=o["ended_at"])
        e = entries[t["idx"]]
        if status == "COMPLETED":
            e.update(file=row["output_file"], sha256=row["output_sha256"], row_count=row["rows_written"])
        e.update(cgroup_memory_peak_bytes=o["cgroup_memory_peak"], assigned_bytes=t["assigned"],
                 attempt=t["attempt"], mutation_bypass_preflight=mutation_bypass_preflight)
        if status != "COMPLETED":
            e["error"] = row["reason"]

    running = []
    queue = list(pending)
    stopped = False
    while queue or running:
        still = []
        for t in running:
            if t["task"].poll():
                finish(t)
            else:
                still.append(t)
        running = still
        while queue and not stopped:
            if stop_file is not None and Path(stop_file).exists():
                stopped = True
                break
            if len(running) >= max(1, int(workers)):
                break
            idx = queue[0]
            if "prepared" not in entries[idx]:
                entries[idx]["prepared"] = True
                t = prepare(idx)
                if t is None:
                    queue.pop(0)
                    continue
                entries[idx]["_task"] = t
            t = entries[idx]["_task"]
            in_use = sum(r["assigned"] for r in running)
            if running and in_use + t["assigned"] > host_budget_bytes:
                break
            queue.pop(0)
            del entries[idx]["_task"]
            t["task"].start()
            running.append(t)
        if stopped and not running:
            break
        if running:
            time.sleep(IR.POLL_SECONDS)
    for e in entries.values():
        e.pop("prepared", None)
        t = e.pop("_task", None)
        if t is not None:        # prepared but never launched because of the stop file
            e["status"] = "NOT_STARTED"
            e["reason"] = "STOP_FILE_BEFORE_LAUNCH"
    results = [entries[i] for i in range(len(jobs))]
    statuses = IR.TERMINAL_STATUSES + ("NOT_STARTED",)
    counts_all = {s: sum(r["status"] == s for r in results) for s in statuses}
    counts = {s: c for s, c in counts_all.items() if c}
    receipt = {"schema": RECEIPT_SCHEMA, "run_id": run_id, "host_role": host_role, "code_sha256": code,
               "code_file_sha256": {m: _sha_file(HERE / f"{m}.py") for m in CODE_FILES},
               "datasets": results, "counts": counts, "counts_all": counts_all,
               "rows_total": sum(r.get("row_count") or 0 for r in results if r["status"] == "COMPLETED"),
               "limits": {"task": task_lim, "host_budget_bytes": host_budget_bytes, "max_concurrency": workers,
                          "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds, "limit_mechanism": label,
                          "resume_skip_statuses": list(RESUME_SKIP_STATUSES)},
               "started_at": started, "ended_at": IR.now_iso(),
               "rule": "profiles read observed data only; no target; fits and frozen bins on train only"}
    final = counts_all["NOT_STARTED"] == 0
    text = _redact(json.dumps(receipt, indent=1, sort_keys=True))
    if final:
        IR.atomic_write_once(out_dir / "PROFILE_RUN_RECEIPT.json", text + "\n")
    else:
        n = len(list(out_dir.glob("PROFILE_RUN_PROGRESS.*.json"))) + 1
        IR.atomic_write_once(out_dir / f"PROFILE_RUN_PROGRESS.{n}.json", text + "\n")
    receipt["final"] = final
    return receipt


def select_job(selector: str, a) -> dict:
    """BANK:selector -> one job. PUBLIC:<panel dir name>, SYNTHETIC:<unit dir name>, FINANCIAL:<index or dataset_id>."""
    bank, _, sel = selector.partition(":")
    bank = bank.upper()
    if bank == "PUBLIC":
        jobs = [j for j in public_jobs(a.public_panels) if Path(j["dir"]).name == sel]
    elif bank == "SYNTHETIC":
        jobs = [j for j in synthetic_jobs(a.synthetic_bank) if Path(j["dir"]).name == sel]
    elif bank == "FINANCIAL":
        doc = json.loads(Path(a.financial_contracts).read_text())
        idx = [i for i, c in enumerate(doc["contracts"]) if str(i) == sel or c["dataset_id"] == sel]
        jobs = [{"bank": "FINANCIAL", "index": i, "contracts_file": str(a.financial_contracts),
                 "root": str(a.financial_root)} for i in idx]
    else:
        jobs = []
    if len(jobs) != 1:
        raise SystemExit(f"REFUSED: selector {selector!r} matches {len(jobs)} datasets")
    return jobs[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--public-panels", type=Path)
    ap.add_argument("--synthetic-bank", type=Path)
    ap.add_argument("--financial-contracts", type=Path)
    ap.add_argument("--financial-root", type=Path)
    ap.add_argument("--workers", type=int, default=2, help="maximum concurrent datasets")
    ap.add_argument("--limit", type=int, default=None, help="per bank, for smoke runs")
    ap.add_argument("--host-budget", type=int, default=8 << 30, help="bytes shared by concurrent datasets")
    ap.add_argument("--task-memory", type=int, default=4 << 30, help="bytes assigned to one dataset at most")
    ap.add_argument("--memory-limit", type=int, default=None,
                    help="hard MemoryMax bytes of the task (assigned = limit / LIMIT_RATIO)")
    ap.add_argument("--wall-seconds", type=float, default=6 * 3600)
    ap.add_argument("--cpu-seconds", type=int, default=6 * 3600)
    ap.add_argument("--heartbeat-seconds", type=float, default=10.0)
    ap.add_argument("--stop-file", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--host-role", default="COORDINATOR")
    ap.add_argument("--slice", default=None)
    ap.add_argument("--smoke-dataset", default=None, help="BANK:selector; runs exactly one dataset")
    a = ap.parse_args(argv)
    if a.worker:
        return worker_main(a.worker)
    if not a.out:
        ap.error("--out is required")
    IR = _load("df_isolated_runner")
    task_memory = a.task_memory
    if a.memory_limit:
        task_memory = int(math.ceil(a.memory_limit / IR.LIMIT_RATIO)) + 1
    if a.smoke_dataset:
        jobs = [select_job(a.smoke_dataset, a)]
    else:
        jobs = []
        if a.public_panels:
            jobs += public_jobs(a.public_panels)[:a.limit]
        if a.synthetic_bank:
            jobs += synthetic_jobs(a.synthetic_bank)[:a.limit]
        if a.financial_contracts:
            jobs += financial_jobs(a.financial_contracts, a.financial_root)[:a.limit]
    r = run(a.out, jobs, a.workers, host_budget_bytes=max(a.host_budget, task_memory), task_memory_bytes=task_memory,
            wall_seconds=a.wall_seconds, cpu_seconds=a.cpu_seconds, heartbeat_seconds=a.heartbeat_seconds,
            stop_file=a.stop_file, resume=a.resume, host_role=a.host_role, slice_=a.slice)
    print(json.dumps({"counts": r["counts"], "rows_total": r["rows_total"], "final": r["final"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
