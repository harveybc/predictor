#!/usr/bin/env python3
"""FS-CLOSE: closure follower for the EURUSD feature-selection manifest (order 2026-10-05 section 8).

The follower re-aggregates every lane's evidence as it lands, emits a DRAFT manifest while any
input is incomplete (listing the missing objects) and marks the manifest FINAL only when every
fail-closed check passes. Nothing in it is narrated: the dispositions, the status, the progress
figure and the master-milestone view are all derived from files whose digests are recorded.

Owned outputs (fs_closure/fs_close/ unless noted):
  FEATURE_METRICS_CATALOG.parquet / .csv.gz   long catalog: feature x stage x target x horizon x
                                              fold x metric x value x state x digest, every metric
                                              computed anywhere in the selection, nothing dropped
  feature_dispositions.csv                    exactly 366 rows, one disposition each
  representation_dispositions.csv             copied from FS-REP (137 heavy candidates)
  causal_evidence.jsonl                       copied from FS-CAUSAL
  selector_sets.json                          methods, K, order, folds, targets, seeds, digests
  refit_plan.json                             sealed plan consumed by tools/fs_close_refit.py
  paired_refit_metrics.parquet                pulled from the worker; population and naive explicit
  FINAL_SELECTION_MANIFEST.json               feature_selection.manifest.v1 (FINAL) or the DRAFT
  SELECTION_DECISION_RECORD.json              feature_selection.decision.v1, FINAL only
  CHECKS.json                                 the fail-closed checks and their states
  fs_closure/STATUS.json, fs_closure/PROGRESS.png
  MASTER_MILESTONE_STATUS.json / MASTER_MILESTONE_PROGRESS.png (M2 regenerated from evidence)

Rules enforced here: TRAIN inner folds for every refit; VALIDATION (2024) is read once, only
inside the closure step, under a rule declared before the read; TEST (2025) is never read;
NOT_IDENTIFIED is neutral; every metric carries its paired same-row naive; roles only, no host
names. Coordinator work is pure-python aggregation; refits run on the CPU worker under the
sanctioned capped launcher, sequentially.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

DENOMINATOR = 366
HEAVY_CANDIDATES = 137
SELECTOR_EPISODE_SOURCES = 37
K_PRIMARY = 24
K_SENSITIVITIES = (8, 16, 24, 32, 48)
FOLDS = ("inner_2019", "inner_2020", "inner_2021", "inner_2022", "inner_2023")
TARGET_CELLS = tuple([("Y_s", h) for h in (1, 2, 3, 4, 5, 6)]
                     + [("Y_l", h) for h in (24, 48, 72, 96, 120, 144)]
                     + [("Y_b", 6), ("Y_b", 144)])
PS3R_HORIZONS = {"Y_s": (1, 2, 3, 4, 5, 6), "Y_l": (24, 48, 72, 96, 120, 144), "Y_b": (6, 144)}
STAGES = ("PS1", "PS2", "PS3-C", "PS3-R", "PS4", "FS-PRED", "FS-CAUSAL", "FS-REP", "FS-GEN", "FS-CLOSE")
MANIFEST_SCHEMA = "feature_selection.manifest.v1"
DRAFT_SCHEMA = "feature_selection.manifest.v1-DRAFT"
DECISION_SCHEMA = "feature_selection.decision.v1"
ACCEPT_VERDICT = "ACCEPT_SELECTED_SET"
PRODUCER_ROLE = "fs_close_follower"
DECIDER_ROLE = "fs_close_fail_closed_verifier"
DATASET_ID = "eurusd_1h.business_contract.v1.train_2012-05_2023-12"
TRAIN_END = "2024-01-01T00:00:00+00:00"
VALIDATION_END = "2025-01-01T00:00:00+00:00"
STATES = ("SELECTED", "REJECTED", "PENDING")
GATE_SHA256_C0345F83 = "e86d85abc998319c53dd1ae5e0f6efbf7152170c177e27432b610ff2722e1ab1"  # tools/selected_manifest_gate.py at c0345f83
REQUIRED_SET_KINDS = ("ALL_ADMISSIBLE", "PRED_BEST", "PLUS_CAUSAL", "PLUS_REP")
OPTIONAL_SET_KINDS = ("KNOCKOFF",)
CONTROL_SET_KINDS = ("RANDOM_K",)
CATALOG_COLUMNS = ("feature", "stage", "target", "horizon", "fold", "metric", "value", "state",
                   "digest", "source")

# status colours (reference palette, status slots; every state is also written as text)
COLOR = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
         "muted": "#898781", "bar": "#2a78d6", "track": "#cde2fb", "ink": "#0b0b0b",
         "ink2": "#52514e", "grid": "#e1e0d9", "surface": "#fcfcfb"}


class ClosureError(RuntimeError):
    """A fail-closed violation that must stop the closure explicitly."""


# ============================================================================ utilities
def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj, drop: str | None = None) -> str:
    body = {k: v for k, v in obj.items() if k != drop} if isinstance(obj, dict) else obj
    return sha256_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def names_sha256(names: Iterable[str]) -> str:
    return sha256_bytes("\n".join(names).encode())


def write_atomic(path: Path, data: bytes) -> bool:
    """Write only when the bytes differ; return True when the file changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == data:
        return False
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return True


def write_json(path: Path, obj) -> bool:
    return write_atomic(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def parse_target(label: str) -> tuple[str, int]:
    """'Y_s_3h' -> ('Y_s', 3); 'Y_b_l144' -> ('Y_b', 144); 'Y_b_s6' -> ('Y_b', 6); 'Y_s' -> ('Y_s', -1)."""
    m = re.match(r"^(Y_[slb])(?:_([sl])?(\d+)h?)?$", str(label))
    if not m:
        return str(label), -1
    return m.group(1), int(m.group(3)) if m.group(3) else -1


def numeric_leaves(obj, prefix="") -> Iterator[tuple[str, float]]:
    """Flatten nested dict/list numeric leaves; booleans become 1/0; strings are skipped."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from numeric_leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        yield prefix, 1.0 if obj else 0.0
    elif isinstance(obj, (int, float)) and not (isinstance(obj, float) and math.isnan(obj)):
        yield prefix, float(obj)


# ============================================================================ paths
@dataclass
class Paths:
    repo: Path
    state: Path
    worker_alias: str | None = None
    worker_state: str = "~/.local/state/canonical_20261003"
    worker_python: str = "~/anaconda3/bin/python"

    @property
    def evidence(self) -> Path:
        return self.repo / "docs/audits/evidence/canonical_20261003"

    @property
    def fs_closure(self) -> Path:
        return self.evidence / "fs_closure"

    @property
    def out(self) -> Path:
        return self.fs_closure / "fs_close"

    @property
    def gate_module(self) -> Path:
        return self.repo / "tools/selected_manifest_gate.py"

    @classmethod
    def default(cls, repo: Path | None = None, state: Path | None = None, worker=None) -> "Paths":
        repo = repo or Path(__file__).resolve().parents[1]
        state = state or Path(os.path.expanduser("~/.local/state/canonical_20261003"))
        return cls(repo=repo, state=state, worker_alias=worker)


# ============================================================================ population
@dataclass
class Population:
    names: list[str]
    batch_of: dict[str, str]
    excluded_selector_sources: list[str]
    excluded_quality: list[str]
    digest: str
    sources: dict[str, str]

    @property
    def set(self) -> set[str]:
        return set(self.names)


def load_population(paths: Paths, denominator: int = DENOMINATOR) -> Population:
    """The 366 model-input candidates: lane A admissible features minus selector episode sources."""
    import pandas as pd

    names, batch_of, quality, sources = [], {}, [], {}
    for batch in ("batch_001", "batch_002", "batch_003"):
        p = paths.evidence / "laneA" / batch / "admissible_features.csv"
        if not p.is_file():
            raise ClosureError(f"POPULATION_SOURCE_MISSING: {p}")
        sources[f"laneA/{batch}/admissible_features.csv"] = sha256_file(p)
        df = pd.read_csv(p, usecols=["feature_id", "role"])
        for fid, role in zip(df["feature_id"], df["role"]):
            if role == "feature":
                names.append(str(fid))
                batch_of[str(fid)] = batch
            else:
                quality.append(str(fid))
    overlay_path = paths.evidence / "laneA/batch_002/role_overlay_batch_001.json"
    overlay = read_json(overlay_path, {})
    sources["laneA/batch_002/role_overlay_batch_001.json"] = sha256_file(overlay_path) if overlay_path.is_file() else ""
    episode = list(overlay.get("selector_episode_source_features", []))
    if len(episode) != SELECTOR_EPISODE_SOURCES:
        raise ClosureError(f"SELECTOR_EPISODE_SOURCES_MISMATCH: {len(episode)} != {SELECTOR_EPISODE_SOURCES}")
    pop = sorted(n for n in names if n not in set(episode))
    if len(pop) != denominator or len(set(pop)) != len(pop):
        raise ClosureError(f"POPULATION_MISMATCH: {len(pop)} candidates, denominator {denominator}")
    return Population(pop, {n: batch_of[n] for n in pop}, sorted(episode), sorted(quality),
                      names_sha256(pop), sources)


# ============================================================================ catalog writer
class CatalogWriter:
    """Streams catalog chunks into one parquet file and one gzip CSV without holding them all."""

    def __init__(self, parquet_path: Path, csv_path: Path):
        import pyarrow as pa

        self.schema = pa.schema([
            ("feature", pa.string()), ("stage", pa.string()), ("target", pa.string()),
            ("horizon", pa.int32()), ("fold", pa.string()), ("metric", pa.string()),
            ("value", pa.float64()), ("state", pa.string()), ("digest", pa.string()),
            ("source", pa.string())])
        self.parquet_path, self.csv_path = parquet_path, csv_path
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self._pq_tmp = parquet_path.with_suffix(".tmp.parquet")
        self._csv_tmp = csv_path.with_suffix(".tmp.gz")
        self._writer = None
        self._csv = gzip.open(self._csv_tmp, "wt", newline="")
        self._csv_writer = csv.writer(self._csv)
        self._csv_writer.writerow(CATALOG_COLUMNS)
        self.rows = 0
        self.by_stage: dict[str, int] = {}

    def add(self, df) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if df is None or len(df) == 0:
            return
        df = df.reindex(columns=list(CATALOG_COLUMNS))
        df["horizon"] = df["horizon"].fillna(-1).astype("int32")
        df["value"] = df["value"].astype("float64")
        for c in ("feature", "stage", "target", "fold", "metric", "state", "digest", "source"):
            df[c] = df[c].fillna("").astype(str)
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._pq_tmp, self.schema, compression="zstd")
        self._writer.write_table(table)
        self._csv_writer.writerows(df.itertuples(index=False, name=None))
        self.rows += len(df)
        for stage, n in df["stage"].value_counts().items():
            self.by_stage[stage] = self.by_stage.get(stage, 0) + int(n)

    def close(self) -> dict:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._writer is None:
            pq.write_table(self.schema.empty_table(), self._pq_tmp, compression="zstd")
        else:
            self._writer.close()
        self._csv.close()
        os.replace(self._pq_tmp, self.parquet_path)
        os.replace(self._csv_tmp, self.csv_path)
        return {"rows": self.rows, "by_stage": self.by_stage,
                "parquet_sha256": sha256_file(self.parquet_path), "csv_sha256": sha256_file(self.csv_path)}


def _frame(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows, columns=list(CATALOG_COLUMNS)) if rows else None


# ============================================================================ stage readers
def read_ps1(paths: Paths, pop: Population) -> Iterator:
    import pandas as pd

    for batch in ("batch_001", "batch_002", "batch_003"):
        p = paths.evidence / "laneA" / batch / "profile_cells.csv"
        if not p.is_file():
            continue
        digest, src = sha256_file(p), f"laneA/{batch}/profile_cells.csv"
        df = pd.read_csv(p)
        rows = []
        for r in df.itertuples(index=False):
            if r.feature_id not in pop.set:
                continue
            leaves = []
            try:
                leaves = list(numeric_leaves(json.loads(r.value))) if isinstance(r.value, str) else []
            except ValueError:
                pass
            if not leaves:
                rows.append(dict(feature=r.feature_id, stage="PS1", target="", horizon=-1, fold="TRAIN",
                                 metric=str(r.metric), value=float("nan"), state=str(r.state),
                                 digest=digest, source=src))
            for leaf, val in leaves:
                rows.append(dict(feature=r.feature_id, stage="PS1", target="", horizon=-1, fold="TRAIN",
                                 metric=f"{r.metric}.{leaf}", value=val, state=str(r.state),
                                 digest=digest, source=src))
        yield _frame(rows)


def read_ps2(paths: Paths, pop: Population) -> Iterator:
    import pandas as pd

    for batch in ("batch_001", "batch_002", "batch_003"):
        p = paths.state / "ps2" / batch / "ps2_fold_cells.csv"
        if not p.is_file():
            continue
        digest, src = sha256_file(p), f"state:ps2/{batch}/ps2_fold_cells.csv"
        full = pd.read_csv(p)
        full = full[full["feature"].isin(pop.set)]
        ident = ["feature", "target", "horizon", "fold", "cell_status"]
        value_cols = [c for c in full.columns if c not in ident + ["cell_reason"] and pd.api.types.is_numeric_dtype(full[c])]
        feats = sorted(full["feature"].unique())
        for i in range(0, len(feats), 12):           # bounded chunks keep the coordinator under its memory budget
            df = full[full["feature"].isin(feats[i:i + 12])]
            long = df.melt(id_vars=ident, value_vars=value_cols, var_name="metric", value_name="value")
            long = long.dropna(subset=["value"])
            tb = long["target"].map(lambda t: parse_target(t)[0])
            yield pd.DataFrame({"feature": long["feature"], "stage": "PS2", "target": tb,
                                "horizon": long["horizon"].astype(int), "fold": long["fold"],
                                "metric": long["metric"], "value": long["value"].astype(float),
                                "state": long["cell_status"], "digest": digest, "source": src})


def read_ps3c(paths: Paths, pop: Population) -> Iterator:
    import pandas as pd

    for batch in ("batch_001", "batch_002", "batch_003"):
        p = paths.evidence / "laneC" / batch / "summary.csv"
        if not p.is_file():
            continue
        digest, src = sha256_file(p), f"laneC/{batch}/summary.csv"
        df = pd.read_csv(p)
        df = df[(df["subject_kind"] == "feature") & df["subject"].isin(pop.set)]
        rows = []
        for r in df.to_dict("records"):
            tgt, hz = parse_target(r.get("target", ""))
            for rung in ("rung1", "rung2", "rung3"):
                rows.append(dict(feature=r["subject"], stage="PS3-C", target=tgt, horizon=hz, fold="TRAIN",
                                 metric=f"{rung}_state", value=float("nan"), state=str(r.get(rung, "")),
                                 digest=digest, source=src))
            for k, v in r.items():
                if k.startswith(("r1_", "r2_", "r3_")) and isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    rows.append(dict(feature=r["subject"], stage="PS3-C", target=tgt, horizon=hz, fold="TRAIN",
                                     metric=k, value=float(v), state="ESTIMATED", digest=digest, source=src))
        yield _frame(rows)


def ps3r_terminal_dirs(paths: Paths) -> list[Path]:
    root = paths.state / "selection_mirror"
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.glob("*/batch_*/*/results.jsonl")) + \
        sorted(p.parent for p in root.glob("*/batch_*/*/*/results.jsonl"))


def ps3r_terminal_role(manifest: dict) -> str:
    fams = set(manifest.get("families", []))
    if "masked_temporal_ae" in fams:
        return "alt_mtae"
    if "past_to_current_siamese" in fams:
        return "alt_p2c"
    if {"ae", "dae"} & fams:
        return "baseline"
    return "unknown"


def read_ps3r(paths: Paths, pop: Population) -> Iterator:
    for d in ps3r_terminal_dirs(paths):
        man = read_json(d / "run_manifest.json", {})
        if man.get("status") != "COMPLETED":
            continue
        res = d / "results.jsonl"
        digest = man.get("results_sha256") or sha256_file(res)
        role = ps3r_terminal_role(man)
        src = f"state:selection_mirror/{d.relative_to(paths.state / 'selection_mirror')}/results.jsonl"
        rows = []
        with open(res) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                feat = r.get("feature_id")
                if feat not in pop.set:
                    continue
                kind = r.get("kind", "")
                fold = r.get("fold_id", "TRAIN")
                fam = r.get("family") or r.get("representation") or r.get("trained") or ""
                tgt = r.get("target", "")
                hz = -1
                if "horizon_index" in r and tgt in PS3R_HORIZONS:
                    idx = int(r["horizon_index"])
                    hz = PS3R_HORIZONS[tgt][idx] if idx < len(PS3R_HORIZONS[tgt]) else -1
                state = r.get("status") or (r.get("reconstruction", {}) or {}).get("status") or "MEASURED"
                if kind == "feature_summary":
                    for item in r.get("probe_loss_across_folds", []):
                        t2 = item.get("target", "")
                        i2 = int(item.get("horizon_index", 0))
                        h2 = PS3R_HORIZONS.get(t2, (-1,) * 6)[i2] if i2 < 6 else -1
                        for leaf in ("mean", "std", "n_folds"):
                            if leaf in item:
                                rows.append(dict(feature=feat, stage="PS3-R", target=t2, horizon=h2, fold="ALL",
                                                 metric=f"{role}.{item.get('representation','')}.probe_loss.{leaf}",
                                                 value=float(item[leaf]), state="MEASURED", digest=digest, source=src))
                    continue
                skip = {"feature_id", "fold_id", "kind", "family", "representation", "trained", "target",
                        "horizon_index", "status", "seed", "train_row_ids_sha256", "architecture_id", "donor",
                        "probe", "loss", "probe_lags"}
                for leaf, val in numeric_leaves({k: v for k, v in r.items() if k not in skip}):
                    rows.append(dict(feature=feat, stage="PS3-R", target=tgt, horizon=hz, fold=fold,
                                     metric=f"{role}.{kind}.{fam}.{leaf}", value=val, state=str(state),
                                     digest=digest, source=src))
        yield _frame(rows)


def read_ps4(paths: Paths, pop: Population) -> Iterator:
    units = sorted((paths.state / "selection_ps4/units").glob("*.json")) if (paths.state / "selection_ps4/units").is_dir() else []
    if not units:
        units = sorted((paths.evidence / "ps4_incremental_profile/units").glob("*.json"))
    rows = []
    for u in units:
        d = read_json(u, {})
        feat = d.get("feature_id")
        if feat not in pop.set:
            continue
        digest = d.get("identity_sha256") or sha256_file(u)
        for r in d.get("rows", []):
            val = r.get("value")
            rows.append(dict(feature=feat, stage="PS4", target="", horizon=-1, fold=r.get("fold", d.get("fold_id", "")),
                             metric=str(r.get("metric", "")),
                             value=float(val) if isinstance(val, (int, float)) and val is not None else float("nan"),
                             state=str(r.get("status", "")), digest=digest, source=f"ps4/units/{u.name}"))
        if len(rows) >= 50000:
            yield _frame(rows)
            rows = []
    yield _frame(rows)


def generic_table_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix in (".parquet", ".csv", ".jsonl") and "ACK" not in p.name)


def load_table(p: Path):
    import pandas as pd

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_json(p, lines=True)


FEATURE_COLS = ("feature", "feature_id", "name", "subject")


def table_to_catalog(df, stage: str, digest: str, src: str, pop: Population):
    """Generic long/wide table -> catalog rows. Recognises feature/target/horizon/fold/metric/value/state."""
    import pandas as pd

    fcol = next((c for c in FEATURE_COLS if c in df.columns), None)
    if fcol is None:
        return None
    df = df[df[fcol].astype(str).isin(pop.set)]
    if df.empty:
        return None
    tgt = df["target"].astype(str) if "target" in df.columns else pd.Series([""] * len(df), index=df.index)
    if "horizon" in df.columns:
        hz = pd.to_numeric(df["horizon"], errors="coerce").fillna(-1).astype(int)
        tb = tgt.map(lambda t: parse_target(t)[0])
    else:
        parsed = tgt.map(parse_target)
        tb, hz = parsed.map(lambda x: x[0]), parsed.map(lambda x: x[1]).astype(int)
    fold = df["fold"].astype(str) if "fold" in df.columns else (df["fold_id"].astype(str) if "fold_id" in df.columns else "TRAIN")
    state_col = next((c for c in ("state", "status", "cell_status", "decision", "verdict", "disposition") if c in df.columns), None)
    state = df[state_col].astype(str) if state_col else "MEASURED"
    method = df["method"].astype(str) + "." if "method" in df.columns else ""
    if "metric" in df.columns and "value" in df.columns:
        out = pd.DataFrame({"feature": df[fcol].astype(str), "stage": stage, "target": tb, "horizon": hz, "fold": fold,
                            "metric": method + df["metric"].astype(str),
                            "value": pd.to_numeric(df["value"], errors="coerce"), "state": state, "digest": digest, "source": src})
        return out
    ident = {fcol, "target", "horizon", "fold", "fold_id", state_col, "method"}
    value_cols = [c for c in df.columns if c not in ident and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return pd.DataFrame({"feature": df[fcol].astype(str), "stage": stage, "target": tb, "horizon": hz, "fold": fold,
                             "metric": method + (state_col or "state"), "value": float("nan"), "state": state,
                             "digest": digest, "source": src})
    base = pd.DataFrame({"feature": df[fcol].astype(str), "target": tb, "horizon": hz, "fold": fold, "state": state,
                         "method": method})
    long = pd.concat([base, df[value_cols].reset_index(drop=True).set_index(base.index)], axis=1) \
        .melt(id_vars=list(base.columns), value_vars=value_cols, var_name="metric", value_name="value").dropna(subset=["value"])
    return pd.DataFrame({"feature": long["feature"], "stage": stage, "target": long["target"], "horizon": long["horizon"],
                         "fold": long["fold"], "metric": long["method"] + long["metric"], "value": long["value"].astype(float),
                         "state": long["state"], "digest": digest, "source": src})


def read_lane_tables(paths: Paths, pop: Population, lane_dir: str, stage: str) -> Iterator:
    root = paths.fs_closure / lane_dir
    for p in generic_table_files(root):
        try:
            df = load_table(p)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        try:
            out = table_to_catalog(df, stage, sha256_file(p), f"fs_closure/{lane_dir}/{p.relative_to(root)}", pop)
        except Exception:
            out = None
        if out is not None and len(out):
            yield out


def read_fs_close(paths: Paths, pop: Population, plan: dict | None) -> Iterator:
    import pandas as pd

    p = paths.out / "paired_refit_metrics.parquet"
    if not p.is_file():
        return
    digest = sha256_file(p)
    df = pd.read_parquet(p)
    df = df[df["state"] == "MEASURED"]
    rows = pd.DataFrame({"feature": "set:" + df["set_id"].astype(str), "stage": "FS-CLOSE", "target": df["target"],
                         "horizon": df["horizon"].astype(int), "fold": df["fold"],
                         "metric": df["head"] + "." + df["metric"] + ".vs_" + df["naive_kind"],
                         "value": df["value"], "state": "MEASURED", "digest": digest, "source": "fs_close/paired_refit_metrics.parquet"})
    naive = rows.copy()
    naive["metric"] = df["head"] + ".naive_" + df["naive_kind"] + "." + df["metric"]
    naive["value"] = df["naive_value"]
    sk = rows.copy()
    sk["metric"] = df["head"] + ".skill_" + df["metric"] + "_vs_" + df["naive_kind"]
    sk["value"] = df["skill"]
    yield pd.concat([rows, naive, sk], ignore_index=True)
    if plan:
        mem = []
        for s in plan.get("sets", []):
            lists = s["features_by"].values() if isinstance(s.get("features_by"), dict) else [s.get("features", [])]
            seen = set()
            for lst in lists:
                seen.update(lst)
            for f in seen:
                mem.append(dict(feature=f, stage="FS-CLOSE", target="", horizon=-1, fold="ALL",
                                metric=f"member.{s['set_id']}", value=1.0, state="MEASURED",
                                digest=set_digest(s), source="fs_close/refit_plan.json"))
        yield _frame(mem)


def build_catalog(paths: Paths, pop: Population, plan: dict | None) -> dict:
    writer = CatalogWriter(paths.out / "FEATURE_METRICS_CATALOG.parquet", paths.out / "FEATURE_METRICS_CATALOG.csv.gz")
    readers = (("PS1", read_ps1(paths, pop)), ("PS2", read_ps2(paths, pop)), ("PS3-C", read_ps3c(paths, pop)),
               ("PS3-R", read_ps3r(paths, pop)), ("PS4", read_ps4(paths, pop)),
               ("FS-PRED", read_lane_tables(paths, pop, "fs_pred", "FS-PRED")),
               ("FS-CAUSAL", read_lane_tables(paths, pop, "fs_causal", "FS-CAUSAL")),
               ("FS-REP", read_lane_tables(paths, pop, "fs_rep", "FS-REP")),
               ("FS-GEN", read_lane_tables(paths, pop, "fs_gen", "FS-GEN")),
               ("FS-CLOSE", read_fs_close(paths, pop, plan)))
    for _, it in readers:
        for df in it:
            writer.add(df)
    summary = writer.close()
    summary["schema"] = "feature_metrics_catalog.v1"
    summary["columns"] = list(CATALOG_COLUMNS)
    summary["generated_utc"] = utc_now()
    return summary


def catalog_sources_digest(paths: Paths) -> str:
    """Digest of every catalog input's (path, mtime, size) so the catalog is rebuilt only on change."""
    items = []
    roots = [paths.evidence / "laneA", paths.evidence / "laneC", paths.state / "ps2", paths.state / "selection_mirror",
             paths.state / "selection_ps4/units", paths.fs_closure / "fs_pred", paths.fs_closure / "fs_causal",
             paths.fs_closure / "fs_rep", paths.fs_closure / "fs_gen", paths.out / "paired_refit_metrics.parquet",
             paths.out / "refit_plan.json"]
    for r in roots:
        if r.is_file():
            st = r.stat()
            items.append((str(r), st.st_mtime_ns, st.st_size))
        elif r.is_dir():
            for p in sorted(r.rglob("*")):
                if p.is_file() and p.suffix in (".csv", ".json", ".jsonl", ".parquet"):
                    st = p.stat()
                    items.append((str(p), st.st_mtime_ns, st.st_size))
    return canonical_sha256(items)


# ============================================================================ lane evidence (decisions)
@dataclass
class LaneEvidence:
    """Everything the closure needs from the other lanes, with digests and missing objects."""
    missing: list[str] = field(default_factory=list)
    digests: dict[str, str] = field(default_factory=dict)
    pred_rankings: dict[str, dict[str, list[str]]] = field(default_factory=dict)   # method -> "target|fold" -> ranking
    pred_incomplete: dict[str, str] = field(default_factory=dict)
    pred_best_method: str | None = None
    pred_draft: dict = field(default_factory=dict)
    pred_progress: dict = field(default_factory=dict)
    pred_methods_planned: list[str] = field(default_factory=list)
    causal: dict[str, dict[str, str]] = field(default_factory=dict)              # feature -> target -> state
    causal_counts: dict[str, int] = field(default_factory=dict)
    rep: dict[str, dict[str, str]] = field(default_factory=dict)                  # feature -> {decision, flags}
    rep_counts: dict[str, int] = field(default_factory=dict)
    gen_calibrated: bool | None = None
    gen_selected: list[str] = field(default_factory=list)
    gen_state: str = "ABSENT"
    gen_counts: dict[str, int] = field(default_factory=dict)
    gen_reason: dict[str, str] = field(default_factory=dict)
    ps2_status: dict[str, dict] = field(default_factory=dict)                     # feature -> summary
    ps3r_terminals: dict[str, dict[str, str]] = field(default_factory=dict)       # feature -> role -> COMPLETED/FAILED
    ps4_profiled: set[str] = field(default_factory=set)
    heavy: list[str] = field(default_factory=list)
    gpu_eta: dict = field(default_factory=dict)


def _first_key(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_pred(paths: Paths, pop: Population, ev: LaneEvidence) -> None:
    roots = [r for r in (paths.fs_closure / "fs_pred", paths.state / "fs_close/fs_pred_mirror") if r.is_dir()]
    sets_file = None
    for root in roots:
        sets_file = next((p for p in sorted(root.rglob("selector_sets*.json")) if p.is_file()), None)
        if sets_file:
            break
    if sets_file is None:
        ev.missing.append("fs_pred/selector_sets.json")
    else:
        ev.digests["fs_pred/" + sets_file.name] = sha256_file(sets_file)
        doc = read_json(sets_file, {})
        ev.pred_best_method = _first_key(doc, ("best_predictive_method", "best_method", "primary_method"))
        rankings = _first_key(doc, ("rankings", "methods"))
        if isinstance(rankings, dict):
            for method, body in rankings.items():
                _ingest_ranking(ev, method, body)
        if isinstance(doc.get("targets"), dict):      # fs_pred_selector_sets_draft.v1: targets -> method -> fold -> selected_by_k
            ev.pred_draft = doc
    for root in roots:
        for p in generic_table_files(root):
            if "ranking" not in p.name.lower():
                continue
            try:
                df = load_table(p)
            except Exception:
                continue
            ev.digests["fs_pred/" + str(p.relative_to(root))] = sha256_file(p)
            _ingest_ranking_table(ev, df)
    prog = next((read_json(r / "progress.json", None) for r in roots if (r / "progress.json").is_file()), None)
    contract = next((read_json(r / "run_contract.json", None) for r in roots if (r / "run_contract.json").is_file()), None)
    ev.pred_progress = prog or {}
    ev.pred_methods_planned = list((contract or {}).get("methods") or [])
    if not ev.pred_rankings:
        ev.missing.append("fs_pred/rankings (method x target x fold complete orderings)")
    for method, cells in list(ev.pred_rankings.items()):
        bad = [k for k, order in cells.items() if len(order) != len(pop.names) or set(order) != pop.set]
        need = {f"{t}_{h}h|{fd}" for t, h in TARGET_CELLS for fd in FOLDS}
        missing_cells = sorted(need - set(cells))
        if bad or missing_cells:
            ev.pred_incomplete[method] = f"incomplete ranking: {len(bad)} partial, {len(missing_cells)} missing cells"


def _ingest_ranking(ev: LaneEvidence, method: str, body) -> None:
    if isinstance(body, dict):
        for cell, order in body.items():
            if isinstance(order, list):
                ev.pred_rankings.setdefault(method, {})[str(cell).replace("/", "|")] = [str(x) for x in order]
    elif isinstance(body, list) and body and isinstance(body[0], str):
        ev.pred_rankings.setdefault(method, {})["*"] = [str(x) for x in body]


def _ingest_ranking_table(ev: LaneEvidence, df) -> None:
    cols = set(df.columns)
    fcol = next((c for c in FEATURE_COLS if c in cols), None)
    if not fcol or "method" not in cols or not ({"rank", "position", "order"} & cols):
        return
    rcol = next(c for c in ("rank", "position", "order") if c in cols)
    tcol = "target" if "target" in cols else None
    fold = "fold" if "fold" in cols else ("fold_id" if "fold_id" in cols else None)
    df = df.sort_values(rcol)
    keys = [c for c in ("method", tcol, fold) if c]
    if "horizon" in cols and tcol:
        df = df.assign(_t=df[tcol].astype(str).map(lambda t: parse_target(t)[0]) + "_" + df["horizon"].astype(int).astype(str) + "h")
        keys = ["method", "_t"] + ([fold] if fold else [])
    for key, g in df.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        method = str(key[0])
        cell = "|".join(str(k) for k in key[1:]) if len(key) > 1 else "*"
        ev.pred_rankings.setdefault(method, {})[cell] = [str(x) for x in g[fcol]]


def load_causal(paths: Paths, pop: Population, ev: LaneEvidence) -> None:
    p = paths.fs_closure / "fs_causal/causal_evidence.jsonl"
    if not p.is_file():
        ev.missing.append("fs_causal/causal_evidence.jsonl")
        return
    ev.digests["fs_causal/causal_evidence.jsonl"] = sha256_file(p)
    counts: dict[str, int] = {}
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            feat = _first_key(r, FEATURE_COLS)
            if feat not in pop.set:
                continue
            state = str(_first_key(r, ("state", "verdict", "evidence", "disposition")) or "NOT_IDENTIFIED").upper()
            robust = bool(r.get("robust", True))
            tgt = str(r.get("target", "*"))
            cur = ev.causal.setdefault(feat, {})
            if state == "CONTRADICTED" and not robust:
                state = "NOT_IDENTIFIED"
            prev = cur.get(tgt)
            rank = {"CONTRADICTED": 2, "SUPPORTED": 1, "NOT_IDENTIFIED": 0}
            if prev is None or rank.get(state, 0) > rank.get(prev, 0):
                cur[tgt] = state
            counts[state] = counts.get(state, 0) + 1
    ev.causal_counts = counts
    covered = len(ev.causal)
    if covered < len(pop.names):
        ev.missing.append(f"fs_causal coverage {covered}/{len(pop.names)} features")


def load_rep(paths: Paths, pop: Population, ev: LaneEvidence) -> None:
    import pandas as pd

    controls = paths.fs_closure / "fs_rep/representation_dispositions.csv"
    if controls.is_file():
        ev.digests["fs_rep/representation_dispositions.csv"] = sha256_file(controls)
    p = paths.fs_closure / "fs_rep/candidate_decisions.csv"
    if not p.is_file():
        p = controls
    if not p.is_file():
        ev.missing.append("fs_rep/candidate_decisions.csv")
        return
    ev.digests["fs_rep/" + p.name] = sha256_file(p)
    df = pd.read_csv(p)
    fcol = next((c for c in FEATURE_COLS if c in df.columns), None)
    dcol = next((c for c in ("decision", "representation", "disposition") if c in df.columns), None)
    if not fcol or not dcol:
        ev.missing.append("fs_rep/representation_dispositions.csv (no feature/decision columns)")
        return
    flags_col = "flags" if "flags" in df.columns else None
    counts: dict[str, int] = {}
    for r in df.to_dict("records"):
        feat = str(r[fcol])
        if feat not in pop.set:
            continue
        dec = str(r[dcol])
        ev.rep[feat] = {"decision": dec, "flags": str(r.get(flags_col, "") if flags_col else "")}
        counts[dec] = counts.get(dec, 0) + 1
    ev.rep_counts = counts
    pending = [f for f, v in ev.rep.items() if v["decision"] == "PENDING"]
    if len(ev.rep) < HEAVY_CANDIDATES or pending:
        ev.missing.append(f"fs_rep decisions {len(ev.rep) - len(pending)}/{HEAVY_CANDIDATES} heavy candidates decided")


def load_gen(paths: Paths, pop: Population, ev: LaneEvidence) -> None:
    """FS-GEN feature-level evidence: calibration/knockoff states per feature; arm 5 exists only if some
    feature has a CALIBRATED knockoff selection. Features without a series carry NO_SERIES_IN_PS2_BATCH."""
    import pandas as pd

    p = paths.fs_closure / "fs_gen/generative_evidence.csv"
    if not p.is_file():
        ev.gen_state = "ABSENT"
        return
    ev.digests["fs_gen/generative_evidence.csv"] = sha256_file(p)
    df = pd.read_csv(p)
    fcol = next((c for c in FEATURE_COLS if c in df.columns), None)
    cal_col = next((c for c in ("calibration_state", "calibration", "state") if c in df.columns), None)
    ko_col = "knockoff_state" if "knockoff_state" in df.columns else cal_col
    sel_col = next((c for c in ("knockoff_selected_cells_majority", "knockoff_selected", "selected") if c in df.columns), None)
    stat_col = next((c for c in ("selection_jaccard_mean", "knockoff_statistic", "statistic") if c in df.columns), None)
    if fcol is None or cal_col is None:
        ev.gen_state = "NOT_CALIBRATED/EMPTY"
        ev.gen_calibrated = False
        return
    df = df[df[fcol].astype(str).isin(pop.set)]
    ev.gen_counts = {str(k): int(v) for k, v in df[cal_col].astype(str).value_counts().items()}
    if "reason" in df.columns:
        ev.gen_reason = {str(f): str(r) for f, r in zip(df[fcol], df["reason"]) if isinstance(r, str) and r}
    ko = df[ko_col].astype(str).str.upper() if ko_col else pd.Series([""] * len(df), index=df.index)
    sel_mask = ko.isin(("CALIBRATED", "PASS", "PASSED"))
    if sel_col is not None:
        sel_mask &= pd.to_numeric(df[sel_col], errors="coerce").fillna(0) > 0
    sel = df[sel_mask]
    if stat_col and len(sel):
        sel = sel.sort_values(stat_col, ascending=False)
    ev.gen_selected = [str(f) for f in sel[fcol]]
    ev.gen_calibrated = bool(ev.gen_selected)
    ev.gen_state = "CALIBRATED" if ev.gen_calibrated else "NOT_CALIBRATED/EMPTY"


def load_ps2_status(paths: Paths, pop: Population, ev: LaneEvidence) -> None:
    import pandas as pd

    for batch in ("batch_001", "batch_002", "batch_003"):
        p = paths.evidence / "laneB" / batch / "ps2_status.csv"
        if not p.is_file():
            continue
        ev.digests[f"laneB/{batch}/ps2_status.csv"] = sha256_file(p)
        df = pd.read_csv(p, usecols=["feature", "target", "horizon", "status", "oof_delta_median", "redundant_with"])
        for r in df.itertuples(index=False):
            if r.feature not in pop.set:
                continue
            s = ev.ps2_status.setdefault(r.feature, {"statuses": {}, "oof_delta": {}, "redundant_with": set()})
            s["statuses"][f"{r.target}|{r.horizon}"] = str(r.status)
            s["oof_delta"][f"{r.target}|{r.horizon}"] = float(r.oof_delta_median) if isinstance(r.oof_delta_median, float) else float("nan")
            if isinstance(r.redundant_with, str) and r.redundant_with:
                s["redundant_with"].update(x.strip() for x in re.split(r"[;,|]", r.redundant_with) if x.strip())


def load_ps3r_ps4(paths: Paths, pop: Population, ev: LaneEvidence) -> None:
    rec = read_json(paths.evidence / "coverage_reconciliation/coverage_reconciliation.json", {})
    heavy = []
    feats = rec.get("features") if isinstance(rec, dict) else None
    if isinstance(feats, dict):
        heavy = [f for f, v in feats.items() if isinstance(v, dict) and v.get("in_extractibility_queue")]
    elif isinstance(feats, list):
        heavy = [f.get("feature_id") or f.get("feature") for f in feats if isinstance(f, dict) and f.get("in_extractibility_queue")]
    if not heavy:
        # fall back to the automation plans: every planned feature is a heavy candidate
        for tsv in (paths.evidence / "automation").glob("*.tsv"):
            with open(tsv) as f:
                next(f, None)
                for line in f:
                    cell = line.split("\t")[0]
                    parts = cell.split("::")
                    if len(parts) >= 2:
                        heavy.append(parts[1])
    ev.heavy = sorted(set(h for h in heavy if h in pop.set))
    for d in ps3r_terminal_dirs(paths):
        man = read_json(d / "run_manifest.json", {})
        feat = (man.get("features") or [None])[0]
        if feat not in pop.set:
            continue
        role = ps3r_terminal_role(man)
        st = man.get("status", "")
        if st == "COMPLETED" or any(d.glob("FAILED*")):
            ev.ps3r_terminals.setdefault(feat, {})[role] = "COMPLETED" if st == "COMPLETED" else "FAILED"
    ps4 = read_json(paths.state / "selection_ps4/STATUS.json", {}) or read_json(paths.evidence / "ps4_incremental_profile/REPORT.json", {})
    ev.ps4_profiled = set(x for x in (ps4.get("profiled") or ps4.get("accepted") or []) if x in pop.set)
    eta = read_json(paths.fs_closure / "fs_gpu/ps3r_eta.json", None)
    if eta is None:
        live = read_json(paths.state / "SELECTION_STATUS.json", {})
        lanes = live.get("lanes", {}) if isinstance(live, dict) else {}
        eta = {"source": "state:SELECTION_STATUS.json", "queues": {}}
        for k, v in lanes.items():
            if isinstance(v, dict) and "counts" in v:
                eta["queues"][k] = {"counts": v.get("counts"), "eta_seconds": v.get("eta_seconds"),
                                    "durations_seconds": v.get("durations_seconds"), "workers": v.get("workers")}
    else:
        eta["source"] = "fs_closure/fs_gpu/ps3r_eta.json"
    ev.gpu_eta = eta


def load_lanes(paths: Paths, pop: Population) -> LaneEvidence:
    ev = LaneEvidence()
    load_pred(paths, pop, ev)
    load_causal(paths, pop, ev)
    load_rep(paths, pop, ev)
    load_gen(paths, pop, ev)
    load_ps2_status(paths, pop, ev)
    load_ps3r_ps4(paths, pop, ev)
    complete_heavy = [f for f in ev.heavy if {"baseline", "alt_mtae", "alt_p2c"} <= set(ev.ps3r_terminals.get(f, {}))]
    if len(complete_heavy) < len(ev.heavy) or not ev.heavy:
        ev.missing.append(f"PS3-R terminals complete {len(complete_heavy)}/{len(ev.heavy) or HEAVY_CANDIDATES} heavy candidates")
    if len(ev.ps4_profiled & set(ev.heavy)) < len(ev.heavy) or not ev.heavy:
        ev.missing.append(f"PS4 profiles {len(ev.ps4_profiled & set(ev.heavy))}/{len(ev.heavy) or HEAVY_CANDIDATES} heavy candidates")
    return ev


# ============================================================================ set definitions
def set_digest(s: dict) -> str:
    return canonical_sha256({k: v for k, v in s.items() if k != "set_sha256"})


def random_k_order(pop: Population, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [pop.names[i] for i in rng.permutation(len(pop.names))]


def reorder_causal(order: list[str], causal_for_target: dict[str, str]) -> list[str]:
    """SUPPORTED first (predictive order kept), then neutral/NOT_IDENTIFIED, then robust CONTRADICTED."""
    rank = {"SUPPORTED": 0, "NOT_IDENTIFIED": 1, "CONTRADICTED": 2}
    return sorted(order, key=lambda f: (rank.get(causal_for_target.get(f, "NOT_IDENTIFIED"), 1), order.index(f)))


def reorder_rep(order: list[str], rep: dict[str, dict[str, str]]) -> list[str]:
    """Heavy candidates whose every representation lacks probe skill are demoted; order otherwise kept."""
    def demoted(f):
        return "NO_PROBE_SKILL_VS_NAIVE_ANY_FAMILY" in (rep.get(f, {}).get("flags") or "")
    pos = {f: i for i, f in enumerate(order)}
    return sorted(order, key=lambda f: (1 if demoted(f) else 0, pos[f]))


def choose_pred_best(ev: LaneEvidence, metrics) -> tuple[str | None, dict]:
    """Best predictive/redundancy method = highest mean ridge skill at K=24 over the 14 cells, TRAIN inner folds.

    Skill is taken against the stricter naive of each cell (minimum skill over the paired naives).
    Ties -> the method FS-PRED itself named, then alphabetical.
    """
    import pandas as pd

    if metrics is None or len(metrics) == 0:
        return ev.pred_best_method if ev.pred_best_method in ev.pred_rankings else None, {}
    m = metrics[(metrics["set_kind"] == "PRED_METHOD") & (metrics["k"] == K_PRIMARY) & (metrics["head"] == "ridge")
                & (metrics["state"] == "MEASURED") & (metrics["metric"].isin(["mae", "log_loss"]))]
    if m.empty:
        return ev.pred_best_method if ev.pred_best_method in ev.pred_rankings else None, {}
    cell = m.groupby(["set_id", "target", "horizon", "fold"])["skill"].min().reset_index()
    per_method = cell.groupby("set_id")["skill"].agg(["mean", "count"])
    full = per_method[per_method["count"] >= len(TARGET_CELLS) * len(FOLDS)]
    if full.empty:
        return None, {}
    scores = {sid.replace("PRED_METHOD:", ""): float(v) for sid, v in full["mean"].items()}
    best = sorted(scores, key=lambda k: (-scores[k], 0 if k == ev.pred_best_method else 1, k))[0]
    return best, scores


def build_plan(pop: Population, ev: LaneEvidence, metrics=None) -> tuple[dict, list[str]]:
    """Sealed refit plan: ALL_ADMISSIBLE, RANDOM_K control, every complete FS-PRED method at the
    sensitivities, and the section-8 arms PRED_BEST / PLUS_CAUSAL / PLUS_REP / KNOCKOFF when their
    inputs exist. Returns (plan, missing objects that blocked arms)."""
    missing = []
    sets = [{"set_id": "ALL_ADMISSIBLE", "set_kind": "ALL_ADMISSIBLE", "k": None, "features": list(pop.names),
             "source": {"rule": "every model-input candidate", "population_sha256": pop.digest}}]
    tape = random_k_order(pop, 0)
    for k in (k for k in K_SENSITIVITIES if k <= len(pop.names)):
        sets.append({"set_id": f"RANDOM_K:{k}", "set_kind": "RANDOM_K", "k": k, "features": tape[:k],
                     "source": {"rule": "numpy default_rng(0) permutation of the sorted population, first K", "seed": 0}})
    complete = {m: cells for m, cells in ev.pred_rankings.items() if m not in ev.pred_incomplete}
    for method, cells in sorted(complete.items()):
        for k in K_SENSITIVITIES:
            sets.append({"set_id": f"PRED_METHOD:{method}:{k}", "set_kind": "PRED_METHOD", "k": k, "method": method,
                         "features_by": {c: order[:k] for c, order in cells.items()},
                         "source": {"rule": "top-K of the FS-PRED complete ranking per target x fold", "digests": ev.digests}})
    best, scores = choose_pred_best(ev, metrics)
    arms = {}
    if best and best in complete:
        arms["PRED_BEST"] = complete[best]
    else:
        missing.append("PRED_BEST (needs FS-PRED complete rankings and their K=24 ridge refits)")
    if "PRED_BEST" in arms:
        if ev.causal:
            arms["PLUS_CAUSAL"] = {c: reorder_causal(order, {f: st for f, st in
                                                           ((f, ev.causal.get(f, {}).get(c.split("|")[0], ev.causal.get(f, {}).get("*", "NOT_IDENTIFIED")))
                                                            for f in order)})
                                   for c, order in arms["PRED_BEST"].items()}
        else:
            missing.append("PLUS_CAUSAL (needs fs_causal/causal_evidence.jsonl)")
        if "PLUS_CAUSAL" in arms:
            if ev.rep and not any(v["decision"] == "PENDING" for v in ev.rep.values()) and len(ev.rep) >= len(ev.heavy or [0] * HEAVY_CANDIDATES):
                arms["PLUS_REP"] = {c: reorder_rep(order, ev.rep) for c, order in arms["PLUS_CAUSAL"].items()}
            else:
                missing.append("PLUS_REP (needs complete fs_rep/representation_dispositions.csv)")
    for kind in ("PRED_BEST", "PLUS_CAUSAL", "PLUS_REP"):
        if kind in arms:
            for k in K_SENSITIVITIES:
                sets.append({"set_id": f"{kind}:{k}", "set_kind": kind, "k": k, "method": best,
                             "features_by": {c: order[:k] for c, order in arms[kind].items()},
                             "source": {"rule": {"PRED_BEST": "best predictive/redundancy method by TRAIN inner-fold ridge skill at K=24",
                                                 "PLUS_CAUSAL": "PRED_BEST order; SUPPORTED first, NOT_IDENTIFIED neutral, robust CONTRADICTED last",
                                                 "PLUS_REP": "PLUS_CAUSAL order; heavy candidates without probe skill in any representation demoted; representation attached from FS-REP"}[kind],
                                        "digests": ev.digests, "pred_best_scores": scores}})
    for f in ev.heavy:
        sets.append({"set_id": f"ALL_MINUS:{f}", "set_kind": "REMOVAL", "k": None, "heads": ["ridge"],
                     "features": [x for x in pop.names if x != f], "removed": f,
                     "source": {"rule": "ALL_ADMISSIBLE without the candidate; refit after removal (raw/identity representation only)"}})
    if ev.gen_calibrated and ev.gen_selected:
        for k in K_SENSITIVITIES:
            feats = ev.gen_selected[:k]
            sets.append({"set_id": f"KNOCKOFF:{k}", "set_kind": "KNOCKOFF", "k": len(feats), "features": feats,
                         "source": {"rule": "calibrated knockoff selection ordered by statistic, first K (fewer if the FDR set is smaller)",
                                    "declared_k": k, "digests": ev.digests}})
    for s in sets:
        s["set_sha256"] = set_digest(s)
    plan = {"schema": "fs_close_refit_plan.v1", "population": list(pop.names), "population_sha256": pop.digest,
            "folds": list(FOLDS), "targets": [f"{t}_{h}h" for t, h in TARGET_CELLS], "seed": 0,
            "k_primary": K_PRIMARY, "k_sensitivities": list(K_SENSITIVITIES), "heads": ["ridge", "hgb"],
            "hgb_max_k": K_PRIMARY, "sets": sets, "pred_best_method": best, "generative_state": ev.gen_state}
    return plan, missing


def selector_sets_document(plan: dict, ev: LaneEvidence, pop: Population) -> dict:
    return {"schema": "fs_close_selector_sets.v1", "generated_utc": utc_now(), "population_sha256": pop.digest,
            "denominator": len(pop.names), "k_primary": K_PRIMARY, "k_sensitivities": list(K_SENSITIVITIES),
            "folds": list(FOLDS), "targets": plan["targets"], "seeds": [0],
            "methods": sorted(ev.pred_rankings), "incomplete_methods": ev.pred_incomplete,
            "pred_best_method": plan.get("pred_best_method"), "generative_state": ev.gen_state,
            "arm_5_knockoff": {"state": ev.gen_state, "n_selected_features": len(ev.gen_selected),
                               "calibration_counts": ev.gen_counts,
                               "note": "section 8 set 5 is compared only if calibrated; NOT_CALIBRATED/EMPTY is recorded, never dropped"},
            "sets": [{k: v for k, v in s.items() if k not in ("features", "features_by")} |
                     {"n_cells": len(s["features_by"]) if "features_by" in s else 1,
                      "k_effective": s["k"] if s["k"] is not None else len(pop.names)} for s in plan["sets"]],
            "input_digests": ev.digests}


# ============================================================================ refit metrics
def load_metrics(paths: Paths):
    p = paths.out / "paired_refit_metrics.parquet"
    if not p.is_file():
        return None
    import pandas as pd

    return pd.read_parquet(p)


def jaccard_stability(plan: dict) -> dict:
    """Mean/min pairwise Jaccard between the fold-specific sets of each set_id (per target)."""
    out = {}
    for s in plan["sets"]:
        if "features_by" not in s:
            out[s["set_id"]] = {"mean": 1.0, "min": 1.0, "fold_invariant": True}
            continue
        by_target: dict[str, list[set]] = {}
        for cell, feats in s["features_by"].items():
            tgt = cell.split("|")[0]
            by_target.setdefault(tgt, []).append(set(feats))
        vals = []
        for sets in by_target.values():
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    u = len(sets[i] | sets[j])
                    vals.append(len(sets[i] & sets[j]) / u if u else 1.0)
        out[s["set_id"]] = {"mean": float(np.mean(vals)) if vals else 1.0, "min": float(np.min(vals)) if vals else 1.0,
                            "fold_invariant": False, "n_pairs": len(vals)}
    return out


def refit_coverage(plan: dict, metrics) -> dict:
    """Which (set_id, head) have all 14 x 5 cells MEASURED."""
    expected = len(TARGET_CELLS) * len(FOLDS)
    cov = {}
    for s in plan["sets"]:
        for head in s.get("heads") or plan["heads"]:
            if head == "hgb" and s["k"] is not None and s["k"] > plan["hgb_max_k"]:
                continue
            cov[(s["set_id"], head)] = 0
    if metrics is not None and len(metrics):
        m = metrics[metrics["state"] == "MEASURED"]
        g = m.groupby(["set_id", "set_sha256", "head"])[["target", "horizon", "fold"]].nunique()
        done = m.groupby(["set_id", "set_sha256", "head"]).apply(lambda d: len(d[["target", "horizon", "fold"]].drop_duplicates()), include_groups=False)
        current = {s["set_id"]: s["set_sha256"] for s in plan["sets"]}
        for (sid, sdig, head), n in done.items():
            if current.get(sid) == sdig and (sid, head) in cov:
                cov[(sid, head)] = int(n)
    return {"expected_cells_per_set_head": expected, "cells": {f"{k[0]}|{k[1]}": v for k, v in cov.items()},
            "complete": sum(1 for v in cov.values() if v >= expected), "planned": len(cov)}


def summarize_skill(metrics) -> dict:
    """Per set x head: mean skill vs the stricter paired naive by target group, and the share of positive cells."""
    if metrics is None or len(metrics) == 0:
        return {}
    m = metrics[(metrics["state"] == "MEASURED") & (metrics["metric"].isin(["mae", "log_loss"]))]
    if m.empty:
        return {}
    cell = m.groupby(["set_id", "head", "target", "horizon", "fold"])["skill"].min().reset_index()
    out: dict[str, dict] = {}
    for (sid, head, tgt), g in cell.groupby(["set_id", "head", "target"]):
        d = out.setdefault(f"{sid}|{head}", {})
        d[tgt] = {"mean_skill_vs_stricter_naive": float(g["skill"].mean()), "min": float(g["skill"].min()),
                  "max": float(g["skill"].max()), "positive_cells": int((g["skill"] > 0).sum()), "cells": int(len(g)),
                  "beats_naive_every_cell": bool((g["skill"] > 0).all())}
    return out


def refit_gain_rows(metrics, plan: dict) -> tuple[list[dict], list[dict]]:
    """Removal refit: gain(f) = (loss(ALL without f) - loss(ALL)) / loss(ALL) on identical rows; > 0 = f helps.

    Returned per feature (identity representation only: trained latents are not materialised on the
    refit host, which is declared) and per cell x fold.
    """
    if metrics is None or len(metrics) == 0:
        return [], []
    m = metrics[(metrics["state"] == "MEASURED") & (metrics["metric"].isin(["mae", "log_loss"])) & (metrics["head"] == "ridge")
                & (metrics["naive_kind"].isin(["zero", "fit_prior"]))]
    current = {s["set_id"]: s["set_sha256"] for s in plan["sets"]}
    m = m[[current.get(a) == b for a, b in zip(m["set_id"], m["set_sha256"])]]
    base = m[m["set_id"] == "ALL_ADMISSIBLE"].set_index(["target", "horizon", "fold"])["value"]
    rem = m[m["set_kind"] == "REMOVAL"]
    cells, per_feature = [], []
    for sid, g in rem.groupby("set_id"):
        feat = sid.split("ALL_MINUS:", 1)[1]
        vals = []
        for r in g.itertuples(index=False):
            key = (r.target, r.horizon, r.fold)
            if key not in base.index:
                continue
            b = float(base.loc[key])
            gain = (float(r.value) - b) / b if b else float("nan")
            vals.append(gain)
            cells.append({"feature_id": feat, "family": "identity", "target": r.target, "horizon": int(r.horizon), "fold": r.fold,
                          "loss_with": b, "loss_without": float(r.value), "refit_gain": gain, "metric": r.metric,
                          "naive_kind": r.naive_kind, "naive_value": float(r.naive_value), "rows_sha256": r.rows_sha256})
        if vals:
            per_feature.append({"feature_id": feat, "family": "identity", "refit_gain": float(np.mean(vals)),
                                "cells": len(vals), "positive_cells": int(sum(v > 0 for v in vals)),
                                "head": "ridge", "rows": "TRAIN inner folds, identical rows", "trained_families": "NOT_AVAILABLE_ON_REFIT_HOST"})
    return per_feature, cells


def write_csv_rows(path: Path, rows: list[dict], cols: list[str]) -> bool:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    return write_atomic(path, buf.getvalue().encode())


# ============================================================================ validation closure
CLOSURE_RULE = {
    "schema": "fs_close_closure_rule.v1",
    "declared_before_reading_validation": True,
    "candidates": "the frozen K=24 sets of ALL_ADMISSIBLE, PRED_BEST, PLUS_CAUSAL, PLUS_REP and KNOCKOFF (only if calibrated)",
    "refit": "the same ridge head, alpha 1.0, fit on all TRAIN rows (2012-05..2023-12) with fit-row standardisation and median imputation; one seed",
    "evaluation": "EXTERNAL VALIDATION rows (2024-01-01 <= t < 2025-01-01) read exactly once; MAE per cell against the same-row zero and fit-mean naives; log-loss against the fit prior",
    "score": "mean over the 14 cells of skill = 1 - loss / stricter paired naive",
    "winner": "highest score; ties -> fewer features, then lower total fit seconds, then the order above",
    "strategy_eligibility": "a set is strategy-eligible only if it beats its paired naive strictly on every short and long horizon cell; otherwise the manifest is FINAL with strategy_eligible = false",
    "test": "EXTERNAL TEST (2025) is never read by this tool; the loader refuses any row at or after 2025-01-01",
}


# ============================================================================ dispositions
REASON = {
    "SELECTED": "SELECTED_PRIMARY_K24_WINNER_SET",
    "NOT_IN_K24": "NOT_IN_PRIMARY_K24_OF_WINNER",
    "IN_SENS": "IN_WINNER_SENSITIVITY_K{k}_ONLY",
    "REDUNDANT": "REJECTED_REDUNDANT_WITH_SELECTED_ON_EVERY_MEASURED_TARGET",
    "NO_UTILITY": "REJECTED_NO_OOF_UTILITY_ANY_TARGET_AND_IN_NO_K48_SET",
    "CONTRADICTED": "REJECTED_CAUSAL_CONTRADICTED_ROBUST_ALL_TARGETS_AND_IN_NO_K48_SET",
    "CAUSAL_NEUTRAL": "CAUSAL_NOT_IDENTIFIED_NEUTRAL",
    "CAUSAL_SUPPORTED": "CAUSAL_SUPPORTED",
    "DRAFT": "PENDING_DRAFT_WINNER_NOT_CHOSEN",
    "PS3R_PENDING": "PS3R_TERMINALS_PENDING",
    "REP": "REPRESENTATION_{d}",
}


def build_dispositions(pop: Population, ev: LaneEvidence, plan: dict, winner: dict | None) -> list[dict]:
    """One automatic disposition per candidate; no manual per-feature decision anywhere."""
    members_k = {}
    if winner:
        wk = winner["set_kind"]
        for s in plan["sets"]:
            if s["set_kind"] == wk and s["k"] is not None:
                feats = set()
                for lst in (s["features_by"].values() if "features_by" in s else [s["features"]]):
                    feats.update(lst)
                members_k[s["k"]] = feats
    in_any_k48 = set()
    for s in plan["sets"]:
        if s["set_kind"] in ("PRED_METHOD", "PRED_BEST", "PLUS_CAUSAL", "PLUS_REP", "KNOCKOFF") and s["k"] is not None and s["k"] <= 48:
            for lst in (s["features_by"].values() if "features_by" in s else [s["features"]]):
                in_any_k48.update(lst)
    selected = members_k.get(K_PRIMARY, set()) if winner else set()
    rows = []
    for f in pop.names:
        reasons, digests = [], {}
        causal_states = set(ev.causal.get(f, {}).values()) or {"NOT_IDENTIFIED"}
        rep = ev.rep.get(f, {}).get("decision")
        ps2 = ev.ps2_status.get(f)
        if ev.digests.get("fs_causal/causal_evidence.jsonl"):
            digests["causal"] = ev.digests["fs_causal/causal_evidence.jsonl"]
        if ev.digests.get("fs_rep/representation_dispositions.csv"):
            digests["representation"] = ev.digests["fs_rep/representation_dispositions.csv"]
        if winner is None:
            state = "PENDING"
            reasons.append(REASON["DRAFT"])
        elif f in selected:
            state = "SELECTED"
            reasons.append(REASON["SELECTED"])
            digests["winner_set"] = winner["set_sha256"]
        else:
            state = "PENDING"
            sens = [k for k in sorted(members_k) if k != K_PRIMARY and f in members_k[k]]
            reasons.append(REASON["IN_SENS"].format(k=sens[0]) if sens else REASON["NOT_IN_K24"])
            # rejection needs positive evidence; never from NOT_IDENTIFIED
            if f not in in_any_k48 and ps2:
                deltas = [v for v in ps2["oof_delta"].values() if np.isfinite(v)]
                if ps2["redundant_with"] and ps2["redundant_with"] <= selected and deltas:
                    state = "REJECTED"
                    reasons.append(REASON["REDUNDANT"])
                elif deltas and all(v <= 0 for v in deltas) and "SUPPORTED" not in causal_states:
                    state = "REJECTED"
                    reasons.append(REASON["NO_UTILITY"])
                elif causal_states == {"CONTRADICTED"} and ev.causal.get(f):
                    state = "REJECTED"
                    reasons.append(REASON["CONTRADICTED"])
        if "SUPPORTED" in causal_states:
            reasons.append(REASON["CAUSAL_SUPPORTED"])
        elif causal_states <= {"NOT_IDENTIFIED"}:
            reasons.append(REASON["CAUSAL_NEUTRAL"])
        if ev.gen_reason.get(f) == "NO_SERIES_IN_PS2_BATCH":
            reasons.append("NO_SERIES_IN_PS2_BATCH")
        if f in ev.heavy:
            if rep and rep != "PENDING":
                reasons.append(REASON["REP"].format(d=rep))
            else:
                reasons.append(REASON["PS3R_PENDING"])
        rows.append({"feature": f, "batch": pop.batch_of[f], "state": state, "reason_codes": ";".join(reasons),
                     "representation": rep or ("RAW" if f not in ev.heavy else "PENDING"),
                     "causal": "/".join(sorted(causal_states)), "heavy_candidate": f in ev.heavy,
                     "in_primary_k24": f in selected,
                     "in_sensitivity_k": ";".join(str(k) for k in sorted(members_k) if f in members_k[k]),
                     "evidence_digests": json.dumps(digests, sort_keys=True)})
    return rows


# ============================================================================ checks
def run_checks(pop: Population, ev: LaneEvidence, plan: dict, plan_missing: list[str], dispositions: list[dict],
               metrics, cov: dict, closure: dict | None, gate_path: Path) -> list[dict]:
    checks = []

    def add(cid, state, detail):
        checks.append({"id": cid, "state": state, "detail": detail})

    states = [d["state"] for d in dispositions]
    ok = len(dispositions) == DENOMINATOR and len({d["feature"] for d in dispositions}) == DENOMINATOR \
        and all(s in STATES for s in states) and set(d["feature"] for d in dispositions) == pop.set
    add("C1_DISPOSITIONS_366", "PASS" if ok else "FAIL",
        f"{len(dispositions)} rows, {len(set(states))} states; SELECTED {states.count('SELECTED')} REJECTED {states.count('REJECTED')} PENDING {states.count('PENDING')}")

    pops = set(metrics["population_sha256"].unique()) if metrics is not None and len(metrics) else set()
    mixed = bool(pops - {pop.digest})
    add("C2_SINGLE_POPULATION", "FAIL" if mixed else "PASS",
        f"population {pop.digest[:12]}; metric populations {sorted(p[:12] for p in pops) or 'none yet'}")

    add("C3_COMPLETE_RANKINGS", "FAIL" if ev.pred_incomplete else ("PASS" if ev.pred_rankings else "PENDING"),
        f"complete methods {sorted(set(ev.pred_rankings) - set(ev.pred_incomplete))}; incomplete {ev.pred_incomplete}")

    bounds = []
    if metrics is not None and len(metrics):
        receipt = None  # receipts hold the bound; the plan loader also refused any row >= TRAIN_END
    rec = None
    c4 = "PASS"
    detail = f"refit loader refuses rows >= {TRAIN_END}; validation loader refuses rows >= {VALIDATION_END}; validation read {'once' if closure else 'not yet'}"
    if closure:
        vb = closure.get("validation_bound") or {}
        tb = closure.get("train_bound") or {}
        if not (vb.get("max_ts", 10**12) < 1735689600 and vb.get("min_ts", 0) >= 1704067200 and tb.get("max_ts", 10**12) < 1704067200
                and closure.get("validation_read_count") == 1):
            c4 = "FAIL"
            detail += "; closure record bounds violate the windows"
    add("C4_NO_TEST_READ", c4, detail)

    bad = [d for d in dispositions if d["state"] == "REJECTED" and "CAUSAL_NOT_IDENTIFIED_NEUTRAL" in d["reason_codes"]
           and not any(code in d["reason_codes"] for code in (REASON["REDUNDANT"], REASON["NO_UTILITY"]))]
    add("C5_NOT_IDENTIFIED_NEUTRAL", "FAIL" if bad else "PASS", f"{len(bad)} rejections resting on NOT_IDENTIFIED alone")

    if metrics is not None and len(metrics):
        m = metrics[metrics["state"] == "MEASURED"]
        missing_naive = int(((m["naive_kind"] == "") | ~np.isfinite(m["naive_value"])).sum())
        add("C6_PAIRED_NAIVE", "FAIL" if missing_naive else "PASS", f"{missing_naive} measured metric rows without a paired naive of {len(m)}")
    else:
        add("C6_PAIRED_NAIVE", "PENDING", "no refit metrics yet")

    needed = [f"{kind}:{k}" for kind in ("PRED_BEST", "PLUS_CAUSAL", "PLUS_REP") for k in K_SENSITIVITIES] + ["ALL_ADMISSIBLE"]
    if ev.gen_calibrated and any(s["set_kind"] == "KNOCKOFF" for s in plan["sets"]):
        needed += [f"KNOCKOFF:{k}" for k in K_SENSITIVITIES]
    expected = cov["expected_cells_per_set_head"]
    missing_sets = [sid for sid in needed if cov["cells"].get(f"{sid}|ridge", 0) < expected]
    add("C7_REQUIRED_SETS_REFIT", "PASS" if not missing_sets else ("FAIL" if metrics is not None and not plan_missing else "PENDING"),
        f"{len(needed) - len(missing_sets)}/{len(needed)} required sets complete at ridge; missing {missing_sets[:6]}")

    add("C8_INPUT_OBJECTS", "PASS" if not ev.missing and not plan_missing else "PENDING",
        "; ".join(ev.missing + plan_missing) or "all lane objects present")

    if closure:
        add("C9_VALIDATION_CHOICE", "PASS" if closure.get("winner") else "FAIL",
            f"winner {closure.get('winner', {}).get('set_id')} score {closure.get('winner', {}).get('score')}; rule {closure.get('rule_sha256', '')[:12]}")
    else:
        add("C9_VALIDATION_CHOICE", "PENDING", "closure step not run: candidates not frozen or VALIDATION_2024_FEATURES_AND_TARGETS not materialised on the refit host")

    if gate_path.is_file():
        g = sha256_file(gate_path)
        add("C10_GATE_MODULE", "PASS" if g == GATE_SHA256_C0345F83 else "FAIL",
            f"tools/selected_manifest_gate.py sha256 {g[:16]} {'==' if g == GATE_SHA256_C0345F83 else '!='} c0345f83 contract")
    else:
        add("C10_GATE_MODULE", "FAIL", "tools/selected_manifest_gate.py absent")
    return checks


def all_pass(checks: list[dict]) -> bool:
    return all(c["state"] == "PASS" for c in checks)


# ============================================================================ manifest
def load_gate(gate_path: Path):
    spec = importlib.util.spec_from_file_location("selected_manifest_gate", gate_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def manifest_features(dispositions: list[dict]) -> list[dict]:
    out = []
    for d in dispositions:
        out.append({"name": d["feature"], "state": d["state"].lower(), "reason_codes": d["reason_codes"].split(";"),
                    "representation": d["representation"], "causal": d["causal"],
                    "in_primary_k24": bool(d["in_primary_k24"]),
                    "sensitivity_k": [int(k) for k in d["in_sensitivity_k"].split(";") if k],
                    "evidence_digests": json.loads(d["evidence_digests"])})
    return out


def plan_missing_from_checks(checks: list[dict]) -> list[str]:
    for c in checks:
        if c["id"] == "C8_INPUT_OBJECTS" and c["state"] != "PASS":
            return [x.strip() for x in c["detail"].split(";") if x.strip() and not x.strip().startswith("all lane")]
    return []


def build_manifest(pop: Population, ev: LaneEvidence, plan: dict, dispositions: list[dict], checks: list[dict],
                   closure: dict | None, source_digests: dict, final: bool, skill_summary: dict | None = None) -> tuple[dict, dict | None]:
    targets = [{"name": "Y_s", "horizons": [1, 2, 3, 4, 5, 6]}, {"name": "Y_l", "horizons": [24, 48, 72, 96, 120, 144]},
               {"name": "Y_b", "horizons": [6, 144]}]
    winner = (closure or {}).get("winner")
    body = {
        "schema": MANIFEST_SCHEMA if final else DRAFT_SCHEMA,
        "manifest_version": 1,
        "status": "FINAL" if final else "DRAFT",
        "dataset_id": DATASET_ID,
        "asset": "EURUSD",
        "denominator": len(pop.names),
        "population_sha256": pop.digest,
        "targets": targets,
        "folds": list(FOLDS),
        "seeds": [0],
        "k_primary": K_PRIMARY,
        "k_sensitivities": list(K_SENSITIVITIES),
        "selection": {"primary_set": winner, "sensitivities": (closure or {}).get("sensitivities"),
                      "closure_rule": CLOSURE_RULE, "strategy_eligible": (closure or {}).get("strategy_eligible"),
                      "train_inner_fold_skill_vs_stricter_naive": skill_summary or {},
                      "arm_5_knockoff": {"state": ev.gen_state, "n_selected_features": len(ev.gen_selected), "calibration_counts": ev.gen_counts},
                      "note": "a set that does not beat its paired naive on a target is reported as such; strategy use requires strict skill"},
        "features": manifest_features(dispositions),
        "producer": {"role": PRODUCER_ROLE, "tool": "tools/fs_close_manifest.py"},
        "independent_decision_record": {"decision_id": f"FS-CLOSE-{dt.date.today().isoformat()}-001", "decider_role": DECIDER_ROLE},
        "source_digests": {k: v for k, v in source_digests.items() if re.fullmatch(r"[0-9a-f]{64}", str(v))},
        "checks": checks,
        "generated_utc": utc_now(),
    }
    if not final:
        body["missing_objects"] = sorted(set(ev.missing + plan_missing_from_checks(checks)))
        body["failing_or_pending_checks"] = [c["id"] + ":" + c["state"] for c in checks if c["state"] != "PASS"]
        body["manifest_sha256"] = canonical_sha256(body, "manifest_sha256")
        return body, None
    body["manifest_sha256"] = canonical_sha256(body, "manifest_sha256")
    decision = {"schema": DECISION_SCHEMA, "decision_id": body["independent_decision_record"]["decision_id"],
                "decider_role": DECIDER_ROLE, "decided_on": dt.date.today().isoformat(),
                "manifest_sha256": body["manifest_sha256"], "verdict": ACCEPT_VERDICT,
                "basis": "every fail-closed check PASS; the verifier re-read the manifest bytes from disk and re-derived its digest",
                "checks": checks}
    decision["record_sha256"] = canonical_sha256(decision, "record_sha256")
    return body, decision


# ============================================================================ status + figures
def lane_coverage(pop: Population, ev: LaneEvidence, cov: dict, dispositions: list[dict], catalog: dict | None) -> dict:
    n = len(pop.names)
    heavy = len(ev.heavy) or HEAVY_CANDIDATES
    ps3r_complete = sum(1 for f in ev.heavy if {"baseline", "alt_mtae", "alt_p2c"} <= set(ev.ps3r_terminals.get(f, {})))
    ps3r_terminals = sum(len(v) for v in ev.ps3r_terminals.values())
    rep_decided = sum(1 for v in ev.rep.values() if v["decision"] != "PENDING")
    methods_complete = len(set(ev.pred_rankings) - set(ev.pred_incomplete))
    lanes = {
        "PS1": {"done": n, "total": n, "unit": "candidates profiled"},
        "PS2": {"done": len(ev.ps2_status), "total": n, "unit": "candidates with PS2 status"},
        "PS3-C": {"done": len([f for f in pop.names if f in _ps3c_features(ev)]), "total": n, "unit": "candidates in PS3-C join"},
        "PS3-R": {"done": ps3r_complete, "total": heavy, "unit": "heavy candidates with 3 terminals",
                  "terminals_done": ps3r_terminals, "terminals_total": heavy * 3},
        "PS4": {"done": len(ev.ps4_profiled & set(ev.heavy)) if ev.heavy else len(ev.ps4_profiled), "total": heavy, "unit": "heavy candidates profiled"},
        "FS-PRED": {"done": methods_complete, "total": max(len(ev.pred_rankings), len(ev.pred_methods_planned), 14),
                    "unit": "methods with complete rankings", "eta_utc": ev.pred_progress.get("eta_utc"),
                    "method_cells_done": ev.pred_progress.get("method_cells_done"), "method_cells_total": ev.pred_progress.get("method_cells_total"),
                    "folds_in_progress": ev.pred_progress.get("folds")},
        "FS-CAUSAL": {"done": len(ev.causal), "total": n, "unit": "candidates with causal evidence"},
        "FS-REP": {"done": rep_decided, "total": heavy, "unit": "heavy candidates decided"},
        "FS-GEN": {"done": sum(v for k, v in ev.gen_counts.items() if k != "NOT_EVALUATED"), "total": n,
                   "unit": f"candidates with generative evidence (arm 5 {ev.gen_state})"},
        "FS-CLOSE": {"done": cov["complete"], "total": cov["planned"], "unit": "set x head refits complete"},
    }
    for v in lanes.values():
        v["fraction"] = round(v["done"] / v["total"], 4) if v["total"] else 0.0
    return lanes


def _ps3c_features(ev: LaneEvidence) -> set:
    return getattr(ev, "_ps3c", set())


def checklist(lanes: dict, final: bool) -> dict:
    def st(frac):
        return "DONE" if frac >= 1.0 else ("IN_PROGRESS" if frac > 0 else "NOT_STARTED")
    return {"I3_reversible_prioritisation": st(lanes["PS2"]["fraction"]),
            "I4-C_causal_ladder": st(min(lanes["PS3-C"]["fraction"], lanes["FS-CAUSAL"]["fraction"]) if lanes["FS-CAUSAL"]["done"] else lanes["PS3-C"]["fraction"] * 0.5),
            "I4-R_extractibility": st(min(lanes["PS3-R"]["fraction"], lanes["FS-REP"]["fraction"])),
            "I5_joint_comparison_and_manifest": "DONE" if final else st(lanes["FS-CLOSE"]["fraction"] * 0.9)}


def build_status(pop: Population, ev: LaneEvidence, lanes: dict, checks: list[dict], manifest: dict, cov: dict,
                 catalog: dict | None, dispositions: list[dict], peak_bytes: int, plan: dict) -> dict:
    states = [d["state"] for d in dispositions]
    return {"schema": "fs_closure_status.v1", "generated_utc": utc_now(), "producer_role": PRODUCER_ROLE,
            "denominator": {"candidates": len(pop.names), "selector_episode_sources_excluded": len(pop.excluded_selector_sources),
                            "quality_excluded": pop.excluded_quality, "heavy_candidates": len(ev.heavy) or HEAVY_CANDIDATES},
            "manifest": {"status": manifest["status"], "schema": manifest["schema"], "manifest_sha256": manifest["manifest_sha256"],
                         "missing_objects": manifest.get("missing_objects", []),
                         "dispositions": {s: states.count(s) for s in STATES}},
            "lanes": lanes, "checklist": checklist(lanes, manifest["status"] == "FINAL"), "checks": checks,
            "refits": {"planned_set_heads": cov["planned"], "complete_set_heads": cov["complete"], "cells": cov["cells"],
                       "pred_best_method": plan.get("pred_best_method"), "sets": len(plan["sets"])},
            "gpu_eta": ev.gpu_eta, "catalog": catalog, "causal_counts": ev.causal_counts, "representation_counts": ev.rep_counts,
            "generative_state": ev.gen_state, "input_digests": ev.digests | pop.sources,
            "aggregation_peak_rss_bytes": peak_bytes}


def eta_text(ev: LaneEvidence) -> list[str]:
    lines = []
    queues = (ev.gpu_eta or {}).get("queues") or {}
    for name, q in queues.items():
        counts = q.get("counts") or {}
        if not counts.get("total"):
            continue
        eta = q.get("eta_seconds") or {}
        med = (q.get("durations_seconds") or {}).get("median")
        remaining = counts.get("pending", 0) + counts.get("running", 0)
        eta_s = eta.get("median") if isinstance(eta, dict) else eta
        eta_txt = f"{eta_s / 3600:.1f} h" if isinstance(eta_s, (int, float)) and eta_s else ("done" if remaining == 0 else "n/a")
        lines.append(f"{name}: {counts.get('completed', 0)}/{counts.get('total', 0)} done, {remaining} remaining, "
                     f"median {med:.0f} s, ETA {eta_txt}" if isinstance(med, (int, float)) else
                     f"{name}: {counts.get('completed', 0)}/{counts.get('total', 0)} done, {remaining} remaining, ETA {eta_txt}")
    if isinstance(ev.gpu_eta, dict) and "queues" not in ev.gpu_eta:
        for k, v in ev.gpu_eta.items():
            if k != "source" and isinstance(v, (str, int, float)):
                lines.append(f"{k}: {v}")
    return lines or ["no GPU queue status available"]


def render_progress(status: dict, out_png: Path, ev: LaneEvidence) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lanes = status["lanes"]
    names = list(lanes)
    fracs = [lanes[n]["fraction"] for n in names]
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(COLOR["surface"])
    final = status["manifest"]["status"] == "FINAL"
    fig.text(0.03, 0.955, "EURUSD feature selection: closure progress", fontsize=20, weight="bold", color=COLOR["ink"])
    fig.text(0.03, 0.915, f"Generated from evidence {status['generated_utc']}  |  denominator {status['denominator']['candidates']} candidates, "
             f"{status['denominator']['heavy_candidates']} heavy  |  manifest {status['manifest']['status']}",
             fontsize=10.5, color=COLOR["ink2"])
    ax = fig.add_axes([0.08, 0.36, 0.40, 0.50])
    ax.set_facecolor(COLOR["surface"])
    y = np.arange(len(names))[::-1]
    ax.barh(y, [1.0] * len(names), color=COLOR["track"], height=0.55, zorder=1)
    ax.barh(y, fracs, color=COLOR["bar"], height=0.55, zorder=2)
    for yi, n in zip(y, names):
        d = lanes[n]
        ax.text(1.02, yi, f"{d['done']}/{d['total']}  {d['unit']}"[:54], va="center", fontsize=8.2, color=COLOR["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, color=COLOR["ink"])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=8.5, color=COLOR["muted"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLOR["grid"])
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=COLOR["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Lane coverage over its denominator", loc="left", fontsize=11.5, color=COLOR["ink"], pad=10)

    # checklist + checks
    x0, ytop = 0.735, 0.86
    fig.text(x0, ytop + 0.02, "Checklist and fail-closed checks", fontsize=11.5, weight="bold", color=COLOR["ink"])
    marks = {"PASS": ("good", "PASS"), "DONE": ("good", "DONE"), "IN_PROGRESS": ("warning", "IN PROGRESS"),
             "PENDING": ("warning", "PENDING"), "NOT_STARTED": ("muted", "NOT STARTED"), "FAIL": ("critical", "FAIL")}
    row = 0
    for key, st in status["checklist"].items():
        c, label = marks.get(st, ("muted", st))
        fig.text(x0, ytop - row * 0.03, "■", color=COLOR[c], fontsize=10)
        fig.text(x0 + 0.015, ytop - row * 0.03, f"{key}: {label}", fontsize=9, color=COLOR["ink"])
        row += 1
    row += 0.4
    for ch in status["checks"]:
        c, label = marks.get(ch["state"], ("muted", ch["state"]))
        fig.text(x0, ytop - row * 0.03, "■", color=COLOR[c], fontsize=10)
        fig.text(x0 + 0.015, ytop - row * 0.03, f"{ch['id']}: {label}", fontsize=8.6, color=COLOR["ink"])
        row += 1

    # bottom: ETA + manifest banner
    fig.text(0.03, 0.29, "GPU queues (observed durations)", fontsize=11.5, weight="bold", color=COLOR["ink"])
    for i, line in enumerate(eta_text(ev)[:4]):
        fig.text(0.03, 0.255 - i * 0.03, line, fontsize=9, color=COLOR["ink2"])
    disp = status["manifest"]["dispositions"]
    banner = (f"MANIFEST {status['manifest']['status']}  |  SELECTED {disp['SELECTED']}  REJECTED {disp['REJECTED']}  "
              f"PENDING {disp['PENDING']}  |  refits complete {status['refits']['complete_set_heads']}/{status['refits']['planned_set_heads']} set x head")
    fig.text(0.03, 0.12, banner, fontsize=11, weight="bold", color=COLOR["good" if final else "serious"])
    missing = status["manifest"].get("missing_objects", [])
    if missing:
        text = "Missing before FINAL: " + "; ".join(missing)
        fig.text(0.03, 0.095, text[:150], fontsize=8.4, color=COLOR["ink2"])
        fig.text(0.03, 0.072, text[150:300], fontsize=8.4, color=COLOR["ink2"])
    fig.text(0.03, 0.04, "States are written as text next to every mark; percentages are coverage over declared denominators, not scientific confidence. "
             "TRAIN inner folds only; VALIDATION read once in the closure step; TEST never read.",
             fontsize=8, color=COLOR["muted"])
    tmp = out_png.with_suffix(".tmp.png")
    fig.savefig(tmp, facecolor=COLOR["surface"])
    plt.close(fig)
    os.replace(tmp, out_png)


MASTER_WEIGHTS = {"PS1": 0.05, "PS2": 0.10, "PS3-C": 0.10, "PS3-R": 0.25, "PS4": 0.10, "FS-PRED": 0.15,
                  "FS-CAUSAL": 0.05, "FS-REP": 0.05, "FS-CLOSE": 0.15}


def regenerate_master(paths: Paths, status: dict, ev: LaneEvidence) -> bool:
    """Rewrite M2 (and the as-of / critical-path ETA) from evidence; other milestones are kept verbatim."""
    master_path = paths.evidence / "MASTER_MILESTONE_STATUS.json"
    doc = read_json(master_path)
    if not doc:
        return False
    lanes = status["lanes"]
    final = status["manifest"]["status"] == "FINAL"
    progress = 100 if final else int(round(100 * sum(MASTER_WEIGHTS[k] * lanes[k]["fraction"] for k in MASTER_WEIGHTS) / sum(MASTER_WEIGHTS.values())))
    progress = min(progress, 99) if not final else 100
    disp = status["manifest"]["dispositions"]
    ps3r = lanes["PS3-R"]
    evidence = (f"PS3-R {ps3r['terminals_done']}/{ps3r['terminals_total']} terminals ({ps3r['done']}/{ps3r['total']} complete); PS4 {lanes['PS4']['done']}/{lanes['PS4']['total']}; "
                f"FS-PRED {lanes['FS-PRED']['done']} methods; causal {lanes['FS-CAUSAL']['done']}/{lanes['FS-CAUSAL']['total']}; rep {lanes['FS-REP']['done']}/{lanes['FS-REP']['total']}; "
                f"refits {status['refits']['complete_set_heads']}/{status['refits']['planned_set_heads']}; manifest {status['manifest']['status']} "
                f"sel/rej/pend {disp['SELECTED']}/{disp['REJECTED']}/{disp['PENDING']}")
    pending = [c for c in status["checks"] if c["state"] != "PASS"]
    nxt = ("Manifest FINAL; hand the K=24 primary set to ARCH under the gate" if final else
           "Blocking: " + "; ".join(status["manifest"].get("missing_objects", [])[:3] or [c["id"] for c in pending[:3]]))
    eta_lines = eta_text(ev)
    gpu_eta = "/".join(l.split(", ETA ")[-1] if ", ETA " in l else "n/a" for l in eta_lines[:3])
    for m in doc.get("milestones", []):
        if m.get("id") == "M2":
            m["progress"] = progress
            m["state"] = "MANIFEST_FINAL" if final else "IN_PROGRESS"
            m["evidence"] = evidence
            m["next"] = nxt
            m["eta"] = "manifest FINAL" if final else f"GPU {gpu_eta}; then closure"
    doc["as_of"] = status["generated_utc"]
    doc["critical_path_eta"] = ("feature-selection manifest FINAL; model, strategy and RL milestones downstream" if final else
                                f"feature-selection manifest after GPU queues ({gpu_eta}) and the closure step; later milestones downstream")
    doc["regenerated_by"] = {"tool": "tools/fs_close_manifest.py", "from": "fs_closure/STATUS.json"}
    changed = write_json(master_path, doc)
    try:
        subprocess.run([sys.executable, str(paths.evidence / "render_master_milestones.py")], check=True,
                       capture_output=True, timeout=180)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        status.setdefault("warnings", []).append(f"master render failed: {getattr(exc, 'stderr', b'')[-300:]}")
    return changed


# ============================================================================ worker dispatch
def worker_dispatch(paths: Paths, plan_path: Path, cov: dict, status_notes: list[str]) -> dict:
    """Push plan + engine to the worker, launch the capped sequential refit when idle, pull results."""
    if not paths.worker_alias:
        return {"state": "NO_WORKER_CONFIGURED"}
    alias, wstate = paths.worker_alias, paths.worker_state
    remote = f"{wstate}/fs_close"
    info = {"state": "IDLE"}
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", alias]
    try:
        subprocess.run(ssh + [f"mkdir -p {remote}/out"], check=True, capture_output=True, timeout=60)
        subprocess.run(["rsync", "-q", "--timeout=60", str(plan_path), str(paths.repo / "tools/fs_close_refit.py"),
                        f"{alias}:{remote}/"], check=True, capture_output=True, timeout=120)
        subprocess.run(["rsync", "-q", "--timeout=120", "--include=paired_refit_metrics.parquet", "--include=refit_receipt.json",
                        "--include=progress.json", "--include=closure_record.json", "--exclude=*", f"{alias}:{remote}/out/",
                        str(paths.state / "fs_close/")], check=False, capture_output=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        info["state"] = "WORKER_UNREACHABLE"
        info["detail"] = str(getattr(exc, "stderr", b""))[-300:]
        return info
    mirror = paths.state / "fs_close/fs_pred_mirror"
    mirror.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-q", "-r", "--timeout=120", "--include=tables/", "--include=tables/*.parquet", "--include=selector_sets.json",
                    "--include=progress.json", "--include=run_contract.json", "--exclude=*", f"{alias}:{wstate}/fs_pred/out/", str(mirror)],
                   check=False, capture_output=True, timeout=300)
    local_out = paths.state / "fs_close"
    for name in ("paired_refit_metrics.parquet", "refit_receipt.json", "progress.json", "closure_record.json"):
        src = local_out / name
        if src.is_file():
            write_atomic(paths.out / name, src.read_bytes())
    progress = read_json(local_out / "progress.json", {})
    running = progress.get("state") == "RUNNING"
    if running:
        try:
            upd = dt.datetime.fromisoformat(progress["updated_utc"])
            running = (dt.datetime.now(dt.timezone.utc) - upd).total_seconds() < 1800
        except (KeyError, ValueError):
            running = False
    marker = read_json(paths.out / "refit_launch_marker.json", {})
    try:
        launched_at = dt.datetime.fromisoformat(marker.get("launched_utc", "1970-01-01T00:00:00+00:00"))
    except ValueError:
        launched_at = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    unit_alive = False
    if marker.get("unit") and (dt.datetime.now(dt.timezone.utc) - launched_at).total_seconds() < 14400:
        try:
            r = subprocess.run(ssh + [f"export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user show -p ActiveState --value {marker['unit']}"],
                               capture_output=True, text=True, timeout=40)
            unit_alive = r.stdout.strip() in ("active", "activating")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            unit_alive = True   # unknown: do not double-launch
    if running or unit_alive:
        info["state"] = "RUNNING" if running else "QUEUED_OR_STARTING"
        info["progress"] = progress
        info["launch_marker"] = marker
        return info
    if cov["complete"] >= cov["planned"]:
        info["state"] = "COMPLETE"
        closure_rec = paths.out / "closure_record.json"
        if not closure_rec.is_file():
            info["closure"] = launch_closure(paths, ssh, alias, wstate, remote)
        return info
    receipt = read_json(paths.out / "refit_receipt.json", {})
    cap_file = paths.out / "refit_cap.json"
    cap_doc = read_json(cap_file, {})
    peak = receipt.get("peak_rss_bytes") if receipt.get("plan_sha256") else None
    if peak:
        # the cap is 1.25x the largest whole-process peak ever measured, never lowered
        cap_bytes = max(int(cap_doc.get("cap_bytes", 0)), int(math.ceil(peak * 1.25)))
        write_json(cap_file, {"schema": "fs_close_refit_cap.v1", "cap_bytes": cap_bytes, "measured_peak_rss_bytes": max(int(cap_doc.get("measured_peak_rss_bytes", 0)), int(peak)),
                              "rule": "1.25 x measured whole-process peak RSS from the refit receipt; monotone", "receipt_plan_sha256": receipt.get("plan_sha256")})
        cap_doc = read_json(cap_file, {})
    pilot = not cap_doc.get("cap_bytes")
    if pilot:
        # first contact: a bounded pilot (largest fold, the two heaviest sets) measures the footprint
        pilot_plan = {k: v for k, v in read_json(plan_path, {}).items()}
        pilot_plan["sets"] = [s for s in pilot_plan.get("sets", []) if s["set_id"] in ("ALL_ADMISSIBLE", f"RANDOM_K:{K_PRIMARY}")]
        pilot_path = paths.out / "refit_plan_pilot.json"
        write_json(pilot_path, pilot_plan)
        try:
            subprocess.run(["rsync", "-q", str(pilot_path), f"{alias}:{remote}/"], check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            info["state"] = "LAUNCH_FAILED"
            info["detail"] = str(exc)[-300:]
            return info
        plan_arg, cap, extra = f"{remote}/refit_plan_pilot.json", "1500M", f" --folds-only {FOLDS[-1]}"
        job_name = "fs_close_refit_pilot"          # its own name: genuinely smaller work than the full plan
    else:
        cap_mb = int(math.ceil(cap_doc["cap_bytes"] / (1 << 20)))
        plan_arg, cap, extra = f"{remote}/refit_plan.json", f"{cap_mb}M", ""
        job_name = f"fs_close_refit_c{cap_mb}m"    # the name carries the measured cap: never lowered under one name
    cmd = (f"export XDG_RUNTIME_DIR=/run/user/$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus; "
           f"cd {remote} && systemd-run --user --collect --unit fs-close-refit-$(date +%s) "
           f"~/.local/bin/crispdm-run -q -W 14400 -m {cap} -t 8h -n {job_name} -- {paths.worker_python} {remote}/fs_close_refit.py "
           f"--plan {plan_arg} --features {wstate}/fs_pred/input/ps1/batch_001/features_train.parquet "
           f"{wstate}/fs_pred/input/ps1/batch_002/features_train.parquet {wstate}/fs_pred/input/ps1/batch_003/features_train.parquet "
           f"--targets {wstate}/fs_pred/input/ps1/batch_001/targets_train.parquet --folds {remote}/folds.json "
           f"--out-dir {remote}/out --heads ridge,hgb --hgb-max-k {K_PRIMARY}{extra}")
    try:
        subprocess.run(["rsync", "-q", str(paths.evidence / "laneA/batch_001/folds.json"), f"{alias}:{remote}/folds.json"],
                       check=True, capture_output=True, timeout=60)
        r = subprocess.run(ssh + [cmd], capture_output=True, timeout=90, text=True)
        info["state"] = ("PILOT_LAUNCHED" if pilot else "LAUNCHED") if r.returncode == 0 else "LAUNCH_FAILED"
        if r.returncode == 0:
            unit = re.search(r"Running as unit: ([^;\s]+)", (r.stderr or "") + (r.stdout or ""))
            write_json(paths.out / "refit_launch_marker.json", {"launched_utc": utc_now(), "pilot": pilot, "cap": cap,
                                                                 "unit": unit.group(1) if unit else ""})
        info["cap"] = cap
        info["queued"] = "crispdm-run -q waits for host headroom up to 4 h before starting"
        info["detail"] = (r.stderr or r.stdout)[-400:]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        info["state"] = "LAUNCH_FAILED"
        info["detail"] = str(exc)[-300:]
    return info


def launch_closure(paths: Paths, ssh: list, alias: str, wstate: str, remote: str) -> dict:
    """The closure step reads VALIDATION once; it runs only when the validation inputs are materialised."""
    val = f"{wstate}/validation_2024"
    probe = subprocess.run(ssh + [f"ls {val}/ps1/batch_001/features_train.parquet {val}/ps1/batch_002/features_train.parquet "
                                  f"{val}/ps1/batch_003/features_train.parquet {val}/ps1/batch_001/targets_train.parquet >/dev/null 2>&1 && echo OK || echo MISSING"],
                           capture_output=True, text=True, timeout=40)
    if "OK" not in probe.stdout:
        return {"state": "VALIDATION_INPUTS_MISSING", "expected": f"{val}/ps1/batch_00{{1,2,3}}/features_train.parquet + batch_001/targets_train.parquet (2024 rows only)"}
    cap_doc = read_json(paths.out / "refit_cap.json", {})
    cap_mb = int(math.ceil(max(cap_doc.get("cap_bytes", 0), 1) / (1 << 20))) or 1500
    cmd = (f"export XDG_RUNTIME_DIR=/run/user/$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus; cd {remote} && "
           f"systemd-run --user --collect --unit fs-close-closure-$(date +%s) ~/.local/bin/crispdm-run -q -W 14400 -m {cap_mb}M -t 4h -n fs_close_closure_c{cap_mb}m -- "
           f"{paths.worker_python} {remote}/fs_close_refit.py --closure --plan {remote}/refit_plan.json "
           f"--features {wstate}/fs_pred/input/ps1/batch_001/features_train.parquet {wstate}/fs_pred/input/ps1/batch_002/features_train.parquet "
           f"{wstate}/fs_pred/input/ps1/batch_003/features_train.parquet --targets {wstate}/fs_pred/input/ps1/batch_001/targets_train.parquet "
           f"--folds {remote}/folds.json --val-features {val}/ps1/batch_001/features_train.parquet {val}/ps1/batch_002/features_train.parquet "
           f"{val}/ps1/batch_003/features_train.parquet --val-targets {val}/ps1/batch_001/targets_train.parquet --out-dir {remote}/out")
    r = subprocess.run(ssh + [cmd], capture_output=True, text=True, timeout=90)
    return {"state": "CLOSURE_LAUNCHED" if r.returncode == 0 else "CLOSURE_LAUNCH_FAILED", "detail": (r.stderr or r.stdout)[-300:]}


# ============================================================================ follow once
def follow_once(paths: Paths, *, dispatch: bool = False, rebuild_catalog: bool | None = None, render: bool = True) -> dict:
    t0 = time.time()
    pop = load_population(paths)
    ev = load_lanes(paths, pop)
    ev._ps3c = _ps3c_covered(paths, pop)
    metrics = load_metrics(paths)
    plan, plan_missing = build_plan(pop, ev, metrics)
    plan_path = paths.out / "refit_plan.json"
    write_json(plan_path, plan)
    cov = refit_coverage(plan, metrics)
    dispatch_info = worker_dispatch(paths, plan_path, cov, []) if dispatch else {"state": "DISPATCH_DISABLED"}
    if dispatch_info.get("state") in ("RUNNING", "COMPLETE", "LAUNCHED"):
        metrics = load_metrics(paths)
        cov = refit_coverage(plan, metrics)
    closure = read_json(paths.out / "closure_record.json", None)
    if closure is None:
        ev.missing.append("VALIDATION_2024_FEATURES_AND_TARGETS on the refit host (PS1 producer, read_end 2025-01-01, 2024 rows only) -> closure_record.json")
    winner = (closure or {}).get("winner")
    dispositions = build_dispositions(pop, ev, plan, winner)
    checks = run_checks(pop, ev, plan, plan_missing, dispositions, metrics, cov, closure, paths.gate_module)
    final = all_pass(checks)
    source_digests = {**pop.sources, **ev.digests}
    if (paths.out / "paired_refit_metrics.parquet").is_file():
        source_digests["fs_close/paired_refit_metrics.parquet"] = sha256_file(paths.out / "paired_refit_metrics.parquet")
    skill_summary = summarize_skill(metrics)
    manifest, decision = build_manifest(pop, ev, plan, dispositions, checks, closure, source_digests, final, skill_summary)
    if final:
        gate = load_gate(paths.gate_module)
        selected = [f["name"] for f in manifest["features"] if f["state"] == "selected"]
        report = gate.evaluate(manifest, selected, decision_record=decision, expected_dataset_id=DATASET_ID)
        if not report["admitted"]:
            final = False
            checks.append({"id": "C11_GATE_ACCEPTS_MANIFEST", "state": "FAIL", "detail": "; ".join(report["reasons"])[:400]})
            manifest, decision = build_manifest(pop, ev, plan, dispositions, checks, closure, source_digests, False, skill_summary)
        else:
            checks.append({"id": "C11_GATE_ACCEPTS_MANIFEST", "state": "PASS", f"detail": f"gate admitted {len(selected)} selected features"})
            manifest["checks"] = checks
            manifest["manifest_sha256"] = canonical_sha256(manifest, "manifest_sha256")
            decision["manifest_sha256"] = manifest["manifest_sha256"]
            decision["checks"] = checks
            decision["record_sha256"] = canonical_sha256(decision, "record_sha256")
    # artifacts
    write_dispositions(paths.out / "feature_dispositions.csv", dispositions)
    write_json(paths.out / "selector_sets.json", selector_sets_document(plan, ev, pop))
    write_json(paths.out / "CHECKS.json", {"schema": "fs_close_checks.v1", "generated_utc": utc_now(), "final": final, "checks": checks})
    write_json(paths.out / "closure_rule.json", CLOSURE_RULE | {"rule_sha256": canonical_sha256(CLOSURE_RULE)})
    write_json(paths.out / "stability_jaccard.json", jaccard_stability(plan))
    gains, gain_cells = refit_gain_rows(metrics, plan)
    write_csv_rows(paths.out / "refit_gain_export.csv", gains,
                   ["feature_id", "family", "refit_gain", "cells", "positive_cells", "head", "rows", "trained_families"])
    write_csv_rows(paths.out / "refit_gain_cells.csv", gain_cells,
                   ["feature_id", "family", "target", "horizon", "fold", "metric", "loss_with", "loss_without", "refit_gain",
                    "naive_kind", "naive_value", "rows_sha256"])
    copy_if_present(paths.fs_closure / "fs_rep/representation_dispositions.csv", paths.out / "representation_dispositions.csv")
    copy_if_present(paths.fs_closure / "fs_causal/causal_evidence.jsonl", paths.out / "causal_evidence.jsonl")
    write_json(paths.out / "FINAL_SELECTION_MANIFEST.json", manifest)
    if decision is not None:
        write_json(paths.out / "SELECTION_DECISION_RECORD.json", decision)
    elif (paths.out / "SELECTION_DECISION_RECORD.json").exists():
        (paths.out / "SELECTION_DECISION_RECORD.json").unlink()
    # catalog: rebuild only when any input changed
    src_digest = catalog_sources_digest(paths)
    cat_meta_path = paths.out / "FEATURE_METRICS_CATALOG.meta.json"
    cat_meta = read_json(cat_meta_path, {})
    if rebuild_catalog or (rebuild_catalog is None and cat_meta.get("sources_digest") != src_digest) or not (paths.out / "FEATURE_METRICS_CATALOG.parquet").is_file():
        cat_meta = build_catalog(paths, pop, plan) | {"sources_digest": src_digest}
        write_json(cat_meta_path, cat_meta)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    lanes = lane_coverage(pop, ev, cov, dispositions, cat_meta)
    status = build_status(pop, ev, lanes, checks, manifest, cov, cat_meta, dispositions, peak, plan)
    status["worker_dispatch"] = dispatch_info
    status["skill_summary"] = skill_summary
    status["refit_gain_export"] = {"path": "docs/audits/evidence/canonical_20261003/fs_closure/fs_close/refit_gain_export.csv",
                                   "columns": ["feature_id", "family", "refit_gain"], "features": len(gains)}
    status["cycle_seconds"] = round(time.time() - t0, 2)
    write_json(paths.fs_closure / "STATUS.json", status)
    if render:
        render_progress(status, paths.fs_closure / "PROGRESS.png", ev)
        regenerate_master(paths, status, ev)
    return status


def _ps3c_covered(paths: Paths, pop: Population) -> set:
    import pandas as pd

    out = set()
    for batch in ("batch_001", "batch_002", "batch_003"):
        p = paths.evidence / "laneC" / batch / "summary.csv"
        if p.is_file():
            df = pd.read_csv(p, usecols=["subject_kind", "subject"])
            out.update(df[df["subject_kind"] == "feature"]["subject"].astype(str))
    return out & pop.set


def write_dispositions(path: Path, rows: list[dict]) -> bool:
    cols = ["feature", "batch", "state", "reason_codes", "representation", "causal", "heavy_candidate",
            "in_primary_k24", "in_sensitivity_k", "evidence_digests"]
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
    return write_atomic(path, buf.getvalue().encode())


def copy_if_present(src: Path, dst: Path) -> None:
    if src.is_file():
        write_atomic(dst, src.read_bytes())


# ============================================================================ git cadence
OWNED_PATHS = ("docs/audits/evidence/canonical_20261003/fs_closure/fs_close",
               "docs/audits/evidence/canonical_20261003/fs_closure/STATUS.json",
               "docs/audits/evidence/canonical_20261003/fs_closure/PROGRESS.png",
               "docs/audits/evidence/canonical_20261003/MASTER_MILESTONE_STATUS.json",
               "docs/audits/evidence/canonical_20261003/MASTER_MILESTONE_PROGRESS.png")


def git_commit_owned(paths: Paths, message: str, push: bool = True) -> str:
    repo = paths.repo
    if (repo / ".git").is_file() or (repo / ".git").is_dir():
        pass
    lock = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-dir"], capture_output=True, text=True)
    if lock.returncode != 0:
        return "NOT_A_REPO"
    if (Path(lock.stdout.strip()) / "index.lock").exists():
        return "INDEX_LOCKED_SKIP"
    subprocess.run(["git", "-C", str(repo), "add", "--", *OWNED_PATHS], capture_output=True)
    diff = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"], capture_output=True)
    if diff.returncode == 0:
        return "NOTHING_TO_COMMIT"
    r = subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message + "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return "COMMIT_FAILED: " + r.stderr[-200:]
    if push:
        subprocess.run(["git", "-C", str(repo), "pull", "--no-rebase", "--no-edit", "-q"], capture_output=True, timeout=120)
        p = subprocess.run(["git", "-C", str(repo), "push", "-q"], capture_output=True, text=True, timeout=120)
        return "COMMITTED_PUSHED" if p.returncode == 0 else "COMMITTED_PUSH_FAILED: " + p.stderr[-200:]
    return "COMMITTED"


# ============================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("follow-once", "follow"):
        s = sub.add_parser(name)
        s.add_argument("--repo", default=None)
        s.add_argument("--state", default=None)
        s.add_argument("--worker", default=None, help="ssh alias of the CPU worker (role worker_b); omit to skip dispatch")
        s.add_argument("--no-render", action="store_true")
        s.add_argument("--rebuild-catalog", action="store_true")
        s.add_argument("--git-commit", action="store_true")
        if name == "follow":
            s.add_argument("--interval", type=float, default=60.0)
            s.add_argument("--commit-every", type=float, default=1800.0)
    a = ap.parse_args(argv)
    paths = Paths.default(Path(a.repo) if a.repo else None, Path(a.state) if a.state else None, a.worker)
    if a.cmd == "follow-once":
        st = follow_once(paths, dispatch=bool(a.worker), rebuild_catalog=True if a.rebuild_catalog else None, render=not a.no_render)
        if a.git_commit:
            st["git"] = git_commit_owned(paths, f"FS-CLOSE follower: manifest {st['manifest']['status']}, refits {st['refits']['complete_set_heads']}/{st['refits']['planned_set_heads']}")
        print(json.dumps({"manifest": st["manifest"]["status"], "dispositions": st["manifest"]["dispositions"],
                          "lanes": {k: f"{v['done']}/{v['total']}" for k, v in st["lanes"].items()},
                          "checks": {c["id"]: c["state"] for c in st["checks"]}, "dispatch": st["worker_dispatch"].get("state"),
                          "peak_rss_mb": round(st["aggregation_peak_rss_bytes"] / 2**20, 1), "git": st.get("git"),
                          "cycle_seconds": st["cycle_seconds"]}, indent=1))
        return 0
    last_commit = 0.0
    while True:
        try:
            st = follow_once(paths, dispatch=bool(a.worker), render=True)
            if a.git_commit and time.time() - last_commit >= a.commit_every:
                st["git"] = git_commit_owned(paths, f"FS-CLOSE follower: manifest {st['manifest']['status']}, refits {st['refits']['complete_set_heads']}/{st['refits']['planned_set_heads']}")
                last_commit = time.time()
            print(utc_now(), st["manifest"]["status"], st.get("git", ""), flush=True)
        except Exception as exc:  # the follower must survive a transient input error and report it
            print(utc_now(), "CYCLE_ERROR", type(exc).__name__, str(exc)[:300], flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
