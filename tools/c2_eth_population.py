"""Lane C2: the ETH 4h population, its frozen feature order, the split it is bound to, and the labels.

Everything lane C2 fits reads TRAIN rows only. The binding is refused, never patched, when a digest disagrees:

* the view: predictor `b1f8a74f` `examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`,
  sha256 `1b447c66...` (18,085 rows, 90 columns);
* the features: variant A of M03's admissible declaration `f3c0beca...` (83 features, order `all_admissible_control`),
  carried by the FROZEN_DEVELOPMENT manifest (feature-eng `b90c4b3`, file sha `fdff0c85...`, canonical `30b78078...`);
* the split: M07's `SPLIT_eth4h_l24_h6_v1.json` (predictor `satoshi/f2-eth-forecast-20261001` `13ef175f`, sha
  `116a5b64...`): declared calendar split TRAIN [0, 13699), VALIDATION [13699, 15895), TEST [15895, 18085) never read;
  window 24, horizons 1..6, purge h_max = 6; train origins [23, 13669] with regular 4 h steps over the 24-step input
  and the h_max target support (13,415 windows, 232 excluded for irregular steps);
* the target as M07 defines it: `Y_h = sum_{k=1..h} z(log_return_1[t+k])`, `z = (x - mu) / sigma`, mu and sigma of
  the `log_return_1` column on TRAIN rows only. The raw log return `log(CLOSE[t+h]/CLOSE[t])` is kept beside it so
  every error can be reported in both units (M07 z-units and log-return units).

Nothing here is governed: the view is a git-pinned DEVELOPMENT resource (manifest `resource.availability_class`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VIEW_SHA256 = "1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f"
VIEW_COMMIT = "b1f8a74f19f6d164be6367b6d57e726c6ff6b95a"
VIEW_PATH = "examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv"
DATASET_ID = "financial_data.project3.ethusdt_4h_tech_stat.model_ready.v1"
MANIFEST_FILE_SHA256 = "fdff0c85fc376cd6930cede4981a64b076022bd701c6a045339689e3ab892d4c"
MANIFEST_CANONICAL_SHA256 = "30b780784c9ec4f04ec6d58bec056e7dd9b15706595489856bf8e336202b3a67"
DECLARATION_SHA256 = "f3c0becaa1655ac16a567aef6ac0e67ecbb89ddc5b204e05d5a8dac2394b527d"
SPLIT_FILE_SHA256 = "116a5b645fe08138c49a558c206dd970528128bcd8370bb9ec8cd62b20a61e25"
SPLIT_AUTHORITY = "M07 (agent afcf115024ffa1381), predictor satoshi/f2-eth-forecast-20261001 13ef175f"
BAR_SECONDS = 14400
LANE_B_INNER_FOLDS = (  # feature-eng b90c4b3 runs/eth_4h/run_summary.json: expanding, purge 60 rows
    ("inner_1", (0, 7474), (7534, 9589)),
    ("inner_2", (0, 9529), (9589, 11644)),
    ("inner_3", (0, 11584), (11644, 13699)),
)
LANE_B_HORIZON_BARS = {"Y_s@4h": 1, "Y_l@24h": 6, "Y_l@144h": 36}


@dataclass(frozen=True)
class PopulationSpec:
    """Everything that distinguishes one population from another: the digests it must match, its bar width and how its
    train rows and target scale are declared. ETH 4h (M07 split file) is the default; a population whose frozen manifest
    declares its own split (lane B, e.g. EURUSD 1h) uses split_mode 'manifest'."""
    name: str
    view_sha256: str
    manifest_file_sha256: str
    declaration_sha256: str
    dataset_id: str
    view_path: str
    view_commit: str
    bar_seconds: int
    split_mode: str = "m07_split_file"        # or "manifest"
    split_file_sha256: str | None = None
    window: int = 24
    hmax: int = 6
    availability_class: str = "DEVELOPMENT"
    derived: str | None = None                # "range_v1": lane B vD1h recipes computed from bar t only


ETH_4H_SPEC = PopulationSpec(
    name="ETH_4h_variant_A", view_sha256=VIEW_SHA256, manifest_file_sha256=MANIFEST_FILE_SHA256, declaration_sha256=DECLARATION_SHA256,
    dataset_id=DATASET_ID, view_path=VIEW_PATH, view_commit=VIEW_COMMIT, bar_seconds=BAR_SECONDS, split_file_sha256=SPLIT_FILE_SHA256)

EURUSD_1H_SPEC = PopulationSpec(   # lane B frozen manifest feature-eng satoshi/b-source-transform-coverage-20261001 (eurusd_1h.v1)
    name="EURUSD_1h_variant_A", view_sha256="72b8271d2a6ab7fc5de8c64dc626f128dbb11a1127c333e4039edecbdcc41783",
    manifest_file_sha256="d36cb91e895673bd2e64738b804ae50e2fb826541ceebc4fe8fd24a3df2f42c5",
    declaration_sha256="6f8fed4197c0d1824969768b5bd69521da115bc7122f49ccd79f897f74e9ccf0",
    dataset_id="git:heuristic-strategy:939f5e6e:tests/data/eurusd_hour_2005_2020.csv", view_path="tests/data/eurusd_hour_2005_2020.csv",
    view_commit="939f5e6e9819c976a6bbb07876cb195bf1374c35", bar_seconds=3600, split_mode="manifest", window=1, hmax=6)

EURUSD_1H_RANGE_SPEC = PopulationSpec(   # lane B vD1h (range family; valid for h = 1 only per lane B), same view and split as variant A
    name="EURUSD_1h_vD1h_range", view_sha256=EURUSD_1H_SPEC.view_sha256, manifest_file_sha256="2985f5891b66a66e524fab2b11561d62bf782af6db678c3605661ec98c04dc57",
    declaration_sha256=None, dataset_id=EURUSD_1H_SPEC.dataset_id, view_path=EURUSD_1H_SPEC.view_path, view_commit=EURUSD_1H_SPEC.view_commit,
    bar_seconds=3600, split_mode="manifest", window=1, hmax=6, derived="range_v1")


def add_range_features(frame):
    """lane B vD1h recipes (bar t only, nothing fitted): ln(H/L); (C-L)/(H-L) with NaN when H == L, filled with the neutral 0.5 and
    COUNTED (a declared deviation: lane B leaves NaN); ln(C/O). Returns the number of filled rows."""
    o, h, l, c = (frame[k].to_numpy(dtype=np.float64) for k in ("OPEN", "HIGH", "LOW", "CLOSE"))
    frame["log_high_low"] = np.log(h / l)
    with np.errstate(invalid="ignore", divide="ignore"):
        loc = (c - l) / (h - l)
    filled = int(np.sum(~np.isfinite(loc)))
    frame["close_location"] = np.where(np.isfinite(loc), loc, 0.5)
    frame["log_close_open"] = np.log(c / o)
    return filled


class PopulationRefusal(ValueError):
    """A binding that does not match its pinned digest is refused with a named code."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_view(path, expected_sha: str | None = VIEW_SHA256) -> pd.DataFrame:
    actual = sha256_file(path)
    if expected_sha is not None and actual != expected_sha:
        raise PopulationRefusal(f"VIEW_SHA256_MISMATCH: {actual} != {expected_sha}")
    frame = pd.read_csv(path)
    if frame.columns[0] != "DATE_TIME":
        raise PopulationRefusal("VIEW_FIRST_COLUMN_NOT_DATE_TIME")
    frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
    frame.attrs["sha256"] = actual
    return frame


def load_manifest(path, expected_sha: str | None = MANIFEST_FILE_SHA256, declaration_sha: str | None = DECLARATION_SHA256) -> dict:
    actual = sha256_file(path)
    if expected_sha is not None and actual != expected_sha:
        raise PopulationRefusal(f"MANIFEST_SHA256_MISMATCH: {actual} != {expected_sha}")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("status") != "FROZEN_DEVELOPMENT":
        raise PopulationRefusal(f"MANIFEST_NOT_FROZEN: {doc.get('status')}")
    if declaration_sha is not None and doc.get("admissible_declaration_sha256") != declaration_sha:
        raise PopulationRefusal("MANIFEST_DECLARATION_SHA_MISMATCH")
    features = list(doc["features"])
    if len(features) != int(doc["feature_count"]):
        raise PopulationRefusal("MANIFEST_FEATURE_COUNT_MISMATCH")
    doc["_file_sha256"] = actual
    return doc


def load_split(path, expected_sha: str | None = SPLIT_FILE_SHA256) -> dict:
    actual = sha256_file(path)
    if expected_sha is not None and actual != expected_sha:
        raise PopulationRefusal(f"SPLIT_SHA256_MISMATCH: {actual} != {expected_sha}")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != "f2.eth_forecast_split.v1":
        raise PopulationRefusal("SPLIT_SCHEMA_UNKNOWN")
    doc["_file_sha256"] = actual
    return doc


def epoch_seconds(frame: pd.DataFrame) -> np.ndarray:
    """UTC epoch seconds of DATE_TIME, independent of the datetime resolution pandas chose (ns, us or s)."""
    return frame["DATE_TIME"].to_numpy().astype("datetime64[s]").astype(np.int64)


def m07_origins(times: np.ndarray, origin_lo: int, origin_hi: int, window: int, hmax: int,
                sample_seconds: int = BAR_SECONDS):
    """M07's `windows_for`: origins whose 24-step input and h_max target support are regular (verbatim rule)."""
    origins, gaps = [], 0
    for o in range(max(origin_lo, window - 1), origin_hi + 1):
        if o + hmax >= len(times):
            break
        span = times[o - window + 1:o + hmax + 1]
        if np.any(np.diff(span) != sample_seconds):
            gaps += 1
            continue
        origins.append(o)
    return np.asarray(origins, dtype=np.int64), gaps


def m07_row_ids_sha256(origins: np.ndarray, times: np.ndarray) -> str:
    return sha256_text("\n".join(f"eth4h:row{o}:{t}" for o, t in zip(origins.tolist(), times[origins].tolist())))


@dataclass
class Population:
    """TRAIN rows of the view with the frozen features, M07's standardization and both target units."""

    frame: pd.DataFrame                      # the full view (validation/test rows are present but never read)
    features: list
    train_rows: tuple                        # [start, end)
    mu: float
    sigma: float
    times: np.ndarray
    origins: np.ndarray                      # M07's scored train origins
    gap_excluded_windows: int
    bindings: dict = field(default_factory=dict)
    bar_seconds: int = BAR_SECONDS

    @property
    def train_end(self) -> int:
        return int(self.train_rows[1])

    def X(self, rows: np.ndarray) -> np.ndarray:
        return self.frame[self.features].to_numpy(dtype=np.float64)[rows]

    def feature_matrix_train(self) -> np.ndarray:
        return self.frame[self.features].to_numpy(dtype=np.float64)[: self.train_end]

    def raw_log_return(self, h: int) -> np.ndarray:
        """log(CLOSE[t+h]/CLOSE[t]) at every TRAIN row whose label lies inside TRAIN at exactly h regular bars."""
        close = self.frame["CLOSE"].to_numpy(dtype=np.float64)
        n = self.train_end
        y = np.full(n, np.nan)
        t = np.arange(n - h)
        regular = (self.times[t + h] - self.times[t]) == h * self.bar_seconds
        y[t[regular]] = np.log(close[t[regular] + h]) - np.log(close[t[regular]])
        return y

    def m07_target(self, h: int) -> np.ndarray:
        """Y_h as M07 defines it: cumulative standardized log_return_1 over (t, t+h], on TRAIN rows; NaN off-support."""
        z = (self.frame["log_return_1"].to_numpy(dtype=np.float64) - self.mu) / self.sigma
        csum = np.concatenate([[0.0], np.cumsum(z[: self.train_end])])
        n = self.train_end
        y = np.full(n, np.nan)
        t = np.arange(n - h)
        regular = (self.times[t + h] - self.times[t]) == h * self.bar_seconds
        tt = t[regular]
        y[tt] = csum[tt + h + 1] - csum[tt + 1]
        return y

    def z_to_log_return(self, y_z: np.ndarray, h: int) -> np.ndarray:
        return self.sigma * y_z + h * self.mu


def bind_population(view_path, manifest_path, split_path=None, *, expect_view=VIEW_SHA256, expect_manifest=MANIFEST_FILE_SHA256,
                    expect_split=SPLIT_FILE_SHA256, spec: PopulationSpec | None = None) -> Population:
    """Bind a population. With `spec=None` this is the ETH 4h binding exactly as before (the expect_* arguments override
    the digests). With a `PopulationSpec` the digests, bar width, and split source come from the spec."""
    if spec is None:
        spec = ETH_4H_SPEC
        digests = dict(view=expect_view, manifest=expect_manifest, split=expect_split, declaration=DECLARATION_SHA256)
    else:
        digests = dict(view=spec.view_sha256, manifest=spec.manifest_file_sha256, split=spec.split_file_sha256, declaration=spec.declaration_sha256)
    frame = load_view(view_path, digests["view"])
    derived_filled = add_range_features(frame) if spec.derived == "range_v1" else None
    manifest = load_manifest(manifest_path, digests["manifest"], digests["declaration"])
    if manifest["resource"]["sha256"] != frame.attrs["sha256"]:
        raise PopulationRefusal("MANIFEST_RESOURCE_SHA_MISMATCH")
    features = list(manifest["features"])
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise PopulationRefusal(f"FEATURES_MISSING_FROM_VIEW: {missing[:5]}")
    times = epoch_seconds(frame)
    if "log_return_1" not in frame.columns:      # derived causally from CLOSE (row t uses rows <= t)
        close = frame["CLOSE"].to_numpy(dtype=np.float64)
        frame["log_return_1"] = np.concatenate([[0.0], np.diff(np.log(close))])
    window, hmax = int(spec.window), int(spec.hmax)
    split = split_target = None
    if spec.split_mode == "m07_split_file":
        split = load_split(split_path, digests["split"])
        if split["source_sha256"] != frame.attrs["sha256"]:
            raise PopulationRefusal("SPLIT_SOURCE_SHA_MISMATCH")
        tr = tuple(int(v) for v in split["declared_split"]["train_rows"])
        if tuple(int(v) for v in manifest["split"]["train"]["rows"]) != tr:
            raise PopulationRefusal("MANIFEST_AND_SPLIT_DISAGREE_ON_TRAIN_ROWS")
        va = [int(v) for v in split["declared_split"]["validation_rows"]]
        te = [int(v) for v in split["declared_split"]["test_rows"]]
        window, hmax = int(split["window"]), int(split["purge_bars"])
        authority, split_sha, split_target = SPLIT_AUTHORITY, split["_file_sha256"], split["target"]
    elif spec.split_mode == "manifest":
        tr = tuple(int(v) for v in manifest["split"]["train"]["rows"])
        va = [int(v) for v in manifest["split"]["validation"]["rows"]]
        te = [int(v) for v in manifest["split"]["test"]["rows"]]
        authority, split_sha = f"frozen manifest {manifest['_file_sha256'][:8]} ({manifest['split'].get('declared_by', '')[:60]})", manifest["_file_sha256"]
    else:
        raise PopulationRefusal(f"UNKNOWN_SPLIT_MODE: {spec.split_mode}")
    lr1 = frame["log_return_1"].to_numpy(dtype=np.float64)[: tr[1]]
    mu, sigma = float(np.mean(lr1)), float(np.std(lr1))  # ddof=0, as M07's StandardScaler
    if split_target is not None:
        for name, ours, theirs in (("mu", mu, split_target["mu"]), ("sigma", sigma, split_target["sigma"])):
            if not np.isclose(ours, float(theirs), rtol=1e-6, atol=1e-12):
                raise PopulationRefusal(f"M07_{name.upper()}_NOT_REPRODUCED: {ours} vs {theirs}")
    if split is not None:
        tsplit = split["splits"]["train"]
        lo, hi = int(tsplit["origin_rows"][0]), int(tsplit["origin_rows"][1])
    else:                                          # origins: declared TRAIN rows whose label support stays inside TRAIN
        lo, hi = 0, tr[1] - 1 - hmax
    origins, gaps = m07_origins(times, lo, hi, window, hmax, spec.bar_seconds)
    if split is not None:
        if len(origins) != int(tsplit["windows"]) or gaps != int(tsplit["gap_excluded_windows"]):
            raise PopulationRefusal(f"M07_TRAIN_ORIGINS_NOT_REPRODUCED: {len(origins)}/{gaps} vs {tsplit['windows']}/{tsplit['gap_excluded_windows']}")
        row_ids_sha = m07_row_ids_sha256(origins, times)
        if row_ids_sha != tsplit["row_ids_sha256"]:
            raise PopulationRefusal("M07_TRAIN_ROW_IDS_SHA_NOT_REPRODUCED")
    else:
        row_ids_sha = sha256_text("\n".join(f"{spec.name}:row{o}:{t}" for o, t in zip(origins.tolist(), times[origins].tolist())))
    bindings = {
        "view": {"path": spec.view_path, "commit": spec.view_commit, "sha256": frame.attrs["sha256"], "rows": int(len(frame)),
                 "dataset_id": spec.dataset_id, "availability_class": spec.availability_class},
        "manifest": {"file_sha256": manifest["_file_sha256"], "canonical_sha256": manifest["manifest_sha256_canonical"],
                     "declaration_sha256": manifest.get("admissible_declaration_sha256"), "variant": manifest["variant"], "features": len(features),
                     "derived": spec.derived, "derived_close_location_filled_rows": derived_filled},
        "split": {"file_sha256": split_sha, "authority": authority, "train_rows": list(tr), "validation_rows": va, "test_rows": te,
                  "test_status": "PROTECTED_NEVER_READ", "window": window, "purge_bars": hmax,
                  "train_origins": [int(origins.min()), int(origins.max())], "train_windows": int(len(origins)),
                  "gap_excluded_windows": int(gaps), "train_row_ids_sha256": row_ids_sha},
        "target": {"definition": "Y_h = sum_{k=1..h} z(log_return_1[t+k]); mu, sigma on TRAIN rows only", "mu": mu, "sigma": sigma, "feature": "log_return_1"},
        "population_spec": spec.name,
    }
    return Population(frame=frame, features=features, train_rows=tr, mu=mu, sigma=sigma, times=times,
                      origins=origins, gap_excluded_windows=gaps, bindings=bindings, bar_seconds=spec.bar_seconds)


# ----------------------------------------------------------------------------------------------- splits inside TRAIN


def contiguous_blocks(origins: np.ndarray, k: int = 5):
    """k contiguous time blocks over the scored origins (equal counts); names B1..Bk."""
    parts = np.array_split(np.asarray(origins), k)
    return [(f"B{i + 1}", int(p[0]), int(p[-1])) for i, p in enumerate(parts)]


def blocked_splits(origins: np.ndarray, h: int, purge: int = 6, k: int = 5):
    """Held-out within TRAIN, blocked by time: evaluate on block i, fit on the other origins outside an embargo of
    `purge + h` rows on each side of the block (labels of fit rows never overlap the block's rows)."""
    origins = np.asarray(origins)
    out = []
    for name, lo, hi in contiguous_blocks(origins, k):
        embargo = purge + h
        fit = origins[(origins < lo - embargo) | (origins > hi + embargo)]
        ev = origins[(origins >= lo) & (origins <= hi)]
        out.append({"name": name, "fit": fit, "eval": ev, "eval_rows": [lo, hi], "embargo": embargo,
                    "protocol": f"blocks{k}"})
    return out


def lane_b_splits(origins: np.ndarray, h: int, train_end: int):
    """Lane B's three expanding inner folds (purge 60 rows), restricted to M07's scored origins and to origins whose
    label stays inside the fold's own rows."""
    origins = np.asarray(origins)
    out = []
    for name, (tr_lo, tr_hi), (va_lo, va_hi) in LANE_B_INNER_FOLDS:
        fit = origins[(origins >= tr_lo) & (origins + h < tr_hi)]
        ev = origins[(origins >= va_lo) & (origins + h < min(va_hi, train_end))]
        out.append({"name": name, "fit": fit, "eval": ev, "eval_rows": [int(va_lo), int(va_hi) - 1], "embargo": 60,
                    "protocol": "laneB"})
    return out


# ----------------------------------------------------------------------------------------------- reference naives


def naive_predictions(pop: Population, y_raw_by_h: dict, h: int, rows: np.ndarray) -> dict:
    """Reference predictions on the identical rows, in raw log-return units:
    zero (price persistence), last-h-bars return persistence, 24 h seasonal (the h-bar return one 24 h period
    earlier; for h > 6 the first multiple of 6 bars that is >= h, so the lagged return is observed at t)."""
    n = pop.train_end
    y = y_raw_by_h[h]
    out = {"naive_zero": np.zeros(len(rows))}
    period = max(1, 86400 // int(pop.bar_seconds))  # bars per 24 h (6 at 4 h, 24 at 1 h)
    seasonal_lag = period * int(np.ceil(h / period))  # 24 h seasonal for h <= period; the first 24 h multiple >= h otherwise
    for name, lag in (("naive_last_return", h), ("naive_seasonal_24h", seasonal_lag)):
        src = rows - lag
        ok = src >= 0
        pred = np.full(len(rows), np.nan)
        pred[ok] = y[src[ok]]
        # the lagged return must itself be fully observed at t: its label row src+h <= t, true for lag >= h
        pred[~np.isfinite(pred)] = 0.0
        out[name] = pred
    return out
