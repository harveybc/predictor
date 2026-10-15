"""I6-DMO single fit: masked normalized loss, same-weight fit/inner stopping, receipt.

Entry point for the automation lane (one subprocess per unit)::

    python -m tools.i6dmo.train --design <design.json> --bundle <frozen_bundle_dir> \
        --out <new_unit_dir> --device {cpu,gpu} [--json]

Exit codes: 0 COMPLETED; 2 CONTRACT_ERROR (design invalid, out dir not empty;
nothing trained, no receipt); 3 FAILED_TYPED (typed disposition, receipt
written); 4 FAILED_UNTYPED (unexpected exception, receipt written). The
receipt is written last, atomically; its presence is the unit terminal.

Training coordinate: ``y_h = (p(t+h) - p(t)) / naive_fit_h`` where
``naive_fit_h`` is the fit-only masked mean ``|p(t+h) - p(t)|``. The per-head
loss is the masked mean absolute error in that coordinate, i.e. exactly
``MAE_h / naive_fit_h``; supported heads are averaged equally. Invalid labels
are removed by boolean selection before any arithmetic, so NaN never enters
the loss or gradients. A zero scale, an unsupported fit head or a constant fit
feature is a typed disposition, never an epsilon. Inputs are z-scored with
fit-only statistics. Early stopping monitors ``0.5 * (fit_loss + inner_loss)``
evaluated at the same end-of-epoch weights; outer VALIDATION is predicted only
after the best weights are restored and never influences any choice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

# Imported FIRST so the static-algorithm determinism env is in place before TensorFlow
# initializes in this interpreter (`python -m tools.i6dmo.train` is the production child;
# the init-order witness in tools.i6dmo.model refuses to score records recorded otherwise).
from tools.i6dmo import model as _model_determinism  # noqa: F401
from tools.i6dmo import design as D

RECEIPT_SCHEMA = "i6dmo.train_receipt.v1"
BUNDLE_SCHEMA = "i6dmo.data_bundle.v1"
SPLITS = ("fit", "inner", "validation")
FINAL_SPLITS = ("fit", "inner")
ARRAYS = ("row_id", "x", "origin_price", "target_price", "mask")
OVERRIDABLE = ("max_epochs", "patience")
CODE_FILES = ("tools/i6dmo/design.py", "tools/i6dmo/model.py", "tools/i6dmo/train.py",
              "predictor_plugins/modular_temporal/layers.py",
              "predictor_plugins/modular_temporal/components.py",
              "predictor_plugins/modular_temporal/assembly.py",
              "predictor_plugins/modular_temporal/registry.py",
              "predictor_plugins/modular_temporal/common.py")
REPO = Path(__file__).resolve().parents[2]


class TrainDisposition(Exception):
    """Typed terminal disposition; message starts with the stable code."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


class ContractError(Exception):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _epoch(text):
    return int(datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())


# --------------------------------------------------------------------- design + bundle

def effective_from_design(path):
    try:
        return D.effective_config(D.load_design(path))
    except D.DesignError as exc:
        raise ContractError("DESIGN_INVALID", str(exc)) from None


def required_splits(effective):
    """The model splits a bundle must carry: no outer validation in a FINAL_REFIT."""
    return FINAL_SPLITS if effective.get("mode") == "FINAL_REFIT" else SPLITS


MARKET_TIME_ARRAYS = ("offsets_hours", "closure_crossed")


def is_market_time(effective):
    return bool((effective.get("design") or {}).get("market_time"))


def market_spacing_hours(effective_or_design):
    """The design's ``spacing_open_market_hours`` — the bound BOTH seams derive (RB05 item 1)."""
    doc = effective_or_design
    block = (doc.get("market_time") if isinstance(doc, dict) else None) or \
            ((doc.get("design") or {}).get("market_time") if isinstance(doc, dict) else None)
    if not block or "spacing_open_market_hours" not in block:
        raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS",
                               "the market-time spacing bound must come from the sealed design")
    return float(block["spacing_open_market_hours"])


def market_time_inputs(arrays, design=None):
    """[observations, offsets_hours, closure_crossed] with the declared conventions enforced.

    Typed ORDINAL_OR_INVALID_OFFSETS (review B-4) when the last offset is not exactly 0, any
    offset is positive, the first adjacent delta is nonzero, the first closure flag is nonzero,
    or the adjacent elapsed hours fall below the DESIGN's ``spacing_open_market_hours`` — the
    same bound the encoder enforces, derived from the design at both seams (RB05 item 1), never
    a fixed floor. Elapsed hours can only exceed the market spacing (closures stretch them).
    """
    minimum = market_spacing_hours(design) if design is not None else None
    offsets = np.asarray(arrays["offsets_hours"], dtype=np.float64)
    closure = np.asarray(arrays["closure_crossed"], dtype=np.float64)
    x = np.asarray(arrays["x"])
    if offsets.ndim != 2 or offsets.shape != closure.shape or len(offsets) != len(x) \
            or offsets.shape[1] != x.shape[1]:
        raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS",
                               f"per-example time inputs must be (n, slots): got {offsets.shape} "
                               f"and {closure.shape} for x {x.shape}")
    if offsets.size:
        if not np.all(offsets[:, -1] == 0):
            raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS",
                                   "the last offset must be exactly 0 (the origin tick)")
        if not np.all(offsets <= 0):
            raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS",
                                   f"offsets must be <= 0; max {float(offsets.max())}")
        deltas = np.diff(offsets, axis=1)
        if not np.all(deltas > 0):
            raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS",
                                   "offsets must increase towards the origin")
        observed = float(deltas.min())
        bound = minimum if minimum is not None else 2.0   # tightest declared grid when unbound
        if observed < bound:
            raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS",
                                   f"adjacent elapsed hours {observed} below the design's "
                                   f"spacing_open_market_hours {bound}: ordinal slot indices or a "
                                   "compressed ladder")
        if not np.all(closure[:, 0] == 0):
            raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS", "the first closure flag must be 0")
        if not np.all(np.isin(closure, (0.0, 1.0))):
            raise TrainDisposition("ORDINAL_OR_INVALID_OFFSETS", "closure flags must be 0/1")
    return [x, offsets.astype(np.float32), closure.astype(np.float32)]


def load_market_split_arrays(bundle_dir, split, design_path):
    """One authenticated market split, as the trainer consumes it."""
    from tools.i6dmo import market_grid
    market_design = market_grid.market_design(_elapsed_view(design_path))
    try:
        arrays = market_grid.load_market_split(bundle_dir, split, design=market_design)
    except Exception as exc:  # DataContractError carries a typed code
        raise TrainDisposition(getattr(exc, "code", "MARKET_BUNDLE_REFUSED"), str(exc)[:400]) from None
    return arrays


def _elapsed_view(design_path):
    """The data lane's market design is derived from the elapsed design; rebuild that view."""
    from tools.i6dmo import data as data_mod
    sealed = D.load_design(design_path) if not isinstance(design_path, dict) else design_path
    base = json.loads(_canonical(sealed))
    elapsed = json.loads((Path(__file__).resolve().parents[2]
                          / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json")
                         .read_text(encoding="utf-8"))
    if D.family(base).startswith("long") and \
            base["horizons_hours"] == D.LONG_HORIZONS:
        return data_mod.long_data_view(elapsed)
    return elapsed


def _load_market_bundle(bundle_dir, effective, design_path):
    """Authenticate a market bundle and apply the same model-side boundary checks."""
    from tools.i6dmo import market_grid
    root = Path(bundle_dir)
    # Read the declared schema BEFORE any body verification so a DS2/ASOF bundle under a
    # market-time design is refused as a schema mismatch, not as a digest error.
    try:
        declared = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainDisposition("BUNDLE_UNREADABLE", str(exc)[:300]) from None
    if declared.get("schema") != market_grid.BUNDLE_SCHEMA:
        raise TrainDisposition("BUNDLE_SCHEMA",
                               f"a market-time design requires {market_grid.BUNDLE_SCHEMA}, got "
                               f"{declared.get('schema')!r}")
    try:
        manifest = market_grid.load_market_manifest(root)
    except Exception as exc:
        raise TrainDisposition(getattr(exc, "code", "BUNDLE_UNREADABLE"), str(exc)[:300]) from None
    expected_rule = effective["design"]["market_time"]["rule"]
    if (manifest.get("input_sampling") or {}).get("rule") != expected_rule:
        raise TrainDisposition("MARKET_RULE_MISMATCH",
                               f"bundle rule {(manifest.get('input_sampling') or {}).get('rule')!r} "
                               f"!= design {expected_rule!r}")
    identity = D.check_feature_identity(manifest["features"])
    if manifest.get("feature_manifest_sha256") != identity["manifest_sha256"]:
        raise TrainDisposition("FEATURE_IDENTITY_MISMATCH", "declared feature manifest differs")
    if list(manifest.get("horizons_hours") or []) != effective["horizons_hours"]:
        raise TrainDisposition("BUNDLE_HORIZONS", "bundle horizons differ from the design")
    slots = effective["design"]["market_time"]["observations"]
    test_start = _epoch(effective["splits"]["test"]["start"])
    horizons = np.asarray(effective["horizons_hours"], dtype=np.int64) * 3600
    out = {"kind": "market", "manifest": manifest, "feature_identity": identity,
           "manifest_sha256": file_sha256(root / "BUNDLE.json")}
    for name in SPLITS:
        if name in manifest.get("splits", {}):
            arrays = load_market_split_arrays(root, name, design_path)
        else:
            raise TrainDisposition("BUNDLE_SCHEMA", f"market bundle lacks the {name} split")
        rows = len(arrays["row_id"])
        if rows < 1 or arrays["x"].shape != (rows, slots, len(identity["features"])):
            raise TrainDisposition("BUNDLE_SHAPE", f"{name}: x must be (n, {slots}, 19)")
        for key in MARKET_TIME_ARRAYS:
            if key not in arrays or np.asarray(arrays[key]).shape != (rows, slots):
                raise TrainDisposition("BUNDLE_SCHEMA", f"{name}: {key} must be (n, {slots})")
        origin = np.asarray(arrays["row_id"], dtype=np.int64)
        if (origin >= test_start).any():
            raise TrainDisposition("TEST_FORBIDDEN", f"{name}: origin at or after the TEST start")
        start, end = (_epoch(effective["splits"][name][k]) for k in ("start", "end"))
        if ((origin < start) | (origin >= end)).any():
            raise TrainDisposition("SPLIT_BOUNDARY_VIOLATION", f"{name}: origin outside its split")
        mask = np.asarray(arrays["mask"], dtype=bool)
        crossing = mask & (origin[:, None] + horizons[None, :] >= end)
        if crossing.any():
            code = "PURGE_VIOLATION" if name == "fit" else "SPLIT_BOUNDARY_VIOLATION"
            raise TrainDisposition(code, f"{name}: {int(crossing.sum())} valid labels reach {end}")
        if not np.isfinite(arrays["x"]).all() or not np.isfinite(arrays["origin_price"]).all():
            raise TrainDisposition("NONFINITE_INPUT", name)
        if not np.isfinite(np.asarray(arrays["target_price"])[mask]).all():
            raise TrainDisposition("NONFINITE_LABEL", f"{name}: a valid label is not finite")
        kept = {"_spacing_open_market_hours": market_spacing_hours(effective),
                "row_id": origin, "x": np.asarray(arrays["x"], dtype=np.float64),
                "origin_price": np.asarray(arrays["origin_price"], dtype=np.float64),
                "target_price": np.asarray(arrays["target_price"], dtype=np.float64), "mask": mask,
                "offsets_hours": np.asarray(arrays["offsets_hours"], dtype=np.float64),
                "closure_crossed": np.asarray(arrays["closure_crossed"], dtype=np.float64)}
        market_time_inputs(kept, design=effective)   # conventions + the design's spacing bound
        out[name] = kept
    return out


def load_data_bundle(bundle_dir, effective, design_path=None):
    """Authenticate the frozen data bundle and run the model-side boundary checks."""
    if is_market_time(effective):
        return _load_market_bundle(bundle_dir, effective, design_path or effective["design"])
    expected = required_splits(effective)
    root = Path(bundle_dir)
    try:
        manifest = json.loads((root / "BUNDLE.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainDisposition("BUNDLE_UNREADABLE", str(exc)) from None
    if manifest.get("schema") == "i6dmo.market_bundle.v1":
        raise TrainDisposition("BUNDLE_SCHEMA",
                               "a market-time bundle is never adopted by an elapsed design")
    required = {"schema", "features", "feature_manifest_sha256", "horizons_hours", "offsets_hours", "splits"}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise TrainDisposition("BUNDLE_SCHEMA", f"BUNDLE.json requires {sorted(required)}")
    if manifest["schema"] != BUNDLE_SCHEMA:
        raise TrainDisposition("BUNDLE_SCHEMA",
                               f"schema must be {BUNDLE_SCHEMA}; a market-time bundle is never "
                               "adopted by an elapsed design (got "
                               f"{manifest['schema']!r})")
    splits = manifest["splits"]
    if not isinstance(splits, dict):
        raise TrainDisposition("BUNDLE_SCHEMA", "splits must be an object")
    if any("test" in str(name).lower() for name in splits):
        raise TrainDisposition("TEST_FORBIDDEN", "a TEST split is never read by a fit; the guarded "
                               "final module alone consumes TEST bundles, without training")
    if set(splits) != set(expected):
        raise TrainDisposition("BUNDLE_SCHEMA", f"splits must be exactly {list(expected)}")
    try:
        identity = D.check_feature_identity(manifest["features"])
    except D.DesignError as exc:
        raise TrainDisposition(exc.code, str(exc)) from None
    if manifest["feature_manifest_sha256"] != identity["manifest_sha256"]:
        raise TrainDisposition("FEATURE_IDENTITY_MISMATCH", "declared feature_manifest_sha256 differs")
    if manifest["horizons_hours"] != effective["horizons_hours"]:
        raise TrainDisposition("BUNDLE_HORIZONS", "bundle horizons differ from the design")
    if manifest["offsets_hours"] != effective["offsets_hours"]:
        raise TrainDisposition("BUNDLE_OFFSETS", "bundle input offsets differ from the design")
    test_start = _epoch(effective["splits"]["test"]["start"])
    horizons = np.asarray(effective["horizons_hours"], dtype=np.int64) * 3600
    steps, n_features = len(effective["offsets_hours"]), len(identity["features"])
    out = {"manifest": manifest, "manifest_sha256": file_sha256(root / "BUNDLE.json"),
           "feature_identity": identity}
    for name in expected:
        entry = splits[name]
        path = root / entry["path"]
        if Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
            raise TrainDisposition("BUNDLE_SCHEMA", f"{name} path must be relative inside the bundle")
        if file_sha256(path) != entry["sha256"]:
            raise TrainDisposition("BUNDLE_DIGEST_MISMATCH", name)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAYS):
                raise TrainDisposition("BUNDLE_SCHEMA", f"{name} arrays must be exactly {list(ARRAYS)}")
            arrays = {k: archive[k] for k in ARRAYS}
        rows = len(arrays["row_id"])
        shapes = {"row_id": (rows,), "x": (rows, steps, n_features), "origin_price": (rows,),
                  "target_price": (rows, len(horizons)), "mask": (rows, len(horizons))}
        if rows < 1 or any(arrays[k].shape != s for k, s in shapes.items()):
            raise TrainDisposition("BUNDLE_SHAPE", f"{name}: expected {shapes}")
        if arrays["row_id"].dtype.kind not in "iu" or len(np.unique(arrays["row_id"])) != rows:
            raise TrainDisposition("BUNDLE_ROW_ID", f"{name}: row_id must be unique integers")
        if arrays["mask"].dtype != np.bool_:
            raise TrainDisposition("BUNDLE_SCHEMA", f"{name}: mask must be boolean")
        origin = arrays["row_id"].astype(np.int64)
        if (origin >= test_start).any():
            raise TrainDisposition("TEST_FORBIDDEN", f"{name}: origin at or after the TEST start")
        start, end = (_epoch(effective["splits"][name][k]) for k in ("start", "end"))
        if ((origin < start) | (origin >= end)).any():
            raise TrainDisposition("SPLIT_BOUNDARY_VIOLATION", f"{name}: origin outside [{start}, {end})")
        label_time = origin[:, None] + horizons[None, :]
        crossing = arrays["mask"] & (label_time >= end)
        if crossing.any():
            code = "PURGE_VIOLATION" if name == "fit" else "SPLIT_BOUNDARY_VIOLATION"
            raise TrainDisposition(code, f"{name}: {int(crossing.sum())} valid labels reach {end}")
        if not np.isfinite(arrays["x"]).all():
            raise TrainDisposition("NONFINITE_INPUT", name)
        if not np.isfinite(arrays["origin_price"]).all():
            raise TrainDisposition("NONFINITE_INPUT", f"{name}: origin_price")
        if not np.isfinite(arrays["target_price"][arrays["mask"]]).all():
            raise TrainDisposition("NONFINITE_LABEL", f"{name}: a valid label is not finite")
        out[name] = arrays
    return out


# --------------------------------------------------------------------- matched common population
#
# NEXT_CONTRASTS_PLAN section 2 (SC05/SC06): both matched arms train, monitor and score on the
# FROZEN common population owned by the data lane (tools/i6dmo/common_population.py,
# INTERFACE_COMMON_POPULATION.json): per-split ordered origin row ids, per-head masks, ONE shared
# FIT-only scaler and ONE shared FIT-only naive loss scale per horizon, read through the
# per-arm accessor ``arm_tensors(pop, sealed_design, split)``. The model side never refits
# statistics per arm, never drops or adds rows and never substitutes another mask.

COMMON_POPULATION_SCHEMA = "i6dmo.common_population.v1"
MATCHED_SCHEMA = "i6dmo.matched_fit.v1"
#: Population split -> trainer split key (DEVELOPMENT 2024 is the trainer's "validation" split).
POPULATION_SPLITS = {"FIT": "fit", "INNER": "inner", "DEVELOPMENT": "validation"}
#: Sealed effective identities of the CLOSED full-population MG2/MG12 campaign (defence in depth;
#: design.matched_arm already refuses their families as MATCHED_IDENTITY_REQUIRED).
CLOSED_FULL_POPULATION_EFFECTIVE = {
    "920332aed3997b56a0fce19ced25c2d48abeff77ebb227ce6cff368c311ad768": "MG2",
    "0853f6d95bc3b70166a469011ba7a8d7f92e740089378e355bb6692c3810dd2f": "MG12",
}
_NAIVE_REL_TOL = 1e-12


def _common_population_module():
    """The data lane's module; its absence is a typed integration state, never a fallback."""
    import importlib
    try:
        return importlib.import_module("tools.i6dmo.common_population")
    except ModuleNotFoundError as exc:
        if exc.name != "tools.i6dmo.common_population":
            raise
        raise TrainDisposition("COMMON_POPULATION_MODULE_MISSING",
                               "tools/i6dmo/common_population.py (data lane) is not installed") from None


def matched_arm(effective_or_design):
    """``mg1``/``mg2m`` through the data lane's typed guard; anything else is refused."""
    doc = effective_or_design.get("design", effective_or_design) \
        if "effective_config_sha256" in effective_or_design else effective_or_design
    if effective_or_design.get("effective_config_sha256") in CLOSED_FULL_POPULATION_EFFECTIVE:
        raise TrainDisposition("MATCHED_IDENTITY_REQUIRED",
                               "the closed full-population identity is never a matched arm")
    try:
        return D.matched_arm(doc)
    except D.DesignError as exc:
        raise TrainDisposition(exc.code, str(exc)[:400]) from None


def _ids_digest(ids):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes()).hexdigest()


def _array_digest(value):
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(_canonical([list(value.shape), str(value.dtype)]).encode())
    digest.update(value.tobytes() if value.dtype.kind != "U" else
                  "\x00".join(value.reshape(-1).tolist()).encode())
    return digest.hexdigest()


def validate_population(population, horizons, features):
    """Model-side defensive validation of the frozen common population (typed, never repaired)."""
    def bad(detail):
        raise TrainDisposition("COMMON_POPULATION_INVALID", detail)

    if getattr(population, "schema", None) != COMMON_POPULATION_SCHEMA:
        bad(f"schema {getattr(population, 'schema', None)!r} != {COMMON_POPULATION_SCHEMA}")
    if tuple(int(h) for h in population.horizons) != tuple(int(h) for h in horizons):
        raise TrainDisposition("COMMON_POPULATION_HORIZONS",
                               f"population horizons {tuple(population.horizons)} != design {tuple(horizons)}")
    if set(population.splits) != set(POPULATION_SPLITS):
        bad(f"splits {sorted(population.splits)} != {sorted(POPULATION_SPLITS)}")
    width = len(horizons)
    for name in POPULATION_SPLITS:
        sp = population.splits[name]
        if sp.name != name:
            bad(f"split {name} carries name {sp.name!r}")
        ids = np.asarray(sp.origin_row_ids)
        if ids.dtype != np.int64 or ids.ndim != 1 or len(ids) < 1:
            bad(f"{name}: origin_row_ids must be a nonempty 1-D int64 array")
        if len(ids) > 1 and not np.all(np.diff(ids) > 0):
            bad(f"{name}: origin_row_ids must be strictly increasing (ordered, unique)")
        for key in ("head_mask", "entry_mask", "management_mask"):
            m = np.asarray(getattr(sp, key))
            if m.dtype != np.bool_ or m.shape != (len(ids), width):
                bad(f"{name}: {key} must be bool ({len(ids)}, {width}), got {m.dtype} {m.shape}")
        head = np.asarray(sp.head_mask)
        if not head.any(axis=1).all():
            raise TrainDisposition("COMMON_ROW_WITHOUT_HEAD",
                                   f"{name}: {int((~head.any(axis=1)).sum())} retained origins "
                                   "have no common valid head")
        if (np.asarray(sp.entry_mask) & ~head).any() or (np.asarray(sp.management_mask) & ~head).any():
            bad(f"{name}: entry/management masks must be subsets of the head mask")
        counts = {int(k): int(v) for k, v in dict(sp.counts).items()}
        if counts != {int(h): int(head[:, j].sum()) for j, h in enumerate(horizons)}:
            bad(f"{name}: per-head counts {counts} disagree with the head mask")
    scaler = population.scaler
    if list(scaler.feature_names) != list(features):
        raise TrainDisposition("SCALER_FEATURE_MISMATCH",
                               "shared scaler features differ from the sealed feature manifest")
    mean, std = np.asarray(scaler.mean), np.asarray(scaler.std)
    if mean.dtype != np.float64 or std.dtype != np.float64 or mean.shape != (len(features),) \
            or std.shape != (len(features),):
        bad("shared scaler mean/std must be float64 per feature")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        bad("shared scaler statistics must be finite")
    if not (std > 0).all():
        raise TrainDisposition("CONSTANT_FEATURE_IN_FIT",
                               f"shared scaler std <= 0 at {np.flatnonzero(~(std > 0)).tolist()}")
    if not isinstance(scaler.n_rows, (int, np.integer)) or int(scaler.n_rows) < 1 or \
            not isinstance(scaler.fit_row_digest, str):
        bad("shared scaler must record n_rows and fit_row_digest")
    scale = {int(k): v for k, v in dict(population.naive_loss_scale).items()}
    if set(scale) != {int(h) for h in horizons}:
        bad(f"naive_loss_scale horizons {sorted(scale)} != {list(horizons)}")
    unsupported = [h for h, v in sorted(scale.items()) if v is None]
    if unsupported:
        raise TrainDisposition("HEAD_UNSUPPORTED_IN_FIT",
                               f"shared naive loss scale typed unsupported for heads {unsupported}")
    if not all(isinstance(v, float) and math.isfinite(v) and v > 0 for v in scale.values()):
        raise TrainDisposition("ZERO_NAIVE_SCALE", f"shared naive loss scale {scale}")
    return population


def load_matched_population(path, expected_digest, horizons, features, input_dir=None):
    """``load_common_population`` (data lane, digest-verified) plus the model-side validation."""
    cp = _common_population_module()
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise TrainDisposition("COMMON_POPULATION_DIGEST", "an expected 64-hex digest is required")
    try:
        population = cp.load_common_population(path, expected_digest=expected_digest,
                                               input_dir=input_dir)
    except TrainDisposition:
        raise
    except Exception as exc:  # the data lane's typed errors keep their code
        raise TrainDisposition(getattr(exc, "code", None) or "COMMON_POPULATION_REFUSED",
                               str(exc)[:400]) from None
    if getattr(population, "digest", None) != expected_digest:
        raise TrainDisposition("COMMON_POPULATION_DIGEST",
                               f"loaded digest {getattr(population, 'digest', None)} != expected")
    return validate_population(population, horizons, features)


def arm_split(population, design, split):
    """The data lane's per-arm accessor for one population split, typed on refusal."""
    cp = _common_population_module()
    try:
        return cp.arm_tensors(population, design, split)
    except Exception as exc:
        raise TrainDisposition(getattr(exc, "code", None) or "ARM_TENSORS_REFUSED",
                               str(exc)[:400]) from None


def _boundary_checks(key, origin, mask, effective):
    test_start = _epoch(effective["splits"]["test"]["start"])
    if (origin >= test_start).any():
        raise TrainDisposition("TEST_FORBIDDEN", f"{key}: common origin at or after the TEST start")
    start, end = (_epoch(effective["splits"][key][k]) for k in ("start", "end"))
    if ((origin < start) | (origin >= end)).any():
        raise TrainDisposition("SPLIT_BOUNDARY_VIOLATION", f"{key}: common origin outside its split")
    horizons = np.asarray(effective["horizons_hours"], dtype=np.int64) * 3600
    crossing = mask & (origin[:, None] + horizons[None, :] >= end)
    if crossing.any():
        raise TrainDisposition("PURGE_VIOLATION" if key == "fit" else "SPLIT_BOUNDARY_VIOLATION",
                               f"{key}: {int(crossing.sum())} common labels reach {end}")


def common_split_arrays(tensors, split_population, effective, key):
    """Trainer arrays for EXACTLY the frozen common rows of one split, re-verified model-side.

    The accessor's rows must be the population's ordered origins and its masks the common
    masks (COMMON_POPULATION_ARM_MISMATCH otherwise); labels outside the common head mask are
    NaN by construction and kept NaN so they can never reach a loss, a monitor or a score;
    offsets must be the ACTUAL signed UTC offsets of the slot instants (OFFSETS_NOT_ACTUAL_UTC);
    the design's spacing bound and the split/purge/TEST boundaries are enforced again.
    """
    want = np.asarray(split_population.origin_row_ids, dtype=np.int64)
    common = np.asarray(split_population.head_mask, dtype=bool)
    if not np.array_equal(np.asarray(tensors.row_id, np.int64), want):
        raise TrainDisposition("COMMON_POPULATION_ARM_MISMATCH", f"{key}: arm rows are not the common rows")
    for name in ("head_mask", "entry_mask", "management_mask"):
        if not np.array_equal(np.asarray(getattr(tensors, name), bool),
                              np.asarray(getattr(split_population, name), bool)):
            raise TrainDisposition("COMMON_POPULATION_ARM_MISMATCH", f"{key}: arm {name} differs")
    slots = effective["design"]["market_time"]["observations"]
    n_features = len(effective["feature_manifest"]["features"])
    x = np.asarray(tensors.x, dtype=np.float64)
    if x.shape != (len(want), slots, n_features):
        raise TrainDisposition("BUNDLE_SHAPE", f"{key}: x must be (n, {slots}, {n_features}), got {x.shape}")
    target = np.asarray(tensors.target_price, dtype=np.float64)
    if target.shape != common.shape:
        raise TrainDisposition("BUNDLE_SHAPE", f"{key}: target_price must be {common.shape}")
    target = np.where(common, target, np.nan)
    kept = {"_spacing_open_market_hours": market_spacing_hours(effective), "row_id": want.copy(),
            "x": x, "origin_price": np.asarray(tensors.origin_price, dtype=np.float64),
            "target_price": target, "mask": common.copy(),
            "offsets_hours": np.asarray(tensors.offsets_hours, dtype=np.float64),
            "closure_crossed": np.asarray(tensors.closure_crossed, dtype=np.float64),
            "slot_instants": np.asarray(tensors.slot_instants, dtype=np.int64),
            "entry_mask": np.asarray(split_population.entry_mask, dtype=bool).copy(),
            "week_ids": np.asarray(tensors.week_ids)}
    if not np.isfinite(kept["x"]).all() or not np.isfinite(kept["origin_price"]).all():
        raise TrainDisposition("NONFINITE_INPUT", key)
    if not np.isfinite(target[common]).all():
        raise TrainDisposition("NONFINITE_LABEL", f"{key}: a common valid label is not finite")
    actual = (kept["slot_instants"] - want[:, None]).astype(np.float64) / 3600.0
    if actual.shape != kept["offsets_hours"].shape or not np.array_equal(actual, kept["offsets_hours"]):
        raise TrainDisposition("OFFSETS_NOT_ACTUAL_UTC",
                               f"{key}: offsets_hours differ from (slot instant - t) in hours")
    _boundary_checks(key, want, common, effective)
    market_time_inputs(kept, design=effective)
    return kept


def load_matched_data(population, effective):
    """Per-split arm arrays on the frozen common population, via the data lane's accessor."""
    out = {}
    for name, key in POPULATION_SPLITS.items():
        out[key] = common_split_arrays(arm_split(population, effective["design"], name),
                                       population.splits[name], effective, key)
    digests = {key: {k: _array_digest(v) for k, v in sorted(out[key].items())
                     if isinstance(v, np.ndarray)} for key in POPULATION_SPLITS.values()}
    out["source"] = {"kind": "common_population.arm_tensors",
                     "arm_arrays_sha256": hashlib.sha256(_canonical(digests).encode()).hexdigest()}
    return out


def matched_scales(population, effective, fit):
    """Shared FIT-only statistics from the population; no per-arm refit of anything.

    Input z-score statistics come from ``population.scaler`` (union of both arms' input-source
    bars for the common FIT origins) and the loss scale per horizon from
    ``population.naive_loss_scale`` (FIT-only naive on the common FIT mask). The arm's exact FIT
    prices on the common FIT mask must reproduce that naive (NAIVE_SCALE_MISMATCH otherwise).
    """
    horizons = effective["horizons_hours"]
    naive = [float(dict(population.naive_loss_scale)[h]) for h in horizons]
    mask = fit["mask"]
    recomputed = []
    for j, h in enumerate(horizons):
        sel = mask[:, j]
        value = math.fsum(np.abs(fit["target_price"][sel, j] - fit["origin_price"][sel]).tolist()) / int(sel.sum())
        recomputed.append(value)
        if abs(value - naive[j]) > _NAIVE_REL_TOL * max(1.0, abs(naive[j])):
            raise TrainDisposition("NAIVE_SCALE_MISMATCH",
                                   f"h{h}: arm FIT naive {value!r} != shared {naive[j]!r}")
    scaler = population.scaler
    scales = {"schema": "i6dmo.scales.v1", "source_split": "fit",
              "source": "common_population.shared_fit_only",
              "rows": int(len(mask)), "input_mean": np.asarray(scaler.mean, np.float64).tolist(),
              "input_std": np.asarray(scaler.std, np.float64).tolist(), "naive_mae": naive,
              "counts": mask.sum(axis=0).astype(int).tolist(),
              "scaler": {"fit_row_digest": scaler.fit_row_digest, "n_rows": int(scaler.n_rows),
                         "feature_names": list(scaler.feature_names)},
              "common_population_sha256": population.digest}
    scales["scales_sha256"] = hashlib.sha256(_canonical(scales).encode()).hexdigest()
    return scales, recomputed


def verify_matched_identity(identity, population_digest, arm=None):
    """Refuse any fit identity that was not trained on THIS frozen common population.

    A cached full-population MG2 (or any other) fit carries no matched identity, or another
    population digest, and is typed NOT_A_MATCHED_FIT; it can only ever be a descriptive
    subset score (REUSED_MODEL_SUBSET_SCORE), never a matched arm.
    """
    identity = identity or {}
    if identity.get("effective_config_sha256") in CLOSED_FULL_POPULATION_EFFECTIVE:
        raise TrainDisposition("NOT_A_MATCHED_FIT", "the closed full-population identity is "
                               "never a matched arm")
    if identity.get("common_population_sha256") != population_digest:
        raise TrainDisposition("NOT_A_MATCHED_FIT",
                               "the fit identity does not bind this common population "
                               f"(got {identity.get('common_population_sha256')!r}); a cached "
                               "full-population fit is never a matched arm")
    if arm is not None and identity.get("matched_arm") != arm:
        raise TrainDisposition("NOT_A_MATCHED_FIT",
                               f"fit arm {identity.get('matched_arm')!r} != expected {arm!r}")
    return identity


def _prepare_matched(common_population, effective):
    arm = matched_arm(effective)
    if os.environ.get("I6DMO_TEST_MAX_ROWS"):
        raise TrainDisposition("MATCHED_SUBSET_FORBIDDEN",
                               "a row subset would break the frozen common population")
    features = effective["feature_manifest"]["features"]
    population = load_matched_population(common_population["path"],
                                         common_population["expected_digest"],
                                         effective["horizons_hours"], features,
                                         input_dir=common_population.get("input_dir"))
    data = load_matched_data(population, effective)
    scales, recomputed = matched_scales(population, effective, data["fit"])
    splits = {}
    for name, key in POPULATION_SPLITS.items():
        sp = population.splits[name]
        splits[name] = {"trainer_split": key, "rows": int(len(sp.origin_row_ids)),
                        "row_ids_sha256": _ids_digest(sp.origin_row_ids),
                        "head_mask_sha256": _array_digest(np.asarray(sp.head_mask, bool)),
                        "counts": {str(h): int(c) for h, c in sorted(dict(sp.counts).items())},
                        "population_split_digest": sp.digest}
    sealed = ((population.identities or {}).get("arms") or {}).get(arm) or {}
    matched = {"schema": MATCHED_SCHEMA, "arm": arm,
               "rule": effective["design"]["market_time"]["rule"],
               "slots": effective["design"]["market_time"]["observations"],
               "common_population_sha256": population.digest,
               "population_schema": population.schema,
               "population_source_verified": common_population.get("input_dir") is not None,
               "population_arm_identity": {k: sealed.get(k) for k in
                                           ("experiment_id", "experiment_design_sha256",
                                            "experiment_seal_sha256", "data_identity_sha256")},
               "splits": splits,
               "scaler": dict(scales["scaler"], source="common_population (shared, FIT-only)"),
               "naive_loss_scale": {str(h): float(dict(population.naive_loss_scale)[h])
                                    for h in effective["horizons_hours"]},
               "naive_arm_cross_check": {str(h): v for h, v in
                                         zip(effective["horizons_hours"], recomputed)},
               "arm_source": data.pop("source"),
               "loss": "equal-head mean of per-head masked MAE in delta/naive_loss_scale "
                       "coordinates on the COMMON masks",
               "monitor": "0.5*(FIT+INNER) on the common masks at the same end-of-epoch weights; "
                          "DEVELOPMENT is predicted only after restore and never selects"}
    return data, scales, matched


# --------------------------------------------------------------------- scales and coordinates

def trainable_rows(arrays):
    """Rows with at least one valid head; the others are counted, not trained."""
    keep = arrays["mask"].any(axis=1)
    kept = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(keep) else v)
            for k, v in arrays.items() if isinstance(v, (np.ndarray, int, float))}
    return kept, int((~keep).sum())


def fit_scales(fit):
    """Fit-only input z-score statistics and per-head naive MAE."""
    x = np.asarray(fit["x"], dtype=np.float64).reshape(-1, fit["x"].shape[-1])
    mean, std = x.mean(axis=0), x.std(axis=0)
    if not (np.isfinite(std).all() and (std > 0).all()):
        bad = np.flatnonzero(~(std > 0)).tolist()
        raise TrainDisposition("CONSTANT_FEATURE_IN_FIT", f"feature indices {bad}")
    mask = fit["mask"]
    counts = mask.sum(axis=0)
    if (counts == 0).any():
        raise TrainDisposition("HEAD_UNSUPPORTED_IN_FIT", f"head indices {np.flatnonzero(counts == 0).tolist()}")
    naive = []
    for h in range(mask.shape[1]):
        sel = mask[:, h]
        delta = np.abs(fit["target_price"][sel, h].astype(np.float64) - fit["origin_price"][sel].astype(np.float64))
        naive.append(math.fsum(delta.tolist()) / int(sel.sum()))
    if not all(math.isfinite(v) and v > 0 for v in naive):
        raise TrainDisposition("ZERO_NAIVE_SCALE", f"fit naive MAE per head {naive}")
    scales = {"schema": "i6dmo.scales.v1", "source_split": "fit", "rows": int(len(mask)),
              "input_mean": mean.tolist(), "input_std": std.tolist(), "naive_mae": naive,
              "counts": counts.astype(int).tolist()}
    scales["scales_sha256"] = hashlib.sha256(_canonical(scales).encode()).hexdigest()
    return scales


def transform(arrays, scales):
    """(x float32 z-scored, y in the training coordinate with NaN where invalid, mask)."""
    mean = np.asarray(scales["input_mean"], dtype=np.float64)
    std = np.asarray(scales["input_std"], dtype=np.float64)
    naive = np.asarray(scales["naive_mae"], dtype=np.float64)
    x = ((np.asarray(arrays["x"], dtype=np.float64) - mean) / std).astype(np.float32)
    mask = np.asarray(arrays["mask"], dtype=bool)
    delta = np.where(mask, arrays["target_price"] - arrays["origin_price"][:, None], np.nan)
    with np.errstate(invalid="ignore"):
        y = np.where(mask, delta / naive[None, :], np.nan)
    return x, y, mask


def to_price(pred, arrays, scales):
    """Inverse of the training coordinate: original USD per EUR."""
    naive = np.asarray(scales["naive_mae"], dtype=np.float64)
    return np.asarray(arrays["origin_price"], dtype=np.float64)[:, None] + np.asarray(pred, np.float64) * naive


# --------------------------------------------------------------------- loss and metrics

def masked_normalized_loss(y, pred, mask):
    """Equal-weight mean over supported heads of each head's masked MAE (TF).

    Invalid labels are removed with ``tf.boolean_mask`` before subtraction;
    heads without a valid label in this batch are excluded, not counted as 0.
    Returns ``(loss, per_head_sums, per_head_counts)``.
    """
    import tensorflow as tf

    mask = tf.cast(mask, tf.bool)
    sums, counts = [], []
    for h in range(pred.shape[1]):
        m = mask[:, h]
        p = tf.boolean_mask(pred[:, h], m)
        t = tf.boolean_mask(tf.cast(y[:, h], pred.dtype), m)
        sums.append(tf.reduce_sum(tf.abs(p - t)))
        counts.append(tf.reduce_sum(tf.cast(m, pred.dtype)))
    sums, counts = tf.stack(sums), tf.stack(counts)
    active = counts > 0
    means = tf.where(active, tf.math.divide_no_nan(sums, counts), tf.zeros_like(sums))
    loss = tf.reduce_sum(means) / tf.reduce_sum(tf.cast(active, pred.dtype))
    return loss, sums, counts


def masked_split_metrics(y, pred, mask):
    """Frozen-weight split loss from global per-head sums/counts (float64, fsum)."""
    y, pred, mask = np.asarray(y, np.float64), np.asarray(pred, np.float64), np.asarray(mask, bool)
    if not np.isfinite(y[mask]).all():
        raise TrainDisposition("NONFINITE_LABEL", "a valid label is not finite")
    if not np.isfinite(pred[mask]).all():
        raise TrainDisposition("NONFINITE_PREDICTION", "a prediction on a valid label is not finite")
    sums, counts, per_head = [], [], []
    for h in range(y.shape[1]):
        sel = mask[:, h]
        s = math.fsum(np.abs(pred[sel, h] - y[sel, h]).tolist())
        sums.append(s)
        counts.append(int(sel.sum()))
        per_head.append(s / counts[-1] if counts[-1] else None)
    active = [v for v in per_head if v is not None]
    if not active:
        raise TrainDisposition("NO_SUPPORTED_HEAD", "split has no valid label")
    loss = math.fsum(active) / len(active)
    return {"sums": sums, "counts": counts, "per_head": per_head, "loss": loss,
            "active_heads": len(active)}


MONITOR_EVAL = "i6dmo.monitor_eval.v2"
MONITOR_PARTITION = "sequential_row_id_fixed"
MONITOR_INSTANCE = "canonical_loader_rebuild"
#: THE canonical monitor-input materialization, shared by the fit path and the replay: the
#: bundle arrays are the source of truth, normalization happens in float64 on CPU (numpy),
#: and there is exactly ONE cast to float32, in materialize_monitor_inputs, before any
#: forward. Any other derivation (e.g., a float32 pipeline) yields different bytes and is
#: refused by the input witness.
MONITOR_INPUT = "i6dmo.monitor_input.v1:float64_cpu_normalize_single_float32_cast"
MONITOR_FORWARD = "i6dmo.monitor_forward.v1"


def _parts(x):
    """Model inputs as a list (market-time models take three tensors)."""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _rows(x):
    return len(_parts(x)[0])


def _take(x, idx):
    """Index every model input consistently (a list stays a list)."""
    if isinstance(x, (list, tuple)):
        return [np.asarray(part)[idx] for part in x]
    return np.asarray(x)[idx]


def _as_float32(x):
    if isinstance(x, (list, tuple)):
        return [np.ascontiguousarray(np.asarray(part, dtype=np.float32)) for part in x]
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def _input_bytes(x):
    """The exact bytes fed to the forward, across every input tensor."""
    return b"".join(np.ascontiguousarray(part).tobytes() for part in _parts(x))


def monitor_forward(model, x32, batch_size):
    """THE monitor's one dedicated forward (fit-time recording and replay call exactly this).

    Pure eager calls on the canonical loader instance over the canonical sequential
    partition — created independently of any training step, so no training-side traced
    graph or grappler-remapped function cluster can be what the monitor measures.
    Float32 in, per-batch float64 out, concatenated in order.
    """
    import tensorflow as tf
    parts = []
    for start in range(0, _rows(x32), batch_size):
        chunk = [tf.constant(part[start:start + batch_size]) for part in _parts(x32)]
        out = model(chunk if len(chunk) > 1 else chunk[0], training=False)
        parts.append(np.concatenate([np.asarray(o, dtype=np.float64) for o in out], axis=1))
    width = len(model.outputs)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, width))


def monitor_forward_identity():
    import inspect
    return {"id": MONITOR_FORWARD,
            "source_sha256": hashlib.sha256(inspect.getsource(monitor_forward).encode()).hexdigest()}


def materialize_monitor_inputs(rows, scales):
    """The ONE shared monitor-input materialization (see MONITOR_INPUT).

    For a market-time split the model input is the three-tensor list
    ``[observations, offsets_hours, closure_crossed]``; elapsed splits keep a single tensor.
    """
    x, y, mask = transform(rows, scales)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if "offsets_hours" in rows:
        design = {"market_time": {"spacing_open_market_hours": rows["_spacing_open_market_hours"]}} \
            if "_spacing_open_market_hours" in rows else None
        return market_time_inputs({**rows, "x": x}, design=design), y, mask
    return x, y, mask
#: Pre-declared, PROSPECTIVE monitor comparison floor (v2, approved by Satoshi after eight
#: rounds): scope is ONLY the cross-process GPU comparison of the monitored value and the
#: per-batch per-head float64 partial sums; everything else stays BITWISE (CPU end-to-end,
#: in-session self-check, reload identity, saved-vs-reloaded predictions, input witness
#: bytes, weight digests, forward identity, determinism flags). Bound |delta| <= atol +
#: rtol*|recorded| with rtol 5e-5 / atol 1e-9, calibrated from the recorded witness
#: distribution (observed max 1.82e-5) with margin against flapping yet far below any
#: scientific effect. Justification: the eight-round exclusion list (cuDNN/cuBLAS autotune
#: and init order, TF32, weights-at-witness digests, post-normalization input bytes,
#: dedicated forward function identity, grappler remapping/layout/auto-mixed-precision)
#: leaves bit-stable deltas that are a property of in-session vs fresh-instance GPU
#: contexts of this TensorFlow build. Applied prospectively: no scientific fit has run
#: under the corrected protocol and no past run is re-graded. Activates ONLY with the
#: static-algorithm determinism flags recorded and equal on both sides.
MONITOR_TOLERANCE = {
    "version": "v2", "rtol": 5e-5, "atol": 1e-9, "label": "NUMERICALLY_EQUIVALENT",
    "scope": "cross_process_gpu_monitored_value_and_partial_sums_only",
    "calibration": "recorded witness distribution across eight excluded-mechanism rounds, max 1.82e-5",
    "justification": ("a reproduced, build/context-specific numerical discrepancy between "
                      "in-session and fresh-instance GPU executions of this pinned TensorFlow "
                      "build, cause not demonstrated (eight mechanisms excluded: autotune/init "
                      "order, TF32, weights, input bytes, forward identity, grappler "
                      "remapping/layout/AMP), bounded by the versioned NUMERICALLY_EQUIVALENT "
                      "protocol"),
    "prospective": True, "requires_static_flags": True,
}


def _floor_allowance(reference):
    return MONITOR_TOLERANCE["atol"] + MONITOR_TOLERANCE["rtol"] * abs(reference)


def floor_satisfied(actual, reference):
    """The sealed formula: abs(actual - reference) <= atol + rtol * abs(reference)."""
    return abs(actual - reference) <= _floor_allowance(reference)


#: This session's process identity: recorded at fit time and compared at replay so the
#: GPU-and-cross-process scope is PROVEN, never inferred from an environment flag.
import uuid as _uuid
PROCESS_IDENTITY = {"uuid": _uuid.uuid4().hex, "pid": os.getpid()}

#: Largest observed relative partial-sum discrepancy across the eight excluded-mechanism
#: rounds (GPU_REPLAY.json at 8f8c4c1f, head 6h). Used ONLY to REPORT decision impact.
MEASURED_MONITOR_UNCERTAINTY_REL = 1.8244613879748877e-5


def _gpu_visible():
    import tensorflow as tf
    return bool(tf.config.list_physical_devices("GPU"))


def floor_protocol_identity():
    """The sealed NUMERICALLY_EQUIVALENT monitor-replay protocol identity.

    Seals the formula with its reference direction, rtol/atol, scope, dtype, label,
    required flags and runtime version. A record without this identity, or with a
    different one, can never use the floor: old receipts keep their original FAIL.
    """
    import tensorflow as tf
    from tools.i6dmo.model import keras_version
    body = {"id": "i6dmo.monitor_numeric_equivalence.v2",
            "formula": "abs(actual - reference) <= atol + rtol * abs(reference)",
            "reference": "recorded_fit_time_value", "actual": "fresh_process_replay_recomputation",
            "rtol": MONITOR_TOLERANCE["rtol"], "atol": MONITOR_TOLERANCE["atol"],
            "scope": "gpu_cross_process_monitor_replay_only", "dtype": "float64",
            "label": "NUMERICALLY_EQUIVALENT",
            "required_flags": current_determinism_flags(),
            "runtime": {"tensorflow": tf.__version__, "keras": keras_version()}}
    return dict(body, protocol_sha256=hashlib.sha256(_canonical(body).encode()).hexdigest())


def stopping_decision_impact(monitored_history, *, min_delta, patience, relative_uncertainty):
    """REPORT (never enforce) whether the measured replay uncertainty could change decisions.

    A negative control for near-tie early stopping: with per-epoch uncertainty
    u_e = relative_uncertainty * |monitored_e|, could the best epoch flip, or could a
    patience/stop decision (threshold min_delta) land differently across contexts?
    Training min_delta, patience and checkpoint selection stay untouched.
    """
    values = [float(v) for v in monitored_history]
    if not values:
        return {"could_change_best_epoch": None, "could_change_patience_or_stop": None,
                "relative_uncertainty": relative_uncertainty}
    unc = [relative_uncertainty * abs(v) for v in values]
    best = min(range(len(values)), key=lambda i: (values[i], i))
    flip_best = any(i != best and values[i] - values[best] <= unc[i] + unc[best]
                    for i in range(len(values)))
    reference, flip_patience = math.inf, False
    for i, value in enumerate(values):
        margin = (reference - min_delta) - value          # >0 significant, <=0 stale
        if reference < math.inf and abs(margin) <= unc[i] + relative_uncertainty * abs(reference):
            flip_patience = True
        if value < reference - min_delta:
            reference = value
    return {"could_change_best_epoch": flip_best, "could_change_patience_or_stop": flip_patience,
            "relative_uncertainty": relative_uncertainty, "min_delta": min_delta,
            "patience": patience, "epochs": len(values), "best_epoch": best + 1,
            "note": "reported negative control; training decisions are untouched and this report "
                    "never claims cross-context decision invariance"}


def current_determinism_flags():
    """The running process's canonical determinism/static-algorithm flags."""
    from tools.i6dmo.model import determinism_flags
    return determinism_flags()


def _ids_sha(ids):
    import hashlib as _h
    ids = np.ascontiguousarray(np.asarray(ids, dtype=np.int64))
    return _h.sha256(ids.tobytes()).hexdigest()


def evaluate_monitored(model, fit, inner, batch_size, ids=None):
    """THE monitored-loss evaluation procedure (training and replay share it, DMO10).

    One fixed path over the CANONICAL FIXED SEQUENTIAL partition: the split
    arrays in row_id order, cut into consecutive batches of ``batch_size``
    (``predict_matrix``; float32 inputs, float64 outputs per batch, concatenated
    in order), then :func:`masked_split_metrics` global per-head float64
    ``fsum`` sums/counts, equal-weight mean over supported heads, and
    ``0.5 * (fit_loss + inner_loss)``. The training shuffle exists for gradient
    steps ONLY and never reaches monitoring: on GPU a row's float32 forward
    value legitimately varies with batch composition (GEMM tiling), so only
    this fixed partition is bitwise comparable across sessions.

    Returns ``(fit_metrics, inner_metrics, monitored, witness)``. The witness
    carries the sha256 of the exact consumed row-id batch sequence, per-batch
    per-head float64 partial sums/counts, and a code witness (sha256 of this
    function's source plus a nonce derived from the partition parameters,
    computed INSIDE this function) so a stale, cached or fallback evaluation
    path is detectable at replay (``ids=None`` records positional ids).
    """
    import inspect
    names = ("fit", "inner")
    splits = {"fit": fit, "inner": inner}
    id_arrays = {}
    for index, name in enumerate(names):
        n = _rows(splits[name][0])
        given = None if ids is None else np.asarray(ids[index], dtype=np.int64)
        if given is not None and len(given) != n:
            raise TrainDisposition("MONITOR_PATH_MISMATCH", f"{name} ids length {len(given)} != rows {n}")
        id_arrays[name] = np.arange(n, dtype=np.int64) if given is None else given
    nonce = hashlib.sha256(_canonical(
        {"procedure": MONITOR_EVAL, "batch_size": batch_size, "partition": MONITOR_PARTITION,
         "n_fit": len(fit[0]), "n_inner": len(inner[0]),
         "fit_ids_sha256": _ids_sha(id_arrays["fit"]),
         "inner_ids_sha256": _ids_sha(id_arrays["inner"])}).encode()).hexdigest()
    code_witness = hashlib.sha256(
        (inspect.getsource(evaluate_monitored) + nonce).encode()).hexdigest()
    witness = {"ids": "positional" if ids is None else "row_id", "nonce": nonce,
               "code_witness": code_witness, "forward": monitor_forward_identity()}
    results = {}
    for name in names:
        x, y, mask = splits[name]
        x = _as_float32(x)                        # the EXACT bytes fed below
        pred = monitor_forward(model, x, batch_size)
        results[name] = masked_split_metrics(y, pred, mask)
        y64, mb_all = np.asarray(y, np.float64), np.asarray(mask, bool)
        batches = []
        for b, start in enumerate(range(0, _rows(x), batch_size)):
            stop = start + batch_size
            pb, yb, mb = pred[start:stop], y64[start:stop], mb_all[start:stop]
            sums = [math.fsum(np.abs(pb[mb[:, h], h] - yb[mb[:, h], h]).tolist())
                    for h in range(yb.shape[1])]
            counts = [int(mb[:, h].sum()) for h in range(yb.shape[1])]
            batches.append({"batch": b, "start": start, "rows": int(len(pb)),
                            "row_ids_sha256": _ids_sha(id_arrays[name][start:stop]),
                            "input_sha256": hashlib.sha256(
                                _input_bytes(_take(x, slice(start, stop)))).hexdigest(),
                            "partial_sums": sums, "partial_counts": counts})
        witness[name] = {"row_ids_sha256": _ids_sha(id_arrays[name]), "batches": batches,
                         # Input witness: post-normalization float32 bytes, digested before any
                         # forward; batch-0 rows individually, the first rows byte-complete.
                         "input_sha256": hashlib.sha256(_input_bytes(x)).hexdigest(),
                         "batch0_input_row_sha256": [
                             hashlib.sha256(_input_bytes(_take(x, slice(i, i + 1)))).hexdigest()
                             for i in range(min(batch_size, _rows(x)))],
                         "batch0_input_rows_hex": [
                             _input_bytes(_take(x, slice(i, i + 1))).hex()
                             for i in range(min(8, _rows(x)))],
                         # Row-level forward witness: batch-0 per-row per-head float64 hex, so a
                         # replay can tell a weight effect (localized rows) from a different
                         # matmul/conv execution mode (uniform small deltas on every row).
                         "batch0_predictions_hex": [[float(v).hex() for v in row]
                                                    for row in pred[:batch_size]]}
    monitored = 0.5 * (results["fit"]["loss"] + results["inner"]["loss"])
    return results["fit"], results["inner"], monitored, witness


def monitor_self_check(model, fit, inner, record, expected, ids=None):
    """Recording-time self-check: a SEPARATE recompute must reproduce ``expected`` bitwise.

    Called in-session before any receipt is written, on FRESHLY materialized
    input copies (never the same tensor object) and through the full
    ``evaluate_monitored`` path, so the comparison is always between two
    independent computations and never the stored value against itself. A
    mismatch is a typed MONITOR_SELF_CHECK_FAILED, never silently accepted.
    """
    def _copy(split):
        x, y, mask = split
        x = ([np.array(part, copy=True) for part in x] if isinstance(x, (list, tuple))
             else np.array(x, copy=True))
        return (x, np.array(y, copy=True), np.array(mask, copy=True))

    fit, inner = _copy(fit), _copy(inner)
    fm, im, monitored, report = replay_monitored(model, fit, inner, record, ids=ids)
    if report["status"] != "EXACT":
        raise TrainDisposition("MONITOR_SELF_CHECK_FAILED",
                               "the in-session self-check is EXACT by declaration; the sealed "
                               f"floor applies only cross-process on GPU, got {report['status']}")
    if monitored != expected:
        raise TrainDisposition("MONITOR_SELF_CHECK_FAILED",
                               f"inference-path recompute {monitored!r} != recorded {expected!r} "
                               "(bitwise, same device, same session)")
    return fm, im, monitored


def replay_monitored(model, fit, inner, record, batch_size=None, ids=None):
    """Recompute the monitored loss through the recorded identical procedure.

    ``record`` is TRAINING.json's ``monitor_evaluation``. A different procedure
    name, batch size, graph, partition or monitor instance is a typed
    MONITOR_PATH_MISMATCH (mixed v1/v2 records are refused): a one-pass or
    permuted recompute of a batched training monitor is a different reduction
    order, not a replay. ``model`` must be the canonical loader-path instance
    (``load_dmo``). When the record carries a witness, the recompute is
    compared batch by batch and the FIRST divergent batch is reported with its
    row-id digest and the pair of partial sums (MONITOR_WITNESS_DIVERGENCE).
    """
    if not isinstance(record, dict) or record.get("procedure") != MONITOR_EVAL:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"recorded procedure {record.get('procedure') if isinstance(record, dict) else record!r}"
                               f" is not {MONITOR_EVAL} (mixed or old records are never comparable)")
    if record.get("monitor_instance", MONITOR_INSTANCE) != MONITOR_INSTANCE:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"monitor instance {record.get('monitor_instance')!r} is not the "
                               f"canonical {MONITOR_INSTANCE}")
    recorded_forward = record.get("monitor_forward")
    if recorded_forward is not None and recorded_forward != monitor_forward_identity():
        raise TrainDisposition("MONITOR_FORWARD_MISMATCH",
                               f"record produced by forward {recorded_forward} but the monitor's "
                               f"one dedicated forward is {monitor_forward_identity()}; a value "
                               "measured through any other forward (e.g., a training-step graph) "
                               "is never comparable")
    if record.get("input_materialization", MONITOR_INPUT) != MONITOR_INPUT:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"input materialization {record.get('input_materialization')!r} is "
                               f"not the canonical {MONITOR_INPUT}; the bundle arrays with one "
                               "float64 CPU normalization and a single float32 cast are the truth")
    if record.get("graph", "inference") != "inference":
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"a {record.get('graph')}-graph-evaluated monitor is not comparable; "
                               "only the shared inference path is")
    if record.get("partition") != MONITOR_PARTITION:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"monitor partition {record.get('partition')!r} is not the canonical "
                               f"{MONITOR_PARTITION}; a permuted/shuffled monitor evaluation is a "
                               "different batch composition and is never comparable")
    recorded = record.get("batch_size")
    if type(recorded) is not int or recorded < 1:
        raise TrainDisposition("MONITOR_PATH_MISMATCH", "recorded batch_size missing")
    if batch_size is not None and batch_size != recorded:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               f"replay batch_size {batch_size} != recorded {recorded}; "
                               "the monitored value is only comparable through the identical path")
    flags = record.get("determinism_flags")
    tolerance_active, context = False, "unproven"
    if flags is not None:
        running = current_determinism_flags()
        if flags.get("tf_imported_before_flags") is True or running["tf_imported_before_flags"] is True:
            side = "recorded" if flags.get("tf_imported_before_flags") is True else "running"
            raise TrainDisposition("DETERMINISM_INIT_ORDER_INVALID",
                                   f"the {side} process imported TensorFlow before the static-"
                                   "algorithm flags were set; its kernel selection is ungoverned "
                                   "and the monitored value is not scoreable")
        if flags != running:
            raise TrainDisposition("DETERMINISM_FLAGS_MISMATCH",
                                   f"recorded flags {flags} != running process {running}; a record "
                                   "scored under other kernel-selection flags is never comparable")
    recording = record.get("recording") or {}
    if recording.get("process_uuid") == PROCESS_IDENTITY["uuid"]:
        context = "same_process"
    elif recording.get("device") == "gpu":
        context = "gpu_cross_process" if _gpu_visible() else "cross_device"
    elif recording.get("device") == "cpu":
        context = "cpu_cross_process"
    protocol = record.get("floor_protocol")
    if protocol is not None:
        current = floor_protocol_identity()
        if protocol != current:
            raise TrainDisposition("MONITOR_PROTOCOL_MISMATCH",
                                   f"record sealed under protocol {protocol.get('protocol_sha256')} "
                                   f"but this process computes {current['protocol_sha256']}; the "
                                   "numeric-equivalence rule never transfers across identities")
        # Scope is PROVEN: GPU-recorded AND a different process AND a GPU-visible replay —
        # never inferred from an environment flag. Everything else stays exact.
        tolerance_active = context == "gpu_cross_process"
    stored = record.get("witness")
    if stored is not None and stored.get("ids") == "row_id" and ids is None:
        raise TrainDisposition("MONITOR_PATH_MISMATCH",
                               "the recorded witness binds row ids; the replay must supply the same ids")
    fm, im, monitored, witness = evaluate_monitored(model, fit, inner, recorded, ids=ids)
    report = {"status": "EXACT", "within_floor": [], "row_forward_diagnostics": {}}
    try:
        if stored is not None:
            report = _compare_witness(stored, witness, tolerance_active)
        recorded_monitored = record.get("monitored")
        if recorded_monitored is not None and monitored != recorded_monitored:
            if not (tolerance_active and floor_satisfied(monitored, recorded_monitored)):
                raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                                       f"monitored {monitored!r} vs recorded {recorded_monitored!r} "
                                       f"exceeds the sealed allowance "
                                       f"{_floor_allowance(recorded_monitored)}")
            report["status"] = "NUMERICALLY_EQUIVALENT"
            report["monitored_abs_delta"] = abs(monitored - recorded_monitored)
    except TrainDisposition as exc:
        if exc.code == "MONITOR_WITNESS_DIVERGENCE" and context == "cross_device":
            raise TrainDisposition("CROSS_DEVICE_NOT_COMPARABLE",
                                   "a GPU-recorded monitor replayed without that GPU class is "
                                   "recorded separately, never graded equal or defective: "
                                   + str(exc)) from None
        raise
    report["context"] = context
    return fm, im, monitored, report


def _row_forward_diff(stored_hex, fresh_hex):
    """Batch-0 per-row diff with a delta-pattern verdict (uniform => execution mode)."""
    divergent, deltas = [], []
    for i, (ra, rb) in enumerate(zip(stored_hex, fresh_hex)):
        if ra == rb:
            continue
        divergent.append(i)
        for ha, hb in zip(ra, rb):
            va, vb = float.fromhex(ha), float.fromhex(hb)
            if va != vb:
                deltas.append(abs(va - vb) / max(abs(va), abs(vb), 1e-300))
    if not divergent:
        return None
    uniform = (len(divergent) == min(len(stored_hex), len(fresh_hex)) and deltas
               and max(deltas) <= 10 * min(deltas))
    return {"n_rows": len(stored_hex), "n_divergent_rows": len(divergent),
            "first_divergent_rows": divergent[:5],
            "max_relative_delta": max(deltas) if deltas else 0.0,
            "min_relative_delta": min(deltas) if deltas else 0.0,
            "uniform_delta_pattern": uniform,
            "interpretation": ("UNIFORM small deltas on every row: a different matmul/conv "
                               "execution mode (e.g., TF32 vs FP32), not weights" if uniform else
                               "LOCALIZED row divergence: weight or data effect, not a uniform "
                               "execution-mode rounding")}


def _compare_input_witness(name, a, b):
    """Byte identity of the post-normalization inputs; a mismatch names rows and bytes."""
    if a.get("input_sha256") is None:
        return
    if a["input_sha256"] == b["input_sha256"]:
        return
    rows = [i for i, (ra, rb) in enumerate(zip(a.get("batch0_input_row_sha256") or [],
                                               b["batch0_input_row_sha256"])) if ra != rb]
    detail = (f"{name}: post-normalization input bytes differ (recorded {a['input_sha256']} vs "
              f"recomputed {b['input_sha256']}); the monitor was fed a different tensor than the "
              f"canonical materialization rebuilds")
    bad_batches = [rb["batch"] for ra, rb in zip(a.get("batches") or [], b["batches"])
                   if ra.get("input_sha256") != rb["input_sha256"]]
    detail += f"; divergent batches {bad_batches[:5]} of {len(b['batches'])}"
    if rows:
        detail += f"; batch 0 first divergent rows {[f'row {i}' for i in rows[:5]]} of {len(b['batch0_input_row_sha256'])}"
        stored_hex, fresh_hex = a.get("batch0_input_rows_hex") or [], b["batch0_input_rows_hex"]
        for i, (ha, hb) in enumerate(zip(stored_hex, fresh_hex)):
            if ha != hb:
                sa = np.frombuffer(bytes.fromhex(ha), dtype=np.float32)
                sb = np.frombuffer(bytes.fromhex(hb), dtype=np.float32)
                elems = np.flatnonzero(sa.tobytes() != sb.tobytes() if len(sa) != len(sb)
                                       else sa != sb)[:4]
                pairs = [(int(e), float(sa[e]), float(sb[e])) for e in elems]
                detail += (f"; row {i} byte-level first differing elements "
                           f"(index, recorded, recomputed): {pairs}")
                break
    raise TrainDisposition("INPUT_WITNESS_MISMATCH", detail)


def _compare_witness(stored, fresh, tolerance_active=False):
    """Batch-by-batch comparison; returns {"status", "within_floor", "row_forward_diagnostics"}.

    With ``tolerance_active`` (static-algorithm flags confirmed on both sides), a batch
    whose structure matches exactly and whose per-head partial sums each satisfy
    |delta| <= atol + rtol*|recorded| is WITHIN_DECLARED_FLOOR, reported never refused.
    The row-level hex forward witness is a recorded DIAGNOSTIC only: reported, never
    pass/fail at row level. Beyond the floor stays a hard MONITOR_WITNESS_DIVERGENCE.
    """
    within_floor, diagnostics = [], {}
    if stored.get("forward") is not None and stored["forward"] != fresh["forward"]:
        raise TrainDisposition("MONITOR_FORWARD_MISMATCH",
                               f"witness forward {stored['forward']} != the monitor's dedicated "
                               f"forward {fresh['forward']}")
    if stored.get("code_witness") != fresh["code_witness"]:
        raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                               "code witness differs: the recorded evaluation ran different "
                               "evaluate_monitored source or partition parameters (stale/cached/"
                               f"fallback path): recorded={stored.get('code_witness')} "
                               f"recomputed={fresh['code_witness']}")
    for name in ("fit", "inner"):
        a, b = stored.get(name) or {}, fresh[name]
        _compare_input_witness(name, a, b)
        row_diff = None
        if a.get("batch0_predictions_hex"):
            row_diff = _row_forward_diff(a["batch0_predictions_hex"], b["batch0_predictions_hex"])
            if row_diff:
                diagnostics[name] = row_diff
        if a.get("row_ids_sha256") != b["row_ids_sha256"]:
            raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                                   f"{name} row-id sequence differs: recorded "
                                   f"{a.get('row_ids_sha256')} recomputed {b['row_ids_sha256']}")
        for ra, rb in zip(a.get("batches") or [], b["batches"]):
            if ra == rb:
                continue
            structural = {k: ra.get(k) for k in ("batch", "start", "rows", "row_ids_sha256",
                                                 "input_sha256", "partial_counts")}
            if tolerance_active and structural == {k: rb.get(k) for k in structural}:
                recorded_sums = ra.get("partial_sums") or []
                pairs = list(zip(recorded_sums, rb["partial_sums"]))
                if pairs and all(floor_satisfied(v, u) for u, v in pairs):
                    within_floor.append({"split": name, "batch": rb["batch"],
                                         "max_abs_delta": max(abs(u - v) for u, v in pairs),
                                         "max_allowed": max(_floor_allowance(u) for u, _ in pairs),
                                         "floor": {"rtol": MONITOR_TOLERANCE["rtol"],
                                                   "atol": MONITOR_TOLERANCE["atol"]}})
                    continue
            detail = (f"{name} batch {rb['batch']} (start {rb['start']}, row_ids "
                      f"{rb['row_ids_sha256']}): recorded partial_sums={ra.get('partial_sums')} "
                      f"counts={ra.get('partial_counts')} vs recomputed "
                      f"partial_sums={rb['partial_sums']} counts={rb['partial_counts']}")
            if rb["batch"] == 0 and row_diff:
                detail += (f"; row-level forward witness: rows "
                           f"{['row ' + str(i) for i in row_diff['first_divergent_rows']]} of "
                           f"{row_diff['n_rows']} diverge ({row_diff['n_divergent_rows']} total), "
                           f"relative deltas [{row_diff['min_relative_delta']:.3e}, "
                           f"{row_diff['max_relative_delta']:.3e}], uniform_delta_pattern="
                           f"{row_diff['uniform_delta_pattern']}: {row_diff['interpretation']}")
            raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE", detail)
        if len(a.get("batches") or []) != len(b["batches"]):
            raise TrainDisposition("MONITOR_WITNESS_DIVERGENCE",
                                   f"{name} batch count differs: {len(a.get('batches') or [])} "
                                   f"recorded vs {len(b['batches'])} recomputed")
    return {"status": "NUMERICALLY_EQUIVALENT" if within_floor else "EXACT",
            "within_floor": within_floor, "row_forward_diagnostics": diagnostics}


class MeanMonitorStopper:
    """Monitor ``0.5 * (fit + inner)``; patience vs ``reference - min_delta``; keep lowest.

    Rule identity ``i6dmo.mean_monitor_stop.v1``: an epoch is stale unless its
    monitored loss is below the last significant reference minus ``min_delta``;
    stopping happens when ``stale >= patience`` or at ``max_epochs``. The
    restored checkpoint is the lowest monitored epoch (first tie wins), which
    includes improvements smaller than ``min_delta``.
    """

    RULE = "i6dmo.mean_monitor_stop.v1"

    def __init__(self, patience, min_delta, max_epochs):
        self.patience, self.min_delta, self.max_epochs = patience, min_delta, max_epochs
        self.history, self.best_epoch, self.best_weights = [], None, None
        self.best_monitored, self.reference, self.stale = math.inf, math.inf, 0
        self.stop_reason = None

    def update(self, fit_loss, inner_loss, weights):
        monitored = 0.5 * (fit_loss + inner_loss)
        if not math.isfinite(monitored):
            raise TrainDisposition("NONFINITE_LOSS", f"monitored loss {monitored}")
        epoch = len(self.history) + 1
        if monitored < self.best_monitored:
            self.best_monitored, self.best_epoch = monitored, epoch
            self.best_weights = [np.array(w, copy=True) for w in weights]
        if monitored < self.reference - self.min_delta:
            self.reference, self.stale = monitored, 0
        else:
            self.stale += 1
        self.history.append({"epoch": epoch, "fit_loss": fit_loss, "inner_loss": inner_loss,
                             "monitored": monitored, "stale": self.stale, "best_epoch": self.best_epoch})
        if self.stale >= self.patience:
            self.stop_reason = "patience"
        elif epoch >= self.max_epochs:
            self.stop_reason = "max_epochs"
        return self.stop_reason is not None

    def identity(self):
        return {"rule": self.RULE, "patience": self.patience, "min_delta": self.min_delta,
                "max_epochs": self.max_epochs, "monitor": "0.5*(fit_loss+inner_loss)",
                "restore": "lowest_monitored_first_tie"}


def _monitor_model(bundle, workdir):
    """A FRESH canonical-loader-path instance holding the training model's current weights.

    Monitoring (and therefore early stopping) never reads the trained in-session
    instance: training on GPU acquires cuDNN/compilation state that can alter
    subsequent in-process forwards, so the monitored value is computed on the
    same save -> build_dmo -> load_weights instance the replay uses.
    """
    from tools.i6dmo import model as M
    path = Path(workdir) / "monitor.weights.h5"
    bundle.forecast_model.save_weights(path)
    fresh = M.build_dmo(bundle.config)
    fresh.forecast_model.load_weights(path)
    return fresh.forecast_model


def fit_dmo(bundle, fit, inner, training, seed, progress=None, ids=None, record_device="cpu"):
    """One fit with AdamW; returns the audit dict. Restores the selected weights."""
    import tempfile
    import tensorflow as tf
    from predictor_plugins.modular_temporal import weights_hash
    from tools.i6dmo.model import predict_matrix

    model = bundle.forecast_model
    tf.keras.utils.set_random_seed(seed)
    order_rng = np.random.default_rng(seed)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=training["learning_rate"],
                                          weight_decay=training["weight_decay"])
    x_fit, y_fit, m_fit = fit
    x_in, y_in, m_in = inner
    variables = model.trainable_variables
    optimizer.build(variables)  # optimizer slots exist before the traced step

    @tf.function(reduce_retracing=True)
    def step(x, y, m):
        with tf.GradientTape() as tape:
            pred = tf.concat(model(x, training=True), axis=1)
            loss, _, _ = masked_normalized_loss(y, pred, m)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    stopper = MeanMonitorStopper(training["patience"], training["min_delta"], training["max_epochs"])
    batch, updates, rows = training["batch_size"], 0, []
    best_witness, loader_overhead = None, []
    monitor_dir = tempfile.TemporaryDirectory(prefix="i6dmo_monitor_")
    initial = weights_hash(model)
    started = time.monotonic()
    while True:
        perm = order_rng.permutation(_rows(x_fit))
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            chunk = [tf.constant(part) for part in _parts(_take(x_fit, idx))]
            loss = float(step(chunk if len(chunk) > 1 else chunk[0],
                              tf.constant(y_fit[idx], tf.float32), tf.constant(m_fit[idx])))
            if not math.isfinite(loss):
                raise TrainDisposition("NONFINITE_LOSS", f"training batch loss {loss}")
            updates += 1
        # The monitored value early stopping reads comes ONLY from the shared inference path on
        # the canonical sequential partition AND the canonical loader-path instance; the
        # per-epoch permutation above is gradient-only.
        overhead_start = time.monotonic()
        monitor_model = _monitor_model(bundle, monitor_dir.name)
        loader_overhead.append(time.monotonic() - overhead_start)
        fm, im, _, epoch_witness = evaluate_monitored(
            monitor_model, (x_fit, y_fit, m_fit), (x_in, y_in, m_in), batch, ids=ids)
        was_best = stopper.best_monitored
        stop = stopper.update(fm["loss"], im["loss"], model.get_weights())
        if stopper.best_monitored < was_best:
            from tools.i6dmo.model import weight_tensor_digests
            # Weights-at-witness: the exact tensors of the instance the witness evaluated.
            epoch_witness["weights_at_witness"] = weight_tensor_digests(monitor_model)
            best_witness = epoch_witness
        row = dict(stopper.history[-1], updates=updates, weights_sha256=weights_hash(model),
                   fit=fm, inner=im, elapsed_s=time.monotonic() - started)
        rows.append(row)
        if progress is not None:
            progress({k: row[k] for k in ("epoch", "fit_loss", "inner_loss", "monitored", "stale",
                                          "best_epoch", "updates", "elapsed_s")})
        if stop:
            break
    last = weights_hash(model)
    model.set_weights(stopper.best_weights)
    restored = weights_hash(model)
    if restored != rows[stopper.best_epoch - 1]["weights_sha256"]:
        raise TrainDisposition("RESTORE_MISMATCH", "restored weights differ from the selected epoch")
    record = {"procedure": MONITOR_EVAL, "batch_size": batch, "graph": "inference",
              "partition": MONITOR_PARTITION, "monitor_instance": MONITOR_INSTANCE,
              "input_materialization": MONITOR_INPUT,
              "monitor_forward": monitor_forward_identity(),
              "floor_protocol": floor_protocol_identity(),
              "recording": {"device": record_device, "process_uuid": PROCESS_IDENTITY["uuid"],
                            "pid": PROCESS_IDENTITY["pid"]},
              "decision_impact": stopping_decision_impact(
                  [row["monitored"] for row in rows], min_delta=training["min_delta"],
                  patience=training["patience"],
                  relative_uncertainty=MEASURED_MONITOR_UNCERTAINTY_REL),
              "determinism_flags": current_determinism_flags(),
              "declared_tolerance": dict(MONITOR_TOLERANCE),
              "witness": best_witness,
              "loader_overhead_s": {"epochs": len(loader_overhead),
                                    "mean_s": math.fsum(loader_overhead) / len(loader_overhead),
                                    "max_s": max(loader_overhead),
                                    "total_s": math.fsum(loader_overhead)},
              "prediction_dtype": "float32 inputs, float64 per-batch outputs",
              "accumulation": "global per-head float64 math.fsum sums/counts, "
                              "equal-weight mean over supported heads"}
    best_row = rows[stopper.best_epoch - 1]
    fm_best, im_best, monitored_best = monitor_self_check(
        _monitor_model(bundle, monitor_dir.name), (x_fit, y_fit, m_fit), (x_in, y_in, m_in),
        record, best_row["monitored"], ids=ids)
    monitor_dir.cleanup()
    record.update(self_check="PASS", best_epoch=stopper.best_epoch, monitored=monitored_best,
                  fit={"sums": fm_best["sums"], "counts": fm_best["counts"]},
                  inner={"sums": im_best["sums"], "counts": im_best["counts"]})
    if int(optimizer.iterations.numpy()) != updates:
        raise TrainDisposition("UPDATE_COUNT_MISMATCH", "optimizer iterations disagree with updates")
    return {"history": rows, "best_epoch": stopper.best_epoch, "best_monitored": stopper.best_monitored,
            "monitor_evaluation": record,
            "stop_reason": stopper.stop_reason, "epochs_completed": len(rows), "updates": updates,
            "initial_weights_sha256": initial, "last_weights_sha256": last,
            "restored_weights_sha256": restored, "early_stop": stopper.identity(),
            "fit_seconds": time.monotonic() - started}


# --------------------------------------------------------------------- unit

def _cgroup_peak():
    try:
        line = next(l for l in Path("/proc/self/cgroup").read_text().splitlines() if l.startswith("0::"))
        return int(Path("/sys/fs/cgroup" + line[3:].strip(), "memory.peak").read_text().strip())
    except (OSError, StopIteration, ValueError):
        return None


def _resources(device):
    import tensorflow as tf
    usage = resource.getrusage(resource.RUSAGE_SELF)
    gpu_peak = None
    if device == "gpu" and tf.config.list_physical_devices("GPU"):
        try:
            gpu_peak = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
        except (ValueError, RuntimeError, KeyError):
            gpu_peak = None
    return {"tf_gpu_memory_peak_bytes": gpu_peak, "cpu_user_s": usage.ru_utime, "cpu_sys_s": usage.ru_stime, "max_rss_kib": usage.ru_maxrss,
            "cgroup_memory_peak_bytes": _cgroup_peak(), "device": device,
            "visible_gpus": len(tf.config.list_physical_devices("GPU")),
            "gpu_device_names": _gpu_device_names(device),
            "tensorflow_version": tf.__version__}


def _gpu_device_names(device):
    """GPU class actually used (device_name per visible GPU); recorded so R0 and R2 can be pinned
    to the same class (R2-18). Empty on CPU."""
    import tensorflow as tf
    if device != "gpu":
        return []
    names = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            names.append(str(tf.config.experimental.get_device_details(gpu).get("device_name")))
        except (ValueError, RuntimeError):
            names.append(None)
    return names


def _code_identity():
    out = {}
    for rel in CODE_FILES:
        path = REPO / rel
        out[rel] = file_sha256(path) if path.exists() else None
    return out


def _summary(arrays, pred_price):
    """Model-side pooled original-price MAE vs naive on identical valid rows (not authoritative)."""
    rows = []
    for h in range(arrays["mask"].shape[1]):
        sel = arrays["mask"][:, h]
        n = int(sel.sum())
        if not n:
            rows.append({"count": 0, "mae": None, "naive": None, "relative_error": None})
            continue
        target = arrays["target_price"][sel, h]
        mae = math.fsum(np.abs(pred_price[sel, h] - target).tolist()) / n
        naive = math.fsum(np.abs(arrays["origin_price"][sel] - target).tolist()) / n
        rows.append({"count": n, "mae": mae, "naive": naive,
                     "relative_error": (mae - naive) / naive if naive > 0 else None})
    return rows


def r2_import_into(bundle, common_population, effective, r2_donors):
    """R2 initialization of a freshly seed-built matched ``bundle`` (OPS runner hook, section 9.5).

    ``r2_donors`` = {"donors": {feature: {"terminal": {path, sha256}}}, "objective": <ruled>,
    "provenance": {campaign_id, ruling_seal_sha256, code_pin, units}} from the OPS store. The
    learned-corpus digests are recomputed HERE from the authenticated population, so a donor binds
    exactly this population's FIT windows and purged INNER selection rows."""
    from tools.i6dmo import model as M
    from tools.i6dmo import r2_runner
    population = load_matched_population(common_population["path"], common_population["expected_digest"],
                                         effective["horizons_hours"], effective["feature_manifest"]["features"],
                                         input_dir=common_population.get("input_dir"))
    provenance = dict(r2_donors["provenance"], corpus=r2_runner.corpora(population, effective))
    context = M.branch_donor_context(bundle, population_sha256=population.digest,
                                     scaler_sha256=r2_runner.scaler_sha256(population),
                                     objective=r2_donors["objective"], provenance=provenance)
    try:
        return M.import_branch_donors(bundle, r2_donors["donors"], context)
    except M.DonorImportError as exc:
        raise TrainDisposition(exc.code, str(exc)[:400]) from None


def train_unit(design_path, bundle_dir, out_dir, device, *, training_overrides=None,
               common_population=None, r2_donors=None):
    """Run one unit end to end; returns the receipt (also written as RECEIPT.json).

    ``training_overrides`` (``max_epochs``/``patience`` only) exist for software
    tests; any override stamps the receipt ``scientific: false``.

    ``common_population`` ({"path", "expected_digest"[, "input_dir"]}) selects the MATCHED
    path: the sealed matched design's arm tensors come from the data lane's frozen common
    population (``bundle_dir`` must then be None or that same directory), with the common
    per-head masks, the shared FIT-only scaler and naive loss scale, and a capacity disclosure.
    """
    started, wall0 = datetime.now(timezone.utc), time.monotonic()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ContractError("OUT_DIR_NOT_EMPTY", str(out))
    if device not in ("cpu", "gpu"):
        raise ContractError("DEVICE", "device must be cpu or gpu")
    overrides = dict(training_overrides or {})
    if set(overrides) - set(OVERRIDABLE):
        raise ContractError("OVERRIDE", f"only {OVERRIDABLE} may be overridden in software tests")
    if common_population is not None:
        if not isinstance(common_population, dict) or not {"path", "expected_digest"} <= \
                set(common_population) or set(common_population) - {"path", "expected_digest",
                                                                     "input_dir"}:
            raise ContractError("MATCHED_ARGS", "common_population needs path and expected_digest "
                                "(and optionally input_dir)")
        if bundle_dir is not None and Path(bundle_dir).resolve() != Path(common_population["path"]).resolve():
            raise ContractError("MATCHED_ARGS", "a matched fit reads its arm tensors from the common "
                                "population; --bundle must be omitted or name that directory")
    elif bundle_dir is None:
        raise ContractError("BUNDLE_REQUIRED", "--bundle is required outside the matched path")
    if r2_donors is not None and common_population is None:
        raise ContractError("R2_ARGS", "an R2 fit imports donors into a matched arm (--common-population)")
    effective = effective_from_design(design_path)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": RECEIPT_SCHEMA, "experiment_id": effective["experiment_id"],
               "started_at": started.isoformat(), "scientific": not overrides and device == "gpu",
               "overrides": overrides,
               "identity": {"design_sha256": effective["design_sha256"],
                            "effective_config_sha256": effective["effective_config_sha256"],
                            "code_files_sha256": _code_identity()},
               "artifacts": {}}
    try:
        from tools.i6dmo.model import enable_determinism
        receipt["determinism"] = enable_determinism()
        import tensorflow as tf
        if device == "gpu" and not tf.config.list_physical_devices("GPU"):
            raise TrainDisposition("NO_GPU_VISIBLE", "a GPU fit was requested; CPU fallback is forbidden")
        receipt["device"] = device
        matched = None
        if common_population is None:
            data = load_data_bundle(bundle_dir, effective)
            receipt["identity"].update(bundle_manifest_sha256=data["manifest_sha256"],
                                       feature_manifest_sha256=data["feature_identity"]["manifest_sha256"])
            subset = os.environ.get("I6DMO_TEST_MAX_ROWS")
            if subset:   # software tests only; already stamped scientific=false by the overrides
                limit = int(subset)
                receipt["overrides"] = {**receipt.get("overrides", {}), "max_rows": limit}
                receipt["scientific"] = False
                for name in required_splits(effective):
                    data[name] = {k: (v[:limit] if isinstance(v, np.ndarray) and v.ndim >= 1 else v)
                                  for k, v in data[name].items()}
            fit_rows, fit_dropped = trainable_rows(data["fit"])
            inner_rows, inner_dropped = trainable_rows(data["inner"])
            scales = fit_scales(fit_rows)
        else:
            data, scales, matched = _prepare_matched(common_population, effective)
            receipt["identity"].update(
                bundle_manifest_sha256=matched["arm_source"]["arm_arrays_sha256"],
                feature_manifest_sha256=effective["feature_manifest"]["manifest_sha256"],
                common_population_sha256=matched["common_population_sha256"],
                matched_arm=matched["arm"])
            # Every common origin has at least one common head by construction (validated);
            # nothing is dropped or added on the model side.
            fit_rows, fit_dropped = data["fit"], 0
            inner_rows, inner_dropped = data["inner"], 0
        training = {**effective["training"], **overrides}
        from tools.i6dmo import model as M
        placement = "/GPU:0" if device == "gpu" else "/CPU:0"

        def progress(event):
            _atomic_json(out / "PROGRESS.json", {**event, "max_epochs": training["max_epochs"],
                                                 "updated_at": datetime.now(timezone.utc).isoformat()})

        with tf.device(placement):
            bundle = M.build_dmo(effective["model"])
            if matched is not None:
                try:
                    matched["capacity"] = M.capacity_record(bundle)
                except M.CapacityError as exc:
                    raise TrainDisposition(exc.code, str(exc)) from None
                if matched["capacity"]["arm"] != matched["arm"]:
                    raise TrainDisposition("CAPACITY_UNREPORTED_CHANGE",
                                           f"built {matched['capacity']['arm']} network for arm {matched['arm']}")
                receipt["identity"]["capacity_sha256"] = matched["capacity"]["capacity_sha256"]
            r2_record = None
            if r2_donors is not None:  # R2: authenticated donors into the R0 seed build, then the R0 recipe
                r2_record = r2_import_into(bundle, common_population, effective, r2_donors)
                receipt["identity"]["r2_import_sha256"] = r2_record["import_sha256"]
            result = fit_dmo(bundle, materialize_monitor_inputs(fit_rows, scales),
                             materialize_monitor_inputs(inner_rows, scales),
                             training, effective["seed"], progress,
                             ids=(fit_rows["row_id"], inner_rows["row_id"]), record_device=device)
            t0 = time.monotonic()
            predictions = {}
            for name in required_splits(effective):
                x, _, _ = materialize_monitor_inputs(data[name], scales)
                predictions[name] = to_price(M.predict_matrix(bundle.forecast_model, x),
                                             data[name], scales)
            predict_s = time.monotonic() - t0
        model_doc = M.save_dmo(bundle, out / "model", extra={"scales_sha256": scales["scales_sha256"]})
        _atomic_json(out / "SCALES.json", scales)
        _atomic_json(out / "TRAINING.json", {
            "schema": "i6dmo.training.v1", "settings": training, "rows": {
                "fit_trainable": len(fit_rows["row_id"]), "fit_without_valid_head": fit_dropped,
                "inner_trainable": len(inner_rows["row_id"]), "inner_without_valid_head": inner_dropped},
            **result})
        (out / "predictions").mkdir(exist_ok=True)
        for name in required_splits(effective):
            np.savez(out / "predictions" / f"{name}.npz", row_id=data[name]["row_id"],
                     horizons_hours=np.asarray(effective["horizons_hours"], np.int64),
                     origin_price=data[name]["origin_price"], predicted_price=predictions[name],
                     target_price=data[name]["target_price"], mask=data[name]["mask"])
        files = ["model/forecast_model.keras", "model/MODEL.json", "SCALES.json", "TRAINING.json",
                 *[f"predictions/{n}.npz" for n in required_splits(effective)]]
        if matched is not None:
            _atomic_json(out / "CAPACITY.json", matched["capacity"])
            files.append("CAPACITY.json")
        if r2_record is not None:
            _atomic_json(out / "R2_IMPORT.json", r2_record)
            files.append("R2_IMPORT.json")
        receipt["artifacts"] = {rel: file_sha256(out / rel) for rel in files}
        receipt["identity"].update(weights_sha256=model_doc["weights_sha256"],
                                   model_archive_sha256=model_doc["archive_sha256"],
                                   scales_sha256=scales["scales_sha256"])
        receipt["training"] = {k: result[k] for k in ("best_epoch", "best_monitored", "stop_reason",
                                                      "epochs_completed", "updates", "early_stop")}
        receipt["model_side_summary"] = {
            "note": "pooled original-price MAE/naive on identical valid rows; the independent replay governs",
            "horizons_hours": effective["horizons_hours"],
            **{name: _summary(data[name], predictions[name]) for name in required_splits(effective)}}
        receipt["timings"] = {"fit_s": result["fit_seconds"], "predict_s": predict_s}
        if is_market_time(effective):
            mt = effective["design"]["market_time"]
            receipt["market_time"] = {**mt, "slot_order_is_not_wall_clock": True,
                                      "time_inputs": list(MARKET_TIME_ARRAYS),
                                      "bundle_schema": "i6dmo.market_bundle.v1"}
        if matched is not None:
            receipt["matched"] = matched
        receipt.update(status="COMPLETED", code="COMPLETED")
    except TrainDisposition as exc:
        receipt.update(status="FAILED_TYPED", code=exc.code, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - every unit needs a terminal
        import traceback
        receipt.update(status="FAILED_UNTYPED", code="UNHANDLED_EXCEPTION",
                       detail=f"{type(exc).__name__}: {exc}"[:2000],
                       traceback=traceback.format_exc()[-2500:])
    receipt.setdefault("timings", {})["wall_s"] = time.monotonic() - wall0
    try:
        receipt["resources"] = _resources(device)
    except Exception:  # noqa: BLE001
        receipt["resources"] = None
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(out / "RECEIPT.json", receipt)
    return receipt


EXIT = {"COMPLETED": 0, "FAILED_TYPED": 3, "FAILED_UNTYPED": 4}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.train",
                                     description="One I6-DMO fit from a frozen data bundle (no overrides).")
    parser.add_argument("--design", required=True)
    parser.add_argument("--bundle", default=None,
                        help="frozen data bundle (omit, or the population dir, for a matched fit)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--common-population", default=None,
                        help="matched arm: the data lane's frozen common population")
    parser.add_argument("--common-population-digest", default=None,
                        help="matched arm: the expected common-population digest (64 hex)")
    parser.add_argument("--input-dir", default=None,
                        help="matched arm: retained input directory; verifies the sealed source")
    parser.add_argument("--r2-donors", default=None,
                        help="R2 fit: JSON with this campaign's authenticated pretraining terminal "
                             "references, the ruled objective and the campaign provenance")
    args = parser.parse_args(argv)
    matched = None
    if (args.common_population is None) != (args.common_population_digest is None):
        print(json.dumps({"status": "CONTRACT_ERROR", "code": "MATCHED_ARGS",
                          "detail": "--common-population and --common-population-digest go together"}))
        return 2
    if args.common_population is not None:
        matched = {"path": args.common_population, "expected_digest": args.common_population_digest}
        if args.input_dir is not None:
            matched["input_dir"] = args.input_dir
    elif args.input_dir is not None:
        print(json.dumps({"status": "CONTRACT_ERROR", "code": "MATCHED_ARGS",
                          "detail": "--input-dir belongs to a matched fit"}))
        return 2
    overrides = None
    raw = os.environ.get("I6DMO_TEST_TRAINING_OVERRIDES")
    if raw:  # software tests only: stamps the receipt scientific=false; no CLI flag exists
        try:
            overrides = json.loads(raw)
        except ValueError:
            print(json.dumps({"status": "CONTRACT_ERROR", "code": "OVERRIDE",
                              "detail": "I6DMO_TEST_TRAINING_OVERRIDES is not JSON"}))
            return 2
    try:
        r2_donors = json.loads(Path(args.r2_donors).read_text()) if args.r2_donors else None
        receipt = train_unit(args.design, args.bundle, args.out, args.device,
                             training_overrides=overrides, common_population=matched, r2_donors=r2_donors)
    except ContractError as exc:
        print(json.dumps({"status": "CONTRACT_ERROR", "code": exc.code, "detail": str(exc)}))
        return 2
    if args.json:
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    else:
        print(f"{receipt['status']} {receipt['code']}")
    return EXIT[receipt["status"]]


if __name__ == "__main__":
    sys.exit(main())
