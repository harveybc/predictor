"""Private, fit-free data preparation for the bounded resolution comparison.

API: prepare(input_dir, out_dir, base_design_path) returns MANIFEST.json's dict.
load(out_dir, group, spacing, slots, horizons=None, *, input_dir=None)
returns (arrays, scales,
metadata). arrays maps FIT/INNER/VALIDATION to RAW x[n, slots, 19], bool mask,
origin_price[n], target_price[n, heads] (NaN when masked), integer UTC-second
row_ids/origin_ts, offsets_hours[n, slots], closure_crossed and week_ids.
No normalization, feature selection, model fitting or TEST loading occurs.

scales contains input_mean, std (also input_std for trainer interoperability),
naive_mae and horizons lists plus scales_sha256. It is FIT-only and common to
every treatment in its group. Horizon projection drops all-invalid rows and
slices naive_mae, never refitting the group input scaler. metadata retains the
full group's scales_sha256 and support identity for that projection.

RD01/02: generic data designs, exact calendar-ranked observed slots and C1
masks. RD03: deduplicated union of all common-FIT treatment source bars.
RD04: one long support frozen across all ten treatments, including H72-only
loads. RD05: no future-dependent statistics. RD06: private, source/code/file
authenticated replay and TEST refusal. See test_resolution_data.py for the
corresponding structural, perturbation and corruption acceptance evidence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from tools.i6dmo import data, market_grid
from tools.i6n_swarm.common import atomic_json, file_sha256, sha256_json

SCHEMA = "i6dmo.resolution_data.v1"
GROUPS = {
    "short": dict(slots=(13,), spacings=(2, 4), horizons=(2, 3, 4, 5, 6, 12)),
    "long": dict(slots=(6, 9, 13, 18, 24), spacings=(2, 4), horizons=(18, 24, 36, 48, 60, 72)),
}
REPO = Path(__file__).resolve().parents[2]
CODE_FILES = ("tools/i6dmo/resolution_data.py", "tools/i6dmo/data.py",
              "tools/i6dmo/market_grid.py", "tools/i6dmo/provenance.py",
              "tools/i6n_contracts/eligibility.py", "tools/i6n_contracts/feature_availability.py",
              "tools/i6n_swarm/common.py")
COMMON_KEYS = {"mask", "origin_price", "target_price", "row_ids", "origin_ts", "week_ids"}
INPUT_KEYS = {"x", "offsets_hours", "closure_crossed"}


class ResolutionDataError(ValueError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (":" + detail if detail else ""))


def _digest(array):
    a = np.ascontiguousarray(array)
    header = sha256_json(dict(dtype=a.dtype.str, shape=list(a.shape)))
    return hashlib.sha256(header.encode() + a.tobytes()).hexdigest()


def _code():
    return {name: file_sha256(REPO / name) for name in CODE_FILES}


def _design(base, spacing, slots, horizons):
    doc = copy.deepcopy(base)
    # Only the generic data contract is used; no experiment seal is generated.
    for key in data.SEAL_KEYS:
        doc.pop(key, None)
    doc["input"].update(spacing_hours=spacing, observations=slots,
                        history_span_hours=spacing * (slots - 1))
    doc["horizons_hours"] = list(horizons)
    return data.validate_design(doc)


def _private_destination(out, source):
    out, source = Path(out).resolve(), Path(source).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ResolutionDataError("OUTPUT_SOURCE_OVERLAP")
    anchor = out
    while not anchor.exists():
        anchor = anchor.parent
    check = subprocess.run(["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, check=False)
    if check.returncode == 0:
        raise ResolutionDataError("PRIVATE_OUTPUT_INSIDE_GIT")
    return out


def _save(root, name, arrays):
    path = root / name
    with path.open("xb") as handle:
        os.chmod(path, 0o600)
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return file_sha256(path)


def _npz(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ResolutionDataError("ARRAY_FILE_INVALID", Path(path).name) from exc


def _seal_scales(body):
    return dict(body, scales_sha256=sha256_json(body))


def _require_heads(group, split, counts, horizons):
    if len(counts) != len(horizons):
        raise ResolutionDataError("HEAD_COUNTS_INVALID", group + ":" + split)
    for h, count in zip(horizons, counts):
        if count <= 0:
            raise ResolutionDataError("HEAD_UNSUPPORTED", f"{group}:{split}:h{h}")


def _validate_scales(scales, group, horizons):
    mean = np.asarray(scales["input_mean"], dtype=np.float64)
    std = np.asarray(scales["std"], dtype=np.float64)
    alias = np.asarray(scales["input_std"], dtype=np.float64)
    if mean.shape != (19,) or std.shape != (19,) or not np.isfinite(mean).all() \
            or not np.isfinite(std).all() or np.any(std <= 0) or not np.array_equal(std, alias):
        raise ResolutionDataError("FEATURE_SCALE_INVALID", group)
    naive = np.asarray(scales["naive_mae"], dtype=np.float64)
    if scales["horizons"] != list(horizons) or naive.shape != (len(horizons),) \
            or not np.isfinite(naive).all() or np.any(naive <= 0):
        raise ResolutionDataError("NAIVE_SCALE_INVALID", group)


def _common_arrays(bundle, origins, week_ids, mask, horizons):
    p = bundle.price[np.searchsorted(bundle.ts, origins)]
    y = np.full(mask.shape, np.nan, dtype=np.float64)
    for j, h in enumerate(horizons):
        keep = mask[:, j]
        stamps = origins[keep] + h * data.HOUR
        ix = np.searchsorted(bundle.ts, stamps)
        if np.any(ix >= len(bundle.ts)) or not np.array_equal(bundle.ts[ix], stamps):
            raise ResolutionDataError("EXACT_TARGET_MISSING")
        y[keep, j] = bundle.price[ix]
    return dict(mask=mask, origin_price=p, target_price=y, row_ids=origins,
                origin_ts=origins.copy(), week_ids=week_ids)


def _build_group(bundle, base, name, root, files):
    spec = GROUPS[name]
    origins = split = weeks = common = None
    treatments = {}
    # Spool indices, not feature tensors; retain at most one market table.
    for spacing in spec["spacings"]:
        for slots in spec["slots"]:
            key = f"s{spacing}.n{slots}"
            design = _design(base, spacing, slots, spec["horizons"])
            table = market_grid.build_market_table(bundle, design)
            if common is None:
                origins, split, weeks = table.origin_ts.copy(), table.split.copy(), table.week_id.copy()
                common = np.ones_like(table.label_ok)
            if not (np.array_equal(origins, table.origin_ts) and np.array_equal(split, table.split)
                    and np.array_equal(weeks, table.week_id)):
                raise ResolutionDataError("CANDIDATES_DIFFER", key)
            valid = table.label_ok & table.input_ok[:, None]
            common &= valid
            exclusion = dict(candidate_row_ids=table.origin_ts, split=table.split,
                             input_reason=np.asarray([r or "INPUT_OK" for r in table.input_reason]),
                             head_valid=valid)
            excluded_name = f"{name}.{key}.exclusions.npz"
            files[excluded_name] = _save(root, excluded_name, exclusion)
            spool = f".{name}.{key}.indices.npz"
            _save(root, spool, dict(slot_index=table.slot_index))
            treatments[key] = dict(spacing=spacing, slots=slots, exclusions_file=excluded_name,
                exclusions_sha256=sha256_json({k: _digest(v) for k, v in exclusion.items()}),
                data_design_sha256=sha256_json({k: design[k] for k in data.DATA_CONTRACT_KEYS}),
                before_intersection=table.counts)
            del table, exclusion, valid
    keep = common.any(axis=1)
    selections = {s: np.flatnonzero((split == s) & keep) for s in data.SPLITS}
    if not len(selections["FIT"]):
        raise ResolutionDataError("NO_COMMON_FIT_ROWS", name)
    for s, sel in selections.items():
        _require_heads(name, s, common[sel].sum(axis=0).tolist(), spec["horizons"])
    common_files, support = {}, {}
    for s, sel in selections.items():
        arrays = _common_arrays(bundle, origins[sel], weeks[sel], common[sel], spec["horizons"])
        filename = f"{name}.{s}.npz"
        files[filename] = _save(root, filename, arrays)
        common_files[s] = filename
        support[s] = dict(rows=len(sel), counts=common[sel].sum(axis=0).astype(int).tolist(),
                          row_ids_sha256=_digest(origins[sel]), mask_sha256=_digest(common[sel]),
                          exclusions=int(((split == s) & ~keep).sum()))
    exclusion = dict(candidate_row_ids=origins, split=split, common_mask=common, keep=keep)
    group_excluded = f"{name}.exclusions.npz"
    files[group_excluded] = _save(root, group_excluded, exclusion)
    exclusions_digest = sha256_json({k: _digest(v) for k, v in exclusion.items()})
    del exclusion
    union = np.zeros(len(bundle.ts), bool)
    for key, treatment in treatments.items():
        spool = root / f".{name}.{key}.indices.npz"
        indices = _npz(spool)["slot_index"]
        treatment["files"] = {}
        for s, sel in selections.items():
            ix = indices[sel]
            if np.any(ix < 0):
                raise ResolutionDataError("INPUT_SLOT_MISSING", key)
            instants = bundle.ts[ix]
            if not np.array_equal(instants[:, -1], origins[sel]):
                raise ResolutionDataError("ORIGIN_NOT_LAST_SLOT", key)
            offsets = (instants - origins[sel, None]).astype(np.float64) / data.HOUR
            flags = np.zeros(ix.shape, bool)
            flags[:, 1:] = np.diff(instants, axis=1) > treatment["spacing"] * data.HOUR
            filename = f"{name}.{key}.{s}.npz"
            files[filename] = _save(root, filename, dict(x=bundle.x[ix], offsets_hours=offsets,
                                                         closure_crossed=flags))
            treatment["files"][s] = filename
            if s == "FIT":
                union[ix.ravel()] = True
        del indices
        spool.unlink()
    bars = bundle.ts[union]
    fit_end = int(data.split_bounds(base)[0].end.timestamp())
    if not len(bars) or np.any(bars >= fit_end):
        raise ResolutionDataError("SCALER_NOT_FIT_ONLY", name)
    values = bundle.x[union]
    with np.errstate(over="ignore", invalid="ignore"):
        mean, std = values.mean(axis=0).tolist(), values.std(axis=0).tolist()
    fit = _npz(root / common_files["FIT"])
    naive = []
    for j in range(len(spec["horizons"])):
        eligible = fit["mask"][:, j]
        errors = np.abs(fit["target_price"][eligible, j] - fit["origin_price"][eligible])
        try:
            value = math.fsum(errors.tolist()) / len(errors)
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise ResolutionDataError("NAIVE_SCALE_INVALID", f"{name}:h{spec['horizons'][j]}") from exc
        naive.append(value)
    scales = dict(input_mean=mean, std=std, input_std=std,
                  naive_mae=naive, horizons=list(spec["horizons"]))
    _validate_scales(scales, name, spec["horizons"])
    scales = _seal_scales(scales)
    scaler_file = f"{name}.scaler_rows.npz"
    files[scaler_file] = _save(root, scaler_file, dict(source_bar_ts=bars))
    return dict(horizons=list(spec["horizons"]), treatments=treatments, common_files=common_files,
                support=support, support_sha256=sha256_json(support), scales=scales,
                scaler_rows=len(bars), scaler_rows_file=scaler_file, scaler_rows_sha256=_digest(bars),
                exclusions_file=group_excluded, exclusions_sha256=exclusions_digest)


def prepare(input_dir, out_dir, base_design_path):
    """Build both fixed groups privately; existing authenticated preparation is adopted.

    A failed build leaves a private sibling .building directory for diagnosis.
    No partial population is published, overwritten or automatically treated as valid.
    """
    out = _private_destination(out_dir, input_dir)
    base_path = Path(base_design_path).resolve()
    base = data.validate_design(json.loads(base_path.read_text()))
    if any(base["business"].get(k) != 12 for k in
           ("entry_delay_after_open_hours", "force_flat_before_close_hours")):
        raise ResolutionDataError("BUSINESS_OFFSETS_NOT_12H")
    design = _design(base, 4, 24, GROUPS["long"]["horizons"])
    bundle = data.load_input(input_dir, design)
    source = dict(input_dir=str(Path(input_dir).resolve()), files=bundle.files,
                  base_design_sha256=file_sha256(base_path))
    if base_path.is_relative_to(REPO):
        source["base_design_relpath"] = str(base_path.relative_to(REPO))
    else:
        source["base_design_path"] = str(base_path)
    code = _code()
    if out.exists():
        if not (out / "MANIFEST.json").is_file():
            raise ResolutionDataError("OUTPUT_NOT_NEW")
        manifest = _verify(out)
        if manifest["source"] != source or manifest["code_files"] != code:
            raise ResolutionDataError("PREPARATION_IDENTITY_CHANGED")
        return manifest
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    files, groups = {}, {}
    for name in GROUPS:
        groups[name] = _build_group(bundle, base, name, staging, files)
    # Refuse a source/code mutation during a long build before publishing anything.
    if code != _code() or any(file_sha256(Path(input_dir) / n) != h for n, h in bundle.files.items()) \
            or file_sha256(base_path) != source["base_design_sha256"]:
        raise ResolutionDataError("PREPARATION_SOURCE_CHANGED")
    manifest = dict(schema=SCHEMA, source=source, code_files=code, base_design=base,
                    feature_names=list(bundle.features), groups=groups, files=files,
                    test_start_ts=int(data.test_start(base).timestamp()), test_read=False, fits_run=0)
    manifest["manifest_sha256"] = sha256_json(manifest)
    atomic_json(staging / "MANIFEST.json", manifest, mode=0o600)
    if out.exists():
        raise ResolutionDataError("OUTPUT_ALREADY_CREATED")
    staging.rename(out)
    return manifest


def _verify(root, input_dir=None):
    root = Path(root)
    try:
        manifest = json.loads((root / "MANIFEST.json").read_text())
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if manifest.get("schema") != SCHEMA or manifest.get("test_read") is not False \
                or manifest.get("fits_run") != 0 or manifest.get("manifest_sha256") != sha256_json(body):
            raise ResolutionDataError("MANIFEST_DIGEST")
        base = data.validate_design(manifest["base_design"])
        if manifest["test_start_ts"] != int(data.test_start(base).timestamp()) \
                or len(manifest["feature_names"]) != 19:
            raise ResolutionDataError("CONTRACT_CHANGED")
        if manifest["code_files"] != _code():
            raise ResolutionDataError("CODE_DIGEST")
        source = manifest["source"]
        input_root = Path(source["input_dir"] if input_dir is None else input_dir)
        refs = {input_root / n: h for n, h in source["files"].items()}
        if "base_design_relpath" in source:
            relative = Path(source["base_design_relpath"])
            base_path = (REPO / relative).resolve()
            if relative.is_absolute() or not base_path.is_relative_to(REPO):
                raise ResolutionDataError("SOURCE_PATH_ESCAPE")
        else:
            base_path = Path(source["base_design_path"])
        refs[base_path] = source["base_design_sha256"]
        if any(not p.is_file() or file_sha256(p) != h for p, h in refs.items()):
            raise ResolutionDataError("SOURCE_DIGEST")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or not (root / name).is_file() \
                    or file_sha256(root / name) != digest:
                raise ResolutionDataError("FILE_DIGEST", name)
        if set(manifest["groups"]) != set(GROUPS):
            raise ResolutionDataError("GROUPS_CHANGED")
        for name, group in manifest["groups"].items():
            if group["horizons"] != list(GROUPS[name]["horizons"]) \
                    or group["support_sha256"] != sha256_json(group["support"]):
                raise ResolutionDataError("SUPPORT_DIGEST", name)
            for split in data.SPLITS:
                _require_heads(name, split, group["support"][split]["counts"], group["horizons"])
            scales = group["scales"]
            _validate_scales(scales, name, group["horizons"])
            if scales != _seal_scales({k: v for k, v in scales.items() if k != "scales_sha256"}):
                raise ResolutionDataError("SCALES_DIGEST", name)
        return manifest
    except ResolutionDataError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ResolutionDataError("MANIFEST_INVALID", type(exc).__name__) from exc


def _check_arrays(arr, split, slots, spacing, manifest, group):
    n, nh = len(arr["row_ids"]), len(group["horizons"])
    ids, mask, offsets = arr["row_ids"], arr["mask"], arr["offsets_hours"]
    expected = dict(x=(n, slots, 19), mask=(n, nh), origin_price=(n,), target_price=(n, nh),
                    row_ids=(n,), origin_ts=(n,), offsets_hours=(n, slots),
                    closure_crossed=(n, slots), week_ids=(n,))
    if set(arr) != set(expected) or any(arr[k].shape != shape for k, shape in expected.items()):
        raise ResolutionDataError("ARRAY_SHAPE")
    if ids.dtype != np.int64 or mask.dtype != np.bool_ or not np.array_equal(ids, arr["origin_ts"]) \
            or np.any(np.diff(ids) <= 0) or not mask.any(axis=1).all():
        raise ResolutionDataError("ROWS_OR_MASK_INVALID")
    if not np.isfinite(arr["x"]).all() or not np.isfinite(offsets).all() \
            or np.any(offsets > 0) or np.any(offsets[:, -1] != 0) \
            or np.any(np.diff(offsets, axis=1) < spacing):
        raise ResolutionDataError("INPUT_NOT_CAUSAL")
    flags = np.zeros(offsets.shape, bool)
    flags[:, 1:] = np.diff(offsets, axis=1) > spacing
    if not np.array_equal(flags, arr["closure_crossed"]):
        raise ResolutionDataError("CLOSURE_FLAGS_INVALID")
    if not np.array_equal(np.isnan(arr["target_price"]), ~mask) \
            or not np.isfinite(arr["target_price"][mask]).all():
        raise ResolutionDataError("TARGET_MASK_INVALID")
    bounds = next(b for b in data.split_bounds(manifest["base_design"]) if b.split_id == split)
    targets = ids[:, None] + np.asarray(group["horizons"])[None, :] * data.HOUR
    if np.any(ids < int(bounds.start.timestamp())) or np.any(ids >= int(bounds.end.timestamp())) \
            or np.any(targets[mask] >= int(bounds.end.timestamp())) \
            or np.any(targets[mask] >= manifest["test_start_ts"]):
        raise ResolutionDataError("SPLIT_OR_TEST_LEAK")
    support = group["support"][split]
    if support["rows"] != n or support["row_ids_sha256"] != _digest(ids) \
            or support["mask_sha256"] != _digest(mask) or support["counts"] != mask.sum(axis=0).tolist():
        raise ResolutionDataError("SUPPORT_DIGEST", split)


def load(out_dir, group, spacing, slots, horizons=None, *, input_dir=None):
    """Authenticated RAW arrays, common FIT-only scales and support metadata.

    The original input files must remain available and byte-identical. A
    repository-owned base design resolves relative to the executing REPO,
    allowing a frozen-runtime checkout without weakening its byte-hash check.
    An explicit input_dir relocates source reads only; all original source
    file hashes remain mandatory and metadata exposes those hashes, not the
    executing host's relocated path. A bad override never falls back.
    Invalid or repeated horizon requests are refused, never silently sorted.
    """
    if group not in GROUPS or type(spacing) is not int or type(slots) is not int \
            or spacing not in GROUPS[group]["spacings"] or slots not in GROUPS[group]["slots"]:
        raise ResolutionDataError("TREATMENT_NOT_DECLARED")
    full_heads = list(GROUPS[group]["horizons"])
    requested = full_heads if horizons is None else list(horizons)
    if not requested or any(type(h) is not int or h not in full_heads for h in requested) \
            or len(set(requested)) != len(requested):
        raise ResolutionDataError("HORIZONS_NOT_DECLARED")
    manifest = _verify(out_dir, input_dir=input_dir)
    info = manifest["groups"][group]
    treatment = info["treatments"][f"s{spacing}.n{slots}"]
    columns = [full_heads.index(h) for h in requested]
    arrays = {}
    for split in data.SPLITS:
        common = _npz(Path(out_dir) / info["common_files"][split])
        inputs = _npz(Path(out_dir) / treatment["files"][split])
        if set(common) != COMMON_KEYS or set(inputs) != INPUT_KEYS:
            raise ResolutionDataError("ARRAY_KEYS")
        arr = dict(common, **inputs)
        _check_arrays(arr, split, slots, spacing, manifest, info)
        selected_mask = arr["mask"][:, columns]
        keep = selected_mask.any(axis=1)
        arr = {k: v[keep] for k, v in arr.items()}
        arr["mask"] = selected_mask[keep]
        arr["target_price"] = arr["target_price"][:, columns]
        arrays[split] = arr
    full_scales = info["scales"]
    scales = _seal_scales(dict(input_mean=full_scales["input_mean"], std=full_scales["std"],
        input_std=full_scales["input_std"], naive_mae=[full_scales["naive_mae"][j] for j in columns],
        horizons=requested))
    metadata = dict(schema=SCHEMA, manifest_sha256=manifest["manifest_sha256"], group=group,
                    spacing=spacing, slots=slots, horizons=requested, full_horizons=full_heads,
                    features=list(manifest["feature_names"]),
                    source_files=dict(manifest["source"]["files"]),
                    group_support_sha256=info["support_sha256"],
                    group_scales_sha256=full_scales["scales_sha256"],
                    counts={s: a["mask"].sum(axis=0).tolist() for s, a in arrays.items()},
                    test_read=False, raw_inputs=True)
    return arrays, scales, metadata
