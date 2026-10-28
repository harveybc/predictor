"""Fit-free continuation master and retrospective calendar-fold data.

prepare(input_dir, out_dir, base_design_path) writes private NPZ + MANIFEST.
periods(out_dir, cadence) returns frozen, group-independent calendar periods.
load(out, group, spacing, slots, horizons=None, *, input_dir=None, period=None)
returns (arrays, scales, metadata). arrays has ONLY FIT and VALIDATION, with
RAW x, mask, original-price targets/naive origins, integer UTC row_ids and
origin_ts, actual offsets_hours, closure_crossed and week_ids. No INNER exists.
Static FIT uses 2020..2023; dynamic FIT uses exactly four calendar years
ending at the first evaluation week's market_open, with mature labels.

Only two max-24-slot tables per group are built, once. Every planned shorter
window is an exact suffix; all windows/spacings share this conservative
support. Master business labels are not TRAIN statistics. Scales are computed
only on the requested fold's common FIT source-bar union, never on the master
or VALIDATION, and stay fixed when projecting heads. preflight performs the
same fold/statistics checks without materializing feature windows.

Cadence VALIDATION is reused for stopping by explicit owner instruction:
these are RETROSPECTIVE DEVELOPMENT comparisons, not prospective OOS evidence.
CD01..CD06 acceptance and refusal tests live in test_continuation_data.py.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from tools.i6dmo import data, market_grid, resolution_data as R
from tools.i6n_contracts import eligibility as E
from tools.i6n_swarm.common import atomic_json, file_sha256, sha256_json

SCHEMA = "i6dmo.continuation_data.v1"
REPO = Path(__file__).resolve().parents[2]
GROUPS = {name: tuple(spec["horizons"]) for name, spec in R.GROUPS.items()}
SLOTS, SPACINGS = (6, 9, 13, 18, 24), (2, 4)
CADENCES = ("QUARTERLY", "MONTHLY", "WEEKLY")
CODE_FILES = (*R.CODE_FILES, "tools/i6dmo/continuation_data.py")
QUALIFICATION = "RETROSPECTIVE_DEVELOPMENT_NOT_PROSPECTIVE_OOS"


class ContinuationDataError(ValueError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (":" + detail if detail else ""))


class ContinuationDataSkip(ContinuationDataError):
    """An authentic calendar period has no scored pair for the requested heads."""
    disposition = "SKIP"


def _code():
    return {name: file_sha256(REPO / name) for name in CODE_FILES}


def _read(path):
    try:
        return R._npz(path)
    except R.ResolutionDataError as exc:
        raise ContinuationDataError(exc.code, Path(path).name) from exc


def _bounds(base):
    return tuple(int(data._instant(base["splits"][k][edge]).timestamp()) for k, edge in
                 (("train", "start"), ("train", "end"), ("validation", "start"), ("validation", "end")))


def _calendar_periods(calendar, base):
    _, weeks = E.load_calendar(calendar)
    _, _, start, end = _bounds(base)
    evaluation = [w for w in weeks if w.operational_start.timestamp() < end
                  and w.force_flat.timestamp() > start]
    result = {}
    for cadence in CADENCES:
        grouped = {}
        for w in evaluation:
            d = w.operational_start
            key = (f"{d.year}-Q{(d.month - 1) // 3 + 1}" if cadence == "QUARTERLY" else
                   f"{d.year}-{d.month:02d}" if cadence == "MONTHLY" else w.week_id)
            grouped.setdefault(key, []).append(w)
        result[cadence] = []
        for key, batch in grouped.items():
            cutoff = batch[0].market_open
            p = dict(id=f"{cadence}-{key}", cadence=cadence,
                train_start=E.iso(cutoff.replace(year=cutoff.year - 4)), train_end=E.iso(cutoff),
                validation_start=data.iso(max(start, int(batch[0].operational_start.timestamp()))),
                validation_end=data.iso(min(end, int(batch[-1].market_close.timestamp()))),
                week_ids=[w.week_id for w in batch])
            p["period_sha256"] = sha256_json(p)
            result[cadence].append(p)
        if [wid for p in result[cadence] for wid in p["week_ids"]] != [w.week_id for w in evaluation]:
            raise ContinuationDataError("CALENDAR_PARTITION_INVALID", cadence)
    return result


def _selection(frame, start, end, horizons, *, label_end=None, week_ids=None):
    selected = (frame["row_ids"] >= start) & (frame["row_ids"] < end)
    if week_ids is not None:
        selected &= np.isin(frame["week_ids"], week_ids)
    ix = np.flatnonzero(selected)
    mask = frame["mask"][ix].copy()
    if label_end is not None:
        targets = frame["row_ids"][ix, None] + np.asarray(horizons) * data.HOUR
        mask &= targets < label_end
    keep = mask.any(axis=1)
    return ix[keep], mask[keep]


def _require_heads(mask, group, split):
    for h, count in zip(GROUPS[group], mask.sum(axis=0)):
        if not count:
            raise ContinuationDataError("HEAD_UNSUPPORTED", f"{group}:{split}:h{h}")


def _master(bundle, base, group, staging, files):
    origins = weeks = common = None
    spool, exclusions = {}, {}
    broad = E.SplitBounds("FIT", data._instant(data.iso(int(bundle.ts[0]))),
        data.test_start(base), label_available_by=data.test_start(base))
    for spacing in SPACINGS:
        design = R._design(base, spacing, 24, GROUPS[group])
        table = market_grid.build_market_table(bundle, design, evaluation_splits=(broad,))
        if common is None:
            origins, weeks = table.origin_ts.copy(), table.week_id.copy()
            common = np.ones_like(table.label_ok)
        if not np.array_equal(origins, table.origin_ts) or not np.array_equal(weeks, table.week_id):
            raise ContinuationDataError("MASTER_CANDIDATES_DIFFER", group)
        common &= table.label_ok & table.input_ok[:, None]
        spool[spacing] = f".{group}.s{spacing}.indices.npz"
        R._save(staging, spool[spacing], dict(slot_index=table.slot_index))
        name = f"{group}.s{spacing}.exclusions.npz"
        files[name] = R._save(staging, name, dict(origin_ts=table.origin_ts, input_ok=table.input_ok,
            input_reason=np.asarray([r or "INPUT_OK" for r in table.input_reason]), head_valid=table.label_ok))
        exclusions[str(spacing)] = dict(file=name, counts=table.counts,
                                       evaluation_identity=table.identity)
        del table
    keep = common.any(axis=1)
    frame = R._common_arrays(bundle, origins[keep], weeks[keep], common[keep], GROUPS[group])
    train_start, train_end, val_start, val_end = _bounds(base)
    for label, start, end in (("FIT", train_start, train_end), ("VALIDATION", val_start, val_end)):
        _, mask = _selection(frame, start, end, GROUPS[group], label_end=end)
        _require_heads(mask, group, label)
    master_file = f"{group}.master.npz"
    files[master_file] = R._save(staging, master_file, frame)
    indices = {}
    for spacing in SPACINGS:
        path = staging / spool[spacing]
        rows = _read(path)["slot_index"][keep]
        if np.any(rows < 0):
            raise ContinuationDataError("MASTER_INPUT_MISSING", group)
        name = f"{group}.s{spacing}.indices.npz"
        files[name] = R._save(staging, name, dict(slot_index=rows))
        indices[str(spacing)] = name
        path.unlink()
    support = dict(rows=len(frame["row_ids"]), row_ids_sha256=R._digest(frame["row_ids"]),
        mask_sha256=R._digest(frame["mask"]), counts=frame["mask"].sum(axis=0).tolist(),
        excluded_origins=int((~keep).sum()))
    return dict(master_file=master_file, indices=indices, horizons=list(GROUPS[group]),
                support=support, support_sha256=sha256_json(support), exclusions=exclusions)


def prepare(input_dir, out_dir, base_design_path):
    """Read retained pre-TEST input once; publish a four-table master, not global scales."""
    try:
        out = R._private_destination(out_dir, input_dir)
    except R.ResolutionDataError as exc:
        raise ContinuationDataError(exc.code) from exc
    base_path = Path(base_design_path).resolve()
    base = data.validate_design(json.loads(base_path.read_text()))
    if any(base["business"].get(k) != 12 for k in
           ("entry_delay_after_open_hours", "force_flat_before_close_hours")):
        raise ContinuationDataError("BUSINESS_OFFSETS_NOT_12H")
    bundle = data.load_input(input_dir, R._design(base, 4, 24, GROUPS["long"]))
    source = dict(input_dir=str(Path(input_dir).resolve()), files=bundle.files,
                  base_design_sha256=file_sha256(base_path))
    if base_path.is_relative_to(REPO):
        source["base_design_relpath"] = str(base_path.relative_to(REPO))
    else:
        source["base_design_path"] = str(base_path)
    code = _code()
    if out.exists():
        manifest = _verify(out)
        if manifest["source"] != source or manifest["code_files"] != code:
            raise ContinuationDataError("PREPARATION_IDENTITY_CHANGED")
        return manifest
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    files = {"SOURCE.npz": R._save(staging, "SOURCE.npz", dict(ts=bundle.ts, x=bundle.x, price=bundle.price))}
    groups = {g: _master(bundle, base, g, staging, files) for g in GROUPS}
    if code != _code() or any(file_sha256(Path(input_dir) / n) != h for n, h in bundle.files.items()) \
            or file_sha256(base_path) != source["base_design_sha256"]:
        raise ContinuationDataError("PREPARATION_SOURCE_CHANGED")
    manifest = dict(schema=SCHEMA, source=source, code_files=code, files=files, groups=groups,
        base_design=base, calendar=bundle.calendar_config, features=list(bundle.features),
        periods=_calendar_periods(bundle.calendar_config, base), test_read=False, fits_run=0,
        test_start_ts=int(data.test_start(base).timestamp()),
        stopping_qualification=QUALIFICATION, train_statistics="ON_DEMAND_PER_FOLD_ONLY")
    manifest["manifest_sha256"] = sha256_json(manifest)
    atomic_json(staging / "MANIFEST.json", manifest, mode=0o600)
    if out.exists():
        raise ContinuationDataError("OUTPUT_ALREADY_CREATED")
    staging.rename(out)
    return manifest


def _verify(out, input_dir=None):
    root = Path(out)
    try:
        manifest = json.loads((root / "MANIFEST.json").read_text())
        body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if manifest.get("schema") != SCHEMA or manifest.get("manifest_sha256") != sha256_json(body) \
                or manifest.get("test_read") is not False or manifest.get("fits_run") != 0:
            raise ContinuationDataError("MANIFEST_DIGEST")
        base = data.validate_design(manifest["base_design"])
        data.provenance.check_manifest(manifest["features"])
        if manifest["code_files"] != _code():
            raise ContinuationDataError("CODE_DIGEST")
        if manifest["test_start_ts"] != int(data.test_start(base).timestamp()) \
                or manifest["periods"] != _calendar_periods(manifest["calendar"], base):
            raise ContinuationDataError("CALENDAR_PERIOD_IDENTITY")
        source = manifest["source"]
        source_root = Path(source["input_dir"] if input_dir is None else input_dir)
        refs = {source_root / name: digest for name, digest in source["files"].items()}
        if "base_design_relpath" in source:
            rel = Path(source["base_design_relpath"])
            base_path = (REPO / rel).resolve()
            if rel.is_absolute() or not base_path.is_relative_to(REPO):
                raise ContinuationDataError("SOURCE_PATH_ESCAPE")
        else:
            base_path = Path(source["base_design_path"])
        refs[base_path] = source["base_design_sha256"]
        if any(not p.is_file() or file_sha256(p) != h for p, h in refs.items()):
            raise ContinuationDataError("SOURCE_DIGEST")
        if json.loads((source_root / "CALENDAR.json").read_text()) != manifest["calendar"] \
                or data.validate_design(json.loads(base_path.read_text())) != base:
            raise ContinuationDataError("SOURCE_CONTRACT_IDENTITY")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or not (root / name).is_file() or file_sha256(root / name) != digest:
                raise ContinuationDataError("FILE_DIGEST", name)
        if set(manifest["groups"]) != set(GROUPS):
            raise ContinuationDataError("GROUP_IDENTITY")
        for g, info in manifest["groups"].items():
            if info["horizons"] != list(GROUPS[g]) or info["support_sha256"] != sha256_json(info["support"]):
                raise ContinuationDataError("SUPPORT_DIGEST", g)
        return manifest
    except ContinuationDataError:
        raise
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise ContinuationDataError("MANIFEST_INVALID", type(exc).__name__) from exc


def periods(out, cadence, *, input_dir=None):
    """Return all actual market-week periods, including partial-year support."""
    cadence = str(cadence).upper()
    if cadence not in CADENCES:
        raise ContinuationDataError("CADENCE_NOT_DECLARED", cadence)
    return copy.deepcopy(_verify(out, input_dir)["periods"][cadence])


def _period(manifest, period):
    if period is None:
        a, b, c, d = _bounds(manifest["base_design"])
        return dict(id="STATIC", cadence="STATIC", train_start=data.iso(a), train_end=data.iso(b),
                    validation_start=data.iso(c), validation_end=data.iso(d), week_ids=None)
    if not isinstance(period, dict) or period.get("cadence") not in CADENCES \
            or period not in manifest["periods"][period["cadence"]]:
        raise ContinuationDataError("PERIOD_IDENTITY")
    start, end = (data._instant(period[k]) for k in ("train_start", "train_end"))
    if start != end.replace(year=end.year - 4):
        raise ContinuationDataError("TRAIN_NOT_FOUR_CALENDAR_YEARS")
    return copy.deepcopy(period)


def _fold(out, group, input_dir, period):
    if group not in GROUPS:
        raise ContinuationDataError("GROUP_NOT_DECLARED")
    manifest = _verify(out, input_dir)
    period = _period(manifest, period)
    info = manifest["groups"][group]
    frame = _read(Path(out) / info["master_file"])
    if set(frame) != R.COMMON_KEYS or frame["row_ids"].dtype != np.int64 \
            or not np.array_equal(frame["row_ids"], frame["origin_ts"]) \
            or np.any(np.diff(frame["row_ids"]) <= 0) or frame["mask"].dtype != np.bool_ \
            or R._digest(frame["row_ids"]) != info["support"]["row_ids_sha256"] \
            or R._digest(frame["mask"]) != info["support"]["mask_sha256"]:
        raise ContinuationDataError("MASTER_SUPPORT_INVALID")
    train_start, train_end = (_epoch(period[k]) for k in ("train_start", "train_end"))
    _, _, val_start, val_end = _bounds(manifest["base_design"])
    train = _selection(frame, train_start, train_end, GROUPS[group], label_end=train_end)
    validation = _selection(frame, val_start, val_end, GROUPS[group], label_end=val_end,
                            week_ids=period["week_ids"])
    if not len(validation[0]):
        raise ContinuationDataSkip("EMPTY_VALIDATION_PERIOD", f"{group}:{period['id']}")
    _require_heads(train[1], group, "FIT")
    if period["id"] == "STATIC":
        _require_heads(validation[1], group, "VALIDATION")
    elif train_end > int(frame["row_ids"][validation[0]].min()):
        raise ContinuationDataError("TRAIN_CUTOFF_AFTER_SCORE")
    source = _read(Path(out) / "SOURCE.npz")
    if set(source) != {"ts", "x", "price"} or source["ts"].dtype != np.int64 \
            or source["x"].shape != (len(source["ts"]), 19) \
            or np.any(np.diff(source["ts"]) <= 0) or np.any(source["ts"] >= manifest["test_start_ts"]):
        raise ContinuationDataError("SOURCE_ARRAYS_OR_TEST_LEAK")
    if source["ts"][0] > train_start - 92 * data.HOUR:
        raise ContinuationDataError("TRAIN_COVERAGE_INCOMPLETE")
    union = np.zeros(len(source["ts"]), bool)
    for spacing in SPACINGS:
        ix = _read(Path(out) / info["indices"][str(spacing)])["slot_index"][train[0]]
        if ix.shape != (len(train[0]), 24) or np.any(ix < 0) or np.any(ix >= len(union)):
            raise ContinuationDataError("MASTER_INDICES_INVALID")
        union[ix.ravel()] = True
    if np.any(source["ts"][union] >= train_end):
        raise ContinuationDataError("SCALER_NOT_TRAIN_ONLY")
    values = source["x"][union]
    with np.errstate(over="ignore", invalid="ignore"):
        mean, std = values.mean(axis=0).tolist(), values.std(axis=0).tolist()
    fit_ids, mask = train
    naive = []
    for j, h in enumerate(GROUPS[group]):
        used = fit_ids[mask[:, j]]
        errors = np.abs(frame["target_price"][used, j] - frame["origin_price"][used])
        try:
            naive.append(math.fsum(errors.tolist()) / len(errors))
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise ContinuationDataError("NAIVE_SCALE_INVALID", f"{group}:h{h}") from exc
    scales = dict(input_mean=mean, std=std, input_std=std, naive_mae=naive, horizons=list(GROUPS[group]))
    try:
        R._validate_scales(scales, group, GROUPS[group])
    except R.ResolutionDataError as exc:
        raise ContinuationDataError(exc.code, group) from exc
    return manifest, period, info, frame, source, {"FIT": train, "VALIDATION": validation}, R._seal_scales(scales)


def _epoch(text):
    return int(data._instant(text).timestamp())


def _heads(group, horizons):
    if group not in GROUPS:
        raise ContinuationDataError("GROUP_NOT_DECLARED")
    requested = list(GROUPS[group]) if horizons is None else list(horizons)
    if not requested or len(set(requested)) != len(requested) \
            or any(type(h) is not int or h not in GROUPS[group] for h in requested):
        raise ContinuationDataError("HORIZONS_NOT_DECLARED")
    return requested, [GROUPS[group].index(h) for h in requested]


def _project(selections, columns, group, period):
    result = {}
    for split, (ix, mask) in selections.items():
        mask = mask[:, columns]
        keep = mask.any(axis=1)
        result[split] = ix[keep], mask[keep]
    if not len(result["VALIDATION"][0]):
        raise ContinuationDataSkip("EMPTY_VALIDATION_PERIOD", f"{group}:{period['id']}")
    return result


def preflight(out, group, horizons=None, *, input_dir=None, period=None):
    """Typed pre-fit checks of coverage, authentic periods, masks and TRAIN scales."""
    heads, columns = _heads(group, horizons)
    manifest, p, info, frame, source, selections, scales = _fold(out, group, input_dir, period)
    selected = _project(selections, columns, group, p)
    return dict(state="PASS", group=group, period=p, horizons=heads,
        manifest_sha256=manifest["manifest_sha256"], group_support_sha256=info["support_sha256"],
        group_scales_sha256=scales["scales_sha256"],
        counts={s: m.sum(axis=0).tolist() for s, (_, m) in selected.items()},
        test_read=False, validation_used_for_stopping=True, qualification=QUALIFICATION)


def load(out, group, spacing, slots, horizons=None, *, input_dir=None, period=None):
    """Materialize one exact suffix and fold; no market table is rebuilt here."""
    heads, columns = _heads(group, horizons)
    if type(spacing) is not int or spacing not in SPACINGS or type(slots) is not int or slots not in SLOTS:
        raise ContinuationDataError("TREATMENT_NOT_DECLARED")
    manifest, p, info, frame, source, selections, full_scales = _fold(out, group, input_dir, period)
    selections = _project(selections, columns, group, p)
    indices = _read(Path(out) / info["indices"][str(spacing)])["slot_index"]
    arrays = {}
    for split, (rows, mask) in selections.items():
        ix = indices[rows, -slots:]
        if ix.shape != (len(rows), slots) or np.any(ix < 0) or np.any(ix >= len(source["ts"])):
            raise ContinuationDataError("MASTER_INDICES_INVALID")
        instants = source["ts"][ix]
        origins = frame["row_ids"][rows]
        if not np.array_equal(instants[:, -1], origins) \
                or np.any(instants > origins[:, None]) or np.any(np.diff(instants, axis=1) < spacing * data.HOUR):
            raise ContinuationDataError("INPUT_NOT_EXACT_CAUSAL")
        x = source["x"][ix]
        if not np.isfinite(x).all():
            raise ContinuationDataError("INPUT_NOT_FINITE")
        flags = np.zeros(ix.shape, bool)
        flags[:, 1:] = np.diff(instants, axis=1) > spacing * data.HOUR
        target = frame["target_price"][rows][:, columns].copy()
        target[~mask] = np.nan
        if not np.isfinite(target[mask]).all() or np.any(
                (origins[:, None] + np.asarray(heads) * data.HOUR)[mask] >=
                (_epoch(p["train_end"]) if split == "FIT" else manifest["test_start_ts"])):
            raise ContinuationDataError("TARGET_MATURITY_OR_TEST_LEAK")
        arrays[split] = dict(x=x, mask=mask, origin_price=frame["origin_price"][rows], target_price=target,
            row_ids=origins, origin_ts=origins.copy(), offsets_hours=(instants - origins[:, None]) / data.HOUR,
            closure_crossed=flags, week_ids=frame["week_ids"][rows])
    scales = R._seal_scales(dict(input_mean=full_scales["input_mean"], std=full_scales["std"],
        input_std=full_scales["input_std"], naive_mae=[full_scales["naive_mae"][j] for j in columns], horizons=heads))
    meta = dict(schema=SCHEMA, manifest_sha256=manifest["manifest_sha256"], features=manifest["features"],
        source_files=dict(manifest["source"]["files"]), group=group, spacing=spacing, slots=slots,
        horizons=heads, full_horizons=list(GROUPS[group]), group_support_sha256=info["support_sha256"],
        group_scales_sha256=full_scales["scales_sha256"], scales_sha256=scales["scales_sha256"], period=p,
        train_start=p["train_start"], train_end=p["train_end"], validation_used_for_stopping=True,
        qualification=QUALIFICATION, test_read=False,
        counts={s: a["mask"].sum(axis=0).tolist() for s, a in arrays.items()},
        validation_counts=arrays["VALIDATION"]["mask"].sum(axis=0).tolist())
    return arrays, scales, meta
