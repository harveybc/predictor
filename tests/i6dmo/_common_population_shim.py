"""LOCAL TEST SHIM of the data lane's ``tools/i6dmo/common_population.py`` interface.

Model-lane tests only. DELETE this file once the data lane's module exists in the integration
worktree. It implements exactly the interface the model lane codes against:

* ``load_common_population(path, *, expected_digest) -> CommonPopulation`` with
  ``schema="i6dmo.common_population.v1"``, ``horizons=(2,3,4,5,6,12)``,
  ``splits: dict[str, SplitPopulation]`` for FIT/INNER/DEVELOPMENT, ``scaler: SharedScaler``,
  ``naive_loss_scale: dict[int, float]``, ``identities: dict`` and ``digest``;
* the per-arm accessor ``arm_split_arrays(population, arm_dir, split, *, design) -> dict``
  (row_id, x, offsets_hours, closure_crossed, slot_instants, origin_price, target_price, mask,
  week_ids) covering at least the common rows.

``build_fixture`` freezes a tiny common population from a REAL MG2 market bundle (built by the
data lane's ``market_grid`` from synthetic inputs) and the same synthetic hourly input for the
25-slot MG1 arm. The on-disk format here is the shim's own; nothing outside the tests reads it.
Importing this module never imports TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, Mapping, Tuple

import numpy as np

SCHEMA = "i6dmo.common_population.v1"
HORIZONS = (2, 3, 4, 5, 6, 12)
SPLITS = ("FIT", "INNER", "DEVELOPMENT")
BUNDLE_SPLIT = {"FIT": "fit", "INNER": "inner", "DEVELOPMENT": "validation"}
SHIM = True


class CommonPopulationError(ValueError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class SplitPopulation:
    name: str
    origin_row_ids: np.ndarray
    head_mask: np.ndarray
    entry_mask: np.ndarray
    management_mask: np.ndarray
    counts: Dict[int, int]
    digest: str


@dataclass(frozen=True)
class SharedScaler:
    feature_names: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    fit_row_digest: str
    n_rows: int


@dataclass(frozen=True)
class CommonPopulation:
    schema: str
    horizons: Tuple[int, ...]
    splits: Dict[str, SplitPopulation]
    scaler: SharedScaler
    naive_loss_scale: Dict[int, float]
    identities: dict
    digest: str


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _arr_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(_canonical([list(value.shape), str(value.dtype)]).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def _split_digest(name, ids, head, entry, mgmt):
    return hashlib.sha256(_canonical({"name": name, "ids": _arr_sha(ids), "head": _arr_sha(head),
                                      "entry": _arr_sha(entry), "mgmt": _arr_sha(mgmt)}).encode()).hexdigest()


def _body(meta, arrays):
    return {"meta": meta, "arrays": {k: _arr_sha(v) for k, v in sorted(arrays.items())}}


def write_population(out_dir, splits, scaler, naive, identities):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    arrays, split_meta = {}, {}
    for name in SPLITS:
        ids, head, entry, mgmt = splits[name]
        arrays.update({f"{name}_ids": ids.astype(np.int64), f"{name}_head": head.astype(bool),
                       f"{name}_entry": entry.astype(bool), f"{name}_mgmt": mgmt.astype(bool)})
        split_meta[name] = {"digest": _split_digest(name, ids.astype(np.int64), head, entry, mgmt),
                            "counts": {str(h): int(head[:, j].sum()) for j, h in enumerate(HORIZONS)}}
    arrays["scaler_mean"] = np.asarray(scaler["mean"], np.float64)
    arrays["scaler_std"] = np.asarray(scaler["std"], np.float64)
    meta = {"schema": SCHEMA, "horizons": list(HORIZONS), "splits": split_meta,
            "scaler": {"feature_names": list(scaler["feature_names"]),
                       "fit_row_digest": scaler["fit_row_digest"], "n_rows": int(scaler["n_rows"])},
            "naive_loss_scale": {str(h): float(naive[h]) for h in HORIZONS},
            "identities": identities}
    np.savez(out / "POPULATION.npz", **arrays)
    digest = hashlib.sha256(_canonical(_body(meta, arrays)).encode()).hexdigest()
    (out / "POPULATION.json").write_text(json.dumps(dict(meta, digest=digest), indent=1, sort_keys=True))
    return digest


def load_common_population(path, *, expected_digest):
    root = Path(path)
    doc = json.loads((root / "POPULATION.json").read_text())
    with np.load(root / "POPULATION.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = {k: v for k, v in doc.items() if k != "digest"}
    digest = hashlib.sha256(_canonical(_body(meta, arrays)).encode()).hexdigest()
    if digest != doc.get("digest") or digest != expected_digest:
        raise CommonPopulationError("COMMON_POPULATION_DIGEST",
                                    f"content {digest} / declared {doc.get('digest')} / expected {expected_digest}")
    splits = {}
    for name in SPLITS:
        ids = arrays[f"{name}_ids"]
        head, entry, mgmt = (arrays[f"{name}_{k}"] for k in ("head", "entry", "mgmt"))
        splits[name] = SplitPopulation(name, ids, head, entry, mgmt,
                                       {int(k): int(v) for k, v in meta["splits"][name]["counts"].items()},
                                       meta["splits"][name]["digest"])
    sc = meta["scaler"]
    return CommonPopulation(SCHEMA, tuple(meta["horizons"]), splits,
                            SharedScaler(tuple(sc["feature_names"]), arrays["scaler_mean"],
                                         arrays["scaler_std"], sc["fit_row_digest"], sc["n_rows"]),
                            {int(k): float(v) for k, v in meta["naive_loss_scale"].items()},
                            meta["identities"], digest)


def write_arm(out_dir, rule, per_split):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    for name, arrays in per_split.items():
        np.savez(out / f"{name}.npz", **arrays)
    (out / "ARM.json").write_text(json.dumps({"schema": "shim.arm.v1", "rule": rule}))
    return out


def arm_split_arrays(population, arm_dir, split, *, design):
    root = Path(arm_dir)
    doc = json.loads((root / "ARM.json").read_text())
    rule = ((design or {}).get("market_time") or {}).get("rule")
    if doc["rule"] != rule:
        raise CommonPopulationError("ARM_RULE_MISMATCH", f"{doc['rule']} != {rule}")
    with np.load(root / f"{split}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def install():
    """Expose this shim as ``tools.i6dmo.common_population`` ONLY when the real module is absent."""
    import tools.i6dmo as pkg
    name = "tools.i6dmo.common_population"
    if name in sys.modules:
        return sys.modules[name]
    if importlib.util.find_spec(name) is not None:
        raise RuntimeError("the data lane's common_population module exists: delete this shim")
    module = sys.modules[__name__]
    sys.modules[name] = module
    setattr(pkg, "common_population", module)
    return module


# --------------------------------------------------------------------------- fixture builder

def _subsample(n_total, n):
    if n_total <= n:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, n).round().astype(int))


def build_fixture(root, *, mg2_bundle, input_dir, sizes=(96, 40, 64), exclude_every=4,
                  designs=None):
    """Freeze a tiny common population + MG1/MG2 shim arm sources.

    Common rows: MG2 market-bundle origins whose 25 hourly MG1 slots are the 25 preceding
    expected open-market bars (synthetic input has no unplanned gaps), subsampled. The common
    head mask is the arms' identical label mask minus a frozen extra exclusion on the 12 h head
    of every ``exclude_every``-th row, so some labels are valid under an arm but outside the
    common population (the masked-out perturbation targets).
    """
    from tools.i6dmo import market_grid

    root = Path(root)
    with np.load(Path(input_dir) / "INPUT.npz", allow_pickle=False) as z:
        ts, xs = z["ts"].astype(np.int64), z["x"].astype(np.float64)
    pos = {int(t): i for i, t in enumerate(ts)}
    manifest = market_grid.load_market_manifest(mg2_bundle)
    features = tuple(manifest["features"])
    splits, mg1_arm, mg2_arm, fit_bars = {}, {}, {}, set()
    for name, size in zip(SPLITS, sizes):
        raw = market_grid.load_market_split(mg2_bundle, BUNDLE_SPLIT[name])
        origin = raw["row_id"].astype(np.int64)
        ok = []
        for r, t in enumerate(origin):
            i = pos.get(int(t))
            if i is None or i < 24:
                continue
            hourly = ts[i - 24:i + 1]
            if not np.array_equal(raw["slot_instants"][r], hourly[::2]):
                continue                     # MG2 slots must be every other MG1 slot
            if not raw["mask"][r].any():
                continue
            ok.append(r)
        ok = np.asarray(ok)[_subsample(len(ok), size)]
        head = raw["mask"][ok].copy()
        extra = np.zeros_like(head)
        extra[::exclude_every, HORIZONS.index(12)] = True
        extra &= head
        common = head & ~extra
        keep = common.any(axis=1)
        ok, common, head = ok[keep], common[keep], head[keep]
        ids = origin[ok]
        entry = raw["entry_mask"][ok] & common
        splits[name] = (ids, common, entry, common.copy())
        idx = np.asarray([pos[int(t)] for t in ids])
        slot_idx = idx[:, None] + np.arange(-24, 1)[None, :]
        instants = ts[slot_idx]
        offsets = (instants - ids[:, None]).astype(np.float64) / 3600.0
        closure = np.zeros(offsets.shape, np.int64)
        closure[:, 1:] = (np.diff(offsets, axis=1) > 1.0).astype(np.int64)
        shared = {"row_id": ids, "origin_price": raw["origin_price"][ok],
                  "target_price": raw["target_price"][ok], "mask": head,
                  "week_ids": raw["week_ids"][ok]}
        mg1_arm[name] = dict(shared, x=xs[slot_idx], offsets_hours=offsets,
                             closure_crossed=closure, slot_instants=instants)
        mg2_arm[name] = dict(shared, x=raw["x"][ok], offsets_hours=raw["offsets_hours"][ok],
                             closure_crossed=raw["closure_crossed"][ok],
                             slot_instants=raw["slot_instants"][ok])
        if name == "FIT":
            fit_bars.update(int(b) for b in instants.reshape(-1))
    bars = np.asarray(sorted(fit_bars), np.int64)
    rows = np.asarray([pos[int(b)] for b in bars])
    union = xs[rows]
    fit_ids, fit_mask, _, _ = splits["FIT"]
    fit = mg1_arm["FIT"]
    naive = {}
    for j, h in enumerate(HORIZONS):
        sel = fit_mask[:, j]
        import math
        naive[h] = math.fsum(np.abs(fit["target_price"][sel, j] - fit["origin_price"][sel]).tolist()) / int(sel.sum())
    scaler = {"feature_names": features, "mean": union.mean(axis=0), "std": union.std(axis=0),
              "fit_row_digest": hashlib.sha256(bars.tobytes()).hexdigest(), "n_rows": int(len(bars))}
    identities = {"calendar": {"calendar_id": manifest["calendar"]["calendar_id"]},
                  "arms": {arm: {"rule": rule, "design_sha256": (designs or {}).get(arm)}
                           for arm, rule in (("MG1", "MG1_MARKET_EXACT"), ("MG2", "MG2_MARKET_EXACT"))},
                  "shim": True}
    digest = write_population(root / "population", splits, scaler, naive, identities)
    write_arm(root / "arm_mg1", "MG1_MARKET_EXACT", mg1_arm)
    write_arm(root / "arm_mg2", "MG2_MARKET_EXACT", mg2_arm)
    return {"population": root / "population", "digest": digest, "arm_mg1": root / "arm_mg1",
            "arm_mg2": root / "arm_mg2", "extra_excluded": True}


def perturb_arm(src, dst, edits):
    """Copy a shim arm source, applying ``edits``: [(split, array, (row, col...), value)]."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    (dst / "ARM.json").write_text((src / "ARM.json").read_text())
    for name in SPLITS:
        with np.load(src / f"{name}.npz", allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        for split, key, index, value in edits:
            if split == name:
                arrays[key][index] = value
        np.savez(dst / f"{name}.npz", **arrays)
    return dst
