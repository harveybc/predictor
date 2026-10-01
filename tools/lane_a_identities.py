"""Regenerate candidate, donor and fused-materialization identities under the integrated engine,
and remeasure fused materialization with 24 retained steps (engineering pilot, no fitting).

    python tools/lane_a_identities.py --npz <validation.npz> --data-manifest <DATA_MANIFEST.json>
        --m04-fixtures tests/fixtures/m04_3ceabfad --out <dir> [--measure-rows 512 --batch 64]

* Candidate: M04's pinned default flat point corrected to the approved design
  (model.branch_steps 24, core time factors [2,2,1]; everything else as pinned),
  mapped through M04's own ``from_flat`` with the REAL ordered channel names read
  from the NPZ (checked against the data manifest's channel_order_sha256 when the
  manifest records how it was computed; reported either way).
* Donor identities: every branch donor manifest digest (weight-independent), the
  ordered-list digest over them, and the core manifest digest WITHOUT upstream
  weight hashes (the full core identity binds the donor weights M02 produces).
* Fused materialization identity: fusion identity, per-row shape/dtype, branch
  grid and the ordered branch-identity digest.
* Size: analytic bytes for the declared train/validation windows, and a measured
  memmap write of ``--measure-rows`` real validation windows through the R0
  fusion model (file bytes, seconds/row, peak RSS). Random R0 weights: this
  prices I/O and memory only, not any representation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--data-manifest", required=True)
    p.add_argument("--m04-fixtures", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--measure-rows", type=int, default=512)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(a.data_manifest).read_text())
    with np.load(a.npz, allow_pickle=False) as z:
        names = [str(n) for n in z["feature_names"]]
        windows_val = z["windows"]
        rows = windows_val[:a.measure_rows].astype("float32")
    order_sha = {"comma_joined_m04_recipe": hashlib.sha256(",".join(names).encode()).hexdigest(),
                 "json_compact": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
                 "newline_joined": hashlib.sha256("\n".join(names).encode()).hexdigest()}

    from predictor_plugins import modular_config as mc
    from predictor_plugins import modular_temporal as mt
    fx = Path(a.m04_fixtures)
    spec = importlib.util.spec_from_file_location("m04_space", fx / "modular_search_space.py")
    m04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m04)
    space = json.loads((fx / "ecl_l24_h24_search_space_v1.json").read_text())
    flat = json.loads((fx / "ecl_l24_h24_default_r0_v1.json").read_text())
    flat.update({"train.seed": 2021, "model.branch_steps": 24, "core.time_factor_0": 2,
                 "core.time_factor_1": 2, "core.time_factor_2": 1})
    if flat.get("train.loss") == "huber":
        flat.setdefault("train.huber_delta", 1.0)
    base = {"feature_names": names, "window": manifest["window"], "sample_hours": manifest["sample_hours"],
            "horizons": manifest["horizons"], "target_feature_indices": list(range(len(names))),
            "objective": {"metric": "MAE", "split": "validation", "higher_is_better": False},
            "evaluator_fixed": {}}
    nested = m04.from_flat(flat, base, space)
    model_config = nested["model"]
    t0 = time.monotonic()
    b = mt.build_modular(model_config)
    build_seconds = time.monotonic() - t0
    branch_digests = [mt._digest(b.donor_manifest("branch", n)) for n in b.branch_models]
    core = b.donor_manifest("core")
    core_wf = dict(core)
    core_wf["upstream"] = {"branches": [e["manifest"] for e in core["upstream"]["branches"]],
                           "fusion": core["upstream"]["fusion"]["identity"]}
    per_row = [int(d) for d in b.fusion_model.output.shape[1:]]
    fused_identity = {"fusion": b.component_manifests()["fusion"]["plugin"], "row_shape": per_row,
                      "dtype": "float32", "grid": list(b.branch_time_grid),
                      "branch_identities_sha256": mt._digest(branch_digests)}
    bytes_per_row = int(np.prod(per_row)) * 4
    train_n = manifest["splits"]["train"]["windows"]
    val_n = manifest["splits"]["validation"]["windows"]

    path = out / "fused_probe.npy"
    mm = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(len(rows), *per_row))
    t0 = time.monotonic()
    for s in range(0, len(rows), a.batch):
        mm[s:s + a.batch] = np.asarray(b.fusion_model(rows[s:s + a.batch], training=False))
    mm.flush()
    seconds = time.monotonic() - t0
    file_bytes = path.stat().st_size
    del mm
    path.unlink()
    report = {
        "engine": {"keras_version": mt.keras_version(), "components": {
            r: mt.describe_component(r, n)["version"] for r, n in
            (("branch", "causal_conv1d"), ("fusion", "sequence_concat"), ("core", "transformer_conv"),
             ("head", "forecast"))}},
        "data": {"dataset_id": manifest["dataset_id"], "channels": len(names),
                 "manifest_channel_order_sha256": manifest.get("channel_order_sha256"),
                 "recomputed_channel_order_sha256": order_sha,
                 "train_windows": train_n, "validation_windows": val_n},
        "candidate": {"flat_point": flat, "corrections_from_pinned_default": {
            "model.branch_steps": [12, 24], "core.time_factors": [[2, 1, 1], [2, 2, 1]]},
            "model_config_sha256": mt.config_digest(model_config),
            "candidate_sha256_m04": m04.config_identity(nested),
            "flat_keys_m01": len(mc.flatten(model_config))},
        "donors": {"branch_count": len(branch_digests), "branch_manifest_sha256_first": branch_digests[:2],
                   "branch_manifest_list_sha256": mt._digest(branch_digests),
                   "core_manifest_weight_free_sha256": mt._digest(core_wf),
                   "note": "full core identity also binds upstream donor weight hashes (set by M02's donors)"},
        "fused_materialization": {
            "identity": fused_identity, "identity_sha256": mt._digest(fused_identity),
            "bytes_per_row": bytes_per_row,
            "analytic_bytes": {"train": train_n * bytes_per_row, "validation": val_n * bytes_per_row},
            "previous_12_step_analytic_bytes": {"train": train_n * bytes_per_row // 2,
                                                "validation": val_n * bytes_per_row // 2},
            "measured": {"rows": len(rows), "batch": a.batch, "file_bytes": file_bytes,
                         "npy_header_bytes": file_bytes - len(rows) * bytes_per_row,
                         "seconds": seconds, "seconds_per_row": seconds / len(rows),
                         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
            "scope": "engineering pilot: R0 random weights, real validation windows; prices I/O and memory only"},
        "build_seconds_F": build_seconds,
    }
    (out / "LANE_A_IDENTITIES.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("data", "donors")}, indent=1)[:2000])
    print(json.dumps(report["fused_materialization"], indent=1))


if __name__ == "__main__":
    main()
