#!/usr/bin/env python
"""Export a modular temporal encoder to a framework-neutral .npz (modular_encoder_export.v1).

Run with the CAMPAIGN interpreter (the one whose Keras major.minor matches the
bundle): engine bundles at pin 3ecdb256 are saved under Keras 3.13 and
``load_bundle`` refuses a mismatch by design. The export carries every branch
and core weight by Keras path, the normalized config, a fixed reference input
and the reference fused/latent outputs, with sha256 per array, so a torch
consumer (agent-multi ``rl_temporal.keras_import``) can import the weights and
prove fidelity without Keras.

    python tools/export_modular_encoder_npz.py --bundle <dir> --out encoder.npz
    python tools/export_modular_encoder_npz.py --config <modular_config.json> --out encoder.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXPORT_SCHEMA = "modular_encoder_export.v1"


# encoder-changing options the torch consumer (agent-multi rl_temporal) does not implement yet
TORCH_UNSUPPORTED_ENCODER_KEYS = ("input_normalization",)


def _identity(bundle, provenance, bypass):
    from predictor_plugins.modular_temporal import config_digest, regime_summary
    from predictor_plugins.modular_temporal.common import _digest
    c = bundle.config
    unsupported = [k for k in TORCH_UNSUPPORTED_ENCODER_KEYS if c.get(k)]
    if unsupported:
        raise ValueError(f"NOT_SUPPORTED_BY_TORCH_CONSUMER: {unsupported} change the encoder and the torch "
                         "extractor does not implement them; refusing a silently different export")
    manifests = {name: _digest(bundle.donor_manifest("branch", name)) for name in bundle.branch_models}
    manifests["core"] = _digest(bundle.donor_manifest("core"))
    return {"config_sha256": config_digest(c), "regimes": regime_summary(c), "provenance": provenance,
            "provenance_bypass": bypass, "component_manifest_sha256": manifests,
            "head_only_options": {k: c[k] for k in ("target_residual",) if c.get(k)}}


def export_bundle(bundle, path: Path, *, seed: int = 0, batch: int = 3, identity=None) -> dict:
    import keras
    import tensorflow as tf

    c = bundle.config
    arrays = {}
    for name, model in bundle.branch_models.items():
        for w in model.weights:
            arrays[f"branch:{name}:{w.path}"] = np.asarray(w.numpy(), dtype=np.float32)
    for w in bundle.core_model.weights:
        arrays[f"core:{w.path}"] = np.asarray(w.numpy(), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(int(batch), int(c["window"]), len(c["feature_names"]))).astype(np.float32)
    arrays["reference:input"] = x
    arrays["reference:fused"] = np.asarray(bundle.fusion_model(x), dtype=np.float32)
    arrays["reference:latent"] = np.asarray(bundle.encoder_model(x), dtype=np.float32)
    meta = {"schema": EXPORT_SCHEMA, "modular_config": c,
            "versions": {"keras": keras.__version__, "tensorflow": tf.__version__, "python": sys.version.split()[0]},
            "weights_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
            "reference_seed": int(seed), "reference_batch": int(batch), "identity": identity}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, __meta__=np.array(json.dumps(meta, sort_keys=True, default=str)), **arrays)
    receipt = {**meta, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "array_count": len(arrays)}
    receipt.pop("weights_sha256")
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", help="saved bundle directory (bundle.json + forecast_model.keras)")
    src.add_argument("--config", help="modular config JSON: build a fresh (random-init) encoder and export it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--receipt", help="write the receipt JSON here (default: <out>.receipt.json)")
    ap.add_argument("--allow-unknown-provenance", action="store_true",
                    help="export a bundle whose provenance is not OPERATIONAL; recorded as a bypass")
    args = ap.parse_args(argv)
    from predictor_plugins.modular_temporal.assembly import build_modular
    from predictor_plugins.modular_temporal.bundle import load_bundle

    if args.bundle:
        bypass = "UNKNOWN_ALLOWED" if args.allow_unknown_provenance else None
        bundle, document = load_bundle(args.bundle, require_contract=None if bypass else "OPERATIONAL")
        source = {"kind": "bundle", "path": args.bundle, "weights_sha256": document["weights_sha256"],
                  "keras_version_saved": document["keras_version"]}
        identity = _identity(bundle, document["provenance"], bypass)
    else:
        config = json.loads(Path(args.config).read_text())
        import tensorflow as tf
        tf.random.set_seed(int(args.seed))
        bundle = build_modular(config)
        source = {"kind": "config_random_init", "path": args.config, "seed": int(args.seed),
                  "label": "RANDOM_INIT_NOT_PRETRAINED"}
        identity = _identity(bundle, {"conditioning_contract": "NOT_APPLICABLE_RANDOM_INIT"}, None)
    out = Path(args.out)
    receipt = export_bundle(bundle, out, seed=args.seed, identity=identity)
    receipt["source"] = source
    rpath = Path(args.receipt) if args.receipt else out.with_suffix(out.suffix + ".receipt.json")
    rpath.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"out": str(out), "sha256": receipt["sha256"], "arrays": receipt["array_count"],
                      "versions": receipt["versions"], "receipt": str(rpath)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
