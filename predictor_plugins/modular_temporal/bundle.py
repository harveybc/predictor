"""Whole-model bundle serialization: archive + canonical config + component manifests."""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from .assembly import build_modular
from .common import _copy, _file_hash, _major_minor, keras_version, weights_hash
from .config import _normalize

keras = tf.keras
from .provenance import BUNDLE_V1, BUNDLE_V2, bundle_provenance, complete

BUNDLE_SCHEMA = BUNDLE_V2


def save_bundle(bundle, directory, provenance=None):
    """Write forecast archive, canonical config, component manifests and weight identity."""
    declared = complete(provenance)                    # validated before anything is written
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "forecast_model.keras"
    bundle.forecast_model.save(archive)
    document = {"schema": BUNDLE_SCHEMA, "config": _normalize(bundle.config),
                "keras_version": keras_version(), "provenance": declared,
                "components": bundle.component_manifests(),
                "weights_sha256": weights_hash(bundle.forecast_model),
                "archive_sha256": _file_hash(archive)}
    (out / "bundle.json").write_text(json.dumps(document, sort_keys=True, indent=2) + "\n",
                                     encoding="utf-8")
    return document


def load_bundle(directory, require_contract=None):
    """Rebuild the architecture from the saved config and restore the saved weights.

    Refuses a Keras major.minor mismatch before deserializing, a tampered archive
    or weights, and a rebuilt graph that does not reproduce the archived outputs.
    Donor paths are not re-read (the archive holds the selected weights); their
    manifests stay in bundle.json as provenance. Regimes return as trainability.
    """
    src = Path(directory)
    document = json.loads((src / "bundle.json").read_text(encoding="utf-8"))
    if document.get("schema") not in (BUNDLE_V1, BUNDLE_V2):
        raise ValueError("Unsupported bundle schema")
    document["provenance"] = bundle_provenance(document)        # v1: UNKNOWN, migrated=False
    if require_contract is not None and document["provenance"]["conditioning_contract"] != require_contract:
        raise ValueError(f"CONDITIONING_CONTRACT_NOT_{require_contract}: bundle declares "
                         f"{document['provenance']['conditioning_contract']}; UNKNOWN is never treated as "
                         f"{require_contract}")
    saved = document.get("keras_version")
    if saved is None or _major_minor(saved) != _major_minor(keras_version()):
        raise ValueError(f"Bundle was saved under Keras {saved}; running Keras {keras_version()} "
                         "(major.minor must match; re-export in the campaign environment)")
    archive = src / "forecast_model.keras"
    if _file_hash(archive) != document["archive_sha256"]:
        raise ValueError("Bundle archive hash mismatch")
    stored = keras.models.load_model(archive, compile=False, safe_mode=True)
    if weights_hash(stored) != document["weights_sha256"]:
        raise ValueError("Bundle weights hash mismatch")
    config = _copy(document["config"])
    regimes = {}
    for spec in [*config["branches"], config["core"]]:
        regimes[spec.get("name", "core")] = spec["regime"]
        spec.update(regime="R0", donor=None)
        spec.pop("freeze_epochs", None)
        spec.pop("unfreeze_learning_rate", None)
    config["regime"] = None
    rebuilt = build_modular(config)
    rebuilt.forecast_model.set_weights(stored.get_weights())
    if weights_hash(rebuilt.forecast_model) != document["weights_sha256"]:
        raise ValueError("Rebuilt architecture does not hold the saved weights")
    probe = np.random.default_rng(0).normal(
        size=(2, *rebuilt.forecast_model.input_shape[1:])).astype("float32")
    if not np.allclose(rebuilt.forecast_model(probe), stored(probe), atol=1e-6):
        raise ValueError("Rebuilt architecture does not reproduce the archived outputs")
    for name, model in rebuilt.branch_models.items():
        model.trainable = regimes[name] != "R1"
    rebuilt.core_model.trainable = regimes["core"] != "R1"
    rebuilt.config = _copy(document["config"])
    return rebuilt, document
