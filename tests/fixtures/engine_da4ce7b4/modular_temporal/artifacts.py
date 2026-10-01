"""Donor manifests, regime loading, and integrity-checked Keras serialization."""

from pathlib import Path
import json

import tensorflow as tf

from .common import _copy, _digest, _file_hash, _json, _positive_int, weights_hash

keras = tf.keras

def _manifest(role, config, spec, plugin, model, input_grid, output_grid):
    return {"schema": 1, "role": role, "plugin": plugin, "params": _copy(spec["params"]),
            "features": _copy(spec.get("features", config["feature_names"])),
            "feature_names": _copy(config["feature_names"]),
            "name": spec.get("name", role), "sample_hours": config["sample_hours"],
            "input_shape": list(model.input_shape[1:]), "output_shape": list(model.output_shape[1:]),
            "input_grid": list(input_grid), "output_grid": list(output_grid)}


def _apply_regime(model, spec, manifest):
    if spec["regime"] != "R0":
        loaded = load_donor(spec["donor"], manifest)
        try:
            model.set_weights(loaded.get_weights())
        except ValueError as exc:
            raise ValueError("Donor weights incompatible with component manifest") from exc
    model.trainable = spec["regime"] != "R1"


def _upstream(branches, manifests, fusion, identity):
    return {"branches": [{"manifest": manifests[name], "weights_sha256": weights_hash(model)}
                         for name, model in branches.items()],
            "fusion": {"identity": identity, "weights_sha256": weights_hash(fusion)}}


def _donor_path(path):
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError("Donor path must end in .keras")
    return path, path.with_suffix(".manifest.json")


def save_donor(model, path, manifest):
    """Save a selected component and a digest-bound identity sidecar.

    Parameters
    ----------
    model : keras.Model
        Component after restoring its selected training checkpoint.
    path : str or pathlib.Path
        Destination ending in ``.keras``.
    manifest : dict
        Expected component identity from :meth:`ModularBundle.donor_manifest`.

    Returns
    -------
    dict
        Sidecar document containing manifest, archive and ordered weight hashes.
    """
    path, sidecar = _donor_path(path)
    manifest = _copy(manifest)
    _check_manifest_model(manifest, model)
    model.save(path)
    document = {"schema": 1, "manifest": manifest, "manifest_sha256": _digest(manifest),
                "model_sha256": _file_hash(path), "weights_sha256": weights_hash(model)}
    sidecar.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return document


def _check_manifest_model(manifest, model):
    required = {"schema", "role", "plugin", "params", "features", "feature_names", "name",
                "sample_hours", "input_shape", "output_shape", "input_grid", "output_grid"}
    if manifest.get("role") == "core":
        required.add("upstream")
    if set(manifest) != required or manifest.get("schema") != 1 or manifest.get("role") not in ("branch", "core"):
        raise ValueError("Invalid donor manifest schema")
    if (list(model.input_shape[1:]) != manifest["input_shape"]
            or list(model.output_shape[1:]) != manifest["output_shape"]):
        raise ValueError("Donor manifest/model shapes differ")


def load_donor(path, expected_manifest):
    """Verify a donor's identity and bytes before safe Keras deserialization.

    Parameters
    ----------
    path : str or pathlib.Path
        Donor archive ending in ``.keras``.
    expected_manifest : dict
        Exact identity required by the requesting branch or core.

    Returns
    -------
    keras.Model
        Loaded model with verified input/output shapes and weights.

    Raises
    ------
    ValueError
        If the sidecar, manifest, archive, weights or requested identity differs.
    FileNotFoundError
        If the archive or sidecar is missing.
    """
    path, sidecar = _donor_path(path)
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid donor manifest JSON") from exc
    required = {"schema", "manifest", "manifest_sha256", "model_sha256", "weights_sha256"}
    if not isinstance(document, dict) or set(document) != required or document["schema"] != 1:
        raise ValueError("Invalid donor manifest schema")
    if _digest(document["manifest"]) != document["manifest_sha256"]:
        raise ValueError("Donor manifest hash mismatch")
    if _json(document["manifest"]) != _json(expected_manifest):
        raise ValueError("Donor manifest mismatch (features/config/grid/upstream)")
    if _file_hash(path) != document["model_sha256"]:
        raise ValueError("Donor archive hash mismatch")
    model = keras.models.load_model(path, compile=False, safe_mode=True)
    _check_manifest_model(document["manifest"], model)
    if weights_hash(model) != document["weights_sha256"]:
        raise ValueError("Donor weights hash mismatch")
    return model
