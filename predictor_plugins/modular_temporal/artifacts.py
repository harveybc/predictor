"""Donor manifests, regime loading, and integrity-checked Keras serialization."""

from pathlib import Path
import json

import tensorflow as tf

from .common import (_copy, _digest, _file_hash, _json, _major_minor, _positive_int, keras_version,
                     weights_hash)

keras = tf.keras

def _manifest(role, config, spec, plugin, model, input_grid, output_grid, params):
    """Component identity. ``params`` are EFFECTIVE (declared defaults resolved)."""
    return {"schema": 1, "role": role, "plugin": plugin, "params": _copy(params),
            "features": _copy(spec.get("features", config["feature_names"])),
            "feature_names": _copy(config["feature_names"]),
            "name": spec.get("name", role), "sample_hours": config["sample_hours"],
            "input_shape": list(model.input_shape[1:]), "output_shape": list(model.output_shape[1:]),
            "input_grid": list(input_grid), "output_grid": list(output_grid)}


DONOR_CONTRACTS = ("OPERATIONAL", "UNKNOWN_ALLOWED")


def _apply_regime(model, spec, manifest, donor_contract="OPERATIONAL"):
    """R1/R2 load the declared donor; the conditioning contract is enforced HERE, by the engine.
    ``donor_contract`` OPERATIONAL (the default when the config omits it) refuses UNKNOWN and
    SYNTHETIC_OFFLINE donors by name; only the explicit UNKNOWN_ALLOWED bypasses, and it is then part
    of the normalized config (and so of bundle.json)."""
    if donor_contract not in DONOR_CONTRACTS:
        raise ValueError(f"donor_contract must be one of {DONOR_CONTRACTS}")
    if spec["regime"] != "R0":
        loaded = load_donor(spec["donor"], manifest,
                            require_contract=None if donor_contract == "UNKNOWN_ALLOWED" else donor_contract)
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


def save_donor(model, path, manifest, declared_params=None, objective=None, provenance=None):
    """Save a selected component and a digest-bound identity sidecar.

    Parameters
    ----------
    model : keras.Model
        Component after restoring its selected training checkpoint.
    path : str or pathlib.Path
        Destination ending in ``.keras``.
    manifest : dict
        Expected component identity from :meth:`ModularBundle.donor_manifest`.
    declared_params : dict, optional
        The literal params as written in the configuration, kept as provenance
        (identity itself is over effective params).
    provenance : dict, optional
        ``conditioning_contract`` / ``learned_corpus`` / ``reconstruction`` (see
        :mod:`.provenance`); undeclared fields are written as UNKNOWN, explicitly.
        Validated BEFORE anything is written. The sidecar is schema 2.

    Returns
    -------
    dict
        Sidecar document containing manifest, archive and ordered weight hashes.
    """
    from .provenance import DONOR_SCHEMA, complete
    path, sidecar = _donor_path(path)
    manifest = _copy(manifest)
    _check_manifest_model(manifest, model)
    provenance = {**complete(provenance), "keras_version": keras_version()}
    model.save(path)
    if declared_params is not None:
        provenance["declared_params"] = _copy(declared_params)
    if objective is not None:
        # the objective that trained these weights: provenance, NOT part of the architecture manifest
        provenance["objective"] = _copy(objective)
    document = {"schema": DONOR_SCHEMA, "manifest": manifest, "manifest_sha256": _digest(manifest),
                "provenance": provenance,
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


def load_donor(path, expected_manifest, require_contract=None):
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
        A missing archive or sidecar is also a ValueError: an explicitly
        requested donor that is absent is an error, never a fallback.
    """
    path, sidecar = _donor_path(path)
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"Requested donor is missing: {path.name} or its manifest sidecar")
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid donor manifest JSON") from exc
    required = {"schema", "manifest", "manifest_sha256", "model_sha256", "weights_sha256"}
    if (not isinstance(document, dict) or not required <= set(document) <= required | {"provenance"}
            or document["schema"] not in (1, 2) or (document["schema"] == 2 and "provenance" not in document)):
        raise ValueError("Invalid donor manifest schema")
    if require_contract is not None:
        from .provenance import donor_provenance
        contract = donor_provenance(path)["conditioning_contract"]
        if contract != require_contract:
            raise ValueError(f"CONDITIONING_CONTRACT_NOT_{require_contract}: donor {path.name} declares "
                             f"{contract}; UNKNOWN is never treated as {require_contract}")
    saved = (document.get("provenance") or {}).get("keras_version")
    if saved is not None and _major_minor(saved) != _major_minor(keras_version()):
        raise ValueError(f"Donor was saved under Keras {saved}; running Keras {keras_version()} "
                         "(major.minor must match)")
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
