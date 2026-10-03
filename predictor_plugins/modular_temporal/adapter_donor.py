"""Typed TRAIN-only NPZ adapter output -> engine branch donor that satisfies the donor contract.

The single adapter implementation lives in feature-extractor (``app/npz_encoder_adapter.py``,
architecture ``conv1d_ae_v1``, branch satoshi/typed-npz-adapter-20261002). It writes
``encoder.keras`` and ``manifest.json`` (input_sha256, train_row_ids_sha256, weight_sha256, ...).
This module does not train anything: it verifies that output and re-expresses it as a schema-2
engine donor for the ``npz_adapter_conv`` branch plugin.

Refusals (``ValueError`` with a code prefix), all before any donor byte is written:

* ``ADAPTER_MANIFEST_INVALID``     manifest keys/architecture are not the adapter's typed contract
* ``ADAPTER_WEIGHTS_MISMATCH``     encoder.keras weights do not hash to the manifest's weight_sha256
* ``ADAPTER_SPLIT_NOT_TRAIN``      the NPZ given as the TRAIN split declares another split
* ``ADAPTER_ROWS_NOT_TRAIN_SPLIT`` manifest train_row_ids_sha256 differs from the TRAIN split's row ids
* ``ADAPTER_INPUT_NOT_TRAIN_SPLIT`` manifest input_sha256 differs from the TRAIN NPZ file bytes
* ``ADAPTER_SHAPE_MISMATCH``       steps/channels/features/filters disagree with the branch

What is carried and what is not (recorded in the conversion receipt):

* carried: the two Conv1D kernels and biases of the adapter encoder, unchanged bytes;
* dropped: Flatten + Dense(latent). They collapse time to one vector, which cannot satisfy the
  branch's rank-three right-edge grid contract;
* re-padded: ``same`` -> ``causal``. With unchanged kernels the branch output at step t equals the
  adapter's trunk at t-2 (away from the left edge), so the representation is look-ahead free and the
  engine's alignment probe passes. The donor is therefore declared OPERATIONAL.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ADAPTER_ARCHITECTURE = "conv1d_ae_v1"
ADAPTER_PLUGIN = "npz_adapter_conv"
ADAPTER_MANIFEST_KEYS = frozenset({
    "input_sha256", "train_row_ids_sha256", "architecture_id", "seed", "n_rows", "steps", "channels",
    "latent_dim", "epochs", "reconstruction_mae", "reconstruction_mse", "weight_sha256", "encoder_path"})
CONVERSION_SCHEMA = "predictor.modular.adapter_donor.v1"


def _refuse(code, detail):
    raise ValueError(f"{code}: {detail}")


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_ids_sha256(row_ids):
    """The adapter's row-id digest, byte for byte (strings: newline-joined UTF-8; integers: <i8)."""
    row_ids = np.asarray(row_ids)
    if row_ids.dtype.kind in "US":
        return hashlib.sha256("\n".join(str(i) for i in row_ids.tolist()).encode("utf-8")).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(row_ids, dtype="<i8").tobytes()).hexdigest()


def adapter_weight_sha256(model):
    """The adapter's encoder weight digest: concatenated little-endian float32 weight bytes."""
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(np.ascontiguousarray(weight, dtype="<f4").tobytes())
    return digest.hexdigest()


def train_split_identity(train_npz):
    """Identity of the TRAIN split the cell trains on: file sha, row-id digest, shape, features."""
    with np.load(train_npz, allow_pickle=False) as z:
        if "split" in z.files and str(z["split"]) != "train":
            _refuse("ADAPTER_SPLIT_NOT_TRAIN", f"{Path(train_npz).name} declares split {str(z['split'])!r}")
        key = "x" if "x" in z.files else "windows"
        shape = tuple(int(s) for s in z[key].shape)
        ids = z["row_ids"] if "row_ids" in z.files else np.arange(shape[0], dtype=np.int64)
        if ids.dtype.kind in "US":
            ids = ids.astype(str)
        features = [str(f) for f in z["feature_names"].tolist()] if "feature_names" in z.files else None
        dataset_id = str(z["dataset_id"]) if "dataset_id" in z.files else None
    return {"sha256": _sha_file(train_npz), "row_ids_sha256": row_ids_sha256(ids), "n_rows": shape[0],
            "steps": shape[1], "channels": shape[2], "feature_names": features, "dataset_id": dataset_id,
            "first_row_id": str(ids[0]), "last_row_id": str(ids[-1])}


def read_adapter_output(adapter_dir):
    """Load and verify the adapter output: exact manifest keys, architecture and weight digest."""
    import tensorflow as tf

    adapter_dir = Path(adapter_dir)
    manifest = json.loads((adapter_dir / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != ADAPTER_MANIFEST_KEYS or manifest["architecture_id"] != ADAPTER_ARCHITECTURE:
        _refuse("ADAPTER_MANIFEST_INVALID", f"keys {sorted(manifest)} / architecture "
                                            f"{manifest.get('architecture_id')!r}")
    encoder = tf.keras.models.load_model(adapter_dir / manifest["encoder_path"], compile=False, safe_mode=True)
    if adapter_weight_sha256(encoder) != manifest["weight_sha256"]:
        _refuse("ADAPTER_WEIGHTS_MISMATCH", f"{adapter_dir / manifest['encoder_path']} does not hash to "
                                            f"{manifest['weight_sha256']}")
    convs = [layer for layer in encoder.layers if isinstance(layer, tf.keras.layers.Conv1D)]
    if len(convs) != 2 or any(c.padding != "same" or c.strides != (1,) or c.dilation_rate != (1,) for c in convs):
        _refuse("ADAPTER_MANIFEST_INVALID", "conv1d_ae_v1 encoder must hold two stride-1 'same' Conv1D layers")
    return manifest, encoder, convs


def _r0_config(config):
    """Same model, every component R0 without donors: only to obtain the branch identity to bind."""
    c = json.loads(json.dumps(config))
    c["regime"] = None
    for spec in [*c["branches"], c["core"]]:
        spec["regime"], spec["donor"] = "R0", None
        for key in ("freeze_epochs", "unfreeze_learning_rate"):
            spec.pop(key, None)
    return c


def convert_adapter_donor(adapter_dir, train_npz, config, branch_name, out_path):
    """Verify the adapter output against the TRAIN split and write an engine donor for one branch.

    Parameters
    ----------
    adapter_dir : path
        Directory holding the adapter's ``manifest.json`` and encoder.
    train_npz : path
        The TRAIN split NPZ the cell trains on. Its row ids and bytes must be those the adapter saw.
    config : dict
        The cell's engine configuration; ``branch_name`` must use plugin ``npz_adapter_conv``.
    out_path : path
        Donor destination ending in ``.keras``; ``<stem>.conversion.json`` is written beside it.

    Returns
    -------
    dict
        The conversion receipt (also written to disk).
    """
    from . import save_donor, weights_hash
    from .assembly import build_modular

    manifest, encoder, convs = read_adapter_output(adapter_dir)
    split = train_split_identity(train_npz)
    if manifest["train_row_ids_sha256"] != split["row_ids_sha256"]:
        _refuse("ADAPTER_ROWS_NOT_TRAIN_SPLIT", f"adapter rows {manifest['train_row_ids_sha256']} != TRAIN split "
                                                f"rows {split['row_ids_sha256']}")
    if manifest["input_sha256"] != split["sha256"] or manifest["n_rows"] != split["n_rows"]:
        _refuse("ADAPTER_INPUT_NOT_TRAIN_SPLIT", f"adapter input {manifest['input_sha256']} != TRAIN NPZ "
                                                 f"{split['sha256']}")
    spec = next((b for b in config["branches"] if b["name"] == branch_name), None)
    if spec is None or spec.get("plugin") != ADAPTER_PLUGIN:
        _refuse("ADAPTER_SHAPE_MISMATCH", f"branch {branch_name!r} must use plugin {ADAPTER_PLUGIN}")
    filters = int(convs[0].filters)
    kernel = int(convs[0].kernel_size[0])
    params = {"filters": 32, "kernel_size": 3, **spec.get("params", {})}
    if (manifest["steps"] != config["window"] or manifest["channels"] != len(spec["features"])
            or (split["steps"], split["channels"]) != (manifest["steps"], manifest["channels"])
            or (split["feature_names"] is not None and split["feature_names"] != list(spec["features"]))
            or params["filters"] != filters or params["kernel_size"] != kernel
            or int(convs[1].filters) != filters or int(convs[1].kernel_size[0]) != kernel):
        _refuse("ADAPTER_SHAPE_MISMATCH", f"adapter steps/channels/filters/kernel {manifest['steps']}/"
                                          f"{manifest['channels']}/{filters}/{kernel} vs branch {branch_name}")
    bundle = build_modular(_r0_config(config))
    branch = bundle.branch_models[branch_name]
    carried = [w for conv in convs for w in conv.get_weights()]
    branch.set_weights(carried)
    if [w.tobytes() for w in branch.get_weights()] != [w.tobytes() for w in carried]:
        raise ValueError("ADAPTER_WEIGHTS_MISMATCH: carried kernels changed during transfer")
    provenance = {
        "conditioning_contract": "OPERATIONAL",
        "learned_corpus": {"kind": "TRAIN_ONLY", "dataset_id": split["dataset_id"] or f"npz:{split['sha256']}",
                           "data_sha256": split["sha256"], "pretrained_weights_source": None,
                           "support": (f"TRAIN NPZ rows {split['first_row_id']}..{split['last_row_id']} "
                                       f"({split['n_rows']} windows, row_ids sha256 {split['row_ids_sha256']})")},
        "reconstruction": {"state": "MEASURED", "mae_z": manifest["reconstruction_mae"],
                           "mse_z": manifest["reconstruction_mse"],
                           "scope": "adapter conv1d_ae_v1 full autoencoder on its TRAIN rows; the carried "
                                    "trunk has no decoder of its own"}}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    document = save_donor(branch, out_path, bundle.donor_manifest("branch", branch_name),
                          declared_params=spec.get("params", {}), provenance=provenance,
                          objective={"kind": "adapter_reconstruction", "loss": "mse",
                                     "architecture_id": ADAPTER_ARCHITECTURE, "seed": manifest["seed"],
                                     "epochs": manifest["epochs"]})
    receipt = {"schema": CONVERSION_SCHEMA, "adapter": {"dir": str(Path(adapter_dir).resolve()),
                                                        "manifest": manifest,
                                                        "manifest_sha256": _sha_file(Path(adapter_dir) / "manifest.json"),
                                                        "encoder_sha256": _sha_file(Path(adapter_dir) / manifest["encoder_path"])},
               "train_split": split, "branch": branch_name, "plugin": ADAPTER_PLUGIN,
               "carried": [{"layer": c.name, "kernel_shape": list(c.get_weights()[0].shape)} for c in convs],
               "dropped": ["Flatten", "Dense(latent)"], "padding": {"adapter": "same", "branch": "causal",
                                                                   "lag_steps": 2 * (kernel // 2)},
               "donor": {"path": str(out_path.resolve()), "model_sha256": document["model_sha256"],
                         "weights_sha256": document["weights_sha256"],
                         "manifest_sha256": document["manifest_sha256"]},
               "branch_weights_sha256": weights_hash(branch)}
    out_path.with_suffix(".conversion.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt
