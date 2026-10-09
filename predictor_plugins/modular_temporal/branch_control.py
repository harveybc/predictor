"""Branch-only Dense-versus-Conv control for the modular temporal model."""

from dataclasses import dataclass

import tensorflow as tf

from .assembly import ModularBundle, build_modular
from .common import _copy, _digest, weights_hash
from .config import _normalize


@dataclass(frozen=True)
class BranchOnlyControl:
    """Two modular bundles differing only in their branch implementations."""

    conv: ModularBundle
    dense: ModularBundle
    report: dict


def _differences(left, right, path=""):
    """Return deterministic leaf paths whose JSON-compatible values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_differences(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = []
        if len(left) != len(right):
            return [path]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.extend(_differences(left_item, right_item, f"{path}[{index}]"))
        return paths
    return [] if left == right else [path]


def _arm_config(base, plugin, params):
    config = _copy(base)
    for branch in config["branches"]:
        branch.update(plugin=plugin, params=_copy(params), regime="R0", donor=None)
    return _normalize(config)


def _head(bundle):
    return bundle.forecast_model.get_layer("forecast_head")


def build_branch_only_control(
    config,
    *,
    seed,
    conv_params=None,
    dense_params=None,
):
    """Build a causal branch-only control with a shared temporal downstream.

    Both arms use identical inputs, routing, sequence fusion, temporal core,
    forecast head, horizons, and initial downstream weights. Only each branch's
    implementation and its private parameters differ. Donor-backed regimes are
    refused here because they would introduce a second experimental factor.

    Parameters
    ----------
    config : dict
        Base modular configuration. Every branch must be in regime R0.
    seed : int
        Nonnegative initializer seed recorded in the report.
    conv_params : dict, optional
        Parameters for ``causal_conv1d``.
    dense_params : dict, optional
        Parameters for ``causal_dense_sequence``.

    Returns
    -------
    BranchOnlyControl
        Executable Conv and Dense-sequence bundles plus an attribution report.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    base = _normalize(config)
    if any(branch["regime"] != "R0" or branch["donor"] is not None
           for branch in base["branches"]):
        raise ValueError("Branch-only initialization control requires every branch in R0")

    conv_params = {"channels": 16, "kernel_size": 3, **(conv_params or {})}
    dense_params = {
        "context": 3,
        "hidden_units": [16],
        "channels": 16,
        "activation": "gelu",
        "use_bias": True,
        **(dense_params or {}),
    }
    if (type(conv_params.get("channels")) is not int
            or type(dense_params.get("channels")) is not int
            or conv_params["channels"] != dense_params["channels"]):
        raise ValueError("Conv and Dense branches must expose the same channel width")

    conv_config = _arm_config(base, "causal_conv1d", conv_params)
    dense_config = _arm_config(base, "causal_dense_sequence", dense_params)
    differences = _differences(conv_config, dense_config)
    allowed = all(
        path.startswith("branches[")
        and (path.endswith(".plugin") or ".params." in path)
        for path in differences
    )
    if not differences or not allowed:
        raise ValueError("Branch-only arms differ outside branch plugin parameters")

    tf.keras.utils.set_random_seed(seed)
    conv = build_modular(conv_config)
    tf.keras.utils.set_random_seed(seed)
    dense = build_modular(dense_config)
    if (conv.branch_time_grid != dense.branch_time_grid
            or conv.core_time_grid != dense.core_time_grid
            or conv.encoder_model.output_shape != dense.encoder_model.output_shape
            or conv.forecast_model.output_shape != dense.forecast_model.output_shape):
        raise ValueError("Branch-only arms do not expose identical temporal contracts")

    dense.core_model.set_weights(conv.core_model.get_weights())
    conv_head, dense_head = _head(conv), _head(dense)
    dense_head.set_weights(conv_head.get_weights())
    if (weights_hash(conv.core_model) != weights_hash(dense.core_model)
            or weights_hash(conv_head) != weights_hash(dense_head)):
        raise ValueError("Shared downstream weights differ at initialization")

    conv_manifests = conv.component_manifests()
    dense_manifests = dense.component_manifests()
    conv_core_contract = {
        key: value for key, value in conv_manifests["core"].items()
        if key != "upstream"
    }
    dense_core_contract = {
        key: value for key, value in dense_manifests["core"].items()
        if key != "upstream"
    }
    shared = {
        "seed": seed,
        "fusion_identity_sha256": _digest(conv_manifests["fusion"]),
        "core_config_sha256": _digest(conv_core_contract),
        "head_config_sha256": _digest(conv_manifests["head"]),
        "core_initial_weights_sha256": weights_hash(conv.core_model),
        "head_initial_weights_sha256": weights_hash(conv_head),
        "branch_time_grid": list(conv.branch_time_grid),
        "core_time_grid": list(conv.core_time_grid),
    }
    if (_digest(conv_manifests["fusion"]) != _digest(dense_manifests["fusion"])
            or _digest(conv_core_contract) != _digest(dense_core_contract)
            or _digest(conv_manifests["head"]) != _digest(dense_manifests["head"])):
        raise ValueError("Fusion, core, or head configuration differs between arms")
    report = {
        "schema": "predictor.i6d.branch_only_control.v1",
        "only_branch_configuration_differs": True,
        "configuration_differences": differences,
        "shared": shared,
        "arms": {
            "CONV": {
                "plugin": "causal_conv1d",
                "params": _copy(conv_params),
                "parameters": conv.forecast_model.count_params(),
            },
            "DENSE_SEQUENCE": {
                "plugin": "causal_dense_sequence",
                "params": _copy(dense_params),
                "parameters": dense.forecast_model.count_params(),
            },
        },
    }
    return BranchOnlyControl(conv=conv, dense=dense, report=report)
