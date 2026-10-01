"""End-to-end Keras model assembly and its public component bundle."""

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from .artifacts import _apply_regime, _manifest, _upstream
from .common import _copy, _digest, _json, _partition
from .components import TemporalComponent, effective_params
from .config import _normalize, regime_summary
from .layers import FeatureSelect
from .registry import _resolve

keras = tf.keras

def _validate_component(component, input_shapes, grid, output_channels=None):
    if not isinstance(component, TemporalComponent) or not isinstance(component.model, keras.Model):
        raise ValueError("Plugin must return TemporalComponent(Keras model, time_grid)")
    model = component.model
    if len(model.inputs) != len(input_shapes) or any(tuple(t.shape[1:]) != tuple(s) for t, s in zip(model.inputs, input_shapes)):
        raise ValueError("Plugin input shape does not match contract")
    if (len(model.outputs) != 1 or len(model.output_shape) != 3
            or model.output_shape[1] != len(grid) or model.output_shape[2] is None):
        raise ValueError("Plugin must output a fixed rank-three sequence")
    if tuple(component.time_grid) != tuple(grid):
        raise ValueError("Plugin time grid does not match right-edge grid contract")
    if output_channels is not None and model.output_shape[2] != output_channels:
        raise ValueError("Plugin channel contract mismatch")
    return model


def probe_alignment(model, input_grid, output_grid, *, seed=0, label="component"):
    """Behavioural time-grid check; equal shapes alone do not establish alignment.

    Perturbs every input position in one batched forward pass. An output labelled
    with right edge t may not change when an input strictly after t changes (no
    look-ahead), and the first output whose right edge is at or after the
    perturbed time must change (every input position participates). A component
    that reverses, shifts or drops time fails here even with the declared shape.
    """
    length, channels = int(model.input_shape[1]), int(model.input_shape[2])
    if length != len(input_grid) or int(model.output_shape[1]) != len(output_grid):
        raise ValueError(f"{label}: probe grid does not match model shape")
    base = np.random.default_rng(seed).normal(size=(1, length, channels)).astype("float32")
    batch = np.repeat(base, length + 1, axis=0)
    for i in range(length):
        batch[i + 1, i, :] += 3.0
    out = np.asarray(model(batch, training=False), dtype="float64")
    moved = np.max(np.abs(out[1:] - out[:1]), axis=2)
    scale = max(1.0, float(np.max(np.abs(out[0]))))
    for i, t in enumerate(input_grid):
        for k, edge in enumerate(output_grid):
            if edge < t and moved[i, k] > 1e-5 * scale:
                raise ValueError(f"{label}: time alignment violated; output at {edge} "
                                 f"depends on the later input at {t}")
        first = next((k for k, edge in enumerate(output_grid) if edge >= t), None)
        if first is None or moved[i, first] <= 1e-7 * scale:
            raise ValueError(f"{label}: time alignment violated; input at {t} does not reach "
                             f"the output whose right edge covers it")
    return {"checked_inputs": length, "checked_outputs": len(output_grid)}


def config_digest(config):
    """SHA-256 of the canonical (sorted, compact, NaN-free) normalized configuration."""
    return _digest(_normalize(config))


def canonical_config_json(config):
    """Deterministic serialization of the normalized configuration."""
    return _json(_normalize(config))


class BudgetExceeded(ValueError):
    """A configuration over a declared cap: the candidate is DEFERRED by name, never truncated to fit."""

    def __init__(self, dimension, measured, cap):
        self.dimension, self.measured, self.cap = dimension, measured, cap
        super().__init__(f"BUDGET_EXCEEDED_DEFERRED: {dimension} {measured} > cap {cap}; the candidate is "
                         "deferred with this reason (no input truncation, no temporal collapse)")


def _budget(bundle):
    """Measured shape budget of a built bundle."""
    c = bundle.config
    widths = [int(m.output_shape[-1]) for m in bundle.branch_models.values()]
    fused = bundle.fusion_model.output_shape
    branch_params = int(sum(m.count_params() for m in bundle.branch_models.values()))
    core_params = int(bundle.core_model.count_params())
    total = int(bundle.forecast_model.count_params())
    routed = sorted({f for spec in c["branches"] for f in spec["features"]})
    return {"raw_channels": len(c["feature_names"]), "routed_channels": len(routed),
            "excluded_features": _copy(c.get("excluded_features", {})),
            "unrouted_features": [f for f in c["feature_names"] if f not in routed
                                  and f not in c.get("excluded_features", {})],
            "branches": len(widths), "branch_widths": widths, "fused_width": int(fused[-1]),
            "fused_time": int(fused[1]), "latent_shape": [int(d) for d in bundle.core_model.output_shape[1:]],
            "materialization_bytes_per_row": int(fused[1]) * int(fused[-1]) * 4, "dtype": "float32",
            "parameters": {"branches": branch_params, "core": core_params,
                           "head": total - branch_params - core_params, "total": total}}


def _analytic_budget(c):
    from .registry import _resolve
    widths = []
    for spec in c["branches"]:
        factory, _ = _resolve("branch", spec, c["entry_point_groups"])
        params = effective_params(factory, spec["params"], {"output_steps": c["branch_steps"]})
        if "channels" not in params:
            raise ValueError("analytic budget needs a branch component that declares its channel width")
        widths.append(int(params["channels"]))
    routed = sorted({f for spec in c["branches"] for f in spec["features"]})
    return {"raw_channels": len(c["feature_names"]), "routed_channels": len(routed),
            "excluded_features": _copy(c.get("excluded_features", {})),
            "unrouted_features": [f for f in c["feature_names"] if f not in routed
                                  and f not in c.get("excluded_features", {})], "branches": len(widths),
            "branch_widths": widths, "fused_width": sum(widths), "fused_time": c["branch_steps"],
            "latent_shape": [c["output_steps"], c["output_channels"]],
            "materialization_bytes_per_row": c["branch_steps"] * sum(widths) * 4, "dtype": "float32"}


def _enforce_caps(budget, caps):
    measured = {"max_branches": budget["branches"], "max_fused_width": budget["fused_width"],
                "max_materialization_bytes_per_row": budget["materialization_bytes_per_row"],
                "max_parameters": (budget.get("parameters") or {}).get("total")}
    for key in sorted(caps):
        value = measured[key]
        if value is not None and value > caps[key]:
            raise BudgetExceeded(key[len("max_"):], value, caps[key])


def measure_budget(config, *, build=True):
    """Shape budget of any config: measured from the built graph (and checked against the analytic
    prediction), or analytic only with ``build=False``. Caps in the config are reported, not enforced."""
    c = _normalize(config)
    analytic = _analytic_budget(c)
    if not build:
        return {**analytic, "parameters": None, "measured": False, "config_sha256": _digest(c)}
    probe_free = _copy(c)
    probe_free.pop("budget_caps", None)
    probe_free["alignment_probe"] = False
    measured = _budget(build_modular(probe_free))
    keys = ("branch_widths", "fused_width", "fused_time", "latent_shape", "materialization_bytes_per_row")
    return {**measured, "measured": True, "config_sha256": _digest(c),
            "analytic_matches_measured": all(analytic[k] == measured[k] for k in keys),
            "caps": _copy(c.get("budget_caps"))}


@dataclass
class ModularBundle:
    """Related Keras models produced from one validated modular configuration.

    Attributes
    ----------
    branch_models : dict[str, keras.Model]
        Independently trainable per-branch feature extractors.
    fusion_model : keras.Model
        Raw channel concatenation that preserves the shared branch time grid.
    core_model : keras.Model
        Fused sequence encoder that returns the compressed temporal latent.
    encoder_model : keras.Model
        Complete input-to-latent path.
    forecast_model : keras.Model
        Complete input-to-direct-forecast model.
    """
    config: dict
    branch_models: dict
    fusion_model: keras.Model
    core_model: keras.Model
    encoder_model: keras.Model
    forecast_model: keras.Model
    branch_time_grid: tuple
    core_time_grid: tuple
    _branch_manifests: dict = field(repr=False)
    _core_manifest: dict = field(repr=False)
    _fusion_component: keras.Model = field(repr=False)
    _fusion_identity: dict = field(repr=False)
    _head_manifest: dict = field(repr=False)

    def component_manifests(self):
        """Identity, version, effective parameters and tensor/time contract of every component."""
        shapes = self._fusion_component.input_shape
        fusion = {"schema": 1, "role": "fusion", "plugin": _copy(self._fusion_identity),
                  "input_shapes": [list(s[1:]) for s in shapes] if isinstance(shapes, list)
                  else [list(shapes[1:])],
                  "output_shape": list(self._fusion_component.output_shape[1:]),
                  "grid": list(self.branch_time_grid)}
        return {"schema": 1, "config_sha256": config_digest(self.config),
                "branches": {n: self.donor_manifest("branch", n) for n in self.branch_models},
                "fusion": fusion, "core": self.donor_manifest("core"),
                "head": _copy(self._head_manifest), "regimes": regime_summary(self.config),
                "budget": _budget(self)}

    def donor_manifest(self, role, name=None):
        """Return the component identity required to save or validate a donor.

        Parameters
        ----------
        role : {"branch", "core"}
            Component type whose weights will be exported.
        name : str or None, optional
            Required branch name; must be omitted for the core.

        Returns
        -------
        dict
            Identity containing configuration, plugin, tensor shapes and grids.
            Core identity also includes the current upstream branch and fusion
            weight digests.
        """
        if role == "branch":
            if name not in self._branch_manifests:
                raise ValueError("Unknown branch name")
            return _copy(self._branch_manifests[name])
        if role != "core" or name is not None:
            raise ValueError("Expected branch/name or core without name")
        manifest = _copy(self._core_manifest)
        manifest["upstream"] = _upstream(self.branch_models, self._branch_manifests,
                                          self._fusion_component, self._fusion_identity)
        return manifest


def build_modular(config: dict) -> ModularBundle:
    """Assemble the configured branches, fusion, core, and forecast head.

    Parameters
    ----------
    config : dict
        A configuration from :func:`default_config` or an explicit compatible
        mapping. ``sample_hours`` is mandatory and the physical input window
        must span at least 24 hours.

    Returns
    -------
    ModularBundle
        Connected Keras models with shared live component weights.

    Raises
    ------
    ValueError
        If a plugin, regime, donor, temporal grid, or tensor shape violates its
        declared contract. No implicit plugin or donor fallback is performed.
    """
    c = _normalize(config)
    input_grid = tuple((i + 1) * c["sample_hours"] for i in range(c["window"]))
    branch_grid = _partition(input_grid, c["branch_steps"])
    core_grid = _partition(branch_grid, c["output_steps"])
    inputs = keras.Input((c["window"], len(c["feature_names"])), name="observations")
    branches, manifests, sequences = {}, {}, []
    groups = c["entry_point_groups"]
    for spec in c["branches"]:
        name = spec["name"]
        shape = (c["window"], len(spec["features"]))
        factory, identity = _resolve("branch", spec, groups)
        component = factory(input_shape=shape, time_grid=input_grid, output_steps=c["branch_steps"],
                            name=name, params=_copy(spec["params"]))
        model = _validate_component(component, [shape], branch_grid)
        effective = effective_params(factory, spec["params"], {"output_steps": c["branch_steps"]})
        manifest = _manifest("branch", c, spec, identity, model, input_grid, branch_grid, effective)
        _apply_regime(model, spec, manifest)
        if c["alignment_probe"]:
            probe_alignment(model, input_grid, branch_grid, label="branch " + name)
        branches[name], manifests[name] = model, manifest
        local = FeatureSelect([c["feature_names"].index(f) for f in spec["features"]],
                              name="select_" + name)(inputs)
        sequences.append(model(local))
    factory, fusion_identity = _resolve("fusion", c["fusion"], groups)
    fusion_identity["params"] = effective_params(factory, c["fusion"]["params"], {})
    shapes = [tuple(x.shape[1:]) for x in sequences]
    component = factory(input_shapes=shapes, time_grid=branch_grid, name="sequence_fusion", params=_copy(c["fusion"]["params"]))
    fusion = _validate_component(component, shapes, branch_grid, sum(s[1] for s in shapes))
    if fusion.weights:
        raise ValueError("Raw sequence fusion cannot contain weights")
    fused = fusion(sequences)
    fusion_model = keras.Model(inputs, fused, name="fusion_model")
    factory, identity = _resolve("core", c["core"], groups)
    component = factory(input_shape=tuple(fused.shape[1:]), time_grid=branch_grid,
                        output_steps=c["output_steps"], output_channels=c["output_channels"],
                        name="temporal_core", params=_copy(c["core"]["params"]))
    core = _validate_component(component, [tuple(fused.shape[1:])], core_grid, c["output_channels"])
    effective = effective_params(factory, c["core"]["params"], {
        "input_steps": c["branch_steps"], "output_steps": c["output_steps"],
        "output_channels": c["output_channels"]})
    core_manifest = _manifest("core", c, c["core"], identity, core, branch_grid, core_grid, effective)
    core_manifest["upstream"] = _upstream(branches, manifests, fusion, fusion_identity)
    _apply_regime(core, c["core"], core_manifest)
    if c["alignment_probe"]:
        probe_alignment(core, branch_grid, core_grid, label="core")
    latent = core(fused)
    encoder = keras.Model(inputs, latent, name="encoder_model")
    factory, head_identity = _resolve("head", c["head"], groups)
    forecast_grid = tuple(input_grid[-1] + h * c["sample_hours"] for h in c["horizons"])
    component = factory(input_shape=tuple(latent.shape[1:]), time_grid=forecast_grid,
                        horizons=c["horizons"], target_count=c["target_count"],
                        name="forecast_head", params=_copy(c["head"]["params"]))
    head = _validate_component(component, [tuple(latent.shape[1:])], forecast_grid, c["target_count"])
    model = keras.Model(inputs, head(latent), name="forecast_model")
    head_manifest = {"schema": 1, "role": "head", "plugin": head_identity,
                     "params": effective_params(factory, c["head"]["params"], {}),
                     "horizons": _copy(c["horizons"]), "target_count": c["target_count"],
                     "input_shape": list(head.input_shape[1:]), "output_shape": list(head.output_shape[1:]),
                     "input_grid": list(core_grid), "output_grid": list(forecast_grid)}
    bundle = ModularBundle(c, branches, fusion_model, core, encoder, model, branch_grid,
                           core_grid, manifests, core_manifest, fusion, fusion_identity, head_manifest)
    if c.get("budget_caps"):
        _enforce_caps(_budget(bundle), c["budget_caps"])
    return bundle
