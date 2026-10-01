"""End-to-end Keras model assembly and its public component bundle."""

from dataclasses import dataclass, field

import tensorflow as tf

from .artifacts import _apply_regime, _manifest, _upstream
from .common import _copy, _partition
from .components import TemporalComponent
from .config import _normalize
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
        manifest = _manifest("branch", c, spec, identity, model, input_grid, branch_grid)
        _apply_regime(model, spec, manifest)
        branches[name], manifests[name] = model, manifest
        local = FeatureSelect([c["feature_names"].index(f) for f in spec["features"]],
                              name="select_" + name)(inputs)
        sequences.append(model(local))
    factory, fusion_identity = _resolve("fusion", c["fusion"], groups)
    fusion_identity["params"] = _copy(c["fusion"]["params"])
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
    core_manifest = _manifest("core", c, c["core"], identity, core, branch_grid, core_grid)
    core_manifest["upstream"] = _upstream(branches, manifests, fusion, fusion_identity)
    _apply_regime(core, c["core"], core_manifest)
    latent = core(fused)
    encoder = keras.Model(inputs, latent, name="encoder_model")
    factory, _ = _resolve("head", c["head"], groups)
    forecast_grid = tuple(input_grid[-1] + h * c["sample_hours"] for h in c["horizons"])
    component = factory(input_shape=tuple(latent.shape[1:]), time_grid=forecast_grid,
                        horizons=c["horizons"], target_count=c["target_count"],
                        name="forecast_head", params=_copy(c["head"]["params"]))
    head = _validate_component(component, [tuple(latent.shape[1:])], forecast_grid, c["target_count"])
    model = keras.Model(inputs, head(latent), name="forecast_model")
    return ModularBundle(c, branches, fusion_model, core, encoder, model, branch_grid,
                         core_grid, manifests, core_manifest, fusion, fusion_identity)
