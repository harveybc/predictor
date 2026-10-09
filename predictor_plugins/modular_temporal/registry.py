"""Built-in component registry and external entry-point resolution."""

from importlib.metadata import entry_points

from .components import (
    causal_conv1d,
    causal_dense_sequence,
    forecast,
    sequence_concat,
    transformer_conv,
)
from .dense_control import causal_window_dense

DEFAULTS = {"modular.branch": "causal_conv1d", "modular.core": "transformer_conv",
            "modular.fusion": "sequence_concat", "modular.head": "forecast",
            "modular.control_branch": "causal_window_dense"}


BUILTINS = {"modular.branch": {"causal_conv1d": causal_conv1d,
                                "causal_dense_sequence": causal_dense_sequence},
            "modular.control_branch": {"causal_window_dense": causal_window_dense},
            "modular.core": {"transformer_conv": transformer_conv},
            "modular.fusion": {"sequence_concat": sequence_concat},
            "modular.head": {"forecast": forecast}}


def _resolve(role, spec, groups):
    """Resolve one component through its entry-point group, deterministically.

    Built-in names have a fixed identity whether or not the distribution is
    installed; an installed entry point that reuses a built-in name must point at
    the very same factory (no shadowing). Any other name must be published by
    exactly one entry point. Unknown, ambiguous or undeclared plugins fail.
    """
    default_group = "modular." + role
    group = groups[role]
    name = spec["plugin"]
    matches = list(entry_points(group=group, name=name))
    builtin = BUILTINS[default_group].get(name) if group == default_group else None
    if builtin is not None:
        for ep in matches:
            if ep.load() is not builtin:
                raise ValueError(f"Entry point {group}:{name} shadows the built-in component")
        factory, implementation = builtin, "predictor_plugins.modular_temporal:" + name
        distribution, dist_version = "predictor", None
    else:
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one plugin {group}:{name}; got {len(matches)}")
        ep = matches[0]
        factory, implementation = ep.load(), ep.value
        distribution = ep.dist.name if ep.dist else None
        dist_version = ep.dist.version if ep.dist else None
    if getattr(factory, "component_role", None) != role or not isinstance(
            getattr(factory, "component_version", None), str):
        raise ValueError(f"Plugin {group}:{name} lacks a component() declaration for role {role}")
    identity = {"group": group, "name": name, "implementation": implementation,
                "version": factory.component_version}
    if builtin is None:
        identity.update(distribution=distribution, distribution_version=dist_version)
    return factory, identity


def describe_component(role, name, group=None):
    """Declared version, parameter names and tensor/time contract of one component."""
    factory, identity = _resolve(role, {"plugin": name}, {role: group or "modular." + role})
    return {**identity, "parameters": list(factory.component_parameters),
            "contract": factory.component_contract}
