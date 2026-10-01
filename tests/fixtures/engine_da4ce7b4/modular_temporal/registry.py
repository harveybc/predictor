"""Built-in component registry and external entry-point resolution."""

from importlib.metadata import entry_points

from .components import causal_conv1d, forecast, sequence_concat, transformer_conv

DEFAULTS = {"modular.branch": "causal_conv1d", "modular.core": "transformer_conv",
            "modular.fusion": "sequence_concat", "modular.head": "forecast"}


BUILTINS = {"modular.branch": {"causal_conv1d": causal_conv1d},
            "modular.core": {"transformer_conv": transformer_conv},
            "modular.fusion": {"sequence_concat": sequence_concat},
            "modular.head": {"forecast": forecast}}


def _resolve(role, spec, groups):
    default_group = "modular." + role
    group = groups[role]
    name = spec["plugin"]
    if group == default_group and name in BUILTINS[default_group]:
        return BUILTINS[default_group][name], {"group": group, "name": name,
                                               "implementation": "modular_temporal.v1:" + name}
    matches = list(entry_points(group=group, name=name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one plugin {group}:{name}; got {len(matches)}")
    ep = matches[0]
    identity = {"group": group, "name": name, "implementation": ep.value,
                "distribution": ep.dist.name if ep.dist else None,
                "version": ep.dist.version if ep.dist else None}
    return ep.load(), identity
