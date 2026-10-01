"""Versioned nested modular configuration <-> the optimizer's flat parameter interface.

The DEAP/DOIN optimizer plugins treat a candidate as a flat mapping
``{parameter_name: value}`` written into the run config. The modular predictor
is configured by a nested document (schema ``predictor.modular.v1``, see
``predictor_plugins.modular_temporal``). This module is the explicit, reversible
mapping between the two:

``modular.<key>``                      top-level scalar/list settings (window,
                                       sample_hours, horizons, regime, ...);
                                       ``modular.branch_steps`` is DERIVED: it
                                       must equal ``modular.window`` (branches
                                       keep every step) and anything else fails
``modular.branch_order``               the ordered branch names (keeps order)
``branches.<name>.<field>``            features, plugin, regime, donor
``branches.<name>.params.<p>[.<q>]``   branch plugin parameters
``core.<field>`` / ``core.params.<p>`` core plugin, regime, donor, parameters
``fusion.plugin`` / ``fusion.params.<p>``
``head.plugin`` / ``head.params.<p>``

``flatten(config)`` and ``unflatten(flat)`` are exact inverses on normalized
configurations. ``apply_flat_overrides(config, run_config)`` takes every key of a
flat run config that belongs to this namespace and applies it; a key naming an
unknown branch, an unknown field or a malformed path fails. Plugin-specific
parameter names and conditional combinations (for example heads that do not
divide d_model, or four compression widths with three time factors) are checked
by the component factories when the model is built -- which the predictor
plugin does in ``build_model``, before any fit.

Relation to the DOIN candidate layer (M04, ``tools/modular_search_space.py``):
that module is a tied search-space projection (schema ``modular.candidate.v1``,
uniform branch parameters, contiguous feature groups, indexed stage scalars and
``train.*`` evaluator settings). This module is the complete reversible encoding
of the model schema ``predictor.modular.v1``. The two are layers that compose;
the facade does NOT accept M04's keys directly. The only integration path is
M04 ``from_flat`` -> nested candidate -> its ``model`` -> the facade's ``modular``
config. The shared ``core.`` prefix cannot shadow: this grammar only accepts
``core.plugin|regime|donor`` and ``core.params.<p>``, so a raw M04 key such as
``core.d_model`` or ``core.stage_count`` FAILS loudly instead of being ignored,
and M04's ``branch.``/``model.``/``train.`` prefixes are outside this namespace
(``tests/test_modular_flat_parity_m04.py`` pins both facts on the 3ceabfad fixtures).
"""
from __future__ import annotations

import copy
import json
import re

TOP = "modular."
PREFIXES = ("modular.", "branches.", "core.", "fusion.", "head.")
_TOP_KEYS = {"schema", "window", "sample_hours", "feature_names", "branch_steps",
             "output_steps", "output_channels", "horizons", "target_count", "regime",
             "alignment_probe", "entry_point_groups"}
_OPTIONAL_TOP_KEYS = {"budget_caps", "excluded_features", "donor_contract"}   # flattened only when present (digest-neutral)
_FIELDS = {"branch": {"features", "plugin", "regime", "donor"},
           "core": {"plugin", "regime", "donor"},
           "fusion": {"plugin"}, "head": {"plugin"}}
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def is_modular_key(key) -> bool:
    return isinstance(key, str) and key.startswith(PREFIXES)


def _flat_params(prefix, params, out):
    for key, value in params.items():
        if not isinstance(key, str) or not _IDENT.match(key):
            raise ValueError(f"Parameter name {key!r} cannot be addressed by a flat key")
        if isinstance(value, dict) and value:
            _flat_params(f"{prefix}{key}.", value, out)
        else:
            out[prefix + key] = copy.deepcopy(value)


def flatten(config: dict) -> dict:
    """Normalized nested config -> flat {dotted_key: value}, deterministic key order."""
    from predictor_plugins.modular_temporal import _normalize
    c = _normalize(config)
    out = {}
    for key in sorted(_TOP_KEYS):
        if key == "entry_point_groups":
            for role, group in sorted(c[key].items()):
                out[f"{TOP}entry_point_groups.{role}"] = group
        else:
            out[TOP + key] = copy.deepcopy(c[key])
    for key in sorted(_OPTIONAL_TOP_KEYS & set(c)):
        out[TOP + key] = copy.deepcopy(c[key])
    out[TOP + "branch_order"] = [b["name"] for b in c["branches"]]
    for spec in c["branches"]:
        for field in sorted(_FIELDS["branch"]):
            out[f"branches.{spec['name']}.{field}"] = copy.deepcopy(spec[field])
        _flat_params(f"branches.{spec['name']}.params.", spec["params"], out)
    for role in ("core", "fusion", "head"):
        for field in sorted(_FIELDS[role]):
            out[f"{role}.{field}"] = copy.deepcopy(c[role][field])
        _flat_params(f"{role}.params.", c[role]["params"], out)
    return dict(sorted(out.items()))


def _set_param(params, path, value, key):
    node = params
    for part in path[:-1]:
        if not _IDENT.match(part):
            raise ValueError(f"Malformed flat key {key!r}")
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Flat key {key!r} descends into a non-mapping parameter")
    if not path or not _IDENT.match(path[-1]):
        raise ValueError(f"Malformed flat key {key!r}")
    node[path[-1]] = copy.deepcopy(value)


def _apply(nested, key, value):
    parts = key.split(".")
    if parts[0] == "modular":
        if len(parts) == 3 and parts[1] == "entry_point_groups":
            nested.setdefault("entry_point_groups", {})[parts[2]] = value
        elif len(parts) == 2 and parts[1] == "branch_order":
            raise ValueError("modular.branch_order is structural; change branches in the nested config")
        elif len(parts) == 2 and parts[1] in (_TOP_KEYS | _OPTIONAL_TOP_KEYS) - {"entry_point_groups"}:
            nested[parts[1]] = copy.deepcopy(value)
        else:
            raise ValueError(f"Unknown modular flat key {key!r}")
        return
    if parts[0] == "branches":
        if len(parts) < 3:
            raise ValueError(f"Malformed flat key {key!r}")
        spec = next((b for b in nested["branches"] if b.get("name") == parts[1]), None)
        if spec is None:
            raise ValueError(f"Flat key {key!r} names an unknown branch {parts[1]!r}")
        role, rest = "branch", parts[2:]
    elif parts[0] in ("core", "fusion", "head"):
        spec = nested.setdefault(parts[0], {})
        role, rest = parts[0], parts[1:]
    else:
        raise ValueError(f"Unknown flat key {key!r}")
    if rest and rest[0] == "params":
        _set_param(spec.setdefault("params", {}), rest[1:], value, key)
    elif len(rest) == 1 and rest[0] in _FIELDS[role]:
        spec[rest[0]] = copy.deepcopy(value)
    else:
        raise ValueError(f"Unknown field in flat key {key!r} for role {role}")


def unflatten(flat: dict) -> dict:
    """Flat mapping produced by ``flatten`` -> normalized nested config."""
    from predictor_plugins.modular_temporal import _normalize
    order = flat.get(TOP + "branch_order")
    if not isinstance(order, list) or not order:
        raise ValueError("modular.branch_order is required to rebuild the branch list")
    nested = {"branches": [{"name": name} for name in order]}
    for key, value in flat.items():
        if not is_modular_key(key):
            raise ValueError(f"Flat key {key!r} is outside the modular namespace")
        if key == TOP + "branch_order":
            continue
        _apply(nested, key, value)
    return _normalize(nested)


def apply_flat_overrides(config: dict, run_config: dict) -> tuple[dict, dict]:
    """Apply every modular-namespace key of a flat run config to a nested config.

    Returns (normalized nested config, the applied overrides). Unknown branches,
    fields or malformed keys fail; plugin parameter validity is checked at build.
    """
    from predictor_plugins.modular_temporal import _normalize
    nested = copy.deepcopy(config)
    applied = {}
    for key in sorted(k for k in run_config if is_modular_key(k)):
        _apply(nested, key, run_config[key])
        applied[key] = copy.deepcopy(run_config[key])
    return _normalize(nested), applied


def dumps(config: dict) -> str:
    """Deterministic JSON of the normalized configuration (sorted keys, no NaN)."""
    from predictor_plugins.modular_temporal import canonical_config_json
    return canonical_config_json(config)


def loads(text: str) -> dict:
    from predictor_plugins.modular_temporal import _normalize
    return _normalize(json.loads(text))


def budget(config, *, build=True):
    """Measured shape budget (raw channels, branches, fused width = sum of branch widths, fused time,
    materialization bytes per row, parameters) read from the engine, for M04's budget model."""
    from predictor_plugins.modular_temporal import measure_budget
    return measure_budget(config, build=build)
