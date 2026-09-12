"""C31 (order 2026-09-11): ONE resolver decides which plugin runs.

Before this module there were two independent lookups. `app/main.py`
chose the model with `predictor_plugin`; the eligibility identity
looked the predictor up under the legacy key `plugin`. A config
carrying `predictor_plugin="cnn"` with an inherited `plugin="ann"`
therefore TRAINED A CNN AND RECORDED AN ANN — the identity described a
run that never happened.

Two independent lookups can always disagree, so there is now one. It
returns a WITNESS, and both the loader and the identity consume that
same witness: whatever is executed is what is bound.

Three further rules it enforces:

  * the canonical key is the one the executor reads. A legacy alias may
    still be present, but if it names a DIFFERENT plugin the run
    refuses before a model object exists;
  * a name registered by more than one distribution refuses. "The first
    one wins" is not a policy, it is an accident that depends on
    installation order;
  * a plugin of THIS repository must resolve to a file inside THIS
    checkout. The audit found the identity depending on a sibling
    checkout's installed metadata; a registry entry pointing somewhere
    else is exactly that, and it refuses.

When the distribution is not installed at all, the resolver falls back
to the entry points THIS repository declares in `setup.py`, and records
which source it used. A clean checkout then reproduces the same
resolution without needing another checkout installed beside it.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]

#: role -> (canonical config key, entry-point group, legacy aliases).
#: The canonical key is the one `app/main.py` actually reads.
PLUGIN_ROLES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "predictor": ("predictor_plugin", "predictor.plugins", ("plugin",)),
    "optimizer": ("optimizer_plugin", "optimizer.plugins", ()),
    "pipeline": ("pipeline_plugin", "pipeline.plugins", ()),
    "preprocessor": ("preprocessor_plugin", "preprocessor.plugins", ()),
    "target": ("target_plugin", "target.plugins", ()),
}

REGISTRY = "INSTALLED_ENTRY_POINT_REGISTRY"
DECLARED = "REPOSITORY_SETUP_PY_DECLARATION"

#: the distribution this checkout is. Used ONLY to break a duplicate
#: registration explicitly, never to widen what may be loaded.
LOCAL_DISTRIBUTION = "predictor"


def _distribution_name(ep) -> str:
    dist = getattr(ep, "dist", None)
    name = getattr(dist, "name", None) or getattr(
        getattr(dist, "metadata", None), "get", lambda _k: None)("Name")
    return str(name) if name else "UNAVAILABLE"


class PluginResolutionRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


# ------------------------------------------------------------- naming
def canonical_name(config: dict, role: str) -> str | None:
    """The plugin name for `role`, with every alias agreeing.

    An alias that names a different plugin is not a preference to
    resolve — it is two configurations pretending to be one.
    """
    if role not in PLUGIN_ROLES:
        raise PluginResolutionRefusal(f"unknown plugin role {role!r}")
    key, _group, aliases = PLUGIN_ROLES[role]
    name = config.get(key)
    for alias in aliases:
        other = config.get(alias)
        if other in (None, ""):
            continue
        if name in (None, ""):
            raise PluginResolutionRefusal(
                f"{role}: the legacy key {alias!r} names {other!r} but "
                f"the canonical key {key!r} is unset — the executor "
                f"reads {key!r}, so a run would silently use a default "
                f"while the identity recorded {other!r}")
        if str(other) != str(name):
            raise PluginResolutionRefusal(
                f"{role}: {key}={name!r} and legacy {alias}={other!r} "
                "disagree. The executor obeys the canonical key, so "
                "this run would train one plugin and record another")
    return str(name) if name not in (None, "") else None


# --------------------------------------------------------- registries
def _installed_entry_points(group: str) -> list:
    from importlib import metadata
    try:
        return list(metadata.entry_points(group=group))
    except TypeError:                                   # older API
        return list(metadata.entry_points().get(group, []))


def _declared_entry_points(group: str,
                           setup_py: Path | None = None) -> dict:
    """The entry points THIS repository declares, read from setup.py.

    Parsed with `ast`, never executed: reading a build script by
    running it would make the identity depend on side effects.
    """
    setup_py = setup_py or (CODE_ROOT / "setup.py")
    if not setup_py.is_file():
        return {}
    try:
        tree = ast.parse(setup_py.read_text())
    except SyntaxError as exc:
        raise PluginResolutionRefusal(
            f"setup.py does not parse ({exc.__class__.__name__}), so "
            "the declared entry points cannot be read")
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords or []:
            if kw.arg != "entry_points":
                continue
            try:
                mapping = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                continue
            for spec in mapping.get(group, []) or []:
                if "=" in spec:
                    name, value = spec.split("=", 1)
                    out[name.strip()] = value.strip()
    return out


# ------------------------------------------------------------ witness
def resolve(role: str, name: str, *,
            code_root: Path | None = None) -> dict:
    """Resolve ONE role to the exact file that will be executed."""
    _key, group, _aliases = PLUGIN_ROLES[role]
    root = Path(code_root or CODE_ROOT)

    matches = [ep for ep in _installed_entry_points(group)
               if ep.name == str(name)]
    values = sorted({ep.value for ep in matches})
    duplicate_policy, rejected = "NONE", []
    distributions = sorted({_distribution_name(ep) for ep in matches})

    # E1 (audit 2026-09-12): the policy only fired when two
    # distributions disagreed about the VALUE. Two distributions
    # publishing the SAME (group, name, value) left two matches and
    # `matches[0]` was consumed — silently, and in installation order.
    # The resolved module is usually the same file, so nothing broke;
    # but "usually" is not a policy and the second distribution was not
    # even recorded. Every distribution publishing the name is now
    # bound, whether or not they agree.
    if len(values) == 1 and len(distributions) > 1:
        duplicate_policy = "IDENTICAL_VALUE_ALL_DISTRIBUTIONS_BOUND"
        rejected = [{"value": values[0], "distribution": d,
                     "note": "publishes the same value; bound, not "
                             "discarded"}
                    for d in distributions]

    if len(values) > 1:
        # An explicit, BOUND policy — never "the first one". The
        # entry-point groups here are shared with sibling applications
        # (`preprocessor.plugins` is also published by gym-fx and by
        # the standalone preprocessor), so two distributions really do
        # register the same name with different classes and the winner
        # would otherwise depend on installation order.
        #
        # The policy: the distribution THIS checkout belongs to wins,
        # because a run of this repository must execute this
        # repository's plugin. It applies only when it picks exactly
        # one; anything else refuses, and the rejected alternatives are
        # recorded either way.
        local = [ep for ep in matches
                 if _distribution_name(ep) == LOCAL_DISTRIBUTION]
        local_values = sorted({ep.value for ep in local})
        if len(local_values) != 1:
            raise PluginResolutionRefusal(
                f"{role}: the name {name!r} is registered in group "
                f"{group!r} by more than one distribution, pointing at "
                f"{values}, and the local-distribution policy does not "
                f"select exactly one ({local_values or 'none local'}). "
                "Taking the first would make the run depend on "
                "installation order, so this refuses")
        duplicate_policy = "LOCAL_DISTRIBUTION_WINS"
        rejected = [{"value": ep.value,
                     "distribution": _distribution_name(ep)}
                    for ep in matches if ep.value != local_values[0]]
        matches = local

    if matches:
        source, value = REGISTRY, matches[0].value
    else:
        declared = _declared_entry_points(group)
        if str(name) not in declared:
            raise PluginResolutionRefusal(
                f"{role}: entry point {name!r} is registered in neither "
                f"the installed registry nor this repository's setup.py "
                f"for group {group!r} — a plugin that cannot be "
                "resolved cannot be reviewed")
        source, value = DECLARED, declared[str(name)]

    module_name = value.split(":")[0].strip()
    import importlib.util as _ilu
    try:
        spec = _ilu.find_spec(module_name)
    except (ImportError, ValueError, ModuleNotFoundError) as exc:
        spec = None
        if source == REGISTRY:
            raise PluginResolutionRefusal(
                f"{role}: module {module_name!r} could not be located "
                f"({exc.__class__.__name__})")
    origin = Path(spec.origin) if (spec and spec.origin) else None
    if origin is None:
        # A declared-but-unimportable module is still locatable by its
        # dotted path inside this checkout; that is the file the run
        # would execute once the package is installed.
        candidate = root / (module_name.replace(".", "/") + ".py")
        if candidate.is_file():
            origin = candidate
    if origin is None or not origin.is_file():
        raise PluginResolutionRefusal(
            f"{role}: module {module_name!r} has no file origin — it "
            "cannot be hashed, so it cannot be reviewed")

    origin = origin.resolve()
    try:
        rel = str(origin.relative_to(root))
        inside = True
    except ValueError:
        rel, inside = str(origin), False
    if not inside and _is_local_group(group):
        raise PluginResolutionRefusal(
            f"{role}: {group}:{name} resolves to {origin.name} OUTSIDE "
            "this checkout. A plugin of this repository must come from "
            "this repository; an identity that binds a sibling "
            "checkout's file describes someone else's code")
    return {
        "role": role,
        "duplicate_policy": duplicate_policy,
        "rejected_duplicates": rejected,
        "publishing_distributions": distributions,
        "config_key": PLUGIN_ROLES[role][0],
        "entry_point_group": group,
        "entry_point_name": str(name),
        "entry_point_value": value,
        "resolution_source": source,
        "module": module_name,
        "origin": str(origin),
        "origin_id": rel,
        "inside_checkout": inside,
    }


def _is_local_group(group: str) -> bool:
    """Groups this repository itself publishes."""
    return group in {g for _k, g, _a in PLUGIN_ROLES.values()}


def resolve_all(config: dict, *,
                code_root: Path | None = None) -> dict:
    """Every declared role, resolved once, for everyone."""
    out: dict[str, dict] = {}
    for role in sorted(PLUGIN_ROLES):
        name = canonical_name(config, role)
        if not name:
            continue
        out[role] = resolve(role, name, code_root=code_root)
    return out


def load_from_witness(witness: dict):
    """Load the class named by a witness — the SAME file the identity
    bound. The loader and the identity never look things up twice."""
    import importlib
    value = witness["entry_point_value"]
    module_name, _, attr = value.partition(":")
    module = importlib.import_module(module_name.strip())
    obj = module
    for part in (attr or "").strip().split(".") if attr else []:
        obj = getattr(obj, part)
    return obj


def declared_plugin_params(witness: dict) -> dict:
    """The plugin's declared defaults, read WITHOUT importing it.

    C34: `SUBMIT_ONLY` needs a plugin's `plugin_params` to merge the
    contract those defaults belong to — and importing the module to
    get them pulls in TensorFlow, which is execution by any honest
    reading. The values are a class-level literal, so they can be read
    from the source with `ast` instead.

    The file is parsed, never executed. A module whose defaults are
    computed rather than declared cannot be read this way, and that
    refuses: guessing a contract is worse than admitting we cannot
    read it.
    """
    origin = Path(witness["origin"])
    value = witness["entry_point_value"]
    _module, _, attr = value.partition(":")
    class_name = (attr or "").split(".")[0].strip()
    try:
        tree = ast.parse(origin.read_text())
    except (OSError, SyntaxError) as exc:
        raise PluginResolutionRefusal(
            f"{witness['role']}: cannot read the declared parameters of "
            f"{origin.name} ({type(exc).__name__})")
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            targets = (stmt.targets if isinstance(stmt, ast.Assign)
                       else [stmt.target] if isinstance(stmt, ast.AnnAssign)
                       else [])
            for t in targets:
                if isinstance(t, ast.Name) and t.id == "plugin_params":
                    node_value = (stmt.value if isinstance(stmt, ast.Assign)
                                  else stmt.value)
                    try:
                        params = ast.literal_eval(node_value)
                    except (ValueError, SyntaxError):
                        raise PluginResolutionRefusal(
                            f"{witness['role']}: {class_name}."
                            "plugin_params is computed, not declared, so "
                            "it cannot be read without executing the "
                            "module — and SUBMIT_ONLY does not execute")
                    if not isinstance(params, dict):
                        raise PluginResolutionRefusal(
                            f"{witness['role']}: {class_name}."
                            "plugin_params is not a mapping")
                    return params
    raise PluginResolutionRefusal(
        f"{witness['role']}: {class_name} in {origin.name} declares no "
        "plugin_params; its contract cannot be merged without executing "
        "the module")
