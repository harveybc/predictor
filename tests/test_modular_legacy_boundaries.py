"""Legacy boundaries around the opt-in modular predictor (M01).

Run from a checkout these check the declared entry points; run from a copy of
this file OUTSIDE the checkout, in an isolated environment with the package
installed non-editable (M01_REQUIRE_INSTALLED=1), they check the installed
registry and prove the installed code, not the checkout, was imported.
"""
import json
import os
from importlib import metadata
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = json.loads((FIXTURES / "legacy_entry_points_dc72170e.json").read_text())["entry_points"]
ADDED = {"predictor.plugins": {"modular_temporal": "predictor_plugins.predictor_plugin_modular:Plugin"}}
# M04 registers its opt-in optimizer in its own lane; allowed, never a changed legacy value.
ALLOWED_OTHER_ADDITIONS = {"optimizer.plugins": {"modular_doin_optimizer"}}
MODULAR_GROUPS = {"modular.branch": "causal_conv1d", "modular.fusion": "sequence_concat",
                  "modular.core": "transformer_conv", "modular.head": "forecast"}
REQUIRE_INSTALLED = os.environ.get("M01_REQUIRE_INSTALLED") == "1"


def _installed(group):
    try:
        dist = metadata.distribution("predictor")
    except metadata.PackageNotFoundError:
        return None
    rows = [ep for ep in dist.entry_points if ep.group == group]
    if "modular_temporal" not in {ep.name for ep in dist.entry_points if ep.group == "predictor.plugins"}:
        return None           # some other (stale) predictor distribution is installed
    return {ep.name: ep.value for ep in rows}


def _declared(group):
    from app.plugin_resolver import _declared_entry_points
    return _declared_entry_points(group)


def _tables():
    tables = {}
    for group in [*LEGACY, *MODULAR_GROUPS]:
        installed = _installed(group)
        if REQUIRE_INSTALLED:
            assert installed is not None, "the M01 predictor distribution is not installed"
        tables[group] = installed if installed is not None else _declared(group)
    return tables


def test_installed_code_is_what_runs_when_required():
    import predictor_plugins
    import app
    if not REQUIRE_INSTALLED:
        pytest.skip("checkout run; the isolated installed run sets M01_REQUIRE_INSTALLED=1")
    for module in (predictor_plugins, app):
        assert "site-packages" in module.__file__, module.__file__


def test_every_legacy_entry_point_is_byte_identical_and_only_opt_in_names_are_added():
    tables = _tables()
    for group, legacy in LEGACY.items():
        now = tables[group]
        for name, value in legacy.items():
            assert now.get(name) == value, (group, name, now.get(name), value)
        added = set(now) - set(legacy)
        expected = set(ADDED.get(group, {}))
        assert expected <= added and added - expected <= ALLOWED_OTHER_ADDITIONS.get(group, set()), \
            (group, added)
        for name, value in ADDED.get(group, {}).items():
            assert now[name] == value
    for group, name in MODULAR_GROUPS.items():
        assert tables[group] == {name: f"predictor_plugins.modular_temporal:{name}"}


def test_legacy_names_resolve_to_legacy_classes_never_to_the_modular_plugin():
    from app.plugin_loader import load_plugin
    from app.plugin_resolver import resolve
    for name, value in LEGACY["predictor.plugins"].items():
        if name == "base":        # historically declared with a broken module path; unchanged
            continue
        witness = resolve("predictor", name)
        assert witness["entry_point_value"] == value
        assert "predictor_plugin_modular" not in witness["origin"]
    ann, _ = load_plugin("predictor.plugins", "ann")
    modular, params = load_plugin("predictor.plugins", "modular_temporal")
    assert ann.__module__ == "predictor_plugins.predictor_plugin_ann"
    assert modular.__module__ == "predictor_plugins.predictor_plugin_modular"
    assert "modular" in params
    with pytest.raises(BaseException):
        resolve("predictor", "modular_temporal_typo")


def test_modular_components_resolve_through_their_entry_points():
    from predictor_plugins import modular_temporal as mt
    for group, name in MODULAR_GROUPS.items():
        role = group.split(".")[1]
        eps = list(metadata.entry_points(group=group, name=name))
        if REQUIRE_INSTALLED:
            assert len(eps) == 1 and eps[0].load() is mt.BUILTINS[group][name]
        assert mt.describe_component(role, name)["version"] == "1.0.0"
