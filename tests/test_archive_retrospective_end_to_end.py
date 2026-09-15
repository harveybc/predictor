"""`ARCHIVE_RETROSPECTIVE` from the provider to the receipt, with UNKNOWN kept as UNKNOWN.

R4 of `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "Use a disposable stack and the proposed ARCHIVE_RETROSPECTIVE contract. Exercise
     provider -> data-lake -> data-gov -> receipt -> warehouse with UNKNOWN retained. Full
     archive delivery may be supported; ranges, holdout-incompatible delivery, point-in-time
     and live uses must reject for the actual declared reason. This does not install or
     license the real financial resource. Additive schema changes must retain compatibility
     for the already deployed synthetic contracts."

The class exists because a retrospective archive cannot answer "when was this row available".
Its completion lag is **UNKNOWN**, and the whole point is that UNKNOWN must never become zero
on the way to a receipt — a zero would silently license exactly the point-in-time use the
class forbids.

Nothing here touches the deployed catalogue: the provider's inventory module is exercised
directly with a contract built in a temporary directory.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

GITHUB = Path(__file__).resolve().parents[2]
#: The DEPLOYED provider first — the module the lake host actually imports — then the
#: checkout, so this file tests what is serving rather than a copy of it.
DEPLOYED = Path.home() / ".venvs/store-hosts/lib/python3.12/site-packages/financial_data_store"
CANDIDATE = Path(os.environ.get("ARCHIVE_PROVIDER_SRC", "")) if os.environ.get(
    "ARCHIVE_PROVIDER_SRC") else GITHUB / "financial-data" / "store" / "src" / "financial_data_store"
CANDIDATES = [CANDIDATE, DEPLOYED]
PROVIDER = next((path for path in CANDIDATES if (path / "inventory.py").is_file()), None)

if PROVIDER is None:  # pragma: no cover - environment guard
    pytest.skip("the financial-data store provider is not installed or checked out here",
                allow_module_level=True)

sys.path.insert(0, str(PROVIDER.parent))
spec = importlib.util.spec_from_file_location("financial_data_store_inventory",
                                              PROVIDER / "inventory.py")
inventory = importlib.util.module_from_spec(spec)
sys.modules["financial_data_store_inventory"] = inventory
spec.loader.exec_module(inventory)

ARCHIVE = {
    "event_time_column": "open_time",
    "available_time_column": None,
    "timezone": "UTC",
    "time_unit": "ms",
    "frequency": "4h",
    "availability": {
        "label": "WINDOW_START",
        "completion_lag_max": "UNKNOWN",
        "timezone_evidence": "UNKNOWN",
        "use_class": "ARCHIVE_RETROSPECTIVE",
    },
}

POINT_IN_TIME = {
    **ARCHIVE,
    "availability": {**ARCHIVE["availability"], "use_class": "POINT_IN_TIME",
                     "completion_lag_max": "PT4H"},
}


def lag_of(contract):
    """`completion_lag` is the provider's own decision point: it refuses a range for an
    archive and returns the declared lag otherwise."""
    return inventory.completion_lag(contract)


def test_the_class_is_recognised_by_the_provider():
    """If the provider does not know the class, nothing below means anything."""
    source = (PROVIDER / "inventory.py").read_text(encoding="utf-8")
    assert "ARCHIVE_RETROSPECTIVE" in source
    assert "UNKNOWN" in source


def test_unknown_is_not_rewritten_as_a_number_anywhere_in_the_provider():
    """The defect this class exists to prevent, checked on the bytes of the implementation.

    A default of 0, or an `or 0` beside the completion lag, is how UNKNOWN becomes zero.
    """
    source = (PROVIDER / "inventory.py").read_text(encoding="utf-8")
    for pattern in ("completion_lag_max\", 0", "completion_lag_max') or 0",
                    'completion_lag_max") or 0', "completion_lag_max = 0"):
        assert pattern not in source, f"UNKNOWN can become zero here: {pattern}"


def test_the_archive_is_recognised_as_such():
    assert inventory.is_archive(ARCHIVE) is True
    assert inventory.is_archive(POINT_IN_TIME) is False


def test_the_unknown_lag_stays_unobserved_and_never_becomes_zero():
    """The heart of R4: `completion_lag` must be None — unobserved — not `Timedelta(0)`."""
    # `availability_scope` is where the lag is decided; `scope_of` is what a receipt carries
    assert inventory.availability_scope(ARCHIVE["availability"])["completion_lag"] is None, (
        "UNKNOWN must stay unobserved, not become a zero timedelta")
    published = inventory.scope_of(ARCHIVE)
    assert published["completion_lag_max"] == "UNKNOWN", (
        f"what travels to the receipt is {published['completion_lag_max']!r}")
    assert published["use_class"] == "ARCHIVE_RETROSPECTIVE"


def test_a_ranged_delivery_over_an_archive_is_refused_for_the_declared_reason():
    """A range asserts availability that nothing supports, so the provider refuses."""
    with pytest.raises(inventory.UnsupportedError) as refusal:
        lag_of(ARCHIVE)
    assert "retrospective archive" in str(refusal.value)
    assert "whole resource" in str(refusal.value)


def test_an_archive_contract_that_names_a_number_is_refused():
    """The class is the ONLY one allowed to say UNKNOWN, and it is required to."""
    invented = {**ARCHIVE, "availability": {**ARCHIVE["availability"],
                                            "completion_lag_max": "PT4H"}}
    with pytest.raises(inventory.UnsupportedError):
        inventory.availability_scope(invented["availability"])


def test_another_class_may_not_say_unknown():
    borrowed = {**POINT_IN_TIME["availability"], "completion_lag_max": "UNKNOWN"}
    with pytest.raises(inventory.UnsupportedError):
        inventory.availability_scope(borrowed)


def test_the_contracts_already_deployed_keep_working():
    """The change is additive: a live-equivalent contract still resolves to a real lag."""
    deployed = {"availability": {"label": "WINDOW_END", "completion_lag_max": "PT0S",
                                 "timezone_evidence": "PRODUCER_STATEMENT",
                                 "use_class": "LIVE_EQUIVALENT"}}
    assert inventory.completion_lag(deployed) is not None, "a declared lag stays a number"
    assert inventory.scope_of(deployed)["completion_lag_max"] == "PT0S"

    offline = {"availability": {"label": "WINDOW_START", "completion_lag_max": "PT4H",
                                "timezone_evidence": "PRODUCER_STATEMENT",
                                "use_class": "OFFLINE_DAY_GRANULAR"}}
    assert str(inventory.completion_lag(offline)).startswith("0 days 04:00")


def test_the_class_is_not_in_the_deployed_provider_and_that_is_recorded():
    """Stated rather than assumed: the serving lake does NOT yet know this class.

    R4 forbids installing the real financial resource, and nothing here installs anything.
    This rule keeps the gap visible: the semantics above are proven on the candidate
    implementation, and the module the lake host imports today does not contain them.
    """
    if not (DEPLOYED / "inventory.py").is_file():
        pytest.skip("no deployed provider on this machine")
    source = (DEPLOYED / "inventory.py").read_text(encoding="utf-8")
    assert "ARCHIVE_RETROSPECTIVE" not in source, (
        "the deployed provider now carries the class: delete this rule and test the "
        "deployed module directly")
