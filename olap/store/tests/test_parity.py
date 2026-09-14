"""The packaged provider must not drift from the query plugin the service actually runs."""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEPLOYED = REPO / "olap" / "lake" / "query_plugins" / "sql_query.py"
PACKAGED = REPO / "olap" / "store" / "src" / "predictor_olap_store" / "query.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_packaged_query_plugin_is_the_deployed_one():
    assert DEPLOYED.is_file() and PACKAGED.is_file()
    assert _sha256(PACKAGED) == _sha256(DEPLOYED), (
        "the provider copy and the running query plugin differ; make the change in one place "
        "and copy it, or finish the migration and delete the copy")


def test_the_provider_declares_the_capabilities_the_inventory_implements():
    import sys

    sys.path.insert(0, str(REPO / "olap" / "store" / "src"))
    from predictor_olap_store.provider import CAPABILITIES, PredictorOlapStore

    store = PredictorOlapStore()
    for capability in CAPABILITIES:
        assert callable(getattr(store, capability)), capability
    assert store.capabilities() == CAPABILITIES
    identity = store.source_identity()
    assert identity["distribution"] == "predictor-olap-store"
