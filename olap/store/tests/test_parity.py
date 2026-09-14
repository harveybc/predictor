"""The packaged provider must not drift from the query plugin the service actually runs."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "olap" / "store" / "src"
DEPLOYED = REPO / "olap" / "lake" / "query_plugins" / "sql_query.py"
PACKAGED = SRC / "predictor_olap_store" / "query.py"

sys.path.insert(0, str(SRC))
import predictor_olap_store as pkg  # noqa: E402


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def test_the_packaged_query_plugin_matches_its_declared_source():
    assert PACKAGED.is_file()
    assert _sha256(PACKAGED) == pkg.SOURCE_SHA256, (
        "the packaged module no longer matches the digest it declares; update both or "
        "stop claiming it is a copy")


def test_it_matches_the_deployed_plugin_when_that_tree_is_present():
    """On a checkout that also carries the service tree, the copy must equal what runs."""
    if not DEPLOYED.is_file():
        pytest.skip(f"{DEPLOYED.relative_to(REPO)} is not in this branch; "
                    f"the declared source is {pkg.SOURCE_REVISION}")
    assert _sha256(PACKAGED) == _sha256(DEPLOYED), (
        "the provider copy and the running query plugin differ; make the change in one "
        "place and copy it, or finish the migration and delete the copy")


def test_the_declared_source_revision_still_holds_that_file():
    """The pinned revision is checked against Git, not taken on trust."""
    out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob",
                          f"{pkg.SOURCE_REVISION}:{pkg.SOURCE_PATH}"],
                         capture_output=True)
    if out.returncode != 0:
        pytest.skip("the declared source revision is not present in this clone")
    assert _sha256_bytes(out.stdout) == pkg.SOURCE_SHA256


def test_the_provider_declares_the_capabilities_it_implements():
    from predictor_olap_store.provider import CAPABILITIES, PredictorOlapStore

    store = PredictorOlapStore()
    for capability in CAPABILITIES:
        assert callable(getattr(store, capability)), capability
    assert store.capabilities() == CAPABILITIES
    assert store.source_identity()["distribution"] == "predictor-olap-store"
