"""The packaged provider must not drift from the query plugin the service actually runs.

The pin is checked against Git, not against whatever happens to be in this checkout: when
this package was made, the deployed revision was reachable only from the runtime worktree
and was not an ancestor of any branch, so a checkout comparison alone would have compared
the package against code the service does not run. That is exactly the drift this file
exists to catch, and the parity run against the live host found it: the branch copy
answered `/api/v1/discover` with row counts where the deployed plugin answers
`row_count_status: NOT_SCANNED` and lists the `gov_metric_current` view.
"""

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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob(revision: str, path: str):
    out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{revision}:{path}"],
                         capture_output=True)
    return out.stdout if out.returncode == 0 else None


def test_the_packaged_module_matches_the_digest_it_declares():
    assert PACKAGED.is_file()
    assert _sha256(PACKAGED.read_bytes()) == pkg.SOURCE_SHA256, (
        "the packaged module no longer matches the digest it declares; update both or stop "
        "calling it a copy")


def test_the_declared_revision_really_holds_those_bytes():
    blob = _blob(pkg.SOURCE_REVISION, pkg.SOURCE_PATH)
    if blob is None:
        pytest.skip(f"revision {pkg.SOURCE_REVISION} is not in this clone")
    assert _sha256(blob) == pkg.SOURCE_SHA256


def test_the_checkout_is_compared_and_any_difference_is_named():
    """If this branch carries that file, it must be the pinned one — or say which it is."""
    if not DEPLOYED.is_file():
        pytest.skip(f"{DEPLOYED.relative_to(REPO)} is not in this branch")
    here = _sha256(DEPLOYED.read_bytes())
    if here != pkg.SOURCE_SHA256:
        pytest.skip(f"this branch carries {here[:16]}… at {DEPLOYED.relative_to(REPO)}, while "
                    f"the package pins {pkg.SOURCE_SHA256[:16]}… from {pkg.SOURCE_REVISION[:12]}; "
                    "the package follows what the service runs, not what this branch holds")
    assert here == pkg.SOURCE_SHA256


def test_the_provider_declares_the_capabilities_it_implements():
    from predictor_olap_store.provider import CAPABILITIES, PredictorOlapStore

    store = PredictorOlapStore()
    for capability in CAPABILITIES:
        assert callable(getattr(store, capability)), capability
    assert store.capabilities() == CAPABILITIES
    assert store.source_identity()["distribution"] == "predictor-olap-store"
