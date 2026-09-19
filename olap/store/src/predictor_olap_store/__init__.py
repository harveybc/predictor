"""The predictor OLAP cube, packaged as a provider for a reusable warehouse host.

`predictor_olap_store.query` was copied verbatim from the query plugin the production
warehouse runs. It now carries one candidate change on top of it (see PENDING_REVIEW);
the deployed revision is unchanged and is still pinned below. It is copied rather than
moved so the running service keeps loading exactly what it loads today.

Pinned source: `olap/lake/query_plugins/sql_query.py` at revision
`a7a86e906fd5d4e4057237f014f8f95d0cf6ea6a`, SHA-256
`ded983563077a2aaac6893bf4acde0a2ce84dd41e084a0be10a0cae613f0567e`.
That revision is the one deployed on the warehouse service; at the time of packaging it was
reachable only from the runtime worktree, not from any branch, which is why the package
pins it explicitly and the tests check the pin against Git rather than against a checkout.
The copy disappears when the service tree and that revision are integrated.
"""

from .provider import PredictorOlapStore, backend

__all__ = ["PredictorOlapStore", "backend", "__version__", "SOURCE_SHA256",
           "SOURCE_REVISION", "SOURCE_PATH", "MODULE_SHA256", "PENDING_REVIEW",
           "ENVELOPE_SOURCE_PATH", "ENVELOPE_SOURCE_SHA256"]
__version__ = "0.1.1"

#: The digest of the deployed query plugin this module was copied from, and the revision
#: that holds it. Both are checked by the package's tests. These describe PRODUCTION and are
#: not touched by a candidate change: nothing here deploys anything.
SOURCE_SHA256 = "ded983563077a2aaac6893bf4acde0a2ce84dd41e084a0be10a0cae613f0567e"
SOURCE_REVISION = "a7a86e906fd5d4e4057237f014f8f95d0cf6ea6a"
SOURCE_PATH = "olap/lake/query_plugins/sql_query.py"

#: The campaign-envelope loader, packaged so the DuckDB provider can consume it
#: through a declared dependency instead of importing a checkout. It is a COPY of
#: `olap/campaign_envelope.py`, and the parity test fails the moment they differ:
#: two loaders would eventually disagree about what an envelope means.
ENVELOPE_SOURCE_PATH = "olap/campaign_envelope.py"
ENVELOPE_SOURCE_SHA256 = "d518cedce2617a454fcc49e5be304c0622a5d7a6f9e14f6593844ca1d4f96969"

#: The digest of the module in THIS branch. It differs from SOURCE_SHA256 exactly when a
#: candidate change is awaiting production review, and PENDING_REVIEW says which one. The
#: package's tests require the two to be consistent: a divergence without a stated reason is
#: a failure, and a stated reason without a divergence is one too.
MODULE_SHA256 = "7beab42dd7e647fc3593621013249e576d392a7d451f76b2d8403c771f9e489d"
PENDING_REVIEW = (
    "S2 availability-contract dimension: additive `gov_availability_contract` table, the "
    "`gov_delivery_availability` view and `write_availability_contracts` / "
    "`resolve_delivery_availability`, the reader VERIFYING the retained bytes and "
    "VALIDATING their temporal semantics against the producer's own rules before it "
    "displays anything (U2, V1). Additive and idempotent; no existing table, column, row or "
    "terminal digest is altered. Proved on SQLite and on a disposable PostgreSQL stack; NOT "
    "deployed. Candidate manifest: docs/audits/work_plan/SATOSHI_S2_PRODUCTION_CANDIDATE.md"
)
