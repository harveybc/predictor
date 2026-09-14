"""The predictor OLAP cube, packaged as a provider for a reusable warehouse host.

`predictor_olap_store.query` is the deployed query plugin verbatim. It is copied, not
rewritten, so that the provider and the running service cannot drift silently:
`tests/test_parity.py` compares their SHA-256 whenever the deployed file is present in the
checkout, and against `SOURCE_SHA256` otherwise.

Pinned source: `predictor/olap/lake/query_plugins/sql_query.py` at revision
`939fb413378b8e4c8a62432de3a5005bbf44ea4a` (branch `satoshi/c166-c184-20260913`), SHA-256
`47b011613ca5c0cef3ee68d8dca98f80e6b64285d7625935f936312753b6df0e`. The service tree is being integrated into the default
branch; until it is, that revision is the declared installable origin of this module.
"""

from .provider import PredictorOlapStore, backend

__all__ = ["PredictorOlapStore", "backend", "__version__", "SOURCE_SHA256",
           "SOURCE_REVISION", "SOURCE_PATH"]
__version__ = "0.1.0"

#: The digest of the deployed query plugin this module was copied from, and the revision
#: that holds it. Both are checked by the package's parity test.
SOURCE_SHA256 = "47b011613ca5c0cef3ee68d8dca98f80e6b64285d7625935f936312753b6df0e"
SOURCE_REVISION = "939fb413378b8e4c8a62432de3a5005bbf44ea4a"
SOURCE_PATH = "olap/lake/query_plugins/sql_query.py"
