"""The predictor OLAP cube, packaged as a provider for a reusable warehouse host.

`predictor_olap_store.query` is the query plugin the production warehouse actually runs,
copied verbatim. It is copied rather than moved so the running service keeps loading exactly
what it loads today.

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
           "SOURCE_REVISION", "SOURCE_PATH"]
__version__ = "0.1.0"

#: The digest of the deployed query plugin this module was copied from, and the revision
#: that holds it. Both are checked by the package's tests.
SOURCE_SHA256 = "ded983563077a2aaac6893bf4acde0a2ce84dd41e084a0be10a0cae613f0567e"
SOURCE_REVISION = "a7a86e906fd5d4e4057237f014f8f95d0cf6ea6a"
SOURCE_PATH = "olap/lake/query_plugins/sql_query.py"
