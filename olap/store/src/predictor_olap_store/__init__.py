"""The predictor OLAP cube, packaged as a provider for a reusable warehouse host.

`predictor_olap_store.query` is the deployed query plugin verbatim
(`predictor/olap/lake/query_plugins/sql_query.py`). It is copied, not rewritten, so that the
provider and the running service cannot drift silently: `tests/test_parity.py` compares
their SHA-256. The copy disappears when the host migration completes.
"""

from .provider import PredictorOlapStore, backend

__all__ = ["PredictorOlapStore", "backend", "__version__"]
__version__ = "0.1.0"
