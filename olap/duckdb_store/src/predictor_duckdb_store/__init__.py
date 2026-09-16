"""The predictor OLAP cube on DuckDB.

D2 of `MUSASHI_DUCKDB_WAREHOUSE_MIGRATION_ORDER_2026_09_16.md`. This is an **external provider
through the data-warehouse host's existing plugin interface** — the same entry-point group, the
same capability names, the same governed result contracts. It creates no second governance API.

It deliberately does NOT reimplement the cube's logic. `predictor_olap_store.query.Plugin`
already holds the governed schema, the availability-contract dimension and the temporal
validation the producer's contract requires; this package subclasses it and replaces only what
the engine decides: how a connection is made, which schema qualifies a name, and the fact that
DuckDB admits exactly one writer. A second copy of those rules would agree with the first until
the day it did not, and the whole point of the dimension is that two readers cannot disagree.
"""

from .provider import PredictorDuckdbStore, backend

__all__ = ["PredictorDuckdbStore", "backend", "__version__"]
__version__ = "0.1.0"
