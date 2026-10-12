"""I6-N lane D: candidate archive -> DOIN external_payload -> chain -> OLAP projection.

Modules: ``records`` (documents and projection), ``outbox`` (durable local state),
``publisher`` (record + pump), ``etl`` (finalized reference -> verified bytes -> warehouse ->
readback -> ACK), ``sink`` (warehouse clients), ``snapshot`` (physical copies with digests).
Requires doin-core and doin-node at the lane D branch on the import path.
"""

import sys
from pathlib import Path

_OLAP_SRC = Path(__file__).resolve().parents[2] / "olap" / "store" / "src"
if _OLAP_SRC.is_dir() and str(_OLAP_SRC) not in sys.path:
    sys.path.insert(0, str(_OLAP_SRC))
