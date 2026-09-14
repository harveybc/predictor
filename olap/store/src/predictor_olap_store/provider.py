"""`predictor_olap` — the cube as an installable backend of a warehouse host.

It owns the star schema, the governed `gov_*` append-only tables and the reporting ETL.
It owns no HTTP route and no governance decision. Everything it exposes is read-only SQL
plus the two append-only writes the campaign protocol needs; nothing here can truncate the
cube, and the host has no route that would ask it to.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .query import Plugin as _Query

CAPABILITIES = ("describe", "storage", "discover", "query",
                "write_metrics", "write_terminal", "terminal_digests")


def _source_commit() -> str | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            try:
                out = subprocess.run(["git", "-C", str(parent), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, timeout=10)
                return out.stdout.strip() if out.returncode == 0 else None
            except OSError:
                return None
    return None


class PredictorOlapStore(_Query):
    def capabilities(self):
        return CAPABILITIES

    def source_identity(self):
        from . import __version__

        return {"kind": "python_distribution", "distribution": "predictor-olap-store",
                "version": __version__, "module": __name__, "source_commit": _source_commit()}


def backend():
    return PredictorOlapStore()
