"""Warehouse sinks for the archive ETL.

``DirectSink`` calls a provider backend in-process (throwaway SQLite/DuckDB in tests, or the
owner process itself). ``HttpSink`` calls the warehouse host routes
``/api/v2/doin/projections[/revoke]`` with a bearer token obtained at call time from the
caller's ``token_provider`` (e.g. an environment variable set by an existing service's
EnvironmentFile). The token is never stored in the object or rendered.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable


class DirectSink:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def write(self, document: dict) -> dict:
        return self.backend.write_doin_projection(document)

    def read(self, document: dict) -> dict:
        return self.backend.read_doin_projection(document)

    def revoke(self, document: dict) -> dict:
        return self.backend.revoke_doin_projection(document)


def _urllib(method: str, url: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, method=method, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - configured URL
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class HttpSink:
    def __init__(self, base_url: str, *, token_provider: Callable[[], str], transport: Callable | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._token = token_provider
        self._transport = transport or _urllib

    def __repr__(self) -> str:
        return "HttpSink(<configured url>)"

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Authorization": "Bearer " + str(self._token())}
        data = None
        if body is not None:
            data = json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        status, raw = self._transport(method, self._base + path, data=data, headers=headers)
        if status not in (200, 201):
            raise RuntimeError(f"WAREHOUSE_HTTP_{status}: {raw[:300].decode('utf-8', 'replace')}")
        return json.loads(raw)

    def write(self, document: dict) -> dict:
        return self._call("POST", "/api/v2/doin/projections", document)

    def read(self, document: dict) -> dict:
        query = urllib.parse.urlencode([(k, v) for k, values in document.items()
                                        for v in (values if isinstance(values, list) else [values])])
        return self._call("GET", "/api/v2/doin/projections?" + query)

    def revoke(self, document: dict) -> dict:
        return self._call("POST", "/api/v2/doin/projections/revoke", document)
