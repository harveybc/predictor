"""Authenticated JSON-over-HTTP front for the coordinator ledger (stdlib only).

Workers on other hosts call the same methods as ``Ledger``. The bearer token
comes from a private 0600 file; the bind address comes from the private
campaign config. Only the whitelisted methods are served.

Run (installed as the campaign's ledger unit, never by hand on a worker)::

    python -m tools.i6n_swarm.lease_service --ledger <root>/state/ledger.sqlite3 \
        --bind <private address> --port <port> --token-file <0600 file>
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

from tools.i6n_swarm import ledger as L

METHODS = ("ensure_unit", "claim", "renew", "complete", "report", "recover", "adopt_prior", "unit", "units",
           "attempts", "terminal", "dag_nodes", "dag_event", "dag_events", "put_search_state", "search_state", "running_on_host", "register_prior", "mark_wait", "terminals",
           "next_claimable")
ERRORS = {"ReceiptMismatch": L.ReceiptMismatch, "BudgetExhausted": L.BudgetExhausted, "LedgerError": L.LedgerError}


class LedgerUnavailable(ConnectionError):
    pass


class LedgerAuthError(PermissionError):
    pass


class LedgerRequestError(ValueError):
    pass


def _token(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise PermissionError("ledger token file must not be group/world readable")
    return path.read_text().strip()


def make_handler(ledger, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
                return self._send(401, dict(error="LedgerAuthError", detail="bad token"))
            name = self.path.rsplit("/", 1)[-1]
            if not self.path.startswith("/v1/") or name not in METHODS:
                return self._send(404, dict(error="LedgerRequestError", detail="unknown method"))
            try:
                length = int(self.headers.get("Content-Length", "0"))
                kwargs = json.loads(self.rfile.read(length) or b"{}")
                result = getattr(ledger, name)(**kwargs)
            except L.LedgerError as exc:
                return self._send(409, dict(error=type(exc).__name__, detail=str(exc)))
            except (TypeError, ValueError) as exc:
                return self._send(400, dict(error="LedgerRequestError", detail=str(exc)))
            return self._send(200, dict(result=result))
    return Handler


def start_server(ledger, host, port, token_file):
    server = ThreadingHTTPServer((host, port), make_handler(ledger, _token(token_file)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class LedgerClient:
    """Same call surface as ``Ledger``; network failures raise LedgerUnavailable."""

    def __init__(self, url, token_file, timeout=15.0):
        self.url = url.rstrip("/")
        self.token = Path(token_file).read_text().strip()
        self.timeout = timeout

    def _call(self, name, kwargs):
        request = urllib.request.Request(f"{self.url}/v1/{name}", data=json.dumps(kwargs, allow_nan=False).encode(),
                                         headers={"Authorization": f"Bearer {self.token}",
                                                  "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())["result"]
        except urllib.error.HTTPError as exc:
            try:
                body = json.loads(exc.read())
            except ValueError:
                body = dict(error="LedgerRequestError", detail=str(exc))
            if exc.code == 401:
                raise LedgerAuthError(body.get("detail")) from None
            raise ERRORS.get(body.get("error"), LedgerRequestError)(body.get("detail")) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise LedgerUnavailable(str(exc)) from exc

    def __getattr__(self, name):
        if name not in METHODS:
            raise AttributeError(name)
        return lambda *args, **kwargs: self._call(name, self._kwargs(name, args, kwargs))

    @staticmethod
    def _kwargs(name, args, kwargs):
        import inspect
        params = list(inspect.signature(getattr(L.Ledger, name)).parameters)[1:]
        bound = dict(zip(params, args))
        bound.update(kwargs)
        return bound


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", required=True)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(L.Ledger(args.ledger), _token(args.token_file)))
    server.serve_forever()


if __name__ == "__main__":
    main()
