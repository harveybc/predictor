"""Deploy checks that receive their token ONLY through the process environment.

Run under the existing unit's EnvironmentFile, never by reading a token file:

    systemd-run --user --wait --pipe --collect -p EnvironmentFile=<existing env file> \
        -p WorkingDirectory=<pinned predictor checkout> <python> -m tools.i6n_archive.probe lake \
        --url http://127.0.0.1:<lake-cas-port> --token-env DATA_GOV_LAKE_TOKEN

``lake``       GET /api/v1/host (byte_upload/byte_read declared) and a 3-byte PUT/GET probe.
``warehouse``  GET /api/v1/host (doin projection capabilities) and read-only counts.
Output is JSON without the token; the token is never printed, logged or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse

from tools.i6n_archive.sink import _urllib

COUNTS = ("gov_report", "gov_metric", "gov_doin_projection")


def _token(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"TOKEN_ENV_MISSING: {name}; run under the unit's EnvironmentFile")
    return value


def _get(transport, url: str, token: str):
    status, raw = transport("GET", url, headers={"Authorization": "Bearer " + token})
    return status, raw


def probe_lake(url: str, token_env: str, transport=_urllib) -> dict:
    token = _token(token_env)
    base = url.rstrip("/")
    status, raw = _get(transport, base + "/api/v1/host", token)
    caps = json.loads(raw).get("capabilities", []) if status == 200 else []
    body = b"i6n"
    digest = hashlib.sha256(body).hexdigest()
    put_status, _ = transport("PUT", f"{base}/api/v1/objects/{digest}", data=body,
                              headers={"Authorization": "Bearer " + token,
                                       "Content-Type": "application/octet-stream"})
    get_status, got = _get(transport, f"{base}/api/v1/objects/{digest}", token)
    ok = status == 200 and {"byte_upload", "byte_read"} <= set(caps) and put_status in (200, 201) \
        and get_status == 200 and hashlib.sha256(got).hexdigest() == digest
    return {"check": "lake", "host_status": status, "capabilities_ok": {"byte_upload", "byte_read"} <= set(caps),
            "put_status": put_status, "get_status": get_status, "ok": ok}


def probe_warehouse(url: str, token_env: str, transport=_urllib) -> dict:
    token = _token(token_env)
    base = url.rstrip("/")
    status, raw = _get(transport, base + "/api/v1/host", token)
    caps = set(json.loads(raw).get("capabilities", [])) if status == 200 else set()
    wanted = {"write_doin_projection", "read_doin_projection", "revoke_doin_projection"}
    counts = {}
    for table in COUNTS:
        sql = f"SELECT count(*) AS n FROM {table} LIMIT 1"
        s, r = _get(transport, base + "/api/v1/query?" + urllib.parse.urlencode({"sql": sql}), token)
        counts[table] = json.loads(r)["rows"][0]["n"] if s == 200 else f"HTTP_{s}"
    return {"check": "warehouse", "host_status": status, "doin_capabilities": sorted(caps & wanted),
            "doin_capabilities_ok": wanted <= caps, "counts": counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=("lake", "warehouse"))
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-env", required=True, help="NAME of the environment variable, never a value")
    args = parser.parse_args(argv)
    result = (probe_lake if args.target == "lake" else probe_warehouse)(args.url, args.token_env)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("ok", result.get("doin_capabilities_ok")) else 3


if __name__ == "__main__":
    sys.exit(main())
