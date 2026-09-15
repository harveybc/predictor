"""Bounded, non-scientific check of a governed synthetic delivery and terminal."""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def measure(raw, expected_sha):
    if sha(raw) != expected_sha:
        raise ValueError("unexpected synthetic input identity")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    values = [float(row["value"]) for row in rows]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("finite, nonempty values required")
    return {"rows": len(rows), "mean": sum(values) / len(values),
            "minimum": min(values), "maximum": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    token = args.key_file.read_text().strip()

    def request(path, body=None, extra=None):
        headers = {"Authorization": "Bearer " + token, **(extra or {})}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(args.base_url.rstrip("/") + path,
                                     data=None if body is None else encoded(body), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as exc:
            # Body can contain submitted metadata; preserve it only in the private run.
            (args.output / "http-error.json").write_bytes(exc.read())
            raise RuntimeError(f"HTTP {exc.code} at {path}") from None

    def send(path, body=None, extra=None):
        status, headers, raw = request(path, body, extra)
        return status, json.loads(raw)

    def save(name, value):
        (args.output / name).write_bytes(encoded(value))

    def utc():
        return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    started = utc()
    wall, cpu = time.monotonic(), time.process_time()
    config = {"lake": "governance_smoke", "resource": "panel.csv", "role": "x",
              "expected_sha256": args.expected_sha256,
              "expected_contract_sha256": args.expected_contract_sha256,
              "measurement": "rows_mean_minimum_maximum.v1"}
    code = Path(__file__).read_bytes()
    code_manifest = {"files": [{"path": Path(__file__).name, "sha256": sha(code)}]}
    save("code-manifest.json", code_manifest)
    (args.output / Path(__file__).name).write_bytes(code)
    campaign = {
        "schema": "governed_campaign.v1", "campaign_key": "store-adoption-" + str(time.time_ns()),
        "classification": "NON_GOVERNING", "project": "predictor",
        "code_identity": {"kind": "file_manifest", "value": sha(encoded(code_manifest))},
        "config_sha256": sha(encoded(config)), "input_mode": "DATASETS",
        "synthetic_spec_sha256": None, "units": ["transport-check"],
        "datasets": [{"lake": config["lake"], "resource": config["resource"],
                      "role": "x", "from": None, "to": None}], "terminal_lake": "olap_cube"}
    save("config.json", config)
    save("campaign.json", campaign)
    status, receipt = send("/api/v2/campaigns", campaign)
    assert status == 201
    save("campaign-receipt.json", receipt)
    digest = receipt["campaign_sha256"]
    headers = {"X-Campaign-SHA256": digest, "X-Unit-ID": "transport-check"}
    query = urllib.parse.urlencode({key: config[key] for key in ("lake", "resource", "role")})
    status, delivery_headers, raw = request("/api/v2/download?" + query, extra=headers)
    assert status == 200 and sha(raw) == delivery_headers["X-Content-SHA256"]
    assert delivery_headers["X-Availability-Contract-SHA256"] == args.expected_contract_sha256
    measurements = measure(raw, args.expected_sha256)
    (args.output / "input.csv").write_bytes(raw)
    save("measurements.json", measurements)
    delivery = delivery_headers["X-Delivery-ID"]
    status, confirmation = send(f"/api/v2/deliveries/{delivery}/confirm",
                               {"schema": "delivery_confirmation.v1", "sha256": sha(raw),
                                "bytes": len(raw), "cached": False}, headers)
    assert status == 200 and confirmation["state"] == "VERIFIED_TRANSFER"
    save("delivery-confirmation.json", confirmation)
    terminal = {
        "schema": "governed_terminal.v1", "generation": 1, "status": "COMPLETED", "reason": None,
        "started_at": started, "finished_at": utc(),
        "costs": {"wall_seconds": time.monotonic() - wall, "cpu_seconds": time.process_time() - cpu},
        "deliveries": [delivery],
        "artifacts": [{"role": "measurements", "sha256": sha(encoded(measurements)),
                       "bytes": len(encoded(measurements))}],
        "metrics": [{"metric": name, "split": None, "horizon": None, "unit": None,
                     "value": value, "std_dev": None, "min_value": None, "max_value": None}
                    for name, value in measurements.items()],
        "tags": {"purpose": "NON_GOVERNING_TRANSPORT_MECHANICS", "input": "synthetic_two_rows"}}
    save("terminal.json", terminal)
    endpoint = f"/api/v2/campaigns/{digest}/units/transport-check/terminal"
    status, terminal_receipt = send(endpoint, terminal, headers)
    assert status == 201
    save("terminal-receipt.json", terminal_receipt)
    status, repeat = send(endpoint, terminal, headers)
    assert status == 200 and repeat["terminal_sha256"] == terminal_receipt["terminal_sha256"]
    save("idempotent-repeat.json", repeat)
    status, reconciliation = send(f"/api/v2/campaigns/{digest}/reconcile", extra=headers)
    assert status == 200
    assert all(reconciliation[k] == [] for k in ("missing_units", "accounting_only", "lake_only"))
    save("reconciliation.json", reconciliation)
    print(json.dumps({"campaign_sha256": digest, "terminal_sha256": terminal_receipt["terminal_sha256"],
                      "measurements": measurements, "reconciled": True, "classification": "NON_GOVERNING"}))


if __name__ == "__main__":
    main()
