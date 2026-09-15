#!/usr/bin/env python3
"""One governed delivery, from whichever machine runs this, through data-gov.

Musashi's instruction of 2026-09-15: check the identity contract that exists, configure the
workers with it, and prove a governed delivery **from each machine**.

Nothing here reimplements the protocol: it drives `data-gov`'s own `DataGovClient`, the same
class every consumer uses, so what is proven is the deployed path — campaign, authenticated
download with its content hash, confirmation, and a terminal with the cost of the work.

The point of running it from gamma or dragon is that those machines hold no copy of any
dataset. They reach the real lake over the operator's tunnel and the bytes arrive governed,
which is what makes `scp` unnecessary.

usage:
  worker_delivery_probe.py --gov-url URL --api-key-file FILE --lake LAKE --resource NAME
      --out RECEIPT.json [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def load_client():
    checkout = Path(os.environ.get("DATA_GOV_CHECKOUT")
                    or Path.home() / "Documents/GitHub/data-gov").expanduser()
    module = checkout / "app" / "client.py"
    if not module.is_file():
        raise SystemExit(f"data-gov checkout not found at {checkout} (set DATA_GOV_CHECKOUT). "
                         "The checkout is used as the CLIENT LIBRARY; no store runs here.")
    # imported as part of its own package: `app.client` uses relative imports, so loading
    # the file on its own would break the very client this probe exists to exercise
    sys.path.insert(0, str(checkout))
    from app.client import DataGovClient

    return DataGovClient


def code_identity_of(repo_root: Path) -> dict:
    """The commit this probe ran from, and whether its tree was dirty."""
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                          capture_output=True, text=True)
    commit = head.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"{repo_root} is not a git checkout: data-gov requires a commit to "
                         "identify the code a campaign ran from (pass --repo-root)")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                           cwd=repo_root, capture_output=True, text=True).stdout.strip()
    return {"kind": "git_commit", "value": commit + ("-dirty" if dirty else "")}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gov-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--lake", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path.home() / ".cache" / "data-gov")
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="the checkout whose commit identifies this code; data-gov "
                             "refuses a campaign without one, which is why a probe copied "
                             "outside a git tree cannot run")
    args = parser.parse_args(argv)

    host = socket.gethostname()
    client_class = load_client()
    api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    experiment = f"worker-delivery-{host}-{int(time.time())}"
    client = client_class(base_url=args.gov_url, api_key=api_key,
                          experiment_key=experiment)

    # the campaign schema is exact-keyed: data-gov refuses an invented field, which is how a
    # probe like this one is kept from quietly becoming its own protocol
    code_identity = code_identity_of(args.repo_root or Path(__file__).resolve().parents[1])
    config_sha256 = hashlib.sha256(json.dumps(
        {"probe": "worker_delivery", "host": host, "lake": args.lake,
         "resource": args.resource}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    campaign = {
        "schema": "governed_campaign.v1",
        "campaign_key": f"worker-delivery-{host}-{int(time.time())}",
        "classification": "NON_GOVERNING",
        "project": "predictor",
        "code_identity": code_identity,
        "config_sha256": config_sha256,
        "input_mode": "DATASETS",
        "synthetic_spec_sha256": None,
        "units": [f"{host}-probe-1"],
        "datasets": [{"lake": args.lake, "resource": args.resource, "role": "probe",
                      "from": None, "to": None}],
        "terminal_lake": "olap_cube",
    }
    status, submitted = client.submit_campaign(campaign)
    if status not in (200, 201):
        raise SystemExit(f"data-gov refused the campaign: {status} {submitted}")
    campaign_sha = submitted.get("campaign_sha256") or submitted.get("sha256")
    unit_id = campaign["units"][0]

    started = time.monotonic()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # the GOVERNED path: it carries the campaign and unit headers and confirms the delivery
    # only after the bytes verify locally
    status, delivery = client.governed_download(
        campaign_sha, unit_id, args.lake, args.resource, "probe", args.cache_dir)
    wall = time.monotonic() - started
    if status != 200:
        raise SystemExit(f"the governed download failed: {status} {delivery}")

    path = Path(delivery["path"]) if isinstance(delivery, dict) and delivery.get("path") else None
    local_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None

    receipt = {
        "schema": "worker_delivery_probe.v1",
        "at": now(), "host": host, "gov_url": args.gov_url,
        "experiment_key": experiment, "campaign_sha256": campaign_sha, "unit_id": unit_id,
        "lake": args.lake, "resource": args.resource,
        "delivery": {key: value for key, value in (delivery or {}).items()
                     if key in ("sha256", "bytes", "cached", "delivery", "state",
                                "source_sha256", "path")},
        "bytes_on_disk_sha256": local_sha,
        "wall_seconds": round(wall, 3),
        "note": "the bytes were obtained through data-gov, not copied between machines",
    }
    if local_sha and delivery.get("sha256") and local_sha != delivery["sha256"]:
        receipt["refusal"] = "the bytes on disk do not match the delivered digest"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    printable = json.loads(json.dumps(receipt).replace(str(Path.home()), "~"))
    print(json.dumps(printable, indent=1))
    return 0 if not receipt.get("refusal") else 1


if __name__ == "__main__":
    raise SystemExit(main())
