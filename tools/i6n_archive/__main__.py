"""CLI: ``python -m tools.i6n_archive {status|pump|etl} --config <private.json>``.

The private config (never committed) names the archive root, the lake object routes, the
warehouse host, the node's chain file and the reference spool. Tokens are read from the
environment variables it names (set by the EnvironmentFile of an existing configured
service); the CLI never reads or prints a token file.

* ``status``  read-only: the outbox states; no network, no TF, no training.
* ``pump``    PUT/GET/verify bodies, then write reference transactions to the spool that the
              DOIN node ingests (one ``<tx_id>.json`` per reference, exclusive create).
* ``etl``     verify the node's chain file, resolve finalized bodies, project, read back, ACK.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _token(name: str):
    def provider() -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"TOKEN_ENV_MISSING: {name} (run under the configured service environment)")
        return value

    return provider


def _spool_submitter(spool: Path):
    spool.mkdir(parents=True, exist_ok=True)

    def submit(tx) -> None:
        data = tx.model_dump_json().encode("utf-8")
        target = spool / f"{tx.id}.json"
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if target.read_bytes() != data:
                raise RuntimeError("SPOOL_CONFLICT: same tx id, different bytes")
            return
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    return submit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "pump", "etl"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    from tools.i6n_archive.publisher import ArchivePublisher

    publisher = ArchivePublisher.from_outbox(Path(config["root"]) / "outbox.sqlite") \
        if (Path(config["root"]) / "outbox.sqlite").exists() else None
    if args.command == "status":
        print(json.dumps(publisher.status() if publisher else {"state": "NO_OUTBOX"}, sort_keys=True))
        return 0
    from doin_node.archive.external import ExternalPayloadArchive, HttpByteStore

    lake = config["lake"]
    store = HttpByteStore(lake["url"], token_provider=_token(lake["token_env"]))
    archive = ExternalPayloadArchive({lake["store_id"]: store}, write_store_id=lake["store_id"])
    if args.command == "pump":
        if publisher is None:
            print(json.dumps({"state": "NO_OUTBOX"}))
            return 0
        result = publisher.pump(archive, _spool_submitter(Path(config["submit_spool"])))
        print(json.dumps(result, sort_keys=True))
        return 0 if not result["errors"] else 3
    from doin_node.blockchain.chain import Chain

    from tools.i6n_archive.etl import ArchiveETL
    from tools.i6n_archive.outbox import Outbox
    from tools.i6n_archive.sink import HttpSink

    chain = Chain()
    chain.load(config["chain"]["chain_json"])
    outbox = publisher.outbox if publisher else Outbox(Path(config["root"]) / "inbox.sqlite")
    warehouse = config["warehouse"]
    etl = ArchiveETL(outbox, archive, HttpSink(warehouse["url"], token_provider=_token(warehouse["token_env"])),
                     confirmation_depth=int(config.get("confirmation_depth", 6)))
    result = etl.run(chain.blocks)
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["errors"] else 3


if __name__ == "__main__":
    sys.exit(main())
