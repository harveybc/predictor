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
from collections import Counter
import hashlib
import json
import os
import tempfile
import sys
from pathlib import Path
import urllib.parse
import urllib.request

from doin_core.archive.body import ArchiveRefusal
from doin_node.archive.external import FINALIZED, canonical_references

from tools.i6n_archive import records


CLOSURE_SCHEMA = "i6n.island.archive_closure.v1"


class ClosurePending(RuntimeError):
    """Archive work is incomplete but may close on a later autonomous tick."""


class ClosureRefusal(RuntimeError):
    """Archive evidence contradicts the closure contract."""


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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


def _fetch_json(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _load_chain_snapshot(chain_config: dict, fetch_json=_fetch_json):
    """Load a canonical snapshot from JSON or a live SQLite-backed node API."""
    chain_json = chain_config.get("chain_json")
    if chain_json:
        from doin_node.blockchain.chain import Chain

        chain = Chain()
        chain.load(chain_json)
        tip = chain.blocks[-1]
        return chain.blocks, {
            "chain_height": len(chain.blocks), "tip_hash": tip.hash,
            "genesis_hash": chain.blocks[0].hash,
            "network_id": chain_config.get("network_id"),
            "finalized_height": len(chain.blocks) - 1 - int(chain_config.get("confirmation_depth", 6)),
            "finalized_hash": None,
        }

    base_url = chain_config.get("url")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValueError("chain requires chain_json or an HTTP(S) url")
    base_url = base_url.rstrip("/")
    before = fetch_json(f"{base_url}/chain/status")
    height = before.get("chain_height")
    if type(height) is not int or height < 1:
        raise RuntimeError("CHAIN_STATUS_INVALID: positive chain_height required")

    raw_blocks = []
    start = 0
    while start < height:
        query = urllib.parse.urlencode({"from": start, "to": min(height - 1, start + 49)})
        page = fetch_json(f"{base_url}/chain/blocks?{query}")
        blocks = page.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            raise RuntimeError("CHAIN_PAGE_INVALID: missing blocks")
        raw_blocks.extend(blocks)
        start += len(blocks)
    after = fetch_json(f"{base_url}/chain/status")
    identity = ("chain_height", "tip_hash", "genesis_hash", "network_id",
                "finalized_height", "finalized_hash")
    if any(before.get(key) != after.get(key) for key in identity):
        raise RuntimeError("CHAIN_MOVED_DURING_SNAPSHOT")
    if len(raw_blocks) != height:
        raise RuntimeError("CHAIN_SNAPSHOT_LENGTH_MISMATCH")

    from doin_core.models import Block

    parsed = [Block.model_validate(block) for block in raw_blocks]
    if parsed[-1].hash != before.get("tip_hash"):
        raise RuntimeError("CHAIN_SNAPSHOT_TIP_MISMATCH")
    return parsed, {key: before.get(key) for key in identity}


def _load_chain_blocks(chain_config: dict, fetch_json=_fetch_json):
    """Compatibility wrapper returning only blocks for callers and focused tests."""
    return _load_chain_snapshot(chain_config, fetch_json=fetch_json)[0]


def _metric_key(metric: dict) -> tuple:
    return (
        metric["metric"], float(metric["value"]) if metric["value"] is not None else None,
        metric.get("unit"), metric.get("split"), metric.get("horizon"),
    )


def _read_terminal_digests(paths: list[str], expected: dict) -> tuple[list[str], tuple]:
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ClosureRefusal("TERMINALS_REQUIRE_FOUR_DISTINCT_PATHS")
    digests = []
    identities = set()
    baseline = None
    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClosurePending(f"TERMINAL_UNAVAILABLE[{index}]: {type(exc).__name__}") from exc
        if (document.get("schema") != "i6n.island.terminal.v1"
                or set(document) != {"schema", "payload", "payload_sha256"}
                or not isinstance(document.get("payload"), dict)
                or document.get("payload_sha256") != _digest(document["payload"])):
            raise ClosureRefusal(f"TERMINAL_INVALID[{index}]")
        payload = document["payload"]
        if (payload.get("status") != "COMPLETE" or payload.get("horizon") != expected["horizon"]
                or payload.get("scientific_config_sha256") != expected["scientific_config_sha256"]
                or payload.get("network_id") != expected["network_id"]
                or payload.get("genesis_hash") != expected["genesis_hash"]):
            raise ClosureRefusal(f"TERMINAL_IDENTITY_MISMATCH[{index}]")
        identity = payload.get("island_id")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ClosureRefusal("TERMINAL_ISLAND_DUPLICATE")
        identities.add(identity)
        chain_identity = tuple(payload.get(name) for name in (
            "chain_height", "tip_hash", "finalized_height", "finalized_hash"))
        if baseline is None:
            baseline = chain_identity
        elif chain_identity != baseline:
            raise ClosureRefusal("TERMINAL_CHAIN_MISMATCH")
        digests.append(document["payload_sha256"])
    return sorted(digests), baseline


def _blocker_count(paths: list[str]) -> int:
    count = 0
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            count += sum(1 for child in path.rglob("*") if child.is_file())
        elif path.exists():
            count += 1
    return count


def _verify_candidate_readback(outbox, archive, sink, blocks, *, confirmation_depth: int,
                               horizon: int) -> dict:
    try:
        refs = canonical_references(blocks, confirmation_depth=confirmation_depth)
    except ArchiveRefusal as exc:
        raise ClosureRefusal(f"CHAIN_INVALID: {exc}") from exc
    finalized = [ref for ref in refs if ref.finality == FINALIZED]
    finalized_by_tx = {ref.tx_id: ref for ref in finalized}
    if len(finalized_by_tx) != len(finalized):
        raise ClosureRefusal("DUPLICATE_FINALIZED_REFERENCE_TX")

    # Resolve every finalized reference, including references not owned by this outbox.
    for ref in finalized:
        try:
            archive.resolve(ref.reference)
        except Exception as exc:  # noqa: BLE001 - typed closure evidence records the boundary
            raise ClosurePending(f"FINALIZED_REFERENCE_UNRESOLVED: {ref.tx_id}: {type(exc).__name__}") from exc

    items = outbox.items()
    if not items:
        raise ClosurePending("NO_CANDIDATE_ITEMS")
    report_count = 0
    metric_count = 0
    candidate_count = 0
    for item in items:
        missing = [name for name, evidence in item["states"].items() if not evidence]
        if missing:
            raise ClosurePending(f"CANDIDATE_PENDING: {item['item_id']}: {','.join(missing)}")
        tx_id = item.get("reference_tx_id")
        if tx_id not in finalized_by_tx:
            raise ClosurePending(f"CANDIDATE_REFERENCE_NOT_FINALIZED: {item['item_id']}")
        try:
            resolved = archive.resolve(item["reference"])
        except Exception as exc:  # noqa: BLE001
            raise ClosurePending(f"CANDIDATE_BODY_UNRESOLVED: {item['item_id']}: {type(exc).__name__}") from exc
        if (resolved.body_digest != item["body_digest"]
                or resolved.manifest_digest != item["manifest_digest"]):
            raise ClosureRefusal(f"CANDIDATE_BODY_IDENTITY_MISMATCH: {item['item_id']}")
        if resolved.payload_schema != records.ISLAND_CANDIDATE_SCHEMA:
            raise ClosureRefusal(f"NON_CANDIDATE_ITEM_IN_CAMPAIGN_OUTBOX: {item['item_id']}")
        if resolved.payload.get("horizon_hours") != horizon:
            raise ClosureRefusal(f"CANDIDATE_HORIZON_MISMATCH: {item['item_id']}")
        candidate_count += 1
        units = records.units_for(resolved.payload_schema, resolved.payload)
        reports = [unit["report"] for unit in units]
        digests = sorted({report["report_sha256"] for report in reports})
        readback = sink.read({"report_sha256": digests})
        active = {
            (projection["row_key_sha256"], projection["report_sha256"])
            for projection in readback.get("projections", []) if projection.get("state") == "PROJECTED"
        }
        stored: dict[str, Counter] = {}
        for metric in readback.get("metrics", []):
            stored.setdefault(metric["report_sha256"], Counter())[_metric_key(metric)] += 1
        for report in reports:
            key = (report["tags"]["row_key_sha256"], report["report_sha256"])
            wanted = Counter(_metric_key(metric) for metric in report["metrics"])
            if key not in active or stored.get(report["report_sha256"]) != wanted:
                raise ClosurePending(f"CANDIDATE_READBACK_MISMATCH: {report['report_sha256']}")
            report_count += 1
            metric_count += sum(wanted.values())

    bad_projections = [row for row in outbox.projections()
                       if row["state"] not in {"PROJECTED", "DUPLICATE_NOT_PROJECTED", "REVOKED"}]
    if bad_projections:
        raise ClosurePending(f"PROJECTION_PENDING_OR_ERROR: {len(bad_projections)}")
    return {
        "candidate_items": candidate_count, "candidate_reports": report_count,
        "candidate_metric_rows": metric_count, "finalized_references_resolved": len(finalized),
    }


def produce_archive_closure(config: dict, publisher, archive, sink, blocks, chain_status: dict,
                            etl_result: dict, pump_result: dict) -> dict:
    """Emit the H4 closure only after independently re-verifying every durable boundary."""
    closure = config.get("closure")
    if not isinstance(closure, dict):
        raise ClosureRefusal("CLOSURE_CONFIG_REQUIRED")
    expected = closure.get("expected")
    if not isinstance(expected, dict):
        raise ClosureRefusal("CLOSURE_EXPECTED_REQUIRED")
    required = {"horizon", "scientific_config_sha256", "network_id", "genesis_hash"}
    if set(expected) != required or expected["horizon"] != 4:
        raise ClosureRefusal("CLOSURE_EXPECTED_INVALID")
    if pump_result.get("errors"):
        raise ClosurePending("PUMP_ERRORS_REMAIN")
    if etl_result.get("errors") or etl_result.get("chain_verified") is not True:
        raise ClosurePending("ETL_ERRORS_REMAIN")
    for name in ("network_id", "genesis_hash"):
        if chain_status.get(name) != expected[name]:
            raise ClosureRefusal(f"CHAIN_{name.upper()}_MISMATCH")
    for name in ("chain_height", "finalized_height"):
        if type(chain_status.get(name)) is not int or chain_status[name] < 0:
            raise ClosurePending(f"CHAIN_{name.upper()}_UNAVAILABLE")
    for name in ("tip_hash", "finalized_hash"):
        if not isinstance(chain_status.get(name), str) or not chain_status[name]:
            raise ClosurePending(f"CHAIN_{name.upper()}_UNAVAILABLE")

    terminal_paths = closure.get("terminal_receipts")
    if not isinstance(terminal_paths, list):
        raise ClosureRefusal("CLOSURE_TERMINALS_REQUIRED")
    terminal_digests, terminal_chain = _read_terminal_digests(terminal_paths, expected)
    observed_chain = tuple(chain_status.get(name) for name in (
        "chain_height", "tip_hash", "finalized_height", "finalized_hash"))
    if observed_chain != terminal_chain:
        raise ClosureRefusal("CLOSURE_CHAIN_DOES_NOT_MATCH_TERMINALS")
    status = publisher.status()
    if (status.get("items", 0) < 1 or status.get("green") != status.get("items")
            or status.get("missing") != [] or status.get("submitted_unfinalized") != 0):
        raise ClosurePending("ARCHIVE_ITEMS_NOT_GREEN")

    guards = closure.get("blocker_paths", {})
    if not isinstance(guards, dict) or any(
            name not in {"pending", "error", "quarantine"} or not isinstance(paths, list)
            or any(not isinstance(path, str) for path in paths)
            for name, paths in guards.items()):
        raise ClosureRefusal("CLOSURE_BLOCKER_PATHS_INVALID")
    blocker_counts = {name: _blocker_count(guards.get(name, []))
                      for name in ("pending", "error", "quarantine")}
    if any(blocker_counts.values()):
        raise ClosurePending("BLOCKER_PATHS_NOT_EMPTY: " + json.dumps(blocker_counts, sort_keys=True))

    verified = _verify_candidate_readback(
        publisher.outbox, archive, sink, blocks,
        confirmation_depth=int(config.get("confirmation_depth", 6)), horizon=expected["horizon"],
    )
    payload = {
        "status": "GREEN", "horizon": expected["horizon"],
        "scientific_config_sha256": expected["scientific_config_sha256"],
        "terminal_receipt_sha256s": terminal_digests,
        "network_id": chain_status["network_id"], "genesis_hash": chain_status["genesis_hash"],
        "chain_height": chain_status["chain_height"], "tip_hash": chain_status["tip_hash"],
        "finalized_height": chain_status["finalized_height"],
        "finalized_hash": chain_status["finalized_hash"],
        "archive": {
            "items": status["items"], "green": status["green"], "missing": [],
            "submitted_unfinalized": 0, **verified,
        },
        "olap": {
            "status": "GREEN", "chain_verified": True, "readback_verified": True,
            "projected_reports": verified["candidate_reports"],
            "metric_rows": verified["candidate_metric_rows"],
        },
        "blockers": blocker_counts,
    }
    envelope = {"schema": CLOSURE_SCHEMA, "payload": payload, "payload_sha256": _digest(payload)}
    receipt = Path(closure["receipt"])
    if not receipt.is_absolute():
        raise ClosureRefusal("CLOSURE_RECEIPT_MUST_BE_ABSOLUTE")
    if receipt.is_file():
        try:
            existing = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClosureRefusal(f"CLOSURE_RECEIPT_INVALID: {type(exc).__name__}") from exc
        if existing != envelope:
            raise ClosureRefusal("CLOSURE_RECEIPT_CONFLICT")
    else:
        _atomic_json(receipt, envelope)
    return envelope


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status", "pump", "etl", "close"))
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
    from tools.i6n_archive.etl import ArchiveETL
    from tools.i6n_archive.outbox import Outbox
    from tools.i6n_archive.sink import HttpSink

    if args.command == "close" and publisher is None:
        print(json.dumps({"state": "WAIT", "reason": "NO_OUTBOX"}, sort_keys=True))
        return 4
    pump_result = {"stored": 0, "submitted": 0, "errors": []}
    if args.command == "close" and publisher is not None:
        pump_result = publisher.pump(archive, _spool_submitter(Path(config["submit_spool"])))
    blocks, chain_status = _load_chain_snapshot(config["chain"])
    outbox = publisher.outbox if publisher else Outbox(Path(config["root"]) / "inbox.sqlite")
    warehouse = config["warehouse"]
    sink = HttpSink(warehouse["url"], token_provider=_token(warehouse["token_env"]))
    etl = ArchiveETL(outbox, archive, sink,
                     confirmation_depth=int(config.get("confirmation_depth", 6)))
    result = etl.run(blocks)
    if args.command == "close":
        try:
            envelope = produce_archive_closure(
                config, publisher, archive, sink, blocks, chain_status, result, pump_result)
        except ClosurePending as exc:
            print(json.dumps({"state": "WAIT", "reason": str(exc), "pump": pump_result,
                              "etl": result}, sort_keys=True))
            return 4
        except ClosureRefusal as exc:
            print(json.dumps({"state": "REFUSED", "reason": str(exc)}, sort_keys=True))
            return 2
        print(json.dumps({"state": "GREEN", "receipt": envelope}, sort_keys=True))
        return 0
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["errors"] else 3


if __name__ == "__main__":
    sys.exit(main())
