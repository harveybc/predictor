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
from tools.i6n_outbox_snapshot import (
    SNAPSHOT_SCHEMA,
    STABLE_FIELDS as OUTBOX_STABLE_FIELDS,
    _inspect as inspect_outbox_snapshot,
)


CLOSURE_SCHEMA = "i6n.island.archive_closure.v2"
CLOSURE_OFFCHAIN_SCHEMA = "i6n.island.archive_closure.v3"


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


def _read_terminal_digests(paths: list[str], expected: dict) -> tuple[dict[str, str], tuple, dict[str, dict]]:
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ClosureRefusal("TERMINALS_REQUIRE_FOUR_DISTINCT_PATHS")
    digests = {}
    identities = set()
    baseline = None
    terminals = {}
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
        evaluated = payload.get("evaluated_candidate_count")
        if type(evaluated) is not int or evaluated < 1:
            raise ClosureRefusal(f"TERMINAL_EVALUATED_COUNT_INVALID[{index}]")
        terminals[identity] = payload
        chain_identity = tuple(payload.get(name) for name in (
            "chain_height", "tip_hash", "finalized_height", "finalized_hash"))
        if baseline is None:
            baseline = chain_identity
        elif chain_identity != baseline:
            raise ClosureRefusal("TERMINAL_CHAIN_MISMATCH")
        digests[identity] = document["payload_sha256"]
    return digests, baseline, terminals


def _verify_terminal_ancestor(blocks, chain_status: dict, terminal_chain: tuple) -> str:
    """Prove that the closure snapshot is the terminal snapshot or its descendant."""
    terminal_height, terminal_tip, terminal_finalized_height, terminal_finalized_hash = terminal_chain
    current = tuple(chain_status.get(name) for name in (
        "chain_height", "tip_hash", "finalized_height", "finalized_hash"))
    if current == terminal_chain:
        return "EXACT"
    if terminal_height < 1 or chain_status["chain_height"] < terminal_height:
        raise ClosureRefusal("TERMINAL_CHAIN_NOT_ANCESTOR")
    if len(blocks) != chain_status["chain_height"]:
        raise ClosureRefusal("CLOSURE_CHAIN_LENGTH_MISMATCH")
    if blocks[terminal_height - 1].hash != terminal_tip:
        raise ClosureRefusal("TERMINAL_TIP_NOT_ANCESTOR")
    if terminal_finalized_height < -1 or terminal_finalized_height >= len(blocks):
        raise ClosureRefusal("TERMINAL_FINALIZED_CHECKPOINT_INVALID")
    if terminal_finalized_height == -1 and terminal_finalized_hash != '':
        raise ClosureRefusal('TERMINAL_FINALIZED_CHECKPOINT_INVALID')
    if terminal_finalized_height >= 0 and blocks[terminal_finalized_height].hash != terminal_finalized_hash:
        raise ClosureRefusal("TERMINAL_FINALIZED_HASH_NOT_ANCESTOR")
    if chain_status["finalized_height"] < terminal_finalized_height:
        raise ClosureRefusal("CHAIN_FINALITY_REGRESSED")
    return "DESCENDANT_VERIFIED"


def _as_result_map(value, island_ids: set[str], field: str) -> dict[str, dict]:
    if isinstance(value, dict) and set(value) == island_ids and all(
            isinstance(item, dict) for item in value.values()):
        return value
    if len(island_ids) == 1 and isinstance(value, dict):
        return {next(iter(island_ids)): value}
    raise ClosureRefusal(f"{field.upper()}_ISLAND_SET_MISMATCH")


def _verify_outbox_seals(paths: list[str], terminals: dict[str, dict],
                         terminal_digests: dict[str, str], publishers: dict) -> dict[str, dict]:
    """Bind four independent immutable snapshots to four optimizer terminals."""
    if len(paths) != 4 or len(set(paths)) != 4:
        raise ClosureRefusal("OUTBOX_SEALS_REQUIRE_FOUR_DISTINCT_PATHS")
    if len({id(publisher) for publisher in publishers.values()}) != 4:
        raise ClosureRefusal("OUTBOX_PUBLISHERS_NOT_INDEPENDENT")
    outbox_paths = [str(Path(publisher.outbox.path).resolve()) for publisher in publishers.values()]
    if len(set(outbox_paths)) != 4:
        raise ClosureRefusal("OUTBOX_PATHS_NOT_INDEPENDENT")

    seals = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClosureRefusal("OUTBOX_SEAL_INVALID_JSON") from exc
        if (not isinstance(document, dict) or document.get("schema") != SNAPSHOT_SCHEMA
                or not isinstance(document.get("payload"), dict)
                or _digest(document["payload"]) != document.get("payload_sha256")):
            raise ClosureRefusal("OUTBOX_SEAL_ENVELOPE_INVALID")
        payload = document["payload"]
        island_id = payload.get("island_id")
        if island_id not in terminals or island_id in seals:
            raise ClosureRefusal("OUTBOX_SEAL_ISLAND_SET_MISMATCH")
        if payload.get("terminal_receipt_sha256") != terminal_digests[island_id]:
            raise ClosureRefusal(f"OUTBOX_SEAL_TERMINAL_MISMATCH: {island_id}")
        observed = inspect_outbox_snapshot(publishers[island_id].outbox.path, terminals[island_id])
        if (any(payload.get(field) != observed[field] for field in OUTBOX_STABLE_FIELDS)
                or payload.get("terminal_receipt_sha256") != terminal_digests[island_id]):
            raise ClosureRefusal(f"OUTBOX_SEAL_CONTENT_MISMATCH: {island_id}")
        seals[island_id] = dict(payload, seal_sha256=document["payload_sha256"])

    if set(seals) != set(terminals):
        raise ClosureRefusal("OUTBOX_SEAL_ISLAND_SET_MISMATCH")
    for field, refusal in (
        ("producer_peer_id", "OUTBOX_PRODUCER_PEERS_NOT_DISTINCT"),
        ("items_sha256", "OUTBOX_ITEM_SETS_NOT_DISTINCT"),
    ):
        if len({seal[field] for seal in seals.values()}) != 4:
            raise ClosureRefusal(refusal)
    return seals


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
                               horizon: int, require_finalized: bool = True) -> dict:
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
        required_states = ('LOCAL_COMPLETE', 'OBJECT_DURABLE', 'OLAP_READBACK_VERIFIED')
        missing = [name for name, evidence in item["states"].items()
                   if not evidence and (require_finalized or name in required_states)]
        if missing:
            raise ClosurePending(f"CANDIDATE_PENDING: {item['item_id']}: {','.join(missing)}")
        tx_id = item.get("reference_tx_id")
        if require_finalized and tx_id not in finalized_by_tx:
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
    policy = closure.get('candidate_finality_policy', 'FINALIZED')
    if policy not in ('FINALIZED', 'OFFCHAIN_VERIFIED'):
        raise ClosureRefusal('CANDIDATE_FINALITY_POLICY_INVALID')
    require_finalized = policy == 'FINALIZED'
    expected = closure.get("expected")
    if not isinstance(expected, dict):
        raise ClosureRefusal("CLOSURE_EXPECTED_REQUIRED")
    required = {"horizon", "scientific_config_sha256", "network_id", "genesis_hash"}
    if set(expected) != required or expected["horizon"] != 4:
        raise ClosureRefusal("CLOSURE_EXPECTED_INVALID")
    for name in ("network_id", "genesis_hash"):
        if chain_status.get(name) != expected[name]:
            raise ClosureRefusal(f"CHAIN_{name.upper()}_MISMATCH")
    for name in ("chain_height", "finalized_height"):
        if type(chain_status.get(name)) is not int or chain_status[name] < (-1 if name == 'finalized_height' else 0):
            raise ClosurePending(f"CHAIN_{name.upper()}_UNAVAILABLE")
    for name in ("tip_hash", "finalized_hash"):
        absent_finality = name == 'finalized_hash' and chain_status['finalized_height'] == -1
        if not isinstance(chain_status.get(name), str) or (not absent_finality and not chain_status[name]) or (absent_finality and chain_status[name] != ''):
            raise ClosurePending(f"CHAIN_{name.upper()}_UNAVAILABLE")

    terminal_paths = closure.get("terminal_receipts")
    if not isinstance(terminal_paths, list):
        raise ClosureRefusal("CLOSURE_TERMINALS_REQUIRED")
    terminal_digests, terminal_chain, terminals = _read_terminal_digests(terminal_paths, expected)
    chain_relation = _verify_terminal_ancestor(blocks, chain_status, terminal_chain)
    if not require_finalized:
        # The corpus can close without later blocks, but champion ordering still
        # has to satisfy the repaired consensus contract over this exact chain.
        from doin_core.consensus.champion import advance_champions
        best, seen = {}, set()
        for block in blocks:
            directions = {tx.domain_id: False for tx in block.transactions if tx.domain_id}
            best, seen = advance_champions(block, best, seen, directions)

    fleet_mode = isinstance(publisher, dict)
    publishers = publisher if fleet_mode else {next(iter(terminals)): publisher}
    if fleet_mode and set(publishers) != set(terminals):
        raise ClosureRefusal("OUTBOX_ISLAND_SET_MISMATCH")
    seals = {}
    if fleet_mode:
        seal_paths = closure.get("outbox_seals")
        if not isinstance(seal_paths, list):
            raise ClosureRefusal("OUTBOX_SEALS_REQUIRED")
        seals = _verify_outbox_seals(
            seal_paths, terminals, terminal_digests, publishers,
        )
    island_ids = set(publishers)
    pump_results = _as_result_map(pump_result, island_ids, "pump")
    etl_results = _as_result_map(etl_result, island_ids, "etl")
    if any(result.get("errors") for result in pump_results.values()):
        raise ClosurePending("PUMP_ERRORS_REMAIN")
    if any(result.get("errors") or result.get("chain_verified") is not True
           for result in etl_results.values()):
        raise ClosurePending("ETL_ERRORS_REMAIN")

    statuses = {}
    for island_id, candidate_publisher in publishers.items():
        status = candidate_publisher.status()
        if require_finalized:
            ready = (status.get('items', 0) >= 1 and status.get('green') == status.get('items')
                     and status.get('missing') == [] and status.get('submitted_unfinalized') == 0)
        else:
            items = candidate_publisher.outbox.items()
            ready = bool(items) and all(all(item['states'][key] for key in
                         ('LOCAL_COMPLETE', 'OBJECT_DURABLE', 'OLAP_READBACK_VERIFIED')) for item in items)
            status = dict(status, corpus_verified=sum(all(item['states'][key] for key in
                          ('LOCAL_COMPLETE', 'OBJECT_DURABLE', 'OLAP_READBACK_VERIFIED')) for item in items))
        if not ready:
            raise ClosurePending(f"ARCHIVE_ITEMS_NOT_GREEN: {island_id}")
        if fleet_mode and status["items"] != terminals[island_id]["evaluated_candidate_count"]:
            raise ClosureRefusal(f"OUTBOX_CANDIDATE_COUNT_MISMATCH: {island_id}")
        statuses[island_id] = status

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

    verified_by_island = {}
    for island_id, candidate_publisher in publishers.items():
        verified_by_island[island_id] = _verify_candidate_readback(
            candidate_publisher.outbox, archive, sink, blocks,
            confirmation_depth=int(config.get("confirmation_depth", 6)), horizon=expected["horizon"],
            require_finalized=require_finalized,
        )
    aggregate_status = {
        "items": sum(status["items"] for status in statuses.values()),
        "green": sum(status["green"] for status in statuses.values()),
    }
    aggregate_verified = {
        key: sum(item[key] for item in verified_by_island.values())
        for key in ("candidate_items", "candidate_reports", "candidate_metric_rows")
    }
    aggregate_verified["finalized_references_resolved"] = max(
        item["finalized_references_resolved"] for item in verified_by_island.values())
    terminal_chain_payload = dict(zip(
        ("chain_height", "tip_hash", "finalized_height", "finalized_hash"), terminal_chain))
    payload = {
        "status": "GREEN", "horizon": expected["horizon"],
        "scientific_config_sha256": expected["scientific_config_sha256"],
        "terminal_receipt_sha256s": sorted(terminal_digests.values()),
        "network_id": chain_status["network_id"], "genesis_hash": chain_status["genesis_hash"],
        "chain_height": chain_status["chain_height"], "tip_hash": chain_status["tip_hash"],
        "finalized_height": chain_status["finalized_height"],
        "finalized_hash": chain_status["finalized_hash"],
        "terminal_chain": terminal_chain_payload,
        "chain_relation": chain_relation,
        "archive": {
            "items": aggregate_status["items"], "green": aggregate_status["green"],
            "missing": [], "submitted_unfinalized": 0, **aggregate_verified,
            "islands": {
                island_id: {
                    "items": statuses[island_id]["items"],
                    "green": statuses[island_id]["green"],
                    **({"snapshot_seal": seals[island_id]} if fleet_mode else {}),
                    **verified_by_island[island_id],
                }
                for island_id in sorted(statuses)
            },
        },
        "olap": {
            "status": "GREEN", "chain_verified": True, "readback_verified": True,
            "projected_reports": aggregate_verified["candidate_reports"],
            "metric_rows": aggregate_verified["candidate_metric_rows"],
        },
        "blockers": blocker_counts,
    }
    schema = CLOSURE_SCHEMA
    if not require_finalized:
        schema = CLOSURE_OFFCHAIN_SCHEMA
        payload['candidate_finality_policy'] = policy
        payload['archive']['corpus_verified'] = sum(s['corpus_verified'] for s in statuses.values())
        payload['archive']['green'] = sum(s['green'] for s in statuses.values())
        payload['archive']['submitted_unfinalized'] = sum(s['submitted_unfinalized'] for s in statuses.values())
    envelope = {"schema": schema, "payload": payload, "payload_sha256": _digest(payload)}
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

    island_config = config.get("islands")
    if island_config is not None:
        if (not isinstance(island_config, list) or len(island_config) != 4
                or any(not isinstance(item, dict)
                       or set(item) != {"island_id", "root", "submit_spool"}
                       for item in island_config)):
            raise ValueError("islands must contain four island_id/root/submit_spool objects")
        if len({item["island_id"] for item in island_config}) != 4:
            raise ValueError("islands must contain four distinct island_id values")
        publishers = {}
        missing = []
        for item in island_config:
            outbox_path = Path(item["root"]) / "outbox.sqlite"
            if outbox_path.exists():
                publishers[item["island_id"]] = ArchivePublisher.from_outbox(outbox_path)
            else:
                missing.append(item["island_id"])
    else:
        root = Path(config["root"])
        publisher = ArchivePublisher.from_outbox(root / "outbox.sqlite") \
            if (root / "outbox.sqlite").exists() else None
        publishers = {"legacy": publisher} if publisher is not None else {}
        missing = [] if publisher is not None else ["legacy"]
    if args.command == "status":
        if island_config is None:
            result = publishers["legacy"].status() if publishers else {"state": "NO_OUTBOX"}
        else:
            result = {
                "state": "READY" if not missing else "WAIT_OUTBOXES",
                "missing": sorted(missing),
                "islands": {name: publisher.status() for name, publisher in sorted(publishers.items())},
            }
        print(json.dumps(result, sort_keys=True))
        return 0
    from doin_node.archive.external import ExternalPayloadArchive, HttpByteStore

    lake = config["lake"]
    store = HttpByteStore(lake["url"], token_provider=_token(lake["token_env"]))
    archive = ExternalPayloadArchive({lake["store_id"]: store}, write_store_id=lake["store_id"])
    if args.command == "pump":
        if not publishers:
            print(json.dumps({"state": "NO_OUTBOX"}))
            return 0
        if island_config is None:
            result = publishers["legacy"].pump(
                archive, _spool_submitter(Path(config["submit_spool"])))
        else:
            by_id = {item["island_id"]: item for item in island_config}
            result = {
                name: publisher.pump(
                    archive, _spool_submitter(Path(by_id[name]["submit_spool"])))
                for name, publisher in sorted(publishers.items())
            }
        print(json.dumps(result, sort_keys=True))
        errors = result.get("errors") if island_config is None else [
            error for item in result.values() for error in item["errors"]]
        return 0 if not errors else 3
    from tools.i6n_archive.etl import ArchiveETL
    from tools.i6n_archive.outbox import Outbox
    from tools.i6n_archive.sink import HttpSink

    if args.command == "close" and missing:
        print(json.dumps({"state": "WAIT", "reason": "MISSING_OUTBOXES",
                          "islands": sorted(missing)}, sort_keys=True))
        return 4
    pump_result = {"stored": 0, "submitted": 0, "errors": []}
    if args.command == "close":
        if island_config is None:
            pump_result = publishers["legacy"].pump(
                archive, _spool_submitter(Path(config["submit_spool"])))
        else:
            by_id = {item["island_id"]: item for item in island_config}
            pump_result = {
                name: publisher.pump(
                    archive, _spool_submitter(Path(by_id[name]["submit_spool"])))
                for name, publisher in sorted(publishers.items())
            }
    blocks, chain_status = _load_chain_snapshot(config["chain"])
    warehouse = config["warehouse"]
    sink = HttpSink(warehouse["url"], token_provider=_token(warehouse["token_env"]))
    if island_config is None:
        outbox = publishers["legacy"].outbox if publishers else Outbox(Path(config["root"]) / "inbox.sqlite")
        etl = ArchiveETL(
            outbox, archive, sink,
            confirmation_depth=int(config.get("confirmation_depth", 6)))
        offchain = etl.run_offchain()
        result = etl.run(blocks)
        result['offchain'] = offchain
        result['errors'].extend(offchain['errors'])
    else:
        result = {}
        for name, publisher in sorted(publishers.items()):
            etl = ArchiveETL(publisher.outbox, archive, sink,
                             confirmation_depth=int(config.get('confirmation_depth', 6)))
            offchain = etl.run_offchain()
            result[name] = etl.run(blocks)
            result[name]['offchain'] = offchain
            result[name]['errors'].extend(offchain['errors'])
    if args.command == "close":
        try:
            envelope = produce_archive_closure(
                config,
                publishers if island_config is not None else publishers["legacy"],
                archive, sink, blocks, chain_status, result, pump_result)
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
    errors = result.get("errors") if island_config is None else [
        error for item in result.values() for error in item["errors"]]
    return 0 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())
