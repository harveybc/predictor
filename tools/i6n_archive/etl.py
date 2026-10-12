"""Finalized reference -> verified bytes -> warehouse projection -> exact readback -> ACK.

Order, per run:

1. verify the chain handed in as canonical (genesis, heights, links, merkle roots, header
   hashes, every reference transaction's id/payload binding). A chain that fails is not
   read at all;
2. revoke: every block this inbox projected that is no longer on the chain gets a REVOKED
   event in the warehouse (history kept) and the affected items lose CHAIN_FINALIZED and
   OLAP_READBACK_VERIFIED; an orphaned reference is released for resubmission;
3. for each FINALIZED reference (depth >= confirmation_depth), in chain order: resolve the
   body (external: GET by locator and digest; full: rebuild from the transaction), verify
   bytes, manifest and payload schema, derive the projection from those bytes only, write it,
   read it back and compare exactly; only then set OLAP_READBACK_VERIFIED.

Fork choice is the node's job: the block list passed in is the node's canonical chain. A hash
proves integrity, not availability: an unresolvable body is an error, never a green state.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Sequence

from doin_core.archive.body import ArchiveRefusal
from doin_node.archive.external import FINALIZED, canonical_references

from tools.i6n_archive import records

SCHEMA_WRITE = "i6n.doin_projection.write.v1"
SCHEMA_REVOKE = "i6n.doin_projection.revoke.v1"
OFFCHAIN_VERIFIED = "OFFCHAIN_VERIFIED"
NO_BLOCK = "0" * 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric_key(m: dict) -> tuple:
    return (m["metric"], float(m["value"]) if m["value"] is not None else None, m.get("unit"),
            m.get("split"), m.get("horizon"))


class ArchiveETL:
    def __init__(self, outbox: Any, archive: Any, sink: Any, *, confirmation_depth: int = 6) -> None:
        self.outbox = outbox
        self.archive = archive
        self.sink = sink
        self.confirmation_depth = confirmation_depth
        self.crash_before_ack = False  # test hook: simulates a power loss after projection

    def run(self, blocks: Sequence[Any]) -> dict:
        summary: dict[str, Any] = {"schema": "i6n.archive.etl_run.v1", "chain_verified": False,
                                   "references": 0, "finalized": 0, "projected_reports": 0,
                                   "duplicate_reports": 0, "revoked_blocks": [], "acked": 0, "errors": []}
        try:
            refs = canonical_references(blocks, confirmation_depth=self.confirmation_depth)
        except ArchiveRefusal as exc:
            summary["errors"].append({"stage": "CHAIN", "error": f"ArchiveRefusal: {exc}"})
            return summary
        summary["chain_verified"] = True
        summary["references"] = len(refs)
        canonical_hashes = {b.hash for b in blocks}
        chain_tx_ids = {tx.id for b in blocks for tx in b.transactions}
        tip = blocks[-1]
        self._revoke(canonical_hashes, chain_tx_ids, tip, summary)
        done = self.outbox.done_keys()  # watermark: work is proportional to new references
        summary["skipped_done"] = 0
        for ref in refs:
            if ref.finality != FINALIZED:
                continue
            summary["finalized"] += 1
            if (ref.tx_id, ref.block_hash) in done:
                summary["skipped_done"] += 1
                continue
            try:
                self._project(ref, tip, summary)
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad reference must not block the others
                summary["errors"].append({"stage": "PROJECT", "tx_id": ref.tx_id, "block_hash": ref.block_hash,
                                          "error": f"{type(exc).__name__}: {exc}"[:400]})
        return summary

    # -- reorg -------------------------------------------------------------------------------
    def _revoke(self, canonical_hashes: set, chain_tx_ids: set, tip: Any, summary: dict) -> None:
        for block_hash in sorted(self.outbox.active_blocks() - canonical_hashes - {NO_BLOCK}):
            detail = {"reason": "REORG", "observed_tip_hash": tip.hash, "observed_tip_height": tip.header.index}
            self.sink.revoke({"schema": SCHEMA_REVOKE, "block_hash": block_hash, **detail})
            self.outbox.mark_block_revoked(block_hash, detail)
            self.outbox.forget_block(block_hash)
            summary["revoked_blocks"].append(block_hash)
        for item in self.outbox.items():
            final = item["states"]["CHAIN_FINALIZED"]
            if not final or final["block_hash"] in canonical_hashes:
                continue
            detail = {"block_hash": final["block_hash"], "observed_tip_hash": tip.hash}
            self.outbox.clear_state(item["item_id"], "CHAIN_FINALIZED", "REVOKED_REORG", detail)
            if item["states"]["OLAP_READBACK_VERIFIED"]:
                self.outbox.clear_state(item["item_id"], "OLAP_READBACK_VERIFIED", "REVOKED_REORG", detail)
            if item["archive_mode"] == "full_payload" and item["states"]["OBJECT_DURABLE"]:
                self.outbox.clear_state(item["item_id"], "OBJECT_DURABLE", "REVOKED_REORG", detail)
            if item["reference_tx_id"]:
                self.outbox.forget_done(item["reference_tx_id"], NO_BLOCK)  # off-chain readback may be reinstated
            if item["reference_tx_id"] and item["reference_tx_id"] not in chain_tx_ids:
                self.outbox.clear_reference(item["item_id"], "TX_NOT_ON_CANONICAL_CHAIN")

    # -- projection ----------------------------------------------------------------------------
    def _item_for(self, resolved: Any, mode: str) -> dict | None:
        return self.outbox.item_by_body(resolved.body_digest, mode)

    def _project(self, ref: Any, tip: Any, summary: dict) -> None:
        resolved = self.archive.resolve(ref.reference)
        units = records.units_for(resolved.payload_schema, resolved.payload)
        mode = ref.reference["archive_mode"]
        item = self._item_for(resolved, mode)
        if item is not None and not item["states"]["CHAIN_FINALIZED"]:
            self.outbox.set_state(item["item_id"], "CHAIN_FINALIZED", {
                "block_hash": ref.block_hash, "height": ref.height, "tx_id": ref.tx_id, "depth": ref.depth,
                "confirmation_depth": self.confirmation_depth, "tip_hash": tip.hash,
                "chain_verified": True, "verified_at": _now()})
            if mode == "full_payload" and not item["states"]["OBJECT_DURABLE"]:
                self.outbox.set_state(item["item_id"], "OBJECT_DURABLE", {
                    "archive_mode": "full_payload", "durable_in": "FINALIZED_CHAIN_TRANSACTION",
                    "block_hash": ref.block_hash, "tx_id": ref.tx_id, "body_digest": resolved.body_digest})
        self._write_and_readback(units, resolved, mode, item, block_hash=ref.block_hash, height=ref.height,
                                 tx_id=ref.tx_id, depth=ref.depth, finality=FINALIZED,
                                 locator=ref.reference["locator"], summary=summary)

    def _write_and_readback(self, units, resolved, mode, item, *, block_hash, height, tx_id, depth, finality,
                            locator, summary) -> None:
        reports = []
        for unit in units:
            report = unit["report"]
            projection = dict(unit["identity"], **unit["extra"],
                              row_key_sha256=report["tags"]["row_key_sha256"],
                              report_sha256=report["report_sha256"], block_hash=block_hash,
                              block_height=height, tx_id=tx_id, body_digest=resolved.body_digest,
                              manifest_digest=resolved.manifest_digest, locator=locator,
                              archive_mode=mode, finality_state=finality,
                              confirmation_depth=self.confirmation_depth, depth=depth)
            receipt = self.sink.write({"schema": SCHEMA_WRITE, "report": report, "projection": projection})
            state = receipt["state"]
            self.outbox.upsert_projection(
                projection_key=receipt["projection_key"], item_id=item["item_id"] if item else None,
                row_key_sha256=projection["row_key_sha256"], report_sha256=report["report_sha256"],
                block_hash=block_hash, tx_id=tx_id, finality_state=finality, state=state, receipt=receipt)
            if state == "PROJECTED":
                summary["projected_reports"] += 1
            elif state == "DUPLICATE_NOT_PROJECTED":
                summary["duplicate_reports"] += 1
            reports.append(report)
        if self.crash_before_ack:
            raise SystemExit("simulated power loss before readback/ACK")
        verified_rows = self._readback(reports)
        self.outbox.mark_done(tx_id, block_hash, item["item_id"] if item else None, len(reports))
        if item is not None:
            self.outbox.set_state(item["item_id"], "OLAP_READBACK_VERIFIED", {
                "reports": len(reports), "metric_rows": verified_rows, "block_hash": block_hash,
                "tx_id": tx_id, "finality_state": finality, "chain_finalized": finality == FINALIZED,
                "verified_at": _now()})
            summary["acked"] += 1

    # -- off-chain readback: durable, hash-verified bytes; NO chain finality claimed ----------
    def run_offchain(self) -> dict:
        """Project items whose bytes are durable in the lake and whose reference went to the node
        spool, before any chain finalizes them. finality_state OFFCHAIN_VERIFIED, block NO_BLOCK,
        chain_finalized=false; superseded (never duplicated) by the FINALIZED projection later.
        Failures stay visible in ``errors`` and are retried by the next run; nothing is refit."""
        summary: dict[str, Any] = {"schema": "i6n.archive.etl_offchain_run.v1", "candidates": 0,
                                   "projected_reports": 0, "duplicate_reports": 0, "acked": 0,
                                   "readback_verified": 0, "skipped_done": 0, "errors": []}
        done = self.outbox.done_keys()
        for item in self.outbox.items():
            if item["states"]["CHAIN_FINALIZED"] or not item["states"]["OBJECT_DURABLE"]:
                continue
            reference, tx_id = item["reference"], item["reference_tx_id"]
            if not reference or not tx_id or reference.get("archive_mode") != "external_payload":
                continue  # the reference must exist (spooled) and point at the lake
            summary["candidates"] += 1
            if (tx_id, NO_BLOCK) in done:
                summary["skipped_done"] += 1
                continue
            try:
                resolved = self.archive.resolve(reference)
                if resolved.body_digest != item["body_digest"]:
                    raise ArchiveRefusal("DIGEST_MISMATCH: lake body differs from the outbox item")
                units = records.units_for(resolved.payload_schema, resolved.payload)
                before = summary["acked"]
                self._write_and_readback(units, resolved, "external_payload", item, block_hash=NO_BLOCK, height=0,
                                         tx_id=tx_id, depth=0, finality=OFFCHAIN_VERIFIED,
                                         locator=reference["locator"], summary=summary)
                summary["readback_verified"] += summary["acked"] - before
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 - visible and retried next run, never refit
                summary["errors"].append({"stage": "OFFCHAIN", "item_id": item["item_id"],
                                          "error": f"{type(exc).__name__}: {exc}"[:400]})
        return summary

    def _readback(self, reports: list[dict]) -> int:
        digests = sorted({r["report_sha256"] for r in reports})
        read = self.sink.read({"report_sha256": digests})
        active = {(p["row_key_sha256"], p["report_sha256"]) for p in read["projections"] if p["state"] == "PROJECTED"}
        stored: dict[str, Counter] = {}
        for m in read["metrics"]:
            stored.setdefault(m["report_sha256"], Counter())[_metric_key(m)] += 1
        rows = 0
        for report in reports:
            key = (report["tags"]["row_key_sha256"], report["report_sha256"])
            if key not in active:
                raise ArchiveRefusal("READBACK_NOT_CANONICAL: no active projection for " + key[1][:12])
            wanted = Counter(_metric_key(m) for m in report["metrics"])
            if stored.get(report["report_sha256"]) != wanted:
                raise ArchiveRefusal("READBACK_MISMATCH: " + report["report_sha256"][:12])
            rows += sum(wanted.values())
        return rows
