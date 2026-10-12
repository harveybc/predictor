"""Publisher API for lane B: record every candidate locally, then pump bytes and references.

``record_candidate`` never blocks a fit: it validates, builds the canonical body bytes and
writes one outbox item (LOCAL_COMPLETE). ``pump`` is independent and replayable:

1. external_payload: PUT body + manifest to the lake, GET both back, compare digests
   (doin-node ``ExternalPayloadArchive.store_body``) -> OBJECT_DURABLE;
2. only then build the reference transaction (deterministic timestamp fixed at record time,
   so a resubmission after a crash has the same tx id) and hand it to ``submit``.

A failed PUT/GET leaves the item at LOCAL_COMPLETE and submits nothing. ``full_payload``
keeps the body inside the transaction; its OBJECT_DURABLE is set by the ETL when that
transaction is finalized, never before.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from doin_core.archive.external import ExternalBody, build_external_body, reference_payload
from doin_node.archive.external import build_reference_transaction

from tools.i6n_archive import records
from tools.i6n_archive.outbox import STATES, Outbox

MODES = ("external_payload", "full_payload")


class ArchivePublisher:
    def __init__(self, root: str | Path, *, store_id: str, domain_id: str, producer_peer_id: str,
                 mode: str = "external_payload", clock: Callable[[], datetime] | None = None) -> None:
        if mode not in MODES:
            raise ValueError("ARCHIVE_MODE")
        self.root = Path(root)
        self.mode = mode
        self.store_id = store_id
        self.domain_id = domain_id
        self.producer_peer_id = producer_peer_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.outbox = Outbox(self.root / "outbox.sqlite")
        self.outbox.bind_meta({"store_id": store_id, "domain_id": domain_id,
                               "producer_peer_id": producer_peer_id, "mode": mode})

    @classmethod
    def from_outbox(cls, path: str | Path) -> "ArchivePublisher":
        path = Path(path)
        if path.name != "outbox.sqlite":
            raise ValueError("OUTBOX_FILE_NAME")
        meta = Outbox(path).meta()
        return cls(path.parent, store_id=meta["store_id"], domain_id=meta["domain_id"],
                   producer_peer_id=meta["producer_peer_id"], mode=meta["mode"])

    # -- B calls this -----------------------------------------------------------------------
    def record_candidate(self, candidate: dict, fold_receipts: list[dict]) -> dict:
        """Every candidate at its terminal disposition, with every physical attempt's receipt.
        The search outcome is NOT part of it: see ``record_decision``."""
        payload = records.round_payload(candidate, fold_receipts)
        return self._record(records.ROUND_SCHEMA, payload, payload["search_candidate_sha256"])

    def record_decision(self, decision: dict) -> dict:
        """Append one search outcome event (winner/finalist/control) for (experiment, plan, h)."""
        payload = records.validate_decision(decision)
        key = f"DECISION:{payload['plan_sha256'][:16]}:h{payload['horizon_hours']}:{payload['sequence']}"
        return self._record(records.DECISION_SCHEMA, payload, key)

    def _record(self, payload_schema: str, payload: dict, key: str) -> dict:
        units = records.units_for(payload_schema, payload)
        row_reports = {u["report"]["tags"]["row_key_sha256"]: u["report"]["report_sha256"] for u in units}
        external = build_external_body(domain_id=self.domain_id, producer_peer_id=self.producer_peer_id,
                                       payload_schema=payload_schema, payload=payload)
        created = self.outbox.insert_item(
            item_id=external.body_digest, experiment=payload["experiment"],
            search_candidate_sha256=key, archive_mode=self.mode,
            body=external.body, manifest=external.manifest, body_digest=external.body_digest,
            manifest_digest=external.manifest_digest, tx_timestamp=self._clock().isoformat(),
            row_reports=row_reports)
        item = self.outbox.item(external.body_digest)
        return {"item_id": item["item_id"], "body_digest": external.body_digest,
                "manifest_digest": external.manifest_digest, "created": created, "states": item["states"]}

    # -- independent pump --------------------------------------------------------------------
    def _external(self, item: dict) -> ExternalBody:
        body, manifest = self.outbox.bytes_of(item["item_id"])
        parsed = json.loads(body)
        return ExternalBody(body=body, body_digest=item["body_digest"], manifest=manifest,
                            manifest_digest=item["manifest_digest"], domain_id=parsed["domain_id"],
                            payload_schema=parsed["payload_schema"], payload=parsed["payload"])

    def pump(self, archive: Any, submit: Callable[[Any], None], *, _after_store: Callable[[], None] | None = None) -> dict:
        summary: dict[str, Any] = {"stored": 0, "submitted": 0, "errors": []}
        for item in self.outbox.items():
            if item["archive_mode"] != self.mode:
                continue
            external = self._external(item)
            try:
                if self.mode == "external_payload" and not item["states"]["OBJECT_DURABLE"]:
                    ref = archive.store_body(external, mode="external_payload")
                    if _after_store is not None:
                        _after_store()
                    self.outbox.set_state(item["item_id"], "OBJECT_DURABLE",
                                          {"store_id": self.store_id, "locator": ref["locator"],
                                           "body_digest": ref["body_digest"],
                                           "manifest_digest": ref["manifest_digest"],
                                           "readback": "GET_EXACT_DIGEST_VERIFIED",
                                           "verified_at": datetime.now(timezone.utc).isoformat()})
                    summary["stored"] += 1
                    item = self.outbox.item(item["item_id"])
            except Exception as exc:  # noqa: BLE001 - recorded, item stays LOCAL_COMPLETE
                summary["errors"].append({"item_id": item["item_id"], "stage": "PUT_GET",
                                          "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            if item["reference_tx_id"] or item["states"]["CHAIN_FINALIZED"]:
                continue
            if self.mode == "external_payload" and not item["states"]["OBJECT_DURABLE"]:
                continue
            ref = reference_payload(external, mode=self.mode,
                                    store_id=self.store_id if self.mode == "external_payload" else None)
            tx = build_reference_transaction(ref, peer_id=self.producer_peer_id,
                                             timestamp=datetime.fromisoformat(item["tx_timestamp"]))
            try:
                submit(tx)
            except Exception as exc:  # noqa: BLE001 - retried by the next pump
                summary["errors"].append({"item_id": item["item_id"], "stage": "SUBMIT",
                                          "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            self.outbox.set_reference(item["item_id"], tx.id, ref)
            summary["submitted"] += 1
        return summary

    # -- read-only status --------------------------------------------------------------------
    def status(self) -> dict:
        items = self.outbox.items()
        per_state = {s: sum(1 for i in items if i["states"][s]) for s in STATES}
        missing = [{"item_id": i["item_id"], "missing": [s for s in STATES if not i["states"][s]]}
                   for i in items if not i["green"]]
        return {"schema": "i6n.archive.status.v1", "mode": self.mode, "items": len(items),
                "per_state": per_state, "green": sum(1 for i in items if i["green"]), "missing": missing,
                "submitted_unfinalized": sum(1 for i in items if i["reference_tx_id"]
                                             and not i["states"]["CHAIN_FINALIZED"])}
