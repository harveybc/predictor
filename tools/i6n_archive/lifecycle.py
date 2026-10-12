"""One status that separates every stage a fit and its record pass through.

Fold lifecycle, from B's read-only status (``tools.i6n_swarm.status.compute``):

* ``QUEUED``    not running (cause kept: QUEUED, WAIT_RESOURCE, WAIT_DEPENDENCY);
* ``CLAIMED``   RUNNING with a lease, but no evaluator progress observed yet;
* ``UPDATING``  RUNNING and the evaluator reports epochs (its own measured ETA);
* ``COMPLETED`` the fold's terminal is verified COMPLETE; ``FAILED``/``INTERRUPTED`` as reported.

Candidate archive lifecycle, from lane D's outbox (never inferred from the fit):

* ``ARCHIVED``          OBJECT_DURABLE: body PUT to the lake, re-read and hashed;
* ``READBACK_VERIFIED`` OLAP_READBACK_VERIFIED: every projected report read back exactly;
* ``chain_finalized``   a separate boolean; an off-chain readback keeps it false.

ETA is copied from B with its kind (measured | projected | unknown) and its sample count
``basis_n``; anything without a sample count is reported as unknown. Read-only: no TF, no
training, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FOLD_STATES = ("QUEUED", "CLAIMED", "UPDATING", "COMPLETED", "FAILED", "INTERRUPTED")
ARCHIVE_STATES = ("NOT_RECORDED", "LOCAL_COMPLETE", "ARCHIVED", "READBACK_VERIFIED")


def fold_lifecycle(fold: dict) -> str:
    state = fold.get("state")
    if state == "COMPLETE":
        return "COMPLETED"
    if state in ("FAILED", "INTERRUPTED"):
        return state
    if state == "RUNNING":
        eta = fold.get("eta") or {}
        if eta.get("kind") == "measured" and eta.get("source") == "evaluator_epochs":
            return "UPDATING"
        return "CLAIMED" if fold.get("lease") else "QUEUED"
    return "QUEUED"


def _eta(eta: Any) -> dict:
    eta = dict(eta or {})
    kind = eta.get("kind")
    if kind not in ("measured", "projected") or (kind == "projected" and not eta.get("basis_n")):
        return {"kind": "unknown", "seconds": None, "basis_n": eta.get("basis_n", 0) or 0,
                "cause": eta.get("cause") or "no measured sample"}
    return {"kind": kind, "seconds": eta.get("seconds"), "low": eta.get("low"), "high": eta.get("high"),
            "basis_n": eta.get("basis_n"), "source": eta.get("source")}


def archive_lifecycle(item: dict | None) -> str:
    if item is None:
        return "NOT_RECORDED"
    states = item["states"]
    if states.get("OLAP_READBACK_VERIFIED"):
        return "READBACK_VERIFIED"
    if states.get("OBJECT_DURABLE"):
        return "ARCHIVED"
    return "LOCAL_COMPLETE"


def compute(swarm_status: dict, outbox: Any | None, now: str | None = None) -> dict:
    levels = swarm_status.get("levels", {})
    folds = []
    fold_counts = {s: 0 for s in FOLD_STATES}
    for fold in levels.get("fold", []):
        life = fold_lifecycle(fold)
        fold_counts[life] += 1
        folds.append({"fit_id": fold.get("fit_id"), "dag_node": fold.get("dag_node"), "horizon": fold.get("horizon"),
                      "fold": fold.get("fold"), "lifecycle": life, "state": fold.get("state"),
                      "cause": fold.get("cause"), "attempts": fold.get("attempts"),
                      "lease": bool(fold.get("lease")), "eta": _eta(fold.get("eta")) if fold.get("eta") else None})
    items = {}
    if outbox is not None:
        for item in outbox.items():
            items[item["search_candidate_sha256"]] = item
    archive_counts = {s: 0 for s in ARCHIVE_STATES}
    candidates = []
    for cand in levels.get("candidate", []):
        item = items.get(cand.get("search_candidate_sha256"))
        life = archive_lifecycle(item)
        archive_counts[life] += 1
        final = (item or {}).get("states", {}).get("CHAIN_FINALIZED") if item else None
        readback = (item or {}).get("states", {}).get("OLAP_READBACK_VERIFIED") if item else None
        candidates.append({"search_candidate_sha256": cand.get("search_candidate_sha256"),
                           "dag_node": cand.get("dag_node"), "fit_state": cand.get("state"),
                           "folds_complete": cand.get("folds_complete"), "folds_total": cand.get("folds_total"),
                           "archive": life, "chain_finalized": bool(final),
                           "readback_finality": (readback or {}).get("finality_state")})
    decisions = [i for k, i in items.items() if str(k).startswith("DECISION:")]
    nodes = [{"node_id": n.get("node_id"), "state": n.get("state"), "cause": n.get("cause"),
              "counts": n.get("counts"), "eta": _eta(n.get("eta"))} for n in levels.get("horizon", [])]
    swarm = levels.get("swarm", {})
    return {"schema": "i6n.archive.lifecycle.v1", "campaign_id": swarm_status.get("campaign_id"),
            "generated_at": now or datetime.now(timezone.utc).isoformat(),
            "fold_lifecycle_counts": fold_counts, "archive_lifecycle_counts": archive_counts,
            "chain_finalized_candidates": sum(1 for c in candidates if c["chain_finalized"]),
            "decision_events": {"recorded": len(decisions),
                                "readback_verified": sum(1 for d in decisions if d["states"]["OLAP_READBACK_VERIFIED"])},
            "swarm": {"state": swarm.get("state"), "cause": swarm.get("cause"), "eta": _eta(swarm.get("eta"))},
            "nodes": nodes, "candidates": candidates, "folds": folds,
            "publication": levels.get("publication")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, help="B status JSON; computed read-only when omitted")
    parser.add_argument("--write", action="store_true", help="also write <root>/archive/LIFECYCLE.json")
    args = parser.parse_args(argv)
    if args.status_json:
        swarm_status = json.loads(args.status_json.read_text())
    else:
        from tools.i6n_swarm import status as swarm_status_module  # integration checkout only

        swarm_status = swarm_status_module.compute(args.campaign_root)
    outbox_path = args.campaign_root / "archive" / "outbox.sqlite"
    outbox = None
    if outbox_path.exists():
        from tools.i6n_archive.outbox import Outbox

        outbox = Outbox(outbox_path)
    result = compute(swarm_status, outbox)
    text = json.dumps(result, sort_keys=True, indent=1)
    if args.write:
        target = args.campaign_root / "archive" / "LIFECYCLE.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(text)
        tmp.replace(target)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
