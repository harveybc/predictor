"""Lane C reproducers for the lane D review (disposable instances only, no network)."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT), str(ROOT / "olap/store/src")]
import tests.test_i6n_archive as T  # noqa: E402
from tools.i6n_archive import records  # noqa: E402

out = {}


def flow(pub, archive, etl, chain):
    pub.pump(archive, chain.submit)
    chain.mint(empty_after=6)
    return etl.run(chain.blocks)


# R1: the winner is only known after the search; re-recording the candidate with won=True
with tempfile.TemporaryDirectory() as d:
    pub, archive, etl, wh, _ = T._rig(Path(d))
    chain = T.LocalChain()
    c0 = T.candidate(0, won=False)
    pub.record_candidate(c0, [T.receipt(c0)])
    first = flow(pub, archive, etl, chain)
    c1 = T.candidate(0, won=True)
    pub.record_candidate(c1, [T.receipt(c1)])
    second = flow(pub, archive, etl, chain)
    status = pub.status()
    with wh.engine().connect() as c:
        won = c.exec_driver_sql("SELECT value FROM gov_doin_metric_canonical WHERE derived_metric='won'").fetchall()
    out["R1_won_update"] = {"first_errors": first["errors"], "second_errors": [e["error"][:90] for e in second["errors"]],
                            "items": status["items"], "green": status["green"],
                            "canonical_won_values": [r[0] for r in won]}

# R2: two attempts of one fold in one payload: accepted at record time, never projectable
with tempfile.TemporaryDirectory() as d:
    pub, archive, etl, wh, _ = T._rig(Path(d))
    chain = T.LocalChain()
    c0 = T.candidate(0)
    r1 = T.receipt(c0)
    r2 = json.loads(json.dumps(r1))
    r2["attempt"] = {"index": 2, "parent_status_sha256": "a" * 64, "fencing_token": "tok-2"}
    rec = pub.record_candidate(c0, [r1, r2])
    run = flow(pub, archive, etl, chain)
    out["R2_two_attempts_same_fold"] = {"recorded": bool(rec["item_id"]),
                                        "etl_errors": [e["error"][:90] for e in run["errors"]],
                                        "green": pub.status()["green"]}

# R3: VALIDATION (outer-week) rows on an INNER fold are accepted
c0 = T.candidate(0)
r = T.receipt(c0, fold="2023Q1")
r["rows"].append(dict(r["rows"][3], split="VALIDATION", n=7))
try:
    records.round_payload(c0, [r])
    out["R3_inner_fold_validation_rows"] = "ACCEPTED"
except records.ArchiveRecordError as exc:
    out["R3_inner_fold_validation_rows"] = f"REFUSED: {exc}"

# R4: C.2 fold ids are accepted, TEST-like ids refused
for fold in ("2023Q1_DCA", "2023Q4", "W2024-12-29", "W2025-01-05"):
    c0 = T.candidate(0)
    try:
        records.round_payload(c0, [T.receipt(c0, fold=fold)])
        out[f"R4_fold_{fold}"] = "ACCEPTED"
    except records.ArchiveRecordError as exc:
        out[f"R4_fold_{fold}"] = f"REFUSED: {exc}"

print(json.dumps(out, indent=1, default=str))
