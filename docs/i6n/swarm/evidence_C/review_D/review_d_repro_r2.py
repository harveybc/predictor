"""Lane C round-2 reproducers for lane D.3 (disposable instances only, no network)."""
import json, sys, tempfile
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT), str(ROOT / "olap/store/src")]
import tests.test_i6n_archive as T  # noqa: E402
from tools.i6n_archive import records  # noqa: E402

out = {}
PLAN = T.PLAN


def cand(i, **kw):
    c = T.candidate(i, **kw)
    c.pop("won", None)
    return c


def flow(pub, archive, etl, chain):
    pub.pump(archive, chain.submit)
    chain.mint(empty_after=6)
    return etl.run(chain.blocks)


def decision(seq, winner, others, supersedes=None):
    outcomes = [{"search_candidate_sha256": winner, "outcome": "WINNER"}] + \
               [{"search_candidate_sha256": o, "outcome": "NOT_SELECTED"} for o in others]
    return {"schema": records.DECISION_SCHEMA, "experiment": "i6n-swarm-test", "plan_sha256": PLAN,
            "horizon_hours": 12, "sequence": seq, "supersedes_decision_sha256": supersedes,
            "basis": {"rule": "min mean fold fitness"}, "outcomes": outcomes}


# R1: old reproducer is refused by design; the decision-event path works and supersedes append-only
with tempfile.TemporaryDirectory() as d:
    pub, archive, etl, wh, _ = T._rig(Path(d))
    chain = T.LocalChain()
    try:
        legacy = T.candidate(0)
        legacy["won"] = True  # the D.2 field, set explicitly (the D.3 fixture ignores won=)
        pub.record_candidate(legacy, [T.receipt(legacy)])
        out["R1_won_field"] = "ACCEPTED"
    except records.ArchiveRecordError as exc:
        out["R1_won_field"] = "REFUSED: " + str(exc)[:40]
    a, b = cand(0), cand(1)
    pub.record_candidate(a, [T.receipt(a)])
    pub.record_candidate(b, [T.receipt(b)])
    s1 = flow(pub, archive, etl, chain)
    d1 = pub.record_decision(decision(1, a["search_candidate_sha256"], [b["search_candidate_sha256"]]))
    s2 = flow(pub, archive, etl, chain)
    pub.record_decision(decision(2, b["search_candidate_sha256"], [a["search_candidate_sha256"]],
                                 supersedes=d1["body_digest"]))
    s3 = flow(pub, archive, etl, chain)
    with wh.engine().connect() as c:
        cur = c.exec_driver_sql("SELECT search_candidate_sha256, outcome, decision_fold, won FROM"
                                " gov_doin_candidate_decision_current ORDER BY 1").fetchall()
        hist = c.exec_driver_sql("SELECT count(*) FROM gov_doin_metric_canonical WHERE derived_metric='won'").scalar()
    winner = [r for r in cur if r[1] == "WINNER"]
    out["R1_decisions"] = {"errors": [len(s1["errors"]), len(s2["errors"]), len(s3["errors"])],
                           "green": pub.status()["green"], "items": pub.status()["items"],
                           "current_winner_is_b": len(winner) == 1 and winner[0][0] == b["search_candidate_sha256"],
                           "current_decision_fold": sorted({r[2] for r in cur}), "won_history_rows": hist}
    # same sequence, different outcome -> refused at record time, never at the ETL
    try:
        pub.record_decision(decision(2, a["search_candidate_sha256"], [b["search_candidate_sha256"]]))
        out["R1_same_sequence_conflict"] = "ACCEPTED"
    except records.ArchiveRecordError as exc:
        out["R1_same_sequence_conflict"] = "REFUSED: " + str(exc)[:30]
    # changed candidate content after archiving -> refused at record time
    try:
        a2 = cand(0, fitness=0.5)
        pub.record_candidate(a2, [T.receipt(a2)])
        out["R1_rerecord_changed"] = "ACCEPTED"
    except records.ArchiveRecordError as exc:
        out["R1_rerecord_changed"] = "REFUSED: " + str(exc)[:30]

# R2: two attempts of one fold, different sums: lowest attempt projected, all kept, green
with tempfile.TemporaryDirectory() as d:
    pub, archive, etl, wh, _ = T._rig(Path(d))
    chain = T.LocalChain()
    c0 = cand(0)
    r1 = T.receipt(c0)
    r2 = T.receipt(c0, scale=2.0)
    r2["attempt"] = {"index": 2, "parent_status_sha256": "a" * 64, "fencing_token": "tok-2"}
    pub.record_candidate(c0, [r2, r1])
    run = flow(pub, archive, etl, chain)
    with wh.engine().connect() as c:
        sab = c.exec_driver_sql("SELECT value FROM gov_doin_metric_canonical WHERE derived_metric='sum_abs_error'"
                                " AND split='TRAIN' AND head_offset_hours=12").fetchall()
        hist = c.exec_driver_sql("SELECT tags_json FROM gov_report WHERE phase='I6N_SWARM_DOIN_CANDIDATE'").fetchall()
    h = json.loads(json.loads(hist[0][0])["attempt_history_json"])
    out["R2_two_attempts"] = {"errors": run["errors"], "green": pub.status()["green"],
                              "projected_train_h12_sum_abs": [r[0] for r in sab],
                              "attempt1_value": r1["rows"][1]["sum_abs_error"],
                              "history": [(x["index"], x["canonical"]) for x in h]}
    bad = T.receipt(c0)
    bad["attempt"] = {"index": 2, "parent_status_sha256": None, "fencing_token": "t"}
    try:
        records.round_payload(c0, [r1, bad])
        out["R2_attempt_without_parent"] = "ACCEPTED"
    except records.ArchiveRecordError as exc:
        out["R2_attempt_without_parent"] = "REFUSED: " + str(exc)[:30]

# R3: VALIDATION on an inner fold
c0 = cand(0)
r = T.receipt(c0, fold="2023Q1")
r["rows"].append(dict(r["rows"][3], split="VALIDATION", n=7))
try:
    records.round_payload(c0, [r])
    out["R3_inner_fold_validation_rows"] = "ACCEPTED"
except records.ArchiveRecordError as exc:
    out["R3_inner_fold_validation_rows"] = "REFUSED: " + str(exc)[:30]

# R4: C.2 fold ids
for fold in ("2023Q1_DCA", "2023Q4", "W2024-12-29", "W2025-01-05"):
    try:
        records.round_payload(c0, [T.receipt(c0, fold=fold)])
        out[f"R4_fold_{fold}"] = "ACCEPTED"
    except records.ArchiveRecordError as exc:
        out[f"R4_fold_{fold}"] = "REFUSED: " + str(exc)[:30]
print(json.dumps(out, indent=1, default=str))
