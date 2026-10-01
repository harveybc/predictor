"""tools/m06_gov_mapping.py: a proposal only, on the governed grain, from the retained M04 evidence."""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import m06_gov_mapping as G  # noqa: E402

RES = ROOT / "docs/audits/evidence/MODULAR_CAMPAIGN_20260930/RESULTS"


def load():
    q = json.load(open(RES / "m04_QUEUE_batch1_v3.from_6ced97e7.json"))
    d = RES / "m04_verify_receipts"
    rec = {f[:-5]: json.load(open(d / f)) for f in os.listdir(d) if f.endswith(".json")}
    return q, rec


def test_one_terminal_per_verified_candidate_on_the_unique_grain():
    q, rec = load()
    p = G.propose(q, rec, None)
    assert p["status"] == "PROPOSAL_NO_ROWS_WRITTEN" and p["counts"]["terminals"] == 16
    keys = {(u["gov_terminal"]["campaign_sha256"], u["gov_terminal"]["unit_id"], u["gov_terminal"]["generation"])
            for u in p["units"]}
    assert len(keys) == 16                                  # UNIQUE(campaign_sha256, unit_id, generation)


def test_metric_rows_are_aggregate_plus_24_horizons_with_terminal_keys():
    q, rec = load()
    u = G.propose(q, rec, None)["units"][0]
    rows = u["gov_terminal_metric"]
    assert len(rows) == 6 * 25
    assert all(re.fullmatch(r"[A-Za-z0-9._:-]+", r["metric"]) for r in rows)
    assert {r["horizon"] for r in rows} == {None, *range(1, 25)}
    assert {r["split"] for r in rows} == {"validation"} and {r["unit"] for r in rows} == {"z_train"}


def test_without_a_governed_delivery_the_terminal_is_non_governing_and_the_gap_is_named():
    q, rec = load()
    p = G.propose(q, rec, None)
    assert all(u["gov_terminal"]["classification"] == "NON_GOVERNING" for u in p["units"])
    assert all(u["gov_terminal_dataset"] == [] for u in p["units"])
    assert any(g.startswith("NO_GOVERNED_DELIVERY") for g in p["gaps"])


def test_the_aggregate_mae_row_equals_the_queue_objective():
    q, rec = load()
    p = G.propose(q, rec, None)
    obj = {c["cid"]: c["objective"] for c in q["candidates"]}
    for u in p["units"]:
        agg = next(r for r in u["gov_terminal_metric"] if r["metric"] == "MAE" and r["horizon"] is None)
        assert agg["value"] == obj[u["gov_terminal"]["unit_id"]]
