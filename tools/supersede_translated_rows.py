#!/usr/bin/env python3
"""C12: mark the previously translated rows, additively.

The first backfill loaded 60 facts from envelopes that had merely
translated a producer's summary — they were never checked against
the producer's own artifact. They are NOT deleted and NOT
rewritten: two additive columns record that they are
`TRANSLATED_SUMMARY_NON_AUTHORITATIVE` and name the
producer-verified envelope that now supersedes them, so a reader
can see both the history and which row to trust.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap import campaign_envelope as ce  # noqa: E402
from tools.backfill_campaign_envelopes import (  # noqa: E402
    _engine, counts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envelope-dir", required=True, type=Path)
    ap.add_argument("--receipt", required=True, type=Path)
    ap.add_argument("--as-of", required=True)
    a = ap.parse_args(argv)

    from sqlalchemy import text
    engine = _engine()

    # the producer-verified envelope per campaign, which is what
    # supersedes the translated rows
    superseding = {}
    for p in sorted(a.envelope_dir.glob("envelope-*.json")):
        doc = json.loads(p.read_text())
        ce.validate_envelope(doc)
        state = ce.envelope_authority_state(doc)
        if state == ce.PRODUCER_BOUND:
            superseding[doc["campaign_key"]] = \
                doc["envelope_sha256"]

    before = counts(engine)
    marked, linked = {}, {}
    with engine.begin() as conn:
        conn.exec_driver_sql(ce.ENVELOPE_DDL)
        # every row without an authority state predates the
        # producer binding; that is exactly the translated set
        rows = conn.execute(text(
            f"SELECT campaign_key, count(*) AS n FROM "
            f"{ce.SCHEMA}.fact_campaign_unit "
            "WHERE authority_state IS NULL "
            "GROUP BY campaign_key")).mappings().all()
        for r in rows:
            marked[r["campaign_key"]] = int(r["n"])
        conn.execute(text(
            f"UPDATE {ce.SCHEMA}.fact_campaign_unit "
            "SET authority_state = :s "
            "WHERE authority_state IS NULL"),
            {"s": ce.TRANSLATED})
        for campaign_key, envelope_sha in sorted(
                superseding.items()):
            res = conn.execute(text(
                f"UPDATE {ce.SCHEMA}.fact_campaign_unit "
                "SET superseded_by_envelope_sha256 = :e "
                "WHERE campaign_key = :k "
                "AND authority_state = :s "
                "AND envelope_sha256 <> :e"),
                {"e": envelope_sha, "k": campaign_key,
                 "s": ce.TRANSLATED})
            linked[campaign_key] = res.rowcount or 0
    after = counts(engine)

    receipt = {
        "schema": "crispdm.translated_row_supersession.v1",
        "as_of": a.as_of,
        "rows_marked_translated": marked,
        "rows_marked_total": sum(marked.values()),
        "rows_linked_to_superseding_envelope": linked,
        "superseding_envelopes": superseding,
        "counts_before": before,
        "counts_after": after,
        "deletions": "NONE — no row was deleted and no metric "
                     "was rewritten; two additive columns carry "
                     "the label and the link",
    }
    receipt["receipt_sha256"] = ce._sha(receipt,
                                        "receipt_sha256")
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=1,
                                    sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("counts_before",
                                   "counts_after")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
