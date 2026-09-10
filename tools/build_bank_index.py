#!/usr/bin/env python3
"""Build the common index of the three evidence banks.

Joins what each producer already published — the T2 public
manifest and census, the T1/M4 synthetic generator descriptors,
and the financial-data incremental census — without downloading,
scoring or re-profiling anything, and without letting any bank
inherit another's authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olap.bank_index import (build_index, financial_bank,  # noqa
                             public_bank, synthetic_bank)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indexed-at", required=True)
    ap.add_argument("--t2-manifest", type=Path)
    ap.add_argument("--t2-census", type=Path)
    ap.add_argument("--t2-adjudication", type=Path)
    ap.add_argument("--t1-inventory", type=Path)
    ap.add_argument("--m4-design", type=Path)
    ap.add_argument("--financial-census-summary", type=Path)
    ap.add_argument("--financial-census-receipt", type=Path)
    ap.add_argument("--local-inventory", type=Path)
    ap.add_argument("--financial-census-document", type=Path,
                    help="the FULL content-addressed census, so "
                         "the index carries its appearances and "
                         "conceptual variables as rows")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args(argv)

    pub = (public_bank(a.t2_manifest, a.t2_census,
                       a.t2_adjudication)
           if a.t2_manifest and a.t2_census else None)
    syn = (synthetic_bank(a.t1_inventory, a.m4_design)
           if (a.t1_inventory or a.m4_design) else None)
    fin = (financial_bank(a.financial_census_summary,
                          a.financial_census_receipt,
                          a.local_inventory,
                          a.financial_census_document)
           if (a.financial_census_summary
               and a.financial_census_receipt) else None)
    doc = build_index(a.indexed_at, pub, syn, fin)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, indent=1,
                                   sort_keys=True) + "\n")
    print(json.dumps({
        "index_sha256": doc["index_sha256"],
        "banks": {k: v["authority_class"]
                  for k, v in doc["banks"].items()},
        "rows": doc["row_count"],
        "cardinality": doc["cardinality_by_kind_and_authority"],
        "public_series_admissible": doc["banks"].get(
            "public_forecasting", {}).get(
                "series_admissible_total"),
        "synthetic_generators": doc["banks"].get(
            "synthetic_known_mechanism", {}).get(
                "generator_count"),
        "financial_variables": doc["banks"].get(
            "financial_domain", {}).get("coverage", {}).get(
                "conceptual_variables"),
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
