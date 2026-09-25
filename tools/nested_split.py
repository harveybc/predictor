#!/usr/bin/env python
"""Declare, once, the nested split WP26 needs, and freeze it before a single weight is fitted.

WP26 of the M5PHET work plan (2026-09-24, revision 3). The round of 2026-09-25 (WP06 stage 5) scored 72 stages on one
sealed population and then reported the best of them. That is selection on the holdout: the seal stopped being held out
the moment it ranked candidates, and the 4.50 % the table published may be partly the search fitting that population's
noise. The only instrument that can answer it is a second population that no search, no candidate and no earlier round
used to rank anything.

What this tool declares, and what it refuses to pretend:

* the source slice is split into an **inner region** (fitting, validation, and any search) and an **outer holdout**,
  contiguous by time, disjoint from each other and disjoint from the population the earlier round sealed;
* the earlier round's seal already consumes the **tail** of the slice. So the outer block cannot be the final block by
  time, and this tool takes instead the last contiguous block **before** that seal's first history row. That costs two
  things, and they are written into the frozen file rather than left to be discovered: the outer block is an
  **earlier week** than the published comparison scored on (a different regime, however mildly), and the inner region
  is **smaller** than the region the earlier fits trained on, so every number confirmed here is confirmed under less
  training data than the number it confirms;
* the residual this construction cannot remove is stated by name, ``OUTER_ROWS_WERE_TRAINING_ROWS_OF_THE_EARLIER_ROUND``:
  the outer rows were never scored on and never ranked anything, but the earlier round's fits did read them as training
  rows. The re-fits WP26 dispatches never read them at all. A block untouched in *every* sense does not exist inside
  this slice, and one taken from outside it would be another file and another provenance;
* the freeze is a freeze. The file is written once and this tool refuses to overwrite it, because a split that can be
  rewritten after a fit is not a holdout.

The outer seal is computed with the same machinery every stage of the comparison uses
(``tools/fit_pipeline_spec.py``'s population rule and ``m5phet_evaluation``'s protocol and corpus seal), so a stage
fitted against the derived file with the declared ``--holdout-fraction`` recomputes the identical digest and can be
refused by ``--expect-seal`` when it does not.

Usage::

    python tools/nested_split.py --data slice.csv --derived-csv <dir>/inner_outer.csv \\
        --out <evidence>/nested_split.json --evaluation-src <M5PHET/evaluation/src> \\
        --seal-window 197 --horizon 60 --outer-rows 10080 --sealed-at 2026-09-25T18:00:00Z \\
        --earlier-holdout-fraction 0.2 --earlier-seal 33820b552ddf
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "predictor.nested_split.v1"

#: the residual no construction inside this slice can remove, named so it cannot be paraphrased away
RESIDUAL = "OUTER_ROWS_WERE_TRAINING_ROWS_OF_THE_EARLIER_ROUND"


def load_fit_harness():
    """The fitting tool's own population rule and protocol builder, so the split is sealed by the same code."""
    spec = importlib.util.spec_from_file_location("fit_pipeline_spec", REPO_ROOT / "tools" / "fit_pipeline_spec.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="the source slice the earlier round used")
    parser.add_argument("--derived-csv", required=True, type=Path,
                        help="where to write the inner+outer file the WP26 fits read (source rows before the earlier seal)")
    parser.add_argument("--out", required=True, type=Path, help="the frozen declaration; refused if it exists")
    parser.add_argument("--evaluation-src", required=True, type=Path)
    parser.add_argument("--seal-window", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--outer-rows", type=int, required=True,
                        help="rows of the derived file that form the outer holdout, the last block by time")
    parser.add_argument("--sealed-at", required=True)
    parser.add_argument("--target", default="Global_active_power")
    parser.add_argument("--minimum-rows", type=int, default=1000)
    parser.add_argument("--earlier-holdout-fraction", type=float, default=0.2,
                        help="the holdout fraction the earlier round declared on the SOURCE file")
    parser.add_argument("--earlier-seal", default=None, help="the earlier round's seal prefix, recorded and checked")
    args = parser.parse_args(argv)

    if args.out.exists():
        print(f"REFUSED: {args.out} already exists. A split that can be rewritten after a fit is not a holdout; "
              f"delete it deliberately or declare another path.", file=sys.stderr)
        return 2

    harness = load_fit_harness()
    if str(args.evaluation_src) not in sys.path:
        sys.path.insert(0, str(args.evaluation_src))

    source = harness.read_csv(args.data)
    source["target_index"] = source["columns"].index(args.target)
    source_rows = len(source["values"])

    # what the earlier round sealed, recomputed from its own rule -- not copied from its README
    earlier = harness.sealed_population(source, holdout_fraction=args.earlier_holdout_fraction,
                                        seal_window=args.seal_window, horizon=args.horizon)
    earlier_start = earlier["holdout_start"]
    earlier_origins = set(int(value) for value in earlier["origins"])

    # the derived file: every source row BEFORE the first row the earlier seal reads. Its own tail is the outer block.
    lines = args.data.read_text().splitlines()
    header, body = lines[0], lines[1:]
    if len(body) != source_rows:
        raise SystemExit(f"the file holds {len(body)} data lines and {source_rows} parsed rows; refusing to slice it")
    args.derived_csv.parent.mkdir(parents=True, exist_ok=True)
    args.derived_csv.write_text("\n".join([header] + body[:earlier_start]) + "\n")

    derived = harness.read_csv(args.derived_csv)
    derived["target_index"] = derived["columns"].index(args.target)
    derived_rows = len(derived["values"])
    if derived_rows != earlier_start:
        raise SystemExit(f"the derived file holds {derived_rows} rows and the earlier holdout starts at {earlier_start}")

    holdout_fraction = args.outer_rows / derived_rows
    outer = harness.sealed_population(derived, holdout_fraction=holdout_fraction,
                                      seal_window=args.seal_window, horizon=args.horizon)
    protocol, seal = harness.build_protocol_and_seal(
        args.evaluation_src, outer["rows"], outer["labels"], data_path=args.derived_csv,
        sealed_at=args.sealed_at, seal_window=args.seal_window, horizon=args.horizon,
        holdout_fraction=holdout_fraction, minimum_rows=args.minimum_rows)

    # disjointness, checked rather than asserted in prose: no outer origin is an origin of the earlier seal, and no
    # row the outer population reads (its history and its horizon) reaches into the earlier holdout.
    outer_origins = [int(value) for value in outer["origins"]]
    overlap = sorted(set(outer_origins) & earlier_origins)
    highest_row_read = max(outer_origins) + args.horizon
    if overlap or highest_row_read >= earlier_start:
        raise SystemExit(f"REFUSED: the outer block is not disjoint from the earlier seal "
                         f"({len(overlap)} shared origins; highest row read {highest_row_read}, earlier holdout "
                         f"starts at {earlier_start})")
    if args.earlier_seal and not (harness.build_protocol_and_seal(
            args.evaluation_src, earlier["rows"], earlier["labels"], data_path=args.data,
            sealed_at="2026-09-25T00:00:00Z", seal_window=args.seal_window, horizon=args.horizon,
            holdout_fraction=args.earlier_holdout_fraction, minimum_rows=args.minimum_rows)[1]
            .seal.startswith(args.earlier_seal)):
        raise SystemExit("REFUSED: the earlier round's seal does not reproduce from this file and this rule; the split "
                         "would be declared against a population that is not the one it claims to be disjoint from")

    inner_rows = derived_rows - args.outer_rows
    stamps = derived["stamps"]
    frozen = {
        "schema": SCHEMA,
        "frozen_at": args.sealed_at,
        "why": ("WP26: the winner of the 2026-09-25 search was selected ON the population it was scored on. This split "
                "declares an outer holdout that ranked nothing, so the winner can be scored once on rows no search "
                "read as an objective."),
        "rule": {
            "sentence": (f"Take the source slice's rows before the earlier round's holdout begins "
                         f"(rows 0 .. {earlier_start - 1}); the INNER region is the first {inner_rows} of them and the "
                         f"OUTER holdout is the last {args.outer_rows}. The outer population is every forecast origin "
                         f"inside the outer block whose whole {args.seal_window}-row sealing window and whole "
                         f"{args.horizon}-row horizon lie inside it -- the same population rule, sealing window and "
                         f"horizon the earlier round declared, applied to a disjoint block."),
            "outer_is_not_the_final_block": (
                f"The earlier round's seal consumes the tail (source rows {earlier_start} .. {source_rows - 1}), so the "
                f"final block by time is not available. The outer block is the last contiguous block BEFORE it."),
            "what_that_costs": [
                (f"a different week: the outer block runs {stamps.iloc[inner_rows].isoformat()} .. "
                 f"{stamps.iloc[-1].isoformat()}, one week earlier than the population the published 4.50 % was "
                 f"measured on; nothing here shows the two weeks are the same regime"),
                (f"less training data: the inner region holds {inner_rows} rows against the {earlier_start} rows the "
                 f"earlier fits trained on, {100 * (1 - inner_rows / earlier_start):.0f} % fewer, so every number "
                 f"confirmed here is confirmed under a smaller fit than the one it confirms"),
            ],
            "residual": {
                "code": RESIDUAL,
                "sentence": ("The outer rows ranked nothing and were scored on by nothing, but the earlier round's "
                             "fits did read them as TRAINING rows. The re-fits this split governs never read them at "
                             "all -- the derived file's holdout begins where the outer block begins. A block untouched "
                             "in every sense does not exist inside this slice, and one taken from outside it would be "
                             "another file and another provenance."),
            },
        },
        "source": {"path": str(args.data), "sha256": sha256(args.data), "rows": source_rows,
                   "first_stamp": source["stamps"].iloc[0].isoformat(),
                   "last_stamp": source["stamps"].iloc[-1].isoformat()},
        "derived": {"path": str(args.derived_csv), "sha256": sha256(args.derived_csv), "rows": derived_rows,
                    "holdout_fraction": holdout_fraction,
                    "how": f"the first {earlier_start} data lines of the source file, header preserved, bytes otherwise "
                           f"unchanged"},
        "inner": {"rows": inner_rows, "row_range": [0, inner_rows - 1],
                  "first_stamp": stamps.iloc[0].isoformat(), "last_stamp": stamps.iloc[inner_rows - 1].isoformat(),
                  "used_for": "fitting, the fit's own validation split, and any search; never scored on here"},
        "outer": {"rows": args.outer_rows, "row_range": [inner_rows, derived_rows - 1],
                  "first_stamp": stamps.iloc[inner_rows].isoformat(), "last_stamp": stamps.iloc[-1].isoformat(),
                  "sealed_origins": len(outer["rows"]), "dropped_for_non_finite": outer["dropped"],
                  "first_origin": outer["rows"][0], "last_origin": outer["rows"][-1],
                  "seal": seal.seal, "seal_window": args.seal_window, "horizon": args.horizon,
                  "protocol_digest": protocol.digest, "sealed_at": args.sealed_at,
                  "minimum_rows": args.minimum_rows,
                  "scored_by": "once per fitted stage, after the fit, never during it"},
        "earlier_round": {
            "seal": None if args.earlier_seal is None else args.earlier_seal,
            "holdout_fraction": args.earlier_holdout_fraction,
            "holdout_start_row": earlier_start, "sealed_origins": len(earlier["rows"]),
            "disjoint": True,
            "checked": (f"no outer origin is an origin of the earlier seal, and the highest source row the outer "
                        f"population reads is {highest_row_read}, below the earlier holdout's first row "
                        f"{earlier_start}"),
        },
        "execution_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "outer_seal": seal.seal, "protocol": protocol.digest,
                      "outer_origins": len(outer["rows"]), "inner_rows": inner_rows,
                      "holdout_fraction": holdout_fraction}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
