"""The search's own decision record (WP06 stage 5 + the `chosen_by: SEARCH` clause of WP23).

One record, for the search itself. Its option set is every representation the search actually *evaluated* -- a point it
refused is not an option, because no fit of it exists and nothing could have chosen it -- and its choice is the one
whose held-out MAE was smallest. It carries no probabilities, because a search produced no distribution, and its `why`
names the objective and the budget, without which the winner is the winner of an unnamed contest.

Two rounds of one search are one search: the second drew from a declared sub-range of the same space, with the same
objective and the same seal, and its points are options of the same choice.

Usage: search_decision.py <decisions dir> <out index.json> <ledger.json> [<ledger.json> ...]
"""
import json
import sys
from pathlib import Path

from m5phet import decide

AS_OF = "2026-09-25T12:00:00+00:00"

decisions_dir = Path(sys.argv[1]).expanduser()
index_path = Path(sys.argv[2])
ledgers = [json.loads(Path(argument).read_text()) for argument in sys.argv[3:]]
ledger = ledgers[0]
evaluations = [entry for book in ledgers for entry in book["evaluations"]]

measured = sorted((entry for entry in evaluations if entry["status"] == "OK"), key=lambda entry: entry["mae"])
refused = [entry for entry in evaluations if entry["status"] != "OK"]
if len(measured) < 2:
    raise SystemExit(f"a choice needs at least two evaluated points; the ledger carries {len(measured)}")


def label(entry):
    point = entry["point"]
    return (f"window {point['window']}, lags {point['lags']}, transform {point['transform']}, differencing order "
            f"{point['differencing_order']}, columns {', '.join(point['features'])}")


options = [[entry["candidate_id"], label(entry)] for entry in sorted(measured, key=lambda e: e["candidate_id"])]
# The tie rule is declared, not left to the sort: two representations can produce the same graph under this
# single-block harness -- the lag list decides nothing beyond `max(lag) <= window`, so two specs differing only in
# their lags are fitted identically -- and a winner picked by whichever happened to sort first would be a choice
# nobody made. Ties are broken by the smallest candidate id and the tie is written into the record's `why`.
lowest = measured[0]["mae"]
tied = sorted((entry for entry in measured if entry["mae"] == lowest), key=lambda e: e["candidate_id"])
best = tied[0]
tie_sentence = ("" if len(tied) == 1 else
                f" {len(tied)} points tied at this error ({', '.join(entry['candidate_id'] for entry in tied)}); the "
                f"tie is broken by the smallest candidate id, and it exists because this single-block harness fits one "
                f"contiguous window, so two representations differing only in their declared lags are the same graph.")

state = decide.decision_state("representation_search", {
    "dataset": Path(ledger["data"]).name,
    "seal": "33820b552ddf",
    "protocol": "d0ebd9a4bc75",
    "sealed_rows": 9824,
    "target": "Global_active_power",
    "horizon_steps": 60,
    "sealing_window": 197,
    "space": "window 3..197, lags from the design job's motivated set, transform, differencing order, column subset",
    "evaluated": len(measured),
    "refused": len(refused),
    "rounds": len(ledgers),
})

entry = decide.search_choice(
    kind="representation",
    question="searched_candidate",
    options=options,
    chosen=best["candidate_id"],
    state_text=state,
    objective=ledger["objective"],
    budget=(" ".join(f"Round {number}: {book['budget']['sentence']} Windows "
                     f"{book['space']['window']['low']}..{book['space']['window']['high']}, "
                     f"{book['dispatched_fits']} fits dispatched."
                     for number, book in enumerate(ledgers, start=1))
            + f" Over the rounds together {len(measured)} points were measured and {len(refused)} were refused "
              f"by name"),
    why=("a genetic search over the declared representation space (DEAP, the operator set "
         "optimizer_plugins/default_optimizer.py already uses in this repository) fitted every legal point once "
         "through tools/fit_pipeline_spec.py, with everything but the representation held at baseline_hand's own "
         "configuration and with the identical seal asserted before each fit, and kept the point whose held-out error "
         "was smallest. Nothing was tuned to make a point win and no refused point was moved to a legal one." +
         tie_sentence),
    as_of=AS_OF,
    record_dir=decisions_dir)

if entry["status"] != "OK":
    raise SystemExit(json.dumps(entry, indent=2, sort_keys=True))

index = {"record": Path(entry["record_path"]).stem,
         "kind": entry["decision"]["kind"], "question": entry["decision"]["question"],
         "chosen": entry["decision"]["chosen"], "chosen_by": entry["decision"]["chosen_by"],
         "stage": best["stage"], "mae": best["mae"], "options": len(options), "tied": len(tied),
         "state_sha256": entry["decision"]["state_sha256"],
         "decisions_dir": str(decisions_dir)}
index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
print(json.dumps(index, indent=2, sort_keys=True))
