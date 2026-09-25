"""Link every stage's decision records to ITS row of the WP06 stage-5 closure table (WP23 step 1).

Four sources now, one rule. Laya's ten records go to the `laya_chosen` row; the human records written for the stages
a person configured go to the rows of the stages they describe; the records the rank-1 searched stage carries for what a PERSON held fixed in
it go to its row; and the search's own record goes to the row of the stage it chose. Nothing else is linked: a record that does not belong to a stage in this table has no row, and a stage
whose row is not COMPARABLE and ranked is refused by `outcome` itself.

The outcomes go into a directory of their own, for the same reason the previous round's did: the earlier outcomes bind
rows of a table this one supersedes, and merging them would count the same decisions twice under two different ranks.

Laya's ten records are named by the PREVIOUS round's link index rather than by a pipeline spec: the `laya_chosen`
spec is not among the artefacts kept in this repository, and the round that linked those records kept every digest it
linked. Reading them from there binds this round to the same ten records, which is the point.

Usage: link_outcomes.py <table.json> <human index> <previous link_outcomes.json> <search index> <winner index>
       <out dir> <summary.json>
"""
import json
import sys
from pathlib import Path

from m5phet.decide import outcome

table = json.loads(Path(sys.argv[1]).read_text())
human_index = json.loads(Path(sys.argv[2]).read_text())
previous = json.loads(Path(sys.argv[3]).read_text())
search_index = json.loads(Path(sys.argv[4]).read_text())
winner_index = json.loads(Path(sys.argv[5]).read_text())
out_dir = Path(sys.argv[6]).expanduser()
LAYA_DECISIONS = Path("~/.local/state/m5phet/decisions").expanduser()
HUMAN_DECISIONS = Path("~/.local/state/m5phet/decisions-wp06-stage34-20260925").expanduser()

rows = {row["stage"]: row for area in table["areas"] if area["area"] == "forecast" for row in area["rows"]}
summary = {"linked": [], "refused": []}


def link(path, stage, chooser, kind, question):
    if stage not in rows:
        summary["refused"].append({"record": Path(path).stem, "stage": stage, "chosen_by": chooser, "kind": kind,
                                   "question": question, "status": "REFUSED", "refusal": "STAGE_NOT_IN_TABLE",
                                   "why": f"no row of this table carries the stage {stage!r}"})
        return
    result = outcome(str(path), rows[stage], out_dir=str(out_dir))
    entry = {"record": Path(path).stem, "stage": stage, "chosen_by": chooser, "kind": kind, "question": question,
             "status": result["status"]}
    if result["status"] == "OK":
        entry["outcome"] = Path(result["record_path"]).stem
        entry["rank"] = result["outcome"]["rank"]
        entry["best_ranked_option"] = result["outcome"].get("best_ranked_option")
        summary["linked"].append(entry)
    else:
        entry.update({"refusal": result.get("refusal"), "why": result.get("why")})
        summary["refused"].append(entry)


laya_records = sorted({entry["record"] for entry in previous["linked"] + previous["refused"]
                       if entry["chosen_by"] == "LAYA"})
for digest in laya_records:
    record = json.loads((LAYA_DECISIONS / f"{digest}.json").read_text())
    link(LAYA_DECISIONS / f"{digest}.json", "laya_chosen", "LAYA", record["kind"], record["question"])

for item in sorted(human_index["written"], key=lambda entry: (entry["stage"], entry["kind"], entry["record"])):
    link(HUMAN_DECISIONS / f"{item['record']}.json", item["stage"], "HUMAN", item["kind"], item["question"])

WINNER_DECISIONS = Path("~/.local/state/m5phet/decisions-wp06-stage5-20260925").expanduser()
for item in sorted(winner_index["written"], key=lambda entry: (entry["kind"], entry["record"])):
    link(WINNER_DECISIONS / f"{item['record']}.json", winner_index["stage"], "HUMAN", item["kind"], item["question"])

search_dir = Path(search_index["decisions_dir"]).expanduser()
link(search_dir / f"{search_index['record']}.json", search_index["stage"], "SEARCH",
     search_index["kind"], search_index["question"])

Path(sys.argv[7]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"linked": len(summary["linked"]), "refused": len(summary["refused"]),
                  "refusals": sorted({(entry["stage"], entry["refusal"]) for entry in summary["refused"]}),
                  "by_chooser": {chooser: sum(1 for entry in summary["linked"] if entry["chosen_by"] == chooser)
                                 for chooser in ("LAYA", "HUMAN", "SEARCH")}},
                 default=list, indent=1))
