"""Link every stage's decision records to ITS row of the WP06 stage-3/4 closure table (WP23 step 1).

Two sources, one rule. Laya's records are the ten the `laya_chosen` pipeline spec names, linked to the `laya_chosen`
row. The human records are the ones written for the stages a person configured, linked to the row of the stage they
describe. Nothing else is linked: a record that does not belong to a stage in this table has no row, and a stage whose
row is not COMPARABLE and ranked is refused by `outcome` itself.

The outcomes go into a directory of their own. The previous round's outcomes bind rows of a two-stage table that this
one supersedes; merging them would count the same decisions twice under two different ranks.
"""
import json
import sys
from pathlib import Path

from m5phet.decide import outcome

table = json.loads(Path(sys.argv[1]).read_text())
human_index = json.loads(Path(sys.argv[2]).read_text())
laya_spec = json.loads(Path(sys.argv[3]).read_text())
out_dir = Path(sys.argv[4]).expanduser()
LAYA_DECISIONS = Path("~/.local/state/m5phet/decisions").expanduser()
HUMAN_DECISIONS = Path("~/.local/state/m5phet/decisions-wp06-stage34-20260925").expanduser()

rows = {row["stage"]: row for area in table["areas"] if area["area"] == "forecast" for row in area["rows"]}
summary = {"linked": [], "refused": []}


def link(path, stage, chooser, kind, question):
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


for digest in sorted(laya_spec["decisions"]):
    record = json.loads((LAYA_DECISIONS / f"{digest}.json").read_text())
    link(LAYA_DECISIONS / f"{digest}.json", "laya_chosen", "LAYA", record["kind"], record["question"])

for item in sorted(human_index["written"], key=lambda entry: (entry["stage"], entry["kind"], entry["record"])):
    link(HUMAN_DECISIONS / f"{item['record']}.json", item["stage"], "HUMAN", item["kind"], item["question"])

Path(sys.argv[5]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({"linked": len(summary["linked"]), "refused": len(summary["refused"]),
                  "refusals": sorted({(entry["stage"], entry["refusal"]) for entry in summary["refused"]})},
                 default=list, indent=1))
