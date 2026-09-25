"""Link the in-spec decision records to the laya_chosen closure-table row (WP23 step 1)."""
import json, sys
from pathlib import Path
from m5phet.decide import outcome

table = json.loads(Path(sys.argv[1]).read_text())
spec = json.loads(Path(sys.argv[2]).read_text())
decisions = Path(sys.argv[3])
out_dir = Path(sys.argv[4])
KINDS = {"feature_preprocessing", "feature_grouping", "group_extractor"}

rows = {r["stage"]: r for area in table["areas"] if area["area"] == "forecast" for r in area["rows"]}
in_spec = set(spec["decisions"])
summary = {"linked": [], "refused": [], "skipped_not_in_spec": []}
for path in sorted(decisions.glob("*.json")):
    record = json.loads(path.read_text())
    if record.get("kind") not in KINDS:
        continue
    if path.stem not in in_spec:
        summary["skipped_not_in_spec"].append({"record": path.stem, "kind": record["kind"],
                                               "chosen": record["chosen"]})
        continue
    result = outcome(str(path), rows["laya_chosen"], out_dir=str(out_dir))
    entry = {"record": path.stem, "kind": record["kind"], "question": record["question"],
             "chosen": record["chosen"], "stage": "laya_chosen", "status": result["status"]}
    if result["status"] == "OK":
        entry["outcome"] = Path(result["record_path"]).stem
        summary["linked"].append(entry)
    else:
        entry["refusal"] = result.get("refusal")
        entry["why"] = result.get("why")
        summary["refused"].append(entry)
print(json.dumps(summary, indent=2, sort_keys=True))
