"""Build the WP06 stage-5 closure table over EVERY stage that exists: the six measured before this round, the two
that were attempted and refused, the control refit of `baseline_hand` under this round's harness, and every point the
search evaluated.

The reasons the two refused candidates carry are read from the previous round's table, not retyped: a refusal that is
paraphrased on its second printing is a different refusal.

Usage: build_table.py <previous table.json> <out dir> <control out-dir> <search out-dir> [<search out-dir> ...]
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREDICTOR = HERE.parents[3]
M5PHET = Path("/home/harveybc/Documents/GitHub/.worktrees/m5phet-wp06-search")
EVIDENCE = PREDICTOR / "docs" / "audits" / "evidence"

#: the stages measured before this round, with the report each one is read from
EARLIER = {
    "baseline_hand": EVIDENCE / "M5PHET_WP18_STEP7_20260925" / "baseline_hand" / "report.json",
    "laya_chosen": EVIDENCE / "M5PHET_WP18_STEP7_20260925" / "laya_chosen" / "report.json",
    "quantile_hand": EVIDENCE / "M5PHET_WP18_STEP7_20260925" / "quantile_hand" / "report.json",
    "quantile_hand_95": EVIDENCE / "M5PHET_WP06_STAGES34_20260925" / "stages" / "quantile_hand_95" / "report.json",
    "candidate_seasonal_lag_74": (EVIDENCE / "M5PHET_WP06_STAGES34_20260925" / "stages"
                                  / "candidate_seasonal_lag_74" / "report.json"),
    "candidate_short_memory": (EVIDENCE / "M5PHET_WP06_STAGES34_20260925" / "stages"
                               / "candidate_short_memory" / "report.json"),
}

previous = json.loads(Path(sys.argv[1]).read_text())
out_dir = Path(sys.argv[2])
control_dir = Path(sys.argv[3])
search_dirs = [Path(argument) for argument in sys.argv[4:]]
out_dir.mkdir(parents=True, exist_ok=True)

not_measured = []
for area in previous["areas"]:
    for row in area["rows"]:
        if row["status"] == "NO_NEW_MEASUREMENT":
            not_measured.append((row["stage"], area["area"], row["reason"]))

reports = dict(EARLIER)
reports["control_baseline_hand_refit"] = control_dir / "report.json"
for search_dir in search_dirs:
    ledger = json.loads((search_dir / "ledger.json").read_text())
    for entry in ledger["evaluations"]:
        if entry["status"] == "OK":
            reports[entry["stage"]] = search_dir / "stages" / entry["stage"] / "report.json"

command = [sys.executable, "-m", "evaluation.compare_stages"]
for stage, path in sorted(reports.items()):
    command += ["--report", f"{stage}={path}"]
for stage, area, reason in not_measured:
    command += ["--not-measured", f"{stage}={area}:{reason}"]
command += ["--out", str(out_dir / "table.json"), "--markdown", str(out_dir / "table.md")]

finished = subprocess.run(command, cwd=M5PHET, capture_output=True, text=True)
if finished.returncode != 0:
    sys.stderr.write(finished.stdout + finished.stderr)
    raise SystemExit(finished.returncode)

table = json.loads((out_dir / "table.json").read_text())
rows = [row for area in table["areas"] if area["area"] == "forecast" for row in area["rows"]]
ranked = sorted((row for row in rows if row["status"] == "MEASURED"), key=lambda row: row["rank"])
print(json.dumps({"stages": len(rows), "measured": len(ranked), "not_measured": len(not_measured),
                  "rank_1": {"stage": ranked[0]["stage"], "model_error": ranked[0]["model_error"]},
                  "baseline_hand_rank": next(row["rank"] for row in rows if row["stage"] == "baseline_hand"),
                  "best_searched": next(({"stage": row["stage"], "rank": row["rank"],
                                          "model_error": row["model_error"]}
                                         for row in ranked if row["stage"].startswith("searched_")), None),
                  "table": str(out_dir / "table.md")}, indent=1))
