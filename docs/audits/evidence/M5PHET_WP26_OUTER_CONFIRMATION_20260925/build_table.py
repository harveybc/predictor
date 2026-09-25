"""Build the WP26 outer-seal closure table from the twenty reports the confirmation run wrote.

Every row is one (stage, seed) fit scored on the frozen OUTER seal. They are twenty rows and not four because a mean
is not a measurement: the table shows what each fit measured, and `confirmation.json` beside it carries the mean, the
standard deviation and the paired interval over the seeds.

Usage: build_table.py <evidence dir>
"""
import json
import subprocess
import sys
from pathlib import Path

M5PHET = Path("/home/harveybc/Documents/GitHub/M5PHET")

evidence = Path(sys.argv[1]).resolve()
confirmation = json.loads((evidence / "confirmation.json").read_text())
table_dir = evidence / "table"
table_dir.mkdir(parents=True, exist_ok=True)

reports = {}
for stage, seeds in sorted(confirmation["runs"].items()):
    for seed, record in sorted(seeds.items(), key=lambda item: int(item[0])):
        if record["status"] != "OK":
            continue
        reports[f"{stage}__seed{seed}"] = evidence / "stages" / f"{stage}__seed{seed}" / "report.json"

command = [sys.executable, "-m", "evaluation.compare_stages"]
for stage, path in sorted(reports.items()):
    command += ["--report", f"{stage}={path}"]
command += ["--out", str(table_dir / "table.json"), "--markdown", str(table_dir / "table.md")]

finished = subprocess.run(command, cwd=M5PHET, capture_output=True, text=True)
if finished.returncode != 0:
    sys.stderr.write(finished.stdout + finished.stderr)
    raise SystemExit(finished.returncode)

table = json.loads((table_dir / "table.json").read_text())
rows = [row for area in table["areas"] if area["area"] == "forecast" for row in area["rows"]]
ranked = sorted((row for row in rows if row["status"] == "MEASURED"), key=lambda row: row["rank"])
print(json.dumps({
    "stages": len(rows), "measured": len(ranked),
    "rank_1": {"stage": ranked[0]["stage"], "model_error": ranked[0]["model_error"]},
    "best_baseline_hand_rank": min(row["rank"] for row in ranked if row["stage"].startswith("baseline_hand")),
    "worst_baseline_hand_rank": max(row["rank"] for row in ranked if row["stage"].startswith("baseline_hand")),
    "comparable": sorted({row["comparability"].split(":")[0] for row in ranked}),
    "table": str(table_dir / "table.md")}, indent=1))
