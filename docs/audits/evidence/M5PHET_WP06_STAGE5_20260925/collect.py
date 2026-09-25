"""Copy what a reader needs out of the run directory, and digest what is left behind.

Per stage: its evaluation report, the predictor configuration the spec implied, the run manifest (which carries the
representation BY VALUE) and the training history. Not copied: `predictions.csv` (9824 rows per stage) and the fitted
graph, which stay in the run directory; their digests are written here so the copy can be checked against them.

The seal and the protocol are identical for every stage of this comparison -- that is what makes it one comparison --
so one copy of each is kept at the top level instead of sixty identical ones.

Usage: collect.py <control out-dir> <evidence dir> <search out-dir> [<search out-dir> ...]
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

control_dir = Path(sys.argv[1])
evidence = Path(sys.argv[2])
search_dirs = [Path(argument) for argument in sys.argv[3:]]
(evidence / "stages").mkdir(parents=True, exist_ok=True)

KEPT = ("report.json", "config.json", "history.json")
DIGESTED = ("predictions.csv", "seal.json", "protocol.json", "fitted/model.keras", "fitted/fit_manifest.json")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def collect(run_dir, name):
    target = evidence / "stages" / name
    target.mkdir(parents=True, exist_ok=True)
    for filename in KEPT:
        shutil.copy2(run_dir / filename, target / filename)
    shutil.copy2(run_dir / "fitted" / "fit_manifest.json", target / "fit_manifest.json")
    lines = []
    for filename in DIGESTED:
        path = run_dir / filename
        if path.exists():
            lines.append(f"{digest(path)}  {filename}")
    (target / "artifact_digests.sha256").write_text("\n".join(sorted(lines)) + "\n")
    return name


collected = [collect(control_dir, "control_baseline_hand_refit")]
for search_dir in search_dirs:
    ledger = json.loads((search_dir / "ledger.json").read_text())
    for entry in ledger["evaluations"]:
        if entry["status"] == "OK":
            collected.append(collect(search_dir / "stages" / entry["stage"], entry["stage"]))
    shutil.copy2(search_dir / "ledger.json", evidence / f"ledger_{search_dir.name}.json")
    shutil.copytree(search_dir / "specs", evidence / "specs", dirs_exist_ok=True)

shutil.copy2(control_dir / "seal.json", evidence / "seal.json")
shutil.copy2(control_dir / "protocol.json", evidence / "protocol.json")
print(json.dumps({"stages_collected": len(collected), "seal": json.loads((evidence / "seal.json").read_text())["seal"]},
                 indent=1))
