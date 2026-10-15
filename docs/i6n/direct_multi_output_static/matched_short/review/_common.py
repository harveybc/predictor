"""Shared helpers for the Retsu review probes (no fits, no TEST, CPU only).

Paths are resolved at run time and REDACTED in every committed output:
  $REPO        this review worktree
  $POPULATION  $EXPERIMENT_STORAGE_ROOT/i6dmo/matched-short-20261015/population (read only)
  $INPUT       the retained input directory, resolved by path from the closed campaigns'
               existing private deployment spec (coordinator role data_root + the sealed relpath)
  $SCRATCH     /var/tmp scratch for copies and outputs that never enter git
"""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("RETSU_CODE_ROOT") or Path(__file__).resolve().parents[5])
REVIEW = Path(__file__).resolve().parent
OUT = REVIEW / "out"
MATCHED = REPO / "docs/i6n/direct_multi_output_static/matched_short"
_OVERRIDE_POP, _OVERRIDE_INPUT = os.environ.get("RETSU_POPULATION"), os.environ.get("RETSU_INPUT")
SCRATCH = Path(os.environ.get("RETSU_SCRATCH", "/var/tmp/retsu-review-scratch-20261015"))
SEAL = json.loads((MATCHED / "POPULATION_SEAL.json").read_text())
DIGEST = SEAL["population_digest"]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def storage_root():
    if _OVERRIDE_POP:
        return Path(_OVERRIDE_POP).parent
    root = os.environ.get("EXPERIMENT_STORAGE_ROOT")
    if not root:
        conf = Path.home() / ".config/environment.d/90-experiment-storage.conf"
        for line in conf.read_text().splitlines():
            if line.startswith("EXPERIMENT_STORAGE_ROOT="):
                root = line.split("=", 1)[1].strip().strip('"')
    return Path(root)


POPULATION = Path(_OVERRIDE_POP) if _OVERRIDE_POP else \
    storage_root() / SEAL["private_location"].split("$EXPERIMENT_STORAGE_ROOT/", 1)[1]
_SPEC = Path(os.environ.get("RETSU_CLOSED_SPEC",
                            "/var/tmp/i6dmo/campaigns/market-time-report-recovery-20261015/private/deployment.json"))
_EFF = _SPEC.parent.parent / "EFFECTIVE_CONFIG.json"


def input_dir():
    if _OVERRIDE_INPUT:
        return Path(_OVERRIDE_INPUT)
    spec = json.loads(_SPEC.read_text())
    eff = json.loads(_EFF.read_text())
    return Path(spec["roles"]["coordinator"]["data_root"]) / eff["data_pins"]["short_input"]["relpath"]


INPUT = input_dir()
DESIGNS = {arm: MATCHED / f"EXPERIMENT.{arm}.json" for arm in ("mg1", "mg2m")}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def redact(obj):
    text = json.dumps(obj, sort_keys=True, indent=1, default=str)
    for real, ph in ((str(POPULATION), "$POPULATION"), (str(INPUT), "$INPUT"), (str(SCRATCH), "$SCRATCH"),
                     (str(REPO), "$REPO"), (str(storage_root()), "$EXPERIMENT_STORAGE_ROOT"),
                     (str(Path.home()), "$HOME")):
        text = text.replace(real, ph)
    return text


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(redact(obj) + "\n")
    print(redact({"wrote": f"out/{name}"}))
