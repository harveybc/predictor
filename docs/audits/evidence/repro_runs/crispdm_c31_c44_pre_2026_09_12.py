"""PRE for C31-C44 (order 2026-09-11).

CONFESSION FIRST, because it changes how this file should be read. The
order says "antes de editar, congelar" — freeze before editing — and I
did that for the recovery addendum (PRE-R) and NOT for this block. I
started correcting C31 before writing this. So the counterexamples
below are reproduced against the AUDITED BASE COMMITS in clean
detached worktrees, not against a working tree frozen before my first
keystroke:

    predictor       16fbbbfdac57e5c8ee67ea2dedf1e41817f8061e
    financial-data  cf2e408f8
    lts             1587457

The base commits are immutable and public, so the evidence is as
strong; the ORDER of my actions was wrong, and that is the defect, not
the evidence.

Every item drives the PRE code through its own public API. Private
paths are redacted to `~`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
RUNTIME = HOME / "Documents/GitHub/.runtime"
PRE_PREDICTOR = RUNTIME / "predictor-pre-c31-c44"
PRE_FINANCIAL = RUNTIME / "financial-data-pre-c31-c44"

FAILURES: list[str] = []
PY = sys.executable


def redact(v) -> str:
    return str(v).replace(str(HOME), "~")


def item(n: int, title: str) -> None:
    print(f"\n=== PRE {n}: {title} ===")


def expect(cond: bool, msg: str) -> None:
    print(f"  [{'PRE-REPRODUCED' if cond else 'NOT-REPRODUCED'}] {msg}")
    if not cond:
        FAILURES.append(f"{msg}")


#: the installed checkout, appended AFTER the PRE root so that PRE
#: MODULES win while the entry-point REGISTRY is discoverable.
#:
#: This is not a convenience: it is item 3 made operational. A PRE
#: worktree carries no `predictor.egg-info` of its own, so on its own
#: it resolves zero plugins and every subject derivation refuses before
#: reaching the defect under test. The identity depending on ANOTHER
#: checkout being installed beside it is precisely the finding.
INSTALLED_CHECKOUT = HOME / "Documents/GitHub/predictor"


def in_pre(code: str, *, root: Path = PRE_PREDICTOR,
           env: dict | None = None,
           with_registry: bool = False) -> subprocess.CompletedProcess:
    """Run a snippet against the PRE checkout, never the corrected one."""
    path = str(root)
    if with_registry:
        path = f"{root}{os.pathsep}{INSTALLED_CHECKOUT}"
    full_env = {**os.environ, "PYTHONPATH": path,
                "CUDA_VISIBLE_DEVICES": "", **(env or {})}
    return subprocess.run([PY, "-c", code], capture_output=True,
                          text=True, cwd=str(root), env=full_env,
                          timeout=600)


def world(tmp: Path, *, x_cols="DATE_TIME,SAME,f1",
          y_cols="DATE_TIME,SAME,TARGET",
          partitions=("train", "validation", "test")) -> dict:
    cfg = {"plugin": "ann", "target_column": "TARGET"}
    for role in partitions:
        (tmp / f"x_{role}.csv").write_text(
            x_cols + "\n" + ",".join(["1"] * len(x_cols.split(","))) + "\n")
        (tmp / f"y_{role}.csv").write_text(
            y_cols + "\n" + ",".join(["2"] * len(y_cols.split(","))) + "\n")
        cfg[f"x_{role}_file"] = str(tmp / f"x_{role}.csv")
        cfg[f"y_{role}_file"] = str(tmp / f"y_{role}.csv")
    return cfg


# --------------------------------------------------------------- 1
item(1, 'predictor_plugin="cnn" + plugin="ann": CNN runs, C20 records ANN')
src = (PRE_PREDICTOR / "eligibility/consumed.py").read_text()
main_src = (PRE_PREDICTOR / "app/main.py").read_text()
identity_key = '"predictor": ("plugin", "predictor.plugins")' in src
executor_key = "config.get('predictor_plugin', 'default_predictor')" \
    in main_src
print(f"  identity looks the predictor up under 'plugin' : {identity_key}")
print(f"  executor selects it with 'predictor_plugin'    : {executor_key}")
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    cfg = world(tmp)
    cfg["predictor_plugin"] = "cnn"
    cfg["plugin"] = "ann"
    out = in_pre(
        "import json,sys;"
        "sys.path.insert(0,'.');"
        "from eligibility.consumed import resolve_plugin_modules;"
        f"cfg=json.loads({json.dumps(json.dumps(cfg))});"
        "r=resolve_plugin_modules(cfg);"
        "print(json.dumps({k:v['entry_point_name'] "
        "for k,v in r.items()}))", with_registry=True)
    recorded = out.stdout.strip().splitlines()[-1] if out.stdout.strip() \
        else out.stderr.strip()[:160]
    print(f"  identity records                              : {recorded}")
expect(identity_key and executor_key and '"ann"' in recorded,
       "the executor would train CNN while the identity records ANN")

# --------------------------------------------------------------- 2
item(2, "mutating predictor_plugins/common/base.py leaves C20 unchanged")
out = in_pre(
    "import sys,json,hashlib,pathlib;"
    "sys.path.insert(0,'.');"
    "from eligibility.consumed import CONSUMING_CODE;"
    "print(json.dumps(list(CONSUMING_CODE)))")
consuming = json.loads(out.stdout.strip().splitlines()[-1])
print(f"  modules in the PRE identity: {len(consuming)}")
print(f"  covers predictor_plugins/common/base.py: "
      f"{'predictor_plugins/common/base.py' in consuming}")
expect("predictor_plugins/common/base.py" not in consuming
       and len(consuming) == 12,
       "the identity covers 12 files and not the shared modules the "
       "executed plugin imports")

# --------------------------------------------------------------- 3
item(3, "a clean checkout without predictor.egg-info fails the C20 tests")
egg = (PRE_PREDICTOR / "predictor.egg-info")
print(f"  the PRE checkout carries its own egg-info : {egg.exists()}")
out = in_pre(
    "import sys;sys.path.insert(0,'.');"
    "from importlib import metadata;"
    "eps=list(metadata.entry_points(group='predictor.plugins'));"
    "print(len(eps))")
print(f"  entry points visible from the PRE checkout: "
      f"{out.stdout.strip().splitlines()[-1] if out.stdout.strip() else 'ERROR'}")
expect(not egg.exists(),
       "the PRE checkout has NO egg-info of its own, so its plugin "
       "resolution depends on a DIFFERENT checkout being installed")

# --------------------------------------------------------------- 4
item(4, "the same column SAME on x and y produces only the x subject")
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    cfg = world(tmp)
    out = in_pre(
        "import json,sys;sys.path.insert(0,'.');"
        "from eligibility.consumed import resolve_consumed_subjects;"
        f"cfg=json.loads({json.dumps(json.dumps(cfg))});"
        f"c=resolve_consumed_subjects(cfg, repo_root='{tmp}');"
        "print(json.dumps([(s['side'],s['column'],s['subject_id']) "
        "for s in c['subjects']]))", with_registry=True)
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() \
        else out.stderr.strip()[-200:]
    try:
        subs = json.loads(line)
    except ValueError:
        subs = []
    for s in subs:
        print(f"    side={s[0]} column={s[1]} id={s[2]}")
    sides = {(s[0], s[1]) for s in subs}
expect(("x", "SAME") in sides and ("y", "SAME") not in sides,
       "the y-side subject for a column that also exists on x is "
       "simply absent")

# --------------------------------------------------------------- 5
item(5, "a y schema that differs across partitions is accepted")
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    cfg = world(tmp)
    (tmp / "y_test.csv").write_text("DATE_TIME,TARGET,EXTRA\n1,2,3\n")
    out = in_pre(
        "import json,sys;sys.path.insert(0,'.');"
        "from eligibility.consumed import resolve_consumed_subjects;"
        f"cfg=json.loads({json.dumps(json.dumps(cfg))});"
        f"c=resolve_consumed_subjects(cfg, repo_root='{tmp}');"
        "print('ACCEPTED', len(c['subjects']))", with_registry=True)
    accepted = "ACCEPTED" in out.stdout
    print(f"  outcome: {'ACCEPTED' if accepted else out.stderr.strip()[-160:]}")
expect(accepted,
       "train/validation/test may carry different y schemas and the "
       "gate says nothing")

# --------------------------------------------------------------- 6
item(6, "a FAILED run then a COMPLETE run of one campaign: the second "
        "refuses")
ce = (PRE_PREDICTOR / "olap/campaign_envelope.py").read_text()
frozen = ('"result_class": doc["result_class"]' in ce
          and '"design_sha256": str(ident["design_sha256"])' in ce
          and "already exists with a DIFFERENT identity" in ce)
has_run_dim = "dim_campaign_run" in ce
print(f"  dim_campaign freezes result_class/design/code : {frozen}")
print(f"  a per-run dimension exists                    : {has_run_dim}")
dead = HOME / ".local/share/predictor/olap_outbox/failed"
reasons = sorted(dead.glob("*.reason")) if dead.is_dir() else []
for r in reasons[:1]:
    print(f"  the live consequence on disk: "
          f"{redact(r.read_text().strip())[:120]}")
expect(frozen and not has_run_dim,
       "the campaign dimension freezes execution facts, so a second "
       "legitimate attempt collides with its own first")

# --------------------------------------------------------------- 7
item(7, "loading a NEW observation then an OLD one makes the OLD current")
inv = (PRE_PREDICTOR / "olap/inventory_rows.py").read_text()
by_load = inv.count("loaded_at DESC")
has_observed = "observed_at" in inv
print(f"  `current` views ordered by loaded_at DESC : {by_load}")
print(f"  any artifact chronology column           : {has_observed}")
expect(by_load >= 4 and not has_observed,
       "`current` means the latest LOAD, so re-importing an old census "
       "today supersedes a newer observation")

# --------------------------------------------------------------- 8
item(8, "a characterization of other bytes with the same descriptor "
        "collides")
ch = (PRE_PREDICTOR / "olap/characterization.py").read_text()
start = ch.index('row["observation_sha256"] = _sha({')
body = ch[start:start + 400]
print("  the PRE observation identity is built from:")
for key in ("variable_id", "partition_key", "descriptor", "value_text",
            "contract", "bank"):
    print(f"    {key}: {f'\"{key}\"' in body}")
for absent in ("source_sha256", "window", "code_identity"):
    print(f"    {absent}: {absent in body}")
expect("source_sha256" not in body and "window" not in body,
       "the identity omits the data, so two different files with the "
       "same descriptive number are ONE observation")

# --------------------------------------------------------------- 9
item(9, "an intermediate NaN removes a position and changes lag 1")
out = in_pre(
    "import sys;sys.path.insert(0,'.');"
    "import numpy as np;"
    "from olap.characterization import characterize_series as cs;"
    "t=np.arange(400.0);clean=np.sin(2*np.pi*t/20.0);"
    "holed=clean.copy();holed[200]=np.nan;"
    "f=lambda v: {r['descriptor']: r['value'] for r in cs(v,"
    "variable_id='v',partition_key='p',"
    "bank_authority='SYNTHETIC_KNOWN_MECHANISM_CALIBRATION_ONLY',"
    "measured_at='2026-09-12T00:00:00Z')};"
    "a=f(clean);b=f(holed);"
    "print(a['autocorrelation_lag1'], b['autocorrelation_lag1'],"
    " a['spectral_centroid'], b['spectral_centroid'])")
line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() \
    else out.stderr.strip()[-200:]
print(f"  clean acf1, holed acf1, clean spectrum, holed spectrum:")
print(f"    {line}")
try:
    parts = [float(x) for x in line.split()]
    shifted = abs(parts[0] - parts[1]) > 1e-9
    spectrum_computed = parts[3] is not None
except (ValueError, IndexError):
    shifted, spectrum_computed = False, False
expect(shifted and spectrum_computed,
       "one NaN shifts the reported lag-1 autocorrelation AND a "
       "spectrum is still computed over a compacted axis")

# --------------------------------------------------------------- 10
item(10, "DATE_TIME enters C30 as a numeric variable")
runner = (PRE_PREDICTOR / "tools/run_characterization.py").read_text()
by_position = "header[:limit]" in runner
excludes = "NON_FEATURE_COLUMNS" in runner or "DATE_TIME" in runner
print(f"  columns taken by POSITION (header[:limit]) : {by_position}")
print(f"  any temporal exclusion in the driver       : {excludes}")
expect(by_position and not excludes,
       "the first N columns are measured by position, so a timestamp "
       "gets a mean, an entropy and a spectrum")

# --------------------------------------------------------------- 11
item(11, "an outbox failure leaves no gap file: results_dir was None")
main_pre = (PRE_PREDICTOR / "app/main.py").read_text()
eager = 'results_dir=shared.get("results_dir")' in main_pre
lazy = "results_dir=lambda:" in main_pre
sets_later = 'shared["results_dir"]' in main_pre
print(f"  results_dir read EAGERLY at the call site : {eager}")
print(f"  read lazily                               : {lazy}")
print(f"  the run sets it LATER in the body         : {sets_later}")
term = (PRE_PREDICTOR / "olap/terminal.py").read_text()
fixed_name = 'OPERATIONAL_GAP_NAME = "OLAP_OUTBOX_OPERATIONAL_GAP.json"' \
    in term
print(f"  the gap file has ONE fixed name           : {fixed_name}")
expect(eager and not lazy and sets_later and fixed_name,
       "results_dir is captured as None before the body runs, so an "
       "outbox failure writes the gap nowhere; and the fixed filename "
       "would let a second run erase the first")

# --------------------------------------------------------------- 12
item(12, "the heartbeat stays healthy=false on a known dead-letter and "
         "does not distinguish it from a dead loader")
ob = (PRE_PREDICTOR / "olap/outbox.py").read_text()
naive = '"healthy": c[FAILED] == 0,' in ob
has_states = "DEAD_LETTER_UNADJUDICATED" in ob
has_freshness = "process_fresh" in ob
print(f"  healthy is literally `failed == 0`  : {naive}")
print(f"  dead-letter adjudication exists     : {has_states}")
print(f"  process freshness is measured       : {has_freshness}")
expect(naive and not has_states and not has_freshness,
       "one historical refusal makes the alarm red forever, and it is "
       "the SAME red a stopped loader would show")

# ---------------------------------------------------------------
print(f"\n=== PRE SUMMARY: 12 items; {len(FAILURES)} NOT reproduced ===")
for f in FAILURES:
    print(f"  NOT-REPRODUCED: {f}")
print("Reproduced against the audited base commits in clean detached "
      "worktrees; see the confession at the top of this file.")
sys.exit(1 if FAILURES else 0)
