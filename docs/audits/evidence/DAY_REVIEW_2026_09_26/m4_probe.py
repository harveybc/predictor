"""Read-only M4 audit reproducer, 2026-09-26. NOT an approval or a run record.

Reads exact Git blobs; extracts unchanged function ASTs without importing the
runner or executing its CLI. All run documents are synthetic and IN MEMORY.
No training, generator construction, real authority records, or filesystem
writes. Only NumPy/SciPy statistics run. Use an address-space cap of 512 MiB.

env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= prlimit --as=536870912 --cpu=30 -- \
  python3 -B /tmp/m4_readonly_audit_0de54534.py
"""
import ast
import copy
import fnmatch
import hashlib
import json
import resource
import subprocess
import time
from pathlib import PurePosixPath
from types import SimpleNamespace as NS

import numpy as np
from scipy import stats

REPO = "/home/harveybc/Documents/GitHub/agent-multi"
REV = "0de54534"
FILES = {}
DIRS = set()


def blob(path):
    return subprocess.check_output(
        ["git", "-C", REPO, "show", REV + ":" + path], text=True)


class MemoryPath(PurePosixPath):
    def is_file(self):
        return str(self) in FILES

    def exists(self):
        return self.is_file() or str(self) in DIRS

    def read_text(self):
        return FILES[str(self)]

    def mkdir(self, **kwargs):
        DIRS.add(str(self))

    def glob(self, pattern):
        return [MemoryPath(p) for p in sorted(FILES)
                if fnmatch.fnmatch(p, str(self / pattern))]


def put(path, doc):
    FILES[str(path)] = json.dumps(doc)


def extract(path, names, env):
    tree = ast.parse(blob(path), filename=path)
    nodes = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 REV + ":" + path, "exec"), env)
    return env


base = dict(Path=MemoryPath, hashlib=hashlib, json=json, np=np,
            stats=stats, time=time, REPO=REPO)
pe = extract("tools/m4_confirmation_protocol.py", {
    "_selfsha", "holm", "_contrast_widths", "sixteen_contrasts",
    "width_heterogeneity", "ConfirmationProtocolRefusal"},
    dict(base, CONFIRMATION_PER_SLOT=48))
cp = NS(**{k: v for k, v in pe.items() if not k.startswith("__")})
me = extract("tools/m4_residual_capacity.py",
             {"_self_sha", "_strict_json_file", "M4Refusal"}, dict(base))
m4 = NS(**{k: v for k, v in me.items() if not k.startswith("__")})
successor = json.loads(blob(
    "docs/research/model_capacity/M4_CONFIRMATION_SUCCESSOR_2026_09_10.json"))
design = json.loads(blob(
    "docs/research/model_capacity/M4_SEALED_DESIGN_V5_2026_09_09.json"))


def iv_unit(role, fam, nz, width, gi, seed):
    return dict(unit_id=f"intervention::{role}::{fam}::{nz}::w{width}::g{gi}::s{seed}",
                role=role, family=fam, noise_coord=nz, width=width,
                generator_index=gi, model_seed=seed,
                generator_id=f"{role}-{fam}-{nz}-g{gi}")


rn = NS(_iv_unit=iv_unit, _safe=lambda s: s.replace("::", "__"))
ce = extract("tools/m4_confirmation_runner.py", {
    "ConfirmationRunnerRefusal", "_sha_bytes", "confirmation_units",
    "materialize_census", "refuse_foreign_role_record",
    "verify_confirmation_run", "unit_record_path", "read_complete_unit_record",
    "new_accounting", "execute_confirmation_units", "verify_role_disjointness",
    "execute_confirmation", "write_pre_result_ledger"},
    dict(base, cp=cp, m4=m4, rn=rn,
         CHECKPOINT_KINDS=("initialization", "calibration_stop",
                           "pre_stop", "post_stop_bounded")))
cp.bind_calibration_evidence = lambda root: {"design": design}
cp.verify_confirmation_successor = lambda root: copy.deepcopy(successor)
census = ce["materialize_census"](successor, design)
print("EXACT_CENSUS", census["units_total"], census["census_sha256"])


def world():
    FILES.clear()
    DIRS.clear()
    ledger = dict(census_sha256=census["census_sha256"], gates={})
    ledger["ledger_sha256"] = cp._selfsha(ledger, "ledger_sha256")
    put("/virtual/CONFIRMATION_PRE_RESULT_LEDGER.json", ledger)


def summary(fam, nz, width, gi, seed, effect, alias=None):
    u = iv_unit("CONFIRMATION", fam, nz, width, gi, seed)
    rec = dict(unit_id=u["unit_id"], arms={
        "initialization": {"restricted_endpoint": 0, "updates_done": 0},
        "calibration_stop": {"restricted_endpoint": effect, "updates_done": 400}},
        paired_primary_difference=effect)
    name = alias or rn._safe(u["unit_id"])
    put(f"/virtual/intervention/{name}_summary.json", rec)


def verify():
    return ce["verify_confirmation_run"](REPO, "/virtual", successor)


world()
for width in (16, 64):
    for gi in range(39):
        for seed in range(3):
            summary("sine", "clean", width, gi, seed, 8 + gi % 5)
r = verify()
v = r["analysis"]["contrasts"]["intervention_effect::sine::clean"]
assert v["reject_at_alpha"] and r["records_verified"] == 234
print("F1_SUMMARIES_WITHOUT_RAW_LOGS_OR_AUTHORITY",
      json.dumps(dict(records_verified=r["records_verified"],
                      n_generators=v["n_generators"],
                      p_holm=v["p_holm"], reject=v["reject_at_alpha"],
                      raw_logs=0, record_hashes=0, gates={})))

world()
for width in (16, 64):
    for gi in range(9000, 9039):
        for copy_id in range(3):
            summary("sine", "clean", width, gi, 0, 8 + gi % 5,
                    alias=f"w{width}_g{gi}_copy{copy_id}")
r = verify()
v = r["analysis"]["contrasts"]["intervention_effect::sine::clean"]
assert v["n_generators"] == 39 and v["reject_at_alpha"]
print("F2_DUPLICATE_S0_AND_OUT_OF_CENSUS_G9000",
      json.dumps(dict(n_generators=v["n_generators"], reject=v["reject_at_alpha"],
                      unique_seeds=1, gi_min=9000)))

world()
for gi in range(30):
    for seed in range(3):
        summary("sine", "clean", 16, gi, seed, 8 + gi % 5)
r = verify()
v = r["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
assert len(r["confirmation_incomplete"]) == 21 and v["reject_at_alpha"]
print("F3_CHECKPOINT_CONTRAST_IGNORES_ATTRITION_FLOOR",
      json.dumps(dict(incomplete_slots=21, n_generators=v["n_generators"],
                      p_holm=v["p_holm"], reject=v["reject_at_alpha"])))

world()
for gi in range(30):
    for fam in ("sine", "chirp"):
        for seed in (0, 1):
            summary(fam, "clean", 16, gi, seed, 8 + gi % 5)
r = verify()
v = r["analysis"]["contrasts"]["checkpoint_effect::primary_pair"]
assert v["n_generators"] == 30 and v["reject_at_alpha"]
print("F3_DISTINCT_GENERATORS_COLLAPSE_BY_G_INDEX",
      json.dumps(dict(distinct_family_generators=60, all_missing_seed_2=True,
                      reported_n=v["n_generators"], reject=v["reject_at_alpha"])))

world()
u = iv_unit("DEVELOPMENT", "sine", "clean", 16, 0, 0)
rec = {"unit_id": u["unit_id"]}
rec["record_sha256"] = m4._self_sha(rec, "record_sha256")
put(ce["unit_record_path"]("/virtual", u["unit_id"]), rec)


def forbidden(*args, **kwargs):
    raise AssertionError("Must not construct generators or train in this audit")


ce["gb"] = NS(generate=forbidden, consumer_verify=forbidden)
rn._run_intervention_unit_v5 = forbidden
rn._limits = forbidden
r = ce["execute_confirmation_units"](
    design, [u], "/virtual", ce["new_accounting"](),
    expect_role="DEVELOPMENT", allow_confirmation=False,
    prior_digests={"synthetic_collision_digest"})
assert r["census_complete"] and r["generators_disjointness_verified"] == 0
print("F4_RESUME_ACCEPTS_UID_AND_HASH_ONLY",
      json.dumps({k: r[k] for k in (
          "units_complete", "units_new_this_session", "census_complete",
          "generators_disjointness_verified")}))

# Gate probes use a sentinel returned by a mocked record-reader, not approval
# documents. No schema, author, decision, external path, or real record exists.
rn._excl_json = put
ce["os"] = NS(chmod=lambda *args: None)
ce["prior_role_digest_census"] = lambda: set()
cp.require_both_records = lambda _: {
    "review": {"record_sha256": "TEST_SENTINEL_REVIEW"},
    "execution": {"record_sha256": "TEST_SENTINEL_EXECUTION"}}
heads = []
for head in ("1" * 40, "2" * 40):
    FILES.clear()
    DIRS.clear()
    ce["_git"] = lambda root, *args: NS(
        returncode=0, stdout="" if args[0] == "status" else head)
    ce["execute_confirmation"](REPO, "/virtual")
    ledger = json.loads(FILES["/virtual/CONFIRMATION_PRE_RESULT_LEDGER.json"])
    heads.append(ledger["gates"]["executing_head"])
assert len(set(heads)) == 2
print("F5_SAME_AUTHORITY_SENTINELS_ACCEPT_DIFFERENT_CODE_HEADS", heads)

def absent(_):
    raise cp.ConfirmationProtocolRefusal("ABSENT synthetic record path")

FILES.clear()
DIRS.clear()
cp.require_both_records = absent
try:
    ce["execute_confirmation"](REPO, "/virtual")
except SystemExit:
    pass
else:
    raise AssertionError("Absent record must refuse")
assert not FILES and not DIRS
print("CONTROL_ABSENT_RECORD_REFUSES_BEFORE_WRITES", True)
try:
    ce["verify_role_disjointness"]({"collision"}, {"collision"})
except SystemExit:
    pass
else:
    raise AssertionError("Collision must refuse")
print("CONTROL_ARRAY_COMPARATOR_REJECTS_COLLISION", True)
print("PEAK_RSS_MIB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2))
print("NO_TRAINING_NO_GENERATORS_NO_REAL_RECORDS_NO_FILESYSTEM_WRITES")
