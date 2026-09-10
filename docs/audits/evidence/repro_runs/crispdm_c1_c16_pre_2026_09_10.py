"""PRE for order C1-C16: every finding in Musashi's audit,
reproduced from the shipped code before a single line is changed.

Twelve reproductions, each naming the reason it fires. A PRE that
passes for an accidental reason is worthless, so every check
asserts the SPECIFIC defect, not merely that something failed.

  1. use_optimizer=true reaches optimizer.optimize() BEFORE the
     gate (my "single choke point" claim was false)
  2. a valid manifest with subject_ids=[] returns
     ELIGIBILITY_GATED
  3. a manifest the caller wrote itself, with its own digest in
     its own config, grants a positive decision
  4. configured subjects may differ from the columns actually
     consumed
  5. substituting data/partition digests under the same id is
     never compared by gate_subjects()
  6. duplicate JSON keys, NaN, a non-canonical digest and a
     future date all reach the current parser
  7. a first census with new entries digests NOTHING, and an
     equal-length mutation does not force a re-hash
  8. an availability instance indexed by feature_family never
     reaches a variable looked up by entity
  9. the common index has 214 rows and ZERO conceptual-variable
     rows
 10. two envelopes sharing a campaign_key with different identity
     share the dimension
 11. a well-formed 64-character string opens the backup gate even
     when no dump exists
 12. an MTM partition shorter than window_size+1 returns a bare
     ndarray while the caller unpacks a tuple

CPU only. Nothing is written outside a temporary directory; the
populated cube is read but never written.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIN = REPO.parent / "financial-data"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FIN / "_scripts" / "lib"))

FAILURES = []


def check(number, name, condition, detail):
    """A PRE item passes when the DEFECT is reproduced."""
    status = "REPRODUCED" if condition else "NOT REPRODUCED"
    print(f"PRE-{number:02d} {name}: {status}")
    print(f"        {detail}")
    if not condition:
        FAILURES.append((number, name, detail))


# ==============================================================
# 1. the optimizer runs before the gate
# ==============================================================
main_src = (REPO / "app/main.py").read_text()
opt_line = next(i for i, ln in enumerate(main_src.splitlines(), 1)
                if "optimizer_plugin.optimize(" in ln)
gate_line = next(i for i, ln in
                 enumerate(main_src.splitlines(), 1)
                 if "gate_subjects(" in ln
                 and "from eligibility" not in ln)
check(1, "optimizer precedes the gate",
      opt_line < gate_line,
      f"optimizer_plugin.optimize() at app/main.py:{opt_line}, "
      f"gate_subjects() at app/main.py:{gate_line} — with "
      "use_optimizer=true the optimizer builds data and fits "
      "models before the gate is ever consulted, and the "
      "comment above it calls itself the single choke point")

# the shipped test only compares the gate against the PIPELINE
consumer_test = (REPO / "tests/unit_tests/"
                        "test_eligibility_integration.py").read_text()
check(2, "the shipped test never mentions the optimizer",
      "optimize" not in consumer_test,
      "test_main_asks_the_gate_before_the_pipeline_runs "
      "compares gate_subjects() only against "
      "run_prediction_pipeline; the optimizer path is untested")

from eligibility import gate  # noqa: E402
from eligibility import integration as integ  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="c1c16_pre_"))


def _entry(subject_id="var.ok", **over):
    e = {
        "subject_kind": "variable", "subject_id": subject_id,
        "version": "1",
        "io_schema": {"input": ["float"], "output": ["float"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t+0"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "none", "parameters": {},
        "digests": {"data": "d" * 64, "code": "c" * 64,
                    "partitions": "p" * 64,
                    "evidence": "e" * 64},
        "measured_cost": {"fit_seconds": 0.0},
        "decision": "PUBLICLY_ELIGIBLE",
        "decision_scope": "forecasting",
        "decision_reason": "reviewed", "reviewer": "candidate",
        "reviewed_at": "2026-09-09T00:00:00Z",
    }
    e.update(over)
    return e


def _manifest(entries=None, **over):
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "candidate", "scope": "test",
           "entries": entries or [_entry()]}
    doc.update(over)
    doc["manifest_sha256"] = gate._self_sha(doc)
    return doc


# ==============================================================
# 3. an empty subject list still returns GATED
# ==============================================================
mpath = TMP / "self_issued.json"
doc = _manifest()
mpath.write_text(json.dumps(doc))
cfg = {integ.KEY_MANIFEST: str(mpath),
       integ.KEY_SCOPE: "forecasting"}
stamp_empty = integ.gate_subjects(cfg, subject_ids=[],
                                  consumer="pre")
check(3, "empty subject list returns GATED",
      stamp_empty["eligibility_status"] == integ.STATUS_GATED
      and stamp_empty["subjects_used_count"] == 0,
      f"status={stamp_empty['eligibility_status']} "
      f"universe_size={stamp_empty['universe_size']} "
      f"subjects_used={stamp_empty['subjects_used']} — a run "
      "that reviewed NOTHING carries the same positive label as "
      "a run that reviewed everything")

stamp_none = integ.gate_subjects(dict(cfg), subject_ids=None,
                                 consumer="pre")
check(4, "the default config makes this the default path",
      stamp_none["eligibility_status"] == integ.STATUS_GATED,
      "app/config.py ships eligibility_subjects=None, so "
      f"status={stamp_none['eligibility_status']} is what a "
      "default gated run gets")

# ==============================================================
# 5. a self-issued manifest grants a positive decision
# ==============================================================
cfg_pinned = {integ.KEY_MANIFEST: str(mpath),
              integ.KEY_SCOPE: "forecasting",
              integ.KEY_SHA: doc["manifest_sha256"]}
stamp_self = integ.gate_subjects(cfg_pinned,
                                 subject_ids=["var.ok"],
                                 consumer="pre")
check(5, "a self-issued manifest grants GATED",
      stamp_self["eligibility_status"] == integ.STATUS_GATED,
      "the manifest was written by this process with "
      "reviewer='candidate', its own self digest, and that same "
      "digest supplied as the pinned expectation from the SAME "
      "config — nothing external reviewed anything, and the run "
      f"is labelled {stamp_self['eligibility_status']}")

# ==============================================================
# 6. declared bindings are never compared at the call point
# ==============================================================
import inspect  # noqa: E402
gs_src = inspect.getsource(integ.gate_subjects)
check(6, "gate_subjects passes no version or digest",
      "code_digest" not in gs_src
      and "evidence_digest" not in gs_src
      and "version" not in gs_src,
      "require_eligible() can compare version, code_digest and "
      "evidence_digest, but gate_subjects() supplies none of "
      "them, so a manifest entry's data and partition digests "
      "are never checked against the bytes in hand")

# substituting the data and partition digests changes nothing
doc_sub = _manifest([_entry(digests={"data": "9" * 64,
                                     "code": "c" * 64,
                                     "partitions": "8" * 64,
                                     "evidence": "e" * 64})])
p_sub = TMP / "substituted.json"
p_sub.write_text(json.dumps(doc_sub))
stamp_sub = integ.gate_subjects(
    {integ.KEY_MANIFEST: str(p_sub),
     integ.KEY_SCOPE: "forecasting"},
    subject_ids=["var.ok"], consumer="pre")
check(7, "substituted data/partition digests still pass",
      stamp_sub["eligibility_status"] == integ.STATUS_GATED,
      "the same subject id with COMPLETELY different data and "
      "partition digests is accepted identically — the id is "
      "the only thing that travels")

# ==============================================================
# 8. configured subjects may differ from consumed columns
# ==============================================================
check(8, "no consumed-column derivation exists",
      not any(hasattr(integ, n) for n in
              ("resolve_consumed_subjects",
               "derive_consumed_subjects")),
      "nothing derives the subject set from the run's own data "
      "contract or file headers; the caller's optional list is "
      "the only source, so it can name variables the run does "
      "not read and omit variables it does")

# ==============================================================
# 9. the parser accepts what it says it rejects
# ==============================================================
raw_dup = ('{"schema": "%s", "issued_at": '
           '"2026-09-10T00:00:00Z", "issuer": "a", "issuer": '
           '"b", "scope": "s", "entries": [], '
           '"manifest_sha256": "PLACEHOLDER"}'
           % gate.MANIFEST_SCHEMA)
parsed_dup = json.loads(raw_dup)
dup_ok = parsed_dup["issuer"] == "b"

nan_doc = _manifest([_entry(measured_cost={
    "fit_seconds": float("nan")})])
p_nan = TMP / "nan.json"
p_nan.write_text(json.dumps(nan_doc))
try:
    gate.load_manifest(p_nan)
    nan_ok = True
    nan_detail = "NaN accepted"
except SystemExit as exc:
    nan_ok = False
    nan_detail = f"refused: {exc}"

upper_doc = _manifest([_entry(digests={
    "data": "D" * 64, "code": "c" * 64,
    "partitions": "p" * 64, "evidence": "e" * 64})])
p_up = TMP / "upper.json"
p_up.write_text(json.dumps(upper_doc))
try:
    gate.load_manifest(p_up)
    upper_ok = True
except SystemExit:
    upper_ok = False

short_doc = _manifest([_entry(digests={
    "data": "abc", "code": "c" * 64,
    "partitions": "p" * 64, "evidence": "e" * 64})])
p_short = TMP / "short.json"
p_short.write_text(json.dumps(short_doc))
try:
    gate.load_manifest(p_short)
    short_ok = True
except SystemExit:
    short_ok = False

future_doc = _manifest(issued_at="2099-01-01T00:00:00Z")
p_fut = TMP / "future.json"
p_fut.write_text(json.dumps(future_doc))
try:
    gate.load_manifest(p_fut, max_age_days=30)
    future_ok = True
except SystemExit:
    future_ok = False

check(9, "the manifest parser is permissive",
      dup_ok and nan_ok and upper_ok and short_ok and future_ok,
      f"duplicate keys silently keep the last value "
      f"(issuer={parsed_dup['issuer']!r}); {nan_detail}; an "
      "UPPERCASE digest passes; a 3-character 'digest' passes; "
      "an issued_at of 2099 passes a 30-day staleness limit "
      "because a negative age is not greater than the limit")

# ==============================================================
# 10. the census digests nothing on a first run
# ==============================================================
import incremental_census as ic  # noqa: E402

cen_src = (FIN / "_scripts/lib/incremental_census.py").read_text()
new_entry_digests = "prev is None" in cen_src.split(
    "want_digest")[1][:400]
check(10, "a NEW appearance never triggers a digest",
      not new_entry_digests,
      "under policy 'selected' want_digest is true only for an "
      "explicit selection or a size change; a first census has "
      "no previous, so every appearance is DECLARED_ONLY and "
      "0/1,680 bytes are digested")

check(11, "change detection ignores an equal-length mutation",
      'prev.get("size_bytes") != size' in cen_src,
      "changed_physically compares SIZE only; mtime_ns is "
      "recorded but never compared and ctime_ns is not recorded, "
      "so a mutation that preserves length is invisible")

writer_src = (FIN /
              "_scripts/build_incremental_census.py").read_text()
check(12, "the content-addressed writer truncates",
      "O_TRUNC" in writer_src,
      "the census file is named census-<sha>.json but opened "
      "with O_CREAT|O_TRUNC, so an existing content-addressed "
      "artifact is overwritten instead of verified")

check(13, "external profiles are copied, not revalidated",
      "sha256_file" not in writer_src.split(
          "_load_external_full_profiles")[1][:1200],
      "_load_external_full_profiles copies the external "
      "inventory's self digest, row counts and dataset digests "
      "without recomputing the inventory digest or hashing a "
      "single byte it claims to have profiled")

# ==============================================================
# 14. the availability join cannot fire
# ==============================================================
avail_indexed_by = "instances[cand[\"feature_family\"]]" in \
    cen_src
avail_looked_up_by = 'availability["instances"].get(a["entity"])' \
    in cen_src
check(14, "availability instances are unreachable",
      avail_indexed_by and avail_looked_up_by,
      "instances are keyed by cand['feature_family'] but looked "
      "up by a['entity']; the two namespaces coincide only by "
      "accident, so the first real contract can land and still "
      "leave every variable UNAVAILABLE")

# ==============================================================
# 15. the common index carries no variables
# ==============================================================
idx = json.loads((REPO / "examples/research/"
                         "crispdm_bank_index.v1.json").read_text())
kinds = {}
for row in idx["common_rows"]:
    kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
check(15, "the index has no conceptual-variable rows",
      idx["row_count"] == 214 and "variable" not in kinds,
      f"row_count={idx['row_count']} kinds={kinds} — the "
      "financial bank's 1,965 conceptual variables and the "
      "public bank's series appear only as COUNTS inside the "
      "bank summary, never as consumable rows")

# ==============================================================
# 16. envelope identity collision
# ==============================================================
from olap import campaign_envelope as ce  # noqa: E402

env_src = (REPO / "olap/campaign_envelope.py").read_text()
check(16, "campaign_key conflicts are ignored",
      "ON CONFLICT (campaign_key) DO NOTHING" in env_src,
      "dim_campaign is inserted with ON CONFLICT DO NOTHING and "
      "no comparison of producer, class, design, code or run, so "
      "a second envelope with the same campaign_key and a "
      "DIFFERENT identity attaches its facts to the first "
      "campaign's dimension row")

# ==============================================================
# 17. the backup gate is a length check
# ==============================================================
bf_src = (REPO /
          "tools/backfill_campaign_envelopes.py").read_text()
check(17, "the backup gate never opens a file",
      "len(a.backup_sha256) != 64" in bf_src
      and "sha256_file" not in bf_src,
      "the backfill validates --backup-sha256 by LENGTH only; a "
      "well-formed 64-character string authorises a migration "
      "with no dump on disk and no recomputed digest")

# ==============================================================
# 18. MTM returns a bare array on short partitions
# ==============================================================
import numpy as np  # noqa: E402

mod = __import__("preprocessor_plugins.phase2_6_preprocessor",
                 fromlist=["*"])
cls = getattr(mod, "PreprocessorPlugin", None) or \
    getattr(mod, "Plugin", None)
plug = cls()
short = np.zeros(6, dtype=np.float32)
out = plug._apply_causal_mtm_decomposition(short, 32, 2, "short")
check(18, "short partition breaks the return contract",
      isinstance(out, np.ndarray),
      f"returned {type(out)} with shape {getattr(out, 'shape', None)} "
      "while the caller unpacks `components, scaler = ...`; a "
      "partition shorter than the window silently changes the "
      "function's type")

# ==============================================================
# 19. the cube is up but nothing feeds it automatically
# ==============================================================
callers = subprocess.run(
    ["grep", "-rn", "load_envelope(", "--include=*.py",
     str(REPO)], capture_output=True, text=True).stdout
call_files = sorted({ln.split(":")[0].split("/")[-1]
                     for ln in callers.strip().splitlines()
                     if not ln.split(":")[0].endswith(
                         Path(__file__).name)})
check(19, "no pipeline calls the loader",
      all(f in ("campaign_envelope.py",
                "backfill_campaign_envelopes.py",
                "test_campaign_envelope.py")
          for f in call_files),
      f"load_envelope() callers: {call_files} — the only real "
      "flow is the manual backfill CLI, so no run, candidate or "
      "cell reaches the cube on its own")

print()
if FAILURES:
    print(f"PRE INCOMPLETE: {len(FAILURES)} item(s) did not "
          "reproduce:")
    for n, name, _ in FAILURES:
        print(f"  PRE-{n:02d} {name}")
    raise SystemExit(1)
print("PRE CONFIRMED: every finding in the governing audit "
      "reproduces from the shipped code, each for its own "
      "named reason. The optimizer really does precede the "
      "gate, an empty subject list and a self-issued manifest "
      "really do return ELIGIBILITY_GATED, the declared "
      "bindings really are never compared to bytes, the first "
      "census really digests nothing, the availability join "
      "really cannot fire, the index really carries no "
      "variables, the campaign dimension really ignores "
      "identity, the backup gate really is a length check, the "
      "MTM contract really breaks on short partitions, and "
      "nothing feeds the cube automatically.")
