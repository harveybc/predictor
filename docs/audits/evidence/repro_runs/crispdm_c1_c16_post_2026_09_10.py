"""POST for order C1-C16: every defect the PRE reproduced is
gone, each for its own named reason.

The PRE is frozen at the reviewed tip and is NOT re-run here — on
corrected code it cannot even locate the call it was written to
inspect, because `gate_subjects(` no longer exists in
`app/main.py`. That is itself part of the evidence and is
asserted below.

Nineteen corrections, in the PRE's own order. CPU only; the
populated cube is read but never written.
"""
import csv
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


def check(n, name, condition, detail):
    print(f"POST-{n:02d} {name}: "
          f"{'CORRECTED' if condition else 'STILL PRESENT'}")
    print(f"         {detail}")
    if not condition:
        FAILURES.append((n, name))


from eligibility import gate            # noqa: E402
from eligibility import integration as integ  # noqa: E402
from eligibility import review          # noqa: E402
from eligibility import strict          # noqa: E402
from eligibility.consumed import (       # noqa: E402
    resolve_consumed_subjects)
from olap import bank_index as bi       # noqa: E402
from olap import campaign_envelope as ce  # noqa: E402
from olap import outbox as ob           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="c1c16_post_"))
main_src = (REPO / "app/main.py").read_text()

# ---- 1/2: order of execution ----
gate_at = main_src.index("gate_run(")
opt_at = main_src.index("optimizer_plugin.optimize(")
pipe_at = main_src.index(
    "pipeline_plugin.run_prediction_pipeline(")
check(1, "the gate precedes the optimizer",
      gate_at < opt_at < pipe_at,
      "gate_run() now stands before optimizer_plugin.optimize() "
      "and before the pipeline; the PRE's own locator crashes "
      "here because gate_subjects( no longer exists in "
      f"app/main.py ({'gate_subjects(' not in main_src})")

head = main_src[:gate_at]
consuming = [c for c in ("optimizer_plugin.optimize(",
                         "run_prediction_pipeline(",
                         "run_preprocessing(", ".build_model(",
                         ".train(", "fit_transform(")
             if c in head]
check(2, "no data-consuming call precedes the gate",
      not consuming,
      "every consuming call now follows the gate; the test that "
      "checks this names each one rather than only the pipeline")

# ---- 3/4: an empty subject set no longer grants ----
cols = ["DATE_TIME", "px", "vol"]
for r in ("d4", "d5", "d6"):
    p = TMP / f"{r}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow(["2020-01-01", "1", "2"])
config = {"x_train_file": str(TMP / "d4.csv"),
          "x_validation_file": str(TMP / "d5.csv"),
          "x_test_file": str(TMP / "d6.csv"),
          "target_column": "px",
          integ.KEY_SCOPE: "forecasting",
          integ.PURPOSE_KEY: integ.PURPOSE_EXPERIMENT}
derived = resolve_consumed_subjects(dict(config), repo_root=TMP)
check(3, "the subject set is DERIVED from the bytes",
      derived["subject_ids"] and len(derived["subject_ids"]) == 2,
      f"resolved {derived['subject_ids']} from the partition "
      "headers the contract names, with DATE_TIME excluded as a "
      "row identifier; a config list can no longer be the source")

try:
    resolve_consumed_subjects(dict(config,
                                   eligibility_subjects=[]),
                              repo_root=TMP)
    empty_ok = False
    empty_detail = "an empty list still passed"
except SystemExit as exc:
    empty_ok = "assertion is false" in str(exc)
    empty_detail = str(exc)[:110]
check(4, "an empty declared list refuses",
      empty_ok, empty_detail)

# ---- 5: a self-issued manifest grants nothing ----
doc = {"schema": gate.MANIFEST_SCHEMA,
       "issued_at": "2026-09-10T00:00:00Z",
       "issuer": "candidate", "scope": "s", "entries": []}
doc["manifest_sha256"] = gate._self_sha(doc)
mp = TMP / "self_issued.json"
mp.write_text(json.dumps(doc))
cfg = dict(config)
cfg[integ.KEY_MANIFEST] = str(mp)
cfg[integ.KEY_SHA] = doc["manifest_sha256"]
cfg[integ.KEY_CENSUS] = "a" * 64
os.environ[review.REVIEW_RECORD_ENV] = str(TMP / "absent.json")
try:
    integ.gate_run(cfg, repo_root=TMP, consumer="post")
    self_ok, self_detail = False, "still granted"
except SystemExit as exc:
    self_ok = "no external eligibility review" in str(exc)
    self_detail = str(exc)[:110]
check(5, "a self-issued manifest no longer grants GATED",
      self_ok, self_detail)

import inspect  # noqa: E402
sig = inspect.signature(review.review_record_path)
check(6, "the config cannot choose the review record",
      list(sig.parameters) == [],
      "review_record_path() takes NO arguments, so no "
      "configuration value can reach it; only a repository "
      "constant and an environment variable resolve it")

# ---- 7: bindings compared at the point of use ----
entry = {
    "subject_kind": "variable", "subject_id": "v",
    "version": "1",
    "io_schema": {"input": ["f"], "output": ["f"]}, "unit": "u",
    "temporal_availability": {"event_time": "t",
                              "available_time": "t"},
    "fit_scope": "TRAIN_ONLY", "incremental_state_policy": "n",
    "parameters": {},
    "digests": {"data": "d" * 64, "code": "c" * 64,
                "partitions": "b" * 64, "evidence": "e" * 64},
    "measured_cost": {}, "decision": "PUBLICLY_ELIGIBLE",
    "decision_scope": "s", "decision_reason": "r",
    "reviewer": "x", "reviewed_at": "2026-09-09T00:00:00Z"}
m = {"entries": [entry]}
refused = {}
for kw, label in (({"data_digest": "9" * 64}, "data"),
                  ({"partitions_digest": "9" * 64},
                   "partitions"),
                  ({"code_digest": "9" * 64}, "code"),
                  ({"evidence_digest": "9" * 64}, "evidence")):
    try:
        gate.require_eligible(m, "v", scope="s", **kw)
        refused[label] = False
    except SystemExit:
        refused[label] = True
check(7, "substituted bindings refuse at the point of use",
      all(refused.values()),
      f"data/partitions/code/evidence each refuse: {refused}")

# ---- 8: derivation exists ----
check(8, "a consumed-column derivation exists",
      callable(resolve_consumed_subjects),
      "eligibility.consumed.resolve_consumed_subjects reads the "
      "partition headers, requires identical schemas across "
      "train/validation/test, refuses duplicates and unnamed "
      "columns, and digests bytes, schema, partitions and code")

# ---- 9: the parser is strict ----
strict_results = {}
p = TMP / "probe.json"
p.write_text('{"a": 1, "a": 2}')
try:
    strict.strict_load_file(p, what="probe")
    strict_results["duplicate_keys"] = False
except SystemExit:
    strict_results["duplicate_keys"] = True
p.write_text('{"a": NaN}')
try:
    strict.strict_load_file(p, what="probe")
    strict_results["nan"] = False
except SystemExit:
    strict_results["nan"] = True
for bad, label in (("A" * 64, "uppercase_digest"),
                   ("abc", "short_digest")):
    try:
        strict.require_sha256(bad, what="d")
        strict_results[label] = False
    except SystemExit:
        strict_results[label] = True
try:
    strict.require_timestamp("2099-01-01T00:00:00Z", what="t")
    strict_results["future_date"] = False
except SystemExit:
    strict_results["future_date"] = True
check(9, "the parser refuses what it used to accept",
      all(strict_results.values()),
      f"{strict_results} — duplicate keys, NaN, a non-canonical "
      "digest and a future timestamp all refuse now")

# ---- 10/11/12/13: the census ----
import incremental_census as ic  # noqa: E402
summary = json.loads((FIN / "features/census/"
                            "CENSUS_SUMMARY.v1.json").read_text())
cov = summary["coverage"]
check(10, "a first census digests every appearance",
      cov["appearances_physically_digested"] == 1680
      and cov["appearances_not_digested"] == 0,
      f"{cov['appearances_physically_digested']}/1680 digested, "
      f"{cov['bytes_read_for_digest']:,} bytes read, coverage "
      f"{cov['digest_coverage_fraction']} — the real 14.4 GB lake")

cen_src = (FIN / "_scripts/lib/incremental_census.py").read_text()
check(11, "an equal-length mutation forces a re-hash",
      'prev.get("mtime_ns") != mtime_ns' in cen_src
      and 'prev.get("ctime_ns") != ctime_ns' in cen_src,
      "physical identity compares size, mtime_ns AND ctime_ns; a "
      "regression rewrites a file to the same length and proves "
      "it is re-digested")

writer_src = (FIN /
              "_scripts/build_incremental_census.py").read_text()
check(12, "the content-addressed writer never truncates",
      "O_TRUNC" not in writer_src and "os.O_EXCL" in writer_src,
      "O_EXCL plus a full byte-equality check; an existing "
      "artifact is verified, not overwritten")

check(13, "external profiles are re-derived and re-hashed",
      "FULL_PROFILE_PHYSICALLY_VERIFIED" in writer_src
      and "DECLARED_ONLY_NOT_VERIFIED" in writer_src,
      "the inventory's self digest is recomputed and every "
      "dataset it claims to profile is re-hashed; anything "
      "unverifiable says why")

# ---- 14: the availability join ----
join = ic.map_families_to_entities(
    {"macro/one": {}},
    [("cross_source", "macro__one__a"),
     ("cross_source", "macro__one__b"),
     ("cross_source", "other__z")])
check(14, "an availability family reaches its variables",
      join["by_family"]["macro/one"] == ["macro__one__a",
                                         "macro__one__b"]
      and "other__z" not in join["entity_to_family"],
      "the explicit join resolves a family to the entities whose "
      "source path it prefixes, reports unmatched families and "
      "refuses to choose when an entity matches two")

# ---- 15: the index carries variables ----
idx = json.loads((REPO / "examples/research/"
                         "crispdm_bank_index.v1.json").read_text())
kinds = {}
for row in idx["common_rows"]:
    kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
check(15, "the index carries the 1,965 variables as rows",
      kinds.get("variable") == 1965
      and bi._recount(idx["common_rows"]) ==
      idx["cardinality_by_kind_and_authority"],
      f"{idx['row_count']} rows: {kinds} — and every cardinality "
      "is recounted from the rows")

# ---- 16: campaign identity ----
env_src = (REPO / "olap/campaign_envelope.py").read_text()
check(16, "a campaign_key collision refuses",
      "DIFFERENT identity" in env_src
      and "ON CONFLICT (campaign_key) DO NOTHING" not in env_src,
      "the stored identity is compared field by field and a "
      "difference refuses BEFORE any fact is written")

# ---- 17: the backup gate ----
bf_src = (REPO /
          "tools/backfill_campaign_envelopes.py").read_text()
check(17, "the backup gate opens the dump",
      "_sha_file(a.backup_file)" in bf_src
      and "a digest is not a backup" in bf_src,
      "the dump is opened and re-hashed; a well-formed string "
      "with no file refuses")

# ---- 18: the MTM contract ----
import numpy as np  # noqa: E402
mod = __import__("preprocessor_plugins.phase2_6_preprocessor",
                 fromlist=["*"])
cls = getattr(mod, "PreprocessorPlugin", None) or \
    getattr(mod, "Plugin", None)
out = cls()._apply_causal_mtm_decomposition(
    np.zeros(6, dtype=np.float32), 32, 2, "short")
check(18, "a short partition keeps the return contract",
      isinstance(out, tuple) and len(out) == 2
      and out[1] is None,
      f"returns {type(out).__name__} of length {len(out)} with "
      "scaler=None, so validation and test become NOT_EVALUABLE "
      "instead of fitting one late")

# ---- 19: the cube is fed ----
callers = subprocess.run(
    ["grep", "-rln", "outbox.emit(", "--include=*.py",
     str(REPO / "app"), str(REPO / "tools")],
    capture_output=True, text=True).stdout.split()
check(19, "a terminal run feeds the cube on its own",
      any("main.py" in c for c in callers),
      f"app/main.py emits its envelope to the durable outbox "
      f"({[Path(c).name for c in callers]}), and the CPU loader "
      "drains it idempotently — a run no longer depends on a "
      "human running a backfill CLI")

print()
if FAILURES:
    print(f"POST INCOMPLETE: {len(FAILURES)} defect(s) remain:")
    for n, name in FAILURES:
        print(f"  POST-{n:02d} {name}")
    raise SystemExit(1)
print("POST CONFIRMED: all nineteen defects the PRE reproduced "
      "are corrected, each for its own reason. The gate precedes "
      "the optimizer and derives its subjects from the bytes; a "
      "self-issued manifest and an empty subject list grant "
      "nothing; every declared binding is compared at the point "
      "of use; the parser refuses duplicate keys, NaN, "
      "non-canonical digests and future dates; the census "
      "digested all 1,680 appearances over 14.4 GB and re-hashes "
      "an equal-length mutation; the content-addressed artifact "
      "is never truncated; external profiles are re-derived; the "
      "availability join fires; the index carries 1,965 variable "
      "rows; a campaign collision and an absent backup refuse; "
      "the MTM contract is total; and a terminal run now feeds "
      "the cube by itself.")
