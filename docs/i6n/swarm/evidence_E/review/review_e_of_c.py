"""Lane E review of lane C: reproducers on C@9f055497 (fabricated rows; no fit, no data, no TEST)."""
import json, hashlib
from tools import i6n_weekly_rollup as wr
from tools import i6n_external_clock_audit_v2_resume as resume

out = {}

def sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

# C1: A.2 candidate_identity_sha256 is the FULL evaluator identity (cell config incl. evaluation_week_id,
# context incl. fold, support): one arm refit weekly yields a different id per week. The accumulator binds
# all rows to ONE candidate_identity_sha256.
weeks = {"W2024-01-07": {"n": 3, "origins_sha256": "1" * 64}, "W2024-01-14": {"n": 2, "origins_sha256": "2" * 64}}
acc = wr.AnnualAccumulator(experiment="X", candidate_identity_sha256=sha({"week": "W2024-01-07", "arm": "best"}),
                           horizon_hours=12, role="MANAGEMENT", expected=weeks)
def row(week, n, arm_week_id):
    return dict(schema_version="i6n.candidate_receipt.v1", experiment="X", candidate_identity_sha256=arm_week_id,
                fold=week, horizon_hours=12, head_offset_hours=12, role="MANAGEMENT", population="S",
                split="VALIDATION", metric="ABS_AND_SQUARED_ERROR_SUMS", unit="PRICE", scale="ORIGINAL_PRICE",
                n=n, sum_abs_error=0.003 * n, sum_sq_error=1e-5 * n, naive_sum_abs_error=0.0031 * n,
                naive_sum_sq_error=1.1e-5 * n, status="SCORED", support_sha256=weeks[week]["origins_sha256"],
                c2_receipt_sha256=None)
res = {}
for w in weeks:
    try:
        res[w] = acc.add(row(w, weeks[w]["n"], sha({"week": w, "arm": "best"})))
    except wr.RollupError as exc:
        res[w] = f"REFUSED {exc.code}"
out["C1_same_arm_two_weekly_refits_A_identities"] = res
out["C1_annual_state"] = acc.summary()["state"]

# C2: the 429 retry rule is bound to "exactly one pending case named by --only-date", not to a date and
# not to an attempt count: any single 429 date passes, and the same date passes again on every rerun.
def prior(date):
    return dict(status="INCONCLUSIVE", evidence_role="holdout", dukascopy_sha256=None, date=date,
                reason=resume.RATE_LIMITED)
out["C2_429_is_transport_failure_any_date"] = [resume.transport_failure(prior(d)) for d in ("2019-10-28", "2021-02-01")]
out["C2_attempt_cap_in_code"] = any(k in open(resume.__file__).read() for k in ("MAX_RATE_LIMITED", "max_attempts", "attempt_cap"))
print(json.dumps(out, indent=1))
