"""Lane H: deterministic CAUSAL Kalman family (successor of the historical MLE ``local_level_kalman``).

DEVELOPMENT code. Nothing here is a confirmed result and no output of this module is a
claim of predictive improvement; selection is made elsewhere, by out-of-train utility.

Why a successor. The historical ``local_level_kalman`` (tools/df_operators.py, concentrated
MLE through ``scipy.optimize.minimize_scalar``) was not reproducible on one of three CPUs
(docs/audits/evidence/repro_runs/d2_support_r1/r4/R4_PORTABILITY_POLICY.md; cause not
isolated). This module removes every ingredient that can differ between machines:

* parameters come from a CLOSED FORM (second-moment estimator on differences, summed with
  ``math.fsum`` so the result does not depend on summation order) or from a PRE-DECLARED
  ratio; there is no optimiser, no iteration, no tolerance;
* the filter is a scalar Python recursion over ``float`` using only ``+ - * /`` and
  ``math.sqrt`` (IEEE correctly rounded); no BLAS, no ``numpy`` reduction, no libm
  transcendental on the artifact path. numpy is used only to hold arrays;
* the environment (interpreter, numpy, thread variables, CPU vendor/flag digest) is
  recorded next to the artifact but is NOT part of the sealed digest, so equal digests on
  two machines mean the numbers are identical, not that the environments were.

Contract reused from tools/df_operators.py (C135): typed reasons AVAILABLE / WARMUP /
MISSING_INPUT / FUTURE_UNAVAILABLE, ``OperatorRefusal``, sealed canonical-JSON artifact,
``fit`` / ``transform_batch`` / ``init_state`` / ``step`` / ``transform_chunk`` /
``save_state`` / ``load_state``, FROZEN fit semantics, delay measured never compensated,
zero look-ahead. Differences, stated: the D2 ``FitSnapshot`` objects are bound to the D2
dataset contracts (TRAIN/CALIBRATION/CONFIRMATION); this lane binds to the lane-F2 split
artefact instead (``binding`` below carries role, row range, split digest and the matrix
digest recomputed here), and only the role TRAIN may fit.

Models (per column, independent):

``kalman_local_level``         y_t = mu_t + e_t, mu_t = mu_{t-1} + eta_t.
``kalman_local_linear_trend``  adds beta_t = beta_{t-1} + zeta_t, mu_t = mu_{t-1} + beta_{t-1} + eta_t.

Closed-form estimators (autocovariances g_k of the centred differences, divisor n):
local level    d = dy:   g0 = q + 2r, g1 = -r             =>  r = -g1,  q = g0 + 2 g1
local trend    d = d2y:  g0 = qs + 2q + 6r, g1 = -q - 4r, g2 = r
                                                           =>  r = g2, q = -g1 - 4 g2, qs = g0 + 2 g1 + 2 g2
Negative estimates are clipped to a declared floor and the clip is recorded per column
(``r_clipped`` ...). ``declared_ratio`` instead fixes q/r (and qs/r) in advance and takes r as
the mean standardized innovation power of the train pass (a closed form too).

Outputs per column: obs (original), level (filtered state), slope (LLT only), innov (one-step
prediction error), zinnov (innov / sqrt(F)), state_var (filtered variance of the level).
``state_var`` is a model covariance under the declared Gaussian model. It is NOT a
probability of being right and no consumer may call it one.

The backward (RTS) smoother exists ONLY as ``smoother_control`` returning a
``NonCausalControlOutput``; it cannot be turned into an eligible input and no causal path
accepts its artifact kind.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ARTIFACT_SCHEMA = "df_kalman_family.artifact.v1"
STATE_SCHEMA = "df_kalman_family.state.v1"
OUTPUT_SCHEMA = "df_kalman_family.output.v1"

LOCAL_LEVEL = "kalman_local_level"
LOCAL_LINEAR_TREND = "kalman_local_linear_trend"
SMOOTHER_CONTROL_KIND = "kalman_rts_smoother_noncausal_control"
CAUSAL_KINDS = (LOCAL_LEVEL, LOCAL_LINEAR_TREND)

AVAILABLE = "AVAILABLE"
WARMUP = "WARMUP"
MISSING_INPUT = "MISSING_INPUT"
FUTURE_UNAVAILABLE = "FUTURE_UNAVAILABLE"
REASONS = (AVAILABLE, WARMUP, MISSING_INPUT, FUTURE_UNAVAILABLE)
_REASON_DTYPE = "<U18"

OUTPUTS = {LOCAL_LEVEL: ("obs", "level", "innov", "zinnov", "state_var"),
           LOCAL_LINEAR_TREND: ("obs", "level", "slope", "innov", "zinnov", "state_var", "slope_var")}

PARAM_SOURCES = ("moments_train", "declared_ratio")
# the predeclared grid (nothing outside it is an operator of this family)
GRID = {
    "param_source": (str, PARAM_SOURCES),
    "ratio_level": (float, (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0)),
    "ratio_slope": (float, (0.0, 1e-06, 1e-04, 1e-02)),
    "q_floor_ratio": (float, (1e-06,)),
    "r_floor_ratio": (float, (1e-06,)),
    "p0_scale": (float, (1.0,)),
    "warmup": (int, (10,)),
}
MIN_TRAIN_ROWS = 50

COVARIANCE_SEMANTICS = ("MODEL_STATE_VARIANCE_UNDER_THE_DECLARED_GAUSSIAN_MODEL; "
                        "NOT_A_PROBABILITY_OF_BEING_RIGHT")


class OperatorRefusal(Exception):
    """Typed refusal; every message starts with REFUSED (same convention as df_operators)."""

    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


class OperatorAbstain(OperatorRefusal):
    pass


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


# ----------------------------------------------------------------------------------------------
# spec
# ----------------------------------------------------------------------------------------------
def validate_spec(spec) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"kind", "params"}:
        raise OperatorRefusal("spec must be exactly {'kind', 'params'}")
    kind = spec["kind"]
    if kind == SMOOTHER_CONTROL_KIND:
        raise OperatorRefusal("the RTS smoother is a NON_CAUSAL control, not an operator: it has no spec, "
                              "no fit and no transform; use smoother_control on a fitted causal artifact")
    if kind not in CAUSAL_KINDS:
        raise OperatorRefusal(f"unknown kind {kind!r}; causal kinds are {list(CAUSAL_KINDS)}")
    params = spec["params"]
    if not isinstance(params, dict) or set(params) != set(GRID):
        raise OperatorRefusal(f"params must be exactly {sorted(GRID)}")
    for key, (typ, allowed) in GRID.items():
        value = params[key]
        if type(value) is not typ:
            raise OperatorRefusal(f"param {key} must be exactly {typ.__name__} (bool/int/float are not interchangeable)")
        if typ is float and not math.isfinite(value):
            raise OperatorRefusal(f"param {key} must be finite")
        if value not in allowed:
            raise OperatorRefusal(f"param {key}={value!r} is outside the predeclared grid {list(allowed)}")
    declared = params["param_source"] == "declared_ratio"
    if declared and params["ratio_level"] <= 0.0:
        raise OperatorRefusal("declared_ratio requires ratio_level > 0")
    if not declared and (params["ratio_level"] != 0.0 or params["ratio_slope"] != 0.0):
        raise OperatorRefusal("moments_train takes no ratios (ratio_level and ratio_slope must be 0.0)")
    if declared and kind == LOCAL_LINEAR_TREND and params["ratio_slope"] <= 0.0:
        raise OperatorRefusal("declared_ratio for the local linear trend requires ratio_slope > 0")
    if declared and kind == LOCAL_LEVEL and params["ratio_slope"] != 0.0:
        raise OperatorRefusal("the local level model has no slope ratio (ratio_slope must be 0.0)")
    return spec


def default_spec(kind: str, **overrides) -> dict:
    params = {"param_source": "moments_train", "ratio_level": 0.0, "ratio_slope": 0.0,
              "q_floor_ratio": 1e-06, "r_floor_ratio": 1e-06, "p0_scale": 1.0, "warmup": 10}
    params.update(overrides)
    return {"kind": kind, "params": params}


# ----------------------------------------------------------------------------------------------
# environment record (never part of the sealed digest)
# ----------------------------------------------------------------------------------------------
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def environment_record(host_role: str | None = None) -> dict:
    flags_digest, vendor = None, None
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("flags") and flags_digest is None:
                flags_digest = hashlib.sha256(" ".join(sorted(line.split(":", 1)[1].split())).encode()).hexdigest()
            if line.startswith("vendor_id") and vendor is None:
                vendor = line.split(":", 1)[1].strip()
    except OSError:
        pass
    threads = {v: os.environ.get(v) for v in THREAD_VARS}
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "machine": platform.machine(), "system": platform.system(), "numpy": np.__version__,
            "float_info_epsilon": sys.float_info.epsilon, "byteorder": sys.byteorder,
            "thread_variables": threads,
            "single_thread_ok": all(v == "1" for v in threads.values() if v is not None) and
                                threads["OMP_NUM_THREADS"] == "1" and threads["OPENBLAS_NUM_THREADS"] == "1" and
                                threads["MKL_NUM_THREADS"] == "1",
            "cpu_vendor": vendor, "cpu_flags_sha256": flags_digest, "host_role": host_role,
            "algorithm": "scalar IEEE double recursion, + - * / and sqrt only; no BLAS, no optimiser"}


# ----------------------------------------------------------------------------------------------
# input checks
# ----------------------------------------------------------------------------------------------
def _real_matrix(x, where: str, allow_nan: bool) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind not in "fiu" or a.dtype == np.bool_:
        raise OperatorRefusal(f"{where} must be a real numeric array")
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise OperatorRefusal(f"{where} must be 2-D (T, V)")
    if np.isinf(a).any():
        raise OperatorRefusal(f"{where} holds +/-inf")
    if not allow_nan and np.isnan(a).any():
        raise OperatorRefusal(f"{where} holds NaN")
    return a


def matrix_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# estimation (closed form; math.fsum makes each sum independent of evaluation order)
# ----------------------------------------------------------------------------------------------
def _autocov(d: list, max_lag: int) -> list:
    n = len(d)
    m = math.fsum(d) / n
    e = [v - m for v in d]
    out = []
    for k in range(max_lag + 1):
        out.append(math.fsum(e[i] * e[i - k] for i in range(k, n)) / n)
    return out


def _moments_ll(y: list):
    d = [y[i] - y[i - 1] for i in range(1, len(y))]
    g = _autocov(d, 1)
    return g, len(d)


def _moments_llt(y: list):
    d = [y[i] - 2.0 * y[i - 1] + y[i - 2] for i in range(2, len(y))]
    g = _autocov(d, 2)
    return g, len(d)


def _ll_pass_unit_r(y: list, lam: float) -> float:
    """Mean standardized innovation power of the local level filter run with r=1, q=lam."""
    level, var = y[0], 1.0
    terms = []
    for t in range(1, len(y)):
        vp = var + lam
        f = vp + 1.0
        v = y[t] - level
        terms.append(v * v / f)
        k = vp / f
        level = level + k * v
        var = (1.0 - k) * vp
    return math.fsum(terms) / len(terms)


def _llt_pass_unit_r(y: list, lam_l: float, lam_s: float) -> float:
    level, slope = y[0], 0.0
    p11, p12, p22 = 1.0, 0.0, 1.0
    terms = []
    for t in range(1, len(y)):
        a1 = level + slope
        q11 = p11 + 2.0 * p12 + p22 + lam_l
        q12 = p12 + p22
        q22 = p22 + lam_s
        f = q11 + 1.0
        v = y[t] - a1
        terms.append(v * v / f)
        k1, k2 = q11 / f, q12 / f
        level = a1 + k1 * v
        slope = slope + k2 * v
        p11 = q11 - k1 * q11
        p12 = q12 - k1 * q12
        p22 = q22 - k2 * q12
    return math.fsum(terms) / len(terms)


def _fit_column(kind: str, params: dict, y: list):
    """Return a per-column fitted dict or an abstain string."""
    n_need = 3 if kind == LOCAL_LEVEL else 4
    if len(y) < n_need:
        return "TRAIN_TOO_SHORT"
    if kind == LOCAL_LEVEL:
        g, n = _moments_ll(y)
        g0 = g[0]
    else:
        g, n = _moments_llt(y)
        g0 = g[0]
    if not g0 > 0.0:
        return "CONSTANT_TRAIN_SERIES: difference variance is not positive; noise not identifiable"
    col = {"moments": {"n": n, "g": [float(v) for v in g]}}
    if params["param_source"] == "moments_train":
        if kind == LOCAL_LEVEL:
            r_raw, q_raw = -g[1], g[0] + 2.0 * g[1]
            qs_raw = None
        else:
            r_raw, q_raw, qs_raw = g[2], -g[1] - 4.0 * g[2], g[0] + 2.0 * g[1] + 2.0 * g[2]
        r_floor = params["r_floor_ratio"] * g0
        r = r_raw if r_raw > r_floor else r_floor
        q_floor = params["q_floor_ratio"] * r
        q = q_raw if q_raw > q_floor else q_floor
        col.update({"r": float(r), "q": float(q), "r_raw": float(r_raw), "q_raw": float(q_raw),
                    "r_clipped": bool(not r_raw > r_floor), "q_clipped": bool(not q_raw > q_floor)})
        if kind == LOCAL_LINEAR_TREND:
            qs = qs_raw if qs_raw > q_floor else q_floor
            col.update({"qs": float(qs), "qs_raw": float(qs_raw), "qs_clipped": bool(not qs_raw > q_floor)})
    else:
        lam_l, lam_s = params["ratio_level"], params["ratio_slope"]
        if kind == LOCAL_LEVEL:
            r = _ll_pass_unit_r(y, lam_l)
        else:
            r = _llt_pass_unit_r(y, lam_l, lam_s)
        if not r > 0.0:
            return "ZERO_INNOVATION_POWER: observation noise not identifiable"
        col.update({"r": float(r), "q": float(lam_l * r), "r_clipped": False, "q_clipped": False})
        if kind == LOCAL_LINEAR_TREND:
            col.update({"qs": float(lam_s * r), "qs_clipped": False})
    col["p0"] = float(params["p0_scale"] * col["r"])
    col["ratio_level_effective"] = float(col["q"] / col["r"])
    if kind == LOCAL_LINEAR_TREND:
        col["ratio_slope_effective"] = float(col["qs"] / col["r"])
    return col


def fit(spec: dict, x_train, binding: dict, host_role: str | None = None) -> dict:
    """Fit on TRAIN rows only. ``binding`` = {dataset_id, role, row_range, split_sha256, column_ids, units}."""
    validate_spec(spec)
    if not isinstance(binding, dict) or binding.get("role") != "TRAIN":
        raise OperatorRefusal("only role TRAIN may fit parameters; Q/R are never estimated on validation or test")
    rr = binding.get("row_range")
    if (not isinstance(rr, (list, tuple)) or len(rr) != 2 or not all(type(v) is int for v in rr) or rr[0] < 0
            or rr[1] <= rr[0]):
        raise OperatorRefusal("binding.row_range must be [start, end) integers")
    x = _real_matrix(x_train, "fit matrix", allow_nan=False)
    if x.shape[0] < MIN_TRAIN_ROWS:
        raise OperatorRefusal(f"fit matrix needs >= {MIN_TRAIN_ROWS} rows")
    if x.shape[0] != rr[1] - rr[0]:
        raise OperatorRefusal("fit matrix rows differ from binding.row_range")
    names = binding.get("column_ids") or [f"c{j}" for j in range(x.shape[1])]
    if len(names) != x.shape[1]:
        raise OperatorRefusal("binding.column_ids length differs from the matrix width")
    cols, status, reason = [], "FITTED", None
    for j in range(x.shape[1]):
        res = _fit_column(spec["kind"], spec["params"], x[:, j].tolist())
        if isinstance(res, str):
            status, reason = "ABSTAIN", f"column {names[j]}: {res}"
            cols = []
            break
        cols.append(res)
    kind = spec["kind"]
    fitted = {} if status == "ABSTAIN" else {
        "per_column": cols,
        "estimator": ("closed_form_second_moments_on_differences" if spec["params"]["param_source"] == "moments_train"
                      else "declared_ratio_with_closed_form_innovation_power"),
        "initial_state_rule": ("level = first finite observation of the stream; "
                               + ("slope = 0; " if kind == LOCAL_LINEAR_TREND else "")
                               + "initial variance = p0 = p0_scale * r (per column, recorded); innovation at the "
                                 "initialising row = 0 and typed WARMUP"),
        "fitted_state_digest_domain": "canonical JSON of per_column"}
    doc = {"schema": ARTIFACT_SCHEMA, "spec": copy.deepcopy(spec), "status": status, "abstain_reason": reason,
           "fit_binding": {"dataset_id": binding.get("dataset_id"), "role": "TRAIN", "row_range": list(rr),
                           "split_sha256": binding.get("split_sha256"), "column_ids": list(names),
                           "matrix_sha256": matrix_sha256(x), "units": binding.get("units"),
                           "scale_note": binding.get("scale_note")},
           "fit_rows": int(x.shape[0]), "n_columns": int(x.shape[1]), "fitted": fitted,
           "fitted_state_digest": _sha(fitted.get("per_column")) if status == "FITTED" else None,
           "meta": _meta(kind), "code_sha256": code_sha256()}
    doc["artifact_sha256"] = _sha(doc)
    out = copy.deepcopy(doc)
    out["environment"] = environment_record(host_role)   # outside the seal on purpose
    return out


def _meta(kind: str) -> dict:
    return {"kind": kind, "outputs": list(OUTPUTS[kind]), "causal": True, "look_ahead": 0, "delay_rows": 0,
            "warmup_policy": "WARMUP while the count of finite observations consumed is <= params.warmup",
            "missing_policy": "PREDICT_ONLY: a NaN row runs the prediction step (variance grows); its outputs are "
                              "MISSING_INPUT; nothing is forward-filled",
            "covariance_semantics": COVARIANCE_SEMANTICS,
            "units": "outputs obs/level/slope/innov in the units of the input column; state_var in units squared; "
                     "zinnov dimensionless",
            "availability": "output row t is available exactly when input row t is (delay 0, never compensated)",
            "fit_mode": "FROZEN_PARAMETERS_FIT_ON_TRAIN_ROWS_ONLY (closed form, no optimiser)"}


def verify_artifact(a) -> dict:
    if not isinstance(a, dict) or a.get("schema") != ARTIFACT_SCHEMA:
        raise OperatorRefusal("unknown artifact schema")
    body = {k: a[k] for k in a if k not in ("artifact_sha256", "environment")}
    validate_spec(a["spec"])
    if a["meta"] != _meta(a["spec"]["kind"]):
        raise OperatorRefusal("artifact meta differs from the kind's declared meta")
    if a["code_sha256"] != code_sha256():
        raise OperatorRefusal("artifact was produced by different operator code")
    try:
        d = _sha(body)
    except (TypeError, ValueError) as exc:
        raise OperatorRefusal(f"artifact is not canonical JSON: {exc}")
    if d != a["artifact_sha256"]:
        raise OperatorRefusal("artifact digest does not re-derive (mutated artifact)")
    if a["status"] == "FITTED" and a["fitted_state_digest"] != _sha(a["fitted"]["per_column"]):
        raise OperatorRefusal("fitted state digest does not re-derive")
    return a


def _usable(a: dict) -> dict:
    verify_artifact(a)
    if a["status"] != "FITTED":
        raise OperatorAbstain(f"artifact is a typed ABSTAIN: {a['abstain_reason']}")
    return a


# ----------------------------------------------------------------------------------------------
# filter core (scalar, pure python)
# ----------------------------------------------------------------------------------------------
NAN = float("nan")


def _new_col_state(kind: str) -> dict:
    s = {"row": 0, "n_obs": 0, "started": False, "level": 0.0, "var": 0.0}
    if kind == LOCAL_LINEAR_TREND:
        s.update({"slope": 0.0, "p11": 0.0, "p12": 0.0, "p22": 0.0})
    return s


def _step_ll(st: dict, c: dict, x: float, warm: int):
    st["row"] += 1
    if x != x:
        if st["started"]:
            st["var"] = st["var"] + c["q"]
            return (NAN, st["level"], NAN, NAN, st["var"]), MISSING_INPUT
        return (NAN, NAN, NAN, NAN, NAN), MISSING_INPUT
    st["n_obs"] += 1
    if not st["started"]:
        st["started"] = True
        st["level"], st["var"] = x, c["p0"]
        return (x, x, 0.0, 0.0, st["var"]), WARMUP
    vp = st["var"] + c["q"]
    f = vp + c["r"]
    v = x - st["level"]
    k = vp / f
    st["level"] = st["level"] + k * v
    st["var"] = (1.0 - k) * vp
    z = v / math.sqrt(f)
    return (x, st["level"], v, z, st["var"]), (WARMUP if st["n_obs"] <= warm else AVAILABLE)


def _step_llt(st: dict, c: dict, x: float, warm: int):
    st["row"] += 1
    if x != x:
        if st["started"]:
            a1 = st["level"] + st["slope"]
            q11 = st["p11"] + 2.0 * st["p12"] + st["p22"] + c["q"]
            q12 = st["p12"] + st["p22"]
            q22 = st["p22"] + c["qs"]
            st["level"], st["p11"], st["p12"], st["p22"] = a1, q11, q12, q22
            return (NAN, a1, st["slope"], NAN, NAN, q11, q22), MISSING_INPUT
        return (NAN,) * 7, MISSING_INPUT
    st["n_obs"] += 1
    if not st["started"]:
        st["started"] = True
        st["level"], st["slope"] = x, 0.0
        st["p11"], st["p12"], st["p22"] = c["p0"], 0.0, c["p0"]
        return (x, x, 0.0, 0.0, 0.0, st["p11"], st["p22"]), WARMUP
    a1 = st["level"] + st["slope"]
    q11 = st["p11"] + 2.0 * st["p12"] + st["p22"] + c["q"]
    q12 = st["p12"] + st["p22"]
    q22 = st["p22"] + c["qs"]
    f = q11 + c["r"]
    v = x - a1
    k1, k2 = q11 / f, q12 / f
    st["level"] = a1 + k1 * v
    st["slope"] = st["slope"] + k2 * v
    st["p11"] = q11 - k1 * q11
    st["p12"] = q12 - k1 * q12
    st["p22"] = q22 - k2 * q12
    z = v / math.sqrt(f)
    return (x, st["level"], st["slope"], v, z, st["p11"], st["p22"]), (WARMUP if st["n_obs"] <= warm else AVAILABLE)


def _stepper(kind):
    return _step_ll if kind == LOCAL_LEVEL else _step_llt


# ----------------------------------------------------------------------------------------------
# output container
# ----------------------------------------------------------------------------------------------
@dataclass
class KalmanOutput:
    """Result of the FORWARD causal filter. Only this type may feed an arm."""
    kind: str
    column_ids: list
    arrays: dict                       # name -> (T, V) float64
    reasons: np.ndarray                # (T, V) <U18
    availability: object               # (T,) int64 or None: identical to the input rows' availability (delay 0)
    artifact_sha256: str
    causal_forward: bool = True
    eligible_for_inputs: bool = True
    covariance_semantics: str = COVARIANCE_SEMANTICS

    def digest(self) -> str:
        h = hashlib.sha256()
        for name in sorted(self.arrays):
            h.update(name.encode())
            h.update(np.ascontiguousarray(self.arrays[name], dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.reasons).tobytes())
        if self.availability is not None:
            h.update(np.ascontiguousarray(self.availability, dtype="<i8").tobytes())
        return h.hexdigest()

    def reason_counts(self) -> dict:
        return {r: int((self.reasons == r).sum()) for r in REASONS}

    def eligible_mask(self) -> np.ndarray:
        return self.reasons == AVAILABLE


@dataclass
class NonCausalControlOutput:
    """Smoothed (backward) series: a REJECTION control. It carries no eligibility, ever."""
    kind: str
    column_ids: list
    arrays: dict
    reasons: np.ndarray
    artifact_sha256: str
    causal_forward: bool = False
    eligible_for_inputs: bool = False
    label: str = "NON_CAUSAL_NEGATIVE_CONTROL"


def eligible_matrix(out, names=None, outputs=None):
    """Stack chosen outputs into a (T, V*k) matrix and column names ``<col>__kf_<output>``.

    The ONLY gate to arm inputs: refuses anything that is not a forward causal KalmanOutput."""
    if isinstance(out, NonCausalControlOutput) or not isinstance(out, KalmanOutput) or not out.causal_forward \
            or not out.eligible_for_inputs:
        raise OperatorRefusal("this output is not an eligible input: only the forward causal filter produces "
                              "eligible inputs; the backward smoother is a non-causal control")
    outputs = list(outputs or [o for o in OUTPUTS[out.kind] if o != "obs"])
    for o in outputs:
        if o not in out.arrays:
            raise OperatorRefusal(f"output {o} not produced by {out.kind}")
    cols = list(range(len(out.column_ids))) if names is None else [out.column_ids.index(n) for n in names]
    mats, labels = [], []
    for j in cols:
        for o in outputs:
            mats.append(out.arrays[o][:, j])
            labels.append(f"{out.column_ids[j]}__kf_{o}")
    return np.stack(mats, axis=1), labels


# ----------------------------------------------------------------------------------------------
# batch, state, step
# ----------------------------------------------------------------------------------------------
def _avail_check(availability, T):
    if availability is None:
        return None
    a = np.asarray(availability)
    if a.dtype.kind not in "iu" or a.shape != (T,):
        raise OperatorRefusal("availability must be an integer vector with one entry per row")
    if T > 1 and not (np.diff(a.astype(np.int64)) > 0).all():
        raise OperatorRefusal("availability timestamps must strictly increase")
    return a.astype(np.int64)


def init_state(a: dict, n_columns: int | None = None) -> dict:
    _usable(a)
    V = a["n_columns"]
    if n_columns is not None and n_columns != V:
        raise OperatorRefusal("stream width differs from the fitted width")
    kind = a["spec"]["kind"]
    return {"schema": STATE_SCHEMA, "artifact_sha256": a["artifact_sha256"], "next_row": 0,
            "columns": [_new_col_state(kind) for _ in range(V)]}


def _check_state(a: dict, state: dict) -> None:
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA or \
            state.get("artifact_sha256") != a["artifact_sha256"]:
        raise OperatorRefusal("state is not bound to this artifact")
    if len(state["columns"]) != a["n_columns"]:
        raise OperatorRefusal("state width differs from the artifact")


def step(a: dict, state: dict, row) -> tuple:
    """One tick. Returns (outputs, reasons, new_state); state is mutated in place and returned."""
    _usable(a)
    _check_state(a, state)
    kind, warm = a["spec"]["kind"], a["spec"]["params"]["warmup"]
    r = np.asarray(row)
    if r.ndim != 1 or r.shape[0] != a["n_columns"] or r.dtype.kind not in "fiu":
        raise OperatorRefusal("row must be a real vector of the fitted width")
    vals = r.astype(np.float64).tolist()
    if any(math.isinf(v) for v in vals):
        raise OperatorRefusal("row holds +/-inf")
    fn = _stepper(kind)
    outs, reasons = [], []
    for j, v in enumerate(vals):
        o, why = fn(state["columns"][j], a["fitted"]["per_column"][j], v, warm)
        outs.append(o)
        reasons.append(why)
    state["next_row"] += 1
    return outs, reasons, state


def transform_batch(a: dict, X, availability=None, state: dict | None = None, *, _return_state: bool = False):
    """Forward causal filter over (T, V). Output row t depends only on rows <= t."""
    _usable(a)
    x = _real_matrix(X, "transform matrix", allow_nan=True)
    T, V = x.shape
    if V != a["n_columns"]:
        raise OperatorRefusal("matrix width differs from the fitted width")
    avail = _avail_check(availability, T)
    kind, warm = a["spec"]["kind"], a["spec"]["params"]["warmup"]
    names = OUTPUTS[kind]
    if state is None:
        state = init_state(a)
    else:
        _check_state(a, state)
    fn = _stepper(kind)
    arrays = {n: np.empty((T, V), dtype=np.float64) for n in names}
    reasons = np.empty((T, V), dtype=_REASON_DTYPE)
    cols = x.T.tolist()
    for j in range(V):
        c, st = a["fitted"]["per_column"][j], state["columns"][j]
        buf = [[0.0] * T for _ in names]
        rj = [None] * T
        for t in range(T):
            o, why = fn(st, c, cols[j][t], warm)
            for k in range(len(names)):
                buf[k][t] = o[k]
            rj[t] = why
        for k, n in enumerate(names):
            arrays[n][:, j] = buf[k]
        reasons[:, j] = rj
    state["next_row"] += T
    out = KalmanOutput(kind=kind, column_ids=list(a["fit_binding"]["column_ids"]), arrays=arrays, reasons=reasons,
                       availability=avail, artifact_sha256=a["artifact_sha256"])
    return (out, state) if _return_state else out


def transform_chunk(a: dict, state: dict, X, availability=None):
    """Continue a stream from a durable state over a contiguous later chunk."""
    return transform_batch(a, X, availability, state=state, _return_state=True)


def state_digest(state: dict) -> str:
    return _sha(state)


def save_state(state: dict) -> bytes:
    body = copy.deepcopy(state)
    return _canonical({"state": body, "state_sha256": _sha(body)})


def load_state(blob: bytes, a: dict) -> dict:
    _usable(a)
    try:
        doc = json.loads(blob.decode())
    except Exception as exc:
        raise OperatorRefusal(f"state blob is not JSON: {exc}")
    if not isinstance(doc, dict) or set(doc) != {"state", "state_sha256"} or _sha(doc["state"]) != doc["state_sha256"]:
        raise OperatorRefusal("state digest does not re-derive (mutated or truncated state)")
    _check_state(a, doc["state"])
    return doc["state"]


# ----------------------------------------------------------------------------------------------
# controls
# ----------------------------------------------------------------------------------------------
def identity_control(X) -> np.ndarray:
    """Identity control: the original observation, unchanged (a copy, bitwise equal)."""
    return np.array(_real_matrix(X, "identity input", allow_nan=True), copy=True)


def steady_state_gain(lam: float) -> float:
    """Steady-state gain of the local level filter, K solves K^2/(1-K) = lam = q/r."""
    return (-lam + math.sqrt(lam * lam + 4.0 * lam)) / 2.0


def ewma_comparable(a: dict, X) -> dict:
    """Causal EWMA whose smoothing constant equals the Kalman local level's steady-state gain per column
    (a comparator with the same effective bandwidth and none of the adaptive state). Output: level, innov."""
    _usable(a)
    x = _real_matrix(X, "ewma input", allow_nan=True)
    T, V = x.shape
    if V != a["n_columns"]:
        raise OperatorRefusal("matrix width differs from the fitted width")
    lvl = np.empty((T, V))
    inn = np.empty((T, V))
    gains = []
    for j in range(V):
        c = a["fitted"]["per_column"][j]
        k = steady_state_gain(c["q"] / c["r"])
        gains.append(k)
        level, started = 0.0, False
        for t, v in enumerate(x[:, j].tolist()):
            if v != v:
                lvl[t, j] = level if started else NAN
                inn[t, j] = NAN
                continue
            if not started:
                level, started = v, True
                lvl[t, j], inn[t, j] = v, 0.0
                continue
            e = v - level
            level = level + k * e
            lvl[t, j], inn[t, j] = level, e
    return {"level": lvl, "innov": inn, "gains": gains, "causal": True}


def permutation_control(Z: np.ndarray, boundaries, seed: int) -> np.ndarray:
    """Destroy the temporal/target relation of a block of output columns with EQUAL capacity: rows are
    permuted jointly (each column keeps its exact marginal) inside each declared partition separately,
    so no row of one partition ever carries values of another."""
    Z = np.asarray(Z, dtype=np.float64)
    out = np.empty_like(Z)
    rng = np.random.RandomState(seed)
    lo = 0
    for hi in list(boundaries) + [Z.shape[0]]:
        if hi <= lo:
            raise OperatorRefusal("partition boundaries must be strictly increasing inside the matrix")
        out[lo:hi] = Z[lo:hi][rng.permutation(hi - lo)]
        lo = hi
    return out


def noise_control(Z: np.ndarray, fit_rows: int, seed: int) -> np.ndarray:
    """Gaussian noise with the per-column mean/std of the first ``fit_rows`` rows (TRAIN only): same capacity,
    no information. Uses the legacy RandomState stream (stable across numpy versions)."""
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z[:fit_rows].mean(axis=0)
    sd = Z[:fit_rows].std(axis=0)
    rng = np.random.RandomState(seed)
    return mu[None, :] + sd[None, :] * rng.standard_normal(Z.shape)


def smoother_control(a: dict, X) -> NonCausalControlOutput:
    """Backward (Rauch-Tung-Striebel) smoother with the FITTED parameters. NON-CAUSAL: every smoothed value at t
    uses rows > t. It exists to be rejected; its output type has no eligibility and ``eligible_matrix`` refuses it.
    NaN is refused (a control is run on complete series)."""
    _usable(a)
    x = _real_matrix(X, "smoother input", allow_nan=False)
    T, V = x.shape
    if V != a["n_columns"]:
        raise OperatorRefusal("matrix width differs from the fitted width")
    kind = a["spec"]["kind"]
    names = ("level", "state_var") if kind == LOCAL_LEVEL else ("level", "slope", "state_var", "slope_var")
    arrays = {n: np.empty((T, V)) for n in names}
    for j in range(V):
        c = a["fitted"]["per_column"][j]
        y = x[:, j].tolist()
        if kind == LOCAL_LEVEL:
            lev, var, plev, pvar = [0.0] * T, [0.0] * T, [0.0] * T, [0.0] * T
            lev[0], var[0] = y[0], c["p0"]
            for t in range(1, T):
                pvar[t] = var[t - 1] + c["q"]
                plev[t] = lev[t - 1]
                f = pvar[t] + c["r"]
                kk = pvar[t] / f
                lev[t] = plev[t] + kk * (y[t] - plev[t])
                var[t] = (1.0 - kk) * pvar[t]
            sl, sv = lev[:], var[:]
            for t in range(T - 2, -1, -1):
                jg = var[t] / pvar[t + 1]
                sl[t] = lev[t] + jg * (sl[t + 1] - plev[t + 1])
                sv[t] = var[t] + jg * jg * (sv[t + 1] - pvar[t + 1])
            arrays["level"][:, j], arrays["state_var"][:, j] = sl, sv
        else:
            F = np.array([[1.0, 1.0], [0.0, 1.0]])
            Q = np.array([[c["q"], 0.0], [0.0, c["qs"]]])
            xf, Pf, xp, Pp = [None] * T, [None] * T, [None] * T, [None] * T
            xf[0] = np.array([y[0], 0.0])
            Pf[0] = np.array([[c["p0"], 0.0], [0.0, c["p0"]]])
            for t in range(1, T):
                xp[t] = F @ xf[t - 1]
                Pp[t] = F @ Pf[t - 1] @ F.T + Q
                f = Pp[t][0, 0] + c["r"]
                kg = Pp[t][:, 0] / f
                xf[t] = xp[t] + kg * (y[t] - xp[t][0])
                Pf[t] = Pp[t] - np.outer(kg, Pp[t][0, :])
            xs, Ps = xf[:], Pf[:]
            for t in range(T - 2, -1, -1):
                jg = Pf[t] @ F.T @ np.linalg.inv(Pp[t + 1])
                xs[t] = xf[t] + jg @ (xs[t + 1] - xp[t + 1])
                Ps[t] = Pf[t] + jg @ (Ps[t + 1] - Pp[t + 1]) @ jg.T
            arrays["level"][:, j] = [s[0] for s in xs]
            arrays["slope"][:, j] = [s[1] for s in xs]
            arrays["state_var"][:, j] = [p[0, 0] for p in Ps]
            arrays["slope_var"][:, j] = [p[1, 1] for p in Ps]
    reasons = np.full((T, V), FUTURE_UNAVAILABLE, dtype=_REASON_DTYPE)
    return NonCausalControlOutput(kind=SMOOTHER_CONTROL_KIND, column_ids=list(a["fit_binding"]["column_ids"]),
                                  arrays=arrays, reasons=reasons, artifact_sha256=a["artifact_sha256"])


# ----------------------------------------------------------------------------------------------
# cost and latency (measured, never assumed)
# ----------------------------------------------------------------------------------------------
def measure_cost(a: dict, X, repeats: int = 3) -> dict:
    """CPU seconds, wall seconds and per-tick latency of batch and tick-by-tick paths on X."""
    import resource
    _usable(a)
    x = _real_matrix(X, "cost input", allow_nan=True)
    rows = x.shape[0]
    best_batch_cpu, best_batch_wall = None, None
    for _ in range(repeats):
        c0, w0 = time.process_time(), time.perf_counter()
        transform_batch(a, x)
        c, w = time.process_time() - c0, time.perf_counter() - w0
        best_batch_cpu = c if best_batch_cpu is None else min(best_batch_cpu, c)
        best_batch_wall = w if best_batch_wall is None else min(best_batch_wall, w)
    st = init_state(a)
    lat = []
    for t in range(min(rows, 2000)):
        w0 = time.perf_counter_ns()
        step(a, st, x[t])
        lat.append(time.perf_counter_ns() - w0)
    lat.sort()
    return {"rows": int(rows), "columns": int(x.shape[1]), "batch_cpu_seconds_best": best_batch_cpu,
            "batch_wall_seconds_best": best_batch_wall,
            "batch_cpu_microseconds_per_row_per_column": 1e6 * best_batch_cpu / (rows * x.shape[1]),
            "tick_latency_ns_median": int(lat[len(lat) // 2]), "tick_latency_ns_p99": int(lat[int(0.99 * (len(lat) - 1))]),
            "tick_latency_note": "full row of all columns including artifact verification on every call",
            "max_rss_kib_process": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
