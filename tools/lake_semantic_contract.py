#!/usr/bin/env python3
"""C91 (order 2026-09-12): a semantic contract for every measured column,
derived and published BEFORE any statistic is recomputed.

The previous round measured `announcement_datetime_local_utc` — an int64
column holding the minimum-int64 sentinel of a missing datetime — as a
finite number: mean -9.22e18, missingness 0, entropy 0. Nothing had
established what the column MEANT before it was averaged.

For each column this module records the physical Arrow type, the logical
type, unit, role and license the census declares, the null/sentinel
policy and where it came from, the physical null count and the counts of
declared and candidate sentinels. It then assigns exactly one state:

  NON_NUMERIC               the physical type is not integer or float;
  SEMANTIC_TYPE_UNRESOLVED  the physical type is numeric but the column
                            is named as a time, date or identifier — a
                            number stored for a time is not a measurement
                            until a contract says how to read it;
  MISSING_POLICY_UNRESOLVED numeric, but it holds integer extreme
                            sentinels or infinities and no policy declares
                            what they mean. An extreme integer is NEVER
                            declared null by this module: without a
                            contract it stays unresolved;
  NUMERIC_MEASURABLE        numeric, not named as time or identifier, and
                            no undeclared sentinel or infinity present.

The census declares physical_type, unit, role and license as UNKNOWN for
every variable, so NUMERIC_MEASURABLE rests on the physical type alone and
says so. Nothing here converts timestamps, categories or identifiers to
float64.
"""
from __future__ import annotations

import re

import numpy as np

CONTRACT = "crispdm.lake_semantic_contract.v1"
NUMERIC_MEASURABLE = "NUMERIC_MEASURABLE"
NON_NUMERIC = "NON_NUMERIC"
SEMANTIC_TYPE_UNRESOLVED = "SEMANTIC_TYPE_UNRESOLVED"
MISSING_POLICY_UNRESOLVED = "MISSING_POLICY_UNRESOLVED"
STATES = (NUMERIC_MEASURABLE, NON_NUMERIC, SEMANTIC_TYPE_UNRESOLVED,
          MISSING_POLICY_UNRESOLVED)

#: names that denote a time, a date or an identifier. Declared, and
#: conservative in the direction of refusal: a false match costs a
#: measurement, never admits a meaningless one.
TEMPORAL_OR_ID_NAME = re.compile(
    r"(^|_)(date|datetime|time|timestamp|ts|epoch|id|uuid)(_|$)", re.I)

INTEGER_SENTINELS = {
    "INT64_MIN": np.iinfo(np.int64).min, "INT64_MAX": np.iinfo(np.int64).max,
    "INT32_MIN": np.iinfo(np.int32).min, "INT32_MAX": np.iinfo(np.int32).max,
}

#: no policy exists in the census today; the hook is here so a declared
#: policy can be consumed, never guessed.
DECLARED_POLICIES: dict = {}


def column_contract(column_array, column: str, variable: dict,
                    rows_used: int) -> dict:
    import pyarrow as pa
    arr = column_array.slice(0, rows_used)
    ty = arr.type
    declared = {k: variable.get(k, "UNDECLARED")
                for k in ("physical_type", "unit", "role", "license",
                          "semantics")}
    policy = DECLARED_POLICIES.get(variable.get("variable_id"))
    out = {
        "contract": CONTRACT,
        "column": column,
        "arrow_type": str(ty),
        "declared_logical_type": declared["physical_type"],
        "declared_unit": declared["unit"],
        "declared_role": declared["role"],
        "declared_license": declared["license"],
        "declared_semantics": declared["semantics"],
        "null_policy": (policy or {"declared": None,
                                   "source": "NONE_DECLARED_IN_CENSUS"}),
        "rows_evaluated": int(len(arr)),
        "physical_nulls": int(arr.null_count),
        "declared_sentinel_counts": {},
        "sentinel_candidate_counts": {},
        "infinite_count": 0,
        "reasons": [],
    }
    numeric = pa.types.is_integer(ty) or pa.types.is_floating(ty)
    if not numeric:
        out["state"] = NON_NUMERIC
        out["reasons"].append(f"physical type {ty} is not integer or float")
        return out
    if TEMPORAL_OR_ID_NAME.search(column):
        out["reasons"].append("numeric physical type for a column named as "
                              "a time, date or identifier")
    values = arr.drop_null().to_numpy(zero_copy_only=False)
    if pa.types.is_integer(ty):
        for label, s in INTEGER_SENTINELS.items():
            k = int(np.count_nonzero(values == s))
            if k:
                out["sentinel_candidate_counts"][label] = k
    else:
        out["infinite_count"] = int(np.count_nonzero(np.isinf(values)))
    if out["reasons"]:
        out["state"] = SEMANTIC_TYPE_UNRESOLVED
    elif (out["sentinel_candidate_counts"] or out["infinite_count"]) \
            and policy is None:
        out["state"] = MISSING_POLICY_UNRESOLVED
        out["reasons"].append("extreme integer sentinels or infinities are "
                              "present and no policy declares what they "
                              "mean")
    else:
        out["state"] = NUMERIC_MEASURABLE
        out["reasons"].append("numeric physical type; the census declares "
                              "logical type, unit and role as "
                              f"{declared['physical_type']}, so this rests "
                              "on the physical type alone")
    return out
