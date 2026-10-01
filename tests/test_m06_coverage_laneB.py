import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import m06_coverage_laneB as C  # noqa: E402


def test_counts_are_derived_from_the_files_not_typed():
    den = json.dumps({"old": {"distinct_rows": 10, "covered": 3}, "new": {"files": 7}, "note": "n"}).encode()
    led = b"state,evaluated,selected\nprofiled,False,False\nprofiled,False,False\nexcluded,False,False\n"
    src = b"provider,status,point_in_time\nA,PROFILED,NOT_ADMISSIBLE: no availability contract\nB,RETAINED_BYTES,POINT_IN_TIME_ADMISSIBLE\n"
    dag = json.dumps({"dag_sha256": "x", "denominators": {"v3_sources": 2}, "not_claimed": []}).encode()
    o = C.build(den, led, src, dag, {k: "r@c:p" for k in ("denominators", "transform_ledger", "source_table", "feature_dag")})
    assert o["transform_ledger"]["by_state"] == {"profiled": 2, "excluded": 1}
    assert o["sources"]["point_in_time_admissible"] == ["B"] and o["new_denominator"] == {"files": 7}
    assert o["inputs"]["source_table"]["sha256"] == C.sha(src)
