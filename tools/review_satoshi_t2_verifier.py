#!/usr/bin/env python3
"""Read-only witnesses for the peer's pinned T2 verifier, not an admission tool."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer-repo", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "-C", args.peer_repo, "show",
         "08371722eef13e359dc5a981eaa3e3e3b29c9f23:tools/expert_guided_clean_round.py"],
        text=True,
    )
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in {"label_mode_of", "verify_t2_arm"}]
    if len(functions) != 2:
        raise ValueError("PINNED_FUNCTIONS_MISSING")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "peer-pinned-verifier", "exec"), namespace)
    result = json.loads(Path(args.result).read_text())
    unit = json.loads(Path(args.units).read_text())["units"][0]
    verify = namespace["verify_t2_arm"]
    rows = []

    def check(name, candidate, expected_unit=unit):
        try:
            errors = verify(candidate, expected_unit)
            rows.append({"case": name, "accepted": not errors, "errors": errors})
        except Exception as exc:
            rows.append({"case": name, "accepted": False, "exception": type(exc).__name__})

    check("real_pinned_legacy_control", result)
    check("explicit_zigzag_control", dict(result, labels={"mode": "zigzag"}))
    check("explicit_opportunity_negative_control", dict(result, labels={"mode": "opportunity"}))
    check("unknown_mode_negative_control", dict(result, labels={"mode": "oracle"}))
    check("explicit_null_labels_not_absent", dict(result, labels=None))
    check("malformed_labels_type", dict(result, labels=[]))
    check("arbitrary_producer_missing_labels", dict(result, producer_commit="unknown-new-producer"))
    changed_unit = copy.deepcopy(unit)
    changed_unit.update(name="wrong_arm", identity="wrong_identity", list="shortlist_strict",
                        dropout=0.3, wd=0.01)
    check("same_result_wrong_list_regularization_identity", result, changed_unit)
    assert rows[0]["accepted"] and rows[1]["accepted"]
    assert not rows[2]["accepted"] and not rows[3]["accepted"]
    report = {"scope": "bounded read-only probes, no admission or science changes",
                      "peer_commit": "08371722eef13e359dc5a981eaa3e3e3b29c9f23",
                      "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "result_sha256": hashlib.sha256(Path(args.result).read_bytes()).hexdigest(),
                      "cases": rows}
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
