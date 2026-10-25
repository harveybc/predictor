"""Read-only synthetic witnesses against pinned producer functions; no fits/data access."""
import argparse
import ast
import json
import math
import subprocess
from types import SimpleNamespace as NS

PIN = "9742ccdf827008c0ff70c6b833246a46aab6747c"


def load_functions(repo, path, names, namespace):
    source = subprocess.check_output(
        ["git", "-C", repo, "show", f"{PIN}:{path}"], text=True
    )
    tree = ast.parse(source)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, f"{PIN}:{path}", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    # Only scalar operations used by the selected producer functions are needed.
    namespace = {"np": NS(isfinite=math.isfinite, clip=lambda x, lo, hi: min(hi, max(lo, x)), nan=math.nan),
                 "NOP": 0, "OPEN_L": 1, "OPEN_S": 2, "E": NS(CLOSE="CLOSE", HOLD="HOLD")}
    load_functions(args.repo, "app/short_horizon/expert_guided/multires.py",
                   {"origin_label", "close_label"}, namespace)
    load_functions(args.repo, "tools/expert_guided_e3m.py", {"row_weight"}, namespace)
    th = NS(min_votes=1, theta_f=.5, theta_mae=.5, theta_vol=.25, theta_close=.1, f_close_take=.9)
    tb = NS(votes={1: [1], -1: [0]}, legs={"f_open": {1: [1.0], -1: [-1.0]},
            "mae": {1: [.8], -1: [.8]}, "leg_sigma": [10.0]})
    lab = namespace["origin_label"](tb, 0, th)
    weight = namespace["row_weight"](NS(weight=lambda b, r: lab[1]), "multires", True, None, 0, 0)
    assert lab[0] == 0 and lab[3] == "nop_graded" and weight == 0
    findings = {"risk_rejected_nop": {"mae": .8, "limit": .5, "f_open": 1.0, "loss_weight": weight}}
    tb.votes[1][0] = 0
    tb.legs["mae"][1][0] = .1
    absent = namespace["origin_label"](tb, 0, th)
    tb.votes[1][0] = 1
    present = namespace["origin_label"](tb, 0, th)
    assert absent[0] == 0 and present[0] == 1
    findings["no_vote_not_economic_negative"] = {"same_economic_quantities": True,
                                                 "without_vote": absent[3], "with_vote": present[3]}
    tb.legs.update(f_close={1: [.95]}, remaining={1: [.05]})
    assert namespace["close_label"](tb, 0, 1, th)[0] == "CLOSE"
    findings["close_reference"] = "close_label uses precomputed close-price fitness, not next-open fill"
    print(json.dumps({"pin": PIN, "scope": "isolated producer-function witnesses; not end-to-end replay",
                      "witnesses": findings}, indent=2))


if __name__ == "__main__":
    main()
