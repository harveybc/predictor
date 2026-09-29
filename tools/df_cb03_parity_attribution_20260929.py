"""CB03: attribute the framework's parity gap to one cause at a time.

`--- the ladder ---`

  N1  laya 0.2.1,  budget 1024/256, author envelope   the published-cell reproduction
  N2  laya 0.3.11, budget 1024/256, author envelope   + the SDK the framework pins
  N3  laya 0.3.11, budget  512/192, author envelope   + the budget the framework hard-codes
  N4  laya 0.3.11, budget  512/192, framework envelope + the state the framework serialises
  F1  our framework, through m5phet.runtime.run                the framework itself

Each rung adds exactly one change, so each step's label disagreement and maximum
probability difference belong to that change and to nothing else.  N4 against F1
is the only pair that holds everything constant, so it -- and only it -- measures
the wrapper's own fidelity.

  python tools/df_cb03_parity_attribution_20260929.py --dir <artifact dir> --out <json>
"""
import argparse
import json
import os


def rows_of(artifact):
    """position -> (predicted, probabilities), skipping anything not answered."""
    out = {}
    for row in artifact["per_row"]:
        if row.get("dropped") or row.get("refused"):
            continue
        if row.get("predicted") is None:
            continue
        out[row["i"]] = (row["predicted"], row["probabilities"])
    return out


def step(left_name, left, right_name, right, change):
    a, b = rows_of(left), rows_of(right)
    shared = sorted(set(a) & set(b))
    agree = sum(1 for i in shared if a[i][0] == b[i][0])
    max_abs = 0.0
    for i in shared:
        max_abs = max(max_abs, max(abs(x - y) for x, y in zip(a[i][1], b[i][1])))
    flips = [{"i": i, "from": a[i][0], "to": b[i][0]} for i in shared if a[i][0] != b[i][0]]
    return {
        "from": left_name, "to": right_name, "the_one_change": change,
        "rows_compared": len(shared),
        "rows_only_in_left": sorted(set(a) - set(b)),
        "rows_only_in_right": sorted(set(b) - set(a)),
        "label_agreement": agree,
        "label_agreement_fraction": round(agree / max(1, len(shared)), 6),
        "label_flips": len(flips),
        "max_abs_probability_difference": max_abs,
        "identical": agree == len(shared) and max_abs == 0.0,
        "flips": flips[:40],
    }


def near_tie_profile(artifact, reference):
    """Of the rows where two runs disagree, how close was the losing option?

    A disagreement on a row whose top two options are nearly tied is a different
    fact from a disagreement on a confident row, and averaging them would hide it."""
    a, b = rows_of(artifact), rows_of(reference)
    margins = []
    for i in sorted(set(a) & set(b)):
        if a[i][0] == b[i][0]:
            continue
        probs = sorted(b[i][1], reverse=True)
        margins.append(round(probs[0] - probs[1], 6))
    margins.sort()
    return {"disagreements": len(margins),
            "reference_top2_margin_min": margins[0] if margins else None,
            "reference_top2_margin_median": margins[len(margins) // 2] if margins else None,
            "reference_top2_margin_max": margins[-1] if margins else None,
            "margins": margins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def load(name):
        path = os.path.join(args.dir, name)
        return json.load(open(path)) if os.path.isfile(path) else None

    n1 = load("native_published400.json")
    n2 = load("v_sdk0311_ckptbudget_author.json")
    n3 = load("v_sdk0311_fwbudget_author.json")
    n4 = load("v_sdk0311_fwbudget_fwenvelope.json")
    f1 = load("parity_400.json")
    if n1 is None or f1 is None:
        raise SystemExit("REFUSED: the reproduction and the framework run are both required")

    def accuracy(artifact, key="metrics"):
        if artifact is None:
            return None
        if key in artifact:
            return artifact[key]["accuracy"]
        return artifact.get("framework_accuracy_on_answered")

    ladder = []
    for left_name, left, right_name, right, change in (
        ("N1_laya021_ckptbudget_author", n1, "N2_laya0311_ckptbudget_author", n2,
         "the SDK version the framework pins (0.2.1 -> 0.3.11)"),
        ("N2_laya0311_ckptbudget_author", n2, "N3_laya0311_fwbudget_author", n3,
         "the sequence budget the framework hard-codes (1024/256 -> 512/192)"),
        ("N3_laya0311_fwbudget_author", n3, "N4_laya0311_fwbudget_fwenvelope", n4,
         "the state envelope the framework serialises ({article} -> {asset,body,headline})"),
        ("N4_laya0311_fwbudget_fwenvelope", n4, "F1_m5phet_runtime", f1,
         "the framework itself: provider, runtime, contract validation, per-row calls"),
    ):
        if left is None or right is None:
            ladder.append({"from": left_name, "to": right_name, "the_one_change": change,
                           "status": "NOT_MEASURED"})
            continue
        ladder.append(step(left_name, left, right_name, right, change))

    out = {
        "schema": "cb03_parity_attribution.v1",
        "accuracies": {
            "N1_laya021_ckptbudget_author": accuracy(n1),
            "N2_laya0311_ckptbudget_author": accuracy(n2),
            "N3_laya0311_fwbudget_author": accuracy(n3),
            "N4_laya0311_fwbudget_fwenvelope": accuracy(n4),
            "F1_m5phet_runtime": f1.get("framework_accuracy_on_answered"),
        },
        "end_to_end": step("N1_laya021_ckptbudget_author", n1, "F1_m5phet_runtime", f1,
                           "everything the framework changes, together"),
        "ladder": ladder,
        "framework_disagreements_against_the_reproduction": near_tie_profile(f1, n1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    printable = json.loads(json.dumps(out))
    for entry in printable["ladder"]:
        entry.pop("flips", None)
    printable["end_to_end"].pop("flips", None)
    printable["framework_disagreements_against_the_reproduction"].pop("margins", None)
    print(json.dumps(printable, indent=1))


if __name__ == "__main__":
    main()
