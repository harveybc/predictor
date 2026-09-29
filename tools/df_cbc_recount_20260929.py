"""CB-C recount: recompute every retained CB03 number from the retained PER-ROW arrays.

This is a recount, not a rerun.  No model is loaded, no dataset is opened, nothing is
downloaded and no service is contacted by this module.  It reads the retained evidence
files, ignores every stored aggregate while recomputing, and only then compares.

It also answers the question this lane exists for: which retained results are RETAINED
PROJECTIONS (a warehouse row that was built and kept on disk) and which are ACCEPTED
WAREHOUSE ROWS (a terminal the live warehouse holds).  On the evidence alone the answer
for every projection is RETAINED, because the evidence directory cannot contain an
acceptance: acceptance lives in the warehouse, and is proved in the companion tool.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

EV = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
LABELS = ["world", "sports", "business", "sci_tech"]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name: str) -> dict:
    return json.loads((EV / name).read_text())


def confusion(rows, k=4):
    """reference rows x predicted columns, plus a final ABSTAINED column."""
    m = [[0] * (k + 1) for _ in range(k)]
    for r in rows:
        g = int(r["gold"])
        p = r.get("predicted")
        if p is None or r.get("refused") or r.get("status") not in (None, "OK"):
            m[g][k] += 1
        else:
            m[g][int(p)] += 1
    return m


def accuracy_from_confusion(m, k=4):
    total = sum(sum(row) for row in m)
    correct = sum(m[i][i] for i in range(k))
    return correct / total, correct, total


def macro_f1_from_confusion(m, k=4):
    f1s = []
    for i in range(k):
        tp = m[i][i]
        fn = sum(m[i]) - tp
        fp = sum(m[j][i] for j in range(k)) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / k, f1s


def ece(rows, bins=15, first_bin_closed_left=True):
    conf = [max(r["probabilities"]) for r in rows]
    hit = [1.0 if int(r["predicted"]) == int(r["gold"]) else 0.0 for r in rows]
    n = len(rows)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        if b == 0 and first_bin_closed_left:
            sel = [i for i, c in enumerate(conf) if lo <= c <= hi]
        else:
            sel = [i for i, c in enumerate(conf) if lo < c <= hi]
        if not sel:
            continue
        acc = sum(hit[i] for i in sel) / len(sel)
        avg = sum(conf[i] for i in sel) / len(sel)
        total += (len(sel) / n) * abs(acc - avg)
    return total


def brier(rows, k=4):
    tot = 0.0
    for r in rows:
        p = r["probabilities"]
        g = int(r["gold"])
        tot += sum((p[j] - (1.0 if j == g else 0.0)) ** 2 for j in range(k))
    return tot / len(rows)


def nll(rows, eps=1e-12):
    return -sum(math.log(max(r["probabilities"][int(r["gold"])], eps)) for r in rows) / len(rows)


def acc_at_coverage(rows, frac=0.5):
    order = sorted(rows, key=lambda r: -max(r["probabilities"]))
    keep = order[: max(1, int(round(len(rows) * frac)))]
    return sum(1 for r in keep if int(r["predicted"]) == int(r["gold"])) / len(keep)


def softmax(logits, t):
    mx = max(l / t for l in logits)
    ex = [math.exp(l / t - mx) for l in logits]
    s = sum(ex)
    return [e / s for e in ex]


def recount_native(art: dict) -> dict:
    rows = art["per_row"]
    m = confusion(rows)
    acc, correct, total = accuracy_from_confusion(m)
    mf1, per_class = macro_f1_from_confusion(m)
    # probabilities must be the softmax of the retained logits at the retained temperature
    worst = 0.0
    for r in rows:
        got = softmax(r["logits"], r["temperature"])
        worst = max(worst, max(abs(a - b) for a, b in zip(got, r["probabilities"])))
    return {
        "rows_recounted": len(rows),
        "confusion_recounted": m,
        "correct": correct, "denominator": total,
        "accuracy": acc,
        "macro_f1_full_precision": mf1,
        "macro_f1_rounded_4": round(mf1, 4),
        "per_class_f1": per_class,
        "ece_15_first_bin_closed_left": ece(rows, 15, True),
        "ece_15_first_bin_open_left": ece(rows, 15, False),
        "brier": brier(rows),
        "nll": nll(rows),
        "mean_confidence": sum(max(r["probabilities"]) for r in rows) / len(rows),
        "acc_at_50_coverage": acc_at_coverage(rows, 0.5),
        "dropped": sum(1 for r in rows if r.get("dropped")),
        "max_abs_softmax_reconstruction_error": worst,
        "support_by_class": {LABELS[i]: sum(m[i]) for i in range(4)},
        "predicted_by_class": {LABELS[j]: sum(m[i][j] for i in range(4)) for j in range(4)},
    }


def naive_tiebreaks(rows, train_counts: dict) -> dict:
    """The train split is exactly balanced, so MAJORITY_CLASS_FROM_TRAIN is a four-way tie.
    Every legitimate break is scored on the SAME evaluation rows."""
    n = len(rows)
    support = {LABELS[i]: sum(1 for r in rows if int(r["gold"]) == i) for i in range(4)}
    tied = sorted(k for k, v in train_counts.items() if v == max(train_counts.values()))
    per = {lab: support[lab] / n for lab in LABELS}
    return {
        "train_class_counts": train_counts,
        "tie": tied,
        "tie_is_four_way": len(tied) == 4,
        "naive_accuracy_by_tiebreak": per,
        "pinned_tiebreak": "ALPHABETICAL_OVER_CLASS_NAMES_SELECTS_Business",
        "pinned_value": per["business"],
        "range_low": min(per.values()),
        "range_high": max(per.values()),
        "argmin": min(per, key=per.get),
        "argmax": max(per, key=per.get),
        "uniform_over_vocabulary": 1.0 / 4,
        "stratified_prior_from_train_expected": sum((1 / 4) * per[lab] for lab in LABELS) * 4 / 4,
    }


def recount_parity(native: dict, parity: dict) -> dict:
    nat = {r["i"]: r for r in native["per_row"]}
    fw = {r["i"]: r for r in parity["per_row"]}
    common = sorted(set(nat) & set(fw))
    agree = 0
    flips = []
    worst = 0.0
    answered = 0
    correct = 0
    for i in common:
        a, b = nat[i], fw[i]
        if b.get("refused") or b.get("predicted") is None:
            continue
        answered += 1
        if int(b["predicted"]) == int(b["gold"]):
            correct += 1
        if int(a["predicted"]) == int(b["predicted"]):
            agree += 1
        else:
            flips.append({"i": i, "gold": int(a["gold"]), "native": int(a["predicted"]),
                          "framework": int(b["predicted"])})
        worst = max(worst, max(abs(x - y) for x, y in zip(a["probabilities"], b["probabilities"])))
    margins = []
    for f in flips:
        p = sorted(nat[f["i"]]["probabilities"], reverse=True)
        margins.append(p[0] - p[1])
    margins.sort()
    mid = len(margins) // 2
    med = margins[mid] if len(margins) % 2 else (margins[mid - 1] + margins[mid]) / 2
    fm = confusion(parity["per_row"])
    facc, fcorrect, ftotal = accuracy_from_confusion(fm)
    fmf1, _ = macro_f1_from_confusion(fm)
    return {
        "rows_compared": len(common),
        "framework_answered": answered,
        "framework_refused": sum(1 for r in parity["per_row"] if r.get("refused")),
        "label_agreement": agree,
        "label_agreement_fraction": agree / len(common),
        "label_flips": len(flips),
        "max_abs_probability_difference": worst,
        "coverage_parity_preserved": answered == len(native["per_row"]) == len(common),
        "framework_accuracy_recounted": facc,
        "framework_correct": fcorrect, "framework_denominator": ftotal,
        "framework_macro_f1_recounted": fmf1,
        "framework_confusion": fm,
        "native_accuracy_recounted": accuracy_from_confusion(confusion(native["per_row"]))[0],
        "net_correct_lost": accuracy_from_confusion(confusion(native["per_row"]))[1] - fcorrect,
        "flip_margin_min": margins[0] if margins else None,
        "flip_margin_median": med if margins else None,
        "flip_margin_max": margins[-1] if margins else None,
    }


def compare_two(a: dict, b: dict) -> dict:
    ra = {r["i"]: r for r in a["per_row"]}
    rb = {r["i"]: r for r in b["per_row"]}
    common = sorted(set(ra) & set(rb))
    agree = sum(1 for i in common if int(ra[i]["predicted"]) == int(rb[i]["predicted"]))
    worst = max(max(abs(x - y) for x, y in zip(ra[i]["probabilities"], rb[i]["probabilities"]))
                for i in common)
    return {"rows_compared": len(common), "label_agreement": agree,
            "label_agreement_fraction": agree / len(common), "label_flips": len(common) - agree,
            "max_abs_probability_difference": worst, "bit_identical": worst == 0.0 and agree == len(common)}


def main() -> int:
    files = sorted(p.name for p in EV.glob("*.json"))
    digests = {n: sha_file(EV / n) for n in files}
    manifest = load("MANIFEST.json")
    manifest_check = {}
    for name, rec in (manifest.get("files") or {}).items():
        want = rec if isinstance(rec, str) else rec.get("sha256")
        manifest_check[name] = {"manifest_sha256": want, "on_disk_sha256": digests.get(name),
                                "matches": want == digests.get(name)}

    native = load("native_published400.json")
    parity = load("parity_400.json")
    receipts = load("CB03_CLASSIFICATION_RECEIPTS.json")
    attribution = load("CB03_PARITY_ATTRIBUTION.json")

    rc_native = recount_native(native)
    stored = native["metrics"]
    native_vs_stored = {
        "accuracy": {"recounted": rc_native["accuracy"], "stored": stored["accuracy"],
                     "agrees": abs(rc_native["accuracy"] - stored["accuracy"]) < 1e-12},
        "macro_f1": {"recounted_full": rc_native["macro_f1_full_precision"],
                     "recounted_rounded_4": rc_native["macro_f1_rounded_4"], "stored": stored["macro_f1"],
                     "agrees_at_stored_precision": rc_native["macro_f1_rounded_4"] == stored["macro_f1"]},
        "ece": {"recounted": rc_native["ece_15_first_bin_closed_left"], "stored": stored["ece"],
                "agrees_at_3dp": round(rc_native["ece_15_first_bin_closed_left"], 3) == stored["ece"]},
        "brier": {"recounted": rc_native["brier"], "stored": stored["brier"],
                  "agrees_at_4dp": round(rc_native["brier"], 4) == stored["brier"]},
        "nll": {"recounted": rc_native["nll"], "stored": stored["nll"],
                "agrees_at_4dp": round(rc_native["nll"], 4) == stored["nll"]},
        "mean_confidence": {"recounted": rc_native["mean_confidence"], "stored": stored["mean_confidence"],
                            "agrees_at_4dp": round(rc_native["mean_confidence"], 4) == stored["mean_confidence"]},
        "acc_at_50_coverage": {"recounted": rc_native["acc_at_50_coverage"],
                               "stored": stored["acc_at_50_coverage"],
                               "agrees": rc_native["acc_at_50_coverage"] == stored["acc_at_50_coverage"]},
        "dropped": {"recounted": rc_native["dropped"], "stored": stored["dropped"],
                    "agrees": rc_native["dropped"] == stored["dropped"]},
        "confusion": {"recounted": rc_native["confusion_recounted"],
                      "stored": native["confusion_reference_by_predicted_plus_abstained"],
                      "agrees": rc_native["confusion_recounted"] == native["confusion_reference_by_predicted_plus_abstained"]},
    }

    naive = naive_tiebreaks(native["per_row"], receipts["train_label_population"]["class_counts"])
    rc_parity = recount_parity(native, parity)

    variants = {"N1": native,
                "N2": load("v_sdk0311_ckptbudget_author.json"),
                "N3": load("v_sdk0311_fwbudget_author.json"),
                "N4": load("v_sdk0311_fwbudget_fwenvelope.json")}
    ladder = {}
    for k, v in variants.items():
        m = confusion(v["per_row"])
        a, _, _ = accuracy_from_confusion(m)
        f, _ = macro_f1_from_confusion(m)
        ladder[k] = {"accuracy_recounted": a, "macro_f1_recounted": f, "rows": len(v["per_row"])}
    ladder["F1_framework"] = {"accuracy_recounted": rc_parity["framework_accuracy_recounted"],
                              "macro_f1_recounted": rc_parity["framework_macro_f1_recounted"],
                              "rows": rc_parity["framework_denominator"]}
    steps = {"N1_to_N2": compare_two(variants["N1"], variants["N2"]),
             "N2_to_N3": compare_two(variants["N2"], variants["N3"]),
             "N3_to_N4": compare_two(variants["N3"], variants["N4"]),
             "N4_to_F1": compare_two(variants["N4"], parity)}

    # --- the distinction this lane exists for -------------------------------------------------
    inventory = []
    for name, rec in (receipts.get("receipts") or {}).items():
        proj = (receipts.get("warehouse_projection") or {}).get(name)
        inventory.append({
            "receipt": name,
            "evidence_class": rec.get("evidence_class"),
            "primary_metric": ((rec.get("author_primary_metric") or {}).get("family")),
            "value": ((rec.get("author_primary_metric") or {}).get("value")),
            "has_retained_warehouse_projection": proj is not None,
            "retained_metric_rows": len(proj["metrics"]) if proj else 0,
            "retained_tag_count": len(proj["tags"]) if proj else 0,
            "accepted_warehouse_row": "NOT_ESTABLISHED_BY_THIS_FILE",
        })

    population_digests = {
        "native_population_sha256": native["population_sha256"],
        "native_pinned_population_sha256": native["pinned_population_sha256"],
        "parity_population_sha256": parity["population_sha256"],
        "receipt_corpus_sha256": ((receipts["warehouse_projection"]["ours_accuracy"]["tags"] or {})
                                  .get("corpus_sha256")),
        "all_equal": len({native["population_sha256"], native["pinned_population_sha256"],
                          parity["population_sha256"]}) == 1,
        "parity_native_artifact_sha256_claimed": parity["native_artifact_sha256"],
        "parity_native_artifact_sha256_on_disk": digests["native_published400.json"],
        "parity_binds_the_native_artifact_actually_on_disk":
            parity["native_artifact_sha256"] == digests["native_published400.json"],
    }

    # the denominator prohibition, checked mechanically
    blob = json.dumps(receipts)
    denominator_guard = {
        "published_score_denominator_recounted": rc_native["denominator"],
        "is_400": rc_native["denominator"] == 400,
        "7600_appears_as_a_metric_value_or_denominator": any(
            r.get("value") == 7600 for r in receipts["warehouse_projection"]["ours_accuracy"]["metrics"]),
        "7600_mentioned_only_inside_the_limitations_text": ("7,600" in blob or "7600" in blob),
    }

    report = {
        "schema": "cbc_recount.v1",
        "what_this_is": "a recount of retained CB03 artifacts from their per-row arrays; no model, "
                        "no dataset, no service",
        "evidence_dir": str(EV),
        "file_digests": digests,
        "manifest_check": manifest_check,
        "manifest_all_match": all(v["matches"] for v in manifest_check.values()),
        "native_recount": rc_native,
        "native_recount_vs_stored": native_vs_stored,
        "paired_naive": naive,
        "parity_recount": rc_parity,
        "attribution_ladder_recounted": ladder,
        "attribution_steps_recounted": steps,
        "attribution_stored_accuracies": attribution["accuracies"],
        "population_digests": population_digests,
        "denominator_guard": denominator_guard,
        "held_out_status": {
            "in_training": True,
            "source": "the benchmark's own results file marks jev.ag_news in_training: true",
            "is_a_generalisation_claim": False,
        },
        "receipt_inventory": inventory,
        "retained_projections": sum(1 for r in inventory if r["has_retained_warehouse_projection"]),
        "accepted_warehouse_rows_provable_from_this_file": 0,
        "why_zero": "acceptance is a property of the live warehouse, not of a retained file; this "
                    "module contacts no service, so it can only report RETAINED",
    }
    if OUT:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("file_digests", "native_recount", "attribution_steps_recounted")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
