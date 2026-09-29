"""CB-C: the receipts that were outstanding, and the one correction the recount forced.

Three things this adds to CB03's three receipts, all from RETAINED per-row arrays.  No
model is loaded, no dataset is scored, nothing is downloaded.

  1. the CORRECTED tie-break range.  CB03 pinned the naive at 0.180000 (alphabetical
     over class names -> Business, 72/400) and named World, 103/400 = 0.2575, as the
     strongest of the four legitimate breaks.  Recounting the population's own support
     from the retained per-row gold labels gives world 103, sports 123, business 72,
     sci_tech 102, so the STRONGEST break is SPORTS at 123/400 = 0.307500.  The honest
     statement is 0.9525 against a naive between 0.180000 and 0.307500, and the pinned
     0.180000 is still the one carried, because the tie was broken before any score
     existed.  Every receipt written here says so in its own limitations field.
  2. a MACRO_F1 warehouse projection.  CB03 built the macro-F1 receipt and projected
     only the accuracy one, so a warehouse reader would have found one family stored
     and might have read the other off it.
  3. OUR FRAMEWORK's own measured score as a receipt of its own: 0.9350 accuracy on the
     identical 400 rows, from the retained parity artefact.  This is the number that
     must not be lost behind the native reproduction's 0.9525.

  python tools/df_cbc_receipts_20260929.py \
      --evidence docs/audits/evidence/cb03_20260929 \
      --out-dir docs/audits/evidence/cbc_reconcile_20260929
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from app import classification_receipt as cr            # noqa: E402
import df_cb03_classification_receipt_20260929 as B      # noqa: E402

VOCAB = B.VOCABULARY
PINNED_400 = B.PINNED_400

#: recounted from the retained per-row gold labels, not copied from a return
SUPPORT_RECOUNTED = None          # filled in main()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def confusion_from_rows(rows, k=4):
    m = [[0] * (k + 1) for _ in range(k)]
    for r in rows:
        g = int(r["gold"])
        p = r.get("predicted")
        if p is None or r.get("refused") or r.get("status") not in (None, "OK"):
            m[g][k] += 1
        else:
            m[g][int(p)] += 1
    return m


def macro_f1(m, k=4):
    out = []
    for i in range(k):
        tp = m[i][i]
        fn = sum(m[i]) - tp
        fp = sum(m[j][i] for j in range(k)) - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(out) / k


def probability_metrics(rows, bins=15):
    """NLL, Brier and ECE from the probabilities the path actually reported.

    The framework's payload rounds every probability to four decimals
    (`probability_decimals: 4`), so these are the metrics of the reported
    distribution, and the receipt says so rather than renormalising behind the
    reader's back."""
    n = len(rows)
    nll = -sum(math.log(max(r["probabilities"][int(r["gold"])], 1e-12)) for r in rows) / n
    brier = sum(sum((p - (1.0 if j == int(r["gold"]) else 0.0)) ** 2
                    for j, p in enumerate(r["probabilities"])) for r in rows) / n
    conf = [max(r["probabilities"]) for r in rows]
    hit = [1.0 if int(r["predicted"]) == int(r["gold"]) else 0.0 for r in rows]
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if (lo <= c <= hi if b == 0 else lo < c <= hi)]
        if not sel:
            continue
        ece += (len(sel) / n) * abs(sum(hit[i] for i in sel) / len(sel)
                                    - sum(conf[i] for i in sel) / len(sel))
    return {"nll": nll, "brier": brier, "ece": ece, "ece_bins": bins}


def tiebreak_table(rows) -> dict:
    n = len(rows)
    support = {VOCAB[i]: sum(1 for r in rows if int(r["gold"]) == i) for i in range(4)}
    per = {k: v / n for k, v in support.items()}
    return {"support": support, "naive_accuracy_by_tiebreak": per,
            "weakest": min(per, key=per.get), "strongest": max(per, key=per.get),
            "low": min(per.values()), "high": max(per.values())}


def limitations(tb: dict, macro_value: float, author_rounded) -> str:
    return (
        "AG News is NOT held out for this checkpoint: the benchmark's own results file marks the "
        "suite in_training true, so this is an in-training-mix score and NO generalisation claim "
        "follows from it. The evaluation population is the first 400 rows of the official test "
        "split in file order and is NOT balanced (world %d, sports %d, business %d, sci_tech %d). "
        "The 7,600-row official test split is a SEPARATE experiment: 7,600 is never the "
        "denominator of this score, whose denominator is 400. The train split is exactly balanced "
        "at 30,000 per class, so MAJORITY_CLASS_FROM_TRAIN is a FOUR-WAY TIE; CB01 broke it "
        "alphabetically over the class names (selects Business) before any score existed, which "
        "gives the carried 0.180000 and is the WEAKEST of the four legitimate breaks. CORRECTION "
        "made by recount on 2026-09-29: the STRONGEST break is %s at %.6f, not World at 0.257500 "
        "as the CB03 return stated; World is the second strongest. The honest statement is "
        "therefore this score against a paired naive BETWEEN %.6f AND %.6f on the same rows, and "
        "the range must be quoted rather than either end alone. The recipe applies the "
        "checkpoint's own per-bucket temperatures, whose fitting split the author does not "
        "publish, so the posterior is UNCALIBRATED and calibrated is false; ECE is reported as "
        "measured with its bin count and is not evidence of calibration. Latency is NOT "
        "comparable: a single BLAS thread by admission policy. Macro-F1 is carried at full "
        "precision (%.12f); the author's harness rounds it to %s."
        % (tb["support"]["world"], tb["support"]["sports"], tb["support"]["business"],
           tb["support"]["sci_tech"], tb["strongest"].capitalize(), tb["high"],
           tb["low"], tb["high"], macro_value, author_rounded))


def framework_document(parity, native, train_digest, *, family, value, name, secondary,
                       limits, scorer_sha):
    m = confusion_from_rows(parity["per_row"])
    total = sum(sum(row) for row in m)
    abstained = sum(row[-1] for row in m)
    fw = parity["framework"]
    protocol = {"provider": fw["provider"], "task_id": fw["task_id"],
                "questions_as_built": fw["questions_as_built"],
                "backend_identity": fw["backend_identity"],
                "backend_declared_budget": fw["backend_declared_budget"],
                "state_envelope_fields": fw["state_envelope_fields"],
                "population_sha256": parity["population_sha256"]}
    return {
        "task_id": B.TASK_ID,
        "corpus_class": "PUBLIC_BENCHMARK",
        "corpus_id": B.CORPUS_ID,
        "corpus_sha256": parity["population_sha256"],
        "evidence_class": "MEASUREMENT",
        "supervision_regime": "ZERO_SHOT_PROMPTED",
        "labelled_rows_fit_head": False,
        "provider": "M5PHET_RUNTIME_news_signal_%s_sdk_%s" % (
            fw["provider"], fw["backend_identity"]["sdk_version"]),
        "checkpoint": "%s@%s" % (native["checkpoint"], native["checkpoint_revision"]),
        "checkpoint_sha256": native["checkpoint_weights_sha256"],
        "author_primary_metric": {"family": family, "name": name, "value": value,
                                  "denominator_policy": "FULL_POPULATION_ABSTENTION_WRONG"},
        "paired_naive": {"family": family, "policy": "MAJORITY_CLASS_FROM_TRAIN",
                         "value": (B.PINNED_NAIVE_ACCURACY if family == "ACCURACY"
                                   else B.PINNED_NAIVE_MACRO_F1),
                         "seed": None,
                         "train_label_population_sha256": train_digest,
                         "evaluation_population_sha256": parity["population_sha256"]},
        "class_vocabulary": list(VOCAB),
        "per_class_confusion": [list(r) for r in m],
        "probability_semantics": "SOFTMAX_POSTERIOR_UNCALIBRATED",
        "calibrated": False,
        "calibration_split": {"split_id": "NONE", "population_sha256": None, "rows": 0,
                              "fitted_parameters": None},
        "probability_metrics": probability_metrics(parity["per_row"]),
        "secondary_metrics": dict(secondary),
        "abstention": {"abstained": abstained, "answered": total - abstained,
                       "coverage": abstained / total,
                       "denominator_policy": "FULL_POPULATION_ABSTENTION_WRONG",
                       "abstention_rule": ("the provider may refuse a row; on this population it "
                                           "refused none, so coverage is 400/400 and the "
                                           "denominator is the full population either way")},
        "population": {"total": total, "answered": total - abstained, "abstained": abstained,
                       "independent_units": total, "repeats": 1, "clustered_by": "NONE"},
        "evaluation_split": B.EVALUATION_SPLIT,
        "evaluation_population_sha256": parity["population_sha256"],
        "protocol_sha256": cr.sha256_of(protocol),
        "scorer_sha256": scorer_sha,
        "seed": None,
        "declared_fields": [],
        "limitations": limits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache", default=os.path.expanduser(
        "~/.local/state/crispdm-data-foundation/classification-extension-20260928/"
        "cache/sota_benchmarks"))
    args = ap.parse_args()

    ev = Path(args.evidence)
    native = json.loads((ev / "native_published400.json").read_text())
    parity = json.loads((ev / "parity_400.json").read_text())
    if native["population_sha256"] != PINNED_400 or parity["population_sha256"] != PINNED_400:
        raise SystemExit("REFUSED: an artefact is not the pinned 400-row population")
    if parity["native_artifact_sha256"] != sha_file(ev / "native_published400.json"):
        raise SystemExit("REFUSED: the parity artefact does not bind the native bytes on disk")

    train_digest, train_rows, train_counts = B.train_population_digest(args.cache)
    if train_digest == PINNED_400:
        raise SystemExit("REFUSED: the train and evaluation populations share a digest")
    if len(set(train_counts.values())) != 1:
        raise SystemExit("REFUSED: the train split is not balanced, so the tie-break note is wrong")

    tb = tiebreak_table(native["per_row"])
    nat_conf = native["confusion_reference_by_predicted_plus_abstained"]
    if confusion_from_rows(native["per_row"]) != [list(r) for r in nat_conf]:
        raise SystemExit("REFUSED: the retained confusion does not match its own per-row array")
    nat_macro = macro_f1(nat_conf)
    limits = limitations(tb, nat_macro, native["metrics"]["macro_f1"])

    native_accuracy = cr.build_receipt(B.document_from(
        native, train_digest, family="ACCURACY", value=native["metrics"]["accuracy"],
        name="accuracy", secondary={"MACRO_F1": nat_macro}, limitations=limits))
    native_macro = cr.build_receipt(B.document_from(
        native, train_digest, family="MACRO_F1", value=nat_macro, name="macro-F1",
        secondary={"ACCURACY": native["metrics"]["accuracy"]}, limitations=limits))

    fw_conf = confusion_from_rows(parity["per_row"])
    fw_acc = sum(fw_conf[i][i] for i in range(4)) / sum(sum(r) for r in fw_conf)
    fw_macro = macro_f1(fw_conf)
    scorer = sha_file(ROOT / "tools/df_cb03_m5phet_parity_20260929.py")
    fw_limits = (limits + " THIS RECEIPT IS OUR FRAMEWORK'S OWN SCORE, NOT the native "
                 "reproduction: %0.4f accuracy against the native reproduction's %0.4f on the "
                 "identical 400 rows, 387/400 label agreement, 13 disagreements, maximum absolute "
                 "probability difference %0.6f. The whole gap is the state envelope: the shipped "
                 "provider serialises {asset, headline, body} and requires all three non-empty, so "
                 "it CANNOT present the author's {\"article\": text} string. Given the same input "
                 "string the framework agrees with the native reference on 400 of 400 labels to "
                 "5.02e-05, which is half of the last decimal the SDK itself retains "
                 "(probability_decimals 4). The probabilities this receipt scores are therefore "
                 "the provider's four-decimal payload, not a renormalised distribution. Two "
                 "framework defects are reported and NOT fixed: the news-shaped classification "
                 "entry point (the entire measured gap) and a sequence budget hard-coded at "
                 "512/192 below this checkpoint's declared 1024/256, harmless here because the "
                 "longest sequence is 232 tokens and waiting for a longer input."
                 % (fw_acc, native["metrics"]["accuracy"],
                    parity["parity"]["max_abs_probability_difference"]))
    framework_accuracy = cr.build_receipt(framework_document(
        parity, native, train_digest, family="ACCURACY", value=fw_acc, name="accuracy",
        secondary={"MACRO_F1": fw_macro}, limits=fw_limits, scorer_sha=scorer))

    receipts = {"cbc_native_accuracy": native_accuracy,
                "cbc_native_macro_f1": native_macro,
                "cbc_framework_accuracy": framework_accuracy}

    # every projection, this time, and not only the accuracy one
    projections = {k: {"tags": cr.terminal_tags(v), "metrics": cr.terminal_metrics(v)}
                   for k, v in receipts.items()}

    # the refusals, run rather than described
    refusals = {}
    for label, call in (
            ("read_native_accuracy_as_MAP", lambda: cr.read_metric(native_accuracy, "MAP")),
            ("compare_accuracy_to_macro_f1", lambda: cr.compare(native_accuracy, native_macro)),
            ("read_framework_accuracy_as_MAP", lambda: cr.read_metric(framework_accuracy, "MAP")),
            ("compare_framework_accuracy_to_native_macro_f1",
             lambda: cr.compare(framework_accuracy, native_macro))):
        try:
            call()
            refusals[label] = "NOT REFUSED - this is a defect"
        except (cr.MetricNotCarried, cr.IncomparableMetrics, cr.IncomparableProtocol) as exc:
            refusals[label] = str(exc)

    # the one comparison the contract DOES admit: same family, same task, same population
    try:
        framework_vs_native = cr.compare(framework_accuracy, native_accuracy)
    except (cr.IncomparableMetrics, cr.IncomparableProtocol) as exc:
        framework_vs_native = {"refused": str(exc)}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema": "cbc_reconciled_receipts.v1",
        "metric_contract": cr.SCHEMA,
        "produced_by": "tools/df_cbc_receipts_20260929.py",
        "what_this_is": "receipts rebuilt from RETAINED per-row arrays with the corrected "
                        "tie-break range, plus the framework receipt CB03 did not build; "
                        "no model ran and no dataset was scored here",
        "tiebreak_recount": tb,
        "cb03_stated_strongest_tiebreak": {"class": "World", "value": 0.2575,
                                           "verdict": "UNDERSTATED: Sports is stronger at 0.3075"},
        "train_label_population": {"sha256": train_digest, "rows": train_rows,
                                   "class_counts": train_counts},
        "receipts": receipts,
        "warehouse_projection": projections,
        "refusals_exercised": refusals,
        "framework_against_native": framework_vs_native,
        "identities": {k: v["metric_identity_sha256"] for k, v in receipts.items()},
        "receipt_sha256": {k: v["receipt_sha256"] for k, v in receipts.items()},
        "acceptance_state": {k: "RETAINED_PROJECTION_NOT_AN_ACCEPTED_WAREHOUSE_ROW"
                             for k in receipts},
        "not_projected_and_why": {
            "author_published_accuracy": (
                "CB03's PUBLISHED_REFERENCE receipt is retained and is deliberately NOT written "
                "as warehouse metric rows. It shares its metric_identity_sha256 with our "
                "ACCURACY measurement, which is exactly what makes the two comparable - and "
                "exactly what would let any aggregation keyed on metric identity, which is what "
                "the contract's own query template prescribes, average a published number "
                "together with a measured one. It is carried instead as read-only tags on the "
                "native accuracy terminal, where a reader can find it and an aggregate cannot.")},
    }
    (out / "CBC_RECEIPTS.json").write_text(json.dumps(bundle, indent=1, sort_keys=True))
    print(json.dumps({
        "tiebreak_recount": tb,
        "native_accuracy": cr.read_metric(native_accuracy, "ACCURACY"),
        "native_macro_f1": cr.read_metric(native_macro, "MACRO_F1"),
        "framework_accuracy": cr.read_metric(framework_accuracy, "ACCURACY"),
        "framework_against_native": framework_vs_native,
        "receipt_sha256": bundle["receipt_sha256"],
        "identities": bundle["identities"],
        "projection_metric_rows": {k: len(v["metrics"]) for k, v in projections.items()},
        "refusals_exercised": refusals,
        "written": str(out / "CBC_RECEIPTS.json"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
