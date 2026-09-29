"""CB03: turn the native reproduction into classification receipts on the existing path.

`app/classification_receipt.py` is CB04's contract, enforced rather than described.
This does not write a second one: it fills the document that contract requires from
the arrays the native run actually produced, and lets the contract refuse anything
that would let a later reader mistake what was measured.

Three receipts come out of one run, and they are deliberately three and not one:

  1. OUR measurement, evidence_class MEASUREMENT, on the 400-row published
     population, with the paired train-derived majority naive scored on exactly
     those rows.
  2. the AUTHOR's published value, evidence_class PUBLISHED_REFERENCE, with its
     source artefact.  A published number is its own receipt and is never
     labelled as our result.
  3. our measurement of the same run's macro-F1 as a PRIMARY metric, so the
     contract's own refusal can be exercised: accuracy and macro-F1 are different
     metrics with different identities, and a query for one cannot return the
     other even when both are ours and both came from the same 400 rows.

It also records what the contract refuses, by running the refusals rather than
describing them.

  python tools/df_cb03_classification_receipt_20260929.py \
      --native native_published400.json --out-dir docs/audits/evidence/cb03_20260929
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import classification_receipt as cr  # noqa: E402

GOVERNED_TRAIN = "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee"
PINNED_400 = "b4c5f991060bcefcc69fac9339b32086dcd0674ac15e44f41eeeeb7c9e782324"

TASK_ID = "agnews_topic_zhang2015_laya_published_row.v1"
CORPUS_ID = "agnews_zhang2015_test_first400_file_order"
EVALUATION_SPLIT = "test_first400_file_order"

#: The author's option keys, in the author's order.  This is what the model was
#: shown, and the confusion below is indexed by it.  Reordering it is a different
#: task and a different label_order_sha256.
VOCABULARY = ["world", "sports", "business", "sci_tech"]

#: CB01, docs/contracts/classification_naive_baselines.v1.json, pinned BEFORE any
#: score existed.  The train split is exactly balanced at 30,000 per class, so the
#: majority is a FOUR-WAY TIE; CB01 broke it alphabetically over the class names,
#: which selects Business and gives 0.180000 on these 400 rows.  The tie-break is
#: recorded because it is the weakest of the four: World would give 103/400.
PINNED_NAIVE_ACCURACY = 0.18
PINNED_NAIVE_MACRO_F1 = 0.07627118644067797
PINNED_NAIVE_TIE_BREAK = "ALPHABETICAL_OVER_CLASS_NAMES_SELECTS_Business"
STRONGEST_TIE_BREAK_ACCURACY = 103 / 400

#: The published cell this reproduction is measured against.
PUBLISHED = {
    "value": 0.9525,
    "macro_f1": 0.9468,
    "source": ("NandhaKishorM/laya, research/results/app_benchmark_results.json, "
               "suites['jev.ag_news']['typed-decisions'], committed at 010bacef "
               "(release 0.3.7, 2026-09-23); the run itself is stamped 2026-09-19 "
               "11:57:16, device cpu, laya 0.2.1, n_per_task 400, seed 13"),
    "in_training": True,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def population_digest(rows):
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def train_population_digest(cache):
    """The train population's own identity, re-hashed here from the delivered bytes.

    The naive is fitted on these labels and must not be fitted on the rows it is
    scored on; the contract refuses a train digest equal to the evaluation digest,
    so the digest has to be real rather than a placeholder."""
    import pandas as pd
    path = os.path.join(cache, "%s.parquet" % GOVERNED_TRAIN)
    if sha256_file(path) != GOVERNED_TRAIN:
        raise SystemExit("REFUSED: AG News train bytes are not the delivered bytes")
    frame = pd.read_parquet(path)
    rows = [[i, str(t), int(l)] for i, (t, l) in
            enumerate(zip(frame["text"], frame["label"]))]
    counts = {}
    for _i, _t, label in rows:
        counts[VOCABULARY[label]] = counts.get(VOCABULARY[label], 0) + 1
    return population_digest(rows), len(rows), counts


def derived_macro_f1(confusion):
    """Macro-F1 from the confusion under the contract's own definition, UNROUNDED.

    The author's harness rounds every metric to four decimals, and the contract
    refuses a carried metric that disagrees with its own confusion by more than
    1e-6.  Both are right: the receipt therefore carries the full-precision value
    and records the author's rounded figure beside it, instead of loosening a
    check that exists to catch exactly this kind of quiet disagreement."""
    size = len(confusion)
    support = [sum(row) for row in confusion]
    predicted = [sum(confusion[r][c] for r in range(size)) for c in range(size)]
    f1 = []
    for i in range(size):
        true_positive = confusion[i][i]
        precision = true_positive / predicted[i] if predicted[i] else 0.0
        recall = true_positive / support[i] if support[i] else 0.0
        f1.append(0.0 if precision + recall == 0
                  else 2 * precision * recall / (precision + recall))
    return sum(f1) / size if size else 0.0


def document_from(native, train_digest, *, family, value, name, secondary, limitations,
                  evidence_class="MEASUREMENT", provider=None, declared_fields=()):
    metrics = native["metrics"]
    confusion = native["confusion_reference_by_predicted_plus_abstained"]
    total = sum(sum(row) for row in confusion)
    abstained = sum(row[-1] for row in confusion)
    protocol = {
        "prompt": native["prompt"],
        "label_order": VOCABULARY,
        "checkpoint": native["checkpoint"],
        "checkpoint_revision": native["checkpoint_revision"],
        "checkpoint_weights_sha256": native["checkpoint_weights_sha256"],
        # the first artefact predates the attribution options; its budget is the
        # one `option_fit` recorded and its envelope is the author's, by construction
        "budget": native.get("budget_used") or {
            "max_len": native["option_fit"]["max_len"],
            "head_max_len": native["option_fit"]["head_max_len"]},
        "state_envelope": native.get("state_envelope", "author"),
        "device": native["device"],
        "population_sha256": native["population_sha256"],
        "harness_sha256": native["harness_sha256"],
    }
    return {
        "task_id": TASK_ID,
        "corpus_class": "PUBLIC_BENCHMARK",
        "corpus_id": CORPUS_ID,
        "corpus_sha256": native["population_sha256"],
        "evidence_class": evidence_class,
        "supervision_regime": "ZERO_SHOT_PROMPTED",
        "labelled_rows_fit_head": False,
        "provider": provider or ("NATIVE_AUTHOR_HARNESS_laya_%s" % native["laya_version"]),
        "checkpoint": "%s@%s" % (native["checkpoint"], native["checkpoint_revision"]),
        "checkpoint_sha256": native["checkpoint_weights_sha256"],
        "author_primary_metric": {
            "family": family,
            "name": name,
            "value": value,
            "denominator_policy": "FULL_POPULATION_ABSTENTION_WRONG",
        },
        "paired_naive": {
            "family": family,
            "policy": "MAJORITY_CLASS_FROM_TRAIN",
            # CB01's pinned values on exactly these 400 rows, of the SAME family
            "value": (PINNED_NAIVE_ACCURACY if family == "ACCURACY"
                      else PINNED_NAIVE_MACRO_F1),
            "seed": None,
            "train_label_population_sha256": train_digest,
            "evaluation_population_sha256": native["population_sha256"],
        },
        "class_vocabulary": list(VOCABULARY),
        "per_class_confusion": [list(row) for row in confusion],
        "probability_semantics": "SOFTMAX_POSTERIOR_UNCALIBRATED",
        "calibrated": False,
        "calibration_split": {"split_id": "NONE", "population_sha256": None, "rows": 0,
                              "fitted_parameters": None},
        "probability_metrics": {"nll": metrics["nll"], "brier": metrics["brier"],
                                "ece": metrics["ece"], "ece_bins": 15},
        "secondary_metrics": dict(secondary),
        "abstention": {
            "abstained": abstained,
            "answered": total - abstained,
            "coverage": abstained / total,
            "denominator_policy": "FULL_POPULATION_ABSTENTION_WRONG",
            "abstention_rule": ("the author's recipe offers no refusal token: a row is dropped "
                                "only when the rendered options do not fit the sequence budget, "
                                "and no row did"),
        },
        "population": {"total": total, "answered": total - abstained, "abstained": abstained,
                       "independent_units": total, "repeats": 1, "clustered_by": "NONE"},
        "evaluation_split": EVALUATION_SPLIT,
        "evaluation_population_sha256": native["population_sha256"],
        "protocol_sha256": cr.sha256_of(protocol),
        "scorer_sha256": native["harness_sha256"],
        "seed": None,
        "declared_fields": sorted(declared_fields),
        "limitations": limitations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--cache", default=os.path.expanduser(
        "~/.local/state/crispdm-data-foundation/classification-extension-20260928/"
        "cache/sota_benchmarks"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    native = json.load(open(args.native))
    if native["population_sha256"] != PINNED_400:
        raise SystemExit("REFUSED: the native artifact is not the pinned 400-row population")
    train_digest, train_rows, train_counts = train_population_digest(args.cache)
    if train_digest == native["population_sha256"]:
        raise SystemExit("REFUSED: the train and evaluation populations share a digest")

    macro = derived_macro_f1(native["confusion_reference_by_predicted_plus_abstained"])
    common = ("AG News is NOT held out for this checkpoint: the author's own results file marks "
              "the suite in_training true, so this is an in-training-mix score and no "
              "generalisation claim follows from it. The population is the first 400 rows of the "
              "official test split in file order, which is NOT balanced (72/102/123/103), so the "
              "paired train-derived majority naive on these rows is 0.180000 and not the full "
              "split's 0.250000; the full 7,600-row test split is a separate experiment and its "
              "denominator is never attached to this score. The train split is exactly balanced "
              "at 30,000 per class, so the majority is a four-way tie; CB01 broke it "
              "alphabetically (%s) before any score existed, which is the WEAKEST of the four "
              "choices - the strongest, World, would give %.4f. The author's recipe applies the "
              "checkpoint's own per-bucket temperatures, whose fitting split the author does not "
              "publish, so the posterior is declared UNCALIBRATED and calibrated false; ECE is "
              "reported as measured with its bin count, not as evidence of calibration. Latency "
              "is not comparable: this ran under a single BLAS thread by admission policy. "
              "Macro-F1 is carried at full precision (%.12f); the author's harness rounds it to "
              "%s, and the contract refuses a carried metric that disagrees with its own "
              "confusion, so the rounding is recorded here rather than absorbed."
              % (PINNED_NAIVE_TIE_BREAK, STRONGEST_TIE_BREAK_ACCURACY, macro,
                 native["metrics"]["macro_f1"]))

    ours = cr.build_receipt(document_from(
        native, train_digest, family="ACCURACY", value=native["metrics"]["accuracy"],
        name="accuracy", secondary={"MACRO_F1": macro}, limitations=common))

    ours_macro = cr.build_receipt(document_from(
        native, train_digest, family="MACRO_F1", value=macro,
        name="macro-F1", secondary={"ACCURACY": native["metrics"]["accuracy"]},
        limitations=common))

    published_doc = document_from(
        native, train_digest, family="ACCURACY", value=PUBLISHED["value"], name="accuracy",
        secondary={},
        evidence_class="PUBLISHED_REFERENCE",
        provider="AUTHOR_PUBLISHED_RUN_laya_0.2.1",
        declared_fields=("author_primary_metric",),
        limitations=("the author's published value, quoted from its artefact and never labelled "
                     "as our measured result: author_primary_metric is the declaration here. "
                     "Source: %s. The author publishes NO per-class confusion, no per-row "
                     "predictions and no population digest for this cell, so the confusion, "
                     "abstention, population and probability metrics carried in this receipt are "
                     "OURS - they are present because the contract requires every field, and "
                     "they are not the author's numbers. The author's published macro-F1 is "
                     "%.4f. %s" % (PUBLISHED["source"], PUBLISHED["macro_f1"], common)))
    published = cr.build_receipt(published_doc)

    # ---- the refusals, run rather than described --------------------------------
    refusals = {}
    try:
        cr.read_metric(ours, "MAP")
        refusals["read_accuracy_receipt_as_MAP"] = "NOT REFUSED - this is a defect"
    except cr.MetricNotCarried as exc:
        refusals["read_accuracy_receipt_as_MAP"] = str(exc)
    try:
        cr.compare(ours, ours_macro)
        refusals["compare_accuracy_to_macro_f1"] = "NOT REFUSED - this is a defect"
    except cr.IncomparableMetrics as exc:
        refusals["compare_accuracy_to_macro_f1"] = str(exc)

    comparison = {}
    try:
        comparison = cr.compare(ours, published)
    except (cr.IncomparableMetrics, cr.IncomparableProtocol) as exc:
        comparison = {"refused": str(exc)}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema": "cb03_classification_receipts.v1",
        "metric_contract": cr.SCHEMA,
        "train_label_population": {"sha256": train_digest, "rows": train_rows,
                                   "class_counts": train_counts},
        "receipts": {"ours_accuracy": ours, "ours_macro_f1": ours_macro,
                     "author_published_accuracy": published},
        "warehouse_projection": {
            "ours_accuracy": {"tags": cr.terminal_tags(ours),
                              "metrics": cr.terminal_metrics(ours)},
        },
        "refusals_exercised": refusals,
        "ours_against_published": comparison,
        "identities_differ": {
            "ours_accuracy": ours["metric_identity_sha256"],
            "ours_macro_f1": ours_macro["metric_identity_sha256"],
            "author_published_accuracy": published["metric_identity_sha256"],
        },
    }
    path = out / "CB03_CLASSIFICATION_RECEIPTS.json"
    path.write_text(json.dumps(bundle, indent=1, sort_keys=True))
    print(json.dumps({
        "receipt_sha256": {k: v["receipt_sha256"] for k, v in bundle["receipts"].items()},
        "identities_differ": bundle["identities_differ"],
        "ours_accuracy": cr.read_metric(ours, "ACCURACY"),
        "ours_macro_f1": cr.read_metric(ours_macro, "MACRO_F1"),
        "naive_accuracy_same_rows": ours["paired_naive"]["value"],
        "refusals_exercised": refusals,
        "ours_against_published": comparison,
        "written": str(path),
    }, indent=1))


if __name__ == "__main__":
    main()
