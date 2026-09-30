"""What the contract accepted before this round. Run at the base tip to reproduce.

No model runs here. Every number is fabricated; the point is only what the
producer contract admits.
"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app import classification_receipt as cr

H = {n: hashlib.sha256(n.encode()).hexdigest() for n in
     ("eval", "train", "corpus", "ckpt", "protocol", "scorer")}
matrix = []
for i in range(3):
    row = [0, 0, 0, 1]; row[i] = 8; row[(i + 1) % 3] = 1; matrix.append(row)
total = sum(sum(r) for r in matrix); abst = sum(r[-1] for r in matrix)
doc = {
    "task_id": "news_relevance_eurusd.v1", "corpus_class": "BUSINESS_HELD_OUT",
    "corpus_id": "cb04_business_news.v1", "corpus_sha256": H["corpus"],
    "evidence_class": "MEASUREMENT", "supervision_regime": "ZERO_SHOT_PROMPTED",
    "labelled_rows_fit_head": False,
    "provider": "canned_answer_table.v1", "checkpoint": "canned_answer_table.v1",
    "checkpoint_sha256": H["ckpt"],
    "author_primary_metric": {"family": "MACRO_F1", "name": "macro-F1",
                              "value": 8 / 9, "denominator_policy": "ANSWERED_ONLY"},
    "paired_naive": {"family": "MACRO_F1", "policy": "MAJORITY_CLASS_FROM_TRAIN",
                     "value": 0.25, "seed": None,
                     "train_label_population_sha256": H["train"],
                     "evaluation_population_sha256": H["eval"]},
    "class_vocabulary": ["related", "unrelated", "unclear"],
    "per_class_confusion": matrix,
    "probability_semantics": "SOFTMAX_POSTERIOR_UNCALIBRATED", "calibrated": False,
    "calibration_split": {"split_id": "NONE", "population_sha256": None,
                          "rows": 0, "fitted_parameters": None},
    "abstention": {"abstained": abst, "answered": total - abst,
                   "coverage": abst / total, "denominator_policy": "ANSWERED_ONLY",
                   "abstention_rule": "the table has no answer for these rows"},
    "population": {"total": total, "answered": total - abst, "abstained": abst,
                   "independent_units": total, "repeats": 1, "clustered_by": "NONE"},
    "evaluation_split": "test", "evaluation_population_sha256": H["eval"],
    "declared_fields": [], "limitations": "answers came from a constant table",
    "protocol_sha256": H["protocol"], "scorer_sha256": H["scorer"], "seed": "1"}

if "--declared" in sys.argv:
    # the same table, declaring what it is: a constant answer path, no weights,
    # and the no-checkpoint sentinel instead of a 64-hex digest
    doc["evidence_class"] = "TRANSPORT_TEST_NOT_SCIENCE"
    doc["checkpoint"] = "NO_CHECKPOINT_SERVED"
    doc["checkpoint_sha256"] = "NO_CHECKPOINT_SERVED"
    doc["answering_path"] = {"path_id": "canned_answer_table.v1",
                            "kind": "NON_MODEL_CONSTANT", "weights_present": False,
                            "served_checkpoint": "NO_CHECKPOINT_SERVED",
                            "served_checkpoint_sha256": "NO_CHECKPOINT_SERVED",
                            "attestation": "OBSERVED_FROM_ANSWERING_PATH"}

out = {}
receipt = None
try:
    receipt = cr.build_receipt(doc)
    out["receipt_built"] = True
    out["evidence_role"] = receipt.get("evidence_role")
    out["stored_checkpoint_sha256"] = receipt["checkpoint_sha256"]
    out["stored_evidence_class"] = receipt["evidence_class"]
    out["declares_the_answering_path"] = "answering_path" in receipt
    out["tags_name_the_answering_path"] = [
        t for t in cr.terminal_tags(receipt)
        if "answering" in t or "weights" in t or t == "evidence_role"]
except Exception as error:
    out["receipt_built"] = False; out["refusal"] = f"{type(error).__name__}: {error}"
try:
    if receipt is None:
        raise RuntimeError("no receipt was built, so no badge was asked for")
    badge = cr.provider_quality_badge("canned_answer_table.v1", [receipt])
    out["badge_issued"] = True
    out["badge_value"] = badge["business_evidence"][0]["value"]
    out["badge_names_the_answering_path"] = any(
        "answering" in k or "weights" in k for k in badge["business_evidence"][0])
except Exception as error:
    out["badge_issued"] = False; out["badge_refusal"] = f"{type(error).__name__}: {error}"
print(json.dumps(out, indent=2))
