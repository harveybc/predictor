"""Classification receipts: the metric contract, enforced rather than described.

Why this module exists
----------------------
A classification number written into the warehouse is read later by somebody who
was not in the room. The failure this module is built against is not a wrong
model: it is a right number read as the wrong thing. MAP, accuracy and macro-F1
all live between zero and one. A stored `0.61` that came from MAP and is read as
accuracy is a false claim that no amount of prose in a document prevents.

So the separation is mechanical, in three independent places:

* a **distinct metric key** per family (`classification.map`,
  `classification.accuracy`, `classification.macro_f1`, ...);
* a **distinct unit token** per family, so the value never travels beside a unit
  that would fit another family;
* a **metric identity digest** over name, family, definition, unit, averaging,
  denominator policy and label order, so two values agree only when everything
  that decides their meaning agrees.

And two refusals: :func:`read_metric` refuses to hand a caller a family the
receipt does not carry, naming both families, and :func:`compare` refuses across
families, again naming both. Nothing coerces, defaults or falls back.

What a receipt must carry
-------------------------
The seven facts CB04 names are each their own field and none is reconstructed at
read time: the author's primary metric under the author's own name; a paired
naive of the same family, fitted on train labels and scored on exactly the same
evaluation rows; the class vocabulary; the per-class confusion; the probability
semantics together with a separate `calibrated` boolean; abstention coverage with
its denominator policy; and the calibration split, kept distinct from the
evaluation split. Consistency between fields is checked, and a contradiction is
refused - but a check never replaces a field.

What this module refuses to produce
-----------------------------------
A provider quality badge from the router prompt corpus, and a badge from a
declaration. The router corpus is 19 prompts repeated five times; it measures
which envelope the router chooses, not whether any classifier answers correctly.
A router record is not even representable as a classification receipt here
(:func:`build_router_reliability_record` returns a different schema that carries
no metric family at all), and the badge path refuses it by name. No badge, at any
value, carries execution authority: `execution_authority` is the constant
`NONE`.

Provenance, added 2026-09-29
----------------------------
A fourth separation, and the one this module could not make before: `answering_path`
says which code path produced the answers, whether weights were loaded, which
checkpoint served and how that was established. `app.classification_provenance`
holds it. Three consequences here: a receipt may not quote a checkpoint the
answering path did not serve; a path with no weights may not carry
`evidence_class` MEASUREMENT, and is stored as a declared test instead of being
discarded; and :func:`provider_quality_badge` refuses evidence whose answering
path was not a model or was never observed. None of this matches a word - a
declared test may be named anything and an honest measurement's own prose may
contain any word.

Scope. This is a producer contract. The warehouse stays a generic store; nothing
here rewrites a historical row or reinterprets an existing number. The store
boundary check `admit_classification_terminal` reads only tags, so it binds a
terminal from any producer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

#: The contract document is the single source of truth. This module reads it; it
#: does not keep a second copy of the vocabulary that could drift from it.
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "docs/contracts/classification_metrics.v1.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text())
SCHEMA = CONTRACT["schema"]

# imported after CONTRACT so the gate can read the vocabulary from the contract
# document rather than keeping a second copy of it
from app import classification_provenance as provenance  # noqa: E402

ROUTER_RECORD_SCHEMA = "router_reliability.v1"

_HEX64 = re.compile(r"[0-9a-f]{64}")
_METRIC_KEY = re.compile(r"[A-Za-z0-9._:-]+")

#: Fields the builder derives; a document is not asked to supply them.
_DERIVED = ("schema", "receipt_id", "receipt_sha256", "metric_identity_sha256")


class ClassificationReceiptError(Exception):
    """Base class. Every refusal in this module is named and carries its reason."""


class ReceiptRefused(ClassificationReceiptError):
    """A receipt that would misstate its own content is not stored."""


class MetricNotCarried(ClassificationReceiptError):
    """A reader asked a receipt for a metric family it does not carry."""


class IncomparableMetrics(ClassificationReceiptError):
    """Two different metric families were placed on the same axis."""


class IncomparableProtocol(ClassificationReceiptError):
    """Same family, different task, population, label order or denominator."""


class BadgeRefused(ClassificationReceiptError):
    """A provider quality claim was not supported by the evidence offered."""

    def __init__(self, refusal: str, detail: str):
        super().__init__(f"{refusal}: {detail}")
        self.refusal = refusal
        self.detail = detail


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def canonical_text(body) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_of(body) -> str:
    return hashlib.sha256(canonical_text(body).encode("ascii")).hexdigest()


def _require(condition, message):
    if not condition:
        raise ReceiptRefused(message)


def _finite(value, where):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{where} must be a number")
    value = float(value)
    _require(math.isfinite(value), f"{where} must be finite, not {value!r}")
    return value


def _hex64(value, where):
    _require(isinstance(value, str) and _HEX64.fullmatch(value),
             f"{where} must be a sha256 hex digest")
    return value


def _checkpoint_digest(value):
    """A sha256 digest, or a declared sentinel saying no checkpoint served.

    Before this round the field was an unconditional hex64, so a path that had
    loaded no weights had to invent one - and an invented digest is shaped
    exactly like a real one. The sentinels are declared in the contract and the
    answering path decides which of them is legal.
    """
    _require(isinstance(value, str) and
             (_HEX64.fullmatch(value) or value in provenance.CHECKPOINT_SENTINELS),
             f"checkpoint_sha256 must be a sha256 digest or one of "
             + ", ".join(provenance.CHECKPOINT_SENTINELS) + f", not {value!r}")
    return value


def family_spec(family: str) -> dict:
    spec = CONTRACT["metric_families"].get(family)
    if spec is None:
        raise ReceiptRefused(
            f"{family!r} is not a declared metric family; the contract declares "
            + ", ".join(sorted(CONTRACT["metric_families"])))
    return spec


def metric_key(family: str, *, naive: bool = False) -> str:
    """`classification.macro_f1` / `classification.naive_macro_f1`.

    A separate key per family is the first of the three separations: a query for
    accuracy cannot return a MAP value, because it does not select it.
    """
    key = family_spec(family)["metric"]
    if naive:
        head, _, tail = key.rpartition(".")
        key = f"{head}.{CONTRACT['paired_naive']['metric_prefix'].split('.')[-1]}{tail}"
    _require(_METRIC_KEY.fullmatch(key), f"metric key {key!r} is not storable")
    return key


def metric_identity_sha256(receipt_or_document) -> str:
    """Digest over everything that decides what the primary number *means*.

    Two values are the same measurement only if their family, the author's own
    name for it, its unit, its definition, its averaging, the denominator policy
    and the label order all agree. Change any of those and this digest changes,
    so no later reader can line the two up by name alone.
    """
    primary = receipt_or_document["author_primary_metric"]
    family = primary["family"]
    spec = family_spec(family)
    labels = receipt_or_document["class_vocabulary"]
    labels = labels["labels"] if isinstance(labels, dict) else labels
    return sha256_of({
        "schema": SCHEMA,
        "family": family,
        "author_name": primary["name"],
        "metric": spec["metric"],
        "unit": spec["unit"],
        "kind": spec["kind"],
        "definition": spec["definition"],
        "denominator_policy": primary["denominator_policy"],
        "label_order_sha256": sha256_of(list(labels)),
    })


# --------------------------------------------------------------------------- #
# validation, field by field
# --------------------------------------------------------------------------- #

def _validate_primary(document) -> dict:
    primary = document["author_primary_metric"]
    _require(isinstance(primary, dict), "author_primary_metric must be an object")
    for name in ("family", "name", "value", "denominator_policy"):
        _require(name in primary, f"author_primary_metric.{name} is required")
    spec = family_spec(primary["family"])
    _require(isinstance(primary["name"], str) and primary["name"],
             "author_primary_metric.name must be the author's own name for the metric")
    value = _finite(primary["value"], "author_primary_metric.value")
    low, high = spec["range"]
    _require(low <= value <= high,
             f"author_primary_metric.value {value} is outside the declared range "
             f"{low}..{high} for {primary['family']}")
    _require(primary["denominator_policy"] in CONTRACT["abstention"]["denominator_policies"],
             "author_primary_metric.denominator_policy must be one of "
             + ", ".join(sorted(CONTRACT["abstention"]["denominator_policies"])))
    return {"family": primary["family"], "name": primary["name"], "value": value,
            "denominator_policy": primary["denominator_policy"],
            "metric": spec["metric"], "unit": spec["unit"]}


def _validate_naive(document, primary) -> dict:
    naive = document["paired_naive"]
    _require(isinstance(naive, dict), "paired_naive must be an object")
    for name in ("family", "policy", "value", "evaluation_population_sha256"):
        _require(name in naive, f"paired_naive.{name} is required")
    _require(naive["family"] == primary["family"],
             f"paired_naive family {naive['family']} does not match the author's primary "
             f"metric family {primary['family']}; a naive of a different family is not a "
             f"pair, it is a second metric")
    policies = CONTRACT["paired_naive"]["policies"]
    _require(naive["policy"] in policies,
             f"paired_naive.policy {naive['policy']!r} is not a declared policy: "
             + ", ".join(sorted(policies)))
    spec = family_spec(naive["family"])
    value = _finite(naive["value"], "paired_naive.value")
    low, high = spec["range"]
    _require(low <= value <= high,
             f"paired_naive.value {value} is outside the declared range {low}..{high}")
    evaluation = document["evaluation_population_sha256"]
    _require(naive["evaluation_population_sha256"] == evaluation,
             "the paired naive must be scored on the same rows as the model: "
             "paired_naive.evaluation_population_sha256 does not equal "
             "evaluation_population_sha256")
    train = naive.get("train_label_population_sha256")
    if naive["policy"] in ("MAJORITY_CLASS_FROM_TRAIN", "STRATIFIED_PRIOR_FROM_TRAIN"):
        _hex64(train, "paired_naive.train_label_population_sha256")
        _require(train != evaluation,
                 "a train-derived naive fitted on the evaluation rows is not a baseline; "
                 "train_label_population_sha256 equals evaluation_population_sha256")
    return {"family": naive["family"], "policy": naive["policy"], "value": value,
            "seed": None if naive.get("seed") is None else str(naive["seed"]),
            "train_label_population_sha256": train,
            "evaluation_population_sha256": evaluation,
            "metric": metric_key(naive["family"], naive=True), "unit": spec["unit"]}


def _validate_vocabulary(document) -> dict:
    labels = document["class_vocabulary"]
    _require(isinstance(labels, list) and labels,
             "class_vocabulary must be the ordered, non-empty label list")
    _require(all(isinstance(label, str) and label for label in labels),
             "class_vocabulary labels must be non-empty strings")
    _require(len(set(labels)) == len(labels), "class_vocabulary contains a duplicate label")
    return {"labels": list(labels), "size": len(labels),
            "label_order_sha256": sha256_of(list(labels))}


def _validate_confusion(document, vocabulary, population) -> dict:
    matrix = document["per_class_confusion"]
    size = vocabulary["size"]
    _require(isinstance(matrix, list) and len(matrix) == size,
             f"per_class_confusion must have one row per class: {size} expected, "
             f"{len(matrix) if isinstance(matrix, list) else type(matrix).__name__} given")
    for index, row in enumerate(matrix):
        _require(isinstance(row, list) and len(row) == size + 1,
                 f"per_class_confusion row {index} must have {size} predicted columns "
                 f"plus the ABSTAINED column")
        for cell in row:
            _require(isinstance(cell, int) and not isinstance(cell, bool) and cell >= 0,
                     "per_class_confusion cells must be non-negative integer counts")
    total = sum(sum(row) for row in matrix)
    abstained = sum(row[-1] for row in matrix)
    _require(total == population["total"],
             f"per_class_confusion does not close over the population: it counts {total} "
             f"rows, population.total is {population['total']}")
    _require(abstained == population["abstained"],
             f"the ABSTAINED column counts {abstained} rows, population.abstained is "
             f"{population['abstained']}")
    support = [sum(row) for row in matrix]
    predicted = [sum(matrix[i][j] for i in range(size)) for j in range(size)]
    correct = [matrix[i][i] for i in range(size)]
    per_class_abstained = [row[-1] for row in matrix]
    return {"orientation": CONTRACT["per_class_confusion"]["orientation"],
            "abstention_column": "ABSTAINED",
            "matrix": [list(row) for row in matrix],
            "support": support, "predicted": predicted, "correct": correct,
            "abstained": per_class_abstained,
            "confusion_sha256": sha256_of({"labels": vocabulary["labels"],
                                           "matrix": [list(row) for row in matrix]})}


def _validate_population(document) -> dict:
    population = document["population"]
    _require(isinstance(population, dict), "population must be an object")
    for name in ("total", "answered", "abstained", "independent_units", "repeats"):
        _require(name in population, f"population.{name} is required")
        _require(isinstance(population[name], int) and not isinstance(population[name], bool)
                 and population[name] >= 0, f"population.{name} must be a count")
    _require(population["total"] > 0, "population.total must be positive")
    _require(population["answered"] + population["abstained"] == population["total"],
             "population.answered + population.abstained must equal population.total; "
             "an abstention is not a prediction and is not lost")
    repeats = population["repeats"]
    _require(repeats >= 1, "population.repeats must be at least 1")
    units = population["independent_units"]
    _require(units * repeats == population["total"],
             f"population.independent_units {units} times repeats {repeats} must equal "
             f"population.total {population['total']}; repeated prompts are not independent "
             f"examples")
    clustered = population.get("clustered_by", "NONE")
    _require(isinstance(clustered, str) and clustered,
             "population.clustered_by is required")
    if repeats > 1:
        _require(clustered != "NONE",
                 "population.clustered_by must name the unit repeats are clustered within "
                 "whenever repeats is greater than 1")
    return {"total": population["total"], "answered": population["answered"],
            "abstained": population["abstained"], "independent_units": units,
            "repeats": repeats, "clustered_by": clustered}


def _validate_abstention(document, population) -> dict:
    abstention = document["abstention"]
    _require(isinstance(abstention, dict), "abstention must be an object")
    for name in ("abstained", "answered", "coverage", "denominator_policy", "abstention_rule"):
        _require(name in abstention, f"abstention.{name} is required")
    _require(abstention["abstained"] == population["abstained"],
             "abstention.abstained disagrees with population.abstained")
    _require(abstention["answered"] == population["answered"],
             "abstention.answered disagrees with population.answered")
    expected = population["abstained"] / population["total"]
    coverage = _finite(abstention["coverage"], "abstention.coverage")
    _require(abs(coverage - expected) <= 1e-9,
             f"abstention.coverage {coverage} is not abstained/total ({expected}); "
             f"coverage is how much of the population abstained, stated, not inferred")
    policies = CONTRACT["abstention"]["denominator_policies"]
    _require(abstention["denominator_policy"] in policies,
             "abstention.denominator_policy must be one of " + ", ".join(sorted(policies)))
    _require(isinstance(abstention["abstention_rule"], str) and abstention["abstention_rule"],
             "abstention.abstention_rule must say what an abstention is")
    return {"abstained": population["abstained"], "answered": population["answered"],
            "coverage": coverage, "denominator_policy": abstention["denominator_policy"],
            "answered_fraction": population["answered"] / population["total"],
            "abstention_rule": abstention["abstention_rule"]}


def _validate_probability(document) -> tuple:
    semantics = document["probability_semantics"]
    declared = CONTRACT["probability_semantics"]["values"]
    _require(semantics in declared,
             f"probability_semantics {semantics!r} is not declared: "
             + ", ".join(sorted(declared)))
    calibrated = document["calibrated"]
    _require(isinstance(calibrated, bool),
             "calibrated must be a boolean of its own; it is not read off the semantics string")

    split = document["calibration_split"]
    _require(isinstance(split, dict), "calibration_split must be an object")
    for name in ("split_id", "population_sha256", "rows", "fitted_parameters"):
        _require(name in split, f"calibration_split.{name} is required")
    none_value = CONTRACT["calibration_split"]["none_value"]
    is_none = split["split_id"] == none_value

    calibrated_values = set(CONTRACT["probability_semantics"]["calibrated_values"])
    if calibrated:
        _require(not is_none,
                 "calibrated is true but calibration_split is NONE; a calibrated probability "
                 "names the split its calibration was fitted on")
        _require(semantics in calibrated_values,
                 f"calibrated is true while probability_semantics is {semantics}; the two "
                 f"fields contradict each other")
    else:
        _require(semantics not in calibrated_values,
                 f"probability_semantics {semantics} names a calibrated posterior while "
                 f"calibrated is false; the two fields contradict each other")

    if not is_none:
        _hex64(split["population_sha256"], "calibration_split.population_sha256")
        _require(split["split_id"] != document["evaluation_split"],
                 f"the calibration split {split['split_id']!r} is the evaluation split; "
                 f"calibration is kept distinct from evaluation")
        _require(split["population_sha256"] != document["evaluation_population_sha256"],
                 "the calibration population is the evaluation population; calibration is "
                 "kept distinct from evaluation")
        _require(isinstance(split["rows"], int) and split["rows"] > 0,
                 "calibration_split.rows must be a positive count")

    probability_metrics = document.get("probability_metrics") or {}
    _require(isinstance(probability_metrics, dict), "probability_metrics must be an object")
    declared_metrics = CONTRACT["probability_semantics"]["probability_metrics"]
    cleaned = {}
    if probability_metrics:
        posterior = semantics.startswith("SOFTMAX_POSTERIOR") or semantics == "ISOTONIC_CALIBRATED"
        _require(posterior,
                 f"probability_semantics {semantics} is not a posterior over the vocabulary, "
                 f"so it may not carry NLL, Brier or ECE: an entropy-derived confidence is "
                 f"not P(correct)")
        bins = probability_metrics.get("ece_bins")
        for name, value in probability_metrics.items():
            if name == "ece_bins":
                continue
            key = f"classification.{name}"
            _require(key in declared_metrics,
                     f"probability metric {name!r} is not declared: "
                     + ", ".join(sorted(declared_metrics)))
            cleaned[name] = _finite(value, f"probability_metrics.{name}")
        if "ece" in cleaned:
            _require(isinstance(bins, int) and bins > 1,
                     "probability_metrics.ece requires a declared ece_bins")
            cleaned["ece_bins"] = bins
    return semantics, calibrated, {
        "split_id": split["split_id"],
        "population_sha256": split["population_sha256"],
        "rows": split["rows"],
        "fitted_parameters": split["fitted_parameters"]}, cleaned


def _validate_confusion_agreement(primary, secondary, confusion, abstention, population):
    """A headline metric that contradicts its own confusion is refused.

    The confusion is carried in full and the metric is carried in full: neither
    is inferred from the other. This only refuses the case where the two, both
    present, disagree - which is exactly the case a reader cannot detect later.
    """
    derivable = {}
    correct = sum(confusion["correct"])
    if abstention["denominator_policy"] == "ANSWERED_ONLY":
        denominator = population["answered"]
    else:
        denominator = population["total"]
    if denominator:
        derivable["ACCURACY"] = correct / denominator

    size = len(confusion["support"])
    per_class_f1 = []
    for index in range(size):
        true_positive = confusion["correct"][index]
        predicted = confusion["predicted"][index]
        support = confusion["support"][index]
        if abstention["denominator_policy"] == "ANSWERED_ONLY":
            support = support - confusion["abstained"][index]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        per_class_f1.append(0.0 if precision + recall == 0
                            else 2 * precision * recall / (precision + recall))
    derivable["MACRO_F1"] = sum(per_class_f1) / size if size else 0.0

    carried = dict(secondary)
    carried.setdefault(primary["family"], primary["value"])
    for family, value in carried.items():
        if family not in derivable:
            continue
        if abs(value - derivable[family]) > 1e-6:
            raise ReceiptRefused(
                f"{family} is carried as {value} but the per-class confusion in the same "
                f"receipt gives {derivable[family]:.6f} under denominator policy "
                f"{abstention['denominator_policy']}; the two fields contradict each other")


def _validate_secondary(document, vocabulary) -> dict:
    secondary = document.get("secondary_metrics") or {}
    _require(isinstance(secondary, dict), "secondary_metrics must be an object")
    cleaned = {}
    for family, value in secondary.items():
        spec = family_spec(family)
        value = _finite(value, f"secondary_metrics.{family}")
        low, high = spec["range"]
        _require(low <= value <= high,
                 f"secondary_metrics.{family} {value} is outside {low}..{high}")
        cleaned[family] = value
    return cleaned


def _validate_provenance(document):
    evidence = document["evidence_class"]
    _require(evidence in CONTRACT["evidence_classes"],
             f"evidence_class {evidence!r} is not declared: "
             + ", ".join(sorted(CONTRACT["evidence_classes"])))
    corpus_class = document["corpus_class"]
    _require(corpus_class in CONTRACT["corpus_classes"],
             f"corpus_class {corpus_class!r} is not declared: "
             + ", ".join(sorted(CONTRACT["corpus_classes"])))
    _require(corpus_class != "ROUTER_PROMPT_CORPUS",
             "the router prompt corpus measures which envelope the router chooses, not "
             "whether a classifier answers correctly; a router corpus receipt may not carry "
             "a classification metric at all (use build_router_reliability_record)")
    _require(document["corpus_id"] != CONTRACT["router_corpus"]["corpus_id"],
             f"corpus_id {document['corpus_id']!r} is the router prompt corpus; it is not a "
             f"classification benchmark")

    regime = document["supervision_regime"]
    _require(regime in CONTRACT["supervision_regimes"],
             f"supervision_regime {regime!r} is not declared: "
             + ", ".join(CONTRACT["supervision_regimes"]))
    fits_head = document.get("labelled_rows_fit_head")
    _require(isinstance(fits_head, bool), "labelled_rows_fit_head is required")
    if fits_head:
        _require(regime in ("TRAINED_HEAD_ON_FROZEN_ENCODER", "FULL_FINETUNE"),
                 f"labelled rows fit a downstream head, so the regime is not "
                 f"{regime}: it is not zero-shot, even on the same test rows")

    declared_fields = document["declared_fields"]
    _require(isinstance(declared_fields, list),
             "declared_fields must be the list of fields that are declarations")
    known = set(CONTRACT["required_receipt_fields"])
    for field in declared_fields:
        _require(field in known,
                 f"declared_fields names {field!r}, which is not a receipt field")
    _require(isinstance(document["limitations"], str) and document["limitations"],
             "limitations is required; an unqualified receipt claims more than it measured")


def build_receipt(document: dict) -> dict:
    """Validate one classification evaluation and return its sealed receipt.

    Refuses, by name, any document that would let a later reader mistake what was
    measured. The returned object is the receipt: the warehouse projection is
    :func:`terminal_tags` and :func:`terminal_metrics`.
    """
    _require(isinstance(document, dict), "a receipt document must be an object")
    for field in CONTRACT["required_receipt_fields"]:
        if field in _DERIVED:
            continue
        _require(field in document, f"{field} is required and was not given")

    for name in ("task_id", "corpus_id", "provider", "checkpoint", "evaluation_split"):
        _require(isinstance(document[name], str) and document[name], f"{name} is required")
    _hex64(document["corpus_sha256"], "corpus_sha256")
    _checkpoint_digest(document["checkpoint_sha256"])
    _hex64(document["evaluation_population_sha256"], "evaluation_population_sha256")

    _validate_provenance(document)
    # where the evidence came from, before anything about what it says
    path = provenance.build_answering_path(document["answering_path"])
    provenance.quoted_record_check(path=path, checkpoint=document["checkpoint"],
                                   checkpoint_sha256=document["checkpoint_sha256"])
    evidence_role = provenance.promotion_check(
        path=path, evidence_class=document["evidence_class"],
        corpus_class=document["corpus_class"])
    primary = _validate_primary(document)
    naive = _validate_naive(document, primary)
    vocabulary = _validate_vocabulary(document)
    population = _validate_population(document)
    confusion = _validate_confusion(document, vocabulary, population)
    abstention = _validate_abstention(document, population)
    semantics, calibrated, calibration, probability_metrics = _validate_probability(document)
    secondary = _validate_secondary(document, vocabulary)
    _validate_confusion_agreement(primary, secondary, confusion, abstention, population)

    receipt = {
        "schema": SCHEMA,
        "task_id": document["task_id"],
        "corpus_class": document["corpus_class"],
        "corpus_id": document["corpus_id"],
        "corpus_sha256": document["corpus_sha256"],
        "evidence_class": document["evidence_class"],
        "supervision_regime": document["supervision_regime"],
        "labelled_rows_fit_head": document["labelled_rows_fit_head"],
        "provider": document["provider"],
        "checkpoint": document["checkpoint"],
        "checkpoint_sha256": document["checkpoint_sha256"],
        "answering_path": path,
        "evidence_role": evidence_role,
        "author_primary_metric": {k: primary[k] for k in
                                  ("family", "name", "value", "denominator_policy",
                                   "metric", "unit")},
        "paired_naive": naive,
        "class_vocabulary": vocabulary,
        "per_class_confusion": confusion,
        "probability_semantics": semantics,
        "calibrated": calibrated,
        "calibration_split": calibration,
        "probability_metrics": probability_metrics,
        "secondary_metrics": secondary,
        "abstention": abstention,
        "population": population,
        "evaluation_split": document["evaluation_split"],
        "evaluation_population_sha256": document["evaluation_population_sha256"],
        "protocol_sha256": _hex64(document.get("protocol_sha256", ""), "protocol_sha256"),
        "scorer_sha256": _hex64(document.get("scorer_sha256", ""), "scorer_sha256"),
        "seed": None if document.get("seed") is None else str(document["seed"]),
        "declared_fields": sorted(document["declared_fields"]),
        "limitations": document["limitations"],
        "authorises_broker_deployment": False,
        "execution_authority": "NONE",
    }
    receipt["metric_identity_sha256"] = metric_identity_sha256(receipt)
    receipt["receipt_id"] = (f"{receipt['task_id']}:{receipt['corpus_id']}:"
                             f"{receipt['provider']}:{receipt['author_primary_metric']['family']}")
    receipt["receipt_sha256"] = sha256_of(receipt)
    return receipt


# --------------------------------------------------------------------------- #
# reading, and refusing to misread
# --------------------------------------------------------------------------- #

def carried_families(receipt: dict) -> set:
    return {receipt["author_primary_metric"]["family"]} | set(receipt.get("secondary_metrics") or {})


def read_metric(receipt: dict, family: str) -> float:
    """The value of one metric family, or a refusal naming both families.

    There is no nearest match and no default. A caller that asks a MAP receipt
    for accuracy gets an exception that names MAP and accuracy, not a float.
    """
    family_spec(family)
    primary = receipt["author_primary_metric"]
    if primary["family"] == family:
        return primary["value"]
    secondary = receipt.get("secondary_metrics") or {}
    if family in secondary:
        return secondary[family]
    raise MetricNotCarried(
        f"this receipt carries {primary['family']} (the author calls it "
        f"{primary['name']!r}) and does not carry {family}; {primary['family']} and {family} "
        f"are different metrics and neither substitutes for the other")


def compare(left: dict, right: dict) -> dict:
    """Compare two receipts, or refuse by name.

    Refuses across metric families (:class:`IncomparableMetrics`) and across
    protocols within one family (:class:`IncomparableProtocol`).
    """
    left_family = left["author_primary_metric"]["family"]
    right_family = right["author_primary_metric"]["family"]
    if left_family != right_family:
        raise IncomparableMetrics(
            f"{left_family} and {right_family} are different metrics: "
            f"{family_spec(left_family)['kind']} against "
            f"{family_spec(right_family)['kind']}. Both lie in a shared numeric interval and "
            f"neither is a version of the other, so no difference, ratio or ranking between "
            f"{left_family} and {right_family} is defined here")
    for field, why in (("task_id", "a different task"),
                       ("evaluation_population_sha256", "a different evaluation population"),
                       ("metric_identity_sha256", "a different metric identity")):
        if left[field] != right[field]:
            raise IncomparableProtocol(
                f"both receipts carry {left_family} but on {why} "
                f"({field}: {left[field]} against {right[field]})")
    return {
        "family": left_family,
        "author_name": left["author_primary_metric"]["name"],
        "unit": left["author_primary_metric"]["unit"],
        "left": left["author_primary_metric"]["value"],
        "right": right["author_primary_metric"]["value"],
        "difference": left["author_primary_metric"]["value"] - right["author_primary_metric"]["value"],
        "left_naive": left["paired_naive"]["value"],
        "right_naive": right["paired_naive"]["value"],
        "population": left["population"]["total"],
        "independent_units": left["population"]["independent_units"],
    }


# --------------------------------------------------------------------------- #
# the warehouse projection: the existing governed terminal, nothing new
# --------------------------------------------------------------------------- #

#: Above this canonical size the full matrix stays in the receipt document and the
#: terminal carries only its digest. A 77-class matrix does not belong in a tag.
CONFUSION_TAG_BUDGET_BYTES = 8192


def _require_receipt(receipt):
    """Only a sealed classification receipt has a warehouse projection.

    A router reliability record reaches this function only if somebody is trying
    to write a router number where a classifier number is read. There is no
    conversion: it is refused by name.
    """
    _require(isinstance(receipt, dict), "a receipt must be an object")
    schema = receipt.get("schema")
    if schema == ROUTER_RECORD_SCHEMA:
        raise ReceiptRefused(
            f"a {ROUTER_RECORD_SCHEMA} record has no classification metric projection: it "
            f"measures the router over {receipt.get('prompts')} prompts repeated "
            f"{receipt.get('repeats')} times and is not a classifier score")
    _require(schema == SCHEMA, f"expected a {SCHEMA} receipt, not {schema!r}")


def terminal_tags(receipt: dict) -> dict:
    """The contract context, as string tags on the existing governed terminal."""
    _require_receipt(receipt)
    confusion = receipt["per_class_confusion"]
    matrix_text = canonical_text({"labels": receipt["class_vocabulary"]["labels"],
                                  "matrix": confusion["matrix"]})
    tags = {
        "metric_contract": SCHEMA,
        "task_id": receipt["task_id"],
        "corpus_class": receipt["corpus_class"],
        "corpus_id": receipt["corpus_id"],
        "corpus_sha256": receipt["corpus_sha256"],
        "evidence_class": receipt["evidence_class"],
        "supervision_regime": receipt["supervision_regime"],
        "provider": receipt["provider"],
        "checkpoint": receipt["checkpoint"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        **provenance.provenance_tags(receipt["answering_path"], receipt["evidence_role"]),
        "class_vocabulary_size": str(receipt["class_vocabulary"]["size"]),
        "label_order_sha256": receipt["class_vocabulary"]["label_order_sha256"],
        "evaluation_split": receipt["evaluation_split"],
        "evaluation_population_sha256": receipt["evaluation_population_sha256"],
        "train_label_population_sha256": str(receipt["paired_naive"]["train_label_population_sha256"]),
        "calibration_split": receipt["calibration_split"]["split_id"],
        "calibration_population_sha256": str(receipt["calibration_split"]["population_sha256"]),
        "probability_semantics": receipt["probability_semantics"],
        "calibrated": "true" if receipt["calibrated"] else "false",
        "denominator_policy": receipt["abstention"]["denominator_policy"],
        "population_total": str(receipt["population"]["total"]),
        "population_answered": str(receipt["population"]["answered"]),
        "population_abstained": str(receipt["population"]["abstained"]),
        "independent_units": str(receipt["population"]["independent_units"]),
        "repeats": str(receipt["population"]["repeats"]),
        "clustered_by": receipt["population"]["clustered_by"],
        "author_primary_metric_family": receipt["author_primary_metric"]["family"],
        "author_primary_metric_name": receipt["author_primary_metric"]["name"],
        "naive_policy": receipt["paired_naive"]["policy"],
        "metric_identity_sha256": receipt["metric_identity_sha256"],
        "confusion_sha256": confusion["confusion_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "scorer_sha256": receipt["scorer_sha256"],
        "seed": str(receipt["seed"]),
        "declared_fields": ",".join(receipt["declared_fields"]) or "NONE",
        "receipt_sha256": receipt["receipt_sha256"],
        "class_vocabulary_json": canonical_text(receipt["class_vocabulary"]["labels"]),
        "abstention_rule": receipt["abstention"]["abstention_rule"],
        "limitations": receipt["limitations"],
        "execution_authority": receipt["execution_authority"],
        "confusion_matrix_json": (matrix_text
                                  if len(matrix_text.encode()) <= CONFUSION_TAG_BUDGET_BYTES
                                  else "CONFUSION_IN_RECEIPT_ONLY"),
    }
    missing = [name for name in CONTRACT["required_context_tags"] if name not in tags]
    _require(not missing, "the terminal projection is missing required tags: "
                          + ", ".join(missing))
    return tags


def _row(metric, value, split, unit):
    return {"metric": metric, "value": float(value), "split": split, "horizon": None,
            "std_dev": None, "min_value": None, "max_value": None, "unit": unit}


def terminal_metrics(receipt: dict) -> list:
    """The metric rows, in the existing nine-field governed metric schema.

    The primary metric comes first, then its paired naive under a key of its own,
    then coverage, then the per-class confusion summary. Every row carries the
    unit of its own family, so no row can be read as another family's.
    """
    _require_receipt(receipt)
    split = receipt["evaluation_split"]
    primary = receipt["author_primary_metric"]
    rows = [_row(primary["metric"], primary["value"], split, primary["unit"]),
            _row(receipt["paired_naive"]["metric"], receipt["paired_naive"]["value"],
                 split, receipt["paired_naive"]["unit"])]
    for family, value in sorted((receipt.get("secondary_metrics") or {}).items()):
        spec = family_spec(family)
        rows.append(_row(spec["metric"], value, split, spec["unit"]))
    for name, value in sorted((receipt.get("probability_metrics") or {}).items()):
        if name == "ece_bins":
            continue
        key = f"classification.{name}"
        rows.append(_row(key, value, split,
                         CONTRACT["probability_semantics"]["probability_metrics"][key]))
    rows.append(_row("classification.abstention_coverage", receipt["abstention"]["coverage"],
                     split, "abstained_fraction"))
    rows.append(_row("classification.answered_fraction",
                     receipt["abstention"]["answered_fraction"], split, "answered_fraction"))
    confusion = receipt["per_class_confusion"]
    for index in range(receipt["class_vocabulary"]["size"]):
        for kind in ("support", "predicted", "correct", "abstained"):
            rows.append(_row(f"classification.{kind}.class_{index}",
                             confusion[kind][index], split, "rows"))
    return rows


# --------------------------------------------------------------------------- #
# the router: measured, published, and kept out of the classifier's ledger
# --------------------------------------------------------------------------- #

def build_router_reliability_record(*, corpus_id, correct, stored_verdicts,
                                    prompts, repeats, evidence_class,
                                    completion_pass=None, note=None) -> dict:
    """How often the router picks the right envelope. Deliberately not a receipt.

    This returns `router_reliability.v1`, a different schema that carries no
    metric family, no class vocabulary and no provider quality. That is the
    point: there is no route by which this number becomes a classifier's score,
    because it is never written in the shape a classifier's score is written in.
    """
    _require(evidence_class in CONTRACT["evidence_classes"],
             f"evidence_class {evidence_class!r} is not declared")
    for name, value in (("correct", correct), ("stored_verdicts", stored_verdicts),
                        ("prompts", prompts), ("repeats", repeats)):
        _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                 f"{name} must be a count")
    _require(prompts * repeats == stored_verdicts,
             f"{prompts} prompts times {repeats} repeats is {prompts * repeats}, not "
             f"{stored_verdicts} stored verdicts")
    _require(correct <= stored_verdicts, "correct cannot exceed the stored verdicts")
    return {
        "schema": ROUTER_RECORD_SCHEMA,
        "corpus_id": corpus_id,
        "measures": CONTRACT["router_corpus"]["measures"],
        "evidence_class": evidence_class,
        "correct_verdicts": correct,
        "stored_verdicts": stored_verdicts,
        "prompts": prompts,
        "repeats": repeats,
        "independent_units": prompts,
        "clustered_by": "prompt",
        "completion_pass": completion_pass,
        "rate_over_stored_verdicts": correct / stored_verdicts if stored_verdicts else None,
        "is_classifier_quality": False,
        "authorises_broker_deployment": False,
        "execution_authority": "NONE",
        "note": note or ("a rate over repeated prompts on development evidence; it is not a "
                         "classifier accuracy and not an independent-example proportion"),
    }


# --------------------------------------------------------------------------- #
# the provider quality badge, and the two things it will not come from
# --------------------------------------------------------------------------- #

def provider_quality_badge(provider: str, evidence: list) -> dict:
    """A quality badge for one provider, or a named refusal.

    Refuses a router record, a declaration, a recount, a published reference, a
    transport fixture, and any evidence set without a measurement on a held-out
    business corpus with a same-row train-derived naive. Whatever it returns
    carries no execution authority.
    """
    _require(isinstance(provider, str) and provider, "provider is required")
    _require(isinstance(evidence, list) and evidence, "at least one evidence record is required")

    for record in evidence:
        if not isinstance(record, dict):
            raise BadgeRefused("BADGE_REFUSED_DECLARATION_IS_NOT_MEASUREMENT",
                               "an evidence record that is not a record")
        if record.get("schema") == ROUTER_RECORD_SCHEMA or \
                record.get("corpus_id") == CONTRACT["router_corpus"]["corpus_id"] or \
                record.get("corpus_class") == "ROUTER_PROMPT_CORPUS":
            raise BadgeRefused(
                "BADGE_REFUSED_ROUTER_SCORE_IS_NOT_CLASSIFIER_QUALITY",
                f"the evidence includes {record.get('corpus_id')}, which is "
                f"{CONTRACT['router_corpus']['prompts']} prompts repeated "
                f"{CONTRACT['router_corpus']['repeats']} times and measures the router, not "
                f"any classifier")
        if record.get("schema") != SCHEMA:
            raise BadgeRefused("BADGE_REFUSED_DECLARATION_IS_NOT_MEASUREMENT",
                               f"an evidence record of schema {record.get('schema')!r} is not "
                               f"a {SCHEMA} receipt")
        if record["evidence_class"] == "TRANSPORT_TEST_NOT_SCIENCE":
            raise BadgeRefused("BADGE_REFUSED_TRANSPORT_TEST_IS_NOT_SCIENCE",
                               f"{record['receipt_id']} carries fabricated transport values")
        if record["evidence_class"] != "MEASUREMENT":
            raise BadgeRefused("BADGE_REFUSED_DECLARATION_IS_NOT_MEASUREMENT",
                               f"{record['receipt_id']} is {record['evidence_class']}, not a "
                               f"measurement executed in this programme")
        # and the label is not the evidence: what answered decides
        refusal = provenance.badge_provenance_refusal(record)
        if refusal is not None:
            raise BadgeRefused(*refusal)

    business = [r for r in evidence if r["corpus_class"] == "BUSINESS_HELD_OUT"]
    if not business:
        raise BadgeRefused(
            "BADGE_REFUSED_NO_BUSINESS_CORPUS_MEASUREMENT",
            "public benchmark accuracy does not establish that the provider is useful on "
            "this programme's own business material; no BUSINESS_HELD_OUT measurement was "
            "offered")
    for record in business:
        if record["paired_naive"]["policy"] not in ("MAJORITY_CLASS_FROM_TRAIN",
                                                    "STRATIFIED_PRIOR_FROM_TRAIN",
                                                    "DEVELOPMENT_FIXED_KEYWORD_RULE"):
            raise BadgeRefused("BADGE_REFUSED_NO_PAIRED_NAIVE",
                               f"{record['receipt_id']} carries no train-derived or "
                               f"development-fixed naive on the same rows")

    return {
        "schema": "provider_quality_badge.v1",
        "provider": provider,
        "supported_by": [r["receipt_sha256"] for r in evidence],
        "business_evidence": [
            {"receipt_id": r["receipt_id"],
             "task_id": r["task_id"],
             "corpus_id": r["corpus_id"],
             "metric_family": r["author_primary_metric"]["family"],
             "metric_name": r["author_primary_metric"]["name"],
             "unit": r["author_primary_metric"]["unit"],
             "value": r["author_primary_metric"]["value"],
             "naive_policy": r["paired_naive"]["policy"],
             "naive_value": r["paired_naive"]["value"],
             "abstention_coverage": r["abstention"]["coverage"],
             "population_total": r["population"]["total"],
             "independent_units": r["population"]["independent_units"],
             "metric_identity_sha256": r["metric_identity_sha256"],
             "answering_path_id": r["answering_path"]["path_id"],
             "answering_path_kind": r["answering_path"]["kind"],
             "served_checkpoint_sha256": r["answering_path"]["served_checkpoint_sha256"],
             "provenance_attestation": r["answering_path"]["attestation"],
             "limitations": r["limitations"]}
            for r in business],
        "router_evidence": "EXCLUDED_BY_CONTRACT",
        "authorises_broker_deployment": False,
        "execution_authority": "NONE",
        "note": "a measured comparison against a same-row naive on named corpora; it is not "
                "a deployment decision and grants nothing",
    }
