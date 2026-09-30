"""Per-row metric identity: a secondary row does not inherit the primary's identity.

The defect this module owns
---------------------------
`classification_metrics.v1` computes `metric_identity_sha256` from
`author_primary_metric`, and `app.classification_receipt.terminal_tags` puts it on
the terminal. A terminal-level digest binds the receipt's **primary** metric, so
every other row a receipt projects - its paired naive, its secondary families, its
probability metrics, its coverage and its per-class confusion - reaches the
warehouse carrying an identity that is not its own.

Proved against the live warehouse on 2026-09-29, not argued:

    identity 1756f877 (ACCURACY)  native-accuracy     classification.macro_f1 = 0.9467546527629132
    identity a62f184c (MACRO_F1)  native-macro-f1     classification.macro_f1 = 0.9467546527629132
    identity 1756f877 (ACCURACY)  framework-accuracy  classification.macro_f1 = 0.9252733932274245

One macro-F1 value under two identities, and one identity over two different
macro-F1 values. Nothing lets one metric be read as another - every row's key and
unit are right - but the contract's own guidance, *two rows agree only when their
identity digests agree*, is **unsound for secondary rows**.

Three separations, and the third is the one that bites
------------------------------------------------------
1. **Its own identity per row.** :func:`row_identity_sha256` digests only what
   decides what the number in that row *means*: family, metric key, unit, kind,
   definition, governing denominator policy, label order. Not the author's label
   for it, not the row's role, not the provider, not the value.

2. **Evidence class separated from identity.** A published accuracy and a measured
   accuracy of one definition share an identity - that is what makes them
   comparable, and precisely why they may not be averaged. So provenance is a
   *separate dimension*: `evidence_class` is out of the identity and into the
   occurrence key, and an aggregate that mixes evidence classes in one group is
   refused by name.

3. **No double counting between terminals.** This is the real risk. Two terminals
   of one campaign can carry the *same* measurement - here the two native
   terminals each store accuracy 0.9525, macro-F1 0.9467546527629132 and the same
   confusion, one as primary and one as secondary. An aggregate keyed on identity
   would average that measurement twice. :func:`metric_occurrence_sha256` names
   *which measurement* a row reports, deliberately without the row's role and
   without the terminal, campaign or receipt that carried it, so two reports of
   one measurement collapse to one occurrence - while a genuine replicate under a
   different protocol, scorer or seed stays its own occurrence.

What this module does not do
----------------------------
It does not edit `classification_metrics.v1`, does not change
`app/classification_receipt.py` - whose digest the sealed CB04 business corpus
manifest pins - and never rewrites a stored row. It is a successor contract:
`docs/contracts/classification_row_identity.v1.json`. A correction to an accepted
row travels as a superseding generation carrying these tags and its reason.

Nothing here carries execution authority.
"""

from __future__ import annotations

import json
from pathlib import Path

from app import classification_receipt as cr

CONTRACT_PATH = (Path(__file__).resolve().parents[1]
                 / "docs/contracts/classification_row_identity.v1.json")
CONTRACT = json.loads(CONTRACT_PATH.read_text())
SCHEMA = CONTRACT["schema"]
ROW_ROLES = tuple(CONTRACT["row_roles"])
TAG_BUDGET_BYTES = int(CONTRACT["tag_budget_bytes"])
OVER_BUDGET = CONTRACT["over_budget_sentinel"]

_METRIC_CONTRACT = cr.SCHEMA


class RowIdentityRefused(cr.ClassificationReceiptError):
    """A row, a terminal or an aggregate that would misstate what it counts."""

    def __init__(self, refusal: str, detail: str):
        super().__init__(f"{refusal}: {detail}")
        self.refusal = refusal
        self.detail = detail


def _refuse(refusal: str, detail: str):
    if refusal not in CONTRACT["refusals"]:
        raise AssertionError(f"{refusal} is not a declared refusal of {SCHEMA}")
    raise RowIdentityRefused(refusal, detail)


# --------------------------------------------------------------------------- #
# the two digests
# --------------------------------------------------------------------------- #

def row_identity_sha256(*, family, metric, unit, kind, definition,
                        denominator_policy, label_order_sha256) -> str:
    """What the number in one row means, and nothing else.

    `family` is ``None`` for a row that carries no metric family: a probability
    metric, a coverage share, a confusion count. The metric key is in the digest,
    so `classification.support.class_0` and `classification.support.class_1` are
    different identities without a separate index field.
    """
    return cr.sha256_of({
        "schema": SCHEMA,
        "metric_contract": _METRIC_CONTRACT,
        "family": family,
        "metric": metric,
        "unit": unit,
        "kind": kind,
        "definition": definition,
        "denominator_policy": denominator_policy,
        "label_order_sha256": label_order_sha256,
    })


def metric_occurrence_sha256(*, row_identity_sha256, task_id, corpus_id, corpus_sha256,
                             provider, checkpoint_sha256, supervision_regime,
                             evidence_class, evaluation_split,
                             evaluation_population_sha256, protocol_sha256,
                             scorer_sha256, seed) -> str:
    """Which measurement a row reports: the double-count guard.

    The row's role, and the terminal, campaign, unit, generation and receipt that
    carried it, are all absent on purpose. A double count *between terminals* is
    the failure this key exists to catch, and a key that carried the terminal
    could not catch it.
    """
    return cr.sha256_of({
        "schema": SCHEMA,
        "row_identity_sha256": row_identity_sha256,
        "task_id": task_id,
        "corpus_id": corpus_id,
        "corpus_sha256": corpus_sha256,
        "provider": provider,
        "checkpoint_sha256": checkpoint_sha256,
        "supervision_regime": supervision_regime,
        "evidence_class": evidence_class,
        "evaluation_split": evaluation_split,
        "evaluation_population_sha256": evaluation_population_sha256,
        "protocol_sha256": protocol_sha256,
        "scorer_sha256": scorer_sha256,
        "seed": None if seed is None else str(seed),
    })


# --------------------------------------------------------------------------- #
# the identity of every row a receipt projects
# --------------------------------------------------------------------------- #

def _governing_denominator_policy(receipt) -> str:
    governing = receipt["abstention"]["denominator_policy"]
    declared = receipt["author_primary_metric"]["denominator_policy"]
    if declared != governing:
        _refuse("PRIMARY_AND_ABSTENTION_DENOMINATOR_POLICIES_DISAGREE",
                f"author_primary_metric.denominator_policy is {declared} and "
                f"abstention.denominator_policy is {governing}; the contract checks every "
                f"carried value against the confusion under the abstention policy, so a "
                f"second policy on the primary would digest a meaning the check does not use")
    return governing


#: the description of one projected row, shared by the producer and the reader so
#: the two cannot drift: everything it needs is either the metric key itself or a
#: field that reaches the warehouse as a tag.
_COVERAGE_UNITS = {"classification.abstention_coverage": "abstained_fraction",
                   "classification.answered_fraction": "answered_fraction"}
_CONFUSION_KINDS = ("support", "predicted", "correct", "abstained")


def _row_context(*, denominator_policy, label_order_sha256, naive_policy,
                 probability_semantics, calibrated) -> dict:
    return {"denominator_policy": denominator_policy,
            "label_order_sha256": label_order_sha256,
            "naive_policy": naive_policy,
            "probability_semantics": probability_semantics,
            "calibrated": bool(calibrated)}


def context_from_receipt(receipt) -> dict:
    return _row_context(
        denominator_policy=_governing_denominator_policy(receipt),
        label_order_sha256=receipt["class_vocabulary"]["label_order_sha256"],
        naive_policy=receipt["paired_naive"]["policy"],
        probability_semantics=receipt["probability_semantics"],
        calibrated=receipt["calibrated"])


def context_from_tags(tags) -> dict:
    """The same context, from the tags a `classification_metrics.v1` terminal stores."""
    return _row_context(
        denominator_policy=tags["denominator_policy"],
        label_order_sha256=tags["label_order_sha256"],
        naive_policy=tags["naive_policy"],
        probability_semantics=tags["probability_semantics"],
        calibrated=str(tags["calibrated"]).lower() == "true")


def describe_row(metric: str, context: dict, *, primary_family=None) -> dict:
    """What one metric key means, by rule, from the key and the stored context.

    `primary_family` only decides whether a family-bearing row is labelled PRIMARY
    or SECONDARY. It is NOT in the identity digest - that is the point - so a
    reader who does not know it still recomputes the same identity.
    """
    families = cr.CONTRACT["metric_families"]
    probability_units = cr.CONTRACT["probability_semantics"]["probability_metrics"]

    for family, spec in families.items():
        if spec["metric"] == metric:
            role = "PRIMARY" if family == primary_family else "SECONDARY"
            return {"metric": metric, "row_role": role, "family": family,
                    "unit": spec["unit"], "kind": spec["kind"],
                    "definition": spec["definition"], **context}
        head, _, tail = spec["metric"].rpartition(".")
        if metric == f"{head}.naive_{tail}":
            return {"metric": metric, "row_role": "PAIRED_NAIVE", "family": family,
                    "unit": spec["unit"], "kind": spec["kind"],
                    "definition": (f"{spec['definition']}; scored for the paired naive "
                                   f"under policy {context['naive_policy']} fitted on "
                                   f"train labels only"), **context}

    if metric in probability_units:
        return {"metric": metric, "row_role": "PROBABILITY", "family": None,
                "unit": probability_units[metric], "kind": "PROBABILITY_METRIC",
                "definition": (f"{metric} over the predicted probabilities of the "
                               f"evaluation population, with probability semantics "
                               f"{context['probability_semantics']} and calibrated="
                               f"{'true' if context['calibrated'] else 'false'}"),
                **context}

    if metric in _COVERAGE_UNITS:
        return {"metric": metric, "row_role": "COVERAGE", "family": None,
                "unit": _COVERAGE_UNITS[metric], "kind": "POPULATION_SHARE",
                "definition": (f"{metric} as a share of the full evaluation population, "
                               f"under the receipt's declared abstention rule"), **context}

    parts = metric.split(".")
    if len(parts) == 3 and parts[0] == "classification" and parts[1] in _CONFUSION_KINDS:
        index = int(parts[2].split("_")[1])
        return {"metric": metric, "row_role": "CONFUSION_CELL", "family": None,
                "unit": "rows", "kind": "CONFUSION_COUNT",
                "definition": (f"count of evaluation rows in the {parts[1]} cell of class "
                               f"index {index} of the declared label order"), **context}

    _refuse("ROW_IDENTITY_TAGS_MISSING",
            f"{metric} is not a row this contract identifies by rule; a producer that "
            f"projects it must extend docs/contracts/classification_row_identity.v1.json "
            f"rather than leave the row without an identity of its own")


def _identity_inputs(receipt) -> list:
    """Every projected row, described in `terminal_metrics` order.

    The order is asserted against `terminal_metrics` itself, so a row can never be
    given an identity that belongs to a different row.
    """
    context = context_from_receipt(receipt)
    primary_family = receipt["author_primary_metric"]["family"]
    rows = [describe_row(row["metric"], context, primary_family=primary_family)
            for row in cr.terminal_metrics(receipt)]
    projected = [row["metric"] for row in cr.terminal_metrics(receipt)]
    if projected != [row["metric"] for row in rows]:
        raise AssertionError("the row identities are not aligned with terminal_metrics")
    return rows


def metric_rows_with_identity(receipt: dict) -> list:
    """`terminal_metrics` rows, each with its OWN identity, role and occurrence.

    The nine stored metric fields are returned unchanged - this adds fields beside
    them and alters none - so the warehouse row stays exactly the row the existing
    schema accepts.
    """
    cr._require_receipt(receipt)
    stored = cr.terminal_metrics(receipt)
    described = _identity_inputs(receipt)
    subject = {
        "task_id": receipt["task_id"],
        "corpus_id": receipt["corpus_id"],
        "corpus_sha256": receipt["corpus_sha256"],
        "provider": receipt["provider"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "supervision_regime": receipt["supervision_regime"],
        "evidence_class": receipt["evidence_class"],
        "evaluation_split": receipt["evaluation_split"],
        "evaluation_population_sha256": receipt["evaluation_population_sha256"],
        "protocol_sha256": receipt["protocol_sha256"],
        "scorer_sha256": receipt["scorer_sha256"],
        "seed": receipt["seed"],
    }
    out = []
    for row, described_row in zip(stored, described):
        identity = row_identity_sha256(
            family=described_row["family"], metric=described_row["metric"],
            unit=described_row["unit"], kind=described_row["kind"],
            definition=described_row["definition"],
            denominator_policy=described_row["denominator_policy"],
            label_order_sha256=described_row["label_order_sha256"])
        if row["unit"] != described_row["unit"]:
            raise AssertionError(f"{row['metric']} is stored with unit {row['unit']} and "
                                 f"identified with unit {described_row['unit']}")
        out.append(dict(
            row,
            row_role=described_row["row_role"],
            family=described_row["family"],
            kind=described_row["kind"],
            denominator_policy=described_row["denominator_policy"],
            evidence_class=receipt["evidence_class"],
            reported_under_name=(receipt["author_primary_metric"]["name"]
                                 if described_row["row_role"] == "PRIMARY" else None),
            row_identity_sha256=identity,
            metric_occurrence_sha256=metric_occurrence_sha256(
                row_identity_sha256=identity, **subject),
        ))
    return out


# --------------------------------------------------------------------------- #
# the additive tags, and the refusal of a terminal without them
# --------------------------------------------------------------------------- #

def _fits(mapping) -> str:
    text = cr.canonical_text(mapping)
    return text if len(text.encode()) <= TAG_BUDGET_BYTES else OVER_BUDGET


def row_identity_tags(receipt: dict) -> dict:
    """Tags a `classification_metrics.v1` terminal must carry as well.

    The two map digests always cover EVERY projected row. The JSON maps carry the
    rows that fit a tag and otherwise the declared sentinel: a 77-class confusion
    contributes 308 rows, and any one of them is recomputable from these tags with
    :func:`recompute_from_tags`.
    """
    rows = metric_rows_with_identity(receipt)
    identities = {row["metric"]: row["row_identity_sha256"] for row in rows}
    occurrences = {row["metric"]: row["metric_occurrence_sha256"] for row in rows}
    roles = {row["metric"]: row["row_role"] for row in rows}
    small = [row["metric"] for row in rows if row["row_role"] != "CONFUSION_CELL"]
    tags = {
        "row_identity_contract": SCHEMA,
        "metric_row_count": str(len(rows)),
        "metric_row_identity_map_sha256": cr.sha256_of(identities),
        "metric_row_occurrence_map_sha256": cr.sha256_of(occurrences),
        "metric_row_identity_rule": (
            "app/classification_row_identity.py:row_identity_sha256 over the identity_inputs "
            "declared in docs/contracts/classification_row_identity.v1.json; recompute any "
            "row from the stored tags with recompute_from_tags(tags, metric_key)"),
        "metric_row_identity_json": _fits({k: identities[k] for k in small}),
        "metric_row_occurrence_json": _fits({k: occurrences[k] for k in small}),
        "metric_row_roles_json": _fits({k: roles[k] for k in small}),
        "author_primary_metric_denominator_policy":
            receipt["author_primary_metric"]["denominator_policy"],
        "metric_identity_sha256_scope":
            "TERMINAL_LEVEL_PRIMARY_ONLY_DO_NOT_USE_FOR_SECONDARY_ROWS",
        "evidence_class_is_a_separate_dimension_from_metric_identity": "true",
        "aggregation_rule": "; ".join(CONTRACT["aggregation_rule"]),
        "double_count_guard": "metric_occurrence_sha256",
        "probability_semantics_of_probability_rows": receipt["probability_semantics"],
    }
    missing = [name for name in CONTRACT["required_row_identity_tags"] if name not in tags]
    if missing:
        raise AssertionError("row identity tags incomplete: " + ", ".join(missing))
    return tags


def terminal_tags_with_row_identity(receipt: dict) -> dict:
    """What a producer under this contract writes: the existing tags plus these.

    `app.classification_receipt.terminal_tags` is called unmodified, so a terminal
    written here is the terminal the old function produced with strictly more
    fields beside it. No existing tag changes value.
    """
    base = cr.terminal_tags(receipt)
    extra = row_identity_tags(receipt)
    clash = sorted(set(base) & set(extra))
    if clash:
        raise AssertionError("row identity tags would overwrite contract tags: "
                             + ", ".join(clash))
    return {**base, **extra}


def assert_row_identity_tags(tags: dict) -> dict:
    """Refuse a contract terminal whose secondary rows have no identity of their own."""
    if tags.get("metric_contract") != _METRIC_CONTRACT:
        _refuse("ROW_IDENTITY_TAGS_MISSING",
                f"this is not a {_METRIC_CONTRACT} terminal: metric_contract is "
                f"{tags.get('metric_contract')!r}")
    missing = [name for name in CONTRACT["required_row_identity_tags"] if name not in tags]
    if missing:
        _refuse("ROW_IDENTITY_TAGS_MISSING",
                f"a {_METRIC_CONTRACT} terminal without {', '.join(missing)}; its secondary "
                f"rows would inherit the primary's metric_identity_sha256, which is not "
                f"their own, and an aggregate keyed on it would combine rows that measure "
                f"different things or the same thing twice")
    return {"row_identity_contract": tags["row_identity_contract"],
            "metric_row_count": int(tags["metric_row_count"]),
            "metric_row_identity_map_sha256": tags["metric_row_identity_map_sha256"],
            "metric_row_occurrence_map_sha256": tags["metric_row_occurrence_map_sha256"]}


# --------------------------------------------------------------------------- #
# reader side: any row's identity, from the stored tags alone
# --------------------------------------------------------------------------- #

def recompute_from_tags(tags: dict, metric_key: str, *, unit=None) -> dict:
    """The identity and occurrence of one stored row, from the terminal's tags alone.

    A reader holding nothing but the warehouse obtains the identity of any row,
    including the 308 confusion cells of a 77-class matrix that no tag could hold,
    and checks the result against `metric_row_identity_map_sha256`. It goes through
    the same :func:`describe_row` the producer used, so reader and producer cannot
    disagree about what a row means.

    `unit` is accepted only to be cross-checked against the rule; it never decides.
    """
    assert_row_identity_tags(tags)
    described = describe_row(metric_key, context_from_tags(tags),
                             primary_family=tags.get("author_primary_metric_family"))
    if unit is not None and unit != described["unit"]:
        raise AssertionError(f"{metric_key} is stored with unit {unit} and identified by "
                             f"rule with unit {described['unit']}")
    identity = row_identity_sha256(
        family=described["family"], metric=metric_key, unit=described["unit"],
        kind=described["kind"], definition=described["definition"],
        denominator_policy=described["denominator_policy"],
        label_order_sha256=described["label_order_sha256"])
    known = (json.loads(tags["metric_row_identity_json"])
             if tags["metric_row_identity_json"] != OVER_BUDGET else {})
    if metric_key in known and known[metric_key] != identity:
        raise AssertionError(f"{metric_key}: recomputed identity {identity} does not match "
                             f"the stored map entry {known[metric_key]}")
    return {"metric": metric_key, "row_role": described["row_role"],
            "family": described["family"], "kind": described["kind"],
            "row_identity_sha256": identity,
            "metric_occurrence_sha256": metric_occurrence_sha256(
                row_identity_sha256=identity, task_id=tags["task_id"],
                corpus_id=tags["corpus_id"], corpus_sha256=tags["corpus_sha256"],
                provider=tags["provider"], checkpoint_sha256=tags["checkpoint_sha256"],
                supervision_regime=tags["supervision_regime"],
                evidence_class=tags["evidence_class"],
                evaluation_split=tags["evaluation_split"],
                evaluation_population_sha256=tags["evaluation_population_sha256"],
                protocol_sha256=tags["protocol_sha256"],
                scorer_sha256=tags["scorer_sha256"],
                seed=None if tags.get("seed") in (None, "None") else tags["seed"])}


# --------------------------------------------------------------------------- #
# aggregation: deduplicate first, refuse rather than average
# --------------------------------------------------------------------------- #

def deduplicate(rows, *, tolerance: float = 0.0) -> dict:
    """One row per occurrence, with what was dropped and what conflicted.

    `rows` are stored rows carrying `metric_occurrence_sha256` and `value`, each
    with whatever provenance the caller has (`unit_id`, `terminal_sha256`); those
    are reported, never digested.
    """
    kept, duplicates, conflicts = {}, [], []
    for row in rows:
        occurrence = row["metric_occurrence_sha256"]
        first = kept.get(occurrence)
        if first is None:
            kept[occurrence] = row
            continue
        if abs(float(first["value"]) - float(row["value"])) > tolerance:
            conflicts.append({
                "refusal": "CONFLICTING_VALUES_FOR_ONE_OCCURRENCE",
                "metric_occurrence_sha256": occurrence, "metric": row["metric"],
                "values": [float(first["value"]), float(row["value"])],
                "carried_by": [first.get("unit_id"), row.get("unit_id")]})
            continue
        duplicates.append({
            "refusal": "DOUBLE_COUNT_SAME_OCCURRENCE_ACROSS_TERMINALS",
            "metric_occurrence_sha256": occurrence, "metric": row["metric"],
            "value": float(row["value"]),
            "first_seen_on": first.get("unit_id"), "duplicate_on": row.get("unit_id"),
            "counted": "ONCE"})
    return {"kept": list(kept.values()), "duplicates_dropped": duplicates,
            "conflicts": conflicts, "rows_in": len(rows), "rows_counted": len(kept)}


def aggregation_groups(rows, *, tolerance: float = 0.0, strict: bool = True) -> dict:
    """Groups an aggregate may reduce: one per (row identity, evidence class).

    Deduplicated on occurrence before any mean. A value conflict inside one
    occurrence is a refusal, not an average; with `strict` it is raised.
    """
    deduped = deduplicate(rows, tolerance=tolerance)
    if strict and deduped["conflicts"]:
        _refuse("CONFLICTING_VALUES_FOR_ONE_OCCURRENCE",
                "; ".join(f"{c['metric']} carries {c['values']} under one occurrence "
                          f"{c['metric_occurrence_sha256'][:8]}"
                          for c in deduped["conflicts"]))
    groups = {}
    for row in deduped["kept"]:
        key = (row["row_identity_sha256"], row["evidence_class"])
        groups.setdefault(key, []).append(row)
    mixed = {}
    for identity, evidence in groups:
        mixed.setdefault(identity, set()).add(evidence)
    warnings = [{"refusal": "EVIDENCE_CLASSES_MIXED_IN_ONE_AGGREGATE",
                 "row_identity_sha256": identity,
                 "evidence_classes": sorted(classes),
                 "kept": "SPLIT_INTO_SEPARATE_GROUPS_NOT_WEIGHTED"}
                for identity, classes in mixed.items() if len(classes) > 1]
    return {"groups": [{"row_identity_sha256": identity, "evidence_class": evidence,
                        "metric": members[0]["metric"], "n_occurrences": len(members),
                        "values": [float(m["value"]) for m in members],
                        "carried_by": sorted(str(m.get("unit_id")) for m in members)}
                       for (identity, evidence), members in sorted(groups.items())],
            "duplicates_dropped": deduped["duplicates_dropped"],
            "conflicts": deduped["conflicts"],
            "evidence_class_separations": warnings,
            "rows_in": deduped["rows_in"], "rows_counted": deduped["rows_counted"]}
