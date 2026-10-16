"""Typed adapter for the I6-DMO design (``EXPERIMENT.short.json``); pure, no fit.

The design is not a legacy predictor config. This module parses it strictly
(duplicate keys, NaN/Infinity, unknown and missing fields are refused at every
level), enforces the fixed decisions of the owner order (section C) as exact
values, and resolves a sealed effective config whose identity covers the whole
design. ``validate-design`` reads only the design file and the bundled
availability contract: no data, network, TEST rows, credentials or training.

Two families share this adapter. The short design is the committed
``EXPERIMENT.short.json``. The long design is never hand-written: the
``seal-long-design`` command derives it deterministically from the validated
short design plus a valid short closure terminal (whose unit id, identity and
file digest are recorded, never its metrics or eligibility). It mirrors every
short decision except the fields PLAN.md section 5 fixes for the long family
(``LONG_FROM_PLAN``) and is tamper-evident through ``seal_sha256``. A field
PLAN.md leaves open is listed in ``LONG_PLAN_OPEN_FIELDS`` and yields the typed
``LONG_DESIGN_UNDECIDED`` naming that field; nothing is improvised.

Typed refusals are :class:`DesignError` with a stable leading code.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from tools.i6n_contracts import feature_availability as fa

DESIGN_SCHEMA = "i6dmo.design.v1"
EFFECTIVE_SCHEMA = "i6dmo.effective_config.v1"
HORIZONS = [2, 3, 4, 5, 6, 12]
LONG_HORIZONS = [18, 24, 36, 48, 60, 72]
SHORT_ID = "I6-DMO-SHORT-DS2-STATIC-R0"
LONG_ID = "I6-DMO-LONG-DS12-STATIC-R0"
SHORT_HOURLY_ID = "I6-DMO-SHORT-HOURLY-DS1-STATIC-R0"
#: The already-reviewed numerical policy, shared verbatim by every family's effective config.
#: UNCHANGED by the market-time work: no broadened tolerance, DMO10 not reopened.
MONITOR_FLOOR_DECLARATION = {
    "version": "v2", "rtol": 5e-5, "atol": 1e-9,
    "scope": "cross-process GPU comparison of the monitored value and per-batch "
             "per-head float64 partial sums ONLY; CPU end-to-end, self-check, reload, "
             "predictions, input bytes, weight digests, forward identity and flags "
             "stay bitwise",
    "justification": "a reproduced, build/context-specific numerical discrepancy "
                     "between in-session and fresh-instance GPU executions of this "
                     "pinned TensorFlow build (cause not demonstrated; eight "
                     "mechanisms excluded: autotune/init order, TF32, weights, input "
                     "bytes, forward identity, grappler remapping/layout/AMP; "
                     "observed max 1.82e-5), bounded by the versioned "
                     "NUMERICALLY_EQUIVALENT protocol "
                     "i6dmo.monitor_numeric_equivalence.v2",
    "prospective": True,
}
MG2_ID = "I6-DMO-SHORT-MG2-MARKET-EXACT-R0"
MG12_ID = "I6-DMO-LONG-MG12-MARKET-EXACT-R0"
MARKET_TIME_SCHEMA = "i6dmo.market_time_design.v1"
#: MARKET_TIME_PAIR_PLAN.md section 2, frozen. ``history_span_hours`` belongs to the OLD elapsed
#: schema and keeps its meaning there; a market-time design has no such field at all.
MARKET_TIME_SCOPE = {
    "mg2": {"experiment_id": MG2_ID, "rule": "MG2_MARKET_EXACT", "observations": 13,
            "spacing_open_market_hours": 2, "span_open_market_hours": 24,
            "horizons_hours": [2, 3, 4, 5, 6, 12], "family": "short_mg2"},
    "mg12": {"experiment_id": MG12_ID, "rule": "MG12_MARKET_EXACT", "observations": 13,
             "spacing_open_market_hours": 12, "span_open_market_hours": 144,
             "horizons_hours": [18, 24, 36, 48, 60, 72], "family": "long_mg12"},
}
#: Fields of the elapsed schema that contradict market-time sampling if present.
FIXED_ELAPSED_FIELDS = ("input", "history_span_hours", "spacing_hours", "observations",
                        "offsets_hours")
#: Section 3, frozen: the ONLY model extension, sealed by formula/dtype/shape/version.
TIME_ENCODER_IDENTITY = {
    "layer": "i6dmo.DynamicPhysicalTimeEncoder", "version": "1.0.0",
    "position": "immediately_after_branch_fusion",
    "inputs": ["fused_values(batch, slots, 4*n_features)",
               "offsets_hours(batch, slots) signed UTC hours from t, last exactly 0",
               "closure_crossed(batch, slots) 0/1, first slot 0"],
    "formula": ("code[s, c] = sin(offset_hours[s] * 10000^(-2*(c//2)/C)) for even c, cos(...) for "
                "odd c, added to the C fused value channels; evaluated on ACTUAL signed offsets, "
                "never ordinal slot indices"),
    "context_channels": ["adjacent_elapsed_hours_over_24", "closure_crossed"],
    "context_rule": ("appended AFTER the encoded values and BEFORE the core; slot 0 has adjacent "
                     "delta 0 and closure flag 0"),
    "guards": ("typed ORDINAL_OR_INVALID_OFFSETS when the last offset is not exactly 0, any "
               "offset is > 0, the first closure flag is not 0, or an adjacent elapsed delta is "
               "below the design's spacing_open_market_hours (ordinal slot indices, future "
               "offsets and compressed ladders are refused, not merely numerically different)"),
    "min_adjacent_hours": "design.market_time.spacing_open_market_hours",
    "unit_conversion_hours": 24, "fitted": False, "dtype": "float32",
    "output_shape": "(batch, slots, C + 2)  # 19 branches -> 76 + 2 = 78",
    "per_example_inputs_required": True,
    "slot_order_is_not_wall_clock": True,
}
STLFSI4 = "fred.stress.stlfsi4.level"
UNDECIDED = "UNDECIDED"
#: PLAN.md section 5, verbatim decisions for the long family; everything else mirrors the short design.
LONG_FROM_PLAN = {
    ("experiment_id",): LONG_ID,
    ("horizons_hours",): LONG_HORIZONS,          # "[18, 24, 36, 48, 60, 72] physical hours"
    ("input", "history_span_hours"): 144,        # "initial 144-hour inclusive span"
    ("input", "spacing_hours"): 12,              # "input spacing 12 h"
}
#: Asserted by PLAN.md section 5 and kept equal to the short values: "13 observations",
#: "same small modular family", "Every hourly origin still gets exact endpoint heads" (issuance 1 h).
#: Fields PLAN.md leaves open for the long family. Reading of PLAN.md (2a440265): none. Its
#: "Freeze its support/cost design before fitting" is satisfied by sealing the mirrored business
#: (per-head masks) and execution (admission/cost) blocks; a reviewer who rules otherwise adds the
#: path here and the seal then stops with LONG_DESIGN_UNDECIDED.
LONG_PLAN_OPEN_FIELDS_DEFAULT = ()
LONG_PLAN_OPEN_FIELDS = LONG_PLAN_OPEN_FIELDS_DEFAULT
FINAL_SUFFIX = "-FINAL"
#: PLAN.md section 5 final-refit definition: "fit once on [2021-01-01, 2025-01-01) using the
#: sealed inner/checkpoint rule, then freeze weights throughout TEST" (one scheduled pre-TEST
#: refit). The sealed inner rule is the same eight-week chronological tail, so every final
#: sample and valid label is strictly before TEST. No field is left open today.
FINAL_TRAIN = {"start": "2021-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"}
FINAL_PLAN_OPEN_FIELDS_DEFAULT = ()
FINAL_PLAN_OPEN_FIELDS = FINAL_PLAN_OPEN_FIELDS_DEFAULT
FINAL_RULE = ("final design = validated short/long design with the PLAN section 5 final-refit "
              "window [2021-01-01, 2025-01-01) and its eight-week inner tail; no outer validation "
              "split; weights frozen throughout TEST; derived without reading any outcome")
#: Section 3 of MUSASHI_TO_SATOSHI_STATIC_BASELINE_CLOSE_AND_MARKET_GRID_20261015: the paired
#: short HOURLY-input control — same 24 elapsed-hour span, 25 observations at hourly spacing
#: including t, six heads, same model recipe and static cadence, sealed as its own identity.
HOURLY_FROM_PLAN = {
    ("experiment_id",): SHORT_HOURLY_ID,
    ("input", "spacing_hours"): 1,
    ("input", "observations"): 25,
}
HOURLY_RULE = ("hourly control = validated short design with hourly spacing (25 observations over "
               "the same 24 elapsed hours, t included); six heads, model recipe and static cadence "
               "mirrored; only what 25-vs-13 observations forces changes in the model (each head "
               "flattens a 25x4 instead of a 13x4 core sequence); sealed as its own identity, "
               "prepared without fitting")
HOURLY_AVAILABILITY_DECLARATION = (
    "DECLARED TRAIN/validation availability differences vs the 13-observation DS2 design: hourly "
    "exact slots across weekend-adjacent windows differ (a window needing 25 consecutive hourly "
    "observations loses more weekend-adjacent origins than the two-hour grid, and gains none); "
    "the exact per-split counts are owned by the data lane's market-grid support study and are "
    "not duplicated here; comparison is only valid under the frozen common-support rules")
#: Frozen BEFORE any scoring; the sealed hourly design binds this document's sha256.
COMMON_SUPPORT_RULES = {
    "schema": "i6dmo.common_support_rules.v1",
    "frozen_before_scoring": True,
    "population": ("intersection of the (origin row_id, head) pairs that are valid under BOTH "
                   "designs' bundles, per split, on identical row ids; nothing outside the "
                   "intersection enters either side of the comparison"),
    "metrics": ("per-head pooled original-price MAE and the SHARED naive computed on exactly the "
                "intersection rows (sums/counts, never means of weekly means); relative error "
                "(MAE_h-naive_h)/naive_h, negative better"),
    "reused_model_rule": ("a model trained on its own full TRAIN population and scored on the "
                          "common support is labelled REUSED_MODEL_SUBSET_SCORE and is NOT a "
                          "model trained on the common TRAIN population; that contrast would be "
                          "a separate sealed design and identity"),
    "distinct_contrasts": ("encoder/R2 and retraining-cadence contrasts are distinct later sealed "
                           "designs; changes are never combined and attributed to downsampling"),
    "no_test": "TEST rows never enter the comparison; development 2024 disclosure applies to both",
}
#: NEXT_CONTRASTS_PLAN_20261015 section 2: the matched short MG1-vs-MG2 control. Two NEW,
#: independently sealed market-time identities. The closed EXPERIMENT.mg2.json (full-population
#: MG2) and the ELAPSED EXPERIMENT.short_hourly.json are NOT these designs and are never accepted
#: where a matched identity is required (typed MATCHED_IDENTITY_REQUIRED).
MATCHED_SCHEMA = "i6dmo.matched_market_time_design.v1"
MATCHED_CONTRAST_ID = "I6-DMO-SHORT-MG1-VS-MG2-MATCHED-R0"
MG1_MATCHED_ID = "I6-DMO-SHORT-MG1-MATCHED-R0"
MG2_MATCHED_ID = "I6-DMO-SHORT-MG2-MATCHED-R0"
MATCHED_SCOPE = {
    "mg1": {"experiment_id": MG1_MATCHED_ID, "rule": "MG1_MARKET_EXACT", "observations": 25,
            "spacing_open_market_hours": 1, "span_open_market_hours": 24,
            "horizons_hours": [2, 3, 4, 5, 6, 12], "family": "short_mg1_matched",
            "role": "CONTROL", "partner": "mg2m"},
    "mg2m": {"experiment_id": MG2_MATCHED_ID, "rule": "MG2_MARKET_EXACT", "observations": 13,
             "spacing_open_market_hours": 2, "span_open_market_hours": 24,
             "horizons_hours": [2, 3, 4, 5, 6, 12], "family": "short_mg2_matched",
             "role": "MATCHED_REFERENCE", "partner": "mg1"},
}
LONG_MATCHED_CONTRAST_ID = "I6-DMO-LONG-MG1-VS-MG12-MATCHED-R0"
LONG_MATCHED_SCOPE = {
    "mg1l": {"experiment_id": "I6-DMO-LONG-MG1-MATCHED-R0", "rule": "MG1_MARKET_EXACT",
             "observations": 145, "spacing_open_market_hours": 1, "span_open_market_hours": 144,
             "horizons_hours": [18, 24, 36, 48, 60, 72], "family": "long_mg1_matched",
             "role": "CONTROL", "partner": "mg12m"},
    "mg12m": {"experiment_id": "I6-DMO-LONG-MG12-MATCHED-R0", "rule": "MG12_MARKET_EXACT",
              "observations": 13, "spacing_open_market_hours": 12, "span_open_market_hours": 144,
              "horizons_hours": [18, 24, 36, 48, 60, 72], "family": "long_mg12_matched",
              "role": "MATCHED_REFERENCE", "partner": "mg1l"},
}
CONTEXT_BRIDGE_CONTRAST_ID = "I6-DMO-LONG-SHORT-INPUT-CONTEXT-BRIDGE-R0"
CONTEXT_BRIDGE_SCOPE = {
    "mg12c": {"experiment_id": "I6-DMO-LONG-MG12-CONTEXT-BRIDGE-CONTROL-R0",
              "rule": "MG12_MARKET_EXACT", "observations": 13,
              "spacing_open_market_hours": 12, "span_open_market_hours": 144,
              "horizons_hours": [18, 24, 36, 48, 60, 72],
              "family": "long_mg12_bridge_control", "role": "CONTROL", "partner": "mg2l"},
    "mg2l": {"experiment_id": "I6-DMO-LONG-MG2-CONTEXT-BRIDGE-R0",
             "rule": "MG2_MARKET_EXACT", "observations": 13,
             "spacing_open_market_hours": 2, "span_open_market_hours": 24,
             "horizons_hours": [18, 24, 36, 48, 60, 72],
             "family": "long_mg2_bridge", "role": "TREATMENT", "partner": "mg12c"},
}
ALL_MATCHED_SCOPE = {**MATCHED_SCOPE, **LONG_MATCHED_SCOPE, **CONTEXT_BRIDGE_SCOPE}
MATCHED_FAMILIES = tuple(v["family"] for v in ALL_MATCHED_SCOPE.values())
#: Frozen BEFORE either fit (plan section 2 "Common population"); both sealed designs bind its sha256.
MATCHED_POPULATION_RULES = {
    "schema": "i6dmo.matched_population_rules.v1",
    "contrast_id": MATCHED_CONTRAST_ID,
    "population_schema": "i6dmo.common_population.v1",
    "splits": {"FIT": "TRAIN [2020-01-01, 2024-01-01) minus the chronological last eight weeks",
               "INNER": "the chronological last eight TRAIN weeks",
               "DEVELOPMENT": "full 2024 [2024-01-01, 2025-01-01) (the data lane's VALIDATION split)"},
    "population": ("per split separately, the intersection of the (origin row_id, horizon) pairs "
                   "valid under BOTH treatments (exact market slots, C1 MANAGEMENT label "
                   "support, split purge and label maturity); only origins with >= 1 common "
                   "valid head are retained; frozen with ordered row ids, per-head counts, "
                   "entry/management masks and typed exclusions BEFORE either fit"),
    "consumers": ("both trainers, checkpoint monitors, annual scorers and weekly diagnostics "
                  "consume exactly these masks"),
    "targets": "exact ELAPSED p(t+h); no smoothing, interpolation or next-observed-row targets",
    "missing_bars": "a missing open-session bar is a counted exclusion, never filled",
    "unsupported_heads": "typed, never filled",
    "scaler": ("ONE FIT-only feature mean/std (ddof=0) over the UNION of MG1/MG2 input-source "
               "bars of the common FIT origins, with its exact source-row digest; the same "
               "statistics serve both arms; INNER/DEVELOPMENT cannot change them"),
    "loss_scale": "FIT-only naive MAE per horizon on the common FIT mask, shared by both arms",
    "full_population_fit_substitutes": False,
    "test": "no TEST row, unit, metadata scan or derived artifact",
}
#: Execution-role runtime the matched designs pin (both GPU roles run it; the reviewed bounded GPU
#: replay ran under it). Not the coordinator's own install. Technical-lead ruling 2026-10-15.
MATCHED_RUNTIME = {"tensorflow": "2.21.0", "keras": "3.13.2"}
#: Seals superseded BEFORE any execution (runtime pin 2.18.0/3.9.0 taken from the coordinator);
#: no fit, report or campaign used them. Refused as SUPERSEDED, never re-accepted.
MATCHED_SUPERSEDED_SEALS = {
    "4786951ae1024f1761848d9e8fc36d519e0de5aed136cce5d3bbda77f4e75e5a": "mg1 tf2.18.0/keras3.9.0",
    "6e8689d4798b1c75333f054ab58bb0204c03b19ae1efd71378a0d6785f86da68": "mg2m tf2.18.0/keras3.9.0",
}
MATCHED_POPULATION_RULES["rules_sha256"] = hashlib.sha256(json.dumps(
    MATCHED_POPULATION_RULES, sort_keys=True, separators=(",", ":"),
    allow_nan=False).encode("utf-8")).hexdigest()
LONG_MATCHED_POPULATION_RULES = copy.deepcopy(MATCHED_POPULATION_RULES)
LONG_MATCHED_POPULATION_RULES.pop("rules_sha256")
LONG_MATCHED_POPULATION_RULES["contrast_id"] = LONG_MATCHED_CONTRAST_ID
LONG_MATCHED_POPULATION_RULES["scaler"] = (
    "ONE FIT-only feature mean/std (ddof=0) over the UNION of MG1-long/MG12 input-source "
    "bars of the common FIT origins, with its exact source-row digest; the same "
    "statistics serve both arms; INNER/DEVELOPMENT cannot change them")
LONG_MATCHED_POPULATION_RULES["rules_sha256"] = hashlib.sha256(json.dumps(
    LONG_MATCHED_POPULATION_RULES, sort_keys=True, separators=(",", ":"),
    allow_nan=False).encode("utf-8")).hexdigest()
CONTEXT_BRIDGE_POPULATION_RULES = copy.deepcopy(MATCHED_POPULATION_RULES)
CONTEXT_BRIDGE_POPULATION_RULES.pop("rules_sha256")
CONTEXT_BRIDGE_POPULATION_RULES.update({
    "contrast_id": CONTEXT_BRIDGE_CONTRAST_ID,
    "scaler": ("ONE FIT-only feature mean/std (ddof=0) over the UNION of MG12c/MG2l input-source "
               "bars of the common FIT origins, with its exact source-row digest; the same "
               "statistics serve both arms; INNER/DEVELOPMENT cannot change them"),
    "input_recipe": ("13 observations including t in both arms; MG12c at spacing12/span144 "
                     "versus MG2l at spacing2/span24 open-market hours, with identical long "
                     "elapsed heads; a whole input-recipe comparison, not separate causal "
                     "effects of span and spacing"),
    "slot_subset_required": False,
    "capacity_delta": 0,
})
CONTEXT_BRIDGE_POPULATION_RULES["rules_sha256"] = hashlib.sha256(json.dumps(
    CONTEXT_BRIDGE_POPULATION_RULES, sort_keys=True, separators=(",", ":"),
    allow_nan=False).encode("utf-8")).hexdigest()


def matched_arm_set(arms):
    """Infer exactly one supported contrast; mixed, partial and combined pairs are refused."""
    keys = set(arms)
    for scope in (MATCHED_SCOPE, LONG_MATCHED_SCOPE, CONTEXT_BRIDGE_SCOPE):
        if keys == set(scope):
            return tuple(scope)
    raise DesignError("ARMS_REQUIRED", "exactly mg1/mg2m, mg1l/mg12m or mg12c/mg2l")


def matched_population_rules(arm):
    """Frozen rules of an explicit matched arm, never a short-family global override."""
    if arm in MATCHED_SCOPE:
        return copy.deepcopy(MATCHED_POPULATION_RULES)
    if arm in LONG_MATCHED_SCOPE:
        return copy.deepcopy(LONG_MATCHED_POPULATION_RULES)
    if arm in CONTEXT_BRIDGE_SCOPE:
        return copy.deepcopy(CONTEXT_BRIDGE_POPULATION_RULES)
    raise DesignError("MATCHED_ARM_UNKNOWN", repr(arm))


SHORT_CLOSURE_UNIT = "short.report"
SHORT_CLOSURE_STATUSES = ("COMPLETE", "NOT_ELIGIBLE")
TERMINAL_SCHEMA = "i6dmo.terminal.v1"
LONG_RULE = ("long design = validated short design with PLAN.md section 5 long fields; sealed after a "
             "valid short closure terminal; no validation/TEST outcome is read")


class DesignError(ValueError):
    """Typed refusal; ``code`` is the stable leading token of the message."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


COMMON_SUPPORT_RULES["rules_sha256"] = _sha(COMMON_SUPPORT_RULES)


# ------------------------------------------------------------------ schema
# Leaves: ("fixed", value) exact (type-strict) value; ("type", python types);
# ("enum", values). Objects are dicts of key -> spec; every key is required.

def _fixed(value):
    return ("fixed", value)


def _typ(*types):
    return ("type", types)


_TS = _typ(str)
SCHEMA = {
    "schema": _fixed(DESIGN_SCHEMA),
    "status": _fixed("DESIGN_NOT_RUNTIME_CONFIG"),
    "experiment_id": _fixed(SHORT_ID),
    "asset": _fixed("EURUSD"),
    "seed": _fixed(6102026),
    "repetitions": _fixed(1),
    "evaluation_mode": _fixed("BUSINESS_STATIC_BASELINE"),
    "retraining_frequency": _fixed("NONE"),
    "input": {
        "value_mode": _fixed("RAW"),
        "history_span_hours": _fixed(24),
        "spacing_hours": _fixed(2),
        "observations": _fixed(13),
        "include_origin": _fixed(True),
        "issuance_hours": _fixed(1),
        "smoothing": _fixed(False),
        "require_exact_timestamps": _fixed(True),
    },
    "horizons_hours": _fixed(HORIZONS),
    "feature_policy": {
        "source": _fixed("active_19_feature_point_in_time_manifest"),
        "exclude": ("exclude", None),
        "reject_stlfsi4_aliases": _fixed(True),
        "development_selection_used_2024": _fixed(True),
        "unresolved_provenance_blocks_confirmatory_test": _fixed(True),
    },
    "splits": {
        "train": {"start": _fixed("2020-01-01T00:00:00Z"), "end": _fixed("2024-01-01T00:00:00Z")},
        "inner_tail_weeks": _fixed(8),
        "validation": {"start": _fixed("2024-01-01T00:00:00Z"), "end": _fixed("2025-01-01T00:00:00Z")},
        "test": {"start": _fixed("2025-01-01T00:00:00Z"), "end": _fixed("2026-01-01T00:00:00Z"),
                 "open": ("test_open", None)},
        "end_exclusive": _fixed(True),
        "purge_by_label_availability": _fixed(True),
    },
    "business": {
        "calendar": _TS,
        "entry_delay_after_open_hours": _fixed(12),
        "force_flat_before_close_hours": _fixed(12),
        "entry_cutoff": _TS,
        "per_output_masks": _fixed(True),
        "keep_row_if_any_head_valid": _fixed(True),
        "preserve_observed_past_input_history": _fixed(True),
    },
    "model": {
        "regime": _fixed("R0"),
        "branch": {"type": _fixed("causal_conv1d"), "filters": _fixed(4), "kernel_size": _fixed(3),
                   "stride": _fixed(1)},
        "fusion": _fixed("concatenate_filters"),
        "positional_encoding": _fixed("immediately_after_fusion"),
        "core": {"type": _fixed("causal_conv1d"), "filters": _fixed([8, 4]), "kernel_sizes": _fixed([3, 3]),
                 "strides": _fixed([1, 1]), "transformer_blocks": _fixed(0)},
        "heads": {"independent_weights": _fixed(True), "readout": _fixed("flatten_core_sequence"),
                  "hidden_units": _fixed([8]), "hidden_activation": _fixed("relu"),
                  "output_units": _fixed(1), "output_activation": _fixed("linear")},
    },
    "training": {
        "optimizer": _fixed("AdamW"),
        "learning_rate": _fixed(0.001),
        "weight_decay": _fixed(0.0001),
        "batch_size": _fixed(256),
        "max_epochs": _fixed(200),
        "patience": _fixed(10),
        "min_delta": _fixed(1e-8),
        "loss": _fixed("equal_head_masked_mae_over_fit_only_naive"),
        "monitor": _fixed("mean_end_of_epoch_fit_and_inner_loss"),
        "restore_best_weights": _fixed(True),
        "outer_validation_for_checkpoint_selection": _fixed(False),
    },
    "metrics": {
        "scale": _fixed("original_price_USD_per_EUR"),
        "relative_error": _fixed("(MAE_h-naive_h)/naive_h"),
        "negative_is_better": _fixed(True),
        "per_horizon": _fixed(True),
        "annual_aggregation": _fixed("pooled_absolute_error_sums_and_counts"),
        "weekly_diagnostics": _fixed(True),
        "olap_write_and_readback": _fixed(True),
    },
    "execution": {
        "first_scientific_gpu": _TS,
        "one_heavy_job_per_host": _fixed(True),
        "secondary_gpu": _TS,
        "preferred_aux_gpu": _TS,
        "adopt_first_full_fit_as_cost_measurement": _fixed(True),
        "cpu_fallback": _fixed(False),
        "new_scientific_identity": _fixed(True),
        "resume_retired_doin_campaign": _fixed(False),
    },
}


_HEX = ("hex64", None)


def _long_schema():
    schema = copy.deepcopy(SCHEMA)
    for path, value in LONG_FROM_PLAN.items():
        node = schema
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = _fixed(value)
    schema["derived_from"] = {
        "rule": _fixed(LONG_RULE),
        "short_experiment_id": _fixed(SHORT_ID),
        "short_design_sha256": _HEX,
        "short_terminal": {"unit_id": _fixed(SHORT_CLOSURE_UNIT), "identity_sha256": _HEX,
                           "terminal_file_sha256": _HEX},
    }
    schema["seal_sha256"] = _HEX
    return schema


LONG_SCHEMA = _long_schema()


def _final_schema(fam):
    schema = copy.deepcopy(LONG_SCHEMA if fam == "long" else SCHEMA)
    base_id = LONG_ID if fam == "long" else SHORT_ID
    schema["experiment_id"] = _fixed(base_id + FINAL_SUFFIX)
    schema["splits"]["train"] = {"start": _fixed(FINAL_TRAIN["start"]), "end": _fixed(FINAL_TRAIN["end"])}
    del schema["splits"]["validation"]
    schema["derived_from"] = {
        "rule": _fixed(FINAL_RULE),
        "base_experiment_id": _fixed(base_id),
        "base_design_sha256": _HEX,
    }
    if fam == "long":  # the long base already carries derived_from/seal; the final replaces them
        schema["derived_from"]["base_short_terminal"] = LONG_SCHEMA["derived_from"]["short_terminal"]
    schema["seal_sha256"] = _HEX
    return schema


FINAL_SCHEMAS = {fam: _final_schema(fam) for fam in ("short", "long")}


def _hourly_schema():
    schema = copy.deepcopy(SCHEMA)
    for path, value in HOURLY_FROM_PLAN.items():
        node = schema
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = _fixed(value)
    schema["derived_from"] = {
        "rule": _fixed(HOURLY_RULE),
        "base_experiment_id": _fixed(SHORT_ID),
        "base_design_sha256": _HEX,
        "availability_declaration": _fixed(HOURLY_AVAILABILITY_DECLARATION),
        "common_support_rules_sha256": _fixed(COMMON_SUPPORT_RULES["rules_sha256"]),
    }
    schema["seal_sha256"] = _HEX
    return schema


HOURLY_SCHEMA = _hourly_schema()


def _market_time_schema(key):
    scope = MARKET_TIME_SCOPE[key]
    base = copy.deepcopy(SCHEMA)
    schema = {
        "schema": _fixed(MARKET_TIME_SCHEMA),
        "status": base["status"],
        "experiment_id": _fixed(scope["experiment_id"]),
        "asset": base["asset"],
        "seed": base["seed"],
        "repetitions": base["repetitions"],
        "evaluation_mode": base["evaluation_mode"],
        "retraining_frequency": base["retraining_frequency"],
        "market_time": {
            "rule": _fixed(scope["rule"]),
            "observations": _fixed(scope["observations"]),
            "spacing_open_market_hours": _fixed(scope["spacing_open_market_hours"]),
            "span_open_market_hours": _fixed(scope["span_open_market_hours"]),
            "include_origin": _fixed(True),
            "elapsed_span_fixed": _fixed(False),
            "require_exact_market_slots": _fixed(True),
        },
        "horizons_hours": _fixed(scope["horizons_hours"]),
        "feature_policy": base["feature_policy"],
        "splits": base["splits"],
        "business": base["business"],
        "model": base["model"],
        "training": base["training"],
        "metrics": base["metrics"],
        "execution": base["execution"],
        "sealed_identities": {
            "base_short_design_sha256": _HEX,
            "time_encoder": {k: _fixed(v) for k, v in TIME_ENCODER_IDENTITY.items()},
            "feature_manifest_sha256": _HEX,
            "availability_contract_sha256": _HEX,
            "calendar": {"calendar_id": _typ(str), "version": _typ(str),
                         "status": _fixed("ASSUMED_NOT_BROKER")},
            "candidate_sha256": _HEX,
            "support_study_sha256": _HEX,
            "seed": _fixed(6102026),
            "runtime": {"tensorflow": _typ(str), "keras": _typ(str)},
        },
        "seal_sha256": _HEX,
    }
    return schema


MARKET_TIME_SCHEMAS = {MARKET_TIME_SCOPE[k]["family"]: _market_time_schema(k)
                       for k in MARKET_TIME_SCOPE}


def _matched_schema(arm):
    """Matched design schema: explicit market-time scope and contrast-specific population rules."""
    scope = ALL_MATCHED_SCOPE[arm]
    partner = ALL_MATCHED_SCOPE[scope["partner"]]
    rules = matched_population_rules(arm)
    base = copy.deepcopy(SCHEMA)
    return {
        "schema": _fixed(MATCHED_SCHEMA),
        "status": base["status"],
        "experiment_id": _fixed(scope["experiment_id"]),
        "asset": base["asset"], "seed": base["seed"], "repetitions": base["repetitions"],
        "evaluation_mode": base["evaluation_mode"],
        "retraining_frequency": base["retraining_frequency"],
        "market_time": {
            "rule": _fixed(scope["rule"]),
            "observations": _fixed(scope["observations"]),
            "spacing_open_market_hours": _fixed(scope["spacing_open_market_hours"]),
            "span_open_market_hours": _fixed(scope["span_open_market_hours"]),
            "include_origin": _fixed(True),
            "elapsed_span_fixed": _fixed(False),
            "require_exact_market_slots": _fixed(True),
        },
        "horizons_hours": _fixed(scope["horizons_hours"]),
        "feature_policy": base["feature_policy"], "splits": base["splits"],
        "business": base["business"], "model": base["model"], "training": base["training"],
        "metrics": base["metrics"], "execution": base["execution"],
        "matched_population": {
            "arm": _fixed(arm), "role": _fixed(scope["role"]),
            "contrast_id": _fixed(rules["contrast_id"]),
            "partner_arm": _fixed(scope["partner"]),
            "partner_experiment_id": _fixed(partner["experiment_id"]),
            "rules_sha256": _fixed(rules["rules_sha256"]),
            "full_population_fit_substitutes": _fixed(False),
        },
        "sealed_identities": {
            "base_short_design_sha256": _HEX,
            "time_encoder": {k: _fixed(v) for k, v in TIME_ENCODER_IDENTITY.items()},
            "feature_manifest_sha256": _HEX,
            "availability_contract_sha256": _HEX,
            "calendar": {"calendar_id": _typ(str), "version": _typ(str),
                         "status": _fixed("ASSUMED_NOT_BROKER")},
            "data_design_sha256": _HEX,
            "data_identity_sha256": _HEX,
            "partner_data_identity_sha256": _HEX,
            "seed": _fixed(6102026),
            "runtime": {"tensorflow": _typ(str), "keras": _typ(str)},
        },
        "seal_sha256": _HEX,
    }


MATCHED_SCHEMAS = {ALL_MATCHED_SCOPE[k]["family"]: _matched_schema(k) for k in ALL_MATCHED_SCOPE}


def mode(doc):
    """``FINAL_REFIT`` for a derived final design, otherwise ``VALIDATION``."""
    eid = doc.get("experiment_id") if isinstance(doc, dict) else None
    return "FINAL_REFIT" if isinstance(eid, str) and eid.endswith(FINAL_SUFFIX) else "VALIDATION"


def family(doc):
    """``long``/``short_hourly`` for those experiment ids, otherwise ``short``."""
    eid = doc.get("experiment_id") if isinstance(doc, dict) else None
    if eid in (LONG_ID, LONG_ID + FINAL_SUFFIX):
        return "long"
    if eid == SHORT_HOURLY_ID:
        return "short_hourly"
    if eid == MG2_ID:
        return "short_mg2"
    if eid == MG12_ID:
        return "long_mg12"
    for scope in ALL_MATCHED_SCOPE.values():
        if eid == scope["experiment_id"]:
            return scope["family"]
    return "short"


def matched_arm(doc):
    """Arm key of a VALIDATED short or long matched design; anything else is refused.

    The closed full-population MG2 (``short_mg2``), the ELAPSED hourly control and every other
    family raise ``MATCHED_IDENTITY_REQUIRED``: none can stand in for a matched identity.
    """
    fam = family(doc)
    for arm, scope in ALL_MATCHED_SCOPE.items():
        if scope["family"] == fam:
            validate_design(doc)
            return arm
    eid = doc.get("experiment_id") if isinstance(doc, dict) else None
    raise DesignError("MATCHED_IDENTITY_REQUIRED",
                      f"family {fam!r} ({eid!r}) is not a declared matched identity; a "
                      "full-population or elapsed design never substitutes for one")


def _same(a, b):
    """Type-strict equality (True is not 1; 1 is not 1.0)."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def stlfsi4_alias(name):
    """True when a feature name denotes STLFSI4 under any spelling/punctuation."""
    return isinstance(name, str) and "stlfsi" in re.sub(r"[^a-z0-9]", "", name.lower())


def _check(node, spec, path):
    if isinstance(spec, dict):
        if not isinstance(node, dict):
            raise DesignError("TYPE", f"{path or '<root>'} must be an object")
        unknown = sorted(set(node) - set(spec))
        if unknown:
            raise DesignError("UNKNOWN_FIELD", f"{path or '<root>'}: {unknown}")
        missing = sorted(set(spec) - set(node))
        if missing:
            raise DesignError("MISSING_FIELD", f"{path or '<root>'}: {missing}")
        for key, sub in spec.items():
            _check(node[key], sub, f"{path}.{key}" if path else key)
        return
    kind, arg = spec
    if kind == "type":
        if not isinstance(node, arg) or isinstance(node, bool) and bool not in arg:
            raise DesignError("TYPE", f"{path} must be {[t.__name__ for t in arg]}")
    elif kind == "fixed":
        if type(node) is not type(arg) and not (isinstance(arg, list) and isinstance(node, list)):
            raise DesignError("TYPE", f"{path} must be {type(arg).__name__}")
        if not _same(node, arg):
            raise DesignError("FIXED_VALUE", f"{path} is fixed to {arg!r} by the owner order; got {node!r}")
    elif kind == "hex64":
        if not isinstance(node, str) or not re.fullmatch(r"[0-9a-f]{64}", node):
            raise DesignError("TYPE", f"{path} must be a lowercase sha256 hex digest")
    elif kind == "test_open":
        if node is not False:
            raise DesignError("TEST_FORBIDDEN", f"{path} must be false; TEST opens only through final gates")
    elif kind == "exclude":
        if not isinstance(node, list) or any(not isinstance(v, str) for v in node):
            raise DesignError("TYPE", f"{path} must be a list of feature names")
        if node != [STLFSI4]:
            raise DesignError("STLFSI4_EXCLUSION", f"{path} must be exactly [{STLFSI4!r}]; got {node!r}")


def _undecided(node, path=""):
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _undecided(v, f"{path}.{k}" if path else k)]
    return [path] if node == UNDECIDED else []


def _seal_body(doc):
    return {k: v for k, v in doc.items() if k != "seal_sha256"}


def validate_design(doc):
    """Validate a parsed short/long validation or derived final design. Raises DesignError."""
    fam, final = family(doc), mode(doc) == "FINAL_REFIT" if isinstance(doc, dict) else False
    if final:
        open_paths = _undecided(doc)
        if open_paths:
            raise DesignError("FINAL_DESIGN_UNDECIDED", f"PLAN leaves {open_paths} open; final units must wait")
        _check(doc, FINAL_SCHEMAS[fam], "")
        if doc["seal_sha256"] != _sha(_seal_body(doc)):
            raise DesignError("FINAL_DESIGN_SEAL", "the sealed final design was modified")
    elif fam in ("short_mg2", "long_mg12"):
        contradictions = sorted(f for f in FIXED_ELAPSED_FIELDS if f in doc)
        if contradictions:
            raise DesignError("MARKET_TIME_CONTRADICTION",
                              f"{contradictions} are fixed-elapsed fields of {DESIGN_SCHEMA}; a "
                              "market-time design declares open-market spacing only (the old "
                              "history_span_hours keeps its elapsed meaning in its own schema)")
        open_paths = _undecided(doc)
        if open_paths:
            raise DesignError("MARKET_TIME_UNDECIDED", f"open fields {open_paths}; never improvised")
        _check(doc, MARKET_TIME_SCHEMAS[fam], "")
        if doc["seal_sha256"] != _sha(_seal_body(doc)):
            raise DesignError("MARKET_TIME_SEAL", "the sealed market-time design was modified")
    elif fam in MATCHED_FAMILIES:
        contradictions = sorted(f for f in FIXED_ELAPSED_FIELDS if f in doc)
        if contradictions:
            raise DesignError("MARKET_TIME_CONTRADICTION",
                              f"{contradictions} are fixed-elapsed fields of {DESIGN_SCHEMA}; a "
                              "matched market-time design declares open-market spacing only")
        if doc.get("schema") != MATCHED_SCHEMA:
            raise DesignError("MATCHED_SCHEMA_REQUIRED",
                              f"schema must be {MATCHED_SCHEMA}; a relabelled market-time or "
                              "elapsed design is not a matched identity")
        open_paths = _undecided(doc)
        if open_paths:
            raise DesignError("MATCHED_DESIGN_UNDECIDED", f"open fields {open_paths}; never improvised")
        _check(doc, MATCHED_SCHEMAS[fam], "")
        if doc["seal_sha256"] != _sha(_seal_body(doc)):
            raise DesignError("MATCHED_DESIGN_SEAL", "the sealed matched design was modified")
        if doc["seal_sha256"] in MATCHED_SUPERSEDED_SEALS:
            raise DesignError("MATCHED_IDENTITY_SUPERSEDED",
                              f"seal {doc['seal_sha256'][:12]} "
                              f"({MATCHED_SUPERSEDED_SEALS[doc['seal_sha256']]}) was superseded "
                              "before execution; use the current matched_short design")
    elif fam == "short_hourly":
        open_paths = _undecided(doc)
        if open_paths:
            raise DesignError("HOURLY_DESIGN_UNDECIDED", f"open fields {open_paths}; prepare, never improvise")
        _check(doc, HOURLY_SCHEMA, "")
        if doc["seal_sha256"] != _sha(_seal_body(doc)):
            raise DesignError("HOURLY_DESIGN_SEAL", "the sealed hourly control design was modified")
    elif fam == "long":
        open_paths = _undecided(doc)
        if open_paths:
            raise DesignError("LONG_DESIGN_UNDECIDED", f"PLAN.md leaves {open_paths} open; long.fit must wait")
        _check(doc, LONG_SCHEMA, "")
        if doc["seal_sha256"] != _sha(_seal_body(doc)):
            raise DesignError("LONG_DESIGN_SEAL", "the sealed long design was modified")
    else:
        _check(doc, SCHEMA, "")
    return json.loads(_canonical(doc))


def final_refit_design(base_design):
    """Derive the typed FINAL_REFIT design from a validated validation design (pure, deterministic).

    TRAIN becomes the PLAN final window [2021-01-01, 2025-01-01) with the same eight-week inner
    tail; the outer validation split is removed; everything else (features, model, training,
    business, metrics, execution, horizons, input grid, seed) is mirrored unchanged. A field in
    ``FINAL_PLAN_OPEN_FIELDS`` is a typed ``FINAL_DESIGN_UNDECIDED``; nothing is improvised.
    """
    base = validate_design(base_design)
    if mode(base) == "FINAL_REFIT" or family(base) not in ("short", "long"):
        raise DesignError("FINAL_FROM_VALIDATION_ONLY",
                          "a final design derives from the short/long validation design; the hourly "
                          "control's final policy is a separate later decision")
    open_paths = [".".join(path) for path in FINAL_PLAN_OPEN_FIELDS]
    if open_paths:
        raise DesignError("FINAL_DESIGN_UNDECIDED", f"PLAN leaves {open_paths} open; final units must wait")
    doc = copy.deepcopy(base)
    doc["experiment_id"] = base["experiment_id"] + FINAL_SUFFIX
    doc["splits"]["train"] = dict(FINAL_TRAIN)
    del doc["splits"]["validation"]
    derived = {"rule": FINAL_RULE, "base_experiment_id": base["experiment_id"],
               "base_design_sha256": design_sha256(base)}
    if family(base) == "long":
        derived["base_short_terminal"] = copy.deepcopy(base["derived_from"]["short_terminal"])
    doc["derived_from"] = derived
    doc.pop("seal_sha256", None)
    doc["seal_sha256"] = _sha(_seal_body(doc))
    return validate_design(doc)


def _reject_duplicates(pairs):
    keys = [k for k, _ in pairs]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise DesignError("DUPLICATE_KEY", str(dup))
    return dict(pairs)


def _reject_constant(token):
    raise DesignError("NONFINITE", f"{token} is not allowed in a design")


def load_design(path):
    """Parse strictly and validate one design file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DesignError("UNREADABLE", f"{path}: {exc.strerror}") from None
    try:
        doc = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise DesignError("INVALID_JSON", str(exc)) from None
    return validate_design(doc)


def design_sha256(doc):
    return _sha(doc)


def check_feature_identity(features, *, availability=True):
    """The 19-feature identity: no STLFSI4 alias, exact gated manifest, availability gate."""
    if not isinstance(features, (list, tuple)) or any(not isinstance(f, str) for f in features):
        raise DesignError("FEATURE_IDENTITY_MISMATCH", "features must be a list of names")
    aliases = [f for f in features if stlfsi4_alias(f)]
    if aliases:
        raise DesignError("STLFSI4_ALIAS", f"STLFSI4 cannot return under any name: {aliases}")
    expected = list(fa.MANIFEST_NO_STLFSI4)
    if list(features) != expected:
        raise DesignError("FEATURE_IDENTITY_MISMATCH",
                          f"expected the gated 19-feature manifest {fa.manifest_sha256(expected)} in order")
    report = {"features": list(features), "count": len(features),
              "manifest_sha256": fa.manifest_sha256(features), "excluded": [STLFSI4]}
    if availability:
        try:
            gate = fa.gate(list(features))
        except fa.AvailabilityGateError as exc:
            raise DesignError(exc.code, str(exc)) from None
        report.update(availability_contract_sha256=gate["contract_sha256"],
                      unverified_features=gate["unverified_features"])
    return report


def _iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def effective_config_sha256(effective):
    return _sha({k: v for k, v in effective.items() if k != "effective_config_sha256"})


def _market_time_effective(d, features):
    """Effective config for a sealed market-time design (variable elapsed offsets)."""
    mt, m, t = d["market_time"], d["model"], d["training"]
    fam = family(d)
    s = d["splits"]
    train_start, train_end = _parse(s["train"]["start"]), _parse(s["train"]["end"])
    inner_start = train_end - timedelta(weeks=s["inner_tail_weeks"])
    effective = {
        "schema": EFFECTIVE_SCHEMA, "family": fam, "mode": "VALIDATION", "design": d,
        "design_sha256": design_sha256(d), "experiment_id": d["experiment_id"], "seed": d["seed"],
        "horizons_hours": list(d["horizons_hours"]),
        "market_time": dict(mt),
        "offsets_hours": None,   # variable per example; NEVER a fixed grid
        "feature_manifest": features,
        "splits": {"fit": {"start": _iso(train_start), "end": _iso(inner_start)},
                   "inner": {"start": _iso(inner_start), "end": _iso(train_end)},
                   "validation": dict(s["validation"]), "test": dict(s["test"]),
                   "end_exclusive": True,
                   "label_rule": "valid label requires origin+h < end of its own split "
                                 "(fit: < inner start); targets stay exact ELAPSED t+h"},
        "model": {"schema": "i6dmo.model.v1", "features": features["features"],
                  "slots": mt["observations"], "offsets_hours": None,
                  "market_time": dict(mt),
                  "time_encoder": copy.deepcopy(TIME_ENCODER_IDENTITY),
                  "horizons_hours": list(d["horizons_hours"]),
                  "branch": {"filters": m["branch"]["filters"], "kernel_size": m["branch"]["kernel_size"],
                             "strides": m["branch"]["stride"], "padding": "causal", "activation": "relu"},
                  "core": {"filters": m["core"]["filters"], "kernel_sizes": m["core"]["kernel_sizes"],
                           "strides": m["core"]["strides"], "padding": "causal", "activation": "relu"},
                  "heads": {"hidden_units": m["heads"]["hidden_units"][0],
                            "hidden_activation": m["heads"]["hidden_activation"],
                            "output_activation": m["heads"]["output_activation"]},
                  "positional_encoding": "dynamic_physical_time_after_fusion",
                  "regime": m["regime"], "seed": d["seed"]},
        "training": {"optimizer": t["optimizer"], "learning_rate": t["learning_rate"],
                     "weight_decay": t["weight_decay"], "batch_size": t["batch_size"],
                     "max_epochs": t["max_epochs"], "patience": t["patience"],
                     "min_delta": t["min_delta"], "loss": t["loss"],
                     "training_coordinate": "delta_from_origin_over_fit_naive_mae",
                     "input_scaling": "fit_only_zscore_per_feature",
                     "monitor": "0.5*(fit_loss+inner_loss) at the same end-of-epoch weights, "
                                "computed ONLY through the shared inference evaluation path",
                     "patience_rule": "stale += 1 unless monitored < reference - min_delta",
                     "restore": "lowest monitored epoch, first tie wins",
                     "batch_order": "seeded_permutation_per_epoch",
                     "rows_without_valid_head": "excluded from fit/inner training, counted",
                     "monitor_comparison_floor": MONITOR_FLOOR_DECLARATION,
                     "outer_validation_for_checkpoint_selection": False},
        "sealed_identities": copy.deepcopy(d["sealed_identities"]),
    }
    if fam in MATCHED_FAMILIES:
        effective["matched_population"] = copy.deepcopy(d["matched_population"])
        effective["matched_population_rules"] = matched_population_rules(matched_arm(d))
    if fam in (scope["family"] for scope in CONTEXT_BRIDGE_SCOPE.values()):
        # The control recipe equals legacy MG12; only new configs carry its distinct identity.
        effective["model"]["family"] = fam
    effective["effective_config_sha256"] = effective_config_sha256(effective)
    return effective


def effective_config(design):
    """Resolve a validated design to the sealed effective config used by train."""
    d = validate_design(design)
    features = check_feature_identity(list(fa.MANIFEST_NO_STLFSI4))
    if family(d) in ("short_mg2", "long_mg12") + MATCHED_FAMILIES:
        return _market_time_effective(d, features)
    s = d["splits"]
    train_start, train_end = _parse(s["train"]["start"]), _parse(s["train"]["end"])
    inner_start = train_end - timedelta(weeks=s["inner_tail_weeks"])
    inp = d["input"]
    offsets = list(range(-inp["history_span_hours"], 1, inp["spacing_hours"]))
    if len(offsets) != inp["observations"] or offsets[-1] != 0:
        raise DesignError("INPUT_GRID", "span/spacing do not give the declared observations ending at t")
    m, t = d["model"], d["training"]
    splits_out = {
        "fit": {"start": _iso(train_start), "end": _iso(inner_start)},
        "inner": {"start": _iso(inner_start), "end": _iso(train_end)},
        "test": dict(s["test"]),
        "end_exclusive": True,
        "label_rule": "valid label requires origin+h < end of its own split (fit: < inner start)",
    }
    if mode(d) == "VALIDATION":
        splits_out["validation"] = dict(s["validation"])
    effective = {
        "schema": EFFECTIVE_SCHEMA,
        "family": family(d),
        "mode": mode(d),
        "design": d,
        "design_sha256": design_sha256(d),
        "experiment_id": d["experiment_id"],
        "seed": d["seed"],
        "horizons_hours": list(d["horizons_hours"]),
        "offsets_hours": offsets,
        "feature_manifest": features,
        "splits": splits_out,
        "model": {
            "schema": "i6dmo.model.v1",
            "features": features["features"],
            "offsets_hours": offsets,
            "horizons_hours": list(d["horizons_hours"]),
            "branch": {"filters": m["branch"]["filters"], "kernel_size": m["branch"]["kernel_size"],
                       "strides": m["branch"]["stride"], "padding": "causal", "activation": "relu"},
            "core": {"filters": m["core"]["filters"], "kernel_sizes": m["core"]["kernel_sizes"],
                     "strides": m["core"]["strides"], "padding": "causal", "activation": "relu"},
            "heads": {"hidden_units": m["heads"]["hidden_units"][0],
                      "hidden_activation": m["heads"]["hidden_activation"],
                      "output_activation": m["heads"]["output_activation"]},
            "positional_encoding": "physical_offset_sinusoid_after_fusion",
            "regime": m["regime"],
            "seed": d["seed"],
        },
        "training": {
            "optimizer": t["optimizer"], "learning_rate": t["learning_rate"],
            "weight_decay": t["weight_decay"], "batch_size": t["batch_size"],
            "max_epochs": t["max_epochs"], "patience": t["patience"], "min_delta": t["min_delta"],
            "loss": t["loss"],
            "training_coordinate": "delta_from_origin_over_fit_naive_mae",
            "input_scaling": "fit_only_zscore_per_feature",
            "monitor": "0.5*(fit_loss+inner_loss) at the same end-of-epoch weights, computed ONLY "
                       "through the shared inference evaluation path (i6dmo.monitor_eval.v1); the "
                       "unambiguous implementation of the design's same-weight mean fit/inner rule — "
                       "selection semantics unchanged, numeric path unified, bitwise self-checked",
            "patience_rule": "stale += 1 unless monitored < reference - min_delta",
            "restore": "lowest monitored epoch, first tie wins",
            "batch_order": "seeded_permutation_per_epoch",
            "rows_without_valid_head": "excluded from fit/inner training, counted",
            "monitor_comparison_floor": MONITOR_FLOOR_DECLARATION,
            "outer_validation_for_checkpoint_selection": False,
        },
    }
    effective["effective_config_sha256"] = effective_config_sha256(effective)
    return effective


def validate_design_report(path):
    """Pure validate-design result: PASS with identities, or FAILED with a typed code."""
    try:
        peek = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        peek = None
    if isinstance(peek, dict) and peek.get("schema") == "i6dmo.r2_contrast_design.v1":
        # The R0-vs-R2 contract (tools/i6dmo/r2_design.py) is validated FOR EXECUTION here: an
        # objective still UNDECIDED_REQUIRES_REVIEW is a typed R2_OBJECTIVE_UNDECIDED refusal.
        from tools.i6dmo import r2_design
        try:
            result = r2_design.validate_r2_design(peek, for_execution=True)
        except (r2_design.R2DesignError, DesignError) as exc:
            return {"schema": "i6dmo.validate_design.v1", "status": "FAILED", "code": exc.code,
                    "detail": str(exc), "pure": True}
        return {"schema": "i6dmo.validate_design.v1", "status": "PASS", "pure": True,
                "family": "r2_contrast", "contrast_id": peek.get("contrast_id"),
                "seal_sha256": peek.get("seal_sha256"), "r2": result, "test_open": False}
    try:
        effective = effective_config(load_design(path))
    except DesignError as exc:
        return {"schema": "i6dmo.validate_design.v1", "status": "FAILED", "code": exc.code,
                "detail": str(exc), "pure": True}
    fm = effective["feature_manifest"]
    return {
        "schema": "i6dmo.validate_design.v1", "status": "PASS", "pure": True,
        "family": effective["family"], "mode": effective["mode"],
        "experiment_id": effective["experiment_id"],
        "design_sha256": effective["design_sha256"],
        "effective_config_sha256": effective["effective_config_sha256"],
        "feature_count": fm["count"], "feature_manifest_sha256": fm["manifest_sha256"],
        "availability_contract_sha256": fm.get("availability_contract_sha256"),
        "unverified_features": fm.get("unverified_features", []),
        "horizons_hours": effective["horizons_hours"], "offsets_hours": effective["offsets_hours"],
        "splits": {k: v for k, v in effective["splits"].items() if k in ("fit", "inner", "validation")
                   and isinstance(v, dict)},
        "test_open": False,
        "disclosure": "19-feature set was developed using 2024; outer VALIDATION is development, "
                      "not an untouched holdout; unverified producers block confirmatory TEST",
    }


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _short_closure(terminal_path):
    """Read only schema/unit_id/status/identity of the short closure terminal."""
    try:
        term = json.loads(Path(terminal_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DesignError("SHORT_NOT_TERMINAL_VALID", f"unreadable terminal: {exc}") from None
    if not isinstance(term, dict) or term.get("schema") != TERMINAL_SCHEMA:
        raise DesignError("SHORT_NOT_TERMINAL_VALID", f"terminal schema must be {TERMINAL_SCHEMA}")
    if term.get("unit_id") != SHORT_CLOSURE_UNIT:
        raise DesignError("SHORT_NOT_TERMINAL_VALID", f"closure unit must be {SHORT_CLOSURE_UNIT}")
    if term.get("status") not in SHORT_CLOSURE_STATUSES:
        raise DesignError("SHORT_NOT_TERMINAL_VALID", f"short closure status must be one of "
                          f"{SHORT_CLOSURE_STATUSES} (a scientific terminal)")
    identity = term.get("identity_sha256")
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity):
        raise DesignError("SHORT_NOT_TERMINAL_VALID", "terminal identity_sha256 missing")
    return {"unit_id": SHORT_CLOSURE_UNIT, "identity_sha256": identity,
            "terminal_file_sha256": _file_sha(terminal_path)}


def seal_long_design(short_design_path, short_terminal_path, out_path):
    """Derive, seal and write ``EXPERIMENT.long.json`` deterministically (idempotent).

    Only the short design and the closure terminal's schema/unit/status/identity
    are read. The long hyperparameters are a pure function of the short design
    and ``LONG_FROM_PLAN``; short metrics or eligibility never enter.
    """
    short = load_design(short_design_path)
    if family(short) != "short":
        raise DesignError("SHORT_DESIGN_REQUIRED", "seal-long-design derives from the short design")
    closure = _short_closure(short_terminal_path)
    doc = copy.deepcopy(short)
    for path, value in LONG_FROM_PLAN.items():
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = copy.deepcopy(value)
    open_paths = []
    for path in LONG_PLAN_OPEN_FIELDS:
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = UNDECIDED
        open_paths.append(".".join(path))
    if open_paths:
        raise DesignError("LONG_DESIGN_UNDECIDED", f"PLAN.md leaves {open_paths} open; nothing written")
    doc["derived_from"] = {"rule": LONG_RULE, "short_experiment_id": SHORT_ID,
                           "short_design_sha256": design_sha256(short), "short_terminal": closure}
    doc["seal_sha256"] = _sha(_seal_body(doc))
    validate_design(doc)
    effective_config(doc)
    text = json.dumps(doc, sort_keys=True, indent=2) + "\n"
    out = Path(out_path)
    if out.exists():
        if out.read_text(encoding="utf-8") != text:
            raise DesignError("LONG_DESIGN_EXISTS", f"{out} holds a different long design; never overwritten")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out)
    return {"schema": "i6dmo.seal_long_design.v1", "status": "SEALED", "experiment_id": LONG_ID,
            "design_sha256": design_sha256(doc), "seal_sha256": doc["seal_sha256"],
            "file_sha256": _file_sha(out), "derived_from": doc["derived_from"]}


def hourly_control_design(short_design):
    """Derive the sealed hourly-input control design from the validated short design (pure)."""
    base = validate_design(short_design)
    if family(base) != "short" or mode(base) == "FINAL_REFIT":
        raise DesignError("SHORT_DESIGN_REQUIRED",
                          "the hourly control derives from the elapsed short design")
    doc = copy.deepcopy(base)
    for path, value in HOURLY_FROM_PLAN.items():
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = copy.deepcopy(value)
    doc["derived_from"] = {"rule": HOURLY_RULE, "base_experiment_id": SHORT_ID,
                           "base_design_sha256": design_sha256(base),
                           "availability_declaration": HOURLY_AVAILABILITY_DECLARATION,
                           "common_support_rules_sha256": COMMON_SUPPORT_RULES["rules_sha256"]}
    doc["seal_sha256"] = _sha(_seal_body(doc))
    return validate_design(doc)


def seal_hourly_control(short_design_path, out_dir):
    """Write EXPERIMENT.short_hourly.json + the frozen COMMON_SUPPORT_RULES.json (idempotent)."""
    doc = hourly_control_design(load_design(short_design_path))
    effective_config(doc)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, payload in (("EXPERIMENT.short_hourly.json", doc),
                          ("COMMON_SUPPORT_RULES.json", COMMON_SUPPORT_RULES)):
        text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        path = out / name
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise DesignError("HOURLY_DESIGN_EXISTS", f"{path} holds different content")
        else:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        written[name] = str(path)
    return {"schema": "i6dmo.seal_hourly_control.v1", "status": "SEALED",
            "experiment_id": SHORT_HOURLY_ID, "design_sha256": design_sha256(doc),
            "common_support_rules_sha256": COMMON_SUPPORT_RULES["rules_sha256"],
            "files": written, "fits_run": 0}


def market_time_design(short_design, key, candidate_dir):
    """Derive one sealed market-time design (MG2 or MG12) from the validated short design."""
    import tensorflow as tf
    from tools.i6dmo.model import keras_version
    base = validate_design(short_design)
    if family(base) != "short" or mode(base) == "FINAL_REFIT":
        raise DesignError("SHORT_DESIGN_REQUIRED",
                          "the market-time designs derive from the elapsed short design")
    scope = MARKET_TIME_SCOPE[key]
    candidates = Path(candidate_dir)
    name = "CANDIDATE.short.json" if key == "mg2" else "CANDIDATE.long.json"
    candidate_path = candidates / name
    try:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        study_bytes = (candidates / "SUPPORT_STUDY.json").read_bytes()
    except OSError as exc:
        raise DesignError("MARKET_TIME_CANDIDATE_MISSING", str(exc)) from None
    if candidate.get("input_sampling", {}).get("rule") != scope["rule"]:
        raise DesignError("MARKET_TIME_CANDIDATE_MISMATCH",
                          f"{name} declares rule {candidate.get('input_sampling', {}).get('rule')!r}, "
                          f"not {scope['rule']}")
    if candidate.get("horizons_hours") != scope["horizons_hours"]:
        raise DesignError("MARKET_TIME_CANDIDATE_MISMATCH", f"{name} horizons differ from the plan")
    features = check_feature_identity(list(fa.MANIFEST_NO_STLFSI4))
    calendar = candidate["calendar"]
    doc = {
        "schema": MARKET_TIME_SCHEMA,
        "status": base["status"],
        "experiment_id": scope["experiment_id"],
        "asset": base["asset"], "seed": base["seed"], "repetitions": base["repetitions"],
        "evaluation_mode": base["evaluation_mode"],
        "retraining_frequency": base["retraining_frequency"],
        "market_time": {"rule": scope["rule"], "observations": scope["observations"],
                        "spacing_open_market_hours": scope["spacing_open_market_hours"],
                        "span_open_market_hours": scope["span_open_market_hours"],
                        "include_origin": True, "elapsed_span_fixed": False,
                        "require_exact_market_slots": True},
        "horizons_hours": list(scope["horizons_hours"]),
        "feature_policy": copy.deepcopy(base["feature_policy"]),
        "splits": copy.deepcopy(base["splits"]),
        "business": copy.deepcopy(base["business"]),
        "model": copy.deepcopy(base["model"]),
        "training": copy.deepcopy(base["training"]),
        "metrics": copy.deepcopy(base["metrics"]),
        "execution": copy.deepcopy(base["execution"]),
        "sealed_identities": {
            "base_short_design_sha256": design_sha256(base),
            "time_encoder": copy.deepcopy(TIME_ENCODER_IDENTITY),
            "feature_manifest_sha256": features["manifest_sha256"],
            "availability_contract_sha256": features["availability_contract_sha256"],
            "calendar": {"calendar_id": calendar["calendar_id"], "version": calendar["version"],
                         "status": "ASSUMED_NOT_BROKER"},
            "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "support_study_sha256": hashlib.sha256(study_bytes).hexdigest(),
            "seed": base["seed"],
            "runtime": {"tensorflow": tf.__version__, "keras": keras_version()},
        },
    }
    doc["seal_sha256"] = _sha(_seal_body(doc))
    return validate_design(doc)


def seal_market_time(short_design_path, candidate_dir, out_dir):
    """Write EXPERIMENT.mg2.json and EXPERIMENT.mg12.json (deterministic, idempotent, no fit)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"schema": "i6dmo.seal_market_time.v1", "status": "SEALED", "fits_run": 0}
    short = load_design(short_design_path)
    for key, name in (("mg2", "EXPERIMENT.mg2.json"), ("mg12", "EXPERIMENT.mg12.json")):
        doc = market_time_design(short, key, candidate_dir)
        effective_config(doc)
        text = json.dumps(doc, sort_keys=True, indent=2) + "\n"
        path = out / name
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise DesignError("MARKET_TIME_DESIGN_EXISTS", f"{path} holds different content")
        else:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        result[key] = {"experiment_id": doc["experiment_id"], "design_sha256": design_sha256(doc),
                       "seal_sha256": doc["seal_sha256"], "file": str(path),
                       "effective_config_sha256": effective_config(doc)["effective_config_sha256"]}
    return result


#: The real input's versioned ASSUMED calendar (same identity the closed MG2/MG12 designs sealed).
MATCHED_CALENDAR = {"calendar_id": "FX_NY_17_ASSUMED_NOT_BROKER", "version": "i6n-v1"}


def matched_market_time_design(short_design, arm, data_designs, *, calendar=None, runtime=None):
    """Derive ONE sealed short or long matched design; pure, no data read.

    ``data_designs`` maps both arms to their data-lane DESIGN documents
    (:func:`tools.i6dmo.common_population.matched_data_design`); this design binds its own data
    design digest and data identity plus the partner's data identity, and the frozen
    contrast-specific population rules. ``calendar``/``runtime`` default to the real calendar and
    the execution-role pins ``MATCHED_RUNTIME`` (not the sealing host's install).
    """
    from tools.i6dmo import common_population as CP
    base = validate_design(short_design)
    if family(base) != "short" or mode(base) == "FINAL_REFIT":
        raise DesignError("SHORT_DESIGN_REQUIRED",
                          "the matched designs derive from the elapsed short design")
    if arm not in ALL_MATCHED_SCOPE:
        raise DesignError("MATCHED_ARM_UNKNOWN", f"{arm!r}; one of {sorted(ALL_MATCHED_SCOPE)}")
    matched_arm_set(data_designs)
    scope = ALL_MATCHED_SCOPE[arm]
    rules = matched_population_rules(arm)
    if set(data_designs) != {arm, scope["partner"]}:
        raise DesignError("ARMS_REQUIRED", f"{arm} requires partner {scope['partner']}")
    own, partner = data_designs[arm], data_designs[scope["partner"]]
    for key, doc in ((arm, own), (scope["partner"], partner)):
        try:
            CP.validate_matched_data_design(doc, arm=key)
        except CP.CommonPopulationError as exc:
            raise DesignError(exc.code, str(exc)) from None
        if doc["derives_from_design_sha256"] != design_sha256(base):
            raise DesignError("MATCHED_DATA_DESIGN_MISMATCH",
                              f"{key} data design derives from another short design")
    features = check_feature_identity(list(fa.MANIFEST_NO_STLFSI4))
    cal = dict(calendar or MATCHED_CALENDAR)
    doc = {
        "schema": MATCHED_SCHEMA,
        "status": base["status"],
        "experiment_id": scope["experiment_id"],
        "asset": base["asset"], "seed": base["seed"], "repetitions": base["repetitions"],
        "evaluation_mode": base["evaluation_mode"],
        "retraining_frequency": base["retraining_frequency"],
        "market_time": {"rule": scope["rule"], "observations": scope["observations"],
                        "spacing_open_market_hours": scope["spacing_open_market_hours"],
                        "span_open_market_hours": scope["span_open_market_hours"],
                        "include_origin": True, "elapsed_span_fixed": False,
                        "require_exact_market_slots": True},
        "horizons_hours": list(scope["horizons_hours"]),
        "feature_policy": copy.deepcopy(base["feature_policy"]),
        "splits": copy.deepcopy(base["splits"]),
        "business": copy.deepcopy(base["business"]),
        "model": copy.deepcopy(base["model"]),
        "training": copy.deepcopy(base["training"]),
        "metrics": copy.deepcopy(base["metrics"]),
        "execution": copy.deepcopy(base["execution"]),
        "matched_population": {
            "arm": arm, "role": scope["role"], "contrast_id": rules["contrast_id"],
            "partner_arm": scope["partner"],
            "partner_experiment_id": ALL_MATCHED_SCOPE[scope["partner"]]["experiment_id"],
            "rules_sha256": rules["rules_sha256"],
            "full_population_fit_substitutes": False,
        },
        "sealed_identities": {
            "base_short_design_sha256": design_sha256(base),
            "time_encoder": copy.deepcopy(TIME_ENCODER_IDENTITY),
            "feature_manifest_sha256": features["manifest_sha256"],
            "availability_contract_sha256": features["availability_contract_sha256"],
            "calendar": {"calendar_id": cal["calendar_id"], "version": cal["version"],
                         "status": "ASSUMED_NOT_BROKER"},
            "data_design_sha256": _sha(own),
            "data_identity_sha256": own["data_identity_sha256"],
            "partner_data_identity_sha256": partner["data_identity_sha256"],
            "seed": base["seed"],
            "runtime": dict(runtime) if runtime is not None else dict(MATCHED_RUNTIME),
        },
    }
    doc["seal_sha256"] = _sha(_seal_body(doc))
    return validate_design(doc)


def _write_once(path, payload, code):
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise DesignError(code, f"{path.name} holds different content; never overwritten")
    else:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


def seal_matched_short(short_design_path, out_dir, *, calendar=None, runtime=None):
    """Write DESIGN.mg1/mg2m, MATCHED_POPULATION_RULES and EXPERIMENT.mg1/mg2m (idempotent, no fit).

    Never touches EXPERIMENT.short_hourly.json, EXPERIMENT.mg2.json or EXPERIMENT.mg12.json.
    """
    return _seal_matched(short_design_path, out_dir, MATCHED_SCOPE,
                         "i6dmo.seal_matched_short.v1", calendar=calendar, runtime=runtime)


def seal_matched_long(short_design_path, out_dir, *, calendar=None, runtime=None):
    """Seal the distinct RAW 144-market-hour MG1-long/MG12 contrast, without data or fits."""
    return _seal_matched(short_design_path, out_dir, LONG_MATCHED_SCOPE,
                         "i6dmo.seal_matched_long.v1", calendar=calendar, runtime=runtime)


def seal_context_bridge(short_design_path, out_dir, *, calendar=None, runtime=None):
    """Seal fresh MG12 control/MG2 long-head identities; no data, fit or old-source rewrite."""
    return _seal_matched(short_design_path, out_dir, CONTEXT_BRIDGE_SCOPE,
                         "i6dmo.seal_context_bridge.v1", calendar=calendar, runtime=runtime)


def _seal_matched(short_design_path, out_dir, scope, schema, *, calendar=None, runtime=None):
    from tools.i6dmo import common_population as CP
    short = load_design(short_design_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_designs = {arm: CP.matched_data_design(short, arm) for arm in scope}
    rules = matched_population_rules(next(iter(scope)))
    result = {"schema": schema, "status": "SEALED", "fits_run": 0,
              "contrast_id": rules["contrast_id"],
              "matched_population_rules_sha256": rules["rules_sha256"]}
    _write_once(out / "MATCHED_POPULATION_RULES.json", rules, "MATCHED_DESIGN_EXISTS")
    for arm in scope:
        doc = matched_market_time_design(short, arm, data_designs, calendar=calendar, runtime=runtime)
        effective = effective_config(doc)
        _write_once(out / f"DESIGN.{arm}.json", data_designs[arm], "MATCHED_DESIGN_EXISTS")
        _write_once(out / f"EXPERIMENT.{arm}.json", doc, "MATCHED_DESIGN_EXISTS")
        result[arm] = {"experiment_id": doc["experiment_id"], "family": family(doc),
                       "design_sha256": design_sha256(doc), "seal_sha256": doc["seal_sha256"],
                       "effective_config_sha256": effective["effective_config_sha256"],
                       "data_design_sha256": doc["sealed_identities"]["data_design_sha256"],
                       "data_identity_sha256": doc["sealed_identities"]["data_identity_sha256"],
                       "experiment_file_sha256": _file_sha(out / f"EXPERIMENT.{arm}.json"),
                       "data_design_file_sha256": _file_sha(out / f"DESIGN.{arm}.json")}
    return result


def run_seal_matched_short(args):
    try:
        result = seal_matched_short(args.short_design, args.out_dir)
    except DesignError as exc:
        result = {"schema": "i6dmo.seal_matched_short.v1", "status": "FAILED", "code": exc.code,
                  "detail": str(exc)}
    print(json.dumps(result, sort_keys=True) if args.json
          else f"{result['status']} {result.get('mg1', {}).get('design_sha256') or result.get('code')}")
    return 0 if result["status"] == "SEALED" else 2


def run_seal_market_time(args):
    try:
        result = seal_market_time(args.short_design, args.candidate_dir, args.out_dir)
    except DesignError as exc:
        result = {"schema": "i6dmo.seal_market_time.v1", "status": "FAILED", "code": exc.code,
                  "detail": str(exc)}
    print(json.dumps(result, sort_keys=True) if args.json
          else f"{result['status']} {result.get('mg2', {}).get('design_sha256') or result.get('code')}")
    return 0 if result["status"] == "SEALED" else 2


def run_seal_hourly_control(args):
    try:
        result = seal_hourly_control(args.short_design, args.out_dir)
    except DesignError as exc:
        result = {"schema": "i6dmo.seal_hourly_control.v1", "status": "FAILED", "code": exc.code,
                  "detail": str(exc)}
    print(json.dumps(result, sort_keys=True) if args.json
          else f"{result['status']} {result.get('design_sha256') or result.get('code')}")
    return 0 if result["status"] == "SEALED" else 2


def run_seal_long_design(args):
    try:
        result = seal_long_design(args.short_design, args.short_terminal, args.out)
    except DesignError as exc:
        result = {"schema": "i6dmo.seal_long_design.v1", "status": "FAILED", "code": exc.code,
                  "detail": str(exc)}
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"{result['status']} {result.get('design_sha256') or result.get('code')}")
    return 0 if result["status"] == "SEALED" else 2


def run_validate_design(args):
    report = validate_design_report(args.design)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"{report['status']} {report.get('design_sha256') or report.get('code')}")
    return 0 if report["status"] == "PASS" else 2


def add_cli(subparsers):
    """Register ``validate-design`` and ``seal-long-design`` on the shared CLI parser.

    ``seal-long`` (campaign sealing) belongs to the ops lane; this name does not collide.
    """
    p = subparsers.add_parser("validate-design", help="pure typed validation of an I6-DMO design")
    p.add_argument("--design", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=run_validate_design)
    q = subparsers.add_parser("seal-long-design", help="derive and seal EXPERIMENT.long.json after short closure")
    q.add_argument("--short-design", required=True)
    q.add_argument("--short-terminal", required=True, help="TERMINAL.json of the short.report unit")
    q.add_argument("--out", required=True)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=run_seal_long_design)
    h = subparsers.add_parser("seal-hourly-control",
                              help="seal the paired short hourly-input control design (no fit)")
    h.add_argument("--short-design", required=True)
    h.add_argument("--out-dir", required=True)
    h.add_argument("--json", action="store_true")
    h.set_defaults(func=run_seal_hourly_control)
    m = subparsers.add_parser("seal-market-time",
                              help="seal the MG2/MG12 market-time pair designs (no fit)")
    m.add_argument("--short-design", required=True)
    m.add_argument("--candidate-dir", required=True)
    m.add_argument("--out-dir", required=True)
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=run_seal_market_time)
    k = subparsers.add_parser("seal-matched-short",
                              help="seal the matched short MG1/MG2 designs (no data, no fit)")
    k.add_argument("--short-design", required=True)
    k.add_argument("--out-dir", required=True)
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=run_seal_matched_short)
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.design")
    add_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
