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


def mode(doc):
    """``FINAL_REFIT`` for a derived final design, otherwise ``VALIDATION``."""
    eid = doc.get("experiment_id") if isinstance(doc, dict) else None
    return "FINAL_REFIT" if isinstance(eid, str) and eid.endswith(FINAL_SUFFIX) else "VALIDATION"


def family(doc):
    """``long`` for the long experiment ids, otherwise ``short`` (validated as such)."""
    eid = doc.get("experiment_id") if isinstance(doc, dict) else None
    return "long" if eid in (LONG_ID, LONG_ID + FINAL_SUFFIX) else "short"


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
    if mode(base) == "FINAL_REFIT":
        raise DesignError("FINAL_FROM_VALIDATION_ONLY", "a final design derives from the validation design")
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


def effective_config(design):
    """Resolve a validated design to the sealed effective config used by train."""
    d = validate_design(design)
    features = check_feature_identity(list(fa.MANIFEST_NO_STLFSI4))
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
            "outer_validation_for_checkpoint_selection": False,
        },
    }
    effective["effective_config_sha256"] = effective_config_sha256(effective)
    return effective


def validate_design_report(path):
    """Pure validate-design result: PASS with identities, or FAILED with a typed code."""
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
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.design")
    add_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
