"""Guarded final TEST path: gates, TEST bundle construction, predictions, final hooks.

This module replaces the unconditional FINAL_TEST_HOOK_NOT_PREPARED placeholder
(AUD-DMO-03). It is the ONLY code allowed to touch a TEST bundle, and it never
trains: the final weights come from the typed FINAL_REFIT units and are frozen.

Every entry point guards first. ``guard_test(root)`` admits the campaign only
when, in that campaign's ``gates/`` directory:

- ``TEST_READINESS.json`` is READY (ops writes it only when every gate passed);
- ``PAIR_VALIDATION.json`` is PASS (validation eligibility, strategy lane);
- ``PROCEDURE_FREEZE.json`` is PASS and its ``final_fit_policy`` equals the
  typed FINAL_REFIT protocol (frozen procedure, never a different window);
- ``DATA_PROVENANCE.json`` is PASS and its evidence resolves BOTH FRED
  publication clocks (``fred_clocks`` entries not UNVERIFIED for
  ``fred.rates.dgs3mo.diff_5d`` and ``fred.rates.dgs7.logret_5d``).

A failed guard is a typed :class:`FinalBlocked` naming the exact missing or
failing object; the ``run_unit`` hook turns it into a FAILED terminal with
disposition ``FINAL_BLOCKED_GATE:<name>``, never a generic placeholder.

Execution conventions are NEVER re-derived here: the final backtest consumes
the strategy lane's sealed receipt through
``strategy_adapter.execution_from_policy(load_execution_policy(root))`` and
refuses while ``gates/EXECUTION_POLICY.json`` is absent or tampered
(AUD-DMO-06 successor, lane 3).

TEST arrays are supplied by the data lane's exact-DS2 materializer (or by
synthetic fixtures in software tests); this module authenticates, formats and
binds them to the readiness receipt. The strategy lane keeps the backtest
exports; the final backtest consumes the predictions written here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tools.i6dmo import design as D
from tools.i6dmo import train as T

TEST_BUNDLE_SCHEMA = "i6dmo.data_bundle.v1"
FRED_CLOCK_FEATURES = ("fred.rates.dgs3mo.diff_5d", "fred.rates.dgs7.logret_5d")
REQUIRED_GATES = ("PAIR_VALIDATION", "PROCEDURE_FREEZE", "DATA_PROVENANCE")


class FinalBlocked(Exception):
    """Typed guard refusal; the message names the exact missing/failing object."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _read_json(path, code, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalBlocked(code, f"{what} at {Path(path).name}: {exc}") from None


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_test(root):
    """Admit the final TEST path for one campaign root, or raise typed FinalBlocked."""
    gates = Path(root) / "gates"
    readiness = _read_json(gates / "TEST_READINESS.json", "TEST_READINESS_MISSING",
                           "gates/TEST_READINESS.json (ops tick writes it when every gate passes)")
    if readiness.get("state") != "READY" or not readiness.get("readiness_sha256"):
        raise FinalBlocked("TEST_READINESS_NOT_READY", f"state={readiness.get('state')}")
    receipts = {}
    for name in REQUIRED_GATES:
        receipt = _read_json(gates / f"{name}.json", f"GATE_MISSING_{name}", f"gates/{name}.json")
        if receipt.get("status") != "PASS":
            raise FinalBlocked(f"GATE_NOT_PASS_{name}", f"status={receipt.get('status')}")
        if any(str(s).lower() == "test" for s in receipt.get("splits_read", [])):
            raise FinalBlocked(f"GATE_TEST_FEEDBACK_{name}", "a pre-TEST gate may not read TEST")
        receipts[name] = receipt
    evidence = receipts["DATA_PROVENANCE"].get("evidence") or {}
    clocks = evidence.get("fred_clocks") or {}
    unresolved = [f for f in FRED_CLOCK_FEATURES
                  if (clocks.get(f) or {}).get("status") in (None, "UNVERIFIED", "STILL_UNRESOLVED")]
    if unresolved:
        raise FinalBlocked("DATA_PROVENANCE_FRED_CLOCKS",
                           f"publication clocks unresolved for {unresolved}; TEST stays closed")
    policy = (receipts["PROCEDURE_FREEZE"].get("evidence") or {}).get("final_fit_policy") or {}
    expected = {"train": [D.FINAL_TRAIN["start"], D.FINAL_TRAIN["end"]], "inner_tail_weeks": 8,
                "weights_frozen_through_test": True}
    if {k: policy.get(k) for k in expected} != expected:
        raise FinalBlocked("PROCEDURE_FREEZE_POLICY_MISMATCH",
                           f"frozen final_fit_policy {policy} != typed FINAL_REFIT {expected}")
    return {"schema": "i6dmo.final_guard.v1", "state": "READY",
            "readiness_sha256": readiness["readiness_sha256"],
            "gates": {name: receipts[name].get("receipt_sha256") for name in REQUIRED_GATES}}


def guard_execution_policy(root):
    """The sealed execution policy, via the strategy lane's loader; typed refusals."""
    from tools.i6dmo import strategy_adapter
    try:
        receipt = strategy_adapter.load_execution_policy(root)
    except Exception as exc:  # StrategyAdapterError EXECUTION_POLICY_TAMPERED
        raise FinalBlocked("EXECUTION_POLICY_TAMPERED", str(exc)) from None
    if receipt is None:
        raise FinalBlocked("EXECUTION_POLICY_MISSING",
                           "gates/EXECUTION_POLICY.json (strategy lane sealed receipt) is absent")
    try:
        params = strategy_adapter.execution_from_policy(receipt)
    except Exception as exc:
        raise FinalBlocked("EXECUTION_POLICY_INVALID", str(exc)) from None
    return receipt, params


def _require_readiness(readiness):
    if not isinstance(readiness, dict) or readiness.get("state") != "READY" \
            or not readiness.get("readiness_sha256"):
        raise FinalBlocked("TEST_READINESS_MISSING",
                           "pass the dict returned by guard_test(root); TEST is never built without it")
    return readiness


def design_file_for(design_doc, directory, name="design.json"):
    """Write a validated design document to a file for the trainer CLI; returns the path."""
    doc = D.validate_design(design_doc)
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return path


def build_test_bundle(final_design, split_arrays, out_dir, *, readiness):
    """Write the guarded TEST bundle: one ``test`` split, bound to the readiness receipt.

    ``split_arrays`` maps ``{"test": arrays}`` with the ordinary bundle arrays
    (row_id, x, origin_price, target_price, mask) produced by the data lane's
    exact-DS2 materializer for the TEST year (synthetic in software tests).
    The ordinary trainer refuses this bundle (TEST_FORBIDDEN); only
    :func:`generate_test_predictions` consumes it, without training.
    """
    readiness = _require_readiness(readiness)
    final = D.validate_design(final_design)
    if D.mode(final) != "FINAL_REFIT":
        raise FinalBlocked("FINAL_DESIGN_REQUIRED", "the TEST bundle binds to the sealed final design")
    if set(split_arrays) != {"test"}:
        raise FinalBlocked("TEST_SPLIT_ONLY", f"got {sorted(split_arrays)}")
    effective = D.effective_config(final)
    test = effective["splits"]["test"]
    start = int(datetime.strptime(test["start"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    end = int(datetime.strptime(test["end"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    arrays = split_arrays["test"]
    origin = np.asarray(arrays["row_id"], dtype=np.int64)
    if ((origin < start) | (origin >= end)).any():
        raise FinalBlocked("TEST_WINDOW_VIOLATION", "a TEST origin lies outside the declared TEST year")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "test.npz"
    np.savez(path, **arrays)
    manifest = {
        "schema": TEST_BUNDLE_SCHEMA,
        "test_bundle": True,
        "readiness_sha256": readiness["readiness_sha256"],
        "gates": readiness["gates"],
        "design_sha256": D.design_sha256(final),
        "features": effective["feature_manifest"]["features"],
        "feature_manifest_sha256": effective["feature_manifest"]["manifest_sha256"],
        "horizons_hours": effective["horizons_hours"],
        "offsets_hours": effective["offsets_hours"],
        "splits": {"test": {"path": path.name, "sha256": _file_sha(path), "n_rows": int(len(origin))}},
    }
    (out / "BUNDLE.json").write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return out


def _load_test_bundle(bundle_dir, readiness):
    readiness = _require_readiness(readiness)
    root = Path(bundle_dir)
    manifest = _read_json(root / "BUNDLE.json", "TEST_BUNDLE_MISSING", "guarded TEST bundle manifest")
    if manifest.get("schema") != TEST_BUNDLE_SCHEMA or manifest.get("test_bundle") is not True:
        raise FinalBlocked("TEST_BUNDLE_SCHEMA", "not a guarded TEST bundle")
    if manifest.get("readiness_sha256") != readiness["readiness_sha256"]:
        raise FinalBlocked("TEST_BUNDLE_READINESS_MISMATCH",
                           "the bundle binds a different readiness receipt")
    entry = manifest["splits"]["test"]
    path = root / entry["path"]
    if _file_sha(path) != entry["sha256"]:
        raise FinalBlocked("TEST_BUNDLE_DIGEST_MISMATCH", entry["path"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return manifest, arrays


def generate_test_predictions(final_unit_dir, test_bundle_dir, out_dir, *, readiness):
    """Predict the TEST year with the frozen final model; write test.npz + a receipt.

    No training, no scale refitting: the final unit's fit-only SCALES.json and
    restored weights are reused exactly, and the output binds their hashes.
    """
    readiness = _require_readiness(readiness)
    unit = Path(final_unit_dir)
    receipt = _read_json(unit / "RECEIPT.json", "FINAL_UNIT_RECEIPT_MISSING", "final refit receipt")
    if receipt.get("status") != "COMPLETED":
        raise FinalBlocked("FINAL_UNIT_NOT_COMPLETED", f"status={receipt.get('status')}")
    scales = _read_json(unit / "SCALES.json", "FINAL_UNIT_SCALES_MISSING", "fit-only scales")
    manifest, arrays = _load_test_bundle(test_bundle_dir, readiness)
    if (receipt.get("identity") or {}).get("design_sha256") != manifest["design_sha256"]:
        raise FinalBlocked("FINAL_DESIGN_MISMATCH",
                           "the TEST bundle binds a different sealed final design than the trained unit")
    from tools.i6dmo import model as M
    bundle, document = M.load_dmo(unit / "model")
    if document["weights_sha256"] != receipt["identity"]["weights_sha256"]:
        raise FinalBlocked("FINAL_WEIGHTS_MISMATCH", "unit receipt and model archive disagree")
    x, _, _ = T.transform(arrays, scales)
    predicted = T.to_price(M.predict_matrix(bundle.forecast_model, x), arrays, scales)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "test.npz", row_id=arrays["row_id"],
             horizons_hours=np.asarray(manifest["horizons_hours"], np.int64),
             origin_price=arrays["origin_price"], predicted_price=predicted,
             target_price=arrays["target_price"], mask=arrays["mask"])
    result = {"schema": "i6dmo.final_test_predictions.v1", "rows": int(len(arrays["row_id"])),
              "horizons_hours": manifest["horizons_hours"],
              "weights_sha256": document["weights_sha256"],
              "scales_sha256": scales.get("scales_sha256"),
              "design_sha256": manifest["design_sha256"],
              "readiness_sha256": readiness["readiness_sha256"],
              "predictions_sha256": _file_sha(out / "test.npz"), "trained": False}
    (out / "PREDICTIONS.json").write_text(json.dumps(result, sort_keys=True, indent=1) + "\n",
                                          encoding="utf-8")
    return result


# ------------------------------------------------------------------ ops hook (final_test)

def _campaign_root(spec):
    root = (spec.get("paths") or {}).get("campaign_root") or spec.get("campaign_root")
    if not root:
        raise FinalBlocked("TEST_READINESS_MISSING", "spec carries no campaign_root to locate gates/")
    return Path(root)


def run_unit(spec, out_dir, progress_path):
    """Ops hook for the ``final.pair_test`` unit (unit_runner HOOKS["final_test"]).

    Guards first; a refusal is a typed FAILED terminal naming the blocking
    object (``FINAL_BLOCKED_GATE:<code>``). With all guards green it generates
    frozen-model TEST predictions for both families and delegates the business
    backtest to the strategy lane's stable exports; absent final artifacts are
    named individually, never collapsed into a placeholder.
    """
    started = time.time()
    out_dir = Path(out_dir)
    artifacts, reports, gates = {}, [], {}

    def terminal(status, disposition):
        return dict(schema="i6dmo.terminal.v1", unit_id=spec.get("unit_id"),
                    identity_sha256=spec.get("identity_sha256"), status=status, disposition=disposition,
                    artifacts=artifacts, eligibility={}, gates=gates, reports=reports,
                    costs=dict(wall_seconds=time.time() - started, fit_seconds=None,
                               score_seconds=time.time() - started, cgroup_peak_bytes=None, gpu_uuid=None))

    try:
        if spec.get("kind") != "final_test":
            raise FinalBlocked("KIND_INVALID", str(spec.get("kind")))
        root = _campaign_root(spec)
        readiness = guard_test(root)
        gates.update(readiness["gates"], readiness_sha256=readiness["readiness_sha256"])
        policy_receipt, policy_params = guard_execution_policy(root)
        gates.update(execution_policy_sha256=policy_receipt["receipt_sha256"],
                     costs_sha256=policy_params["costs_sha256"],
                     cost_verification=policy_params["cost_verification"])
        upstream = spec.get("upstream_dirs") or {}
        test_bundles = spec.get("test_bundles") or {}
        predictions = {}
        for family in ("short", "long"):
            unit = upstream.get(f"final.{family}_refit")
            if not unit:
                raise FinalBlocked("FINAL_UNIT_MISSING",
                                   f"spec.upstream_dirs lacks final.{family}_refit (its trained unit dir)")
            bundle = test_bundles.get(family)
            if not bundle:
                raise FinalBlocked("TEST_BUNDLE_MISSING",
                                   f"spec.test_bundles lacks the guarded {family} TEST bundle")
            result = generate_test_predictions(unit, bundle, out_dir / f"predictions_{family}",
                                               readiness=readiness)
            rel = f"predictions_{family}/test.npz"
            artifacts[f"{family}_test_predictions"] = {"path": rel, "sha256": result["predictions_sha256"]}
            predictions[family] = result
        backtest = spec.get("_backtest_runner")  # tests inject a fake; default = strategy lane exports
        if backtest is None:
            from tools.i6dmo import strategy_adapter
            backtest = getattr(strategy_adapter, "run_final_backtest", None)
            if backtest is None:
                raise FinalBlocked("FINAL_BACKTEST_EXPORT_MISSING",
                                   "strategy lane export run_final_backtest(spec, predictions, "
                                   "execution_params, out_dir) is not yet published; the sealed policy "
                                   "is verified and the TEST predictions are complete and retained")
        result = backtest(spec, predictions, policy_params, out_dir)
        reports.extend(result.get("reports", []))
        gates["pair_status"] = result.get("pair_status", "PASS")
        return terminal("COMPLETE", result.get("disposition", "FINAL_TEST_COMPLETE"))
    except FinalBlocked as blocked:
        return terminal("FAILED", f"FINAL_BLOCKED_GATE:{blocked.code}")
    except (D.DesignError, T.TrainDisposition) as exc:
        return terminal("FAILED", f"FINAL_BLOCKED_GATE:{getattr(exc, 'code', 'ERROR')}")
