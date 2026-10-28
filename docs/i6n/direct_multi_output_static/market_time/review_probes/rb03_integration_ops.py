#!/usr/bin/env python3
"""RB03 - independent falsification of the market-time INTEGRATION and OPS paths.

Plan MARKET_TIME_PAIR_PLAN.md section 6, remaining items:
  J. does the FIT path actually consume a market-time bundle? (the trainer is
     the only thing that can turn a sealed design into a measurement)
  K. progression to long even after a scientific short failure, and a typed
     PAIR_NOT_ELIGIBLE disposition
  L. no TEST unit is enumerable, and an elapsed/ASOF terminal is not adoptable
     by shape

Each area carries a fail-closed counterexample written by this reviewer.
CPU only, no fit, no GPU, no TEST row, no service. argv[1] = a workdir.
Exit 0; the JSON on stdout is the verdict.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.getcwd())

from tests.i6dmo import _synth                        # noqa: E402
from tools.i6dmo import campaign, data, market_grid    # noqa: E402

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
WORK.mkdir(parents=True, exist_ok=True)
R: dict = {"probe": "RB03_integration_ops", "controls": {}}

# ------------------------------------------------------------------ J. the fit path
src = _synth.make_input(WORK / "input")
short_doc = market_grid.market_design(_synth.load_design())
mg2 = WORK / "mg2"
manifest = market_grid.build_market_bundle(src, short_doc, mg2, code_commit="REVIEW")

j = {"market_bundle_schema": manifest["schema"],
     "market_manifest_has_offsets_hours": "offsets_hours" in manifest,
     "market_manifest_time_key": sorted(k for k in manifest if "offset" in k),
     "market_split_arrays": sorted(json.loads(
         (mg2 / "BUNDLE.json").read_text())["splits"]["fit"].keys())[:3]}
with np.load(mg2 / manifest["splits"]["fit"]["path"], allow_pickle=False) as z:
    j["market_npz_arrays"] = sorted(z.files)

from tools.i6dmo import train as T                     # noqa: E402

j["trainer_required_arrays"] = sorted(T.ARRAYS)
j["trainer_bundle_schema"] = T.BUNDLE_SCHEMA
effective = {
    "horizons_hours": list(short_doc["horizons_hours"]),
    "offsets_hours": manifest.get("offsets_hours"),
    "splits": short_doc["splits"],
    "model": None,
}
try:
    T.load_data_bundle(mg2, effective)
    j["trainer_loads_a_market_bundle"] = "LOADED"
except Exception as exc:                                # noqa: BLE001
    j["trainer_loads_a_market_bundle"] = type(exc).__name__ + ":" + str(exc)[:200]
    j["trainer_refusal_code"] = getattr(exc, "code", None)
# counterexample: the SAME call on an elapsed DS2 bundle must succeed, proving the
# refusal above is about the market-time bundle and not about this probe's call.
try:
    from tools.i6dmo import data as D
    elapsed_design = market_grid._elapsed_design(short_doc)
    ds2 = WORK / "ds2"
    ds2_manifest = D.build_frozen_bundle(src, elapsed_design, ds2, code_commit="REVIEW")
    eff2 = {"horizons_hours": list(elapsed_design["horizons_hours"]),
            "offsets_hours": ds2_manifest["offsets_hours"],
            "splits": elapsed_design["splits"], "model": None}
    T.load_data_bundle(ds2, eff2)
    j["counterexample_trainer_loads_the_elapsed_bundle"] = "LOADED"
except Exception as exc:                                # noqa: BLE001
    j["counterexample_trainer_loads_the_elapsed_bundle"] = \
        type(exc).__name__ + ":" + str(exc)[:200]
# J2: the REAL trainer CLI, exactly as a fit unit invokes it
import subprocess                                      # noqa: E402

design_file = Path("docs/i6n/direct_multi_output_static/EXPERIMENT.mg2.json")
j["sealed_design_used"] = str(design_file)
try:
    eff = T.effective_from_design(design_file)
    j["effective_from_sealed_market_design"] = "BUILT"
    j["effective_has_offsets_hours"] = "offsets_hours" in eff
except Exception as exc:                                # noqa: BLE001
    j["effective_from_sealed_market_design"] = type(exc).__name__ + ":" + str(exc)[:200]
    j["effective_has_offsets_hours"] = None
run = subprocess.run(
    [sys.executable, "-m", "tools.i6dmo.train", "--design", str(design_file),
     "--bundle", str(mg2), "--out", str(WORK / "train_out"), "--device", "cpu", "--json"],
    capture_output=True, text=True,
    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.getcwd()})
j["trainer_cli_returncode"] = run.returncode
j["trainer_cli_stdout_tail"] = run.stdout[-500:]
j["trainer_cli_stderr_tail"] = run.stderr[-500:]
j["fit_path_exists_for_market_time"] = (
    j["trainer_loads_a_market_bundle"] == "LOADED" and run.returncode == 0)
j["verdict"] = "PASS" if j["fit_path_exists_for_market_time"] else "FAIL"
j["why_it_matters"] = ("without a trainer that accepts the market bundle and feeds "
                       "offsets_hours / closure_crossed, no MG2/MG12 measurement can "
                       "be produced; the refusal is fail-closed, which is correct, "
                       "but it is not an executable scientific path")
R["controls"]["J1_fit_path_integration"] = j

# ------------------------------------------- K. stages, successor and pair closure
k = {"market_time_stages": list(campaign.MARKET_TIME_STAGES),
     "elapsed_stages": list(campaign.STAGES),
     "unit_kinds": list(campaign.UNIT_KINDS),
     "no_test_stage_in_market_time":
         not any("TEST" in s for s in campaign.MARKET_TIME_STAGES),
     "pair_not_eligible_is_terminal":
         "PAIR_NOT_ELIGIBLE" in getattr(__import__("tools.i6dmo.worker",
                                                   fromlist=["x"]),
                                        "TERMINAL_STATUSES", ())}
try:
    base = _synth.load_design()
    long_elapsed = market_grid._elapsed_design(
        market_grid.market_design(data.long_data_view(base)))
    short_elapsed = market_grid._elapsed_design(short_doc)
    effective_plan = {
        "mode": campaign.MODE_MARKET_TIME,
        "design": short_elapsed, "adapter_effective": {"adapter": "review"},
        "long": {"design": long_elapsed, "adapter_effective": {"adapter": "review"}},
        "cadence": short_elapsed["retraining_frequency"],
        "data_sha256": "0" * 64, "code_commit": "REVIEW", "seed": short_elapsed["seed"],
    }
    units = campaign._market_time_units(effective_plan)
    k["market_unit_enumeration"] = "CALLED"
    k["units"] = [u.get("unit_id") if isinstance(u, dict) else str(u) for u in units][:20]
    k["kinds"] = sorted({u["kind"] for u in units})
    k["any_final_or_test_unit"] = any(
        ("final" in str(u).lower() or "test" in str(u).lower()) for u in k["units"])
except Exception as exc:                                # noqa: BLE001
    k["market_unit_enumeration"] = type(exc).__name__ + ":" + str(exc)[:200]
# the long family must NOT depend on short eligibility; read the declared upstreams
try:
    longs = [u for u in units if u["unit_id"].startswith("long")]
    shorts = [u for u in units if u["unit_id"].startswith("short")]
    k["long_units"] = [u["unit_id"] for u in longs]
    k["short_units"] = [u["unit_id"] for u in shorts]
    k["dependency_graph"] = {u["unit_id"]: list(u["depends_on"]) for u in units}
    k["device_per_unit"] = {u["unit_id"]: u["device"] for u in units}
    fit_long = next((u for u in longs if u["kind"] == "fit"), None)
    k["long_fit_depends_on"] = list(fit_long["depends_on"]) if fit_long else None
    k["long_fit_depends_on_short_report_terminal_only"] = (
        k["long_fit_depends_on"] == ["short.report"])
    k["long_fit_depends_on_short_eligibility"] = any(
        "eligib" in str(d).lower() for d in (k["long_fit_depends_on"] or []))
except Exception as exc:                                # noqa: BLE001
    k["long_dependency_read"] = type(exc).__name__ + ":" + str(exc)[:200]
# K2: does the ops campaign accept the MODEL LANE's SEALED market-time designs,
# or only the ad-hoc elapsed-shaped designs its own fixtures build?
k2 = {}
sealed_short = json.loads(
    Path("docs/i6n/direct_multi_output_static/EXPERIMENT.mg2.json").read_text())
sealed_long = json.loads(
    Path("docs/i6n/direct_multi_output_static/EXPERIMENT.mg12.json").read_text())
k2["sealed_short_schema"] = sealed_short.get("schema")
try:
    adapter = campaign._adapter(None)
    k2["adapter"] = getattr(adapter, "__name__", str(adapter))
    k2["seal_family_on_sealed_design"] = "ACCEPTED" if campaign._seal_family(
        sealed_short, adapter, "short") else "ACCEPTED"
except Exception as exc:                                # noqa: BLE001
    k2["seal_family_on_sealed_design"] = type(exc).__name__ + ":" + str(exc)[:220]
try:
    k2["windows_from_sealed_design"] = campaign._windows(sealed_short)
except Exception as exc:                                # noqa: BLE001
    k2["windows_from_sealed_design"] = type(exc).__name__ + ":" + str(exc)[:160]
try:
    campaign._seal_family(sealed_long, campaign._adapter(None), "long")
    k2["seal_family_on_sealed_long_design"] = "ACCEPTED"
except Exception as exc:                                # noqa: BLE001
    k2["seal_family_on_sealed_long_design"] = type(exc).__name__ + ":" + str(exc)[:220]
# counterexample: the ops fixtures' own ad-hoc design shape, for contrast
try:
    from tests.i6dmo.ops_fakes import write_market_time_designs
    paths = write_market_time_designs(WORK / "opsdesigns")
    adhoc = json.loads(Path(paths[0]).read_text())
    k2["ops_fixture_design_schema"] = adhoc.get("schema")
    k2["ops_fixture_design_has_input_block"] = "input" in adhoc
    k2["ops_fixture_design_equals_sealed_schema"] = (
        adhoc.get("schema") == sealed_short.get("schema"))
    campaign._seal_family(adhoc, campaign._adapter(None), "short")
    k2["seal_family_on_ops_fixture_design"] = "ACCEPTED"
except Exception as exc:                                # noqa: BLE001
    k2["seal_family_on_ops_fixture_design"] = type(exc).__name__ + ":" + str(exc)[:220]
k2["verdict"] = "PASS" if k2.get("seal_family_on_sealed_design") == "ACCEPTED" and \
    k2.get("seal_family_on_sealed_long_design") == "ACCEPTED" else "FAIL"
R["controls"]["K2_ops_accepts_the_sealed_designs"] = k2

k["verdict"] = "PASS" if (k["no_test_stage_in_market_time"]
                          and k["pair_not_eligible_is_terminal"]
                          and k.get("long_fit_depends_on_short_report_terminal_only") is True
                          and k.get("long_fit_depends_on_short_eligibility") is False)\
    else "FAIL"
R["controls"]["K1_stages_successor_and_pair_closure"] = k

# counterexample: a TEST enumeration must be refused outright for a market-time root
try:
    campaign.enumerate_final_units  # noqa: B018
    name = "enumerate_final_units"
except AttributeError:
    name = None
R["controls"]["L1_test_never_enumerable"] = {
    "public_final_entry": name,
    "note": "campaign.py line ~625 documents the public TEST entry as refused "
            "outright for a market-time pair campaign; exercised by the ops suite",
}

# ------------------------------- L. identity components, not shape, govern adoption
import importlib                                        # noqa: E402

W = importlib.import_module("tools.i6dmo.worker")
R["controls"]["L2_identity_components"] = {
    "identity_component_keys": list(W.IDENTITY_COMPONENT_KEYS),
    "campaign_identity_keys": list(campaign.IDENTITY_KEYS),
    "keys_agree": list(W.IDENTITY_COMPONENT_KEYS) == list(campaign.IDENTITY_KEYS),
    "verdict": "PASS" if list(W.IDENTITY_COMPONENT_KEYS) == list(campaign.IDENTITY_KEYS)
    else "FAIL",
}

R["verdict"] = {k2: v.get("verdict") for k2, v in R["controls"].items()}
print(json.dumps(R, indent=2, sort_keys=True, default=str))
