"""Independent reviewer probe RP06: sealed execution-policy receipt (no owner gate,
tamper refused), exact-bundle identity refusing ASOF/superseded pins, and a
five-entry TRACEABILITY.json pointer spot-check.

Exit 0 iff every control holds.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tools.i6dmo import campaign, strategy_adapter as S  # noqa: E402

results = {}


def check(name, ok, detail):
    results[name] = {"ok": bool(ok), "detail": detail}


work = Path(sys.argv[1])

INSTRUMENT = {
    "instrument_id": "EURUSD", "config_version": "reviewer-rp06",
    "price_decimals": 5, "point_size": 0.00001, "pip_size": 0.0001,
    "pip_cost_parameter": 0.00001, "lot_units": 100000,
    "costs": {"spread_price": 0.00013, "slippage_price": 0.00002,
              "commission_per_lot_round_trip": 0.0, "swap_per_lot_per_day": 0.0},
    "cost_verification": "NOT_BROKER_VERIFIED",
}

# --- seal + reload round trip, deterministic, no approval prompt in the path ---
receipt = S.seal_execution_policy(work, INSTRUMENT)
loaded = S.load_execution_policy(work)
check("policy_seal_roundtrip", loaded == receipt
      and receipt["business_rules"]["min_valid_long_heads"] == 2
      and receipt["simulation_assumptions"]["statement"].startswith("SIMULATION_ASSUMPTION")
      and receipt["reporting"]["periods_per_year"] == 260,
      {"receipt_sha256": receipt["receipt_sha256"],
       "min_valid_long_heads": receipt["business_rules"]["min_valid_long_heads"],
       "cost_verification": receipt["instrument"]["cost_verification"]})

# --- reviewer mutation: one changed byte in the sealed receipt is refused typed ---
path = work / "gates" / "EXECUTION_POLICY.json"
doc = json.loads(path.read_text())
doc["business_rules"]["min_valid_long_heads"] = 1  # keep the stale digest
path.write_text(json.dumps(doc, sort_keys=True))
try:
    S.load_execution_policy(work)
    check("policy_tamper_refused", False, "tampered receipt accepted")
except S.StrategyAdapterError as exc:
    check("policy_tamper_refused", exc.code == "EXECUTION_POLICY_TAMPERED", f"typed {exc.code}")

# --- no owner-approval gate remains in the campaign gate table ---
gate_text = json.dumps(campaign.GATES)
check("no_owner_approval_gate",
      "STRATEGY_OWNER_RULINGS" not in gate_text and "WAIT_OWNER" not in gate_text
      and campaign.GATES["EXECUTION_POLICY"][2] is True,
      {"gates": sorted(campaign.GATES), "execution_policy_auto": campaign.GATES["EXECUTION_POLICY"][2]})

# --- exact-bundle identity: ASOF-diagnostic and superseded pins are refused ---
def pin(identity, rule="DS2_EXACT"):
    return {"short_bundle": {"relpath": "bundles/x", "sha256": "0" * 64,
                             "bundle_sha256": identity, "input_sampling": rule}}

refusals = {}
for name, identity, rule in (
        ("superseded", "5221615d" + "0" * 56, "DS2_EXACT"),
        ("asof_arm_1", "3e42c3f5" + "0" * 56, "DS2_EXACT"),
        ("asof_arm_2", "e88f72aa" + "0" * 56, "DS2_EXACT"),
        ("asof_rule", "a" * 64, "DS2_ASOF_LAST_AVAILABLE")):
    try:
        campaign._check_bundle_pins(pin(identity, rule))
        refusals[name] = "ACCEPTED"
    except campaign.Refused as exc:
        refusals[name] = str(exc).split(":")[0]
try:
    campaign._check_bundle_pins(pin("b" * 64))
    refusals["exact_ok"] = "ACCEPTED"
except campaign.Refused as exc:
    refusals["exact_ok"] = str(exc).split(":")[0]
check("bundle_identity_refusals",
      refusals == {"superseded": "SUPERSEDED_BUNDLE", "asof_arm_1": "ASOF_DIAGNOSTIC_ARM",
                   "asof_arm_2": "ASOF_DIAGNOSTIC_ARM", "asof_rule": "BUNDLE_SAMPLING",
                   "exact_ok": "ACCEPTED"}, refusals)

# --- TRACEABILITY.json: spot-check five entries' code/evidence/test pointers ---
doc = json.loads((ROOT / "docs/i6n/direct_multi_output_static/TRACEABILITY.json").read_text())
entries = doc["entries"]
picked = ["DMO01", "DMO02", "DMO10", "DMO12", "DMO20"]
spot = {}
for key in picked:
    e = entries[key]
    missing = [p for p in list(e.get("code", [])) + list(e.get("evidence", []))
               if not (ROOT / p).exists()]
    bad_tests = []
    for t in e.get("tests", []):
        file_part, _, selector = t.partition("::")
        fp = ROOT / file_part
        if not fp.exists():
            bad_tests.append(t)
            continue
        if selector:
            # selectors are prefixes or -k substrings of collected test names
            stem = re.escape(selector.rstrip("_"))
            text = fp.read_text()
            if not re.search(rf"def [a-z0-9_]*{stem}", text) and stem not in text:
                bad_tests.append(t)
    spot[key] = {"missing_files": missing, "unresolved_tests": bad_tests,
                 "declared_broken": e.get("broken_pointers", [])}
check("traceability_spot_check",
      doc["broken"] == [] and all(not v["missing_files"] and not v["unresolved_tests"]
                                  for v in spot.values()),
      spot)

ok = all(v["ok"] for v in results.values())
print(json.dumps({"probe": "RP06_policy_final_traceability", "ok": ok,
                  "controls": results}, indent=1))
sys.exit(0 if ok else 1)
