"""Independent reviewer control RP01: exact DS2 alignment (positive + fail-closed mutations).

Bounded synthetic fixtures only; no fit, no GPU, no TEST rows, no services.
Exit 0 iff every control behaves as the sealed design requires.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tests.i6dmo import _synth  # noqa: E402
from tools.i6dmo import data  # noqa: E402

results = {}


def check(name, ok, detail):
    results[name] = {"ok": bool(ok), "detail": detail}


design = _synth.load_design()

# --- control 1 (positive): every selected slot value is the EXACT bar, table-wide ---
missing_bar = _synth.epoch("2022-06-15T13:00:00+00:00")
nonfinite = (_synth.epoch("2021-09-08T09:00:00+00:00"), 5)
src = _synth.make_input(Path(sys.argv[1]) / "input", missing={missing_bar},
                        nonfinite=[nonfinite])
bundle = data.load_input(src, design)
table = data.build_origin_table(bundle, design)
sel = table.slot_index >= 0
ts_sel = bundle.ts[np.where(sel, table.slot_index, 0)]
instants = np.broadcast_to(table.slot_instants[:, :, None], table.slot_index.shape)
stale = int(np.count_nonzero(sel & (ts_sel != instants)))
check("exact_everywhere", stale == 0,
      f"{stale} selected slot values whose source bar is not the exact slot instant "
      f"(over {int(sel.sum())} selected values)")

# --- control 2: a missing exact slot is a COUNTED exclusion, never filled ---
rec = table.record(missing_bar + 2 * 3600)  # origin whose t-2h slot is the dropped bar
counted = sum(table.counts[s]["origins"]["INPUT_SLOT_NOT_OBSERVED"] for s in table.counts)
check("missing_slot_counted", (not rec.input_ok) and rec.input_reason == "INPUT_SLOT_NOT_OBSERVED"
      and counted > 0,
      f"origin input_ok={rec.input_ok} reason={rec.input_reason}; "
      f"INPUT_SLOT_NOT_OBSERVED counted={counted}")

# --- control 3: a non-finite exact value is excluded, not bridged by an older value ---
nf_origin = nonfinite[0]  # the NaN bar itself as current-slot origin
rec_nf = table.record(nf_origin)
check("nonfinite_excluded", (not rec_nf.input_ok)
      and rec_nf.input_reason == "INPUT_VALUE_NOT_FINITE",
      f"input_ok={rec_nf.input_ok} reason={rec_nf.input_reason}")

# --- control 4 (reviewer mutation): the ASOF arm WOULD select a stale bar here; the
# default must not. Demonstrates the two treatments differ on this exact fixture. ---
offsets = [o * 3600 for o in bundle.window.offsets_hours()]
origins = np.asarray([missing_bar + 2 * 3600], dtype=np.int64)
exact_idx, _ = data.slot_index(bundle.ts, bundle.x, origins, offsets)
asof_idx, _ = data.asof_arm_slot_index(bundle.ts, bundle.x, origins, offsets)
slot = bundle.window.offsets_hours().index(-2)
check("default_is_exact_not_asof",
      (exact_idx[0, slot] == -1).all() and (asof_idx[0, slot] >= 0).all()
      and data.asof_slot_index is data.slot_index,
      "missing slot: exact index all -1, ASOF arm selects an older bar; "
      "asof_slot_index alias bound to the exact indexer")

# --- control 5: sampling rule authenticated at load (stale/ASOF manifest REJECTED) ---
try:
    data.authenticate_sampling({"input_sampling": {"rule": "DS2_ASOF_LAST_AVAILABLE"}}, design)
    check("asof_manifest_rejected", False, "ASOF-rule manifest was ACCEPTED for the exact design")
except data.DataContractError as exc:
    check("asof_manifest_rejected", exc.code == "SAMPLING_RULE_MISMATCH", f"typed {exc.code}")
ok_rule = data.authenticate_sampling({"input_sampling": {"rule": "DS2_EXACT"}}, design)
check("exact_manifest_accepted", ok_rule == "DS2_EXACT", f"rule={ok_rule}")

# --- control 6 (reviewer mutation): a tampered rule string fails typed, not silently ---
try:
    data.authenticate_sampling({"input_sampling": {"rule": "DS2_EXACTX"}}, design)
    check("tampered_rule_rejected", False, "tampered rule accepted")
except data.DataContractError as exc:
    check("tampered_rule_rejected", exc.code == "SAMPLING_RULE_MISMATCH", f"typed {exc.code}")

ok = all(v["ok"] for v in results.values())
print(json.dumps({"probe": "RP01_alignment", "ok": ok, "controls": results}, indent=1))
sys.exit(0 if ok else 1)
