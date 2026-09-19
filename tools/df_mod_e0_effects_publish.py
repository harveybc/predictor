#!/usr/bin/env python3
"""RP18: publish a contrast table (effects v2) to governance and the warehouse by the normal route — a
NON_GOVERNING campaign whose units are the contrasts (one per architecture), each with one terminal
carrying the metric rows (value, replicate SD, n) and, in its tags, the state, the exact member list and
the design/closure identities. Nothing is rewritten: the earlier effects file stays where it was; this
is the successor table, reconciled through the outbox.

    python tools/df_mod_e0_effects_publish.py --effects EFFECTS.json --run-id RUN --api-key-file KEY [--gov-url URL] --out RECEIPT.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str, where: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load("df_utility_run")
GR = _load("governed_run")


def rows_for(a: str, p: dict, eff: dict) -> tuple:
    rows, states, members = [], {}, {}

    def put(name, est, unit="mase"):
        st = est.get("state") if isinstance(est, dict) else None
        states[name] = st or "NOT_ESTIMABLE"
        if st == "ESTIMATED" and est.get("value") is not None:
            rows.append({**R._metric(name, est["value"], unit), "split": eff["split"], "std_dev": est.get("sd_replicates"),
                         "min_value": min(est["per_replicate"].values()) if est.get("per_replicate") else None,
                         "max_value": max(est["per_replicate"].values()) if est.get("per_replicate") else None})
            rows.append({**R._metric(name + ".n_replicates", est.get("n_replicates") or est.get("n_observed") or 0, "count"), "split": eff["split"]})
        members[name] = {"n_expected": est.get("n_expected"), "n_observed": est.get("n_observed", est.get("n_replicates")), "missing": est.get("missing")}
    for r, d in p["COMMON"]["d"].items():
        put(f"mod_e0.effect.d_common_r{r}", d)
    put("mod_e0.effect.gamma_common_pair", p["COMMON"]["gamma_common_pair"])
    put("mod_e0.effect.gamma_factorial", p["FACT"]["gamma_factorial"])
    for r, f in p["FACT"].items():
        if r == "gamma_factorial":
            continue
        put(f"mod_e0.effect.fusion_2x2_r{r}", f["fusion"])
        put(f"mod_e0.effect.readout_2x2_r{r}", f["readout"])
        put(f"mod_e0.effect.interaction_2x2_r{r}", f["interaction"])
    for r, ro in p["READOUT"].items():
        put(f"mod_e0.effect.readout_sequence_fusion_r{r}", ro["sequence_fusion"])
        put(f"mod_e0.effect.readout_summary_fusion_r{r}", ro["summary_fusion"])
    for h, e in p["H2"]["e"].items():
        put(f"mod_e0.effect.h2_e_h{h}", e)
    if p["H2"]["slope"] is not None:
        rows.append({**R._metric("mod_e0.effect.h2_slope", p["H2"]["slope"], "mase_per_level"), "split": eff["split"]})
        states["mod_e0.effect.h2_slope"] = "ESTIMATED"
    else:
        states["mod_e0.effect.h2_slope"] = p["H2"]["state"]
    if p["DONOR"]:
        for r, d in p["DONOR"].items():
            put(f"mod_e0.effect.donor_delta_r{r}", d)
    return rows, states, members


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="the run whose closure the effects come from; the campaign key is <run-id>-effects-v2")
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--outbox-dir", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suffix", default="-effects-v2", help="campaign key suffix (e.g. -effects-v3-composed for the RP26 composition)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"REFUSED: {args.out} exists")
    eff = json.loads(args.effects.read_text())
    if eff.get("schema") != "df_mod_e0_arch_effects.v2":
        raise SystemExit("REFUSED: not an effects v2 document")
    code_identity = GR.strict_code_identity(REPO)
    gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), args.run_id)
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(args.outbox_dir or GR.DEFAULT_OUTBOX)).resolve())
    key = f"{args.run_id}{args.suffix}"
    design_sha = eff["design_sha256"]
    merged = None
    if str(design_sha).startswith("MERGED:"):
        merged = eff.get("sources") or {}
        design_sha = (merged.get("stage") or {}).get("design") or eff["design_sha256"]
    units = [f"effects__{a}" for a in eff["per_arch"]]
    eff_sha = hashlib.sha256(args.effects.read_bytes()).hexdigest()
    status, reg = gov.submit_campaign({"schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING", "project": "predictor",
                                       "code_identity": code_identity, "config_sha256": design_sha, "input_mode": "SYNTHETIC",
                                       "synthetic_spec_sha256": design_sha, "units": units, "datasets": [], "terminal_lake": "olap_cube"})
    if status not in (200, 201):
        raise SystemExit(f"REFUSED: campaign {key} refused: http {status} {reg}")
    sha = reg["campaign_sha256"]
    receipt = {"schema": "df_mod_e0_effects_publish.v1", "campaign_key": key, "campaign_sha256": sha, "http": status, "effects_sha256": eff_sha, "units": {}}
    for a in eff["per_arch"]:
        unit = f"effects__{a}"
        rows, states, members = rows_for(a, eff["per_arch"][a], eff)
        now = R.now_iso()
        terminal = R._terminal(status="COMPLETED", reason=None, cost={"cpu_seconds": 0.0, "wall_seconds": 0.0}, metrics=rows, started=now, finished=now,
                               tags={"purpose": "MOD_E0_ARCH_EFFECTS_V2", "proposal": "P-MOD", "grants": "NONE", "classification": "NON_GOVERNING",
                                     "phase": "DEVELOPMENT", "arch": a, "design_sha256": design_sha, "composed_design": str(eff["design_sha256"]), "effects_sha256": eff_sha,
                                     **({"composition_validated": json.dumps(eff.get("validated"), default=str)[:1500], "contradictions": json.dumps(eff.get("contradictions"), default=str)[:800]} if merged is not None else {}),
                                     "closure": str(eff.get("closure")), "split": eff["split"], "replicates": json.dumps(eff["replicates"]),
                                     "effect_states": json.dumps(states, sort_keys=True), "members": json.dumps(members, sort_keys=True, default=str)[:4000],
                                     "supersedes": "RP14_ARCH_STAGE_EFFECTS.json (v1, withdrawn reading; see SATOSHI_RP16_ERRATA_2026_09_19.md)",
                                     "output_sha256": eff_sha})
        GR._require_reconciled(gov, sha, unit, before_run=True)
        outbox.put({"campaign_sha256": sha, "unit_id": unit, "terminal": terminal})
        flushed = GR._send_pending(gov, outbox)
        receipt["units"][unit] = {"rows": len(rows), "states": states, "pending_after_flush": flushed["pending"]}
    rs, rb = gov.reconcile_campaign(sha)
    receipt["reconciliation"] = {"http": rs, **{k: rb.get(k) for k in ("missing_units", "accounting_only", "lake_only")}}
    args.out.write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "units"} | {"units": {u: v["rows"] for u, v in receipt["units"].items()}}, indent=1, default=str))
    return 0 if not rb.get("missing_units") else 1


if __name__ == "__main__":
    raise SystemExit(main())
