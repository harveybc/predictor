#!/usr/bin/env python3
"""RP13: backfill of the metrics contract for an EXECUTED MOD-E0 campaign, from what its files
allow and nothing else, published as governed successor terminals (next generation) through the
outbox. No training, no attempt is modified.

Derivable from the files: every data/target metric (the generator and the contract reproduce the
series exactly: the closure verified it), and the model descriptors of the SAVED checkpoint (the
restored best): parameters, bytes, norms, singular-value descriptors, graph, plus gradients and
activations on the same fixed batches the instrumented runs use (a function of the saved weights
and the regenerated inputs). NOT derivable: the initial state, the scheduled epochs and the final
(pre-restore) epoch — no checkpoint was saved — and any gradient/activation trajectory: those are
NO_MEDIDO with the reason, never reconstructed.

    python tools/df_mod_e0_backfill.py --root RUN_ROOT --out-dir DIR [--publish --api-key-file KEY --gov-url URL]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

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


E = _load("df_mod_e0")
M = _load("df_mod_e0_metrics")
RUN = _load("df_mod_e0_run")
R = _load("df_utility_run")

SCHEMA = "df_mod_e0_backfill.v1"
BACKFILL_VERSION = "v2_grains_snr_total"      # RP20: a successor generation per backfill version; a unit is skipped only if it already carries THIS version
NOT_DERIVABLE = {"state": M.NO_MEDIDO, "reason": "no checkpoint was saved at this point of the executed run; not reconstructible from the files"}


def backfill_attempt(attempt: Path) -> dict:
    """The derivable metrics of one attempt; the record is read, never written."""
    t0 = time.process_time()
    rec = json.loads((attempt / "cell.json").read_bytes())
    gen = E.generate(int(rec["level"]), int(rec["r"]), int(rec["seed"]))
    periods = [gen["params"]["groups"][g]["period"] for g in gen["params"]["latent_groups"]]
    pilot = rec.get("exposure") == "NO_TEST_ACCESS"
    prep = E.prepare(gen["x"], gen["oracle"], periods, int(rec["window"]), int(rec["horizon"]), test_access=not pilot)
    P = prep["parts"]
    s = rec.get("scale") or prep["scale"]
    dv = int((rec.get("profiles") or {}).get("descriptor_version") or rec.get("descriptor_version") or 1)
    data = M.data_metrics(gen, prep, list(P), dv)
    model = E.build_modular(list(rec["assignment"]), int(rec["window"]), gen["x"].shape[1], fusion=rec["fusion"], seed=int(rec["seed"]), arch=str(rec.get("arch") or E.DEFAULT_ARCH))
    model.load_weights(str(attempt / "weights.weights.h5"))
    if rec["hypothesis"] == "H3" and rec["arm"] in ("sequence", "summary"):
        E.freeze_extractor(model)                                 # the arm trained with a frozen extractor: counts reflect it
    batch = int((rec.get("training") or {}).get("rule", {}).get("batch") or E.TRAINING["batch"])
    loss = (rec.get("training") or {}).get("rule", {}).get("loss", E.TRAINING["loss"])
    Xtr, ytr = E._sx(P["train"]["X"], s), E._sy(P["train"]["y"], s)
    Xva = E._sx(P["validation"]["X"], s)
    best = M.model_metrics(model, grad_batch=(Xtr[:batch], ytr[:batch]), act_batch=Xva[:32], loss=loss)
    checkpoints = {"initial": dict(NOT_DERIVABLE), "final": dict(NOT_DERIVABLE), "best": best}
    for e in M.CHECKPOINT_EPOCHS:
        if e <= int(rec["training"]["epochs"]):
            checkpoints[f"epoch_{e}"] = dict(NOT_DERIVABLE)
    doc = {"schema": SCHEMA, "cell_id": rec["cell_id"], "attempt": attempt.name, "cell_sha256": hashlib.sha256((attempt / "cell.json").read_bytes()).hexdigest(),
           "weights_sha256": hashlib.sha256((attempt / "weights.weights.h5").read_bytes()).hexdigest(),
           "data_metrics": data, "model_metrics": {"schema": M.SCHEMA, "schedule_epochs": list(M.CHECKPOINT_EPOCHS), "checkpoints": checkpoints,
                                                    "fixed_batches": {"gradients": f"first {batch} training windows", "activations": "first 32 validation windows"},
                                                    "note": "best = the saved (restored) checkpoint; other checkpoints NO_MEDIDO: not saved by the executed run"},
           "derivable": {"data": "ALL (generator + contract reproduce the consumed series)", "model_best": "ALL descriptors, gradients and activations on the fixed batches",
                         "model_trajectory": "NONE (initial / scheduled epochs / final not saved)"},
           "cpu_seconds": round(time.process_time() - t0, 3)}
    rows, states = M.terminal_rows({"data_metrics": data, "model_metrics": doc["model_metrics"], "parameters": rec.get("parameters")})
    doc["rows"] = [{**R._metric(n, v, u), "split": sp} for n, v, u, sp in rows]
    doc["states"] = states
    return doc


def find_sent_envelope(outbox_root: Path, campaign_sha256: str, unit_id: str) -> tuple:
    """The highest-generation envelope already SENT for (campaign, unit)."""
    best = None
    for path in sorted((outbox_root / "sent").glob("*.json")):
        try:
            env = json.loads(path.read_text(encoding="ascii"))
        except (ValueError, UnicodeDecodeError):
            continue
        if env.get("campaign_sha256") == campaign_sha256 and env.get("unit_id") == unit_id:
            gen = int((env.get("terminal") or {}).get("generation", 1))
            if best is None or gen > best[0]:
                best = (gen, path, env)
    return best


def successor_terminal(base: dict, doc: dict, reason: str, envelope_path: Path, generation: int) -> dict:
    """The same outcome, instants, costs, deliveries, artifacts and tags; the metric rows extended
    with the backfilled contract rows; the states extended; generation + 1."""
    known = {(m["metric"], m["split"]) for m in base.get("metrics") or []}
    extra = [m for m in doc["rows"] if (m["metric"], m["split"]) not in known]
    try:
        states = json.loads((base.get("tags") or {}).get("metric_states") or "{}")
    except ValueError:
        states = {}
    states.update(doc["states"])
    successor = {k: v for k, v in base.items() if k != "generation"}
    successor["metrics"] = list(base.get("metrics") or []) + extra
    successor["generation"] = generation
    successor["tags"] = {**(base.get("tags") or {}), "metric_states": json.dumps(states, sort_keys=True), "backfill": reason,
                         "backfill_schema": SCHEMA, "backfill_version": BACKFILL_VERSION, "backfill_cell_sha256": doc["cell_sha256"],
                         "supersedes_envelope_sha256": hashlib.sha256(envelope_path.read_bytes()).hexdigest(),
                         "supersedes_generation": str(generation - 1)}
    return successor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="send the successor terminals through the outbox (governance writes only)")
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--outbox-dir", default=None)
    parser.add_argument("--reason", default="RP13 backfill: contract D/Y/M/G rows derived from the retained arrays and saved checkpoint; trajectory NO_MEDIDO")
    args = parser.parse_args(argv)
    GR = _load("governed_run")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out_dir / "BACKFILL.json"
    if receipt_path.exists():
        raise SystemExit(f"REFUSED: {receipt_path} exists")
    registrations = json.loads((args.root / "CAMPAIGNS.json").read_text())
    docs = {}
    for attempt in sorted(p for p in (args.root / "attempts").iterdir() if (p / "cell.json").is_file()):
        target = args.out_dir / f"{attempt.name}.json"
        if target.is_file():
            docs[attempt.name] = json.loads(target.read_text())
            continue
        doc = backfill_attempt(attempt)
        target.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float) + "\n")
        docs[attempt.name] = doc
        print(json.dumps({"backfilled": attempt.name, "rows": len(doc["rows"]), "cpu": doc["cpu_seconds"]}), flush=True)
    receipt = {"schema": SCHEMA, "root": str(args.root), "units": len(docs), "rows_total": sum(len(d["rows"]) for d in docs.values()),
               "cpu_seconds": round(sum(d["cpu_seconds"] for d in docs.values()), 3), "published": None}
    if args.publish:
        if not args.api_key_file:
            raise SystemExit("REFUSED: --api-key-file is required to publish")
        run_id = next(iter(registrations)).rsplit("-mod-e0-", 1)[0]
        gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), run_id)
        outbox_root = Path(os.path.expanduser(args.outbox_dir or GR.DEFAULT_OUTBOX)).resolve()
        outbox = GR.TerminalOutbox(outbox_root)
        units_by_sha = {}
        for key, reg in registrations.items():
            units_by_sha[reg["campaign_sha256"]] = key
        published, failed, skipped = [], [], []
        for unit_id, doc in docs.items():
            found = None
            for sha_c in units_by_sha:
                found = find_sent_envelope(outbox_root, sha_c, unit_id)
                if found:
                    break
            if not found:
                skipped.append({"unit_id": unit_id, "why": "no sent envelope in the outbox"})
                continue
            gen, path, env = found
            if (env["terminal"].get("tags") or {}).get("backfill_version") == BACKFILL_VERSION:
                skipped.append({"unit_id": unit_id, "why": f"generation {gen} already carries backfill {BACKFILL_VERSION}"})
                continue
            succ = successor_terminal(env["terminal"], doc, args.reason, path, gen + 1)
            outbox.put({"campaign_sha256": env["campaign_sha256"], "unit_id": unit_id, "terminal": succ})
            try:
                flushed = GR._send_pending(gov, outbox)
                published.append({"unit_id": unit_id, "generation": gen + 1, "rows": len(succ["metrics"]), "pending_after": flushed["pending"]})
            except Exception as e:  # noqa: BLE001
                failed.append({"unit_id": unit_id, "error": str(e)[:200]})
        recon = {k: gov.reconcile_campaign(v["campaign_sha256"])[1] for k, v in registrations.items()}
        receipt["published"] = {"units": len(published), "failed": failed, "skipped": skipped, "details": published,
                                "reconciliation": {k: {"missing": len(v.get("missing_units") or []), "accounting_only": v.get("accounting_only"), "lake_only": v.get("lake_only")}
                                                   for k, v in recon.items()}}
    receipt_path.write_text(json.dumps(receipt, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "published"} | {"published": None if receipt["published"] is None else
                     {k: v for k, v in receipt["published"].items() if k != "details"}}, indent=1, default=str))
    return 0 if not (receipt["published"] or {}).get("failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
