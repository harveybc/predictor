#!/usr/bin/env python3
"""RP48 independent review: temporary copies, disposable HTTP stack, offline broker.

Run on reviewed code with trading-stack Python, CPU only. No production writes,
no scientific fitting, no service adoption. Child computation alone is stubbed
in the failure-path probe; campaign/delivery/outbox/reconciliation are real.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rl_probe(repo, out):
    t = load("rp48_broker_fixture", repo / "tests/test_e3_weekly_runtime.py")
    with tempfile.TemporaryDirectory() as tmp:
        frame = t._frame(70, slope=0.0, gap=0.0)
        # A temporary opening-price jump causes a real Margin rejection. Later
        # bars return to affordable prices; the long-only model keeps asking.
        frame.loc[1:5, ["OPEN", "HIGH", "LOW", "CLOSE"]] = 1000.0
        env, cfg = t._env(Path(tmp), frame)
        ctrl = t.C.WeeklyLongFlatController(
            [t._release(0, model=t.RT.ConstantModel(t.C.LONG))],
            size_fraction=0.8, latency_bars=2)
        doc = t.RT.run_weekly(env, ctrl, t._bars(frame), config=cfg,
                             bar_step=t.HOUR, max_steps=30)
        out.write_text(json.dumps({
            "submissions": doc["plugin"]["submissions"], "events": doc["plugin"]["events"],
            "fills": doc["fills"], "final": doc["final"],
            "decisions": [{k: r.get(k) for k in ("env_bar", "reason", "price", "units_before", "broker_terminal_status")}
                          for r in doc["records"]]}, indent=2))


def probe(repo):
    sys.path.insert(0, str(repo / "tools"))
    import df_e1_pilot as P
    import df_e1_close as C
    import df_e1_governed as G
    import df_public_lake_adopt as A
    import df_e1_innovation as I
    import governed_run as GR
    import numpy as np
    results = {"reviewed_code": "f83d63566309258b8c0aacddb6aba031c89ab7e0",
               "execution_checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    rehearsal = repo / "docs/audits/evidence/d3_k5_20260917/RP46/REHEARSAL.json"
    results["named_rehearsal"] = {"exists": rehearsal.is_file(),
        "binding": A._rehearsal_binding(rehearsal, Path("unused"), {})["accepted"]}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        freeze = json.loads((repo / "docs/audits/evidence/d3_k5_20260917/RP34/PILOT_FREEZE.json").read_text())
        source = Path(freeze["root"])
        results["historical_originals"] = {"checked": len(freeze["entries"]), "changed": [n for n, v in freeze["entries"].items()
            if not (source / n).is_file() or P.sha_file(source / n) != v["sha256"]]}
        copied = base / "historical-copy"
        copied.mkdir()
        for name in ("DESIGN.json", "DATA.json", "DATA.npz"):
            shutil.copy2(source / name, copied / name)
        attempt = copied / "attempts" / "pilot_ae"
        shutil.copytree(source / "attempts" / "pilot_ae", attempt)
        job = json.loads((attempt / "job.json").read_text())
        score = json.loads((attempt / "cell.json").read_text())
        accepted, why = P._closure_verdict(attempt, job, score)
        prior = (attempt / "outcome.json").read_bytes()
        (attempt / "outcome.json").unlink()
        fresh, problem = P._closure_verdict(attempt, job, score)
        (attempt / "outcome.json").write_bytes(prior)
        results["fresh_child_verification_order"] = {
            "with_outcome_accepted": accepted is not None, "with_outcome_refusal": why,
            "without_outcome_accepted": fresh is not None, "without_outcome_refusal": problem,
            "scope": "same conserved AE result, only parent outcome absent as on first completion; no fitting"}

        local = base / "local-claims"
        local.mkdir()
        design = "d" * 64
        delivery = {"campaign_sha256": "c" * 64, "campaign_key": "not-registered",
                    "delivery_id": "a" * 64, "sha256": "b" * 64,
                    "at": "2026-09-20T01:00:00Z", "host": "fixture", "code_identity": {"value": "e" * 40}}
        terminal = {"campaign_sha256": "c" * 64, "terminal_sha256": "f" * 64,
                    "accepted_at": "2026-09-20T02:00:00Z", "status": "NOT_A_REAL_STATUS",
                    "reconciliation": {"http": 200}}
        (local / "DELIVERIES.json").write_text(json.dumps({"design_sha256": design, "units": {"u": delivery}}))
        (local / "TERMINAL_RECEIPTS.json").write_text(json.dumps({"units": {"u": terminal}}))
        status, facts = C._governance(local, "u", {"design_sha256": design, "started_at": "2020-01-01T00:00:00Z"})
        results["local_claims_without_accounting"] = {"status": status, "problems": facts["problems"],
            "real_campaign_created": False, "terminal_payload_present": False, "actor_or_resource_present": False,
            "reconciliation_lists_present": False, "terminal_status": terminal["status"]}

        t = load("rp48_http_fixture", repo / "tests/test_df_e1_governed_route.py")

        class Factory:
            def mktemp(self, name):
                return Path(tempfile.mkdtemp(prefix=name, dir=base))

        stack_generator = t.stack.__wrapped__(Factory())
        stack = next(stack_generator)
        try:
            root = base / "runner"
            design = t._design(stack, root)
            failed = {"outcome": "RESOURCE_EXCEEDED", "reason": "audit child fixture", "score": None,
                      "cost": {"cpu_seconds": 0.0, "wall_seconds": 0.0}}
            with patch.object(P, "run_isolated", return_value=failed):
                report = t._run_governed(stack, root, design, run_id="rp48-review-failure", pilot_only=True)
            deliveries = json.loads((root / "DELIVERIES.json").read_text())["units"]
            token = t.KEY.read_text().strip()
            population = {}
            for uid, rec in deliveries.items():
                gov = GR.GovHttp(stack["url"], token, rec["campaign_key"])
                http, body = gov.reconcile_campaign(rec["campaign_sha256"])
                population[uid] = {"http": http, "missing_units": body.get("missing_units")}
            summary = P._governed_summary(root, report)
            results["runner_failure_population"] = {
                "registered": sorted(deliveries), "reported": [x["unit_id"] for x in report["terminals"]],
                "population_reconciliations": population, "summary": summary,
                "terminal_receipts_persisted": (root / "TERMINAL_RECEIPTS.json").exists(),
                "scope": "actual runner, disposable HTTP governance and DuckDB; costly child replaced by failed fixture"}
            results["empty_report_summary"] = P._governed_summary(root, {"terminals": []})
            results["accepted_unit_closer_status"] = C._governance(root, "pilot_ae", {"design_sha256": design["design_sha256"]})[0]
        finally:
            stack_generator.close()

        doc = json.loads((repo / "docs/audits/evidence/d3_k5_20260917/RP44/INNOVATION_RECOVERY_dragon.json").read_text())
        data = I.generate(
            **{k: doc["declared"][k] for k in ("realisations", "window", "distant_lag", "short_support", "channels", "amplitude", "noise", "seed")})
        j = int(len(data["y"]) * 0.7)
        amp, var = data["declared"]["amplitude"], data["declared"]["noise"] ** 2
        y = data["y"][j:]
        z = data["recovery_x"][j:, data["innovation_position"], 0]
        raw = I._score(amp * z, y)
        optimal = I._score(amp * z / (1 + var), y)
        results["noise_floor_check_without_fitting"] = {
            "published_copy_r2": doc["noise_floor"]["r2_of_the_known_solution"], "recomputed_copy": raw,
            "conditional_mean": optimal, "theoretical_bayes_mse": var + amp ** 2 * var / (1 + var),
            "theoretical_copy_mse": var * (1 + amp ** 2),
            "published_snr": doc["snr"], "target_power_snr": amp ** 2 / var,
            "scope": "known Gaussian generator; observed noisy-input Bayes rule, not latent-oracle access or a fit"}

        out = base / "rl.json"
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--repo", str(repo),
                                "--out", str(out), "--rl"], capture_output=True, text=True, timeout=90)
        results["broker_margin_probe"] = (json.loads(out.read_text()) if child.returncode == 0 else
            {"exit": child.returncode, "stderr_tail": child.stderr[-2000:]})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rl", action="store_true")
    a = ap.parse_args()
    if a.rl:
        rl_probe(a.repo.resolve(), a.out)
    else:
        a.out.write_text(json.dumps(probe(a.repo.resolve()), indent=2, default=str) + "\n")
