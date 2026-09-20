#!/usr/bin/env python3
"""Independent RP40 review. Temporary fixtures; no service writes or training.

Run with trading-stack Python, --repo pointing to the reviewed checkout and
--out pointing to the review results. The optional RL child uses the real
offline gym-fx environment and broker, never a venue.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rl_probe(repo, out):
    t = load("rp40_runtime_fixture", repo / "tests/test_e3_weekly_runtime.py")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for fraction, latency in [(0.1, 1), (0.8, 1), (0.1, 3)]:
            root = Path(tmp) / f"{fraction}-{latency}"
            root.mkdir()
            frame = t._frame(120, slope=0.0, gap=0.0)
            env, cfg = t._env(root, frame, size_fraction=0.5)
            ctrl = t.C.WeeklyLongFlatController(
                [t._release(0, model=t.RT.ConstantModel(t.C.LONG))],
                size_fraction=fraction, latency_bars=latency)
            doc = t.RT.run_weekly(env, ctrl,
                frame.assign(DATE_TIME=t._times(frame)).set_index("DATE_TIME"),
                bar_step=t.HOUR, max_steps=20)
            first = next(r for r in doc["records"] if r["reason"] == "OPEN_LONG")
            fill = doc["fills"][0]
            rows.append({"controller_fraction": fraction, "latency_bars": latency,
                         "decision": first, "first_fill": fill,
                         "broker_config_position_size": cfg["position_size"],
                         "actual_units": doc["final"]["units"]})
    out.write_text(json.dumps(rows, indent=2))


def probe(repo):
    sys.path.insert(0, str(repo / "tools"))
    import df_e1_pilot as P
    import df_e1_close as C
    import df_e1_governed as G
    import df_public_lake_adopt as A
    import df_e1_receiver as R
    import governed_run as GR
    import numpy as np

    results = {"reviewed_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "scope": "temporary orchestration fixtures plus real offline RL broker; no scientific training"}
    frozen = json.loads((repo / "docs/audits/evidence/d3_k5_20260917/RP34/PILOT_FREEZE.json").read_text())
    differences = []
    for name, expected in frozen["entries"].items():
        file = Path(frozen["root"]) / name
        if not file.is_file() or P.sha_file(file) != expected["sha256"] or file.stat().st_size != expected["bytes"]:
            differences.append(name)
    results["historical_originals"] = {"files_checked": len(frozen["entries"]),
        "differences": differences, "scope": "current hashes and sizes vs RP34 frozen manifest; not a fresh inference replay"}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "empty-receipts"
        root.mkdir()
        before = C._governance(root, "never-registered", {})[0]
        for name in ("DELIVERIES.json", "TERMINAL_RECEIPTS.json"):
            (root / name).touch()
        after, facts = C._governance(root, "never-registered", {})
        results["empty_files_governance"] = {"before": before, "after": after, "facts": facts}

        design = json.loads((repo / "docs/audits/evidence/d3_k5_20260917/RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json").read_text())
        root = base / "cached-prepare"
        root.mkdir()
        # This branch never parses DATA.npz: no raw panel is needed to expose it.
        (root / "DATA.npz").write_bytes(b"audit cached payload, not a scientific tensor")
        (root / "DATA.json").write_text(json.dumps({"design_sha256": design["design_sha256"],
            "data_sha256": P.sha_file(root / "DATA.npz")}))
        failed = {"outcome": "RESOURCE_EXCEEDED", "reason": "isolated audit child fixture",
                  "cost": {"cpu_seconds": 0.0, "wall_seconds": 0.0}, "score": None}
        with patch.object(G, "require_delivery", side_effect=RuntimeError("no delivery")) as guard, \
             patch.object(G, "acquire", side_effect=RuntimeError("no campaign")) as acquire, \
             patch.object(G, "report_terminal", side_effect=RuntimeError("no transport")) as report, \
             patch.object(GR, "strict_code_identity", return_value={"audit_fixture": True}), \
             patch.object(P, "run_isolated", return_value=failed) as child:
            prepared = P.prepare(design, root)
            doc = P.run(design, root=root, run_id="audit-only", cap_seconds=14400,
                        already_spent=0, trace=lambda *a, **kw: None)
            results["cached_runner_without_governance"] = {
                "prepared": prepared is not None, "children_dispatched": child.call_count,
                "delivery_checks": guard.call_count, "registrations": acquire.call_count,
                "terminal_reports": report.call_count, "local_terminals": len(list((root / "TERMINALS").glob("*.json"))),
                "stopped": doc["stopped"], "child_was_stubbed": True}

        payload = base / "panel.fixture"
        payload.write_bytes(b"synthetic route fixture")
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        calls = []

        class RouteClient:
            def __init__(self, *a):
                self.ranged = a[-1].endswith("-ranged")

            def submit_campaign(self, campaign):
                calls.append("register-ranged" if self.ranged else "register")
                return (403, {}) if self.ranged else (201, {"campaign_sha256": "a" * 64})

            def governed_download(self, *a):
                calls.append("download")
                return 200, {"path": str(payload), "sha256": digest,
                             "availability_use": "UNDECLARED", "availability_label": "UNKNOWN"}

            def __getattr__(self, name):
                raise AssertionError(f"Unexpected route call {name}")

        with patch.object(GR, "GovHttp", RouteClient), \
             patch.object(GR, "strict_code_identity", return_value={"audit_fixture": True}), \
             patch.object(A, "http_json", return_value=(403, {"reason": "fixture refusal"}, {})):
            route = A.route_checks("http://unused", "fixture", cache_dir=base,
                run_id="audit", resource="fixture/panel", expect_sha=digest)
        results["route_download_only"] = {"calls": calls,
            "delivered_bytes_verified": route["delivered_bytes_verified"],
            "refusals_hold": route["refusals_hold"], "terminal_or_reconcile_called": False,
            "transport_was_stubbed": True}

        config, key = base / "runtime.json", base / "fixture.key"
        original = {"lakes": [], "policies": [], "principals": {}}
        config.write_text(json.dumps(original))
        key.write_text("NOT_A_REAL_CREDENTIAL")
        proposed = {**original, "lakes": [{"lake_id": A.LAKE_ID}]}
        state = base / "adopt"
        with patch.object(A, "RUNTIME_CONFIG", config), patch.object(A, "API_KEY_FILE", key), \
             patch.object(A, "inventory", return_value={}), \
             patch.object(A, "successor_config", return_value=proposed), \
             patch.object(A, "lake_entry", return_value={"declared": {}}), \
             patch.object(A, "run", side_effect=subprocess.TimeoutExpired("fixture restart", 180)):
            try:
                A.adopt(state, principals=[])
            except subprocess.TimeoutExpired:
                pass
        results["adoption_restart_exception"] = {
            "config_restored": json.loads(config.read_text()) == original,
            "receipt_written": (state / "RECEIPT.json").exists(),
            "backup_preserved": (state / "5055.runtime.backup.json").exists(),
            "production_touched": False}
        with patch.object(A, "adopt", return_value={"adopted": False, "rolled_back": True}), \
             contextlib.redirect_stdout(io.StringIO()):
            code = A.main(["adopt", "--state-dir", str(base / "unused")])
        results["failed_adoption_exit_code"] = code

        # Each window has a different level, but all rows of a window equal it.
        X = np.repeat(np.arange(10, dtype=float), 60)[:, None]
        tasks, x = R._tasks(X, X, np.arange(59, 600, 60), 60, 1, 0, distant_lag=50, seed=1)
        results["distant_lag_counterexample"] = {
            "declared_solvable_by_seven": tasks["distant_lag"]["solvable_by_seven_sample_receiver"],
            "last_sample_copy_r2": R._score(x[:, -1, 0], tasks["distant_lag"]["y"])["r2"],
            "scope": "counterexample to universal diagnostic label, not a household result"}
        results["diagnostic_updates_arithmetic"] = [{"requested": u, "batches_per_epoch": 63,
            "actual_if_fit_completes": math.ceil(u / 63) * 63} for u in (400, 1600)]

        rl_out = base / "rl.json"
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--repo", str(repo),
                            "--out", str(rl_out), "--rl"], capture_output=True, text=True, timeout=180)
        if p.returncode:
            results["rl_probe_error"] = {"exit": p.returncode, "stderr_tail": p.stderr[-2000:]}
        else:
            results["real_broker_size_latency"] = json.loads(rl_out.read_text())
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rl", action="store_true")
    args = ap.parse_args()
    if args.rl:
        rl_probe(args.repo.resolve(), args.out)
    else:
        args.out.write_text(json.dumps(probe(args.repo.resolve()), indent=2, default=str) + "\n")
