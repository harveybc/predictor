#!/usr/bin/env python3
"""Read-only audit of the f57f8ef shared recipe; mutations stay in a temp directory."""
import argparse
import copy
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("hs", "predictor-source", "pool-root", "admitted", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    import numpy as np
    import pandas as pd

    clean = a.hs / "docs/short_horizon/clean_round"
    t3 = clean / "evidence/t3"
    pool = a.pool_root / "eurusd_h1_full_pool_s"
    recipe = read(clean / "CONSUMER_RECIPE.json")
    sources = {
        "spec": clean / "CLEAN_ROUND_SPEC.json",
        "scaler": t3 / "results/t3_unit_C/SCALER.json",
        "groups": t3 / "groups/unit_C.json",
        "list": pool / "selection_v4/FEATURE_LIST_shortlist_strict.json",
        "manifest": t3 / "T3_MANIFEST.json",
        "decision": t3 / "T3_DECISION.json",
        "combination": t3 / "musashi_delivery/COMBINATION_DECISION.json",
        "c-result": t3 / "results/t3_unit_C/E3M.json",
    }
    sc, result = read(sources["scaler"]), read(sources["c-result"])
    groups = read(sources["groups"])
    checks = {}
    checks["recipe_matches_sealed_scaler"] = all(
        recipe["scaler"][k] == sc[k]
        for k in ("mean", "std", "fit_period", "fit_rows", "policy", "stationarity")
    ) and recipe["scaler"]["artifact_sha256"] == sha(sources["scaler"])
    checks["recipe_columns_and_window_match"] = (
        recipe["tensor"]["order"] == sc["features"]
        and recipe["tensor"]["window"] == sc["window"]
    )
    names = list(groups["groups"])
    columns = recipe["list"]["columns"]
    checks["actual_recipe_groups_match_C_architecture"] = (
        names == result["arch"]["group_names"]
        and [[columns.index(c) for c in groups["groups"][g]] for g in names] == result["arch"]["groups"]
        and [groups["kinds"].get(g, "conv") for g in names] == result["arch"]["group_kinds"]
        and [groups["units"].get(g, 2) for g in names] == result["arch"]["group_units"]
    )
    checks["admitted_and_calendar_digests"] = all(
        sha(a.admitted / n) == result["meta"]["sources"]["sha256"][n]
        for n in ("eurusd.parquet", "CLOSURE_INTERVALS.json")
    )
    checks["actual_store_and_manifest_digests"] = (
        sha(pool / "full_pool_eurusd_h1.parquet") == recipe["store"]["sha256"]
        and sha(pool / "MANIFEST.json") == result["meta"]["sources"]["sha256"]["MANIFEST.json"]
    )
    pm = read(pool / "MANIFEST.json")
    checks["merged_parent_files_match"] = all(
        sha(a.pool_root / folder / meta["file"]) == meta["sha256"]
        for folder, meta in pm["inputs"].items()
    )
    checks["selection_ledger_digest"] = sha(pool / "selection_v4/LEDGER.jsonl") == recipe["identities"]["selection_ledger_sha256"]
    # All original results are checked against the independently retained decision join.
    join = read(t3 / "musashi_delivery/READY_FOR_T3_DECISION_AUDIT.json")
    checks["all43_results_match_retained_join"] = len(join["units"]) == 43 and all(
        sha(t3 / "results" / unit / "E3M.json") == meta["result_sha256"]
        for unit, meta in join["units"].items()
    )
    ledger = [json.loads(line) for line in (pool / "selection_v4/LEDGER.jsonl").read_text().splitlines()]
    pairing = {}
    signature = ("support_hash", "val_hash", "rows_train", "rows_val", "seed", "k_folds")
    for stage, expected in (("contrast", 714), ("encoding", 72), ("rescue", 120)):
        paired = defaultdict(list)
        for row in ledger:
            if row["stage"] == stage and row["family"] != "NONE":
                paired[tuple(row.get(k) for k in ("task", "model", "fold", "family", "ctx"))].append(row)
        mismatches = sum(len({tuple(row[k] for k in signature) for row in rows}) != 1 for rows in paired.values())
        pairing[stage] = {"groups": len(paired), "mismatches": mismatches,
                          "arm_counts": dict(Counter(len(rows) for rows in paired.values()))}
        checks[stage + "_support_paired"] = len(paired) == expected and mismatches == 0
    checks["ledger_unique_and_error_free"] = len(ledger) == len({r["key"] for r in ledger}) == 3882 and not any(r.get("error") for r in ledger)
    report = {"schema": "musashi_shared_recipe_audit.v1", "checks": checks,
              "paired_support": pairing,
              "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
              "inputs_sha256": {k: sha(p) for k, p in sources.items()},
              "recipe_sha256": sha(clean / "CONSUMER_RECIPE.json"),
              "science_commit": recipe["identities"]["science_commit"],
              "request_scope": "recipe, existing decisions, upstream fit boundaries and consumer integration",
              "new_policy_fits": 0, "test_scored": False, "original_artifacts_changed": False}
    with tempfile.TemporaryDirectory(prefix="recipe-audit-") as td:
        tmp = Path(td)
        base = {k: read(p) for k, p in sources.items()}

        def emit(case, mutate=None):
            d = copy.deepcopy(base)
            if mutate:
                mutate(d)
            root = tmp / case
            root.mkdir()
            args = []
            for k, value in d.items():
                p = root / sources[k].name
                # Preserve baseline bytes so honest derivation has exactly the original identity.
                p.write_bytes(sources[k].read_bytes() if value == base[k] else json.dumps(value).encode())
                args += ["--" + k, str(p)]
            out = root / "RECIPE.json"
            r = subprocess.run([sys.executable, str(a.hs / "tools/expert_guided_consumer_recipe.py"),
                                "emit", *args, "--out", str(out), "--science-commit", report["science_commit"]],
                               capture_output=True, text=True, timeout=30)
            rec = read(out) if out.exists() else None
            return {"exit_code": r.returncode, "accepted": r.returncode == 0,
                    "recipe_identity": rec.get("identity") if rec else None,
                    "same_as_delivered": sha(out) == report["recipe_sha256"] if rec else False,
                    "error": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else None}

        tests = {"honest_baseline": emit("honest_baseline")}
        mutations = {
            "scaler_mean_changed": lambda d: d["scaler"]["mean"].__setitem__(0, d["scaler"]["mean"][0] + 10),
            "scaler_policy_rolling": lambda d: d["scaler"].__setitem__("policy", "rolling_z"),
            "group_kind_changed": lambda d: d["groups"]["kinds"].__setitem__("gmm/logret_1+rv_24+log_range", "conv"),
            "model_depth_changed": lambda d: d["spec"]["model"].__setitem__("depth", 9),
            "store_digest_changed": lambda d: d["spec"]["store"].__setitem__("sha256", "0" * 64),
            "manifest_identity_changed": lambda d: d["manifest"].__setitem__("identity", "other-manifest"),
            "replay_failed": lambda d: d["combination"].update(state="REPLAY_CORRECTION_REQUIRED", engineering_verified=False, checkpoint_probe_replay="FAIL", full_trading_metrics_exact=False),
            "negative_control_bad_result": lambda d: d["c-result"].__setitem__("device", "changed"),
            "negative_control_bad_column": lambda d: d["scaler"]["features"].__setitem__(0, "NOT_A_FEATURE"),
        }
        for name, mut in mutations.items():
            tests[name] = emit(name, mut)
        report["emitter_mutations"] = tests
        assert tests["honest_baseline"]["same_as_delivered"], tests["honest_baseline"]
        assert not tests["negative_control_bad_result"]["accepted"]
        assert not tests["negative_control_bad_column"]["accepted"]

        # Re-run the native decision, without touching the original decision or group file.
        out = tmp / "DECISION.json"
        r = subprocess.run([sys.executable, str(a.hs / "tools/expert_guided_t3_manifest.py"),
                            "decide", "--manifest", str(sources["manifest"]), "--results", str(t3 / "results"),
                            "--out", str(out)], capture_output=True, text=True, timeout=60)
        checks["per_group_decision_reproduced"] = r.returncode == 0 and sha(out) == sha(sources["decision"])

    # The selected store columns are already transformed. No additional differencing happened in C.
    report["transform_boundary"] = {
        "already_transformed_store_columns": [c for c in columns if "@" in c],
        "additional_consumer_differences": sc["stationarity"]["differenced"],
        "recipe_omits_stage_for_suffix_operations": True,
    }
    frame = pd.read_parquet(pool / "full_pool_eurusd_h1.parquet", columns=["timestamp", *columns])
    assert (pd.to_datetime(frame.timestamp, utc=True) < pd.Timestamp("2025-01-01", tz="UTC")).all()
    ts = pd.to_datetime(frame.timestamp, utc=True)
    report["observed_row_window"] = {
        "adjacent_gaps_gt_one_hour": int((ts.diff() > pd.Timedelta(hours=1)).sum()),
        "largest_adjacent_gap_hours": float(ts.diff().dt.total_seconds().max() / 3600),
        "physical_span_hours_of_13_stride2_min": float((ts - ts.shift(24)).dt.total_seconds().dropna().min() / 3600),
        "physical_span_hours_of_13_stride2_max": float((ts - ts.shift(24)).dt.total_seconds().dropna().max() / 3600),
        "semantics": "consecutive observed rows; closures consume no tensor slots, elapsed time can exceed 24h",
    }
    consumer = load_module("audited_clean_round", a.hs / "tools/expert_guided_clean_round.py")
    spec = read(sources["spec"])
    arm = next(x for x in spec["stages"]["A"] if x["name"] == "strict_fam_w13")
    cmd = consumer.e3m_cmd(spec, arm, "<output>", "<package>")
    planned_groups = Path(arm["groups"])
    report["next_queue_consumer"] = {
        "command": cmd,
        "uses_recipe_argument": any("recipe" in x.lower() for x in cmd),
        "planned_group_file_sha256": sha(planned_groups) if planned_groups.exists() else None,
        "selected_C_groups_sha256": sha(sources["groups"]),
        "matches_C_groups": planned_groups.exists() and sha(planned_groups) == sha(sources["groups"]),
        "code_recomputes_stationarity_and_scaler": True,
    }

    # Real-data learned-transform witness. Only the INNER block is perturbed;
    # repeat the native GMM fit with its unchanged configuration to expose fit-scope dependence.
    sys.path.insert(0, str(a.predictor_source))
    from tools import layer2_pool_h1 as l2
    manifest = read(a.pool_root / "eurusd_h1_layer2_pool_s/MANIFEST.json")
    raw = pd.read_parquet(pool / "full_pool_eurusd_h1.parquet", columns=["timestamp", "OPEN", "HIGH", "LOW", "CLOSE"])
    raw = raw.rename(columns={k: "Bid" + k.title() for k in ("OPEN", "HIGH", "LOW", "CLOSE")})
    P = l2.parents(raw)
    cut = pd.Timestamp("2023-07-01", tz="UTC")
    end = pd.Timestamp(manifest["fit_period"]["end_exclusive"])
    fit = np.asarray(P.index < end)
    inner = np.asarray((P.index >= cut) & (P.index < end))
    before = np.asarray(P.index < cut)
    p0 = manifest["family_meta"]["gmm"]["params"]
    f0, _ = l2.apply_gmm(P, p0)
    store_gmm = frame[list(f0.columns)].to_numpy()
    report["learned_transform_witness"] = {
        "layer2_sha256": sha(a.predictor_source / "tools/layer2_pool_h1.py"),
        "upstream_fit_period": manifest["fit_period"],
        "downstream_fit_period": sc["fit_period"],
        "selected_affected_columns": [c for c in columns if c.startswith(("rg.gmm.", "kf.", "fd."))],
        "recomputed_parent_posterior_matches_store": bool(np.array_equal(f0.to_numpy(), store_gmm, equal_nan=True)),
        "initial_max_abs_store_delta": float(np.nanmax(np.abs(f0.to_numpy() - store_gmm))),
        "inner_rows_used_in_upstream_fit": int(inner.sum()),
    }
    # Preserve the actual native seed, settings and row count; this is a diagnostic fit only.
    P1 = P.copy()
    P1.loc[inner, "logret_1"] *= 3.0
    P1.loc[inner, "rv_24"] *= 3.0
    P1.loc[inner, "log_range"] *= 3.0
    base_fit = l2.fit_gmm(P, fit)
    changed_fit = l2.fit_gmm(P1, fit)
    a0, _ = l2.apply_gmm(P, base_fit)
    a1, _ = l2.apply_gmm(P1, changed_fit)
    frozen, _ = l2.apply_gmm(P1, base_fit)
    x, y = a0.to_numpy()[before], a1.to_numpy()[before]
    changed = ~(np.equal(x, y) | (np.isnan(x) & np.isnan(y)))
    report["learned_transform_witness"].update(
        native_refit_max_abs_store_delta=float(np.nanmax(np.abs(a0.to_numpy() - store_gmm))),
        frozen_apply_prefix_unchanged=bool(np.array_equal(x, frozen.to_numpy()[before], equal_nan=True)),
        refit_changed_prefix_cells=int(changed.sum()),
        refit_changed_prefix_rows=int(changed.any(axis=1).sum()),
        refit_max_abs_prefix_delta=float(np.nanmax(np.abs(x - y))),
        diagnostic_gmm_fits=2, policy_fits=0,
    )
    assert report["learned_transform_witness"]["frozen_apply_prefix_unchanged"]
    assert changed.any(), "fit-scope witness did not reproduce"
    assert all(checks.values()), checks
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": checks, "mutations_accepted": [k for k,v in tests.items() if v["accepted"]],
                      "learned_transform_witness": report["learned_transform_witness"],
                      "queue_matches_C": report["next_queue_consumer"]["matches_C_groups"]}, indent=2))


if __name__ == "__main__":
    main()
