#!/usr/bin/env python3
"""Rebuild pinned tensors and replay saved FULL checkpoints on retained probe rows."""
import argparse
from pathlib import Path
import sys

import expert_package_worker as w


def compare(actual, expected):
    if actual is None or expected is None:
        return {"exact": actual is expected, "max_abs_delta": None}
    if len(actual) != len(expected) or any(len(a) != len(b) for a, b in zip(actual, expected)):
        raise ValueError("REPLAY_PROBE_SHAPE")
    delta = max((abs(float(a) - float(b)) for row, target in zip(actual, expected)
                 for a, b in zip(row, target)), default=0.0)
    return {"exact": actual == expected, "max_abs_delta": delta}


def replay(plan):
    source = Path(plan["source_root"])
    if plan.get("source_reference_config"):
        cfg = w.read(plan["source_reference_config"])
        def verify_source():
            if cfg.get("t3_manifest", {}).get("identity") == "t3v2_7db1953f99ad":
                from close_t3_v5 import verify_source as verify_v5
                verify_v5(source, cfg)
            else:
                from t3_peer_preflight import verify_inputs
                verify_inputs(source, cfg)
            if w.runtime_versions(cfg["python"]) != cfg["runtime_versions"]:
                raise ValueError("REPLAY_RUNTIME_VERSIONS")
    else:
        cfg = w.read(source / "ops/CONFIG.private.json")
        def verify_source():
            w.verify_pins(cfg)
    verify_source()
    sys.path.insert(0, str(source / "code"))
    import numpy as np
    import torch
    from app.short_horizon.expert_guided import data as DG, temporal as T
    requested = w.read(source / "in/FEATURE_LIST_shortlist_strict.json")["features"]
    fixed = cfg.get("t3_manifest", {}).get("fixed", {})
    store = fixed.get("store_dir", "full_pool_s")
    dropout = fixed.get("regularization", {}).get("dropout", .1)
    ds = DG.load_store(source / "data/admitted", source / "data" / store / "full_pool_eurusd_h1.parquet",
                       requested, scaling="fixed_train", contract="consolidated", nonstationary="diff",
                       win=13, stride=2, verify=True, end="2025-01-01")
    if plan.get("trading_replay"):
        import dataclasses
        from tools import expert_guided_e3m as S
        from app.short_horizon.expert_guided import env as E
        context = S.Ctx(ds)
        execution = dataclasses.replace(S.MUSASHI_CFG, min_orders_per_week=2.0)
    outcomes = {}
    for name, item in plan["units"].items():
        run = Path(item["run"])
        for filename, expected in item["hashes"].items():
            if w.digest(run / filename) != expected:
                raise ValueError("REPLAY_INPUT_CHANGED:" + name + ":" + filename)
        result, scaler, probes = [w.read(run / f) for f in ("E3M.json", "SCALER.json", "PROBE.json")]
        DG.verify_scaler(scaler, ds)
        if (result["meta"]["support_hash"] != ds.meta["support_hash"]
                or result["agent_state"]["mode"] != "none" or result["labels"] != {"mode": "zigzag"}):
            raise ValueError("REPLAY_SUPPORT_OR_OBSERVATION_CONTRACT")
        stats = ds.meta["scaling"]["features"]
        if any([stats[f][field] for f in scaler["features"]] != scaler[field] for field in ("mean", "std")):
            raise ValueError("REPLAY_SCALER_NOT_EXACT")
        modules, policies = {}, {}
        for seed in (20261019, 20261020, 20261021):
            policies[seed] = {}
            for module in w.MODULES:
                key = f"FULL_{module}_{seed}"
                probe = probes["probe"][key]
                rows = np.asarray(probe["rows"], dtype=int)
                if (probe["state"] != "zeros" or rows.size == 0 or rows.min() < 0 or rows.max() >= len(ds.windows)
                        or [int(ds.row_ts[r]) for r in rows] != probe["row_ts_ns"]):
                    raise ValueError("REPLAY_PROBE_ROW_OR_TIME")
                X = np.concatenate((ds.windows[rows].reshape(len(rows), -1).astype(np.float32),
                                    np.zeros((len(rows), 9), np.float32)), axis=1)
                model = T.TemporalPolicyNet(len(requested), 13, 2, n_reg=2 if module.startswith("open_") else 0,
                                            n_state=9, dropout=dropout, **result["arch"])
                model.load_state_dict(torch.load(run / (key + ".pt"), map_location="cpu", weights_only=True), strict=True)
                policy = T.TorchPolicy(model)
                policies[seed][module] = policy
                logits, reg = policy(X)
                modules[key] = {"logits": compare(logits.tolist(), probe["logits"]),
                                "reg": compare(None if reg is None else reg.tolist(), probe["reg"])}
        for filename, expected in item["hashes"].items():
            if w.digest(run / filename) != expected:
                raise ValueError("REPLAY_INPUT_CHANGED_DURING_FORWARD")
        outcomes[name] = {"checkpoints": len(modules), "exact": all(x["logits"]["exact"] and x["reg"]["exact"] for x in modules.values()),
                          "modules": modules, "result_sha256": item["hashes"]["E3M.json"]}
        if plan.get("trading_replay"):
            trading = {}
            for seed, pols in policies.items():
                frozen = result["trials"]["FULL"][str(seed)]["arbiter"]
                composed = S.Composed("FULL", pols, frozen["arbiter"], frozen["tau"], mask=E.state_mask("none"))
                trading[str(seed)] = {}
                for period in ("TRAIN", "INNER_A", "OBS_2024"):
                    actual = S.jsonable(S.summary(context.rollout(period, composed(context), execution)[0]))
                    expected = result["trials"]["FULL"][str(seed)]["trading"][period]
                    changed = {k: {"expected": expected.get(k), "actual": actual.get(k)}
                               for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)}
                    trading[str(seed)][period] = {"exact": actual == expected, "differences": changed}
                    print({"unit": name, "seed": seed, "period": period, "trading_exact": actual == expected}, flush=True)
            outcomes[name]["trading"] = trading
            outcomes[name]["trading_exact"] = all(v["exact"] for seed in trading.values() for v in seed.values())
        print({"unit": name, "exact": outcomes[name]["exact"]}, flush=True)
    verify_source()
    return {"schema": "t3_checkpoint_probe_replay.v1", "state": "PASS" if all(x["exact"] and x.get("trading_exact", True) for x in outcomes.values()) else "FAIL",
            "units": outcomes, "count": len(outcomes), "checkpoints": sum(x["checkpoints"] for x in outcomes.values()),
            "torch_version": torch.__version__, "torch_cpu_threads": torch.get_num_threads(),
            "threshold": "exact equality; no tolerance", "training_performed": False, "test_scored": False,
            "full_trading_metrics_replayed": bool(plan.get("trading_replay")),
            "scope": "saved-checkpoint forward on retained probe rows; optional native TRAIN/INNER_A/OBS_2024 rollouts"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    result = replay(w.read(a.plan))
    w.atomic(a.out, result)
    raise SystemExit(0 if result["state"] == "PASS" else 1)
