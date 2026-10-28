#!/usr/bin/env python3
"""Bounded CPU wiring witness on real admitted bars; no scientific fit or score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(a.agent_root))
    import app
    from app.frozen_recipe_inputs import FrozenRecipeInputs
    from env_plugins.frozen_recipe import FrozenRecipeEnv
    from agent_plugins.sac_agent import Plugin as Agent
    rec = json.loads(a.recipe.read_text())
    if sha(a.recipe) != a.recipe_sha:
        raise ValueError("ENGINE_PROBE_RECIPE_PIN")
    if sha(a.bars) != rec["business"]["calendar"]["admitted_sha256"]:
        raise ValueError("ENGINE_PROBE_BARS_PIN")
    inputs = FrozenRecipeInputs.load(rec, a.store, a.manifest)
    end = pd.Timestamp(rec["split"]["test_closed_from"], tz="UTC")
    bars = pd.read_parquet(a.bars, filters=[("available_at", "<", end)])
    stamps = pd.DatetimeIndex(bars.available_at).as_unit("ns").asi8
    if not (np.diff(stamps) > 0).all():
        raise ValueError("ENGINE_PROBE_BAR_CLOCK")
    if not np.array_equal(stamps, pd.DatetimeIndex(bars.timestamp).as_unit("ns").asi8 + 3600 * 10**9):
        raise ValueError("ENGINE_PROBE_AVAILABILITY")
    rows = np.searchsorted(inputs.row_ts_ns, stamps)
    bounded = np.minimum(rows, len(inputs.row_ts_ns) - 1)
    matched = (rows < len(inputs.row_ts_ns)) & (inputs.row_ts_ns[bounded] == stamps)
    valid = matched & inputs.valid[bounded]
    counts = {}
    for name, (lo, hi) in rec["split"]["periods"].items():
        mask = (stamps >= pd.Timestamp(lo, tz="UTC").value) & (stamps < pd.Timestamp(hi, tz="UTC").value)
        counts[name] = {"bars": int(mask.sum()), "valid_windows": int((valid & mask).sum()),
                        "no_valid_window": int((~valid & mask).sum())}
    source = pd.read_parquet(a.store, columns=["timestamp"] + rec["list"]["columns"],
                             filters=[("timestamp", "<", end)])
    source_times = pd.DatetimeIndex(source.timestamp).as_unit("ns").asi8 + 3600 * 10**9
    nonfinite = {}
    for name, (lo, hi) in rec["split"]["periods"].items():
        mask = (source_times >= pd.Timestamp(lo, tz="UTC").value) & (source_times < pd.Timestamp(hi, tz="UTC").value)
        n = (~np.isfinite(source.loc[mask, rec["list"]["columns"]].to_numpy())).sum(axis=0)
        nonfinite[name] = {column: int(count) for column, count in zip(rec["list"]["columns"], n) if count}
    # First eligible bounded prefix in TRAIN, selected without prices or outcomes.
    w = rec["tensor"]["window"]
    context = (w["observations"] - 1) * w["stride_rows"]
    first = next((i for i in range(context, len(bars) - 8)
                  if valid[i:i + 9].all() and (inputs.period[bounded[i:i + 9]] == "TRAIN").all()), None)
    if first is None:
        raise ValueError("ENGINE_PROBE_NO_TRAIN_PREFIX")
    part = bars.iloc[first - context:first + 9].copy()
    frame = part.rename(columns={"available_at": "DATE_TIME", "BidOpen": "OPEN", "BidHigh": "HIGH",
                                 "BidLow": "LOW", "BidClose": "CLOSE"})
    frame = frame[["DATE_TIME", "OPEN", "HIGH", "LOW", "CLOSE"]]
    frame["VOLUME"] = 0.
    app.__path__.insert(0, str(a.gym_root / "app"))
    sys.path.insert(0, str(a.gym_root))
    from app.env import GymFxEnv
    from broker_plugins.default_broker import Plugin as Broker
    from data_feed_plugins.default_data_feed import Plugin as Feed
    from preprocessor_plugins.feature_window_preprocessor import Plugin as Preprocessor
    from strategy_plugins.shared_execution_envelope import Plugin as Strategy
    from reward_plugins.pnl_reward import Plugin as Reward
    from metrics_plugins.trading_metrics import Plugin as Metrics
    cfg = dict(date_column="DATE_TIME", price_column="CLOSE", headers=True,
               timestamp_semantics="completed_bar_available_at", timeframe="1h", execution_transition="next_open_v2",
               feature_columns=["CLOSE"], feature_scaling="none", feature_clip=0,
               window_size=w["observations"], observation_stride_bars=w["stride_rows"],
               require_full_observation_window=True, include_price_window=False, include_agent_state=False,
               initial_cash=10000., max_entries_per_day=1, action_space_mode="continuous",
               continuous_action_contract="target_exposure_hysteresis_v2", observation_encoder="shared_recipe",
               shared_recipe=rec, net_arch=[rec["representation"]["model"]["head"]], device="cpu",
               buffer_size=1, train_seed=7)
    with tempfile.TemporaryDirectory(prefix="recipe-engine-witness-") as tmp:
        path = Path(tmp) / "bars.csv"
        frame.to_csv(path, index=False)
        cfg["input_data_file"] = str(path)
        base = GymFxEnv(cfg, Feed(cfg), Broker(cfg), Strategy(cfg), Preprocessor(cfg), Reward(cfg), Metrics(cfg))
        env = FrozenRecipeEnv(base, inputs)
        try:
            agent = Agent(cfg)
            model = agent.build(env, cfg)
            obs, info = env.reset(seed=7)
            captures = []
            hook = model.actor.features_extractor.register_forward_pre_hook(
                lambda m, args: captures.append(args[0]["features"].detach().cpu().numpy().copy()))
            for j in range(8):
                expected = inputs.windows[rows[first + j]]
                if not np.array_equal(obs["features"], expected):
                    raise ValueError("ENGINE_PROBE_INPUT")
                model.predict(obs, deterministic=True)
                if not np.array_equal(captures[-1][0], expected):
                    raise ValueError("ENGINE_PROBE_ACTOR")
                # HOLD only: engineering clock witness, not an untrained-policy backtest.
                obs, reward, done, truncated, info = env.step(np.array([0.], np.float32))
                if not np.isfinite(reward) or truncated:
                    raise ValueError("ENGINE_PROBE_TRANSITION")
                if pd.Timestamp(info["action_time"]).value != stamps[first + j]:
                    raise ValueError("ENGINE_PROBE_ACTION_CLOCK")
            hook.remove()
            if not np.array_equal(obs["features"], inputs.windows[rows[first + 8]]):
                raise ValueError("ENGINE_PROBE_FINAL_INPUT")
            if base.bridge.trade_count != 0:
                raise ValueError("ENGINE_PROBE_UNEXPECTED_TRADE")
        finally:
            env.close()
    return dict(schema="sac_recipe_engine_witness.v1", state="PASS", recipe=rec["identity"],
                scientific_fits=0, engine_steps=8, trades=0, test_read=False,
                input_exact=True, actor_input_exact=True, action_clock_exact=True,
                observations="features only", period_support=counts, nonfinite_source_cells=nonfinite,
                scientific_accounting_parity_verified=False,
                first_observation=pd.Timestamp(stamps[first], tz="UTC").isoformat(),
                pins={"recipe": sha(a.recipe), "bars": sha(a.bars), "store": sha(a.store),
                      "manifest": sha(a.manifest), "bridge": sha(a.agent_root / "env_plugins/frozen_recipe.py"),
                      "engine": sha(a.gym_root / "app/env.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("agent-root", "gym-root", "recipe", "store", "manifest", "bars", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recipe-sha", required=True)
    args = parser.parse_args()
    result = verify(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
