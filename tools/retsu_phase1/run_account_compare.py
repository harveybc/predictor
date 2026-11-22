#!/usr/bin/env python
"""Same long/short strategy on the change-champion price and on CLOSE persistence.

The individual is the plugin default, declared here and not fit on 2024.
Bars are the hourly CLOSE repeated into open/high/low because the STL series
has no high or low. Both arms use that feed.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

os.environ.setdefault("STRATEGY_QUIET", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
PHASE = Path(os.environ.get("RETSU_ACCOUNT_DIR", "/var/tmp/retsu-phase1/followup/account"))
INDIVIDUAL = (5.0, 0.9, 2.0, 0.5, 2.0, 24.0)


def load_predictions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, parse_dates=["DATE_TIME"])
    frame["DATE_TIME"] = pd.to_datetime(frame["DATE_TIME"], utc=True).dt.tz_localize(None)
    frame = frame.set_index("DATE_TIME").sort_index()
    hourly_cols = [col for col in frame.columns if col.startswith("h")]
    daily_cols = [col for col in frame.columns if col.startswith("d")]
    hourly = frame[hourly_cols]
    daily = frame[daily_cols]
    price = frame[hourly_cols[0]].copy()
    # Persistence files are constant across horizons, so column 0 is the origin CLOSE.
    base = pd.DataFrame({
        "open": price.to_numpy(),
        "high": price.to_numpy(),
        "low": price.to_numpy(),
        "close": price.to_numpy(),
        "volume": 0.0,
    }, index=price.index)
    base.index.name = "datetime"
    return hourly, daily, base


def summarize(name: str, profit: float, stats: dict, trades: list[dict]) -> dict:
    sides = Counter()
    return {
        "arm": name,
        "individual": list(INDIVIDUAL),
        "profit_usd": profit,
        "final_balance_usd": 10000.0 + profit,
        "num_trades": stats.get("num_trades", 0),
        "win_pct": stats.get("win_pct", 0),
        "max_dd": stats.get("max_dd", 0),
        "sharpe": stats.get("sharpe", 0),
        "sides": dict(sides),
        "trades": [
            {
                "open_dt": str(trade.get("open_dt")),
                "close_dt": str(trade.get("close_dt")),
                "pnl": trade.get("pnl"),
                "pips": trade.get("pips"),
                "duration": trade.get("duration"),
                "max_dd": trade.get("max_dd"),
            }
            for trade in trades
        ],
    }


def main() -> None:
    from app.plugins.plugin_long_short_predictions import Plugin

    rows = []
    for name in ("seed7", "pretrained"):
        naive_h, naive_d, base = load_predictions(PHASE / f"{name}_naive.csv")
        model_h, model_d, _base_model = load_predictions(PHASE / f"{name}_model.csv")
        if not naive_h.index.equals(model_h.index):
            raise RuntimeError(f"INDEX_MISMATCH {name}")
        plugin = Plugin()
        os.chdir(PHASE)
        profit, stats = plugin.evaluate_candidate(
            INDIVIDUAL, base, model_h, model_d, {"show_trades": False, "hourly_predictions_file": "provided", "daily_predictions_file": "provided"}
        )
        rows.append(summarize(f"{name}_model", profit, stats, list(plugin.trades)))
        plugin = Plugin()
        profit, stats = plugin.evaluate_candidate(
            INDIVIDUAL, base, naive_h, naive_d, {"show_trades": False, "hourly_predictions_file": "provided", "daily_predictions_file": "provided"}
        )
        rows.append(summarize(f"{name}_naive", profit, stats, list(plugin.trades)))
    payload = {
        "strategy": "plugin_long_short_predictions",
        "individual_declared": list(INDIVIDUAL),
        "individual_source": "plugin_params defaults; time_horizon is unused because predictions are provided",
        "feed": "2024 hours with both short and long change labels; OHLC equals CLOSE",
        "prediction_columns_are_prices": True,
        "arms": rows,
    }
    (PHASE / "account_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(row["arm"], "profit", round(row["profit_usd"], 2), "trades", row["num_trades"], "sharpe", row["sharpe"])
    print("ACCOUNT_COMPARE_OK")


if __name__ == "__main__":
    main()
