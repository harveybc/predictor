from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("PROJECT3_ROOT", "/home/harveybc/Documents/GitHub/financial-data"))
TARGET_TIMEFRAMES = ("5m", "15m", "1h", "4h")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def log(machine: str, message: str) -> None:
    path = ROOT / "_logs" / machine / "stage22_trading_features_worker.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} {message}\n")


def notify(event: str, title: str, message: str) -> None:
    script = ROOT / "_scripts" / "telegram_notify.py"
    if not script.exists():
        return
    import subprocess
    import sys

    try:
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--event",
                f"stage22:trading:{event}",
                "--title",
                title,
                "--message",
                message,
                "--min-interval-minutes",
                "20",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except Exception:
        return


def read_asset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError("missing timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / down.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def rolling_autocorr(series: pd.Series, window: int, lag: int) -> pd.Series:
    return series.rolling(window).corr(series.shift(lag))


def hurst_proxy(series: pd.Series, window: int = 200) -> pd.Series:
    # Fast state-memory proxy: persistent positive autocorrelation lifts the
    # estimate above 0.5, anti-persistence pushes it below 0.5. Full R/S Hurst
    # is reserved for the heavier Stage 2.3/2.4 passes.
    return (0.5 + 0.5 * rolling_autocorr(series, window, 1)).clip(0, 1)


def compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else None

    for horizon in [1, 5, 10, 20, 60]:
        out[f"return_{horizon}"] = close.pct_change(horizon)
        out[f"log_return_{horizon}"] = np.log(close / close.shift(horizon))

    for period in [10, 20, 50, 100, 200]:
        out[f"sma_{period}"] = close.rolling(period).mean()
        out[f"ema_{period}"] = ema(close, period)
        out[f"close_sma_ratio_{period}"] = close / out[f"sma_{period}"] - 1

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    for period in [7, 14, 21]:
        out[f"rsi_{period}"] = rsi(close, period)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    out["stoch_k"] = 100 * (close - low14) / (high14 - low14)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()
    out["williams_r_14"] = -100 * (high14 - close) / (high14 - low14)
    typical = (high + low + close) / 3
    mean_typical_14 = typical.rolling(14).mean()
    mean_abs_dev_14 = (typical - mean_typical_14).abs().rolling(14).mean()
    out["cci_14"] = (typical - mean_typical_14) / (0.015 * mean_abs_dev_14)
    for period in [10, 20, 60]:
        out[f"roc_{period}"] = close.pct_change(period)
    out["mom_10"] = close.diff(10)
    out["mom_20"] = close.diff(20)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_middle"] = bb_mid
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_pct_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]
    tr = true_range(df)
    out["atr_14"] = tr.rolling(14).mean()
    out["natr_14"] = out["atr_14"] / close
    for period in [10, 20, 60]:
        out[f"hist_vol_{period}"] = out["log_return_1"].rolling(period).std() * np.sqrt(period)

    out["ema_cross_10_50"] = (out["ema_10"] - out["ema_50"]) / close
    out["ema_cross_20_100"] = (out["ema_20"] - out["ema_100"]) / close
    out["trend_slope_50"] = np.log(close.replace(0, np.nan)).diff(50) / 50
    out["trend_strength_50"] = out["trend_slope_50"].abs()

    if volume is not None:
        signed = np.sign(close.diff()).fillna(0)
        out["obv"] = (signed * volume.fillna(0)).cumsum()
        out["obv_delta_20"] = out["obv"].diff(20)
        for period in [10, 20]:
            out[f"volume_sma_{period}"] = volume.rolling(period).mean()
        out["volume_ratio_20"] = volume / out["volume_sma_20"]
        out["vwap_60"] = (typical * volume).rolling(60).sum() / volume.rolling(60).sum()
        mf_raw = typical * volume
        positive = mf_raw.where(typical.diff() > 0, 0).rolling(14).sum()
        negative = mf_raw.where(typical.diff() < 0, 0).rolling(14).sum().abs()
        out["mfi_14"] = 100 - (100 / (1 + positive / negative))
    return sanitize(out)


def compute_statistical(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    close = df["close"].astype(float)
    returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    out["log_return_1"] = returns
    for window in [20, 60, 252]:
        out[f"roll_mean_ret_{window}"] = returns.rolling(window).mean()
        out[f"roll_std_ret_{window}"] = returns.rolling(window).std()
        out[f"roll_skew_ret_{window}"] = returns.rolling(window).skew()
        out[f"roll_kurt_ret_{window}"] = returns.rolling(window).kurt()
    out["realized_var_12"] = returns.pow(2).rolling(12).sum()
    out["realized_var_48"] = returns.pow(2).rolling(48).sum()
    out["autocorr_lag1_100"] = rolling_autocorr(returns, 100, 1)
    out["autocorr_lag5_100"] = rolling_autocorr(returns, 100, 5)
    out["sqret_autocorr_lag1_100"] = rolling_autocorr(returns.pow(2), 100, 1)
    vol20 = returns.rolling(20).std()
    vol252 = returns.rolling(252).std()
    out["vol_regime_high"] = (vol20 > vol252.rolling(252).quantile(0.75)).astype("float")
    out["vol_regime_low"] = (vol20 < vol252.rolling(252).quantile(0.25)).astype("float")
    out["hurst_proxy_200"] = hurst_proxy(returns, 200)
    out["zscore_close_100"] = (close - close.rolling(100).mean()) / close.rolling(100).std()
    return sanitize(out)


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if col == "timestamp":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def validate_features(df: pd.DataFrame) -> dict[str, Any]:
    values = df.drop(columns=["timestamp"], errors="ignore")
    inf_count = int(np.isinf(values.to_numpy(dtype="float64", na_value=np.nan)).sum())
    all_nan_cols = [col for col in values.columns if values[col].isna().all()]
    return {
        "rows": int(len(df)),
        "columns": int(len(values.columns)),
        "inf_count": inf_count,
        "all_nan_columns": all_nan_cols[:50],
        "all_nan_column_count": len(all_nan_cols),
    }


def write_docs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Stage 2.2 Trading Asset Features",
                "",
                "Technical and statistical features computed causally from Stage 2.1 trading-asset OHLCV data.",
                "",
                "Citations: Stefan Jansen, Machine Learning for Algorithmic Trading, Ch. 4; Ruey Tsay, Analysis of Financial Time Series; Marcos Lopez de Prado, Advances in Financial Machine Learning.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "data_dictionary.md").write_text(
        "\n".join(
            [
                "# Data Dictionary",
                "",
                "- `technical.parquet`: returns, moving averages, MACD, RSI, stochastic, Bollinger, ATR, volume and trend features.",
                "- `statistical.parquet`: rolling moments, realized variance, autocorrelation, volatility regime flags, Hurst proxy, close z-score.",
                "- Warmup-window NaNs are expected at the start of each series.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def process_asset_tf(asset: str, tf: str, machine: str) -> dict[str, Any]:
    source = ROOT / "features" / "trading_asset_data" / asset / f"{tf}.parquet"
    if not source.exists():
        return {"asset": asset, "timeframe": tf, "status": "missing_source", "source": rel(source)}
    out_dir = ROOT / "features" / "trading_asset_features" / asset / tf
    write_docs(out_dir)
    try:
        df = read_asset(source)
        technical = compute_technical(df)
        statistical = compute_statistical(df)
        tech_path = out_dir / "technical.parquet"
        stat_path = out_dir / "statistical.parquet"
        technical.to_parquet(tech_path, index=False)
        statistical.to_parquet(stat_path, index=False)
        result = {
            "asset": asset,
            "timeframe": tf,
            "status": "ok",
            "source": rel(source),
            "technical_path": rel(tech_path),
            "statistical_path": rel(stat_path),
            "technical_validation": validate_features(technical),
            "statistical_validation": validate_features(statistical),
        }
    except Exception as exc:
        result = {
            "asset": asset,
            "timeframe": tf,
            "status": "failed",
            "source": rel(source),
            "error": f"{type(exc).__name__}: {exc}",
        }
    log(machine, f"asset={asset} tf={tf} status={result['status']}")
    return result


def discover_assets(args_assets: list[str] | None) -> list[str]:
    if args_assets:
        return sorted(args_assets)
    root = ROOT / "features" / "trading_asset_data"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def write_report(machine: str, universe: str, results: list[dict[str, Any]]) -> None:
    summary = {
        "generated_at": utc_now(),
        "stage": "Stage 2.2",
        "machine": machine,
        "universe": universe,
        "jobs_total": len(results),
        "jobs_ok": sum(item["status"] == "ok" for item in results),
        "jobs_failed": sum(item["status"] == "failed" for item in results),
        "results": results,
    }
    out_json = ROOT / "_metadata" / f"stage22_trading_features_{machine}_{universe}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    out_md = ROOT / "_logs" / "supervisor_reports" / f"stage22_trading_features_{machine}_{universe}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Stage 2.2 Trading Features - {machine} {universe}",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"- Jobs total: {summary['jobs_total']}",
        f"- Jobs ok: {summary['jobs_ok']}",
        f"- Jobs failed: {summary['jobs_failed']}",
        "",
        "| Asset | TF | Status | Technical cols | Statistical cols |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["asset"],
                    item["timeframe"],
                    item["status"],
                    str(item.get("technical_validation", {}).get("columns", "")),
                    str(item.get("statistical_validation", {}).get("columns", "")),
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--assets", nargs="*")
    args = parser.parse_args()

    assets = discover_assets(args.assets)
    log(args.machine, f"START stage22 trading universe={args.universe} assets={len(assets)}")
    notify(
        f"{args.machine}:{args.universe}:start",
        f"Project 3 Stage 2.2 {args.machine} started",
        f"machine: {args.machine}\nstage: Stage 2.2\ncurrent_task: trading technical/statistical features\nassets: {len(assets)}",
    )
    results = [process_asset_tf(asset, tf, args.machine) for asset in assets for tf in TARGET_TIMEFRAMES]
    write_report(args.machine, args.universe, results)
    log(args.machine, f"DONE stage22 trading universe={args.universe} jobs={len(results)}")
    notify(
        f"{args.machine}:{args.universe}:finish",
        f"Project 3 Stage 2.2 {args.machine} finished",
        f"machine: {args.machine}\nstage: Stage 2.2\nstatus: finished\ndeliverable_path: _metadata/stage22_trading_features_{args.machine}_{args.universe}.json\njobs: {len(results)}",
    )


if __name__ == "__main__":
    main()
