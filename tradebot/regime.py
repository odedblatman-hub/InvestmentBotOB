"""Market regime detection and the per-regime strategies.

Every feature at date t uses only prices up to and including t (no look-ahead).
Regimes, checked in priority order:
  crash     - freefall: deep drawdown below the trend, or a sharp drop while vol is extreme
  volatile  - "jittery": realized vol in the top of its own 1-year range
  sideways  - choppy, no direction: low Kaufman efficiency ratio
  calm      - everything else: quiet, directional market
  warmup    - not enough history yet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ("calm", "sideways", "volatile", "crash")


def features(close: pd.Series, rp: dict, sp: dict) -> pd.DataFrame:
    ret = close.pct_change()
    vol = ret.rolling(rp["vol_window"]).std() * np.sqrt(252)
    vol_rank = vol.rolling(rp["vol_lookback"], min_periods=60).rank(pct=True)
    sma = close.rolling(sp["trend_window"]).mean()
    peak = close.rolling(rp["dd_lookback"], min_periods=1).max()
    n = rp["er_window"]
    er = (close - close.shift(n)).abs() / close.diff().abs().rolling(n).sum()
    bb_mean = close.rolling(sp["bb_window"]).mean()
    bb_std = close.rolling(sp["bb_window"]).std()
    return pd.DataFrame(
        {
            "close": close,
            "vol": vol,
            "vol_rank": vol_rank,
            "sma": sma,
            "dd": close / peak - 1,
            "er": er,
            "z": (close - bb_mean) / bb_std,
            "mom20": close / close.shift(20) - 1,
        }
    )


def classify(f: pd.DataFrame, rp: dict) -> pd.Series:
    crash = ((f["dd"] <= -rp["crash_dd"]) & (f["close"] < f["sma"])) | (
        (f["mom20"] <= -rp["crash_mom"]) & (f["vol_rank"] >= 0.9)
    )
    volatile = f["vol_rank"] >= rp["vol_high"]
    sideways = f["er"] < rp["er_low"]
    out = np.select([crash, volatile, sideways], ["crash", "volatile", "sideways"], default="calm")
    warm = f[["vol_rank", "sma", "er", "z"]].isna().any(axis=1)
    return pd.Series(np.where(warm, "warmup", out), index=f.index, name="regime")


def exposure(f: pd.DataFrame, regime: pd.Series, sp: dict) -> pd.Series:
    """Target long exposure in [0, 1] for each day, chosen by the day's regime."""
    up = (f["close"] > f["sma"]).astype(float)
    calm = up  # trend following: fully in while above the trend, out otherwise
    sideways = (0.5 - f["z"] * sp["mr_scale"]).clip(0, 1)  # mean reversion: buy dips, trim rips
    volatile = up * (sp["target_vol"] / f["vol"]).clip(0, 1)  # vol targeting, only with the trend
    exp = np.select(
        [regime == "calm", regime == "sideways", regime == "volatile"],
        [calm, sideways, volatile],
        default=0.0,  # crash and warmup -> cash
    )
    return pd.Series(np.nan_to_num(exp), index=f.index, name="exposure")


def signals(close: pd.Series, cfg: dict, overrides: dict | None = None) -> pd.DataFrame:
    sp = {**cfg["strategy"], **(overrides or {})}
    rp = cfg["regime"]
    f = features(close, rp, sp)
    f["regime"] = classify(f, rp)
    f["exposure"] = exposure(f, f["regime"], sp).clip(0, cfg["risk"]["max_position_pct"])
    return f
