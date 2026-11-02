"""Price data loading. Returns a daily close-price Series indexed by date."""
from __future__ import annotations

import pandas as pd


def load_prices(cfg: dict) -> pd.Series:
    g = cfg["general"]
    if g["data_source"] == "csv":
        df = pd.read_csv(g["csv_path"], parse_dates=["Date"]).set_index("Date").sort_index()
        close = df["Close"]
    elif g["data_source"] == "yfinance":
        try:
            import yfinance as yf
        except ImportError as e:  # pragma: no cover
            raise SystemExit("yfinance is not installed: pip install -r requirements.txt") from e
        df = yf.download(g["symbol"], start=g["start"], auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"no data downloaded for {g['symbol']}")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):  # yfinance returns MultiIndex columns
            close = close.iloc[:, 0]
    else:
        raise ValueError(f"unknown data_source {g['data_source']!r}")
    close = close.astype(float).dropna()
    close = close[close > 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = "close"
    if len(close) < 300:
        raise RuntimeError(f"only {len(close)} price rows; need at least 300 for the indicators")
    return close
