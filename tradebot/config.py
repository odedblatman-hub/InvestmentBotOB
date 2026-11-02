"""Configuration loading/saving.

Config lives in a TOML file. Anything not set there falls back to DEFAULTS.
"""
from __future__ import annotations

import copy
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "general": {
        "symbol": "SPY",
        "data_source": "yfinance",  # "yfinance" or "csv"
        "csv_path": "data/prices.csv",  # used when data_source == "csv" (columns: Date, Close)
        "start": "2005-01-01",
        "state_dir": "state",
        "proposals_dir": "proposals",
    },
    "broker": {
        "kind": "paper",  # "paper" or "alpaca"
        "initial_cash": 100_000.0,
        "slippage_bps": 5.0,
        # Alpaca: keys come from env vars ALPACA_API_KEY / ALPACA_API_SECRET, never from this file.
        "alpaca_base_url": "https://paper-api.alpaca.markets",
        "allow_live": False,  # must be true to point at a non-paper URL
    },
    "risk": {
        "max_drawdown": 0.15,  # handbrake: halt when equity falls this far below its peak
        "max_position_pct": 1.0,  # max fraction of equity in the position (1.0 = no leverage)
        "max_order_pct": 0.5,  # max fraction of equity in a single order
        "kill_switch_file": "KILL",  # relative to state_dir; create this file to stop all orders
    },
    "regime": {
        "vol_window": 20,  # days of returns for realized volatility
        "vol_lookback": 252,  # window used to rank today's vol against the past
        "vol_high": 0.80,  # vol percentile above which the market is "volatile" (jittery)
        "er_window": 20,  # Kaufman efficiency-ratio window
        "er_low": 0.25,  # ER below this = choppy/"sideways"
        "dd_lookback": 252,
        "crash_dd": 0.15,  # drawdown from 1y high that, below trend, counts as freefall
        "crash_mom": 0.10,  # or a 20-day drop this large while vol is extreme
    },
    "strategy": {
        "trend_window": 200,  # SMA used by trend filters
        "bb_window": 20,  # mean-reversion lookback
        "mr_scale": 0.25,  # exposure change per 1 std dev of z-score in sideways markets
        "target_vol": 0.12,  # annualized vol target in volatile markets
        "cost_bps": 5.0,  # backtest cost per unit of turnover
    },
    "gate": {
        "enforce": True,  # refuse to trade unless the walk-forward test beats buy & hold
        "metric": "sharpe",  # "sharpe", "calmar" or "cagr"
        "train_years": 3,
        "test_years": 1,
        "min_fold_win_rate": 0.5,  # fraction of out-of-sample periods that must beat buy & hold
        "max_age_days": 7,  # re-run the backtest when the cached result is older than this
        "grid": {"trend_window": [100, 200], "target_vol": [0.10, 0.15]},
    },
    "nightly": {
        "min_improvement": 0.05,  # proposal must beat current config's OOS metric by this much
        "max_proposals": 3,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None and Path(path).exists():
        with open(path, "rb") as fh:
            cfg = _merge(cfg, tomllib.load(fh))
    cfg["_path"] = str(path) if path else None
    return cfg


def state_dir(cfg: dict) -> Path:
    p = Path(cfg["general"]["state_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- minimal TOML writer (tomllib is read-only) ---------------------------------------

def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v)} to TOML")


def dump_toml(cfg: dict) -> str:
    lines: list[str] = []

    def section(prefix: str, d: dict) -> None:
        scalars = {k: v for k, v in d.items() if not isinstance(v, dict)}
        tables = {k: v for k, v in d.items() if isinstance(v, dict)}
        if scalars:
            lines.append(f"[{prefix}]")
            lines.extend(f"{k} = {_fmt(v)}" for k, v in scalars.items())
            lines.append("")
        for k, v in tables.items():
            section(f"{prefix}.{k}", v)

    for name, body in cfg.items():
        if name.startswith("_"):
            continue
        section(name, body)
    return "\n".join(lines)


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def get_dotted(cfg: dict, key: str) -> Any:
    node = cfg
    for p in key.split("."):
        node = node[p]
    return node


def save_config(cfg: dict, path: str | Path) -> Path | None:
    """Write cfg to path, backing up the previous file. Returns the backup path."""
    path = Path(path)
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
    path.write_text(dump_toml(cfg))
    return backup
