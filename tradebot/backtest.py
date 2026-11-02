"""Walk-forward backtest and the trading gate.

For each out-of-sample (test) period, strategy parameters are picked using ONLY the
preceding training period, then evaluated on the test period they never saw. The
strategy is compared with buy & hold over the same test period. The 15% drawdown
handbrake is simulated exactly as it works live: once tripped, the strategy sits in
cash for the rest of that period.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import state_dir
from .regime import REGIMES, signals

TRADING_DAYS = 252


def simulate(close: pd.Series, exposure: pd.Series, cost_bps: float, max_dd: float | None) -> pd.DataFrame:
    """Daily simulation. Exposure decided at close t is held over day t+1."""
    ret = close.pct_change().fillna(0.0).to_numpy()
    target = exposure.to_numpy()
    n = len(ret)
    pos = np.zeros(n)
    strat = np.zeros(n)
    equity, peak, prev, halted = 1.0, 1.0, 0.0, False
    for t in range(1, n):
        p = 0.0 if halted else target[t - 1]
        r = p * ret[t] - abs(p - prev) * cost_bps / 1e4
        equity *= 1 + r
        peak = max(peak, equity)
        if max_dd is not None and equity / peak - 1 <= -max_dd:
            halted = True
        pos[t], strat[t], prev = p, r, p
    return pd.DataFrame({"ret": strat, "pos": pos}, index=close.index)


def metrics(daily_ret: pd.Series) -> dict:
    r = daily_ret.fillna(0.0)
    eq = (1 + r).cumprod()
    years = max(len(r) / TRADING_DAYS, 1e-9)
    total = float(eq.iloc[-1] - 1) if len(eq) else 0.0
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min()) if len(eq) else 0.0
    calmar = cagr / abs(mdd) if mdd < 0 else (cagr * 100 if cagr > 0 else 0.0)
    return {"total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": mdd, "calmar": calmar}


def _grid(cfg: dict) -> list[dict]:
    grid = cfg["gate"].get("grid") or {}
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def walk_forward(close: pd.Series, cfg: dict) -> dict:
    g, cost, max_dd = cfg["gate"], cfg["strategy"]["cost_bps"], cfg["risk"]["max_drawdown"]
    metric = g["metric"]
    combos = _grid(cfg)
    sigs = [signals(close, cfg, c) for c in combos]  # causal indicators: safe to compute once

    start, end = close.index[0], close.index[-1]
    test_start = start + pd.DateOffset(years=g["train_years"])
    folds, oos_parts, bench_parts, regime_parts = [], [], [], []
    while test_start + pd.DateOffset(months=3) <= end:
        test_end = min(test_start + pd.DateOffset(years=g["test_years"]), end + pd.Timedelta(days=1))
        train = (close.index >= test_start - pd.DateOffset(years=g["train_years"])) & (close.index < test_start)
        test = (close.index >= test_start) & (close.index < test_end)

        scores = [
            metrics(simulate(close[train], s["exposure"][train], cost, max_dd)["ret"])[metric] for s in sigs
        ]
        best = int(np.nanargmax(scores))
        sim = simulate(close[test], sigs[best]["exposure"][test], cost, max_dd)
        bench = close[test].pct_change().fillna(0.0)
        m_s, m_b = metrics(sim["ret"]), metrics(bench)
        folds.append(
            {
                "test_start": str(test_start.date()),
                "test_end": str(close.index[test][-1].date()),
                "params": combos[best],
                "strategy": m_s,
                "benchmark": m_b,
                "beat": m_s[metric] > m_b[metric],
                "handbrake_tripped": bool(m_s["max_dd"] <= -max_dd),
                "avg_exposure": float(sim["pos"].mean()),
            }
        )
        oos_parts.append(sim["ret"])
        bench_parts.append(bench)
        regime_parts.append(sigs[best]["regime"][test].shift(1))  # regime that set each day's position
        test_start = test_end

    if not folds:
        raise RuntimeError("not enough history for a single walk-forward period")

    oos, bench = pd.concat(oos_parts), pd.concat(bench_parts)
    regimes = pd.concat(regime_parts)
    by_regime = {}
    for name in REGIMES:
        mask = regimes == name
        if mask.any():
            by_regime[name] = {
                "days": int(mask.sum()),
                "strategy_return": float((1 + oos[mask]).prod() - 1),
                "benchmark_return": float((1 + bench[mask]).prod() - 1),
            }

    overall_s, overall_b = metrics(oos), metrics(bench)
    win_rate = sum(f["beat"] for f in folds) / len(folds)
    passed = overall_s[metric] > overall_b[metric] and win_rate >= g["min_fold_win_rate"]

    # parameters for live trading: fit on the most recent training window
    recent = close.index >= end - pd.DateOffset(years=g["train_years"])
    live_scores = [metrics(simulate(close[recent], s["exposure"][recent], cost, max_dd)["ret"])[metric] for s in sigs]

    return {
        "created": time.time(),
        "data_end": str(end.date()),
        "metric": metric,
        "passed": bool(passed),
        "fold_win_rate": win_rate,
        "strategy": overall_s,
        "benchmark": overall_b,
        "by_regime": by_regime,
        "folds": folds,
        "live_params": combos[int(np.nanargmax(live_scores))],
    }


def gate_path(cfg: dict) -> Path:
    return state_dir(cfg) / "gate.json"


def run_gate(close: pd.Series, cfg: dict) -> dict:
    result = walk_forward(close, cfg)
    gate_path(cfg).write_text(json.dumps(result, indent=2))
    return result


def load_or_run_gate(close: pd.Series, cfg: dict) -> dict:
    p = gate_path(cfg)
    if p.exists():
        cached = json.loads(p.read_text())
        if time.time() - cached["created"] < cfg["gate"]["max_age_days"] * 86400:
            return cached
    return run_gate(close, cfg)


def format_report(res: dict) -> str:
    m = res["metric"]
    pct = lambda x: f"{x * 100:7.2f}%"  # noqa: E731
    lines = [
        f"Walk-forward backtest ({len(res['folds'])} out-of-sample periods, data to {res['data_end']})",
        f"{'period':<25}{'strategy ' + m:>16}{'buy&hold ' + m:>16}{'strat DD':>10}  beat",
    ]
    for f in res["folds"]:
        lines.append(
            f"{f['test_start'] + '..' + f['test_end']:<25}{f['strategy'][m]:>16.2f}{f['benchmark'][m]:>16.2f}"
            f"{pct(f['strategy']['max_dd']):>10}  {'yes' if f['beat'] else 'no'}"
            + ("  [handbrake]" if f["handbrake_tripped"] else "")
        )
    s, b = res["strategy"], res["benchmark"]
    lines += [
        "",
        f"{'':<14}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>9}",
        f"{'strategy':<14}{pct(s['cagr']):>9}{s['sharpe']:>9.2f}{pct(s['max_dd']):>9}{s['calmar']:>9.2f}",
        f"{'buy & hold':<14}{pct(b['cagr']):>9}{b['sharpe']:>9.2f}{pct(b['max_dd']):>9}{b['calmar']:>9.2f}",
        "",
        "By regime (out-of-sample):",
    ]
    for name, r in res["by_regime"].items():
        lines.append(
            f"  {name:<10}{r['days']:>6} days   strategy {pct(r['strategy_return'])}   buy&hold {pct(r['benchmark_return'])}"
        )
    lines += [
        "",
        f"Periods beating buy & hold: {res['fold_win_rate'] * 100:.0f}%",
        f"GATE: {'PASSED - trading allowed' if res['passed'] else 'FAILED - bot will stay in cash'}",
        f"Live parameters: {res['live_params']}",
    ]
    return "\n".join(lines)
