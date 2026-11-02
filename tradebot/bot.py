"""One trading cycle. Run it once a day after the close (see scripts/crontab.example)."""
from __future__ import annotations

import json
import math
import time

import pandas as pd

from .backtest import load_or_run_gate
from .broker import GuardedBroker, make_broker
from .config import state_dir
from .regime import signals
from .risk import Order, RiskManager


def journal(cfg: dict, entry: dict) -> None:
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with open(state_dir(cfg) / "journal.jsonl", "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def run_once(cfg: dict, close: pd.Series, broker=None) -> dict:
    symbol = cfg["general"]["symbol"]
    risk = RiskManager(cfg)
    if risk.kill_switch_engaged():
        entry = {"action": "skip", "reason": "kill switch engaged"}
        journal(cfg, entry)
        return entry

    guarded = GuardedBroker(broker or make_broker(cfg), risk)
    price = float(close.iloc[-1])
    gate = load_or_run_gate(close, cfg)

    sig = signals(close, cfg, gate["live_params"]).iloc[-1]
    target = float(sig["exposure"])
    notes = []
    if not gate["passed"]:
        if cfg["gate"]["enforce"]:
            target = 0.0
            notes.append("gate failed: strategy did not beat buy & hold out-of-sample -> staying in cash")
        else:
            notes.append("WARNING gate failed but gate.enforce = false; trading anyway")

    account = guarded.account(price)
    dd = risk.update_equity(account.equity)
    if risk.halted:
        target = 0.0
        notes.append(f"handbrake: {risk.state['halt_reason']} -> exiting, no new buys until `reset`")

    target_qty = math.floor(target * account.equity / price)
    delta = target_qty - account.position_qty
    entry = {
        "action": "hold",
        "date": str(close.index[-1].date()),
        "symbol": symbol,
        "price": price,
        "regime": sig["regime"],
        "target_exposure": target,
        "equity": account.equity,
        "drawdown": dd,
        "position_qty": account.position_qty,
        "target_qty": target_qty,
        "gate_passed": gate["passed"],
        "notes": notes,
    }
    if delta != 0:
        # Split into slices no bigger than max_order_pct; each slice is risk-checked on its own.
        side = "buy" if delta > 0 else "sell"
        slice_max = max(1, math.floor(cfg["risk"]["max_order_pct"] * account.equity / price))
        remaining, orders = abs(delta), []
        while remaining > 0:
            order = Order(symbol, side, min(remaining, slice_max))
            decision, fill = guarded.submit(order, price)
            orders.append({"side": side, "qty": order.qty, "risk_check": decision.reason, "fill": fill})
            if not decision.allowed:
                break
            remaining -= order.qty
        filled = all(o["fill"] for o in orders)
        entry.update(action="order" if filled else ("partial" if orders[0]["fill"] else "rejected"), orders=orders)
    journal(cfg, entry)
    return entry
