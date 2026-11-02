"""Risk manager: the handbrake (max drawdown) and the kill switch.

`check_order` is called by GuardedBroker immediately before EVERY order is sent.
  * Kill switch file present -> every order is rejected.
  * Drawdown from peak equity >= max_drawdown -> the bot halts (persisted). While halted,
    only orders that reduce the position are allowed, so it can get flat but never add risk.
    A human has to run `python -m tradebot reset` to resume.
  * Size limits per order and per position, and basic sanity checks.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from .config import state_dir


@dataclass
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    qty: int


@dataclass
class Account:
    equity: float
    cash: float
    position_qty: int


@dataclass
class Decision:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg["risk"]
        self.dir = state_dir(cfg)
        self.state_file = self.dir / "risk_state.json"
        self.kill_file = self.dir / self.cfg["kill_switch_file"]
        self.state = {"peak_equity": None, "halted": False, "halt_reason": None, "halted_at": None}
        if self.state_file.exists():
            self.state.update(json.loads(self.state_file.read_text()))

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=2))

    # --- kill switch -------------------------------------------------------------
    def kill_switch_engaged(self) -> bool:
        return self.kill_file.exists()

    def engage_kill_switch(self, reason: str = "manual") -> Path:
        self.kill_file.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason}\n")
        return self.kill_file

    def reset(self, equity: float | None = None) -> None:
        """Human-initiated: clear the halt and kill switch, and re-base the peak."""
        if self.kill_file.exists():
            self.kill_file.unlink()
        self.state.update(halted=False, halt_reason=None, halted_at=None, peak_equity=equity)
        self._save()

    # --- handbrake ---------------------------------------------------------------
    def update_equity(self, equity: float) -> float:
        """Track peak equity; trip the handbrake on breach. Returns current drawdown."""
        peak = self.state["peak_equity"]
        if peak is None or equity > peak:
            peak = self.state["peak_equity"] = equity
        dd = equity / peak - 1 if peak > 0 else 0.0
        if dd <= -self.cfg["max_drawdown"] and not self.state["halted"]:
            self.state.update(
                halted=True,
                halt_reason=f"drawdown {dd:.1%} breached limit {-self.cfg['max_drawdown']:.0%}",
                halted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        self._save()
        return dd

    @property
    def halted(self) -> bool:
        return bool(self.state["halted"])

    # --- the pre-order check -----------------------------------------------------
    def check_order(self, order: Order, account: Account, price: float) -> Decision:
        if self.kill_switch_engaged():
            return Decision(False, f"kill switch engaged ({self.kill_file})")
        if order.side not in ("buy", "sell"):
            return Decision(False, f"bad side {order.side!r}")
        if not (isinstance(order.qty, int) and order.qty > 0):
            return Decision(False, f"bad quantity {order.qty!r}")
        if not (math.isfinite(price) and price > 0):
            return Decision(False, f"bad price {price!r}")
        if not (math.isfinite(account.equity) and account.equity > 0):
            return Decision(False, f"bad account equity {account.equity!r}")

        self.update_equity(account.equity)
        reduces = order.side == "sell" and order.qty <= account.position_qty
        if self.halted and not reduces:
            return Decision(False, f"handbrake: trading halted ({self.state['halt_reason']}); only exits allowed")
        if order.side == "sell" and not reduces:
            return Decision(False, "short selling is not allowed")

        value = order.qty * price
        if not reduces and value > self.cfg["max_order_pct"] * account.equity + 1e-9:
            return Decision(False, f"order ${value:,.0f} exceeds {self.cfg['max_order_pct']:.0%} of equity")
        if order.side == "buy":
            new_pos_value = (account.position_qty + order.qty) * price
            if new_pos_value > self.cfg["max_position_pct"] * account.equity + 1e-9:
                return Decision(False, f"position would exceed {self.cfg['max_position_pct']:.0%} of equity")
            if value > account.cash + 1e-9:
                return Decision(False, f"insufficient cash (${account.cash:,.0f} < ${value:,.0f})")
        return Decision(True, "ok")
