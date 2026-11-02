"""Brokers. The bot only ever talks to GuardedBroker, which runs the risk check first."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .config import state_dir
from .risk import Account, Decision, Order, RiskManager


class PaperBroker:
    """Simulated account stored in state/paper_account.json. Fills at last price +/- slippage."""

    def __init__(self, cfg: dict):
        self.path: Path = state_dir(cfg) / "paper_account.json"
        self.slippage = cfg["broker"]["slippage_bps"] / 1e4
        self.symbol = cfg["general"]["symbol"]
        if self.path.exists():
            self.acct = json.loads(self.path.read_text())
        else:
            self.acct = {"cash": float(cfg["broker"]["initial_cash"]), "qty": 0}
            self._save()

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.acct, indent=2))

    def account(self, price: float) -> Account:
        return Account(self.acct["cash"] + self.acct["qty"] * price, self.acct["cash"], self.acct["qty"])

    def submit(self, order: Order, price: float) -> dict:
        fill = price * (1 + self.slippage if order.side == "buy" else 1 - self.slippage)
        sign = 1 if order.side == "buy" else -1
        self.acct["cash"] -= sign * order.qty * fill
        self.acct["qty"] += sign * order.qty
        self._save()
        return {"status": "filled", "fill_price": fill, "qty": order.qty, "side": order.side}


class AlpacaBroker:
    """Minimal Alpaca REST adapter (market orders, whole shares).

    Keys are read from the ALPACA_API_KEY / ALPACA_API_SECRET environment variables.
    Points at the paper endpoint unless broker.allow_live = true.
    """

    def __init__(self, cfg: dict):
        b = cfg["broker"]
        self.base = b["alpaca_base_url"].rstrip("/")
        if "paper-api" not in self.base and not b["allow_live"]:
            raise SystemExit("Refusing live Alpaca URL: set broker.allow_live = true if you really mean it.")
        self.key = os.environ.get("ALPACA_API_KEY")
        self.secret = os.environ.get("ALPACA_API_SECRET")
        if not (self.key and self.secret):
            raise SystemExit("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables.")
        self.symbol = cfg["general"]["symbol"]

    def _req(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(
            self.base + path,
            method=method,
            data=json.dumps(body).encode() if body else None,
            headers={"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as e:
            if e.code == 404 and path.startswith("/v2/positions/"):
                return None
            raise

    def account(self, price: float) -> Account:
        a = self._req("GET", "/v2/account")
        pos = self._req("GET", f"/v2/positions/{self.symbol}")
        qty = int(float(pos["qty"])) if pos else 0
        return Account(float(a["equity"]), float(a["cash"]), qty)

    def submit(self, order: Order, price: float) -> dict:
        return self._req(
            "POST",
            "/v2/orders",
            {"symbol": order.symbol, "qty": str(order.qty), "side": order.side, "type": "market", "time_in_force": "day"},
        )


class GuardedBroker:
    """Wraps a broker so that no order can reach it without passing RiskManager.check_order."""

    def __init__(self, broker, risk: RiskManager):
        self._broker = broker
        self.risk = risk

    def account(self, price: float) -> Account:
        return self._broker.account(price)

    def submit(self, order: Order, price: float) -> tuple[Decision, dict | None]:
        account = self._broker.account(price)  # fresh numbers for the check
        decision = self.risk.check_order(order, account, price)
        if not decision.allowed:
            return decision, None
        return decision, self._broker.submit(order, price)


def make_broker(cfg: dict):
    kind = cfg["broker"]["kind"]
    if kind == "paper":
        return PaperBroker(cfg)
    if kind == "alpaca":
        return AlpacaBroker(cfg)
    raise ValueError(f"unknown broker kind {kind!r}")
