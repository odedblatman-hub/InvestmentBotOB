"""Command line: python -m tradebot <command>"""
from __future__ import annotations

import argparse
import json
import sys

from . import backtest, bot, nightly
from .config import load_config
from .data import load_prices
from .regime import signals
from .risk import RiskManager


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tradebot", description="Regime-switching trading bot (paper by default).")
    ap.add_argument("-c", "--config", default="config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backtest", help="walk-forward backtest + gate verdict")
    sub.add_parser("regime", help="show today's regime and target exposure")
    sub.add_parser("run", help="one trading cycle (run daily after the close)")
    sub.add_parser("nightly", help="review and write improvement proposals (never applies them)")
    a = sub.add_parser("apply", help="apply a nightly proposal to the config")
    a.add_argument("proposal_id")
    sub.add_parser("status", help="risk state, kill switch, account")
    k = sub.add_parser("kill", help="engage the kill switch: block every order")
    k.add_argument("reason", nargs="?", default="manual")
    sub.add_parser("reset", help="clear handbrake + kill switch (human decision)")
    args = ap.parse_args(argv)

    if args.cmd == "apply":
        print(nightly.apply_proposal(args.config, args.proposal_id))
        return 0

    cfg = load_config(args.config)

    if args.cmd == "kill":
        print(f"Kill switch engaged: {RiskManager(cfg).engage_kill_switch(args.reason)}")
        return 0
    if args.cmd == "reset":
        RiskManager(cfg).reset()
        print("Handbrake and kill switch cleared. Peak equity will re-base on the next run.")
        return 0
    if args.cmd == "status":
        r = RiskManager(cfg)
        print(json.dumps({"kill_switch": r.kill_switch_engaged(), **r.state}, indent=2))
        return 0

    close = load_prices(cfg)
    if args.cmd == "backtest":
        res = backtest.run_gate(close, cfg)
        print(backtest.format_report(res))
    elif args.cmd == "regime":
        gate = backtest.load_or_run_gate(close, cfg)
        last = signals(close, cfg, gate["live_params"]).iloc[-1]
        print(f"{close.index[-1].date()}  {cfg['general']['symbol']}  close {last['close']:.2f}")
        print(f"regime: {last['regime']}   target exposure: {last['exposure']:.0%}")
        print(f"vol {last['vol']:.1%} (rank {last['vol_rank']:.0%}), drawdown {last['dd']:.1%}, "
              f"efficiency {last['er']:.2f}, z {last['z']:.2f}")
    elif args.cmd == "run":
        print(json.dumps(bot.run_once(cfg, close), indent=2, default=str))
    elif args.cmd == "nightly":
        report, proposals = nightly.review(cfg, close)
        print(report.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
