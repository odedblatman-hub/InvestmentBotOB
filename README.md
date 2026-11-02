# InvestmentBotOB

A regime-switching trading bot for a single symbol (default `SPY`). It has a drawdown
handbrake, a kill switch, a walk-forward backtest that must pass before it trades, and a
nightly review that suggests changes but never applies them by itself.

**It runs in paper-trading mode by default.** Nothing here is financial advice. Backtests
overstate real results, so run it on paper for months before you consider real money.

## What it does

| Feature | Where | How |
|---|---|---|
| Reads the market regime | `tradebot/regime.py` | Each day is labeled **calm** (quiet trend), **sideways** (choppy, low efficiency ratio), **volatile** (realized vol in the top 20% of its 1-year range) or **crash** (deep drawdown below the 200-day trend, or a sharp drop while vol is extreme). |
| Uses a different strategy per regime | `tradebot/regime.py` | calm → trend following · sideways → mean reversion (Bollinger z-score) · volatile → volatility targeting, only above the trend · crash → cash. Long-only, no leverage. |
| Handbrake | `tradebot/risk.py` | When equity falls 15% below its peak, the bot halts. After that it can only exit positions, and a human must run `reset` before it trades again. |
| Kill switch | `tradebot/risk.py` | `python -m tradebot kill` creates `state/KILL`. Every order is checked against it right before it is sent, inside `GuardedBroker`, so code cannot bypass the check. |
| Pre-order checks | `tradebot/risk.py` | Size per order (≤50% of equity, and large orders are split), size per position (≤100%), no shorting, cash available, sane price and quantity. |
| Backtest gate | `tradebot/backtest.py` | Walk-forward: parameters are chosen on 3 years of data, then tested on the following year, period by period. If the strategy doesn't beat buy & hold out of sample (Sharpe by default, in ≥50% of periods and overall), the bot **stays in cash**. |
| Night shift | `tradebot/nightly.py` | Lists what went wrong (gate failures, regimes that lagged, handbrake trips, blocked orders), tests single-parameter changes and writes `proposals/<date>.md`. **You** decide whether to apply them. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.toml config.toml      # edit the symbol, cash, limits, etc.
python -m pytest -q                     # 12 tests
```

## Daily use

```bash
python -m tradebot backtest      # walk-forward report + gate verdict (PASSED / FAILED)
python -m tradebot regime        # today's regime and target exposure
python -m tradebot run           # one trading cycle: run daily after the close
python -m tradebot nightly       # review + proposals (writes proposals/YYYYMMDD.md)
python -m tradebot apply 20261102-1   # apply a proposal you approved (config gets backed up)
python -m tradebot status        # handbrake / kill switch state
python -m tradebot kill "reason" # stop everything immediately
python -m tradebot reset         # human decision: clear handbrake + kill switch
```

Every cycle is logged to `state/journal.jsonl`. To automate the daily run and the night
shift, see `scripts/crontab.example`.

## Using a real broker (Alpaca)

Set `broker.kind = "alpaca"` and export `ALPACA_API_KEY` / `ALPACA_API_SECRET`. The bot
uses Alpaca's **paper** endpoint and refuses a live URL unless you also set
`broker.allow_live = true`. The Alpaca adapter is small and has not been tested against
the live API, so test it on paper first.

## Data

Prices come from `yfinance` by default. To use your own data, set `data_source = "csv"`
and point `csv_path` at a file with `Date,Close` columns (adjusted closes).

## Honest limitations

- The gate is often **FAILED** for broad indexes. Beating buy & hold on a risk-adjusted
  basis is hard, and in that case the bot sits in cash. That is by design.
- In the backtest, the handbrake resets at the start of each test period (as if you ran
  `reset`), so drawdowns across stitched periods can exceed 15%. In live trading, it
  stays halted until you reset it.
- The night shift tries many tweaks on the same history. That is a form of overfitting,
  so treat its proposals as hypotheses, not upgrades.
- It trades one symbol, daily, with market orders. There is no intraday logic, no
  taxes, and no handling of corporate actions beyond adjusted prices.
