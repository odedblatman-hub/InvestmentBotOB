import copy
import json
import tomllib

import numpy as np
import pandas as pd
import pytest

from tradebot import backtest, bot, nightly
from tradebot.broker import GuardedBroker, PaperBroker
from tradebot.config import DEFAULTS, dump_toml, load_config
from tradebot.regime import signals
from tradebot.risk import Account, Order, RiskManager


def synthetic_prices(seed=0, years=12):
    """Regime-switching random walk: calm uptrends, chop, jittery spells and crashes."""
    rng = np.random.default_rng(seed)
    segs = []
    for _ in range(years):
        segs += [(0.0006, 0.007, 120), (0.0, 0.010, 60), (0.0002, 0.022, 40), (0.0005, 0.008, 25)]
        if rng.random() < 0.35:
            segs.append((-0.006, 0.030, 30))  # freefall
    rets = np.concatenate([rng.normal(mu, sd, n) for mu, sd, n in segs])
    idx = pd.bdate_range("2008-01-01", periods=len(rets))
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx, name="close")


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(DEFAULTS)
    c["general"]["state_dir"] = str(tmp_path / "state")
    c["general"]["proposals_dir"] = str(tmp_path / "proposals")
    return c


@pytest.fixture(scope="module")
def close():
    return synthetic_prices()


def test_no_lookahead(cfg, close):
    full = signals(close, cfg)
    cut = signals(close.iloc[:1500], cfg)
    pd.testing.assert_frame_equal(full.iloc[:1500], cut)


def test_regimes_detected(cfg, close):
    reg = signals(close, cfg)["regime"]
    counts = reg.value_counts()
    for name in ("calm", "sideways", "volatile", "crash"):
        assert counts.get(name, 0) > 20, counts
    # in a crash the bot should hold no exposure
    assert (signals(close, cfg)["exposure"][reg == "crash"] == 0).all()


def test_simulate_uses_previous_day_signal():
    close = pd.Series([100.0, 110.0, 121.0], index=pd.bdate_range("2020-01-01", periods=3))
    exp = pd.Series([0.0, 1.0, 1.0], index=close.index)
    sim = backtest.simulate(close, exp, cost_bps=0, max_dd=None)
    assert sim["ret"].tolist() == pytest.approx([0.0, 0.0, 0.10])  # day-1 gain not captured


def test_simulated_handbrake_goes_to_cash():
    close = pd.Series([100, 90, 80, 70, 100, 120.0], index=pd.bdate_range("2020-01-01", periods=6))
    sim = backtest.simulate(close, pd.Series(1.0, index=close.index), cost_bps=0, max_dd=0.15)
    assert sim["pos"].iloc[-2:].tolist() == [0.0, 0.0]


def test_walk_forward_report(cfg, close):
    res = backtest.walk_forward(close, cfg)
    assert len(res["folds"]) >= 8
    assert isinstance(res["passed"], bool)
    assert "GATE" in backtest.format_report(res)


def test_kill_switch_blocks_everything(cfg):
    r = RiskManager(cfg)
    r.engage_kill_switch("test")
    acct = Account(100_000, 50_000, 100)
    assert not r.check_order(Order("SPY", "sell", 10), acct, 500).allowed
    assert not r.check_order(Order("SPY", "buy", 1), acct, 500).allowed
    r.reset()
    assert r.check_order(Order("SPY", "buy", 1), acct, 500).allowed


def test_handbrake_halts_buys_allows_exits(cfg):
    r = RiskManager(cfg)
    r.update_equity(100_000)
    acct = Account(84_000, 40_000, 100)  # -16%
    d = r.check_order(Order("SPY", "buy", 1), acct, 440)
    assert not d.allowed and "handbrake" in d.reason
    assert r.check_order(Order("SPY", "sell", 100), acct, 440).allowed
    assert RiskManager(cfg).halted  # persisted across restarts


def test_size_limits(cfg):
    r = RiskManager(cfg)
    acct = Account(100_000, 100_000, 0)
    assert not r.check_order(Order("SPY", "buy", 120), acct, 500).allowed  # 60% > 50% per order
    assert not r.check_order(Order("SPY", "sell", 5), acct, 500).allowed  # no shorting
    assert r.check_order(Order("SPY", "buy", 100), acct, 500).allowed


def test_run_once_slices_orders_and_respects_gate(cfg, close):
    gate = backtest.run_gate(close, cfg)
    gate.update(passed=True, live_params={})
    backtest.gate_path(cfg).write_text(json.dumps(gate))
    # find a date where the strategy wants to be fully invested
    sig = signals(close, cfg)
    day = sig.index[(sig["exposure"] == 1.0) & (sig.index > sig.index[600])][0]
    entry = bot.run_once(cfg, close[:day])
    assert entry["action"] == "order" and len(entry["orders"]) >= 2
    acct = PaperBroker(cfg).account(close[day])
    assert acct.position_qty == entry["target_qty"]

    gate["passed"] = False
    backtest.gate_path(cfg).write_text(json.dumps(gate))
    entry = bot.run_once(cfg, close[:day])
    assert entry["target_qty"] == 0 and PaperBroker(cfg).account(close[day]).position_qty == 0

    RiskManager(cfg).engage_kill_switch()
    assert bot.run_once(cfg, close[:day])["action"] == "skip"


def test_guarded_broker_never_bypasses_risk(cfg):
    r = RiskManager(cfg)
    r.engage_kill_switch()
    g = GuardedBroker(PaperBroker(cfg), r)
    decision, fill = g.submit(Order("SPY", "buy", 1), 100.0)
    assert not decision.allowed and fill is None
    assert PaperBroker(cfg).account(100.0).position_qty == 0


def test_nightly_proposes_but_does_not_apply(cfg, close, tmp_path):
    cfg["nightly"]["min_improvement"] = -99  # force proposals for the test
    cfg["gate"]["min_fold_win_rate"] = 0.0
    cfg["gate"]["grid"] = {}
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(dump_toml(cfg))
    before = cfg_path.read_text()
    report, proposals = nightly.review(load_config(cfg_path), close)
    assert report.exists() and cfg_path.read_text() == before
    if proposals:
        msg = nightly.apply_proposal(str(cfg_path), proposals[0]["id"])
        assert "Applied" in msg
        new = tomllib.loads(cfg_path.read_text())
        section, key = proposals[0]["key"].split(".")
        assert new[section][key] == proposals[0]["to"]


def test_toml_roundtrip():
    assert tomllib.loads(dump_toml(DEFAULTS)) == DEFAULTS
