"""Night shift: review what went wrong and write improvement PROPOSALS. Never applies them.

It re-runs the walk-forward test with the current config, looks at the journal, then tries
one-parameter tweaks. A tweak becomes a proposal only if it improves the out-of-sample
metric by at least nightly.min_improvement, still passes the gate, and does not make the
worst drawdown deeper. You review proposals/<date>.md and apply one with
`python -m tradebot apply <id>`.

Caveat: trying many tweaks on the same history is itself a form of overfitting. Treat
proposals as hypotheses, and prefer ones with a clear reason behind them.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pandas as pd

from .backtest import walk_forward
from .config import get_dotted, load_config, save_config, set_dotted, state_dir

# (dotted key, candidate values)
SEARCH_SPACE: list[tuple[str, list]] = [
    ("regime.vol_high", [0.70, 0.75, 0.85, 0.90]),
    ("regime.er_low", [0.15, 0.20, 0.30, 0.35]),
    ("regime.crash_dd", [0.10, 0.12, 0.20]),
    ("regime.crash_mom", [0.07, 0.15]),
    ("strategy.mr_scale", [0.15, 0.35]),
    ("strategy.bb_window", [10, 40]),
]


def _read_journal(cfg: dict, days: int = 30) -> list[dict]:
    p = state_dir(cfg) / "journal.jsonl"
    if not p.exists():
        return []
    cutoff = time.time() - days * 86400
    out = []
    for line in p.read_text().splitlines():
        e = json.loads(line)
        if time.mktime(time.strptime(e["ts"], "%Y-%m-%d %H:%M:%S")) >= cutoff:
            out.append(e)
    return out


def diagnose(base: dict, entries: list[dict]) -> list[str]:
    findings = []
    m = base["metric"]
    if not base["passed"]:
        findings.append(
            f"Gate is FAILING: OOS {m} {base['strategy'][m]:.2f} vs buy & hold {base['benchmark'][m]:.2f}, "
            f"{base['fold_win_rate']:.0%} of periods won."
        )
    worst = sorted(
        base["by_regime"].items(), key=lambda kv: kv[1]["strategy_return"] - kv[1]["benchmark_return"]
    )
    for name, r in worst[:2]:
        gap = r["strategy_return"] - r["benchmark_return"]
        if gap < 0:
            findings.append(
                f"Regime '{name}' lagged buy & hold by {gap:.1%} over {r['days']} days "
                f"({r['strategy_return']:.1%} vs {r['benchmark_return']:.1%})."
            )
    tripped = [f for f in base["folds"] if f["handbrake_tripped"]]
    if tripped:
        findings.append("Handbrake tripped in: " + ", ".join(f["test_start"][:4] for f in tripped) + ".")
    rejected = [e for e in entries if e.get("action") in ("rejected", "partial")]
    if rejected:
        reasons = {o["risk_check"] for e in rejected for o in e.get("orders", []) if o["risk_check"] != "ok"}
        findings.append(f"{len(rejected)} order(s) blocked by risk checks in the last 30 days: {sorted(reasons)}")
    halted = [e for e in entries if any("handbrake" in n for n in e.get("notes", []))]
    if halted:
        findings.append("Bot is currently halted by the handbrake; it needs a human `reset` to resume.")
    if not findings:
        findings.append("No problems found.")
    return findings


def review(cfg: dict, close: pd.Series) -> tuple[Path, list[dict]]:
    base = walk_forward(close, cfg)
    metric = base["metric"]
    findings = diagnose(base, _read_journal(cfg))

    candidates = []
    for key, values in SEARCH_SPACE:
        current = get_dotted(cfg, key)
        for v in values:
            if v == current:
                continue
            trial = copy.deepcopy(cfg)
            set_dotted(trial, key, v)
            res = walk_forward(close, trial)
            gain = res["strategy"][metric] - base["strategy"][metric]
            if (
                gain >= cfg["nightly"]["min_improvement"]
                and res["passed"]
                and res["strategy"]["max_dd"] >= base["strategy"]["max_dd"] - 1e-9
            ):
                candidates.append(
                    {"key": key, "from": current, "to": v, "gain": gain, "oos": res["strategy"], "win_rate": res["fold_win_rate"]}
                )

    # best tweak per parameter, then top N overall
    best: dict[str, dict] = {}
    for c in candidates:
        if c["key"] not in best or c["gain"] > best[c["key"]]["gain"]:
            best[c["key"]] = c
    proposals = sorted(best.values(), key=lambda c: -c["gain"])[: cfg["nightly"]["max_proposals"]]
    stamp = time.strftime("%Y%m%d")
    for i, p in enumerate(proposals, 1):
        p["id"] = f"{stamp}-{i}"

    out_dir = Path(cfg["general"]["proposals_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stamp}.json").write_text(json.dumps({"base": base["strategy"], "proposals": proposals}, indent=2))

    md = [f"# Night-shift review {time.strftime('%Y-%m-%d')}", "", f"Data through {base['data_end']}.", "", "## What went wrong", ""]
    md += [f"- {f}" for f in findings]
    md += ["", f"## Current config (out-of-sample)", "",
           f"{metric}: {base['strategy'][metric]:.2f} (buy & hold {base['benchmark'][metric]:.2f}), "
           f"CAGR {base['strategy']['cagr']:.1%}, max drawdown {base['strategy']['max_dd']:.1%}, "
           f"gate {'passed' if base['passed'] else 'FAILED'}", "", "## Proposed changes (NOT applied)", ""]
    if proposals:
        for p in proposals:
            md.append(
                f"- **{p['id']}**: `{p['key']}` {p['from']} -> {p['to']}  |  {metric} +{p['gain']:.2f} "
                f"-> {p['oos'][metric]:.2f}, CAGR {p['oos']['cagr']:.1%}, max DD {p['oos']['max_dd']:.1%}, "
                f"periods won {p['win_rate']:.0%}"
            )
        md += ["", "Apply one with: `python -m tradebot apply <id>`  (your config is backed up first)."]
    else:
        md.append("No single-parameter change cleared the improvement bar. Nothing to apply.")
    report = out_dir / f"{stamp}.md"
    report.write_text("\n".join(md) + "\n")
    return report, proposals


def apply_proposal(config_path: str, proposal_id: str) -> str:
    stamp = proposal_id.split("-")[0]
    cfg = load_config(config_path)
    pfile = Path(cfg["general"]["proposals_dir"]) / f"{stamp}.json"
    if not pfile.exists():
        raise SystemExit(f"no proposals file {pfile}")
    match = [p for p in json.loads(pfile.read_text())["proposals"] if p["id"] == proposal_id]
    if not match:
        raise SystemExit(f"proposal {proposal_id} not found in {pfile}")
    p = match[0]
    set_dotted(cfg, p["key"], p["to"])
    backup = save_config(cfg, config_path)
    gate = state_dir(cfg) / "gate.json"
    if gate.exists():
        gate.unlink()  # force a fresh backtest with the new config before the next trade
    return f"Applied {proposal_id}: {p['key']} {p['from']} -> {p['to']}" + (f" (backup: {backup})" if backup else "")
