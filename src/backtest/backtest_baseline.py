#!/usr/bin/env python3
"""
backtest_baseline.py

Part 2 / Phase 3 backtest harness: replay the REAL captured mid-price
path chronologically, compute the Avellaneda-Stoikov-optimal reservation
price and spread at every step (`src/model/avellaneda_stoikov.py`), post
a hypothetical bid/ask at those levels, and check whether the REAL
captured trade tape actually crossed them. A real trade print at or
beyond our quoted price counts as a hypothetical fill.

This is deliberately "naive" in exactly the sense the README's Phase 3
asks for: it is not a matched-queue-position, latency-aware execution
simulator (that level of realism is explicitly deferred to Phase 5,
which the roadmap says should use the sister hawkes-fill-probability
project's fill model instead of a naive always-fill-if-crossed
assumption). What it *is*: a real reference inventory path and a real
naive PnL series, driven end to end by real market data rather than a
synthetic price/order simulation -- which is what Part 2 asks for.

Fill logic per depth-tick step:
  - We are quoting bid = r - spread/2, ask = r + spread/2 at reservation
    price r and total spread from the AS formulas, using the mid s at
    the START of the step (we don't get to see the step's own trades
    before quoting -- quotes are set on last-known state, same causality
    a real quoting loop would respect).
  - Any real trade print with local_ts in (step_start, step_end] whose
    price <= our bid is treated as a hypothetical BUY fill for us (we
    buy 1 unit at our quoted bid price -- trade price at/through our bid
    means an aggressive seller swept down to or past where we were
    resting).
  - Any real trade print in the same window with price >= our ask is a
    hypothetical SELL fill for us (aggressive buyer swept up to/past our
    ask).
  - Fill size is fixed at 1 unit per fill (matches DERIVATION.md's
    "each fill is one unit of the asset, for simplicity" assumption) --
    NOT the real trade's own quantity, to stay consistent with the model
    as derived. A step where multiple trades cross the same side only
    counts the first crossing as a fill (a single resting order can only
    be hit once before it needs to be replaced by a fresh quote at the
    next step's new reservation price).
  - No inventory cap is enforced (matches the base AS derivation, which
    has no |q|<=Q constraint -- that's a GLFT-era refinement, not part
    of this baseline).

Outputs (data/processed/, gitignored -- code only is versioned):
  - backtest_series.csv: per-step s, r, bid, ask, spread, q, cash, pnl,
    fill markers.
  - Summary stats printed to stdout and returned by main().
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.avellaneda_stoikov import ASParams, quotes  # noqa: E402
from src.model.estimate_params import load_mid_series, load_trades  # noqa: E402


def run_backtest(depth_path: str, trades_path: str, params: ASParams):
    times, mids = load_mid_series(depth_path)
    trades = load_trades(trades_path)
    if len(mids) < 2:
        raise ValueError("Not enough mid-price observations to backtest")

    t0 = times[0]
    t_end_horizon = t0 + params.T  # AS horizon T anchored to capture start

    # Pre-sort trades by ts (already chronological from the capture, but
    # be defensive) and use a moving pointer so each trade is only
    # consumed by the one step window it falls in.
    trades_sorted = sorted(trades, key=lambda tr: tr["ts"])
    tr_idx = 0
    n_trades = len(trades_sorted)

    q = 0.0            # inventory, units of asset
    cash = 0.0         # cash, $
    rows = []
    n_bid_fills = 0
    n_ask_fills = 0

    for i in range(len(mids) - 1):
        step_start = times[i]
        step_end = times[i + 1]
        s = mids[i]
        # t measured in seconds since capture start, clipped to [0, T]
        # once we run past the design horizon T we keep quoting at the
        # boundary case tau=0 (pure myopic markup, no inventory-risk
        # widening) rather than extrapolating the formula past its
        # intended domain.
        t_elapsed = min(step_start - t0, params.T)

        bid, ask, r, spread = quotes(s, q, t_elapsed, params)

        bid_filled = False
        ask_filled = False
        while tr_idx < n_trades and trades_sorted[tr_idx]["ts"] <= step_end:
            tr = trades_sorted[tr_idx]
            if trades_sorted[tr_idx]["ts"] > step_start:
                price = tr["price"]
                if not bid_filled and price <= bid:
                    q += 1.0
                    cash -= bid
                    bid_filled = True
                    n_bid_fills += 1
                elif not ask_filled and price >= ask:
                    q -= 1.0
                    cash += ask
                    ask_filled = True
                    n_ask_fills += 1
            tr_idx += 1

        mtm = cash + q * s  # mark-to-market wealth using this step's mid
        rows.append({
            "local_ts": step_start,
            "t_elapsed": t_elapsed,
            "mid": s,
            "reservation_price": r,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "inventory": q,
            "cash": cash,
            "mtm_pnl": mtm,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
        })

    final_mid = mids[-1]
    final_mtm = cash + q * final_mid

    summary = {
        "n_steps": len(rows),
        "capture_span_sec": times[-1] - times[0],
        "n_bid_fills": n_bid_fills,
        "n_ask_fills": n_ask_fills,
        "n_fills_total": n_bid_fills + n_ask_fills,
        "final_inventory": q,
        "inventory_min": min(r["inventory"] for r in rows) if rows else 0.0,
        "inventory_max": max(r["inventory"] for r in rows) if rows else 0.0,
        "final_cash": cash,
        "final_mid": final_mid,
        "final_mtm_pnl": final_mtm,
        "reservation_price_min": min(r["reservation_price"] for r in rows) if rows else None,
        "reservation_price_max": max(r["reservation_price"] for r in rows) if rows else None,
        "spread_min": min(r["spread"] for r in rows) if rows else None,
        "spread_max": max(r["spread"] for r in rows) if rows else None,
        "spread_mean": (sum(r["spread"] for r in rows) / len(rows)) if rows else None,
        "params": {
            "gamma": params.gamma, "sigma": params.sigma,
            "kappa": params.kappa, "A": params.A, "T": params.T,
        },
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--params-json", default="data/processed/estimated_params.json",
                     help="Output of estimate_params.py")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    with open(args.params_json) as f:
        p = json.load(f)["params"]
    params = ASParams(gamma=p["gamma"], sigma=p["sigma"], kappa=p["kappa"],
                       A=p["A"], T=p["T"])

    print("[params] " + ", ".join(f"{k}={v}" for k, v in vars(params).items()))

    rows, summary = run_backtest(args.depth, args.trades, params)

    os.makedirs(args.out_dir, exist_ok=True)
    series_path = os.path.join(args.out_dir, "backtest_series.csv")
    with open(series_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(args.out_dir, "backtest_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== Baseline Avellaneda-Stoikov backtest summary =====")
    for k, v in summary.items():
        if k == "params":
            continue
        if isinstance(v, float):
            print(f"  {k:28s}: {v:,.6f}")
        else:
            print(f"  {k:28s}: {v}")
    print(f"\nWrote {series_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
