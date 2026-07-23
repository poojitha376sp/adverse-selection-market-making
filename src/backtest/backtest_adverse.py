#!/usr/bin/env python3
"""
backtest_adverse.py

Part 3 / Phase 4: three-way backtest -- baseline Avellaneda-Stoikov vs.
the heuristic-overlay and principled adverse-selection variants
(`src/model/avellaneda_stoikov_adverse.py`) -- on the SAME real captured
data, using the SAME chronological replay-and-check-for-real-fills
methodology as Part 2's `backtest_baseline.py` (the fill logic below is a
direct generalization of `backtest_baseline.run_backtest`, parametrized
by a `quote_fn` so all three variants run through identical mechanics --
same causality, same "1 unit per fill, one fill per side per step" rule,
same no-inventory-cap baseline behaviour. Only the quote-generation
function differs between the three runs).

In addition to Part 2's summary stats, this harness also reports the
REALIZED ADVERSE-SELECTION COST per variant -- the number the whole
extension is supposed to move: for every hypothetical fill, look at the
mid-price `horizon_sec` seconds later and charge the maker the loss they
would have avoided by not trading, i.e.

    bid fill (we bought at `bid`):  cost = max(0, bid - mid(t+H))
    ask fill (we sold at `ask`)  :  cost = max(0, mid(t+H) - ask)

summed over all fills. This directly operationalizes README's repeated
definition: "losses on fills immediately followed by adverse price
moves." `horizon_sec` defaults to the same horizon used to LABEL the ML
classifier's training target (2s), so "adverse" means the same thing on
both sides of this project.

The p_informed(t) signal used by the heuristic/principled variants is
looked up CAUSALLY: at backtest step time t, we use the most recent
`ml_informedness_classifier.py`-produced score at or before t (an
as-of/backward lookup) -- never a future score -- consistent with how a
live quoting loop would actually have access to the signal.

Usage:
    python src/backtest/backtest_adverse.py \
        --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
        --trades data/raw/trades_btcusdt_20260723_094653.jsonl \
        --params-json data/processed/estimated_params.json \
        --scores-csv data/processed/ml_informedness_scores.csv \
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from bisect import bisect_right

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.avellaneda_stoikov import ASParams, quotes as baseline_quotes  # noqa: E402
from src.model.avellaneda_stoikov_adverse import (  # noqa: E402
    AdverseParams,
    heuristic_quotes,
    principled_quotes,
)
from src.model.estimate_params import load_mid_series, load_trades  # noqa: E402


def load_p_informed_series(scores_csv: str):
    times, probs = [], []
    with open(scores_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["local_ts"]))
            probs.append(float(row["p_informed"]))
    return times, probs


def make_causal_lookup(times, values):
    """Return a function f(t) -> most recent value at a timestamp <= t
    (as-of/backward lookup, never looks into the future). Falls back to
    the first value if t precedes all observations."""
    def lookup(t: float) -> float:
        idx = bisect_right(times, t) - 1
        if idx < 0:
            return values[0]
        return values[idx]
    return lookup


def run_generic_backtest(depth_path: str, trades_path: str, horizon_T: float,
                          quote_fn, quote_params, p_lookup=None, label: str = "run"):
    """Generalized version of backtest_baseline.run_backtest: identical
    fill/PnL mechanics, but the quote at each step comes from
    `quote_fn(s, q, t_elapsed, p_informed, quote_params)` instead of being
    hardcoded to the plain AS formulas -- so baseline/heuristic/principled
    all share exactly the same causal replay engine. `horizon_T` (the AS
    trading horizon, seconds) is passed separately from `quote_params`
    because `quote_params` is a different type per variant (plain
    ASParams for baseline, AdverseParams for heuristic/principled) while
    the horizon clipping logic is common to all three."""
    times, mids = load_mid_series(depth_path)
    trades = load_trades(trades_path)
    if len(mids) < 2:
        raise ValueError("Not enough mid-price observations to backtest")

    t0 = times[0]
    trades_sorted = sorted(trades, key=lambda tr: tr["ts"])
    tr_idx = 0
    n_trades = len(trades_sorted)

    q = 0.0
    cash = 0.0
    rows = []
    fills = []  # for adverse-selection-cost computation: {ts, side, price}
    n_bid_fills = 0
    n_ask_fills = 0

    for i in range(len(mids) - 1):
        step_start = times[i]
        step_end = times[i + 1]
        s = mids[i]
        t_elapsed = min(step_start - t0, horizon_T)
        p_informed = p_lookup(step_start) if p_lookup is not None else 0.0

        bid, ask, r, spread = quote_fn(s, q, t_elapsed, p_informed, quote_params)

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
                    fills.append({"ts": step_start, "side": "bid", "price": bid})
                elif not ask_filled and price >= ask:
                    q -= 1.0
                    cash += ask
                    ask_filled = True
                    n_ask_fills += 1
                    fills.append({"ts": step_start, "side": "ask", "price": ask})
            tr_idx += 1

        mtm = cash + q * s
        rows.append({
            "local_ts": step_start, "t_elapsed": t_elapsed, "mid": s,
            "reservation_price": r, "bid": bid, "ask": ask, "spread": spread,
            "p_informed": p_informed, "inventory": q, "cash": cash, "mtm_pnl": mtm,
            "bid_filled": bid_filled, "ask_filled": ask_filled,
        })

    final_mid = mids[-1]
    final_mtm = cash + q * final_mid

    summary = {
        "label": label,
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
        "spread_mean": (sum(r["spread"] for r in rows) / len(rows)) if rows else None,
        "spread_min": min(r["spread"] for r in rows) if rows else None,
        "spread_max": max(r["spread"] for r in rows) if rows else None,
    }
    return rows, summary, fills, (times, mids)


def compute_adverse_selection_cost(fills, times, mids, horizon_sec: float = 2.0):
    """For each fill, charge max(0, adverse move) horizon_sec later. See
    module docstring for the exact per-side formula. Returns (total_cost,
    per_fill_costs, n_fills_with_future_data)."""
    per_fill = []
    for fill in fills:
        t_target = fill["ts"] + horizon_sec
        idx = bisect_right(times, t_target)
        if idx >= len(times):
            # No future data this far out (fill too close to end of
            # capture) -- excluded from the cost total, flagged, not
            # silently zero-filled.
            per_fill.append({**fill, "future_mid": None, "cost": None})
            continue
        future_mid = mids[idx]
        if fill["side"] == "bid":
            cost = max(0.0, fill["price"] - future_mid)
        else:
            cost = max(0.0, future_mid - fill["price"])
        per_fill.append({**fill, "future_mid": future_mid, "cost": cost})

    valid_costs = [f["cost"] for f in per_fill if f["cost"] is not None]
    total_cost = sum(valid_costs)
    return total_cost, per_fill, len(valid_costs)


def baseline_quote_fn(s, q, t, p_informed, params: ASParams):
    return baseline_quotes(s, q, t, params)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--params-json", default="data/processed/estimated_params.json")
    ap.add_argument("--scores-csv", default="data/processed/ml_informedness_scores.csv")
    ap.add_argument("--ml-metrics-json", default="data/processed/ml_informedness_metrics.json")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--heuristic-beta", type=float, default=1.0)
    ap.add_argument("--principled-eta", type=float, default=1.0)
    ap.add_argument("--as-cost-horizon-sec", type=float, default=2.0)
    args = ap.parse_args()

    with open(args.params_json) as f:
        p = json.load(f)["params"]
    base_params = ASParams(gamma=p["gamma"], sigma=p["sigma"], kappa=p["kappa"],
                            A=p["A"], T=p["T"])

    with open(args.ml_metrics_json) as f:
        ml_meta = json.load(f)
    expected_adverse_move = ml_meta["feature_meta"]["expected_adverse_move_given_informed"]
    if expected_adverse_move is None:
        expected_adverse_move = 0.0

    adverse_params = AdverseParams(
        base=base_params,
        heuristic_beta=args.heuristic_beta,
        principled_eta=args.principled_eta,
        expected_adverse_move=expected_adverse_move,
    )

    print("[params] " + ", ".join(f"{k}={v}" for k, v in vars(base_params).items()))
    print(f"[adverse params] heuristic_beta={args.heuristic_beta}, "
          f"principled_eta={args.principled_eta}, "
          f"expected_adverse_move=${expected_adverse_move:.4f} "
          f"(from ML training data, see {args.ml_metrics_json})")

    p_times, p_values = load_p_informed_series(args.scores_csv)
    p_lookup = make_causal_lookup(p_times, p_values)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    def run_and_report(name, quote_fn, quote_params, use_p):
        lookup = p_lookup if use_p else None
        rows, summary, fills, (times, mids) = run_generic_backtest(
            args.depth, args.trades, base_params.T,
            quote_fn, quote_params, p_lookup=lookup, label=name,
        )
        as_cost, per_fill_costs, n_valid = compute_adverse_selection_cost(
            fills, times, mids, horizon_sec=args.as_cost_horizon_sec,
        )
        summary["adverse_selection_cost_total"] = as_cost
        summary["adverse_selection_cost_n_fills_scored"] = n_valid
        summary["adverse_selection_cost_per_fill"] = (
            as_cost / n_valid if n_valid > 0 else None
        )

        series_path = os.path.join(args.out_dir, f"backtest_series_{name}.csv")
        with open(series_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

        fills_path = os.path.join(args.out_dir, f"backtest_fills_{name}.json")
        with open(fills_path, "w") as f:
            json.dump(per_fill_costs, f, indent=2, default=str)

        results[name] = summary
        return summary

    # (1) baseline -- p_informed not consumed at all (not even looked up),
    # so this run is bit-for-bit the Part 2 backtest logic (same
    # formulas, same fill mechanics, quote_params is the plain ASParams).
    def _baseline_fn(s, q, t, p, params):
        return baseline_quote_fn(s, q, t, p, params)

    run_and_report("baseline", _baseline_fn, base_params, use_p=False)

    # (2) heuristic overlay
    def _heuristic_fn(s, q, t, p, params):
        return heuristic_quotes(s, q, t, p, params)

    run_and_report("heuristic", _heuristic_fn, adverse_params, use_p=True)

    # (3) principled variant
    def _principled_fn(s, q, t, p, params):
        return principled_quotes(s, q, t, p, params)

    run_and_report("principled", _principled_fn, adverse_params, use_p=True)

    print("\n===== Three-way adverse-selection backtest comparison =====")
    header = f"{'metric':32s}" + "".join(f"{k:>16s}" for k in results.keys())
    print(header)
    metric_keys = [
        "n_fills_total", "n_bid_fills", "n_ask_fills",
        "inventory_min", "inventory_max", "final_inventory",
        "spread_mean", "final_mtm_pnl",
        "adverse_selection_cost_total", "adverse_selection_cost_n_fills_scored",
        "adverse_selection_cost_per_fill",
    ]
    for mk in metric_keys:
        vals = []
        for name in results:
            v = results[name].get(mk)
            if isinstance(v, float):
                vals.append(f"{v:16.4f}")
            else:
                vals.append(f"{str(v):>16s}")
        print(f"{mk:32s}" + "".join(vals))

    comparison_path = os.path.join(args.out_dir, "backtest_adverse_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump({
            "adverse_params": {
                "heuristic_beta": args.heuristic_beta,
                "principled_eta": args.principled_eta,
                "expected_adverse_move": expected_adverse_move,
                "as_cost_horizon_sec": args.as_cost_horizon_sec,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nWrote {comparison_path}")


if __name__ == "__main__":
    main()
