#!/usr/bin/env python3
"""
regime_stress_test.py

Part 4 / Phase 5 (Validation) -- README: "Stress-test both strategies
over the highest-toxicity periods identified by VPIN in the data, since
that's exactly where the extension is supposed to earn its keep."

DEVIATION FROM THE LITERAL README TEXT, STATED EXPLICITLY: the README's
Phase 2 plan was to reuse the informed-flow proxy from the sibling
`order-flow-imbalance` / `vpin-flow-toxicity` repos. `informedness_signal.py`
already documented (Part 3) that this project stayed self-contained
instead and engineered its own OFI-style + quote-imbalance features
directly from this repo's own captured data, rather than a cross-repo
import (those are separate git repos with their own environments; a
runtime dependency between them isn't a practical "one repo, one
`requirements.txt`, one `python script.py`" setup for a student project
submission). Consistent with that Part 3 decision, this stress test uses
this project's OWN Part 3 ML classifier's predicted p_informed(t) as the
regime indicator instead of a VPIN score computed by a different repo.
This is a real, acknowledged deviation from the README's literal
wording, not a silent substitution -- and it means the "regime" here is
exactly the same signal the heuristic/principled quoting variants
already consume, which is arguably the *more* relevant regime definition
for this specific question ("does the extension help exactly where its
own signal says risk is highest") even though it is not the VPIN
definition of toxicity.

Methodology:
  1. Run the SAME full-window (~240s) three-way backtest Part 3 ran
     (`backtest_adverse.py`'s `run_generic_backtest`) -- NOT restricted
     to the held-out slice from `heldout_backtest.py`. This is a
     deliberate, documented choice: the held-out test window is only
     ~71s and (checked directly) contains just 2-3 fills per variant --
     splitting that further into high/low-toxicity buckets leaves 0-1
     fills per bucket, which cannot support any stress-test conclusion
     at all. The regime split here is about characterizing the EXISTING
     three-way comparison's behavior across market conditions, not about
     re-validating the classifier out-of-sample (that is what
     `heldout_backtest.py` is for) -- a materially different question,
     answered on the fuller data that question needs.
  2. Regime indicator: for every backtest step, the SAME causally-looked-
     up p_informed(t) value the heuristic/principled variants already
     condition their quotes on (`backtest_adverse.load_p_informed_series`
     + `make_causal_lookup`). A step (and any fill occurring during it)
     is HIGH-toxicity regime if p_informed(t) is at or above the
     `--regime-quantile` percentile (default: 75th) of the full p_informed
     distribution over the whole capture; otherwise LOW-toxicity regime.
     This threshold is deliberately a QUANTILE of this specific capture's
     own p_informed distribution (which is heavily right-skewed: median
     ~0.019, 75th pct ~0.06, 95th pct ~0.88 -- see printed diagnostics),
     not an arbitrary fixed probability cutoff, so "high toxicity" means
     "the riskiest quartile of this session," a stress-test framing
     consistent with the README's "highest-toxicity periods" language.
  3. Regime membership is computed ONCE from the exogenous p_informed(t)
     signal and applied identically to all three variants' fills --
     baseline, heuristic, and principled fills are bucketed by the SAME
     regime definition even though baseline's own quotes don't react to
     it, so the comparison is apples-to-apples (a fill is "high regime"
     because the signal said so at that moment, independent of which
     variant was quoting).
  4. Within each regime bucket, report n_fills and adverse-selection
     cost (total AND per-fill) per variant -- the per-fill number is the
     one that actually answers the question, since total cost is
     mechanically driven by how many fills happened to land in that
     bucket.

Usage:
    python src/validation/regime_stress_test.py \
        --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
        --trades data/raw/trades_btcusdt_20260723_094653.jsonl \
        --params-json data/processed/estimated_params.json \
        --scores-csv data/processed/ml_informedness_scores.csv \
        --ml-metrics-json data/processed/ml_informedness_metrics.json \
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.backtest.backtest_adverse import (  # noqa: E402
    baseline_quote_fn,
    compute_adverse_selection_cost,
    load_p_informed_series,
    make_causal_lookup,
    run_generic_backtest,
)
from src.model.avellaneda_stoikov import ASParams  # noqa: E402
from src.model.avellaneda_stoikov_adverse import (  # noqa: E402
    AdverseParams,
    heuristic_quotes,
    principled_quotes,
)


def split_fills_by_regime(fills: list[dict], p_lookup, threshold: float):
    high, low = [], []
    for fill in fills:
        p = p_lookup(fill["ts"])
        (high if p >= threshold else low).append({**fill, "p_informed_at_fill": p})
    return high, low


def regime_cost_summary(fills: list[dict], times, mids, horizon_sec: float):
    total, per_fill, n_valid = compute_adverse_selection_cost(fills, times, mids, horizon_sec)
    return {
        "n_fills": len(fills),
        "n_fills_scored": n_valid,
        "adverse_selection_cost_total": total,
        "adverse_selection_cost_per_fill": (total / n_valid) if n_valid > 0 else None,
    }


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
    ap.add_argument("--regime-quantile", type=float, default=0.75,
                     help="Percentile of the full-capture p_informed distribution "
                          "used as the high/low regime cutoff (default: 0.75, i.e. "
                          "the riskiest quartile is 'high toxicity').")
    args = ap.parse_args()

    with open(args.params_json) as f:
        p = json.load(f)["params"]
    base_params = ASParams(gamma=p["gamma"], sigma=p["sigma"], kappa=p["kappa"],
                            A=p["A"], T=p["T"])

    with open(args.ml_metrics_json) as f:
        ml_meta = json.load(f)
    expected_adverse_move = ml_meta["feature_meta"]["expected_adverse_move_given_informed"] or 0.0

    adverse_params = AdverseParams(
        base=base_params,
        heuristic_beta=args.heuristic_beta,
        principled_eta=args.principled_eta,
        expected_adverse_move=expected_adverse_move,
    )

    p_times, p_values = load_p_informed_series(args.scores_csv)
    p_lookup = make_causal_lookup(p_times, p_values)

    threshold = float(np.percentile(p_values, args.regime_quantile * 100))
    print(f"[regime] p_informed distribution over the full ~{base_params.T:.0f}s capture: "
          f"min={min(p_values):.4f} median={np.median(p_values):.4f} "
          f"p75={np.percentile(p_values, 75):.4f} p90={np.percentile(p_values, 90):.4f} "
          f"max={max(p_values):.4f}")
    print(f"[regime] HIGH-toxicity threshold = {args.regime_quantile:.0%} percentile = "
          f"{threshold:.4f}  (a step/fill is HIGH regime if its causal p_informed(t) >= this)")

    variants = {
        "baseline": (lambda s, q, t, pr, params: baseline_quote_fn(s, q, t, pr, params), base_params, False),
        "heuristic": (lambda s, q, t, pr, params: heuristic_quotes(s, q, t, pr, params), adverse_params, True),
        "principled": (lambda s, q, t, pr, params: principled_quotes(s, q, t, pr, params), adverse_params, True),
    }

    results = {}
    for name, (quote_fn, quote_params, use_p) in variants.items():
        lookup = p_lookup if use_p else None
        rows, summary, fills, (times, mids) = run_generic_backtest(
            args.depth, args.trades, base_params.T, quote_fn, quote_params,
            p_lookup=lookup, label=name,
        )
        overall = regime_cost_summary(fills, times, mids, args.as_cost_horizon_sec)

        high_fills, low_fills = split_fills_by_regime(fills, p_lookup, threshold)
        high = regime_cost_summary(high_fills, times, mids, args.as_cost_horizon_sec)
        low = regime_cost_summary(low_fills, times, mids, args.as_cost_horizon_sec)

        results[name] = {
            "n_fills_total": summary["n_fills_total"],
            "overall": overall,
            "high_toxicity_regime": high,
            "low_toxicity_regime": low,
            "high_toxicity_fills_detail": [
                {"ts": f["ts"], "side": f["side"], "p_informed_at_fill": f["p_informed_at_fill"]}
                for f in high_fills
            ],
        }

    print("\n===== Regime stress test: adverse-selection cost, HIGH-toxicity regime only =====")
    header = f"{'metric':32s}" + "".join(f"{k:>16s}" for k in results.keys())
    print(header)
    for mk in ["n_fills", "adverse_selection_cost_total", "adverse_selection_cost_per_fill"]:
        vals = []
        for name in results:
            v = results[name]["high_toxicity_regime"].get(mk)
            vals.append(f"{v:16.4f}" if isinstance(v, float) else f"{str(v):>16s}")
        print(f"{mk:32s}" + "".join(vals))

    print("\n===== Regime stress test: adverse-selection cost, LOW-toxicity regime only =====")
    print(header)
    for mk in ["n_fills", "adverse_selection_cost_total", "adverse_selection_cost_per_fill"]:
        vals = []
        for name in results:
            v = results[name]["low_toxicity_regime"].get(mk)
            vals.append(f"{v:16.4f}" if isinstance(v, float) else f"{str(v):>16s}")
        print(f"{mk:32s}" + "".join(vals))

    # Honest headline verdict: is the per-fill AS-cost improvement bigger
    # in the high-toxicity regime than the low-toxicity regime?
    b_high = results["baseline"]["high_toxicity_regime"]["adverse_selection_cost_per_fill"]
    p_high = results["principled"]["high_toxicity_regime"]["adverse_selection_cost_per_fill"]
    b_low = results["baseline"]["low_toxicity_regime"]["adverse_selection_cost_per_fill"]
    p_low = results["principled"]["low_toxicity_regime"]["adverse_selection_cost_per_fill"]
    verdict_lines = []
    if b_high is not None and p_high is not None:
        delta_high = p_high - b_high
        verdict_lines.append(
            f"HIGH-toxicity regime: baseline per-fill AS cost {b_high:.4f} -> "
            f"principled {p_high:.4f}  (delta {delta_high:+.4f}, "
            f"{'IMPROVED' if delta_high < 0 else 'WORSENED'}, n_fills "
            f"baseline={results['baseline']['high_toxicity_regime']['n_fills']} "
            f"principled={results['principled']['high_toxicity_regime']['n_fills']})"
        )
    if b_low is not None and p_low is not None:
        delta_low = p_low - b_low
        verdict_lines.append(
            f"LOW-toxicity regime:  baseline per-fill AS cost {b_low:.4f} -> "
            f"principled {p_low:.4f}  (delta {delta_low:+.4f}, "
            f"{'IMPROVED' if delta_low < 0 else 'WORSENED'}, n_fills "
            f"baseline={results['baseline']['low_toxicity_regime']['n_fills']} "
            f"principled={results['principled']['low_toxicity_regime']['n_fills']})"
        )
    print("\n===== Headline verdict =====")
    for line in verdict_lines:
        print(line)
    print("Sample sizes in both buckets are tiny (single-digit fills) -- read as a "
          "directional, not statistically powered, finding. See research/RESULTS.md.")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "note": (
            "Regime indicator = this project's OWN Part 3 ML classifier's causal "
            "p_informed(t), NOT a cross-repo VPIN score -- see module docstring for why. "
            "Run on the FULL ~240s backtest window (not the held-out-only slice), because "
            "the held-out slice has too few fills to support a regime split -- see module "
            "docstring."
        ),
        "regime_definition": {
            "quantile": args.regime_quantile,
            "threshold_p_informed": threshold,
            "p_informed_distribution": {
                "min": float(min(p_values)), "median": float(np.median(p_values)),
                "p75": float(np.percentile(p_values, 75)),
                "p90": float(np.percentile(p_values, 90)),
                "max": float(max(p_values)),
            },
        },
        "results": results,
    }
    out_path = os.path.join(args.out_dir, "regime_stress_test.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
