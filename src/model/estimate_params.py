#!/usr/bin/env python3
"""
estimate_params.py

Turns a raw capture (`src/data/collect_market_data.py` output: one
depth@100ms .jsonl and one @trade .jsonl) into the numbers the
Avellaneda-Stoikov model (`src/model/avellaneda_stoikov.py`) needs:
sigma, kappa, A -- plus writes the reconstructed mid-price series to
data/processed/ for the backtest harness to replay.

What's *estimated* from real data vs. what's a *design choice*, stated
plainly (this is the point of Part 2's brief, not an afterthought):

  sigma (volatility)  -- ESTIMATED, real number from the real captured
      mid-price path. Realized-variance estimator: sum of squared
      mid-price increments over the capture window, scaled to a
      per-second variance, sigma = sqrt(that). Units: $ per sqrt(second)
      (Avellaneda-Stoikov's dS_t = sigma dW_t is *arithmetic* Brownian
      motion in raw price units, not log-returns).

  kappa, A (fill-intensity decay/base rate) -- CRUDELY ESTIMATED from
      the real trade tape, method documented in `estimate_kappa_A`
      below. This is NOT a rigorous calibration (we don't have our own
      resting orders or a full order-book-sweep reconstruction to
      measure fill rates directly) -- it is a real-data-informed proxy,
      not an assumed placeholder, but its statistical power over a
      ~4-minute capture is limited and should be read as "plausible
      order of magnitude," not "calibrated."

  gamma (risk aversion), T (horizon) -- NOT estimated at all. These are
      the market maker's own preference/design parameters in the AS
      model, not properties of the market. Illustrative, stated values
      -- see DEFAULT_GAMMA / DEFAULT_T_SECONDS below and the comments
      next to them.

Usage:
    python src/model/estimate_params.py \
        --depth data/raw/depth_btcusdt_TIMESTAMP.jsonl \
        --trades data/raw/trades_btcusdt_TIMESTAMP.jsonl \
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# Design-choice parameters (NOT estimated -- the market maker's own
# preferences, per DERIVATION.md Sec 6: gamma and T never appear as
# something the maker infers from the market, only sigma/kappa/A do).
# ---------------------------------------------------------------------------
DEFAULT_GAMMA = 0.003      # risk aversion. Illustrative, hand-picked so the
                            # resulting spread sits in a plausible range for
                            # BTCUSDT's real price/volatility scale (see
                            # README/backtest notes) -- NOT the gamma=0.1
                            # used in Avellaneda-Stoikov's own toy numerical
                            # example, which was calibrated to a ~$100 stock
                            # and would produce an absurdly wide (tens of
                            # dollars) spread at BTC's $65k scale and 1
                            # $/sqrt(sec) volatility. The *scale-dependence*
                            # of gamma is itself worth noting: this is a
                            # dimensionful risk-aversion coefficient (units
                            # of 1/$), not a universal constant, so it has to
                            # be re-picked per instrument/price-level.
DEFAULT_T_SECONDS = 240.0  # trading horizon in seconds. Illustrative:
                            # chosen to match the ~240s capture window used
                            # for the backtest, framing the whole capture as
                            # a single quoting session that ends when the
                            # capture does (so the inventory-risk widening
                            # term decays to zero by the end of the backtest,
                            # as intended, rather than staying pinned near
                            # its start-of-horizon value throughout).
                            #
                            # Sensitivity note (from a quick gamma/T grid
                            # check against the real capture, see
                            # src/backtest/backtest_baseline.py): because
                            # the fitted kappa here is small (~0.49, itself
                            # a symptom of the crude estimation method --
                            # see estimate_kappa_A), the markup term's
                            # gamma->0 floor 2/kappa is already ~4.1 dollars
                            # -- far above BTCUSDT's real observed mean
                            # spread (~$0.10 in this capture). So the
                            # baseline's hypothetical fill count stays
                            # modest (order of 10 fills over ~240s /
                            # ~2700 real trades) across a wide range of
                            # gamma, not just at this particular value --
                            # a genuine finding about this crude kappa
                            # estimate, reported as such rather than
                            # papered over by cherry-picking gamma to
                            # force more fills.


def load_mid_series(depth_path: str) -> tuple[list[float], list[float]]:
    """Reconstruct top-of-book mid-price series from a depth@100ms capture.

    Same incremental-diff reconstruction approach documented in
    `collect_market_data.summarize_depth`: seed the local book from the
    first diff's own levels, apply every subsequent diff on top. Returns
    (timestamps, mids), both real-valued lists in chronological order.
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    times: list[float] = []
    mids: list[float] = []
    with open(depth_path) as f:
        for line in f:
            rec = json.loads(line)
            raw = rec["raw"]
            for p, qty in raw.get("b") or []:
                p, qty = float(p), float(qty)
                if qty == 0.0:
                    bids.pop(p, None)
                else:
                    bids[p] = qty
            for p, qty in raw.get("a") or []:
                p, qty = float(p), float(qty)
                if qty == 0.0:
                    asks.pop(p, None)
                else:
                    asks[p] = qty
            if bids and asks:
                best_bid, best_ask = max(bids), min(asks)
                if best_bid < best_ask:
                    times.append(rec["local_ts"])
                    mids.append((best_bid + best_ask) / 2.0)
    return times, mids


def load_trades(trades_path: str) -> list[dict]:
    """Parse the raw @trade capture into a list of {ts, price, qty, buyer_is_maker}."""
    out = []
    with open(trades_path) as f:
        for line in f:
            rec = json.loads(line)
            raw = rec["raw"]
            if raw.get("e") != "trade":
                continue
            out.append({
                "ts": rec["local_ts"],
                "price": float(raw["p"]),
                "qty": float(raw["q"]),
                # Binance semantics: m=True means the BUYER was the
                # maker, i.e. this trade was a taker SELL hitting the
                # bid; m=False means a taker BUY hitting the ask.
                "buyer_is_maker": bool(raw["m"]),
            })
    return out


def estimate_sigma(times: list[float], mids: list[float]) -> dict:
    """Realized-volatility estimator for sigma in Avellaneda-Stoikov's
    dS_t = sigma dW_t (arithmetic BM, raw price units, not log-returns).

    sigma^2_per_sec = sum((mid[i] - mid[i-1])^2) / (t[-1] - t[0])

    the standard quadratic-variation estimator of instantaneous variance
    for a driftless diffusion observed at (possibly irregular) discrete
    times -- valid regardless of the sampling grid being uneven, which
    ours is (depth@100ms messages don't arrive on an exact 100ms clock).
    """
    if len(mids) < 2:
        raise ValueError("Need at least 2 mid-price observations to estimate sigma")
    sq_sum = 0.0
    for i in range(1, len(mids)):
        sq_sum += (mids[i] - mids[i - 1]) ** 2
    total_span = times[-1] - times[0]
    if total_span <= 0:
        raise ValueError("Non-positive time span in mid-price series")
    var_per_sec = sq_sum / total_span
    sigma_per_sec = math.sqrt(var_per_sec)
    return {
        "sigma_per_sec": sigma_per_sec,
        "n_mid_obs": len(mids),
        "span_sec": total_span,
        "sum_sq_increments": sq_sum,
        "method": "realized quadratic variation of the raw (non-log) mid-price "
                  "path, scaled to per-second variance; sigma = sqrt(that). "
                  "ESTIMATED from real captured data.",
    }


def estimate_kappa_A(trades: list[dict], depth_times: list[float], depth_mids: list[float]) -> dict:
    """Crude, real-data-informed estimate of (A, kappa) in lambda(delta) = A*exp(-kappa*delta).

    Method (documented explicitly, per the task brief, as a *rough* proxy
    -- not a rigorous calibration):

    1. For every executed trade, find the prevailing mid-price at (or
       just before) the trade's timestamp (nearest earlier depth
       snapshot), and compute the trade's absolute distance from that
       mid: delta_i = |trade_price_i - mid_i|. This approximates "how
       far from mid a resting limit order would have had to sit to be
       the one that filled this print" -- a real market order that
       executes delta away from mid would, if a maker's limit order sat
       at exactly that distance, have swept through and filled it.

    2. Bin trades by threshold distance delta: for a grid of thresholds
       delta_k, count how many trades in the whole capture window
       reached at least that far (|trade_price - mid| >= delta_k), and
       divide by the capture window's duration to get an empirical rate
       lambda_hat(delta_k) = (# trades reaching >= delta_k) / T_window.
       This is the empirical analogue of the model's lambda(delta): the
       rate at which a quote resting at distance delta gets hit.

    3. Fit ln(lambda_hat(delta_k)) = ln(A) - kappa*delta_k by ordinary
       least squares over the thresholds with a nonzero count. Intercept
       gives ln(A), (negative of) slope gives kappa.

    Caveats made explicit, not glossed over:
      - This proxies "quote distance" by "how far the trade print itself
        was from contemporaneous mid," which conflates trade size/sweep
        depth with a maker's *resting* quote distance -- a genuine fill
        model would need our own posted orders and the full book, not
        just trade prints. Treat kappa, A here as an illustrative,
        order-of-magnitude estimate informed by real trade data, not a
        calibrated fill-probability model.
      - Over a ~4 minute capture with mostly small, tightly-clustered
        trade prints (BTCUSDT is a deep, liquid pair), the distances are
        mostly within a few dollars of mid, so the tail thresholds have
        very few observations -- the fitted kappa is noisy.
    """
    if not trades or len(depth_mids) < 2:
        raise ValueError("Need trades and a mid-price series to estimate kappa/A")

    # nearest-earlier-mid lookup via simple linear scan with a moving
    # pointer (both series are already time-sorted).
    j = 0
    n_depth = len(depth_times)
    distances = []
    for tr in trades:
        while j + 1 < n_depth and depth_times[j + 1] <= tr["ts"]:
            j += 1
        mid_at_trade = depth_mids[j]
        distances.append(abs(tr["price"] - mid_at_trade))

    t_window = depth_times[-1] - depth_times[0]
    if t_window <= 0:
        raise ValueError("Non-positive depth time window")

    distances_sorted = sorted(distances)
    max_d = distances_sorted[-1]
    if max_d <= 0:
        raise ValueError("All trades printed exactly at mid -- cannot fit a decay")

    # Use empirical distance quantiles as thresholds (10 buckets) rather
    # than a fixed linear grid, so each threshold has a reasonable
    # sample size even though most trades cluster near mid.
    n_thresholds = 10
    quantiles = [distances_sorted[int(q * (len(distances_sorted) - 1))]
                 for q in [i / (n_thresholds - 1) for i in range(n_thresholds)]]
    thresholds = sorted(set(d for d in quantiles if d > 0))

    xs, ys = [], []
    counts_at_threshold = {}
    for d_k in thresholds:
        count = sum(1 for d in distances if d >= d_k)
        counts_at_threshold[d_k] = count
        if count > 0:
            xs.append(d_k)
            ys.append(math.log(count / t_window))

    if len(xs) < 2:
        raise ValueError("Not enough distinct distance thresholds with hits to fit kappa/A")

    # OLS slope/intercept: ln(lambda) = ln(A) - kappa*delta
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        raise ValueError("Degenerate distance thresholds (zero variance) -- cannot fit slope")
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    kappa = -slope
    A = math.exp(intercept)

    # Guard against a degenerate/wrong-signed fit (e.g. too little data):
    # kappa must be > 0 for lambda(delta) to decay, and the model divides
    # by kappa, so a non-positive fit is unusable as-is.
    fit_is_sane = kappa > 0 and A > 0

    return {
        "kappa": kappa,
        "A": A,
        "fit_is_sane": fit_is_sane,
        "n_trades": len(trades),
        "n_thresholds_used": len(xs),
        "distance_thresholds": thresholds,
        "counts_at_threshold": counts_at_threshold,
        "max_observed_distance": max_d,
        "median_observed_distance": distances_sorted[len(distances_sorted) // 2],
        "capture_window_sec": t_window,
        "method": "OLS fit of ln(empirical hit-rate at distance>=threshold) vs threshold, "
                  "over quantile-spaced thresholds of |trade_price - contemporaneous_mid|. "
                  "CRUDE real-data-informed proxy, NOT a rigorous fill-probability calibration "
                  "-- see estimate_kappa_A docstring.",
    }


@dataclass
class EstimatedParams:
    gamma: float
    sigma: float
    kappa: float
    A: float
    T: float
    gamma_source: str
    T_source: str
    sigma_source: str
    kappa_A_source: str


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", required=True, help="Path to captured depth@100ms .jsonl")
    ap.add_argument("--trades", required=True, help="Path to captured @trade .jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--horizon-sec", type=float, default=DEFAULT_T_SECONDS)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    times, mids = load_mid_series(args.depth)
    trades = load_trades(args.trades)
    print(f"[load] {len(mids)} mid-price observations spanning "
          f"{times[-1] - times[0]:.1f}s; {len(trades)} trade prints")

    sigma_info = estimate_sigma(times, mids)
    kappa_A_info = estimate_kappa_A(trades, times, mids)

    params = EstimatedParams(
        gamma=args.gamma,
        sigma=sigma_info["sigma_per_sec"],
        kappa=kappa_A_info["kappa"] if kappa_A_info["fit_is_sane"] else 1.0,
        A=kappa_A_info["A"] if kappa_A_info["fit_is_sane"] else 5.0,
        T=args.horizon_sec,
        gamma_source="DESIGN CHOICE (illustrative; not estimated from data)",
        T_source="DESIGN CHOICE (illustrative; not estimated from data)",
        sigma_source="ESTIMATED from real captured mid-price data (realized quadratic variation)",
        kappa_A_source=(
            "CRUDE ESTIMATE from real captured trade data (see estimate_kappa_A method)"
            if kappa_A_info["fit_is_sane"] else
            "FIT WAS DEGENERATE on this capture -- fell back to an ILLUSTRATIVE PLACEHOLDER "
            "(kappa=1.0, A=5.0), not a calibrated value. See kappa_A_diagnostics."
        ),
    )

    out = {
        "params": asdict(params),
        "sigma_diagnostics": sigma_info,
        "kappa_A_diagnostics": kappa_A_info,
        "source_files": {"depth": args.depth, "trades": args.trades},
    }

    params_path = os.path.join(args.out_dir, "estimated_params.json")
    with open(params_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    mid_series_path = os.path.join(args.out_dir, "mid_price_series.csv")
    with open(mid_series_path, "w") as f:
        f.write("local_ts,mid_price\n")
        for t, m in zip(times, mids):
            f.write(f"{t},{m}\n")

    print(f"\n[sigma] {sigma_info['sigma_per_sec']:.6f} $/sqrt(sec)  "
          f"(from {sigma_info['n_mid_obs']} obs over {sigma_info['span_sec']:.1f}s) -- ESTIMATED")
    print(f"[kappa,A] kappa={kappa_A_info['kappa']:.6f}  A={kappa_A_info['A']:.6f}  "
          f"fit_is_sane={kappa_A_info['fit_is_sane']} -- "
          f"{'CRUDE ESTIMATE' if kappa_A_info['fit_is_sane'] else 'FALLBACK PLACEHOLDER'}")
    print(f"[gamma] {args.gamma} -- DESIGN CHOICE")
    print(f"[T] {args.horizon_sec}s -- DESIGN CHOICE")
    print(f"\nWrote {params_path}")
    print(f"Wrote {mid_series_path}")


if __name__ == "__main__":
    main()
