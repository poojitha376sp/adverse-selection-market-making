#!/usr/bin/env python3
"""
informedness_signal.py

Part 3 / Phase 4. Builds the feature set + training target that feeds the
ML informedness classifier (`ml_informedness_classifier.py`), computed
directly from THIS repo's own captured L2 depth + trade data
(`src/data/collect_market_data.py` output). No dependency on the sibling
`order-flow-imbalance` / `vpin-flow-toxicity` GitHub repos -- per the task
brief, this project stays self-contained and instead replicates their
lightweight methodology (short-horizon order-flow imbalance + quote
imbalance), which the README's Phase 2 note already anticipated as an
acceptable fallback ("or engineered directly from this project's own
captured data").

Two feature families, both standard, well-known microstructure signals
(see research/CHEATSHEET.md's sister OFI cheatsheet cross-reference for
the underlying literature) -- nothing novel is being invented here:

  1. Trade-flow imbalance ("OFI-style"): net signed trade volume in a
     short trailing window. Binance trade prints carry `m` = "buyer is
     maker"; m=False means an aggressive taker BUY hit the ask (+qty
     signed), m=True means an aggressive taker SELL hit the bid (-qty
     signed). Net signed volume in a short trailing window is the
     simplest, most standard order-flow-imbalance proxy for directional
     pressure.
  2. Quote imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty) at the
     top of book, reconstructed from the incremental depth diffs the
     same way `estimate_params.load_mid_series` does, but retaining
     quantities as well as price. Top-of-book only (single level) -- a
     deliberate simplification, documented as such, not a multi-level
     book-pressure model.

Target definition ("is the flow at this point informed"):

  A point in time is labeled INFORMED if the mid-price moves, over the
  next `horizon_sec` seconds, by more than a volatility-scaled threshold
  -- i.e. a move a hypothetical passive quote resting at that instant
  would not have been compensated for by the ordinary AS spread. This is
  exactly the definition the task brief suggests: "the price moves
  adversely against a hypothetical passive quote over the next short
  horizon by more than some threshold." Concretely:

      threshold = k_sigma * sigma_per_sec * sqrt(horizon_sec)
      informed(t) = 1{ |mid(t+horizon_sec) - mid(t)| > threshold }

  sigma_per_sec is the same realized-volatility estimate Part 2 already
  computes (`estimate_params.estimate_sigma`) -- so the threshold is a
  "how many sigmas of surprise did the price move by" test, not an
  arbitrary dollar cutoff. The label is intentionally SYMMETRIC (absolute
  value): a big move either direction is "informed" flow from the
  perspective of a market maker quoting both sides, since a passive
  maker gets adversely selected whichever way an informed trader was
  leaning. Points within `horizon_sec` of the end of the capture have no
  future price to check and are labeled NaN (excluded from training, but
  still scored at inference time).

  Caveat, stated plainly: this labels *periods of realized toxic price
  movement*, not verified informed trader identity (no ground truth for
  "who was informed" exists in public exchange data) -- the same
  limitation the underlying academic literature (VPIN, OFI-toxicity
  proxies) all share, and one the README's Phase 5 already flags in the
  Optiver-finding discussion (accuracy on this label != guaranteed PnL
  protection).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "net_signed_vol",
    "trade_count",
    "ofi_normalized",
    "quote_imbalance",
    "trailing_vol",
    "mid_momentum",
    "quoted_spread",
]


def reconstruct_book_series(depth_path: str) -> pd.DataFrame:
    """Reconstruct top-of-book (best bid/ask price + qty) at every depth
    snapshot in chronological order. Same incremental-diff book
    reconstruction as `estimate_params.load_mid_series`, but retaining
    quantities (not just price) since quote imbalance needs them.
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows = []
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
                best_bid = max(bids)
                best_ask = min(asks)
                if best_bid < best_ask:
                    rows.append({
                        "local_ts": rec["local_ts"],
                        "mid": (best_bid + best_ask) / 2.0,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "bid_qty": bids[best_bid],
                        "ask_qty": asks[best_ask],
                    })
    df = pd.DataFrame(rows).sort_values("local_ts").reset_index(drop=True)
    return df


def load_trades_df(trades_path: str) -> pd.DataFrame:
    """Parse the raw @trade capture, signing each print by Binance's `m`
    (buyer_is_maker) flag: m=False -> aggressive taker BUY (+qty),
    m=True -> aggressive taker SELL (-qty)."""
    rows = []
    with open(trades_path) as f:
        for line in f:
            rec = json.loads(line)
            raw = rec["raw"]
            if raw.get("e") != "trade":
                continue
            qty = float(raw["q"])
            buyer_is_maker = bool(raw["m"])
            rows.append({
                "local_ts": rec["local_ts"],
                "price": float(raw["p"]),
                "qty": qty,
                "signed_qty": qty * (-1.0 if buyer_is_maker else 1.0),
                "buyer_is_maker": buyer_is_maker,
            })
    df = pd.DataFrame(rows).sort_values("local_ts").reset_index(drop=True)
    return df


def estimate_sigma_per_sec(book: pd.DataFrame) -> float:
    """Same realized quadratic-variation sigma estimator as
    `estimate_params.estimate_sigma`, recomputed here directly on the
    reconstructed book series so this module has no import-time
    dependency on estimate_params (kept independent/self-contained)."""
    mids = book["mid"].to_numpy()
    times = book["local_ts"].to_numpy()
    sq_sum = float(np.sum(np.diff(mids) ** 2))
    span = float(times[-1] - times[0])
    if span <= 0:
        raise ValueError("Non-positive time span in book series")
    return math.sqrt(sq_sum / span)


def build_feature_frame(
    depth_path: str,
    trades_path: str,
    trailing_window_sec: float = 2.0,
    horizon_sec: float = 2.0,
    k_sigma: float = 1.0,
    vol_lookback_obs: int = 20,
    momentum_lookback_obs: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Build the full (features, label) frame, time-aligned to the
    reconstructed book snapshots (one row per depth snapshot).

    All features at row i are computed strictly from information at or
    before local_ts[i] (trailing windows only) -- no look-ahead into the
    feature set itself. Only the LABEL looks forward (that's the point of
    a label). Returns (frame, meta) where meta carries the estimated
    sigma, the resulting threshold, and other provenance info.
    """
    book = reconstruct_book_series(depth_path)
    trades = load_trades_df(trades_path)
    if len(book) < vol_lookback_obs + 2 or trades.empty:
        raise ValueError("Not enough book/trade data to build features")

    ts = book["local_ts"].to_numpy()
    mids = book["mid"].to_numpy()

    # ---- Feature 1: trailing net signed trade volume (OFI-style) ----
    # Vectorized via cumulative sums + searchsorted: cheaper and simpler
    # than a per-row python loop, exact same result as a trailing window
    # sum for each row.
    trade_ts = trades["local_ts"].to_numpy()
    signed = trades["signed_qty"].to_numpy()
    qty = trades["qty"].to_numpy()
    buy_qty = np.where(signed > 0, qty, 0.0)
    sell_qty = np.where(signed < 0, qty, 0.0)

    cum_signed = np.concatenate([[0.0], np.cumsum(signed)])
    cum_buy = np.concatenate([[0.0], np.cumsum(buy_qty)])
    cum_sell = np.concatenate([[0.0], np.cumsum(sell_qty)])
    cum_count = np.arange(len(trade_ts) + 1, dtype=float)

    idx_end = np.searchsorted(trade_ts, ts, side="right")
    idx_start = np.searchsorted(trade_ts, ts - trailing_window_sec, side="right")

    net_signed_vol = cum_signed[idx_end] - cum_signed[idx_start]
    buy_vol = cum_buy[idx_end] - cum_buy[idx_start]
    sell_vol = cum_sell[idx_end] - cum_sell[idx_start]
    trade_count = cum_count[idx_end] - cum_count[idx_start]
    total_vol = buy_vol + sell_vol
    # normalized OFI in [-1, 1]: net / total traded volume in the window
    ofi_normalized = np.divide(net_signed_vol, total_vol,
                                out=np.zeros_like(net_signed_vol), where=total_vol > 0)

    # ---- Feature 2: top-of-book quote imbalance ----
    bid_qty = book["bid_qty"].to_numpy()
    ask_qty = book["ask_qty"].to_numpy()
    denom = bid_qty + ask_qty
    quote_imbalance = np.divide(bid_qty - ask_qty, denom,
                                 out=np.zeros_like(bid_qty), where=denom > 0)

    # ---- Feature 3: trailing realized vol (count-based rolling window;
    # snapshots arrive at an approximately-100ms but not exactly uniform
    # cadence, so this is an approximation of a fixed time window,
    # documented as such) ----
    mid_diff = np.diff(mids, prepend=mids[0])
    trailing_vol = pd.Series(mid_diff).rolling(vol_lookback_obs, min_periods=2).std().to_numpy()

    # ---- Feature 4: trailing momentum (signed mid move over lookback) ----
    mid_momentum = mids - np.concatenate([
        np.full(momentum_lookback_obs, mids[0]), mids[:-momentum_lookback_obs]
    ])

    # ---- Feature 5: currently quoted (real) top-of-book spread -- a
    # liquidity-context feature (thin/wide book at this instant) ----
    quoted_spread = book["best_ask"].to_numpy() - book["best_bid"].to_numpy()

    frame = book.copy()
    frame["net_signed_vol"] = net_signed_vol
    frame["trade_count"] = trade_count
    frame["ofi_normalized"] = ofi_normalized
    frame["quote_imbalance"] = quote_imbalance
    frame["trailing_vol"] = trailing_vol
    frame["mid_momentum"] = mid_momentum
    frame["quoted_spread"] = quoted_spread

    # ---- Label: forward move vs volatility-scaled threshold ----
    sigma_per_sec = estimate_sigma_per_sec(book)
    threshold = k_sigma * sigma_per_sec * math.sqrt(horizon_sec)

    idx_future = np.searchsorted(ts, ts + horizon_sec, side="left")
    n = len(ts)
    valid = idx_future < n
    future_mid = np.full(n, np.nan)
    future_mid[valid] = mids[idx_future[valid]]
    forward_return = future_mid - mids
    informed = np.where(valid, (np.abs(forward_return) > threshold).astype(float), np.nan)

    frame["forward_return"] = forward_return
    frame["informed"] = informed

    # first vol_lookback_obs/momentum_lookback_obs rows have partial/NaN
    # trailing features -- fine for training (dropped via dropna), but we
    # keep them in the returned frame so inference/backtest code can still
    # score every timestamp (using whatever trailing data is available).
    frame["trailing_vol"] = frame["trailing_vol"].fillna(0.0)

    meta = {
        "sigma_per_sec": sigma_per_sec,
        "horizon_sec": horizon_sec,
        "k_sigma": k_sigma,
        "threshold_dollars": threshold,
        "trailing_window_sec": trailing_window_sec,
        "n_rows": n,
        "n_labeled": int(valid.sum()),
        "informed_rate": float(np.nanmean(informed)),
        "expected_adverse_move_given_informed": float(
            np.nanmean(np.abs(forward_return)[informed == 1.0])
        ) if np.nansum(informed == 1.0) > 0 else None,
        "feature_columns": FEATURE_COLUMNS,
        "label_definition": (
            "informed(t) = 1{ |mid(t+horizon_sec) - mid(t)| > k_sigma * sigma_per_sec "
            "* sqrt(horizon_sec) } -- symmetric, volatility-scaled forward price-move "
            "threshold; see module docstring."
        ),
    }
    return frame, meta
