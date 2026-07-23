#!/usr/bin/env python3
"""
avellaneda_stoikov.py

The plain (non-adverse-selection-aware) Avellaneda-Stoikov quoting model,
as derived by hand in `research/DERIVATION.md`. This is Part 2 / Phase 3
of the project: the baseline the Part 3 adverse-selection extension will
sit on top of.

Only two formulas are implemented here (Section 6 of DERIVATION.md):

    reservation price:   r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

    optimal total spread: delta_a + delta_b
                         = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)

with the individual quotes placed symmetrically around r:

    bid = r - (delta_a + delta_b) / 2
    ask = r + (delta_a + delta_b) / 2

State: (s, q, t, gamma, sigma, kappa, A, T). Note `A` (the base fill
intensity in lambda(delta) = A * exp(-kappa*delta)) does not appear in
the reservation-price/spread formulas themselves -- it only matters for
simulating/estimating fill *probabilities* off the quotes, which is what
the backtest harness (`src/backtest/`) uses it for. It is threaded
through this module's dataclass anyway so the full state described in
DERIVATION.md travels together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ASParams:
    """Model parameters, constant over the horizon (see DERIVATION.md Sec 7
    for the explicit acknowledgement that this constancy is the model's
    blind spot -- Part 3 relaxes it).

    gamma : risk aversion (maker's own preference, design choice)
    sigma : per-second mid-price volatility (should be estimated from data)
    kappa : fill-intensity decay-with-distance parameter (illustrative or estimated)
    A     : base fill intensity at zero distance (illustrative or estimated)
    T     : horizon length in seconds (design choice)
    """
    gamma: float
    sigma: float
    kappa: float
    A: float
    T: float


def reservation_price(s: float, q: float, t: float, params: ASParams) -> float:
    """r(s,q,t) = s - q * gamma * sigma^2 * (T - t)"""
    tau = max(params.T - t, 0.0)
    return s - q * params.gamma * params.sigma ** 2 * tau


def optimal_spread(t: float, params: ASParams) -> float:
    """delta_a + delta_b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/kappa)"""
    tau = max(params.T - t, 0.0)
    inventory_term = params.gamma * params.sigma ** 2 * tau
    markup_term = (2.0 / params.gamma) * math.log(1.0 + params.gamma / params.kappa)
    return inventory_term + markup_term


def quotes(s: float, q: float, t: float, params: ASParams) -> tuple[float, float, float, float]:
    """Return (bid, ask, reservation_price, total_spread) for state (s,q,t).

    bid/ask are placed symmetrically around r, per DERIVATION.md Sec 4-6:
    the reservation price is defined as the midpoint of the two optimal
    quote distances, so delta_a == delta_b == total_spread/2 once the
    (q-1)/(q+1) asymmetry has already been folded into r itself.
    """
    r = reservation_price(s, q, t, params)
    spread = optimal_spread(t, params)
    half = spread / 2.0
    return r - half, r + half, r, spread


def fill_intensity(delta: float, params: ASParams) -> float:
    """lambda(delta) = A * exp(-kappa * delta), delta >= 0 distance from mid."""
    if delta < 0:
        return params.A
    return params.A * math.exp(-params.kappa * delta)
