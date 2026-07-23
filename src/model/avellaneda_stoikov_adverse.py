#!/usr/bin/env python3
"""
avellaneda_stoikov_adverse.py

Part 3 / Phase 4: the adverse-selection extension on top of the Part 2
baseline (`avellaneda_stoikov.py`). Both variants consume `p_informed`,
the ML classifier's predicted probability that the flow right now is
informed (`ml_informedness_classifier.py`'s continuous output, NOT its
thresholded binary label -- the quoting extension uses the raw
probability so it can react proportionally rather than as an on/off
switch).

Two variants, per README's Phase 4 / DERIVATION.md Sec 8:

(a) HEURISTIC OVERLAY (`heuristic_quotes`)
    Take the baseline optimal total spread unchanged and scale it by a
    multiplier that increases with predicted informedness:

        total_spread' = total_spread_baseline * (1 + beta * p_informed)

    No change to the reservation price, no economic derivation -- this
    is exactly the "scale the baseline spread by a function of the
    toxicity signal" heuristic the README's Phase 4 describes as
    variant (a). Simple, transparent, easy to reason about, but not
    grounded in re-solving the control problem.

(b) PRINCIPLED VARIANT (`principled_quotes`)
    A documented, numeric APPROXIMATION of adding toxicity as an
    explicit state-dependent adjustment, per DERIVATION.md Sec 7-8. A
    full from-scratch HJB re-solve with p_informed as an added
    stochastic state variable is out of scope for one session (that is
    genuinely GLFT/Cartea-Jaimungal-lineage research, not a one-file
    implementation) -- so this variant instead combines TWO separately
    justified, smaller pieces, and is explicit in these comments that
    it is an approximation, not a rigorous new derivation:

    (b.1) State-dependent effective risk aversion. DERIVATION.md Sec 6
        shows both the inventory-skew term AND the volatility-widening
        term are driven by gamma (risk aversion) multiplying sigma^2 --
        i.e. gamma controls how much the maker is willing to pay (via
        skew and width) to avoid holding risk. Elevated toxicity is
        exactly a situation where holding inventory is riskier than
        sigma alone captures (an informed counterparty is more likely
        to be right about the *direction* the price is about to move,
        which is additional risk sigma^2 -- estimated from realized,
        mostly-uninformed-flow, historical variance -- does not price
        in). Scaling gamma by (1 + eta*p_informed) is a standard,
        easy-to-justify way to inject a state-dependent risk multiplier
        into an already-solved HJB formula: it reuses the *exact* AS
        formulas (so it stays consistent with DERIVATION.md's solved
        g(q,t)) but evaluates them at a state-dependent effective risk
        aversion instead of a constant one -- both the skew and the
        width react coherently (not just the width, unlike (a)), because
        both terms in the AS closed form share the same gamma factor.
        This is a leading-order-style modeling choice, not a re-solved
        HJB with p_informed as a genuine new state variable coupled
        into the PDE's diffusion/jump terms -- an honest re-derivation
        would need to redo Sec 2-5 of DERIVATION.md with p_informed(t)
        as an added correlated state, which is out of scope here.

    (b.2) Glosten-Milgrom-style additive breakeven premium. Per
        DERIVATION.md Sec 8: a risk-neutral, zero-expected-profit maker
        facing a probability p of trading with an informed counterparty
        who costs the maker `L` dollars per unit (in expectation) must
        charge at least `p*L` in extra half-spread on THAT side to break
        even. Approximating `p` by `p_informed` and `L` by the
        classifier's own training-set estimate of the expected forward
        adverse move conditional on being labeled informed
        (`informedness_signal.py`'s `expected_adverse_move_given_informed`,
        threaded through from `ml_informedness_classifier.py`'s output),
        gives a direct, literature-grounded additive term:

            gm_premium = 2 * p_informed * expected_adverse_move

        (the factor of 2 because the premium must be added to BOTH the
        bid and ask side of the total spread, symmetric with the
        symmetric/undirected informedness label used here). This is
        the "prices the risk that the person you just traded with
        already knows something" mechanism DERIVATION.md Sec 8 argues
        is structurally distinct from (and complementary to) pure
        inventory risk -- (b.1) alone does not add this, since it only
        rescales the existing inventory-risk formulas.

    Combined:

        gamma_eff = gamma * (1 + eta * p_informed)
        r, spread_AS  = baseline AS formulas evaluated at gamma_eff
        total_spread' = spread_AS + gm_premium
        bid' = r - total_spread'/2 ,  ask' = r + total_spread'/2

    Both `eta` (risk-aversion sensitivity) and `expected_adverse_move`
    are explicit, inspectable parameters on `AdverseParams` -- not
    hidden magic numbers -- so the backtest can report exactly what was
    used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.model.avellaneda_stoikov import ASParams, quotes


@dataclass(frozen=True)
class AdverseParams:
    """Extends the baseline ASParams with the two extension-specific knobs.

    base                   : the Part 2 baseline ASParams (gamma, sigma, kappa, A, T)
    heuristic_beta         : (a) spread multiplier sensitivity; total_spread
                              scaled by (1 + heuristic_beta * p_informed).
                              beta=1.0 means fully-informed flow (p=1) doubles
                              the baseline spread.
    principled_eta         : (b.1) effective-risk-aversion sensitivity;
                              gamma_eff = gamma * (1 + principled_eta * p_informed).
    expected_adverse_move  : (b.2) the Glosten-Milgrom-style expected loss
                              per unit when trading with informed flow, in
                              dollars -- estimated empirically from the ML
                              classifier's own training data (mean |forward
                              price move| conditional on the informed
                              label), NOT hand-picked.
    """
    base: ASParams
    heuristic_beta: float = 1.0
    principled_eta: float = 1.0
    expected_adverse_move: float = 0.0


def heuristic_quotes(s: float, q: float, t: float, p_informed: float, params: AdverseParams):
    """Variant (a): baseline reservation price unchanged; total spread
    scaled by (1 + heuristic_beta * p_informed). See module docstring."""
    p = max(0.0, min(1.0, p_informed))
    bid, ask, r, spread = quotes(s, q, t, params.base)
    mult = 1.0 + params.heuristic_beta * p
    new_spread = spread * mult
    half = new_spread / 2.0
    return r - half, r + half, r, new_spread


def principled_quotes(s: float, q: float, t: float, p_informed: float, params: AdverseParams):
    """Variant (b): state-dependent effective risk aversion (b.1) plus a
    Glosten-Milgrom-style additive breakeven premium (b.2). See module
    docstring for the full justification and explicit approximation
    caveat."""
    p = max(0.0, min(1.0, p_informed))
    gamma_eff = params.base.gamma * (1.0 + params.principled_eta * p)
    eff_params = ASParams(
        gamma=gamma_eff, sigma=params.base.sigma, kappa=params.base.kappa,
        A=params.base.A, T=params.base.T,
    )
    bid, ask, r, spread_as = quotes(s, q, t, eff_params)
    gm_premium = 2.0 * p * params.expected_adverse_move
    total_spread = spread_as + gm_premium
    half = total_spread / 2.0
    return r - half, r + half, r, total_spread
