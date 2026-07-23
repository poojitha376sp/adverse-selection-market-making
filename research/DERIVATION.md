# Deriving the Avellaneda-Stoikov Reservation Price and Optimal Spread

A first-principles working-through of Avellaneda & Stoikov (2008), done by
hand before writing a line of quoting code. The goal isn't to reproduce the
paper's proof line by line — it's to actually *rebuild* the argument well
enough to know where every term in the final formulas comes from, and,
just as importantly, where the model stops being valid. That second part
is what motivates Part 3 of this project.

Notation follows the cheatsheet (`research/CHEATSHEET.md`, Section 1).

---

## 1. Setup

A single market maker quotes a bid `s − δᵇ` and an ask `s + δᵃ` around a
reference mid-price `s`, `δᵃ, δᵇ ≥ 0` chosen by the maker. Three
ingredients:

**Mid-price.** No drift, pure diffusion:

```
dS_t = σ dW_t
```

(`σ` constant — the maker isn't trying to predict direction, only to
manage the risk of being long or short when the horizon `T` arrives.)

**Order arrivals.** Buy and sell market orders arrive as independent
Poisson processes. A quote posted a distance `δ` from mid gets hit at
intensity

```
λ(δ) = A·e^(−κδ),        A, κ > 0
```

— fewer fills the further you quote from mid, decaying exponentially,
governed by a liquidity/competition parameter `κ` (a denser book / more
competition ⇒ larger `κ` ⇒ your fill probability collapses faster as you
step away from the touch) and a base intensity `A`.

**Inventory and wealth.** Let `q_t` be the maker's inventory (can be
negative — short) and `x_t` cash. `N^bid_t`, `N^ask_t` are the counting
processes for bid-side and ask-side fills, with intensities `λ(δᵇ)` and
`λ(δᵃ)` respectively (each fill is one unit of the asset, for simplicity).
Every bid fill buys one unit at `s − δᵇ`; every ask fill sells one unit at
`s + δᵃ`:

```
dq_t = dN^bid_t − dN^ask_t
dx_t = (s + δᵃ) dN^ask_t − (s − δᵇ) dN^bid_t
```

**Preferences.** CARA (exponential) utility with risk-aversion `γ > 0`,
applied to terminal wealth *marked to market* — cash plus whatever
inventory is left, valued at the terminal mid-price:

```
U = E[ −exp( −γ (x_T + q_T·S_T) ) ]
```

The maker picks `(δᵃ_t, δᵇ_t)_{0≤t≤T}`, adapted to the filtration, to
maximize `U`. Define the value function

```
V(x, s, q, t) = max_{(δᵃ,δᵇ)} E_t[ −exp(−γ(x_T + q_T S_T)) ]
```

with `V(x, s, q, T) = −exp(−γ(x + qs))` as the terminal boundary
condition (no more trading, inventory settles at the current mid — no
penalty or bonus beyond that mark).

---

## 2. The HJB equation

Standard stochastic-control machinery: `V` should satisfy, along an
optimal path, a Hamilton-Jacobi-Bellman equation combining (i) the
diffusive part from `dS_t`, and (ii) two independent jump terms — one for
a bid fill, one for an ask fill — each contributing its Poisson-arrival
expected change in `V`, maximized over the maker's choice of that side's
distance:

```
0 = ∂V/∂t + (σ²/2)·∂²V/∂s²
    + max_{δᵇ≥0} λ(δᵇ)·[ V(x−(s−δᵇ), s, q+1, t) − V(x, s, q, t) ]
    + max_{δᵃ≥0} λ(δᵃ)·[ V(x+(s+δᵃ), s, q−1, t) − V(x, s, q, t) ]
```

The two maximizations decouple cleanly here because `δᵃ` only appears in
the ask-jump term and `δᵇ` only in the bid-jump term — a genuinely useful
simplification, and the reason the bid and ask problems can be solved
side by side rather than jointly.

This is a nonlinear PDE in `(s, q, t)` (nonlinear because of the `max`
operators sitting inside it) coupled across the integer inventory levels
`q`. Solving it directly is not tractable. The next step is the one piece
of real cleverness in the paper: an ansatz that exploits the exponential
utility to strip out the "irrelevant" `x` and `s` dependence and reduce
the problem to something solvable.

---

## 3. The CARA ansatz: why exponential utility linearizes this

Try

```
V(x, s, q, t) = −exp( −γ(x + qs) ) · g(q, t)
```

The `exp(−γ(x+qs))` factor is exactly the CARA utility of *current*
mark-to-market wealth; `g(q,t) > 0` is left as an unknown function
capturing the *residual* value/risk correction from continuing to trade
optimally from here to `T`. (`g(q,T) = 1` matches the terminal condition.)

This works because a bid fill changes `x → x−(s−δᵇ)` and `q → q+1`
*simultaneously*, and the combination `x + qs` transforms very cleanly
under that joint jump:

```
x' + q's = [x − (s−δᵇ)] + (q+1)s = (x + qs) + δᵇ
```

so

```
V(x', s, q+1, t) = −exp(−γ(x+qs+δᵇ))·g(q+1,t) = −exp(−γ(x+qs))·e^{−γδᵇ}·g(q+1,t)
```

Symmetrically for an ask fill, `x'' + q''s = (x+qs) + δᵃ`, giving

```
V(x'', s, q−1, t) = −exp(−γ(x+qs))·e^{−γδᵃ}·g(q−1,t)
```

Every term in the HJB equation now carries the same overall factor
`−exp(−γ(x+qs))`. Dividing it out (it's strictly negative, so `max`
flips to `min`) collapses the PDE in `(x,s,q,t)` down to an equation for
`g` in `(q,t)` alone:

```
0 = g_t + (σ²/2)γ²q²·g
    + min_{δᵇ≥0} λ(δᵇ)·[ e^{−γδᵇ}g(q+1,t) − g(q,t) ]
    + min_{δᵃ≥0} λ(δᵃ)·[ e^{−γδᵃ}g(q−1,t) − g(q,t) ]
```

(The `(σ²/2)γ²q²g` term falls out because `∂²V/∂s² = −γ²q²·V` under the
ansatz, since `g` doesn't depend on `s`.)

This is the linearization the exercise asks for: `x` and `s` have
dropped out of the unknown entirely. What's left is a much smaller
problem — a system of ODEs indexed by the integer inventory level `q`,
each coupled only to its neighbors `q±1`.

---

## 4. Solving the inner optimization: optimal quote distances

For fixed `g(q,t)`, `g(q+1,t)`, minimize over `δᵇ`:

```
f(δᵇ) = A·e^{−κδᵇ}·[ e^{−γδᵇ}g(q+1,t) − g(q,t) ]
       = A·[ e^{−(κ+γ)δᵇ}g(q+1,t) − e^{−κδᵇ}g(q,t) ]
```

`f'(δᵇ) = 0` gives `κ·g(q,t)·e^{−κδᵇ} = (κ+γ)·g(q+1,t)·e^{−(κ+γ)δᵇ}`, i.e.

```
e^{−γδᵇ*} = κ·g(q,t) / [(κ+γ)·g(q+1,t)]

δᵇ* = (1/γ)·ln(1 + γ/κ)  +  (1/γ)·ln[ g(q+1,t) / g(q,t) ]
```

and, by the mirror-image argument on the ask side,

```
δᵃ* = (1/γ)·ln(1 + γ/κ)  +  (1/γ)·ln[ g(q−1,t) / g(q,t) ]
```

This already has a clean interpretation: **each optimal offset is a fixed
"myopic markup" `(1/γ)ln(1+γ/κ)` plus an inventory-dependent correction**
that compares `g` one step further from vs. one step closer to flat
inventory. If holding one more unit (`g(q+1,t)`) is worth *less* than
holding the current amount (`g(q,t)`) — i.e. `q` is already positive and
adding more is unwelcome — the bid backs away (`δᵇ*` grows); if it's
worth *more*, the bid gets more aggressive. Exactly the skewing behavior
you'd want, derived rather than assumed.

Because the reservation price is defined as the midpoint of the two
optimal quotes, `r = s + (δᵃ* − δᵇ*)/2`, we already have, in general form,

```
r(s,q,t) = s + (1/2γ)·ln[ g(q−1,t) / g(q+1,t) ]

δᵃ* + δᵇ* = (2/γ)·ln(1+γ/κ) + (1/γ)·ln[ g(q−1,t)·g(q+1,t) / g(q,t)² ]
```

Everything now hinges on `g(q,t)`.

---

## 5. Pinning down g(q,t): the approximation step

Plugging the optimal `δᵃ*, δᵇ*` back in leaves a *nonlinear* difference
equation for `g(q,t)` (the optimized `λ(δ*)` term involves `g`-ratios
raised to a fractional power `κ/γ` — there is no clean closed form here in
general). This is exactly the point in the derivation where
Avellaneda-Stoikov stop being exact and switch to an asymptotic
approximation, and it's worth being honest about that rather than
pretending the final formula falls out for free.

The approximation: consider the *boundary* case `κ → ∞` — quotes further
than an infinitesimal distance from mid never get filled, so the maker is
effectively locked into inventory `q` until `T` with no further trading.
Then `g(q,t)` has an exact, elementary closed form: since
`S_T − s ~ N(0, σ²(T−t))`,

```
g(q,t) = E[ e^{−γq(S_T−s)} ] = exp( ½·γ²σ²(T−t)·q² )
```

(a standard Gaussian moment-generating-function identity). This satisfies
the terminal condition `g(q,T)=1` and captures exactly the risk a CARA
agent assigns to being marked-to-market on `q` units of pure Brownian
noise for the remaining time `T−t` — a **pure inventory-risk penalty**,
quadratic in `q` and linear in remaining variance `σ²(T−t)`.

Avellaneda-Stoikov's argument (and the one adopted here) is that this
`κ→∞` solution is also the correct **leading-order term** of the true
`g(q,t)` for finite `κ`: continuing to quote and trade perturbs `g` only
at first order in `q` (it lets the maker *partially* offset the
Brownian risk by trading it away), it does not change the dominant `q²`
curvature, which is set by the diffusion the maker is exposed to over
`T−t` regardless of how well it quotes. So we use

```
g(q,t) ≈ exp( ½·γ²σ²(T−t)·q² )
```

as the working approximation — exact in the no-further-trading limit,
leading-order-correct in general, and (this is the part worth checking,
not just asserting) it reproduces the textbook formulas exactly:

```
ln[g(q−1,t)/g(q+1,t)] = ½γ²σ²(T−t)·[(q−1)² − (q+1)²] = −2γ²σ²(T−t)·q

⟹ r(s,q,t) = s + (1/2γ)·(−2γ²σ²(T−t)q) = s − q·γ·σ²·(T−t)          ✓

ln[g(q−1,t)g(q+1,t)/g(q,t)²] = ½γ²σ²(T−t)·[(q−1)²+(q+1)²−2q²] = γ²σ²(T−t)

⟹ δᵃ*+δᵇ* = (2/γ)ln(1+γ/κ) + (1/γ)·γ²σ²(T−t) = γσ²(T−t) + (2/γ)ln(1+γ/κ)   ✓
```

(Both identities were checked symbolically with sympy while writing this
note — see the arithmetic above; they collapse to the target formulas
exactly, not approximately, once `g` is substituted.)

---

## 6. Reading the three terms

```
r(s,q,t) = s − q·γ·σ²·(T−t)

δᵃ+δᵇ = γ·σ²·(T−t) + (2/γ)·ln(1+γ/κ)
```

**`−q·γ·σ²·(T−t)` — the inventory-skew term.** This is the whole
reservation price shifted away from the raw mid, in the direction that
makes the maker *want* to trade back toward `q=0`. If `q>0` (long), the
term is negative — quotes shift down, so the ask becomes relatively more
attractive to hit (encouraging sells) and the bid relatively less
aggressive (discouraging further buys). The size of the shift scales with
how much inventory you're carrying (`q`), how much that inventory could
move against you (`σ²`), how much you personally dislike that risk (`γ`),
and how long you're stuck holding it (`T−t`) — shrinking to zero as the
horizon approaches, since a wrong position matters less the less time it
has left to move.

**`γ·σ²·(T−t)` — the volatility/horizon widening term.** This is exactly
twice the magnitude of the per-unit-inventory skew coefficient above —
which makes sense, since it comes from the *curvature* of the same
`g(q,t) = exp(½γ²σ²(T−t)q²)` risk penalty (the reservation price uses the
first-difference/slope of `ln g` in `q`; the spread uses the
second-difference/curvature). It says: the further the mid-price could
plausibly wander before `T` (`σ²(T−t)`), and the more that wandering hurts
a risk-averse agent (`γ`), the wider both quotes sit from the reservation
price — pure inventory-risk compensation, present even at `q=0`, because
even a flat position risks *becoming* unwanted inventory the instant a
fill lands.

**`(2/γ)·ln(1+γ/κ)` — the liquidity/markup term.** This is what survives
in the trivial one-shot case with **no** inventory-risk consideration at
all (`σ=0`, or equivalently the `q=0`, "myopic" slice of the problem) —
solving `min_δ A e^{−κδ}(e^{−γδ}−1) = 0` directly gives exactly
`δ* = (1/γ)ln(1+γ/κ)` per side (verified by direct differentiation).
Two intuitive checks on it: (i) as `κ→∞` (infinitely liquid/competitive
book, fill probability collapses instantly with distance) the term → 0 —
no room to charge a markup when any spread loses the trade; (ii) as
`γ→0` (risk-neutral), `ln(1+γ/κ) ≈ γ/κ`, so the term → `2/κ` — a maker
with *no* risk aversion at all still charges a positive markup, because
`λ(δ)=Ae^{−κδ}` behaves like a downward-sloping demand curve and `1/κ`
is the standard monopolist-style optimal markup against that elasticity.
Risk aversion (`γ>0`) inflates this baseline markup further via the
concave `ln(1+γ/κ)` — the maker demands extra compensation for taking on
the *risk* of holding an unwanted unit, on top of the pure
market-power markup.

---

## 7. The model's blind spot

Every quantity in these formulas — `A`, `κ`, `σ` — is **held constant**
for the whole horizon `[0,T]`. The HJB equation was set up with
`λ(δ)=Ae^{−κδ}` and `dS_t=σdW_t` as fixed primitives, and the derivation
above never questioned that. Concretely:

- `σ` doesn't move even though realized volatility clusters and spikes
  around informed activity — the widening term `γσ²(T−t)` only reacts
  to volatility *after* it has already happened and been re-estimated
  into a new constant.
- `κ` and `A` don't move even though a burst of one-sided, urgent flow
  (the kind associated with informed trading) changes the *effective*
  fill-intensity profile the maker is actually facing right now.
- There is no state variable at all for "does this order look informed."
  The only state the optimal control conditions on is `(s, q, t)` —
  price, inventory, and time. A market maker running this model exactly
  as derived has literally no mechanism to distinguish a fill from a
  liquidity trader from a fill from someone who knows the mid is about to
  jump; both get the identical quote, because the model was never given
  an input that could tell them apart.

That is the gap: the reservation-price/spread machinery above is a
complete, correct answer to "how should I quote given *inventory* risk
under **constant**, exogenous order-flow and volatility parameters" — and
essentially silent on adverse selection, which is a statement about the
non-constancy and informativeness of that flow. Part 3 of this project
is exactly about replacing the "constants" `A`, `κ`, `σ` (or adding a
fourth term to the spread) with something that reacts to a measured
informedness/toxicity signal, rather than re-deriving a whole new HJB
from scratch for every possible signal specification.

---

## 8. Why an adverse-selection term is needed at all: Glosten-Milgrom

Glosten & Milgrom (1985) make the point in the starkest possible setting:
strip out inventory risk *entirely* — a **risk-neutral** market maker,
facing a **zero-expected-profit** competitive constraint, with **no**
market power and **no** `γ`, `σ`, or `T` anywhere in the model. Each
incoming order is, with some probability, from a trader who knows the
asset's true terminal value and, otherwise, from an uninformative noise
trader. Because the maker can't tell which is which order by order, they
still have to set the ask above and the bid below their current
expectation of value — sized precisely so that expected losses to
informed traders on that side are offset by expected gains from noise
traders. **A strictly positive bid-ask spread survives even though every
other reason for one (inventory risk, market power, order-processing
cost) has been deliberately removed** — spread as a pure adverse-selection
premium, nothing else.

That is the conceptual case for treating adverse selection as an
*additive*, structurally distinct effect rather than something the
Avellaneda-Stoikov machinery already captures implicitly through `σ`: the
`γσ²(T−t)` and `(2/γ)ln(1+γ/κ)` terms above both go to zero as `γ→0`
except for the pure-markup piece, and none of them reference *who* is on
the other side of the trade at all — whereas Glosten-Milgrom's spread
requires no risk aversion or market power whatsoever and is entirely a
function of *how informative* order flow is. The two mechanisms are
complementary, not overlapping: AS prices the risk of carrying inventory
into an uncertain future; Glosten-Milgrom prices the risk that the person
you just traded with already knows something about that future you
don't. A market maker who only implements the AS formulas above is
pricing the first risk correctly and the second one not at all — which is
precisely the mechanism this project adds on top in Part 3, informed by
an empirical toxicity/order-flow signal (from the sister
`order-flow-imbalance` / `vpin-flow-toxicity` projects) rather than
Glosten-Milgrom's stylized fixed-probability-of-informed-trade
assumption.
