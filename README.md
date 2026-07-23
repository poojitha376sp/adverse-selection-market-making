# Market Making with Adverse Selection

QuantFest (IICPC) project. Extends the classic Avellaneda-Stoikov
inventory-risk market-making model to account for the probability that an
incoming order is *informed* — so quoted spreads widen automatically when
flow looks toxic, instead of running a fixed formula blind to who's on the
other side of the trade. This is the difference between a textbook
market-making model and how a real desk actually manages adverse
selection.

Part of a 4-project microstructure suite for QuantFest:
[order-flow-imbalance](https://github.com/poojitha376sp/order-flow-imbalance) ·
[vpin-flow-toxicity](https://github.com/poojitha376sp/vpin-flow-toxicity) ·
[hawkes-fill-probability](https://github.com/poojitha376sp/hawkes-fill-probability)

Status: planning phase.

See [`research/CHEATSHEET.md`](research/CHEATSHEET.md) for the working
reference doc — academic papers, practitioner writeups, relevant
conferences, and what market-making/HFT firms publicly disclose about
inventory-risk and adverse-selection-aware quoting. Kept up to date as
research continues.

---

## Execution Roadmap (4 parts)

Built day by day rather than in one sitting.

- [x] **Part 1 — Foundations** (Phase 1 Research + Phase 2 Data acquisition):
  derive the Avellaneda-Stoikov reservation price/spread from the HJB
  equation by hand, stand up a real L2 + trade data pipeline (reusing the
  informed-flow proxy from order-flow-imbalance / vpin-flow-toxicity). See
  [`research/DERIVATION.md`](research/DERIVATION.md) and
  [`src/data/collect_market_data.py`](src/data/collect_market_data.py).
- [x] **Part 2 — Core Mechanism** (Phase 3 Baseline implementation): the
  standard Avellaneda-Stoikov market maker, backtested for a reference
  PnL/inventory path. Reservation-price/spread formulas implemented in
  [`src/model/avellaneda_stoikov.py`](src/model/avellaneda_stoikov.py);
  parameter estimation (real realized volatility from captured
  mid-price data, a crude real-data-informed fill-intensity fit, plus
  the risk-aversion/horizon design choices) in
  [`src/model/estimate_params.py`](src/model/estimate_params.py); the
  chronological real-data backtest harness in
  [`src/backtest/backtest_baseline.py`](src/backtest/backtest_baseline.py).
  Backtested against a fresh ~240s BTCUSDT capture
  (`data/raw/depth_btcusdt_20260723_094653.jsonl` +
  `data/raw/trades_btcusdt_20260723_094653.jsonl`, gitignored — capture
  is regenerable via `src/data/collect_market_data.py --duration 240`):
  9 hypothetical fills (3 bid, 6 ask), inventory ranged −3 to 0 units,
  naive mark-to-market PnL of −$2.96 over the window. Output series
  written to `data/processed/` (gitignored, code only).
- [ ] **Part 3 — Fitting & Extension** (Phase 4 Adverse-selection
  extension): augment the state with the informedness signal, both the
  heuristic-overlay and principled re-solved variants. The informedness
  signal itself is a **classical ML classifier** (gradient boosting) in
  this part; an online-Bayesian deep-learning version is a documented
  Part 4 stretch. See "AI/ML plan" below and `research/CHEATSHEET.md`.
- [ ] **Part 4 — Validation & Deliverables** (Phase 5 + 6): baseline vs.
  extension backtest, adverse-selection-cost comparison, final write-up.

Stretch goals (full 3-signal integration, multi-asset inventory risk) are
a bonus beyond these 4 parts, not required for core completion.

---

## Plan of Approach

### Phase 1 — Research
- Baseline reference: Avellaneda, Stoikov (2008), *"High-Frequency Trading
  in a Limit Order Book"* — the classic inventory-risk quoting model
  (reservation price + optimal spread from a Hamilton-Jacobi-Bellman
  solution). Understand its blind spot: it treats all incoming flow as
  equally likely to be uninformed.
- Primary extension reference: Guéant, Lehalle, Fernandez-Tapia (2012/2013),
  *"Dealing with the Inventory Risk: A Solution to the Market Making
  Problem"* — closed-form/tractable extensions of Avellaneda-Stoikov.
- Adverse-selection reference: Cartea, Jaimungal, and coauthors on market
  making with informed order flow — how a toxicity/informedness signal
  enters the quoting problem as a state variable that widens the optimal
  spread.
- Write a short internal note deriving the baseline Avellaneda-Stoikov
  reservation price and optimal spread from the HJB equation before
  extending it — the extension only means something if the baseline is
  understood cold.

### Phase 2 — Data acquisition
- Primary candidate: Level-2 order book + trade data from a crypto
  exchange websocket (Binance/Bybit), matching the granularity needed to
  both simulate quoting and estimate an informedness signal.
- Reuse (or directly depend on) the informed-flow proxy signal from
  [order-flow-imbalance](https://github.com/poojitha376sp/order-flow-imbalance)
  and/or [vpin-flow-toxicity](https://github.com/poojitha376sp/vpin-flow-toxicity)
  as the "probability the incoming order is informed" input, rather than
  inventing a fourth, unvalidated proxy.

### Phase 3 — Baseline implementation
- Implement the standard Avellaneda-Stoikov market maker: reservation
  price as a function of inventory and time-to-horizon, optimal
  bid/ask spread as a function of volatility and risk aversion.
- Backtest the baseline on historical/simulated data to get a reference
  PnL, inventory path, and realized adverse-selection cost (losses on
  fills immediately followed by adverse price moves).

### Phase 4 — Adverse-selection extension
- Augment the state space with the informedness signal (OFI/VPIN-derived)
  and re-derive (or numerically solve) the optimal quoting policy so that
  the spread widens and/or skews when the signal indicates elevated
  toxicity.
- Compare two variants: (a) heuristic overlay — scale the baseline spread
  by a function of the toxicity signal; (b) principled — re-solve the
  control problem with toxicity as an explicit state variable, per the
  Guéant-Lehalle-Fernandez-Tapia-style formulation. Report both, since the
  gap between them is itself a useful result.

### AI/ML plan
ML is a core part of this project, not an afterthought — staged so a solid
classical baseline exists before anything heavier (decision recorded
2026-07-23, see `research/CHEATSHEET.md` for full citations). The "AI/ML"
question here is specifically *what generates the informedness signal*
that Phase 4 feeds into the quoting policy:
- **Now (Part 3, classical ML)**: a gradient boosting classifier trained
  on order-flow features (from order-flow-imbalance / vpin-flow-toxicity,
  or engineered directly from this project's own captured data) predicting
  "is this flow currently informed" — this is the signal both the
  heuristic-overlay and principled variants above consume. Classical and
  fast to validate before anything fancier.
- **Later (Part 4 / stretch, deep learning)**: Cartea, Duran-Martin &
  Sánchez-Betancourt's "Detecting Toxic Flow" (CHEATSHEET.md §1) — an
  online Bayesian neural network ("PULSE") updating in under 1ms per
  trade — is the natural upgrade path for the signal generator itself,
  swapped in without changing Phase 4's quoting logic. Keep Optiver's own
  finding front of mind while evaluating it (CHEATSHEET.md §4): their AI
  models could *recognize* adverse selection but still traded at negative
  EV against informed counterparties, so "the classifier is accurate" and
  "the quoting policy that consumes it actually protects PnL" are two
  separate claims — this project's Phase 5 validation needs to check both,
  not assume the second follows from the first.

### Phase 5 — Validation (the part most student projects get wrong)
- Backtest baseline vs. adverse-selection-aware quoting on the same
  held-out data, using a realistic fill-simulation model (ideally the
  fill-probability estimator from
  [hawkes-fill-probability](https://github.com/poojitha376sp/hawkes-fill-probability),
  not a naive "always fill at quoted price" assumption).
- Report inventory risk (variance of inventory path), realized spread
  captured, and adverse-selection cost specifically — the whole point of
  the extension is that this last number should drop relative to the
  baseline without collapsing fill rate to zero.
- Stress-test both strategies over the highest-toxicity periods identified
  by VPIN in the data, since that's exactly where the extension is
  supposed to earn its keep.

### Phase 6 — Deliverables
- Quoting engine supporting both baseline and adverse-selection-aware
  modes, run against the same backtest harness.
- Comparison report: PnL, inventory variance, adverse-selection cost,
  fill rate, spread captured — baseline vs. extension, with a breakdown by
  toxicity regime.
- Write-up connecting results back to the theory: does the data support
  the qualitative prediction (wider spreads in high-toxicity regimes
  reduce adverse-selection losses) and by how much.

### Stretch goals
- Full integration of all three upstream signals (OFI, VPIN, Hawkes fill
  probability) into a single quoting decision, as the "capstone" that ties
  the 4-project suite together.
- Multi-asset inventory risk (quoting correlated instruments
  simultaneously) as a further extension beyond the single-asset case.

---

## Research papers to read next
- Avellaneda, Stoikov (2008) — *High-Frequency Trading in a Limit Order
  Book*
- Guéant, Lehalle, Fernandez-Tapia (2013) — *Dealing with the Inventory
  Risk: A Solution to the Market Making Problem*
- Cartea, Jaimungal, Ricci — *Buy Low Sell High: A High-Frequency Trading
  Perspective* (market making with informed flow)
- Glosten, Milgrom (1985) — *Bid, Ask, and Transaction Prices in a
  Specialist Market with Heterogeneously Informed Traders* (classical
  adverse-selection foundation)
