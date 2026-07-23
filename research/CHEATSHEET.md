# Adverse-Selection Market Making Reference Cheatsheet

Research compiled 2026-07-23. Covers: (1) academic literature, (2) practitioner
explainers, (3) conferences/venues, (4) what market-making/HFT firms disclose
publicly about inventory-risk and adverse-selection-aware quoting. Mirrors the
structure of the companion "Order Flow Imbalance" cheatsheet in the sister
`order-flow-imbalance` project.

---

## 1. Academic papers (core literature)

### Foundational paper

- **Avellaneda, M., & Stoikov, S. (2008). "High-Frequency Trading in a Limit Order Book."** *Quantitative Finance*, 8(3), 217–224. DOI: 10.1080/14697680701381228
  Author copy (Stoikov, Cornell): https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf · RePEc listing: https://ideas.repec.org/a/taf/quantf/v8y2008i3p217-224.html · NYU Scholars record: https://nyuscholars.nyu.edu/en/publications/high-frequency-trading-in-a-limit-order-book · Semantic Scholar: https://www.semanticscholar.org/paper/High-frequency-trading-in-a-limit-order-book-Avellaneda-Stoikov/98e6e6d759ac60bff783c76a8bcc60bb4282b311

  - **Setup:** A single risk-averse dealer with exponential utility (CARA, risk-aversion coefficient γ) quotes bid/ask around a mid-price `s` that follows arithmetic Brownian motion with volatility σ, over a finite horizon `T`. Buy/sell market orders arrive as a Poisson process whose intensity decays exponentially in distance `δ` from the mid-price: `λ(δ) = A·e^(−κδ)`, where `A` is the base arrival rate/intensity scale and `κ` governs how fast fill probability decays with quoted distance (a proxy for local order-book liquidity/depth).
  - **Reservation (indifference) price**, from solving the resulting HJB equation: `r(s, q, t) = s − q·γ·σ²·(T − t)`, where `q` is current inventory (positive = long, negative = short). This is the mid-price adjusted by a term that skews quotes to push inventory back toward zero — more skew for larger |q|, higher risk aversion γ, higher volatility σ, or more time remaining.
  - **Optimal (total) spread** around the reservation price: `δᵃ + δᵇ = γ·σ²·(T − t) + (2/γ)·ln(1 + γ/κ)`. The spread widens with volatility (quadratically), risk aversion, and time-to-horizon, and narrows as `κ` grows (denser/more liquid order book). The individual optimal bid/ask offsets are placed symmetrically around `r(s,q,t)`, not around the raw mid-price `s`.
  - **Why it matters for implementation:** This is the base model an "adverse-selection extension" project sits on top of — it already prices *inventory* risk explicitly (via the `qγσ²(T−t)` skew) but has **no explicit adverse-selection/informed-flow term**: `σ`, `κ`, `A` are treated as constants, and there's no mechanism for the dealer to react to a signal that incoming flow is "toxic." Extending it typically means (a) making `σ` or the mid-price process itself jump/react to detected informed flow, (b) adding an adverse-selection cost term directly into the spread formula, or (c) making `κ`/`A` (fill-probability parameters) conditional on a toxicity signal.

### Primary extension (tractable closed-form solution)

- **Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2013). "Dealing with the Inventory Risk: A Solution to the Market Making Problem."** *Mathematics and Financial Economics*, 7(4), 477–507. DOI: 10.1007/s11579-012-0087-0
  Published version: https://link.springer.com/article/10.1007/s11579-012-0087-0 · arXiv preprint (2011): https://arxiv.org/abs/1105.3115 (PDF: https://arxiv.org/pdf/1105.3115) · HAL: https://hal.science/hal-01393110v1 · EconPapers: https://econpapers.repec.org/RePEc:hal:journl:hal-01393110

  - **What they solved that Avellaneda-Stoikov didn't:** AS's HJB equation is a nonlinear PDE that (in the original paper) was solved via an asymptotic approximation for `κ,A` fixed and only sketched numerically; there was no closed form and no treatment of hard inventory limits. Guéant-Lehalle-Fernandez-Tapia (GLFT) show that, under the same exponential-CARA-utility/Poisson-arrival setup (building on Ho & Stoll's original inventory-risk framework and AS's formalization), a change of variables **transforms the nonlinear HJB system into a system of linear ordinary differential equations** — reducing the problem to something numerically trivial (matrix exponentials) and enabling a genuine **closed-form asymptotic approximation of the optimal quotes** via a spectral (eigenvalue) characterization.
  - They also **explicitly solve the problem under inventory constraints** (a hard bound `q ∈ [−Q, Q]` on the market maker's position), which AS's original treatment did not address, and they characterize the **asymptotic behavior of optimal quotes** as the horizon grows.
  - **Why it matters for implementation:** the GLFT closed-form/asymptotic quotes are the standard "production-grade" version of Avellaneda-Stoikov cited in most practitioner writeups and open-source implementations — if a project wants quotes that are fast to compute and respect inventory limits, this is the paper to implement rather than re-deriving AS's raw HJB numerically.

### Adverse-selection-specific extensions

- **Glosten-Milgrom mechanism, formalized in continuous/algorithmic-trading terms — Guilbaud, F., & Pham, H. (2013). "Optimal High-Frequency Trading with Limit and Market Orders."** *Quantitative Finance*, 13(1), 79–94.
  arXiv: https://arxiv.org/abs/1106.5040 (PDF: https://arxiv.org/pdf/1106.5040) · SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1871969 · Publisher: https://www.tandfonline.com/doi/abs/10.1080/14697688.2012.708779
  - Models the bid-ask spread itself as a finite-state Markov chain (multiples of tick size) and has an agent choosing between posting limit orders and firing market orders to maximize expected utility, trading off **three explicitly named and distinct sources of risk: inventory risk, execution risk, and adverse selection risk** (the risk that the price moves unfavorably right after a limit order is filled). Solved as a mixed regular/impulse stochastic control problem via a quasi-variational-inequality system.
  - Why it matters: one of the first papers to formally decompose HFT market-making risk into these three named buckets within an AS-style stochastic control framework — a natural template for where to "plug in" an adverse-selection cost term next to AS's inventory term.

- **Cartea, Á., & Jaimungal, S. (2015). "Risk Metrics and Fine Tuning of High-Frequency Trading Strategies."** *Mathematical Finance*, 25(3), 576–611. DOI: 10.1111/mafi.12023
  SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2010417 · Publisher: https://onlinelibrary.wiley.com/doi/10.1111/mafi.12023
  - Proposes explicit risk metrics (not just terminal expected-utility maximization) for HFT strategies with short holding periods, letting the trader fine-tune the trade-off between inventory risk and expected profit. The optimal strategy structurally includes a **buffer to cover adverse-selection costs** and adapts quoting to short-term price "momentum" driven by order-flow information.
  - Why it matters: gives a concrete recipe (risk-metric penalization) for building adverse-selection buffers into an AS-style strategy without a full informed-trader game-theoretic model.

- **Cartea, Á., Jaimungal, S., & Ricci, J. (2018). "Algorithmic Trading, Stochastic Control, and Mutually-Exciting Processes."** *SIAM Review*, 60(3), 673–703 (won SIAM's 2018 SIGEST Prize). DOI: 10.1137/18M1176968
  SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3306158 · Author PDF (Oxford-Man Institute): https://oxford-man.ox.ac.uk/wp-content/uploads/2020/05/Algorithmic-Trading-Stochastic-Control-and-Mutually-Exciting-Processes.pdf · ORA: https://ora.ox.ac.uk/objects/uuid:a43beae7-ed40-4848-b092-6aa1aef4ae96
  - Introduces a **multivariate mutually-exciting (Hawkes) process** for buy/sell market order arrivals and limit-order-book shape, so that a burst of one-sided flow makes further same-side flow *more* likely (self- and cross-excitation) — a direct, tractable way of modeling "toxic"/clustered informed flow arriving in bursts, in the same stochastic-control (HJB) tradition as Avellaneda-Stoikov.
  - Why it matters: this is the most concrete, well-known Cartea-Jaimungal-lineage mechanism for making order-flow intensity itself react to recent flow (a Hawkes-style stand-in for "flow looks informed right now, so widen"), fully worked out with HJB-style optimal control.

- **Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*.** Cambridge University Press. ISBN 978-1107091146.
  Publisher/Cambridge frontmatter (table of contents): https://assets.cambridge.org/97811070/91146/frontmatter/9781107091146_frontmatter.pdf · Google Books: https://books.google.com/books/about/Algorithmic_and_High_Frequency_Trading.html?id=5dMmCgAAQBAJ
  - Confirmed table of contents: **Chapter 10 is titled "Market Making"** (Part III, "Algorithmic and High-Frequency Trading"), preceded by Ch. 5 "Stochastic optimal control and stopping" and followed by Ch. 12 "Order imbalance." This is the textbook synthesis of the Cartea-Jaimungal research programme (including adverse-selection/informed-flow considerations) into a single teachable framework; individual chapters draw on the papers above and on the authors' other SSRN/journal papers.
  - Why it matters: the standard textbook reference bridging Avellaneda-Stoikov-style stochastic control with adverse-selection-aware order-flow modeling; useful as a single citable source for the whole extended framework rather than citing many individual papers.

- **Lehalle, C.-A., & Mounjid, O. (2017). "Limit Order Strategic Placement with Adverse Selection Risk and the Role of Latency."** *Market Microstructure and Liquidity*, 3(1), 1750009.
  arXiv: https://arxiv.org/abs/1610.00261 (PDF: https://arxiv.org/pdf/1610.00261)
  - Uses labeled trade data to show market participants condition limit-order placement on observed liquidity imbalance, then builds a stochastic-control framework where an agent monitors and repositions limit orders to reduce adverse selection. Key finding: the value of exploiting a liquidity-imbalance signal to avoid adverse selection **erodes with latency** — if you can't cancel/reinsert fast enough after detecting informed flow, the signal is of little use, giving a formal rationale for "speed as an adverse-selection defense."
  - Why it matters: directly quantifies the adverse-selection/latency trade-off referenced qualitatively by many practitioner sources (see Section 4); a good source for "why fast repricing matters" beyond AS's static parameters.

- **Cartea, Á., & Sánchez-Betancourt, L. (2025). "Brokers and Informed Traders: Dealing with Toxic Flow and Extracting Trading Signals."** *SIAM Journal on Financial Mathematics*, 16(2), 243–270.
  SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4265814 · ORA (Oxford): https://ora.ox.ac.uk/objects/uuid:8238ef01-247a-4cf0-9c43-8b67da33367e · Oxford-Man Institute announcement: https://oxford-man.ox.ac.uk/alvaro-cartea-and-leandro-sanchez-betancourt-release-their-new-publication-brokers-and-informed-traders-dealing-with-toxic-flow-and-extracting-trading-signals/
  - Derives closed-form strategies for a broker/liquidity provider who trades with both an informed trader (who has a trend signal and, on average, trades at the broker's expense) and a noise trader (uninformative flow, on average profitable for the broker). The broker's losses to the informed trader are treated as the "price paid" to extract the informed trader's trend signal, which then feeds into how much flow to internalize vs. externalize (unwind in the lit market) vs. use to speculate.
  - Why it matters: a modern (2025), directly on-topic "adverse selection as information extraction" framework — closed-form, finite- and infinite-horizon — that's a natural citation for "quoting under toxic flow" beyond the original static Glosten-Milgrom setup.

- **Cartea, Á., Duran-Martin, G., & Sánchez-Betancourt, L. "Detecting Toxic Flow."** *Quantitative Finance*, 26(4), 541–561 (2026 publication; preprint 2023).
  arXiv: https://arxiv.org/abs/2312.05827 (PDF: https://arxiv.org/pdf/2312.05827) · Publisher: https://www.tandfonline.com/doi/full/10.1080/14697688.2026.2619539
  - Proposes an online Bayesian learning method ("PULSE") to predict, trade-by-trade, whether an incoming trade is "toxic" (adversely selecting), tested on a proprietary FX transactions dataset; shows it outperforms standard ML/statistical baselines for this classification task.
  - Why it matters: a concrete, published example of the "detect-then-widen" pipeline this project is presumably implementing — classify flow toxicity, then feed that into quoting logic.

### Classical foundation

- **Glosten, L. R., & Milgrom, P. R. (1985). "Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders."** *Journal of Financial Economics*, 14(1), 71–100.
  ScienceDirect: https://www.sciencedirect.com/science/article/pii/0304405X85900443 · RePEc/IDEAS: https://ideas.repec.org/a/eee/jfinec/v14y1985i1p71-100.html · PDF mirrors: https://www.edegan.com/pdfs/Glosten%20Milgrom%20(1985)%20-%20Bid%20Ask%20and%20Transaction%20Prices%20in%20a%20Specialist%20Market%20with%20Heterogeneously%20Informed%20Trades.pdf and https://www.kellogg.northwestern.edu/research/math/papers/570.pdf
  Secondary walkthroughs used to cross-check the mechanism (not primary sources, but useful accessible explainers): Faustian Dreams blog, "Glosten–Milgrom: Spread & Bayesian Updates" — https://faustiandreams.github.io/2022-08-23/glosten-milgrom-model ; Sanmay Das, "A Learning Market-Maker in the Glosten-Milgrom Model" — https://cs.gmu.edu/~sanmay/papers/das-qf-rev3.pdf

  - **Mechanism:** a sequential-trade model with a risk-neutral, zero-expected-profit specialist/market maker facing a stream of single-unit orders, each of which comes from an **informed trader** (who knows the asset's true terminal value) with some probability, or from a **noise/liquidity trader** (uninformative) otherwise. Because the market maker cannot tell which type placed any given order, they set the **ask above and the bid below** their current expectation of value, sized so that the *expected* loss to informed traders on that side is exactly offset by the *expected* gain from noise traders — a zero-expected-profit break-even condition per quote.
  - After each trade, the market maker performs a **Bayesian update**: a trade at the ask raises the posterior probability the true value is high (and vice versa for a trade at the bid), so quotes ratchet toward the revealed information over the sequence of trades; transaction prices form a martingale that converges toward the true value as more informed trades occur.
  - **Headline result:** a **positive bid-ask spread arises purely from information asymmetry**, even though the market maker is risk-neutral and breaks even on expectation — spread is not compensation for inventory or order-processing cost here, it is purely an adverse-selection premium.
  - **Why it matters for implementation:** this is the conceptual origin of "widen the spread when flow looks informed" — the qualitative behavior every later stochastic-control extension (Guilbaud-Pham, Cartea-Jaimungal, Cartea-Sánchez-Betancourt) is trying to reproduce in a continuous-time, inventory-aware setting. It's the natural "why does adverse selection cost money at all" section of any implementation writeup, even though it's a discrete sequential-trade model rather than the continuous stochastic-control style of Avellaneda-Stoikov.

---

## 2. Practitioner articles / blog posts / talks

- **Hummingbot — "Guide to the Avellaneda & Stoikov Strategy"** and **"Technical Deep Dive into the Avellaneda & Stoikov Strategy"**
  https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/ · https://hummingbot.org/blog/technical-deep-dive-into-the-avellaneda--stoikov-strategy/
  Documentation for Hummingbot's open-source market-making bot, which ships an actual Avellaneda-Stoikov strategy implementation. The technical deep-dive page states the reservation-price and optimal-spread formulas explicitly (matching the formulas in Section 1 above) and walks through parameter units and practical calibration (including a scaled `inventory_risk_aversion` knob and an `order_amount_shape_factor` for practical order sizing) — a good "from paper to running bot" reference.

- **jshellen/HFT (GitHub) — "High Frequency Market Making" algorithm collection**
  https://github.com/jshellen/HFT
  Open-source collection of stochastic-optimal-control market-making algorithms solved via finite-difference numerical schemes, explicitly built as Avellaneda-Stoikov variants: **"AS++"** (AS with terminal + running inventory penalties), **"ASAS"** (explicitly adds an adverse-selection factor into the optimal-distance calculation), **"AS+++"** (adds hedging via quasi-variational-inequality conditions and rebate considerations), and **"ASMP"** (integrates Stoikov's micro-price model). The "ASAS" variant is a directly on-topic, publicly available reference implementation for "Avellaneda-Stoikov + adverse selection."

- **Quantt — "Market Making Strategy Guide"**
  https://www.quantt.co.uk/resources/market-making-strategy-guide
  Independent educational site (explicitly states it is not affiliated with the firms it names, used "for identification only"). Frames Avellaneda-Stoikov as "the foundational academic framework," and has an explicit adverse-selection section: spreads must compensate for adverse selection "even in a perfectly competitive market," with detection based on order-pattern/sizing analysis and response via widening spreads for flow that "looks informed" and tightening for flow that looks benign. Useful as an accessible, non-firm-affiliated explainer that keeps the AS-formula and adverse-selection concepts side by side, though it is not a primary/authoritative source and its firm-specific claims (see Section 4) should not be taken as confirmed insider information.

- **Faycal Drissi — "Lecture Notes on Market Microstructure and Algorithmic Trading" (Oxford, 2024)**
  https://www.faycaldrissi.com/files/HFT_2024___Oxford___lecture_notes_2024.pdf
  Publicly posted PDF lecture notes described (per its filename/search indexing) as Oxford course material in the Cartea-Jaimungal tradition of stochastic-control-based algorithmic/HFT modeling. Content could not be machine-extracted in this research pass (binary/PDF-object text did not decode cleanly) — flagged as **existence-confirmed, content not independently verified**; worth opening directly if citing specific formulas from it.

- **GitHub topic pages and reference implementations** (breadth, not all inspected in depth):
  https://github.com/topics/avellaneda-stoikov · https://github.com/topics/market-making
  Numerous open-source Avellaneda-Stoikov implementations exist beyond jshellen/HFT, e.g. `AymenCode/Avellaneda-Stoikov-Market-Making` (simulations + risk analysis + model extensions), `im1235/ISAC` (reinforcement-learning control of the AS risk-aversion parameter), `fedecaccia/avellaneda-stoikov`, `Jungle-Sven/avellaneda_stoikov_mm`. Quality/rigor varies by repo and was not individually verified beyond README descriptions; listed as a discovery starting point, not vetted reference implementations.

- **Dean Markwick (dm13450.github.io)** — checked specifically per the brief (same blogger who covered Order Flow Imbalance for the sister project). **No Avellaneda-Stoikov- or adverse-selection-specific post was found** on this blog via search or direct site fetch; only the OFI post (already cited in the sister cheatsheet) was located. Flagged as **searched, not found** rather than omitted silently.

- **QuantStart** — checked specifically per the brief. No QuantStart article specifically on Avellaneda-Stoikov or adverse-selection market making was found via search (their HFT series, cited in the sister OFI cheatsheet, covers market microstructure, LOB mechanics, and optimal execution generally, but nothing AS-specific surfaced). Flagged as **searched, not found**.

- **QuantInsti (Quantra)** — no free article specific to Avellaneda-Stoikov/adverse-selection market making was found; their general "Market Microstructure for High-Frequency Trading" paid course module (https://www.quantinsti.com/epat/market-microstructure, also referenced in the sister OFI cheatsheet) is the closest match, listed as a known structured-course resource rather than a free deep-dive.

---

## 3. Conferences / venues

**Academic journals:**
- *Quantitative Finance* (Taylor & Francis) — original venue for Avellaneda & Stoikov (2008), Guilbaud & Pham (2013), and Cartea/Duran-Martin/Sánchez-Betancourt's "Detecting Toxic Flow" (2026); the leading applied venue for this literature today.
- *Mathematics and Financial Economics* (Springer) — venue for Guéant, Lehalle & Fernandez-Tapia (2013).
- *Mathematical Finance* (Wiley) — venue for Cartea & Jaimungal's "Risk Metrics and Fine Tuning of High-Frequency Trading Strategies" (2015).
- *SIAM Journal on Financial Mathematics* — venue for Cartea & Sánchez-Betancourt's "Brokers and Informed Traders" (2025) and other Cartea-Jaimungal-lineage stochastic-control papers (e.g., "Algorithmic Trading with Model Uncertainty," Cartea/Donnelly/Jaimungal).
- *SIAM Review* — venue for Cartea, Jaimungal & Ricci's mutually-exciting-processes paper (2018), which won SIAM's SIGEST Prize (a distinction SIAM Review gives to significant papers from its affiliated journals, here from SIAM J. Financial Mathematics).
- *Journal of Financial Economics* — venue for Glosten & Milgrom (1985), the classical microstructure/adverse-selection home journal.
- *Market Microstructure and Liquidity* (World Scientific) — microstructure-specific journal, venue for Lehalle & Mounjid (2017) on adverse selection and latency. As noted in the sister OFI cheatsheet, this journal **ceased publication after Vol. 6 (2020)** — historically important, no longer active. https://www.worldscientific.com/worldscinet/mml

**Academic/mixed conferences (same broad venues identified for the OFI cheatsheet — market microstructure research spans both topics):**
- **"Market Microstructure: Confronting Many Viewpoints"** — biannual Paris conference (Institut Louis Bachelier / Fondation Banque de France); Charles-Albert Lehalle (co-author of the GLFT and Lehalle-Mounjid papers above) is one of the recurring organizers/editors in this series. https://www.wiley.com/en-us/Market+Microstructure:+Confronting+Many+Viewpoints-p-9781119952787
- **Conference on Market Microstructure, Quantitative Trading, High Frequency, and Large Data** — University of Chicago Stevanovich Center, annual academic conference. https://stevanovichcenter.uchicago.edu/conferences/
- **Oxford-Man Institute of Quantitative Finance** — not a conference per se, but the institutional home (via Álvaro Cartea's affiliation) that hosts/publishes several of the adverse-selection/toxic-flow papers above and runs its own seminar series; https://oxford-man.ox.ac.uk/

**Industry/practitioner conferences (reused from the OFI cheatsheet's findings — no market-making/adverse-selection-specific agenda content was independently verified beyond what's already noted there):**
- **Battle of the Quants** — https://battleofthequants.com/ — general systematic-trading industry conference; no adverse-selection-market-making-specific agenda item was verified in this pass.
- **Q-Group** and **Trading Show** — as in the sister cheatsheet, these are known industry venues but **no specific adverse-selection-market-making session was confirmed** via search; not claimed as topical matches.

**Note on distinction (same pattern as the OFI cheatsheet):** the Paris and Chicago conferences and the core journals above are primarily **academic**; Battle of the Quants and Trading Show are primarily **industry/practitioner** and their specific agendas on this exact sub-topic were not verified.

---

## 4. Firms — what's publicly known about adverse-selection / inventory-risk approach

Caveat up front, same as the sister cheatsheet: **no firm publishes its actual production quoting logic, risk-aversion parameters, or toxicity-detection model.** What follows is what's genuinely publicly confirmable — official blog/career content or explicitly-disclosed talks — versus third-party commentary, which is flagged as such rather than presented as firm-confirmed fact.

- **Optiver** — Has an official Technology Blog (https://www.optiver.com/insights/technology-blog/) and a careers-hub article, **"Engineering the three pillars of trading: Pricing, risk, and execution"** (https://optiver.com/working-at-optiver/career-hub/engineering-the-three-pillars-of-trading-pricing-risk-and-execution/), which states that Optiver determines bids/offers by "calculating a theoretical price for the instrument and the margin needed to compensate for the risk involved," uses "cutting-edge risk models" translated into trader-facing limits, and emphasizes that prices and risk change rapidly, requiring continuous re-evaluation. **This article does not explicitly name "adverse selection" or "inventory risk" as distinct terms** — the connection to those concepts is inferential (speed + continuous repricing + risk-based margin), not a direct technical disclosure. A third-party Substack post titled "Optiver's €3.5B Market-Making Engine: Avellaneda-Stoikov Inventory Optimization at Scale" (https://navnoorbawa.substack.com/p/optivers-35b-market-making-engine) exists but is **independent commentary/speculation, not an Optiver publication** — flagged explicitly as unverified and not attributable to Optiver.

- **Hudson River Trading (HRT)** — Official tech blog "The HRT Beat" (https://www.hudsonrivertrading.com/hrtbeat/), same as noted in the sister OFI cheatsheet, with general posts on modeling philosophy and ML-in-trading pitfalls. **No HRT-authored post specifically on adverse selection or inventory-risk quoting mechanics was found.** A specific "profit formula" — `(Bid-Ask Spread × Volume × Fill Rate) − (Adverse Selection Cost + Inventory Risk)` — surfaced in search results attributed to third-party commentary (a Substack/blog analysis of HRT), **not to any HRT-published source**; flagged as unverified third-party framing, not an HRT disclosure.

- **Jane Street** — Public "Tech Talks" series (https://www.janestreet.com/tech-talks/index.html) including a talk on building their internal exchange/crossing engine ("JX"), which covers limit-order-book/matching-engine design but is infrastructure-focused, not signal- or risk-model-focused. **No public Jane Street content specifically on adverse selection or inventory-risk quoting logic was found.**

- **XTX Markets** — As in the sister OFI cheatsheet: publicly describes itself as a fully model-driven market maker generating ML-based price forecasts with no human traders on the desk, and contributed a paper to a Bank of Canada publication, "Use of Artificial Intelligence in Market Making" (https://www.banqueducanada.ca/wp-content/uploads/2026/04/Use-of-Artificial-Intelligence-in-Market-Making-XTX-Markets-020326.pdf) — the most likely candidate among these firms for actual adverse-selection-adjacent policy-level discussion, but its content could not be fully machine-extracted in this research pass; **worth reading directly** if citing.

- **Citadel Securities** — No technical research blog or public content on adverse-selection/inventory-risk methodology found; public content remains limited to generic careers-page descriptions of quantitative researcher/trader roles (as also found for the OFI cheatsheet).

- **Jump Trading, DRW, Tower Research Capital, IMC, Virtu Financial** — Consistent with the sister OFI cheatsheet's finding: **no technical blogs, papers, or public talks specifically addressing adverse selection or inventory-risk quoting methodology were found for any of these five firms.** Their public presence remains recruiting-oriented (careers pages, generic "what we do" descriptions) rather than research-disclosure-oriented. Honest assessment: **nothing firm-specific and adverse-selection-specific exists in the public domain for this group**, beyond what independent third-party commentary (Substack posts, "quant firm guide" sites) speculates — and that speculation should not be cited as confirmed firm practice.

---

## Summary of sourcing confidence

- **High confidence / directly verified:** Avellaneda & Stoikov (2008) reservation-price and optimal-spread formulas (cross-confirmed across the Hummingbot technical deep-dive and standard secondary presentations; the underlying paper's DOI/venue independently confirmed via RePEc/NYU Scholars); Guéant-Lehalle-Fernandez-Tapia (2013) publication details and core contribution (linear-ODE transformation, closed-form/asymptotic quotes, inventory constraints — confirmed via arXiv abstract and Springer/HAL listings); Glosten & Milgrom (1985) citation and mechanism (cross-confirmed via ScienceDirect, RePEc, and two independent secondary explainers); Guilbaud & Pham (2013), Cartea & Jaimungal (2015) "Risk Metrics," Cartea/Jaimungal/Ricci (2018) SIAM Review paper, Lehalle & Mounjid (2017), Cartea & Sánchez-Betancourt (2025), and Cartea/Duran-Martin/Sánchez-Betancourt "Detecting Toxic Flow" — all confirmed via multiple independent sources (SSRN, arXiv, publisher pages, or institutional repositories) for title, venue, and core claims; Cartea/Jaimungal/Penalva textbook's Chapter 10 "Market Making" placement (confirmed via the publisher's own frontmatter/TOC PDF); existence and official nature of Hummingbot's AS documentation, jshellen/HFT's "ASAS" adverse-selection variant, and Optiver's/HRT's/Jane Street's official blog/tech-talk pages.

- **Medium confidence (verify before final citation):** Faycal Drissi's Oxford lecture notes PDF (existence and filename/context confirmed, but content could not be machine-extracted — recommend opening directly before citing specific claims from it); XTX Markets' Bank of Canada paper content (existence confirmed via a URL already surfaced for the sister OFI cheatsheet, content not independently re-extracted here); the general quality/correctness of unvetted third-party GitHub Avellaneda-Stoikov implementations beyond jshellen/HFT (listed as discovery starting points, not code-reviewed); Quantt's guide content (independent educational source, self-declared non-affiliated, but not a primary/peer-reviewed source).

- **Flagged as unconfirmed / not found / explicitly not to be cited as fact:** Dean Markwick's blog (dm13450.github.io) has **no** Avellaneda-Stoikov or adverse-selection post — searched specifically per the brief and not found; same for a QuantStart article specifically on this topic; the third-party Substack claims about Optiver's and HRT's specific quoting formulas/architecture (navnoorbawa.substack.com posts) are **independent commentary, not firm-published sources**, and should not be attributed to Optiver or HRT as confirmed disclosure; no public technical content on adverse-selection or inventory-risk methodology was found for Citadel Securities, Jump Trading, DRW, Tower Research Capital, IMC, or Virtu Financial — absence of evidence is reported as such, not filled in with speculation.
