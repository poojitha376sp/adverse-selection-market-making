# Part 4 Results: Validation & Deliverables

QuantFest write-up. Companion to [`research/DERIVATION.md`](DERIVATION.md)
(theory) and [`research/CHEATSHEET.md`](CHEATSHEET.md) (literature). This
document is Phase 5 (Validation) + Phase 6 (Deliverables) of the
[README roadmap](../README.md#execution-roadmap-4-parts) — the final
part of the core project.

**Headline verdict, up front:** the adverse-selection extension does what
the theory predicts — it widens/adjusts spreads in a way that lowers
*total* realized adverse-selection cost, and that improvement genuinely
concentrates in the highest-toxicity regime, as the extension is supposed
to. But it does so almost entirely by trading *less*, not by pricing
informed flow *better* per trade — a nuance Part 3 already reported
honestly and Part 4's validation confirms, sharpens, and in one place
(the high-toxicity regime specifically) makes more pronounced, not less.
On a genuinely held-out slice of data the classifier never trained on,
the picture is murkier still: a small, directionally-consistent
improvement on only 3 fills — not enough data to call it more than
suggestive. Everything below is one ~240-second BTCUSDT capture. Treat
every number here as a case study, not a validated trading edge.

---

## 1. What Part 4 checked that Part 3 didn't

Part 3's three-way backtest (`src/backtest/backtest_adverse.py`) reported
real, honest numbers: total adverse-selection cost fell from baseline
\$17.88 to heuristic \$10.78 to principled \$9.60, but *per-fill* cost
actually **rose** slightly (\$1.99 → \$2.16 → \$2.40) — the improvement
came from trading less/wider, not pricing informed flow better. That
result was reported as-is in the README, not hidden.

What it didn't check: that backtest replayed the **entire** ~240s capture
(`n_steps=2393`), including the chronologically first 70% of the data
that `ml_informedness_classifier.py` trained its GradientBoostingClassifier
on (`n_train=1661` of 2373 labeled rows — see
`data/processed/ml_informedness_metrics.json`). The classifier's own
ROC-AUC/precision/recall were correctly evaluated only on the held-out
30%, but the backtest that produced the headline \$17.88 → \$9.60 numbers
was not restricted to that same held-out slice. That's a real,
checkable contamination risk, not a rhetorical concern — Part 4 checks it
directly instead of assuming it away.

Two new scripts, both in `src/validation/`:

- **`heldout_backtest.py`** — reruns the identical baseline/heuristic/
  principled comparison restricted to *only* the chronological test rows
  the classifier never trained on.
- **`regime_stress_test.py`** — splits the full backtest window into
  high- vs. low-predicted-informedness regimes (using this project's own
  Part 3 classifier's `p_informed(t)` as the regime indicator — see §3
  for why, stated explicitly) and compares the three variants *within*
  the high-toxicity regime specifically, since per the README's Phase 5,
  "that's exactly where the extension is supposed to earn its keep."

---

## 2. Held-out validation

### Method

`heldout_backtest.py` rebuilds the exact feature frame and chronological
70/30 split `ml_informedness_classifier.py` used, and takes the test
split's `local_ts` bounds as the held-out window:

- **Held-out window:** `local_ts ∈ [1784800191.97, 1784800263.39]`,
  **71.4s span**, 712 rows — the classifier trained on the preceding
  1661 rows (~168.5s) and never saw this slice during fitting.
- The three-way backtest is rerun **inside this window only**, with
  inventory and cash reset to zero and the AS horizon clock restarted at
  the window's start (documented design choice in the script's
  docstring: keeping the original clock would make the inventory-
  widening term artificially small throughout, which has nothing to do
  with what's being validated and would confound the comparison).
- `p_informed(t)` inside the window is read from the *already-produced*
  `ml_informedness_scores.csv` — legitimate reuse, because those scores
  come from *scoring* (not re-fitting) a model that was only ever fit on
  the training rows. Scoring held-out rows afterward is ordinary
  out-of-sample inference, not leakage.

### Result

| metric (held-out, 71.4s window) | baseline | heuristic | principled |
|---|---:|---:|---:|
| n fills | 3 | 3 | 3 |
| inventory range | [-1, 1] | [-1, 1] | [-1, 1] |
| inventory variance | 0.2432 | 0.2432 | 0.2432 |
| mean quoted spread | \$4.76 | \$4.87 | \$4.98 |
| realized spread captured (total) | \$6.47 | \$6.78 | \$7.10 |
| realized spread captured (per fill) | \$2.16 | \$2.26 | \$2.37 |
| **adverse-selection cost (total)** | **\$11.81** | **\$11.50** | **\$11.18** |
| **adverse-selection cost (per fill)** | **\$3.94** | **\$3.83** | **\$3.73** |
| net edge per fill (captured − AS cost) | -\$1.78 | -\$1.57 | -\$1.36 |

*(Full numbers: `data/processed/heldout_backtest_comparison.json`, per-run
series/fills in `data/processed/heldout_backtest_{series,fills}_*.{csv,json}`.)*

### Reading it honestly

- **The fill count did not drop this time.** All three variants filled
  on exactly the same 3 trades. The extra widening from heuristic/
  principled quoting (spread mean +2.3% and +4.6% vs. baseline) wasn't
  large enough to avoid any of the crossing trades in this particular
  71s window — the price moves that triggered these fills were bigger
  than the incremental half-spread. This is a different failure/success
  mode than Part 3's full-window result, where fill count *did* drop
  (9→5→4) and that was the main lever.
- **Per-fill AS cost did improve, modestly:** \$3.94 → \$3.83 → \$3.73,
  about a 5% reduction baseline-to-principled. Realized spread captured
  per fill also rose slightly (\$2.16 → \$2.37). Both move in the
  theory-predicted direction — unlike Part 3's full-window finding where
  per-fill cost *rose*. On just 3 fills this is not a statistically
  meaningful result; it is reported as a directional observation, not a
  validated effect.
- **Net edge per fill stays negative throughout** (-\$1.78 to -\$1.36):
  even the principled variant's captured spread does not cover its
  adverse-selection losses on this slice. The extension mitigates but
  does not eliminate the loss here.
- **Why the held-out window behaved differently:** checking the causal
  `p_informed(t)` at the three held-out fill timestamps gives 0.0257,
  0.0065, and 0.0992 — two of three are in the *low*-toxicity regime by
  the §3 threshold (0.0615). This window happened to be a mostly
  low-toxicity period. That connects directly to §3's finding below: the
  extension's fill-avoidance mechanism has little to filter *for* here,
  because there wasn't much toxic flow to avoid in this particular
  slice — most of the high-toxicity activity in the full capture sits in
  the (in-sample) first 70%, not this held-out tail.

---

## 3. Regime stress test

### Method and an explicit deviation from the README's literal wording

The README's Phase 5 asks to stress-test "over the highest-toxicity
periods identified by VPIN in the data." This project's Phase 2 plan
was to reuse the informed-flow proxy from the sibling
`order-flow-imbalance` / `vpin-flow-toxicity` repos; Part 3 already
documented staying self-contained instead (own OFI-style + quote-
imbalance features, no cross-repo runtime dependency, since those are
separate repos with their own environments — not a practical "one repo,
one `requirements.txt`" setup for this submission). Consistent with that
Part 3 decision, `regime_stress_test.py` uses **this project's own Part 3
ML classifier's predicted `p_informed(t)` as the regime indicator**,
*not* a VPIN score. This is stated here as an explicit, acknowledged
deviation from the README's literal text — not a silent substitution.
Arguably it's the more relevant regime definition for the specific
question being asked ("does the extension help exactly where its own
signal says risk is highest"), even though it isn't the VPIN definition.

Second methodological choice, also explicit: this stress test runs on
the **full ~240s backtest window** (the same one Part 3 used), not the
71.4s held-out-only slice from §2. Checked directly: the held-out slice
contains only 2-3 fills per variant total; splitting that further into
high/low-toxicity buckets leaves 0-1 fills per bucket, which cannot
support any conclusion. §2 and §3 are answering different questions —
§2 is "does the classifier's signal generalize out-of-sample," §3 is
"does the existing (partly in-sample) three-way comparison's behavior
differ by regime" — and each is run on the data that question needs.

**Regime definition:** a step (or fill) is HIGH-toxicity regime if its
causal `p_informed(t)` — the *same* signal the heuristic/principled
quotes already condition on — is at or above the 75th percentile of the
full capture's `p_informed` distribution (median 0.019, p75 0.062, p90
0.229, max 0.983 — heavily right-skewed, so "top quartile" is a
meaningfully riskier bucket, not an arbitrary fixed cutoff). Regime
membership is computed once from this exogenous signal and applied
identically to all three variants' fills, so the comparison is
apples-to-apples.

### Result

**High-toxicity regime (top quartile of p_informed, full window):**

| metric | baseline | heuristic | principled |
|---|---:|---:|---:|
| n fills | 6 | 2 | 1 |
| AS cost (total) | \$12.24 | \$5.45 | \$4.07 |
| **AS cost (per fill)** | **\$2.04** | **\$2.72** | **\$4.07** |

**Low-toxicity regime (bottom 3 quartiles):**

| metric | baseline | heuristic | principled |
|---|---:|---:|---:|
| n fills | 3 | 3 | 3 |
| AS cost (total) | \$5.64 | \$5.33 | \$5.53 |
| **AS cost (per fill)** | **\$1.88** | **\$1.78** | **\$1.84** |

*(Full numbers: `data/processed/regime_stress_test.json`.)*

### Is the improvement concentrated in the high-toxicity regime? Yes and no — reported honestly, both halves

**In TOTAL-cost terms: yes, strongly.** The high-toxicity regime's total
AS cost falls \$12.24 → \$5.45 → \$4.07 (a 67% drop, principled vs.
baseline) — a much bigger proportional improvement than the low-toxicity
regime's \$5.64 → \$5.33 → \$5.53 (roughly flat, no clear trend, well
within noise on 3 fills). That total-cost drop is driven entirely by
**avoided fills**: 6 → 2 → 1 fills in the high regime, vs. a constant 3
fills in the low regime. The extension is correctly widening the most
during self-identified high-toxicity moments and getting run over less
often *there* — the mechanism is working exactly where README's Phase 5
says it should.

**In PER-FILL terms: no — if anything, the opposite, on a very thin
sample.** Conditional on still getting filled during a high-toxicity
moment, per-fill cost gets *worse* going baseline → heuristic →
principled: \$2.04 → \$2.72 → \$4.07. This is the same qualitative
pattern Part 3 reported for the whole window (total cost down, per-fill
cost up) — the regime split shows it is not just present but is *more*
pronounced specifically in the regime meant to showcase the extension.
The important caveat: the principled high-toxicity bucket has **n=1**
fill. A single data point moving the per-fill average is not a trend, it
is an anecdote with a dollar sign on it, and it should not be read as
proof the principled variant prices informed flow worse — only as "not
disproven to price it worse, on data this thin."

**Combined reading:** the extension's real, defensible lever in this
capture is *avoiding* trades during predicted-toxic windows, not
*repricing* the trades it still takes during those windows. That is
consistent with — and sharpens — Part 3's own honest framing.

---

## 4. Connecting back to theory

`DERIVATION.md` §6 shows the AS optimal spread is
`γσ²(T−t) + (2/γ)ln(1+γ/κ)` — a function of *constant* volatility and
liquidity parameters, with **no mechanism at all** to react to whether
flow right now looks informed (§7, "the model's blind spot"). §8 makes
the Glosten-Milgrom case for why a *separate*, additive adverse-selection
term is needed: a positive spread survives purely from information
asymmetry, with no risk aversion or market power required.

The qualitative prediction both sections motivate — **wider/adjusted
spreads in high-toxicity regimes should reduce adverse-selection cost**
— is supported by the data, but through a specific, narrower mechanism
than "the classifier lets the maker price toxic flow correctly." What
actually happened, confirmed twice now (Part 3's full-window result and
Part 4's regime-conditioned re-check):

1. Widening the spread as a function of `p_informed` (heuristic) or
   raising the effective `γ` plus adding a Glosten-Milgrom breakeven
   premium (principled, per `avellaneda_stoikov_adverse.py`'s §(b.1)/
   (b.2)) reduces the probability of getting filled at all during
   detected-toxic windows — this is the AS fill-intensity mechanism
   (`λ(δ)=Ae^{−κδ}`, DERIVATION.md §1) doing exactly what it should: a
   wider `δ` mechanically lowers `λ(δ)`, so fewer of the largest, most
   adverse trades cross the quote.
2. That is a real, quantifiable win in total dollar terms (67% lower
   total AS cost in the high-toxicity regime alone), but it is a
   **volume** effect, not a **pricing** effect. Glosten-Milgrom's own
   mechanism — the breakeven premium *sized correctly enough* that
   losses to informed counterparties are offset by gains from noise
   counterparties *on the trades that still happen* — is not clearly
   showing up here: per-fill cost does not fall in either regime in a
   way that would support "the premium is correctly priced," and in the
   high-toxicity regime it visibly rises (on a thin sample).
3. `AdverseParams`'s own documentation (`avellaneda_stoikov_adverse.py`,
   §b.1-b.2 comments) is explicit that the principled variant is an
   *approximation* — a state-dependent effective-γ trick plus an
   additive GM premium, not a re-solved HJB with `p_informed` as a
   genuine coupled state variable. The honest reading of this data is
   that the approximation captures AS's volume-response lever well and
   Glosten-Milgrom's pricing lever poorly, at least on this capture and
   at these parameter settings (`heuristic_beta=1.0`, `principled_eta=1.0`).
   That is a specific, falsifiable claim a future pass could test by
   varying `principled_eta` and the GM premium size independently, or by
   re-deriving `g(q,t)` with `p_informed(t)` as an actual coupled state —
   the "out of scope" item the module docstring already flags.

---

## 5. Classifier limitations, and what they imply for trusting the extension

Part 3's `ml_informedness_metrics.json`: test ROC-AUC **0.7243**,
precision **1.00**, recall **0.0417** (2 of 48 positive test rows caught)
at the default 0.5 threshold, on a 6.7% positive test class (accuracy
93.5% vs. a 93.3% majority-class-baseline — the classifier's raw
thresholded accuracy is barely above "always predict not-informed").

What this means for trusting §2/§3's results:

- **ROC-AUC 0.72 is a real, better-than-random signal**, not noise — the
  *continuous* `p_informed` score (what the quoting extension actually
  consumes, per `avellaneda_stoikov_adverse.py`'s module docstring) does
  rank-order informed vs. uninformed moments meaningfully better than
  chance. This is consistent with §3's finding that the top quartile of
  `p_informed` genuinely does contain a disproportionate share of the
  costliest fills (6 of 9 baseline fills, and both of the two largest
  individual fill costs, land in the top-quartile bucket).
- **The 0.0417 recall at the 0.5 threshold is close to irrelevant here**,
  because neither the heuristic nor principled quoting variant ever
  thresholds `p_informed` at 0.5 — both consume the raw probability
  continuously (multiplying the spread, or scaling γ and adding a GM
  premium proportionally). The classifier doesn't need to correctly
  *classify* a point as "informed" under a hard threshold to be useful
  here; it needs the *continuous score* to correlate with realized
  adverse moves, which the ROC-AUC and the regime-bucket finding above
  both support, weakly but genuinely.
- **What should NOT be inferred:** that this classifier reliably flags
  *most* informed trades. It doesn't — 96% of true positives were missed
  at the default threshold, and even in continuous-score terms, this is
  a single gradient-boosting model trained on 1661 rows of one ~4-minute
  BTCUSDT capture with a ~7% positive class. The Optiver caution the
  README's AI/ML plan already flags — "the classifier is accurate" and
  "the quoting policy that consumes it actually protects PnL" are
  separate claims — is exactly the gap between §4 point 1 (the volume
  mechanism clearly works) and point 2 (the pricing mechanism doesn't
  clearly show up yet). A more accurate classifier, or a genuinely
  re-solved HJB with `p_informed` as a coupled state, might close that
  gap; this validation pass cannot claim it already has.

---

## 6. Sample-size caveat, stated plainly

Every number in this document — Part 3's and Part 4's alike — comes from
**one ~240-second BTCUSDT capture** (`data/raw/depth_btcusdt_20260723_094653.jsonl`
+ the matching trades file), regenerable but not yet regenerated as an
independent second window. Concretely:

- The full-window three-way backtest has 9/5/4 fills across the three
  variants — single-digit sample sizes for every headline number.
- The held-out-only backtest (§2) has exactly 3 fills per variant.
- The high-toxicity regime bucket (§3) has 6/2/1 fills — the principled
  variant's headline high-regime per-fill number is **one trade**.
- `kappa`/`A` (fill-intensity parameters) are themselves only *crudely*
  estimated from this same ~4-minute trade tape (`estimate_params.py`'s
  own docstring: "plausible order of magnitude, not calibrated").

None of the directional findings above — total cost down, per-fill cost
flat-to-up, improvement concentrated in avoided-fills rather than
better-priced fills — should be read as a validated property of this
quoting strategy. They are what happened, honestly measured, on one
short window. The right next step for anyone extending this project
(noted as a natural stretch beyond the 4-part roadmap) is running the
identical `heldout_backtest.py` / `regime_stress_test.py` pipeline
against multiple independently captured windows and checking whether
the same qualitative pattern (volume effect real, pricing effect not
yet demonstrated) replicates — this document reports one clean,
honestly-run case study, not a backtest with statistical power.

---

## 7. Reproducing these results

```bash
# Part 3 prerequisites (unchanged, rerun first if data/processed/ is empty --
# it is gitignored, code-only is versioned):
python src/model/estimate_params.py --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
    --trades data/raw/trades_btcusdt_20260723_094653.jsonl --out-dir data/processed
python src/model/ml_informedness_classifier.py --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
    --trades data/raw/trades_btcusdt_20260723_094653.jsonl --out-dir data/processed
python src/backtest/backtest_adverse.py --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
    --trades data/raw/trades_btcusdt_20260723_094653.jsonl --out-dir data/processed

# Part 4 (this document):
python src/validation/heldout_backtest.py --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
    --trades data/raw/trades_btcusdt_20260723_094653.jsonl --out-dir data/processed
python src/validation/regime_stress_test.py --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
    --trades data/raw/trades_btcusdt_20260723_094653.jsonl --out-dir data/processed
```

## 8. Deliverables checklist (Phase 6)

- [x] Quoting engine supporting baseline and both adverse-selection-aware
  modes, run against the same backtest harness —
  `src/model/avellaneda_stoikov.py`, `src/model/avellaneda_stoikov_adverse.py`,
  `src/backtest/backtest_adverse.py` (Part 3), reused unchanged by
  `src/validation/heldout_backtest.py` and `regime_stress_test.py` (Part 4).
- [x] Comparison report — PnL, inventory variance, adverse-selection cost
  (total and per-fill), realized spread captured, fill rate, with a
  breakdown by toxicity regime — this document, §2-3.
- [x] Write-up connecting results back to theory, with an honest
  quantified answer ("by how much") — §4.
