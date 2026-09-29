# P2-ETF-DELAY-AGENT-DYNAMICS

Delayed heterogeneous-agent feedback, used to detect proximity to a Hopf
bifurcation — not to predict price directly.

## The idea

Two-type agent models of price formation (Beja & Goldman 1980; Chiarella
1992; Dieci & He 2018 survey) split demand into **fundamentalists**, who pull
price back toward a reference value, and **chartists**, who extrapolate the
recent trend on a **delay** — they need `tau` days of price history to read
"the trend". The resulting dynamics is a delay differential equation, and
delay is exactly the ingredient that can turn a stable market into one with
self-sustained oscillations: as chartist strength or the delay grows, the
system can lose stability through a **Hopf bifurcation** — a pair of complex
characteristic roots crossing the imaginary axis, producing a *growing
oscillation* rather than a monotone runaway.

Let `x(t) = log_price(t) − trend(t)` (trend = EMA of log price; a price-only
"fundamental" proxy — the master dataset has no real fundamentals). The
linearized model is

```
dx/dt = -f·x(t) + (c/τ)·(x(t) - x(t-τ))  =  A·x(t) + B·x(t-τ)
```

`f` = fundamentalist strength, `c` = chartist strength, `τ` = the delay. For
this scalar DDE, **every** characteristic root has the closed form

```
λ_k = A + W_k(B·τ·e^(-A·τ)) / τ        (Lambert W, branch k)
```

and the **principal branch W₀ always gives the dominant (rightmost) root**,
whether it comes out real or as a complex conjugate pair (Asl & Ulsoy,
*"Analysis of a System of Linear Delay Differential Equations"*, ASME J.
Dynamic Systems, 2003). So:

```
λ_dom            = A + W₀(B·τ·e^(-A·τ)) / τ
stability_margin = -Re(λ_dom)     (>0 stable; shrinking = approaching Hopf)
hopf_frequency    = Im(λ_dom)     (nonzero ⇒ a growing OSCILLATION, not a
                                    monotone move, if/when the boundary
                                    is reached — the Hopf signature)
```

`(A, B, τ)` are re-estimated **every trading day** by rolling OLS on the
trailing `FIT_WINDOW` days, independently for each `τ` in `TAU_GRID`, keeping
whichever `τ` explains that ETF's recent dynamics best (EMA-smoothed R²,
so the winning `τ` doesn't flip on noise-level day-to-day differences).

## What it does and does not claim

* **Local, linear, backward-looking.** Not a claim that any ETF obeys one
  fixed DDE — `stability_margin` is a diagnostic from a fit re-run daily, not
  a law of motion. Near the boundary the estimate is inherently noisy (see
  Validation status), which is expected for a small-window fit of a
  slowly-drifting quantity, not a defect unique to this implementation.
* **A Hopf crossing predicts an oscillation, not a direction** — it could
  presage a breakout rally or a crash. The regime-lead diagnostic tests both.
* **The forecast is a small linear model** on the margin and friends, always
  compared against a baseline with **no delay-model features at all**, so the
  delay machinery is credited only for what it adds.

## How the result is kept honest

Same checks as the sibling engines, applied from the start:

1. **Baseline ablation.** Every configuration runs twice on the same dates:
   the delay model (`x, mom5, mom21, vol21` + `margin, margin_chg, omega,
   A_hat, B_hat, tau_norm`), and a baseline using only the first four. Credit
   is the *difference*.
2. **Cross-sectional target.** Features and forward returns are z-scored
   across ETFs each day, so drift/beta cancel.
3. **The product is what is tested**: mean forward return of the top-3 minus
   the equal-weight universe, after cost, plus cross-sectional rank-IC.
4. **Non-overlapping test points** spaced one horizon apart; Sharpe
   annualized by `sqrt(252/horizon)`.
5. **Selection/holdout split.** First 70% of test periods choose among trend
   spans (63/126/252d) × horizons (1/5/20d) × windows (252–1008d). Last 30%
   is a holdout that drives the headline numbers and confidence.
6. **Confidence** (`trainer.confidence_label`): *High* needs holdout IC
   t ≥ 1.65, positive net spread, **and** beating the baseline on both IC and
   spread; *Medium* needs positive IC and spread and beating the baseline on
   both; otherwise *Low* (also *Low* below 30 holdout periods).

## Dashboard readouts (Tab 1)

* **Phase map**: every ETF plotted at (stability margin, fitted delay τ);
  left of zero (shaded) = past the local instability boundary.
* **Systemic fragility**: share of ETFs currently past their own boundary.
* **Oscillatory candidates**: ETFs furthest past the boundary with a nonzero
  Hopf frequency, and the cycle length that implies (`2π/ω`).

## The "pre-crash / pre-breakout" claim is tested, not assumed

`delay_model.regime_lead_diagnostic` compares, per ETF: an **instability
onset** (margin crosses below 0 — the crossing moment, not the whole
below-threshold stretch) against a **large-move event** (`|`10-day forward
return`|` exceeds `2×` 21-day realized vol, scaled), both de-duplicated to
one per 21 quiet days. It reports how often an onset sat in the 20 days
*before* an event against a **per-ETF** chance baseline (computed from that
ETF's own onset/event counts and its own covered history, then pooled the
same way the real hits are — pooling a single chance rate across ETFs would
let an onset on one ETF "explain" an event on another, which cannot happen,
and would overstate the chance baseline).

## Selection sample-size floor (added after the first live run)

The first real run picked, in every universe, the longest trend span, longest
horizon, and largest window — precisely the grid cells with the FEWEST
non-overlapping test periods (a 20-day horizon leaves ~20x fewer periods than
a 1-day horizon over the same history). One universe's winner showed a
striking selection-segment edge (t=+2.06) that collapsed on its own holdout
(t=+0.39, net spread flipped sign) — while a passed-over config with ~9x more
selection periods had a smaller but far more consistent edge that held up on
a holdout roughly 4x larger. A plain t-stat already divides by sqrt(n), but
that doesn't stop a thin-sample cell's REALIZED t-stat from being a much
noisier random variable than a large-sample cell's — so ranking many configs
by raw t-stat alone systematically favors whichever noisy cell got lucky.

`trainer.select_config` now also requires a configuration's selection-segment
period count to clear a floor relative to the largest period count in that
pool (`config.MIN_SELECTION_N_FRACTION`, default 0.15), in addition to an
absolute floor (`MIN_SELECTION_N`, default 50). Re-run against the real grid
from that first live run, this shifted the flagged winner to the
large-sample, holdout-consistent configuration and — tellingly — also
revealed that the universe which had looked most promising no longer clears
the eligibility gate at all once its noisiest cell is excluded, which is the
more honest read of that universe's evidence.

## Repo structure

```
config.py         universes, TREND_SPANS, TAU_GRID, HORIZONS, WINDOWS, LEAD_PARAMS
data_manager.py   master parquet from HF (same as sibling repos)
delay_model.py    rolling delay-feedback fit, Lambert-W dominant root, features,
                  O(1) window ridge fits, walk-forward evaluation, regime-lead test
trainer.py        per-universe orchestration, selection/holdout, live picks, JSON
push_results.py   HfApi.upload_file to the results dataset
us_calendar.py    trading-day helpers
streamlit_app.py  Tab 1 live picks + phase map; Tab 2 backtest grid, curves, lead test
.github/workflows/daily_run.yml   weekday cron after the US close
```

## Running

```bash
pip install -r requirements.txt
python trainer.py                 # writes delay_results_YYYY-MM-DD.json, pushes to HF
streamlit run streamlit_app.py
```

Set `HF_TOKEN`; create the dataset repo `P2SAMAPA/p2-etf-delay-agent-dynamics-results`
or change `config.RESULTS_REPO`. To stop the live configuration re-selecting
daily, set `config.PINNED_CONFIG = {"trend_span": 126, "horizon": 5, "window": 756}`.

## Validation status (please read)

Tested on synthetic data only; this sandbox cannot reach `huggingface.co`.

* **The Lambert-W root is exact**: checked against the DDE characteristic
  equation directly over 2,000 random `(A, B, τ)` (max residual ~1e-16), and
  confirmed to be the *dominant* root against every other branch `k=-6..6`
  over 500 more draws.
* **A designed stable→unstable crossing was recovered**: the rolling fit's
  median winning `τ` matched the true `τ` exactly, and its margin correlated
  0.74 with the true margin. The *exact* crossing day is noisy (estimated
  onset preceded the true one in this run) — expected, not a bug: detecting
  a small quantity crossing zero from a 126-day rolling fit is inherently
  imprecise near the boundary.
* **Ablation validated cleanly**: with the forecast target made to depend on
  the margin feature (and *only* the target — the input price series and
  therefore every feature stayed untouched, to avoid the target's own
  construction mechanically leaking into baseline momentum features), the
  delay model beat the baseline with t ≈ 3.7–4.8 on both the selection and
  holdout segments. Earlier attempts that injected into the return series
  itself gave a false negative (baseline nearly matched the full model)
  purely from that contamination, not from an ablation flaw — worth knowing
  if this diagnostic is extended.
* **Two real bugs were caught and fixed** during this validation, both
  informative about what "chance level" means here:
  - The onset/event de-duplication helper advanced its cooldown timer on
    every flagged day, not just recorded onsets — nearly disabling
    deduplication whenever the flag was true often (as `margin < 0` can be).
    Fixed, and "onset" was redefined as the *crossing moment* rather than
    the whole below-threshold stretch, for the same reason.
  - The event-size threshold was accidentally computed from *log*-volatility
    (used elsewhere as a ridge feature) instead of raw volatility, making
    the threshold negative and every day count as a "large move". Fixed by
    computing a separate raw-vol series for this purpose.
  - The chance-level formula originally pooled onset counts across ETFs
    against a single dataset-wide timeline, letting an onset on one ETF
    contribute to explaining an event on a different ETF. Fixed to compute
    chance per ETF, then aggregate the way real hits are aggregated.
* **Pure noise, all fixes applied**: hit rate ≈ chance level (8–16% vs
  10–16% across universes and trend spans) and false-alarm rate ≈ its own
  chance level — the diagnostic does not manufacture a lead effect from
  nothing. All three universes landed at "Low" confidence. A planted lead
  relationship (forced onset ~15 days before a forced large move) was
  correctly recovered (hit rate 50% vs 29% chance; false-alarm rate 53% vs
  73% chance; median lead 8 days).
* **Full three-universe run** at realistic size (≈4,500 days, up to 43 ETFs,
  36 grid cells: 3 trend spans × 3 horizons × 4 windows) takes about 30
  seconds — no PDE simulation here, so noticeably faster than the sibling
  reaction-diffusion engine.

Nothing here says it works on real markets. The first real run is the actual
test; read the holdout numbers and baseline comparison before the picks.
Realistic rank-IC for a daily cross-sectional signal is small (order
0.01–0.05); with 36 configurations searched, one strong-looking cell is
expected by chance. This is not financial advice.
