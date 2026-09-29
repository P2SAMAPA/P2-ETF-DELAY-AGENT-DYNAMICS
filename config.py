"""
config.py  —  Configuration for P2 DELAY-AGENT-DYNAMICS
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-delay-agent-dynamics-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

TOP_N = 3

# ---------------------------------------------------------------------------
# The heterogeneous-agent delay model, per ETF, per day:
#
#   x(t) = log_price(t) - trend(t)      "deviation from fundamental" proxy;
#                                        trend = EMA(log_price, span). No real
#                                        fundamental data exists in the master
#                                        dataset, so this is a price-only
#                                        proxy, same caveat as the sibling
#                                        engines' valuation surfaces.
#
# Continuous-time behavioral model (Beja-Goldman / Chiarella-style two-type
# agent dynamics: fundamentalists pull x back to 0 at rate f; chartists
# extrapolate the trailing trend (x(t)-x(t-tau))/tau and push in that
# direction with strength c):
#
#   dx/dt = -f*x(t) + (c/tau)*(x(t) - x(t-tau))  =  A*x(t) + B*x(t-tau)
#       where A = c/tau - f,  B = -c/tau
#
# Discretized (forward Euler, dt=1 trading day) and fit by rolling OLS on
# TREND_SPAN's x(t) series, for every tau in TAU_GRID, keeping the tau with
# the best in-window fit:
#
#   x(t) - x(t-1) = A*x(t-1) + B*x(t-1-tau) + noise
#
# The dominant characteristic root of the fitted linear DDE has the closed
# form (Lambert W; Asl & Ulsoy 2003, "Analysis of a System of Linear Delay
# Differential Equations", ASME J. Dyn. Sys.):
#
#   lambda_dom = A + W_0(B*tau*exp(-A*tau)) / tau
#
# Re(lambda_dom) > 0 means the fitted local dynamics are (locally) unstable;
# a Hopf-type bifurcation is where Re(lambda_dom) crosses 0 with a nonzero
# Im(lambda_dom), producing a growing oscillation (limit cycle) rather than a
# monotone runaway. STABILITY_MARGIN = -Re(lambda_dom) is what the engine
# tracks: positive and shrinking means the ETF's own trend-vs-reversion
# feedback is approaching that boundary; negative means it has already
# crossed it (by this local linear approximation). See delay_model.py.
# ---------------------------------------------------------------------------
TREND_SPANS = [63, 126, 252]
TAU_GRID = [5, 8, 13, 21, 34, 55]
FIT_WINDOW = 126
MIN_FIT_DAYS = 60
MARGIN_CHANGE_LOOKBACK = 5
R2_SMOOTH_DAYS = 5   # EMA smoothing of per-tau R^2 before picking the winning tau each day (see delay_model.py)

# ---------------------------------------------------------------------------
# Prediction: pooled ridge regression of the cross-sectionally z-scored
# forward H-day return on cross-sectionally z-scored features, refit at each
# evaluation date on the trailing WINDOW days. Two models share the same
# machinery, exactly as in the sibling engines:
#   baseline : x(t-1), 5d return, 21d return, 21d realized log-vol. No delay
#              model at all — plain trend/mean-reversion/vol features.
#   delayed  : baseline + the fitted delay-model readout (stability margin,
#              its 5-day change, oscillation frequency, A, B, tau).
# The delayed model is only credited for what it adds beyond the baseline.
# ---------------------------------------------------------------------------
HORIZONS = [1, 5, 20]
WINDOWS = [252, 504, 756, 1008]
RIDGE_LAMBDA = 0.01
MIN_TRAIN_DAYS = 60

# Evaluation: test points spaced `horizon` days apart (no overlapping forward
# windows). The first SELECTION_FRACTION of test periods pick the
# configuration; the rest is a holdout used for the reported numbers and the
# confidence label.
SELECTION_FRACTION = 0.70
MIN_HOLDOUT_PERIODS = 30
TRADING_COST_BPS = 15
IC_T_HIGH = 1.65

# Optional: pin the live configuration instead of re-selecting daily, e.g.
# {"trend_span": 126, "horizon": 5, "window": 756}. None = select.
PINNED_CONFIG = None

# Pre-crash / pre-breakout lead diagnostic: does a stability-margin onset
# (crossing below MARGIN_THRESHOLD) tend to sit before a large realized move
# — up OR down, since a Hopf-type crossing predicts a growing OSCILLATION,
# not a specific direction? Both trigger definitions are fixed here, not
# tuned on the data.
LEAD_PARAMS = {
    "margin_threshold": 0.0,       # margin crossing below this = instability onset
    "event_window": 10,            # trading days a "large move" is measured over
    "event_vol_k": 2.0,            # |cum return over event_window| > k * (vol21 * sqrt(event_window)) => event
    "dedup_days": 21,              # minimum gap between two onsets/events of the same kind
    "lead_window": 20,             # onset counts as "leading" an event within this many days before it
}

SERIES_STEP = 3   # decimation of stored time series for the dashboard
