"""
delay_model.py  —  Delayed heterogeneous-agent feedback / Hopf-proximity signal

Idea
----
Two-type agent models of price formation (Beja & Goldman 1980; Chiarella 1992;
Day & Huang 1990; Dieci & He 2018 survey) split demand into fundamentalists, who
push price back toward a reference value, and chartists, who extrapolate the
recent trend — and chartists act on a DELAYED read of the trend (they need `tau`
days of price history to estimate "the trend"). The resulting price dynamics is
a delay differential equation, and delay is exactly the ingredient that can turn
a stable fundamentalist-dominated market into one with self-sustained
oscillations, bubbles and crashes: as chartist strength or the delay grows, the
system can lose stability through a HOPF BIFURCATION — a pair of complex
characteristic roots crossing the imaginary axis, producing a growing
oscillation (limit cycle) rather than a monotone explosion.

This module does NOT predict price directly. It fits the local (A, B, tau) of
that feedback loop to each ETF's own recent price history, computes the
DOMINANT characteristic root of the fitted linear DDE in closed form (via the
Lambert W function — see below), and tracks how close that root sits to the
instability boundary. That distance — "stability margin" — is the operational
version of "how close is this market's feedback loop to a Hopf bifurcation."

The linear DDE and its dominant root
-------------------------------------
Let x(t) = log_price(t) - trend(t) (trend = EMA of log price; "how far price
has drifted from a slow reference"). The linearized two-agent model is

    dx/dt = -f*x(t) + (c/tau)*(x(t) - x(t-tau))  =:  A*x(t) + B*x(t-tau)

f = fundamentalist mean-reversion strength, c = chartist trend-following
strength, tau = the delay (how many days of trend chartists extrapolate). A and
B are what gets fitted; f and c are not separately identified from A, B alone
(A = c/tau - f, B = -c/tau — two equations, two unknowns given tau, so f and c
ARE recoverable, but the engine works with A, B, tau directly since that is
what the stability calculus below needs).

For the scalar DDE dx/dt = A*x(t) + B*x(t-tau), EVERY characteristic root is

    lambda_k = A + W_k(B*tau*e^{-A*tau}) / tau,   k = 0, ±1, ±2, ...

where W_k is the k-th branch of the Lambert W function (w*e^w = z). This is a
classical closed-form result (Asl & Ulsoy, "Analysis of a System of Linear
Delay Differential Equations", ASME J. Dynamic Systems, 2003) — and the
PRINCIPAL branch W_0 always gives the RIGHTMOST (dominant) root, for any real
A, B, tau > 0, whether that root ends up real or (for z < -1/e) a complex
conjugate pair. That means the entire infinite-dimensional stability question
for this linear DDE collapses to one evaluation of scipy's `lambertw`.

    lambda_dom = A + lambertw(B*tau*exp(-A*tau), k=0) / tau
    stability_margin  = -Re(lambda_dom)     (>0 stable, <0 unstable; shrinking
                                              margin = approaching the boundary)
    hopf_frequency    =  Im(lambda_dom)     (nonzero => the instability, if/when
                                              reached, arrives as an OSCILLATION
                                              with this angular frequency, not a
                                              monotone runaway — the Hopf case)

(A, B, tau) are re-estimated every trading day by rolling OLS on the trailing
FIT_WINDOW days, independently for each candidate tau in TAU_GRID, keeping
whichever tau explains that ETF's recent dynamics best (highest in-window R²).

Honest scope
------------
* This is a LOCAL, LINEAR, backward-looking approximation re-fit daily, not a
  claim that any ETF literally obeys one fixed DDE. "Stability margin" is a
  diagnostic built from that local linear fit, not a law of motion.
* A Hopf crossing predicts a GROWING OSCILLATION, not a direction — it could
  presage a breakout rally or a crash. The regime-lead diagnostic below tests
  both.
* The forecast is a small linear model on top of the margin and friends,
  always compared against a baseline with no delay-model features at all, so
  the delay machinery is credited only for what it adds.
"""

import numpy as np
import pandas as pd
from scipy.special import lambertw
from typing import Dict, List, Optional, Tuple

BASE_FEATURES = ["x", "mom5", "mom21", "vol21"]
DELAY_FEATURES = ["margin", "margin_chg", "omega", "A_hat", "B_hat", "tau_norm"]
FEATURE_NAMES = BASE_FEATURES + DELAY_FEATURES
N_BASE = len(BASE_FEATURES)
N_FEAT = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------
def rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing sum over up to `w` rows ending at each row (partial windows early on)."""
    a = np.asarray(a, dtype=float)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    lo = np.maximum(np.arange(a.shape[0]) - w + 1, 0)
    return c[1:] - c[lo]


def trailing_sum(a: np.ndarray, w: int) -> np.ndarray:
    """Trailing sum over exactly `w` rows; NaN until a full window exists."""
    out = rolling_sum(a, w)
    out[: w - 1] = np.nan
    return out


def ema(a: np.ndarray, span: int) -> np.ndarray:
    return pd.DataFrame(a).ewm(span=span, adjust=False).mean().values


def compute_forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    """forward[t] = sum(returns[t+1 : t+1+horizon]) (log return from close t to close t+H); NaN at the tail."""
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    end = n - horizon
    if end > 0:
        fwd[:end] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


# ---------------------------------------------------------------------------
# Dominant characteristic root of dx/dt = A*x(t) + B*x(t-tau)
# ---------------------------------------------------------------------------
def dominant_root(A: np.ndarray, B: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized principal-branch Lambert-W solution. Returns (margin, omega)
    where margin = -Re(lambda_dom) and omega = Im(lambda_dom), same shape as
    A/B. A*tau is clipped defensively before exponentiating (fitted A, B stay
    small in practice, but this guards against a pathological window fit).
    """
    At = np.clip(A * tau, -20.0, 20.0)
    z = B * tau * np.exp(-At)
    z = np.clip(z, -1e6, 1e6)
    w = lambertw(z, k=0)
    lam = A + w / tau
    return -np.real(lam), np.imag(lam)


# ---------------------------------------------------------------------------
# Rolling bivariate OLS (closed form, fully vectorized): for each tau, fit
#   y(t) = A*u(t) + B*w(t) + const   (least squares, over a trailing window)
# where y = delta_x, u = x(t-1), w = x(t-1-tau) — using centered-sum formulas
# so no explicit per-day matrix solve or Python loop is needed.
# ---------------------------------------------------------------------------
def _centered_sums(y, u, w, window):
    n_win = trailing_sum(np.ones_like(y), window)
    Sy, Su, Sw = trailing_sum(y, window), trailing_sum(u, window), trailing_sum(w, window)
    Suu = trailing_sum(u * u, window) - Su * Su / n_win
    Sww = trailing_sum(w * w, window) - Sw * Sw / n_win
    Suw = trailing_sum(u * w, window) - Su * Sw / n_win
    Suy = trailing_sum(u * y, window) - Su * Sy / n_win
    Swy = trailing_sum(w * y, window) - Sw * Sy / n_win
    Syy = trailing_sum(y * y, window) - Sy * Sy / n_win
    return n_win, Suu, Sww, Suw, Suy, Swy, Syy


def rolling_bivariate_ols(y: np.ndarray, u: np.ndarray, w: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (A, B, R2, n_eff), each shape (n, T); NaN where the window/inputs are invalid."""
    ok = np.isfinite(y) & np.isfinite(u) & np.isfinite(w)
    y0, u0, w0 = np.where(ok, y, 0.0), np.where(ok, u, 0.0), np.where(ok, w, 0.0)
    n_win, Suu, Sww, Suw, Suy, Swy, Syy = _centered_sums(y0, u0, w0, window)
    n_eff = trailing_sum(ok.astype(float), window)
    det = Suu * Sww - Suw * Suw
    eps = 1e-9 * (Suu + Sww + 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_det = np.where(np.abs(det) > eps, det, np.nan)
        A = (Suy * Sww - Swy * Suw) / safe_det
        B = (Suu * Swy - Suw * Suy) / safe_det
        ss_res = Syy - A * Suy - B * Swy
        r2 = 1.0 - ss_res / np.where(Syy > 1e-12, Syy, np.nan)
    return A, B, r2, n_eff


def fit_delay_features(log_prices: np.ndarray, trend_span: int, tau_grid: List[int],
                       fit_window: int, min_fit_days: int, margin_lookback: int,
                       r2_smooth: int = 5) -> Dict[str, np.ndarray]:
    """
    For every day and every ETF: pick, among `tau_grid`, the delay whose
    rolling-window fit best explains that ETF's recent dynamics, and return
    that fit's stability margin, its trend over the last `margin_lookback`
    days, the Hopf oscillation frequency, A, B and the winning tau — each
    shape (n, T), NaN before enough history exists.

    The per-tau R² used to pick the winner is smoothed with a short EMA
    (`r2_smooth` days) BEFORE comparing taus. Picking the argmax of the raw
    daily R² makes the winning tau (and therefore the margin estimate) flip
    on noise-level day-to-day differences between near-tied candidates — the
    same failure mode a daily K-means refit had in the sibling barycenter
    engine. Smoothing first trades a little responsiveness for a much less
    jumpy margin estimate; see delay_model tests in the repo notes.
    """
    n, T = log_prices.shape
    trend = ema(log_prices, trend_span)
    x = log_prices - trend
    dx = np.diff(x, axis=0, prepend=x[:1])   # dx[t] = x[t]-x[t-1]; row 0 unused (no x[-1])
    dx[0] = np.nan

    per_tau = []
    for tau in tau_grid:
        u = np.roll(x, 1, axis=0)     # x(t-1)
        u[:1] = np.nan
        wv = np.roll(x, 1 + tau, axis=0)   # x(t-1-tau)
        wv[: 1 + tau] = np.nan
        A, B, r2, n_eff = rolling_bivariate_ols(dx, u, wv, fit_window)
        margin, omega = dominant_root(np.nan_to_num(A), np.nan_to_num(B), float(tau))
        valid = np.isfinite(r2) & (n_eff >= min_fit_days)
        r2_for_selection = pd.DataFrame(np.where(valid, r2, -np.inf)).ewm(span=r2_smooth, adjust=False).mean().values
        per_tau.append({"tau": tau, "A": A, "B": B, "margin": margin, "omega": omega,
                        "valid": valid, "r2_sel": r2_for_selection, "r2_raw": r2})

    best_r2 = np.full((n, T), -np.inf)
    best = {k: np.full((n, T), np.nan) for k in ["margin", "omega", "A", "B", "tau", "r2_raw"]}
    for pt in per_tau:
        take = pt["valid"] & (pt["r2_sel"] > best_r2)
        best_r2 = np.where(take, pt["r2_sel"], best_r2)
        for key, val in [("margin", pt["margin"]), ("omega", pt["omega"]), ("A", pt["A"]), ("B", pt["B"]),
                         ("tau", np.full((n, T), float(pt["tau"]))), ("r2_raw", pt["r2_raw"])]:
            best[key] = np.where(take, val, best[key])

    finite = np.isfinite(best_r2) & (best_r2 > -np.inf)
    for key in best:
        best[key] = np.where(finite, best[key], np.nan)

    margin_prev = np.roll(best["margin"], margin_lookback, axis=0)
    margin_prev[:margin_lookback] = np.nan
    margin_chg = best["margin"] - margin_prev

    return {"x": x, "margin": best["margin"], "margin_chg": margin_chg, "omega": best["omega"],
            "A": best["A"], "B": best["B"], "tau": best["tau"], "r2": best["r2_raw"]}


def build_feature_array(returns: np.ndarray, log_prices: np.ndarray, trend_span: int,
                        tau_grid: List[int], fit_window: int, min_fit_days: int,
                        margin_lookback: int, r2_smooth: int = 5) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Full (n, T, N_FEAT) feature tensor for one trend_span: baseline features + delay-model readout."""
    n, T = returns.shape
    L = 21
    mom5 = trailing_sum(returns, 5)
    mom21 = trailing_sum(returns, L)
    s1, s2 = trailing_sum(returns, L), trailing_sum(returns ** 2, L)
    var = (s2 - s1 ** 2 / L) / (L - 1)
    vol21 = 0.5 * np.log(np.maximum(var, 1e-10))

    dm = fit_delay_features(log_prices, trend_span, tau_grid, fit_window, min_fit_days, margin_lookback, r2_smooth)

    feats = np.full((n, T, N_FEAT), np.nan)
    feats[:, :, 0] = dm["x"]
    feats[:, :, 1] = mom5
    feats[:, :, 2] = mom21
    feats[:, :, 3] = vol21
    feats[:, :, 4] = dm["margin"]
    feats[:, :, 5] = dm["margin_chg"]
    feats[:, :, 6] = dm["omega"]
    feats[:, :, 7] = dm["A"]
    feats[:, :, 8] = dm["B"]
    feats[:, :, 9] = dm["tau"] / tau_grid[len(tau_grid) // 2]   # normalized around a mid-grid tau
    return feats, dm


# ---------------------------------------------------------------------------
# Cross-sectional regression machinery (shared shape with the sibling engines)
# ---------------------------------------------------------------------------
def cs_zscore(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
        out = np.where(sd > 1e-9, (a - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    out[~np.isfinite(a)] = np.nan
    return out


def build_cross_products(Fz: np.ndarray, Yz: np.ndarray) -> Dict:
    n, T, p = Fz.shape
    feat_ok = np.isfinite(Fz).all(axis=(1, 2))
    y_ok = np.isfinite(Yz).all(axis=1)
    ok = feat_ok & y_ok
    F0 = np.where(ok[:, None, None], np.nan_to_num(Fz), 0.0)
    Y0 = np.where(ok[:, None], np.nan_to_num(Yz), 0.0)
    XtX = np.einsum("dtp,dtq->dpq", F0, F0)
    Xty = np.einsum("dtp,dt->dp", F0, Y0)
    return {
        "cXtX": np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)], axis=0),
        "cXty": np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)], axis=0),
        "cN": np.concatenate([[0], np.cumsum(ok.astype(int))]),
        "feat_ok": feat_ok,
    }


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, lam_rel: float) -> np.ndarray:
    p = XtX.shape[0]
    lam = lam_rel * np.trace(XtX) / p + 1e-12
    return np.linalg.solve(XtX + lam * np.eye(p), Xty)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def walk_forward(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                 top_n: int, cfg: Dict) -> Optional[Dict]:
    """
    Evaluate the delay model and the baseline on non-overlapping test points
    spaced `horizon` days apart. At test row i, the model is fit on training
    rows [i-window, i-horizon] (targets fully resolved by i; no leakage).
    """
    n, T, p = Fz.shape
    base_idx = list(range(N_BASE))
    lam = cfg["ridge_lambda"]
    min_train = cfg["min_train_days"]
    cost = cfg["cost_bps"] / 1e4

    rows, ic_m, ic_b, sg_m, sg_b, sn_m, sn_b = [], [], [], [], [], [], []
    prev_m, prev_b = set(), set()
    i = 0
    while i <= n - 1 - horizon:
        a, b = max(0, i - window), i - horizon
        if b < a or not cp["feat_ok"][i]:
            i += 1 if b < a else horizon
            continue
        if cp["cN"][b + 1] - cp["cN"][a] < min_train:
            i += horizon
            continue
        XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
        Xty = cp["cXty"][b + 1] - cp["cXty"][a]
        try:
            beta_m = ridge_solve(XtX, Xty, lam)
            beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], lam)
        except np.linalg.LinAlgError:
            i += horizon
            continue
        f = Fz[i]
        s_m = f @ beta_m
        s_b = f[:, base_idx] @ beta_b
        r = fwd[i]
        if not np.isfinite(r).all():
            i += horizon
            continue
        top_m = set(np.argsort(-s_m)[:top_n].tolist())
        top_b = set(np.argsort(-s_b)[:top_n].tolist())
        g_m = r[list(top_m)].mean() - r.mean()
        g_b = r[list(top_b)].mean() - r.mean()
        c_m = (top_n - len(top_m & prev_m)) / top_n * cost
        c_b = (top_n - len(top_b & prev_b)) / top_n * cost
        rows.append(i)
        ic_m.append(spearman(s_m, r))
        ic_b.append(spearman(s_b, r))
        sg_m.append(g_m)
        sg_b.append(g_b)
        sn_m.append(g_m - c_m)
        sn_b.append(g_b - c_b)
        prev_m, prev_b = top_m, top_b
        i += horizon

    if len(rows) < 10:
        return None
    return {k: np.array(v) for k, v in dict(rows=rows, ic_m=ic_m, ic_b=ic_b, sg_m=sg_m, sg_b=sg_b,
                                              sn_m=sn_m, sn_b=sn_b).items()}


def summarize(wf: Dict, lo: int, hi: int, horizon: int) -> Dict:
    sl = slice(lo, hi)
    ic_m, ic_b = wf["ic_m"][sl], wf["ic_b"][sl]
    sn_m, sn_b, sg_m = wf["sn_m"][sl], wf["sn_b"][sl], wf["sg_m"][sl]
    n = len(ic_m)
    if n < 3:
        return {"n": n}
    ann = np.sqrt(252.0 / horizon)
    sd_net = sn_m.std() + 1e-12
    sd_gross = sg_m.std() + 1e-12
    return {
        "n": int(n),
        "ic_mean": float(ic_m.mean()), "ic_t": tstat(ic_m),
        "ic_base_mean": float(ic_b.mean()), "ic_base_t": tstat(ic_b),
        "net_spread_mean": float(sn_m.mean()), "net_spread_t": tstat(sn_m),
        "gross_spread_mean": float(sg_m.mean()),
        "base_net_spread_mean": float(sn_b.mean()),
        "hit_rate": float((sn_m > 0).mean()),
        "sharpe_net": float(sn_m.mean() / sd_net * ann),
        "sharpe_gross": float(sg_m.mean() / sd_gross * ann),
        "d_ic_mean": float((ic_m - ic_b).mean()), "d_ic_t": tstat(ic_m - ic_b),
        "d_spread_mean": float((sn_m - sn_b).mean()), "d_spread_t": tstat(sn_m - sn_b),
    }


def live_scores(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                cfg: Dict) -> Optional[Dict]:
    """Fit on the latest resolved training window and score the final row (today)."""
    n = Fz.shape[0]
    i = n - 1
    a, b = max(0, i - window), i - horizon
    if b < a or not cp["feat_ok"][i] or cp["cN"][b + 1] - cp["cN"][a] < cfg["min_train_days"]:
        return None
    XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
    Xty = cp["cXty"][b + 1] - cp["cXty"][a]
    base_idx = list(range(N_BASE))
    beta_m = ridge_solve(XtX, Xty, cfg["ridge_lambda"])
    beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], cfg["ridge_lambda"])
    f = Fz[i]
    disp = np.nanmean(np.nanstd(fwd[a:b + 1], axis=1))
    return {
        "score_m": f @ beta_m,
        "score_b": f[:, base_idx] @ beta_b,
        "beta_m": beta_m,
        "dispersion": float(disp) if np.isfinite(disp) else 0.0,
    }


# ---------------------------------------------------------------------------
# Pre-crash / pre-breakout regime-lead diagnostic
# ---------------------------------------------------------------------------
def _onsets(flag: np.ndarray, dedup: int) -> np.ndarray:
    out, last = [], -10 ** 9
    for t in np.where(flag)[0]:
        if t - last > dedup:
            out.append(t)
            last = t   # only advance on a RECORDED onset — advancing on every
                       # flagged day (even unrecorded ones) would require a
                       # `dedup`-day gap with the flag never true at all,
                       # which collapses onset counts whenever the flag is
                       # true often (e.g. margin<0 roughly half the time
                       # under pure noise) rather than de-duplicating bursts
                       # around genuinely separate onsets.
    return np.array(out, dtype=int)


def regime_lead_diagnostic(margin: np.ndarray, returns: np.ndarray, vol21: np.ndarray,
                           dates: List[str], lp: Dict) -> Dict:
    """
    Does a stability-margin onset (the crossing moment, below) tend to sit
    before a large realized move (either direction, since a Hopf crossing
    predicts a growing OSCILLATION, not a sign)?

      instability onset: margin crosses below margin_threshold (the crossing
                          moment, not every day margin happens to sit below
                          it — margin often stays below threshold for a
                          stretch once a window's fit calls the dynamics
                          unstable, so flagging the whole stretch would make
                          "onsets" nearly as common as "days")
      large-move event  : |cumulative return over event_window days| exceeds
                          event_vol_k * (21d realized vol * sqrt(event_window))
      (both de-duplicated per ETF: an onset/event needs `dedup_days` quiet
      days before it counts again; thresholds fixed in advance, per ETF)

    Computed per ETF, then pooled. The chance baseline is ALSO computed per
    ETF (from that ETF's own onset/event counts and its own covered-day
    count), then aggregated the same way the real hits are — a single
    dataset-wide chance rate pooling all ETFs' onset counts would implicitly
    let an onset on one ETF "explain" an event on another, which cannot
    happen, and would overstate how likely a hit is by chance alone.
    """
    n, T = margin.shape
    fwd_cum = np.full((n, T), np.nan)
    W = lp["event_window"]
    if n > W:
        cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
        fwd_cum[: n - W] = cum[W + 1: n + 1] - cum[1: n - W + 1]
    thresh = lp["event_vol_k"] * vol21 * np.sqrt(W)
    event_flag_all = np.abs(fwd_cum) > thresh

    below = margin < lp["margin_threshold"]
    prev_above = np.roll(~below, 1, axis=0)
    prev_above[0] = False
    onset_flag_all = below & prev_above

    LW, dedup = lp["lead_window"], lp["dedup_days"]
    leads = []
    n_onsets = n_events = n_hits = n_false = 0
    chance_hits_expected = chance_false_expected = 0.0
    for j in range(T):
        ok = np.isfinite(margin[:, j]) & np.isfinite(fwd_cum[:, j])
        if not ok.any():
            continue
        start = int(np.argmax(ok))
        ev = _onsets(event_flag_all[:, j] & ok, dedup)
        on = _onsets(onset_flag_all[:, j] & ok, dedup)
        on = on[on >= start]
        ev = ev[ev >= start]
        n_days_j = int(ok.sum())
        n_onsets += len(on)
        n_events += len(ev)

        for d in ev:
            prior = on[(on >= d - LW) & (on < d)]
            if len(prior):
                n_hits += 1
                leads.append(int(d - prior.min()))
        n_false += sum(1 for m in on if not ((ev > m) & (ev <= m + LW)).any())

        # Per-ETF chance baseline: probability a given event's `LW`-day prior
        # window contains at least one of THIS ETF's own onsets (and,
        # symmetrically, a given onset's forward window contains one of this
        # ETF's own events), if onsets/events were placed uniformly at random
        # over this ETF's own covered days.
        p_win_j = min(LW / max(n_days_j, 1), 1.0)
        chance_hit_j = 1 - (1 - p_win_j) ** len(on) if len(on) else 0.0
        chance_false_j = (1 - p_win_j) ** len(ev) if len(ev) else 1.0
        chance_hits_expected += len(ev) * chance_hit_j
        chance_false_expected += len(on) * chance_false_j

    return {
        "available": True,
        "n_instability_onsets": int(n_onsets),
        "n_large_move_events": int(n_events),
        "hit_rate": float(n_hits / n_events) if n_events else None,
        "chance_hit_rate": float(chance_hits_expected / n_events) if n_events else None,
        "median_lead_days": float(np.median(leads)) if leads else None,
        "false_alarm_rate": float(n_false / n_onsets) if n_onsets else None,
        "chance_false_alarm_rate": float(chance_false_expected / n_onsets) if n_onsets else None,
    }
