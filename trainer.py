"""
trainer.py  —  P2 DELAY-AGENT-DYNAMICS trainer

Fits a local delayed heterogeneous-agent feedback model (fundamentalist
mean-reversion vs. delayed chartist trend-following) to each ETF's own price
history, reads off the dominant characteristic root of that fitted linear DDE
via the Lambert W function, and asks whether the resulting "distance to Hopf
bifurcation" (stability margin) — plus its trend, oscillation frequency, and
the fitted feedback strengths — helps rank ETFs by forward return relative to
the universe average, over and above a baseline with no delay-model features
at all. See delay_model.py for the theory.

Design choices (each exists to keep the result honest — same shape as the
sibling engines):
  * Cross-sectional target: features and forward return are z-scored across
    ETFs each day, so market drift/beta cancel.
  * Baseline ablation: every configuration is evaluated for the delay model
    AND a no-delay baseline on the same dates; only the difference is credit.
  * Non-overlapping test points spaced `horizon` days apart; Sharpe annualized
    by sqrt(252/horizon); cost charged on the fraction of the top-N replaced.
  * Selection/holdout split: the first 70% of test periods choose the
    configuration; the last 30% is a holdout used for the reported numbers
    and the confidence label.
  * The "pre-crash/pre-breakout" claim is tested explicitly (regime-lead
    diagnostic), not assumed.
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import delay_model as dmod
from data_manager import load_master_data, validate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d: Dict) -> Dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in d.items()}


def confidence_label(hold: Dict, gate_passed: bool = True) -> str:
    """
    Graded on the HOLDOUT of the selected configuration (a single
    pre-selected config, so a one-sided 5% test is legitimate), and only
    upward if the delay model beats the no-delay baseline on BOTH IC and net
    spread — not just one, which is close to a coin flip on its own.
      High   : holdout rank-IC t >= IC_T_HIGH, net top-N spread > 0, and beats
               the baseline on both IC and net spread.
      Medium : IC > 0, net spread > 0, and beats the baseline on both IC and
               net spread.
      Low    : anything else, or fewer than MIN_HOLDOUT_PERIODS test periods,
               or `gate_passed` is False.

    `gate_passed=False` means NO configuration was profitable on the
    SELECTION segment for this pool — the config being scored here is only
    "the best of a bad lot" (see select_config), not one that cleared its own
    eligibility bar. Good-looking holdout numbers on a config that failed
    that bar are not good evidence of anything systematic, so this caps
    confidence at Low regardless of the holdout stats below — this was found
    to matter in practice: a real run had gate_passed=False together with a
    holdout IC t-stat sitting right at the High threshold (1.48 one day,
    1.72 the next, on otherwise near-identical numbers), which without this
    cap would flip between Medium and High on pure day-to-day noise while
    the selection segment quietly kept failing to validate the config at all.
    """
    if not gate_passed:
        return "Low"
    if hold.get("n", 0) < config.MIN_HOLDOUT_PERIODS:
        return "Low"
    ic, sp = hold["ic_mean"], hold["net_spread_mean"]
    d_ic, d_sp = hold["d_ic_mean"], hold["d_spread_mean"]
    beats_baseline = d_ic > 0 and d_sp > 0
    if hold["ic_t"] >= config.IC_T_HIGH and sp > 0 and beats_baseline:
        return "High"
    if ic > 0 and sp > 0 and beats_baseline:
        return "Medium"
    return "Low"


def select_config(grid: List[Dict], trend_span: Optional[int] = None):
    """
    Choose a configuration from the SELECTION segment only: eligible if
    rank-IC and net top-N spread are both positive there, AND the config's
    selection-segment period count clears both an absolute floor
    (MIN_SELECTION_N) and a floor relative to the largest period count seen
    anywhere in this pool (MIN_SELECTION_N_FRACTION) — see config.py's
    comment on MIN_SELECTION_N_FRACTION for why the relative floor matters:
    a thin-sample cell's t-stat is a noisier random variable than a
    large-sample cell's even though the formula already divides by sqrt(n),
    so ranking by raw t-stat alone systematically favors whichever noisy
    cell got lucky. Among eligible configs, rank-sum of the two t-stats. If
    nothing is eligible, fall back to the size-filtered pool (still
    respecting both floors) and report gate_passed=False.
    """
    pool = [g for g in grid if (trend_span is None or g["trend_span"] == trend_span)
            and g["selection"].get("n", 0) >= config.MIN_SELECTION_N]
    if not pool:
        return None, False
    max_n = max(g["selection"]["n"] for g in pool)
    pool = [g for g in pool if g["selection"]["n"] >= config.MIN_SELECTION_N_FRACTION * max_n]
    if not pool:
        return None, False
    elig = [g for g in pool if g["selection"]["ic_mean"] > 0 and g["selection"]["net_spread_mean"] > 0]
    gate = bool(elig)
    cand = elig if gate else pool
    ic_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["ic_t"]))}
    sp_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["net_spread_t"]))}
    best = max(cand, key=lambda g: (ic_order[id(g)] + sp_order[id(g)], g["selection"]["ic_t"]))
    return best, gate


def make_picks(scores: np.ndarray, tickers: List[str], dispersion: float, horizon: int,
               confidence: str, top_n: int) -> List[Dict]:
    order = np.argsort(-scores)[:top_n]
    return [{
        "ticker": tickers[j],
        "expected_return": round(float(scores[j] * dispersion * 100), 3),
        "score_z": round(float(scores[j]), 4),
        "horizon_days": int(horizon),
        "confidence": confidence,
    } for j in order]


def pattern_diagnostics(dm_out: Dict, tickers: List[str], dates: List[str], returns: np.ndarray) -> Dict:
    """Today's snapshot + fragility index + a decimated time series, for the dashboard."""
    n = len(dates)
    margin, omega, A, B, tau = dm_out["margin"], dm_out["omega"], dm_out["A"], dm_out["B"], dm_out["tau"]
    last = n - 1
    valid_today = np.isfinite(margin[last])
    out = {}
    if valid_today.any():
        j = np.where(valid_today)[0]
        order = np.argsort(margin[last, j])
        snapshot = [{"ticker": tickers[j[k]], "margin": _f(margin[last, j[k]]), "omega": _f(omega[last, j[k]]),
                    "period_days": _f(2 * np.pi / omega[last, j[k]]) if omega[last, j[k]] > 1e-6 else None,
                    "A": _f(A[last, j[k]]), "B": _f(B[last, j[k]]), "tau": _f(tau[last, j[k]])}
                   for k in order]
        out["snapshot"] = snapshot
        out["fragility_index"] = float(np.mean(margin[last, j] < 0))
        out["n_unstable"] = int(np.sum(margin[last, j] < 0))
        out["n_covered"] = int(len(j))

    valid_hist = np.isfinite(margin).any(axis=1)
    idx = np.arange(0, n, config.SERIES_STEP)
    idx = idx[valid_hist[idx]]
    out["series"] = {
        "dates": [dates[i] for i in idx],
        "fragility_index": [_f(np.nanmean(margin[i] < 0)) if np.isfinite(margin[i]).any() else None for i in idx],
        "median_margin": [_f(np.nanmedian(margin[i])) if np.isfinite(margin[i]).any() else None for i in idx],
    }
    return out


def analyze_universe(name: str, tickers: List[str], prices_df: pd.DataFrame) -> Optional[Dict]:
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return None
    px_df = prices_df[available]
    px_df = px_df[~px_df.isna().any(axis=1)]
    if len(px_df) < 400:
        logger.warning(f"Not enough data for {name}")
        return None

    values = px_df.values
    returns = np.diff(np.log(values), axis=0)
    log_prices = np.log(values[1:])
    dates = [d.strftime("%Y-%m-%d") for d in px_df.index[1:]]
    n = len(dates)

    cfg = dict(ridge_lambda=config.RIDGE_LAMBDA, min_train_days=config.MIN_TRAIN_DAYS, cost_bps=config.TRADING_COST_BPS)
    fwd_by_h = {h: dmod.compute_forward_returns(returns, h) for h in config.HORIZONS}
    yz_by_h = {h: dmod.cs_zscore(fwd_by_h[h]) for h in config.HORIZONS}

    L = 21
    s1, s2 = dmod.trailing_sum(returns, L), dmod.trailing_sum(returns ** 2, L)
    var21 = (s2 - s1 ** 2 / L) / (L - 1)
    vol21_raw = np.sqrt(np.maximum(var21, 1e-10))          # raw vol, for the event-size threshold below
    vol21_log = 0.5 * np.log(np.maximum(var21, 1e-10))     # log-vol, matches the ridge feature scale

    grid, wf_store, live_ctx, patterns, dm_outs = [], {}, {}, {}, {}
    for trend_span in config.TREND_SPANS:
        logger.info(f"  [{name}] trend_span={trend_span}: fitting delay-feedback model + reaction ridge grid...")
        Fz_raw, dm_out = dmod.build_feature_array(returns, log_prices, trend_span, config.TAU_GRID,
                                                   config.FIT_WINDOW, config.MIN_FIT_DAYS,
                                                   config.MARGIN_CHANGE_LOOKBACK, config.R2_SMOOTH_DAYS)
        Fz = dmod.cs_zscore(Fz_raw)
        dm_outs[trend_span] = dm_out
        patterns[trend_span] = pattern_diagnostics(dm_out, available, dates, returns)
        patterns[trend_span]["regime_lead"] = dmod.regime_lead_diagnostic(dm_out["margin"], returns, vol21_raw, dates, config.LEAD_PARAMS)

        for h in config.HORIZONS:
            cp = dmod.build_cross_products(Fz, yz_by_h[h])
            for window in config.WINDOWS:
                wf = dmod.walk_forward(Fz, cp, fwd_by_h[h], h, window, config.TOP_N, cfg)
                key = (trend_span, h, window)
                live_ctx[key] = (Fz, cp)
                if wf is None:
                    continue
                m = len(wf["rows"])
                split = int(m * config.SELECTION_FRACTION)
                entry = {
                    "trend_span": trend_span, "horizon": h, "window": window, "n_periods": m,
                    "selection": _clean(dmod.summarize(wf, 0, split, h)),
                    "holdout": _clean(dmod.summarize(wf, split, m, h)),
                    "full": _clean(dmod.summarize(wf, 0, m, h)),
                }
                grid.append(entry)
                wf_store[key] = (wf, split)
        logger.info(f"  [{name}] trend_span={trend_span}: done ({len([g for g in grid if g['trend_span'] == trend_span])} configs)")

    if not grid:
        return None

    def live_for(g):
        Fz, cp = live_ctx[(g["trend_span"], g["horizon"], g["window"])]
        return dmod.live_scores(Fz, cp, fwd_by_h[g["horizon"]], g["horizon"], g["window"], cfg)

    selected, gate = select_config(grid)
    pinned = False
    pin = config.PINNED_CONFIG
    if pin:
        match = [g for g in grid if g["trend_span"] == pin["trend_span"] and g["horizon"] == pin["horizon"] and g["window"] == pin["window"]]
        if match:
            selected, pinned = match[0], True
            gate = True   # a pinned config bypasses the normal selection gate entirely (the
                          # user chose it directly), so it shouldn't be capped as "best of a
                          # bad lot" on the strength of a gate check that never ran for it.
    if selected is None:
        return None

    ls = live_for(selected)
    conf = confidence_label(selected["holdout"], gate)
    picks = base_picks = []
    if ls is not None:
        picks = make_picks(ls["score_m"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)
        base_picks = make_picks(ls["score_b"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)

    trend_picks = {}
    for trend_span in patterns:
        g, tgate = select_config(grid, trend_span)
        if g is None:
            continue
        lst = live_for(g)
        tconf = confidence_label(g["holdout"], tgate)
        trend_picks[str(trend_span)] = {
            "config": {"trend_span": trend_span, "horizon": g["horizon"], "window": g["window"]},
            "gate_passed": tgate,
            "holdout": g["holdout"],
            "confidence": tconf,
            "picks": make_picks(lst["score_m"], available, lst["dispersion"], g["horizon"],
                                tconf, config.TOP_N) if lst is not None else [],
        }

    wf, split = wf_store[(selected["trend_span"], selected["horizon"], selected["window"])]
    curves = {
        "dates": [dates[i] for i in wf["rows"]],
        "delay_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_m"])],
        "baseline_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_b"])],
        "split_index": int(split),
    }

    top = {
        "trend_span": selected["trend_span"], "horizon": selected["horizon"], "window": selected["window"],
        "gate_passed": gate, "pinned": pinned, "confidence": conf,
        "selection": selected["selection"], "holdout": selected["holdout"], "full": selected["full"],
        "n_periods": selected["n_periods"],
    }
    feature_weights = None
    if ls is not None:
        feature_weights = {nm: round(float(w), 5) for nm, w in zip(dmod.FEATURE_NAMES, ls["beta_m"])}

    return {
        "tickers": available, "n_days": n, "last_date": dates[-1],
        "selected": top, "picks": picks, "baseline_picks": base_picks,
        "grid": grid, "trend_picks": trend_picks, "curves": curves,
        "patterns": {str(k): v for k, v in patterns.items()}, "feature_weights": feature_weights,
    }


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date, "algorithm": "DELAY-AGENT-DYNAMICS",
        "top_picks": {}, "selected": {}, "grid": {}, "trend_picks": {}, "baseline_picks": {},
        "curves": {}, "patterns": {}, "feature_weights": {}, "universe_meta": {},
        "settings": {"trading_cost_bps": config.TRADING_COST_BPS, "top_n": config.TOP_N,
                    "selection_fraction": config.SELECTION_FRACTION, "tau_grid": config.TAU_GRID,
                    "fit_window": config.FIT_WINDOW},
    }

    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df)
        if out is None:
            continue
        results["top_picks"][name] = out["picks"]
        results["selected"][name] = out["selected"]
        results["grid"][name] = out["grid"]
        results["trend_picks"][name] = out["trend_picks"]
        results["baseline_picks"][name] = out["baseline_picks"]
        results["curves"][name] = out["curves"]
        results["patterns"][name] = out["patterns"]
        results["feature_weights"][name] = out["feature_weights"]
        results["universe_meta"][name] = {"tickers": out["tickers"], "n_days": out["n_days"], "last_date": out["last_date"]}

        s = out["selected"]
        h = s["holdout"]
        logger.info(f"  ✅ Selected: trend_span={s['trend_span']} / {s['horizon']}d / window {s['window']} "
                    f"(selection gate {'passed' if s['gate_passed'] else 'NOT passed'}) | holdout n={h.get('n')} "
                    f"IC={h.get('ic_mean', 0):+.4f} (t={h.get('ic_t', 0):+.2f}) "
                    f"vs baseline IC={h.get('ic_base_mean', 0):+.4f} | net top-{config.TOP_N} excess "
                    f"{h.get('net_spread_mean', 0) * 1e4:+.1f}bps/period | confidence {s['confidence']}")
        for p in out["picks"]:
            logger.info(f"     {p['ticker']}: {p['expected_return']:+.3f}% excess over {p['horizon_days']}d ({p['confidence']})")

    output_path = f"delay_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")

    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
