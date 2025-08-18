# src/shrinkage_hier.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
import pandas as pd


def _moments_tau2(values: pd.Series, meas_var: pd.Series, min_prior_var: float = 1e-6) -> float:
    """
    Method-of-moments estimate for prior variance tau^2 in a normal-normal model:
        Var(observed means) ≈ tau^2 + mean(measurement variance)
    Guarded to be non-negative (>= min_prior_var).
    """
    v = pd.to_numeric(values, errors="coerce")
    s2 = pd.to_numeric(meas_var, errors="coerce")
    mask = v.notna() & s2.notna()
    if mask.sum() <= 1:
        return max(min_prior_var, 0.0)
    var_obs = float(np.nanvar(v[mask], ddof=1))
    mean_meas = float(np.nanmean(s2[mask]))
    return max(var_obs - mean_meas, min_prior_var)


def _weighted_mean(x: pd.Series, w: pd.Series, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Weighted mean and its measurement variance (1 / sum w).
    Missing-safe; if weights are all invalid, returns (nan, inf).
    """
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return float("nan"), float("inf")
    wsum = float(w[mask].sum())
    mu = float((x[mask] * w[mask]).sum() / max(wsum, eps))
    var = 1.0 / max(wsum, eps)
    return mu, var


def hierarchical_shrink(
    df: pd.DataFrame,
    delta_col: str,
    se_col: str,
    team_col: str = "team",
    driver_col: str = "driver",
    *,
    min_prior_var_team: float = 1e-6,
    min_prior_var_driver: float = 1e-6,
) -> Tuple[pd.Series, pd.Series, pd.Series, Dict[str, float]]:
    """
    Two-level (driver ← team ← field) empirical Bayes shrinkage for a single event's deltas.

    Observations:
        y_i  ~ N(φ_d, s_i^2)
        φ_d  ~ N(θ_team(d), τ_d^2)          (driver random effect around team mean)
        θ_t  ~ N(μ, τ_t^2)                   (team mean random effect around field mean)

    We estimate τ_d^2 and τ_t^2 via method-of-moments on (driver means, team means),
    then compute posterior means with closed-form normal-normal updates.

    Inputs
    ------
    df : DataFrame containing one row per driver×team (for this event)
    delta_col : column name with observed delta (seconds)
    se_col    : column name with observed standard error (seconds)
    team_col  : team identifier column
    driver_col: driver identifier column

    Returns
    -------
    shrunk : posterior mean for each row (driver-level)
    shrink_w : pooling weight on the observed deviation (τ_d^2 / (τ_d^2 + s_i^2))
               Note: smaller weight ⇒ stronger pooling to team mean
    post_sd : posterior SD of φ_d  (≈ sqrt(τ_d^2 * s_i^2 / (τ_d^2 + s_i^2)))
    meta : dict with estimated hyperparameters (tau_team^2, tau_driver^2, mu_field)
    """
    d = df.copy()

    # Sanity & types
    y = pd.to_numeric(d[delta_col], errors="coerce")
    s = pd.to_numeric(d[se_col], errors="coerce")
    team = d[team_col].astype(str)
    driver = d[driver_col].astype(str)

    # Weights from measurement SE
    s2 = (s ** 2).replace([np.inf, -np.inf], np.nan)
    w = 1.0 / s2
    w = w.where(np.isfinite(w), np.nan)

    # ----- TEAM LAYER: team means & field mean -----
    # Driver means for this event are just y with measurement var s2
    # Compute team-level means with precision weights
    team_stats = (
        pd.DataFrame({"y": y, "w": w, "team": team})
        .dropna(subset=["y", "w"])
        .groupby("team", as_index=False)
        .apply(lambda g: pd.Series({
            "m_team": _weighted_mean(g["y"], g["w"])[0],      # weighted mean
            "v_team": _weighted_mean(g["y"], g["w"])[1],      # measurement variance of the mean
        }))
        .reset_index(drop=True)
    )

    if team_stats.empty:
        # Fallback: nothing we can do—return the raw values
        meta = {"tau_team2": float("nan"), "tau_driver2": float("nan"), "mu_field": float("nan")}
        return y, pd.Series(np.nan, index=df.index), s, meta

    mu_field, _ = _weighted_mean(team_stats["m_team"], 1.0 / team_stats["v_team"].replace(0.0, np.nan))
    tau_team2 = _moments_tau2(team_stats["m_team"] - mu_field, team_stats["v_team"], min_prior_var_team)

    # Shrink team means toward field mean
    a = tau_team2 / (tau_team2 + team_stats["v_team"].replace(0.0, 1e-12))
    team_stats["m_team_shrunk"] = mu_field + a * (team_stats["m_team"] - mu_field)
    # (Optional team posterior var available if needed)
    # team_stats["v_team_post"] = tau_team2 * team_stats["v_team"] / (tau_team2 + team_stats["v_team"])

    # Map shrunk team mean to each driver row
    team_to_shrunk = dict(zip(team_stats["team"].astype(str), team_stats["m_team_shrunk"]))
    t_shrunk = team.map(team_to_shrunk)

    # ----- DRIVER LAYER: shrink driver means toward shrunk team means -----
    # For this event, each driver has one y with measurement var s2
    # Estimate τ_d^2 from deviations around shrunk team means
    tau_driver2 = _moments_tau2(y - t_shrunk, s2, min_prior_var_driver)

    b = tau_driver2 / (tau_driver2 + s2.replace(0.0, 1e-12))
    shrunk = t_shrunk + b * (y - t_shrunk)
    # Posterior variance of driver effect (ignoring team-level uncertainty for simplicity)
    post_var = tau_driver2 * s2 / (tau_driver2 + s2.replace(0.0, 1e-12))
    post_sd = np.sqrt(post_var)

    meta = {"tau_team2": float(tau_team2), "tau_driver2": float(tau_driver2), "mu_field": float(mu_field)}
    return shrunk, b, post_sd, meta

# ---- tiny guardrails used by season aggregation pipeline ----
import numpy as np
import pandas as pd

# knobs
N0_ESS = 300            # pseudo-laps to fully trust
ESS_FLOOR = 50          # min assumed laps
SE_FLOOR_FRAC = 0.035   # SE floor for R⊕Q fusion
USE_TEAM_PRIOR = False
TEAM_PRIOR_STRENGTH = 150

def cap_and_fuse_rq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs (required): delta_R, se_R, delta_Q, se_Q
    Output: adds raw_est, raw_se (inverse-variance fused with SE floor)
    """
    d = df.copy()
    for c in ("se_R", "se_Q"):
        d[c] = np.maximum(pd.to_numeric(d[c], errors="coerce"), SE_FLOOR_FRAC)
    wR = 1.0 / (d["se_R"] ** 2)
    wQ = 1.0 / (d["se_Q"] ** 2)
    num = wR * pd.to_numeric(d["delta_R"], errors="coerce") + wQ * pd.to_numeric(d["delta_Q"], errors="coerce")
    den = wR + wQ
    d["raw_est"] = np.where(den > 0, num / den, 0.0)
    d["raw_se"]  = np.sqrt(np.where(den > 0, 1.0 / den, np.inf))
    return d

def apply_ess_shrinkage(df: pd.DataFrame, field_mean: float = 0.0) -> pd.DataFrame:
    """
    Pull raw_est toward field_mean when n_eff is small.
    alpha_ess = n_eff / (n_eff + N0_ESS), with floor on n_eff.
    """
    d = df.copy()
    n = np.maximum(pd.to_numeric(d.get("n_eff", 0), errors="coerce").fillna(0.0), ESS_FLOOR)
    alpha = n / (n + N0_ESS)
    d["alpha_ess"] = alpha
    d["est_shrunk"] = alpha * pd.to_numeric(d["raw_est"], errors="coerce") + (1.0 - alpha) * field_mean
    return d

def _team_col(df: pd.DataFrame) -> str:
    return "Team" if "Team" in df.columns else ("team" if "team" in df.columns else "")

def apply_team_prior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional gentle pull toward team mean until laps accrue.
    alpha_team = n_eff / (n_eff + TEAM_PRIOR_STRENGTH)
    """
    d = df.copy()
    tcol = _team_col(d)
    if not tcol or not USE_TEAM_PRIOR:
        d["alpha_team"] = 1.0
        d["driver_pace_final"] = d.get("est_shrunk", d.get("raw_est"))
        return d

    team_mean = d.groupby(tcol)["est_shrunk"].transform("mean")
    n = np.maximum(pd.to_numeric(d.get("n_eff", 0), errors="coerce").fillna(0.0), 1.0)
    alpha = n / (n + TEAM_PRIOR_STRENGTH)
    d["alpha_team"] = alpha
    d["driver_pace_final"] = alpha * d["est_shrunk"] + (1.0 - alpha) * team_mean
    return d
