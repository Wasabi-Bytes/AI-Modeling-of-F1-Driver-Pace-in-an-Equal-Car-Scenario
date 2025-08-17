# src/model_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import warnings
import math

import numpy as np
import pandas as pd

# Statsmodels for FE/splines + robust SEs
import statsmodels.formula.api as smf
from patsy import bs

# (Keep sklearn around for potential future ridge; not used for OLS inference now)
from sklearn.linear_model import Ridge

# Local modules
from load_data import load_config, load_all_data
from clean_data import clean_event_payload  # uses your existing cleaner

# Silence noisy future warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.*")


# ---------- Paths ----------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- Utilities ----------
def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-").replace("--", "-")


def _get_ref_values(cfg: Dict[str, Any], laps: pd.DataFrame) -> Tuple[str, float]:
    fuel_ref = cfg.get("fuel_ref_lap", "median")
    if fuel_ref == "median":
        L_mid = float(np.nanmedian(pd.to_numeric(laps.get("lap_number", np.nan), errors="coerce")))
    else:
        try:
            L_mid = float(fuel_ref)
        except Exception:
            L_mid = float(np.nanmedian(pd.to_numeric(laps.get("lap_number", np.nan), errors="coerce")))
    tyre_ref = str(cfg.get("tyre_ref_compound", "M")).upper()
    return tyre_ref, L_mid


def _driver_counts(df: pd.DataFrame, driver_col: str = "driver") -> pd.Series:
    return df.groupby(driver_col, dropna=False).size().rename("n")


def _se_from_residuals(resid: np.ndarray, n: int) -> float:
    if n <= 1:
        return float("nan")
    sd = float(np.nanstd(resid, ddof=1)) if np.isfinite(resid).any() else float("nan")
    return sd / math.sqrt(max(n, 1))


def _empirical_bayes_shrinkage(delta: pd.Series, se: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Simple EB shrinkage toward 0 (global mean). Estimates tau^2 via method-of-moments:
      Var(observed) = tau^2 + mean(se^2)  => tau^2 = max(Var(observed) - mean(se^2), eps)
    Returns (shrunk_delta, shrinkage_weight)
    """
    delta = pd.to_numeric(delta, errors="coerce")
    se2 = pd.to_numeric(se, errors="coerce") ** 2
    valid = delta.notna() & se2.notna() & (se2 >= 0)
    if valid.sum() == 0:
        return delta, pd.Series(np.zeros_like(delta), index=delta.index)

    var_obs = float(np.nanvar(delta[valid], ddof=1)) if valid.sum() > 1 else 0.0
    mean_se2 = float(np.nanmean(se2[valid])) if valid.any() else 0.0
    tau2 = max(var_obs - mean_se2, 1e-6)

    w = tau2 / (tau2 + se2.replace(0.0, 1e-9))
    shrunk = w * delta.fillna(0.0)  # shrink toward 0
    return shrunk, w


# ---------- Column coalescers ----------
_TEAM_COL_CANDIDATES = [
    "team", "Team", "Constructor", "ConstructorName", "TeamName",
    "Entrant", "Car", "CarName", "ConstructorTeam"
]
_DRIVER_COL_CANDIDATES = ["driver", "Driver", "DriverNumber", "DriverId", "DriverRef"]
_SESSION_COL_CANDIDATES = ["session", "Session", "phase", "Phase", "q_session"]
_EVENT_COL_CANDIDATES = ["Event", "EventName", "GrandPrix", "GP", "gp"]

def _ensure_driver_column(d: pd.DataFrame) -> pd.DataFrame:
    if "driver" in d.columns:
        d["driver"] = d["driver"].astype(str)
        return d
    for c in _DRIVER_COL_CANDIDATES:
        if c in d.columns:
            d["driver"] = d[c].astype(str)
            return d
    d["driver"] = "UNK"
    return d

def _ensure_team_column(d: pd.DataFrame) -> pd.DataFrame:
    if "team" in d.columns:
        d["team"] = d["team"].astype(str)
        return d
    for c in _TEAM_COL_CANDIDATES:
        if c in d.columns:
            d["team"] = d[c].astype(str)
            return d
    raise ValueError("[race/quali] No team column found; please ensure a standard team field is present.")

def _ensure_session_column(d: pd.DataFrame) -> pd.DataFrame:
    if "session" in d.columns:
        d["session"] = d["session"].astype(str)
        return d
    for c in _SESSION_COL_CANDIDATES:
        if c in d.columns:
            d["session"] = d[c].astype(str)
            return d
    d["session"] = "Q"
    return d

def _ensure_event_column(d: pd.DataFrame) -> pd.DataFrame:
    if "event" in d.columns:
        d["event"] = d["event"].astype(str)
        return d
    for c in _EVENT_COL_CANDIDATES:
        if c in d.columns:
            d["event"] = d[c].astype(str)
            return d
    # Fallback to year+gp if present
    if {"year","gp"}.issubset(d.columns):
        d["event"] = (d["year"].astype(str) + " " + d["gp"].astype(str)).astype(str)
    else:
        d["event"] = "UNKNOWN_EVENT"
    return d


# ============================================================
# =============== RACE METRICS (WITHIN TEAM) =================
# ============================================================

def _prep_race_columns(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()

    # Only consume "pace" laps if lap_ok is available
    if "lap_ok" in d.columns:
        d = d[d["lap_ok"].astype(bool)].copy()

    d = _ensure_driver_column(d)
    d = _ensure_team_column(d)
    d = _ensure_event_column(d)

    needed_cols = ["LapTimeSeconds", "driver", "team", "compound", "lap_on_tyre", "lap_number"]
    for c in needed_cols:
        if c not in d.columns:
            raise ValueError(f"[race_metrics] Missing required column: {c}")

    d["compound"] = d["compound"].astype(str).str.upper()
    d["lap_on_tyre"] = pd.to_numeric(d["lap_on_tyre"], errors="coerce")
    d["lap_number"] = pd.to_numeric(d["lap_number"], errors="coerce")
    d["LapTimeSeconds"] = pd.to_numeric(d["LapTimeSeconds"], errors="coerce")

    # Trim obvious outliers (heavy tails) by stint: > Q3 + 3*IQR within driver×stint
    if "stint_id" in d.columns:
        grp = d.groupby(["driver", "stint_id"])
    else:
        grp = d.groupby(["driver"])  # fallback if stint missing
    q1 = grp["LapTimeSeconds"].transform("quantile", 0.25)
    q3 = grp["LapTimeSeconds"].transform("quantile", 0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    keep = (d["LapTimeSeconds"] <= (q3 + 3 * iqr).fillna(q3 + 10.0))  # generous cap if IQR=0
    d = d.loc[keep].copy()

    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["LapTimeSeconds", "driver", "team", "compound", "lap_on_tyre", "lap_number", "event"])
    # Normalized helper columns
    d["driver_team"] = d["driver"].astype(str) + "@" + d["team"].astype(str)
    d["driver_event"] = d["driver"].astype(str) + "-" + d["event"].astype(str)
    return d


def race_metrics_corrections_team(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Robust correction model with event fixed effects + spline controls.
    Steps:
      1) Fit: lap_time_s ~ C(event) + C(compound)
               + bs(lap_on_tyre, df=DF_AGE) + bs(lap_number, df=DF_LAP)
               + [optional: C(compound):bs(lap_on_tyre, df=DF_AGE_INT)]
      2) Normalize laps: subtract predicted controls (keep intercept & event FE baseline)
      3) Team-demean normalized times → driver deltas
      4) SE per driver from residual spread (HC3 fit for robustness)
    """
    d = _prep_race_columns(race_df)
    if d.empty:
        return pd.DataFrame(columns=["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"])

    # Configurable spline degrees (defaults are sensible)
    df_age = int(cfg.get("race_spline_df_tyre_age", 4))
    df_lap = int(cfg.get("race_spline_df_lap_num", 4))
    df_age_int = int(cfg.get("race_spline_df_tyre_age_interact", 3))  # for compound × age interaction
    use_age_interact = bool(cfg.get("race_use_compound_age_interaction", True))

    d = d.copy()
    d["lap_time_s"] = d["LapTimeSeconds"].astype(float)

    # Build formula with optional interaction (inject df via f-strings)
    base = f"lap_time_s ~ C(event) + C(compound) + bs(lap_on_tyre, df={df_age}) + bs(lap_number, df={df_lap})"
    if use_age_interact:
        base += f" + C(compound):bs(lap_on_tyre, df={df_age_int})"

    m = smf.ols(base, data=d).fit(cov_type="HC3")  # heteroskedasticity-robust

    # Predicted control component
    d["pred"] = m.predict(d)

    # Normalize: remove predicted controls but keep intercept-like baseline
    intercept = float(m.params.get("Intercept", 0.0))
    d["norm_time"] = d["lap_time_s"] - (d["pred"] - intercept)

    # Within-team demeaning
    d["team_mean"] = d.groupby("team", dropna=False)["norm_time"].transform("mean")
    d["demeaned"] = d["norm_time"] - d["team_mean"]

    # Per-driver stats
    grp = d.groupby(["driver", "team"], dropna=False)
    mean_demeaned = grp["demeaned"].mean().rename("race_delta_s")
    n_by_driver = grp.size().rename("race_n")

    # SE from per-driver residual spread
    res = d["demeaned"] - d.groupby(["driver", "team"], dropna=False)["demeaned"].transform("mean")
    se_by_driver = res.groupby([d["driver"], d["team"]]).std() / np.sqrt(
        res.groupby([d["driver"], d["team"]]).count().clip(lower=1)
    )

    out = mean_demeaned.reset_index()
    out["race_se_s"] = out.set_index(["driver", "team"]).index.map(se_by_driver)
    out["race_n"] = out.set_index(["driver", "team"]).index.map(n_by_driver).astype(int).values
    out["race_model"] = "corrections_team(fe+spline)"
    return out[["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"]].sort_values("race_delta_s").reset_index(drop=True)


def race_metrics_ols_team(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Driver-within-team OLS with event FE + non-linear controls and cluster-robust SEs.
    Model:
      lap_time_s ~ C(event) + C(team) + C(driver_team)
                   + bs(lap_on_tyre, df=DF_AGE) + bs(lap_number, df=DF_LAP) + C(compound)
    Cluster-robust SEs (clusters = driver×event) help with stint autocorrelation.
    Deltas are constructed from normalized laps (team-demeaned) rather than raw coefs.
    """
    d = _prep_race_columns(race_df)
    if d.empty:
        return pd.DataFrame(columns=["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"])

    df_age = int(cfg.get("race_spline_df_tyre_age", 4))
    df_lap = int(cfg.get("race_spline_df_lap_num", 4))

    d = d.copy()
    d["lap_time_s"] = d["LapTimeSeconds"].astype(float)

    # Fit with event FE, team FE, driver@team FE, plus non-linear controls
    formula = (
        f"lap_time_s ~ C(event) + C(team) + C(driver_team)"
        + f" + bs(lap_on_tyre, df={df_age}) + bs(lap_number, df={df_lap})"
        + " + C(compound)"
    )

    # Cluster id = driver×event for robust SEs
    d["cluster_id"] = d["driver_event"].astype(str)

    m = smf.ols(formula, data=d).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["cluster_id"]},
        use_t=True
    )

    # Normalized laps using model prediction; keep intercept-like baseline
    intercept = float(m.params.get("Intercept", 0.0))
    d["pred"] = m.predict(d)
    d["norm_time"] = d["lap_time_s"] - (d["pred"] - intercept)

    # Within-team means → deltas (lower = faster)
    d["team_mean"] = d.groupby("team", dropna=False)["norm_time"].transform("mean")
    d["demeaned"] = d["norm_time"] - d["team_mean"]

    grp = d.groupby(["driver", "team"], dropna=False)
    mean_demeaned = grp["demeaned"].mean().rename("race_delta_s")
    n_by_driver = grp.size().rename("race_n")

    # SE per driver from residual spread after normalization
    res = d["demeaned"] - d.groupby(["driver", "team"], dropna=False)["demeaned"].transform("mean")
    se_by_driver = res.groupby([d["driver"], d["team"]]).std() / np.sqrt(
        res.groupby([d["driver"], d["team"]]).count().clip(lower=1)
    )

    out = mean_demeaned.reset_index()
    out["race_se_s"] = out.set_index(["driver", "team"]).index.map(se_by_driver)
    out["race_n"] = out.set_index(["driver", "team"]).index.map(n_by_driver).astype(int).values
    out["race_model"] = "ols_team(fe+spline,clustered)"
    return out[["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"]].sort_values("race_delta_s").reset_index(drop=True)


# ============================================================
# ================= QUALIFYING (EVOLUTION-AWARE) =============
# ============================================================

def _winsorize_series(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    """Winsorize a series to [q_lower, q_upper]."""
    s = s.astype(float)
    lo = s.quantile(lower_q) if 0.0 < lower_q < 0.5 else None
    hi = s.quantile(upper_q) if 0.5 < upper_q < 1.0 else None
    if lo is not None:
        s = s.clip(lower=lo)
    if hi is not None:
        s = s.clip(upper=hi)
    return s


def quali_metrics_within_team(quali_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Evolution-aware quali metric.
    Steps per event:
      1) Keep only pace laps (lap_ok) and finite LapTimeSeconds.
      2) Normalize each lap by its segment median (Q1/Q2/Q3).
      3) Winsorize normalized laps (optional; default upper-tail only).
      4) For each driver×segment, take the best normalized lap.
      5) Compute teammate gap per segment vs team-best normalized lap.
      6) Combine segments using inverse-variance weights where each driver×segment variance
         is the variance of that driver's normalized laps in the segment (post-winsor), divided by n_laps.
      7) Return per-driver (driver, team, quali_delta_s, quali_se_s, quali_k).
    """
    out_cols = ["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"]

    if quali_df is None or len(quali_df) == 0:
        return pd.DataFrame(columns=out_cols)

    d = quali_df.copy()
    d = _ensure_driver_column(d)
    d = _ensure_team_column(d)
    d = _ensure_session_column(d)

    # Lap time & validity
    d["LapTimeSeconds"] = pd.to_numeric(d.get("LapTimeSeconds", d.get("LapTime")), errors="coerce")
    if "lap_ok" in d.columns:
        d = d[d["lap_ok"].astype(bool)]
    if "is_valid" in d.columns:
        d = d[d["is_valid"].astype(bool)]
    d = d[np.isfinite(d["LapTimeSeconds"])]
    if d.empty:
        return pd.DataFrame(columns=out_cols)

    # 1) Segment (Q1/Q2/Q3) normalization by robust center (median)
    seg_med = d.groupby("session")["LapTimeSeconds"].transform("median")
    d["norm_lt"] = d["LapTimeSeconds"] - seg_med

    # 2) Winsorization config (post-normalization)
    use_winsor = bool(cfg.get("quali_winsorize", True))
    q_low = float(cfg.get("quali_winsor_lower_q", 0.00))   # keep default no lower trim
    q_high = float(cfg.get("quali_winsor_upper_q", 0.05))  # light upper-tail trim
    if use_winsor:
        d["norm_lt"] = (
            d.groupby("session", group_keys=False)["norm_lt"]
             .apply(lambda s: _winsorize_series(s, q_low, q_high))
        )

    # 3) Optional top-k AFTER normalization (defaults to using all laps)
    use_topk = bool(cfg.get("quali_use_topk_after_norm", False))
    k = int(cfg.get("quali_top_k", 3))
    if use_topk:
        d = (d.sort_values("norm_lt")
               .groupby(["driver", "team", "session"], as_index=False)
               .head(k))

    # 4) Driver-session best normalized lap
    drv_sess = (
        d.groupby(["driver", "team", "session"], as_index=False)
         .agg(
             best_norm_lt=("norm_lt", "min"),
             laps_n=("norm_lt", "size"),
             laps_sd=("norm_lt", "std")
         )
    )
    # Variance proxy per driver-session: sd^2 / laps_n (guard against zeros)
    drv_sess["var_ds"] = (drv_sess["laps_sd"].fillna(0.0) ** 2) / drv_sess["laps_n"].clip(lower=1)

    # 5) Teammate best per segment (team reference in normalized space)
    team_best = (
        drv_sess.groupby(["team", "session"])["best_norm_lt"]
                .min()
                .rename("team_best_norm")
                .reset_index()
    )
    g = drv_sess.merge(team_best, on=["team", "session"], how="left")
    g["gap_s"] = g["best_norm_lt"] - g["team_best_norm"]

    # 6) Precision-weighted combine across segments per driver
    # Weight w = 1 / var_ds (fallback for zero variance -> small epsilon)
    eps = float(cfg.get("quali_var_epsilon", 1e-6))
    g["w"] = 1.0 / (g["var_ds"].replace(0.0, eps))

    def _combine_driver(df: pd.DataFrame) -> pd.Series:
        W = df["w"].sum()
        if W <= 0 or not np.isfinite(W):
            return pd.Series({"quali_delta_s": np.nan, "quali_se_s": np.nan, "quali_k": 0})
        delta = (df["gap_s"] * df["w"]).sum() / W
        se = math.sqrt(1.0 / W)
        k_sessions = int(df["session"].nunique())
        return pd.Series({"quali_delta_s": float(delta), "quali_se_s": float(se), "quali_k": k_sessions})

    # NOTE: keep compatibility with older pandas (no include_groups kw)
    agg = g.groupby(["driver", "team"]).apply(_combine_driver).reset_index()

    if agg.empty:
        return pd.DataFrame(columns=out_cols)

    return agg[out_cols].sort_values("quali_delta_s").reset_index(drop=True)


# ============================================================
# ================== EVENT-LEVEL COMBINATION =================
# ============================================================

def combine_event_metrics(
    race_df: pd.DataFrame,
    quali_df: Optional[pd.DataFrame],
    wR: float = 0.6,   # kept for backward compatibility; ignored by precision-weighting
    wQ: float = 0.4,   # kept for backward compatibility; ignored by precision-weighting
    apply_bayes_shrinkage: bool = True
) -> pd.DataFrame:
    """
    Precision-weighted event combination:
      delta_event = (delta_R / sigma_R^2 + delta_Q / sigma_Q^2) / (1/sigma_R^2 + 1/sigma_Q^2)
      se_event     = sqrt( 1 / (1/sigma_R^2 + 1/sigma_Q^2) )
    Falls back to whichever side is present. Also emits QA columns:
      - event_wR_eff, event_wQ_eff (effective weights that sum to 1 when both present)
    """
    if quali_df is None:
        quali_df = pd.DataFrame(columns=["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"])

    m = pd.merge(race_df, quali_df, on=["driver", "team"], how="outer")

    # Normalize numeric types
    for col in ["race_delta_s", "quali_delta_s", "race_se_s", "quali_se_s"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    def _combine_row(r):
        dR, sR = r.get("race_delta_s"), r.get("race_se_s")
        dQ, sQ = r.get("quali_delta_s"), r.get("quali_se_s")

        # Handle missing/invalid SEs by treating that side as missing
        validR = (pd.notna(dR) and pd.notna(sR) and float(sR) > 0.0)
        validQ = (pd.notna(dQ) and pd.notna(sQ) and float(sQ) > 0.0)

        if validR and validQ:
            wR_eff = 1.0 / (float(sR) ** 2)
            wQ_eff = 1.0 / (float(sQ) ** 2)
            denom = wR_eff + wQ_eff
            if denom <= 0 or not np.isfinite(denom):
                return pd.Series({"event_delta_s": np.nan, "event_se_s": np.nan, "event_wR_eff": np.nan, "event_wQ_eff": np.nan})
            delta = (float(dR) * wR_eff + float(dQ) * wQ_eff) / denom
            se = math.sqrt(1.0 / denom)
            return pd.Series({
                "event_delta_s": float(delta),
                "event_se_s": float(se),
                "event_wR_eff": float(wR_eff / denom),
                "event_wQ_eff": float(wQ_eff / denom),
            })
        elif validR:
            return pd.Series({
                "event_delta_s": float(dR),
                "event_se_s": float(sR),
                "event_wR_eff": 1.0,
                "event_wQ_eff": 0.0,
            })
        elif validQ:
            return pd.Series({
                "event_delta_s": float(dQ),
                "event_se_s": float(sQ),
                "event_wR_eff": 0.0,
                "event_wQ_eff": 1.0,
            })
        else:
            return pd.Series({
                "event_delta_s": np.nan,
                "event_se_s": np.nan,
                "event_wR_eff": np.nan,
                "event_wQ_eff": np.nan,
            })

    comb = m.apply(_combine_row, axis=1)
    m = pd.concat([m, comb], axis=1)

    # Empirical-Bayes shrinkage (toward 0 = equal to teammate)
    if apply_bayes_shrinkage:
        for col_delta, col_se, out_col, w_col in [
            ("race_delta_s", "race_se_s", "race_delta_s_shrunk", "race_shrink_w"),
            ("quali_delta_s", "quali_se_s", "quali_delta_s_shrunk", "quali_shrink_w"),
            ("event_delta_s", "event_se_s", "event_delta_s_shrunk", "event_shrink_w"),
        ]:
            if col_delta in m.columns and col_se in m.columns:
                shrunk, w = _empirical_bayes_shrinkage(m[col_delta], m[col_se])
                m[out_col] = shrunk
                m[w_col] = w

    return m


# ============================================================
# ===================== ORCHESTRATOR =========================
# ============================================================

def _unpack_clean_payload(
    result: Union[Tuple, Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Supports multiple return shapes from clean_event_payload:
      - (race_df, race_summary, quali_df, quali_summary)
      - (race_df, race_summary, quali_df)
      - (race_df, race_summary)
      - {"race_laps":..., "race_summary":..., "quali_laps":..., "quali_summary":...}
    """
    if isinstance(result, dict):
        dR = result.get("race_laps")
        rS = result.get("race_summary", {})
        dQ = result.get("quali_laps")
        qS = result.get("quali_summary", {})
        return dR, rS, dQ, qS

    if isinstance(result, tuple):
        if len(result) == 4:
            dR, rS, dQ, qS = result
            return dR, rS, dQ, qS
        if len(result) == 3:
            dR, rS, dQ = result
            return dR, rS, dQ, {}
        if len(result) == 2:
            dR, rS = result
            return dR, rS, None, {}
        if len(result) == 1:
            dR = result[0]
            return dR, {}, None, {}
    # Fallback
    raise ValueError("clean_event_payload returned an unsupported structure")


def compute_event_metrics(event: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For a single event: compute race (within-team), quali (within-team),
    combine them into per-event deltas, and apply optional EB shrinkage.
    """
    dR_clean, r_summary, dQ_clean, q_summary = _unpack_clean_payload(clean_event_payload(event, cfg))

    model_choice = str(cfg.get("race_model", "ols_team")).lower()
    if model_choice in ("ols_team", "ridge_team"):
        race_out = race_metrics_ols_team(dR_clean, cfg)
    elif model_choice in ("corrections_team", "corrections"):
        race_out = race_metrics_corrections_team(dR_clean, cfg)
    else:
        # Fallback: raw team-demeaned means on pace laps
        d_tmp = _prep_race_columns(dR_clean)
        d_tmp["team_mean"] = d_tmp.groupby("team")["LapTimeSeconds"].transform("mean")
        d_tmp["demeaned"] = d_tmp["LapTimeSeconds"] - d_tmp["team_mean"]
        grp = d_tmp.groupby(["driver", "team"], dropna=False)["demeaned"]
        mean_ = grp.mean()
        n_ = grp.size()
        race_out = (
            mean_.rename("race_delta_s")
            .reset_index()
            .assign(
                race_se_s=lambda x: x.apply(
                    lambda r: _se_from_residuals(
                        d_tmp.loc[(d_tmp["driver"] == r["driver"]) & (d_tmp["team"] == r["team"]), "demeaned"].to_numpy()
                        - float(mean_.loc[(r["driver"], r["team"])]),
                        int(n_.loc[(r["driver"], r["team"])]),
                    ),
                    axis=1,
                ),
                race_n=lambda x: x.apply(lambda r: int(n_.loc[(r["driver"], r["team"])]), axis=1),
                race_model="raw_team"
            )
        )[["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"]]

    quali_out = quali_metrics_within_team(dQ_clean, cfg) if dQ_clean is not None else pd.DataFrame(
        columns=["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"]
    )

    # Precision-weighted combination now happens inside combine_event_metrics
    wR = float(cfg.get("wR", 0.6))  # ignored by precision weighting but kept for backward compatibility
    wQ = float(cfg.get("wQ", 0.4))  # ignored by precision weighting but kept for backward compatibility
    apply_bayes = bool(cfg.get("apply_bayes_shrinkage", True))
    merged = combine_event_metrics(race_out, quali_out, wR=wR, wQ=wQ, apply_bayes_shrinkage=apply_bayes)

    meta = {
        "year": event.get("year"),
        "gp": event.get("gp"),
        "race_summary": r_summary,
        "quali_summary": q_summary,
    }
    return {"meta": meta, "metrics": merged, "race_only": race_out, "quali_only": quali_out}


# ---------- Save helpers ----------
def save_event_metrics(ev_result: Dict[str, Any], outdir: Path) -> None:
    meta = ev_result["meta"]
    df = ev_result["metrics"]
    race_df = ev_result["race_only"]
    quali_df = ev_result["quali_only"]

    slug = f"{meta['year']}-{_slug(meta['gp'])}"
    _ensure_dir(outdir)

    df.to_csv(outdir / f"{slug}-event_metrics.csv", index=False)
    race_df.to_csv(outdir / f"{slug}-race_metrics.csv", index=False)
    quali_df.to_csv(outdir / f"{slug}-quali_metrics.csv", index=False)


# ---------- Main ----------
def main():
    cfg = load_config("config/config.yaml")

    outdir = _project_root() / "outputs" / "metrics"
    _ensure_dir(outdir)

    events = load_all_data(cfg)
    print(f"[INFO] Loaded {len(events)} events; computing per-event metrics…")

    all_rows = []
    for ev in events:
        res = compute_event_metrics(ev, cfg)
        save_event_metrics(res, outdir)

        meta = res["meta"]
        df = res["metrics"].copy()
        df.insert(0, "year", meta["year"])
        df.insert(1, "gp", meta["gp"])
        all_rows.append(df)

        # Lightweight QA: average effective weights for this event (rows with both sides present)
        try:
            both_mask = df["event_wR_eff"].notna() & df["event_wQ_eff"].notna()
            if both_mask.any():
                wR_mean = float(df.loc[both_mask, "event_wR_eff"].mean())
                wQ_mean = float(df.loc[both_mask, "event_wQ_eff"].mean())
                print(f"[INFO] {meta['year']} {meta['gp']}: avg effective weights -> race={wR_mean:.2f}, quali={wQ_mean:.2f}")
        except Exception:
            pass

        nR = int(res["race_only"].get("race_n", pd.Series(dtype=int)).sum()) if not res["race_only"].empty else 0
        nQ = int(res["quali_only"].get("quali_k", pd.Series(dtype=int)).sum()) if not res["quali_only"].empty else 0
        print(f"[INFO] {meta['year']} {meta['gp']}: metrics computed "
              f"(drivers={df['driver'].nunique()}, race_n={nR}, quali_k={nQ})")

    if all_rows:
        combined = pd.concat(all_rows, axis=0, ignore_index=True)
        combined.to_csv(outdir / "all_events_metrics.csv", index=False)
        print(f"[INFO] Wrote combined metrics to: {outdir / 'all_events_metrics.csv'}")
    else:
        print("[WARN] No events available for metrics.")


if __name__ == "__main__":
    main()
