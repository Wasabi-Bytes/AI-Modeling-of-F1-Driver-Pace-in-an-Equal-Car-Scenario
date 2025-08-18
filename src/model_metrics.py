# src/model_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
from shrinkage_hier import hierarchical_shrink
import warnings
import math
import logging

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

# Logging (quiet by default; respects your YAML logging.level elsewhere)
logger = logging.getLogger(__name__)

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


# ============================================================
# ============================================================
# =============== Track metadata (lazy load) =================
# ============================================================
_TRACK_META_CACHE: Optional[pd.DataFrame] = None

def _project_root() -> Path:  # keep local copy used above
    return Path(__file__).resolve().parent.parent

def _load_track_meta(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Read data/track_meta.csv if present and normalize columns:
      event_key, track_type (lowercased), downforce_index [0,1],
      drs_zones, speed_bias, overtaking_difficulty
    """
    global _TRACK_META_CACHE
    if _TRACK_META_CACHE is not None:
        return _TRACK_META_CACHE

    path = str(cfg.get("paths", {}).get("track_meta", "data/track_meta.csv"))
    f = (_project_root() / path)
    if not f.exists():
        logger.info("[track_meta] No track_meta at %s; running without track controls.", f)
        _TRACK_META_CACHE = None
        return None

    try:
        tm = pd.read_csv(f)
    except Exception as e:
        logger.warning("[track_meta] Failed to read %s (%s); running without track controls.", f, e)
        _TRACK_META_CACHE = None
        return None

    # Normalize column names
    cols = {c.lower(): c for c in tm.columns}
    def _ren(old, new):
        if old in cols and cols[old] != new:
            tm.rename(columns={cols[old]: new}, inplace=True)

    _ren("event_key", "event_key")
    _ren("track_type", "track_type")
    _ren("downforce_index", "downforce_index")
    _ren("drs_zones", "drs_zones")
    _ren("speed_bias", "speed_bias")
    _ren("overtaking_difficulty", "overtaking_difficulty")

    if "event_key" not in tm.columns:
        logger.warning("[track_meta] Missing 'event_key' column; running without track controls.")
        _TRACK_META_CACHE = None
        return None

    # Value normalization + helpful flags
    tm["event_key"] = tm["event_key"].astype(str).str.strip()
    tm["__ekey__"] = tm["event_key"].str.lower().str.strip()
    if "track_type" in tm.columns:
        tm["track_type"] = tm["track_type"].astype(str).str.strip().str.lower()
    if "downforce_index" in tm.columns:
        tm["downforce_index"] = pd.to_numeric(tm["downforce_index"], errors="coerce").clip(0.0, 1.0)
    if "drs_zones" in tm.columns:
        tm["drs_zones"] = pd.to_numeric(tm["drs_zones"], errors="coerce").astype("Int64")
    if "speed_bias" in tm.columns:
        tm["speed_bias"] = pd.to_numeric(tm["speed_bias"], errors="coerce")
    if "overtaking_difficulty" in tm.columns:
        tm["overtaking_difficulty"] = pd.to_numeric(tm["overtaking_difficulty"], errors="coerce").clip(0.0, 1.0)

    _TRACK_META_CACHE = tm
    logger.info("[track_meta] Loaded %d rows from %s", len(tm), f)
    return _TRACK_META_CACHE


def _event_key(year: Any, gp: Any) -> str:
    # Back-compat display key (not used for matching)
    return f"{year} {gp}".strip()


def _attach_event_track_tags(df: Optional[pd.DataFrame], event: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Attach track metadata columns to laps dataframe (no-op if metadata missing).
    Columns attached when available:
      - track_type, downforce_index
      - drs_zones, speed_bias, overtaking_difficulty
    """
    if df is None or len(df) == 0:
        return df

    tm = _load_track_meta(cfg)
    if tm is None or tm.empty:
        return df

    gp = str(event.get("gp", "")).lower().strip()
    year = str(event.get("year", "")).strip()

    # 1) Try direct/substring match against event_key
    ekeys = tm["__ekey__"]
    row = tm.loc[ekeys == gp]
    if row.empty:
        row = tm.loc[ekeys.apply(lambda k: (k in gp) or (gp in k))]

    # 2) Heuristic mapping from GP name to CSV key (handles Monza/Silverstone/etc.)
    if row.empty:
        gp_map = {
            "british": "silverstone",
            "silverstone": "silverstone",
            "italian": "monza",
            "monza": "monza",
            "saudi": "jeddah",
            "saudi arabian": "jeddah",
            "jeddah": "jeddah",
            "emilia": "imola",
            "romagna": "imola",
            "imola": "imola",
            "azerbaijan": "baku",
            "baku": "baku",
            "united states": "usa_cota",
            "austin": "usa_cota",
            "cota": "usa_cota",
            "brazil": "sao_paulo",
            "são paulo": "sao_paulo",
            "sao paulo": "sao_paulo",
            "interlagos": "sao_paulo",
            "mexico": "mexico",
            "mexican": "mexico",
            "qatar": "qatar",
            "losail": "qatar",
            "miami": "miami",
            "las vegas": "vegas",
            "vegas": "vegas",
            "abu dhabi": "abu_dhabi",
            "yas marina": "abu_dhabi",
            "hungary": "hungary",
            "hungarian": "hungary",
            "spain": "spain",
            "spanish": "spain",
            "japan": "japan",
            "japanese": "japan",
            "australia": "australia",
            "australian": "australia",
            "austria": "austria",
            "spielberg": "austria",
            "red bull ring": "austria",
            "canada": "canadian",
            "canadian": "canadian",
            "montreal": "canadian",
            "china": "china",
            "shanghai": "china",
            "monaco": "monaco",
            "belgian": "belgian",
            "spa": "belgian",
            "netherlands": "zandvoort",
            "dutch": "zandvoort",
            "zandvoort": "zandvoort",
            "singapore": "singapore",
            "barcelona": "spain",
            "catalunya": "spain",
        }
        key = None
        for k, v in gp_map.items():
            if k in gp:
                key = v
                break
        if key is not None:
            row = tm.loc[ekeys == key]

    # 3) Fallback to __default__
    if row.empty:
        row = tm.loc[ekeys == "__default__"]

    if row.empty:
        logger.info("[track_meta] No metadata match for gp='%s' (year=%s)", gp, year)
        return df

    r0 = row.iloc[0]
    attach_cols = ["track_type", "downforce_index", "drs_zones", "speed_bias", "overtaking_difficulty"]

    d2 = df.copy()
    for c in attach_cols:
        if c in tm.columns:
            if c not in d2.columns:
                d2[c] = r0.get(c)
            else:
                # fill only missing values if the column already exists
                d2[c] = d2[c].where(d2[c].notna(), r0.get(c))
    return d2


def _append_track_controls_to_formula(base_formula: str, d: pd.DataFrame, cfg: Dict[str, Any]) -> str:
    """
    Optionally extend the model formula with track controls:
      - archetype: + C(track_type)
      - continuous: + bs(downforce_index, df=3)
    Only if cfg['track_effects']['use'] is True and the needed column exists with at least some non-null data.
    """
    te = cfg.get("track_effects", {}) or {}
    if not bool(te.get("use", False)):
        return base_formula

    mode = str(te.get("mode", "archetype")).lower()
    f = base_formula
    if mode == "archetype":
        if "track_type" in d.columns and d["track_type"].notna().any():
            f += " + C(track_type)"
            logger.info("[track_effects] Enabled archetype FE (C(track_type))")
        else:
            logger.info("[track_effects] track_type not present; skipping archetype FE")
    elif mode == "continuous":
        if "downforce_index" in d.columns and d["downforce_index"].notna().any():
            f += " + bs(downforce_index, df=3)"
            logger.info("[track_effects] Enabled continuous control (bs(downforce_index, df=3))")
        else:
            logger.info("[track_effects] downforce_index not present; skipping continuous control")
    else:
        logger.info("[track_effects] Unknown mode '%s'; skipping", mode)
    return f

# ============================================================
# ========== Empirical Bayes shrinkage (smart target) ========
# ============================================================
def _read_shrinkage_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch shrinkage config with sensible defaults."""
    s = cfg.get("shrinkage", {}) or {}
    return {
        "target": str(s.get("target", "field_mean")).lower(),  # "field_mean" | "team_mean" | "zero"
        "include_team_re": bool(s.get("include_team_re", True)),
        "prior_var_mode": str(s.get("prior_var_mode", "moments")).lower(),  # "moments" | "fixed"
        "prior_var_fixed": float(s.get("prior_var_fixed", 0.0)),  # used if prior_var_mode == "fixed"
        "min_prior_var": float(s.get("min_prior_var", 1e-6)),
    }


def _moments_tau2(resid: pd.Series, se2: pd.Series, min_prior_var: float) -> float:
    """
    Method-of-moments for tau^2 using centered residuals:
      Var(observed residuals) ≈ tau^2 + mean(se^2)
    """
    valid = resid.notna() & se2.notna() & (se2 >= 0)
    if valid.sum() <= 1:
        return max(min_prior_var, 0.0)
    var_obs = float(np.nanvar(resid[valid], ddof=1))
    mean_se2 = float(np.nanmean(se2[valid]))
    return max(var_obs - mean_se2, min_prior_var)


def _team_level_shrink(
    delta: pd.Series,
    se: pd.Series,
    team: Optional[pd.Series],
    cfg_sh: Dict[str, Any],
) -> pd.Series:
    """
    Optional team random effect: compute team means, shrink them toward field mean,
    then return the shrunk team mean for each row's team as the per-driver target.
    """
    if team is None or team.isna().all():
        mu_field = float(pd.to_numeric(delta, errors="coerce").mean())
        return pd.Series(mu_field, index=delta.index)

    d = pd.to_numeric(delta, errors="coerce")
    s2 = (pd.to_numeric(se, errors="coerce") ** 2).replace([np.inf, -np.inf], np.nan)

    by_team = pd.DataFrame({"delta": d, "se2": s2, "team": team}).dropna(subset=["delta"])
    if by_team.empty:
        mu_field = float(d.mean())
        return pd.Series(mu_field, index=delta.index)

    tm = by_team.groupby("team", dropna=False)["delta"].mean()
    n_t = by_team.groupby("team", dropna=False)["delta"].size().astype(float)
    mean_se2_t = by_team.groupby("team", dropna=False)["se2"].mean().fillna(0.0)
    se2_team_mean = (mean_se2_t / n_t.clip(lower=1.0)).reindex(tm.index).fillna(0.0)

    mu_field = float(tm.mean())
    resid_t = tm - mu_field
    tau2_team = (
        _moments_tau2(resid_t, se2_team_mean, cfg_sh["min_prior_var"])
        if cfg_sh["prior_var_mode"] == "moments"
        else max(cfg_sh["prior_var_fixed"], cfg_sh["min_prior_var"])
    )
    w_t = tau2_team / (tau2_team + se2_team_mean.replace(0.0, 1e-12))
    tm_shrunk = mu_field + w_t * (tm - mu_field)

    return team.map(tm_shrunk).fillna(mu_field)


def _empirical_bayes_shrinkage_smart(
    delta: pd.Series,
    se: pd.Series,
    team: Optional[pd.Series],
    cfg: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, float]:
    """
    General EB shrinkage:
      - Target can be "field_mean", "team_mean", or "zero"
      - Optional team random effect adjusts the target via shrunk team means
      - Prior variance tau^2 via method-of-moments (default) or fixed
    Returns: (shrunk_delta, shrink_weight, tau2)
    """
    cfg_sh = _read_shrinkage_cfg(cfg)
    d = pd.to_numeric(delta, errors="coerce")
    s2 = (pd.to_numeric(se, errors="coerce") ** 2).replace([np.inf, -np.inf], np.nan)

    # Determine target mean per row
    target_mode = cfg_sh["target"]
    if target_mode == "zero":
        mu = pd.Series(0.0, index=d.index)
    elif target_mode == "team_mean":
        if team is None or team.isna().all():
            mu = pd.Series(float(d.mean()), index=d.index)
        else:
            tmp = pd.DataFrame({"team": team, "d": d})
            mu = tmp.groupby("team", dropna=False)["d"].transform("mean")
    else:  # "field_mean"
        mu = pd.Series(float(d.mean()), index=d.index)

    # Optional team random-effect: refine mu by shrinking team means toward field mean
    if cfg_sh["include_team_re"]:
        mu = _team_level_shrink(d, se, team, cfg_sh)

    # Prior variance tau^2
    if cfg_sh["prior_var_mode"] == "fixed":
        tau2 = max(cfg_sh["prior_var_fixed"], cfg_sh["min_prior_var"])
    else:
        resid = d - mu
        tau2 = _moments_tau2(resid, s2, cfg_sh["min_prior_var"])

    # EB weight and shrink
    w = tau2 / (tau2 + s2.replace(0.0, 1e-12))
    shrunk = mu + w * (d - mu)

    return shrunk, w, float(tau2)


def _empirical_bayes_shrinkage(delta: pd.Series, se: pd.Series):
    """
    Back-compat shim used by older call sites: delegate to the smart EB
    with defaults (field mean target, no team RE).
    """
    shrunk, w, _ = _empirical_bayes_shrinkage_smart(delta, se, team=None, cfg={})
    return shrunk, w


# ============================================================
# =================== Column coalescers ======================
# ============================================================
_TEAM_COL_CANDIDATES = [
    "team", "Team", "Constructor", "ConstructorName", "TeamName",
    "Entrant", "Car", "CarName", "ConstructorTeam",
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
    if {"year", "gp"}.issubset(d.columns):
        d["event"] = (d["year"].astype(str) + " " + d["gp"].astype(str)).astype(str)
    else:
        d["event"] = "UNKNOWN_EVENT"
    return d


# ============================================================
# =============== RACE METRICS (WITHIN TEAM) =================
# ============================================================
def _prep_race_columns(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
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

    # Trim obvious outliers by stint: > Q3 + 3*IQR within driver×stint (generous)
    if "stint_id" in d.columns:
        grp = d.groupby(["driver", "stint_id"])
    else:
        grp = d.groupby(["driver"])
    q1 = grp["LapTimeSeconds"].transform("quantile", 0.25)
    q3 = grp["LapTimeSeconds"].transform("quantile", 0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    keep = (d["LapTimeSeconds"] <= (q3 + 3 * iqr).fillna(q3 + 10.0))
    d = d.loc[keep].copy()

    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["LapTimeSeconds", "driver", "team", "compound", "lap_on_tyre", "lap_number", "event"]
    )

    d["driver_team"] = d["driver"].astype(str) + "@" + d["team"].astype(str)
    d["driver_event"] = d["driver"].astype(str) + "-" + d["event"].astype(str)
    return d


def race_metrics_corrections_team(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Robust correction model with event fixed effects + spline controls (+ optional track controls).
    """
    d = _prep_race_columns(race_df)
    if d.empty:
        return pd.DataFrame(columns=["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"])

    df_age = int(cfg.get("race_spline_df_tyre_age", 4))
    df_lap = int(cfg.get("race_spline_df_lap_num", 4))
    df_age_int = int(cfg.get("race_spline_df_tyre_age_interact", 3))
    use_age_interact = bool(cfg.get("race_use_compound_age_interaction", True))

    d = d.copy()
    d["lap_time_s"] = d["LapTimeSeconds"].astype(float)

    base = f"lap_time_s ~ C(event) + C(compound) + bs(lap_on_tyre, df={df_age}) + bs(lap_number, df={df_lap})"
    # Temperature spline (optional)
    df_temp = int(cfg.get("race_spline_df_track_temp", 3))
    use_temp_comp = bool(cfg.get("race_use_temp_compound_interaction", False))
    if (df_temp > 0) and ("track_temp_c_filled" in d.columns) and d["track_temp_c_filled"].notna().any():
        base += f" + bs(track_temp_c_filled, df={df_temp})"
        if use_temp_comp:
            base += f" + C(compound):bs(track_temp_c_filled, df={df_temp})"

    if use_age_interact:
        base += f" + C(compound):bs(lap_on_tyre, df={df_age_int})"

    # Optional track controls
    formula = _append_track_controls_to_formula(base, d, cfg)

    m = smf.ols(formula, data=d).fit(cov_type="HC3")

    d["pred"] = m.predict(d)
    intercept = float(m.params.get("Intercept", 0.0))
    d["norm_time"] = d["lap_time_s"] - (d["pred"] - intercept)

    d["team_mean"] = d.groupby("team", dropna=False)["norm_time"].transform("mean")
    d["demeaned"] = d["norm_time"] - d["team_mean"]

    grp = d.groupby(["driver", "team"], dropna=False)
    mean_demeaned = grp["demeaned"].mean().rename("race_delta_s")
    n_by_driver = grp.size().rename("race_n")

    res = d["demeaned"] - d.groupby(["driver", "team"], dropna=False)["demeaned"].transform("mean")
    se_by_driver = res.groupby([d["driver"], d["team"]]).std() / np.sqrt(
        res.groupby([d["driver"], d["team"]]).count().clip(lower=1)
    )

    out = mean_demeaned.reset_index()
    out["race_se_s"] = out.set_index(["driver", "team"]).index.map(se_by_driver)
    out["race_n"] = out.set_index(["driver", "team"]).index.map(n_by_driver).astype(int).values
    out["race_model"] = "corrections_team(fe+spline" + (",track" if " + " in formula and formula != base else "") + ")"
    return out[["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"]].sort_values("race_delta_s").reset_index(drop=True)


def race_metrics_ols_team(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Driver-within-team OLS with event FE + non-linear controls and cluster-robust SEs (+ optional track controls).
    """
    d = _prep_race_columns(race_df)
    if d.empty:
        return pd.DataFrame(columns=["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"])

    df_age = int(cfg.get("race_spline_df_tyre_age", 4))
    df_lap = int(cfg.get("race_spline_df_lap_num", 4))

    d = d.copy()
    d["lap_time_s"] = d["LapTimeSeconds"].astype(float)

    base = (
        f"lap_time_s ~ C(event) + C(team) + C(driver_team)"
        + f" + bs(lap_on_tyre, df={df_age}) + bs(lap_number, df={df_lap})"
        + " + C(compound)"
    )

    # Temperature spline (optional)
    df_temp = int(cfg.get("race_spline_df_track_temp", 3))
    use_temp_comp = bool(cfg.get("race_use_temp_compound_interaction", False))
    if (df_temp > 0) and ("track_temp_c_filled" in d.columns) and d["track_temp_c_filled"].notna().any():
        base += f" + bs(track_temp_c_filled, df={df_temp})"
        if use_temp_comp:
            base += f" + C(compound):bs(track_temp_c_filled, df={df_temp})"

    # Optional track controls
    formula = _append_track_controls_to_formula(base, d, cfg)

    d["cluster_id"] = d["driver_event"].astype(str)
    m = smf.ols(formula, data=d).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["cluster_id"]},
        use_t=True,
    )

    intercept = float(m.params.get("Intercept", 0.0))
    d["pred"] = m.predict(d)
    d["norm_time"] = d["lap_time_s"] - (d["pred"] - intercept)

    d["team_mean"] = d.groupby("team", dropna=False)["norm_time"].transform("mean")
    d["demeaned"] = d["norm_time"] - d["team_mean"]

    grp = d.groupby(["driver", "team"], dropna=False)
    mean_demeaned = grp["demeaned"].mean().rename("race_delta_s")
    n_by_driver = grp.size().rename("race_n")

    res = d["demeaned"] - d.groupby(["driver", "team"], dropna=False)["demeaned"].transform("mean")
    se_by_driver = res.groupby([d["driver"], d["team"]]).std() / np.sqrt(
        res.groupby([d["driver"], d["team"]]).count().clip(lower=1)
    )

    out = mean_demeaned.reset_index()
    out["race_se_s"] = out.set_index(["driver", "team"]).index.map(se_by_driver)
    out["race_n"] = out.set_index(["driver", "team"]).index.map(n_by_driver).astype(int).values
    out["race_model"] = "ols_team(fe+spline,clustered" + (",track" if " + " in formula and formula != base else "") + ")"
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
    Evolution-aware quali metric (normalize within Q1/Q2/Q3, precision-weight per segment).
    Track effects are NOT used here (normalization already handles session evolution); we may carry tags only.
    """
    out_cols = ["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"]

    if quali_df is None or len(quali_df) == 0:
        return pd.DataFrame(columns=out_cols)

    d = quali_df.copy()
    d = _ensure_driver_column(d)
    d = _ensure_team_column(d)
    d = _ensure_session_column(d)

    d["LapTimeSeconds"] = pd.to_numeric(d.get("LapTimeSeconds", d.get("LapTime")), errors="coerce")
    if "lap_ok" in d.columns:
        d = d[d["lap_ok"].astype(bool)]
    if "is_valid" in d.columns:
        d = d[d["is_valid"].astype(bool)]
    d = d[np.isfinite(d["LapTimeSeconds"])]
    if d.empty:
        return pd.DataFrame(columns=out_cols)

    # 1) Segment medians & normalization
    seg_med = d.groupby("session")["LapTimeSeconds"].transform("median")
    d["norm_lt"] = d["LapTimeSeconds"] - seg_med

    # 2) Winsorization
    use_winsor = bool(cfg.get("quali_winsorize", True))
    q_low = float(cfg.get("quali_winsor_lower_q", 0.00))
    q_high = float(cfg.get("quali_winsor_upper_q", 0.05))
    if use_winsor:
        d["norm_lt"] = (
            d.groupby("session", group_keys=False)["norm_lt"]
            .apply(lambda s: _winsorize_series(s, q_low, q_high))
        )

    # 3) Optional top-k after normalization
    use_topk = bool(cfg.get("quali_use_topk_after_norm", False))
    k = int(cfg.get("quali_top_k", 3))
    if use_topk:
        d = d.sort_values("norm_lt").groupby(["driver", "team", "session"], as_index=False).head(k)

    # 4) Driver-session best lap
    drv_sess = (
        d.groupby(["driver", "team", "session"], as_index=False)
        .agg(best_norm_lt=("norm_lt", "min"), laps_n=("norm_lt", "size"), laps_sd=("norm_lt", "std"))
    )
    drv_sess["var_ds"] = (drv_sess["laps_sd"].fillna(0.0) ** 2) / drv_sess["laps_n"].clip(lower=1)

    # 5) Team best per segment
    team_best = (
        drv_sess.groupby(["team", "session"])["best_norm_lt"]
        .min()
        .rename("team_best_norm")
        .reset_index()
    )
    g = drv_sess.merge(team_best, on=["team", "session"], how="left")
    g["gap_s"] = g["best_norm_lt"] - g["team_best_norm"]

    # 6) Precision-weighted combine across segments
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

    agg = g.groupby(["driver", "team"]).apply(_combine_driver).reset_index()
    if agg.empty:
        return pd.DataFrame(columns=out_cols)

    return agg[out_cols].sort_values("quali_delta_s").reset_index(drop=True)


# ============================================================
# ================== EVENT-LEVEL COMBINATION =================
# ============================================================
def combine_event_metrics(
    race_df: Optional[pd.DataFrame],
    quali_df: Optional[pd.DataFrame],
    wR: float = 0.6,   # kept for backward compatibility; ignored by precision-weighting
    wQ: float = 0.4,   # kept for backward compatibility; ignored by precision-weighting
    apply_bayes_shrinkage: bool = True,
) -> pd.DataFrame:
    """
    Precision-weighted event combination:

      delta_event = (dR/sR^2 + dQ/sQ^2) / (1/sR^2 + 1/sQ^2)
      se_event    = sqrt( 1 / (1/sR^2 + 1/sQ^2) )

    Robust to missing inputs:
      - If quali is missing/empty -> event == race
      - If race is missing/empty  -> event == quali
    """
    # Coalesce Nones/empties into typed empty frames with join keys
    if race_df is None or len(race_df) == 0:
        race_df = pd.DataFrame(columns=["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"])
    if quali_df is None or len(quali_df) == 0:
        quali_df = pd.DataFrame(columns=["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"])

    # Outer-merge on driver/team so we can fall back to whichever side exists
    m = pd.merge(race_df, quali_df, on=["driver", "team"], how="outer")

    # Normalize numeric types
    for col in ["race_delta_s", "quali_delta_s", "race_se_s", "quali_se_s"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    def _combine_row(r: pd.Series) -> pd.Series:
        dR, sR = r.get("race_delta_s"), r.get("race_se_s")
        dQ, sQ = r.get("quali_delta_s"), r.get("quali_se_s")

        validR = (pd.notna(dR) and pd.notna(sR) and float(sR) > 0.0)
        validQ = (pd.notna(dQ) and pd.notna(sQ) and float(sQ) > 0.0)

        if validR and validQ:
            wR_eff = 1.0 / (float(sR) ** 2)
            wQ_eff = 1.0 / (float(sQ) ** 2)
            denom = wR_eff + wQ_eff
            if denom <= 0 or not np.isfinite(denom):
                return pd.Series(
                    {"event_delta_s": np.nan, "event_se_s": np.nan, "event_wR_eff": np.nan, "event_wQ_eff": np.nan}
                )
            delta = (float(dR) * wR_eff + float(dQ) * wQ_eff) / denom
            se = math.sqrt(1.0 / denom)
            return pd.Series(
                {"event_delta_s": float(delta), "event_se_s": float(se), "event_wR_eff": float(wR_eff / denom), "event_wQ_eff": float(wQ_eff / denom)}
            )
        elif validR:
            return pd.Series({"event_delta_s": float(dR), "event_se_s": float(sR), "event_wR_eff": 1.0, "event_wQ_eff": 0.0})
        elif validQ:
            return pd.Series({"event_delta_s": float(dQ), "event_se_s": float(sQ), "event_wR_eff": 0.0, "event_wQ_eff": 1.0})
        else:
            return pd.Series({"event_delta_s": np.nan, "event_se_s": np.nan, "event_wR_eff": np.nan, "event_wQ_eff": np.nan})

    comb = m.apply(_combine_row, axis=1)
    m = pd.concat([m, comb], axis=1)

    # Optional EB shrinkage on each layer (race/quali/event)
    if apply_bayes_shrinkage:
        for col_delta, col_se, out_col, w_col in [
            ("race_delta_s", "race_se_s", "race_delta_s_shrunk", "race_shrink_w"),
            ("quali_delta_s", "quali_se_s", "quali_delta_s_shrunk", "quali_shrink_w"),
            ("event_delta_s", "event_se_s", "event_delta_s_shrunk", "event_shrink_w"),
        ]:
            if col_delta in m.columns and col_se in m.columns:
                shrunk, w = _empirical_bayes_shrinkage(m[col_delta], m[col_se])  # back-compat path
                m[out_col] = shrunk
                m[w_col] = w

    return m


# ============================================================
# ===================== ORCHESTRATOR =========================
# ============================================================
def _unpack_clean_payload(
    result: Union[Tuple, Dict, Any]
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
    raise ValueError("clean_event_payload returned an unsupported structure")


def compute_event_metrics(event: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    For a single event: compute race (within-team), quali (within-team),
    combine them into per-event deltas, and apply optional EB shrinkage using cfg.
    """
    dR_clean, r_summary, dQ_clean, q_summary = _unpack_clean_payload(clean_event_payload(event, cfg))

    # Attach track tags (no-op if metadata missing)
    dR_clean = _attach_event_track_tags(dR_clean, event, cfg)
    # --- Temperature covariate prep (fill NaNs with event median) ---
    dR_clean = dR_clean.copy()
    if "track_temp_c" not in dR_clean.columns:
        dR_clean["track_temp_c"] = np.nan
    dR_clean["track_temp_c_filled"] = pd.to_numeric(dR_clean["track_temp_c"], errors="coerce")

    # Prefer event/race summaries; fall back to in-sample median
    med_temp = np.nan
    try:
        med_temp = float(
            pd.to_numeric((event.get("weather_summary") or {}).get("median_track_temp_c"), errors="coerce"))
    except Exception:
        pass
    if not np.isfinite(med_temp):
        try:
            med_temp = float(pd.to_numeric((r_summary or {}).get("median_track_temp_c"), errors="coerce"))
        except Exception:
            pass
    if not np.isfinite(med_temp):
        med_temp = float(dR_clean["track_temp_c_filled"].median())

    dR_clean["track_temp_c_filled"] = dR_clean["track_temp_c_filled"].fillna(med_temp)
    if dQ_clean is not None:
        dQ_clean = _attach_event_track_tags(dQ_clean, event, cfg)

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
                        d_tmp.loc[
                            (d_tmp["driver"] == r["driver"]) & (d_tmp["team"] == r["team"]),
                            "demeaned",
                        ].to_numpy()
                        - float(mean_.loc[(r["driver"], r["team"])]),
                        int(n_.loc[(r["driver"], r["team"])]),
                    ),
                    axis=1,
                ),
                race_n=lambda x: x.apply(lambda r: int(n_.loc[(r["driver"], r["team"])]), axis=1),
                race_model="raw_team",
            )
        )[["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"]]

    quali_out = (
        quali_metrics_within_team(dQ_clean, cfg)
        if dQ_clean is not None
        else pd.DataFrame(columns=["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"])
    )

    # Precision-weighted combination (no shrinkage here; we’ll do it after with proper cfg)
    wR = float(cfg.get("wR", 0.6))  # ignored by precision weighting; kept for back-compat signature
    wQ = float(cfg.get("wQ", 0.4))
    merged = combine_event_metrics(race_out, quali_out, wR=wR, wQ=wQ, apply_bayes_shrinkage=False)

    # Apply shrinkage (EB by default; HB if toggled)
    if bool(cfg.get("apply_bayes_shrinkage", True)):
        use_hb = bool(cfg.get("use_hierarchical_shrinkage", False))
        for (col_delta, col_se, out_col, w_col, extra_col) in [
            ("race_delta_s", "race_se_s", "race_delta_s_shrunk", "race_shrink_w", "race_post_sd"),
            ("quali_delta_s", "quali_se_s", "quali_delta_s_shrunk", "quali_shrink_w", "quali_post_sd"),
            ("event_delta_s", "event_se_s", "event_delta_s_shrunk", "event_shrink_w", "event_post_sd"),
        ]:
            if col_delta in merged.columns and col_se in merged.columns:
                if use_hb:
                    shrunk, w, post_sd, meta_hb = hierarchical_shrink(
                        merged[["driver", "team", col_delta, col_se]].rename(columns={
                            col_delta: "delta", col_se: "se"
                        }),
                        delta_col="delta",
                        se_col="se",
                        team_col="team",
                        driver_col="driver",
                    )
                    merged[out_col] = shrunk
                    merged[w_col] = w
                    merged[extra_col] = post_sd
                    # (Optionally stash meta_hb['tau_team2'], meta_hb['tau_driver2'] if you want)
                else:
                    shrunk, w, tau2 = _empirical_bayes_shrinkage_smart(
                        merged[col_delta], merged[col_se], merged.get("team"), cfg
                    )
                    merged[out_col] = shrunk
                    merged[w_col] = w
                    # Keep EB's hyperparameter for reference; no post_sd for EB
                    tau_name = out_col.replace("_delta_s_shrunk", "") + "_tau2"
                    merged[tau_name] = float(tau2)

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
        # Ensure track tags are attached before modeling
        ev = dict(ev)  # shallow copy just in case
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
        print(
            f"[INFO] {meta['year']} {meta['gp']}: metrics computed "
            f"(drivers={df['driver'].nunique()}, race_n={nR}, quali_k={nQ})"
        )

    if all_rows:
        combined = pd.concat(all_rows, axis=0, ignore_index=True)
        combined.to_csv(outdir / "all_events_metrics.csv", index=False)
        print(f"[INFO] Wrote combined metrics to: {outdir / 'all_events_metrics.csv'}")
    else:
        print("[WARN] No events available for metrics.")


if __name__ == "__main__":
    main()
