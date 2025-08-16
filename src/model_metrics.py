# src/model_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import warnings
import math

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder

# Local modules
from load_data import load_config, load_all_data
from clean_data import clean_event_payload  # uses your existing cleaner

# Silence noisy future warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")


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


def _ohe(sparse_ok: bool = False) -> OneHotEncoder:
    # Compat: scikit-learn <1.4 uses `sparse`, >=1.4 prefers `sparse_output`
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_ok)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse_ok)


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
    # last resort: if no column exists, attempt to infer a single team per driver (mode of any available hint)
    if "TeamColor" in d.columns:
        tmp = d.groupby("driver")["TeamColor"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else "UNK")
        d = d.merge(tmp.rename("team"), left_on="driver", right_index=True, how="left")
        d["team"] = d["team"].fillna("UNK").astype(str)
        return d
    raise ValueError("[race/quali] No team column found; please add one in clean_data.py or provide a standard team field.")

def _ensure_session_column(d: pd.DataFrame) -> pd.DataFrame:
    if "session" in d.columns:
        d["session"] = d["session"].astype(str)
        return d
    for c in _SESSION_COL_CANDIDATES:
        if c in d.columns:
            d["session"] = d[c].astype(str)
            return d
    # default single-session if not provided
    d["session"] = "Q"
    return d


# ============================================================
# =============== RACE METRICS (WITHIN TEAM) =================
# ============================================================

def _prep_race_columns(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d = _ensure_driver_column(d)
    d = _ensure_team_column(d)

    needed_cols = ["LapTimeSeconds", "driver", "team", "compound", "lap_on_tyre", "lap_number"]
    for c in needed_cols:
        if c not in d.columns:
            raise ValueError(f"[race_metrics] Missing required column: {c}")

    d["compound"] = d["compound"].astype(str).str.upper()
    d["lap_on_tyre"] = pd.to_numeric(d["lap_on_tyre"], errors="coerce")
    d["lap_number"] = pd.to_numeric(d["lap_number"], errors="coerce")
    d["LapTimeSeconds"] = pd.to_numeric(d["LapTimeSeconds"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["LapTimeSeconds", "driver", "team", "compound", "lap_on_tyre", "lap_number"])
    return d


def race_metrics_corrections_team(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Tyre/fuel/age corrections, followed by team demeaning (removes car layer).
    Produces per-driver, per-event race deltas (lower = faster).
    """
    d = _prep_race_columns(race_df)
    if d.empty:
        return pd.DataFrame(columns=["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"])

    tyre_ref, L_mid = _get_ref_values(cfg, d)
    ref_age = float(cfg.get("tyre_ref_age", 3.0))

    # Learn correction factors: compound + linear age + centered lap_number
    ohe = _ohe()
    Xc = ohe.fit_transform(d[["compound"]])
    Xn = np.column_stack([
        d["lap_on_tyre"].to_numpy(),
        (d["lap_number"].to_numpy() - L_mid),
    ])
    X = np.column_stack([Xc, Xn])
    y = d["LapTimeSeconds"].to_numpy()

    lin = LinearRegression()
    lin.fit(X, y)

    beta = lin.coef_
    n_comp = Xc.shape[1]
    beta_comp = beta[:n_comp]
    beta_age = beta[n_comp + 0]
    beta_fuel = beta[n_comp + 1]

    try:
        Xc_ref = ohe.transform(pd.DataFrame({"compound": [tyre_ref]}))
        if Xc_ref.shape[1] != Xc.shape[1]:
            Xc_ref = np.zeros((1, Xc.shape[1]), dtype=float)
    except Exception:
        Xc_ref = np.zeros((1, Xc.shape[1]), dtype=float)

    comp_effect_actual = (Xc * beta_comp).sum(axis=1)
    comp_effect_ref = float((Xc_ref * beta_comp).sum(axis=1))
    age_effect_actual = beta_age * d["lap_on_tyre"].to_numpy()
    age_effect_ref = beta_age * ref_age
    fuel_effect_actual = beta_fuel * (d["lap_number"].to_numpy() - L_mid)
    fuel_effect_ref = 0.0

    correction = (comp_effect_actual - comp_effect_ref) + (age_effect_actual - age_effect_ref) + (fuel_effect_actual - fuel_effect_ref)
    d["norm_time"] = d["LapTimeSeconds"].to_numpy() - correction

    # Team demean
    d["team_mean"] = d.groupby("team", dropna=False)["norm_time"].transform("mean")
    d["demeaned"] = d["norm_time"] - d["team_mean"]

    # Per-driver stats
    grp = d.groupby(["driver", "team"], dropna=False)
    mean_demeaned = grp["demeaned"].mean().rename("race_delta_s")
    n_by_driver = grp.size().rename("race_n")

    se_by_driver = []
    for (drv, tm), sub in grp:
        resid = sub["demeaned"].to_numpy() - float(mean_demeaned.loc[(drv, tm)])
        se_by_driver.append(_se_from_residuals(resid, int(n_by_driver.loc[(drv, tm)])))

    out = mean_demeaned.reset_index()
    out["race_se_s"] = se_by_driver
    out["race_model"] = "corrections_team"
    return out[["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"]].sort_values("race_delta_s").reset_index(drop=True)


def race_metrics_ols_team(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Team-controlled linear model:
      LapTime ~ C(team) + C(driver_within_team) + controls
    Coefficients for driver_within_team are within-team deltas.
    """
    d = _prep_race_columns(race_df)
    if d.empty:
        return pd.DataFrame(columns=["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"])

    tyre_ref, L_mid = _get_ref_values(cfg, d)
    ref_age = float(cfg.get("tyre_ref_age", 3.0))

    # Build categorical features
    d["drv_team"] = d["driver"].astype(str) + "@" + d["team"].astype(str)

    ohe_team = _ohe()
    ohe_drvteam = _ohe()
    X_team = ohe_team.fit_transform(d[["team"]])
    X_drvteam = ohe_drvteam.fit_transform(d[["drv_team"]])

    # Controls
    X_num = np.column_stack([
        pd.to_numeric(d["lap_on_tyre"], errors="coerce").to_numpy(),
        (pd.to_numeric(d["lap_number"], errors="coerce").to_numpy() - L_mid),
    ])

    # Full design (team + driver-within-team + controls)
    X = np.column_stack([X_team, X_drvteam, X_num])
    y = d["LapTimeSeconds"].to_numpy()

    # Choose regressor (OLS or Ridge)
    model_name = "ols_team"
    reg = LinearRegression()
    if str(cfg.get("race_model", "ols_team")).lower() == "ridge_team":
        alpha = float(cfg.get("regularization_alpha", 1.0))
        reg = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
        model_name = "ridge_team"

    reg.fit(X, y)

    # Extract driver-within-team effects by predicting at reference controls
    drvteam_values = sorted(d["drv_team"].unique())
    X_drvteam_all = ohe_drvteam.transform(pd.DataFrame({"drv_team": drvteam_values}))

    # Null out team terms and use reference controls for numeric.
    X_team_zero = np.zeros((X_drvteam_all.shape[0], X_team.shape[1]), dtype=float)
    X_num_ref = np.column_stack([
        np.full((X_drvteam_all.shape[0],), ref_age, dtype=float),
        np.full((X_drvteam_all.shape[0],), 0.0, dtype=float),  # centered at L_mid
    ])

    X_ref = np.column_stack([X_team_zero, X_drvteam_all, X_num_ref])
    preds = reg.predict(X_ref)

    # Convert drv_team back to (driver, team)
    df_preds = pd.DataFrame({"drv_team": drvteam_values, "pred_ref_s": preds})
    df_preds[["driver", "team"]] = df_preds["drv_team"].str.split("@", n=1, expand=True)

    # Within-team deltas
    df_preds["team_best"] = df_preds.groupby("team")["pred_ref_s"].transform("min")
    df_preds["race_delta_s"] = df_preds["pred_ref_s"] - df_preds["team_best"]

    # SE per driver from residuals
    resid_all = y - reg.predict(X)
    n_by_drvteam = d.groupby(["driver", "team"], dropna=False).size()

    se_list = []
    for _, row in df_preds.iterrows():
        mask = (d["driver"] == row["driver"]) & (d["team"] == row["team"])
        se_list.append(_se_from_residuals(resid_all[mask.to_numpy()], int(n_by_drvteam.get((row["driver"], row["team"]), 0))))
    df_preds["race_se_s"] = se_list
    df_preds["race_n"] = [int(n_by_drvteam.get((r["driver"], r["team"]), 0)) for _, r in df_preds.iterrows()]
    df_preds["race_model"] = model_name

    return df_preds[["driver", "team", "race_delta_s", "race_se_s", "race_n", "race_model"]].sort_values("race_delta_s").reset_index(drop=True)


# ============================================================
# ================= QUALIFYING (WITHIN TEAM) =================
# ============================================================

def quali_metrics_within_team(quali_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute within-team quali gaps by sub-session (Q1/Q2/Q3),
    using best valid lap per driver per sub-session.
    """
    if quali_df is None or len(quali_df) == 0:
        return pd.DataFrame(columns=["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"])

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
        return pd.DataFrame(columns=["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"])

    # Top-k within each driver/session
    k = int(cfg.get("quali_top_k", 3))
    best_per_session = (
        d.sort_values("LapTimeSeconds")
         .groupby(["driver", "team", "session"], as_index=False)
         .head(k)
    )

    # Take the best (min) per driver/session
    best1 = best_per_session.groupby(["driver", "team", "session"], as_index=False)["LapTimeSeconds"].min()

    # Within team per session: subtract team best
    best1["team_best"] = best1.groupby(["team", "session"])["LapTimeSeconds"].transform("min")
    best1["gap_s"] = best1["LapTimeSeconds"] - best1["team_best"]

    # Aggregate over sessions per driver
    agg = best1.groupby(["driver", "team"], dropna=False).agg(
        quali_delta_s=("gap_s", "mean"),
        quali_k=("session", "nunique"),
        quali_sd=("gap_s", "std"),
    ).reset_index()
    agg["quali_se_s"] = agg.apply(
        lambda r: (r["quali_sd"] / math.sqrt(max(int(r["quali_k"]), 1))) if pd.notna(r["quali_sd"]) else float("nan"),
        axis=1,
    )
    return agg[["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"]].sort_values("quali_delta_s").reset_index(drop=True)


# ============================================================
# ================== EVENT-LEVEL COMBINATION =================
# ============================================================

def combine_event_metrics(
    race_df: pd.DataFrame,
    quali_df: Optional[pd.DataFrame],
    wR: float = 0.6,
    wQ: float = 0.4,
    apply_bayes_shrinkage: bool = True
) -> pd.DataFrame:
    if quali_df is None:
        quali_df = pd.DataFrame(columns=["driver", "team", "quali_delta_s", "quali_se_s", "quali_k"])

    m = pd.merge(race_df, quali_df, on=["driver", "team"], how="outer")

    # If one side missing, use the other and its SE
    for col in ["race_delta_s", "quali_delta_s", "race_se_s", "quali_se_s"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    # Handle missing deltas: renormalize weights per row
    def _combine_row(r):
        hasR = pd.notna(r.get("race_delta_s"))
        hasQ = pd.notna(r.get("quali_delta_s"))
        if hasR and hasQ:
            delta = wR * r["race_delta_s"] + wQ * r["quali_delta_s"]
            se = math.sqrt((wR * (r["race_se_s"] or 0.0)) ** 2 + (wQ * (r["quali_se_s"] or 0.0)) ** 2)
        elif hasR:
            delta, se = float(r["race_delta_s"]), float(r["race_se_s"] or np.nan)
        elif hasQ:
            delta, se = float(r["quali_delta_s"]), float(r["quali_se_s"] or np.nan)
        else:
            delta, se = float("nan"), float("nan")
        return pd.Series({"event_delta_s": delta, "event_se_s": se})

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
        # As a last resort, raw team-demeaned means
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
                        int(n_.loc[(r["driver"], r["team"])])
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

    wR = float(cfg.get("wR", 0.6))
    wQ = float(cfg.get("wQ", 0.4))
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
