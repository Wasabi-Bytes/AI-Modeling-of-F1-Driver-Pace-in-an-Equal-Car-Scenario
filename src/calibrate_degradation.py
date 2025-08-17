# src/calibrate_degradation.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import json
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import bs

from load_data import load_config, enable_cache, load_all_data

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.*")


# ---------- Paths / IO ----------
def _proj() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cfg_get(cfg: dict, path: list[str], default=None):
    d = cfg
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# ---------- Track meta (optional) ----------
def _norm_event_key_str(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(cleaned.split())


def _year_to_str_scalar(v) -> str:
    if v is None:
        return ""
    try:
        # handles floats/ints/strings that look like numbers
        iv = int(pd.to_numeric(v))
        return str(iv)
    except Exception:
        return "" if (isinstance(v, float) and np.isnan(v)) else str(v)


def _gp_to_str_scalar(v) -> str:
    if v is None:
        return ""
    return "" if (isinstance(v, float) and np.isnan(v)) else str(v)


def _norm_event_key(
    year: Union[pd.Series, np.ndarray, list, int, float, str, None],
    gp:   Union[pd.Series, np.ndarray, list, str, None]
) -> Union[pd.Series, str]:
    """
    Accepts scalars OR Series/array-likes for year/gp.
    Returns a normalized key (Series if inputs are vector-like, else str).
    """
    is_vec = isinstance(year, (pd.Series, np.ndarray, list)) or isinstance(gp, (pd.Series, np.ndarray, list))
    if is_vec:
        yser = year if isinstance(year, pd.Series) else pd.Series(year)
        gser = gp   if isinstance(gp, pd.Series)   else pd.Series(gp)
        ytxt = yser.map(_year_to_str_scalar)
        gtxt = gser.map(_gp_to_str_scalar)
        return (ytxt + " " + gtxt).map(_norm_event_key_str)
    else:
        ytxt = _year_to_str_scalar(year)
        gtxt = _gp_to_str_scalar(gp)
        return _norm_event_key_str(f"{ytxt} {gtxt}")


def _load_track_meta(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    meta_path = _cfg_get(cfg, ["paths", "track_meta"], None)
    if not meta_path:
        return None
    f = (_proj() / meta_path).resolve()
    if not f.exists():
        print(f"[INFO] track_meta not found at {f}; calibrating global curves only.")
        return None

    try:
        meta = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Failed to read track_meta: {e}; skipping per-archetype calibration.")
        return None

    cols = {c.lower(): c for c in meta.columns}
    if ("year" in cols) and ("gp" in cols):
        meta["event_key_norm"] = _norm_event_key(meta[cols["year"]], meta[cols["gp"]])
    elif "event_key" in cols:
        meta["event_key_norm"] = meta[cols["event_key"]].map(_norm_event_key_str)
    else:
        return None

    out = pd.DataFrame({"event_key_norm": meta["event_key_norm"].astype(str)})
    out["track_type"] = (meta.get("track_type") or meta.get(cols.get("track_type", ""), np.nan))
    out["downforce_index"] = pd.to_numeric(
        meta.get("downforce_index", meta.get(cols.get("downforce_index", ""), np.nan)), errors="coerce"
    )
    return out


# ---------- Data prep ----------
def _collect_pace_laps(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Use load_all_data(cfg) to get race pace laps (lap_ok == True) with required columns.
    """
    if "cache_dir" in cfg:
        enable_cache(cfg["cache_dir"])

    events = load_all_data(cfg)
    rows = []
    for ev in events:
        dR = ev.get("race_laps")
        if dR is None or len(dR) == 0:
            continue
        tmp = dR.copy()
        # Keep strict pace laps only (the loader already computed lap_ok)
        if "lap_ok" in tmp.columns:
            tmp = tmp[tmp["lap_ok"].astype(bool)].copy()
        # event key for join (vectorized-safe)
        tmp["year"] = ev.get("year")
        tmp["gp"] = ev.get("gp")
        tmp["event_key_norm"] = _norm_event_key(tmp["year"], tmp["gp"])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["LapTimeSeconds", "compound", "lap_on_tyre", "lap_number", "Team", "event_key_norm"])

    d = pd.concat(rows, ignore_index=True)

    # Standardize essential columns
    needed = ["LapTimeSeconds", "compound", "lap_on_tyre", "lap_number"]
    for c in needed:
        if c not in d.columns:
            raise ValueError(f"[calibrate_degradation] Missing column: {c}")

    # Team column for fixed effects
    if "Team" not in d.columns:
        cand = [c for c in d.columns if c.lower() in ("team", "constructor", "teamname", "constructorname")]
        if cand:
            d["Team"] = d[cand[0]].astype(str)
        else:
            d["Team"] = "UNK"

    # Types
    d["LapTimeSeconds"] = pd.to_numeric(d["LapTimeSeconds"], errors="coerce")
    d["compound"] = d["compound"].astype(str).str.upper()
    d["lap_on_tyre"] = pd.to_numeric(d["lap_on_tyre"], errors="coerce")
    d["lap_number"] = pd.to_numeric(d["lap_number"], errors="coerce")

    d = d.dropna(subset=["LapTimeSeconds", "lap_on_tyre", "lap_number"])
    return d


# ---------- Residualization ----------
def _residualize(d: pd.DataFrame, df_lapnum: int = 3) -> pd.DataFrame:
    """
    For each compound separately, remove event FE, team FE, and a smooth fuel/lap_number effect.
    Returns d with an added 'resid' column (per-compound residual).
    """
    out_rows = []
    for comp in sorted(d["compound"].dropna().unique()):
        sub = d[d["compound"] == comp].copy()
        if len(sub) < 200:
            continue
        # OLS: LapTime ~ C(event_key_norm) + C(Team) + bs(lap_number)
        # (we deliberately do NOT include lap_on_tyre here; we want that left in residual)
        formula = f"LapTimeSeconds ~ C(event_key_norm) + C(Team) + bs(lap_number, df={df_lapnum})"
        try:
            m = smf.ols(formula, data=sub).fit(cov_type="HC3")
            sub["resid"] = sub["LapTimeSeconds"] - m.predict(sub)
        except Exception:
            # Robust fallback: subtract event+team means + linear lap_number
            sub["ln"] = sub["lap_number"]
            try:
                m2 = smf.ols("LapTimeSeconds ~ C(event_key_norm) + C(Team) + ln", data=sub).fit()
                sub["resid"] = sub["LapTimeSeconds"] - m2.predict(sub)
            except Exception:
                sub["resid"] = sub["LapTimeSeconds"] - sub.groupby(["event_key_norm", "Team"])["LapTimeSeconds"].transform("mean")
        out_rows.append(sub)

    if not out_rows:
        return pd.DataFrame(columns=d.columns.tolist() + ["resid"])
    return pd.concat(out_rows, ignore_index=True)


# ---------- Piecewise fit (non-negative slopes) ----------
def _fit_piecewise_nonneg(x: np.ndarray, y: np.ndarray, k_min: int = 5, k_max: int = 25) -> Tuple[float, float, int, float]:
    """
    Fit y ≈ e * min(x, k) + l * max(x - k, 0), with e >= 0, l >= 0.
    Grid search over integer switch k; least squares for slopes (clipped to >=0).
    Returns (early_slope, late_slope, switch_k, rmse).
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 30:
        return 0.0, 0.0, max(1, k_min), float("nan")

    x = np.clip(x, 1.0, None)
    k_min = int(max(2, k_min))
    k_max = int(max(k_min + 1, k_max, np.nanpercentile(x, 80)))

    best = (0.0, 0.0, k_min, float("inf"))
    for k in range(k_min, int(k_max) + 1):
        X = np.column_stack([np.minimum(x, k), np.maximum(x - k, 0.0)])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            beta = np.maximum(beta, 0.0)  # non-negativity
            yhat = X @ beta
            rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
            if rmse < best[3]:
                best = (float(beta[0]), float(beta[1]), int(k), rmse)
        except Exception:
            continue
    return best


# ---------- Calibration driver ----------
def calibrate_degradation(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict with:
      {
        "global": { "S": {...}, "M": {...}, "H": {...} },
        "by_track_type": { "street": {...}, "permanent": {...}, ... },
        "meta": {...}
      }
    """
    # 1) Data
    d = _collect_pace_laps(cfg)
    if d.empty:
        raise RuntimeError("No pace laps available for calibration.")

    # 2) Join optional track_meta
    meta = _load_track_meta(cfg)
    if meta is not None:
        d = d.merge(meta[["event_key_norm", "track_type"]], on="event_key_norm", how="left")
    else:
        d["track_type"] = np.nan

    # 3) Residualize per compound
    d = _residualize(d, df_lapnum=3)
    d = d.dropna(subset=["resid", "lap_on_tyre"])

    # Cap extreme lap_on_tyre for stability
    d["lot_cap"] = np.clip(d["lap_on_tyre"].astype(float), 1, 35)

    # Helper to fit one slice
    def _fit_compound_slice(df_slice: pd.DataFrame) -> Dict[str, float]:
        e, l, sw, rmse = _fit_piecewise_nonneg(df_slice["lot_cap"].to_numpy(), df_slice["resid"].to_numpy(),
                                               k_min=5, k_max=25)
        return {"early_slope": float(e), "late_slope": float(l), "switch_lap": int(sw), "rmse": float(rmse)}

    # 4) Global curves per compound
    out_global: Dict[str, Dict[str, float]] = {}
    for comp in ("S", "M", "H"):
        sub = d[d["compound"] == comp]
        if len(sub) >= 200:
            out_global[comp] = _fit_compound_slice(sub)

    # 5) Optional per track_type curves (only when enough data)
    out_by_tt: Dict[str, Dict[str, Dict[str, float]]] = {}
    if d["track_type"].notna().any():
        for tt in sorted([t for t in d["track_type"].dropna().unique()]):
            out_by_tt[tt] = {}
            for comp in ("S", "M", "H"):
                sub = d[(d["track_type"] == tt) & (d["compound"] == comp)]
                if len(sub) >= 150:
                    out_by_tt[tt][comp] = _fit_compound_slice(sub)
            if not out_by_tt[tt]:
                out_by_tt.pop(tt, None)

    meta_info = {
        "method": "piecewise_nonneg",
        "residualization": "C(event)+C(Team)+bs(lap_number,df=3)",
        "min_samples_global": 200,
        "min_samples_tt": 150,
        "n_rows_used": int(len(d)),
    }

    return {"global": out_global, "by_track_type": out_by_tt, "meta": meta_info}


def main():
    cfg = load_config("config/config.yaml")

    # Run calibration
    result = calibrate_degradation(cfg)

    # Where to write params
    out_path = _cfg_get(cfg, ["paths", "degradation_params"], "outputs/calibration/degradation_params.json")
    f = (_proj() / out_path).resolve()
    _ensure_dir(f.parent)

    with open(f, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    print(f"[INFO] Wrote calibrated degradation params to: {f}")


if __name__ == "__main__":
    main()
