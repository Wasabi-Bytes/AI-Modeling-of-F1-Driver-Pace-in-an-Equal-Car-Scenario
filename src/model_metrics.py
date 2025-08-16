# src/model_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
import math

import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import LinearRegression
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


def _ohe() -> OneHotEncoder:
    # Compat: scikit-learn <1.4 uses `sparse`, >=1.4 prefers `sparse_output`
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------- Race metrics (OLS) ----------
def race_metrics_ols(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    d = race_df.copy()

    needed_cols = ["LapTimeSeconds", "driver", "compound", "lap_on_tyre", "lap_number"]
    for c in needed_cols:
        if c not in d.columns:
            raise ValueError(f"[race_metrics_ols] Missing required column: {c}")

    d["driver"] = d["driver"].astype(str)
    d["compound"] = d["compound"].astype(str).str.upper()
    d["lap_on_tyre"] = pd.to_numeric(d["lap_on_tyre"], errors="coerce")
    d["lap_number"] = pd.to_numeric(d["lap_number"], errors="coerce")
    d["LapTimeSeconds"] = pd.to_numeric(d["LapTimeSeconds"], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["LapTimeSeconds", "driver", "compound", "lap_on_tyre", "lap_number"])
    if d.empty:
        return pd.DataFrame(columns=["driver", "race_delta_s", "race_se_s", "race_n"])

    tyre_ref, L_mid = _get_ref_values(cfg, d)
    ref_age = float(cfg.get("tyre_ref_age", 3))

    ohe = _ohe()
    cat = d[["driver", "compound"]]
    X_cat = ohe.fit_transform(cat)

    X_num = np.column_stack([
        pd.to_numeric(d["lap_on_tyre"], errors="coerce").to_numpy(),
        (pd.to_numeric(d["lap_number"], errors="coerce").to_numpy() - L_mid),
    ])

    X = np.column_stack([X_cat, X_num])
    y = d["LapTimeSeconds"].to_numpy()

    lin = LinearRegression()
    lin.fit(X, y)

    drivers = sorted(d["driver"].unique())
    compounds = sorted(d["compound"].unique())
    cat_all = pd.DataFrame(
        [(drv, cmpd) for drv in drivers for cmpd in compounds],
        columns=["driver", "compound"]
    )
    X_cat_all = ohe.transform(cat_all)

    ref_mask = (cat_all["compound"].str.upper() == tyre_ref)
    if not ref_mask.any():
        # Fallback: most common compound in data
        tyre_ref = d["compound"].value_counts().idxmax()
        ref_mask = (cat_all["compound"].str.upper() == str(tyre_ref).upper())

    X_cat_ref = X_cat_all[ref_mask.values, :]

    X_num_ref = np.column_stack([
        np.full((X_cat_ref.shape[0],), ref_age, dtype=float),
        np.full((X_cat_ref.shape[0],), 0.0, dtype=float),  # centered at L_mid
    ])

    X_ref = np.column_stack([X_cat_ref, X_num_ref])
    preds = lin.predict(X_ref)

    drv_for_ref = cat_all.loc[ref_mask, "driver"].tolist()
    pred_by_driver = pd.Series(preds, index=drv_for_ref, name="pred_ref_s")

    best = float(pred_by_driver.min())
    deltas = pred_by_driver - best

    resid_all = y - lin.predict(X)
    n_by_driver = _driver_counts(d)
    se_by_driver = []
    for drv in drivers:
        idx = (d["driver"] == drv).to_numpy()
        se_by_driver.append(_se_from_residuals(resid_all[idx], int(n_by_driver.get(drv, 0))))

    out = pd.DataFrame({
        "driver": drivers,
        "race_delta_s": [float(deltas.get(drv, np.nan)) for drv in drivers],
        "race_se_s": se_by_driver,
        "race_n": [int(n_by_driver.get(drv, 0)) for drv in drivers],
        "race_model": "ols",
        "race_ref_compound": str(tyre_ref),
        "race_ref_age": ref_age,
        "race_ref_lap_center": L_mid,
    })
    return out.sort_values("race_delta_s").reset_index(drop=True)


# ---------- Race metrics (Corrections) ----------
def race_metrics_corrections(race_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    d = race_df.copy()
    needed_cols = ["LapTimeSeconds", "driver", "compound", "lap_on_tyre", "lap_number"]
    for c in needed_cols:
        if c not in d.columns:
            raise ValueError(f"[race_metrics_corrections] Missing required column: {c}")

    d["driver"] = d["driver"].astype(str)
    d["compound"] = d["compound"].astype(str).str.upper()
    d["lap_on_tyre"] = pd.to_numeric(d["lap_on_tyre"], errors="coerce")
    d["lap_number"] = pd.to_numeric(d["lap_number"], errors="coerce")
    d["LapTimeSeconds"] = pd.to_numeric(d["LapTimeSeconds"], errors="coerce")

    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["LapTimeSeconds", "driver", "compound", "lap_on_tyre", "lap_number"])
    if d.empty:
        return pd.DataFrame(columns=["driver", "race_delta_s", "race_se_s", "race_n"])

    tyre_ref, L_mid = _get_ref_values(cfg, d)
    ref_age = float(cfg.get("tyre_ref_age", 3.0))

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

    # coefficients
    beta = lin.coef_
    n_comp = Xc.shape[1]
    beta_comp = beta[:n_comp]
    beta_age = beta[n_comp + 0]
    beta_fuel = beta[n_comp + 1]

    # reference compound row
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
    norm_time = d["LapTimeSeconds"].to_numpy() - correction
    d["norm_time"] = norm_time

    grp = d.groupby("driver", dropna=False)
    mean_norm = grp["norm_time"].mean().rename("avg_norm_s")
    n_by_driver = grp.size().rename("n")

    best = float(mean_norm.min())
    deltas = mean_norm - best

    se_by_driver = []
    for drv, sub in grp:
        resid = sub["norm_time"].to_numpy() - float(mean_norm.loc[drv])
        se_by_driver.append(_se_from_residuals(resid, int(n_by_driver.loc[drv])))

    out = pd.DataFrame({
        "driver": mean_norm.index.astype(str),
        "race_delta_s": [float(deltas.loc[drv]) for drv in mean_norm.index],
        "race_se_s": se_by_driver,
        "race_n": [int(n_by_driver.loc[drv]) for drv in mean_norm.index],
        "race_model": "corrections",
        "race_ref_compound": str(tyre_ref),
        "race_ref_age": ref_age,
        "race_ref_lap_center": L_mid,
    }).sort_values("race_delta_s").reset_index(drop=True)
    return out


# ---------- Qualifying metrics ----------
def quali_metrics(quali_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if quali_df is None or len(quali_df) == 0:
        return pd.DataFrame(columns=["driver", "quali_delta_s", "quali_se_s", "quali_n"])

    d = quali_df.copy()
    if "driver" not in d.columns:
        if "Driver" in d.columns:
            d["driver"] = d["Driver"].astype(str)
        elif "DriverNumber" in d.columns:
            d["driver"] = d["DriverNumber"].astype(str)
        else:
            d["driver"] = "UNK"

    d["LapTimeSeconds"] = pd.to_numeric(d.get("LapTimeSeconds", d.get("LapTime")), errors="coerce")
    d["lap_ok"] = d.get("lap_ok", True)

    if "Compound" in d.columns:
        comp = d["Compound"].astype(str).str.upper()
        soft_mask = comp.str.startswith("S")
        if soft_mask.any():
            d = d[soft_mask].copy()

    d = d[(d["lap_ok"].astype(bool)) & np.isfinite(d["LapTimeSeconds"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["driver", "quali_delta_s", "quali_se_s", "quali_n"])

    k = int(cfg.get("quali_top_k", 3))

    rows = []
    for drv, sub in d.groupby("driver"):
        sub_s = sub.sort_values("LapTimeSeconds", ascending=True)
        topk = sub_s.head(k).copy()
        if len(topk) == 0:
            continue
        med = float(np.median(topk["LapTimeSeconds"].to_numpy()))
        sd = float(np.std(topk["LapTimeSeconds"].to_numpy(), ddof=1)) if len(topk) > 1 else float("nan")
        se = sd / math.sqrt(min(len(topk), k))
        rows.append({"driver": str(drv), "med_topk_s": med, "se_topk_s": se, "quali_n": int(len(sub))})
    if not rows:
        return pd.DataFrame(columns=["driver", "quali_delta_s", "quali_se_s", "quali_n"])

    q = pd.DataFrame(rows)
    best = float(q["med_topk_s"].min())
    q["quali_delta_s"] = q["med_topk_s"] - best
    q = q.rename(columns={"se_topk_s": "quali_se_s"})
    return q[["driver", "quali_delta_s", "quali_se_s", "quali_n"]].sort_values("quali_delta_s").reset_index(drop=True)


# ---------- Orchestrator per event ----------
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
    dR_clean, r_summary, dQ_clean, q_summary = _unpack_clean_payload(clean_event_payload(event, cfg))

    model_choice = str(cfg.get("race_model", "ols")).lower()
    if model_choice == "ols":
        race_out = race_metrics_ols(dR_clean, cfg)
    elif model_choice == "corrections":
        race_out = race_metrics_corrections(dR_clean, cfg)
    else:
        grp = dR_clean.groupby("driver", dropna=False)["LapTimeSeconds"]
        mean_ = grp.mean()
        n_ = grp.size()
        best = float(mean_.min())
        race_out = pd.DataFrame({
            "driver": mean_.index.astype(str),
            "race_delta_s": [float(mean_.loc[i] - best) for i in mean_.index],
            "race_se_s": [float(np.std(dR_clean.loc[dR_clean["driver"] == i, "LapTimeSeconds"], ddof=1) /
                                math.sqrt(max(int(n_.loc[i]), 1))) for i in mean_.index],
            "race_n": [int(n_.loc[i]) for i in mean_.index],
            "race_model": "raw",
            "race_ref_compound": "",
            "race_ref_age": np.nan,
            "race_ref_lap_center": np.nan,
        }).sort_values("race_delta_s").reset_index(drop=True)

    quali_out = quali_metrics(dQ_clean, cfg) if dQ_clean is not None else pd.DataFrame(
        columns=["driver", "quali_delta_s", "quali_se_s", "quali_n"]
    )

    merged = pd.merge(race_out, quali_out, on="driver", how="outer")

    meta = {
        "year": event["year"],
        "gp": event["gp"],
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
        nQ = int(res["quali_only"].get("quali_n", pd.Series(dtype=int)).sum()) if not res["quali_only"].empty else 0
        print(f"[INFO] {meta['year']} {meta['gp']}: metrics computed "
              f"(drivers={df['driver'].nunique()}, race_n={nR}, quali_n={nQ})")

    if all_rows:
        combined = pd.concat(all_rows, axis=0, ignore_index=True)
        combined.to_csv(outdir / "all_events_metrics.csv", index=False)
        print(f"[INFO] Wrote combined metrics to: {outdir / 'all_events_metrics.csv'}")
    else:
        print("[WARN] No events available for metrics.")


if __name__ == "__main__":
    main()
