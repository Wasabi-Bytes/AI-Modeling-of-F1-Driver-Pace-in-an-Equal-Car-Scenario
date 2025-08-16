# src/clean_data.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import warnings
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore", category=FutureWarning, module="fastf1.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")


# -------- Paths & Config --------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    cfg_path = _project_root() / config_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------- Generic helpers --------
def _col(df: pd.DataFrame, name: str, default=None):
    return df[name] if name in df.columns else default


def _ensure_seconds(df: pd.DataFrame) -> pd.DataFrame:
    if "LapTimeSeconds" in df.columns:
        return df
    d = df.copy()
    if "LapTime" in d.columns:
        def _to_sec(x):
            try:
                return float(getattr(x, "total_seconds", lambda: float(x))())
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return np.nan
        d["LapTimeSeconds"] = d["LapTime"].apply(_to_sec)
    else:
        d["LapTimeSeconds"] = pd.to_numeric(d.get("LapTime", np.nan), errors="coerce")
    return d


def _ensure_driver_col(df: pd.DataFrame) -> pd.DataFrame:
    if "driver" in df.columns:
        return df
    d = df.copy()
    if "Driver" in d.columns:
        d["driver"] = d["Driver"].astype(str)
    elif "DriverNumber" in d.columns:
        d["driver"] = d["DriverNumber"].astype(str)
    else:
        d["driver"] = d.get("DriverNumber", d.get("Driver", "UNK")).astype(str)
    return d


def _bool_series(df: pd.DataFrame, name: str, default_val: bool = True) -> pd.Series:
    if name in df.columns:
        s = df[name]
        if s.dtype != bool:
            s = s.fillna(default_val).astype(bool)
        return s
    return pd.Series(default_val, index=df.index, dtype=bool)


def _track_status_ok(series: pd.Series, exclude_vsc_sc: bool) -> pd.Series:
    if not exclude_vsc_sc:
        return pd.Series(True, index=series.index) if series is not None else pd.Series(True)

    s = (series.astype(str).fillna("")) if series is not None else pd.Series("", index=None)
    ok_numeric = s.isin({"", "0", "1", "AllClear", "ALLCLEAR"})
    sc_like = s.str.contains("SC", case=False, na=False)
    vsc_like = s.str.contains("VSC", case=False, na=False)
    red_like = s.str.contains("RED", case=False, na=False)
    yel_like = s.str.contains("YEL", case=False, na=False)
    return ok_numeric & ~(sc_like | vsc_like | red_like | yel_like)


def _within_bounds(times: pd.Series, lo: float, hi: float) -> pd.Series:
    return times.ge(lo) & times.le(hi)


def _drop_pit_laps(df: pd.DataFrame) -> pd.Series:
    pin = df["PitInTime"].notna() if "PitInTime" in df.columns else pd.Series(False, index=df.index)
    pout = df["PitOutTime"].notna() if "PitOutTime" in df.columns else pd.Series(False, index=df.index)
    pit_out_lap = _bool_series(df, "PitOutLap", False)
    return ~(pin | pout | pit_out_lap)


def _per_driver_trim(df: pd.DataFrame, time_col: str, p_lo: float, p_hi: float) -> pd.Series:
    if "driver" not in df.columns:
        return pd.Series(True, index=df.index)

    def _mask(group: pd.DataFrame) -> pd.Series:
        vals = group[time_col].dropna()
        if len(vals) < 5:
            return pd.Series(True, index=group.index)
        lo = np.percentile(vals, p_lo)
        hi = np.percentile(vals, p_hi)
        return group[time_col].between(lo, hi, inclusive="both")

    return df.groupby("driver", group_keys=False).apply(_mask)


def _summarize_counts(df: pd.DataFrame, driver_col: str, step_name: str) -> pd.DataFrame:
    if driver_col not in df.columns:
        # create a single bucket to avoid crashes (should not happen after _ensure_driver_col)
        tmp = df.copy()
        tmp[driver_col] = "UNK"
        df = tmp
    g = df.groupby(driver_col).size().rename("n")
    out = g.reset_index()
    out["step"] = step_name
    return out[["driver", "step", "n"]].rename(columns={driver_col: "driver"})


def _combine_summaries(steps: List[pd.DataFrame]) -> pd.DataFrame:
    if not steps:
        return pd.DataFrame(columns=["driver", "step", "n"])
    return pd.concat(steps, ignore_index=True)


# -------- Race cleaning --------
def clean_race_laps(
    laps: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg_bounds = config.get("lap_time_bounds", [60, 180])
    lo, hi = float(cfg_bounds[0]), float(cfg_bounds[1])
    exclude_vsc_sc = bool(config.get("exclude_vsc_sc", True))
    p_lo, p_hi = config.get("trim_percentiles", [5, 95])

    d0 = _ensure_seconds(laps)
    d0 = _ensure_driver_col(d0)

    acc = _bool_series(d0, "IsAccurate", True)
    valid = _bool_series(d0, "LapIsValid", True)
    not_deleted = ~_bool_series(d0, "Deleted", False)
    dA = d0[acc & valid & not_deleted].copy()

    mask_b = _drop_pit_laps(dA)
    dB = dA[mask_b].copy()

    mask_c = _within_bounds(dB["LapTimeSeconds"], lo, hi)
    dC = dB[mask_c].copy()

    track_status_series = dC["track_status"] if "track_status" in dC.columns else _col(dC, "TrackStatus", pd.Series("", index=dC.index))
    mask_d = _track_status_ok(track_status_series, exclude_vsc_sc)
    dD = dC[mask_d].copy()

    mask_e = _per_driver_trim(dD, "LapTimeSeconds", float(p_lo), float(p_hi))
    dE = dD[mask_e].copy()

    s0 = _summarize_counts(d0, "driver", "raw")
    sA = _summarize_counts(dA, "driver", "accurate_valid")
    sB = _summarize_counts(dB, "driver", "no_pits")
    sC = _summarize_counts(dC, "driver", "bounds")
    sD = _summarize_counts(dD, "driver", "no_sc_vsc")
    sE = _summarize_counts(dE, "driver", "trimmed")

    summary = _combine_summaries([s0, sA, sB, sC, sD, sE])
    return dE.reset_index(drop=True), summary.reset_index(drop=True)


# -------- Qualifying cleaning --------
def clean_quali_laps(
    laps: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg_bounds = config.get("lap_time_bounds", [60, 180])
    lo, hi = float(cfg_bounds[0]), float(cfg_bounds[1])
    p_lo, p_hi = config.get("trim_percentiles", [5, 95])
    soft_only = bool(config.get("quali_soft_only", True))

    d0 = _ensure_seconds(laps)
    d0 = _ensure_driver_col(d0)

    acc = _bool_series(d0, "IsAccurate", True)
    valid = _bool_series(d0, "LapIsValid", True)
    not_deleted = ~_bool_series(d0, "Deleted", False)
    dA = d0[acc & valid & not_deleted].copy()

    if soft_only and "Compound" in dA.columns:
        dB = dA[dA["Compound"].astype(str).str.upper().str.startswith("S")].copy()
    else:
        dB = dA

    dC = dB[_within_bounds(dB["LapTimeSeconds"], lo, hi)].copy()

    mask_d = _per_driver_trim(dC, "LapTimeSeconds", float(p_lo), float(p_hi))
    dD = dC[mask_d].copy()

    s0 = _summarize_counts(d0, "driver", "raw")
    sA = _summarize_counts(dA, "driver", "accurate_valid")
    sB = _summarize_counts(dB, "driver", "soft_only" if (soft_only and "Compound" in dA.columns) else "compound_any")
    sC = _summarize_counts(dC, "driver", "bounds")
    sD = _summarize_counts(dD, "driver", "trimmed")

    summary = _combine_summaries([s0, sA, sB, sC, sD])
    return dD.reset_index(drop=True), summary.reset_index(drop=True)


# -------- Per-event wrapper --------
def clean_event_payload(event: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(event)

    race_clean, race_summary = clean_race_laps(event["race_laps"], config)
    out["race_laps_clean"] = race_clean
    out["race_summary"] = race_summary

    if "quali_laps" in event and isinstance(event["quali_laps"], pd.DataFrame):
        q_clean, q_summary = clean_quali_laps(event["quali_laps"], config)
        out["quali_laps_clean"] = q_clean
        out["quali_summary"] = q_summary

    return out


# -------- Manual test runner --------
if __name__ == "__main__":
    from src.load_data import load_config as _ld_cfg, load_all_data

    cfg = _ld_cfg("config/config.yaml")
    events = load_all_data(cfg)

    print(f"[INFO] Loaded {len(events)} events; cleaning…")
    cleaned: List[Dict[str, Any]] = []

    race_totals = []
    quali_totals = []

    for ev in events:
        evc = clean_event_payload(ev, cfg)
        cleaned.append(evc)

        year, gp = evc["year"], evc["gp"]

        rs = evc["race_summary"]
        rt = rs[rs["step"] == "trimmed"].groupby("driver")["n"].sum().sum()
        race_totals.append((year, gp, int(rt)))

        if "quali_summary" in evc:
            qs = evc["quali_summary"]
            qt = qs[qs["step"] == "trimmed"].groupby("driver")["n"].sum().sum()
            quali_totals.append((year, gp, int(qt)))
            print(f"[INFO] {year} {gp}: race clean laps = {int(rt)}, quali clean laps = {int(qt)}")
        else:
            print(f"[INFO] {year} {gp}: race clean laps = {int(rt)}")

    out_dir = _project_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_race_summ = pd.concat([c["race_summary"].assign(event=f'{c["year"]} {c["gp"]}') for c in cleaned], ignore_index=True)
    all_race_summ.to_csv(out_dir / "diagnostics_race_cleaning.csv", index=False)

    if any("quali_summary" in c for c in cleaned):
        all_quali_summ = pd.concat(
            [c["quali_summary"].assign(event=f'{c["year"]} {c["gp"]}') for c in cleaned if "quali_summary" in c],
            ignore_index=True
        )
        all_quali_summ.to_csv(out_dir / "diagnostics_quali_cleaning.csv", index=False)

    print(f"[INFO] Wrote diagnostics to: {out_dir}")
