# src/filters.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import logging


__all__ = [
    "standardize_lap_seconds",
    "is_green",
    "is_wet_compound",
    "derive_and_filter_tags_base",
    "drop_outliers",
    "clean_laps",
]


# ---------------------------- basic utilities ---------------------------- #
def standardize_lap_seconds(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "LapTimeSeconds" not in d.columns:
        if "LapTime" in d.columns:
            def _to_secs(x):
                try:
                    return float(getattr(x, "total_seconds", lambda: float(x))())
                except Exception:
                    try:
                        return float(x)
                    except Exception:
                        return np.nan
            d["LapTimeSeconds"] = d["LapTime"].apply(_to_secs)
        else:
            d["LapTimeSeconds"] = pd.to_numeric(d.get("LapTime", np.nan), errors="coerce")
    return d


def is_green(track_status: pd.Series) -> pd.Series:
    """
    FastF1 TrackStatus: '1' = green. Require pure '1' (no composite codes).
    """
    s = track_status.astype(str).fillna("")
    return s == "1"


def is_wet_compound(series: pd.Series) -> pd.Series:
    names = series.astype(str).str.upper()
    return names.isin({"INTERMEDIATE", "WET", "FULL WET", "EXTREME WET"})


# ----------------------- base tag + base filtering ----------------------- #
def derive_and_filter_tags_base(laps: pd.DataFrame, *, session_kind: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Standardize fields and apply the strict *base* pace filter:
    - green track
    - accurate timing
    - exclude pit in/out laps
    - valid laptime > 0

    Returns (df_with_tags, base_rejection_counts).
    Dry-only and outlier filtering are applied later in `clean_laps`.
    """
    d = standardize_lap_seconds(laps).reset_index(drop=True)

    # canonical identity fields
    if "Driver" in d.columns:
        d["driver"] = d["Driver"].astype(str)
    elif "DriverNumber" in d.columns:
        d["driver"] = d["DriverNumber"].astype(str)
    else:
        d["driver"] = d.get("DriverNumber", d.get("Driver", "UNK")).astype(str)

    d["Team"] = d.get("Team", "UNK").astype(str)
    d["compound"] = d.get("Compound", "UNKNOWN").fillna("UNKNOWN").astype(str)

    if "LapNumber" not in d.columns:
        d = d.sort_values(["driver", "LapTimeSeconds"]).copy()
        d["LapNumber"] = d.groupby("driver").cumcount() + 1
    d["lap_number"] = pd.to_numeric(d["LapNumber"], errors="coerce").fillna(0).astype(int)

    # track status & timing accuracy
    d["track_status"] = d.get("TrackStatus", "").astype(str)
    _green = is_green(d["track_status"])

    is_accurate = d.get("IsAccurate", True)
    if isinstance(is_accurate, pd.Series):
        is_accurate = is_accurate.fillna(True).astype(bool)

    # pit in/out
    for col in ("PitInTime", "PitOutTime"):
        if col not in d.columns:
            d[col] = pd.NaT
    is_outlap = d["PitOutTime"].notna()
    is_inlap = d["PitInTime"].notna()

    # valid time
    has_time = d["LapTimeSeconds"].notna() & (d["LapTimeSeconds"] > 0)

    # base flag
    d["lap_ok_base"] = has_time & is_accurate & (~is_outlap) & (~is_inlap) & _green

    # stint inference
    if "Stint" in d.columns:
        stint = pd.to_numeric(d["Stint"], errors="coerce")
        if stint.isna().any():
            d = d.sort_values(["driver", "LapNumber"]).copy()
            inferred = d["PitOutTime"].notna().groupby(d["driver"]).cumsum()
            stint = stint.fillna(inferred)
        d["stint_id"] = stint.fillna(-1).astype(int)
    else:
        d = d.sort_values(["driver", "LapNumber"]).copy()
        d["stint_id"] = d["PitOutTime"].notna().groupby(d["driver"]).cumsum().astype(int)

    d = d.sort_values(["driver", "stint_id", "LapNumber"]).copy()
    d["lap_on_tyre"] = d.groupby(["driver", "stint_id"]).cumcount() + 1

    # counts before trimming to base-ok
    base_counts = {
        "total_rows": int(len(d)),
        "base_dropped_non_green": int((~_green).sum()),
        "base_dropped_inlap": int(is_inlap.sum()),
        "base_dropped_outlap": int(is_outlap.sum()),
        "base_dropped_inaccurate": int((~is_accurate).sum()),
        "base_dropped_invalid_time": int((~has_time).sum()),
    }

    before = len(d)
    d = d.loc[d["lap_ok_base"]].reset_index(drop=True)
    logging.info(
        f"[filters] {session_kind}: base kept {len(d)}/{before} after green/accurate/no-pit/valid-time."
    )

    # ensure required columns exist
    for c in [
        "LapTimeSeconds", "driver", "Team", "compound",
        "stint_id", "lap_on_tyre", "lap_number", "track_status", "lap_ok_base"
    ]:
        if c not in d.columns:
            d[c] = np.nan if c != "lap_ok_base" else True

    return d, base_counts


# ---------------------------- outlier removal ---------------------------- #
def drop_outliers(
    d: pd.DataFrame,
    method: str = "iqr",
    iqr_k: float = 1.5,
    z_k: float = 3.0
) -> Tuple[pd.DataFrame, int]:
    """
    Deterministic outlier removal on LapTimeSeconds.
    Default = per-driver IQR; alternative = global z-score.
    """
    if d.empty:
        return d, 0

    if method.lower() == "iqr":
        def iqr_mask(s: pd.Series) -> pd.Series:
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
            return (s < lo) | (s > hi)
        mask = d.groupby("driver", group_keys=False)["LapTimeSeconds"].apply(iqr_mask)
        mask = mask.reindex(d.index, fill_value=False)

    else:  # z-score
        x = pd.to_numeric(d["LapTimeSeconds"], errors="coerce")
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            mask = pd.Series(False, index=d.index)
        else:
            z = (x - mu) / sd
            mask = z.abs() > z_k

    dropped = int(mask.sum())
    out = d.loc[~mask].reset_index(drop=True)
    return out, dropped


# ------------------------------- clean_laps ------------------------------- #
def clean_laps(
    d_base: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    session_kind: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Deterministic cleaning:
      - base rules (green/accurate/no-pit/valid-time)
      - optional dry-only (drop wet/intermediates)
      - optional outliers (IQR or z-score)
    Returns (clean_df, rejection_report).
    """
    # base
    d, base_counts = derive_and_filter_tags_base(d_base, session_kind=session_kind)

    # dry-only
    filt_cfg = (cfg.get("filters") or {})
    dry_only = bool(filt_cfg.get("dry_only", True))
    if dry_only and "compound" in d.columns:
        wet_mask = is_wet_compound(d["compound"])
        dropped_wet = int(wet_mask.sum())
        d = d.loc[~wet_mask].reset_index(drop=True)
    else:
        dropped_wet = 0

    # outliers
    drop_flag = bool(filt_cfg.get("drop_outliers", True))
    out_method = str(filt_cfg.get("outlier_method", "iqr"))
    iqr_k = float(filt_cfg.get("iqr_k", 1.5))
    z_k = float(filt_cfg.get("z_k", 3.0))
    if drop_flag:
        d, dropped_outliers = drop_outliers(d, method=out_method, iqr_k=iqr_k, z_k=z_k)
    else:
        dropped_outliers = 0

    d["lap_ok"] = True

    report: Dict[str, int] = {
        **base_counts,
        "dropped_wet": dropped_wet,
        "dropped_outliers": dropped_outliers,
        "kept_total": int(len(d)),
    }

    logging.info(
        f"[clean_laps] {session_kind}: dry_only={dry_only}, outliers={drop_flag} "
        f"(method={out_method}, iqr_k={iqr_k}, z_k={z_k}); kept={report['kept_total']}."
    )

    return d, report
