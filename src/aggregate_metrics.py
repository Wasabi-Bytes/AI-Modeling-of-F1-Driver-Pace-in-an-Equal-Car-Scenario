# src/aggregate_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import math
import numpy as np
import pandas as pd
import yaml

from load_data import load_config

# -------- Paths --------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# -------- Math helpers --------
def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _combine_event_row(row: pd.Series, wR: float, wQ: float) -> Tuple[float, float, int]:
    has_r = np.isfinite(row.get("race_delta_s")) and np.isfinite(row.get("race_se_s"))
    has_q = np.isfinite(row.get("quali_delta_s")) and np.isfinite(row.get("quali_se_s"))

    if has_r and has_q:
        d = wR * row["race_delta_s"] + wQ * row["quali_delta_s"]
        se = math.sqrt((wR**2) * (row["race_se_s"]**2) + (wQ**2) * (row["quali_se_s"]**2))
        n = int(_safe_float(row.get("race_n", 0)) + _safe_float(row.get("quali_n", 0)))
        return d, se, n

    if has_r:
        return row["race_delta_s"], row["race_se_s"], int(_safe_float(row.get("race_n", 0)))

    if has_q:
        return row["quali_delta_s"], row["quali_se_s"], int(_safe_float(row.get("quali_n", 0)))

    return float("nan"), float("nan"), 0

def _ivw_mean(values: np.ndarray, ses: np.ndarray) -> Tuple[float, float, int]:
    mask = np.isfinite(values) & np.isfinite(ses) & (ses > 0)
    if not mask.any():
        return float("nan"), float("nan"), 0
    v = values[mask]
    se = ses[mask]
    w = 1.0 / (se**2)
    mean = float(np.sum(w * v) / np.sum(w))
    agg_se = float(math.sqrt(1.0 / np.sum(w)))
    k = int(mask.sum())
    return mean, agg_se, k

# -------- Main aggregation --------
def main():
    cfg = load_config("config/config.yaml")
    wR = float(cfg.get("wR", 0.6))
    wQ = float(cfg.get("wQ", 0.4))

    metrics_dir = _project_root() / "outputs" / "metrics"
    combined_csv = metrics_dir / "all_events_metrics.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(f"Missing {combined_csv}. Run model_metrics.py first.")

    df = pd.read_csv(combined_csv)
    # Normalize column names just in case
    for col in ["race_delta_s", "race_se_s", "race_n", "quali_delta_s", "quali_se_s", "quali_n"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Per-event combined delta & SE
    rows = []
    for (year, gp, driver), sub in df.groupby(["year", "gp", "driver"], dropna=False):
        # Each (year,gp,driver) should have a single row, but just in case:
        row = sub.iloc[0].copy()
        d, se, n = _combine_event_row(row, wR, wQ)
        rows.append({
            "year": year,
            "gp": gp,
            "driver": driver,
            "event_delta_s": d,
            "event_se_s": se,
            "event_weight_n": n,
            "race_delta_s": row.get("race_delta_s", np.nan),
            "race_se_s": row.get("race_se_s", np.nan),
            "race_n": row.get("race_n", np.nan),
            "quali_delta_s": row.get("quali_delta_s", np.nan),
            "quali_se_s": row.get("quali_se_s", np.nan),
            "quali_n": row.get("quali_n", np.nan),
        })

    event_breakdown = pd.DataFrame(rows)
    # Rank within each event (lower is better)
    event_breakdown["event_rank"] = event_breakdown.groupby(["year", "gp"])["event_delta_s"].rank(method="min")

    # Aggregate across events per driver using inverse-variance weighting
    agg_rows = []
    for driver, sub in event_breakdown.groupby("driver", dropna=False):
        vals = sub["event_delta_s"].to_numpy(dtype=float)
        ses = sub["event_se_s"].to_numpy(dtype=float)
        mean, agg_se, k = _ivw_mean(vals, ses)

        # Sample sizes for transparency
        race_n_sum = int(pd.to_numeric(sub.get("race_n", 0), errors="coerce").fillna(0).sum())
        quali_n_sum = int(pd.to_numeric(sub.get("quali_n", 0), errors="coerce").fillna(0).sum())

        agg_rows.append({
            "driver": driver,
            "agg_delta_s": mean,
            "agg_se_s": agg_se,
            "events_used": k,
            "race_laps_used": race_n_sum,
            "quali_laps_used": quali_n_sum,
        })

    driver_ranking = pd.DataFrame(agg_rows).sort_values("agg_delta_s").reset_index(drop=True)
    driver_ranking["rank"] = driver_ranking["agg_delta_s"].rank(method="min")

    outdir = _project_root() / "outputs" / "aggregate"
    _ensure_dir(outdir)

    event_breakdown.to_csv(outdir / "event_breakdown.csv", index=False)
    driver_ranking.to_csv(outdir / "driver_ranking.csv", index=False)

    print(f"[INFO] Wrote per-event breakdown: {outdir / 'event_breakdown.csv'} "
          f"(rows={len(event_breakdown)})")
    print(f"[INFO] Wrote driver ranking: {outdir / 'driver_ranking.csv'} "
          f"(drivers={driver_ranking['driver'].nunique()})")

if __name__ == "__main__":
    main()
