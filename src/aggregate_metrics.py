# src/aggregate_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
import math

import numpy as np
import pandas as pd

from load_data import load_config, load_all_data  # for event ordering if needed

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")


# ---------- Paths ----------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-").replace("--", "-")


# ---------- Loading ----------
def _load_all_event_metrics(metrics_dir: Path) -> pd.DataFrame:
    """
    Prefer the combined file written by model_metrics.main().
    Fallback: concat per-event files if needed.
    """
    combined = metrics_dir / "all_events_metrics.csv"
    if combined.exists():
        df = pd.read_csv(combined)
        # Keep order of appearance as event index
        df["event_key"] = df["year"].astype(str) + " - " + df["gp"].astype(str)
        # Create stable event order index by first appearance
        order = (
            df[["event_key"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index(names="event_idx")
        )
        df = df.merge(order, on="event_key", how="left")
        return df

    # Fallback: stitch per-event files
    rows = []
    for f in metrics_dir.glob("*-event_metrics.csv"):
        part = pd.read_csv(f)
        # Try to parse year/gp from filename: "YYYY-gp-name-event_metrics.csv"
        stem = f.stem.replace("-event_metrics", "")
        try:
            year, gp_slug = stem.split("-", 1)
            part.insert(0, "year", int(year))
            part.insert(1, "gp", gp_slug.replace("-", " ").title())
        except Exception:
            pass
        rows.append(part)
    if not rows:
        raise FileNotFoundError(f"No event metrics found in {metrics_dir}")
    df = pd.concat(rows, ignore_index=True)
    df["event_key"] = df["year"].astype(str) + " - " + df["gp"].astype(str)
    order = (
        df[["event_key"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index(names="event_idx")
    )
    df = df.merge(order, on="event_key", how="left")
    return df


# ---------- Aggregation ----------
def _choose_delta_and_se(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Prefer shrunk columns when available; fallback to raw event delta/SE.
    """
    if "event_delta_s_shrunk" in df.columns and df["event_delta_s_shrunk"].notna().any():
        delta = pd.to_numeric(df["event_delta_s_shrunk"], errors="coerce")
    else:
        delta = pd.to_numeric(df.get("event_delta_s"), errors="coerce")
    se = pd.to_numeric(df.get("event_se_s"), errors="coerce")
    return delta, se


def _inverse_variance_recency_weights(se: pd.Series, event_idx: pd.Series, max_idx: int, recency_decay: float) -> pd.Series:
    """
    Weight_i = (recency_decay ** events_ago) * (1 / se_i^2)
    With guardrails for missing or zero SE.
    """
    eps = 1e-9
    se2 = se**2
    # If se is missing or 0, use a soft floor to avoid infinite weight
    se2 = se2.where(se2 > 0, other=np.nan)
    base = 1.0 / se2
    # If still all NaN, fall back to uniform base weights
    if base.notna().sum() == 0:
        base = pd.Series(1.0, index=se.index)

    # Recency: newer events get events_ago = 0, older larger number
    events_ago = max_idx - event_idx.fillna(event_idx.max())
    rec = (recency_decay ** events_ago).astype(float)
    rec = rec.clip(lower=0.0, upper=1.0)

    w = rec * base
    # Normalize later per driver; return as-is now
    return w


def aggregate_driver_metrics(events_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input: per-lap/per-driver per-event metrics DataFrame from model_metrics.py output.
    Output:
      - driver_ranking: aggregated equal-car delta per driver with SE and diagnostics
      - event_breakdown: per-event effective weights used and chosen deltas
    """
    # Columns sanity
    required = {"driver", "team", "year", "gp", "event_idx"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing columns in input metrics: {missing}")

    # Pick which deltas to aggregate
    delta, se = _choose_delta_and_se(events_df)

    df = events_df.copy()
    df["event_delta_pick"] = delta
    df["event_se_pick"] = se

    # Weighting
    recency_decay = float(cfg.get("recency_decay", 0.92))
    max_idx = int(df["event_idx"].max())
    df["w_base"] = _inverse_variance_recency_weights(df["event_se_pick"], df["event_idx"], max_idx, recency_decay)

    # Some diagnostics weights (optionally include sample counts)
    # If race_n/quali_k exist, you can incorporate them multiplicatively:
    if "race_n" in df.columns:
        df["w_samples"] = df["race_n"].fillna(0).clip(lower=0)
    else:
        df["w_samples"] = 0.0
    if "quali_k" in df.columns:
        df["w_samples"] = df["w_samples"] + df["quali_k"].fillna(0).clip(lower=0)

    # Final event weight: base weight (recency * inverse variance).
    # You *may* add a small sample-based prior, but keep it simple for transparency.
    df["event_weight"] = df["w_base"]

    # Event-level breakdown for export
    event_breakdown = df[[
        "year", "gp", "event_idx", "driver", "team",
        "event_delta_pick", "event_se_pick", "event_weight",
        "race_n", "quali_k",
        "event_delta_s", "event_delta_s_shrunk", "event_se_s"
    ]].copy().sort_values(["event_idx", "driver"], ignore_index=True)

    # Aggregate per driver
    def _agg_driver(sub: pd.DataFrame) -> pd.Series:
        d = pd.to_numeric(sub["event_delta_pick"], errors="coerce")
        w = pd.to_numeric(sub["event_weight"], errors="coerce")
        se = pd.to_numeric(sub["event_se_pick"], errors="coerce")

        mask = d.notna() & w.notna() & (w > 0)
        if mask.sum() == 0:
            return pd.Series({
                "agg_delta_s": np.nan,
                "agg_se_s": np.nan,
                "events_used": 0,
                "total_weight": 0.0
            })

        d = d[mask]
        w = w[mask]
        se = se[mask]

        # Weighted mean
        w_sum = float(w.sum())
        agg_delta = float((w * d).sum() / w_sum)

        # SE of weighted mean: sqrt(1 / sum(w_invvar)), where w includes recency.
        # Our w = recency / se^2  =>  sum_w = sum(recency / se^2)
        # Approximate SE = sqrt(1 / sum_w)
        agg_se = math.sqrt(1.0 / w_sum) if w_sum > 0 else np.nan

        return pd.Series({
            "agg_delta_s": agg_delta,
            "agg_se_s": agg_se,
            "events_used": int(mask.sum()),
            "total_weight": w_sum
        })

    driver_agg = (
        df.groupby("driver", dropna=False)
          .apply(_agg_driver)
          .reset_index()
          .sort_values("agg_delta_s", ignore_index=True)
    )

    # Also keep team with the largest weight for label (purely cosmetic)
    top_team = (
        df.groupby(["driver", "team"], dropna=False)["event_weight"]
          .sum()
          .reset_index()
          .sort_values(["driver", "event_weight"], ascending=[True, False])
          .drop_duplicates("driver")
          .rename(columns={"team": "label_team"})[["driver", "label_team"]]
    )
    driver_agg = driver_agg.merge(top_team, on="driver", how="left")

    return driver_agg, event_breakdown


# ---------- Main ----------
def main():
    cfg = load_config("config/config.yaml")

    metrics_dir = _project_root() / "outputs" / "metrics"
    out_dir = _project_root() / "outputs" / "aggregate"
    _ensure_dir(out_dir)

    events_df = _load_all_event_metrics(metrics_dir)

    # Ensure numeric and sort by event order
    events_df["event_idx"] = pd.to_numeric(events_df["event_idx"], errors="coerce").fillna(events_df["event_idx"].max()).astype(int)
    events_df = events_df.sort_values(["event_idx", "driver"]).reset_index(drop=True)

    driver_ranking, event_breakdown = aggregate_driver_metrics(events_df, cfg)

    # Save
    driver_ranking.to_csv(out_dir / "driver_ranking.csv", index=False)
    event_breakdown.to_csv(out_dir / "event_breakdown.csv", index=False)

    print(f"[INFO] Wrote: {out_dir / 'driver_ranking.csv'}")
    print(f"[INFO] Wrote: {out_dir / 'event_breakdown.csv'}")


if __name__ == "__main__":
    main()
