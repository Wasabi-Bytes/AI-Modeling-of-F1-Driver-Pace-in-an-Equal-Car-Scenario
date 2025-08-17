# src/aggregate_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
import math

import numpy as np
import pandas as pd

from load_data import load_config

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


# ---------- Weight building pieces ----------
def _choose_delta_and_se(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Prefer shrunk columns when available; fallback to raw event delta/SE."""
    if "event_delta_s_shrunk" in df.columns and df["event_delta_s_shrunk"].notna().any():
        delta = pd.to_numeric(df["event_delta_s_shrunk"], errors="coerce")
    else:
        delta = pd.to_numeric(df.get("event_delta_s"), errors="coerce")
    se = pd.to_numeric(df.get("event_se_s"), errors="coerce")
    return delta, se


def _invvar(se: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Inverse-variance with guards."""
    se = pd.to_numeric(se, errors="coerce")
    se2 = se ** 2
    se2 = se2.where(se2 > 0, np.nan)
    base = 1.0 / se2
    if base.notna().sum() == 0:
        base = pd.Series(1.0, index=se.index)  # uniform fallback
    return base


def _recency_factor(
    df: pd.DataFrame,
    mode: str,
    event_idx_col: str,
    event_date_col: str,
    event_decay: float,
    half_life_days: float
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (recency_factor, events_ago, days_ago).
    If dates are missing for half-life mode, falls back to event-index decay.
    """
    recency = pd.Series(1.0, index=df.index, dtype=float)
    events_ago = pd.Series(np.nan, index=df.index, dtype=float)
    days_ago = pd.Series(np.nan, index=df.index, dtype=float)

    if mode == "date_half_life":
        # Find a usable date column
        cand_cols = [event_date_col, "event_date", "date", "session_date"]
        date_col = next((c for c in cand_cols if c in df.columns), None)
        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            if dates.notna().any():
                max_date = dates.max()
                days_ago = (max_date - dates).dt.total_seconds() / 86400.0
                hl = max(float(half_life_days), 1.0)
                recency = np.power(0.5, days_ago / hl)
                # also compute events_ago for reference if present
                if event_idx_col in df.columns:
                    max_idx = int(pd.to_numeric(df[event_idx_col], errors="coerce").max())
                    events_ago = max_idx - pd.to_numeric(df[event_idx_col], errors="coerce")
                return recency.clip(0.0, 1.0), events_ago, days_ago

    # Fallback or explicit event-index mode
    if event_idx_col in df.columns:
        idx = pd.to_numeric(df[event_idx_col], errors="coerce")
        max_idx = int(idx.max())
        events_ago = max_idx - idx
        recency = np.power(float(event_decay), events_ago.clip(lower=0))
    return recency.clip(0.0, 1.0), events_ago, days_ago


def _sample_factor(df: pd.DataFrame, race_w: float, quali_w: float) -> pd.Series:
    """
    Effective sample size multiplier.
    Use race laps (race_n) and a lightweight proxy for quali contribution (quali_k segments).
    You can tune each via config weights.
    """
    race = pd.to_numeric(df.get("race_n"), errors="coerce").fillna(0.0).clip(lower=0.0)
    quali = pd.to_numeric(df.get("quali_k"), errors="coerce").fillna(0.0).clip(lower=0.0)
    eff = race_w * race + quali_w * quali
    # Ensure at least 1.0 so events with zero counts but valid SE don't get nuked
    return eff.replace(0.0, 1.0)


# ---------- Aggregation ----------
def aggregate_driver_metrics(events_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input: per-driver per-event metrics DataFrame from model_metrics.py output.
    Output:
      - driver_ranking: aggregated equal-car delta per driver with SE and diagnostics
      - event_breakdown: per-event effective weights and components
    """
    required = {"driver", "team", "year", "gp", "event_idx"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing columns in input metrics: {missing}")

    # Pick which deltas/SEs to aggregate
    delta, se = _choose_delta_and_se(events_df)

    df = events_df.copy()
    df["event_delta_pick"] = delta
    df["event_se_pick"] = se

    # --- Config ---
    wcfg = cfg.get("weighting", {}) if isinstance(cfg.get("weighting", {}), dict) else {}
    recency_mode = str(wcfg.get("recency_mode", "event_index")).lower()  # "event_index" | "date_half_life"
    event_decay = float(wcfg.get("event_recency_decay", 0.92))            # per-event index decay
    half_life_days = float(wcfg.get("half_life_days", 120.0))            # date half-life
    race_sample_w = float(wcfg.get("race_sample_weight", 1.0))
    quali_sample_w = float(wcfg.get("quali_sample_weight", 1.0))

    # --- Weight components ---
    invvar = _invvar(df["event_se_pick"])  # 1/SE^2
    recency, events_ago, days_ago = _recency_factor(
        df, recency_mode, "event_idx", "event_date", event_decay, half_life_days
    )
    sample = _sample_factor(df, race_sample_w, quali_sample_w)

    df["w_invvar"] = invvar
    df["w_recency"] = recency
    df["w_sample"] = sample

    # Final event weight = recency * invvar * sample
    df["event_weight"] = df["w_recency"] * df["w_invvar"] * df["w_sample"]
    df["events_ago"] = events_ago
    df["days_ago"] = days_ago

    # Event-level breakdown for export / inspection
    keep_cols = [
        "year", "gp", "event_idx", "driver", "team",
        "event_delta_pick", "event_se_pick",
        "w_invvar", "w_recency", "w_sample", "event_weight",
        "race_n", "quali_k", "events_ago", "days_ago",
        # provenance / reference
        "event_delta_s", "event_delta_s_shrunk", "event_se_s",
        "event_wR_eff", "event_wQ_eff"
    ]
    present_cols = [c for c in keep_cols if c in df.columns]
    event_breakdown = (
        df[present_cols]
        .copy()
        .sort_values(["event_idx", "driver"], ignore_index=True)
    )

    # Aggregate per driver
    def _agg_driver(sub: pd.DataFrame) -> pd.Series:
        d = pd.to_numeric(sub["event_delta_pick"], errors="coerce")
        w = pd.to_numeric(sub["event_weight"], errors="coerce")

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

        # Weighted mean
        w_sum = float(w.sum())
        agg_delta = float((w * d).sum() / w_sum)

        # Approximate SE of weighted mean:
        # Our event weights are w = recency * sample * (1/SE^2).
        # The precision adds, so an optimistic but consistent SE is sqrt(1 / sum(w)).
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

    # Cosmetic label: team with the largest cumulative event_weight
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
    events_df["event_idx"] = pd.to_numeric(events_df["event_idx"], errors="coerce") \
        .fillna(events_df["event_idx"].max()).astype(int)
    events_df = events_df.sort_values(["event_idx", "driver"]).reset_index(drop=True)

    driver_ranking, event_breakdown = aggregate_driver_metrics(events_df, cfg)

    # Save
    driver_ranking.to_csv(out_dir / "driver_ranking.csv", index=False)
    event_breakdown.to_csv(out_dir / "event_breakdown.csv", index=False)

    print(f"[INFO] Wrote: {out_dir / 'driver_ranking.csv'}")
    print(f"[INFO] Wrote: {out_dir / 'event_breakdown.csv'}")


if __name__ == "__main__":
    main()
