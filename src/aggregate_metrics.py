# src/aggregate_metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
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
        # Stable event order by first appearance
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
        # Try parse year/gp from filename: "YYYY-gp-name-event_metrics.csv"
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


# ---------- Track meta helpers ----------
def _norm_event_key_str(s: str) -> str:
    # Lowercase, keep alnum+space, compress spaces
    if not isinstance(s, str):
        return "unknown"
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(cleaned.split())


def _norm_event_key(year: Optional[int], gp: Optional[str]) -> str:
    y = "" if pd.isna(year) else str(int(year))
    g = "" if gp is None else str(gp)
    return _norm_event_key_str(f"{y} {g}")


def _load_track_meta(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    paths = cfg.get("paths", {}) or {}
    meta_path = paths.get("track_meta")
    if not meta_path:
        return None
    f = (_project_root() / meta_path).resolve()
    if not f.exists():
        print(f"[INFO] track_meta not found at {f}; proceeding without archetype/forecast aggregates.")
        return None
    try:
        meta = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Failed to read track_meta: {e}; proceeding without archetype/forecast aggregates.")
        return None

    # Accept either explicit columns or a single 'event_key'
    cols = {c.lower(): c for c in meta.columns}
    # Normalize keys
    if "year" in cols and "gp" in cols:
        meta["event_key_norm"] = meta.apply(
            lambda r: _norm_event_key(r[cols["year"]], r[cols["gp"]]), axis=1
        )
    elif "event_key" in cols:
        meta["event_key_norm"] = meta[cols["event_key"]].map(_norm_event_key_str)
    else:
        # Best-effort: look for something like "Grand Prix" column
        guess_gp_col = next((c for c in meta.columns if "grand" in c.lower() or "prix" in c.lower() or c.lower() == "gp"), None)
        if guess_gp_col is not None and "year" in cols:
            meta["event_key_norm"] = meta.apply(
                lambda r: _norm_event_key(r[cols["year"]], r[guess_gp_col]), axis=1
            )
        else:
            print("[WARN] track_meta has no (year,gp) or event_key; cannot join.")
            return None

    # Standardize expected columns if present
    std = {}
    std["track_type"] = meta.get("track_type", meta.get(cols.get("track_type", ""), np.nan))
    std["downforce_index"] = pd.to_numeric(meta.get("downforce_index", meta.get(cols.get("downforce_index", ""), np.nan)), errors="coerce")
    std["drs_zones"] = pd.to_numeric(meta.get("drs_zones", meta.get(cols.get("drs_zones", ""), np.nan)), errors="coerce")
    std["overtaking_difficulty"] = pd.to_numeric(meta.get("overtaking_difficulty", meta.get(cols.get("overtaking_difficulty", ""), np.nan)), errors="coerce")
    std["speed_bias"] = pd.to_numeric(meta.get("speed_bias", meta.get(cols.get("speed_bias", ""), np.nan)), errors="coerce")
    meta_std = pd.concat([meta[["event_key_norm"]], pd.DataFrame(std)], axis=1)
    return meta_std


def _assign_df_bucket(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Assign low/med/high downforce buckets using tertiles from available meta.
    """
    m = meta.copy()
    if "downforce_index" not in m.columns:
        m["df_bucket"] = np.nan
        return m

    di = pd.to_numeric(m["downforce_index"], errors="coerce")
    if di.notna().sum() < 3:
        # Too little data to split meaningfully
        m["df_bucket"] = np.where(di.notna(), "med", np.nan)
        return m

    q1 = float(di.quantile(1/3))
    q2 = float(di.quantile(2/3))

    def _bucket(x):
        if not np.isfinite(x):
            return np.nan
        if x <= q1:
            return "low"
        elif x <= q2:
            return "med"
        else:
            return "high"

    m["df_bucket"] = di.map(_bucket)
    return m


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


def _read_recency_knobs(cfg: Dict[str, Any]) -> Tuple[str, float, float]:
    """
    Support both legacy `weighting` and new `aggregation` sections.
    Returns (mode, per_event_decay, half_life_days).
    """
    if isinstance(cfg.get("weighting", {}), dict) and cfg["weighting"]:
        w = cfg["weighting"]
        mode = str(w.get("recency_mode", "event_index")).lower()  # "event_index" | "date_half_life"
        return mode, float(w.get("event_recency_decay", 0.92)), float(w.get("half_life_days", 120.0))

    # Fallback to aggregation section
    a = cfg.get("aggregation", {}) or {}
    use_date = bool(a.get("use_date_decay", True))
    mode = "date_half_life" if use_date else "event_index"
    return mode, float(a.get("per_event_decay", 0.92)), float(a.get("half_life_days", 120.0))


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
        cand_cols = [event_date_col, "event_date", "date", "session_date"]
        date_col = next((c for c in cand_cols if c in df.columns), None)
        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            if dates.notna().any():
                max_date = dates.max()
                days_ago = (max_date - dates).dt.total_seconds() / 86400.0
                hl = max(float(half_life_days), 1.0)
                recency = np.power(0.5, days_ago / hl)
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
    """
    race = pd.to_numeric(df.get("race_n"), errors="coerce").fillna(0.0).clip(lower=0.0)
    quali = pd.to_numeric(df.get("quali_k"), errors="coerce").fillna(0.0).clip(lower=0.0)
    eff = race_w * race + quali_w * quali
    return eff.replace(0.0, 1.0)


# ---------- Core aggregation (reusable) ----------
def _aggregate_frame(events_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reusable aggregator:
      returns (driver_agg, event_breakdown) for the given events subset.
    """
    required = {"driver", "team", "event_idx"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Missing columns in input metrics: {missing}")

    delta, se = _choose_delta_and_se(events_df)

    df = events_df.copy()
    df["event_delta_pick"] = delta
    df["event_se_pick"] = se

    # --- Config ---
    mode, event_decay, half_life_days = _read_recency_knobs(cfg)

    wcfg = cfg.get("weighting", {}) if isinstance(cfg.get("weighting", {}), dict) else {}
    race_sample_w = float(wcfg.get("race_sample_weight", 1.0))
    quali_sample_w = float(wcfg.get("quali_sample_weight", 1.0))

    # --- Weight components ---
    invvar = _invvar(df["event_se_pick"])  # 1/SE^2
    recency, events_ago, days_ago = _recency_factor(
        df, mode, "event_idx", "event_date", event_decay, half_life_days
    )
    sample = _sample_factor(df, race_sample_w, quali_sample_w)

    df["w_invvar"] = invvar
    df["w_recency"] = recency
    df["w_sample"] = sample
    df["event_weight"] = df["w_recency"] * df["w_invvar"] * df["w_sample"]
    df["events_ago"] = events_ago
    df["days_ago"] = days_ago

    # Event-level breakdown for transparency
    keep_cols = [
        "year", "gp", "event_idx", "driver", "team",
        "event_delta_pick", "event_se_pick",
        "w_invvar", "w_recency", "w_sample", "event_weight",
        "race_n", "quali_k", "events_ago", "days_ago",
        "event_delta_s", "event_delta_s_shrunk", "event_se_s",
        "event_wR_eff", "event_wQ_eff",
        # optional meta if present
        "track_type", "downforce_index", "df_bucket",
        "drs_zones", "speed_bias", "overtaking_difficulty",
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
        w_sum = float(w.sum())
        agg_delta = float((w * d).sum() / w_sum)
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

    # Cosmetic: label team with largest cumulative weight
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


# ---------- Public aggregations ----------
def aggregate_driver_metrics_global(events_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _aggregate_frame(events_df, cfg)


def aggregate_driver_metrics_by_archetype(events_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns a long table with aggregates by (track_type, df_bucket).
    Requires events_df to have those columns (will be added in main() if track_meta is available).
    """
    if "track_type" not in events_df.columns or "df_bucket" not in events_df.columns:
        # Nothing to do
        return pd.DataFrame(columns=["driver", "agg_delta_s", "agg_se_s", "events_used", "total_weight", "label_team", "track_type", "df_bucket"])

    out_rows = []
    track_types = sorted([t for t in events_df["track_type"].dropna().unique()])
    df_buckets = sorted([b for b in events_df["df_bucket"].dropna().unique()])

    for tt in track_types:
        for db in df_buckets:
            sub_mask = (events_df["track_type"] == tt) & (events_df["df_bucket"] == db)
            sub = events_df.loc[sub_mask].copy()
            if sub.empty:
                continue
            agg, _ = _aggregate_frame(sub, cfg)
            agg["track_type"] = tt
            agg["df_bucket"] = db
            out_rows.append(agg)

    if not out_rows:
        return pd.DataFrame(columns=["driver", "agg_delta_s", "agg_se_s", "events_used", "total_weight", "label_team", "track_type", "df_bucket"])

    by_arch = pd.concat(out_rows, ignore_index=True)
    # Order within each (tt, db)
    by_arch = by_arch.sort_values(["track_type", "df_bucket", "agg_delta_s"], ignore_index=True)
    return by_arch


def forecast_profile(events_df: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Blend archetype-specific and global aggregates for a target event profile:
      delta_forecast = alpha * delta_archetype + (1-alpha) * delta_global
      se^2_forecast  ≈ alpha^2 * se_arch^2 + (1-alpha)^2 * se_global^2  (independence approx)
    Uses cfg['track_effects'].forecast_alpha if present (default 0.7).
    Target profile pulled from paths.track_meta by matching cfg['viz_track'] (year, gp).
    """
    # Need archetype columns prepared
    if ("track_type" not in events_df.columns or "df_bucket" not in events_df.columns
            or events_df["track_type"].notna().sum() == 0
            or events_df["df_bucket"].notna().sum() == 0):
        return None

    # Alpha from config
    te_cfg = cfg.get("track_effects", {}) or {}
    alpha = float(te_cfg.get("forecast_alpha", 0.7))
    alpha = min(max(alpha, 0.0), 1.0)

    # Identify target event from viz_track as a sensible default
    vt = cfg.get("viz_track", {}) or {}
    target_year = vt.get("year", None)
    target_gp = vt.get("grand_prix", None)
    target_key = _norm_event_key(target_year, target_gp)

    # Try to find matching row in events_df with meta attached (via main() join)
    candidates = events_df.copy()
    candidates["event_key_norm"] = candidates.apply(
        lambda r: _norm_event_key(r.get("year"), r.get("gp")), axis=1
    )
    sub = candidates.loc[candidates["event_key_norm"] == target_key]
    if sub.empty:
        # Fall back to most recent event
        sub = candidates.sort_values("event_idx", ascending=False).head(1)

    tt = sub.get("track_type")
    db = sub.get("df_bucket")
    if tt is None or db is None or tt.isna().all() or db.isna().all():
        return None

    target_tt = str(tt.iloc[0])
    target_db = str(db.iloc[0])

    # Build global and by-arch aggregates
    global_agg, _ = _aggregate_frame(events_df, cfg)
    by_arch = aggregate_driver_metrics_by_archetype(events_df, cfg)
    if by_arch.empty:
        return None

    # Select the archetype row for the target profile
    arch_sel = by_arch[(by_arch["track_type"] == target_tt) & (by_arch["df_bucket"] == target_db)]
    if arch_sel.empty:
        return None

    # Merge and blend
    m = global_agg.merge(
        arch_sel[["driver", "agg_delta_s", "agg_se_s"]]
            .rename(columns={"agg_delta_s": "arch_delta_s", "agg_se_s": "arch_se_s"}),
        on="driver", how="left"
    )
    if m["arch_delta_s"].isna().all():
        return None

    def _blend(row):
        dg, sg = row["agg_delta_s"], row["agg_se_s"]
        da, sa = row["arch_delta_s"], row["arch_se_s"]
        if pd.notna(da) and pd.notna(sa):
            delta = alpha * da + (1.0 - alpha) * dg
            # Independence approximation for variance
            var = (alpha ** 2) * (sa ** 2) + ((1.0 - alpha) ** 2) * (sg ** 2)
            se = math.sqrt(var)
            return pd.Series({"forecast_delta_s": delta, "forecast_se_s": se})
        else:
            # Fall back to global if archetype missing
            return pd.Series({"forecast_delta_s": dg, "forecast_se_s": sg})

    blend = m.apply(_blend, axis=1)
    out = pd.concat([m[["driver", "label_team", "agg_delta_s", "agg_se_s"]], blend], axis=1)
    out["track_type"] = target_tt
    out["df_bucket"] = target_db
    out["alpha"] = alpha
    return out.sort_values("forecast_delta_s", ignore_index=True)


# ---------- Main ----------
def main():
    cfg = load_config("config/config.yaml")

    metrics_dir = _project_root() / "outputs" / "metrics"
    out_dir = _project_root() / "outputs" / "aggregate"
    _ensure_dir(out_dir)

    # Load per-event metrics
    events_df = _load_all_event_metrics(metrics_dir)

    # Ensure numeric and sort by event order
    events_df["event_idx"] = pd.to_numeric(events_df["event_idx"], errors="coerce") \
        .fillna(events_df["event_idx"].max()).astype(int)
    events_df = events_df.sort_values(["event_idx", "driver"]).reset_index(drop=True)

    # === Join track_meta (optional) and assign downforce buckets ===
    meta = _load_track_meta(cfg)
    if meta is not None:
        events_df["event_key_norm"] = events_df.apply(
            lambda r: _norm_event_key(r.get("year"), r.get("gp")), axis=1
        )
        events_df = events_df.merge(meta, on="event_key_norm", how="left")
        # Assign df_bucket from the meta distribution (consistent cutpoints)
        meta_with_b = _assign_df_bucket(meta.dropna(subset=["downforce_index"]))
        _ = meta_with_b[["event_key_norm", "downforce_index", "df_bucket"]]  # kept for reference

        # Map df_bucket by position of downforce_index within meta tertiles
        if "downforce_index" in events_df.columns and events_df["downforce_index"].notna().any():
            q1 = float(meta["downforce_index"].quantile(1/3)) if meta["downforce_index"].notna().any() else 0.33
            q2 = float(meta["downforce_index"].quantile(2/3)) if meta["downforce_index"].notna().any() else 0.67

            def _bucket(x):
                if not np.isfinite(x):
                    return np.nan
                if x <= q1:
                    return "low"
                elif x <= q2:
                    return "med"
                else:
                    return "high"

            events_df["df_bucket"] = pd.to_numeric(events_df["downforce_index"], errors="coerce").map(_bucket)
    else:
        # No meta; ensure columns exist
        events_df["track_type"] = np.nan
        events_df["downforce_index"] = np.nan
        events_df["df_bucket"] = np.nan

    # === Global aggregation (existing behavior) ===
    driver_ranking_global, event_breakdown = aggregate_driver_metrics_global(events_df, cfg)

    # === Archetype aggregation (if meta present) ===
    driver_ranking_by_arch = aggregate_driver_metrics_by_archetype(events_df, cfg)

    # === Forecast profile (blend archetype + global) ===
    driver_forecast = forecast_profile(events_df, cfg)

    # ---- Save outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    driver_ranking_global.to_csv(out_dir / "driver_ranking.csv", index=False)
    event_breakdown.to_csv(out_dir / "event_breakdown.csv", index=False)
    print(f"[INFO] Wrote: {out_dir / 'driver_ranking.csv'}")
    print(f"[INFO] Wrote: {out_dir / 'event_breakdown.csv'}")

    if not driver_ranking_by_arch.empty:
        driver_ranking_by_arch.to_csv(out_dir / "driver_ranking_by_archetype.csv", index=False)
        print(f"[INFO] Wrote: {out_dir / 'driver_ranking_by_archetype.csv'}")
    else:
        print("[INFO] No track_meta available or insufficient to compute by-archetype aggregates.")

    if driver_forecast is not None and not driver_forecast.empty:
        driver_forecast.to_csv(out_dir / "driver_forecast.csv", index=False)
        print(f"[INFO] Wrote: {out_dir / 'driver_forecast.csv'}")
    else:
        print("[INFO] Forecast profile not produced (missing/insufficient meta).")


if __name__ == "__main__":
    main()
