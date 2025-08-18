# src/diagnostics_temp.py
from __future__ import annotations
from typing import Callable, List, Dict, Any
import numpy as np
import pandas as pd

def _edges_from_cfg(cfg: Dict[str, Any]) -> List[float]:
    # Defaults to [28, 35] if not set in YAML: weather.temp_bins_c
    bins = ((cfg.get("weather") or {}).get("temp_bins_c") or [28.0, 35.0])
    return [float(b) for b in bins]

def _label_bins(edges: List[float]) -> List[str]:
    # e.g., [28, 35] → ["<28°C", "28–35°C", ">35°C"]
    lo, hi = edges[0], edges[1]
    return [f"<{lo:.0f}°C", f"{lo:.0f}–{hi:.0f}°C", f">{hi:.0f}°C"]

def _assign_bin(track_temp_c: pd.Series, edges: List[float]) -> pd.Series:
    # (-inf, e0], (e0, e1], (e1, inf); NaNs → "unknown"
    s = pd.to_numeric(track_temp_c, errors="coerce")
    labels = _label_bins(edges)
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    out[s.le(edges[0])] = labels[0]
    out[s.gt(edges[0]) & s.le(edges[1])] = labels[1]
    out[s.gt(edges[1])] = labels[2]
    return out.fillna("unknown")

def temp_bin_deltas(
    events: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    build_driver_deltas: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """
    For each temperature bin, filter race laps and run your delta builder:
      build_driver_deltas(laps_df) -> DataFrame with at least
        ["Driver","Team","delta_R","se_R"] (rename happens here)
    Returns tidy table: [temp_bin, Driver, Team, delta, se]
    """
    edges = _edges_from_cfg(cfg)
    bins_label = _label_bins(edges)

    results = []
    for ev in events:
        # prefer race_laps if it already has temps; else use race_interactions
        laps = ev.get("race_laps")
        if laps is None or "track_temp_c" not in laps.columns:
            laps = ev.get("race_interactions")
        if laps is None or "track_temp_c" not in laps.columns:
            continue

        d = laps.copy()
        d["temp_bin"] = _assign_bin(d["track_temp_c"], edges)
        # keep only pace laps if the flag exists
        if "lap_ok" in d.columns:
            d = d[d["lap_ok"] == True].copy()

        for b in bins_label:  # skip "unknown" to keep output crisp
            sub = d[d["temp_bin"] == b]
            if sub.empty:
                continue
            deltas_b = build_driver_deltas(sub)

            # normalize output names for reporting
            cols = deltas_b.columns
            if "delta_R" in cols and "se_R" in cols:
                deltas_b = deltas_b.rename(columns={"delta_R": "delta", "se_R": "se"})
            elif "delta" not in cols or "se" not in cols:
                # best-effort: try common alternatives
                if "se_R" in cols and "delta" in cols:
                    deltas_b = deltas_b.rename(columns={"se_R": "se"})
                elif "delta_R" in cols and "se" in cols:
                    deltas_b = deltas_b.rename(columns={"delta_R": "delta"})

            deltas_b.insert(0, "temp_bin", b)
            results.append(deltas_b[["temp_bin", "Driver", "Team", "delta", "se"]])

    if not results:
        return pd.DataFrame(columns=["temp_bin", "Driver", "Team", "delta", "se"])

    return pd.concat(results, ignore_index=True)

def summarize_temp_bins(bin_table: pd.DataFrame) -> pd.DataFrame:
    """
    Compact pivot to eyeball hot/cool patterns:
    rows=Driver, cols=temp_bin, values=mean delta.
    (Lower delta = faster, consistent with your convention.)
    """
    if bin_table.empty:
        return bin_table
    pivot = bin_table.pivot_table(
        index="Driver", columns="temp_bin", values="delta", aggfunc="mean"
    )
    return pivot.reset_index()
