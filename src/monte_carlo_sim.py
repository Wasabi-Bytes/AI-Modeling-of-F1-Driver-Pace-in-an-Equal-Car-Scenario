from __future__ import annotations

"""
Monte Carlo season simulator using the existing equal-car race engine.

• Scans outputs/metrics/*-event_metrics.csv to determine the season schedule
• For each event, loads event-specific deltas and the track outline/meta
• Runs N Monte Carlo seasons (5000 by default)
• Awards FIA points (configurable) to Drivers and Constructors
• Saves:
    - outputs/mc/driver_title_probs.png
    - outputs/mc/constructor_title_probs.png
    - outputs/mc/driver_points_mean.csv
    - outputs/mc/constructor_points_mean.csv
    - outputs/mc/title_probabilities.csv

Requires only what the repo already has (config.yaml, metrics, track loader).
"""

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local imports from your codebase
from load_data import load_config, enable_cache
from visualize_equal_race import (
    simulate_progress,
    load_driver_ranking_event,
    load_driver_ranking_global,
    load_track_outline,
    _cfg_get,
    _load_viz_track_meta,
    _load_weather_summary_for_viz,
    _list_event_metric_files,
    _get_driver_team_map_from_recent,
)

PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "outputs" / "mc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Points tables ----------------
FIA_POINTS_2010 = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

# ---------------- Helpers ----------------

def _parse_event_from_metrics_path(p: Path) -> Tuple[int, str]:
    """Extract (year, gp_slug) from a metrics filename like '2024-canadian-event_metrics.csv'.
    gp_slug preserves hyphens (e.g. 'sao-paulo')."""
    name = p.name
    base = name.replace("-event_metrics.csv", "")
    year_str, gp_slug = base.split("-", 1)
    return int(year_str), gp_slug


def _season_from_metrics() -> List[Tuple[int, str, Path]]:
    files = _list_event_metric_files()
    season: List[Tuple[int, str, Path]] = []
    for f in files:
        try:
            y, gp = _parse_event_from_metrics_path(f)
            season.append((y, gp, f))
        except Exception:
            continue
    # Keep original order in metrics dir (already sorted newest last in helper) -> sort by filename
    season.sort(key=lambda t: t[2].name)
    return season


def _cfg_for_event(base_cfg: dict, year: int, gp_slug: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    vt = cfg.get("viz_track", {}) or {}
    vt.update({"year": year, "grand_prix": gp_slug})
    cfg["viz_track"] = vt
    return cfg


def _load_team_map_for_event(metrics_csv: Path) -> Dict[str, str]:
    """Prefer team mapping from the per-event metrics file (driver,team)."""
    try:
        df = pd.read_csv(metrics_csv)
        low = {c.lower(): c for c in df.columns}
        if ("driver" in low) and ("team" in low):
            sub = df[[low["driver"], low["team"]]].dropna()
            sub.columns = ["driver", "team"]
            return {str(r.driver): str(r.team) for r in sub.itertuples(index=False)}
    except Exception:
        pass
    # Fallback: infer from most recent race data
    team_map, _, _ = _get_driver_team_map_from_recent()
    return team_map


def _award_points(order: List[str], table: List[int]) -> Dict[str, int]:
    pts: Dict[str, int] = {d: 0 for d in order}
    for pos, drv in enumerate(order, start=1):
        if pos <= len(table):
            pts[drv] += table[pos - 1]
    return pts


def _tiebreak_key(points: Dict[str, int], pos_counts: Dict[str, Dict[int, int]], name: str) -> Tuple:
    # Higher points first, then more wins, then more 2nds, 3rds, ... then alphabetical for stability
    counts = pos_counts.get(name, {})
    return (
        -points.get(name, 0),
        -counts.get(1, 0),
        -counts.get(2, 0),
        -counts.get(3, 0),
        -counts.get(4, 0),
        -counts.get(5, 0),
        name,
    )


def _make_bar(prob_s: pd.Series, title: str, out_png: Path) -> None:
    prob_s = prob_s.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(prob_s.index, prob_s.values)
    plt.ylabel("Probability of Title")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ---------------- Core simulation ----------------

def simulate_season(
    cfg: dict,
    runs: int = 5000,
    use_event_deltas: bool = True,
    base_lap: float | None = None,
    n_laps: int | None = None,
    dt: float | None = None,
    noise_sd: float | None = None,
    points_table: List[int] = FIA_POINTS_2010,
) -> Dict[str, Path]:
    """Run N Monte Carlo seasons across all events found in outputs/metrics.

    Returns dict of file paths written.
    """
    if "cache_dir" in cfg:
        enable_cache(cfg["cache_dir"])

    vizsec = _cfg_get(cfg, ["visualize_equal_race"], {}) or {}
    base_lap = float(base_lap if base_lap is not None else _cfg_get(vizsec, ["base_lap_sec"], 90.0))
    n_laps = int(n_laps if n_laps is not None else _cfg_get(vizsec, ["n_laps"], 20))
    dt = float(dt if dt is not None else _cfg_get(vizsec, ["dt"], 0.5))
    noise_sd = float(noise_sd if noise_sd is not None else _cfg_get(vizsec, ["lap_jitter_sd"], 0.12))
    seed = int(_cfg_get(vizsec, ["seed"], 42))

    season = _season_from_metrics()
    if not season:
        raise RuntimeError("No event_metrics files found in outputs/metrics — run model_metrics.py first.")

    # Determine full driver roster from global ranking (fallback) so we can allocate arrays
    global_rank = load_driver_ranking_global(cfg)
    all_drivers = list(global_rank["driver"].astype(str))

    driver_title_counts: Dict[str, int] = {d: 0 for d in all_drivers}
    team_title_counts: Dict[str, int] = {}

    # For mean points across runs
    driver_points_accum: Dict[str, float] = {d: 0.0 for d in all_drivers}
    team_points_accum: Dict[str, float] = {}

    for run_idx in range(runs):
        # Per-season tallies
        d_points: Dict[str, int] = {d: 0 for d in all_drivers}
        t_points: Dict[str, int] = {}
        d_pos_counts: Dict[str, Dict[int, int]] = {d: {} for d in all_drivers}

        print(f"===== Season {run_idx+1}/{runs} =====")

        for ev_idx, (year, gp_slug, metrics_csv) in enumerate(season):
            cfg_ev = _cfg_for_event(cfg, year, gp_slug)

            # Ranking for this event (event-specific deltas when available)
            if use_event_deltas:
                ranking = load_driver_ranking_event(cfg_ev, gp_slug)
                if ranking is None or ranking.empty:
                    ranking = load_driver_ranking_global(cfg_ev)
            else:
                ranking = load_driver_ranking_global(cfg_ev)

            # Track outline + meta + weather
            xy = load_track_outline(cfg_ev)
            meta = _load_viz_track_meta(cfg_ev)
            weather = _load_weather_summary_for_viz(cfg_ev)

            # Simulate this race (unique run index per event to decorrelate streams)
            _, _, _, drivers, _, _, _, _, orders, _, _, _, _, stats = simulate_progress(
                ranking,
                xy,
                base_lap=base_lap,
                n_laps=n_laps,
                dt=dt,
                noise_sd=noise_sd,
                seed=seed,
                cfg=cfg_ev,
                meta=meta,
                weather_summary=weather,
                run_idx=(run_idx * 1000 + ev_idx),
            )

            final_order_idx = stats.get("finish_order", orders[-1])
            fin = [drivers[i] for i in final_order_idx]

            # --- Console notes per race ---
            top3 = ", ".join(fin[:3]) if len(fin) >= 3 else ", ".join(fin)
            last = fin[-1] if fin else "—"
            print(f"[{year} {gp_slug}] P1–P3: {top3} | Last: {last}")

            # Event-specific team mapping preferred
            team_map = _load_team_map_for_event(metrics_csv)

            # Award points
            pts_dr = _award_points(fin, points_table)
            for drv, p in pts_dr.items():
                d_points[drv] = d_points.get(drv, 0) + int(p)
            for pos, drv in enumerate(fin, start=1):
                d_pos_counts.setdefault(drv, {})[pos] = d_pos_counts.get(drv, {}).get(pos, 0) + 1

            # Constructors: sum both drivers present that event
            for pos, drv in enumerate(fin, start=1):
                team = team_map.get(drv, "UNKNOWN")
                tp = points_table[pos - 1] if pos <= len(points_table) else 0
                t_points[team] = t_points.get(team, 0) + int(tp)

        # --- Season done: pick champions with tie-breaks ---
        if d_points:
            champ_driver = min(d_points.keys(), key=lambda n: _tiebreak_key(d_points, d_pos_counts, n))
            driver_title_counts[champ_driver] = driver_title_counts.get(champ_driver, 0) + 1
            for d, v in d_points.items():
                driver_points_accum[d] = driver_points_accum.get(d, 0.0) + float(v)
            print(f"   >> Driver Champion: {champ_driver}")

        if t_points:
            # Tie-break: most points then alphabetical
            champ_team = max(sorted(t_points.keys()), key=lambda t: (t_points[t], t))
            team_title_counts[champ_team] = team_title_counts.get(champ_team, 0) + 1
            for t, v in t_points.items():
                team_points_accum[t] = team_points_accum.get(t, 0.0) + float(v)
            print(f"   >> Constructor Champion: {champ_team}\n")

    # --- Summaries ---
    driver_probs = pd.Series(driver_title_counts, dtype=float) / float(runs)
    team_probs = pd.Series(team_title_counts, dtype=float) / float(runs)

    driver_points_mean = pd.Series(driver_points_accum, dtype=float) / float(runs)
    team_points_mean = pd.Series(team_points_accum, dtype=float) / float(runs)

    # Drop all-zero entries for tidy outputs
    driver_probs = driver_probs[driver_probs > 0].sort_values(ascending=False)
    team_probs = team_probs[team_probs > 0].sort_values(ascending=False)
    driver_points_mean = driver_points_mean[driver_points_mean > 0].sort_values(ascending=False)
    team_points_mean = team_points_mean[team_points_mean > 0].sort_values(ascending=False)

    # Write
    paths: Dict[str, Path] = {}
    paths["driver_title_png"] = OUT_DIR / "driver_title_probs.png"
    paths["constructor_title_png"] = OUT_DIR / "constructor_title_probs.png"
    _make_bar(driver_probs, "Driver Championship Probability", paths["driver_title_png"])
    _make_bar(team_probs, "Constructor Championship Probability", paths["constructor_title_png"])

    driver_points_mean.to_frame("mean_points").to_csv(OUT_DIR / "driver_points_mean.csv")
    team_points_mean.to_frame("mean_points").to_csv(OUT_DIR / "constructor_points_mean.csv")

    pd.DataFrame({
        "driver": driver_probs.index,
        "driver_title_prob": driver_probs.values,
    }).to_csv(OUT_DIR / "driver_title_probabilities.csv", index=False)

    pd.DataFrame({
        "team": team_probs.index,
        "constructor_title_prob": team_probs.values,
    }).to_csv(OUT_DIR / "constructor_title_probabilities.csv", index=False)

    return paths


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Monte Carlo championship simulator (equal car engine)")
    ap.add_argument("--runs", type=int, default=5000, help="Number of Monte Carlo seasons")
    ap.add_argument("--no-event-deltas", action="store_true", help="Use global ranking instead of per-event deltas")
    ap.add_argument("--base-lap", type=float, default=None, help="Override base lap seconds")
    ap.add_argument("--n-laps", type=int, default=None, help="Override race laps in sim engine")
    ap.add_argument("--dt", type=float, default=None, help="Time step (s) in sim engine")
    ap.add_argument("--noise-sd", type=float, default=None, help="Per-lap jitter SD (s)")
    ap.add_argument("--points", type=str, default="25,18,15,12,10,8,6,4,2,1", help="Comma list for points table")
    args = ap.parse_args()

    cfg = load_config("config/config.yaml")

    points = [int(x) for x in args.points.split(",") if x.strip()]

    paths = simulate_season(
        cfg,
        runs=args.runs,
        use_event_deltas=(not args.no_event_deltas),
        base_lap=args.base_lap,
        n_laps=args.n_laps,
        dt=args.dt,
        noise_sd=args.noise_sd,
        points_table=points,
    )

    # Console summary
    print("[INFO] Monte Carlo finished.")
    for k, p in paths.items():
        print(f"[INFO] Wrote {k}: {p}")


if __name__ == "__main__":
    main()
