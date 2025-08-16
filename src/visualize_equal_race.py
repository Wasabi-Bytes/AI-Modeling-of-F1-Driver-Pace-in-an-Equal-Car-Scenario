# src/visualize_equal_race.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from load_data import load_config, get_track_outline, get_recent_races, load_session, enable_cache

# ------------------- IO & Paths -------------------
PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "outputs" / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Config -------------------
BASE_LAP_SEC = 90.0    # equal-car base lap (seconds)
N_LAPS = 20            # laps to simulate
DT = 0.5               # seconds per animation frame
NOISE_SD = 0.08        # tiny jitter added to per-driver lap time
RANDOM_SEED = 42

# ------------------- Utilities -------------------
def _darken_hex(hex_color: str, factor: float = 0.75) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"

def _path_length(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    ds = np.sqrt(dx * dx + dy * dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, float(s[-1])

def _resample_path(xy: np.ndarray, n: int = 2000) -> np.ndarray:
    s, total = _path_length(xy)
    if total == 0:
        return np.repeat(xy[:1], n, axis=0)
    t = np.linspace(0, total, n)
    x = np.interp(t, s, xy[:, 0])
    y = np.interp(t, s, xy[:, 1])
    return np.column_stack([x, y])

# ------------------- Team colors (2025-ish palette) -------------------
TEAM_COLORS = {
    "Red Bull": "#1E41FF",
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#005AFF",
    "RB": "#2B4562",
    "Sauber": "#006F3C",
    "Haas": "#B6BABD",
    "UNKNOWN": "#888888",
}

# ------------------- Driver -> team mapping helpers -------------------
def _infer_driver_cols(df: pd.DataFrame) -> str:
    for c in ["driver", "Driver", "Abbreviation", "DriverNumber", "DriverId", "BroadcastName", "FullName"]:
        if c in df.columns:
            return c
    return df.columns[0]

def _get_driver_team_map_from_recent() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    try:
        cfg = load_config("config/config.yaml")
        if "cache_dir" in cfg:
            enable_cache(cfg["cache_dir"])
        races = get_recent_races(cfg)
        year, gp = races[0]["year"], races[0]["grand_prix"]
        laps, _ = load_session(year, gp, "R")
        if laps is None or len(laps) == 0:
            raise RuntimeError("No laps to infer mapping")

        dcol = _infer_driver_cols(laps)
        drivers = laps[[dcol]].dropna().astype(str).drop_duplicates()
        team_col = "Team" if "Team" in laps.columns else None
        name_col = None
        for cand in ["Abbreviation", "FullName", "BroadcastName", "Driver", "DriverId"]:
            if cand in laps.columns:
                name_col = cand
                break
        num_col = "DriverNumber" if "DriverNumber" in laps.columns else None

        team_map, name_map, num_map = {}, {}, {}
        for dr in drivers[dcol].tolist():
            sub = laps[laps[dcol].astype(str) == dr]
            if team_col:
                tser = sub[team_col].dropna()
                team = tser.iloc[0] if not tser.empty else "UNKNOWN"
            else:
                team = "UNKNOWN"
            if name_col:
                nser = sub[name_col].dropna()
                nm = nser.iloc[0] if not nser.empty else str(dr)
            else:
                nm = str(dr)
            if num_col:
                nnum = pd.to_numeric(sub[num_col], errors="coerce").dropna()
                nn = int(nnum.iloc[0]) if not nnum.empty else 999
            else:
                nn = 999
            team_map[str(dr)] = str(team)
            name_map[str(dr)] = str(nm)
            num_map[str(dr)] = int(nn)
        return team_map, name_map, num_map
    except Exception:
        return {}, {}, {}

# ------------------- Inputs -------------------
def load_driver_ranking() -> pd.DataFrame:
    path = PROJ / "outputs" / "aggregate" / "driver_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(f"driver_ranking.csv not found at {path}")
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    driver_col = low.get("driver", list(df.columns)[0])
    delta_col = low.get("agg_delta_s")
    se_col = low.get("agg_se_s")
    if delta_col is None:
        for cand in ["delta_s", "agg_delta", "equal_delta_s"]:
            if cand in low:
                delta_col = low[cand]
                break
    if delta_col is None:
        raise ValueError("Could not find aggregated delta column (agg_delta_s) in driver_ranking.csv")

    keep = [driver_col, delta_col] + ([se_col] if se_col else [])
    out = df[keep].copy()
    out.rename(columns={driver_col: "driver", delta_col: "agg_delta_s"}, inplace=True)
    if se_col:
        out.rename(columns={se_col: "agg_se_s"}, inplace=True)
    else:
        out["agg_se_s"] = np.nan
    out["driver"] = out["driver"].astype(str)
    out["agg_delta_s"] = pd.to_numeric(out["agg_delta_s"], errors="coerce")
    return out

def load_montreal_outline(cfg: dict) -> np.ndarray:
    if "cache_dir" in cfg:
        enable_cache(cfg["cache_dir"])
    df = get_track_outline(cfg)
    if df is None or df.empty:
        raise RuntimeError("Could not obtain track outline")
    xy = df[["x", "y"]].to_numpy(dtype=float)
    xy -= xy.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xy, axis=1))
    if scale > 0:
        xy /= scale
    xy = _resample_path(xy, n=2500)
    return xy

# ------------------- Colors per driver -------------------
def assign_colors(drivers: List[str], team_map: Dict[str, str], num_map: Dict[str, int]) -> Dict[str, str]:
    by_team: Dict[str, List[str]] = {}
    for d in drivers:
        t = team_map.get(d, "UNKNOWN")
        by_team.setdefault(t, []).append(d)

    colors = {}
    for team, ds in by_team.items():
        base = TEAM_COLORS.get(team, TEAM_COLORS["UNKNOWN"])
        if len(ds) == 1:
            colors[ds[0]] = base
        else:
            ds_sorted = sorted(ds, key=lambda z: num_map.get(z, 999))
            darker = _darken_hex(base, 0.75)
            colors[ds_sorted[0]] = darker  # senior
            for d in ds_sorted[1:]:
                colors[d] = base
    return colors

# ------------------- Simulation -------------------
def simulate_progress(
    ranking: pd.DataFrame,
    xy_path: np.ndarray,
    base_lap: float,
    n_laps: int,
    dt: float,
    noise_sd: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      positions: (T, D, 2) XY positions
      lap_key:   (T, D)   = completed_laps + lap_progress in [0, n_laps]
      leader_lap: (T,)    = leader's displayed lap number (1..n_laps)
      drivers:   list of driver ids (length D)
    """
    rng = np.random.default_rng(seed)
    drivers = ranking["driver"].tolist()
    deltas = ranking["agg_delta_s"].to_numpy(dtype=float)
    D = len(drivers)

    lap_times = base_lap + deltas + rng.normal(0, noise_sd, size=D)
    lap_times = np.clip(lap_times, 60.0, 180.0)

    path_pts = xy_path.shape[0]
    speed_pts_per_sec = path_pts / lap_times  # how many path points per second

    # Conservative max duration; trim later
    max_time = n_laps * float(np.max(lap_times))
    T_max = int(math.ceil(max_time / dt))
    t = np.arange(T_max) * dt  # (T_max,)

    # Advance in path points for each driver across time
    adv_pts = t[:, None] * speed_pts_per_sec[None, :]  # (T_max, D)
    completed_laps = np.floor(adv_pts / path_pts).astype(int)  # (T_max, D)
    lap_progress = (adv_pts % path_pts) / path_pts            # (T_max, D)
    lap_key = completed_laps + lap_progress                   # overall race progress

    # Stop when all drivers finished n_laps
    done_mask = (completed_laps >= n_laps)
    stop_idx = None
    for i in range(T_max):
        if np.all(done_mask[i]):
            stop_idx = i
            break
    if stop_idx is None:
        stop_idx = T_max - 1

    adv_pts = adv_pts[: stop_idx + 1]
    lap_key = lap_key[: stop_idx + 1]
    completed_laps = completed_laps[: stop_idx + 1]
    lap_progress = lap_progress[: stop_idx + 1]
    T = adv_pts.shape[0]

    # Positions
    idx = (adv_pts.astype(int) % path_pts)  # (T, D)
    positions = np.stack([xy_path[idx, 0], xy_path[idx, 1]], axis=2)  # (T, D, 2)

    # Leader lap to display (leader's completed laps + 1, capped)
    leader_completed = completed_laps.max(axis=1)  # (T,)
    leader_lap = np.minimum(n_laps, leader_completed + 1)

    return positions, lap_key, leader_lap, drivers

# ------------------- Build figure -------------------
def build_animation(
    positions: np.ndarray,
    lap_key: np.ndarray,
    leader_lap: np.ndarray,
    drivers: List[str],
    name_map: Dict[str, str],
    color_map: Dict[str, str],
    xy_path: np.ndarray,
    n_laps: int,
) -> go.Figure:
    """
    Plotly animated figure:
      - left: track outline + cars
      - right: live leaderboard (#1 leader ... #20 last)
      - top: Lap k / N annotation (updates live)
      - legend: driver color key
    """
    T, D, _ = positions.shape

    # Initial order (all same progress) -> keep original order
    order0 = list(range(D))
    labels = [str(dr).upper()[:3] for dr in drivers]
    names = [name_map.get(dr, dr) for dr in drivers]
    colors = [color_map.get(dr, "#888888") for dr in drivers]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.74, 0.26],
        specs=[[{"type": "xy"}, {"type": "scatter"}]],
        horizontal_spacing=0.05,
        subplot_titles=("Equal-Car Montreal Replay", "Leaderboard (live)"),
    )

    # Track outline (kept as first trace so frames index correctly)
    fig.add_trace(
        go.Scatter(
            x=xy_path[:, 0], y=xy_path[:, 1],
            mode="lines",
            line=dict(width=2, color="rgba(80,90,110,0.25)"),
            hoverinfo="skip",
            showlegend=False,
            name="Track",
        ),
        row=1, col=1,
    )

    # Cars (frame 0)
    x0 = positions[0, :, 0]
    y0 = positions[0, :, 1]
    fig.add_trace(
        go.Scatter(
            x=x0, y=y0,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=12, line=dict(width=1, color="#222"), color=colors),
            hovertext=names,
            hoverinfo="text",
            showlegend=True,
            name="Cars",
        ),
        row=1, col=1,
    )

    # Legend/key: one entry per driver with their color
    for nm, col in zip(names, colors):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=col, line=dict(width=1, color="#222")),
                name=nm,
                showlegend=True,
            ),
            row=1, col=1,
        )

    # Leaderboard (frame 0)
    rank_labels0 = [f"{i+1}. {labels[idx]}" for i, idx in enumerate(order0)]
    fig.add_trace(
        go.Scatter(
            x=[1] * D,
            y=list(range(D, 0, -1)),
            mode="text",
            text=rank_labels0,
            textfont=dict(size=14),
            showlegend=False,
            name="Leaderboard",
        ),
        row=1, col=2,
    )

    # Bounds and aspect
    pad = 0.1
    x_all = positions[:, :, 0].flatten()
    y_all = positions[:, :, 1].flatten()
    xmin, xmax = float(x_all.min()) - pad, float(x_all.max()) + pad
    ymin, ymax = float(y_all.min()) - pad, float(y_all.max()) + pad
    fig.update_xaxes(range=[xmin, xmax], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, range=[0, 2], row=1, col=2)
    fig.update_yaxes(showticklabels=False, range=[0, D + 1], row=1, col=2)

    # UI
    fig.update_layout(
        height=720,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(title="Drivers", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.47, y=1.08, xanchor="center",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"fromcurrent": True, "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]),
            ],
        )],
        title="Equal-Car Montreal Replay",
        annotations=[dict(
            text=f"Lap 1 / {n_laps}", showarrow=False, x=0.35, y=1.08, xref="paper", yref="paper",
            font=dict(size=14, color="#1f2d3d")
        )],
    )

    # Frames (keep trace order: 0=track, 1=cars, 2..(1+D)=legend, last=leaderboard)
    frames = []
    for ti in range(T):
        # Order by true race progress (completed_laps + lap_progress)
        order = np.argsort(-lap_key[ti])  # descending -> leader is #1
        rank_labels = [f"{i+1}. {labels[idx]}" for i, idx in enumerate(order)]
        # Lap display for leader
        lap_disp = int(np.clip(leader_lap[ti], 1, n_laps))

        frames.append(go.Frame(
            data=[
                go.Scatter(x=xy_path[:, 0], y=xy_path[:, 1]),                          # track (unchanged)
                go.Scatter(x=positions[ti, :, 0], y=positions[ti, :, 1], text=labels), # cars (update)
                *[go.Scatter() for _ in range(D)],                                      # legend placeholders
                go.Scatter(x=[1] * D, y=list(range(D, 0, -1)), text=rank_labels),       # leaderboard
            ],
            layout=go.Layout(annotations=[dict(
                text=f"Lap {lap_disp} / {n_laps}", showarrow=False, x=0.35, y=1.08, xref="paper", yref="paper",
                font=dict(size=14, color="#1f2d3d")
            )]),
            name=str(ti),
        ))
    fig.frames = frames
    return fig

# ------------------- Main -------------------
def main():
    np.random.seed(RANDOM_SEED)
    cfg = load_config("config/config.yaml")
    if "cache_dir" in cfg:
        enable_cache(cfg["cache_dir"])

    ranking = load_driver_ranking()
    xy = load_montreal_outline(cfg)

    team_map, name_map, num_map = _get_driver_team_map_from_recent()
    for dr in ranking["driver"].tolist():
        name_map.setdefault(dr, dr)
        team_map.setdefault(dr, "UNKNOWN")
        num_map.setdefault(dr, 999)

    color_map = assign_colors(ranking["driver"].tolist(), team_map, num_map)

    positions, lap_key, leader_lap, drivers = simulate_progress(
        ranking, xy, base_lap=BASE_LAP_SEC, n_laps=N_LAPS, dt=DT, noise_sd=NOISE_SD, seed=RANDOM_SEED
    )

    fig = build_animation(
        positions=positions,
        lap_key=lap_key,
        leader_lap=leader_lap,
        drivers=drivers,
        name_map=name_map,
        color_map=color_map,
        xy_path=xy,
        n_laps=N_LAPS,
    )

    out_path = OUT_DIR / "simulation.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"[INFO] Wrote visualization: {out_path}")

if __name__ == "__main__":
    main()
