from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Reuse functions from the single-race engine
from visualize_equal_race import (
    load_config, enable_cache, _cfg_get,
    load_driver_ranking_event, load_driver_ranking_global, _load_viz_track_meta,
    _load_weather_summary_for_viz, load_track_outline,
    simulate_progress, assign_colors, _get_driver_team_map_from_recent,
)

PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "outputs" / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard FIA points for top 10 (no FL, no sprint)
F1_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

@dataclass
class TrackSpec:
    year: int
    grand_prix: str
    label: Optional[str] = None  # override tile title if desired

def _driver_code(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return "???"
    tok = s.replace("-", " ").split()
    if len(tok) == 1:
        return tok[0][:3].upper()
    code = (tok[0][0] + tok[-1][0] + (tok[-1][1] if len(tok[-1]) > 1 else "")).upper()
    return code[:3]

def _points_for_position(pos: int) -> int:
    return F1_POINTS[pos] if 0 <= pos < len(F1_POINTS) else 0

def _deterministic_mode(cfg: dict) -> bool:
    """When True: noise=0, DNFs off, incidents off, deterministic passes."""
    add_noise = bool(_cfg_get(cfg, ["simulation", "add_noise"], True))
    return not add_noise

def _sim_knobs_for_mode(cfg: dict) -> Tuple[float, float, bool, bool]:
    """
    Returns (lap_jitter_sd, incident_rate, disable_dnfs, deterministic_pass)
    honoring the 'deterministic' (add_noise: false) mode contract.
    """
    deterministic = _deterministic_mode(cfg)
    if deterministic:
        return (0.0, 0.0, True, True)
    vizsec = _cfg_get(cfg, ["visualize_equal_race"], {}) or {}
    lap_sd = float(_cfg_get(vizsec, ["lap_jitter_sd"], 0.12))
    incident_rate = float(_cfg_get(vizsec, ["incident_rate"], 0.03))
    disable_dnfs = bool(_cfg_get(vizsec, ["disable_dnfs"], False))
    return (lap_sd, incident_rate, disable_dnfs, False)

def _season_tracks_from_metrics(cfg: dict, target_year: Optional[int]) -> List[TrackSpec]:
    """
    Build the full schedule from outputs/metrics/all_events_metrics.csv.
    If target_year is None, use the most recent year found.
    Ordering is by event_idx (or first appearance).
    """
    src = PROJ / "outputs" / "metrics" / "all_events_metrics.csv"
    if not src.exists():
        return []
    df = pd.read_csv(src)
    if "year" not in df.columns or "gp" not in df.columns:
        return []
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year", "gp"])
    if df.empty:
        return []
    # choose the year
    if target_year is None:
        target_year = int(df["year"].max())
    else:
        target_year = int(target_year)
    season = df[df["year"] == target_year].copy()
    if season.empty:
        # fallback: use most recent available
        target_year = int(df["year"].max())
        season = df[df["year"] == target_year].copy()

    if "event_idx" not in season.columns:
        season = season.copy()
        season["event_key"] = season["year"].astype(str) + " - " + season["gp"].astype(str)
        order = (
            season[["event_key"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .reset_index(names="event_idx")
        )
        season = season.merge(order, on="event_key", how="left")

    order_tbl = (
        season[["gp", "event_idx"]]
        .drop_duplicates(subset=["gp"])
        .sort_values("event_idx", ignore_index=True)
    )
    tracks = [TrackSpec(target_year, str(gp)) for gp in order_tbl["gp"].tolist()]
    return tracks

def _champ_tracks_from_cfg(cfg: dict) -> List[TrackSpec]:
    """
    Priority:
      1) championship.use_all_tracks: true -> infer full schedule from metrics for championship.year (or latest)
      2) championship.tracks list (explicit)
      3) tiny fallback sample
    """
    use_all = bool(_cfg_get(cfg, ["championship", "use_all_tracks"], True))
    year = _cfg_get(cfg, ["championship", "year"], None)
    if use_all:
        tracks = _season_tracks_from_metrics(cfg, year)
        if tracks:
            return tracks

    raw = _cfg_get(cfg, ["championship", "tracks"], None)
    if isinstance(raw, list) and raw:
        out = []
        for r in raw:
            try:
                y = int(r.get("year"))
                gp = str(r.get("grand_prix"))
                lbl = r.get("label")
                out.append(TrackSpec(y, gp, lbl))
            except Exception:
                pass
        if out:
            return out

    # Fallback sample
    return [
        TrackSpec(2024, "Bahrain"),
        TrackSpec(2024, "Saudi Arabian"),
        TrackSpec(2024, "Australian"),
    ]

def _exclude_drivers(cfg: dict, ranking: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude drivers by name or code.
    - Always excludes DOO (as requested).
    - Also honors config.simulation.exclude_drivers list (names or codes).
    """
    cfg_list = _cfg_get(cfg, ["simulation", "exclude_drivers"], []) or []
    exclude = {str(x).strip() for x in cfg_list}
    exclude.add("DOO")  # hard requirement from user
    if exclude and not ranking.empty:
        drivers = ranking["driver"].astype(str)
        codes = drivers.map(_driver_code)
        keep = ~drivers.isin(exclude) & ~codes.isin(exclude)
        ranking = ranking[keep].reset_index(drop=True)
    return ranking

def _load_ranking_for_track(cfg_base: dict, year: int, gp: str) -> pd.DataFrame:
    """
    Clone cfg with target viz_track, try event deltas for that GP, else global ranking.
    Applies simulation.exclude_drivers (e.g., DOO).
    """
    cfg = json.loads(json.dumps(cfg_base))  # deep copy
    cfg.setdefault("viz_track", {})
    cfg["viz_track"]["year"] = year
    cfg["viz_track"]["grand_prix"] = gp

    ranking = load_driver_ranking_event(cfg, str(gp).lower())
    if ranking is None or ranking.empty:
        ranking = load_driver_ranking_global(cfg)

    return _exclude_drivers(cfg, ranking)

def _build_tile_axes(fig: go.Figure, row: int, col: int):
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False, row=row, col=col)

def _track_bounds(xy: np.ndarray, pad: float = 0.08) -> Tuple[float, float, float, float]:
    xmin, xmax = float(xy[:,0].min()), float(xy[:,0].max())
    ymin, ymax = float(xy[:,1].min()), float(xy[:,1].max())
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)

def run_championship():
    cfg = load_config("config/config.yaml")
    if "cache_dir" in cfg:
        enable_cache(cfg["cache_dir"])

    tracks = _champ_tracks_from_cfg(cfg)
    if not tracks:
        print("[WARN] Could not infer schedule; configure championship.tracks or ensure outputs/metrics/all_events_metrics.csv exists.")
        return

    # Mode knobs
    lap_sd, incident_rate, disable_dnfs, deterministic_pass = _sim_knobs_for_mode(cfg)
    vizsec = _cfg_get(cfg, ["visualize_equal_race"], {}) or {}
    base_lap = float(_cfg_get(vizsec, ["base_lap_sec"], 90.0))
    n_laps = int(_cfg_get(vizsec, ["n_laps"], 20))
    dt = float(_cfg_get(vizsec, ["dt"], 0.5))
    seed = int(_cfg_get(vizsec, ["seed"], 42))
    run_idx = int(_cfg_get(vizsec, ["run_idx"], 0))

    # Consistent coloring across tiles
    team_map, name_map, num_map = _get_driver_team_map_from_recent()

    # Storage for each round
    rounds = []  # list of dicts per event
    event_results: List[pd.DataFrame] = []

    max_T = 0  # longest animation length across rounds

    for spec in tracks:
        # Per-track cfg
        cfg_track = json.loads(json.dumps(cfg))
        cfg_track.setdefault("viz_track", {})
        cfg_track["viz_track"]["year"] = spec.year
        cfg_track["viz_track"]["grand_prix"] = spec.grand_prix

        # Ranking (excluding DOO etc.)
        ranking = _load_ranking_for_track(cfg_track, spec.year, spec.grand_prix)

        # Track & meta
        meta = _load_viz_track_meta(cfg_track)
        weather_summary = _load_weather_summary_for_viz(cfg_track)
        xy = load_track_outline(cfg_track)

        # Colors / labels
        for dr in ranking["driver"].tolist():
            team_map.setdefault(dr, "UNKNOWN")
            num_map.setdefault(dr, 999)
        color_map = assign_colors(ranking["driver"].tolist(), team_map, num_map)
        colors = [color_map.get(dr, "#888888") for dr in ranking["driver"].tolist()]
        labels = [_driver_code(dr) for dr in ranking["driver"].tolist()]

        # Simulate the race
        (positions, lap_key, leader_lap, drivers,
         phase_flags, rc_texts, drs_on, drs_banner,
         orders, gaps_panel, zones, alpha_eff, det_eff, stats) = simulate_progress(
            ranking=ranking,
            xy_path=xy,
            base_lap=base_lap,
            n_laps=n_laps,
            dt=dt,
            noise_sd=lap_sd,
            seed=seed,
            cfg=cfg_track,
            meta=meta,
            weather_summary=weather_summary,
            incident_rate=incident_rate,
            disable_dnfs=disable_dnfs,
            start_gain_sd_override=(0.0 if _deterministic_mode(cfg) else None),
            run_idx=run_idx,
            deterministic_pass=deterministic_pass,
        )

        T, D, _ = positions.shape
        max_T = max(max_T, T)

        finish_idx = stats["finish_order"]
        title = spec.label or f"{spec.year} {spec.grand_prix}"

        # Event points
        rows = []
        for pos, idx in enumerate(finish_idx, start=1):
            dr = drivers[idx]
            pts = _points_for_position(pos - 1)
            rows.append({"event": title, "position": pos, "driver": dr, "points": pts})
        event_results.append(pd.DataFrame(rows, columns=["event", "position", "driver", "points"]))

        # Save per-round package for visualization frames
        rounds.append(dict(
            title=title,
            xy=xy,
            positions=positions,  # (T, D, 2)
            drivers=drivers,
            labels=labels,
            colors=colors,
        ))

    # Aggregate standings across all rounds
    results = pd.concat(event_results, ignore_index=True)
    standings = (
        results.groupby("driver", as_index=False)["points"]
        .sum()
        .sort_values(["points", "driver"], ascending=[False, True])
        .reset_index(drop=True)
    )
    standings["rank"] = np.arange(1, len(standings) + 1)

    # Save CSVs
    results_path = OUT_DIR / "championship_event_results.csv"
    standings_path = OUT_DIR / "championship_standings.csv"
    results.to_csv(results_path, index=False)
    standings.to_csv(standings_path, index=False)
    print(f"[INFO] Wrote: {results_path}")
    print(f"[INFO] Wrote: {standings_path}")

    # ---------- Animated tile grid ----------
    n_tiles = len(rounds)
    ncols = min(4, max(2, int(np.ceil(np.sqrt(n_tiles)))))
    nrows = int(np.ceil(n_tiles / ncols))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{"type": "xy"} for _ in range(ncols)] for _ in range(nrows)],
        horizontal_spacing=0.05, vertical_spacing=0.07,
        subplot_titles=[r["title"] for r in rounds] + [""] * (nrows * ncols - n_tiles),
    )

    # We will place 2 traces per tile: (0) static track outline, (1) moving markers
    trace_handles = []  # [(row, col, track_trace_idx, marker_trace_idx)]
    global_trace_idx = 0

    for k, rnd in enumerate(rounds):
        r = (k // ncols) + 1
        c = (k % ncols) + 1
        xy = rnd["xy"]

        # Track outline
        fig.add_trace(
            go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines",
                       line=dict(width=2, color="rgba(80,90,110,0.25)"),
                       hoverinfo="skip", showlegend=False,
                       name=f"Track {rnd['title']}"),
            row=r, col=c
        )
        track_trace_idx = global_trace_idx
        global_trace_idx += 1

        # Initial car positions (t=0)
        x0 = rnd["positions"][0, :, 0]
        y0 = rnd["positions"][0, :, 1]
        fig.add_trace(
            go.Scatter(
                x=x0, y=y0,
                mode="markers+text",
                text=rnd["labels"],
                textposition="middle center",
                marker=dict(size=14, color=rnd["colors"], line=dict(width=1, color="#222")),
                hovertext=rnd["drivers"],
                hoverinfo="text",
                showlegend=False,
                name=f"Cars {rnd['title']}",
            ),
            row=r, col=c
        )
        marker_trace_idx = global_trace_idx
        global_trace_idx += 1

        _build_tile_axes(fig, r, c)
        xmin, xmax, ymin, ymax = _track_bounds(xy, pad=0.08)
        fig.update_xaxes(range=[xmin, xmax], constrain="domain", row=r, col=c)
        fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1, row=r, col=c)

        trace_handles.append((r, c, track_trace_idx, marker_trace_idx))

    # ----- Build frames -----
    # We'll only update the marker traces each frame and reference them by index.
    marker_indices = [m_idx for (_r, _c, _t_idx, m_idx) in trace_handles]

    frames = []
    for ti in range(max_T):
        data_this = []
        for k, rnd in enumerate(rounds):
            pos = rnd["positions"]
            T_k = pos.shape[0]
            t_use = min(ti, T_k - 1)
            x = pos[t_use, :, 0]
            y = pos[t_use, :, 1]
            # Only update x/y; other style props come from the initial trace
            data_this.append(go.Scatter(x=x, y=y))
        frames.append(go.Frame(data=data_this, traces=marker_indices, name=str(ti)))

    fig.frames = frames

    # Play controls
    def _btn(speed: float, label: str):
        # dt is the race simulation step; we map speed to ~visual FPS by inverse scaling
        frame_ms = max(10, int(1000 * dt / speed))
        return dict(
            label=label,
            method="animate",
            args=[None, {"fromcurrent": True, "frame": {"duration": frame_ms, "redraw": True}, "transition": {"duration": 0}}],
        )

    buttons = [
        _btn(5.0, "Play 5×"),
        _btn(10.0, "Play 10×"),
        _btn(20.0, "Play 20×"),
        dict(label="Pause", method="animate",
             args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]),
    ]

    mode_txt = "Deterministic Equal-Car Championship" if _deterministic_mode(cfg) else "Stochastic Equal-Car Championship"
    fig.update_layout(
        height=max(900, 280 * int(np.ceil(n_tiles / min(4, n_tiles)))),
        title=f"{mode_txt} — {n_tiles} Rounds",
        paper_bgcolor="#f6f9fe",
        plot_bgcolor="#f6f9fe",
        showlegend=False,
        margin=dict(l=20, r=20, t=90, b=90),
        updatemenus=[dict(type="buttons", showactive=False, x=0.5, y=1.12, xanchor="center", buttons=buttons)],
    )

    # Footer with final standings (top 14)
    top = min(14, len(standings))
    fig.add_annotation(
        text=(
            f"<b>Championship Standings (Top {top})</b><br>"
            + "<br>".join([f"{int(standings.iloc[i]['rank']):>2}. {standings.iloc[i]['driver']} — {int(standings.iloc[i]['points'])}"
                           for i in range(top)])
        ),
        x=0.5, y=-0.06, xref="paper", yref="paper", showarrow=False,
        font=dict(size=13), align="center"
    )

    out_html = OUT_DIR / "championship.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn", auto_open=False)
    print(f"[INFO] Wrote visualization: {out_html}")

if __name__ == "__main__":
    run_championship()
