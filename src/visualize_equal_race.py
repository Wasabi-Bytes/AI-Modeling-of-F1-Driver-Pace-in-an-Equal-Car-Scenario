# src/visualize_equal_race.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
BASE_LAP_SEC = 90.0
N_LAPS = 20
DT = 0.5
LAP_JITTER_SD = 0.12
DEGRADE_S_PER_LAP = 0.00
START_REACTION_SD = 0.12

USE_EVENT_SPECIFIC_DELTAS = True
TARGET_GP_SUBSTR = "canadian"  # e.g. "british", "austrian", "hungarian"

DRS_TAG = "ⓓ"  # shown in table when a car has DRS this lap

# DRS & blocking
DRS_DETECTION_THRESH_S = 1.0
DRS_ALPHA = 6.0
DRS_BETA = 1.5
DRS_GAMMA = 1.2
DRS_SLIPSTREAM_BONUS = 0.02
DRS_COOLDOWN_AFTER_SC_LAPS = 2
DRS_MIN_ENABLE_LAP = 3
DETECTION_OFFSET_FRACTION = 0.03

P_DEFEND = 0.55
BLOCKING_THRESH_S = 0.8
DEFENSE_TIME_COST = 0.06

# Safety Car / VSC
P_INCIDENT_PER_LAP = 0.03
SC_SHARE = 0.7
VSC_SPEED_FACTOR = 0.65
SC_SPEED_FACTOR = 0.50
SC_DURATION_LAPS_MINMAX = (1, 3)
VSC_DURATION_SEC_MINMAX = (12.0, 32.0)
SC_STAGGER_GAP_FRAC = 0.003

# Reliability (optional)
RELIABILITY_DNF_PER_LAP = 0.0

# Playback buttons
RANDOM_SEED = 42
PLAYBACK_CHOICES = [5.0, 10.0, 20.0]

# UI
SHOW_MARKER_LABELS = False
RC_WRAP_WIDTH = 44

# ------------------- Utilities -------------------
def _darken_hex(hex_color: str, factor: float = 0.75) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"

def _wrap_line(s: str, width: int = 44) -> str:
    out, line = [], []
    for word in str(s).split():
        if sum(len(w)+1 for w in line+[word]) > width:
            out.append(" ".join(line)); line = [word]
        else:
            line.append(word)
    if line: out.append(" ".join(line))
    return "<br>".join(out)

def _path_length(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    dx = np.diff(xy[:, 0]); dy = np.diff(xy[:, 1])
    ds = np.sqrt(dx * dx + dy * dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s, float(s[-1])

def _resample_path(xy: np.ndarray, n: int = 2000) -> np.ndarray:
    s, total = _path_length(xy)
    if total == 0: return np.repeat(xy[:1], n, axis=0)
    t = np.linspace(0, total, n)
    x = np.interp(t, s, xy[:, 0]); y = np.interp(t, s, xy[:, 1])
    return np.column_stack([x, y])

def _curvature(xy: np.ndarray) -> np.ndarray:
    P = xy.shape[0]
    if P < 3: return np.zeros(P, dtype=float)
    v = np.diff(xy, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    d_ang = np.diff(ang, prepend=ang[0])
    d_ang = np.abs(np.mod(d_ang + np.pi, 2*np.pi) - np.pi)
    k = 7
    pad = np.pad(d_ang, (k, k), mode="wrap")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.convolve(pad, ker, mode="same")[k:-k]
    sm_full = np.empty(P, dtype=float); sm_full[:-1] = sm; sm_full[-1] = sm[0]
    return sm_full

def _find_drs_zones(xy: np.ndarray, min_frac: float = 0.08, curv_thresh: float = 0.002) -> List[Tuple[int, int, int]]:
    P = xy.shape[0]
    curv = _curvature(xy)
    straight = curv < curv_thresh

    zones = []
    i = 0
    while i < P:
        if straight[i]:
            j = (i + 1) % P
            while j != i and straight[j]:
                j = (j + 1) % P
                if j == 0 and not straight[0]:
                    break
            end = (j - 1) % P
            zones.append((i, end))
            if j > i: i = j
            else: break
        else:
            i += 1

    if zones and zones[0][0] == 0 and zones[-1][1] == P - 1:
        merged = (zones[-1][0], zones[0][1])
        zones = [merged] + zones[1:-1]

    def _seg_len(a, b): return (b - a + 1) if a <= b else (P - a) + (b + 1)
    min_len = max(1, int(min_frac * P))
    zones = [(a, b) for (a, b) in zones if _seg_len(a, b) >= min_len]
    zones.sort(key=lambda z: _seg_len(z[0], z[1]), reverse=True)
    zones = zones[:3]

    det_off = int(max(1, DETECTION_OFFSET_FRACTION * P))
    out = []
    for a, b in zones:
        det = (a - det_off) % P
        out.append((a, b, det))
    return out

# ------------------- Team colors -------------------
TEAM_COLORS = {
    "Red Bull": "#1E41FF", "Ferrari": "#DC0000", "Mercedes": "#00D2BE",
    "McLaren": "#FF8700", "Aston Martin": "#006F62", "Alpine": "#0090FF",
    "Williams": "#005AFF", "RB": "#2B4562", "Sauber": "#006F3C",
    "Haas": "#B6BABD", "UNKNOWN": "#888888",
}

# ------------------- Driver mapping helpers -------------------
def _infer_driver_cols(df: pd.DataFrame) -> str:
    for c in ["driver","Driver","Abbreviation","DriverNumber","DriverId","BroadcastName","FullName"]:
        if c in df.columns: return c
    return df.columns[0]

def _get_driver_team_map_from_recent() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    try:
        cfg = load_config("config/config.yaml")
        if "cache_dir" in cfg: enable_cache(cfg["cache_dir"])
        races = get_recent_races(cfg)
        year, gp = races[0]["year"], races[0]["grand_prix"]
        laps, _ = load_session(year, gp, "R")
        if laps is None or len(laps) == 0: raise RuntimeError("No laps to infer mapping")
        dcol = _infer_driver_cols(laps)
        drivers = laps[[dcol]].dropna().astype(str).drop_duplicates()
        team_col = next((c for c in ["team","Team","Constructor","TeamName","ConstructorName"] if c in laps.columns), None)
        name_col = next((c for c in ["Abbreviation","FullName","BroadcastName","Driver","DriverId"] if c in laps.columns), None)
        num_col  = next((c for c in ["DriverNumber","Number","CarNumber"] if c in laps.columns), None)

        team_map, name_map, num_map = {}, {}, {}
        for dr in drivers[dcol].tolist():
            sub = laps[laps[dcol].astype(str) == dr]
            tser = sub[team_col].dropna() if team_col else pd.Series([], dtype=str)
            nser = sub[name_col].dropna() if name_col else pd.Series([], dtype=str)
            nnum = pd.to_numeric(sub[num_col], errors="coerce").dropna() if num_col else pd.Series([], dtype=float)
            team_map[str(dr)] = str(tser.iloc[0]) if not tser.empty else "UNKNOWN"
            name_map[str(dr)] = str(nser.iloc[0]) if not nser.empty else str(dr)
            num_map[str(dr)]  = int(nnum.iloc[0]) if not nnum.empty else 999
        return team_map, name_map, num_map
    except Exception:
        return {}, {}, {}

# ------------------- Inputs -------------------
def _load_weights_from_config(cfg: dict) -> Tuple[float, float]:
    wR = float(cfg.get("wR", cfg.get("wr", 0.6)))
    wQ = float(cfg.get("wQ", cfg.get("wq", 0.4)))
    return wR, wQ

def load_driver_ranking_global() -> pd.DataFrame:
    path = PROJ / "outputs" / "aggregate" / "driver_ranking.csv"
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    driver_col = low.get("driver", list(df.columns)[0])
    delta_col = low.get("agg_delta_s") or low.get("equal_delta_s") or low.get("delta_s") or low.get("agg_delta")
    se_col = low.get("agg_se_s")
    keep = [driver_col, delta_col] + ([se_col] if se_col else [])
    out = df[keep].copy()
    out.rename(columns={driver_col: "driver", delta_col: "agg_delta_s"}, inplace=True)
    if se_col: out.rename(columns={se_col: "agg_se_s"}, inplace=True)
    else: out["agg_se_s"] = np.nan
    out["driver"] = out["driver"].astype(str)
    out["agg_delta_s"] = pd.to_numeric(out["agg_delta_s"], errors="coerce").fillna(out["agg_delta_s"].median())
    out["agg_se_s"] = pd.to_numeric(out["agg_se_s"], errors="coerce")
    return out

def _list_event_metric_files() -> List[Path]:
    mdir = PROJ / "outputs" / "metrics"
    return sorted(mdir.glob("*-event_metrics.csv")) if mdir.exists() else []

def load_driver_ranking_event(cfg: dict, gp_substr: str) -> Optional[pd.DataFrame]:
    gp_substr = gp_substr.lower()
    files = _list_event_metric_files()
    if not files: return None
    pick = None
    for f in files[::-1]:
        if gp_substr in f.name.lower(): pick = f; break
    if pick is None: return None
    df = pd.read_csv(pick)
    low = {c.lower(): c for c in df.columns}
    drv = low.get("driver", list(df.columns)[0])
    rcol = low.get("race_delta_s"); qcol = low.get("quali_delta_s")
    if drv is None or rcol is None: return None
    wR, wQ = _load_weights_from_config(cfg)
    df["__evt_delta__"] = df[rcol].astype(float) + (wQ * df[qcol].astype(float) if qcol in df.columns else 0.0)
    out = df[[drv, "__evt_delta__"]].copy()
    out.columns = ["driver", "agg_delta_s"]
    out["driver"] = out["driver"].astype(str)
    out["agg_delta_s"] = pd.to_numeric(out["agg_delta_s"], errors="coerce")
    return out.dropna(subset=["agg_delta_s"])

def load_track_outline(cfg: dict) -> np.ndarray:
    if "cache_dir" in cfg: enable_cache(cfg["cache_dir"])
    df = get_track_outline(cfg)  # Montreal by default via config
    xy = df[["x", "y"]].to_numpy(dtype=float)
    xy -= xy.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xy, axis=1))
    if scale > 0: xy /= scale
    return _resample_path(xy, n=2500)

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
            colors[ds_sorted[0]] = _darken_hex(base, 0.75)
            for d in ds_sorted[1:]: colors[d] = base
    return colors

# ------------------- Simulation (same as previous version) -------------------
# (unchanged core logic; only the figure layout changed)
def simulate_progress(
    ranking: pd.DataFrame,
    xy_path: np.ndarray,
    base_lap: float,
    n_laps: int,
    dt: float,
    noise_sd: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    drivers = ranking["driver"].tolist()
    deltas = ranking["agg_delta_s"].to_numpy(dtype=float)
    D = len(drivers)

    P = xy_path.shape[0]
    zones = _find_drs_zones(xy_path)

    def _seg_len(a,b): return (b - a + 1) if a <= b else (P - a) + (b + 1)
    longest = max((_seg_len(a,b) for (a,b,_) in zones), default=int(0.12 * P))
    frac = longest / float(P)
    baseline = 0.12
    scale = max(0.6, min(2.0, 1.0 + 1.5 * (frac - baseline)))
    alpha_eff = DRS_ALPHA * scale
    det_eff = DRS_DETECTION_THRESH_S * (1.0 + 0.6 * (frac - baseline))

    base_driver = base_lap + deltas + rng.normal(0.0, noise_sd, size=D)
    lap_idx = np.arange(n_laps, dtype=float)[:, None]
    degrade = lap_idx * float(DEGRADE_S_PER_LAP)
    eps_lap = rng.normal(0.0, float(LAP_JITTER_SD), size=(n_laps, D))
    lap_times = base_driver[None, :] + degrade + eps_lap
    lap_times = np.clip(lap_times, 60.0, 180.0)
    speed_pts_per_sec = P / lap_times

    curr_pts = np.zeros(D, dtype=float)
    curr_lap = np.zeros(D, dtype=int)
    last_pts = np.zeros(D, dtype=float)
    defended_this_lap = np.zeros(D, dtype=bool)
    drs_eligible = {i: {} for i in range(len(zones))}
    drs_disabled_laps_remaining = DRS_MIN_ENABLE_LAP - 1

    phase = "GREEN"; vsc_ticks_remaining = 0; sc_laps_remaining = 0
    leader_last_completed = 0

    start_penalty = rng.normal(0.0, START_REACTION_SD, size=D)
    start_penalty_pts = start_penalty * speed_pts_per_sec[0, :]
    curr_pts = np.maximum(0.0, curr_pts - np.clip(start_penalty_pts, 0.0, P*0.02))

    positions_list = []; lapkey_list = []; leaderlap_list = []
    phase_flags = []; rc_texts = []; drs_on_flags = []; drs_banner = []
    event_log: List[Tuple[float,str]] = []
    orders = []; gaps_panel = []

    def _order_indices() -> np.ndarray:
        lk = curr_lap.astype(float) + (curr_pts / P)
        return np.argsort(-lk)

    def _gap_seconds(i_follow: int, i_lead: int) -> float:
        if curr_lap[i_follow] != curr_lap[i_lead]: return 99.0
        dp = (curr_pts[i_lead] - curr_pts[i_follow])
        if dp < 0: dp += P
        v = (P / (BASE_LAP_SEC + deltas[i_follow]))  # rough fallback if needed
        return float(dp / max(v, 1e-6))

    def _in_range(prev: float, now: float, target: int) -> bool:
        if now >= prev: return (prev <= target) and (target < now)
        else: return (target >= prev) or (target < now)

    def _is_in_zone(idx: float, z: Tuple[int,int,int]) -> bool:
        a,b,_ = z
        return (idx >= a and idx <= b) if a <= b else (idx >= a or idx <= b)

    def _apply_defense(i_lead: int):
        L = min(curr_lap[i_lead], n_laps-1)
        v = speed_pts_per_sec[L, i_lead]
        dist_pts = DEFENSE_TIME_COST * v
        new_pts = curr_pts[i_lead] - dist_pts
        if new_pts < 0: new_pts += P
        curr_pts[i_lead] = new_pts
        defended_this_lap[i_lead] = True

    def _attempt_pass(i_follow: int, i_lead: int, gap_at_det_s: float, tsec: float):
        L = min(curr_lap[i_follow], n_laps-1)
        pace_lead = lap_times[L, i_lead]
        pace_foll = lap_times[L, i_follow]
        delta_norm = (pace_lead - pace_foll) / BASE_LAP_SEC
        slip = DRS_SLIPSTREAM_BONUS if gap_at_det_s <= (det_eff * 1.2) else 0.0
        defended = 1.0 if defended_this_lap[i_lead] else 0.0
        logit = alpha_eff * delta_norm + DRS_BETA * slip - DRS_GAMMA * defended
        p = 1.0 / (1.0 + math.exp(-logit))
        if np.isfinite(p) and (np.random.random() < p):
            car_len = max(1, int(P * 0.0025))
            curr_pts[i_follow] = (curr_pts[i_lead] + car_len) % P
            curr_pts[i_lead] = (curr_pts[i_lead] - car_len) % P
            event_log.append((tsec, f"PASS: {drivers[i_follow]} → {drivers[i_lead]} (DRS)"))

    def _log(msg: str, tsec: float): event_log.append((tsec, msg))

    sim_time = 0.0
    max_time = n_laps * float(np.max(lap_times))
    T_max = int(math.ceil(max_time / DT)) + 1

    for _ in range(T_max):
        speeds = np.zeros(D, dtype=float)
        active = curr_lap < n_laps
        if active.any():
            L = np.clip(curr_lap, 0, n_laps-1)
            base_speed = speed_pts_per_sec[L, np.arange(D)]
            phase_factor = 1.0 if phase == "GREEN" else (VSC_SPEED_FACTOR if phase == "VSC" else SC_SPEED_FACTOR)
            speeds[active] = base_speed[active] * phase_factor

        if RELIABILITY_DNF_PER_LAP > 0 and np.random.random() < RELIABILITY_DNF_PER_LAP:
            act_idx = np.where(active)[0]
            if len(act_idx) > 0:
                dnf_i = int(np.random.choice(act_idx))
                curr_lap[dnf_i] = n_laps; speeds[dnf_i] = 0.0
                _log(f"DNF: {drivers[dnf_i]}", sim_time)

        last_pts = curr_pts.copy()
        curr_pts[active] = (curr_pts[active] + speeds[active] * DT) % P

        crossed = (curr_pts < last_pts) & active
        if crossed.any():
            curr_lap[crossed] += 1
            defended_this_lap[crossed] = False
            for z in drs_eligible.values():
                for di in list(z.keys()):
                    if crossed[di]: z.pop(di, None)

        leader_completed = int(curr_lap.max() if curr_lap.size else 0)

        if phase == "GREEN" and leader_completed > 0 and np.random.random() < P_INCIDENT_PER_LAP:
            if np.random.random() < SC_SHARE:
                phase = "SC"
                sc_laps_remaining = np.random.randint(SC_DURATION_LAPS_MINMAX[0], SC_DURATION_LAPS_MINMAX[1] + 1)
                order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
                lead = order[0]; base_pos = curr_pts[lead]
                for rank, di in enumerate(order):
                    gap_pts = int(SC_STAGGER_GAP_FRAC * P * rank)
                    pos = base_pos - gap_pts
                    while pos < 0: pos += P
                    curr_pts[di] = pos; curr_lap[di] = curr_lap[lead]
                for z in drs_eligible.values(): z.clear()
                defended_this_lap[:] = False
                _log("SAFETY CAR DEPLOYED (yellow)", sim_time)
            else:
                phase = "VSC"
                vsec = float(np.random.uniform(VSC_DURATION_SEC_MINMAX[0], VSC_DURATION_SEC_MINMAX[1]))
                vsc_ticks_remaining = max(1, int(vsec / DT))
                _log("VIRTUAL SAFETY CAR DEPLOYED", sim_time)

        if phase == "VSC":
            vsc_ticks_remaining -= 1
            if vsc_ticks_remaining <= 0:
                phase = "GREEN"; _log("VSC END — GREEN FLAG", sim_time)
        elif phase == "SC":
            if leader_completed > 0:
                sc_laps_remaining -= 1
                if sc_laps_remaining <= 0:
                    phase = "GREEN"
                    _log("SAFETY CAR IN — GREEN FLAG (DRS delayed)", sim_time)

        if phase == "GREEN":
            order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
            # blocking (outside zones)
            for k in range(1, len(order)):
                i_lead = order[k-1]; i_foll = order[k]
                if curr_lap[i_lead] >= n_laps or curr_lap[i_foll] >= n_laps: continue
                # quick gap approx (fallback)
                dp = curr_pts[i_lead] - curr_pts[i_foll]
                if dp < 0: dp += P
                v = P / (BASE_LAP_SEC + deltas[i_foll])
                gap_s = float(dp / max(v, 1e-6))
                in_any_zone = any(_is_in_zone(curr_pts[i_foll], z) for z in _find_drs_zones(xy_path))
                if (gap_s <= BLOCKING_THRESH_S) and (not in_any_zone) and (not defended_this_lap[i_lead]) and (np.random.random() < P_DEFEND):
                    _apply_defense(i_lead); _log(f"DEFENSE: {drivers[i_lead]} blocks {drivers[i_foll]} (+{DEFENSE_TIME_COST:.02f}s)", sim_time)

        # DRS global enable rule
        drs_enabled = (phase == "GREEN") and (leader_completed + 1 >= DRS_MIN_ENABLE_LAP)
        has_drs = np.zeros(D, dtype=bool)

        # DRS detection & passes
        if phase == "GREEN":
            order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
            for zid, (zs, ze, zd) in enumerate(zones):
                if drs_enabled:
                    for idx in order[1:]:
                        if curr_lap[idx] >= n_laps: continue
                        if _in_range(last_pts[idx], curr_pts[idx], zd):
                            pos = list(order).index(idx)
                            if pos > 0:
                                ahead = order[pos-1]
                                if curr_lap[ahead] >= n_laps: continue
                                # quick gap estimate at detection
                                dp = curr_pts[ahead] - curr_pts[idx]
                                if dp < 0: dp += P
                                v = P / (BASE_LAP_SEC + deltas[idx])
                                gap_s = float(dp / max(v, 1e-6))
                                if gap_s <= det_eff:
                                    drs_eligible[zid][idx] = (ahead, gap_s)
                                    has_drs[idx] = True
                                else:
                                    drs_eligible[zid].pop(idx, None)
                for idx, (ahead, gap_at_det) in list(drs_eligible[zid].items()):
                    if curr_lap[idx] >= n_laps or curr_lap[ahead] >= n_laps:
                        drs_eligible[zid].pop(idx, None); continue
                    if _in_range(last_pts[idx], curr_pts[idx], ze):
                        curr_order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
                        pos = list(curr_order).index(idx)
                        if pos > 0 and curr_order[pos-1] == ahead:
                            _attempt_pass(idx, ahead, gap_at_det, sim_time)
                        drs_eligible[zid].pop(idx, None)

        # Collect frame
        pos_idx = np.mod(curr_pts.astype(int), P)
        frame_xy = np.stack([xy_path[pos_idx, 0], xy_path[pos_idx, 1]], axis=1)
        positions_list.append(frame_xy)

        lk = curr_lap.astype(float) + (curr_pts / P)
        lapkey_list.append(lk)
        leaderlap_list.append(min(n_laps, int(curr_lap.max()) + 1))
        phase_flags.append(phase)
        drs_on_flags.append(bool(drs_enabled))
        drs_banner.append("DRS ENABLED" if drs_enabled else "DRS DISABLED")

        order = list(np.argsort(-lk)); orders.append(order)
        gaps = []
        for r, idx in enumerate(order):
            if r == 0:
                gaps.append("-")
            else:
                lap_diff = (curr_lap[order[0]] - curr_lap[idx])
                if lap_diff > 0:
                    gaps.append(f"+{lap_diff} Lap" + ("s" if lap_diff > 1 else ""))
                else:
                    dp = curr_pts[order[0]] - curr_pts[idx]
                    if dp < 0: dp += P
                    v = P / (BASE_LAP_SEC + deltas[idx])
                    g = float(dp / max(v, 1e-6))
                    gaps.append("+—" if g >= 98.0 else f"+{g:,.3f}")
                if drs_enabled and has_drs[idx]: gaps[-1] += f" {DRS_TAG}"
        gaps_panel.append(gaps)

        recent = [f"t={sim_time:5.1f}s  {m}" for (t, m) in event_log if t <= sim_time]
        wrapped = [_wrap_line(r, width=RC_WRAP_WIDTH) for r in recent[-8:]]
        rc_texts.append("<br>".join(wrapped) if wrapped else "—")

        sim_time += DT
        if (curr_lap >= n_laps).all(): break

    positions = np.stack(positions_list, axis=0)
    lap_key = np.stack(lapkey_list, axis=0)
    leader_lap = np.array(leaderlap_list, dtype=int)
    return (positions, lap_key, leader_lap, drivers,
            phase_flags, rc_texts, drs_on_flags, drs_banner,
            orders, gaps_panel, zones, alpha_eff, det_eff)

# ------------------- Figure -------------------
def _flag_style(phase: str) -> Tuple[str, str]:
    if phase == "SC":  return ("SAFETY CAR", "#ffd400")
    if phase == "VSC": return ("VIRTUAL SAFETY CAR", "#ffb347")
    return ("GREEN FLAG", "#2ecc71")

def _track_color(phase: str) -> str:
    if phase == "SC":  return "rgba(255,212,0,0.70)"
    if phase == "VSC": return "rgba(255,165,0,0.45)"
    return "rgba(80,90,110,0.25)"

def build_animation(
    positions, lap_key, leader_lap, drivers, name_map, color_map, xy_path, n_laps,
    phase_flags, rc_texts, drs_on, drs_banner, orders, gaps_panel, zones, alpha_eff, det_eff
) -> go.Figure:
    T, D, _ = positions.shape
    labels = [str(dr).upper()[:3] for dr in drivers]
    names = [name_map.get(dr, dr) for dr in drivers]
    colors = [color_map.get(dr, "#888888") for dr in drivers]

    # --- Layout: 2 rows x 2 cols. Left col spans both rows for the TRACK.
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy", "rowspan": 2}, {"type": "table"}],
               [None, {"type": "scatter"}]],
        column_widths=[0.62, 0.38],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
        subplot_titles=("Equal-Car Replay", "Leaderboard (live)", "Race Control"),
    )

    # TRACK (trace order 0)
    track_col0 = _track_color(phase_flags[0] if len(phase_flags) else "GREEN")
    fig.add_trace(go.Scatter(x=xy_path[:,0], y=xy_path[:,1], mode="lines",
                             line=dict(width=2, color=track_col0),
                             hoverinfo="skip", showlegend=False, name="Track"), row=1, col=1)

    # CARS (trace order 1)
    mode0 = "markers+text" if SHOW_MARKER_LABELS else "markers"
    fig.add_trace(go.Scatter(x=positions[0,:,0], y=positions[0,:,1], mode=mode0,
                             text=(labels if SHOW_MARKER_LABELS else None), textposition="top center",
                             marker=dict(size=12, line=dict(width=1, color="#222"), color=colors),
                             hovertext=names, hoverinfo="text", showlegend=True, name="Cars"), row=1, col=1)

    # SC marker placeholder (trace order 2)
    fig.add_trace(go.Scatter(), row=1, col=1)

    # Legend entries (trace orders 3..(2+D))
    for nm, col in zip(names, colors):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=col, line=dict(width=1, color="#222")),
                                 name=nm, showlegend=True), row=1, col=1)

    # TABLE (trace order 3+D)
    order0 = orders[0]
    table_positions = list(range(1, D+1))
    table_drivers = [labels[i] for i in order0]
    table_gaps = gaps_panel[0]
    table = go.Table(
        columnwidth=[0.20, 0.40, 0.40],
        header=dict(values=["<b>Position</b>", "<b>Driver</b>", "<b>Time</b>"],
                    fill_color="#2b2f3a", font=dict(color="white", size=13), align="left"),
        cells=dict(values=[table_positions, table_drivers, table_gaps],
                   fill_color="#0f1720", font=dict(color="white", size=12), align="left",
                   height=24)
    )
    fig.add_trace(table, row=1, col=2)

    # RACE CONTROL (trace order 4+D)
    rc0 = rc_texts[0] if len(rc_texts) else "—"
    fig.add_trace(go.Scatter(x=[0.02], y=[0.98], mode="text", text=[rc0],
                             textposition="top left", textfont=dict(size=12),
                             showlegend=False, name="Race Control"),
                  row=2, col=2)

    # DRS zones & detection markers (STATIC overlays; added AFTER animated traces so no index shift)
    for (zs, ze, zd) in zones:
        fig.add_trace(go.Scatter(x=[xy_path[zd,0]], y=[xy_path[zd,1]], mode="markers",
                                 marker=dict(size=8, symbol="triangle-up", color="rgba(30,144,255,0.95)"),
                                 showlegend=False, name="DRS detect"), row=1, col=1)
        if zs <= ze:
            xs = xy_path[zs:ze+1, 0]; ys = xy_path[zs:ze+1, 1]
        else:
            xs = np.r_[xy_path[zs:,0], xy_path[:ze+1,0]]
            ys = np.r_[xy_path[zs:,1], xy_path[:ze+1,1]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(width=6, color="rgba(30,144,255,0.18)"),
                                 showlegend=False, name="DRS zone"), row=1, col=1)

    # Axes/layout
    pad = 0.08
    x_all = positions[:,:,0].ravel(); y_all = positions[:,:,1].ravel()
    xmin, xmax = float(x_all.min())-pad, float(x_all.max())+pad
    ymin, ymax = float(y_all.min())-pad, float(y_all.max())+pad
    fig.update_xaxes(range=[xmin, xmax], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1,
                     showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, range=[0, 1], row=2, col=2)
    fig.update_yaxes(showticklabels=False, range=[0, 1], row=2, col=2)

    # Playback buttons (more top margin so they never clip)
    def _btn(speed: float, label: str):
        frame_ms = max(10, int(1000 * DT / speed))
        return dict(label=label, method="animate",
                    args=[None, {"fromcurrent": True,
                                 "frame": {"duration": frame_ms, "redraw": True},
                                 "transition": {"duration": 0}}])
    buttons = [_btn(s, f"Play {int(s)}×") for s in PLAYBACK_CHOICES]
    buttons.append(dict(label="Pause", method="animate",
                        args=[[None], {"mode": "immediate",
                                       "frame": {"duration": 0, "redraw": False},
                                       "transition": {"duration": 0}}]))

    banner_text, banner_bg = _flag_style(phase_flags[0] if len(phase_flags) else "GREEN")
    drs_tag = drs_banner[0] if len(drs_banner) else "DRS DISABLED"
    fig.update_layout(
        height=860,
        margin=dict(l=16, r=16, t=140, b=20),
        legend=dict(title="Drivers", x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.6)"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.48, y=1.16, xanchor="center", buttons=buttons)],
        title=f"Equal-Car Replay  •  DRSα={alpha_eff:.2f}  det≈{det_eff:.2f}s",
        annotations=[
            dict(text=f"Lap 1 / {N_LAPS}", showarrow=False, x=0.21, y=1.17, xref="paper", yref="paper",
                 font=dict(size=14, color="#1f2d3d")),
            dict(text=banner_text, showarrow=False, x=0.50, y=1.17, xref="paper", yref="paper",
                 font=dict(size=14, color="#1f2d3d"), bgcolor=banner_bg),
            dict(text=drs_tag, showarrow=False, x=0.78, y=1.17, xref="paper", yref="paper",
                 font=dict(size=14, color="#1f2d3d"),
                 bgcolor="#e8f1ff" if ("ENABLED" in drs_tag) else "#ffd6d6"),
        ],
        paper_bgcolor="#f2f6fb"
    )

    # Frames (trace order must match initial add_trace order for animated ones)
    animated_prefix_count = 3 + D   # track(0), cars(1), sc(2), legends(3..2+D) -> then table, RC
    frames = []
    for ti in range(T):
        order = orders[ti]
        table_positions = list(range(1, D+1))
        table_drivers = [labels[i] for i in order]
        table_gaps = gaps_panel[ti]
        lap_disp = int(np.clip(leader_lap[ti], 1, N_LAPS))
        banner_text, banner_bg = _flag_style(phase_flags[ti])
        track_col = _track_color(phase_flags[ti])
        sc_marker = go.Scatter(x=[xy_path[0,0]], y=[xy_path[0,1]], mode="markers",
                               marker=dict(size=14, color="rgba(255,212,0,0.9)", symbol="square"),
                               showlegend=False, name="SC") if phase_flags[ti]=="SC" else go.Scatter()

        # Only update the animated traces: track, cars, sc, (keep legend placeholders as empty),
        # then table + race control.
        frame_data = [
            go.Scatter(x=xy_path[:,0], y=xy_path[:,1], line=dict(width=2, color=track_col)),
            go.Scatter(x=positions[ti,:,0], y=positions[ti,:,1],
                       mode=("markers+text" if SHOW_MARKER_LABELS else "markers"),
                       text=(labels if SHOW_MARKER_LABELS else None)),
            sc_marker,
        ] + [go.Scatter() for _ in range(D)] + [
            go.Table(columnwidth=[0.20,0.40,0.40],
                     header=dict(values=["<b>Position</b>","<b>Driver</b>","<b>Time</b>"],
                                 fill_color="#2b2f3a", font=dict(color="white", size=13), align="left"),
                     cells=dict(values=[table_positions, table_drivers, table_gaps],
                                fill_color="#0f1720", font=dict(color="white", size=12), align="left", height=24)),
            go.Scatter(x=[0.02], y=[0.98], mode="text", text=[rc_texts[ti]],
                       textposition="top left", textfont=dict(size=12)),
        ]

        frames.append(go.Frame(
            data=frame_data,
            layout=go.Layout(annotations=[
                dict(text=f"Lap {lap_disp} / {N_LAPS}", showarrow=False, x=0.21, y=1.17, xref="paper", yref="paper",
                     font=dict(size=14, color="#1f2d3d")),
                dict(text=banner_text, showarrow=False, x=0.50, y=1.17, xref="paper", yref="paper",
                     font=dict(size=14, color="#1f2d3d"), bgcolor=banner_bg),
                dict(text=("DRS ENABLED" if "ENABLED" in drs_banner[ti] else "DRS DISABLED"),
                     showarrow=False, x=0.78, y=1.17, xref="paper", yref="paper",
                     font=dict(size=14, color="#1f2d3d"),
                     bgcolor="#e8f1ff" if ("ENABLED" in drs_banner[ti]) else "#ffd6d6"),
            ]),
            name=str(ti),
        ))
    fig.frames = frames
    return fig

# ------------------- Main -------------------
def main():
    np.random.seed(RANDOM_SEED)
    cfg = load_config("config/config.yaml")
    if "cache_dir" in cfg: enable_cache(cfg["cache_dir"])

    if USE_EVENT_SPECIFIC_DELTAS:
        ranking = load_driver_ranking_event(cfg, TARGET_GP_SUBSTR)
        if ranking is None or ranking.empty:
            print("[WARN] Could not load per-event deltas; falling back to global aggregates.")
            ranking = load_driver_ranking_global()
    else:
        ranking = load_driver_ranking_global()

    xy = load_track_outline(cfg)

    team_map, name_map, num_map = _get_driver_team_map_from_recent()
    for dr in ranking["driver"].tolist():
        name_map.setdefault(dr, dr); team_map.setdefault(dr, "UNKNOWN"); num_map.setdefault(dr, 999)
    color_map = assign_colors(ranking["driver"].tolist(), team_map, num_map)

    (positions, lap_key, leader_lap, drivers,
     phase_flags, rc_texts, drs_on, drs_banner,
     orders, gaps_panel, zones, alpha_eff, det_eff) = simulate_progress(
        ranking, xy, base_lap=BASE_LAP_SEC, n_laps=N_LAPS, dt=DT,
        noise_sd=LAP_JITTER_SD, seed=RANDOM_SEED
    )

    fig = build_animation(
        positions, lap_key, leader_lap, drivers, name_map, color_map, xy, N_LAPS,
        phase_flags, rc_texts, drs_on, drs_banner, orders, gaps_panel, zones, alpha_eff, det_eff
    )

    out_path = OUT_DIR / "simulation.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"[INFO] Wrote visualization: {out_path}")

if __name__ == "__main__":
    main()
