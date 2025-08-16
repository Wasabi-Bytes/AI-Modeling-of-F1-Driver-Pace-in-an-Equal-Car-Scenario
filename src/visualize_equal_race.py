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
# Pace & sim
BASE_LAP_SEC = 90.0        # equal-car base lap (seconds)
N_LAPS = 20                # laps to simulate
DT = 0.5                   # seconds per animation frame (simulation tick)
LAP_JITTER_SD = 0.12       # per-lap random variation (sec)
DEGRADE_S_PER_LAP = 0.00   # tyre fade per lap (sec)
START_REACTION_SD = 0.12   # grid launch / restart reaction noise (sec)

# DRS & blocking
DRS_DETECTION_THRESH_S = 1.0      # within 1.0s at detection = eligible
DRS_SLIPSTREAM_BONUS = 0.02       # extra term in pass probability
DRS_ALPHA = 6.0                   # weight on normalized pace delta
DRS_BETA = 1.5                    # weight on slipstream
DRS_GAMMA = 1.2                   # penalty if leader defended earlier this lap
DRS_COOLDOWN_AFTER_SC_LAPS = 2    # DRS disabled this many full laps after SC
DRS_MIN_ENABLE_LAP = 3            # DRS disabled for first 2 laps
DETECTION_OFFSET_FRACTION = 0.03  # detection point ~3% of lap before zone

# Blocking outside DRS
P_DEFEND = 0.55
BLOCKING_THRESH_S = 0.8           # if follower within this gap, leader may defend
DEFENSE_TIME_COST = 0.06          # defending adds ~0.06s cost to leader

# Safety Car / VSC (stochastic)
P_INCIDENT_PER_LAP = 0.03         # incident chance on a GREEN lap (evaluated when leader completes a lap)
SC_SHARE = 0.7                    # % incidents that are SC (else VSC)
VSC_SPEED_FACTOR = 0.65
SC_SPEED_FACTOR = 0.50
SC_DURATION_LAPS_MINMAX = (1, 3)  # leader-led laps behind SC
VSC_DURATION_SEC_MINMAX = (12.0, 32.0)
SC_STAGGER_GAP_FRAC = 0.003       # small path fraction gap when bunching

# Reliability (off by default)
RELIABILITY_DNF_PER_LAP = 0.0

# Playback
RANDOM_SEED = 42
PLAYBACK_MULTIPLIER = 10.0        # ~10× race-time vs real-time playback

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

def _curvature(xy: np.ndarray) -> np.ndarray:
    """
    Smoothed curvature proxy with SAME length as xy (P).
    """
    P = xy.shape[0]
    if P < 3:
        return np.zeros(P, dtype=float)

    v = np.diff(xy, axis=0)                     # (P-1, 2)
    ang = np.arctan2(v[:, 1], v[:, 0])          # (P-1,)
    d_ang = np.diff(ang, prepend=ang[0])        # (P-1,)
    d_ang = np.abs(np.mod(d_ang + np.pi, 2*np.pi) - np.pi)  # shortest angle

    k = 7
    pad = np.pad(d_ang, (k, k), mode="wrap")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.convolve(pad, ker, mode="same")[k:-k]  # (P-1,)

    sm_full = np.empty(P, dtype=float)
    sm_full[:-1] = sm
    sm_full[-1] = sm[0]
    return sm_full

def _find_drs_zones(
    xy: np.ndarray,
    min_frac: float = 0.08,
    curv_thresh: float = 0.002
) -> List[Tuple[int, int, int]]:
    """
    Auto-detect up to 3 longest low-curvature segments as DRS zones.
    Returns: list of (start_idx, end_idx, detection_idx).
    """
    P = xy.shape[0]
    curv = _curvature(xy)              # length P
    straight = curv < curv_thresh      # bool length P

    zones: List[Tuple[int, int]] = []
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
            if j > i:
                i = j
            else:
                break
        else:
            i += 1

    if zones and zones[0][0] == 0 and zones[-1][1] == P - 1:
        merged = (zones[-1][0], zones[0][1])
        zones = [merged] + zones[1:-1]

    def _seg_len(a, b):
        return (b - a + 1) if a <= b else (P - a) + (b + 1)

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
    """
    Pulls a recent race to infer driver->team, display name, and number.
    Falls back gracefully if columns differ by event.
    """
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
        team_col = None
        for cand in ["team", "Team", "Constructor", "TeamName", "ConstructorName"]:
            if cand in laps.columns:
                team_col = cand
                break
        name_col = None
        for cand in ["Abbreviation", "FullName", "BroadcastName", "Driver", "DriverId"]:
            if cand in laps.columns:
                name_col = cand
                break
        num_col = None
        for cand in ["DriverNumber", "Number", "CarNumber"]:
            if cand in laps.columns:
                num_col = cand
                break

        team_map, name_map, num_map = {}, {}, {}
        for dr in drivers[dcol].tolist():
            sub = laps[laps[dcol].astype(str) == dr]
            if team_col:
                tser = sub[team_col].dropna()
                team = str(tser.iloc[0]) if not tser.empty else "UNKNOWN"
            else:
                team = "UNKNOWN"
            if name_col:
                nser = sub[name_col].dropna()
                nm = str(nser.iloc[0]) if not nser.empty else str(dr)
            else:
                nm = str(dr)
            if num_col:
                nnum = pd.to_numeric(sub[num_col], errors="coerce").dropna()
                nn = int(nnum.iloc[0]) if not nnum.empty else 999
            else:
                nn = 999
            team_map[str(dr)] = team
            name_map[str(dr)] = nm
            num_map[str(dr)] = nn
        return team_map, name_map, num_map
    except Exception:
        return {}, {}, {}

# ------------------- Inputs -------------------
def load_driver_ranking() -> pd.DataFrame:
    """
    Load aggregated equal-car deltas.
    Prefers 'agg_delta_s'; falls back to similarly named columns if needed.
    """
    path = PROJ / "outputs" / "aggregate" / "driver_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(f"driver_ranking.csv not found at {path}")
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}

    driver_col = low.get("driver", list(df.columns)[0])
    delta_col = low.get("agg_delta_s") or low.get("equal_delta_s") or low.get("delta_s") or low.get("agg_delta")
    if delta_col is None:
        raise ValueError("Could not find aggregated delta column (expected 'agg_delta_s') in driver_ranking.csv")

    se_col = low.get("agg_se_s")

    keep = [driver_col, delta_col] + ([se_col] if se_col else [])
    out = df[keep].copy()
    out.rename(columns={driver_col: "driver", delta_col: "agg_delta_s"}, inplace=True)
    if se_col:
        out.rename(columns={se_col: "agg_se_s"}, inplace=True)
    else:
        out["agg_se_s"] = np.nan

    out["driver"] = out["driver"].astype(str)
    out["agg_delta_s"] = pd.to_numeric(out["agg_delta_s"], errors="coerce").fillna(out["agg_delta_s"].median())
    out["agg_se_s"] = pd.to_numeric(out["agg_se_s"], errors="coerce")
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

# ------------------- Simulation (advanced) -------------------
def simulate_progress(
    ranking: pd.DataFrame,
    xy_path: np.ndarray,
    base_lap: float,
    n_laps: int,
    dt: float,
    noise_sd: float,   # baseline per-driver offset
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str], List[bool], List[str]]:
    """
    Returns:
      positions: (T, D, 2) XY positions
      lap_key:   (T, D)   = completed_laps + lap_progress in [0, n_laps]
      leader_lap: (T,)    = leader's displayed lap number (1..n_laps)
      drivers:   list of driver ids (length D)
      phase_flags: (T,)   : "GREEN" | "VSC" | "SC"
      rc_texts:   (T,)   : multiline Race Control text (recent events)
      drs_on:     (T,)   : whether DRS is enabled at this tick
      drs_banner: (T,)   : "DRS ENABLED" | "DRS DISABLED"
    Includes: DRS + blocking + VSC/SC phases with event logging.
    """
    rng = np.random.default_rng(seed)
    drivers = ranking["driver"].tolist()
    deltas = ranking["agg_delta_s"].to_numpy(dtype=float)
    D = len(drivers)

    P = xy_path.shape[0]
    zones = _find_drs_zones(xy_path)  # [(start_idx, end_idx, det_idx), ...]
    zone_ids = list(range(len(zones)))

    # Pace model (per-lap)
    base_driver = base_lap + deltas + rng.normal(0.0, noise_sd, size=D)
    lap_idx = np.arange(n_laps, dtype=float)[:, None]
    degrade = lap_idx * float(DEGRADE_S_PER_LAP)
    eps_lap = rng.normal(0.0, float(LAP_JITTER_SD), size=(n_laps, D))
    lap_times = base_driver[None, :] + degrade + eps_lap
    lap_times = np.clip(lap_times, 60.0, 180.0)
    speed_pts_per_sec = P / lap_times  # (L, D)

    # State
    curr_pts = np.zeros(D, dtype=float)     # 0..P along current lap
    curr_lap = np.zeros(D, dtype=int)       # 0..n_laps
    last_pts = np.zeros(D, dtype=float)
    defended_this_lap = np.zeros(D, dtype=bool)
    drs_eligible = {z: dict() for z in zone_ids}  # zone -> {driver_index: (ahead_index, gap_s_at_det)}
    drs_disabled_laps_remaining = DRS_MIN_ENABLE_LAP - 1  # disable L1 & L2

    # Phases
    phase = "GREEN"   # GREEN, VSC, SC
    vsc_ticks_remaining = 0
    sc_laps_remaining = 0
    leader_last_completed = 0

    # Start reaction jitter: apply once at t=0 as negative progress to some cars
    start_penalty = rng.normal(0.0, START_REACTION_SD, size=D)
    start_penalty_pts = start_penalty * speed_pts_per_sec[0, :]
    curr_pts = np.maximum(0.0, curr_pts - np.clip(start_penalty_pts, 0.0, P*0.02))

    # Storage
    positions_list: List[np.ndarray] = []
    lapkey_list: List[np.ndarray] = []
    leaderlap_list: List[int] = []
    phase_flags: List[str] = []
    drs_on_flags: List[bool] = []
    rc_texts: List[str] = []
    drs_banner: List[str] = []
    event_log: List[Tuple[float, str]] = []  # (time_sec, message)

    # Conservative max steps
    max_time = n_laps * float(np.max(lap_times))
    T_max = int(math.ceil(max_time / dt)) + 1

    def _order_indices() -> np.ndarray:
        lk = curr_lap.astype(float) + (curr_pts / P)
        return np.argsort(-lk)

    def _gap_seconds(i_follow: int, i_lead: int) -> float:
        if curr_lap[i_follow] != curr_lap[i_lead]:
            return 99.0
        dp = (curr_pts[i_lead] - curr_pts[i_follow])
        if dp < 0:
            dp += P
        v = speed_pts_per_sec[min(curr_lap[i_follow], n_laps-1), i_follow]
        if v <= 0:
            return 99.0
        return float(dp / v)

    def _in_range(prev: float, now: float, target: int) -> bool:
        if now >= prev:
            return (prev <= target) and (target < now)
        else:
            return (target >= prev) or (target < now)

    def _is_in_zone(idx: float, z: Tuple[int, int, int]) -> bool:
        a, b, _ = z
        if a <= b:
            return (idx >= a) and (idx <= b)
        else:
            return (idx >= a) or (idx <= b)

    def _apply_defense(i_lead: int):
        v = speed_pts_per_sec[min(curr_lap[i_lead], n_laps-1), i_lead]
        dist_pts = DEFENSE_TIME_COST * v
        new_pts = curr_pts[i_lead] - dist_pts
        if new_pts < 0:
            new_pts += P
        curr_pts[i_lead] = new_pts
        defended_this_lap[i_lead] = True

    def _attempt_pass(i_follow: int, i_lead: int, gap_at_det_s: float, tsec: float):
        L = min(curr_lap[i_follow], n_laps-1)
        pace_lead = lap_times[L, i_lead]
        pace_foll = lap_times[L, i_follow]
        delta_norm = (pace_lead - pace_foll) / base_lap  # positive if follower faster
        slip = DRS_SLIPSTREAM_BONUS if gap_at_det_s <= (DRS_DETECTION_THRESH_S * 1.2) else 0.0
        defended = 1.0 if defended_this_lap[i_lead] else 0.0

        logit = DRS_ALPHA * delta_norm + DRS_BETA * slip - DRS_GAMMA * defended
        p = 1.0 / (1.0 + math.exp(-logit))
        if np.isfinite(p) and (np.random.random() < p):
            # swap by nudging positions
            car_len = max(1, int(P * 0.0025))
            curr_pts[i_follow] = (curr_pts[i_lead] + car_len) % P
            curr_pts[i_lead] = (curr_pts[i_lead] - car_len) % P
            event_log.append((tsec, f"PASS: {drivers[i_follow]} → {drivers[i_lead]} (DRS)"))

    def _log(msg: str, tsec: float):
        event_log.append((tsec, msg))

    sim_time = 0.0
    for tstep in range(T_max):
        # Compute speeds with phase modifiers
        speeds = np.zeros(D, dtype=float)
        active = curr_lap < n_laps
        if active.any():
            L = np.clip(curr_lap, 0, n_laps-1)
            base_speed = speed_pts_per_sec[L, np.arange(D)]
            phase_factor = 1.0
            if phase == "VSC":
                phase_factor = VSC_SPEED_FACTOR
            elif phase == "SC":
                phase_factor = SC_SPEED_FACTOR
            speeds[active] = base_speed[active] * phase_factor

        # Optional DNF
        if RELIABILITY_DNF_PER_LAP > 0 and np.random.random() < RELIABILITY_DNF_PER_LAP:
            act_idx = np.where(active)[0]
            if len(act_idx) > 0:
                dnf_i = int(np.random.choice(act_idx))
                curr_lap[dnf_i] = n_laps
                speeds[dnf_i] = 0.0
                _log(f"DNF: {drivers[dnf_i]}", sim_time)

        # Save previous pts for crossing checks
        last_pts[:] = curr_pts

        # Advance positions
        curr_pts[active] = (curr_pts[active] + speeds[active] * dt) % P

        # Lap crossings
        crossed = (curr_pts < last_pts) & active  # wrapped -> finished a lap
        if crossed.any():
            curr_lap[crossed] += 1
            defended_this_lap[crossed] = False  # reset defense flag per lap
            for z in zone_ids:
                de = drs_eligible[z]
                for di in list(de.keys()):
                    if crossed[di]:
                        de.pop(di, None)

        # Leader's completed laps
        leader_completed = int(curr_lap.max() if curr_lap.size else 0)

        # Incident generator at leader lap completion (GREEN only)
        if phase == "GREEN" and leader_completed > leader_last_completed:
            if drs_disabled_laps_remaining > 0:
                drs_disabled_laps_remaining -= 1
            if np.random.random() < P_INCIDENT_PER_LAP:
                if np.random.random() < SC_SHARE:
                    # Safety Car
                    phase = "SC"
                    sc_laps_remaining = np.random.randint(SC_DURATION_LAPS_MINMAX[0], SC_DURATION_LAPS_MINMAX[1] + 1)
                    # Bunch field to leader with stagger gaps
                    order = _order_indices()
                    lead = order[0]
                    base_pos = curr_pts[lead]
                    for rank, di in enumerate(order):
                        gap_pts = int(SC_STAGGER_GAP_FRAC * P * rank)
                        pos = base_pos - gap_pts
                        while pos < 0:
                            pos += P
                        curr_pts[di] = pos
                        curr_lap[di] = curr_lap[lead]  # bring to leader's lap
                    for z in zone_ids:
                        drs_eligible[z].clear()
                    defended_this_lap[:] = False
                    _log("SAFETY CAR DEPLOYED (yellow)", sim_time)
                else:
                    # VSC
                    phase = "VSC"
                    vsec = float(np.random.uniform(VSC_DURATION_SEC_MINMAX[0], VSC_DURATION_SEC_MINMAX[1]))
                    vsc_ticks_remaining = max(1, int(vsec / dt))
                    _log("VIRTUAL SAFETY CAR DEPLOYED", sim_time)

        # Handle phase timers & transitions
        if phase == "VSC":
            if vsc_ticks_remaining > 0:
                vsc_ticks_remaining -= 1
            if vsc_ticks_remaining <= 0:
                phase = "GREEN"
                _log("VSC END — GREEN FLAG", sim_time)
        elif phase == "SC":
            if leader_completed > leader_last_completed:
                sc_laps_remaining -= 1
                if sc_laps_remaining <= 0:
                    phase = "GREEN"
                    drs_disabled_laps_remaining = max(drs_disabled_laps_remaining, DRS_COOLDOWN_AFTER_SC_LAPS)
                    _log("SAFETY CAR IN — GREEN FLAG (DRS delayed)", sim_time)

        # DRS & blocking under GREEN only
        drs_enabled = (phase == "GREEN") and (drs_disabled_laps_remaining <= 0) and (leader_completed + 1 >= DRS_MIN_ENABLE_LAP)

        if phase == "GREEN":
            order = _order_indices()

            # BLOCKING outside DRS zones
            for k in range(1, len(order)):
                i_lead = order[k-1]
                i_foll = order[k]
                if curr_lap[i_lead] >= n_laps or curr_lap[i_foll] >= n_laps:
                    continue
                gap_s = _gap_seconds(i_foll, i_lead)
                if gap_s <= BLOCKING_THRESH_S:
                    in_any_zone = any(_is_in_zone(curr_pts[i_foll], zones[z]) for z in zone_ids)
                    if (not in_any_zone) and (not defended_this_lap[i_lead]) and (np.random.random() < P_DEFEND):
                        _apply_defense(i_lead)
                        _log(f"DEFENSE: {drivers[i_lead]} blocks {drivers[i_foll]} (+{DEFENSE_TIME_COST:.02f}s)", sim_time)

            # DRS detection & pass attempts
            for z in zone_ids:
                zstart, zend, zdet = zones[z]

                # Detection: mark eligible followers when they cross detection point
                if drs_enabled:
                    order = _order_indices()  # refresh (may change with defenses)
                    for idx in order[1:]:  # followers only
                        if curr_lap[idx] >= n_laps:
                            continue
                        if _in_range(last_pts[idx], curr_pts[idx], zdet):
                            pos = list(order).index(idx)
                            if pos > 0:
                                ahead = order[pos-1]
                                if curr_lap[ahead] >= n_laps:
                                    continue
                                gap_s = _gap_seconds(idx, ahead)
                                if gap_s <= DRS_DETECTION_THRESH_S:
                                    drs_eligible[z][idx] = (ahead, gap_s)
                                else:
                                    drs_eligible[z].pop(idx, None)

                # Zone end: attempt pass for eligible followers
                for idx, (ahead, gap_at_det) in list(drs_eligible[z].items()):
                    if curr_lap[idx] >= n_laps or curr_lap[ahead] >= n_laps:
                        drs_eligible[z].pop(idx, None)
                        continue
                    if _in_range(last_pts[idx], curr_pts[idx], zend):
                        curr_order = _order_indices()
                        pos = list(curr_order).index(idx)
                        if pos > 0 and curr_order[pos-1] == ahead:
                            _attempt_pass(idx, ahead, gap_at_det, sim_time)
                        drs_eligible[z].pop(idx, None)

        # Store frame
        pos_idx = np.mod(curr_pts.astype(int), P)
        frame_xy = np.stack([xy_path[pos_idx, 0], xy_path[pos_idx, 1]], axis=1)
        positions_list.append(frame_xy)

        lk = curr_lap.astype(float) + (curr_pts / P)
        lapkey_list.append(lk)

        leaderlap_list.append(min(n_laps, int(curr_lap.max()) + 1))
        phase_flags.append(phase)
        drs_on_flags.append(bool(drs_enabled))
        drs_banner.append("DRS ENABLED" if drs_enabled else "DRS DISABLED")

        # Race Control text = latest ~8 events
        # (Only include events up to current sim_time)
        recent = [f"t={sim_time:5.1f}s  {m}" for (t, m) in event_log if t <= sim_time]
        rc_block = "\n".join(recent[-8:]) if recent else "—"
        rc_texts.append(rc_block)

        leader_last_completed = leader_completed
        sim_time += dt

        # Exit when everyone has finished
        if (curr_lap >= n_laps).all():
            break

    positions = np.stack(positions_list, axis=0)        # (T, D, 2)
    lap_key = np.stack(lapkey_list, axis=0)             # (T, D)
    leader_lap = np.array(leader_lap, dtype=int) if (leader_lap := leaderlap_list) else np.array([], dtype=int)
    return positions, lap_key, leader_lap, drivers, phase_flags, rc_texts, drs_on_flags, drs_banner

# ------------------- Build figure -------------------
def _flag_style(phase: str) -> Tuple[str, str]:
    """
    Returns (banner_text, banner_bgcolor) for phase
    """
    if phase == "SC":
        return ("SAFETY CAR", "#ffd400")  # yellow
    if phase == "VSC":
        return ("VIRTUAL SAFETY CAR", "#ffb347")  # amber
    return ("GREEN FLAG", "#2ecc71")  # green

def _track_color(phase: str) -> str:
    if phase == "SC":
        return "rgba(255,212,0,0.70)"   # yellow
    if phase == "VSC":
        return "rgba(255,165,0,0.45)"   # orange/amber
    return "rgba(80,90,110,0.25)"       # normal

def build_animation(
    positions: np.ndarray,
    lap_key: np.ndarray,
    leader_lap: np.ndarray,
    drivers: List[str],
    name_map: Dict[str, str],
    color_map: Dict[str, str],
    xy_path: np.ndarray,
    n_laps: int,
    phase_flags: List[str],
    rc_texts: List[str],
    drs_on: List[bool],
    drs_banner: List[str],
) -> go.Figure:
    """
    Plotly animated figure with:
      - left: track outline + cars
      - middle: live leaderboard
      - right: Race Control log (events)
      - top banner: GREEN / VSC / SAFETY CAR and DRS ON/OFF
    """
    T, D, _ = positions.shape

    labels = [str(dr).upper()[:3] for dr in drivers]
    names = [name_map.get(dr, dr) for dr in drivers]
    colors = [color_map.get(dr, "#888888") for dr in drivers]

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.62, 0.20, 0.18],
        specs=[[{"type": "xy"}, {"type": "scatter"}, {"type": "scatter"}]],
        horizontal_spacing=0.04,
        subplot_titles=("Equal-Car Montreal Replay", "Leaderboard (live)", "Race Control"),
    )

    # Track outline (frame 0 uses correct phase color)
    track_col0 = _track_color(phase_flags[0] if len(phase_flags) else "GREEN")
    fig.add_trace(
        go.Scatter(
            x=xy_path[:, 0], y=xy_path[:, 1],
            mode="lines",
            line=dict(width=2, color=track_col0),
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

    # Legend: one entry per driver
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
    order0 = np.argsort(-lap_key[0])
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

    # Race Control panel (frame 0)
    rc0 = rc_texts[0] if len(rc_texts) else "—"
    fig.add_trace(
        go.Scatter(
            x=[1],
            y=[1],
            mode="text",
            text=[rc0],
            textposition="top left",
            textfont=dict(size=12),
            showlegend=False,
            name="Race Control",
        ),
        row=1, col=3,
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
    fig.update_xaxes(showticklabels=False, range=[0, 2], row=1, col=3)
    fig.update_yaxes(showticklabels=False, range=[0, 2], row=1, col=3)

    # UI: ~10x playback
    frame_duration_ms = max(10, int(1000 * DT / PLAYBACK_MULTIPLIER))
    # Top banner (phase + DRS)
    banner_text, banner_bg = _flag_style(phase_flags[0] if len(phase_flags) else "GREEN")
    drs_tag = drs_banner[0] if len(drs_banner) else "DRS DISABLED"
    fig.update_layout(
        height=760,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(title="Drivers", x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.6)"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.47, y=1.10, xanchor="center",
            buttons=[
                dict(label=f"Play ~{int(PLAYBACK_MULTIPLIER)}×", method="animate",
                     args=[None, {"fromcurrent": True,
                                  "frame": {"duration": frame_duration_ms, "redraw": True},
                                  "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"mode": "immediate",
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0}}]),
            ],
        )],
        title="Equal-Car Montreal Replay",
        annotations=[
            dict(  # Lap counter (left)
                text=f"Lap 1 / {n_laps}",
                showarrow=False, x=0.28, y=1.10, xref="paper", yref="paper",
                font=dict(size=14, color="#1f2d3d")
            ),
            dict(  # Phase banner (center)
                text=banner_text,
                showarrow=False, x=0.50, y=1.10, xref="paper", yref="paper",
                font=dict(size=14, color="#1f2d3d"),
                bgcolor=banner_bg
            ),
            dict(  # DRS banner (right)
                text=drs_tag,
                showarrow=False, x=0.73, y=1.10, xref="paper", yref="paper",
                font=dict(size=14, color="#1f2d3d"),
                bgcolor="#e8f1ff" if ("ENABLED" in drs_tag) else "#ffd6d6"
            ),
        ],
    )

    # Frames (trace order: 0=track, 1=cars, 2..(1+D)=legend, (2+D)=leaderboard, (3+D)=RaceControl)
    frames = []
    for ti in range(T):
        order = np.argsort(-lap_key[ti])  # descending -> leader first
        rank_labels = [f"{i+1}. {labels[idx]}" for i, idx in enumerate(order)]
        lap_disp = int(np.clip(leader_lap[ti], 1, n_laps))

        banner_text, banner_bg = _flag_style(phase_flags[ti])
        drs_tag = drs_banner[ti]
        track_col = _track_color(phase_flags[ti])

        frames.append(go.Frame(
            data=[
                go.Scatter(x=xy_path[:, 0], y=xy_path[:, 1], line=dict(width=2, color=track_col)),  # track
                go.Scatter(x=positions[ti, :, 0], y=positions[ti, :, 1], text=labels),              # cars
                *[go.Scatter() for _ in range(D)],                                                  # legend placeholders
                go.Scatter(x=[1] * D, y=list(range(D, 0, -1)), text=rank_labels),                   # leaderboard
                go.Scatter(x=[1], y=[1], text=[rc_texts[ti]], textposition="top left"),             # Race Control
            ],
            layout=go.Layout(annotations=[
                dict(text=f"Lap {lap_disp} / {n_laps}", showarrow=False, x=0.28, y=1.10,
                     xref="paper", yref="paper", font=dict(size=14, color="#1f2d3d")),
                dict(text=banner_text, showarrow=False, x=0.50, y=1.10,
                     xref="paper", yref="paper", font=dict(size=14, color="#1f2d3d"), bgcolor=banner_bg),
                dict(text=drs_tag, showarrow=False, x=0.73, y=1.10,
                     xref="paper", yref="paper", font=dict(size=14, color="#1f2d3d"),
                     bgcolor="#e8f1ff" if ("ENABLED" in drs_tag) else "#ffd6d6"),
            ]),
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

    (positions, lap_key, leader_lap, drivers,
     phase_flags, rc_texts, drs_on, drs_banner) = simulate_progress(
        ranking, xy, base_lap=BASE_LAP_SEC, n_laps=N_LAPS, dt=DT,
        noise_sd=LAP_JITTER_SD, seed=RANDOM_SEED
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
        phase_flags=phase_flags,
        rc_texts=rc_texts,
        drs_on=drs_on,
        drs_banner=drs_banner,
    )

    out_path = OUT_DIR / "simulation.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"[INFO] Wrote visualization: {out_path}")

if __name__ == "__main__":
    main()
