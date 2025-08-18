# src/visualize_equal_race.py
from __future__ import annotations

import math
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from load_data import load_config, get_track_outline, get_recent_races, load_session, enable_cache

# ------------------- IO & Paths -------------------
PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "outputs" / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Defaults (used if config keys missing) -------------------
BASE_LAP_SEC = 90.0
N_LAPS = 20
DT = 0.5

LAP_JITTER_SD = 0.12
START_REACTION_SD = 0.12
START_GAIN_SD = 0.08         # random T1 gain/loss, seconds (small)
START_GAIN_RANK_BIAS = 0.0   # >0 lets mid/back gain a touch on avg

# Event deltas
USE_EVENT_SPECIFIC_DELTAS = True
TARGET_GP_SUBSTR = "canadian"  # e.g. "british", "austrian", "hungarian"

# DRS/Overtaking (base; can be overridden by config + track overrides)
DRS_DETECTION_THRESH_S = 1.0
OVERTAKE_ALPHA_PACE = 6.0      # coefficient on normalized pace gap
OVERTAKE_BETA_DRS = 1.5        # bonus when DRS slipstream active
OVERTAKE_GAMMA_DEFEND = 1.2    # reduction when leader defended earlier in lap
DIRTY_AIR_PENALTY = 0.15       # constant penalty (track-specific, worse at Monaco)
DRS_SLIPSTREAM_BONUS = 0.02    # used to decide if "slip" is active
DRS_COOLDOWN_AFTER_SC_LAPS = 2
DRS_MIN_ENABLE_LAP = 3
DETECTION_OFFSET_FRACTION = 0.03

# --- Calibrated degradation params loader & curve helper ---
def _cfg_get(cfg: dict, path: List[str], default):
    """Get nested config key; path is list of keys."""
    d = cfg
    for k in path:
        if not isinstance(d, dict) or (k not in d):
            return default
        d = d[k]
    return d

def _load_degradation_params(cfg: dict) -> Optional[dict]:
    path = _cfg_get(cfg, ["paths", "degradation_params"], None)
    if not path:
        return None
    f = (PROJ / path).resolve()
    if not f.exists():
        return None
    try:
        with open(f, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None

def _curve_from_params(n_laps: int, params_obj: dict) -> np.ndarray:
    """Build cumulative degradation curve from {early_slope, late_slope, switch_lap}."""
    e = float(params_obj.get("early_slope", 0.0))
    l = float(params_obj.get("late_slope", 0.0))
    sw = int(params_obj.get("switch_lap", 12))
    t = np.arange(n_laps, dtype=float)  # 0..L-1; lap_on_tyre≈t
    early = np.minimum(t, sw) * e
    late = np.maximum(t - sw, 0.0) * l
    return early + late

# Safety Car / VSC (configurable default)
P_INCIDENT_PER_LAP_DEFAULT = 0.03
SC_SHARE = 0.7
VSC_SPEED_FACTOR = 0.65
SC_SPEED_FACTOR = 0.50
SC_DURATION_LAPS_MINMAX = (1, 3)
VSC_DURATION_SEC_MINMAX = (12.0, 32.0)
SC_STAGGER_GAP_FRAC = 0.003

# Reliability (new: specify per-race DNF target; per-lap is derived)
RELIABILITY_MODE = "per_race"      # "per_race" | "per_lap"
RELIABILITY_PER_RACE_DNF = 0.10    # ~10% chance a car DNFs over a typical race
RELIABILITY_TYPICAL_LAPS = 70      # translate to per-lap hazard ~ 1 - (1-0.10)^(1/70) ≈ 0.0015
RELIABILITY_PER_LAP = 0.0          # only used if RELIABILITY_MODE == "per_lap"

# Playback speeds
RANDOM_SEED = 42
PLAYBACK_CHOICES = [5.0, 10.0, 20.0, 100.0]

# UI
SHOW_MARKER_LABELS = False
RC_WRAP_WIDTH = 48
DRS_TAG = "ⓓ"  # marks DRS on leaderboard gaps

# ------------------- Track meta helpers (NEW) -------------------
def _norm_event_key_str(s: str) -> str:
    if not isinstance(s, str):
        return "unknown"
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(cleaned.split())

def _norm_event_key(year: Optional[int], gp: Optional[str]) -> str:
    y = "" if pd.isna(year) else str(int(year))
    g = "" if gp is None else str(gp)
    return _norm_event_key_str(f"{y} {g}")

def _load_viz_track_meta(cfg: dict) -> Optional[dict]:
    """
    Load data/track_meta.csv and extract metadata for the viz target in cfg['viz_track'].
    Expected columns: event_key or (year,gp), plus:
      track_type, downforce_index, drs_zones, speed_bias, overtaking_difficulty
    Returns a dict or None if not available.
    """
    meta_path = _cfg_get(cfg, ["paths", "track_meta"], "data/track_meta.csv")
    f = (PROJ / meta_path).resolve()
    if not f.exists():
        return None
    try:
        meta = pd.read_csv(f)
    except Exception:
        return None

    cols = {c.lower(): c for c in meta.columns}
    if "event_key" in cols:
        meta["event_key_norm"] = meta[cols["event_key"]].map(_norm_event_key_str)
    elif "year" in cols and "gp" in cols:
        meta["event_key_norm"] = meta.apply(lambda r: _norm_event_key(r[cols["year"]], r[cols["gp"]]), axis=1)
    else:
        return None

    vt = _cfg_get(cfg, ["viz_track"], {}) or {}
    target_key = _norm_event_key(vt.get("year"), vt.get("grand_prix"))
    row = meta.loc[meta["event_key_norm"] == target_key]
    if row.empty:
        return None

    r0 = row.iloc[0]
    out = {
        "track_type": str(r0.get(cols.get("track_type", "track_type"), "")).strip().lower() or None,
        "downforce_index": pd.to_numeric(r0.get(cols.get("downforce_index", "downforce_index"), np.nan), errors="coerce"),
        "drs_zones": int(pd.to_numeric(r0.get(cols.get("drs_zones", "drs_zones"), np.nan), errors="coerce")) if not pd.isna(r0.get(cols.get("drs_zones", "drs_zones"), np.nan)) else None,
        "speed_bias": pd.to_numeric(r0.get(cols.get("speed_bias", "speed_bias"), np.nan), errors="coerce"),
        "overtaking_difficulty": pd.to_numeric(r0.get(cols.get("overtaking_difficulty", "overtaking_difficulty"), np.nan), errors="coerce"),
    }
    return out

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

# === Personality loader (ADD) ===
def _load_personality_scores(cfg: dict, drivers: list[str]) -> Dict[str, Dict[str, float]]:
    """
    Read outputs/calibration/personality.csv (or cfg.paths.personality) and
    return {driver: {aggression, defence, risk, *_se}} with safe fallbacks.
    """
    path = _cfg_get(cfg, ["paths", "personality"], "outputs/calibration/personality.csv")
    f = (PROJ / path).resolve()
    if not f.exists():
        return {}

    try:
        df = pd.read_csv(f)
    except Exception:
        return {}

    low = {c.lower(): c for c in df.columns}
    req = ["driver", "aggression", "defence", "risk"]
    if not all(k in low for k in req):
        return {}

    def _clamp01(s):
        return pd.to_numeric(s, errors="coerce").clip(0.0, 1.0)

    out = {}
    for _, r in df.iterrows():
        d = str(r[low["driver"]])
        out[d] = {
            "aggression": float(_clamp01(r[low["aggression"]])) if pd.notna(r[low["aggression"]]) else 0.5,
            "defence":   float(_clamp01(r[low["defence"]]))   if pd.notna(r[low["defence"]])   else 0.5,
            "risk":      float(_clamp01(r[low["risk"]]))      if pd.notna(r[low["risk"]])      else 0.5,
            "aggression_se": float(pd.to_numeric(r.get(low.get("aggression_se",""), np.nan), errors="coerce")) if "aggression_se" in low else np.nan,
            "defence_se":   float(pd.to_numeric(r.get(low.get("defence_se",""),   np.nan), errors="coerce")) if "defence_se" in low else np.nan,
            "risk_se":      float(pd.to_numeric(r.get(low.get("risk_se",""),      np.nan), errors="coerce")) if "risk_se" in low else np.nan,
        }
    for d in drivers:
        out.setdefault(d, {"aggression":0.5,"defence":0.5,"risk":0.5,"aggression_se":np.nan,"defence_se":np.nan,"risk_se":np.nan})
    return out

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
    delta_col = (low.get("agg_delta_s") or low.get("equal_delta_s")
                 or low.get("delta_s") or low.get("agg_delta"))
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
    """Prefer event_delta_s_shrunk if present; fallback to event_delta_s,
    else (race, quali) with config wR/wQ."""
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

    evt = low.get("event_delta_s_shrunk") or low.get("event_delta_s")
    if evt:
        out = df[[drv, evt]].copy()
        out.columns = ["driver", "agg_delta_s"]
        out["driver"] = out["driver"].astype(str)
        out["agg_delta_s"] = pd.to_numeric(out["agg_delta_s"], errors="coerce")
        return out.dropna(subset=["agg_delta_s"])

    rcol = low.get("race_delta_s"); qcol = low.get("quali_delta_s")
    if drv is None or rcol is None:
        return None
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

# ------------------- Track & config helpers -------------------
def _current_gp_name(cfg: dict) -> str:
    try:
        gp = _cfg_get(cfg, ["viz_track", "grand_prix"], "")
        return str(gp)
    except Exception:
        return ""

def _track_type_from_cfg(cfg: dict, meta: Optional[dict] = None) -> str:
    # Prefer metadata if available
    if meta and isinstance(meta.get("track_type", None), str) and meta["track_type"]:
        return str(meta["track_type"]).lower()
    tt = _cfg_get(cfg, ["viz_track", "track_type"], None)
    if isinstance(tt, str) and tt:
        return tt.lower()
    gp = _current_gp_name(cfg).lower()
    if any(k in gp for k in ["monaco", "singapore", "miami", "jeddah", "baku", "vegas", "montr", "canadian"]):
        return "street"
    return "permanent"

def _apply_track_overrides(base: dict, cfg: dict) -> dict:
    """Apply overtaking.track_overrides if GP substring matches."""
    gp = _current_gp_name(cfg).lower()
    ovs = _cfg_get(cfg, ["overtaking", "track_overrides"], {}) or {}
    out = dict(base)
    for key, override in ovs.items():
        if key.lower() in gp:
            for k, v in (override or {}).items():
                out[k] = v
    return out

# ------------------- WEATHER: summary + temp multiplier -------------------
def _extract_weather_summary_from_laps(laps: Optional[pd.DataFrame]) -> Optional[dict]:
    if laps is None or len(laps) == 0:
        return None
    def med(colnames: List[str]):
        for c in colnames:
            if c in laps.columns:
                v = pd.to_numeric(laps[c], errors="coerce")
                if v.notna().any():
                    return float(v.median())
        return np.nan
    return {
        "median_track_temp_c": med(["track_temp_c", "TrackTemp", "TrackTempC"]),
        "median_air_temp_c": med(["air_temp_c", "AirTemp", "AirTempC"]),
        "median_wind_kph": med(["wind_speed_kph", "wind_kph", "WindSpeed", "WindKph"]),
        "median_humidity_pct": med(["humidity_pct", "RelHumidity", "HumidityPct"]),
    }

def _load_weather_summary_for_viz(cfg: dict) -> Optional[dict]:
    """Use the event's weather_summary medians if available; else compute medians from laps."""
    vt = _cfg_get(cfg, ["viz_track"], {}) or {}
    year, gp = vt.get("year"), vt.get("grand_prix")
    try:
        laps, meta = load_session(year, gp, "R")
    except Exception:
        laps, meta = None, None
    # Prefer event weather_summary if present
    ws = None
    if isinstance(meta, dict):
        ws = meta.get("weather_summary") or None
    else:
        # some integrations return an object with attribute
        ws = getattr(meta, "weather_summary", None) if meta is not None else None
    if ws and isinstance(ws, dict):
        # Keep only expected keys (robustness)
        out = {
            "median_track_temp_c": float(pd.to_numeric(ws.get("median_track_temp_c"), errors="coerce")),
            "median_air_temp_c": float(pd.to_numeric(ws.get("median_air_temp_c"), errors="coerce")) if "median_air_temp_c" in ws else np.nan,
            "median_wind_kph": float(pd.to_numeric(ws.get("median_wind_kph"), errors="coerce")) if "median_wind_kph" in ws else np.nan,
            "median_humidity_pct": float(pd.to_numeric(ws.get("median_humidity_pct"), errors="coerce")) if "median_humidity_pct" in ws else np.nan,
        }
        return out
    # Fallback: compute from laps
    return _extract_weather_summary_from_laps(laps)

def _temp_multiplier_fn(cfg: dict, weather_summary: Optional[dict]) -> Callable[[str], float]:
    """
    Build a small multiplier function for tyre-degradation slopes based on track temp.
    Multiplier ~= 1 + sens[compound] * k * (T - baseline), clipped to a small band.
    """
    # Defaults are intentionally small
    te = _cfg_get(cfg, ["temp_effect"], {}) or {}
    use = bool(te.get("use", True))
    base_c = float(te.get("baseline_track_c", 30.0))
    k = float(te.get("per_deg_pct", 0.004))  # +0.4% per °C
    # per-compound sensitivity (Soft a bit more sensitive)
    sens = te.get("compound_sensitivity", {"S": 1.15, "M": 1.00, "H": 0.85})
    clip_lo = float(te.get("clip_low", -0.15))  # allow up to -15%
    clip_hi = float(te.get("clip_high", 0.20))  # and +20%
    T = None
    if weather_summary and np.isfinite(pd.to_numeric(weather_summary.get("median_track_temp_c"), errors="coerce")):
        T = float(pd.to_numeric(weather_summary.get("median_track_temp_c"), errors="coerce"))
    if (not use) or (T is None):
        return lambda _c: 1.0

    dT = T - base_c
    def f(compound: str) -> float:
        s = float(sens.get(str(compound).upper(), 1.0))
        mult = 1.0 + s * k * dT
        return float(np.clip(mult, 1.0 + clip_lo, 1.0 + clip_hi))
    return f

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

# ------------------- Reliability (per-lap from per-race target) -------------------
def per_lap_dnf_probability(cfg: dict, n_laps: int, risk_vec: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns per-driver per-lap DNF probabilities.
    - If risk_vec is None or personality.use == False -> scalar broadcast (reliability mode).
    - If risk_vec provided -> scale per-race DNF by risk in [0,1]:
        p_race_driver = base * (1 + risk_weight*(risk - 0.5))  -> in [0.5*base, 1.5*base] at weight=1
    """
    mode = _cfg_get(cfg, ["reliability", "mode"], RELIABILITY_MODE)
    if str(mode).lower() == "per_lap" and risk_vec is None:
        p_pl = float(_cfg_get(cfg, ["reliability", "per_lap"], RELIABILITY_PER_LAP))
        return np.array([p_pl], dtype=float)

    base = float(_cfg_get(cfg, ["reliability", "per_race_dnf"], RELIABILITY_PER_RACE_DNF))
    typical = int(_cfg_get(cfg, ["reliability", "typical_race_laps"], RELIABILITY_TYPICAL_LAPS))
    typical = max(1, typical)

    if risk_vec is None or not bool(_cfg_get(cfg, ["personality", "use"], False)):
        p_pl = 1.0 - (1.0 - base) ** (1.0 / float(typical))
        return np.array([float(np.clip(p_pl, 0.0, 1.0))], dtype=float)

    amp = float(_cfg_get(cfg, ["personality", "risk_weight"], 1.0))
    mult = 1.0 + amp * (np.asarray(risk_vec, dtype=float) - 0.5)
    mult = np.clip(mult, 0.5, 1.5)
    p_race = np.clip(base * mult, 0.0, 0.9)
    p_pl_vec = 1.0 - (1.0 - p_race) ** (1.0 / float(typical))
    return p_pl_vec

# ------------------- Degradation model (piecewise per compound) -------------------
def _compound_for_all(cfg: dict, rng: np.random.Generator, D: int) -> List[str]:
    """Choose a compound per driver. If degradation.compound_mix exists, sample;
    else use degradation.default_compound (or 'M')."""
    mix = _cfg_get(cfg, ["degradation", "compound_mix"], None)
    if isinstance(mix, dict) and len(mix) > 0:
        keys = list(mix.keys())
        probs = np.array([float(mix[k]) for k in keys], dtype=float)
        probs = probs / probs.sum()
        return rng.choice(keys, size=D, p=probs).tolist()
    default = str(_cfg_get(cfg, ["degradation", "default_compound"], "M")).upper()
    return [default] * D

def _deg_params_for(compound: str, cfg: dict, track_type: str) -> Tuple[float, float, int, float]:
    """Return early_slope, late_slope (s/lap), switch_lap, track_mult."""
    comp = compound.upper()
    block = _cfg_get(cfg, ["degradation", "compounds"], {}) or {}
    default = {"S": (0.035, 0.010, 10), "M": (0.020, 0.012, 14), "H": (0.010, 0.015, 18)}
    e, l, sw = default.get(comp, default["M"])
    if comp in block and isinstance(block[comp], dict):
        e = float(block[comp].get("early_slope", e))
        l = float(block[comp].get("late_slope", l))
        sw = int(block[comp].get("switch_lap", sw))
    mults = _cfg_get(cfg, ["degradation", "track_type_multipliers"], {}) or {}
    track_mult = float(mults.get(track_type, 1.0))
    return e, l, sw, track_mult

def build_degradation_matrix(
    cfg: dict,
    n_laps: int,
    drivers: List[str],
    rng: np.random.Generator,
    meta: Optional[dict] = None,
    temp_mult_fn: Optional[Callable[[str], float]] = None,
) -> np.ndarray:
    """
    Return (n_laps, D) additive degradation in seconds per car.

    Modes:
      - linear/config (default): use config.degradation.compounds + track_type_multipliers
      - calibrated: read outputs/calibration/degradation_params.json and apply compound-specific curves
                    (optionally per track_type). When 'calibrated' is used we *ignore*
                    track_type_multipliers (to avoid double counting), but we still honor
                    degradation.compound_scale if present.
    Applies a small 'temp_mult_fn(compound)' if provided, to scale the slope/curve.
    """
    D = len(drivers)
    track_type = _track_type_from_cfg(cfg, meta=meta).lower()

    compounds = _compound_for_all(cfg, rng, D)
    source = str(_cfg_get(cfg, ["degradation", "source"], "linear")).lower()

    scale_block = _cfg_get(cfg, ["degradation", "compound_scale"], {}) or {}
    def _scale_for(c: str) -> float:
        return float(scale_block.get(str(c).upper(), 1.0))

    # identity multiplier if not provided
    if temp_mult_fn is None:
        temp_mult_fn = lambda _c: 1.0

    out = np.zeros((n_laps, D), dtype=float)

    if source == "calibrated":
        params = _load_degradation_params(cfg)
        if params:
            by_tt = (params.get("by_track_type") or {})
            global_p = (params.get("global") or {})

            for j in range(D):
                c = str(compounds[j]).upper()
                pobj = None
                if track_type and track_type in by_tt:
                    pobj = (by_tt.get(track_type) or {}).get(c)
                if pobj is None:
                    pobj = global_p.get(c)

                tm = float(temp_mult_fn(c))
                if pobj is not None:
                    curve = _curve_from_params(n_laps, pobj)
                    out[:, j] = tm * _scale_for(c) * curve
                else:
                    e, l, sw, _ = _deg_params_for(c, cfg, track_type)
                    t = np.arange(n_laps, dtype=float)
                    out[:, j] = tm * _scale_for(c) * (np.minimum(t, sw) * e + np.maximum(t - sw, 0.0) * l)
            return out

    lap_idx = np.arange(n_laps, dtype=float)[:, None]  # (L,1)
    for j in range(D):
        c = str(compounds[j]).upper()
        e, l, sw, mult = _deg_params_for(c, cfg, track_type)
        tm = float(temp_mult_fn(c))
        early = np.minimum(lap_idx, sw) * e
        late = np.maximum(lap_idx - sw, 0.0) * l
        out[:, j] = tm * _scale_for(c) * mult * (early + late).ravel()
    return out

# ------------------- Overtaking params from config (+ track overrides & META) -------------------
def get_overtake_params(cfg: dict, meta: Optional[dict] = None) -> Dict[str, float]:
    base = {
        "alpha_pace": float(_cfg_get(cfg, ["overtaking", "alpha_pace"], OVERTAKE_ALPHA_PACE)),
        "beta_drs": float(_cfg_get(cfg, ["overtaking", "beta_drs"], OVERTAKE_BETA_DRS)),
        "gamma_defend": float(_cfg_get(cfg, ["overtaking", "gamma_defend"], OVERTAKE_GAMMA_DEFEND)),
        "dirty_air_penalty": float(_cfg_get(cfg, ["overtaking", "dirty_air_penalty"], DIRTY_AIR_PENALTY)),
        "drs_detect_thresh_s": float(_cfg_get(cfg, ["overtaking", "drs_detect_thresh_s"], DRS_DETECTION_THRESH_S)),
        "detection_offset_fraction": float(_cfg_get(cfg, ["overtaking", "detection_offset_fraction"], DETECTION_OFFSET_FRACTION)),
        "drs_min_enable_lap": int(_cfg_get(cfg, ["overtaking", "drs_min_enable_lap"], DRS_MIN_ENABLE_LAP)),
        "drs_cooldown_after_sc_laps": int(_cfg_get(cfg, ["overtaking", "drs_cooldown_after_sc_laps"], DRS_COOLDOWN_AFTER_SC_LAPS)),
    }
    # YAML string overrides via substring match
    base = _apply_track_overrides(base, cfg)

    # Metadata-driven tweaks (optional)
    if meta:
        # Increase dirty-air penalty with overtaking difficulty (0..1) and slightly with downforce
        diff = meta.get("overtaking_difficulty")
        if pd.notna(diff):
            scale = float(np.clip(0.8 + 1.2 * float(diff), 0.5, 2.0))  # 0.8x .. 2.0x
            base["dirty_air_penalty"] = float(np.clip(base["dirty_air_penalty"] * scale, 0.03, 0.60))
        dfi = meta.get("downforce_index")
        if pd.notna(dfi):
            base["dirty_air_penalty"] = float(np.clip(base["dirty_air_penalty"] * (0.9 + 0.3 * float(dfi)), 0.03, 0.60))
    return base

# ------------------- RNG seeding helpers (NEW) -------------------
def _seed_mix32(*parts) -> int:
    """Stable 32-bit hash for seeding."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little") & 0xFFFFFFFF

def _event_key_for_seed(cfg: dict) -> str:
    vt = _cfg_get(cfg, ["viz_track"], {}) or {}
    return f"{vt.get('year','')}-{vt.get('grand_prix','')}".lower()

def _spawn_rng_streams(cfg: dict, global_seed: int, run_idx: int = 0):
    """
    Deterministic streams per event/run using SeedSequence.
    Returns dict of RNGs and a small metadata blob.
    """
    ek = _event_key_for_seed(cfg)
    root = np.random.SeedSequence([int(global_seed), _seed_mix32(ek), int(run_idx)])
    # Independent streams
    names = ["start", "lap", "overtake", "dnf", "incident"]
    children = root.spawn(len(names))
    rngs = {nm: np.random.default_rng(ss) for nm, ss in zip(names, children)}
    meta = {
        "global_seed": int(global_seed),
        "event_key": ek,
        "run_idx": int(run_idx),
        "spawn_keys": {nm: ss.spawn_key for nm, ss in zip(names, children)},
    }
    return rngs, meta

def _finish_order_entropy(start_order: List[int], finish_order: List[int]) -> float:
    """
    Normalized inversion ratio (0=no changes, 1=complete reversal).
    """
    pos0 = {d: i for i, d in enumerate(start_order)}
    seq = [pos0[d] for d in finish_order if d in pos0]
    inv = 0
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] > seq[j]:
                inv += 1
    max_inv = n * (n - 1) // 2
    return (inv / max_inv) if max_inv > 0 else 0.0

# ------------------- Simulation -------------------
def simulate_progress(
    ranking: pd.DataFrame,
    xy_path: np.ndarray,
    base_lap: float,
    n_laps: int,
    dt: float,
    noise_sd: float,
    seed: int,
    cfg: Optional[dict] = None,
    meta: Optional[dict] = None,
    weather_summary: Optional[dict] = None,
    incident_rate: float = P_INCIDENT_PER_LAP_DEFAULT,   # NEW
    start_gain_sd_override: Optional[float] = None,       # NEW
    disable_dnfs: bool = False,                           # NEW
    run_idx: int = 0,
):

    if cfg is None:
        cfg = {}

    # Deterministic, per-run RNG streams
    rngs, seed_meta = _spawn_rng_streams(cfg, seed, run_idx)
    r_start = rngs["start"]
    r_lap = rngs["lap"]
    r_pass = rngs["overtake"]
    r_dnf = rngs["dnf"]
    r_inc = rngs["incident"]

    # --- Meta pace bias (small) ---
    base_lap_eff = float(base_lap)
    if meta and pd.notna(meta.get("speed_bias", np.nan)):
        # Interpret +speed_bias as faster track -> slightly shorter base lap
        sb = float(np.clip(meta["speed_bias"], -1.5, 1.5))
        base_lap_eff *= (1.0 - 0.06 * sb)  # cap ~±9% within range [-1.5,1.5]

    # --- Drivers & deltas ---
    drivers = ranking["driver"].astype(str).tolist()
    deltas = pd.to_numeric(ranking["agg_delta_s"], errors="coerce").to_numpy(dtype=float)
    D = len(drivers)
    P = xy_path.shape[0]

    # --- Overtake/DRS params (+ meta) ---
    otk = get_overtake_params(cfg, meta=meta)
    global DRS_MIN_ENABLE_LAP, DRS_COOLDOWN_AFTER_SC_LAPS, DETECTION_OFFSET_FRACTION
    DRS_MIN_ENABLE_LAP = int(otk["drs_min_enable_lap"])
    DRS_COOLDOWN_AFTER_SC_LAPS = int(otk["drs_cooldown_after_sc_laps"])
    DETECTION_OFFSET_FRACTION = float(otk["detection_offset_fraction"])

    # Detect DRS zones from geometry, then optionally truncate to metadata count
    zones = _find_drs_zones(xy_path)

    # Only truncate when K > 0; K==0 should mean "don't override"
    if meta and isinstance(meta.get("drs_zones", None), (int, np.integer)):
        K = int(meta["drs_zones"])
        if K > 0:
            zones = zones[:K]

    # Fallback to a generic straight if detection/metadata yields none
    if not zones:
        zs = int(0.15 * P);
        ze = int(0.35 * P);
        zd = int(0.12 * P)
        zones = [(zs, ze, zd)]

    det_eff = float(otk["drs_detect_thresh_s"])

    # Scale alpha by length of strongest DRS zone (existing) + count of zones (meta-aware)
    def _seg_len(a, b): return (b - a + 1) if a <= b else (P - a) + (b + 1)
    longest = max((_seg_len(a, b) for (a, b, _) in zones), default=int(0.12 * P))
    frac = longest / float(P)
    baseline = 0.12
    length_scale = max(0.6, min(2.0, 1.0 + 1.5 * (frac - baseline)))
    count_scale = 1.0 + 0.18 * max(0, (len(zones) - 1))  # +18% per extra zone
    count_scale = float(np.clip(count_scale, 0.7, 1.9))
    alpha_eff = float(otk["alpha_pace"]) * length_scale * count_scale

    # --- Personality vectors ---
    use_personality = bool(_cfg_get(cfg, ["personality", "use"], False))
    if use_personality:
        pers = _load_personality_scores(cfg, drivers)
    else:
        pers = {}

    agg = np.array([pers.get(d, {}).get("aggression", 0.5) for d in drivers], dtype=float)
    defence = np.array([pers.get(d, {}).get("defence", 0.5)   for d in drivers], dtype=float)
    risk = np.array([pers.get(d, {}).get("risk", 0.5)         for d in drivers], dtype=float)

    agg_w  = float(_cfg_get(cfg, ["personality", "agg_weight"], 1.0))
    def_w  = float(_cfg_get(cfg, ["personality", "def_weight"], 1.0))

    # Base lap + driver deltas + jitter
    base_driver = base_lap_eff + deltas + r_lap.normal(0.0, noise_sd, size=D)

    # Degradation matrix (compound & track-type aware; meta track_type preferred)
    tmul_fn = _temp_multiplier_fn(cfg, weather_summary)
    degrade = build_degradation_matrix(cfg, n_laps, drivers, r_lap, meta=meta, temp_mult_fn=tmul_fn)  # (L, D)

    # Random lap noise
    eps_lap = r_lap.normal(0.0, float(LAP_JITTER_SD), size=(n_laps, D))

    # Lap time cube
    lap_times = base_driver[None, :] + degrade + eps_lap
    lap_times = np.clip(lap_times, 60.0, 180.0)
    speed_pts_per_sec = P / lap_times

    # Reliability per-lap probability (from per-race target), vectorized with risk
    p_dnf_per_lap_vec = per_lap_dnf_probability(cfg, n_laps, risk if use_personality else None)
    if p_dnf_per_lap_vec.size == 1:
        p_dnf_per_lap_vec = np.repeat(p_dnf_per_lap_vec[0], D)
    if disable_dnfs:
        p_dnf_per_lap_vec[:] = 0.0

    # --- State ---
    curr_pts = np.zeros(D, dtype=float)
    curr_lap = np.zeros(D, dtype=int)
    last_pts = np.zeros(D, dtype=float)
    defended_this_lap = np.zeros(D, dtype=bool)
    drs_eligible = {i: {} for i in range(len(zones))}

    phase = "GREEN"
    vsc_ticks_remaining = 0
    sc_laps_remaining = 0
    drs_disabled_until_lap = DRS_MIN_ENABLE_LAP

    # ---- Run stats ----
    stats = {
        "attempts": 0,
        "passes": 0,
        "dnfs": [],
        "start_gains_sec": None,       # filled after starts
        "grid_order": None,            # indices 0..D-1
        "finish_order": None,          # indices 0..D-1
        "finish_entropy": None,        # normalized inversion ratio
        "seed_meta": seed_meta,
        "timestamp": int(time.time()),
        "weather_summary": weather_summary or {},
    }
    # base "grid" order from pace deltas (lower -> ahead)
    grid_order = list(np.argsort(deltas))
    stats["grid_order"] = grid_order

    # base "grid" order from pace deltas (lower -> ahead)
    grid_order = list(np.argsort(deltas))
    stats["grid_order"] = grid_order

    # NEW: place cars on a staggered grid using pace order
    GRID_GAP_SEC = float(_cfg_get(cfg, ["starts", "grid_gap_sec"], 0.16))  # ~0.16s between rows
    leader_speed_pts_per_sec = float(speed_pts_per_sec[0, grid_order[0]])
    gap_pts = max(1.0, GRID_GAP_SEC * leader_speed_pts_per_sec)

    curr_pts[:] = 0.0
    for rank, di in enumerate(grid_order):
        # P1 near the line; each row ~GRID_GAP_SEC further back
        curr_pts[di] = (P - rank * gap_pts) % P

    # --- Grid & start model ---
    base_start_sd = float(_cfg_get(cfg, ["starts", "start_gain_sd"], START_GAIN_SD))
    if start_gain_sd_override is not None:
        base_start_sd = float(start_gain_sd_override)
    rank_bias = float(_cfg_get(cfg, ["starts", "start_gain_rank_bias"], START_GAIN_RANK_BIAS))
    L0_speed = speed_pts_per_sec[0, :]
    ranks = np.arange(D, dtype=float)
    bias = rank_bias * ((ranks - (D - 1) / 2.0) / max(1.0, D - 1))

    if use_personality:
        scale_vec = (1.0 + 0.6 * agg_w * (agg - 0.5)) * (1.0 - 0.4 * def_w * (defence - 0.5))
        scale_vec = np.clip(scale_vec, 0.5, 1.5)
    else:
        scale_vec = np.ones(D, dtype=float)

    start_gain_sec = r_start.normal(0.0, base_start_sd, size=D) * scale_vec + bias
    stats["start_gains_sec"] = {drivers[i]: float(start_gain_sec[i]) for i in range(D)}
    start_gain_pts = start_gain_sec * L0_speed
    curr_pts = np.maximum(0.0, curr_pts + np.clip(start_gain_pts, -P * 0.02, P * 0.02))

    positions_list = []; lapkey_list = []; leaderlap_list = []
    phase_flags = []; rc_texts = []; drs_on_flags = []; drs_banner = []
    event_log: List[Tuple[float, str]] = []
    orders = []; gaps_panel = []

    def _gap_seconds(i_follow: int, i_lead: int) -> float:
        if curr_lap[i_follow] != curr_lap[i_lead]:
            return 99.0
        dp = (curr_pts[i_lead] - curr_pts[i_follow])
        if dp < 0: dp += P
        L = min(curr_lap[i_follow], n_laps - 1)
        v = (P / lap_times[L, i_follow])
        return float(dp / max(v, 1e-6))

    def _in_range(prev: float, now: float, target: int) -> bool:
        if now >= prev: return (prev <= target) and (target < now)
        else:           return (target >= prev) or (target < now)

    def _apply_defense(i_lead: int):
        L = min(curr_lap[i_lead], n_laps - 1)
        v = speed_pts_per_sec[L, i_lead]
        dist_pts = 0.06 * v  # ~0.06s cost
        new_pts = curr_pts[i_lead] - dist_pts
        if new_pts < 0: new_pts += P
        curr_pts[i_lead] = new_pts
        defended_this_lap[i_lead] = True

    def _attempt_pass(i_follow: int, i_lead: int, gap_at_det_s: float, tsec: float):
        nonlocal stats
        L = min(curr_lap[i_follow], n_laps - 1)
        pace_lead = lap_times[L, i_lead]
        pace_foll = lap_times[L, i_follow]
        delta_norm = (pace_lead - pace_foll) / base_lap_eff
        slip = DRS_SLIPSTREAM_BONUS if gap_at_det_s <= (det_eff * 1.2) else 0.0
        defended = 1.0 if defended_this_lap[i_lead] else 0.0

        # Personality multipliers
        att_mult = 1.0 + 0.8 * agg_w * (agg[i_follow] - 0.5) if use_personality else 1.0
        def_mult = 1.0 + 0.8 * def_w * (defence[i_lead] - 0.5) if use_personality else 1.0

        dirty = float(otk["dirty_air_penalty"])
        logit = (alpha_eff * att_mult) * delta_norm + float(otk["beta_drs"]) * slip - (float(otk["gamma_defend"]) * def_mult) * defended - dirty
        p = 1.0 / (1.0 + math.exp(-logit))
        stats["attempts"] += 1
        if np.isfinite(p) and (r_pass.random() < p):
            car_len = max(1, int(P * 0.0025))
            curr_pts[i_follow] = (curr_pts[i_lead] + car_len) % P
            curr_pts[i_lead] = (curr_pts[i_lead] - car_len) % P
            event_log.append((tsec, f"PASS: {drivers[i_follow]} → {drivers[i_lead]} (DRS)"))
            stats["passes"] += 1

    def _log(msg: str, tsec: float):
        event_log.append((tsec, msg))

    sim_time = 0.0
    max_time = n_laps * float(np.max(lap_times))
    T_max = int(math.ceil(max_time / dt)) + 1  # use dt (arg), not DT (module)

    for _ in range(T_max):
        # base speeds
        speeds = np.zeros(D, dtype=float)
        active = curr_lap < n_laps
        if active.any():
            L = np.clip(curr_lap, 0, n_laps - 1)
            base_speed = speed_pts_per_sec[L, np.arange(D)]
            phase_factor = 1.0 if phase == "GREEN" else (VSC_SPEED_FACTOR if phase == "VSC" else SC_SPEED_FACTOR)
            speeds[active] = base_speed[active] * phase_factor

        last_pts = curr_pts.copy()
        curr_pts[active] = (curr_pts[active] + speeds[active] * dt) % P

        # Lap crossings
        crossed = (curr_pts < last_pts) & active
        if crossed.any():
            curr_lap[crossed] += 1
            defended_this_lap[crossed] = False
            for z in drs_eligible.values():
                for di in list(z.keys()):
                    if crossed[di]:
                        z.pop(di, None)
            indices = np.where(crossed)[0]
            for di in indices:
                if r_dnf.random() < float(p_dnf_per_lap_vec[di]):
                    curr_lap[di] = n_laps
                    _log(f"DNF: {drivers[di]}", sim_time)
                    stats["dnfs"].append(drivers[di])

        leader_completed = int(curr_lap.max() if curr_lap.size else 0)

        # Incidents -> SC/VSC
        if phase == "GREEN" and leader_completed > 0 and r_inc.random() < float(incident_rate):
            if r_inc.random() < SC_SHARE:
                phase = "SC"
                sc_laps_remaining = r_inc.integers(SC_DURATION_LAPS_MINMAX[0], SC_DURATION_LAPS_MINMAX[1] + 1)
                order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
                lead = order[0]; base_pos = curr_pts[lead]
                for rank, di in enumerate(order):
                    gap_pts = int(SC_STAGGER_GAP_FRAC * P * rank)
                    pos = base_pos - gap_pts
                    while pos < 0: pos += P
                    curr_pts[di] = pos; curr_lap[di] = curr_lap[lead]
                for z in drs_eligible.values(): z.clear()
                defended_this_lap[:] = False
                drs_disabled_until_lap = leader_completed + 1 + DRS_COOLDOWN_AFTER_SC_LAPS
                _log("SAFETY CAR DEPLOYED (yellow)", sim_time)
            else:
                phase = "VSC"
                vsec = float(r_inc.uniform(VSC_DURATION_SEC_MINMAX[0], VSC_DURATION_SEC_MINMAX[1]))
                vsc_ticks_remaining = max(1, int(vsec / dt))
                _log("VIRTUAL SAFETY CAR DEPLOYED", sim_time)

        if phase == "VSC":
            vsc_ticks_remaining -= 1
            if vsc_ticks_remaining <= 0:
                phase = "GREEN"
                _log("VSC END — GREEN FLAG", sim_time)
        elif phase == "SC":
            if leader_completed > 0:
                sc_laps_remaining -= 1
                if sc_laps_remaining <= 0:
                    phase = "GREEN"
                    drs_disabled_until_lap = leader_completed + 1 + DRS_COOLDOWN_AFTER_SC_LAPS
                    _log("SAFETY CAR IN — GREEN FLAG (DRS delayed)", sim_time)

        # DRS & blocking logic
        drs_enabled = (phase == "GREEN") and ((leader_completed + 1) >= max(DRS_MIN_ENABLE_LAP, drs_disabled_until_lap))
        has_drs = np.zeros(D, dtype=bool)

        if phase == "GREEN":
            order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
            # simple blocking outside zones
            BLOCKING_THRESH_S = 0.8
            for k in range(1, len(order)):
                i_lead = order[k - 1]; i_foll = order[k]
                if curr_lap[i_lead] >= n_laps or curr_lap[i_foll] >= n_laps:
                    continue
                in_any_zone = any(((zs <= curr_pts[i_foll] <= ze) if zs <= ze
                                   else (curr_pts[i_foll] >= zs or curr_pts[i_foll] <= ze))
                                  for (zs, ze, _) in zones)
                if not in_any_zone:
                    gap_s = _gap_seconds(i_foll, i_lead)
                    # personality-aware defend prob
                    if use_personality:
                        base_defend = 0.55
                        P_DEFEND = float(np.clip(base_defend * (1.0 + 0.6 * def_w * (defence[i_lead] - 0.5)), 0.20, 0.95))
                    else:
                        P_DEFEND = 0.55
                    if (gap_s <= BLOCKING_THRESH_S) and (not defended_this_lap[i_lead]) and (r_pass.random() < P_DEFEND):
                        _apply_defense(i_lead)
                        _log(f"DEFENSE: {drivers[i_lead]} blocks {drivers[i_foll]} (+0.06s)", sim_time)

        # DRS detection & passes
        if phase == "GREEN":
            order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
            for zid, (zs, ze, zd) in enumerate(zones):
                if drs_enabled:
                    for idx in order[1:]:
                        if curr_lap[idx] >= n_laps:
                            continue
                        if _in_range(last_pts[idx], curr_pts[idx], zd):
                            pos = list(order).index(idx)
                            if pos > 0:
                                ahead = order[pos - 1]
                                if curr_lap[ahead] >= n_laps:
                                    continue
                                gap_s = _gap_seconds(idx, ahead)
                                if gap_s <= det_eff:
                                    drs_eligible[zid][idx] = (ahead, gap_s)
                                    has_drs[idx] = True
                                else:
                                    drs_eligible[zid].pop(idx, None)
                for idx, (ahead, gap_at_det) in list(drs_eligible[zid].items()):
                    if curr_lap[idx] >= n_laps or curr_lap[ahead] >= n_laps:
                        drs_eligible[zid].pop(idx, None)
                        continue
                    if _in_range(last_pts[idx], curr_pts[idx], ze):
                        curr_order = np.argsort(-(curr_lap.astype(float) + (curr_pts / P)))
                        pos = list(curr_order).index(idx)
                        if pos > 0 and curr_order[pos - 1] == ahead:
                            _attempt_pass(idx, ahead, gap_at_det, sim_time)
                        drs_eligible[zid].pop(idx, None)

        # Frame buffers
        pos_idx = np.mod(curr_pts.astype(int), P)
        positions_list.append(np.stack([xy_path[pos_idx, 0], xy_path[pos_idx, 1]], axis=1))
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
                    g = _gap_seconds(idx, order[0])
                    gaps.append("+—" if g >= 98.0 else f"+{g:,.3f}")
                if drs_enabled and has_drs[idx]:
                    gaps[-1] += f" {DRS_TAG}"
        gaps_panel.append(gaps)

        recent = [f"t={sim_time:5.1f}s  {m}" for (t, m) in event_log if t <= sim_time]
        wrapped = [_wrap_line(r, width=RC_WRAP_WIDTH) for r in recent[-12:]]
        rc_texts.append(wrapped if wrapped else ["—"])

        sim_time += dt
        if (curr_lap >= n_laps).all():
            break

    positions = np.stack(positions_list, axis=0)
    lap_key = np.stack(lapkey_list, axis=0)
    leader_lap = np.array(leaderlap_list, dtype=int)

    # Finish stats
    final_order = orders[-1] if orders else list(range(D))
    stats["finish_order"] = final_order
    stats["finish_entropy"] = _finish_order_entropy(stats["grid_order"], final_order)

    return (positions, lap_key, leader_lap, drivers,
            phase_flags, rc_texts, drs_on_flags, drs_banner,
            orders, gaps_panel, zones, alpha_eff, det_eff, stats)

# ------------------- Figure -------------------
def _flag_style(phase: str) -> Tuple[str, str]:
    if phase == "SC":  return ("SAFETY CAR", "#ffd400")
    if phase == "VSC": return ("VIRTUAL SAFETY CAR", "#ffb347")
    return ("GREEN FLAG", "#2ecc71")

def _track_color(phase: str) -> str:
    if phase == "SC":  return "rgba(255,212,0,0.70)"
    if phase == "VSC": return "rgba(255,165,0,0.45)"
    return "rgba(80,90,110,0.25)"

def _fmt_weather_overlay(ws: Optional[dict]) -> str:
    if not ws:
        return "Weather: n/a"
    ttrk = ws.get("median_track_temp_c")
    tair = ws.get("median_air_temp_c")
    wind = ws.get("median_wind_kph")
    hum = ws.get("median_humidity_pct")
    def _f(val, suf):
        return f"{val:.0f}{suf}" if (val is not None and np.isfinite(val)) else "—"
    return (f"Weather (median): Track {_f(ttrk,'°C')}  |  Air {_f(tair,'°C')}  |  "
            f"Wind {_f(wind,' kph')}  |  Hum {_f(hum,' %')}")

def build_animation(
    positions, lap_key, leader_lap, drivers, name_map, color_map, xy_path, n_laps,
    phase_flags, rc_texts, drs_on, drs_banner, orders, gaps_panel, zones, alpha_eff, det_eff,
    weather_summary: Optional[dict] = None,
) -> go.Figure:
    T, D, _ = positions.shape
    labels = [str(dr).upper()[:3] for dr in drivers]
    names = [name_map.get(dr, dr) for dr in drivers]
    colors = [color_map.get(dr, "#888888") for dr in drivers]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy", "rowspan": 2}, {"type": "table"}],
               [None, {"type": "table"}]],
        column_widths=[0.60, 0.40],
        row_heights=[0.72, 0.28],
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
        subplot_titles=("Equal-Car Replay", "Leaderboard (live)", "Race Control"),
    )

    track_col0 = _track_color(phase_flags[0] if len(phase_flags) else "GREEN")
    fig.add_trace(go.Scatter(x=xy_path[:,0], y=xy_path[:,1], mode="lines",
                             line=dict(width=2, color=track_col0),
                             hoverinfo="skip", showlegend=False, name="Track"), row=1, col=1)

    mode0 = "markers+text" if SHOW_MARKER_LABELS else "markers"
    fig.add_trace(go.Scatter(x=positions[0,:,0], y=positions[0,:,1], mode=mode0,
                             text=(labels if SHOW_MARKER_LABELS else None), textposition="top center",
                             marker=dict(size=12, line=dict(width=1, color="#222"), color=colors),
                             hovertext=names, hoverinfo="text", showlegend=True, name="Cars"), row=1, col=1)

    fig.add_trace(go.Scatter(), row=1, col=1)

    for nm, col in zip(names, colors):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=12, color=col, line=dict(width=1, color="#222")),
                                 name=nm, showlegend=True), row=1, col=1)

    order0 = orders[0]
    table_positions = list(range(1, len(drivers)+1))
    table_drivers = [labels[i] for i in order0]
    table_gaps = gaps_panel[0]
    leaderboard = go.Table(
        columnwidth=[0.22, 0.38, 0.40],
        header=dict(values=["<b>Position</b>", "<b>Driver</b>", "<b>Time</b>"],
                    fill_color="#2b2f3a", font=dict(color="white", size=13), align="left"),
        cells=dict(values=[table_positions, table_drivers, table_gaps],
                   fill_color="#0f1720", font=dict(color="white", size=12),
                   align="left", height=22)
    )
    fig.add_trace(leaderboard, row=1, col=2)

    rc0_lines = rc_texts[0] if len(rc_texts) else ["—"]
    rc_table = go.Table(
        columnwidth=[1.0],
        header=dict(values=["<b>Race Control</b>"],
                    fill_color="#2b2f3a", font=dict(color="white", size=13), align="left"),
        cells=dict(values=[rc0_lines],
                   fill_color="#0f1720", font=dict(color="white", size=12),
                   align="left", height=22)
    )
    fig.add_trace(rc_table, row=2, col=2)

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

    pad = 0.08
    x_all = positions[:,:,0].ravel(); y_all = positions[:,:,1].ravel()
    xmin, xmax = float(x_all.min())-pad, float(x_all.max())+pad
    ymin, ymax = float(y_all.min())-pad, float(y_all.max())+pad
    fig.update_xaxes(range=[xmin, xmax], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[ymin, ymax], scaleanchor="x", scaleratio=1,
                     showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)

    def _btn(speed: float, label: str):
        frame_ms = max(5, int(1000 * DT / speed))
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
    initial_lap_disp = int(np.clip(leader_lap[0] if len(leader_lap) else 1, 1, n_laps))
    wx_text = _fmt_weather_overlay(weather_summary)

    fig.update_layout(
        height=920,
        margin=dict(l=16, r=16, t=168, b=20),
        legend=dict(title="Drivers", x=0.01, y=0.98, bgcolor="rgba(255,255,255,0.6)"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.48, y=1.20, xanchor="center", buttons=buttons)],
        title=f"Equal-Car Replay  •  DRSα={alpha_eff:.2f}  det≈{det_eff:.2f}s",
        annotations=[
            dict(text=f"Lap {initial_lap_disp} / {n_laps}", showarrow=False, x=0.21, y=1.21, xref="paper", yref="paper",
                 font=dict(size=14, color="#1f2d3d")),
            dict(text=banner_text, showarrow=False, x=0.50, y=1.21, xref="paper", yref="paper",
                 font=dict(size=14, color="#1f2d3d"), bgcolor=banner_bg),
            dict(text=drs_tag, showarrow=False, x=0.78, y=1.21, xref="paper", yref="paper",
                 font=dict(size=14, color="#1f2d3d"),
                 bgcolor="#e8f1ff" if ("ENABLED" in drs_tag) else "#ffd6d6"),
            dict(text=wx_text, showarrow=False, x=0.01, y=1.12, xanchor="left", xref="paper", yref="paper",
                 font=dict(size=13, color="#1f2d3d"),
                 bgcolor="#eef6ff"),
        ],
        paper_bgcolor="#f2f6fb"
    )

    frames = []
    for ti in range(positions.shape[0]):
        order = orders[ti]
        table_positions = list(range(1, len(drivers) + 1))
        table_drivers = [labels[i] for i in order]
        table_gaps = gaps_panel[ti]

        lap_disp = int(np.clip(leader_lap[ti], 1, n_laps))
        banner_text, banner_bg = _flag_style(phase_flags[ti])
        track_col = _track_color(phase_flags[ti])
        sc_marker = (
            go.Scatter(
                x=[xy_path[0, 0]],
                y=[xy_path[0, 1]],
                mode="markers",
                marker=dict(size=14, color="rgba(255,212,0,0.9)", symbol="square"),
                showlegend=False,
                name="SC",
            )
            if phase_flags[ti] == "SC"
            else go.Scatter()
        )
        rc_lines = rc_texts[ti]

        frame_data = [
            go.Scatter(x=xy_path[:, 0], y=xy_path[:, 1], line=dict(width=2, color=track_col)),
            go.Scatter(
                x=positions[ti, :, 0],
                y=positions[ti, :, 1],
                mode=("markers+text" if SHOW_MARKER_LABELS else "markers"),
                text=(labels if SHOW_MARKER_LABELS else None),
            ),
            sc_marker,
        ] + [go.Scatter() for _ in range(len(drivers))] + [
            go.Table(
                columnwidth=[0.22, 0.38, 0.40],
                header=dict(values=["<b>Position</b>", "<b>Driver</b>", "<b>Time</b>"],
                            fill_color="#2b2f3a", font=dict(color="white", size=13), align="left"),
                cells=dict(values=[table_positions, table_drivers, table_gaps],
                           fill_color="#0f1720", font=dict(color="white", size=12),
                           align="left", height=22),
            ),
            go.Table(
                columnwidth=[1.0],
                header=dict(values=["<b>Race Control</b>"],
                            fill_color="#2b2f3a", font=dict(color="white", size=13), align="left"),
                cells=dict(values=[rc_lines],
                           fill_color="#0f1720", font=dict(color="white", size=12),
                           align="left", height=22),
            ),
        ]

        frames.append(
            go.Frame(
                data=frame_data,
                layout=go.Layout(
                    annotations=[
                        dict(text=f"Lap {lap_disp} / {n_laps}", showarrow=False, x=0.21, y=1.21, xref="paper", yref="paper",
                             font=dict(size=14, color="#1f2d3d")),
                        dict(text=banner_text, showarrow=False, x=0.50, y=1.21, xref="paper", yref="paper",
                             font=dict(size=14, color="#1f2d3d"), bgcolor=banner_bg),
                        dict(text=("DRS ENABLED" if "ENABLED" in drs_banner[ti] else "DRS DISABLED"),
                             showarrow=False, x=0.78, y=1.21, xref="paper", yref="paper",
                             font=dict(size=14, color="#1f2d3d"),
                             bgcolor="#e8f1ff" if ("ENABLED" in drs_banner[ti]) else "#ffd6d6"),
                        dict(text=_fmt_weather_overlay(weather_summary), showarrow=False,
                             x=0.01, y=1.12, xanchor="left", xref="paper", yref="paper",
                             font=dict(size=13, color="#1f2d3d"), bgcolor="#eef6ff"),
                    ]
                ),
                name=str(ti),
            )
        )

    fig.frames = frames
    return fig

def _json_default(o):
    """Make numpy/pandas types JSON-serializable."""
    import numpy as _np
    import pandas as _pd
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.ndarray,)):
        return o.tolist()
    if isinstance(o, (_pd.Timestamp, _np.datetime64)):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    # Fallback — last resort stringify
    return str(o)

# ------------------- Main -------------------
def main():
    np.random.seed(RANDOM_SEED)
    cfg = load_config("config/config.yaml")
    if "cache_dir" in cfg: enable_cache(cfg["cache_dir"])

    # Read high-level sim knobs if present
    vizsec = _cfg_get(cfg, ["visualize_equal_race"], {}) or {}
    use_evt = bool(_cfg_get(vizsec, ["use_event_specific_deltas"], USE_EVENT_SPECIFIC_DELTAS))
    gp_sub = str(_cfg_get(vizsec, ["target_gp_substr"], TARGET_GP_SUBSTR)) or TARGET_GP_SUBSTR

    # Base lap/time-step/laps optional overrides
    base_lap = float(_cfg_get(vizsec, ["base_lap_sec"], BASE_LAP_SEC))
    n_laps = int(_cfg_get(vizsec, ["n_laps"], N_LAPS))
    dt = float(_cfg_get(vizsec, ["dt"], DT))
    noise_sd = float(_cfg_get(vizsec, ["lap_jitter_sd"], LAP_JITTER_SD))

    # Seeds & run index for determinism/diversity
    seed = int(_cfg_get(vizsec, ["seed"], RANDOM_SEED))
    run_idx = int(_cfg_get(vizsec, ["run_idx"], 0))

    # Track metadata (optional)
    meta = _load_viz_track_meta(cfg)
    # Weather summary (preferred from event["weather_summary"]; fallback to laps medians)
    weather_summary = _load_weather_summary_for_viz(cfg)

    # Deltas: event-specific or global
    if use_evt:
        ranking = load_driver_ranking_event(cfg, gp_sub.lower())
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
     orders, gaps_panel, zones, alpha_eff, det_eff, stats) = simulate_progress(
        ranking, xy, base_lap=base_lap, n_laps=n_laps, dt=dt,
        noise_sd=noise_sd, seed=seed, cfg=cfg, meta=meta, weather_summary=weather_summary, run_idx=run_idx
    )

    fig = build_animation(
        positions, lap_key, leader_lap, drivers, name_map, color_map, xy, n_laps,
        phase_flags, rc_texts, drs_on, drs_banner, orders, gaps_panel, zones, alpha_eff, det_eff,
        weather_summary=weather_summary
    )

    out_path = OUT_DIR / "simulation.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    print(f"[INFO] Wrote visualization: {out_path}")

    # Per-run log for reproducibility and MC stats
    log_path = OUT_DIR / f"simulation_run_log_run{run_idx}.json"
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2, default=_json_default)
    print(f"[INFO] Wrote run log: {log_path}")

if __name__ == "__main__":
    main()
