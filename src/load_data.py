# src/load_data.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import warnings
import numpy as np
import fastf1
import yaml
import pandas as pd
import logging

logging.getLogger("fastf1").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning, module="fastf1.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")


# -------- Paths & Config --------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    cfg_path = _project_root() / config_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------- FastF1 Cache --------
def enable_cache(cache_dir: str) -> None:
    cache_dir_abs = (_project_root() / cache_dir).resolve()
    cache_dir_abs.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir_abs))


# -------- Recent events (example list; override via config if needed) --------
def get_recent_races(_: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Keep this as a simple example—pull from config if you want dynamic control
    return [
        {"year": 2025, "grand_prix": "Hungarian Grand Prix", "session": "R"},
        {"year": 2025, "grand_prix": "Belgian Grand Prix",   "session": "R"},
        {"year": 2025, "grand_prix": "British Grand Prix",   "session": "R"},
        {"year": 2025, "grand_prix": "Austrian Grand Prix",  "session": "R"},
        {"year": 2025, "grand_prix": "Canadian Grand Prix",  "session": "R"},
    ]


# -------- Helpers --------
def _to_secs(x) -> float:
    try:
        return float(getattr(x, "total_seconds", lambda: float(x))())
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan


def _standardize_lap_seconds(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "LapTimeSeconds" not in d.columns:
        if "LapTime" in d.columns:
            d["LapTimeSeconds"] = d["LapTime"].apply(_to_secs)
        else:
            d["LapTimeSeconds"] = pd.to_numeric(d.get("LapTime", np.nan), errors="coerce")
    return d


def _is_green(track_status: pd.Series) -> pd.Series:
    """
    FastF1 TrackStatus: '1' means green conditions.
    Some weekends show composite codes (e.g., '451'); we require a pure '1' for pace laps.
    """
    s = track_status.astype(str).fillna("")
    return s == "1"


def _derive_and_filter_tags(laps: pd.DataFrame, *, session_kind: str) -> pd.DataFrame:
    """
    Derive standardized tags and apply the strict pace-lap filter (lap_ok) immediately.
    session_kind: 'R' or 'Q' (affects only logging text; logic identical).
    """
    d = _standardize_lap_seconds(laps).reset_index(drop=True)

    # --- Canonical driver/team/compound/event fields ---
    if "Driver" in d.columns:
        d["driver"] = d["Driver"].astype(str)
    elif "DriverNumber" in d.columns:
        d["driver"] = d["DriverNumber"].astype(str)
    else:
        d["driver"] = d.get("DriverNumber", d.get("Driver", "UNK")).astype(str)

    if "Team" in d.columns:
        d["Team"] = d["Team"].astype(str)
    else:
        d["Team"] = d.get("Team", "UNK").astype(str)

    d["compound"] = d.get("Compound", "UNKNOWN").fillna("UNKNOWN").astype(str)

    # Event label if present (optional; used downstream for FEs)
    ev_col = "Event" if "Event" in d.columns else ("EventName" if "EventName" in d.columns else None)
    if ev_col:
        d["Event"] = d[ev_col].astype(str)

    # --- Ensure LapNumber exists and is integer ---
    if "LapNumber" not in d.columns:
        d = d.sort_values(["driver", "LapTimeSeconds"]).copy()
        d["LapNumber"] = d.groupby("driver").cumcount() + 1
    d["lap_number"] = pd.to_numeric(d["LapNumber"], errors="coerce").fillna(0).astype(int)

    # --- Track status & timing accuracy ---
    d["track_status"] = d.get("TrackStatus", "").astype(str)
    is_green = _is_green(d["track_status"])
    is_accurate = d.get("IsAccurate", True)
    if isinstance(is_accurate, (pd.Series,)):
        is_accurate = is_accurate.fillna(True).astype(bool)

    # --- Pit in/out flags to identify in/out laps ---
    for col in ("PitInTime", "PitOutTime"):
        if col not in d.columns:
            d[col] = pd.NaT
    is_outlap = d["PitOutTime"].notna()
    is_inlap = d["PitInTime"].notna()

    # --- Positive, valid lap time ---
    has_time = d["LapTimeSeconds"].notna() & (d["LapTimeSeconds"] > 0)

    # --- Strict pace-lap definition ---
    d["lap_ok"] = has_time & is_accurate & (~is_outlap) & (~is_inlap) & is_green

    # --- Stint inference (robust) ---
    # Prefer provided Stint; otherwise start a new stint immediately after a pit OUT.
    if "Stint" in d.columns:
        stint = pd.to_numeric(d["Stint"], errors="coerce")
        # If some stints are missing, still fall back on pit-based inference for those rows
        missing = stint.isna()
        if missing.any():
            d = d.sort_values(["driver", "LapNumber"]).copy()
            inferred = d["PitOutTime"].notna().groupby(d["driver"]).cumsum()
            stint = stint.fillna(inferred)
        d["stint_id"] = stint.fillna(-1).astype(int)
    else:
        d = d.sort_values(["driver", "LapNumber"]).copy()
        d["stint_id"] = d["PitOutTime"].notna().groupby(d["driver"]).cumsum().astype(int)

    # --- lap_on_tyre counter (1-based within driver×stint) ---
    d = d.sort_values(["driver", "stint_id", "LapNumber"]).copy()
    d["lap_on_tyre"] = d.groupby(["driver", "stint_id"]).cumcount() + 1

    # --- Log drops for visibility, then FILTER immediately ---
    before = len(d)
    dropped = (~d["lap_ok"]).sum()
    kept = d["lap_ok"].sum()
    logging.info(
        f"[load_data] {session_kind}: kept {kept}/{before} pace laps "
        f"({dropped} dropped: non-green/in-out/inaccurate/invalid)."
    )

    d = d.loc[d["lap_ok"]].reset_index(drop=True)

    # Ensure all expected columns exist post-filter
    needed = [
        "LapTimeSeconds", "driver", "Team", "compound",
        "stint_id", "lap_on_tyre", "lap_number", "track_status", "lap_ok"
    ]
    for c in needed:
        if c not in d.columns:
            d[c] = np.nan if c != "lap_ok" else True

    return d


# -------- Session Loading --------
def load_session(year: int, grand_prix: str, session: str) -> Tuple[Optional[pd.DataFrame], Optional[Any]]:
    try:
        ses = fastf1.get_session(year, grand_prix, session)
        ses.load(laps=True, telemetry=False, weather=False)
        laps = ses.laps.reset_index(drop=True)
        return laps, ses
    except Exception as e:
        print(f"[WARN] Failed to load {year} {grand_prix} {session}: {e}")
        return None, None


def load_all_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    enable_cache(config["cache_dir"])
    races = get_recent_races(config)
    out: List[Dict[str, Any]] = []

    for race in races:
        year, gp = race["year"], race["grand_prix"]

        # --- Race ---
        race_laps_raw, _ = load_session(year, gp, "R")
        if race_laps_raw is None or len(race_laps_raw) == 0:
            print(f"[WARN] No race laps for {year} {gp}")
            continue
        race_laps = _derive_and_filter_tags(race_laps_raw, session_kind="R")

        entry: Dict[str, Any] = {
            "year": year,
            "gp": gp,
            "race_laps": race_laps,
        }

        # --- Quali (optional) ---
        if config.get("include_qualifying", True):
            quali_laps_raw, _ = load_session(year, gp, "Q")
            if quali_laps_raw is not None and len(quali_laps_raw) > 0:
                # Quali benefits from the same strict filtering (pace laps only)
                quali_laps = _derive_and_filter_tags(quali_laps_raw, session_kind="Q")
                entry["quali_laps"] = quali_laps
            else:
                entry["quali_laps"] = None

        out.append(entry)

    return out


# -------- Track Outline (fallbacks) --------
def get_track_outline(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    track_cfg = config.get("viz_track", {})
    year = int(track_cfg.get("year"))
    gp = str(track_cfg.get("grand_prix"))
    session = str(track_cfg.get("session", "R"))

    laps, ses = load_session(year, gp, session)
    if ses is None:
        print("[WARN] Could not load session for viz track.")
        return None

    try:
        ci = ses.get_circuit_info()
        if ci is not None and hasattr(ci, "coordinates") and ci.coordinates is not None:
            coords = ci.coordinates
            low = {c.lower(): c for c in coords.columns}
            if "x" in low and "y" in low:
                df = coords[[low["x"], low["y"]]].dropna().reset_index(drop=True)
                df.columns = ["x", "y"]
                return df
    except Exception:
        pass

    try:
        ses = fastf1.get_session(year, gp, session)
        ses.load(laps=True, telemetry=True, weather=False)
        laps = ses.laps
        if laps is None or len(laps) == 0:
            print("[WARN] No laps available to compute fallback XY.")
            return None
        fl = laps.pick_fastest()
        tel = fl.get_telemetry()
        low = {c.lower(): c for c in tel.columns}
        if "x" in low and "y" in low:
            df = tel[[low["x"], low["y"]]].dropna().reset_index(drop=True)
            df.columns = ["x", "y"]
            return df
    except Exception as e:
        print(f"[WARN] Fallback XY failed: {e}")

    print("[WARN] No track outline available.")
    return None


# -------- Manual Test Runner --------
if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    print(f"[INFO] Loaded config from: {(_project_root() / 'config/config.yaml').resolve()}")

    data = load_all_data(cfg)
    print(f"[INFO] Loaded events: {len(data)}")
    if data:
        first = data[0]
        print("[INFO] Example keys in first event:", list(first.keys()))
        cols = list(first["race_laps"].columns)
        needed = ["LapTimeSeconds", "driver", "compound", "stint_id", "lap_on_tyre",
                  "lap_number", "track_status", "lap_ok"]
        print("[INFO] Tagged fields present:", {k: (k in cols) for k in needed})
        print("[INFO] Race laps kept (pace-only):", len(first["race_laps"]))
        if first.get("quali_laps") is not None:
            print("[INFO] Quali laps kept (pace-only):", len(first["quali_laps"]))

    outline = get_track_outline(cfg)
    if outline is not None:
        print(f"[INFO] Track outline points: {len(outline)}")
