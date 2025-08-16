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


# -------- Hardcoded last-5 events (as of 2025-08-16) --------
def get_recent_races(_: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def _derive_race_tags(laps: pd.DataFrame) -> pd.DataFrame:
    d = _standardize_lap_seconds(laps).reset_index(drop=True)

    if "Driver" in d.columns:
        d["driver"] = d["Driver"].astype(str)
    elif "DriverNumber" in d.columns:
        d["driver"] = d["DriverNumber"].astype(str)
    else:
        d["driver"] = d.get("DriverNumber", d.get("Driver", "UNK")).astype(str)

    d["compound"] = d.get("Compound", "UNKNOWN").fillna("UNKNOWN").astype(str)

    if "Stint" in d.columns:
        d["stint_id"] = pd.to_numeric(d["Stint"], errors="coerce").fillna(-1).astype(int)
    else:
        pit_flag = (
            d.get("PitInTime").notna() | d.get("PitOutTime").notna()
            if ("PitInTime" in d or "PitOutTime" in d) else pd.Series(False, index=d.index)
        )
        d = d.sort_values(["driver", "LapNumber"] if "LapNumber" in d.columns else ["driver"]).copy()
        d["stint_id"] = pit_flag.groupby(d["driver"]).cumsum().astype(int)

    if "LapNumber" not in d.columns:
        d["LapNumber"] = d.groupby("driver").cumcount() + 1
    d = d.sort_values(["driver", "stint_id", "LapNumber"]).copy()
    d["lap_on_tyre"] = d.groupby(["driver", "stint_id"]).cumcount() + 1

    d["lap_number"] = pd.to_numeric(d["LapNumber"], errors="coerce").fillna(0).astype(int)

    if "TrackStatus" in d.columns:
        d["track_status"] = d["TrackStatus"].astype(str)
    else:
        d["track_status"] = ""

    if "IsAccurate" in d.columns:
        d["lap_ok"] = d["IsAccurate"].fillna(True).astype(bool)
    elif "LapIsValid" in d.columns:
        d["lap_ok"] = d["LapIsValid"].fillna(True).astype(bool)
    else:
        d["lap_ok"] = True

    for col in ("PitInTime", "PitOutTime"):
        if col not in d.columns:
            d[col] = pd.NaT

    d = d.sort_values(["driver", "LapNumber"]).reset_index(drop=True)
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

        race_laps, _ = load_session(year, gp, "R")
        if race_laps is None or len(race_laps) == 0:
            print(f"[WARN] No race laps for {year} {gp}")
            continue

        race_laps_tagged = _derive_race_tags(race_laps)

        entry: Dict[str, Any] = {
            "year": year,
            "gp": gp,
            "race_laps": race_laps_tagged,
        }

        if config.get("include_qualifying", True):
            quali_laps, _ = load_session(year, gp, "Q")
            entry["quali_laps"] = quali_laps

        out.append(entry)

    return out


# -------- Track Outline (Montreal preferred) --------
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
        needed = ["LapTimeSeconds", "driver", "compound", "stint_id", "lap_on_tyre", "lap_number", "track_status", "lap_ok"]
        print("[INFO] Tagged fields present:", {k: (k in cols) for k in needed})
        print("[INFO] Last 15 race columns (peek):", cols[-15:])

    outline = get_track_outline(cfg)
    if outline is not None:
        print(f"[INFO] Track outline points: {len(outline)}")
