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


# -------- Small helpers --------
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
    Some weekends show composite codes (e.g., '451'); here we require a pure '1' for pace laps.
    """
    s = track_status.astype(str).fillna("")
    return s == "1"


def _derive_and_filter_tags(laps: pd.DataFrame, *, session_kind: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Derive standardized tags and apply the strict pace-lap filter (lap_ok) immediately.
    session_kind: 'R' or 'Q' (affects only logging text; logic identical).

    Returns:
      - filtered laps (lap_ok == True)
      - qa_counts dict (drops by reason; non-exclusive)
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

    # --- QA counters (non-exclusive reasons) ---
    qa_counts = {
        "total_rows": int(len(d)),
        "dropped_non_green": int((~is_green).sum()),
        "dropped_inlap": int(is_inlap.sum()),
        "dropped_outlap": int(is_outlap.sum()),
        "dropped_inaccurate": int((~is_accurate).sum()),
        "dropped_invalid_time": int((~has_time).sum()),
        "dropped_total": int((~d["lap_ok"]).sum()),
        "kept_total": int(d["lap_ok"].sum()),
    }

    # --- Filter immediately for modeling consumption ---
    before = len(d)
    d = d.loc[d["lap_ok"]].reset_index(drop=True)
    kept = len(d)
    dropped = before - kept

    logging.info(
        f"[load_data] {session_kind}: kept {kept}/{before} pace laps "
        f"({dropped} dropped). Reasons (non-exclusive): "
        f"invalid_time={qa_counts['dropped_invalid_time']}, "
        f"inaccurate={qa_counts['dropped_inaccurate']}, "
        f"inlap={qa_counts['dropped_inlap']}, outlap={qa_counts['dropped_outlap']}, "
        f"non_green={qa_counts['dropped_non_green']}."
    )

    # Per-event/team/driver totals of kept laps (lightweight logging)
    try:
        kept_by = d.groupby(["Team", "driver"], dropna=False).size().sort_values(ascending=False).head(10)
        logging.info(f"[load_data] {session_kind}: top-kept Team×Driver (first 10):\n{kept_by}")
    except Exception:
        pass

    # Ensure all expected columns exist post-filter
    needed = [
        "LapTimeSeconds", "driver", "Team", "compound",
        "stint_id", "lap_on_tyre", "lap_number", "track_status", "lap_ok"
    ]
    for c in needed:
        if c not in d.columns:
            d[c] = np.nan if c != "lap_ok" else True

    return d, qa_counts


def _tag_stints_no_filter(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same stint/lap_on_tyre logic to RAW laps (no filtering).
    Used for the interaction table so it includes in/out/neutralized laps but with consistent tags.
    """
    d = _standardize_lap_seconds(laps).reset_index(drop=True)

    # driver / team / compound
    if "Driver" in d.columns:
        d["driver"] = d["Driver"].astype(str)
    elif "DriverNumber" in d.columns:
        d["driver"] = d["DriverNumber"].astype(str)
    else:
        d["driver"] = d.get("DriverNumber", d.get("Driver", "UNK")).astype(str)
    d["Team"] = d.get("Team", "UNK").astype(str)
    d["compound"] = d.get("Compound", "UNKNOWN").fillna("UNKNOWN").astype(str)

    # LapNumber
    if "LapNumber" not in d.columns:
        d = d.sort_values(["driver", "LapTimeSeconds"]).copy()
        d["LapNumber"] = d.groupby("driver").cumcount() + 1
    d["lap_number"] = pd.to_numeric(d["LapNumber"], errors="coerce").fillna(0).astype(int)

    # Pit flags
    for col in ("PitInTime", "PitOutTime"):
        if col not in d.columns:
            d[col] = pd.NaT

    # Stint
    if "Stint" in d.columns:
        stint = pd.to_numeric(d["Stint"], errors="coerce")
        missing = stint.isna()
        if missing.any():
            d = d.sort_values(["driver", "LapNumber"]).copy()
            inferred = d["PitOutTime"].notna().groupby(d["driver"]).cumsum()
            stint = stint.fillna(inferred)
        d["stint_id"] = stint.fillna(-1).astype(int)
    else:
        d = d.sort_values(["driver", "LapNumber"]).copy()
        d["stint_id"] = d["PitOutTime"].notna().groupby(d["driver"]).cumsum().astype(int)

    # lap_on_tyre
    d = d.sort_values(["driver", "stint_id", "LapNumber"]).copy()
    d["lap_on_tyre"] = d.groupby(["driver", "stint_id"]).cumcount() + 1

    return d


def _build_interaction_table(
    laps_raw: pd.DataFrame,
    ses: Any,
    *,
    session_kind: str,
) -> pd.DataFrame:
    """
    Construct an auxiliary interaction table from RAW laps (no lap_ok filter):
      - driver, team, compound, stint_id, lap_on_tyre, lap_number
      - position (if available), lap-to-lap position change
      - pit in/out flags
      - track flags: green/yellow/SC/VSC (from TrackStatus)
      - weather (air/track temps) if available
      - lap_ok + component reasons (for reference)
      - placeholders for drs_active/available, gap_ahead/behind (NaN unless available)
    """
    if laps_raw is None or len(laps_raw) == 0:
        return pd.DataFrame()

    d = _tag_stints_no_filter(laps_raw)

    # Minimal time/validity & flags
    d["track_status"] = d.get("TrackStatus", "").astype(str)
    is_green = _is_green(d["track_status"])
    is_accurate = d.get("IsAccurate", True)
    if isinstance(is_accurate, (pd.Series,)):
        is_accurate = is_accurate.fillna(True).astype(bool)
    is_outlap = d["PitOutTime"].notna()
    is_inlap = d["PitInTime"].notna()
    has_time = d["LapTimeSeconds"].notna() & (d["LapTimeSeconds"] > 0)
    d["lap_ok"] = has_time & is_accurate & (~is_outlap) & (~is_inlap) & is_green

    # Position & lap-to-lap change (if available)
    if "Position" in d.columns:
        d["position"] = pd.to_numeric(d["Position"], errors="coerce")
        d["pos_change"] = d.groupby("driver")["position"].shift(1) - d["position"]
    else:
        d["position"] = np.nan
        d["pos_change"] = np.nan

    # Track flags
    s = d["track_status"].astype(str).fillna("")
    d["flag_yellow"] = s.str.contains("2") | s.str.contains("3")
    d["flag_sc"] = s.str.contains("4")
    d["flag_vsc"] = s.str.contains("5")

    # Weather: merge nearest AirTemp / TrackTemp if available
    d["ambient_temp_c"] = np.nan
    d["track_temp_c"] = np.nan
    try:
        # Weather data is indexed by absolute session time; map via lap mid time if available
        if hasattr(ses, "weather_data") and ses.weather_data is not None and len(ses.weather_data) > 0:
            wx = ses.weather_data.copy()
            # Pick columns robustly
            low = {c.lower(): c for c in wx.columns}
            air = low.get("airtemp")
            track = low.get("tracktemp")
            time_col = low.get("time", "Time")
            if air and track and time_col in wx.columns:
                wx = wx[[time_col, air, track]].rename(columns={air: "ambient_temp_c", track: "track_temp_c"})
                # Estimate a per-lap reference time; fall back to LapStartTime if present
                if "LapStartTime" in d.columns:
                    ref = d["LapStartTime"]
                elif "Time" in d.columns:
                    ref = d["Time"]
                else:
                    ref = None
                if ref is not None:
                    # Merge by nearest time within a tolerance
                    # Convert to pandas Timedelta for merge_asof
                    ref_td = pd.to_timedelta(ref)
                    wx_td = pd.to_timedelta(wx[time_col])
                    dd = d.copy()
                    dd["__ref_time__"] = ref_td
                    wx2 = wx.copy()
                    wx2["__wx_time__"] = wx_td
                    dd = dd.sort_values("__ref_time__")
                    wx2 = wx2.sort_values("__wx_time__")
                    merged = pd.merge_asof(
                        dd, wx2,
                        left_on="__ref_time__", right_on="__wx_time__", direction="nearest", tolerance=pd.Timedelta("120s")
                    )
                    d["ambient_temp_c"] = merged["ambient_temp_c"].values
                    d["track_temp_c"] = merged["track_temp_c"].values
    except Exception:
        pass

    # Placeholders for gaps/DRS (not generally available in Laps without telemetry)
    d["gap_ahead_s"] = np.nan
    d["gap_behind_s"] = np.nan
    d["drs_active"] = pd.NA
    d["drs_available"] = pd.NA

    # Select and rename for clarity
    out_cols = [
        "driver", "Team", "compound", "stint_id", "lap_on_tyre", "lap_number",
        "LapTimeSeconds", "position", "pos_change",
        "gap_ahead_s", "gap_behind_s", "drs_active", "drs_available",
        "flag_yellow", "flag_sc", "flag_vsc", "track_status",
        "ambient_temp_c", "track_temp_c",
        "PitInTime", "PitOutTime",
        "lap_ok",
    ]
    for c in out_cols:
        if c not in d.columns:
            d[c] = np.nan
    out = d[out_cols].rename(columns={"Team": "team"})
    return out.reset_index(drop=True)


def _compute_start_deltas(laps_raw: pd.DataFrame, ses: Any) -> pd.DataFrame:
    """
    Compute start deltas: GridPosition vs position at end of Lap 1.
    Positive value => positions gained on Lap 1.
    """
    if laps_raw is None or len(laps_raw) == 0 or ses is None:
        return pd.DataFrame(columns=["driver", "team", "grid_pos", "pos_end_lap1", "start_delta"])

    # Lap 1 positions from raw laps
    d = _tag_stints_no_filter(laps_raw)
    if "Position" not in d.columns:
        return pd.DataFrame(columns=["driver", "team", "grid_pos", "pos_end_lap1", "start_delta"])

    lap1 = d[d["lap_number"] == 1].copy()
    lap1["pos_end_lap1"] = pd.to_numeric(lap1["Position"], errors="coerce")

    # Grid positions from classification if available
    grid = None
    try:
        res = ses.results
        if res is not None and len(res) > 0:
            low = {c.lower(): c for c in res.columns}
            dn = low.get("drivernumber", None)
            abbr = low.get("abbreviation", None)
            gridcol = low.get("gridposition", None)
            teamcol = low.get("teamname", low.get("team", None))
            # Prefer DriverNumber to map to laps
            if dn and gridcol:
                grid = res[[dn, gridcol] + ([teamcol] if teamcol else [])].rename(
                    columns={dn: "driver", gridcol: "grid_pos"}
                )
                grid["driver"] = grid["driver"].astype(str)
                if teamcol:
                    grid = grid.rename(columns={teamcol: "team"})
            elif abbr and gridcol:
                grid = res[[abbr, gridcol] + ([teamcol] if teamcol else [])].rename(
                    columns={abbr: "driver", gridcol: "grid_pos"}
                )
                grid["driver"] = grid["driver"].astype(str)
                if teamcol:
                    grid = grid.rename(columns={teamcol: "team"})
    except Exception:
        grid = None

    if grid is None:
        # Fallback: use lap1 order as a proxy (not perfect, but better than empty)
        lap1 = lap1.sort_values("pos_end_lap1")
        lap1["grid_pos"] = lap1["pos_end_lap1"]
        lap1["team"] = lap1.get("Team", "UNK").astype(str)
        out = lap1[["driver", "team", "grid_pos", "pos_end_lap1"]].copy()
    else:
        out = pd.merge(
            lap1[["driver", "Team", "pos_end_lap1"]].rename(columns={"Team": "team"}),
            grid[["driver", "grid_pos"] + (["team"] if "team" in grid.columns else [])],
            on="driver", how="left", suffixes=("", "_grid")
        )
        # If team missing from grid, fill from lap1
        if "team" not in out.columns:
            out["team"] = lap1["Team"].astype(str)

    out["grid_pos"] = pd.to_numeric(out["grid_pos"], errors="coerce")
    out["start_delta"] = out["grid_pos"] - out["pos_end_lap1"]
    return out[["driver", "team", "grid_pos", "pos_end_lap1", "start_delta"]].dropna(subset=["pos_end_lap1"]).reset_index(drop=True)


# -------- Session Loading --------
def load_session(year: int, grand_prix: str, session: str) -> Tuple[Optional[pd.DataFrame], Optional[Any]]:
    """
    Load laps and session object.
    NOTE: weather=True so we can merge air/track temps into the interaction table if available.
    """
    try:
        ses = fastf1.get_session(year, grand_prix, session)
        ses.load(laps=True, telemetry=False, weather=True)
        laps = ses.laps.reset_index(drop=True)
        return laps, ses
    except Exception as e:
        print(f"[WARN] Failed to load {year} {grand_prix} {session}: {e}")
        return None, None


def load_all_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of event dicts with:
      - 'race_laps': filtered pace laps for modeling (lap_ok == True)
      - 'race_interactions': raw-lap interaction table for calibration/sim
      - 'race_start_deltas': grid vs end-of-lap-1 deltas
      - (optional) 'quali_laps', 'quali_interactions'
      - 'qa': dict of QA counters for race (and quali if available)
    """
    enable_cache(config["cache_dir"])
    races = get_recent_races(config)
    out: List[Dict[str, Any]] = []

    for race in races:
        year, gp = race["year"], race["grand_prix"]

        # --- Race ---
        race_laps_raw, ses_r = load_session(year, gp, "R")
        if race_laps_raw is None or len(race_laps_raw) == 0:
            print(f"[WARN] No race laps for {year} {gp}")
            continue

        race_laps, qa_r = _derive_and_filter_tags(race_laps_raw, session_kind="R")
        race_inter = _build_interaction_table(race_laps_raw, ses_r, session_kind="R")
        start_deltas = _compute_start_deltas(race_laps_raw, ses_r)

        entry: Dict[str, Any] = {
            "year": year,
            "gp": gp,
            "race_laps": race_laps,
            "race_interactions": race_inter,
            "race_start_deltas": start_deltas,
            "qa": {"race": qa_r},
        }

        # --- Quali (optional) ---
        if config.get("include_qualifying", True):
            quali_laps_raw, ses_q = load_session(year, gp, "Q")
            if quali_laps_raw is not None and len(quali_laps_raw) > 0:
                quali_laps, qa_q = _derive_and_filter_tags(quali_laps_raw, session_kind="Q")
                quali_inter = _build_interaction_table(quali_laps_raw, ses_q, session_kind="Q")
                entry["quali_laps"] = quali_laps
                entry["quali_interactions"] = quali_inter
                entry["qa"]["quali"] = qa_q
            else:
                entry["quali_laps"] = None
                entry["quali_interactions"] = None

        # Optional: persist interaction tables for calibration if diagnostics logging is enabled
        try:
            log_cfg = (config.get("logging") or {})
            if bool(log_cfg.get("write_run_log", False)):
                diag_dir = (_project_root() / str(log_cfg.get("diagnostics_dir", "outputs/diagnostics"))).resolve()
                (diag_dir / "interactions").mkdir(parents=True, exist_ok=True)
                slug = f"{year}-{gp.lower().replace(' ', '-')}"
                race_inter.to_csv(diag_dir / "interactions" / f"{slug}-race_interactions.csv", index=False)
                if entry.get("quali_interactions") is not None:
                    entry["quali_interactions"].to_csv(diag_dir / "interactions" / f"{slug}-quali_interactions.csv", index=False)
                # Start deltas
                if not start_deltas.empty:
                    start_deltas.to_csv(diag_dir / "interactions" / f"{slug}-race_start_deltas.csv", index=False)
        except Exception:
            pass

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
        # Modeling inputs (unchanged)
        cols = list(first["race_laps"].columns)
        needed = ["LapTimeSeconds", "driver", "compound", "stint_id", "lap_on_tyre",
                  "lap_number", "track_status", "lap_ok"]
        print("[INFO] Tagged fields present:", {k: (k in cols) for k in needed})
        print("[INFO] Race laps kept (pace-only):", len(first["race_laps"]))
        if first.get("quali_laps") is not None:
            print("[INFO] Quali laps kept (pace-only):", len(first["quali_laps"]))

        # New artifacts
        inter_cols = list(first["race_interactions"].columns)
        print("[INFO] Race interaction table columns (sample):", inter_cols[:10], "…")
        if not first["race_start_deltas"].empty:
            print("[INFO] Start deltas head:\n", first["race_start_deltas"].head())

    outline = get_track_outline(cfg)
    if outline is not None:
        print(f"[INFO] Track outline points: {len(outline)}")
