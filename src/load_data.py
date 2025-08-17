# src/load_data.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

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


# ====================== NEW: Weather & summaries helpers ======================
def _ensure_weather_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ("ambient_temp_c", "track_temp_c"):
        if c not in d.columns:
            d[c] = np.nan
    return d


def _summarize_event_weather(df: pd.DataFrame) -> Dict[str, Any]:
    d = df
    out: Dict[str, Any] = {}
    for c in ("ambient_temp_c", "track_temp_c"):
        if c in d.columns:
            vals = pd.to_numeric(d[c], errors="coerce").to_numpy()
            mask = ~np.isnan(vals)
            out[f"pct_nonan_{c}"] = float(mask.mean()) if vals.size else 0.0
            out[f"median_{c}"] = float(np.nanmedian(vals[mask])) if mask.any() else float("nan")
        else:
            out[f"median_{c}"] = float("nan")
            out[f"pct_nonan_{c}"] = 0.0
    return out

# ============================================================================


# -------- Fallback recent events (kept for backwards-compat) --------
def get_recent_races(_: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Legacy path used only if no season config is provided.
    return [
        {"year": 2025, "grand_prix": "Hungarian Grand Prix", "session": "R"},
        {"year": 2025, "grand_prix": "Belgian Grand Prix",   "session": "R"},
        {"year": 2025, "grand_prix": "British Grand Prix",   "session": "R"},
        {"year": 2025, "grand_prix": "Austrian Grand Prix",  "session": "R"},
        {"year": 2025, "grand_prix": "Canadian Grand Prix",  "session": "R"},
    ]


# -------- Season enumeration (NEW) --------
def enumerate_season_rounds(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return a list of dicts: [{'year': YEAR, 'round': int, 'event_name': str}, ...]
    Reads season.year and season.include_sprint from config if present.
    Defaults: year=2025, include_sprint=True.
    """
    season_cfg = (config.get("season") or {})
    year = int(season_cfg.get("year", 2025))
    include_sprint = bool(season_cfg.get("include_sprint", True))

    # Get the year's event schedule (no testing)
    sched = fastf1.get_event_schedule(year, include_testing=False)

    # Which weekend formats to include
    if include_sprint:
        mask = sched["EventFormat"].isin(["conventional", "sprint"])
    else:
        mask = sched["EventFormat"].isin(["conventional"])

    # Keep rows with a round number; sort by round
    keep = sched.loc[mask & sched["RoundNumber"].notna()].copy()
    keep["RoundNumber"] = keep["RoundNumber"].astype(int)
    keep = keep.sort_values("RoundNumber")

    events = []
    for _, row in keep.iterrows():
        events.append({
            "year": year,
            "round": int(row["RoundNumber"]),
            "event_name": str(row.get("EventName", "")) or str(row.get("Event", "")),
        })

    logging.info(f"[load_data] Season {year}: enumerated {len(events)} rounds "
                 f"({'conventional+sprint' if include_sprint else 'conventional only'}).")
    return events


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
    """
    d = _standardize_lap_seconds(laps).reset_index(drop=True)

    # Canonical driver/team/compound/event fields
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

    ev_col = "Event" if "Event" in d.columns else ("EventName" if "EventName" in d.columns else None)
    if ev_col:
        d["Event"] = d[ev_col].astype(str)

    # Lap number
    if "LapNumber" not in d.columns:
        d = d.sort_values(["driver", "LapTimeSeconds"]).copy()
        d["LapNumber"] = d.groupby("driver").cumcount() + 1
    d["lap_number"] = pd.to_numeric(d["LapNumber"], errors="coerce").fillna(0).astype(int)

    # Track status & timing accuracy
    d["track_status"] = d.get("TrackStatus", "").astype(str)
    is_green = _is_green(d["track_status"])
    is_accurate = d.get("IsAccurate", True)
    if isinstance(is_accurate, (pd.Series,)):
        is_accurate = is_accurate.fillna(True).astype(bool)

    # Pit in/out
    for col in ("PitInTime", "PitOutTime"):
        if col not in d.columns:
            d[col] = pd.NaT
    is_outlap = d["PitOutTime"].notna()
    is_inlap = d["PitInTime"].notna()

    # Valid time
    has_time = d["LapTimeSeconds"].notna() & (d["LapTimeSeconds"] > 0)

    # Strict pace-lap flag
    d["lap_ok"] = has_time & is_accurate & (~is_outlap) & (~is_inlap) & is_green

    # Stint inference
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

    try:
        kept_by = d.groupby(["Team", "driver"], dropna=False).size().sort_values(ascending=False).head(10)
        logging.info(f"[load_data] {session_kind}: top-kept Team×Driver (first 10):\n{kept_by}")
    except Exception:
        pass

    needed = [
        "LapTimeSeconds", "driver", "Team", "compound",
        "stint_id", "lap_on_tyre", "lap_number", "track_status", "lap_ok"
    ]
    for c in needed:
        if c not in d.columns:
            d[c] = np.nan if c != "lap_ok" else True

    return d, qa_counts


def _tag_stints_no_filter(laps: pd.DataFrame) -> pd.DataFrame:
    d = _standardize_lap_seconds(laps).reset_index(drop=True)

    if "Driver" in d.columns:
        d["driver"] = d["Driver"].astype(str)
    elif "DriverNumber" in d.columns:
        d["driver"] = d["DriverNumber"].astype(str)
    else:
        d["driver"] = d.get("DriverNumber", d.get("Driver", "UNK")).astype(str)
    d["Team"] = d.get("Team", "UNK").astype(str)
    d["compound"] = d.get("Compound", "UNKNOWN").fillna("UNKNOWN").astype(str)

    if "LapNumber" not in d.columns:
        d = d.sort_values(["driver", "LapTimeSeconds"]).copy()
        d["LapNumber"] = d.groupby("driver").cumcount() + 1
    d["lap_number"] = pd.to_numeric(d["LapNumber"], errors="coerce").fillna(0).astype(int)

    for col in ("PitInTime", "PitOutTime"):
        if col not in d.columns:
            d[col] = pd.NaT

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

    d = d.sort_values(["driver", "stint_id", "LapNumber"]).copy()
    d["lap_on_tyre"] = d.groupby(["driver", "stint_id"]).cumcount() + 1

    return d


def _build_interaction_table(
    laps_raw: pd.DataFrame,
    ses: Any,
    *,
    session_kind: str,
    merge_tolerance_seconds: int = 120,   # <-- NEW: config-driven tolerance
) -> pd.DataFrame:
    if laps_raw is None or len(laps_raw) == 0:
        return pd.DataFrame()

    d = _tag_stints_no_filter(laps_raw)

    d["track_status"] = d.get("TrackStatus", "").astype(str)
    is_green = _is_green(d["track_status"])
    is_accurate = d.get("IsAccurate", True)
    if isinstance(is_accurate, (pd.Series,)):
        is_accurate = is_accurate.fillna(True).astype(bool)
    is_outlap = d["PitOutTime"].notna()
    is_inlap = d["PitInTime"].notna()
    has_time = d["LapTimeSeconds"].notna() & (d["LapTimeSeconds"] > 0)
    d["lap_ok"] = has_time & is_accurate & (~is_outlap) & (~is_inlap) & is_green

    if "Position" in d.columns:
        d["position"] = pd.to_numeric(d["Position"], errors="coerce")
        d["pos_change"] = d.groupby("driver")["position"].shift(1) - d["position"]
    else:
        d["position"] = np.nan
        d["pos_change"] = np.nan

    s = d["track_status"].astype(str).fillna("")
    d["flag_yellow"] = s.str.contains("2") | s.str.contains("3")
    d["flag_sc"] = s.str.contains("4")
    d["flag_vsc"] = s.str.contains("5")

    d["ambient_temp_c"] = np.nan
    d["track_temp_c"] = np.nan
    try:
        if hasattr(ses, "weather_data") and ses.weather_data is not None and len(ses.weather_data) > 0:
            wx = ses.weather_data.copy()
            low = {c.lower(): c for c in wx.columns}
            air = low.get("airtemp")
            track = low.get("tracktemp")
            time_col = low.get("time", "Time")
            if air and track and time_col in wx.columns:
                wx = wx[[time_col, air, track]].rename(columns={air: "ambient_temp_c", track: "track_temp_c"})
                if "LapStartTime" in d.columns:
                    ref = d["LapStartTime"]
                elif "Time" in d.columns:
                    ref = d["Time"]
                else:
                    ref = None
                if ref is not None:
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
                        left_on="__ref_time__", right_on="__wx_time__",
                        direction="nearest",
                        tolerance=pd.Timedelta(f"{merge_tolerance_seconds}s")  # <-- NEW: config tolerance
                    )
                    d["ambient_temp_c"] = merged["ambient_temp_c"].values
                    d["track_temp_c"] = merged["track_temp_c"].values
    except Exception:
        pass

    # Guarantee presence even if merge failed (NEW)
    d = _ensure_weather_cols(d)

    d["gap_ahead_s"] = np.nan
    d["gap_behind_s"] = np.nan
    d["drs_active"] = pd.NA
    d["drs_available"] = pd.NA

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
    if laps_raw is None or len(laps_raw) == 0 or ses is None:
        return pd.DataFrame(columns=["driver", "team", "grid_pos", "pos_end_lap1", "start_delta"])

    d = _tag_stints_no_filter(laps_raw)
    if "Position" not in d.columns:
        return pd.DataFrame(columns=["driver", "team", "grid_pos", "pos_end_lap1", "start_delta"])

    lap1 = d[d["lap_number"] == 1].copy()
    lap1["pos_end_lap1"] = pd.to_numeric(lap1["Position"], errors="coerce")

    grid = None
    try:
        res = ses.results
        if res is not None and len(res) > 0:
            low = {c.lower(): c for c in res.columns}
            dn = low.get("drivernumber", None)
            abbr = low.get("abbreviation", None)
            gridcol = low.get("gridposition", None)
            teamcol = low.get("teamname", low.get("team", None))
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
        if "team" not in out.columns:
            out["team"] = lap1["Team"].astype(str)

    out["grid_pos"] = pd.to_numeric(out["grid_pos"], errors="coerce")
    out["start_delta"] = out["grid_pos"] - out["pos_end_lap1"]
    return out[["driver", "team", "grid_pos", "pos_end_lap1", "start_delta"]].dropna(subset=["pos_end_lap1"]).reset_index(drop=True)


# -------- Session Loading --------
def load_session(year: int, gp_or_round: Union[str, int], session: str) -> Tuple[Optional[pd.DataFrame], Optional[Any]]:
    """
    Load laps and session object.
    gp_or_round can be a Grand Prix name (str) or a round number (int).
    NOTE: weather=True so we can merge air/track temps into the interaction table if available.
    """
    try:
        ses = fastf1.get_session(year, gp_or_round, session)
        ses.load(laps=True, telemetry=False, weather=True)
        laps = ses.laps.reset_index(drop=True)
        return laps, ses
    except Exception as e:
        print(f"[WARN] Failed to load {year} {gp_or_round} {session}: {e}")
        return None, None


def load_all_data(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of event dicts with:
      - 'race_laps': filtered pace laps (lap_ok == True)
      - 'race_interactions': raw-lap interaction table
      - 'race_start_deltas': grid vs end-of-lap-1 deltas
      - (optional) 'quali_laps', 'quali_interactions'
      - 'qa': dict of QA counters
    """
    enable_cache(config["cache_dir"])

    # NEW: read weather knobs (one-liner to make them live)
    weather_cfg = (config.get("weather") or {})
    merge_tol_sec = int(weather_cfg.get("merge_tolerance_seconds", 120))

    # Prefer full-season enumeration when config.season is present; otherwise fallback to legacy list
    season_cfg = config.get("season")
    if season_cfg is not None:
        events = enumerate_season_rounds(config)
        # For logs: show first few
        preview = ", ".join([f"R{e['round']}" for e in events[:6]])
        logging.info(f"[load_data] Will attempt Q+R for {len(events)} rounds: {preview}{' …' if len(events) > 6 else ''}")
        gp_iter = [{"year": e["year"], "key": e["round"], "label": e["event_name"] or f"Round {e['round']}"} for e in events]
    else:
        legacy = get_recent_races(config)
        logging.info(f"[load_data] No season config found; using legacy recent races list ({len(legacy)} items).")
        gp_iter = [{"year": r["year"], "key": r["grand_prix"], "label": r["grand_prix"]} for r in legacy]

    out: List[Dict[str, Any]] = []
    include_quali = bool(config.get("include_qualifying", True))

    attempted_sessions = 0
    loaded_events = 0

    # NEW: loaded vs skipped labels
    loaded_labels: List[str] = []
    skipped_labels: List[str] = []

    for ev in gp_iter:
        year, key, label = ev["year"], ev["key"], ev["label"]

        # --- Race ---
        attempted_sessions += 1
        race_laps_raw, ses_r = load_session(year, key, "R")
        if race_laps_raw is None or len(race_laps_raw) == 0:
            print(f"[WARN] No race laps for {year} {label}")
            skipped_labels.append(f"{year} {label}")  # NEW: record skip
            continue

        race_laps, qa_r = _derive_and_filter_tags(race_laps_raw, session_kind="R")
        race_inter = _build_interaction_table(race_laps_raw, ses_r, session_kind="R",
                                              merge_tolerance_seconds=merge_tol_sec)  # NEW: pass tolerance
        start_deltas = _compute_start_deltas(race_laps_raw, ses_r)

        entry: Dict[str, Any] = {
            "year": year,
            "gp": label,
            "race_laps": race_laps,
            "race_interactions": race_inter,
            "race_start_deltas": start_deltas,
            "qa": {"race": qa_r},
        }

        # --- Quali ---
        if include_quali:
            attempted_sessions += 1
            quali_laps_raw, ses_q = load_session(year, key, "Q")
            if quali_laps_raw is not None and len(quali_laps_raw) > 0:
                quali_laps, qa_q = _derive_and_filter_tags(quali_laps_raw, session_kind="Q")
                quali_inter = _build_interaction_table(quali_laps_raw, ses_q, session_kind="Q",
                                                       merge_tolerance_seconds=merge_tol_sec)  # NEW: pass tolerance
                entry["quali_laps"] = quali_laps
                entry["quali_interactions"] = quali_inter
                entry["qa"]["quali"] = qa_q
            else:
                entry["quali_laps"] = None
                entry["quali_interactions"] = None

        # NEW: Weather medians per event (for viz and sim knobs)
        entry["weather_summary"] = _summarize_event_weather(race_inter)

        # Optional diagnostics persistence
        try:
            log_cfg = (config.get("logging") or {})
            if bool(log_cfg.get("write_run_log", False)):
                diag_dir = (_project_root() / str(log_cfg.get("diagnostics_dir", "outputs/diagnostics"))).resolve()
                (diag_dir / "interactions").mkdir(parents=True, exist_ok=True)
                slug_base = str(label).lower().replace(" ", "-")
                slug = f"{year}-{slug_base}"
                race_inter.to_csv(diag_dir / "interactions" / f"{slug}-race_interactions.csv", index=False)
                if entry.get("quali_interactions") is not None:
                    entry["quali_interactions"].to_csv(diag_dir / "interactions" / f"{slug}-quali_interactions.csv", index=False)
                if not start_deltas.empty:
                    start_deltas.to_csv(diag_dir / "interactions" / f"{slug}-race_start_deltas.csv", index=False)
        except Exception:
            pass

        out.append(entry)
        loaded_events += 1
        loaded_labels.append(f"{year} {label}")  # NEW: record loaded

    # NEW: human-friendly summary
    if loaded_labels:
        logging.info("[summary] Loaded GPs (%d): %s", len(loaded_labels), "; ".join(loaded_labels))
    if skipped_labels:
        logging.info("[summary] Skipped (no data yet) (%d): %s", len(skipped_labels), "; ".join(skipped_labels))

    logging.info(f"[load_data] Attempted sessions: {attempted_sessions} "
                 f"(~2 × rounds if Q included). Loaded events: {loaded_events}")

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

        inter_cols = list(first["race_interactions"].columns)
        print("[INFO] Race interaction table columns (sample):", inter_cols[:10], "…")
        if not first["race_start_deltas"].empty:
            print("[INFO] Start deltas head:\n", first["race_start_deltas"].head())

        # NEW: show weather summary presence
        print("[INFO] Weather summary for first event:", first.get("weather_summary", {}))

    outline = get_track_outline(cfg)
    if outline is not None:
        print(f"[INFO] Track outline points: {len(outline)}")
