# src/estimate_personality.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import warnings
import numpy as np
import pandas as pd

from load_data import load_config, enable_cache, load_all_data

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")

# ---------------- Paths & small utils ----------------
PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "outputs" / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _cfg_get(cfg: dict, path: list[str], default=None):
    d = cfg
    for k in path:
        if not isinstance(d, dict) or (k not in d):
            return default
        d = d[k]
    return d


def _to_num_arr(x) -> np.ndarray:
    return np.asarray(pd.to_numeric(x, errors="coerce"))


def _beta_post_mean(k, n, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Vectorized posterior mean for Beta-Binomial with prior Beta(a,b)."""
    k = _to_num_arr(k)
    n = _to_num_arr(n)
    denom = n + a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        p = (k + a) / denom
    # keep within [0,1]
    return np.clip(p, 0.0, 1.0)


def _beta_post_se(k, n, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """Vectorized posterior SE ≈ sqrt( p(1-p)/(n+a+b+1) )."""
    k = _to_num_arr(k)
    n = _to_num_arr(n)
    p = _beta_post_mean(k, n, a, b)
    tot = n + a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (p * (1.0 - p)) / (tot + 1.0)
    var = np.where(np.isfinite(var), var, np.nan)
    return np.sqrt(np.maximum(var, 0.0))


def _minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - lo) / (hi - lo)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_series(df: pd.DataFrame, col: Optional[str], default_value=None, name: Optional[str] = None) -> pd.Series:
    if col is not None and col in df.columns:
        s = df[col]
        if name and s.name != name:
            s = s.rename(name)
        return s
    return pd.Series([default_value] * len(df), index=df.index, name=(name or (col or "col")))


# ---------------- Load interactions (best effort) ----------------
def _find_interaction_csv() -> Optional[Path]:
    candidates = [
        PROJ / "outputs" / "aux" / "interaction_table.csv",
        PROJ / "outputs" / "aux" / "interactions.csv",
        PROJ / "outputs" / "aux" / "interaction_events.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_interactions_or_coarse(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Preferred: interaction table (with gaps/DRS/pos_change).
    Fallback: reconstruct coarse table from race pace laps (position deltas only).
    """
    csv_path = _find_interaction_csv()
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        low = {c.lower(): c for c in df.columns}
        for cand in ["drs_available", "drs_active", "drs", "isdrs"]:
            if cand in low and "drs_available" not in df.columns:
                df = df.rename(columns={low[cand]: "drs_available"})
                break
        return df

    # -------- Coarse fallback using raw laps --------
    if "cache_dir" in cfg:
        enable_cache(cfg["cache_dir"])
    events = load_all_data(cfg)

    rows = []
    for ev in events:
        d = ev.get("race_laps")
        if d is None or d.empty:
            continue
        tmp = d.copy()
        if "lap_ok" in tmp.columns:
            tmp = tmp[tmp["lap_ok"].astype(bool)].copy()

        drv_col = _first_existing_col(tmp, ["driver", "Driver", "Abbreviation", "DriverNumber", "DriverId"])
        team_col = _first_existing_col(tmp, ["Team", "team", "Constructor", "TeamName", "ConstructorName"])
        lap_col = _first_existing_col(tmp, ["LapNumber", "lap_number"])
        pos_col = _first_existing_col(tmp, ["Position", "position"])

        drivers = _ensure_series(tmp, drv_col, name="driver").astype(str)
        teams = _ensure_series(tmp, team_col, default_value="UNK", name="team").astype(str)
        laps = pd.to_numeric(_ensure_series(tmp, lap_col, default_value=np.nan, name="LapNumber"), errors="coerce")
        pos = pd.to_numeric(_ensure_series(tmp, pos_col, default_value=np.nan, name="Position"), errors="coerce")

        tmp2 = pd.DataFrame({
            "driver": drivers,
            "team": teams,
            "LapNumber": laps,
            "Position": pos,
            "year": ev.get("year"),
            "gp": ev.get("gp"),
        }).dropna(subset=["driver", "LapNumber"])

        tmp2 = tmp2.sort_values(["driver", "LapNumber"]).copy()
        tmp2["pos_change"] = tmp2.groupby("driver")["Position"].diff().fillna(0.0)

        tmp2["drs_available"] = np.nan
        tmp2["gap_ahead_s"] = np.nan
        tmp2["gap_behind_s"] = np.nan
        rows.append(tmp2)

    if not rows:
        return pd.DataFrame(columns=["driver", "team", "LapNumber", "Position", "pos_change", "year", "gp"])
    return pd.concat(rows, ignore_index=True)


# ---------------- Personality estimation ----------------
def estimate_personality(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Outputs one row per driver with aggression/defence/risk in [0,1] (+ SEs & counts).
    """
    d = _load_interactions_or_coarse(cfg)
    if d.empty:
        raise RuntimeError("No interaction data available to estimate personality.")

    # Normalize common columns
    low = {c.lower(): c for c in d.columns}
    driver_col = low.get("driver") or low.get("drivernumber") or "driver"
    lap_col = low.get("lapnumber") or "LapNumber"
    pos_col = low.get("position") or "Position"
    team_col = low.get("team") or "team"

    keep = [c for c in [driver_col, team_col, lap_col, pos_col, "pos_change", "drs_available", "gap_ahead_s", "gap_behind_s", "year", "gp"] if c in d.columns]
    d = d[keep].copy()

    ren = {}
    if driver_col in d.columns: ren[driver_col] = "driver"
    if team_col in d.columns: ren[team_col] = "team"
    if lap_col in d.columns: ren[lap_col] = "LapNumber"
    if pos_col in d.columns: ren[pos_col] = "Position"
    d = d.rename(columns=ren)

    for req in ["driver", "LapNumber", "Position"]:
        if req not in d.columns:
            d[req] = np.nan
    d["driver"] = d["driver"].astype(str)
    d["LapNumber"] = pd.to_numeric(d["LapNumber"], errors="coerce")
    d["Position"] = pd.to_numeric(d["Position"], errors="coerce")

    if "pos_change" not in d.columns:
        d = d.sort_values(["driver", "LapNumber"])
        d["pos_change"] = d.groupby("driver")["Position"].diff().fillna(0.0)
    d["pos_change"] = pd.to_numeric(d["pos_change"], errors="coerce").fillna(0.0)

    if "drs_available" in d.columns:
        d["drs_available"] = d["drs_available"].astype(bool)

    # Opportunities & threats
    drs_mask = (d["LapNumber"] >= 3)
    if "drs_available" in d.columns and d["drs_available"].notna().any():
        drs_mask = drs_mask & d["drs_available"]

    opp_mask = (d["Position"] > 1) & drs_mask
    if "gap_ahead_s" in d.columns and d["gap_ahead_s"].notna().any():
        opp_mask = opp_mask | (pd.to_numeric(d["gap_ahead_s"], errors="coerce") <= 1.1)

    d["attack"] = (d["pos_change"] < 0).astype(int)

    if "gap_behind_s" in d.columns and d["gap_behind_s"].notna().any():
        threat_mask = (pd.to_numeric(d["gap_behind_s"], errors="coerce") <= 1.1) & (d["LapNumber"] >= 3)
    elif "drs_available" in d.columns and d["drs_available"].notna().any():
        threat_mask = d["drs_available"].astype(bool) & (d["LapNumber"] >= 3)
    else:
        threat_mask = (d["LapNumber"] >= 3)

    d["lost_pos"] = (d["pos_change"] > 0).astype(int)

    # Aggregates
    g = d.groupby("driver", dropna=False)
    n_opps = g.apply(lambda x: int(opp_mask.loc[x.index].sum())).rename("n_opps")
    n_attacks = g["attack"].sum().rename("n_attacks")
    n_threats = g.apply(lambda x: int(threat_mask.loc[x.index].sum())).rename("n_threats")
    n_defences = (n_threats - g["lost_pos"].sum()).clip(lower=0).rename("n_defences")
    exposure_laps = g.size().rename("exposure_laps")

    # DNF proxy
    if {"year", "gp"}.issubset(d.columns):
        end = d.groupby(["year", "gp"])["LapNumber"].max().rename("Lmax_event")
        dl = d.join(end, on=["year", "gp"])
        fin = dl.groupby(["year", "gp", "driver"])["LapNumber"].max().rename("Lmax_driver").reset_index()
        fin = fin.merge(end.reset_index(), on=["year", "gp"], how="left")
        fin["dnf_flag"] = (fin["Lmax_driver"] <= (fin["Lmax_event"] - 3)).astype(int)
        dnf_events = fin.groupby("driver")["dnf_flag"].sum().rename("dnf_events")
        events_seen = fin.groupby("driver").size().rename("events_seen")
    else:
        dnf_events = pd.Series(0, index=n_opps.index, name="dnf_events")
        events_seen = pd.Series(0, index=n_opps.index, name="events_seen")

    # Rates (smoothed) and normalization
    agg_rate = (n_attacks + 1) / (n_opps + 2)
    def_rate = (n_defences + 1) / (n_threats + 2)
    risk_rate = (dnf_events + 1) / (events_seen + 2)

    agg_score = _minmax01(pd.Series(agg_rate, index=n_opps.index))
    def_score = _minmax01(pd.Series(def_rate, index=n_threats.index))
    risk_score = _minmax01(pd.Series(risk_rate, index=events_seen.index))

    # Vectorized posterior SEs
    agg_se = pd.Series(_beta_post_se(n_attacks, n_opps), index=n_opps.index, name="aggression_se")
    def_se = pd.Series(_beta_post_se(n_defences, n_threats), index=n_opps.index, name="defence_se")
    risk_se = pd.Series(_beta_post_se(dnf_events, events_seen), index=n_opps.index, name="risk_se")

    out = pd.DataFrame({
        "driver": n_opps.index.astype(str),
        "aggression": agg_score.values,
        "aggression_se": agg_se.values,
        "defence": def_score.values,
        "defence_se": def_se.values,
        "risk": risk_score.values,
        "risk_se": risk_se.values,
        "n_attacks": n_attacks.values,
        "n_opps": n_opps.values,
        "n_defences": n_defences.values,
        "n_threats": n_threats.values,
        "n_incidents": np.zeros_like(n_opps.values, dtype=int),
        "exposure_laps": exposure_laps.reindex(n_opps.index).fillna(0).astype(int).values,
    })

    for col in ["aggression", "defence", "risk"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.5)
    for col in ["aggression_se", "defence_se", "risk_se"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.25)

    return out.sort_values("driver").reset_index(drop=True)


def main():
    cfg = load_config("config/config.yaml")
    if "cache_dir" in cfg:
        enable_cache(cfg["cache_dir"])
    df = estimate_personality(cfg)
    out_path = _cfg_get(cfg, ["paths", "personality"], "outputs/calibration/personality.csv")
    out_file = (PROJ / out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"[INFO] Wrote personality scores: {out_file}")


if __name__ == "__main__":
    main()
