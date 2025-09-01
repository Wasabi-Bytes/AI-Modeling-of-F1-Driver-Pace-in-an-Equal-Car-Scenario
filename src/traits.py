# src/traits.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from load_data import load_config, load_all_data  # uses your deterministic cleaner + interaction tables

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")


# ---------------------------------------------------------------------
# Paths & small helpers
# ---------------------------------------------------------------------
def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _norm_key(year: Any, gp: Any) -> str:
    y = "" if pd.isna(year) else str(int(year))
    g = "" if gp is None else str(gp)
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in f"{y} {g}")
    return " ".join(cleaned.split())


def _load_track_meta(cfg: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Reads paths.track_meta if present and returns a normalized meta table with:
      event_key_norm, track_type, drs_zones, speed_bias (as proxy for straight len)
    """
    p = (_project_root() / str((cfg.get("paths") or {}).get("track_meta", "data/track_meta.csv"))).resolve()
    if not p.exists():
        return None
    try:
        m = pd.read_csv(p)
    except Exception:
        return None

    cols = {c.lower(): c for c in m.columns}
    # Build normalized key (prefer year+gp, otherwise event_key)
    if "event_key" in cols:
        m["event_key_norm"] = (
            m[cols["event_key"]]
            .astype(str)
            .str.lower()
            .str.replace(r"[^0-9a-z\s]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    elif {"year", "gp"} <= set(cols.keys()):
        m["event_key_norm"] = m.apply(lambda r: _norm_key(r[cols["year"]], r[cols["gp"]]), axis=1)
    else:
        return None

    # Standardize useful columns if present
    out = pd.DataFrame({"event_key_norm": m["event_key_norm"]})
    for want in ["track_type", "downforce_index", "drs_zones", "speed_bias", "overtaking_difficulty"]:
        c = cols.get(want)
        if c is None:
            out[want] = np.nan
        else:
            if want in {"downforce_index", "drs_zones", "speed_bias", "overtaking_difficulty"}:
                out[want] = pd.to_numeric(m[c], errors="coerce")
            else:
                out[want] = m[c].astype(str).str.lower()
    return out


def _event_drs_detect_thresh(cfg: Dict[str, Any], gp_name: str) -> float:
    """Get detection threshold (seconds) with per-track override."""
    ov = ((cfg.get("overtaking") or {}).get("track_overrides") or {})
    key = gp_name.lower()
    # try substring matching
    for k, v in ov.items():
        if k in key and isinstance(v, dict) and "drs_detect_thresh_s" in v:
            return float(v["drs_detect_thresh_s"])
    return float((cfg.get("overtaking") or {}).get("drs_detect_thresh_s", 1.0))


def _drs_min_enable_lap(cfg: Dict[str, Any]) -> int:
    return int((cfg.get("overtaking") or {}).get("drs_min_enable_lap", 3))


def _drs_cooldown_after_sc(cfg: Dict[str, Any]) -> int:
    return int((cfg.get("overtaking") or {}).get("drs_cooldown_after_sc_laps", 2))


# ---------------------------------------------------------------------
# Dataset construction: pass attempts
# ---------------------------------------------------------------------
def _pair_attacker_defender(inter: pd.DataFrame) -> pd.DataFrame:
    """
    For each lap, pair a driver with the car directly ahead on the same lap.
    Requires columns: driver, position, lap_number.
    """
    d = inter.copy()
    d = d.dropna(subset=["driver", "position", "lap_number"])
    d["position"] = pd.to_numeric(d["position"], errors="coerce")

    att = d[["driver", "team", "lap_number", "position", "LapTimeSeconds", "pos_change",
             "flag_sc", "flag_vsc", "track_status", "compound", "drs_active", "drs_available",
             "gap_ahead_s"]].copy()
    att = att.rename(columns={
        "driver": "attacker",
        "team": "attacker_team",
        "LapTimeSeconds": "attacker_lap_s"
    })
    att["need_def_pos"] = att["position"] - 1

    defn = d[["driver", "team", "lap_number", "position", "LapTimeSeconds"]].copy()
    defn = defn.rename(columns={
        "driver": "defender",
        "team": "defender_team",
        "LapTimeSeconds": "defender_lap_s",
        "position": "def_pos"
    })

    m = att.merge(
        defn,
        left_on=["lap_number", "need_def_pos"],
        right_on=["lap_number", "def_pos"],
        how="left"
    )

    # Keep only valid pairs (must have a defender and not P1)
    m = m[m["defender"].notna()].copy()
    m = m[m["position"] > 1].copy()

    return m


def _estimate_drs_available(df: pd.DataFrame, drs_min_lap: int, sc_cooldown: int, drs_zones: Optional[float]) -> pd.Series:
    """
    Conservative proxy for DRS availability on a given lap:
      - Track has DRS zones (if known)
      - Lap >= min enable lap
      - Not SC/VSC and not within cooldown laps after an SC lap (per driver)
    """
    if df.empty:
        return pd.Series(dtype=bool)

    drs_track_ok = True if pd.isna(drs_zones) else (drs_zones > 0)

    # Per-driver since-last-SC counter
    group = df.sort_values(["attacker", "lap_number"]).groupby("attacker", dropna=False)
    sc_flag = df["flag_sc"].fillna(False).astype(bool)
    # mark laps that are within cooldown after any SC lap for that driver
    cooldown = []
    for _, sub in group:
        since_sc = np.inf
        mask = []
        for _, row in sub.iterrows():
            if bool(row["flag_sc"]):
                since_sc = 0
            else:
                since_sc = since_sc + 1 if np.isfinite(since_sc) else np.inf
            mask.append(since_sc <= sc_cooldown)
        cooldown.extend(mask)
    cooldown = pd.Series(cooldown, index=group.obj.index)

    drs_est = (
        (df["lap_number"] >= drs_min_lap)
        & (~df["flag_sc"].fillna(False))
        & (~df["flag_vsc"].fillna(False))
        & (~cooldown.reindex(df.index).fillna(False))
    )
    return drs_est & drs_track_ok


def build_pass_attempts(events: List[Dict[str, Any]], cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Construct a pass-attempt dataset by pairing each driver with the car ahead
    on each lap and labeling success if the attacker gained >=1 position on that lap.

    Output columns (key ones):
      event_key, year, gp, lap_number, attacker, defender,
      pass_completed (0/1), pace_gap, drs_available, track_type, straight_len (proxy),
      drs_zones_bucket
    """
    rows = []
    meta = _load_track_meta(cfg)

    for ev in events:
        inter = ev.get("race_interactions")
        if inter is None or len(inter) == 0:
            continue

        df = _pair_attacker_defender(inter)
        if df.empty:
            continue

        # Best-effort dry-only (skip clear wet compounds)
        comp = df.get("compound")
        if comp is not None:
            names = comp.astype(str).str.upper()
            df = df[~names.isin({"INTERMEDIATE", "WET", "FULL WET", "EXTREME WET"})].copy()

        # DRS context
        drs_min = _drs_min_enable_lap(cfg)
        sc_cd = _drs_cooldown_after_sc(cfg)

        # Attach track meta (DRS zones, speed bias, track type) via normalized key
        year = ev.get("year")
        gp = ev.get("gp")
        ekey = _norm_key(year, gp)
        drs_z = np.nan
        speed_bias = np.nan
        track_type = np.nan
        if meta is not None:
            row = meta.loc[meta["event_key_norm"] == ekey]
            if not row.empty:
                drs_z = float(row.iloc[0]["drs_zones"]) if pd.notna(row.iloc[0]["drs_zones"]) else np.nan
                speed_bias = float(row.iloc[0]["speed_bias"]) if pd.notna(row.iloc[0]["speed_bias"]) else np.nan
                track_type = str(row.iloc[0]["track_type"]) if pd.notna(row.iloc[0]["track_type"]) else np.nan

        # Use provided drs_available if present, otherwise estimate
        if "drs_available" in df.columns and df["drs_available"].notna().any():
            drs_avail = df["drs_available"].astype(bool)
        else:
            drs_avail = _estimate_drs_available(df, drs_min, sc_cd, drs_z)

        # In-range detector: prefer gap_ahead_s if available; else rely on DRS proxy
        thresh = _event_drs_detect_thresh(cfg, str(gp))
        if "gap_ahead_s" in df.columns and df["gap_ahead_s"].notna().any():
            in_range = df["gap_ahead_s"] <= float(thresh)
        else:
            in_range = drs_avail

        # Exposures only when lap_ok if present and not under VSC/SC
        if "lap_ok" in df.columns:
            in_range = in_range & df["lap_ok"].astype(bool)

        # Pace gap: prefer previous-lap times
        for col in ["attacker_prev_lap_s", "defender_prev_lap_s"]:
            df[col] = np.nan
        df = df.sort_values(["attacker", "lap_number"])
        df["attacker_prev_lap_s"] = df.groupby("attacker")["attacker_lap_s"].shift(1)
        df = df.sort_values(["defender", "lap_number"])
        df["defender_prev_lap_s"] = df.groupby("defender")["defender_lap_s"].shift(1)

        # If previous missing, use same-lap
        a_t = df["attacker_prev_lap_s"].fillna(df["attacker_lap_s"])
        d_t = df["defender_prev_lap_s"].fillna(df["defender_lap_s"])
        pace_gap = d_t - a_t  # + => attacker faster

        # Outcome: pass completed on this lap (attacker moved up)
        y = (pd.to_numeric(df["pos_change"], errors="coerce").fillna(0.0) > 0.0).astype(int)

        # Straight length proxy from meta (speed_bias); also bucketize DRS zones
        if pd.isna(speed_bias):
            straight_len = np.nan
        else:
            straight_len = float(speed_bias)

        if pd.isna(drs_z):
            drs_bucket = "unknown"
        elif drs_z <= 0:
            drs_bucket = "drs_0"
        elif drs_z == 1:
            drs_bucket = "drs_1"
        else:
            drs_bucket = "drs_2plus"

        sub = pd.DataFrame({
            "event_key": [f"{year} {gp}"] * len(df),
            "year": [year] * len(df),
            "gp": [gp] * len(df),
            "lap_number": df["lap_number"].values,
            "attacker": df["attacker"].astype(str).values,
            "defender": df["defender"].astype(str).values,
            "attacker_team": df["attacker_team"].astype(str).values,
            "defender_team": df["defender_team"].astype(str).values,
            "pass_completed": y.values,
            "in_range": in_range.astype(bool).values,
            "drs_available": drs_avail.astype(bool).values,
            "pace_gap": pd.to_numeric(pace_gap, errors="coerce").values,
            "track_type": [track_type] * len(df),
            "straight_len": [straight_len] * len(df),
            "drs_bucket": [drs_bucket] * len(df),
        })
        # keep only in-range exposures
        sub = sub[sub["in_range"]].copy()
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=[
            "event_key", "year", "gp", "lap_number", "attacker", "defender",
            "attacker_team", "defender_team", "pass_completed", "in_range",
            "drs_available", "pace_gap", "track_type", "straight_len", "drs_bucket"
        ])
    out = pd.concat(rows, ignore_index=True)

    # Final cleanups
    out["pace_gap"] = pd.to_numeric(out["pace_gap"], errors="coerce")
    out["drs_available"] = out["drs_available"].astype(bool)
    out["pass_completed"] = out["pass_completed"].astype(int)
    out["track_type"] = out["track_type"].fillna("unknown").astype(str)
    out["drs_bucket"] = out["drs_bucket"].fillna("unknown").astype(str)
    return out


# ---------------------------------------------------------------------
# Trait estimation (aggression/defence) via penalized logistic
# ---------------------------------------------------------------------
def _vectorize_pass_rows(pass_df: pd.DataFrame) -> Tuple[DictVectorizer, np.ndarray, np.ndarray]:
    """
    Build a feature dictionary per row and vectorize with DictVectorizer.
    Features:
      - numeric: pace_gap, straight_len
      - binary: drs_available
      - categorical: track_type, drs_bucket
      - attacker driver FE: attacker=<code>
      - defender driver FE: defender=<code>
    """
    feats: List[Dict[str, Any]] = []
    y = pass_df["pass_completed"].astype(int).to_numpy()

    for _, r in pass_df.iterrows():
        d: Dict[str, Any] = {
            "pace_gap": float(r["pace_gap"]) if pd.notna(r["pace_gap"]) else 0.0,
            "straight_len": float(r["straight_len"]) if pd.notna(r["straight_len"]) else 0.0,
            "drs_available": bool(r["drs_available"]),
            f"track_type={r['track_type']}": 1.0,
            f"drs_bucket={r['drs_bucket']}": 1.0,
            f"attacker={r['attacker']}": 1.0,
            f"defender={r['defender']}": 1.0,
        }
        feats.append(d)

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feats)
    return vec, X, y


def _extract_trait_tables(vec: DictVectorizer, clf: LogisticRegression, pass_df: pd.DataFrame) -> pd.DataFrame:
    """
    Read learned coefficients back into driver-level aggression/defence scores.
    aggression ~= coefficient on attacker=<driver>
    defence   ~= - coefficient on defender=<driver>   (higher = better defending)
    """
    names = vec.get_feature_names_out()
    coefs = clf.coef_.ravel()
    w = dict(zip(names, coefs))

    # Collect present drivers
    attackers = {f"attacker={d}": d for d in pass_df["attacker"].astype(str).unique()}
    defenders = {f"defender={d}": d for d in pass_df["defender"].astype(str).unique()}

    agg_rows = []
    for k, driver in attackers.items():
        agg = float(w.get(k, 0.0))
        agg_rows.append((driver, agg))
    def_rows = []
    for k, driver in defenders.items():
        # invert sign: positive = harder to pass
        dsc = -float(w.get(k, 0.0))
        def_rows.append((driver, dsc))

    df_agg = pd.DataFrame(agg_rows, columns=["driver", "aggression"])
    df_def = pd.DataFrame(def_rows, columns=["driver", "defence"])

    # Center to mean zero for interpretability
    if not df_agg.empty:
        df_agg["aggression"] = df_agg["aggression"] - float(df_agg["aggression"].mean())
    if not df_def.empty:
        df_def["defence"] = df_def["defence"] - float(df_def["defence"].mean())

    # Volume stats
    expo = pass_df.groupby("attacker")["pass_completed"].agg(n_attempts="size", n_success="sum").reset_index()
    expo = expo.rename(columns={"attacker": "driver"})
    expo["success_rate"] = expo["n_success"] / expo["n_attempts"].clip(lower=1)

    traits = df_agg.merge(df_def, on="driver", how="outer").merge(expo, on="driver", how="left")
    return traits.sort_values("aggression", ascending=False, ignore_index=True)


def estimate_driver_traits(cfg: Dict[str, Any], events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    High-level runner:
      - loads events (if not provided)
      - builds pass attempts
      - fits penalized logistic with attacker/defender effects (L2 -> partial pooling)
      - writes pass_dataset.csv, driver_traits.csv, and a pickled overtake model
    """
    if events is None:
        events = load_all_data(cfg)

    pass_df = build_pass_attempts(events, cfg)
    outdir = _project_root() / "outputs" / "traits"
    _ensure_dir(outdir)

    # Save dataset for auditing
    pass_df.to_csv(outdir / "pass_dataset.csv", index=False)

    if len(pass_df) < 50 or pass_df["pass_completed"].sum() == 0:
        # Not enough data — emit empty traits
        empty = pd.DataFrame(columns=["driver", "aggression", "defence", "n_attempts", "n_success", "success_rate"])
        empty.to_csv(outdir / "driver_traits.csv", index=False)
        return {"pass_df": pass_df, "traits": empty, "model": None}

    # Build features
    vec, X, y = _vectorize_pass_rows(pass_df)

    # Hyperparams
    traits_cfg = (cfg.get("personality") or {})
    C = float(traits_cfg.get("l2_C", 1.0))   # L2 strength (inverse)
    max_iter = int(traits_cfg.get("max_iter", 2000))
    seed = int((cfg.get("rng") or {}).get("seed", 2025))

    # Penalized logistic w/ saga (supports sparse + L2)
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="saga",
        max_iter=max_iter,
        random_state=seed,
        n_jobs=None if X.shape[0] < 100000 else -1,
        verbose=0
    )
    clf.fit(X, y)

    # Extract driver traits
    traits = _extract_trait_tables(vec, clf, pass_df)
    traits.to_csv(outdir / "driver_traits.csv", index=False)

    # Persist a compact model bundle for inference
    bundle = {
        "vectorizer": vec,
        "coef_": clf.coef_.ravel(),
        "intercept_": float(clf.intercept_.ravel()[0]),
        "feature_names": vec.get_feature_names_out().tolist(),
        "C": C,
        "max_iter": max_iter,
        "random_state": seed,
        "meta": {"n_rows": int(len(pass_df)), "n_pos": int(y.sum())}
    }
    with open(outdir / "overtake_model.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # Small JSON summary for quick checks
    with open(outdir / "overtake_model_summary.json", "w") as f:
        json.dump(bundle["meta"], f, indent=2)

    return {"pass_df": pass_df, "traits": traits, "model": bundle}


def main():
    cfg = load_config("config/config.yaml")
    res = estimate_driver_traits(cfg)
    n = len(res["pass_df"])
    pos = int(res["pass_df"]["pass_completed"].sum()) if n else 0
    print(f"[INFO] Built pass dataset: n={n}, passes={pos}")
    print(f"[INFO] Wrote traits to: {(_project_root() / 'outputs' / 'traits' / 'driver_traits.csv').resolve()}")


if __name__ == "__main__":
    main()
