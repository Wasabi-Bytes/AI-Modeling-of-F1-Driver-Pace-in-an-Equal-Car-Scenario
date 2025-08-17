# tests/test_minimal.py
from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, timedelta
import sys

import numpy as np
import pandas as pd
import pytest

# ---- Make project modules importable (src/ and repo root) ----
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Optional deps for diagnostics
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Project imports
from load_data import load_config, load_all_data
from clean_data import clean_event_payload
from model_metrics import combine_event_metrics
from aggregate_metrics import aggregate_driver_metrics

# ---------- Helpers ----------
DIAG_DIR = ROOT / "outputs" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def _unpack_clean_payload(result):
    if isinstance(result, dict):
        return (
            result.get("race_laps"),
            result.get("race_summary", {}),
            result.get("quali_laps"),
            result.get("quali_summary", {}),
        )
    if isinstance(result, tuple):
        if len(result) == 4:
            return result
        if len(result) == 3:
            dR, rS, dQ = result
            return dR, rS, dQ, {}
        if len(result) == 2:
            dR, rS = result
            return dR, rS, None, {}
        if len(result) == 1:
            return result[0], {}, None, {}
    raise ValueError("clean_event_payload returned an unsupported structure")


def _get_one_event():
    cfg = load_config("config/config.yaml")
    events = load_all_data(cfg)
    assert len(events) > 0, "No events loaded — run your data loader first."
    return events[-1], cfg


# ---------- 1) Filters & stint invariants ----------
def test_filters_and_stints():
    event, cfg = _get_one_event()
    race_df, _, _, _ = _unpack_clean_payload(clean_event_payload(event, cfg))
    assert isinstance(race_df, pd.DataFrame) and not race_df.empty, "Cleaner returned no race laps."

    # (a) lap_ok must be True if present (non-green/out laps excluded upstream)
    if "lap_ok" in race_df.columns:
        assert race_df["lap_ok"].dropna().astype(bool).all(), "Found laps with lap_ok == False in cleaned data."

    # (b) in/out laps excluded if flags exist
    pit_flags = [c for c in race_df.columns if str(c).lower() in
                 {"ispitinlap", "ispitoutlap", "pitin", "pitout", "pitinlap", "pitoutlap"}]
    for c in pit_flags:
        assert (~race_df[c].fillna(False).astype(bool)).all(), f"Pit in/out laps still present (column {c})."

    # (c) lap_on_tyre: 1-based *values* and monotone within each stint.
    # After filtering, the first observed lap_on_tyre can be >1, so we require min >= 1 (not == 1).
    must_cols = {"driver", "lap_on_tyre"}
    assert must_cols.issubset(race_df.columns), f"Missing columns for stint check: {must_cols - set(race_df.columns)}"

    if "stint_id" in race_df.columns:
        grp = race_df.sort_values(["driver", "stint_id", "lap_number"]).groupby(["driver", "stint_id"])
    else:
        assert "compound" in race_df.columns, "Need compound to infer stints when stint_id is missing."
        grp = race_df.sort_values(["driver", "compound", "lap_number"]).groupby(["driver", "compound"])

    for _, g in grp:
        lot = pd.to_numeric(g["lap_on_tyre"], errors="coerce").dropna().astype(int).values
        if len(lot) == 0:
            continue
        assert lot.min() >= 1, f"lap_on_tyre must be 1-based; found min={lot.min()}."
        # Allow gaps (if early laps were filtered), but forbid decreases
        assert np.all(np.diff(lot) >= 0), "lap_on_tyre must be non-decreasing within a stint."


# ---------- 2) Event combiner edge-cases ----------
def test_event_combiner_missing_and_both():
    # Missing quali → event == race
    race_df = pd.DataFrame({
        "driver": ["DRV"], "team": ["TEAM"],
        "race_delta_s": [0.20], "race_se_s": [0.05], "race_n": [100], "race_model": ["ols_team"]
    })
    m = combine_event_metrics(race_df, quali_df=None)  # pass None so function builds empty quali frame
    assert pytest.approx(0.20, abs=1e-9) == float(m.loc[0, "event_delta_s"])
    assert pytest.approx(0.05, abs=1e-9) == float(m.loc[0, "event_se_s"])

    # Missing race → event == quali
    quali_df = pd.DataFrame({
        "driver": ["DRV"], "team": ["TEAM"],
        "quali_delta_s": [-0.10], "quali_se_s": [0.08], "quali_k": [3]
    })
    m = combine_event_metrics(race_df=None, quali_df=quali_df)  # pass None for the missing side
    assert pytest.approx(-0.10, abs=1e-9) == float(m.loc[0, "event_delta_s"])
    assert pytest.approx(0.08, abs=1e-9) == float(m.loc[0, "event_se_s"])

    # Both present, same SE → arithmetic mean
    race_df = pd.DataFrame({"driver":["DRV"],"team":["TEAM"],"race_delta_s":[0.20],"race_se_s":[0.10],"race_n":[50],"race_model":["ols_team"]})
    quali_df = pd.DataFrame({"driver":["DRV"],"team":["TEAM"],"quali_delta_s":[0.00],"quali_se_s":[0.10],"quali_k":[2]})
    m = combine_event_metrics(race_df, quali_df)
    assert pytest.approx(0.10, abs=1e-9) == float(m.loc[0, "event_delta_s"])
    assert pytest.approx(math.sqrt(1.0 / (1/0.10**2 + 1/0.10**2)), rel=1e-6) == float(m.loc[0, "event_se_s"])


# ---------- 3) Half-life (date) weighting ----------
def test_date_half_life_weight_halves():
    cfg = {
        "weighting": {
            "recency_mode": "date_half_life",
            "half_life_days": 100,
            "race_sample_weight": 1.0,
            "quali_sample_weight": 1.0,
        }
    }
    today = datetime(2025, 1, 1)
    old = (today - timedelta(days=100)).date().isoformat()
    new = today.date().isoformat()

    events_df = pd.DataFrame({
        "driver": ["DRV","DRV"],
        "team":   ["TEAM","TEAM"],
        "year":   [2025, 2025],
        "gp":     ["Old GP", "New GP"],
        "event_idx": [0, 1],
        "event_date": [old, new],
        "event_delta_s": [0.1, 0.1],
        "event_se_s": [0.05, 0.05],
        "race_n": [500, 500],
        "quali_k": [3, 3],
    })

    ranking, breakdown = aggregate_driver_metrics(events_df, cfg)
    w_old = float(breakdown.loc[breakdown["gp"]=="Old GP","event_weight"].iloc[0])
    w_new = float(breakdown.loc[breakdown["gp"]=="New GP","event_weight"].iloc[0])

    assert w_new > 0 and w_old > 0, "Weights must be positive."
    ratio = w_old / w_new
    assert 0.47 <= ratio <= 0.53, f"Half-life check failed: expected ~0.50, got {ratio:.3f}"


# ---------- 4) Reliability per-race → per-lap ----------
def test_reliability_per_race_to_per_lap():
    p_race = 0.10
    L = 70
    p_lap = 1.0 - (1.0 - p_race)**(1.0 / L)
    assert 0.0012 <= p_lap <= 0.0018, f"Expected ~0.0015 per-lap, got {p_lap:.6f}"


# ---------- 5) Residual diagnostics (QQ + influence-ish) ----------
def test_residuals_diagnostics_and_plots(tmp_path: Path = DIAG_DIR):
    if not (HAVE_MPL and HAVE_SCIPY):
        pytest.skip("matplotlib/scipy not available; skipping diagnostic plots.")

    event, cfg = _get_one_event()
    race_df, _, _, _ = _unpack_clean_payload(clean_event_payload(event, cfg))
    assert not race_df.empty, "No race laps for diagnostics."

    use_cols = {"LapTimeSeconds", "driver", "compound"}
    missing = use_cols - set(race_df.columns)
    if missing:
        pytest.skip(f"Residual diagnostic skipped (missing columns: {missing})")

    d = race_df.dropna(subset=["LapTimeSeconds"]).copy()
    d["LapTimeSeconds"] = pd.to_numeric(d["LapTimeSeconds"], errors="coerce")
    d = d[np.isfinite(d["LapTimeSeconds"])]

    if d.empty or d["driver"].nunique() < 4:
        pytest.skip("Not enough laps or drivers for a meaningful QQ diagnostic.")

    d["group_mean"] = d.groupby(["driver", "compound"])["LapTimeSeconds"].transform("mean")
    d["resid"] = d["LapTimeSeconds"] - d["group_mean"]

    # QQ-plot
    z = d["resid"].to_numpy()
    z = z[np.isfinite(z)]
    if z.size < 50:
        pytest.skip("Too few residuals for QQ plot.")
    z_std = (z - np.mean(z)) / (np.std(z, ddof=1) + 1e-12)
    osm, osr = stats.probplot(z_std, dist="norm")
    qq_path = tmp_path / "residuals_qq.png"
    plt.figure(figsize=(5,5))
    plt.scatter(osm[0], osm[1], s=8, alpha=0.6)
    lo, hi = np.percentile(osm[0], [1, 99])
    plt.plot([lo, hi], [lo, hi], lw=2, color="k")
    plt.title("QQ plot of residuals (driver×compound centered)")
    plt.xlabel("Theoretical quantiles"); plt.ylabel("Sample quantiles")
    plt.tight_layout(); plt.savefig(qq_path); plt.close()
    assert qq_path.exists(), "QQ plot not written."

    # Spread by compound (stint if present)
    infl_path = tmp_path / "residuals_spread_by_compound.png"
    plt.figure(figsize=(6,4))
    if "stint_id" in d.columns:
        spread = d.groupby(["compound", "stint_id"])["resid"].std(ddof=1).dropna()
        x = np.arange(spread.shape[0])
        plt.bar(x, spread.values, width=0.8)
        plt.xticks(x, [f"{c}-{s}" for (c, s) in spread.index], rotation=90)
        plt.title("Std(residual) by compound×stint")
    else:
        spread = d.groupby(["compound"])["resid"].std(ddof=1).dropna()
        x = np.arange(spread.shape[0])
        plt.bar(x, spread.values, width=0.6)
        plt.xticks(x, list(spread.index))
        plt.title("Std(residual) by compound")
    plt.ylabel("seconds")
    plt.tight_layout(); plt.savefig(infl_path); plt.close()
    assert infl_path.exists(), "Residual spread plot not written."


# ---------- CLI entry ----------
if __name__ == "__main__":
    ok = True
    for fn in [
        test_filters_and_stints,
        test_event_combiner_missing_and_both,
        test_date_half_life_weight_halves,
        test_reliability_per_race_to_per_lap,
        test_residuals_diagnostics_and_plots,
    ]:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except pytest.skip.Exception as e:
            print(f"[SKIP] {fn.__name__} – {e}")
        except AssertionError as e:
            ok = False
            print(f"[FAIL] {fn.__name__} – {e}")
        except Exception as e:
            ok = False
            print(f"[ERROR] {fn.__name__} – {e}")
    if not ok:
        raise SystemExit(1)
