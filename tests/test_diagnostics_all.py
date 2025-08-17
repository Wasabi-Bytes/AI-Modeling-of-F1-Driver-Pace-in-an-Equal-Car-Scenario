# tests/test_diagnostics_all.py
import os
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for any matplotlib plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------- Test setup helpers ---------------------

PROJ = Path(__file__).resolve().parents[1]
SRC = PROJ / "src"
OUT_DIAG = PROJ / "outputs" / "diagnostics"
OUT_DIAG.mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import visualize_equal_race as viz  # noqa: E402


def _circle_path(n=1800, r=1.0):
    """Simple closed loop to avoid depending on telemetry in tests."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.stack([x, y], axis=1)


def _base_cfg(track_type="permanent", use_personality=False):
    return {
        "paths": {},  # can be filled per-test
        "viz_track": {"year": 2024, "grand_prix": "Unit Test GP", "track_type": track_type},
        "overtaking": {
            "alpha_pace": 6.0,
            "beta_drs": 1.5,
            "gamma_defend": 1.2,
            "dirty_air_penalty": 0.10,  # somewhat generous to allow passing in tests
            "drs_detect_thresh_s": 1.0,
            "detection_offset_fraction": 0.03,
            "drs_min_enable_lap": 1,
            "drs_cooldown_after_sc_laps": 0,
            "track_overrides": {},
        },
        "degradation": {
            "source": "linear",  # some tests switch to 'calibrated'
            "default_compound": "M",
            "compounds": {"M": {"early_slope": 0.02, "late_slope": 0.012, "switch_lap": 10}},
            "track_type_multipliers": {"permanent": 1.0, "street": 1.2},
        },
        "reliability": {"mode": "per_race", "per_race_dnf": 0.02, "typical_race_laps": 60},
        "personality": {"use": bool(use_personality), "agg_weight": 1.0, "def_weight": 1.0, "risk_weight": 1.0},
        "visualize_equal_race": {"seed": 123, "run_idx": 0},
    }


def _ranking(drivers, deltas=None):
    if deltas is None:
        deltas = np.zeros(len(drivers), dtype=float)
    return pd.DataFrame({"driver": drivers, "agg_delta_s": deltas})


def _run_sim(
    ranking,
    cfg,
    meta=None,
    base_lap=90.0,
    n_laps=12,
    dt=0.5,
    noise_sd=0.05,
    seed=123,
    run_idx=0,
    force_no_incidents=True,
    path=None,
):
    path = _circle_path(2500) if path is None else path

    # Optionally disable SC/VSC incidents for determinism in some tests
    if force_no_incidents:
        old = viz.P_INCIDENT_PER_LAP
        viz.P_INCIDENT_PER_LAP = 0.0

    try:
        return viz.simulate_progress(
            ranking=ranking,
            xy_path=path,
            base_lap=base_lap,
            n_laps=n_laps,
            dt=dt,
            noise_sd=noise_sd,
            seed=seed,
            cfg=cfg,
            meta=meta or {},
            run_idx=run_idx,
        )
    finally:
        if force_no_incidents:
            viz.P_INCIDENT_PER_LAP = old


# --------------------- Personality tests ---------------------

def test_personality_aggression_increases_pass_success(monkeypatch):
    """
    Monotonicity check aligned with implementation: higher aggression -> higher pass success,
    holding similar gaps/opportunities.
    """
    drivers = ["D1", "D2", "D3", "D4", "D5", "D6"]
    rank = _ranking(drivers)

    cfg = _base_cfg(use_personality=True)
    # Make DRS eligibility plentiful and passing easier for a stable signal
    cfg["overtaking"]["drs_detect_thresh_s"] = 2.5
    meta = {"drs_zones": 3, "overtaking_difficulty": 0.1}

    def with_aggression(val):
        def _fake(_cfg, ds):
            out = {}
            for d in ds:
                out[d] = {"aggression": (val if d == "D1" else 0.5), "defence": 0.5, "risk": 0.5}
            return out
        return _fake

    monkeypatch.setattr(viz, "_load_personality_scores", with_aggression(0.9))
    passes_hi = []
    for r in range(8):
        *_, stats = _run_sim(rank, cfg, meta=meta, run_idx=r, noise_sd=0.03, n_laps=10, base_lap=80.0)
        passes_hi.append(stats["passes"])

    monkeypatch.setattr(viz, "_load_personality_scores", with_aggression(0.1))
    passes_lo = []
    for r in range(8):
        *_, stats = _run_sim(rank, cfg, meta=meta, run_idx=r, noise_sd=0.03, n_laps=10, base_lap=80.0)
        passes_lo.append(stats["passes"])

    assert np.mean(passes_hi) >= np.mean(passes_lo), (
        f"High aggression did not increase pass success: hi={np.mean(passes_hi):.2f}, lo={np.mean(passes_lo):.2f}"
    )


def test_personality_defence_reduces_passes(monkeypatch):
    """
    Defence symmetry: higher leader defence should reduce realized passes.
    """
    drivers = ["LDR", "ATT", "B1", "B2", "B3", "B4"]
    # Give attacker slight pace advantage to create opportunities
    deltas = np.array([0.00, -0.10, 0.00, 0.00, 0.00, 0.00])
    rank = _ranking(drivers, deltas)

    meta = {"drs_zones": 2, "overtaking_difficulty": 0.3}

    def personalities(def_lead):
        return {
            "LDR": {"aggression": 0.5, "defence": def_lead, "risk": 0.5},
            "ATT": {"aggression": 0.95, "defence": 0.3, "risk": 0.5},
            "B1": {"aggression": 0.5, "defence": 0.5, "risk": 0.5},
            "B2": {"aggression": 0.5, "defence": 0.5, "risk": 0.5},
            "B3": {"aggression": 0.5, "defence": 0.5, "risk": 0.5},
            "B4": {"aggression": 0.5, "defence": 0.5, "risk": 0.5},
        }

    cfg = _base_cfg(use_personality=True)

    # Low defence leader
    monkeypatch.setattr(viz, "_load_personality_scores", lambda _c, ds: personalities(0.2))
    _, _, _, _, _, _, _, _, _, _, _, _, _, stats_lo = _run_sim(
        rank, cfg, meta=meta, run_idx=1, noise_sd=0.02, n_laps=12, base_lap=80.0
    )

    # High defence leader
    monkeypatch.setattr(viz, "_load_personality_scores", lambda _c, ds: personalities(0.9))
    _, _, _, _, _, _, _, _, _, _, _, _, _, stats_hi = _run_sim(
        rank, cfg, meta=meta, run_idx=2, noise_sd=0.02, n_laps=12, base_lap=80.0
    )

    # Should see fewer passes when leader defence is high (probabilistic, keep lenient)
    assert stats_hi["passes"] <= stats_lo["passes"], (
        f"High defence produced >= passes ({stats_hi['passes']}) vs low defence ({stats_lo['passes']})."
    )


# --------------------- HB shrinkage tests (if data available) ---------------------

def test_hier_shrinkage_if_available():
    """
    Posterior means (event_delta_s_shrunk) lie between raw deltas and field/team means;
    pooling increases with higher SE. If files/columns are not present, skip.
    """
    mdir = PROJ / "outputs" / "metrics"
    files = sorted(mdir.glob("*-event_metrics.csv"))
    if not files:
        pytest.skip("No metrics files available for shrinkage tests.")

    df = pd.read_csv(files[-1])
    low = {c.lower(): c for c in df.columns}

    # Require these columns to run meaningful checks
    need = ["driver", "event_delta_s", "event_delta_s_shrunk"]
    if not all(k in low for k in need):
        pytest.skip("Missing shrinkage columns; skipping.")

    drv = low["driver"]
    raw = pd.to_numeric(df[low["event_delta_s"]], errors="coerce")
    shr = pd.to_numeric(df[low["event_delta_s_shrunk"]], errors="coerce")

    # Optional SE column (or use agg_se_s if available)
    se_col = low.get("event_delta_se") or low.get("agg_se_s")
    se = pd.to_numeric(df[se_col], errors="coerce") if se_col else pd.Series(np.nan, index=df.index)

    # Field mean boundary check
    field_mean = float(np.nanmean(raw))
    # Check that shrinked values are between raw and field_mean for most rows (allow some tolerance)
    between = ((shr - raw) * (field_mean - raw)) >= -1e-9
    assert between.mean() > 0.8, "Too many shrinked values outside [raw, field_mean] interval."

    # Pooling increases with higher SE: compare average shrink distance high-SE vs low-SE halves
    if se.notna().sum() > 6:
        med = se.median()
        hi = (se >= med)
        lo = (se < med)
        dist = (shr - raw).abs()
        assert dist[hi].mean() >= dist[lo].mean(), "Pooling did not increase with higher SE."


# --------------------- Degradation calibration tests ---------------------

def test_degradation_calibrated_beats_linear_on_synthetic(tmp_path):
    """
    Generate synthetic 'observed' degradation from calibrated curve + noise,
    then verify calibrated fit has lower residual than linear baseline.
    Also verify piecewise continuity around the switch.
    """
    # Synthetic calibrated params
    params = {
        "global": {"M": {"early_slope": 0.022, "late_slope": 0.015, "switch_lap": 8}},
        "by_track_type": {}
    }
    dp = tmp_path / "degradation_params.json"
    dp.write_text(json.dumps(params), encoding="utf-8")

    cfg_cal = _base_cfg()
    cfg_cal["degradation"]["source"] = "calibrated"
    cfg_cal["paths"]["degradation_params"] = str(dp)

    cfg_lin = _base_cfg()  # linear with default slopes

    drivers = ["X"]
    D = len(drivers)
    n_laps = 18
    rng = np.random.default_rng(0)

    # Calibrated curve (truth)
    deg_cal = viz.build_degradation_matrix(cfg_cal, n_laps, drivers, rng, meta={"track_type": "permanent"})
    # Linear curve
    rng2 = np.random.default_rng(0)
    deg_lin = viz.build_degradation_matrix(cfg_lin, n_laps, drivers, rng2, meta={"track_type": "permanent"})

    # Observed = calibrated + noise
    obs = deg_cal[:, 0] + rng.normal(0, 0.002, size=n_laps)

    sse_cal = float(np.sum((deg_cal[:, 0] - obs) ** 2))
    sse_lin = float(np.sum((deg_lin[:, 0] - obs) ** 2))
    assert sse_cal <= sse_lin, "Calibrated degradation did not reduce residual vs linear on synthetic data."

    # Continuity near switch lap
    sw = params["global"]["M"]["switch_lap"]
    pre = deg_cal[sw - 1, 0]
    post = deg_cal[sw, 0]
    assert abs(post - pre) < 0.05, "Piecewise curve appears discontinuous at switch."

    # Save quick diagnostic plot
    plt.figure()
    plt.plot(range(n_laps), obs, label="obs (syn)")
    plt.plot(range(n_laps), deg_cal[:, 0], label="calibrated")
    plt.plot(range(n_laps), deg_lin[:, 0], label="linear")
    plt.legend()
    plt.xlabel("Lap on tyre")
    plt.ylabel("Cumulative degradation [s]")
    outp = OUT_DIAG / "degradation_calib_vs_linear.png"
    plt.savefig(outp, dpi=120)
    plt.close()


# --------------------- Track effects tests ---------------------

def test_track_effects_passes_vary_with_meta(monkeypatch):
    """
    Lower 'overtaking_difficulty' (lower dirty air) should yield more passes on average.
    """
    drivers = [f"D{i}" for i in range(8)]
    # small spread to create pace gaps
    deltas = np.linspace(-0.15, 0.15, len(drivers))
    rank = _ranking(drivers, deltas)

    cfg = _base_cfg(use_personality=False)

    # Disable incidents for stability
    monkeypatch.setattr(viz, "P_INCIDENT_PER_LAP", 0.0)

    # Two metas: easy vs hard overtaking
    easy_meta = {"drs_zones": 3, "overtaking_difficulty": 0.1}
    hard_meta = {"drs_zones": 1, "overtaking_difficulty": 0.9}

    def avg_passes(meta):
        xs = []
        for r in range(8):
            _, _, _, _, _, _, _, _, _, _, _, _, _, stats = _run_sim(
                rank, cfg, meta=meta, run_idx=r, noise_sd=0.03, n_laps=10, base_lap=82.0
            )
            xs.append(stats["passes"])
        return float(np.mean(xs))

    p_easy = avg_passes(easy_meta)
    p_hard = avg_passes(hard_meta)

    # Easy should have >= passes than hard (allow equality rarely)
    assert p_easy >= p_hard, f"Expected more passes with easier meta; got easy={p_easy}, hard={p_hard}"

    # Save quick bar plot
    plt.figure()
    plt.bar(["easy", "hard"], [p_easy, p_hard])
    plt.ylabel("Mean passes (runs)")
    plt.title("Track meta effect on passes")
    outp = OUT_DIAG / "track_effects_passes.png"
    plt.savefig(outp, dpi=120)
    plt.close()


# --------------------- RNG & diversity tests ---------------------

def test_rng_same_seed_same_outcome(monkeypatch):
    """
    Same seed & run index ⇒ identical outcomes (attempts, passes, finish order).
    """
    drivers = [f"D{i}" for i in range(8)]
    rank = _ranking(drivers)
    cfg = _base_cfg()
    meta = {"drs_zones": 2, "overtaking_difficulty": 0.3}

    monkeypatch.setattr(viz, "P_INCIDENT_PER_LAP", 0.0)

    res1 = _run_sim(rank, cfg, meta=meta, run_idx=7, seed=111, n_laps=10, noise_sd=0.02)
    res2 = _run_sim(rank, cfg, meta=meta, run_idx=7, seed=111, n_laps=10, noise_sd=0.02)

    stats1 = res1[-1]
    stats2 = res2[-1]

    assert stats1["attempts"] == stats2["attempts"]
    assert stats1["passes"] == stats2["passes"]
    assert stats1["finish_order"] == stats2["finish_order"]


def test_rng_different_runidx_diverse_outcomes(monkeypatch):
    """
    Different run index ⇒ different outcomes (often). Use a mild threshold to avoid flakiness.
    """
    drivers = [f"D{i}" for i in range(10)]
    rank = _ranking(drivers)
    cfg = _base_cfg()
    meta = {"drs_zones": 2, "overtaking_difficulty": 0.25}

    monkeypatch.setattr(viz, "P_INCIDENT_PER_LAP", 0.0)

    orders = []
    passes = []
    for r in range(10):
        _, _, _, _, _, _, _, _, _, _, _, _, _, stats = _run_sim(
            rank, cfg, meta=meta, run_idx=r, seed=222, n_laps=10, noise_sd=0.03
        )
        orders.append(tuple(stats["finish_order"]))
        passes.append(stats["passes"])

    unique_orders = len(set(orders))
    # Guardrail: at least 60% unique orders over runs
    assert unique_orders >= math.ceil(0.6 * len(orders)), f"Unique orders too low: {unique_orders}/{len(orders)}"


def test_event_to_event_differences_consistent_with_meta(monkeypatch):
    """
    Event-to-event differences: more passes at low-dirty-air tracks.
    """
    drivers = [f"D{i}" for i in range(8)]
    deltas = np.linspace(-0.10, 0.10, len(drivers))
    rank = _ranking(drivers, deltas)
    cfg = _base_cfg()

    monkeypatch.setattr(viz, "P_INCIDENT_PER_LAP", 0.0)

    metas = [
        {"name": "low_dirty_air", "meta": {"drs_zones": 3, "overtaking_difficulty": 0.1}},
        {"name": "high_dirty_air", "meta": {"drs_zones": 1, "overtaking_difficulty": 0.9}},
    ]

    results = []
    for m in metas:
        ps = []
        for r in range(6):
            _, _, _, _, _, _, _, _, _, _, _, _, _, stats = _run_sim(
                rank, cfg, meta=m["meta"], run_idx=r, seed=333, n_laps=10, noise_sd=0.03
            )
            ps.append(stats["passes"])
        results.append((m["name"], float(np.mean(ps))))

    # Assert ordering
    mean_low = [v for (k, v) in results if k == "low_dirty_air"][0]
    mean_high = [v for (k, v) in results if k == "high_dirty_air"][0]
    assert mean_low >= mean_high, f"Expected more passes for low dirty air; got {mean_low} vs {mean_high}"

    # Save quick plot
    plt.figure()
    plt.bar([r[0] for r in results], [r[1] for r in results])
    plt.ylabel("Mean passes (runs)")
    plt.title("Event-to-event (meta) differences")
    outp = OUT_DIAG / "rng_event_meta_passes.png"
    plt.savefig(outp, dpi=120)
    plt.close()


# --------------------- Quick smoke test of full animate build ---------------------

def test_build_animation_smoke(monkeypatch):
    """
    Smoke test: ensure build_animation constructs a Figure without errors on tiny sim.
    """
    drivers = [f"D{i}" for i in range(6)]
    rank = _ranking(drivers)
    cfg = _base_cfg()
    meta = {"drs_zones": 2, "overtaking_difficulty": 0.3}

    monkeypatch.setattr(viz, "P_INCIDENT_PER_LAP", 0.0)
    (positions, lap_key, leader_lap, drivers_out,
     phase_flags, rc_texts, drs_on, drs_banner,
     orders, gaps_panel, zones, alpha_eff, det_eff, stats) = _run_sim(
        rank, cfg, meta=meta, n_laps=6, noise_sd=0.02
    )
    # Build figure
    fig = viz.build_animation(
        positions, lap_key, leader_lap, drivers_out,
        name_map={d: d for d in drivers_out},
        color_map={d: "#888888" for d in drivers_out},
        xy_path=_circle_path(2500),
        n_laps=6,
        phase_flags=phase_flags,
        rc_texts=rc_texts,
        drs_on=drs_on,
        drs_banner=drs_banner,
        orders=orders,
        gaps_panel=gaps_panel,
        zones=zones,
        alpha_eff=alpha_eff,
        det_eff=det_eff,
    )
    assert hasattr(fig, "to_html"), "Plotly figure not created properly."
