
# Equal-Car F1 Driver Pace Modeling

**What if every F1 driver raced in the same car?**
This project estimates their **true pace** by stripping away constructor advantage and then visualizes an **equal-car race** with realistic degradation, overtaking, reliability, and track effects.

---

## Scope & Goals

* **Objective:** Measure each driver’s underlying pace, independent of car/team effects, and simulate equal-car racing dynamics.
* **Data:** Last 10–12 Grands Prix, weighted by exponential recency decay (≈0.92) or **date half-life** (default ≈120 days).
* **Sessions:** Race (R) and Qualifying (Q).
* **Key Outputs:**

  * Equal-car driver ranking (+ uncertainties)
  * Per-event breakdown tables
  * Equal-car race replay (`outputs/viz/simulation.html`)
  * Monte-Carlo season outcomes (champion odds, win distributions)
  * Calibration artifacts (degradation curves, driver personality)
  * Diagnostics plots & consolidated tests

---

## Pipeline

### 1) Data Loading & Tagging (trustworthy inputs)

* Source: **FastF1** telemetry & session data.
* A single boolean selects usable pace laps: `lap_ok = True` iff:

  * Lap time exists and is positive
  * Timing accurate (`IsAccurate == True`)
  * Not an out-lap and not an in-lap (pit flags)
  * Track green (`TrackStatus == "1"`)
* Rows with `lap_ok == False` are dropped immediately after load.
* **Stint hygiene** (when `Stint` missing): start a stint after pit exit; recompute **1-based** `lap_on_tyre`; validate monotonicity.
* **Guaranteed columns** downstream: `Driver, Team, Event, Compound, Stint, lap_on_tyre, LapNumber, LapTimeSeconds, TrackStatus, lap_ok`.
* **QA:** counts of dropped laps by reason; kept laps per event/team/driver.
* **Track metadata:** `data/track_meta.csv` with
  `event_key, track_type (street/permanent/hybrid), downforce_index [0–1], drs_zones, speed_bias, overtaking_difficulty`.

---

### 2) Race Metrics (robust & comparable)

Two modeling paths (configurable):

**a) Correction-Factor Model — `race_metrics_corrections_team`**

* Event fixed effects absorb cross-track difficulty.
* Non-linear controls: splines for `lap_number` (fuel) & `lap_on_tyre` (degradation); `Compound` factor; optional interaction.
* Outlier guard: driver×stint IQR trimming.
* Normalize laps, then **team-demean** → driver race deltas.
* **Uncertainty:** HC3 robust SEs.

**b) OLS Team Model — `race_metrics_ols_team`**

* Event FEs + team FEs + driver\@team FEs + smooth tyre/fuel controls + compound factor.
* **Cluster-robust SEs** (driver×event clusters).
* Deltas constructed from normalized (team-demeaned) laps.

**Track effects (optional):**

* If `track_effects.use: true`: add **archetype** FEs (street/permanent) or a smooth control for **downforce\_index**.
* Event FEs remain to prevent cross-track leakage into driver deltas.

---

### 3) Qualifying Metrics (evolution-aware)

* Normalize **within Q1/Q2/Q3** by segment median.
* Keep all valid laps; optional winsorization or “top-k after normalization.”
* Best normalized lap per driver×segment; teammate gaps computed per segment.
* **Precision combine** segments (inverse variance).

---

### 4) Event-Level Combination (Race ⊕ Quali)

Precision-weighted mean:

$$
\Delta_{\text{event}}
= \frac{\Delta_R/\sigma_R^2 + \Delta_Q/\sigma_Q^2}{1/\sigma_R^2 + 1/\sigma_Q^2}, \quad
\sigma_{\text{event}} = \sqrt{\frac{1}{1/\sigma_R^2 + 1/\sigma_Q^2}}.
$$

* If one side missing, falls back naturally.
* Effective weights reported.
* **Shrinkage options:**

  * Empirical Bayes (default), or
  * **Hierarchical Bayesian** driver←team←field pooling when `use_hierarchical_shrinkage: true`.
    Outputs add `*_delta_s_shrunk`, pooling factors, posterior SD.

---

### 5) Cross-Event Aggregation (global + archetype + forecast)

* Weighted by **inverse variance × recency decay (or date half-life) × effective sample size**.
* Emit:

  * **Global** aggregates (existing)
  * **Archetype-specific** aggregates (e.g., street vs permanent; downforce buckets)
  * **Forecast blend** for upcoming event:
    $\Delta_{\text{forecast}} = \alpha \cdot \Delta_{\text{archetype}} + (1-\alpha)\cdot \Delta_{\text{global}}$ (configurable $\alpha$)
* A per-event **weights table** is produced for transparency.

---

## Equal-Car Simulation (config-driven realism & diversity)

*(Implemented in `src/visualize_equal_race.py`)*

* **Base pace:** circuit baseline + driver delta + noise.
* **Calibrated tyre degradation:**
  Reads `outputs/calibration/degradation_params.json` when `degradation.source: calibrated`, applying compound-specific piecewise curves; falls back to linear if disabled/missing.
* **Reliability:** specify **per-race DNF rate** (e.g., 10% over \~70 laps) and convert to **per-lap hazard** internally.
* **Overtaking:** logistic probability driven by **pace gap**, **DRS**, **defence**, **dirty-air penalty**, and **track meta**; **driver personality** modulates attempt/hold propensities.
* **Personality (optional):**
  `outputs/calibration/personality.csv` with normalized **aggression**, **defence**, **risk**:

  * **Aggression:** ↑attempt rate
  * **Defence:** ↑hold probability / fewer easy passes against
  * **Risk:** scales DNF hazard around the base per-race target
* **Event-specific deltas (optional):** use `event_delta_s_shrunk` instead of global aggregates.
* **DRS zones:** automatically inferred from track geometry; count/strength can be tuned via `track_meta`.
* **Deterministic but diverse RNG:**
  Per-event/per-run streams (starts, lap noise, overtakes, DNFs, incidents) via seed mixing; same seed **and** run index → identical replay, different run index → diverse outcomes.
* **Run logging:** `outputs/viz/simulation_run_log_run{idx}.json` with seed metadata, attempts, passes, DNFs, start gains, and **finish-order entropy**.
* **Output:** `outputs/viz/simulation.html` (interactive Plotly replay).

---

## Quickstart

1. **Install** dependencies (FastF1, pandas, numpy, statsmodels, plotly, etc.).
2. **Prepare data:** run your existing data ingestion; ensure `data/track_meta.csv` exists.
3. **Run equal-car replay:**

```bash
python -m src.visualize_equal_race
# Output: outputs/viz/simulation.html
# Log   : outputs/viz/simulation_run_log_run0.json
```

*Change `visualize_equal_race.seed` / `run_idx` in the config to reproduce or diversify runs.*

## Diagnostics & Tests

Consolidated into `tests/test_diagnostics_all.py`:

* **Personality:**
  Aggression increases pass success; defence reduces passes (symmetry checks).
* **Hierarchical shrinkage:**
  Posterior means lie between raw deltas and team/field means; pooling ↑ with higher SE.
* **Degradation calibration:**
  Calibrated compound curves reduce residuals vs linear; piecewise smoothness at knots.
* **Track effects:**
  Archetype aggregates differ from global in expected directions.
* **RNG & diversity:**
  Same seed+run index → identical outcomes; different run index → different outcomes; finish-order diversity guard; event-to-event differences align with `track_meta`.
* **Viz smoke test:** figure builds without errors.

All tests currently **pass** on recent weekend data + synthetic fixtures.

---

## Deliverables

**Tables**

* `outputs/aggregate/driver_ranking.csv`
* `outputs/aggregate/event_breakdown.csv`
* `outputs/aggregate/championship_mc_drivers.csv`
* `outputs/aggregate/championship_mc_constructors.csv`

**Calibration & Meta**

* `outputs/calibration/degradation_params.json`
* `outputs/calibration/personality.csv`
* `data/track_meta.csv`

**Visuals & Logs**

* `outputs/viz/simulation.html` (equal-car replay)
* `outputs/viz/simulation_run_log_run{idx}.json`
* `outputs/diagnostics/…` (plots, if enabled)

---

## Limitations & Extensions

**Simplifications**

* No explicit pit-strategy optimization
* Weather not modeled; race evolution (beyond SC/VSC) simplified
* Penalties/sprints excluded

**Planned / Ongoing**

* Additional diagnostics dashboards

---

## References

* **Data:** FastF1
* **Modeling:** statsmodels, scikit-learn
* **Simulation:** Logistic overtaking with DRS & dirty-air; calibrated tyre degradation; Monte-Carlo outcomes
