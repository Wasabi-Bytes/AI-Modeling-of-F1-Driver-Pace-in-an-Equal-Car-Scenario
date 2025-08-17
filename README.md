# Equal-Car F1 Driver Pace Modeling

**What if every F1 driver raced in the same car?**
This project estimates their **true pace** by stripping away constructor advantage.

---

## Scope & Goals

* **Objective**: Measure each driver’s underlying pace, independent of car/team effects.
* **Data**: Last 10–12 Grands Prix, weighted by exponential recency decay (≈0.92) or by a **date half-life** (default ≈120 days).
* **Sessions**: Race (R) and Qualifying (Q).
* **Outputs**:

  * Equal-car driver ranking (with uncertainties)
  * Per-event breakdown tables
  * Montreal equal-start race animation
  * Monte Carlo season outcomes (champion odds, win distributions)

---

## Pipeline

### 1) Data Loading & Tagging (trustworthy inputs)

* Source: FastF1 telemetry & session data.
* Every downstream step only sees **pace laps** via a single boolean:

  * `lap_ok = True` iff:

    * Lap time exists and is positive,
    * Timing is accurate (`IsAccurate == True`),
    * **Not** an out-lap and **not** an in-lap (pit in/out flags),
    * Circuit green: FastF1 track status `"1"` (clear/green).
* Rows with `lap_ok == False` are dropped immediately after load.
* **Stint hygiene** (when `Stint` is missing):

  * Start a new stint right after a pit exit.
  * Recompute `lap_on_tyre` as **1-based** within each `Driver×Stint`.
  * Validate no backward jumps in tyre age; flag if detected.
* **Guaranteed columns** to consumers: Driver, Team, Event name, Compound, Stint, lap\_on\_tyre, LapNumber, LapTimeSeconds, TrackStatus, lap\_ok.
* **QA**: counts of dropped laps by reason; per event/team/driver totals of kept laps.

---

### 2) Race Metrics (robust & comparable)

Two modeling paths (configurable):

**a) Correction-Factor Model — `race_metrics_corrections_team`**

* **Event fixed effects** absorb cross-track difficulty.
* **Non-linear controls**: splines for `lap_number` (fuel burn) and `lap_on_tyre` (degradation); `Compound` as a factor; optional `Compound × tyre-age` interaction.
* **Outlier guard**: trim long-tail laps per driver (× stint) using an IQR rule.
* Normalize laps by removing fitted controls (keep intercept/event baseline), then **demean within team** → driver race deltas.
* **Uncertainty**: heteroskedasticity-robust SEs (HC3).
* Output per driver: delta (s), SE (s), laps used.

**b) OLS Team Model — `race_metrics_ols_team`**

* Design: event FEs + team FEs + driver\@team FEs + splines for tyre-age & lap-number + compound factor.
* **Cluster-robust SEs** (clusters = `driver×event`) to handle stint autocorrelation.
* Deltas constructed from **normalized laps** (team-demeaned), not raw coefficients.
* Same outlier trimming and return shape as (a).

---

### 3) Qualifying Metrics (evolution-aware)

* **Per-segment normalization**: within each of Q1/Q2/Q3, subtract the **segment median** so early vs late runners are fair.
* Keep **all valid laps** and optionally **winsorize** upper-tail outliers (configurable), or alternatively keep **top-k** laps **after** normalization.
* For each driver×segment: take the **best normalized lap**.
* **Teammate gap** per segment = driver best − team best (normalized).
* **Precision combine** across segments: weight each segment by inverse variance of the driver’s normalized laps in that segment.
* Output per driver: delta (s), SE (s), number of segments contributing.

---

### 4) Event-Level Combination (Race ⊕ Quali)

**Precision-weighted** per event:

$$
\Delta_{\text{event}}
= \frac{\Delta_R/\sigma_R^2 + \Delta_Q/\sigma_Q^2}{1/\sigma_R^2 + 1/\sigma_Q^2},
\qquad
\sigma_{\text{event}}
= \sqrt{\frac{1}{1/\sigma_R^2 + 1/\sigma_Q^2}}.
$$

* If one side is missing, the formula naturally falls back to the other.
* We also report effective weights: $w_{R,\text{eff}}$ and $w_{Q,\text{eff}}$.

**Empirical-Bayes shrinkage (optional):**

* Prior mean: **field mean** (default), with options for **team mean** or **zero**.
* Optional **team-level random effect** to stabilize volatile lineups.
* Prior variance $\tau^2$ via **method-of-moments** (configurable minimum or fixed).
* Outputs include shrink weights and $\tau^2$ per layer (race, quali, event).

---

### 5) Cross-Event Aggregation (smarter weighting & decay)

Base event weight:

$$
w
= \underbrace{\frac{1}{\mathrm{SE}^2}}_{\text{precision}}
\times
\underbrace{d(\text{recency})}_{\text{event-index decay or date half-life}}
\times
\underbrace{\text{effective sample size}}_{\text{laps contributing}}.
$$

* **Recency** options:

  * **Event-index decay** (≈0.92 per event), or
  * **Date half-life** (≈120 days by default).
* Aggregated delta = weighted mean; aggregated SE = $\sqrt{1/\sum w}$.
* Outputs: `driver_ranking.csv`, `event_breakdown.csv`.

---

### 6) Equal-Car Simulation (config-driven)

* Lap model: base circuit pace + **driver delta** + noise.
* **Reliability**: specify a **per-race DNF rate** (e.g., 10% over \~70 laps), internally converted to a per-lap probability (≈0.15%/lap for 10%).
* **Degradation**: piecewise/spline **compound-specific** tyre-age loss calibrated from laps.
* **Overtaking**: logistic probability driven by pace gap, DRS, defence, dirty-air penalty; **track-specific overrides** via config.
* **Event-specific deltas** option: use a given event’s $\Delta_{\text{event}}$ instead of global aggregated delta for circuit-style sensitivity.
* Output: interactive `simulation.html` replay (e.g., Montreal).

---

### 7) Monte Carlo Season Simulator

* Simulates full seasons (e.g., 24 races × 5000 seasons).
* Driver pace per race: $\mathcal{N}(\text{agg\_delta}, \text{agg\_se})$ + noise.
* Includes reliability and overtaking dynamics as above.
* Assigns FIA points and produces driver/constructor distributions.

---

## Diagnostics & Tests

* **Filters & stints**: in/out laps and non-green laps excluded; `lap_on_tyre` is 1-based and monotone.
* **Combiner**: returns the available side when the other is missing; reports effective weights.
* **Recency**: date half-life test halves the weight at the configured half-life.
* **Reliability**: per-race → per-lap conversion checks (≈0.15%/lap for 10%/race over \~70 laps).
* **Residuals**: QQ/influence plots to monitor heavy tails post-trim.
* All current minimal tests pass.

---

## Deliverables

**Tables**

* `driver_ranking.csv`
* `event_breakdown.csv`
* `championship_mc_drivers.csv`
* `championship_mc_constructors.csv`

**Visuals**

* `simulation.html` (equal-car race replay)
* Residual diagnostics & plots

**Narrative**

* This README (assumptions, limitations, interpretation)

---

## Limitations & Extensions

**Simplifications (still present)**

* No explicit pit-strategy optimization
* Weather not modeled; quali evolution is handled, but race evolution/safety cars are simplified
* DNFs independent of team/driver
* Penalties/sprints excluded

**Planned / Ongoing**

* Track-type effects and downforce profiles
* Richer compound-specific degradation calibration
* Hierarchical Bayesian shrinkage across seasons
* Strategy & safety-car modeling
* Driver “personality” parameters (aggression, defence, risk)

---

## References

* **Data**: FastF1
* **Models**: statsmodels, scikit-learn
* **Simulation**: logistic overtaking, Monte Carlo outcomes
