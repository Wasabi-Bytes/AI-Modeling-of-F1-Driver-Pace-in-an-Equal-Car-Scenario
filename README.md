# Equal-Car F1 Driver Pace Modeling

**What if every F1 driver raced the same car?**
This project estimates each driver’s **underlying pace** (independent of constructor) and replays an **equal-car race** with realistic tyre wear, overtaking, reliability, weather, and track effects.

---

## Highlights

* **Driver pace estimates** from Race ⊕ Quali with uncertainties & recency weighting.
* **Equal-car replay**: DRS/dirty air, SC/VSC, DNFs, deterministic grid seeding.
* **Weather-aware**: event medians merged; **track temperature** softly scales tyre degradation.
* **Personality (optional)**: aggression ↑attempts, defence ↑holds, risk ↑DNF hazard.
* **Track meta**: track type, downforce index, DRS zone count, speed bias, overtaking difficulty.
* **Reproducible & logged**: per-event/per-run RNG streams and JSON run logs.

---

## Data & Sources

* **FastF1** telemetry/session data (laps, segments, flags).
* **Track metadata**: `data/track_meta.csv` (track\_type, downforce\_index, drs\_zones, speed\_bias, overtaking\_difficulty).
* **Personality scores**: `outputs/calibration/personality.csv` with `driver, aggression, defence, risk ∈ [0,1]`.

---

## Modeling Pipeline

### 1) Data Loading & Tagging

Valid pace laps only (`lap_ok = True`): accurate timing, green track, not in/out-laps.
Stint hygiene reconstructs `lap_on_tyre`. Event-level **weather\_summary** (medians) is attached.

### 2) Race Metrics (configurable path)

* **Corrections model** (`race_metrics_corrections_team`) *or*
* **OLS team model** (`race_metrics_ols_team`)
  Both include event fixed effects, tyre/fuel splines, compound factor, robust SEs (cluster/HC3).
  **Optional weather term:** small spline on `track_temp_c` (+ optional compound interaction).

### 3) Qualifying Metrics (evolution-aware)

Normalize per Q1/Q2/Q3; precision-combine best normalized lap per segment.

### 4) Event-Level Combination (Race ⊕ Quali)

Precision-weighted mean:

$$
\Delta_{\text{event}}
= \frac{\Delta_R/\sigma_R^2 + \Delta_Q/\sigma_Q^2}{1/\sigma_R^2 + 1/\sigma_Q^2},\qquad
\sigma_{\text{event}} = \sqrt{\frac{1}{1/\sigma_R^2 + 1/\sigma_Q^2}} .
$$

### 5) Cross-Event Aggregation (global/archetype/forecast)

Weights combine inverse-variance, recency, and sample size:

$$
w_i \propto \frac{1}{\sigma_i^2}\;\cdot\;\underbrace{\lambda_i}_{\text{recency (half-life or decay)}}\;\cdot\;
\underbrace{n_{i,\text{eff}}}_{\text{effective sample size}},\qquad \sum_i w_i = 1 .
$$

Outputs: **global** aggregates, **archetype** (e.g., street vs permanent), and an optional **forecast blend**:

$$
\Delta_{\text{forecast}} = \alpha\,\Delta_{\text{archetype}} + (1-\alpha)\,\Delta_{\text{global}} .
$$

---

## Equal-Car Simulation (math & mechanics)

*(Implemented in `src/visualize_equal_race.py`)*

### Base Pace per Driver

Per-lap pace (seconds) is:

$$
\text{LapTime}_{i\ell} = \underbrace{B_{\text{track}}}_{\text{base}} + 
\underbrace{\Delta_i}_{\text{driver delta}} + 
\underbrace{D_{\ell}^{(c_i)}}_{\text{degradation}} +
\varepsilon_{i\ell},
$$

with small i.i.d. noise $\varepsilon_{i\ell}\sim \mathcal{N}(0,\sigma^2)$.
Grid is seeded by $\Delta_i$ (faster → ahead) with a small staggered gap.

### Tyre Degradation (piecewise + temperature)

For compound $c$, piecewise-linear wear:

$$
D_{\ell}^{(c)} \;=\; m_T(c)\cdot s_c \cdot
\big[e_c\,\min(\ell, \ell_{\text{sw},c})\;+\;l_c\,\max(0,\,\ell-\ell_{\text{sw},c})\big],
$$

where $e_c$ and $l_c$ are early/late slopes (s/lap), $\ell_{\text{sw},c}$ is the switch lap, $s_c$ is any compound scale.
**Temperature multiplier** (small, bounded):

$$
m_T(c) \;=\; \mathrm{clip}\!\left(1 + s_c^{(T)}\,k\,(T_{\text{track}}-T_0),\, 1+\text{lo},\, 1+\text{hi}\right).
$$

### Overtaking Probability (DRS, personality, dirty air)

At a DRS zone, follower $f$ vs leader $l$:

$$
p(\text{pass}) \;=\; \sigma\!\Big(
\underbrace{\alpha_{\text{eff}}\,a_f}_{\text{attacker}}\cdot \Delta_{\text{norm}}
\;+\; \beta\cdot\mathbf{1}_{\text{DRS}}
\;-\; \underbrace{\gamma\,d_l}_{\text{defender}}
\;-\; \delta_{\text{dirty}}
\Big),
$$

* $\Delta_{\text{norm}} = \frac{\text{LapTime}_l - \text{LapTime}_f}{B_{\text{track}}}$ (positive if follower is faster),
* $a_f = 1 + \kappa_A(\text{aggression}_f-0.5)$, $d_l = 1 + \kappa_D(\text{defence}_l-0.5)$,
* $\alpha_{\text{eff}}$ scales with DRS zone *length* and *count*; $\delta_{\text{dirty}}$ increases with overtaking difficulty & downforce.

### Reliability (DNF hazard with risk)

User sets a **per-race DNF rate** $p_{\text{race}}$ over a typical length $L_{\text{typ}}$.
Per-lap hazard:

$$
p_{\text{lap}} \;=\; 1 - (1 - p_{\text{race}})^{1/L_{\text{typ}}}.
$$

With **risk personality** (if enabled):

$$
p_{\text{race},i} \;=\; \mathrm{clip}\!\big(p_{\text{base}}\,[1 + \omega_R(\text{risk}_i-0.5)],\,0,\,0.9\big),\quad
p_{\text{lap},i} = 1 - (1 - p_{\text{race},i})^{1/L_{\text{typ}}}.
$$

### Starts (small, personality-aware jitter)

Start gain (sec) for driver $i$:

$$
g_i \sim \mathcal{N}\!\big(0,\;\sigma_{\text{start}}\cdot s_i^2\big) + b_{\text{rank}},\quad
s_i = (1 + \eta_A(\text{aggression}_i-0.5))(1 - \eta_D(\text{defence}_i-0.5)).
$$

### Finish-Order Entropy (diversity diagnostic)

Normalized inversion ratio between grid and finish orders (0 = no changes, 1 = full reversal).

---

## Configuration (key knobs)

* **Weather**: merge tolerance; fields; temperature multiplier $(T_0,k,\text{clip})$.
* **Race model**: spline dfs for tyre age/lap number/**track temp**; optional temp×compound interaction; robust SE type.
* **Aggregation**: date half-life or recency decay; effective sample size.
* **Degradation**: linear vs calibrated curves; compound mix/scale; track-type multipliers.
* **Overtaking/DRS**: $\alpha,\beta,\gamma,\delta_{\text{dirty}}$; DRS detection threshold; track overrides; zone inference from geometry.
* **Reliability**: per-race DNF rate and typical laps (implied per-lap hazard).
* **Personality**: enable; weights $(\kappa_A,\kappa_D,\omega_R)$; CSV path for scores.
* **Visualization**: base lap, laps, timestep, seed/run index, weather overlay.

---

## Outputs

* **Tables**:
  `outputs/aggregate/driver_ranking.csv`, per-event breakdowns, season MC summaries.
* **Calibration & Meta**:
  `outputs/calibration/degradation_params.json`, `outputs/calibration/personality.csv`, `data/track_meta.csv`.
* **Replay & Logs**:
  `outputs/viz/simulation.html`, `outputs/viz/simulation_run_log_run{idx}.json`.
* **Diagnostics** (optional): plots under `outputs/diagnostics/`.

---

## Limitations & Roadmap

* No explicit pit-strategy optimization (single-stint abstraction in replay).
* Weather effects simplified to **track-temp scaling** (wind/humidity shown but not modeled yet).
* Penalties/sprints currently excluded.

**Planned**: improved diagnostics dashboard

---

## Credits

* **Data**: FastF1
* **Modeling**: statsmodels, scikit-learn
* **Visualization**: Plotly / Matplotlib
