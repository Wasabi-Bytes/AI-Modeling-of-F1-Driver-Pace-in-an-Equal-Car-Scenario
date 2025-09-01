
# Equal-Car F1 Driver Pace Modeling

**What if every F1 driver raced the same car?**
This project estimates each driver’s **underlying pace** (independent of constructor) and replays an **equal-car race** in two modes:

* **Deterministic (pure pace)** — no randomness, no SC, no DNFs.
* **Stochastic (race-like)** — includes variability, DNFs, and safety cars.

By filtering out weather, SC/VSC, and random jitter, we isolate **driver ability**. Aggression, defence, and risk still shape racing outcomes.

---

## Highlights

* **Driver pace deltas** estimated via mixed-effects regression.
* **Strict data filters**: dry only, green-flag laps, no pit laps, outliers removed.
* **Two simulation modes**: deterministic for rankings, stochastic for realism.
* **Aggression & defence traits**: data-driven pass/hold tendencies.
* **Transparent & reproducible**: per-run configs, rejection logs, diagnostic plots.

---

## Simulation Modes

```
+--------------------+                 +----------------------+
|  Deterministic     |                 |  Stochastic          |
|  (Pure Pace)       |                 |  (Race-like Replay)  |
+--------------------+                 +----------------------+
| - No randomness    |                 | - Adds noise/jitter  |
| - No DNFs          |                 | - DNFs per hazard    |
| - No SC/VSC        |                 | - SC/VSC events      |
| - Grid = pace deltas|                | - Randomized starts  |
| - Overtakes = if p>0.5|              | - Overtakes sampled  |
+--------------------+                 +----------------------+
         |                                         |
         |                                         |
         +----------> Shared Engine <--------------+
                       (Pace, Tyres, Traits)
```

---

## Modeling Pipeline

### 1) Clean Data (`src/load_data.py`, `src/filters.py`)

Keep only valid laps:

* Green flag (`TrackStatus=1`), accurate timing.
* Dry conditions only.
* Exclude in/out-laps and pit entries.
* Reject statistical outliers.

### 2) Pace Models (`src/model_metrics.py`)

Mixed model (default):

$$
\text{LapTime} \sim \text{tyre\_comp} + \text{tyre\_age} + \text{lap} + (1|\text{driver}) + (1|\text{team}) + (1|\text{track}) + (1|\text{event})
$$

* **Driver random effects** → intrinsic pace deltas.
* Robust SEs and shrinkage improve stability for small samples.

### 3) Qualifying Metrics

* Normalize by session evolution (Q1/Q2/Q3).
* Use best valid lap per session.

### 4) Event Combination

Precision-weighted blend of race and quali:

$$
\Delta_{\text{event}} = 
\frac{\Delta_R/\sigma_R^2 + \Delta_Q/\sigma_Q^2}{1/\sigma_R^2 + 1/\sigma_Q^2}
$$

### 5) Cross-Event Aggregation (`src/aggregate_metrics.py`)

Weights = inverse variance × recency decay × effective sample size.
Output: driver ranking with uncertainty.

---

## Equal-Car Simulation (`src/visualize_equal_race.py`)

### Lap Time Model

Each lap for driver *i*:

$$
t_{i\ell} = B_{\text{track}} + \Delta_i + D_{\ell}^{(c)} + \varepsilon_{i\ell}
$$

* $B_{\text{track}}$: track base pace.
* $\Delta_i$: driver’s intrinsic pace delta.
* $D_{\ell}^{(c)}$: tyre degradation curve.
* $\varepsilon_{i\ell}$: noise (0 in deterministic mode).

### Overtaking Model

Probability follower *f* passes leader *l*:

$$
p(\text{pass}) = \sigma\!\Big(\alpha\cdot(\Delta t) - \gamma\cdot d_l + \beta\cdot \mathbf{1}_{\text{DRS}} \Big)
$$

* Deterministic: pass if $p > 0.5$.
* Stochastic: sample Bernoulli($p$).

### Reliability Model

Per-lap DNF hazard for driver *i*:

$$
p_{\ell,i} = 1 - (1 - p_{\text{race},i})^{1/L_{\text{typ}}}
$$

* Risk trait raises $p_{\text{race},i}$.
* In deterministic mode: $p_{\ell,i}=0$.

---

## Configuration (`config/config.yaml`)

* **deterministic:** true/false
* **filters:** green flag only, dry only, pit laps removed, outlier drop
* **weights:** race vs quali blend, recency decay
* **simulation:** add\_noise, dnf\_per\_lap, safety\_car, overtake\_model = deterministic/stochastic
* **personality:** toggle aggression/defence/risk

---

## Outputs

* **Rankings**: `outputs/aggregate/driver_ranking.csv` (pace deltas, SEs, sample counts).
* **Replays**: `outputs/viz/simulation.html` (deterministic or stochastic).
* **Calibration**: tyre degradation curves, driver traits, track meta.
* **Diagnostics**: filter rejections, residuals, CV logs.

---

## Limitations & Roadmap

* Pit strategy simplified (single-stint replay).
* Weather effects only partially modeled (track temp multiplier).
* Penalties/sprints excluded.

**Planned**:

* Bayesian driver model with head-to-head priors.
* More granular overtaking model (sector-based).
* Interactive diagnostics dashboard.

---

## Credits

* **Data**: FastF1
* **Modeling**: statsmodels, scikit-learn
* **Visualization**: Plotly, Matplotlib
