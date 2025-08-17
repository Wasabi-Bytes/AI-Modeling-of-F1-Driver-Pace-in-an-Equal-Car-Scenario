# Equal-Car F1 Driver Pace Modeling

**What if every F1 driver raced in the same car?**  
This project estimates their **true pace** by stripping away constructor advantage.  
We model driver performance relative to teammates, aggregate across races, and simulate equal-car outcomes.

---

## Scope & Goals
- **Objective**: Measure each driver’s underlying pace, independent of car/team effects.  
- **Data**: Last 10–12 Grands Prix, weighted by exponential recency decay (≈0.92).  
- **Sessions**: Race (R) and Qualifying (Q).  
- **Outputs**:
  - Equal-car driver ranking (with uncertainties)  
  - Per-event breakdown tables  
  - Montreal equal-start race animation  
  - Monte Carlo season outcomes (champion odds, win distributions)  

---

## Pipeline

### 1. Data Loading & Tagging (`load_data.py`)
- Source: [FastF1](https://theoehrly.github.io/Fast-F1/) telemetry & session data.  
- Tags each lap with driver, compound, stint, lap number, lap_on_tyre counter.  
- Infers stints when missing (using pit in/out flags).  

---

### 2. Race Metrics (`model_metrics.py`)
Two modeling options (chosen in `config.yaml`):  

**a. Correction-Factor Model (`race_metrics_corrections_team`)**  
- Regress lap time on tyre compound, tyre age, lap number.  
- Subtract fitted effects to normalise laps.  
- Demean within team → driver’s residual pace delta.  
- SE estimated from residual variance.  

**b. OLS Team Model (`race_metrics_ols_team`)**  
- Build design: `team` + `driver@team` + lap controls.  
- Fit OLS regression of lap time on team + driver-within-team.  
- Predict reference laps to compute deltas.  
- SEs from regression residuals.  

---

### 3. Qualifying Metrics
- Keep top-k laps per driver/session.  
- Take best lap in Q1, Q2, Q3.  
- Subtract teammate’s best lap in same session.  
- Average across sessions → driver’s quali delta.  
- SE = variance of gaps / √k.  

---

### 4. Event-Level Combination
\[
\Delta_{event} = w_R \cdot \Delta_{race} + w_Q \cdot \Delta_{quali}
\]  
- Defaults: \(w_R = 0.6, w_Q = 0.4\).  
- Missing values fall back to whichever delta is available.  
- Empirical-Bayes shrinkage pulls small-sample drivers toward “average teammate.”  

---

### 5. Cross-Event Aggregation (`aggregate_metrics.py`)
- Inverse-variance weighting × recency decay:  
  \[
  w = \frac{(decay)^{events\_ago}}{SE^2}
  \]  
- Aggregated delta = weighted mean.  
- Aggregated SE = \(\sqrt{1 / \sum w}\).  
- Outputs:
  - `driver_ranking.csv`  
  - `event_breakdown.csv`  

---

### 6. Equal-Car Simulation (`visualize_equal_race.py`)
- Base lap time + driver delta + noise.  
- Linear tyre degradation term per lap.  
- Logistic overtaking probability based on pace gap, DRS, defence.  
- Random DNFs with fixed per-lap probability (~4%).  
- Output: `simulation.html` interactive replay (Montreal circuit).  

---

### 7. Monte Carlo Season Simulator
- Simulates 24 races × 5000 seasons.  
- Each driver’s pace ~ Normal(agg_delta, agg_se).  
- Adds noise + DNF risk.  
- Assigns FIA points.  
- Outputs:  
  - `championship_mc_drivers.csv` (points, champ %, top-3 %, wins)  
  - `championship_mc_constructors.csv`  

---

## Deliverables
**Tables**  
- `driver_ranking.csv`  
- `event_breakdown.csv`  
- `championship_mc_drivers.csv`  
- `championship_mc_constructors.csv`  

**Visuals**  
- `simulation.html` (equal-car race replay)  
- Diagnostics & plots  

**Narrative**  
- This README (assumptions, limitations, interpretation)  

---

## Limitations & Extensions
**Simplifications**  
- No pit stops or tyre strategy  
- No weather or track evolution  
- DNFs uniform across teams  
- Safety cars, penalties, sprints excluded  

**Planned Extensions**  
- Track-type effects  
- Tyre-specific degradation curves  
- Hierarchical Bayesian shrinkage  
- Pit strategy modeling  
- Calibrated overtaking model & safety-car logic  
- Driver “personality” parameters (aggression, defence, risk)  

---

## References
- **Data**: FastF1  
- **Models**: scikit-learn, statsmodels  
- **Simulation**: logistic overtaking, Monte Carlo outcomes  
