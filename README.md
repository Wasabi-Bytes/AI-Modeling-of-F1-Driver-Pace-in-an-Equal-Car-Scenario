# AI Modeling of F1 Driver Pace in an Equal-Car Scenario

This project estimates Formula 1 drivers’ **true pace** by removing constructor advantages.  
We simulate races as if all drivers competed in identical machinery.

---

## Overview

- **Goal**: Isolate driver skill from car/team effects.  
- **Method**: Normalize lap times, compare teammates, and aggregate across recent events with statistical shrinkage.  
- **Outputs**:
  - Equal-car driver ranking (with uncertainty)
  - Interactive race replay animation (Montreal circuit)
  - Monte Carlo season simulations with championship odds
  - Per-event breakdown tables

---

## Pipeline

### 1. Data Loading
- Source: [FastF1](https://theoehrly.github.io/Fast-F1/) telemetry and session data  
- Sessions: Race (R) and Qualifying (Q)  
- Recent races: last 10–12 Grands Prix, weighted by exponential recency decay (default ≈ 0.92)  

### 2. Cleaning
- Keep only valid laps (exclude pit in/out, SC/VSC/RED flag laps)  
- Bound lap times to [60s, 180s]  
- Trim per-driver to 5th–95th percentile  
- Diagnostics: sample size per driver/event  

### 3. Event Metrics

#### Race Pace Models
**Option A — OLS Regression (team-controlled)**  

\[
LapTime_{it} = \alpha + \beta_{\text{team}(i)} + \delta_{\text{driver}(i)} + f(compound) + g(lap\_on\_tyre) + h(lap\_number) + \varepsilon_{it}
\]

**Option B — Correction-Factor Model**  

\[
norm\_time = raw - \theta_{compound} - \beta f(lap\_on\_tyre) - \gamma g(lap\_number)
\]  

Subtract team mean → driver’s relative pace.  
Uncertainty from residual variance.  

#### Qualifying Pace
- Select best valid lap(s) per Q1/Q2/Q3  
- Normalize by teammate’s best lap  
- Average across sessions  
- Uncertainty = variance / √k  

### 4. Event Combination
Weighted average of race and quali metrics:  

\[
\Delta_{event} = w_R \cdot \Delta_{race} + w_Q \cdot \Delta_{quali}
\]

Default: \(w_R = 0.6, \, w_Q = 0.4\).  
Event uncertainty combines both SEs.  

### 5. Cross-Event Aggregation
- Inverse-variance weighting across events  
- Exponential recency decay (default = 0.92)  
- Shrinkage (Empirical-Bayes) to stabilize small-sample drivers  

**Outputs**:  
- `driver_ranking.csv` (aggregated deltas and uncertainties)  
- `event_breakdown.csv` (per-event statistics)  

---

## Equal-Car Race Simulation

### Lap Model
\[
t_{driver, lap} = BASE\_LAP + \Delta_{driver} + \varepsilon, 
\quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
\]

### Race Dynamics
- Overtakes: logistic probability model based on pace gap, DRS, slipstream  
- DNFs: fixed per-lap probability (~4%)  
- No team/engine effects — pure driver skill  

**Output**: `simulation.html` interactive Montreal replay  

---

## Monte Carlo Season Simulator

- Simulates 24 races × 5000 iterations  
- Driver pace ~ Normal(agg_delta, agg_se)  
- Adds stochastic noise and DNF risk  
- Assigns F1 points per FIA system  

**Outputs**:  
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
- `simulation.html` (equal-car race animation)  
- Diagnostic plots  

**Documentation**  
- README with assumptions and limitations  

---

## Limitations & Future Work

### Current Simplifications
- No pit strategy modeling  
- No weather effects  
- No stochastic SC/VSC/penalties  
- Sprint weekends not modeled  

### Potential Extensions
- Track-type specific effects  
- Hierarchical Bayesian shrinkage  
- Overtaking model calibration to history  
- Pit window and tyre degradation strategy  
- Safety car and incident simulations  

---

## Project Logic (Code Flow)

1. **Data loading/tagging** (`load_data.py`)  
   - Loads races via FastF1, tags compound, stint, tyre age  

2. **Race metrics** (`model_metrics.py`)  
   - OLS team model  
   - Correction-factor model  

3. **Qualifying metrics**  
   - Best laps per Q session, normalized vs teammate  

4. **Event-level aggregation**  
   - Weighted race + quali, empirical-Bayes shrinkage  

5. **Cross-event aggregation** (`aggregate_metrics.py`)  
   - Inverse-variance + recency weighting  

6. **Equal-car simulation** (`visualize_equal_race.py`)  
   - Race dynamics with overtakes, DNFs  
