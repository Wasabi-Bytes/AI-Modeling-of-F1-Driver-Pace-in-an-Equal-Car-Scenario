# Equal Car Simulation – F1 Driver & Constructor Championship Modeling

Built a motorsport analytics pipeline to uncover **true driver performance** by stripping away car and team advantages, simulating a fully equalized Formula 1 grid.  

---

## Project Overview
Using **FastF1 telemetry**, this project models how drivers perform with identical machinery.  
We combine **within-team race pace** and **within-team qualifying pace**, then run **Monte Carlo season simulations** to crown fair Drivers’ and Constructors’ Champions.  

**Highlights:**
- Focus on the **last 10–12 races**, with **recency weighting** to balance form and sample size  
- **Race pace**: lap-time model adjusted for tyre compound, tyre age, fuel load, and normalized within each team  
- **Quali pace**: best valid laps compared relative to teammate(s), adjusted for track evolution across Q1–Q3  
- **Monte Carlo**: thousands of simulated seasons with DNFs & randomness  
- **Visuals**: animated equal-car race replay on Montreal with optional driver face PNGs  

---

## Hypothesis
> With equal cars, **drivers who consistently outperform their teammates** (e.g., Verstappen, Hamilton, Alonso) should rise to the top, while car-driven gaps flatten out.  

---

## Workflow
1. **Load** last 10–12 races via FastF1  
2. **Clean** laps (remove pits, outliers, SC/VSC)  
3. **Model race pace** using either:  
   - **Team-controlled regression** per event:  
     `LapTimeSeconds ~ team + driver_within_team + compound + lap_on_tyre + lap_number`  
     → captures driver deltas relative to teammates, removing car advantage  
   - **Correction-factor model with team demeaning**  
4. **Add quali metric** (best valid laps, teammate-normalized, session-adjusted)  
5. **Aggregate** performance across events with uncertainty weighting + recency decay  
6. **Visualize** equal-car race animation (Montreal)  
7. **Simulate** thousands of seasons (Drivers & Constructors)  

---

## Outputs
- **Driver rankings** under equal cars (teammate-controlled)  
- **Interactive race replay** (Montreal track)  
- **Championship forecasts**: driver win % and constructor standings  

---

## Limitations & Future Extensions
- Current model **does not include** weather effects, pit strategy, or safety car randomness.  
- Designed for **pace inference**, not exact race replication.  
- Extensions that could be added later:
  - Track-type adjustments (low vs high downforce)  
  - Stochastic SC/VSC events  
  - Pit window models and overtaking probability  
  - Sprint weekends & penalty models  
  - Hierarchical/Bayesian shrinkage for small-sample drivers  
