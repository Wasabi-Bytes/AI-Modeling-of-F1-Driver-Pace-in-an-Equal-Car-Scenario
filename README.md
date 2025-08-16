# Equal Car Simulation – F1 Driver & Constructor Championship Modeling

Built a motorsport analytics pipeline to uncover **true driver performance** by stripping away car and team advantages, simulating a fully equalized Formula 1 grid.  

---

## Project Overview
Using **FastF1 telemetry**, this project models how drivers perform with identical machinery.  
We combine **race pace** and **qualifying pace**, then run **Monte Carlo season simulations** to crown fair Drivers’ and Constructors’ Champions.  

**Highlights:**
- Focus on the **last 5 races** for recency & efficiency  
- **Race pace**: lap-time model adjusted for tyre compound, tyre age, and fuel load  
- **Quali pace**: median of top flying laps on Softs (cleaned of track-limit deletions if available)  
- **Monte Carlo**: thousands of simulated seasons with DNFs & randomness  
- **Visuals**: animated race replay on Montreal with optional driver face PNGs  

---

## Hypothesis
> With equal cars, **Lance Stroll, Max Verstappen, or Lewis Hamilton** emerge as leading championship contenders.  

---

## Workflow
1. **Load** last 5 races via FastF1  
2. **Clean** laps (remove pits, outliers, SC/VSC)  
3. **Model race pace** using either:
   - **OLS regression** per event:  
     `LapTimeSeconds ~ driver + compound + lap_on_tyre + lap_number`  
   - **Simple correction factors**: normalize each lap for compound offsets, tyre degradation, and fuel load  
4. **Add quali metric** (median of top-k valid laps)  
5. **Aggregate** performance across events with uncertainty weighting  
6. **Visualize** equal-car race animation (Montreal)  
7. **Simulate** thousands of seasons (Drivers & Constructors)  

---

## Outputs
- **Driver rankings** under equal cars  
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
  - Bayesian shrinkage for small-sample drivers  
