# Equal Car Simulation – F1 Driver & Constructor Championship Modeling

Built a motorsport analytics pipeline to uncover **true driver performance** by stripping away car and team advantages, simulating a fully equalized Formula 1 grid.  

---

## Project Overview
Using **FastF1 telemetry**, this project models how drivers perform with identical machinery.  
We combine **race pace** and **qualifying pace**, then run **Monte Carlo season simulations** to crown fair Drivers’ and Constructors’ Champions.  

**Highlights:**
- Focus on the **last 5 races** for recency & efficiency  
- **Race pace**: clean laps only, no pits/yellows  
- **Quali pace**: median of top flying laps  
- **Monte Carlo**: thousands of simulated seasons with DNFs & randomness  
- **Visuals**: animated race replay on Montreal with optional driver face PNGs  

---

## Hypothesis
> With equal cars, **Lance Stroll, Max Verstappen, or Lewis Hamilton** emerge as leading championship contenders.  


---

## Workflow
1. **Load** last 5 races via FastF1  
2. **Clean** laps (remove pits/outliers)  
3. **Model pace deltas** (race + quali)  
4. **Aggregate** performance across events  
5. **Visualize** equal-car race animation  
6. **Simulate** thousands of seasons (Drivers & Constructors)  

---

## Outputs
- **Driver rankings** under equal cars  
- **Interactive race replay** (Montreal track)  
- **Championship forecasts**: driver win % and constructor standings  
