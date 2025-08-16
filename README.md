# Equal Car Simulation – F1 Driver & Constructor Championship Modeling

Developed a motorsport analytics pipeline to reveal **true driver performance** by removing car and team advantages, simulating a fully equalized Formula 1 grid.

---

## 🚦 Project Overview
This project uses **FastF1 telemetry data** to estimate how drivers would perform if all cars were equal.  
We combine **race pace** and **qualifying pace**, then run **season-long Monte Carlo simulations** to crown a fair Drivers’ Champion and Constructors’ Champion.  

Key features:
- **Last 5 races** only (keeps analysis fresh, easy to run)
- **Race pace modeling**: clean laps only, pit/yellow removed
- **Qualifying pace modeling**: median of top flying laps
- **Equal car ranking**: performance deltas normalized across events
- **Monte Carlo championship**: thousands of simulated seasons with DNFs & luck modeled
- **Visualizations**: animated race replay on a real F1 track map (e.g., Montreal)
- **Driver face PNGs** (optional): cute visualization with faces racing around the track

---

## 🧪 Hypothesis
> If all drivers had equal machinery, **Lance Stroll, Max Verstappen, or Lewis Hamilton** would emerge as the most consistent championship contenders.  
>  
> This project tests that hypothesis by removing car/team influence and simulating a season under equal-car conditions.

---

## 🎯 Roles This Project Targets
This repo demonstrates skills aligned with multiple career paths:
- **Data Analyst / Data Scientist** – cleaning, analysis, storytelling  
- **Machine Learning Engineer (entry-level)** – predictive modeling (lap times, simulations)  
- **Sports / F1 Analytics** – motorsport niche, high recruiter impact  
- **Data Visualization Specialist** – animated race simulations & championship dashboards  

---

## 📂 Repository Structure

equal-car-simulation/
├─ README.md
├─ requirements.txt
├─ config/
│ └─ config.yaml
├─ data/
│ └─ cache/
├─ faces/ (optional driver PNGs)
├─ notebooks/
│ └─ 01_equal_car_simulation.ipynb
├─ outputs/
│ ├─ driver_ranking.csv
│ ├─ championship_mc_drivers.csv
│ ├─ championship_mc_constructors.csv
│ └─ simulation.html
├─ scripts/
│ └─ run_analysis.py
└─ src/
├─ load_data.py
├─ cleaning.py
├─ quali.py
├─ pace_model.py
├─ simulate.py
└─ championship.py


---

## ⚙️ Workflow
1. **Load data** – fetch last 5 races (Race + Quali) via FastF1  
2. **Clean data** – drop pits, invalid laps, outliers  
3. **Calculate pace deltas** – per driver, per event  
4. **Aggregate** – combine Race + Quali pace across events  
5. **Visualize** – animated “equal car” race simulation (Montreal map)  
6. **Monte Carlo** – simulate entire seasons for Drivers’ & Constructors’ Championships  

---

## 📊 Outputs
- **Driver ranking table** (equal-car pace deltas)  
- **Race animation** (interactive HTML, faces optional)  
- **Championship projections** (driver win %s, constructor standings, win distributions)  

---

## 🏁 Skills Demonstrated
- Data cleaning & preprocessing  
- Exploratory & statistical analysis  
- Simulation modeling (Monte Carlo)  
- Data visualization (Plotly animations, Matplotlib)  
- Domain-specific analytics (motorsport / F1)  

---

## 🚧 Next Steps
- Add interactive dashboards (Streamlit or Dash)  
- Expand to full season dataset  
- Introduce track-specific difficulty multipliers  

---
