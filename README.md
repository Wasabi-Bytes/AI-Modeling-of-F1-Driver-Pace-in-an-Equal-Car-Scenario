# AI Modeling of F1 Driver Pace in an Equal-Car Scenario

This project models and simulates Formula 1 driver performance under the assumption that all cars are equal.  
The goal is to estimate which drivers are genuinely fastest, independent of constructor advantage.

---

## How Driver Skill Is Isolated

### Race Metrics
- Lap times are normalized for tyre compound, tyre age, and fuel effect.  
- Drivers are compared only against their teammates, canceling out car advantages.  
- Regression models (linear and OLS variants) estimate within-team deltas, with residual variance used for uncertainty.  

### Qualifying Metrics
- Each driver’s best lap is measured relative to their teammate’s best lap per session.  
- Track evolution (Q1 → Q3) is controlled to avoid bias from rubbering-in.  
- Session deltas are averaged to give qualifying pace independent of machinery.  

---

## Shrinkage & Aggregation
- Event-level teammate deltas are combined across races and seasons.  
- Empirical-Bayes shrinkage reduces noise by pulling estimates toward the grid average.  
- Weights account for recency and variance, so recent consistent performances count more.  

---

## Equal-Car Simulation

Using the aggregated skill deltas, we simulate races where every driver has the same car.

- **Base car performance**: A fixed `BASE_LAP_SEC` is used for all drivers.  
- **Driver skill**: Each driver’s lap time is adjusted by their aggregated delta (driver-only effect).  

**Race dynamics:**
- Random noise models race-day fluctuations.  
- Overtakes use a logistic probability function depending on pace difference and DRS/slipstream effects.  
- DNFs occur with fixed per-lap probability (configurable).  
- No constructor effects: Engine, aero, and team performance are removed — only driver ability matters.  

This allows us to simulate head-to-head races where the true best driver wins on skill alone, not because of the car.

---

## Goals
- Provide a data-driven ranking of F1 drivers independent of machinery.  
- Explore how different drivers would perform if the grid had equal cars.  
- Create realistic race simulations that showcase pure driver talent.  

---

## Limitations & Future Extensions

**Current model does not include:**
- Pit strategy  
- Weather  
- Safety cars  
- Sprint race effects  

It is designed for **driver skill inference**, not exact race replication.

**Future work may add:**
- Track-specific difficulty adjustments  
- Stochastic SC/VSC events  
- Pit window models and overtaking probability curves  
- Penalty and grid-drop models  
- Bayesian shrinkage for rookies and low-sample drivers  

---
