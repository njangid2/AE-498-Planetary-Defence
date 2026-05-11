# Asteroid Deflection & Mission Design Simulations

This repository contains simulation tools and analysis scripts related to asteroid deflection studies, low-thrust mission design, Monte Carlo uncertainty propagation, and future impact probability assessment.

---

## Repository Structure

### `quad/`
This folder contains all code related to quad charts, including plotting scripts, visualization utilities, and supporting analysis used for mission concept presentations and summaries.

---

### `NucleardeltaV/`
This folder contains scripts and notebooks related to nuclear deflection simulations and trajectory propagation.

#### Files

##### `Lowthrust.ipynb`
Notebook for low-thrust mission design and trajectory analysis.

Includes:
- Low-thrust transfer setup
- Mission trajectory analysis

---

##### `prop_1m.py`
Performs a 100k Monte Carlo simulation for:
- Calculating the applied deflection \( \Delta V \)
- Propagating asteroid trajectories to the original date of impact
- Assessing post-deflection impact geometry and uncertainty

---

##### `prop_future.py`
Performs a 100k Monte Carlo simulation for:
- Calculating the applied deflection \( \Delta V \)
- Propagating asteroid trajectories forward to the year 2099
- Evaluating future impact probabilities and long-term orbital evolution after deflection

---
