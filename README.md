# Stochastic_Non_Bayesian_Filtering_Cancer_Dynamics
Stochastic cancer treatment model (Pinho SDE) with UKF, EnKF, and  Particle Filter state estimation — MSc thesis, Karlstad University 2026.

# Stochastic Cancer Cell Population Dynamics — Bayesian Filtering

**MSc Thesis in Mathematics**  
**Karlstad University, Sweden, 2026**  
**Author:** Richard Assiamah  
**Supervisors:** Dr. Nikos Kavallaris and Dr. Yosief Wondmagegne

---

## Overview

This repository contains the Python implementation developed for the 
MSc thesis:

> *Stochastic Modelling and Nonlinear Bayesian Filtering for Cancer 
> Cell Population Dynamics: A Comparative Study of UKF, EnKF, and 
> Particle Filters*

The work extends the deterministic five-state Pinho cancer treatment 
model (Pinho et al., 2013) to a stochastic differential equation (SDE) 
framework and applies three nonlinear Bayesian filters to estimate 
unobserved biological states from partial observations.

---

## The Model

The **Pinho five-state model** describes the combined chemotherapy and 
anti-angiogenic treatment of cancer. The five state variables are:

| Variable | Description |
|----------|-------------|
| x₁ (NC) | Normal (healthy) cells — **unobserved** |
| x₂ (CC) | Cancer cells — observed |
| x₃ (EC) | Endothelial (blood vessel) cells — **unobserved** |
| y  (CA) | Chemotherapy agent concentration — observed |
| w  (AA) | Anti-angiogenic agent concentration — observed |

The deterministic ODE model (Pinho et al., 2013) is extended to an 
Itô SDE with additive Gaussian noise and discretised using the 
**Euler–Maruyama method**.

---

## Filters Implemented

| Filter | Description |
|--------|-------------|
| **UKF** | Unscented Kalman Filter (scaled unscented transform, α=1e-3) |
| **EnKF** | Ensemble Kalman Filter (N=50 ensemble members) |
| **PF** | Bootstrap Particle Filter with SIR resampling (N=1000 particles) |

All filters are run over **30 Monte Carlo simulations** and evaluated 
using **Root Mean Square Error (RMSE)** and **90% confidence intervals** 
on the unobserved states NC (x₁) and EC (x₃).

---

## Repository Structure

---

## Key Results

- **EnKF** achieved the lowest RMSE for both unobserved states:
  NC (x₁): 0.013486 and EC (x₃): 0.006974
- **PF** (1000 particles) achieved comparable accuracy to EnKF but 
  at significantly higher computational cost
- **UKF** (11 sigma points) was the most computationally efficient 
  but recorded higher RMSE for the unobserved states under stochastic 
  conditions
- All three filters successfully reconstructed the unobserved states 
  from observations of CC, CA, and AA only

---

## Requirements

```bash
pip install numpy matplotlib scipy
```

Python 3.8 or higher recommended.

---

## How to Run

```bash
# Run UKF simulation
python pinho_ukf.py

# Run EnKF simulation
python pinho_enkf.py

# Run Particle Filter simulation
python pinho_pf.py

# Run all filters and generate comparison figure
python pinho_all_filters.py
```

Each script runs 30 Monte Carlo simulations and saves the output 
figure as a PNG file in the current directory.

---

## References

1. Pinho, S. T. R., Bacelar, F. S., Andrade, R. F. S., and Freedman, 
   H. I. (2013). *A mathematical model for the effect of anti-angiogenic 
   therapy in the treatment of cancer tumours by chemotherapy.* 
   Nonlinear Analysis: Real World Applications, 14, 815–828.

2. Khalili, P., Vatankhah, R., and Arefi, M. M. (2024). 
   *State/Parameter Identification in Cancerous Models Using Unscented 
   Kalman Filter.* Cybernetics and Systems, 55(8), 2464–2488.

3. Särkkä, S. and Svensson, L. (2023). *Bayesian Filtering and 
   Smoothing* (2nd ed.). Cambridge University Press.

---

## Citation

If you use this code in your own work, please cite:
Assiamah, R. (2026). Stochastic Modelling and Nonlinear Bayesian
Filtering for Cancer Cell Population Dynamics. MSc Thesis,
Karlstad University, Sweden.