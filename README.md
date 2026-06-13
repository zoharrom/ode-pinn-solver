# Physics-Informed Neural Network (PINN) for Stiff Epidemiological ODEs

## Overview

This repository contains a custom Physics-Informed Neural Network (PINN) built in PyTorch. It is designed to solve the problem of the non-linear Susceptible-Infected-Recovered (SIR) differential equations. 

Unlike traditional numerical solvers (e.g., Runge-Kutta) that integrate forward in time, this project utilizes continuous, differentiable deep learning approximations to model compartmental transmission dynamics.

## Engineering Challenges Solved

Standard feedforward PINNs notoriously fail when applied to stiff epidemiological models due to extreme loss landscape valleys. This project successfully implements solutions to three major scientific machine learning roadblocks:

1. **The Disease-Free Equilibrium Trap (Trivial Collapse):** Unconstrained networks will minimize loss by predicting flat lines ($I=0$), artificially satisfying the derivatives while ignoring the initial infection spark.
2. **Vanishing Gradients:** Expanding the time domain over the lifespan of an epidemic saturates standard activation functions (like $\tanh$), causing zeroed gradients and paralyzing the optimizer.
3. **Exploding Gradients:** Applying massive loss weights to force compliance with stiff ODE parameters ($\beta=2.0, \gamma=1.0$) causes second-order optimizers (L-BFGS) to overshoot the optimal valley.

## Mathematical Architecture

To guarantee convergence, the following structural algorithms were engineered into the model:

- **Non-dimensionalization (Time Scaling):** The time domain is mathematically normalized to $\tau \in [0, 1]$. By applying chain-rule scaling directly to the differential residuals, inputs remain safely inside the linear regime of the network's activations, permanently stabilizing the gradients.
- **The Linear Ansatz (Hard Constraints):** Initial conditions are mathematically locked into the forward pass using $Output = IC + \tau \cdot \mathcal{N}(\tau)$. This eliminates competing initial-condition loss penalties and allows the optimizer to focus 100% of its capacity on ODE compliance.
- **Physics-Informed Data Assimilation:** To bypass the trivial equilibrium trap, a sparse scaffolding of numerical data (20 anchor points) is integrated into the loss function. The network uses the data for macro-shape routing, while the continuous physics derivatives resolve the high-fidelity non-linear curves.

## Repository Structure

- `model.py`: The PyTorch PINN architecture with Linear Ansatz hard constraints.
- `physics.py`: The non-dimensionalized ODE residuals computed via automatic differentiation (`torch.autograd`).
- `train.py`: The hybrid optimization loop (Adam $\rightarrow$ L-BFGS) and data assimilation logic.
- `sir_pinn_results.png`: Visual verification comparing the PINN's continuous approximation against a SciPy numerical solver.

## Execution

Ensure dependencies are installed via `pip install -r requirements.txt`.
Run the optimization sequence using:
`python train.py`