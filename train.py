import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from model import SIRPINN
from physics import conservation_loss, sir_residuals

# Epidemic parameters
BETA = 2.0
GAMMA = 1.0
S0, I0, R0 = 0.99, 0.01, 0.0
T_MAX = 15.0

# Hyperparameters
EPOCHS_ADAM = 6000
LR = 1e-3
N_COLLOCATION = 400
N_DATA = 20  # Only 20 scaffolding points to guide the PINN

COMPARTMENT_COLORS = {"S": "#0066CC", "I": "#CC0000", "R": "#228B22"}

def sir_ode(y, t, beta, gamma):
    s, i, r = y
    return [-beta * s * i, beta * s * i - gamma * i, gamma * i]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIRPINN().to(device)
    
    # 1. Physics Collocation Points (Unlabeled - Forces math compliance)
    tau_collocation = torch.linspace(0, 1.0, N_COLLOCATION, device=device).unsqueeze(1)

    # 2. Synthetic Data Scaffolding (Labeled - Prevents Trivial Collapse)
    t_data = np.linspace(0, T_MAX, N_DATA)
    exact_data = odeint(sir_ode, [S0, I0, R0], t_data, args=(BETA, GAMMA))
    tau_data = torch.tensor(t_data / T_MAX, dtype=torch.float32, device=device).unsqueeze(1)
    
    s_data = torch.tensor(exact_data[:, 0:1], dtype=torch.float32, device=device)
    i_data = torch.tensor(exact_data[:, 1:2], dtype=torch.float32, device=device)
    r_data = torch.tensor(exact_data[:, 2:3], dtype=torch.float32, device=device)

    print("Starting Phase 1: Adam Optimization...")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS_ADAM + 1):
        optimizer_adam.zero_grad()

        # Calculate standard physics residuals
        f_s, f_i, f_r = sir_residuals(tau_collocation, model, BETA, GAMMA, T_MAX)
        loss_physics = torch.mean(f_s**2 + f_i**2 + f_r**2)
        loss_cons = conservation_loss(model, tau_collocation)

        # Calculate Data Scaffolding Error
        pred_data = model(tau_data)
        loss_data = torch.mean((pred_data[:, 0:1] - s_data)**2 +
                               (pred_data[:, 1:2] - i_data)**2 +
                               (pred_data[:, 2:3] - r_data)**2)

        # Combine them: The 100.0 weight guarantees the network hits the anchor points
        loss = loss_physics + loss_cons + 100.0 * loss_data
        loss.backward()
        optimizer_adam.step()

        if epoch % 1000 == 0 or epoch == 1:
            print(f"Adam Epoch {epoch:5d} | total={loss.item():.6f} | phys={loss_physics.item():.6f} | data={loss_data.item():.6f}")

    print("\nStarting Phase 2: L-BFGS Optimization...")
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5000, line_search_fn="strong_wolfe")

    def closure():
        optimizer_lbfgs.zero_grad()
        f_s, f_i, f_r = sir_residuals(tau_collocation, model, BETA, GAMMA, T_MAX)
        loss_physics = torch.mean(f_s**2 + f_i**2 + f_r**2)
        loss_cons = conservation_loss(model, tau_collocation)
        
        pred_data = model(tau_data)
        loss_data = torch.mean((pred_data[:, 0:1] - s_data)**2 +
                               (pred_data[:, 1:2] - i_data)**2 +
                               (pred_data[:, 2:3] - r_data)**2)
        
        loss = loss_physics + loss_cons + 100.0 * loss_data
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    final_loss = closure()
    print(f"Final L-BFGS Loss: {final_loss.item():.6f}")

    return model, device

def plot_results(model, device):
    t_eval = np.linspace(0, T_MAX, 500)
    exact = odeint(sir_ode, [S0, I0, R0], t_eval, args=(BETA, GAMMA))

    # Query network using normalized time
    tau_eval = t_eval / T_MAX
    tau_tensor = torch.tensor(tau_eval, dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        pred = model(tau_tensor).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    compartments = [
        ("Susceptible", 0, COMPARTMENT_COLORS["S"]),
        ("Infected", 1, COMPARTMENT_COLORS["I"]),
        ("Recovered", 2, COMPARTMENT_COLORS["R"]),
    ]

    for name, idx, color in compartments:
        # Thick, transparent background line for SciPy
        ax.plot(t_eval, exact[:, idx], color=color, linestyle="-", linewidth=7.0, alpha=0.3, label=f"{name} (SciPy)")
        # Thin, dashed foreground line for the PINN
        ax.plot(t_eval, pred[:, idx], color=color, linestyle="--", linewidth=2.0, label=f"{name} (PINN)")

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Population fraction", fontsize=12)
    ax.set_title(f"SIR Model: PINN vs SciPy (β={BETA}, γ={GAMMA})", fontsize=13)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("sir_pinn_results.png", dpi=150)
    print("Saved plot to sir_pinn_results.png")
    plt.show()

if __name__ == "__main__":
    model, device = train()
    plot_results(model, device)