import torch

def sir_residuals(tau, model, beta, gamma, t_max):
    tau = tau.requires_grad_(True)
    output = model(tau)
    s, i, r = output[:, 0:1], output[:, 1:2], output[:, 2:3]

    # Calculate derivatives with respect to normalized time (tau)
    ds_dtau = torch.autograd.grad(s, tau, grad_outputs=torch.ones_like(s), create_graph=True)[0]
    di_dtau = torch.autograd.grad(i, tau, grad_outputs=torch.ones_like(i), create_graph=True)[0]
    dr_dtau = torch.autograd.grad(r, tau, grad_outputs=torch.ones_like(r), create_graph=True)[0]

    # Chain-rule scaling
    f_s = ds_dtau - t_max * (-beta * s * i)
    f_i = di_dtau - t_max * (beta * s * i - gamma * i)
    f_r = dr_dtau - t_max * (gamma * i)

    return f_s, f_i, f_r

def conservation_loss(model, tau):
    output = model(tau)
    return torch.mean((output.sum(dim=1) - 1.0)**2)