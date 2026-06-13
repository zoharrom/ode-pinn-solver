import torch
import torch.nn as nn

class SIRPINN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_units=64):
        super().__init__()
        layers = [nn.Linear(1, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, 3))
        self.net = nn.Sequential(*layers)
        
        # Hardcode initial conditions
        self.s0 = 0.99
        self.i0 = 0.01
        self.r0 = 0.0

    def forward(self, tau):
        raw_out = self.net(tau)
        
        # Linear Ansatz: At tau=0, the output is exactly the starting conditions.
        s = self.s0 + tau * raw_out[:, 0:1]
        i = self.i0 + tau * raw_out[:, 1:2]
        r = self.r0 + tau * raw_out[:, 2:3]
        
        return torch.cat([s, i, r], dim=1)