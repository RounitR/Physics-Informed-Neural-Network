import torch
import torch.nn as nn

class TrajectoryPINN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1 + latent_dim, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 128),
            nn.Tanh(),

            nn.Linear(128, 2)
        )

    def forward(self, t, z):
        inp = torch.cat([t, z], dim=1)
        return self.net(inp)
