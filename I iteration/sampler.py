import torch.nn as nn
import torch

class Sampler(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Sampler, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.sigma = nn.Parameter(torch.ones(output_dim) * 10)

    def forward(self, x):
        mean = self.network(x)
        sigma = torch.clamp(self.sigma, 1e-3, 50.0)
        dist = torch.distributions.Normal(mean, sigma)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob