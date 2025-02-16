import torch.nn as nn
import os
import torch

import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Normal

class Critic(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
        super(Critic, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.net = nn.Sequential(
            nn.Linear(input_dims + n_agents*n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        q = self.net(torch.cat([state, action], dim = 1))
        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class Actor(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        super(Actor, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.movement_net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 2),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(2))

        self.vocab_net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions - 2)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        action_std = torch.exp(self.log_std).clamp(min=1e-6)
        dist = Normal(self.movement_net(state), action_std)
        movement = dist.rsample()
        #movement = self.movement_net(state)
        utterance = self.vocab_net(state)
        utterance = F.gumbel_softmax(utterance, hard=True)

        pi = torch.cat([movement, utterance], dim = 1)

        return pi
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))