from torch import nn
import torch
import torch.nn.functional as F

import environment

class Actor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim*2 + environment.MEMORY_SIZE + 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 + 2 + environment.VOCAB_SIZE + environment.MEMORY_SIZE)
        )

        self.physical_processor = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.utterance_processor = nn.Sequential(
            nn.Linear(environment.VOCAB_SIZE + environment.MEMORY_SIZE, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        

    def forward(self, physical, utterances, memories, tasks):
        num_agents, num_land_agents, dim = physical.shape
        physical_features = self.physical_processor(physical.view(-1, dim))
        physical_features = physical_features.view(num_agents, num_land_agents, -1)
        physical_features = SoftmaxPooling(dim=1)(physical_features) # shape: (n_agents, 256)

        utterances_features = self.utterance_processor(torch.cat([utterances, memories], dim = 1))
        utterances_features = SoftmaxPooling(dim=0)(utterances_features)
        utterances_features = utterances_features.repeat(num_agents,1) # shape (n_agents, 256)

        output = self.action_net(torch.cat([physical_features, utterances_features, memories, tasks], dim=1))
        actions_means = nn.Tanh()(output[:, :2])
        action_log_std = output[:, 2:4]
        utterances_logits = output[:, 4: 4 + environment.VOCAB_SIZE]
        delta_memories = output[:, 4 + environment.VOCAB_SIZE:]

        return actions_means, action_log_std, utterances_logits, delta_memories, physical_features, utterances_features

class SoftmaxPooling(nn.Module):
    def __init__(self, dim, temperature = 1.0):
        super(SoftmaxPooling, self).__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x):
        weights = F.softmax(x / self.temperature, dim=self.dim)
        pooled = torch.sum(weights * x, dim=self.dim)
        return pooled