from torch import nn
import torch
import torch.nn.functional as F

import environment

class Actor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dim + environment.MEMORY_SIZE + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2 + environment.VOCAB_SIZE + environment.MEMORY_SIZE),
            nn.Tanh()
        )

        self.physical_processor = nn.Sequential(
            nn.Linear(6, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.utterance_processor = nn.Sequential(
            nn.Linear(environment.VOCAB_SIZE + environment.MEMORY_SIZE, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        

    def forward(self, physical, utterances, memories, tasks):
        num_agents, num_land_agents, dim = physical.shape
        physical_features = self.physical_processor(physical.view(-1, dim))
        physical_features = physical_features.view(num_agents, num_land_agents, -1)
        physical_features = SoftmaxPooling(dim=-2)(physical_features) # shape: (n_agents, 256)

        utterances_input = torch.cat([utterances.repeat(num_agents, 1, 1) , 
                                      memories.repeat(1,num_agents).view(num_agents, num_agents, -1)], dim = 2)
        utterances_features = self.utterance_processor(utterances_input)
        utterances_features = SoftmaxPooling(dim=-2)(utterances_features)
        
        output = self.action_net(torch.cat([physical_features, memories, tasks], dim=-1))
        actions_means = output[:, :2]
        utterances_logits = output[:, 2: 2 + environment.VOCAB_SIZE]
        delta_memories = output[:, 2 + environment.VOCAB_SIZE:]
        return actions_means, utterances_logits, delta_memories, physical_features, utterances_features

class SoftmaxPooling(nn.Module):
    def __init__(self, dim, temperature = 1.0):
        super(SoftmaxPooling, self).__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x):
        weights = F.softmax(x / self.temperature, dim=self.dim)
        pooled = torch.sum(weights * x, dim=self.dim)
        return pooled