import torch
from torch import nn, optim
from torch.distributions import Normal
import torch.nn.functional as F

from actor import Actor
import environment

class ActorCritic(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.actor = Actor(hidden_dim)

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim*2 + environment.MEMORY_SIZE + 3, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, state):
        physical = state["physical"]
        utterances = state["utterances"]
        memories = state["memories"]
        tasks = state["tasks"]

        actions_means, utterances_logits, delta_memories, physical_features, utterances_features = self.actor(physical, utterances, memories, tasks)

        action_std = torch.exp(self.log_std).clamp(min=1e-6)
        dist = Normal(actions_means, action_std) #removed an unsqueeze here
        next_utterances = F.gumbel_softmax(utterances_logits, tau=1.0, hard=True)
        #critic should be evaluating actor's action
        return dist, next_utterances, delta_memories, self.critic(torch.cat([physical_features, utterances_features, memories, tasks], dim = -1))

class PPO:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def update(self, states, actions, old_log_probs, returns, advantages):
        actions = torch.FloatTensor([a.tolist() for a in actions]).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device).detach()
        returns = returns.to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        c1=0.5
        c2= 0.01

        # Compute new log probabilities
        dist, utterances, _, values = self.policy(states) # TODO implement utterances loss
        new_log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().mean()

        # PPO loss
        ratio = torch.exp(new_log_probs - old_log_probs) # ratio between log probs
        clipped_ratio = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) # clipped ratio
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean() # ensure small updates 

        # Value loss
        value_loss = F.mse_loss(returns, values.squeeze())

        total_loss = policy_loss +  c1*value_loss  + c2*entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()