import torch
from torch import nn, optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPO:
    def __init__(self, state_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor([s.tolist() for s in states]).to(self.device)
        actions = torch.FloatTensor([a.tolist() for a in actions]).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device).detach()
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Compute new log probabilities
        means, values = self.policy(states)
        stds = torch.exp(self.policy.log_std)
        dist = Normal(means, stds)
        new_log_probs = dist.log_prob(actions).sum(-1)

        # PPO loss
        ratio = (new_log_probs - old_log_probs).exp() # ratio between log probs
        clipped_ratio = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) # clipped ratio
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean() # ensure small updates 

        # Value loss
        value_loss = (returns - values.squeeze()).pow(2).mean()

        total_loss = policy_loss +  value_loss 

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()