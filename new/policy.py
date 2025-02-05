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
            nn.Linear(2 + hidden_dim*2 + environment.MEMORY_SIZE + 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        physical = state["physical"]
        utterances = state["utterances"]
        memories = state["memories"]
        tasks = state["tasks"]

        actions_means, actions_log_std, utterances_logits, delta_memories, physical_features, utterances_features = self.actor(physical, utterances, memories, tasks)

        action_std = torch.exp(actions_log_std)
        dist = Normal(actions_means, action_std) #removed an unsqueeze here
        actions = dist.sample()
        next_utterances = F.gumbel_softmax(utterances_logits, tau=1.0, hard=True)
        #critic should be evaluating actor's action
        return actions_means, actions_log_std, next_utterances, delta_memories, self.critic(torch.cat([actions, physical_features, utterances_features, memories, tasks], dim = 1)), actions 

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
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Compute new log probabilities
        means, log_std, utterances, _, values, _ = self.policy(states) # TODO implement utterances loss
        stds = torch.exp(log_std)
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