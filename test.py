import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from torch.distributions import Normal

import random

import gymnasium as gym

WORLD_SIZE = 2.0
STEP_SIZE = 0.1
MAX_STEPS = 50
VISUALIZE_EVERY = 50

NUM_COLORS = 5
NUM_SHAPES = 5

class EmergentEnv(gym.Env):
    def __init__(self, render_env = False, n_agents = 3):
        self.n_agents = n_agents
        self.agent_pos = torch.zeros((n_agents, 2))
        self.agent_color = torch.randint(0, NUM_COLORS, (n_agents, 1))
        self.agent_shape = torch.randint(0, NUM_SHAPES, (n_agents, 1))

        self.landmark_pos = torch.zeros(2)
        self.landmark_color = torch.randint(0, NUM_COLORS, (1,))
        self.landmark_shape = torch.randint(0, NUM_SHAPES, (1,))
        
        self.observation_space = gym.spaces.Box(-1, 1, (n_agents, 4))
        self.action_space = gym.spaces.Box(-.1, .1, (n_agents, 2))

        self.render_env = render_env


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = (torch.rand((self.n_agents, 2)) * 2 - 1)
        self.agent_color = torch.randint(0, NUM_COLORS, (self.n_agents, 1))
        self.agent_shape = torch.randint(0, NUM_SHAPES, (self.n_agents, 1))

        self.landmark_pos = (torch.rand(2) * 2 - 1)
        self.landmark_color = torch.randint(0, NUM_COLORS, (1,))
        self.landmark_shape = torch.randint(0, NUM_SHAPES, (1,))

        self.current_step = 0

        if self.render_env:
            self.states_traj = [self.agent_pos.clone()]
        
        return self._get_state(), {}

    def _get_state(self):
        return torch.cat([
            self.landmark_pos - self.agent_pos,
            self.agent_pos  # Adding absolute position helps learning
        ], dim=-1)

    def step(self, actions):
        actions = actions * STEP_SIZE
        self.agent_pos += actions
        #self.agent_pos = torch.clamp(self.agent_pos, -1, 1)
        self.current_step += 1
        
        if self.render_env:
            self.states_traj.append(self.agent_pos.clone())

        distances = torch.norm(self.agent_pos - self.landmark_pos, dim=1)
        truncated = self.current_step >= MAX_STEPS
        terminated = (distances < 0.05).all()
        
        rewards = []
        for i in range(self.n_agents):
            dist = distances[i].item()
            action_norm = torch.norm(actions[i]).item()
            reward = -dist - 0.1 * action_norm 
            if dist < 0.05:
                reward += 2.0
            rewards.append(reward)

        observation = self._get_state()
        info = {}
            
        return observation, rewards, terminated, truncated, info
    
    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption(f"PPO")
        clock = pygame.time.Clock()
        
        def scale(pos):
            return int((pos[0] + 1) * 200), int((pos[1] + 1) * 200)
        
        landmark_scaled = scale(self.landmark_pos)
        trajectory = [[scale(p[i]) for p in self.states_traj] for i in range(self.n_agents)]

        for i in range(1, len(self.states_traj)+1):
            screen.fill((255, 255, 255))
            pygame.draw.circle(screen, (0, 255, 0), landmark_scaled, 10)
            
            # Current position
            for traj in trajectory:
                if i-1 < len(traj):
                    pygame.draw.circle(screen, (255, 0, 0), traj[i-1], 5)
                if i > 1 and len(traj) >= 1:
                    pygame.draw.lines(screen, (255, 0, 0), False, traj[:i], 2)
            
            pygame.display.flip()
            clock.tick(20)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        pygame.quit()

# DO NOT TOUCH, FOR WHATEVER REASON,
# THE LAST TANH LAYER IN THE ACTOR MODEL
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

def train():
    gym.register(
        id="Emergent-v0",
        entry_point=EmergentEnv,
    )

    env = gym.make("Emergent-v0", render_env=True)
    state_dim = env.observation_space.shape[-1]
    agent = PPO(state_dim)
    
    episode = 0
    while True:
        agent_states = [[] for _ in range(env.n_agents)]
        agent_actions = [[] for _ in range(env.n_agents)]
        agent_rewards = [[] for _ in range(env.n_agents)]
        agent_old_log_probs = [[] for _ in range(env.n_agents)]
        agent_values = [[] for _ in range(env.n_agents)]

        state, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            with torch.no_grad():
                action_mean, values = agent.policy(torch.FloatTensor(state).to(agent.device))
                action_std = torch.exp(agent.policy.log_std)
                dist = Normal(action_mean, action_std.unsqueeze(0))
                actions = dist.sample()
                log_probs = dist.log_prob(actions.to(agent.device)).sum(dim=-1)

            next_state, rewards, terminated, truncated, _ = env.step(actions.cpu())
            
            for i in range(env.n_agents):
                agent_states[i].append(state[i])
                agent_actions[i].append(actions[i].cpu().numpy())
                agent_rewards[i].append(rewards[i])
                agent_old_log_probs[i].append(log_probs[i].item())
                agent_values[i].append(values[i].item())

            state = next_state

        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []

        for i in range(env.n_agents):
            rewards = agent_rewards[i]
            values = agent_values[i]

            returns = []
            discounted_return = 0
            for r in reversed(rewards):
                discounted_return = r + agent.gamma * discounted_return
                returns.insert(0, discounted_return)
            
            returns = torch.FloatTensor(returns)
            values_tensor = torch.FloatTensor(values)
            advantages = returns - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            all_states.extend(agent_states[i])
            all_actions.extend(agent_actions[i])
            all_old_log_probs.extend(agent_old_log_probs[i])
            all_returns.extend(returns.tolist())
            all_advantages.extend(advantages.tolist())

        agent.update(all_states, all_actions, all_old_log_probs, all_returns, all_advantages)

        if episode % VISUALIZE_EVERY == 0:
            total_reward = sum(sum(r) for r in agent_rewards)
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
            env.render()
        
        episode += 1

if __name__ == "__main__":
    train()