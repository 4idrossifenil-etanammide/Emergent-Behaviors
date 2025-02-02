import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from torch.distributions import Normal

import gymnasium as gym

# Environment parameters
WORLD_SIZE = 2.0
STEP_SIZE = 0.1
MAX_STEPS = 50
VISUALIZE_EVERY = 50

class EmergentEnv(gym.Env):
    def __init__(self, render_env = False):
        self.agent_pos = torch.zeros(2)
        self.landmark_pos = torch.zeros(2)
        
        self.observation_space = gym.spaces.Box(-1, 1, (4,))
        self.action_space = gym.spaces.Box(-.1, .1, (2,))

        self.render_env = render_env


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = (torch.rand(2) - 0.5)
        self.landmark_pos = (torch.rand(2) * 2 - 1)
        self.current_step = 0

        if self.render_env:
            self.states_traj = [self.agent_pos.clone()]
        
        return self._get_state(), {}

    def _get_state(self):
        return torch.cat([
            self.landmark_pos - self.agent_pos,
            self.agent_pos  # Adding absolute position helps learning
        ])

    def step(self, action):
        action = action * STEP_SIZE
        self.agent_pos += action
        #self.agent_pos = torch.clamp(self.agent_pos, -1, 1)
        self.current_step += 1
        
        if self.render_env:
            self.states_traj.append(self.agent_pos.clone())

        distance = torch.norm(self.agent_pos - self.landmark_pos)
        truncated = self.current_step >= MAX_STEPS
        terminated = distance < 0.05
        
        # Enhanced reward function
        reward = -distance - 0.1 * torch.norm(action)  # Penalize large actions
        if distance < 0.05:  # Success bonus
            reward += 2.0

        observation = self._get_state()
        info = {}
            
        return observation, reward, terminated, truncated, info
    
    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption(f"PPO")
        clock = pygame.time.Clock()
        
        # Modified scaling function
        def scale(pos):
            return int((pos[0] + 1) * 200), int((pos[1] + 1) * 200)
        
        landmark_scaled = scale(self.landmark_pos)
        trajectory = [scale(p) for p in self.states_traj]

        for i in range(1, len(trajectory)+1):
            screen.fill((255, 255, 255))
            pygame.draw.circle(screen, (0, 255, 0), landmark_scaled, 10)
            
            # Current position
            pygame.draw.circle(screen, (0, 0, 255), trajectory[i-1], 5)
            
            # Path drawing
            if i > 1:
                pygame.draw.lines(screen, (255, 0, 0), False, trajectory[:i], 2)
            
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

        # Calculate new policy
        means, _ = self.policy(states)
        stds = torch.exp(self.policy.log_std)
        dist = Normal(means, stds)
        new_log_probs = dist.log_prob(actions).sum(-1)

        # PPO loss
        ratio = (new_log_probs - old_log_probs).exp()
        clipped_ratio = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        _, values = self.policy(states)
        value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()

        # Entropy bonus
        entropy_loss = -dist.entropy().mean()

        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

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
    state_dim = env.observation_space.shape[0]  # Extract observation space length directly
    agent = PPO(state_dim)
    
    episode = 0
    while True:
        states, actions, rewards, dones, old_log_probs = [], [], [], [], []
        state, _ = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            with torch.no_grad():
                action_mean, value = agent.policy(torch.FloatTensor(state).to(agent.device))
                action_std = torch.exp(agent.policy.log_std)
                dist = Normal(action_mean, action_std)
                action = dist.sample().cpu()
                log_prob = dist.log_prob(action.to(agent.device)).sum().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward.item())
            dones.append(terminated or truncated)
            old_log_probs.append(log_prob)

            state = next_state

        # Calculate returns and advantages
        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + agent.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std(correction=0) + 1e-8) # DO NOT TOUCH THE correction=0, OTHERWISE IT WILL NOT WORK

        # Update policy
        agent.update(states, actions, old_log_probs, returns, returns)

        # Visualization
        if episode % VISUALIZE_EVERY == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}")
            env.render()
        
        episode += 1

if __name__ == "__main__":
    train()