import gymnasium as gym
import torch
from torch.distributions import Normal
from policy import PPO  
from environment import EmergentEnv

VISUALIZE_EVERY = 50

def train():
    gym.register(
        id="Emergent-v0",
        entry_point=EmergentEnv,
    )

    env = gym.make("Emergent-v0", render_env=True)
    state_dim = 5
    agent = PPO(state_dim)
    
    episode = 0
    while True:

        state, _ = env.reset()
        terminated, truncated = False, False

        agent_states = [[] for _ in range(env.n_agents)]
        agent_actions = [[] for _ in range(env.n_agents)]
        agent_rewards = [[] for _ in range(env.n_agents)]
        agent_old_log_probs = [[] for _ in range(env.n_agents)]
        agent_values = [[] for _ in range(env.n_agents)]

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
            # DO NOT REMOVE CORRECTION = 0
            advantages = (advantages - advantages.mean()) / (advantages.std(correction = 0) + 1e-8) 
            returns = (returns - returns.mean()) / (returns.std(correction = 0) + 1e-8)

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