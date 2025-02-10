import gymnasium as gym
import torch
from torch.distributions import Normal
from policy import PPO  
from environment import EmergentEnv
import environment
import memory

import matplotlib.pyplot as plt

VISUALIZE_EVERY = 50
PLOT_EVERY = 100
MAX_PLOT = 10000

GAMMA = 0.9

def train():
    gym.register(
        id="Emergent-v0",
        entry_point=EmergentEnv,
    )

    env = gym.make("Emergent-v0", render_env=True)
    state_dim = 2 + environment.VOCAB_SIZE + environment.MEMORY_SIZE + 1 # 2 because position are bidimensional, 1 because goal is a scalar
    agent = PPO(state_dim, gamma = GAMMA)
    rewards_history = []
    trainingMemory = memory.Memory()
    
    episode = 0
    while True:

        state, _ = env.reset()
        n_agents = env.unwrapped.n_agents

        terminated, truncated = False, False

        agent_states = {
            "physical": [],
            "utterances": [],
            "memories": [],
            "tasks": []
        } ###

        trainingMemory.add_episode(n_agents)
        agent_actions = [[] for _ in range(n_agents)]###
        agent_rewards = [[] for _ in range(n_agents)]###
        agent_old_log_probs = [[] for _ in range(n_agents)]###
        agent_values = [[] for _ in range(n_agents)]###

        while not (terminated or truncated):
            with torch.no_grad():
                dist, utterances, delta_memories, values= agent.policy({k:v.to(agent.device) for k,v in state.items()})
                actions = dist.sample()
                log_probs = dist.log_prob(actions.to(agent.device)).sum(dim=-1)
                log_probs = log_probs.view(-1)

            next_state, rewards, terminated, truncated, _ = env.step([actions.cpu(), utterances, delta_memories])
            
            for i in range(n_agents):
                trainingMemory.add_step(i, actions[i].cpu().numpy(), rewards[i], log_probs[i].item(), values[i].item())
                agent_actions[i].append(actions[i].cpu().numpy())###
                agent_rewards[i].append(rewards[i])###
                agent_old_log_probs[i].append(log_probs[i].item())###
                agent_values[i].append(values[i].item())###

            trainingMemory.add_step_state(state, agent)
            agent_states["physical"].append(state["physical"].to(agent.device))###
            agent_states["utterances"].append(state["utterances"].to(agent.device))###
            agent_states["memories"].append(state["memories"].to(agent.device))###
            agent_states["tasks"].append(state["tasks"].to(agent.device))###

            state = next_state

        all_actions = [] ###
        all_old_log_probs = []###
        all_returns = []###
        all_advantages = []###

        for i in range(n_agents):
            rewards = trainingMemory.get_reward(episode, i)
            values = trainingMemory.get_values(episode, i)

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

            all_actions.extend(agent_actions[i])
            all_old_log_probs.extend(agent_old_log_probs[i])
            all_advantages.extend(advantages.tolist())###
            trainingMemory.add_objective_functions(i, returns.tolist(), advantages.tolist())

        all_states = {k: torch.cat(v, dim=0) for k,v in trainingMemory.get_states(episode).items()}
        all_returns = torch.cat([ torch.Tensor(trainingMemory.get_returns(episode,i)) for i in range(n_agents)])
        #print(torch.equal(all_returns2, torch.FloatTensor(all_returns)))
        agent.update(all_states, all_actions, all_old_log_probs, all_returns, all_advantages)
        total_reward = sum(sum(r) for r in agent_rewards) / n_agents
        rewards_history.append(total_reward)


        if episode % VISUALIZE_EVERY == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
            env.render()

        if episode % PLOT_EVERY == 0:
            plt.figure(figsize=(10,5))
            plt.plot(rewards_history)
            plt.title('Total Reward per episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.savefig('rewards_plot.png')
            plt.close()
        
        episode += 1


if __name__ == "__main__":
    train()