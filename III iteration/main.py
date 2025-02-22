import numpy as np
from env import MultiAgentCommEnv

from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
import torch

torch.autograd.set_detect_anomaly(True)

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == "__main__":

    env = MultiAgentCommEnv()
    dir_name="weights"
    n_agents = env.num_agents
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space.shape[-1])

    critic_dims = sum(actor_dims)
    n_actions = env.action_space.shape[-1]

    lr = 1E-3
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64,
                           alpha = lr, beta = lr, scenario = dir_name, chkpt_dir = ".")
    
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 10
    RENDER_INTERVAL = 10
    N_GAMES = 30000
    total_steps = 0
    score_history = []
    best_score = -1E10 

    evaluate = False
    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs, _ = env.reset()
        score = 0
        done = False
        episode_step = 0
        while not done:
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(np.concatenate(np.expand_dims(actions, axis=0)))
            done = terminated or truncated

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-PRINT_INTERVAL:])
        if not evaluate:
            if score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = score

        if i % PRINT_INTERVAL == 0 and i > 0:
            print(f"Episode {i}| Average score: {avg_score}")

        if i % RENDER_INTERVAL == 0 and i > 0:
            env.render()

    
