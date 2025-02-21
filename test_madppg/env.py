import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pygame

VOCAB_SIZE = 10

class MultiAgentCommEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Environment parameters
        self.world_size = 2.0  # Size of the square world
        self.num_agents = 2
        self.num_landmarks = 4
        self.episode_length = 25
        self.step_count = 0
        
        # Define color landmarks (RGB)
        self.landmark_colors = np.array([0,1,2,3])
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        
        # Define action spaces (movement + communication)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, 2 + 1), dtype=np.float32)
        self.obs_size = (self.num_landmarks*2) + (self.num_agents*2) + 1 + 2 + (self.num_agents - 1)  # 3 landmarks, 2 agents, 1 color, other agent pos, n-1 communications
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, self.obs_size), dtype=np.float32)
        self.agent_positions = None
        self.landmark_positions = None
        self.communications = np.zeros((self.num_agents))


    def reset(self, seed=None, options=None):
        # Initialize agents and landmarks randomly
        self.agent_positions = np.random.uniform(-1, 1, size=(2, 2))
        self.landmark_positions = np.random.uniform(-1, 1, size=(self.num_landmarks, 2))
        self.step_count = 0
        
        # Reset communications
        self.communications = np.zeros((self.num_agents))
        self.goals = np.random.choice(self.num_landmarks, self.num_agents)
        self.agent_goals = np.array([1,0]) # TODO: Permute over all the possible agents. For now agent 0 has agent 1 and viceversa

        self.trajectories = [self.agent_positions.copy()]
        self.communications_traj = [self.communications.copy()]

        self.distance_to_target_prev = np.linalg.norm(self.agent_positions - self.landmark_positions[self.goals], axis = 1)
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.zeros((2, self.obs_size))
        
        for i in range(self.num_agents):
            
            # Relative positions to landmarks (3 landmarks * 2D)
            rel_positions = (self.agent_positions[i] - self.landmark_positions).flatten()
            agents_rel_positions = (self.agent_positions[i] - self.agent_positions).flatten()
            
            # Target color (3D)
            #target_color = self.landmark_colors[self.goals[i]] # An agent doesn't have to see it's own color, but rather the others target color
            other_agent = self.agent_goals[i]
            target_color = self.landmark_colors[self.goals[other_agent]]
            other_rel_pos = self.agent_positions[other_agent] - self.landmark_positions[self.goals[other_agent]]

            comm = []
            for j in range(self.num_agents):
                if i != j:
                    comm.append([self.communications[j]])

            comm = np.concatenate(comm)
            
            # Concatenate observations
            obs[i] = np.concatenate([
                rel_positions,
                agents_rel_positions,
                np.array([target_color]),
                other_rel_pos,
                comm
            ])
            
        return obs
    
    def step(self, actions):
        self.step_count += 1
        
        # Update positions
        self.agent_positions = np.clip(self.agent_positions + actions[:, :2] * 0.1, -0.98, 0.98)
        self.communications = actions[:, -1]
        
        self.trajectories.append(self.agent_positions.copy())
        self.communications_traj.append(self.communications.copy())

        # Calculate rewards
        #distance_to_target = np.linalg.norm(self.agent_positions - self.landmark_positions[self.goals], axis = 1)
        #rewards = self.distance_to_target_prev - distance_to_target
        #self.distance_to_target_prev = distance_to_target.copy()

        #distance = sum(np.linalg.norm(self.agent_positions - self.landmark_positions[self.goals], axis = 1))
        #rewards = np.repeat(-distance, self.num_agents)

        rewards = np.zeros((self.num_agents))
        for i in range(self.num_agents):
            
            distance = np.linalg.norm(self.agent_positions[i] - self.landmark_positions[self.goals[i]])
            #other_distance = np.linalg.norm(self.agent_positions[self.agent_goals[i]] - self.landmark_positions[self.goals[self.agent_goals[i]]])
            rewards[i] = -distance*0.5
            if distance < 0.15:
                rewards[i] += 1
            if distance < 0.1:
                rewards[i] += 1
            if distance < 0.05:
                rewards[i] += 1 
        
        tmp = rewards[0]
        rewards[0]+=rewards[1]
        rewards[1]+=tmp
        # Check termination
        truncated = self.step_count >= self.episode_length
        #terminated = (np.abs(self.agent_positions - self.landmark_positions[self.goals]) < 0.05).all()
        terminated = np.all([
            np.linalg.norm(self.agent_positions[i] - self.landmark_positions[self.goals[self.agent_goals[i]]]) < 0.05
            for i in range(self.num_agents)
        ])
        
        return self._get_obs(), rewards, terminated, truncated, {}

    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption(f"PPO")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
        scale = lambda x: (int((x[0] + 1) * 200), int((x[1] + 1) * 200))

        landmarks_scaled = [scale(l) for l in self.landmark_positions]
        scaled_trajectory = [[scale(p[i]) for p in self.trajectories] for i in range(self.num_agents)]

        for i in range(1, len(self.trajectories)+1):
            screen.fill((255, 255, 255))
            for k, landmark in enumerate(landmarks_scaled):
                pygame.draw.circle(screen, self.colors[k], landmark, 10)
                text = font.render(str(k), True, (0,0,0))
                screen.blit(text, (landmark[0] - text.get_width() // 2, landmark[1] - text.get_height() // 2))
            
            # Current position
            for j, traj in enumerate(scaled_trajectory):
                if i-1 < len(traj):
                    pygame.draw.circle(screen, (169, 131, 7), traj[i - 1], 5) # Dark yellow agents
                    text = font.render(str(self.goals[j]), True, (0, 0, 0))
                    screen.blit(text, (traj[i - 1][0] - text.get_width() // 2, traj[i - 1][1] - 20))

                    comm = self.communications_traj[i-1][j]
                    comm_text = font.render(str(int(comm)), True, (72,72,72))
                    screen.blit(comm_text, (traj[i-1][0] - comm_text.get_width() // 2, traj[i-1][1] + 20))

                if i > 1 and len(traj) >= 1:
                    #pygame.draw.lines(screen, self.colors_map[self.agent_color[j].item()], False, traj[:i], 2)
                    pass
            
            pygame.display.flip()
            clock.tick(10)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        pygame.quit()