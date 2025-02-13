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
        self.num_landmarks = 3
        self.episode_length = 100
        self.step_count = 0
        
        # Define color landmarks (RGB)
        self.landmark_colors = np.array([
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1]   # Blue
        ])
        
        # Define action spaces (movement + communication)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, 2 + VOCAB_SIZE), dtype=np.float32)
        self.obs_size = 2 + (3*2) + 3 + VOCAB_SIZE  # vel + 3 landmarks * 2D + target color + comm
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, self.obs_size), dtype=np.float32)
        self.agent_positions = None
        self.agent_velocities = None
        self.landmark_positions = None
        self.communications = np.zeros((2, VOCAB_SIZE))


    def reset(self, seed=None, options=None):
        # Initialize agents and landmarks randomly
        self.agent_positions = np.random.uniform(-1, 1, size=(2, 2))
        self.agent_velocities = np.zeros((2, 2))
        self.landmark_positions = np.random.uniform(-1, 1, size=(3, 2))
        self.step_count = 0
        
        # Reset communications
        self.communications = np.zeros((2, VOCAB_SIZE))
        self.goals = np.random.choice(self.num_landmarks, self.num_agents)

        self.trajectories = [self.agent_positions.copy()]
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        obs = np.zeros((2, self.obs_size))
        
        for i in range(self.num_agents):
            # Velocity (2D)
            velocity = self.agent_velocities[i]
            
            # Relative positions to landmarks (3 landmarks * 2D)
            rel_positions = (self.landmark_positions - self.agent_positions[i]).flatten()
            
            # Target color (3D)
            target_color = self.landmark_colors[self.goals[i]]
            
            # Received communication (3D)
            # TODO Change this to the other agent comm or use both
            comm = self.communications[i]
            
            # Concatenate observations
            obs[i] = np.concatenate([
                velocity,
                rel_positions,
                target_color,
                comm
            ])
            
        return obs
    
    def step(self, actions):
        self.step_count += 1
        
        # Update positions
        self.agent_positions += actions[:, :2] * 0.1
        self.communications = actions[:, 2:]
        
        self.trajectories.append(self.agent_positions.copy())

        # Calculate rewards
        rewards = np.zeros((2))
        for i in range(self.num_agents):
            target_pos = self.landmark_positions[self.goals[i]]
            distance = np.linalg.norm(self.agent_positions[i] - target_pos)
            rewards[i] = -distance  # Reward is negative distance to target
        
        # Check termination
        terminated = self.step_count >= self.episode_length
        truncated = False
        
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
                pygame.draw.circle(screen, [int(x) for x in self.landmark_colors[k] * 255], landmark, 10)
                text = font.render(str(k), True, (0,0,0))
                screen.blit(text, (landmark[0] - text.get_width() // 2, landmark[1] - text.get_height() // 2))
            
            # Current position
            for j, traj in enumerate(scaled_trajectory):
                if i-1 < len(traj):
                    pygame.draw.circle(screen, (169, 131, 7), traj[i - 1], 5) # Dark yellow agents
                    text = font.render(str(self.goals[j]), True, (0, 0, 0))
                    screen.blit(text, (traj[i - 1][0] - text.get_width() // 2, traj[i - 1][1] - 20))
                if i > 1 and len(traj) >= 1:
                    #pygame.draw.lines(screen, self.colors_map[self.agent_color[j].item()], False, traj[:i], 2)
                    pass
            
            pygame.display.flip()
            clock.tick(20)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        pygame.quit()