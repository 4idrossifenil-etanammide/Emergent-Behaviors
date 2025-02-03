import torch
import pygame
import random
import gymnasium as gym

WORLD_SIZE = 2.0
STEP_SIZE = 0.1
MAX_STEPS = 50

NUM_COLORS = 5
NUM_SHAPES = 5

MAX_AGENTS = 4
MAX_LANDMARKS = 4

class EmergentEnv(gym.Env):
    def __init__(self, render_env = False):
        self.n_agents = 0
        self.n_landmarks = 0
        self.observation_space = gym.spaces.Box(-1, 1, (self.n_agents, 5))
        self.action_space = gym.spaces.Box(-.1, .1, (self.n_agents, 2))
        
        self.render_env = render_env

        self.colors_map = {i: tuple(torch.randint(0, 256, (3,)).tolist()) for i in range(NUM_COLORS)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.n_agents = random.randint(1, MAX_AGENTS)
        self.n_landmarks = random.randint(1, MAX_LANDMARKS)

        self.agent_pos = (torch.rand((self.n_agents, 2)) * 2 - 1)
        self.agent_color = torch.randint(0, NUM_COLORS, (self.n_agents, 1))
        self.agent_shape = torch.randint(0, NUM_SHAPES, (self.n_agents, 1))

        self.initial_pos = self.agent_pos.clone()

        self.landmark_pos = (torch.rand((self.n_landmarks, 2)) * 2 - 1)
        self.landmark_color = torch.randint(0, NUM_COLORS, (self.n_landmarks, 1))
        self.landmark_shape = torch.randint(0, NUM_SHAPES, (self.n_landmarks, 1))

        self.tasks = torch.randint(0, 2, (self.n_agents, 1)) # 0 - GOTO; 1 - DO NOTHING
        self.goals = torch.randint(0, self.n_landmarks, (self.n_agents, 1))
        self.goals[self.tasks == 1] = -1

        self.current_step = 0

        if self.render_env:
            self.states_traj = [self.agent_pos.clone()]

        self.observation_space = gym.spaces.Box(-1, 1, (self.n_agents, 5))
        self.action_space = gym.spaces.Box(-.1, .1, (self.n_agents, 2))
        
        return self._get_state(), {}

    def _get_state(self):
        goals = self.goals.view(-1)
        goal_pos = self.landmark_pos[goals.long()]
        do_nothing_mask = goals == -1
        goal_pos[do_nothing_mask] = self.initial_pos[do_nothing_mask]  # No movement for DO NOTHING
        return torch.cat([
            goal_pos - self.agent_pos, # relative to goal position
            self.agent_pos,  #  absolute position of agents
            self.tasks.float()  # Adding task information
        ], dim=-1)

    def step(self, actions):
        actions = actions * STEP_SIZE
        self.agent_pos += actions
        #self.agent_pos = torch.clamp(self.agent_pos, -1, 1)
        self.current_step += 1
        
        if self.render_env:
            self.states_traj.append(self.agent_pos.clone())

        goals = self.goals.view(-1)
        goal_pos = self.landmark_pos[goals.long()]
        goal_pos[goals == -1] = self.initial_pos[goals == -1]

        distances = torch.norm(self.agent_pos - goal_pos, dim=1)

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
        font = pygame.font.SysFont(None, 24)
        
        def scale(pos):
            return int((pos[0] + 1) * 200), int((pos[1] + 1) * 200)
        
        landmarks_scaled = [scale(self.landmark_pos[i]) for i in range(self.n_landmarks)]
        trajectory = [[scale(p[i]) for p in self.states_traj] for i in range(self.n_agents)]

        for i in range(1, len(self.states_traj)+1):
            screen.fill((255, 255, 255))
            for k, landmark in enumerate(landmarks_scaled):
                pygame.draw.circle(screen, self.colors_map[self.landmark_color[k].item()], landmark, 10)
                text = font.render(str(k), True, (0,0,0))
                screen.blit(text, (landmark[0] - text.get_width() // 2, landmark[1] - text.get_height() // 2))
            
            # Current position
            for j, traj in enumerate(trajectory):
                if i-1 < len(traj):
                    pygame.draw.circle(screen, self.colors_map[self.agent_color[j].item()], traj[i - 1], 5)
                    if self.tasks[j].item() == 0:
                        goal_index = self.goals[j].item()
                        text = font.render(str(goal_index), True, (0, 0, 0))
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