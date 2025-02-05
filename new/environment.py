import torch
from torch import nn

import pygame
import random
import gymnasium as gym

WORLD_SIZE = 2.0
STEP_SIZE = 0.3
DAMPING = 0.5

MAX_STEPS = 50

NUM_COLORS = 8
NUM_SHAPES = 8

MAX_AGENTS = 4
MAX_LANDMARKS = 4

VOCAB_SIZE = 10
MEMORY_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

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

        all_colors = torch.randperm(NUM_COLORS)[:self.n_agents + self.n_landmarks]
        all_shapes = torch.randperm(NUM_SHAPES)[:self.n_agents + self.n_landmarks]

        self.agent_pos = (torch.rand((self.n_agents, 2)) * 2 - 1)
        self.agent_color = all_colors[:self.n_agents].view(-1, 1)
        self.agent_shape =  all_shapes[:self.n_agents].view(-1, 1)

        self.initial_pos = self.agent_pos.clone()

        self.landmark_pos = (torch.rand((self.n_landmarks, 2)) * 2 - 1)
        self.landmark_color = all_colors[self.n_agents:].view(-1, 1)
        self.landmark_shape =  all_shapes[self.n_agents:].view(-1, 1)

        self.utterances = torch.zeros((self.n_agents, VOCAB_SIZE))
        self.memories = torch.zeros((self.n_agents, MEMORY_SIZE)).to(device)

        #task initialization
        tasks = torch.randint(0, 2, (self.n_agents, 1)) # 0 - GOTO; 1 - DO NOTHING
        self.targets = torch.randint(0, self.n_landmarks, (self.n_agents,1))
        self.targets[tasks == 1] = -1
        flat_targets = self.targets.view(-1)
        goal_pos = self.landmark_pos[flat_targets.long()]
        goal_pos[flat_targets == -1] = self.initial_pos[flat_targets == -1]
        self.goals = torch.cat([tasks, goal_pos], dim = 1) #NB: the goals are the POSITIONS of the goal

        self.velocities = torch.zeros((self.n_agents, 2))

        self.current_step = 0

        if self.render_env:
            self.states_traj = [self.agent_pos.clone()]
            self.utterances_traj = [self.utterances.clone()]

        self.observation_space = gym.spaces.Box(-1, 1, (self.n_agents, 5))
        self.action_space = gym.spaces.Box(-.1, .1, (self.n_agents, 2))
        
        return self._get_state(), {}

    # TODO Implement random rotation matrix
    def _get_state(self):
        pos = []
        for agent_idx in range(self.n_agents):
            pos.append(
                torch.cat([
                    torch.cat([self.agent_pos - self.agent_pos[agent_idx, :], self.landmark_pos - self.agent_pos[agent_idx, :]], dim = 0), # shape [n_agent + n_landmark, 2]
                    torch.cat([self.velocities, torch.zeros((self.n_landmarks, 2))], dim = 0), # shape [n_agent + n_landmark, 2]
                    torch.cat([self.agent_color, self.landmark_color], dim = 0), # shape [n_agent + n_landmark , 1]
                    torch.cat([self.agent_shape, self.landmark_shape], dim = 0) # shape [n_agent + n_landmark, 1]
                ], dim = 1).unsqueeze(0) # shape [n_agent + n_landmark, 6]
            )

        physical = torch.cat(pos, dim=0) # shape [n_agent, n_agent + n_landmark, 6]
        state = {
            "physical": physical,
            "utterances": self.utterances,
            "memories": self.memories,
            "tasks": self.goals.float()
        }
        return state

    def step(self, x):
        actions, utterances, delta_memories = x
        self.utterances = utterances
        self.memories = nn.Tanh()(self.memories + delta_memories + 1E-8)  #why the tanh?

        # Transition dynamics. STEP SIZE is delta_t and DAMPING is damping factor
        self.agent_pos += self.velocities * STEP_SIZE
        self.velocities = self.velocities * DAMPING + actions * STEP_SIZE

        self.current_step += 1
        
        if self.render_env:
            self.states_traj.append(self.agent_pos.clone())
            self.utterances_traj.append(self.utterances.clone())

        distances = torch.norm(self.agent_pos - self.goals[:, 1:], dim=1)

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
                utterance = torch.argmax(self.utterances_traj[i-1][j]).item()
                utterance_text = font.render(str(utterance), True, (255, 0, 0))
                screen.blit(utterance_text, (traj[i - 1][0] - utterance_text.get_width() // 2, traj[i - 1][1] + 20))
                if i-1 < len(traj):
                    pygame.draw.circle(screen, self.colors_map[self.agent_color[j].item()], traj[i - 1], 5)
                    if self.goals[j][0].item() == 0:
                        goal_index = self.targets[j].item()
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