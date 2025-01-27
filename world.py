from physical_processor import PhysicalProcessor
from utterance_processor import UtteranceProcessor
from softmax_pooling import SoftmaxPooling
from action_processor import ActionProcessor
from math import cos, sin
from history import History

import torch
import torch.nn as nn

import random

class World(nn.Module):
    def __init__(self, config: dict, device):
        super(World, self).__init__()

        world_config = config["world"]

        self.height = world_config["height"]
        self.width = world_config["width"]
        self.num_agents = world_config["num_agents"]
        self.num_landmarks = world_config["num_landmarks"]
        self.num_shapes = world_config["num_shapes"]
        self.batch_size = world_config["batch_size"]
        self.memory_size = world_config["memory_size"]
        self.timesteps = world_config["timesteps"]
        self.vocab_size = world_config["vocab_size"]
        self.delta_t= world_config["delta_t"]
        self.damping = world_config["damping"]

        self.hidden_size = world_config["hidden_size"]

        self.device = device

        self.physical_processor = PhysicalProcessor(self.hidden_size).to(self.device)
        self.utterance_processor = UtteranceProcessor(self.hidden_size, self.memory_size, self.vocab_size).to(self.device)
        self.action_processor = ActionProcessor(self.hidden_size, self.memory_size, self.vocab_size, self.width).to(self.device)

    def reset(self):
        self.agents, self.rotation_matrices = self.create_agents_batch()
        self.agents = self.agents.to(self.device)
        self.rotation_matrices = self.rotation_matrices.to(self.device)
        self.landmarks = self.create_landmarks_batch().to(self.device)
        self.goals = self.assign_goals().to(self.device)
        self.utterance = torch.zeros((self.batch_size, self.num_agents, self.vocab_size)).to(self.device)
        self.utterance_memory = torch.zeros((self.batch_size, self.num_agents, self.num_agents, self.memory_size)).to(self.device)
        self.final_memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size)).to(self.device)

    def create_agents_batch(self):
        
        radians = torch.rand(self.batch_size * self.num_agents) * 2 * torch.pi
        sines = torch.sin(radians)
        cosines = torch.cos(radians)
        rotation_matrices = torch.stack([
            torch.stack([cosines, -sines], dim=1),
            torch.stack([sines, cosines], dim=1)
                        ], dim=1)
        rotation_matrices = rotation_matrices.view(self.batch_size, self.num_agents, 2, 2)

        pos = torch.tensor([[[100,100],[500,100],[500, 500]]]).repeat(self.batch_size, 1, 1)
        velocity = torch.zeros((self.batch_size, self.num_agents, 2)) 
        gaze = torch.tensor([[[150,150],[250,250],[350, 350]]]).repeat(self.batch_size, 1, 1)
        color = torch.tensor([[[255,0,0],[0,255,0],[0, 0, 255]]]).repeat(self.batch_size, 1, 1)
        shape = torch.tensor([[[1],[2],[3]]]).repeat(self.batch_size, 1, 1)

        agents = torch.cat((pos, velocity, gaze, color, shape), dim=2)

        self.initial_positions = pos.clone().to(self.device)
        self.initial_gazes = gaze.clone().to(self.device)
        #return agents, rotation_matrices
        return agents, torch.ones((self.batch_size, self.num_agents, 2, 2))
    
    def create_landmarks_batch(self):
        pos = torch.tensor([[[250,300],[350,300]]]).repeat(self.batch_size, 1, 1)
        color = torch.tensor([[[255,255,0],[0,255,255]]]).repeat(self.batch_size, 1, 1)
        shapes = torch.tensor([[[1],[2]]]).repeat(self.batch_size, 1, 1)
        velocity = torch.zeros((self.batch_size, self.num_landmarks, 2))
        gaze = torch.tensor([-1,-1]).repeat(self.batch_size, self.num_landmarks, 1)
        landmarks = torch.cat((pos, velocity, gaze, color, shapes), dim=2)
        return landmarks

    def assign_goals(self):

        goals = torch.tensor([[[1,1,250,300],   # First agent should make second agent go to 250,300 (first landmark)
                               [2,0,350,300],   # Second agent should make first agent look at 350,300
                               [0,2,500,500]]]) # Third agent should make itself do nothing
        return goals.repeat(self.batch_size, 1, 1)
    
    #return relative position observation for one agent
    def get_observation(self, agent_idx):
        relative_agents_pos = self.agents[:, :, :2] - self.agents[:, agent_idx, :2].unsqueeze(1) # p_j - p_i
        relative_agents_pos = torch.einsum('bij,bkj->bki', self.rotation_matrices[:, agent_idx], relative_agents_pos) # R_i^T (p_j - p_i)
        updated_agents_positions = self.agents.clone()
        updated_agents_positions[:, :, :2] = relative_agents_pos

        relative_landmark_pos = self.landmarks[:, :, :2] - self.agents[:, agent_idx, :2].unsqueeze(1)
        relative_landmark_pos = torch.einsum('bij,bkj->bki', self.rotation_matrices[:, agent_idx], relative_landmark_pos)
        updated_landmark_positions = self.landmarks.clone()
        updated_landmark_positions[:, :, :2] = relative_landmark_pos

        obs = torch.cat((updated_agents_positions, updated_landmark_positions), dim=1)
        return obs

    def forward(self):
        total_cost = 0
        history = History(self.agents, self.landmarks)

        for _ in range(self.timesteps):

            all_agents_physical_features = []
            all_agents_utterance_features = []

            for agent_idx in range(self.num_agents):
                agent_obs = self.get_observation(agent_idx)

                physical_features = self.physical_processor(agent_obs)
                physical_features = SoftmaxPooling(dim=1)(physical_features)

                utterance_features, new_mem = self.utterance_processor(self.utterance, self.utterance_memory[:,agent_idx,:,:])
                utterance_features = SoftmaxPooling(dim=1)(utterance_features)

                updated_utterance_memory = self.utterance_memory.clone()
                updated_utterance_memory[:, agent_idx, :, :] = new_mem
                self.utterance_memory = updated_utterance_memory
                
                all_agents_physical_features.append(physical_features.unsqueeze(1))
                all_agents_utterance_features.append(utterance_features.unsqueeze(1))

            all_agents_physical_features = torch.cat(all_agents_physical_features, dim=1)
            all_agents_utterance_features = torch.cat(all_agents_utterance_features, dim=1)

            for agent_idx in range(self.num_agents):
                private_goal = self.goals[:, agent_idx, :] 
                private_memory = self.final_memory[:, agent_idx, :]
                physical_features = all_agents_physical_features[:, agent_idx, :]
                utterance_features = all_agents_utterance_features[:, agent_idx, :]

                v, gaze, utterance, new_memory = self.action_processor(private_goal, private_memory, physical_features, utterance_features)

                updated_final_memory = self.final_memory.clone()
                updated_final_memory[:, agent_idx, :] = new_memory
                self.final_memory = updated_final_memory

                updated_utterance = self.utterance.clone()
                updated_utterance[:, agent_idx, :] = utterance
                self.utterance = updated_utterance

                # Transition dynamics
                updated_agents = self.agents.clone()
                updated_agents[:, agent_idx, :2] = updated_agents[:, agent_idx, :2] + v# * self.delta_t  # Position
                updated_agents[:, agent_idx, 2:4] = v  # Velocity
                updated_agents[:, agent_idx, 4:6] = updated_agents[:, agent_idx, 4:6] + gaze  # Gaze
                self.agents = updated_agents

                history.update(agent_idx, updated_agents, utterance)

            total_cost += self.compute_near_cost()

        return total_cost, history.get_history()
    

    def compute_near_cost(self):
        near_cost = 0
        for b in range(self.batch_size):
            for agent in range(self.num_agents):
                goal_type = self.goals[b, agent, 0]
                target_agent = self.goals[b, agent, 1].long()
                pos_x, pos_y = self.goals[b, agent, 2], self.goals[b, agent, 3]
                
                if goal_type == 2:
                    gaze_x, gaze_y = self.agents[b, target_agent, 4], self.agents[b, target_agent, 5]
                    near_cost += torch.norm(torch.stack([gaze_x - pos_x, gaze_y - pos_y]))
                    near_cost += torch.norm(self.initial_positions[b, target_agent] - self.agents[b, target_agent, :2])
                else:
                    target_pos_x, target_pos_y = self.agents[b, target_agent, :2]
                    near_cost += torch.norm(torch.stack([target_pos_x - pos_x, target_pos_y - pos_y]))
                    near_cost += torch.norm(self.initial_gazes[b, target_agent] - self.agents[b, target_agent, 4:6])
        
        return near_cost
    



