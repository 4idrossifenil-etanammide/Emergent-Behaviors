from typing import Tuple
from typing import List

from agent import Agent

import torch
import torch.nn as nn

class World(nn.Module):
    def __init__(self, config: dict):

        self.height = config["height"]
        self.width = config["width"]
        self.num_agents = config["num_agents"]
        self.num_landmarks = config["num_landmarks"]
        self.num_shapes = config["num_shapes"]
        self.batch_size = config["batch_size"]
        self.memory_size = config["memory_size"]
        self.timesteps = config["timesteps"]

        #create all of the agents and put them in a tensor
        # shape: (batch_size, num_agents, 7)
        self.agents = self.create_agents_batch()
        # shape: (batch_size, num_landmarks, 6)
        self.landmarks = self.create_landmarks_batch()
        # shape: (batch_size, num_agents, 2)
        self.goals = self.assign_goals()
        
        self.communication = torch.zeros((self.batch_size, self.num_agents, 1))

        self.memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size))

    #creates an agent, giving it a random position, a random color, a random gaze and
    # a velocity of 0
    def create_agents_batch(self):
        pos = torch.randint(0, self.width, (self.batch_size, self.num_agents, 2))
        velocity = torch.zeros((self.batch_size, self.num_agents, 2))
        gaze = torch.randint(0, self.width, (self.batch_size, self.num_agents, 2))
        color = torch.randint(0, 256, (self.batch_size, self.num_agents, 3))
        shape = torch.randint(0, self.num_shapes, (self.batch_size, self.num_agents, 1))

        agents = torch.cat((pos, velocity, gaze, color, shape), dim=2)
        return agents
    
    def create_landmarks_batch(self):
        pos = torch.randint(0, self.width, (self.batch_size, self.num_landmarks, 2))
        color = torch.randint(0, 256, (self.batch_size, self.num_landmarks, 3))
        shapes = torch.randint(0, self.num_shapes, (self.batch_size, self.num_landmarks, 1))

        landmarks = torch.cat((pos, color, shapes), dim=2)
        return landmarks

    def assign_goals(self):
        goal_type = torch.randint(0, 3, (self.batch_size, self.num_agents, 1)) # 0 do nothing, 1 look at landmark, 2 move to landmark
        goal_target = torch.randint(0, self.num_landmarks, (self.batch_size, self.num_agents, 1))
        return torch.cat((goal_type, goal_target), dim=2)
    
    def get_observation(self, agent_idx):
        all_agents = self.agents.clone()  
        all_landmarks = self.landmarks.clone()

        # Instead of selecting directly using agent_idx, we select 
        # using the agent_idx and agent_idx + 1 to avoid collapsing a dimension
        private_goal = self.goals[:, agent_idx:agent_idx + 1, :] 
        private_memory = self.memory[:, agent_idx:agent_idx + 1, :]  

        observation = {
            "physical_state": {"agents": all_agents, "landmarks": all_landmarks},
            "communication": self.communication.clone(),
            "private_info": {"goal": private_goal, "memory": private_memory},
        }
        return observation
    
    def forward(self):
        for timestep in self.timesteps:
            #Given that from Figure 3 in the paper the physical features
            #seems to be extracted once for all the agents, I'm doing the same here
            for agent_idx in range(self.num_agents):
                observation = self.get_observation(agent_idx)
