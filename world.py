from physical_processor import PhysicalProcessor
from utterance_processor import UtteranceProcessor
from softmax_pooling import SoftmaxPooling

import torch
import torch.nn as nn

class World(nn.Module):
    def __init__(self, config: dict):
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

        #create all of the agents and put them in a tensor
        # shape: (batch_size, num_agents, 10)
        self.agents = self.create_agents_batch()
        # shape: (batch_size, num_landmarks, 6)
        self.landmarks = self.create_landmarks_batch()
        # shape: (batch_size, num_agents, 2)
        self.goals = self.assign_goals()
        
        self.utterance = torch.zeros((self.batch_size, self.num_agents, 1))

        self.utterance_memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size))
        self.final_memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size))

        self.physical_processor = PhysicalProcessor(config["physical_processor"])
        self.utterance_processor = UtteranceProcessor(config["utterance_processor"], self.memory_size)

        

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

    #TODO Check if the goals are not contradicting each other
    def assign_goals(self):
        goal_type = torch.randint(0, 3, (self.batch_size, self.num_agents, 1)) # 0 do nothing, 1 look at landmark, 2 move to landmark
        goal_target = torch.randint(0, self.num_landmarks, (self.batch_size, self.num_agents, 1))
        return torch.cat((goal_type, goal_target), dim=2)
    
    def forward(self):
        for timestep in range(self.timesteps):
            #Given that from Figure 3 in the paper the physical features
            #seems to be extracted once for all the agents, I'm doing the same here
            agent_physical_features = self.physical_processor(self.agents)
            landmark_physical_features = self.physical_processor(self.landmarks)
            physical_features = torch.cat((agent_physical_features, landmark_physical_features), dim=1)
            physical_features = SoftmaxPooling(dim=1)(physical_features)

            utterance_features, self.utterance_memory = self.utterance_processor(self.utterance, self.utterance_memory)
            utterance_features = SoftmaxPooling(dim=1)(utterance_features)

            for agent_idx in range(self.num_agents):
                private_goal = self.goals[:, agent_idx:agent_idx + 1, :] 
                private_memory = self.final_memory[:, agent_idx:agent_idx + 1, :]
