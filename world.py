from physical_processor import PhysicalProcessor
from utterance_processor import UtteranceProcessor
from softmax_pooling import SoftmaxPooling
from action_processor import ActionProcessor
from math import cos, sin

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

        self.device = device

        self.physical_processor = PhysicalProcessor(config["physical_processor"]).to(self.device)
        self.utterance_processor = UtteranceProcessor(config["utterance_processor"], self.memory_size, self.vocab_size, self.num_agents).to(self.device)
        self.action_processor = ActionProcessor(config["action_processor"], self.memory_size, self.num_landmarks, self.vocab_size).to(self.device)

    def reset(self):
        # shape: (batch_size, num_agents, 10)
        self.agents, self.rotation_matrices = self.create_agents_batch()
        self.agents = self.agents.to(self.device)
        self.rotation_matrices = self.rotation_matrices.to(self.device)
        # shape: (batch_size, num_landmarks, 6) or (batch_size, num_landmarks, 10) 
        self.landmarks = self.create_landmarks_batch().to(self.device)
        # shape: (batch_size, num_agents, 2)
        self.goals = self.assign_goals().to(self.device)
        self.utterance = torch.zeros((self.batch_size, self.num_agents, self.vocab_size)).to(self.device)
        self.utterance_memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size)).to(self.device)
        self.final_memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size)).to(self.device)

    # creates batch_size*num_agents agents, giving them:
    # Random position: (width, width)
    # Zero velocity: (0,0)
    # Random gaze: (width,width)
    # Random color: (255,255,255)
    # Random shape: (num_shapes)
    # and, for each agent, generates the individual rotation matrix
    def create_agents_batch(self):

        pos = torch.randint(0, self.width, (self.batch_size, self.num_agents, 2))
        velocity = torch.zeros((self.batch_size, self.num_agents, 2))
        gaze = torch.randint(0, self.width, (self.batch_size, self.num_agents, 2))
        color = torch.randint(0, 256, (self.batch_size, self.num_agents, 3))
        shape = torch.randint(0, self.num_shapes, (self.batch_size, self.num_agents, 1))
        
        radians = torch.rand(self.batch_size * self.num_agents ) * 2 * torch.pi
        sines = torch.sin(radians)
        cosines = torch.cos(radians)
        rotation_matrices = torch.stack([
            torch.stack([cosines, -sines], dim=1),
            torch.stack([sines, cosines], dim=1)
                        ], dim=1)
        rotation_matrices = rotation_matrices.view(self.batch_size, self.num_agents, 2, 2)

        agents = torch.cat((pos, velocity, gaze, color, shape), dim=2)

        return agents, rotation_matrices
    
    # creates an batch_size*num_landmarks landmarks, giving them:
    # Random position: (width, width)
    # Random color: (255,255,255)
    # Random shape: (num_shapes)
    # if dummyValues is true, also gives them:
    # Zero velocity: (0,0)
    # negative gaze: (-1,-1)
    def create_landmarks_batch(self, dummyValues = True):
        pos = torch.randint(0, self.width, (self.batch_size, self.num_landmarks, 2))
        color = torch.randint(0, 256, (self.batch_size, self.num_landmarks, 3))
        shapes = torch.randint(0, self.num_shapes, (self.batch_size, self.num_landmarks, 1))

        if dummyValues:
            velocity = torch.zeros((self.batch_size, self.num_landmarks, 2))
            gaze = torch.tensor([-1,-1]).repeat(self.batch_size, self.num_landmarks, 1)
            landmarks = torch.cat((pos, velocity, gaze, color, shapes), dim=2)
            return landmarks
        
        landmarks = torch.cat((pos, color, shapes), dim=2)
        return landmarks

    #Assign goals to the agent such that goals are not contradicting.
    #TODO Check if the goals are not contradicting each other
    def assign_goals(self):
        goals = torch.zeros((self.batch_size, self.num_agents, 4))
        for b in range(self.batch_size):
            agents_to_assign = set([i for i in range(self.num_agents)])
            landmarks_to_assign = set([i for i in range(self.num_landmarks)])
            for agent in range(self.num_agents):
                goal_type = random.choice([0,1,2]) # 0: do nothing, 1: go to, 2: look at

                target_agent = random.sample(sorted(agents_to_assign), 1)[0]
                agents_to_assign.remove(target_agent)
                #TODO would like to have agents that have to go to somewhere but also lookat something

                if goal_type == 0:
                    pos_x, pos_y = self.agents[b, target_agent, :2]
                else:
                    target_landmark = random.sample(sorted(landmarks_to_assign), 1)[0] # Multiple agents can have same landmark as objective, so we don't remove it from the set
                    pos_x, pos_y = self.landmarks[b, target_landmark, :2]

                goals[b, agent, :] = torch.tensor([goal_type, target_agent, pos_x, pos_y])

        return goals
    
    def forward(self):
        total_cost = 0
        history = {
            "initial_agents": {
                "positions": self.agents[:, :, :2].clone(),
                "colors": self.agents[:, :, 6:9].clone(),
                "shapes": self.agents[:, :, 9].clone(),
                "gaze": self.agents[:, :, 4:6].clone()
            },
            "initial_landmarks": {
                "positions": self.landmarks[:, :, :2].clone(),
                "colors": self.landmarks[:, :, 6:9].clone(),
                "shapes": self.landmarks[:, :, 9].clone()
            },
            "agents": [
                {
                    "positions": [],
                    "gaze": [],
                    "utterances": []
                } for _ in range(self.num_agents)
            ]
        }

        for _ in range(self.timesteps):

            all_agents_physical_features = []
            all_agents_utterance_fetures = []
            #compute the physical representation rotated
            for i in range(self.num_agents):
                #for every agent i in every batch, calculate the physical features of agents and landmarks
                #using its own rotation matrix and its position for the rotation
                agent_physical_features = self.physical_processor(self.agents, self.rotation_matrices[:,i], self.agents[:,i,:2])
                landmark_physical_features = self.physical_processor(self.landmarks, self.rotation_matrices[:,i], self.agents[:,i,:2])
                physical_features = torch.cat((agent_physical_features, landmark_physical_features), dim=1)
                physical_features = SoftmaxPooling(dim=1)(physical_features)

                #print("utterance memory: ",self.utterance_memory.shape)
                utterance_features, goal_pred = self.utterance_processor(self.utterance, self.utterance_memory[:,i,:])
                utterance_features = SoftmaxPooling(dim=1)(utterance_features)

                #update the memory with the aggregated utterance features
                new_mem = self.utterance_processor.mem_update(utterance_features, self.utterance_memory[:,i,:])
                self.utterance_memory = self.utterance_memory.clone()
                self.utterance_memory[:, i, :] = new_mem

                all_agents_physical_features.append(physical_features.unsqueeze(1))
                all_agents_utterance_fetures.append(utterance_features.unsqueeze(1))

            all_agents_physical_features = torch.cat(all_agents_physical_features, dim=1)
            all_agents_utterance_fetures = torch.cat(all_agents_utterance_fetures, dim=1)

            #is this necessary? might be ideal to do all of the agents at once?
            # I, Tommaso Leonardi, 1914546, still need to thoroughly check this in light of all our updates
            for agent_idx in range(self.num_agents):
                private_goal = self.goals[:, agent_idx, :] 
                private_memory = self.final_memory[:, agent_idx, :]
                physical_features = all_agents_physical_features[:, agent_idx, :]

                v, gaze, utterance, new_memory = self.action_processor(private_goal, private_memory, physical_features, utterance_features)

                #print("DEBUG")

                updated_final_memory = self.final_memory.clone()
                updated_final_memory[:, agent_idx, :] = new_memory
                self.final_memory = updated_final_memory

                updated_utterance = self.utterance.clone()
                updated_utterance[:, agent_idx, :] = utterance
                self.utterance = updated_utterance

                # Transition dynamics
                updated_agents = self.agents.clone()
                updated_agents[:, agent_idx, :2] = updated_agents[:, agent_idx, :2] + v * self.delta_t  # Position
                updated_agents[:, agent_idx, 2:4] = updated_agents[:, agent_idx, 2:4] * self.damping + v * self.delta_t  # Velocity
                updated_agents[:, agent_idx, 4:6] = gaze  # Gaze
                self.agents = updated_agents

                # Record history
                history["agents"][agent_idx]["positions"].append(updated_agents[:, agent_idx, :2].clone())
                history["agents"][agent_idx]["gaze"].append(gaze.clone())
                history["agents"][agent_idx]["utterances"].append(utterance.clone())

            cost = self.compute_cost(goal_pred)
            total_cost += cost
            #print("cost: ", cost)

        return total_cost, history

    #Computes the cost, by summing the joint goal distance, the auxiliary prediction cost
    # and the 
    def compute_cost(self, goal_pred):
        near_cost = self.compute_near_cost()

        #TODO We should have n**2 predictions, not n predictions
        prediction_cost = torch.norm(goal_pred - self.goals).sum()

        #shouldn't symbols be counted also for previous iterations?
        symbol_counts = self.utterance.sum(dim=(0, 1))  # shape: (vocab_size)
        total_count = symbol_counts.sum()

        probs = symbol_counts / (0.1 + total_count - 1) # 0.1 should be alpha. What's the correct value? TODO
        log_probs = torch.log(probs + 1e-10)

        utterance_cost = (symbol_counts * log_probs).sum()

        return near_cost + prediction_cost - utterance_cost
    

    def compute_near_cost(self):
        near_cost = 0
        for b in range(self.batch_size):
            for agent in range(self.num_agents):
                goal_type = self.goals[b, agent, 0]
                target_agent = self.goals[b, agent, 1].long()
                pos_x, pos_y = self.goals[b, agent, 2], self.goals[b, agent, 3]
                
                if goal_type == 2:
                    gaze_x, gaze_y = self.agents[b, target_agent, 4], self.agents[b, target_agent, 5]
                    near_cost += torch.norm(torch.tensor([gaze_x - pos_x, gaze_y - pos_y]))
                else:
                    target_pos_x, target_pos_y = self.agents[b, target_agent, :2]
                    near_cost += torch.norm(torch.tensor([target_pos_x - pos_x, target_pos_y - pos_y]))
        
        return near_cost


