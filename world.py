from physical_processor import PhysicalProcessor
from utterance_processor import UtteranceProcessor
from softmax_pooling import SoftmaxPooling
from action_processor import ActionProcessor

import torch
import torch.nn as nn

import random

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
        self.timesteps=1
        self.vocab_size = world_config["vocab_size"]
        self.delta_t= world_config["delta_t"]
        self.damping = world_config["damping"]

        #create all of the agents and put them in a tensor
        # shape: (batch_size, num_agents, 10)
        self.agents = self.create_agents_batch()
        # shape: (batch_size, num_landmarks, 6) or (batch_size, num_landmarks, 10) 
        self.landmarks = self.create_landmarks_batch()
        # shape: (batch_size, num_agents, 2)
        self.goals = self.assign_goals()
        
        self.utterance = torch.zeros((self.batch_size, self.num_agents, self.vocab_size))

        self.utterance_memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size))

        #WHY final? shouldn't it be the same memory it was before?
        self.final_memory = torch.zeros((self.batch_size, self.num_agents, self.memory_size))

        self.physical_processor = PhysicalProcessor(config["physical_processor"])
        self.utterance_processor = UtteranceProcessor(config["utterance_processor"], self.memory_size, self.vocab_size, self.num_agents)
        self.action_processor = ActionProcessor(config["action_processor"], self.memory_size, self.num_landmarks, self.vocab_size)

        

    # creates an batch_size*num_agents agents, giving them:
    # Random position: (width, width)
    # Zero velocity: (0,0)
    # Random gaze: (width,width)
    # Random color: (255,255,255)
    # Random shape: (num_shapes)
    def create_agents_batch(self):

        pos = torch.randint(0, self.width, (self.batch_size, self.num_agents, 2))
        velocity = torch.zeros((self.batch_size, self.num_agents, 2))
        gaze = torch.randint(0, self.width, (self.batch_size, self.num_agents, 2))
        color = torch.randint(0, 256, (self.batch_size, self.num_agents, 3))
        shape = torch.randint(0, self.num_shapes, (self.batch_size, self.num_agents, 1))

        agents = torch.cat((pos, velocity, gaze, color, shape), dim=2)
        return agents
    
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
            print(landmarks.shape)
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

                if goal_type == 0:
                    pos_x, pos_y = self.agents[b, target_agent, :2]
                else:
                    target_landmark = random.sample(sorted(landmarks_to_assign), 1)[0] # Multiple agents can have same landmark as objective, so we don't remove it from the set
                    pos_x, pos_y = self.landmarks[b, target_landmark, :2]

                goals[b, agent, :] = torch.tensor([goal_type, target_agent, pos_x, pos_y])

        return goals
    
    def forward(self):

        total_cost = 0
        for _ in range(self.timesteps):
            #Given that from Figure 3 in the paper the physical features
            #seems to be extracted once for all the agents, I'm doing the same here
            agent_physical_features = self.physical_processor(self.agents)
            landmark_physical_features = self.physical_processor(self.landmarks)
            physical_features = torch.cat((agent_physical_features, landmark_physical_features), dim=1)
            physical_features = SoftmaxPooling(dim=1)(physical_features)

            utterance_features, self.utterance_memory, goal_pred = self.utterance_processor(self.utterance, self.utterance_memory)
            utterance_features = SoftmaxPooling(dim=1)(utterance_features)

            #print("goal prediction shape: ", goal_pred.shape)

            #is this necessary? might be ideal to do all of the agents at once?
            for agent_idx in range(self.num_agents):
                private_goal = self.goals[:, agent_idx, :] 
                private_memory = self.final_memory[:, agent_idx, :]

                v, gaze, utterance, new_memory = self.action_processor(private_goal, private_memory, physical_features, utterance_features)

                self.final_memory[:, agent_idx, :] = new_memory
                self.utterance[:, agent_idx, :] = utterance

                # Transition dynamics
                self.agents[:, agent_idx, :2] += v * self.delta_t # Position
                self.agents[:, agent_idx, 2:4] = self.agents[:, agent_idx, 2:4] * self.damping + v * self.delta_t # Velocity
                self.agents[:, agent_idx, 4:6] = gaze # Gaze

            cost = self.compute_cost(goal_pred)
            total_cost += cost
            #print("cost: ", cost)

        return total_cost

    #Computes the cost, by summing the joint goal distance, the auxiliary prediction cost
    # and the 
    def compute_cost(self, goal_pred):
        #TODO: fix this to correctly pair agents and goal.
        #Missing check for which goal has to be done (go to? look at?)
        near_cost = torch.norm(self.agents[:, :, :2] - self.goals[:, :, 2:], dim=2).sum()
        prediction_cost = torch.norm(goal_pred - self.goals).sum()

        #shouldn't symbols be counted also for previous iterations?
        symbol_counts = self.utterance.sum(dim=(0, 1))  # shape: (vocab_size)
        total_count = symbol_counts.sum()

        probs = symbol_counts / (0.1 + total_count - 1) # 0.1 should be alpha. What's the correct value? TODO
        log_probs = torch.log(probs + 1e-10)

        #Note: this has to be maximized, so subtract it?
        utterance_cost = (symbol_counts * log_probs).sum()

        return near_cost + prediction_cost + utterance_cost

