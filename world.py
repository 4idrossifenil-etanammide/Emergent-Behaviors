from typing import Tuple
from typing import List

from agent import Agent

import torch
import random

class World:
    def __init__(self, config: dict):

        self.height = config["height"]
        self.width = config["width"]
        self.num_agents = config["num_agents"]
        self.num_landmarks = config["num_landmarks"]

        #create all of the agents and put them in a tensor
        # numAgents x ( dimPosition + dimVelocity + dimGaze + dimColor )
        self.agents = torch.tensor([self.createAgent() for x in range(self.num_agents)]) 
        print(self.agents)

    #creates an agent, giving it a random position, a random color, a random gaze and
    # a velocity of 0
    def createAgent(self):
        pos = [random.randint(0, self.width), random.randint(0, self.height)]
        velocity = [0, 0]
        gaze = [random.randint(0, self.width), random.randint(0, self.height)]
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        return pos + velocity + gaze + color

    def get_size(self):
        return self.size