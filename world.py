from typing import Tuple
from typing import List

from agent import Agent

import torch

class World:
    def __init__(self, size: torch.Tensor, config: dict):

        assert len(size) == 2, "Size must be a 2D tensor!"

        self.size = size 

        self.num_entities = config["num_agents"] + config["num_landmarks"]

    def get_size(self):
        return self.size