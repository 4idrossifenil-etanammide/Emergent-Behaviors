from typing import Tuple
from typing import List

from agent import Agent

class World:
    def __init__(self, size: Tuple[int, int], agents: List[Agent]):
        self.size = size
        self.agents = agents

    def get_size(self):
        return self.size

    def get_agents(self):
        return self.agents 