from typing import Tuple

class Agent:
    def __init__(self, pos: Tuple[int, int], color: Tuple[int, int, int], direction: Tuple[int, int]):
        self.pos = pos
        self.color = color
        self.direction = direction

    def get_pos(self):
        return self.pos
    
    def get_color(self):
        return self.color
    
    def get_direction(self):
        return self.direction
