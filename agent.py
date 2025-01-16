from typing import Tuple

#Each agent has a position, defined as a pair of int (coordinates), a color, defined
# as a triple of int (RGB values), and a direction, defined as a pair of int (coordinates)
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
