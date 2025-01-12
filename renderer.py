from world import World
from agent import Agent

import math
from typing import List
import pygame

class Renderer:
    def __init__(self, world: World):
        self.world = world

        pygame.init()
        self.width, self.height = self.world.get_size()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Project")

    def render(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.renderWorld()

            pygame.display.flip()
            pygame.time.Clock().tick(60)
        
        pygame.quit()

    def renderWorld(self):
        self.screen.fill((0, 0, 0))

        agents = self.world.get_agents()

        for agent in agents:
            self.renderAgent(agent)

    def renderAgent(self, agent: Agent):
        pygame.draw.circle(self.screen, agent.get_color(), agent.get_pos(), 10)

        x, y = agent.get_pos()
        dx = agent.get_direction()[0] - x
        dy = agent.get_direction()[1] - y 

        length = math.sqrt(dx**2 + dy**2)
        if length == 0:
            return  # Avoid division by zero if the player is looking at itself

        # Normalize the direction vector
        dx /= length
        dy /= length

        intersection = None

        # Check intersection with the left boundary (x = 0)
        if dx < 0:  # Looking to the left
            t_left = -x / dx
            y_left = y + t_left * dy
            if 0 <= y_left <= self.height:
                intersection = (0, y_left)

        # Check intersection with the right boundary (x = self.width)
        if dx > 0:  # Looking to the right
            t_right = (self.width - x) / dx
            y_right = y + t_right * dy
            if 0 <= y_right <= self.height:
                intersection = (self.width, y_right)

        # Check intersection with the top boundary (y = 0)
        if dy < 0:  # Looking upwards
            t_top = -y / dy
            x_top = x + t_top * dx
            if 0 <= x_top <= self.width:
                intersection = (x_top, 0)

        # Check intersection with the bottom boundary (y = self.height)
        if dy > 0:  # Looking downwards
            t_bottom = (self.height - y) / dy
            x_bottom = x + t_bottom * dx
            if 0 <= x_bottom <= self.width:
                intersection = (x_bottom, self.height)

        if intersection:
            pygame.draw.line(self.screen, agent.get_color(), (x, y), intersection, 2)