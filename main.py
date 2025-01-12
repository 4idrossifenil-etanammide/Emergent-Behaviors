from world import World
from agent import Agent
from renderer import Renderer

import random

def main():
    agents = []
    n_agent = random.randint(1, 10) 
    positions = [(random.randint(0, 800), random.randint(0, 600)) for _ in range(n_agent)]  
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_agent)]
    directions = [positions[(i+1)%n_agent] for i in range(n_agent)]
    agents = [Agent(positions[i], colors[i], directions[i]) for i in range(n_agent)]

    world = World((800, 600), agents=agents)

    renderer = Renderer(world)
    renderer.render()

if __name__ == "__main__":
    main()