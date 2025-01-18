from world import World
from agent import Agent
from renderer import Renderer


import random
import json

def load_config(path: str  = "config.json"):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def main():
    config = load_config()

    world = World((800, 600), config["world"]) 

#     renderer = Renderer(world)
#     renderer.render()

if __name__ == "__main__":
    main()