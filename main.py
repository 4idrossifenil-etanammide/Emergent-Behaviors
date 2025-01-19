from world import World
from physical_processor import Agent


import random
import json

def load_config(path: str  = "config.json"):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def main():
    config = load_config()

    world = World(config) 

if __name__ == "__main__":
    main()