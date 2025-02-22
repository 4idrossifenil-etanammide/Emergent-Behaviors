from world import World

import json

import torch

torch.autograd.set_detect_anomaly(True)

def load_config(path: str  = "config.json"):
    with open(path, "r") as f:
        config = json.load(f)
    return config

def main(render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config()

    world = World(config, device)
    optimizer = torch.optim.Adam(world.parameters(), lr=1E-3)
    epochs = 1000


    for epoch in range(1, epochs):
        world.reset()
        optimizer.zero_grad()
        loss, history = world()

        loss.backward()
        optimizer.step()

        print(f"[EPOCH: {epoch}] Loss: {loss.item()}")
        
    

if __name__ == "__main__":
    main()