import torch.nn as nn

# In a nutshell, the agents have 10 different values defining their state:
# - pos -> 2
# - velocity -> 2
# - gaze -> 2
# - color -> 3
# - shape -> 1
# Using same logic, landmarks have just 6 (no gaze nor velocity)
class PhysicialProcessor(nn.Module):
    def __init__(self, config):
        super(PhysicialProcessor, self).__init__()
        self.agent_physical_processor = nn.Sequential(
                nn.Linear(10, config["hidden_size"]),
                nn.Dropout(0.1),
                nn.ELU(),
                nn.Linear(config["hidden_size"], 256)
            )

        self.landmark_physical_processor = nn.Sequential(
                nn.Linear(6, config["hidden_size"]),
                nn.Dropout(0.1),
                nn.ELU(),
                nn.Linear(config["hidden_size"], 256)
            )

    def forward(self, x):
        if x.shape[-1] == 10: # if it is an agent
            return self.agent_physical_processor(x)
        else: # If it is a landmark
            return self.landmark_physical_processor(x)
        
