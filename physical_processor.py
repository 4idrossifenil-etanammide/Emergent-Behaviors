import torch.nn as nn
from softmax_pooling import SoftmaxPooling

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
        
        # Perform pooling along the number of entities in that particular batch
        # Given that it can change across different worlds, this makes the physical features all equal
        self.softmax_pooling = SoftmaxPooling(dim=1) 

    def forward(self, x):

        batch, num, dim = x.shape
        x = x.reshape(batch * num, dim)

        if dim == 10: # if it is an agent
            x = self.agent_physical_processor(x)
        else: # If it is a landmark
            x = self.landmark_physical_processor(x)

        x = x.reshape(batch, num, dim)

        # shape: (batch, dim)
        x = self.softmax_pooling(x)
        return x
        
