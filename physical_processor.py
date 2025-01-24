import torch.nn as nn
import torch

# In a nutshell, the agents have 10 different values defining their state:
# - pos -> 2
# - velocity -> 2
# - gaze -> 2
# - color -> 3
# - shape -> 1
# Using same logic, landmarks have just 6 (no gaze nor velocity)
class PhysicalProcessor(nn.Module):
    def __init__(self, hidden_size):
        super(PhysicalProcessor, self).__init__()
        
        self.physical_processor = nn.Sequential(
                nn.Linear(10, hidden_size),
                nn.ELU(),
                nn.Linear(hidden_size, 256)
            )

    def forward(self, x):

        batch, num, dim = x.shape
        x = x.reshape(batch * num, dim).float()

        x = self.physical_processor(x)

        x = x.reshape(batch, num, -1)
        return x
    
        
