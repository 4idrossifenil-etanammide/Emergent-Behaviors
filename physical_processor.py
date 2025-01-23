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
    def __init__(self, config):
        super(PhysicalProcessor, self).__init__()
        
        self.physical_processor = nn.Sequential(
                nn.Linear(10, config["hidden_size"]),
                nn.Dropout(0.1),
                nn.ELU(),
                nn.Linear(config["hidden_size"], 256)
            )

    def forward(self, x, rmatrix):

        x = self.rotate(x, rmatrix)

        batch, num, dim = x.shape
        x = x.reshape(batch * num, dim).float()

        x = self.physical_processor(x)

        x = x.reshape(batch, num, -1)
        return x
    
    def rotate(self, x, rmatrix):
        updated_x = x.clone()

        updated_x[:, :, :2] = updated_x[:, :, :2] @ rmatrix
        updated_x[:, :, 4:6] = updated_x[:, :, 4:6] @ rmatrix

        x = updated_x

        return x
    
        
