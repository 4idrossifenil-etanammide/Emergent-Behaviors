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

    def forward(self, x, rmatrix, agentpos):

        x = self.rotate(x, rmatrix, agentpos)

        batch, num, dim = x.shape
        x = x.reshape(batch * num, dim).float()

        x = self.physical_processor(x)

        x = x.reshape(batch, num, -1)
        return x
    
    def rotate(self, x, rmatrix, agentpos):
        # x: B x NA x 10
        #rmatrix: B x 2 x 2
        #agentpos: B x 2

        updated_x = x.clone()
        agentpos = agentpos.unsqueeze(1)
        rmatrix = rmatrix.unsqueeze(1)

        updated_x[:, :, :2] = torch.matmul(rmatrix,(updated_x[:, :, :2] - agentpos).unsqueeze(3)).squeeze(3)
        updated_x[:, :, 4:6] = torch.matmul(rmatrix,(updated_x[:, :, 4:6] -agentpos).unsqueeze(3)).squeeze(3)

        x = updated_x
        return x
    
        
