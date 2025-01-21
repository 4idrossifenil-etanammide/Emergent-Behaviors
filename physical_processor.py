import torch.nn as nn

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
        
        #>TODO seconda interpreatazione con rete unica

    def forward(self, x):

        batch, num, dim = x.shape
        x = x.reshape(batch * num, dim).float()

        if dim == 10: # if it is an agent
            x = self.agent_physical_processor(x)
        else: # If it is a landmark
            x = self.landmark_physical_processor(x)

        x = x.reshape(batch, num, -1)
        return x
        
