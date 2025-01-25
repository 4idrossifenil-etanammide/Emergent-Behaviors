import torch.nn as nn
import torch
import torch.nn.functional as F

class ActionProcessor(nn.Module):
    def __init__(self, hidden_size, memory_size, vocab_size):
        super(ActionProcessor, self).__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.vocab_size = vocab_size

        self.goal_embedding = nn.Sequential(
            nn.Linear(4, self.hidden_size//2),
            nn.ELU(),
            nn.Linear(self.hidden_size//2, self.hidden_size)
        )

        self.velocity_decoder = nn.Sequential(
            nn.Linear(self.hidden_size + 512 + self.memory_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, 2)
        )

        self.gaze_decoder = nn.Sequential(
            nn.Linear(self.hidden_size + 512 + self.memory_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, 2)
        )

        self.utterance_decoder = nn.Sequential(
            nn.Linear(self.hidden_size + 512 + self.memory_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.vocab_size)
        )

        self.cell = nn.GRUCell(self.hidden_size + 2 * 256, self.memory_size)

    def forward(self, goal, memory, physical_features, utterance_features):

        goal = self.goal_embedding(goal.float())

        x = torch.cat((goal, physical_features, utterance_features), dim=-1)

        new_memory = self.cell(x, memory)

        velocity = self.velocity_decoder(torch.cat((x, new_memory), dim=-1))
        gaze = self.gaze_decoder(torch.cat((x, new_memory), dim=-1))
        utterance_logits = self.utterance_decoder(torch.cat((x, new_memory), dim=-1))
        
        utterance = F.gumbel_softmax(utterance_logits, tau=1.0, hard=True)

        return velocity, gaze, utterance, new_memory

