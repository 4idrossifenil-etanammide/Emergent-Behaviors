import torch.nn as nn
import torch
import torch.nn.functional as F

class ActionProcessor(nn.Module):
    def __init__(self, config, memory_size, num_landmarks, vocab_size):
        super(ActionProcessor, self).__init__()
        self.actions = config["actions"]
        self.embedding_size = config["embedding_size"]
        self.memory_size = memory_size
        self.num_landmarks = num_landmarks
        self.vocab_size = vocab_size

        self.goal_embedding = nn.Sequential(
            nn.Linear(4, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.embedding_size)
        )

        # TODO: To avoid too big value as velocity and gaze, we should use tanh as activation function
        self.velocity_decoder = nn.Sequential(
            nn.Linear(memory_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

        self.gaze_decoder = nn.Sequential(
            nn.Linear(memory_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

        self.utterance_decoder = nn.Sequential(
            nn.Linear(memory_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.vocab_size)
        )

        self.cell = nn.GRUCell(64 + 256 + 32, memory_size) # TODO: Change those ugly hardcoded values

    def forward(self, goal, memory, physical_features, utterance_features):

        goal = self.goal_embedding(goal.float()) # From (batch, 4) to (batch, embedding_size)

        x = torch.cat((goal, physical_features, utterance_features), dim=1)

        new_memory = self.cell(x, memory)

        v = self.velocity_decoder(new_memory)
        gaze = self.gaze_decoder(new_memory)
        utterance_logits = self.utterance_decoder(new_memory)
        utterance = self.gumbel_softmax(utterance_logits)

        return v, gaze, utterance, new_memory

    def gumbel_softmax(self, logits, temperature = 1.0):

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = logits + gumbel_noise
        y_soft = F.softmax(y / temperature, dim=-1)

        shape = y_soft.size()
        _, max_indices = y_soft.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, max_indices, 1.0)
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard

