import torch.nn as nn
import torch
import torch.nn.functional as F

class ActionProcessor(nn.Module):
    def __init__(self, config, memory_size, num_landmarks, vocab_size):
        super(ActionProcessor, self).__init__()
        self.actions = config["actions"]
        self.embedding_size = config["embedding_size"]
        self.feature_size = config["feature_size"]
        self.hidden_size = config["hidden_size"]
        self.goal_size = config["goal_size"]
        self.memory_size = memory_size
        self.action_decoder_input = self.embedding_size + 2 * self.feature_size + self.memory_size
        self.num_landmarks = num_landmarks
        self.vocab_size = vocab_size

        self.goal_embedding = nn.Sequential(
            nn.Linear(self.goal_size, self.hidden_size),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.embedding_size)
        )

        # TODO: To avoid too big value as velocity and gaze, we should use tanh as activation function
        self.action_decoder = nn.Sequential(
            nn.Linear(self.action_decoder_input, self.hidden_size),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 2+2+self.vocab_size)
        )

        self.cell = nn.GRUCell(self.embedding_size + 2 * self.feature_size, memory_size) # TODO: Change those ugly hardcoded values

    def forward(self, goal, memory, physical_features, utterance_features):

        goal = self.goal_embedding(goal.float()) # From (batch, 4) to (batch, embedding_size)

        x = torch.cat((goal, physical_features, utterance_features), dim=-1)

        new_memory = self.cell(x, memory)

        #TODO need to add small noise ( u = psi_u + epsilon)
        action = self.action_decoder(torch.cat((x, new_memory), dim =-1))
        velocity = action[..., :2]
        gaze = action[..., 2:4]
        utterance_logits = action[..., 4:]
        utterance = self.gumbel_softmax(utterance_logits)

        return velocity, gaze, utterance, new_memory

    def gumbel_softmax(self, logits, temperature = 1.0):

        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = logits + gumbel_noise
        y_soft = F.softmax(y / temperature, dim=-1)

        _, max_indices = y_soft.max(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter(-1, max_indices, 1.0)
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard

