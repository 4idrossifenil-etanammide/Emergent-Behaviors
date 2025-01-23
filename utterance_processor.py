import torch.nn as nn
import torch

class UtteranceProcessor(nn.Module):
    def __init__(self, config, memory_size, vocab_size, num_agents):
        super(UtteranceProcessor, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.feature_size = config["feature_size"]
        self.linear_input = self.embedding_size + memory_size
        self.num_agents = num_agents

        # (vocab_size) -> (embedding_size)
        self.embedding = nn.Sequential(
            nn.Linear(vocab_size, self.hidden_size),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.embedding_size)
        )

        self.cell = nn.GRUCell(self.embedding_size, memory_size)

        #(embedding_size + memory_size) -> (feature_size)
        self.linear = nn.Sequential(
            nn.Linear(self.linear_input, self.hidden_size),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.feature_size)
        )

        #Is it fine for num_agents to be a dimension? NO! # TODO change this
        self.goal_predictor = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3 + num_agents + 2)
        )

    def forward(self, x, mem):
        x_batch, x_num, x_dim = x.shape
        x = x.reshape(x_batch * x_num, x_dim)
        x = self.embedding(x)

        mem_batch, mem_num, mem_dim = mem.shape
        mem = mem.reshape(mem_batch * mem_num, mem_dim)

        new_mem = self.cell(x, mem)

        #FIrst i would compute this, and THEN update the memory
        new_x = self.linear(torch.cat((x, new_mem), dim=1))

        new_mem = new_mem.reshape(mem_batch, mem_num, -1)
        new_x = new_x.reshape(x_batch, x_num, -1)

        goal_pred_logits = self.goal_predictor(new_x)
        goal_pred = torch.cat( [torch.argmax(torch.softmax(goal_pred_logits[..., :3], dim=-1), dim=-1).unsqueeze(-1),
                              torch.argmax(torch.softmax(goal_pred_logits[..., 3:3 + self.num_agents], dim=-1), dim =-1).unsqueeze(-1),
                              goal_pred_logits[..., 3 + self.num_agents:]], dim = -1)
        #should this also changfe how the loss function si defined?

        return new_x, new_mem, goal_pred