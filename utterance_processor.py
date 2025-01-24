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

        self.cell = nn.GRUCell(self.hidden_size, memory_size)

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
        # x: B x NA x utDim
        # mem: B x memDim

        #extract the embedding
        x = self.embedding(x)
        x_batch, x_num, x_dim = x.shape

        #prepare the memory
        mem = mem.unsqueeze(1).repeat(1,x_num,1)

        #compute the features
        new_x = self.linear(torch.cat((x, mem), dim=-1))

        #predict the goal for all the agents
        goal_pred_logits = self.goal_predictor(new_x)
        goal_pred = torch.cat( [torch.argmax(torch.softmax(goal_pred_logits[..., :3], dim=-1), dim=-1).unsqueeze(-1),
                              torch.argmax(torch.softmax(goal_pred_logits[..., 3:3 + self.num_agents], dim=-1), dim =-1).unsqueeze(-1),
                              goal_pred_logits[..., 3 + self.num_agents:]], dim = -1)
        #should this also changfe how the loss function si defined? yes!
        return new_x, goal_pred
    
    def mem_update(self, x, mem):
        return self.cell(x, mem)