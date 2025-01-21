import torch.nn as nn
import torch

class UtteranceProcessor(nn.Module):
    def __init__(self, config, memory_size, vocab_size, num_agents):
        super(UtteranceProcessor, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = config["embedding_size"]
        self.num_agents = num_agents

        self.embedding = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.embedding_size)
        )

        self.cell = nn.GRUCell(self.embedding_size, memory_size)

        #TODO fix input size
        self.linear = nn.Sequential(
            nn.Linear(memory_size, memory_size),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(memory_size, memory_size)
        )

        #TODO unify the goals and take as input utterance representation and memory
        self.goal_predictor = nn.Sequential(
            nn.Linear(memory_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )

        self.agent_goal_predictor = nn.Sequential(
            nn.Linear(memory_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_agents)
        )

        self.pos_goal_predictor = nn.Sequential(
            nn.Linear(memory_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, x, mem):
        x_batch, x_num, x_dim = x.shape

        x = x.reshape(x_batch * x_num, x_dim)
        x = self.embedding(x)

        mem_batch, mem_num, mem_dim = mem.shape
        mem = mem.reshape(mem_batch * mem_num, mem_dim)

        new_mem = self.cell(x, mem)

        new_x = self.linear(new_mem)

        new_mem = new_mem.reshape(mem_batch, mem_num, -1)
        new_x = new_x.reshape(x_batch, x_num, -1)

        goal_pred = self.goal_predictor(new_mem)
        goal_pred = goal_pred.reshape(x_batch, x_num, -1)
        goal_pred = torch.softmax(goal_pred, dim=-1)
        goal_pred = torch.argmax(goal_pred, dim=-1)

        agent_goal_pred = self.agent_goal_predictor(new_mem)
        agent_goal_pred = agent_goal_pred.reshape(x_batch, x_num, -1)
        agent_goal_pred = torch.softmax(agent_goal_pred, dim=-1)
        agent_goal_pred = torch.argmax(agent_goal_pred, dim=-1)

        pos_goal_pred = self.pos_goal_predictor(new_mem)
        pos_goal_pred = pos_goal_pred.reshape(x_batch, x_num, -1)

        goal_pred = torch.cat((goal_pred.unsqueeze(-1), agent_goal_pred.unsqueeze(-1), pos_goal_pred), dim=-1)

        return new_x, new_mem, goal_pred