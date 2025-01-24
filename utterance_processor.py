import torch.nn as nn
import torch

class UtteranceProcessor(nn.Module):
    def __init__(self, hidden_size, memory_size, vocab_size):
        super(UtteranceProcessor, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        self.embedding = nn.Sequential(
            nn.Linear(vocab_size, self.hidden_size//2),
            nn.ELU(),
            nn.Linear(self.hidden_size//2, self.hidden_size)
        )

        self.cell = nn.GRUCell(self.hidden_size, memory_size)

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size + self.memory_size, self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, 256)
        )

    def forward(self, x, mem):
        
        x = self.embedding(x)

        x_batch, x_num, x_dim = x.shape
        x = x.reshape(x_batch * x_num, x_dim).float()

        mem_batch, mem_num, mem_dim = mem.shape
        mem = mem.reshape(mem_batch * mem_num, mem_dim)

        new_mem = self.cell(x, mem)
        x = self.linear(torch.cat((x,new_mem), dim=-1))

        x = x.reshape(x_batch, x_num, -1)
        new_mem = new_mem.reshape(mem_batch, mem_num, -1)

        return x, new_mem

