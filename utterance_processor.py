import torch.nn as nn

class UtteranceProcessor(nn.Module):
    def __init__(self, config, memory_size, vocab_size):
        super(UtteranceProcessor, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = config["embedding_size"]

        self.embedding = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.embedding_size)
        )
        
        self.cell = nn.GRUCell(self.embedding_size, memory_size)

        self.linear = nn.Sequential(
            nn.Linear(memory_size, memory_size),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(memory_size, memory_size)
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

        return new_x, new_mem