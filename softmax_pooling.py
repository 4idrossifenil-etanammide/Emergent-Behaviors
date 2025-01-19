import torch.nn as nn
import torch
import torch.nn.functional as F

# Compute a weighted sum over a specified dimension
class SoftmaxPooling(nn.Module):
    def __init__(self, dim, temperature = 1.0):
        super(SoftmaxPooling, self).__init__()
        self.dim = dim

    def forward(self, x):
        weights = F.softmax(x / self.temperature, dim=self.dim)
        pooled = torch.sum(weights * x, dim=self.dim)
        return pooled