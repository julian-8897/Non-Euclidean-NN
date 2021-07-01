import math
import numpy as np
import mobius
import torch.nn as nn
import torch


class HypFF(nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.weights1 = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))
        self.bias1 = nn.Parameter(torch.zeros(2))
        self.weights2 = nn.Parameter(torch.randn(2, 4) / math.sqrt(2))
        self.bias2 = nn.Parameter(torch.zeros(4))

    def forward(self, X):
        a1 = mobius.m_vector_mul(self.weights1, X) + self.bias1
        h1 = a1.sigmoid()
        a2 = mobius.m_vector_mul(self.weights2, X) + self.bias2
        h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)
        return h2

    # Things to do:
    # Project values into "hyperbolic values"
    # find a way to test model
