import numpy as np
import mobius
import torch.nn as nn


class HypFF(nn.Module):

    def __init__(self, weights, bias):
        super(HypFF, self).__init__()
        self.weights = weights
        self.bias = bias
        self.relu = nn.ReLU()

    def linear_transform(self, x):
        #x: inputs
        # Mobius version of the linear transformation
        w_x = mobius.m_vector_mul(self.weights, x)
        res = mobius.m_add(w_x, self.bias)
        return res

    def forward(self, x):
        #x: inputs
        output = self.linear_transform(x)
        output = self.relu(x)
        return output
