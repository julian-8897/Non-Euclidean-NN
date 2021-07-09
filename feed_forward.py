import math
import numpy as np
import mobius
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

ball = geoopt.PoincareBall()


class HypFF(nn.Module):

    def __init__(self):
        super(HypFF, self).__init__()
        self.fc1 = mobius.MobLinear(1, 1)

    def forward(self, x):
        # a2 = mobius.m_vector_mul(self.weights2, X) + self.bias2
        # h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)
        x = ball.mobius_fn_apply(F.relu, x)
        return x

    # Things to do:
    # find a way to test model
    # How to initialize weights
    # refine model, and test a feed forward neural network with sample data


######################################################
model = HypFF()
params = list(model.parameters())
point = torch.randn(1)
input = ball.projx(point)
output = model(input)

print(model)
print(params)
print(point, input)
print("Output is :", output)

# model.zero_grad()
# output.backward(torch.randn(1))

# target = torch.randn(1)
# criteria = nn.MSELoss()

# loss = criteria(output, target)
# print(loss.grad_fn)
