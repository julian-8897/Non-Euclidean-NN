import math
import numpy as np
import mobius
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


class HypFF(nn.Module):

    def __init__(self):
        super(HypFF, self).__init__()
        self.fc1 = mobius.MobLinear(1, 1)
        self.fc2 = mobius.MobLinear(1, 1)

    def forward(self, x):

        ball = geoopt.PoincareBall()
        x = ball.mobius_fn_apply(F.relu, self.fc1(x))
        x = self.fc2(x)
        return x

    # Things to do:
    # find a way to test model
    # How to initialize weights
    # refine model, and test a feed forward neural network with sample data


######################################################

# model.zero_grad()
# output.backward(torch.randn(1))

# target = torch.randn(1)
# criteria = nn.MSELoss()

# loss = criteria(output, target)
# print(loss.grad_fn)
