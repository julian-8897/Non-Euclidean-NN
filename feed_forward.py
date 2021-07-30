import math
import numpy as np
import mobius
import torch
import torch.nn as nn
import geoopt


class HypFF(nn.Module):

    def __init__(self):
        super(HypFF, self).__init__()
        # self.flatten = nn.Flatten()
        self.fc1 = mobius.MobLinear(784, 512)
        self.fc2 = mobius.MobLinear(512, 256)
        self.fc3 = mobius.MobLinear(256, 10)

    def forward(self, x):

        ball = geoopt.PoincareBall()
        # x = self.flatten(x)
        # x = ball.projx(x)
        x = ball.mobius_fn_apply(nn.LeakyReLU(), self.fc1(x))
        x = ball.mobius_fn_apply(nn.LeakyReLU(), self.fc2(x))
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        #x = ball.mobius_fn_apply(nn.LogSoftmax(dim=1), self.fc3(x))
        # x = ball.logmap0(x)
        return x

    # Things to do:
    # find a way to test model
    # How to initialize weights
    # refine model, and test a feed forward neural network with sample data


######################################################
# ball = geoopt.PoincareBall()
# model = HypFF()
# params = list(model.parameters())
# point = torch.Tensor([8.8])
# input = ball.projx(point)
# output = model(input)

# print(model)
# print("Output is :", output)
# model.zero_grad()
# output.backward(torch.randn(1))

# target = torch.randn(1)
# criteria = nn.MSELoss()

# loss = criteria(output, target)
# print(loss.grad_fn)
