import math
import numpy as np
import mobius
import torch
import torch.nn as nn
import geoopt


class HypFF(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(HypFF, self).__init__()
        # self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.fc1 = mobius.MobLinear(self.input_size, self.hidden_size1)
        self.fc2 = mobius.MobLinear(self.hidden_size1, self.hidden_size2)
        self.fc3 = mobius.MobLinear(self.hidden_size2, self.output_size)

    def forward(self, x):

        ball = geoopt.PoincareBall()
        # x = self.flatten(x)
        # x = ball.projx(x)
        x = ball.mobius_fn_apply(nn.ReLU(), self.fc1(x))
        x = ball.mobius_fn_apply(nn.ReLU(), self.fc2(x))
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = ball.mobius_fn_apply(nn.LogSoftmax(dim=1), self.fc3(x))
        #x = self.fc3(x)
        return x

    # Things to do:
    # find a way to test model
    # How to initialize weights
    # refine model, and test a feed forward neural network with sample data


######################################################

