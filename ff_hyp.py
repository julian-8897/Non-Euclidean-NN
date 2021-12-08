import geoopt
import numpy as np
import mobius
import torch.nn as nn


class HypFF(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(HypFF, self).__init__()

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.act_fn = nn.ReLU()
        self.fc1 = mobius.MobLinear(self.input_size, self.hidden_size1)
        self.fc2 = mobius.MobLinear(self.hidden_size1, self.hidden_size2)
        self.fc3 = mobius.MobLinear(self.hidden_size2, self.output_size)

    def forward(self, x):

        ball = geoopt.PoincareBall()
        #x = ball.projx(x)
        # x = self.fc1(x)
        # x = ball.expmap0(self.act_fn(ball.logmap0(x)))
        # x = self.fc2(x)
        # x = ball.expmap0(self.act_fn(ball.logmap0(x)))
        x = ball.mobius_fn_apply(self.act_fn, self.fc1(x))
        x = ball.mobius_fn_apply(self.act_fn, self.fc2(x))
        # x = self.fc1(x)
        # x = self.fc2(x)
        #x = ball.mobius_fn_apply(nn.LogSoftmax(dim=1), self.fc3(x))
        # x = ball.mobius_fn_apply(self.act_fn, self.fc3(x))
        x = self.fc3(x)
        return x
