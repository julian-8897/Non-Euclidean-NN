import geoopt
import hypmath.mobius as mobius
import torch.nn as nn


class HypFF(nn.Module):
    """
    Hyperbolic FF model with changeable parameters such as:
    input size: input size of data
    hidden_size1: size of hidden layer 1
    hidden_size2: size of hidden layer 2
    output size: output size of data
    act_fn: activation function
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, act_fn):
        super(HypFF, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.act_fn = act_fn
        self.fc1 = mobius.MobLinear(self.input_size, self.hidden_size1)
        self.fc2 = mobius.MobLinear(self.hidden_size1, self.hidden_size2)
        self.fc3 = mobius.MobLinear(self.hidden_size2, self.output_size)

    def forward(self, x):

        ball = geoopt.PoincareBall()
        x = ball.mobius_fn_apply(self.act_fn, self.fc1(x))
        x = ball.mobius_fn_apply(self.act_fn, self.fc2(x))
        x = self.fc3(x)
        return x
