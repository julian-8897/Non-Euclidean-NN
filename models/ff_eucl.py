import torch.nn as nn


class EuclFF(nn.Module):
    """
    Euclidean FF model with changeable parameters such as:
    input size: input size of data
    hidden_size1: size of hidden layer 1
    hidden_size2: size of hidden layer 2
    output size: output size of data
    act_fn: activation function
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, act_fn):
        super(EuclFF, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.act_fn = act_fn
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x = self.fc3(x)
        return x
