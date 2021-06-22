import numpy as np
import mobius
import torch.nn as nn


class HypFF(nn.Module):

    def __init__(self):
        pass

    def linear_transform(self, w, x, b):
        w_x = mobius.m_vector_mul(w, x)
        res = mobius.m_add(w_x, b)
        return res
