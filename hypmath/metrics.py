import numpy as np
import math
import geoopt
import torch

ball = geoopt.PoincareBall()


def PoincareDistance(X1, X2):

    # distance = np.arccosh(1.0 + 2.0*(np.linalg.norm(X1-X2)**2) /
    #                       ((1 - np.linalg.norm(X1)**2)*(1 - np.linalg.norm(X2)**2)))

    # distance = 2.0 * \
    #     artanh(np.linalg.norm(
    #         ball.mobius_add(-torch.Tensor(X1), torch.Tensor(X2))))

    return ball.dist(torch.Tensor(X1), torch.Tensor(X2))
