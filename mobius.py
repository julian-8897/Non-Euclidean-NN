"""
Mobius Linear operation for the Poincare ball model
"""
import torch.nn
import geoopt


def make_manifold(ball=None, c=None):
    """
    Parameters:
    ball: geoopt.PoincareBall
    c: float

    """
    if ball is None:
        assert c is not None, "curvature of the ball should be explicitly specified"
        ball = geoopt.PoincareBall(c)

    return ball


def linear_transform(
        input,
        weight,
        bias=None,
        *,
        ball: geoopt.PoincareBall):

    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    # if nonlin is not None:
    #     output = ball.logmap0(output)
    #     output = nonlin(output)
    #     output = ball.expmap0(output)
    return output


class MobLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        ball=None,
        c=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ball = make_manifold(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(
                self.bias, manifold=self.ball)

    def forward(self, input):
        return linear_transform(
            input,
            weight=self.weight,
            bias=self.bias,
            ball=self.ball
        )


# def m_add(x, y):
#     # Mobius addition operation
#     numer = (1.0 + 2.0 * np.dot(x, y) + linalg.norm(y)**2) * \
#         x + (1.0 - linalg.norm(x)**2) * y
#     denom = 1.0 + 2.0 * np.dot(x, y) + linalg.norm(x)**2 * linalg.norm(y)**2
#     return numer/denom


# def m_scalar_mul(r, x):
#     # Mobius scalar multiplication operation
#     res = np.tanh(r * np.arctanh(linalg.norm(x))) * (x/linalg.norm(x))
#     return res


# def m_vector_mul(M, x):
#     # Mobius vector multiplication operation
#     Mx = np.dot(M, x)
#     norm_Mx = linalg.norm(Mx)
#     norm_x = linalg.norm(x)
#     res = np.tanh((norm_Mx/norm_x) * np.arctanh(norm_x)) * (Mx/norm_Mx)
#     return res


# def dist_fn(x, y):
#     # closed-form expression of the distance function of the poincare model
#     numer = linalg.norm(x - y)**2
#     denom = (1.0 - linalg.norm(x)**2)*(1.0 - linalg.norm(y)**2)
#     distance = np.arccosh(1.0 + 2.0*numer/denom)
#     return distance
