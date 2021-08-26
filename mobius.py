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


def linear_transform(input, weight, bias=None,  *, ball: geoopt.PoincareBall):

    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    return output


class MobLinear(torch.nn.Linear):
    def __init__(self, *args, ball=None, c=1.0, **kwargs):
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

    # @torch.no_grad()
    # def reset_parameters(self):
    #     torch.nn.init.eye_(self.weight)
    #     self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
    #     if self.bias is not None:
    #         self.bias.zero_()
