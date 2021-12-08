"""
Mobius Linear operation for the Poincare ball model
"""
# import torch.nn
# import geoopt


# def make_manifold(ball=None, c=None):
#     """
#     Parameters:
#     ball: geoopt.PoincareBall
#     c: float

#     """
#     if ball is None:
#         assert c is not None, "curvature of the ball should be explicitly specified"
#         ball = geoopt.PoincareBall(c)

#     return ball


# def linear_transform(input, weight, bias=None,  *, ball: geoopt.PoincareBall):

#     output = ball.mobius_matvec(weight, input)
#     if bias is not None:
#         output = ball.mobius_add(output, bias)
#     return output


# class MobLinear(torch.nn.Linear):
#     def __init__(self, *args, ball=None, c=1.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.ball = geoopt.PoincareBall()
#         if self.bias is not None:
#             self.bias = geoopt.ManifoldParameter(
#                 self.bias, manifold=self.ball)

#     def forward(self, input):
#         return linear_transform(
#             input,
#             weight=self.weight,
#             bias=self.bias,
#             ball=self.ball
#         )

#     # @torch.no_grad()
#     # def reset_parameters(self):
#     #     torch.nn.init.eye_(self.weight)
#     #     self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
#     #     if self.bias is not None:
#     #         self.bias.zero_()


import torch.nn
import geoopt
import geoopt.manifolds.stereographic.math as pmath

# # package.nn.modules.py
# def create_ball(ball=None, c=None):
#     """
#     Helper to create a PoincareBall.
#     Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
#     In this case you will require same curvature parameters for different layers or end up with nans.
#     Parameters
#     ----------
#     ball : geoopt.PoincareBall
#     c : float
#     Returns
#     -------
#     geoopt.PoincareBall
#     """
#     if ball is None:
#         assert c is not None, "curvature of the ball should be explicitly specified"
#         ball = geoopt.PoincareBall(c)
#     # else trust input
#     return ball


# class MobLinear(torch.nn.Linear):
#     def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         # for manifolds that have parameters like Poincare Ball
#         # we have to attach them to the closure Module.
#         # It is hard to implement device allocation for manifolds in other case.
#         self.ball = create_ball(ball, c)
#         if self.bias is not None:
#             self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
#         self.nonlin = nonlin
#         self.reset_parameters()

#     def forward(self, input):
#         return mobius_linear(
#             input,
#             weight=self.weight,
#             bias=self.bias,
#             nonlin=self.nonlin,
#             ball=self.ball,
#         )

#     @torch.no_grad()
#     def reset_parameters(self):
#         torch.nn.init.eye_(self.weight)
#         self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
#         if self.bias is not None:
#             self.bias.zero_()


# # package.nn.functional.py
# def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
#     output = ball.mobius_matvec(weight, input)
#     if bias is not None:
#         output = ball.mobius_add(output, bias)
#     if nonlin is not None:
#         output = ball.logmap0(output)
#         output = nonlin(output)
#         output = ball.expmap0(output)
#     return output

ball = geoopt.PoincareBall(c=0.0)


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=False,
    hyperbolic_bias=False,
    nonlin=None,
    c=0.0,
):
    if hyperbolic_input == True:
        output = pmath.mobius_matvec(weight, input, c=c)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = ball.expmap0(output)
    if bias is not None:
        if hyperbolic_bias == False:
            bias = ball.expmap0(bias)
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.mobius_fn_apply(nonlin, output)
    output = ball.projx(output)
    return output


class MobLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=False,
        hyperbolic_bias=False,
        nonlin=None,
        c=0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias == True:
                self.ball = manifold = geoopt.PoincareBall(c=c)
                self.bias = geoopt.ManifoldParameter(
                    self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(ball.expmap0(self.bias.normal_() / 4))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            c=0.0,
        )

    # def extra_repr(self):
    #     info = super().extra_repr()
    #     info += "c={}, hyperbolic_input={}".format(
    #         self.ball.c, self.hyperbolic_input)
    #     if self.bias is not None:
    #         info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
    #     return info
