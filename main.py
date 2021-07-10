import feed_forward
import numpy as np
import torch
import geoopt

ball = geoopt.PoincareBall()
model = feed_forward.HypFF()
params = list(model.parameters())
point = torch.randn(1)
input = ball.projx(point)
output = model(input)

print(model)
print("Output is :", output)
# for i in range(0, 100):
#     a = np.random.uniform(low=-0.01, high=0.01, size=1)
#     b = np.random.uniform(low=-0.01, high=0.01, size=1)

#     res = mobius.m_add(-a, a)
#     print(res)
