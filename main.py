import numpy as np
import mobius_ops

for i in range(0, 100):
    a = np.random.uniform(low=-0.01, high=0.01, size=1)
    b = np.random.uniform(low=-0.01, high=0.01, size=1)

    res = mobius_ops.m_add(-a, a)
    print(res)
