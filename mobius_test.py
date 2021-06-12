import numpy as np
import mobius_ops.py

for i in range(0, 10000):
    a = np.random.uniform(low=-0.01, high=0.01, size=1)
    b = np.random.uniform(low=-0.01, high=0.01, size=1)

    res = m_add(-a, b)
    print(res)
