"""
Mobius operations for the Poincare model
"""

import numpy as np
from numpy import linalg


def m_add(x, y):
    numer = (1.0 + 2.0 * np.dot(x, y) + linalg.norm(y)**2) * \
        x + (1.0 - linalg.norm(x)**2) * y
    denom = 1.0 + 2.0 * np.dot(x, y) + linalg.norm(x)**2 * linalg.norm(y)**2
    return numer/denom


def m_scalar_mul(r, x):
    res = np.tanh(r * np.arctanh(linalg.norm(x))) * (x/linalg.norm(x))
    return res


def m_vector_mul(M, x):
    Mx = M.dot(x)
    norm_Mx = linalg.norm(Mx)
    norm_x = linalg.norm(x)
    res = np.tanh((norm_Mx/norm_x) * np.arctanh(norm_x)) * (Mx/norm_Mx)
    return res
