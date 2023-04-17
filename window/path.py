import numpy as np

A = np.array([[0, 0, 0, 1],
              [1, 1, 1, 1],
              [0, 0, 1, 0],
              [3, 2, 1, 0]])

inv_A = np.linalg.inv(A)

def get_coef(x1, x2, v1, v2):
    return inv_A @ np.array([x1, x2, v1, v2])

def get_poly(x1, x2, v1, v2):
    coef = get_coef(x1, x2, v1, v2)
    def P(t):
        return coef[0]*t**3 + coef[1]*t**2 + coef[2]*t**1 + coef[3]
    return P

def get_path(x1, x2, v1, v2):
    P = get_poly(x1[0], x2[0], v1[0], v2[0])
    Q = get_poly(x1[1], x2[1], v1[1], v2[1])
    S = get_poly(x1[2], x2[2], v1[2], v2[2])
    def u(t):
        return np.array([P(t), Q(t), S(t)])
    return u