import numpy as np


A = np.array([[0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [5, 4, 3, 2, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [20, 12, 6, 2, 0, 0]]
                )

A_inv = np.linalg.inv(A)

def get_path(x0, x1, v0, v1, a0, a1, L, T, n_point):

    x0_ad = x0/L
    x1_ad = x1/L

    v0_ad = T*v0/L
    v1_ad = T*v1/L

    a0_ad = T**2*a0/L
    a1_ad = T**2*a1/L

    X = np.vstack([x0_ad, x1_ad, v0_ad, v1_ad, a0_ad, a1_ad])

    Y = A_inv @ X
    
    t = np.linspace(0, 1, n_point)

    M = np.vstack([t**(6-n) for n in range(1, 7)])
    U = L*np.transpose(Y) @ M

    return T*t, U