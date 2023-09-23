import numpy as np

A = np.array([[0, 0, 1],
              [1, 1, 1],
              [2, 1, 0]]
                )

A_inv = np.linalg.inv(A)

def get_path(x0, x1, n, L, n_point):
    
    x0 = np.array(x0)
    x1 = np.array(x1)
    n = np.array(n)

    X = np.vstack([x0, x1, n])

    Y = A_inv @ X
    
    t = np.linspace(0, 1, n_point)

    M = np.vstack([t**(3-n) for n in range(1, 4)])
    U = np.transpose(Y) @ M

    U_after_window = x1.reshape(3, -1) + np.outer(n/np.linalg.norm(n), np.linspace(0, L, n_point))
    return np.concatenate([U, U_after_window.reshape(3, -1)], axis=1)