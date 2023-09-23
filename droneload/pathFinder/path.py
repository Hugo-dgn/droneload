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

    X = np.vstack([x0/L, x1/L, n])

    Y = A_inv @ X
    
    t = np.linspace(0, 1, n_point)

    M = np.vstack([t**(3-n) for n in range(1, 4)])
        
    U = L*np.transpose(Y) @ M

    return U