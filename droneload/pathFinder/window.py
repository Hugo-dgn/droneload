import numpy as np

e_1 = np.array([1, 0, 0], dtype=np.float64)
e_2 = np.array([0, 1, 0], dtype=np.float64)
e_3 = np.array([0, 0, 1], dtype=np.float64)

def _gram_shmitt(v):
    v = np.array(v, dtype=np.float64)/np.linalg.norm(v)
    candidate_basis = [[v, e_2, e_3], [v, e_1, e_3], [v, e_1, e_2]]
    V = None
    for A in candidate_basis:
        matrix = np.column_stack(A)
        det = np.linalg.det(matrix)
        if not np.isclose(det, 0):
            V = matrix
            break

    e1 = V[:,0]
    e2 = V[:,1] - np.dot(V[:,1], e1) * e1
    e3 = V[:,2] - np.dot(V[:,2], e1) * e1 - np.dot(V[:,2], e2) * e2

    T = np.column_stack((e1/np.linalg.norm(e1), e2/np.linalg.norm(e2), e3/np.linalg.norm(e3)))

    if np.linalg.det(T) < 0:
        T = T @ np.array([  [1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]])
    return T


class Window:
    def __init__(self, corners):
        self.corners = np.array(corners)
        self.n = np.cross(self.corners[1]-self.corners[0], self.corners[2]-self.corners[0])
        self.p = np.mean(corners, axis=0)