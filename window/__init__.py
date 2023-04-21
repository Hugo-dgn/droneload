from .path import get_path
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
    def __init__(self, position, vector, scale=1, theta=0):
        self.position = np.array(position)
        self.vector = np.array(vector)
        self.scale = scale
        self.basis = _gram_shmitt(self.vector)
        self.norme = np.linalg.norm(vector)
        self.rotat = np.array([[1, 0, 0],
                               [0, np.cos(theta), np.sin(theta)],
                               [0, -np.sin(theta), np.cos(theta)]])

    def get_corner(self):
        fact_e2 = self.scale/2*self.norme
        fact_e3 = 1/(2*self.scale)*self.norme
        top_r = fact_e2*e_2 + fact_e3*e_3
        top_l = fact_e2*e_2 - fact_e3*e_3
        bottom_r = -fact_e2*e_2 + fact_e3*e_3
        bottom_l = -fact_e2*e_2 - fact_e3*e_3

        corner_window_base = np.column_stack((top_r, top_l, bottom_l, bottom_r))
        corner = self.basis @ self.rotat @ corner_window_base + np.column_stack([self.position for _ in range(4)])
        return corner
