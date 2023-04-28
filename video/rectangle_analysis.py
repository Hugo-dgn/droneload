import scipy
import numpy as np

def rectangle_similarity_score(points, minimun_area=1000):
    if len(points) < 4 or len(points)>6:
        return float('inf'), None
    points = points.squeeze()
    hull = scipy.spatial.ConvexHull(points)

    if len(hull.vertices) != 4:
        return float('inf'), hull
    
    score = 0

    poly = [np.array(points[i]) for i in hull.vertices]
    a, b, c, d = poly

    v1 = b-a
    v2 = c-b
    v3 = d-c
    v4 = a-d

    score += 10/abs(np.linalg.det(np.column_stack([v1, v2])))

    score += 0.1*abs(1 - abs(np.dot(v1, v3)/np.linalg.norm(v1)/np.linalg.norm(v3)))
    score += 0.1*abs(1 - abs(np.dot(v2, v4)/np.linalg.norm(v2)/np.linalg.norm(v4)))

    return score, poly


def get_v1_v2(rect):
    a, b, c, d = rect

    _v1 = b-a
    _v2 = c-b

    e_x = np.array([1, 0])
    if abs(np.dot(_v1, e_x)) >  np.dot(_v2, e_x):
        v1 = _v1
        v2 = _v2
    else:
        v1 = _v2
        v2 = _v1
    
    return v1, v2

def find_normal(rect, alpha):
    v1, v2 = get_v1_v2(rect)

    v1_norme = np.linalg.norm(v1)
    v2_norme = np.linalg.norm(v2)

    v1_dot_v2 = np.dot(v1, v2)

    if v1_dot_v2 == 0:
        return np.array([0, 0, 1])

    A = v1_norme**2 - alpha**2*v2_norme**2

    z1_abs = np.sqrt(0.5*(-A + np.sqrt(A**2+4*alpha**2*v1_dot_v2**2)))

    p = np.append(v1, -z1_abs)
    q = np.append(v2, v1_dot_v2/z1_abs)

    n = np.cross(p, q)

    n = n/np.linalg.norm(n)

    return n

def find_dist(rect, alpha, area, normal):
    v1, v2 = get_v1_v2(rect)
    e_z = np.array([0, 0, 1])
    n_v1_3d = np.linalg.norm(v1)/np.sqrt(1-np.dot(normal, e_z)**2)

