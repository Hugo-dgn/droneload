import scipy
import numpy as np
import locals

def rectangle_similarity_score(points):
    if len(points) < 4 or len(points)>6:
        return float('inf'), None
    points = points.squeeze()
    hull = scipy.spatial.ConvexHull(points)

    if len(hull.vertices) != 4:
        return float('inf'), hull.vertices
    
    score = 0

    poly = [np.array(points[i]) for i in hull.vertices]
    a, b, c, d = poly

    v1 = b-a
    v2 = c-b
    v3 = d-c
    v4 = a-d

    inverse_A = 2*1/((abs(np.linalg.det(np.column_stack([v1, v2]))) + abs(np.linalg.det(np.column_stack([v3, v4]))))+0.0000001)
    score += locals.alpha_1*inverse_A**2

    score += locals.alpha_2*((np.linalg.norm(v1+v3)*inverse_A)*2 + (np.linalg.norm(v2+v4)*inverse_A)*2)
    return score, poly


def get_v1_v2(rect):
    a, b, c, d = rect

    _v1 = b-a
    _v2 = c-b

    e_x = np.array([1, 0])
    if abs(np.dot(_v1, e_x)) < np.dot(_v2, e_x):
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

    A = alpha**2*v1_norme**2 - v2_norme**2
    
    norme_product = v1_norme*v2_norme

    z = 0.5*norme_product*(-A/norme_product + np.sqrt((A/norme_product)**2+4*alpha**2*(v1_dot_v2/norme_product)**2))/alpha**2

    p = np.append(v1, z)
    q = np.append(v2, -v1_dot_v2/z)

    n = np.cross(p, q)

    n = n/np.linalg.norm(n)
    return n