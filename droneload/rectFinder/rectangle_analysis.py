import scipy
import numpy as np

import cv2

import droneload.rectFinder.calibration as calibration

def rectangle_similarity_score(points):

    l = calibration.get_image_size()

    if len(points) < 4 or len(points)>6:
        return float('inf'), None
    points = points.squeeze()
    hull = scipy.spatial.ConvexHull(points)

    if len(hull.vertices) != 4:
        return float('inf'), hull
    
    score = 0

    poly = [np.array(points[i]) for i in hull.vertices]
    a, b, c, d = poly

    v1 = (b-a)/l
    v2 = (c-b)/l
    v3 = (d-c)/l
    v4 = (a-d)/l

    score += 1/abs(np.linalg.det(np.column_stack([v1, v2])))

    score += 0.1*abs(1 - abs(np.dot(v1, v3)/np.linalg.norm(v1)/np.linalg.norm(v3)))**2
    score += 0.1*abs(1 - abs(np.dot(v2, v4)/np.linalg.norm(v2)/np.linalg.norm(v4)))**2

    return score, poly


def get_v1_v2(rect):

    l = calibration.get_image_size()

    a, b, c, d = rect/l

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
    v1, v2 = get_v1_v2(np.array(rect))

    v1_norme = np.linalg.norm(v1)
    v2_norme = np.linalg.norm(v2)

    v1_dot_v2 = np.dot(v1, v2)

    if v1_dot_v2 == 0:
        A = v1_norme*v2_norme
        return np.array([0, 0, A]), np.array([0, 0, A])

    A = v1_norme**2 - alpha**2*v2_norme**2

    z1 = 0.5*(-A + np.sqrt(A**2+4*alpha**2*v1_dot_v2**2))
    p1 = np.append(v1, z1)
    q1 = np.append(v2, -v1_dot_v2/z1)
    n1 = np.cross(p1, q1)

    z2 = 0.5*(-A + np.sqrt(A**2+4*alpha**2*v1_dot_v2**2))
    p2 = np.append(v1, z2)
    q2 = np.append(v2, -v1_dot_v2/z2)
    n2 = np.cross(p2, q2)

    return n1, n2

def get_3D_vecs(objpts, imgpts):
    
    mtx = calibration.get_mtx()
    dist = calibration.get_dist()
    
    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(objpts, imgpts, mtx, dist)
    
    return retval, rvecs, tvecs, inliers

def find_center_2D(rect):
    rect = np.array(rect)

    if rect.shape == (4, 2):
        return (rect[0] + rect[1] + rect[2] + rect[3])/4

def find_center_3D(center_2D, dist):
    x = np.array([center_2D[0], center_2D[1], dist])
    cmatrix = calibration.get_camera_matrix()
    return np.linalg.inv(cmatrix) @ x