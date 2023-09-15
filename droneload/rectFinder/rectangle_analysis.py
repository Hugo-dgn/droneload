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

def get_3D_vecs(target_rect, rect):
    
    mtx = calibration.get_mtx()
    dist = calibration.get_dist()
    
    objpts = target_rect.corners.astype(np.float32)
    imgpts = rect.corners.astype(np.float32)
    
    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(objpts, imgpts, mtx, dist)
    
    return retval, rvecs, tvecs, inliers

def find_pos_3D(objpts, tvecs, rvecs):
    if len(objpts.shape) != 2 or objpts.shape[1] != 3:
        raise ValueError("objpts must be a matrix of shape (n, 3)")
    
    R, _ = cv2.Rodrigues(rvecs)
    
    t = np.array([[tvecs[0][0]], [tvecs[2][0]], [tvecs[1][0]]])

    points = tvecs + R@objpts.T
    
    correction_matrice = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])
    
    points = correction_matrice@points
    
    return points.T