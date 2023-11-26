import scipy
import numpy as np

import cv2

import droneload.rectFinder.calibration as calibration

def rectangle_similarity_score(points):

    l = calibration.get_image_size()

    if len(points) < 4:
        return float('inf'), None
    points = points.squeeze()
    hull = scipy.spatial.ConvexHull(points)
    
    if len(hull.vertices) != 4:
        return float('inf'), None
    
    score = 0

    poly = [np.array(points[i]) for i in hull.vertices]
    a, b, c, d = poly

    v1 = (b-a)/l
    v2 = (c-b)/l
    v3 = (d-c)/l
    v4 = (a-d)/l

    score += 1e-5/abs(np.linalg.det(np.column_stack([v1, v2])))**2
    
    nv1 = np.linalg.norm(v1)
    nv2 = np.linalg.norm(v2)
    nv3 = np.linalg.norm(v3)
    nv4 = np.linalg.norm(v4)
    
    cos1 = np.dot(v1, v2)/(nv1*nv2)
    cos2 = np.dot(v2, v3)/(nv2*nv3)
    cos3 = np.dot(v3, v4)/(nv3*nv4)
    cos4 = np.dot(v4, v1)/(nv4*nv1)
    
    score += cos1**2 + cos2**2 + cos3**2 + cos4**2

    return score, poly