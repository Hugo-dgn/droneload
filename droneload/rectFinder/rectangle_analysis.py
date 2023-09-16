import scipy
import numpy as np

import cv2

import droneload.rectFinder.calibration as calibration

def rectangle_similarity_score(points):

    l = calibration.get_image_size()

    if len(points) != 4:
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

    score += 1/abs(np.linalg.det(np.column_stack([v1, v2])))**2
    
    score += (np.linalg.norm(v1)*np.linalg.norm(v3)/(abs(np.dot(v1, v3))+1e-10))**2
    score += (np.linalg.norm(v2)*np.linalg.norm(v4)/(abs(np.dot(v2, v4))+1e-10))**2

    return score, poly