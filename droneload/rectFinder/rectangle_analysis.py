import numpy as np

import droneload.rectFinder.calibration as calibration

def rectangle_similarity_score(points):
    l = calibration.get_image_size()

    if len(points) != 4:
        return float('inf'), None
    points = points.squeeze()
    score = 0
    a, b, c, d = points

    v1 = (b-a)/l
    v2 = (c-b)/l
    v3 = (d-c)/l
    v4 = (a-d)/l
    
    nv1_carre = np.sum(v1**2)
    nv2_carre = np.sum(v2**2)
    nv3_carre = np.sum(v3**2)
    nv4_carre = np.sum(v4**2)
    
    cos1_carre = np.dot(v1, v2)**2/(nv1_carre*nv2_carre)
    cos2_carre = np.dot(v2, v3)**2/(nv2_carre*nv3_carre)
    cos3_carre = np.dot(v3, v4)**2/(nv3_carre*nv4_carre)
    cos4_carre = np.dot(v4, v1)**2/(nv4_carre*nv1_carre)
    
    score += cos1_carre + cos2_carre + cos3_carre + cos4_carre

    return score, points