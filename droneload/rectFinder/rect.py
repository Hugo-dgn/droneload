import numpy as np
import cv2

from droneload.rectFinder.calibration import get_image_size
from droneload.rectFinder.calibration import get_dist, get_mtx
CURRENT_ID = 0

"""
Transform coordinate from solverPnP to classic coordinate system²
"""
correction_matrice = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])

_current_rects = []


def rotate_list(l, n):
    return np.concatenate((l[n:], l[:n]))

def _find_best_fit(corners2D1, corners2D2, current_best_score, current_best_fit):
    l = get_image_size()
    
    for _ in range(4):
        score = np.sum(np.linalg.norm(corners2D1 - corners2D2, axis=1))/l
        if score < current_best_score:
            current_best_score = score
            current_best_fit = corners2D1.copy()
        corners2D1 = rotate_list(corners2D1, 1)
    
    return current_best_score, current_best_fit


def get_current_rects(info=False):
    if info:
        return _current_rects.copy()
    else:
        return [rect for rect, _, _ in _current_rects]

def get_main_rect(min_success=0, max_last_seen = 0):
    """
    Return the rect with the highest nb_fit
    """
    if len(_current_rects) == 0:
        return None
    
    main_rect = max(_current_rects, key=lambda x: x[2])
    if main_rect[2] < min_success or main_rect[1] > max_last_seen:
        return None
    else:
        return main_rect[0]

def remove_old_rects(max_count):
    global _current_rects
    _current_rects = [(rect, count+1, nb_fit) for rect, count, nb_fit in _current_rects if count < max_count]


class Rect:
    
    """
    This class represents a 2D rectangle using cv2 coordinate convention.
    The classic oder begin with the top left corner and goes clockwise.
    The top left corner is the corner closer to the origine
    """
    
    def __init__(self, corners2D, corners3D = None):
        if corners3D is not None:
            self.corners3D = np.array(corners3D)
            self.classic_oder_3D()
        else:
            self.corners3D = None
            
        self.corners2D = np.array(corners2D)
        self.classic_oder2D()
        
        global CURRENT_ID
        self.id = CURRENT_ID
        CURRENT_ID += 1
    
    def define_3D(self, corners3D):
        self.corners3D = np.array(corners3D)
        self.classic_oder_3D()
    
    def classic_oder2D(self):
        
        v1 = self.corners2D[1] - self.corners2D[0]
        v2 = self.corners2D[2] - self.corners2D[1]
        
        if np.cross(v1, v2) < 0:
            self.corners2D = self.corners2D[::-1]
        
        top_left_index = np.argmin(np.linalg.norm(self.corners2D, axis=1))
        self.corners2D = rotate_list(self.corners2D, top_left_index)
        
    def classic_oder_3D(self):
        mask = [True, False, True]
        v1 = self.corners3D[1][mask] - self.corners3D[0][mask]
        v2 = self.corners3D[2][mask] - self.corners3D[1][mask]
        
        if np.cross(v1, v2) > 0:
            self.corners3D = self.corners3D[::-1]
            
        top_left_index = np.argmin(np.linalg.norm(self.corners3D[:,mask], axis=1))
        self.corners3D = rotate_list(self.corners3D, top_left_index)
        
    def center2D(self):
        return np.mean(self.corners2D, axis=0)
    
    def center3D(self):
        return np.mean(self.corners3D, axis=0)
    
    def similarity(self, rect): 
        l = get_image_size()
        
        bestscore, bestfit = _find_best_fit(self.corners2D, rect.corners2D, float('inf'), self.corners2D.copy())
        bestscore, bestfit = _find_best_fit(self.corners2D[::-1], rect.corners2D, bestscore, bestfit)
        
        #add the difference of the area of the two quadrilaterals
        area1 = cv2.contourArea(self.corners2D)
        area2 = cv2.contourArea(rect.corners2D)
        bestscore += abs(area1 - area2)/l**2
        
        return bestscore, bestfit

    def compute(self):
        
        dist = get_dist()
        mtx = get_mtx()
        
        objpts = np.array(self.corners3D, dtype=np.float32)
        imgpts = np.array(self.corners2D, dtype=np.float32)
        
        
        retval, rvecs, tvecs = cv2.solvePnP(objpts, imgpts, mtx, dist)

        pos = (correction_matrice @ tvecs).reshape(3)
        
        dist = np.linalg.norm(pos)
    
        return pos, retval, rvecs, tvecs
    
    def fit(self, tol):
        min_score = float('inf')
        best_index = None
        best_id = None
        best_nb_fit = None
        for i, (rect, last, nb_fit) in enumerate(_current_rects):
            score, _ = self.similarity(rect)
            if score < tol and score < min_score:
                min_score = score 
                best_id = rect.id
                best_index = i   
                best_nb_fit = nb_fit   
        return best_index, min_score, best_id, best_nb_fit

def fit(rects, tol):
    fit_dict = {}
    
    for rect in rects:
        index, score, id, nb_fit = rect.fit(tol)
        if index is None:
            _current_rects.append((rect, 0, 0))
        else:
            if id not in fit_dict:
                fit_dict[id] = [rect, score, index, nb_fit]
            else:
                if score < fit_dict[id][1]:
                    fit_dict[id] = [rect, score, index, nb_fit]
    
    for id, (rect, score, index, nb_fit) in fit_dict.items():
        rect.id = id
        _current_rects[index] = (rect, 0, nb_fit+1)
    
    