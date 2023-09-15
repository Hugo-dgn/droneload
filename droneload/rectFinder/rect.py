import numpy as np

from droneload.rectFinder.calibration import get_image_size

CURRENT_ID = 0

def rotate_list(l, n):
    return np.concatenate((l[n:], l[:n]))

def _find_best_fit(corners1, corners2, current_best_score, current_best_fit):
    l = get_image_size()
    
    for _ in range(4):
        score = np.linalg.norm(corners1 - corners2)/l
        if score < current_best_score:
            current_best_score = score
            current_best_fit = corners1.copy()
        corners1 = rotate_list(corners1, 1)
    
    return current_best_score, current_best_fit

_current_rects = []

def remove_old_rects(max_count):
    global _current_rects
    _current_rects = [(rect, count+1) for rect, count in _current_rects if count < max_count]

def get_current_rects():
    return _current_rects.copy()


class Rect2D:
    
    """
    This class represents a 2D rectangle using cv2 coordinate convention.
    The classic oder begin with the top left corner and goes clockwise.
    The top left corner is the corner closer to the origine
    """
    
    def __init__(self, corners):
        self.corners = np.array(corners)
        if not self.corners.shape == (4, 2):
            message = "corners must be a 4x2 numpy array"
            raise ValueError(message)
        
        self.classic_oder()
        
        global CURRENT_ID
        self.id = CURRENT_ID
        CURRENT_ID += 1
    
    def classic_oder(self):
        
        v1 = self.corners[1] - self.corners[0]
        v2 = self.corners[2] - self.corners[1]
        
        if np.cross(v1, v2) < 0:
            self.corners = self.corners[::-1]
        
        top_left_index = np.argmin(np.linalg.norm(self.corners, axis=1))
        self.corners = rotate_list(self.corners, top_left_index)
    
    def center(self):
        return np.mean(self.corners, axis=0)
    
    def similarity(self, rect): 
        bestscore, bestfit = _find_best_fit(self.corners, rect.corners, float('inf'), self.corners.copy())
        bestscore, bestfit = _find_best_fit(self.corners[::-1], rect.corners, bestscore, bestfit)
        
        return bestscore, bestfit
    
    def fit(self, tol):
        for i, (rect, last) in enumerate(_current_rects):
            score, fit = self.similarity(rect)
            if score < tol:
                self.corners = fit
                self.id = rect.id
                _current_rects[i] = (self, 0)
                return True
        
        _current_rects.append((self, 0))
        return False
        
        

class Rect3D:
    
    """
    This class represents a 3D rectangle using classic 3D coordinate convention.
    The classic oder begin with the top left corner and goes clockwise when projecting to the (x, z) plane.
    """
    
    def __init__(self, corners) -> None:
        self.corners = np.array(corners)
        if not self.corners.shape == (4, 3):
            message = "corners of Rect3D must be a 4x3 numpy array"
            raise ValueError(message)
        
        self.classic_oder()
    
    def classic_oder(self):
        
        projection = self.corners[:, [0, 2]]
        
        projection -= 2*np.array([0, np.max(projection.reshape(-1))])
        
        v1 = projection[1] - projection[0]
        v2 = projection[2] - projection[1]
        
        if np.cross(v1, v2) > 0:
            self.corners = self.corners[::-1]
            projection = projection[::-1]
        
        top_left_index = np.argmin(np.linalg.norm(projection, axis=1))
        self.corners = rotate_list(self.corners, top_left_index)