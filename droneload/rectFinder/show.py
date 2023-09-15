import cv2
import numpy as np

import droneload.rectFinder.calibration as calibration

def draw_rectangles(frame, rects):
    for rect in rects:
        pts = rect.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        frame = cv2.putText(frame, f"{rect.id}", rect.corners[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def draw_coordinate(frame, center_2D, rvecs, tvecs):
    
    mtx = calibration.get_mtx()
    dist = calibration.get_dist()
    
    axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    
    imgpts = imgpts.astype(int)
    center_2D = center_2D.astype(int)
    
    center_2D = tuple(center_2D.ravel())
    
    frame = cv2.line(frame, center_2D, tuple(imgpts[0].ravel()), (255,0,0), 5)
    frame = cv2.line(frame, center_2D, tuple(imgpts[1].ravel()), (0,255,0), 5)
    frame = cv2.line(frame, center_2D, tuple(imgpts[2].ravel()), (0,0,255), 5)
    
    return frame
