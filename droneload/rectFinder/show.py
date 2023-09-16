import cv2
import numpy as np
import matplotlib.pyplot as plt

from droneload.pathFinder.window import Window

import droneload.rectFinder.calibration as calibration
from droneload.rectFinder.rect import get_current_rects, get_main_rect

class Scene:
    ax = None
    lim_x = [-30, 30]
    lim_y = [-30, 30]
    lim_z = [-30, 30]

def draw_rectangles(frame, rects):
    for rect in rects:
        pts = rect.corners2D.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        frame = cv2.putText(frame, f"{rect.id}", rect.corners2D[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def draw_main_rectangle(frame):
    main_rect = get_main_rect()
    if main_rect is not None:
        pts = main_rect.corners2D.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        
        frame = cv2.putText(frame, f"{main_rect.id}", main_rect.corners2D[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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
    

def draw_scene(ax, pause = 0.001):
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    
    rects = get_current_rects()
    for rect, life in rects:
        corners = rect.corners3D
        window = Window(corners)
        corners = window.corners.copy().T
        corners = np.column_stack([corners[:,0], corners[:,1], corners[:,2], corners[:,3], corners[:,0]])
        ax.plot3D(corners[0,:], corners[1,:], corners[2,:], 'green')
    ax.scatter([0], [0], [0], c='r', marker='o')
    plt.pause(pause)