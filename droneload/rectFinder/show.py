import cv2
import numpy as np
import matplotlib.pyplot as plt

from droneload.pathFinder.window import Window

import droneload.rectFinder.calibration as calibration
from droneload.rectFinder.rect import get_current_rects

class Scene:
    ax = None
    lim_x = [-30, 30]
    lim_y = [-30, 30]
    lim_z = [-30, 30]

def draw_rectangles(frame, rects):
    for rect in rects:
        pts = rect.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        frame = cv2.putText(frame, f"{rect.id}", rect.corners[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def draw_coordinate(frame, center_2D, rvecs, tvecs):
    
    mtx = calibration.get_mtx()
    dist = calibration.get_dist()
    
    Scene.axis = np.float32([[3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)
    imgpts, jac = cv2.projectPoints(Scene.axis, rvecs, tvecs, mtx, dist)
    
    imgpts = imgpts.astype(int)
    center_2D = center_2D.astype(int)
    
    center_2D = tuple(center_2D.ravel())
    
    frame = cv2.line(frame, center_2D, tuple(imgpts[0].ravel()), (255,0,0), 5)
    frame = cv2.line(frame, center_2D, tuple(imgpts[1].ravel()), (0,255,0), 5)
    frame = cv2.line(frame, center_2D, tuple(imgpts[2].ravel()), (0,0,255), 5)
    
    return frame

def init_scene(lim_x=None, lim_y=None, lim_z=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if lim_x is not None:
        Scene.lim_x = lim_x
    if lim_y is not None:
        Scene.lim_y = lim_y
    if lim_z is not None:
        Scene.lim_z = lim_z
        
    Scene.ax = ax
    
    Scene.ax.set_xlim(Scene.lim_x)
    Scene.ax.set_ylim(Scene.lim_y)
    Scene.ax.set_zlim(Scene.lim_z)
    
    Scene.ax.set_xlabel('x (m)')
    Scene.ax.set_ylabel('y (m)')
    Scene.ax.set_zlabel('z (m)')
    


def draw_scene():
    
    if Scene.ax is None:
        message = "the scene wasn't init, use droneload.rectFinder.init_scene"
        assert AssertionError(message)
    Scene.ax.clear()
    
    Scene.ax.set_xlim([-30, 30])
    Scene.ax.set_ylim([-30, 30])
    Scene.ax.set_zlim([-30, 30])
    
    Scene.ax.set_xlabel('x (m)')
    Scene.ax.set_ylabel('y (m)')
    Scene.ax.set_zlabel('z (m)')
    Scene.ax.scatter([0], [0], [0], c='r', marker='o')
    
    rects = get_current_rects()
    for rect, life in rects:
        corners = rect._updim.corners
        window = Window(corners)
        corners = window.corners.copy().T
        corners = np.column_stack([corners[:,0], corners[:,1], corners[:,2], corners[:,3], corners[:,0]])
        Scene.ax.plot3D(corners[0,:], corners[1,:], corners[2,:], 'green')
    Scene.ax.scatter([0], [0], [0], c='r', marker='o')
    plt.pause(0.0001)