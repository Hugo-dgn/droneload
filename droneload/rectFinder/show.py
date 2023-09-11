import cv2
import numpy as np

def draw_rectangles(frame, rects):
    for rect in np.array(rects):
        pts = rect.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
