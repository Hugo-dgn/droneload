import numpy as np
import cv2
import heapq

from droneload.rectFinder.rectangle_analysis import rectangle_similarity_score
import droneload.rectFinder.calibration as calibration
from droneload.rectFinder.rect import Rect

import time


def get_contours_canny(image, seuil, kernel_size):
    blur_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    contours = cv2.Canny(blur_image, seuil, seuil*2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    contours = cv2.dilate(contours, kernel, iterations=1)
    
    return contours

def get_lines(contours, rminLineLength, rmaxLineGap, threshold):
    
    l = calibration.get_image_size()
    
    line_image = np.zeros_like(contours)
    minLineLength = l*rminLineLength
    maxLineGap = l*rmaxLineGap
    lines = cv2.HoughLinesP(contours,rho = 1,theta = 1*np.pi/180,threshold = threshold,minLineLength = minLineLength,maxLineGap = maxLineGap)
    if lines is None:
        return line_image
    
    cv2.polylines(line_image, lines.reshape(-1, 2, 2), False, (255,255,255), 1)
    
    return line_image

def find_rectangles(image, tol):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # keep only the 10 largest contours to maximize performance
    # using a heap to avoid sorting the whole list
    # important to use heapq.nlargest as it is optimized
    contours = heapq.nlargest(10, contours, key = cv2.contourArea)

    rects = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        score, rect = rectangle_similarity_score(approx)
        if score < tol :
            rect = Rect(rect)
            rects.append(rect)
    return rects

def undistort(img):
    mtx = calibration.get_mtx()
    dist = calibration.get_dist()
    
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst