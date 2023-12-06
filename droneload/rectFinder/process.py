import numpy as np
import scipy
import cv2

import droneload.rectFinder.rectangle_analysis as rectangle_analysis
import droneload.rectFinder.calibration as calibration
from droneload.rectFinder.rect import Rect


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.uint16)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.uint16)

def get_contours_sobel(image, seuil):
    gradient_x = scipy.signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    gradient_y = scipy.signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

    contours = np.sqrt(gradient_x**2 + gradient_y**2)
    seuil = 50
    contours[contours < seuil] = 0
    contours[contours >= seuil] = 255
    return contours.astype(np.uint8)

def get_contours_canny(image, seuil, kernel_size):
    blur_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    contours = cv2.Canny(blur_image, seuil, seuil*2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    contours = cv2.dilate(contours, kernel, iterations=1)
    return contours

def get_lines(contours):
    
    l = calibration.get_image_size()
    
    line_image = np.zeros_like(contours)
    minLineLength = l/10
    maxLineGap = l/50
    threshold = 50
    lines = cv2.HoughLinesP(contours,rho = 1,theta = 1*np.pi/180,threshold = threshold,minLineLength = minLineLength,maxLineGap = maxLineGap)
    if lines is None:
        return line_image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),2)
    
    return line_image
            

def find_rectangles(image, tol):
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        score, rect = rectangle_analysis.rectangle_similarity_score(approx)
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