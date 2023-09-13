import numpy as np
import scipy
import cv2

import droneload.rectFinder.rectangle_analysis as rectangle_analysis
import droneload.rectFinder.calibration as calibration


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.uint16)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.uint16)

def get_contours_sobel(image, seuil=20):
    gradient_x = scipy.signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    gradient_y = scipy.signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

    contours = np.sqrt(gradient_x**2 + gradient_y**2)
    seuil = 50
    contours[contours < seuil] = 0
    contours[contours >= seuil] = 255
    return contours.astype(np.uint8)

def find_rectangles(image, tol):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        score, rect = rectangle_analysis.rectangle_similarity_score(approx)
        if score < tol :
            rects.append(rect)

    return np.array(rects, dtype=np.float32)

def undistort(img):
    mtx = calibration.get_mtx()
    dist = calibration.get_dist()
    
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst