import numpy as np
import scipy
import cv2

import video.rectangle_analysis as rectangle_analysis


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#####matrice de lissage

#uniforme
moy_kernel = np.array([[1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]])

##gaussien
k = 3
sigma = 0.5
gaussien_kernel = cv2.getGaussianKernel(k, sigma)
gaussien_kernel = np.outer(gaussien_kernel, gaussien_kernel.transpose())

def get_contours_sobel(image, seuil=20):
    #image = scipy.signal.convolve2d(image, gaussien_kernel, mode='same', boundary='symm')

    gradient_x = scipy.signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    gradient_y = scipy.signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

    contours = np.sqrt(gradient_x**2 + gradient_y**2)

    seuil = 50
    contours[contours < seuil] = 0
    contours[contours >= seuil] = 255

    return contours

def get_contours_canny(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    return edges

def find_rectangle(image, tol=0.01):
    image = np.uint8(image)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        score, rect = rectangle_analysis.rectangle_similarity_score(approx)
        if score < tol :
            rects.append(rect)

    return rects
