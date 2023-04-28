import numpy as np
import scipy
import cv2

import video.rectangle_analysis as rectangle_analysis

import time


sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.uint16)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.uint16)

#####matrice de lissage

#uniforme
moy_kernel = np.array([[1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]], dtype=np.uint16)

##gaussien
k = 3
sigma = 0.5
gaussien_kernel = cv2.getGaussianKernel(k, sigma)
gaussien_kernel = np.outer(gaussien_kernel, gaussien_kernel.transpose()).astype(np.uint16)

def get_contours_sobel(image, seuil=20):
    #image = scipy.signal.convolve2d(image, gaussien_kernel, mode='same', boundary='symm')
    gradient_x = scipy.signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    gradient_y = scipy.signal.convolve2d(image, sobel_y, mode='same', boundary='symm')

    contours = np.sqrt(gradient_x**2 + gradient_y**2)
    seuil = 50
    contours[contours < seuil] = 0
    contours[contours >= seuil] = 255
    return contours.astype(np.uint8)

def max_pool_2d_numpy(input_array, pool_size=(2, 2), strides=None, padding='VALID'):
    if strides is None:
        strides = pool_size
    output_shape = (input_array.shape[0] // strides[0], input_array.shape[1] // strides[1])
    output_array = np.zeros(output_shape, dtype=np.uint8)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            start_i = i * strides[0]
            start_j = j * strides[1]
            end_i = start_i + pool_size[0]
            end_j = start_j + pool_size[1]
            output_array[i, j] = np.max(input_array[start_i:end_i, start_j:end_j])
    return output_array

def get_contours_canny(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    return edges

def find_rectangle(image, tol=0.01):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        score, rect = rectangle_analysis.rectangle_similarity_score(approx)
        if score < tol :
            rects.append(rect)

    return np.array(rects)


def convexe_part(image):
    convexe_matrix = np.pad(image, pad_width=1, mode='constant', constant_values=255)
    convexe_value = 1
    bord_value = 255
    interieur_value = 0
    def dfs(point, value):
        if convexe_matrix[point] != bord_value and convexe_matrix[point] != value:
            convexe_matrix[point] = value
            dfs((point[0]+1, point[1]), value)
            dfs((point[0]-1, point[1]), value)
            dfs((point[0], point[1]+1), value)
            dfs((point[0], point[1]-1), value)
    
    for i, line in enumerate(convexe_matrix):
        for j, val in enumerate(line):
            if val == interieur_value:
                dfs((i, j), convexe_value)
                convexe_value += 1
    return convexe_matrix