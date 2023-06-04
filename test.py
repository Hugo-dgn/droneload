import cv2
import timeit

import video.process as process

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

contours = process.get_contours_sobel(image)
process.find_rectangle(contours, 0.1)

def get_rect():
    contours = process.get_contours_sobel(image)
    process.find_rectangle(contours, 0.1)
    return contours

num_runs = 10
execution_time = timeit.timeit(get_rect, number=num_runs)

print(execution_time)