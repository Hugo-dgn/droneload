import cv2
import numpy as np


def circles_in_img(img):
    circle_list = []  # Initialize circle_list as an empty list

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,  # Increase minDist to make circles less sensitive
        param1=10,  # Increase param1 to make edge detection less sensitive
        param2=100,   # Decrease param2 to make circle center detection less sensitive
        minRadius=10,  # Adjust minRadius to limit the size range of detected circles
        maxRadius=200,  # Adjust maxRadius to limit the size range of detected circles
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            circle_list.append([x, y, r])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"R={r}", (x - r, y - r - 10),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return circle_list
