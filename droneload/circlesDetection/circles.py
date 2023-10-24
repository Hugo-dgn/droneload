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
        minDist=20,  # Augmenter minDist = augmenter sensibilité de détection générale
        param1=10,  # Augmenter param1 = diminuer sensibilité de détection des bords
        param2=100,   # Augmenter param2 = augmenter sensibilité de détection des centres
        minRadius=10,  # minRadius permet de limiter la plage de cerlces détectés
        maxRadius=200,  # idem : Seuls les cercles tq minRadius < rayon < maxRadius seront detectés
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            circle_list.append([x, y, r])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"R={r}", (x - r, y - r - 10),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return circle_list
