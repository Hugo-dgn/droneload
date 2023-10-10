import cv2
import numpy as np

image = cv2.imread('Circles.jpg')

def circles_in_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circle_list = []

        for (x, y, r) in circles:
            circle_list.append([x, y, r])

        print("Detected circles:")
        for i, circle in enumerate(circle_list):
            print(f"Circle {i + 1}: Center = ({circle[0]}, {circle[1]}), Radius = {circle[2]}")

        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (255, 0, 0), 4) 
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1) 

        cv2.imshow("Detected Circles", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles detected in the image.")
