import cv2
import numpy as np

# Load the image
image = cv2.imread('Circles.webp')

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the yellow color in HSV
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Create a mask to extract only the yellow regions
mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

# Apply Gaussian blur to the mask to reduce noise
blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)

# Use Hough Circle Transform to detect circles in the blurred mask
circles = cv2.HoughCircles(
    blurred_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0
)

# If circles are found, draw them on the original image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, r = circle[0], circle[1], circle[2]
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

# Display the original image with detected circles
cv2.imshow('Yellow Circles Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()