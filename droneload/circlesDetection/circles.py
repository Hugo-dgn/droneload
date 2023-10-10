import cv2
import numpy as np
import math


def circle_similarity(circle1, circle2, tol):
    """
    Calculate the similarity score between two circles based on IoU.

    Args:
        circle1 (tuple): A tuple containing (x, y, radius) for the first circle.
        circle2 (tuple): A tuple containing (x, y, radius) for the second circle.
        tolerance (float): The similarity tolerance threshold (between 0 and 1).

    Returns:
        bool: True if the circles are similar, False otherwise.
    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2

    # Calculate the Intersection over Union (IoU) score
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    iou = min(r1, r2) / max(r1, r2)

    return iou >= tol


def find_circles(img, tol):
    """
    Find circles in an image based on a specified similarity tolerance.

    Args:
        image_path (str): The path to the image.
        tolerance (float): The similarity tolerance threshold (between 0 and 1).
        min_radius (int): Minimum radius of circles to search for (default is 0).
        max_radius (int): Maximum radius of circles to search for (default is None).

    Returns:
        list: A list of circles that meet the similarity threshold.
    """
    img_height, img_width = img.shape
    img = cv2.GaussianBlur(img, (9, 9), 2)

    min_radius = 0
    max_radius = min(img_height, img_width)

    # Initialize an accumulator matrix to store circle votes
    accumulator = np.zeros((img_height, img_width, max_radius), dtype=np.uint8)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(img, threshold1=50, threshold2=150)

    # Find non-zero edge pixels (potential circle centers)
    edge_pixels = np.argwhere(edges > 0)

    # Iterate through potential circle centers
    for center_y, center_x in edge_pixels:
        for radius in range(min_radius, max_radius):
            # Calculate circle parameters
            theta = np.linspace(0, 2 * np.pi, 100)
            x = center_x + radius * np.cos(theta)
            y = center_y + radius * np.sin(theta)

            # Ensure the circle coordinates are within the image bounds
            x = np.clip(x, 0, img_width - 1).astype(int)
            y = np.clip(y, 0, img_height - 1).astype(int)

            # Vote for the circle in the accumulator
            accumulator[y, x, radius] += 1

    # Find circles that meet the similarity threshold
    detected_circles = []
    for radius in range(min_radius, max_radius):
        for y in range(img_height):
            for x in range(img_width):
                if accumulator[y, x, radius] > 0:
                    circle = (x, y, radius)
                    # Check similarity with other detected circles
                    is_similar = True
                    for detected_circle in detected_circles:
                        if circle_similarity(circle, detected_circle, tol):
                            is_similar = False
                            break
                    if is_similar:
                        detected_circles.append(circle)

    return detected_circles


image = cv2.imread('cirlces.jpeg')
find_circles(image, 0.8)
