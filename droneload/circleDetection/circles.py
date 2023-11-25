import cv2
import numpy as np
import yaml

calibration_file = 'calibration.yml'

real_radius = 2.1  # entrer ici taille reelle du cercle


def load_calibration_data():
    with open(calibration_file, 'r') as file:
        calib_data = yaml.load(file, Loader=yaml.FullLoader)

    calib_data = calib_data[0]
    camera_matrix = np.array(calib_data['mtx'])
    distortion_coeff = np.array(calib_data['dist'])

    return camera_matrix, distortion_coeff


def calculate_real_radius_and_distance(imgpts, camera_matrix, distortion_coeff):
    imgpts = np.array(imgpts, dtype=float)
    angles = np.linspace(0, 2 * np.pi, 10)
    objpts = np.array([[real_radius * np.cos(theta), real_radius *
                      np.sin(theta), 0] for theta in angles], dtype=np.float32)

    retval, rvecs, tvecs = cv2.solvePnP(
        objpts, imgpts, camera_matrix, distortion_coeff)

    if retval:
        distance = np.linalg.norm(tvecs)
        return real_radius, distance


def detect_circles_and_measure(img):
    circle_list = []
    camera_matrix, distortion_coeff = load_calibration_data()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (9,9) taille du noyau, 4 = intensité du flou : intensité trop faible (=2) => probleme detection quand on s'approche trop de la camera
    gray = cv2.GaussianBlur(gray, (13, 13), 6)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,  # Augmenter minDist = augmenter sensibilité générale
        param1=10,  # Augmenter param1 = diminuer sensibilité de detection des bords
        param2=100,  # Augmenter param2 = augmenter sensibilité de detection du centre
        minRadius=10,  # Seuls les cercles tq minDist < rayon < maxDist sont detectes
        maxRadius=200
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        imgpts = []

        for (x, y, r) in circles:
            circle_list.append([x, y, r])
            circle_points = []
            for i in range(10):
                angle = i * 2 * np.pi / 10
                circle_x = int(x + r * np.cos(angle))
                circle_y = int(y + r * np.sin(angle))
                circle_points.append([circle_x, circle_y])

                imgpt = np.array([[circle_x, circle_y]], dtype=float)
                imgpts.append(imgpt)

        real_radii, distances = calculate_real_radius_and_distance(
            imgpts, camera_matrix, distortion_coeff)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"R={real_radii:.2f} cm", (circle_x - r, circle_y - r - 10),
                    font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Distance={distances:.2f} cm", (circle_x - r, circle_y - r + 20),
                    font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return circle_list