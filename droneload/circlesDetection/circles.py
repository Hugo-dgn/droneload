import cv2
import numpy as np
import yaml

calibration_file = 'D:\CS_2A\DroneLoadCode\droneload\calibration.yml'

real_radius = 2.8  # entrer ici taille reelle du cercle


def load_calibration_data():
    # Charger les données de calibration à partir du fichier calibration.yml
    with open(calibration_file, 'r') as file:
        calib_data = yaml.load(file, Loader=yaml.FullLoader)

    # Accéder aux données de la matrice de caméra (mtx) et de la distorsion (dist)
    # Accéder au premier élément du tuple (le dictionnaire)
    calib_data = calib_data[0]
    camera_matrix = np.array(calib_data['mtx'])
    distortion_coeff = np.array(calib_data['dist'])

    return camera_matrix, distortion_coeff


def calculate_real_radius_and_distance(imgpts, camera_matrix, distortion_coeff):
    imgpts = np.array(imgpts, dtype=float)  # Convertir en tableau NumPy
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
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=10,
        param2=100,
        minRadius=10,
        maxRadius=200
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        imgpts = []  # Créez une liste vide pour stocker les coordonnées des cercles

        for (x, y, r) in circles:
            circle_list.append([x, y, r])

            # Échantillonnez 10 points le long du contour du cercle
            circle_points = []
            for i in range(10):
                angle = i * 2 * np.pi / 10
                circle_x = int(x + r * np.cos(angle))
                circle_y = int(y + r * np.sin(angle))
                circle_points.append([circle_x, circle_y])

                imgpt = np.array([[circle_x, circle_y]], dtype=float)
                imgpts.append(imgpt)

        # Reste du code pour calculer le rayon réel et la distance
        real_radii, distances = calculate_real_radius_and_distance(
            imgpts, camera_matrix, distortion_coeff)
        print(real_radii, distances)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (circle_x, circle_y) in enumerate(circle_points):
            cv2.putText(img, f"R={real_radii:.2f} cm", (circle_x - r, circle_y - r - 10),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"Distance={distances:.2f} cm", (circle_x - r, circle_y - r + 20),
                        font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.circle(img, (x, y), r, (0, 255, 0), 4)

    return circle_list
