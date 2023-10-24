import cv2
import numpy as np
import yaml

calibration_file = 'D:\CS_2A\DroneLoadCode\droneload\calibration.yml'

real_radius = 2.8  # entrer ici taille reelle du cercle


def load_calibration_data():
    with open(calibration_file, 'r') as file:
        calib_data = yaml.load(file, Loader=yaml.FullLoader)

    camera_matrix = np.array(calib_data['mtx'])
    distortion_coeff = np.array(calib_data['dist'])

    return camera_matrix, distortion_coeff


def calculate_real_radius_and_distance(imgpts, camera_matrix, distortion_coeff):
    angles = np.linspace(0, 2 * np.pi, 10)
    objpts = np.array([[real_radius * np.cos(theta), real_radius *
                      np.sin(theta), 0] for theta in angles], dtype=np.float32)

    retval, rvecs, tvecs = cv2.solvePnP(
        objpts, imgpts, camera_matrix, distortion_coeff)

    if retval:
        distance = np.linalg.norm(tvecs)
        return real_radius, distance


def detect_circles_and_measure():
    cap = cv2.VideoCapture(0)
    camera_matrix, distortion_coeff = load_calibration_data()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erreur lors de la capture de l'image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            for (x, y, r) in circles:
                imgpts = np.array([[x, y]], dtype=float)

                real_radius, distance = calculate_real_radius_and_distance(
                    imgpts, camera_matrix, distortion_coeff)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Rayon réel = {real_radius:.2f} cm", (
                    x - r, y - r - 10), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Distance à la caméra = {distance:.2f} cm", (
                    x - r, y - r + 20), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

        cv2.imshow("Circles Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
