import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar


def read_qr_code(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    decoded_objects = pyzbar.decode(img)

    for obj in decoded_objects:
        qr_data = obj.data.decode('utf-8')

        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array(points), returnPoints=False)
            for j in range(0, len(hull)):
                cv2.line(img, tuple(hull[j][0]), tuple(
                    hull[(j+1) % len(hull)][0]), (0, 0, 255), 3)

        print("QR Code Data:", qr_data)
    cv2.imshow("QR Code Scanner", img)
