import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar


def read_qr_code(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    decoded_objects = pyzbar.decode(img)

    return decoded_objects
