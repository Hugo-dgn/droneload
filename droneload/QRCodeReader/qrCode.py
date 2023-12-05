import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar


def read_qr_code(img):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Decode the QR codes present in the grayscale image
    decoded_objects = pyzbar.decode(gray)

    # Return the decoded QR code information
    return decoded_objects
