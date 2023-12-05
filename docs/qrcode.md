# QR Code Reading Package Documentation

## Overview
This package is designed to read QR codes from images using computer vision techniques. It relies on the `cv2`, `numpy (np)`, and `pyzbar` libraries for image processing and QR code decoding functionalities.

## Functions Description
### `read_qr_code(img)`
- Reads QR codes from an input image.
- Converts the input image to grayscale.
- Utilizes the `pyzbar.decode()` function from the `pyzbar` library to decode QR codes within the image.
- Returns a list of decoded objects (`decoded_objects`) containing information extracted from the QR codes.

## Notes to the Users
- Ensure that the input image contains valid QR codes for accurate decoding.
- The `read_qr_code()` function returns decoded information from QR codes found in the input image.

This package serves the purpose of extracting information encoded in QR codes from images, utilizing OpenCV, NumPy, and the Pyzbar library to facilitate QR code reading through computer vision techniques.