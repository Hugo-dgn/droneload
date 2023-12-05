# Circle Detection Package Documentation

## Overview
This package provides functionality to detect circles in an image (gathered from the camera), measures its distance from the camera, and overlay the measurements onto the image. It utilizes the OpenCV library for computer vision tasks and requires the `cv2`, `numpy`, and `yaml` packages.


## Functions description
### `load_calibration_data()`
- Loads calibration data from a YAML file (`calibration.yml`) containing camera matrix and distortion coefficients.
- Returns `camera_matrix` and `distortion_coeff`.

### `calculate_real_radius_and_distance(imgpts, camera_matrix, distortion_coeff)`
- Calculates the distance of detected circles in the image.
- `imgpts`: Image points of the detected circles.
- `camera_matrix` and `distortion_coeff`: Calibration parameters.
- Returns the `real_radius` and `distance` values.

### `detect_circles_and_measure(img)`
- Detects circles in the input image and measures their distance to camera.
- Utilizes calibration data and `cv2.HoughCircles` for circle detection.
- Returns a list of detected circles' coordinates (`circle_list`).


## Overall Used Method: Circle Detection and Measurement
### Method Overview
This method employs computer vision techniques, primarily the Hough Circle Transform, to detect circles within an image. It includes the following key operations:

1. **Loading Calibration Data**
   - Utilizes a YAML file (`calibration.yml`) containing camera matrix and distortion coefficients for the calibration process.

2. **Circle Detection**
   - Converts the input image to grayscale.
   - Applies Gaussian blur to reduce noise.
   - Uses the Hough Circle Transform (`cv2.HoughCircles`) to identify circular shapes in the image based on defined parameters such as edge detection method, minimum distance between detected circles, and other parameters.

3. **Circle Measurement**
   - Upon circle detection, retrieves the detected circle coordinates and iterates through each detected circle.
   - Calculates points along the circumference of each circle using trigonometric functions.
   - Generates image points (`imgpts`) based on the circle coordinates and circumference points.

4. **Calculating Real Radius and Distance**
   - Utilizes the calibration data (camera matrix and distortion coefficients) and the detected image points to calculate the real radius and distance of the detected circles via the Perspective-n-Point (PnP) algorithm (`cv2.solvePnP`).
   - The PnP algorithm estimates the rotation and translation vectors, enabling the determination of real-world measurements.

5. **Visualizing Measurements**
   - Overlays text on the image with the calculated real radii and distances for each detected circle.
   - Utilizes OpenCV's `cv2.putText` to display this information directly onto the image.

### How Hough Circle Transform Works
The Hough Circle Transform is a technique used for detecting circular shapes within an image. It operates by:

- **Edge Detection**: Initially, edges in the image are identified using techniques like Canny edge detection.
  
- **Accumulator Array**: Utilizes an accumulator array to determine possible circle centers. Each pixel in the accumulator array represents a possible circle center, and the value in each pixel denotes how many edges support a circle centered at that point.

- **Voting Procedure**: Votes are cast in the accumulator array for each edge pixel, incrementing the appropriate circle center's accumulator cell.
  
- **Peak Detection**: After voting is complete, peak values in the accumulator array represent potential circle centers, with their respective radii based on the size of the accumulator cell.

- **Parameter Tuning**: The Hough Circle Transform requires careful parameter tuning (such as minimum and maximum radii, minimum distance between centers, Canny edge detection thresholds, etc.) to effectively detect circles in various image conditions.

By iterating through these steps, the Hough Circle Transform can identify circular shapes within the image, allowing subsequent analysis and measurements, as done in this method.

## Notes to the users
- Ensure that the calibration file (`calibration.yml`) contains valid camera matrix and distortion coefficients for accurate measurements.
- The `detect_circles_and_measure()` function returns a list of detected circles' coordinates and overlays their measurements on the input image.

This package is designed to facilitate circle detection, measurement, and visualization, leveraging OpenCV's functionalities and calibration data for real-world measurements.