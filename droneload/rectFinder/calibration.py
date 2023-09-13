import yaml
import numpy as np

class Calibration:
    mtx = None
    dist = None
    l = None

def calibration(path):
    with open(path, "r") as f:
        calibration = yaml.load(f, Loader=yaml.FullLoader)[0]
    
    Calibration.mtx = np.array(calibration["mtx"])
    Calibration.dist = np.array(calibration["dist"])

def calibrate_image_size(height):
    Calibration.l = height


def get_mtx():
    return Calibration.mtx

def get_dist():
    return Calibration.dist

def get_image_size():
    return Calibration.l