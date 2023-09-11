_f = None
_l = None

def calibrate_focal(f):
    global _f
    _f = f

def calibrate_image_size(lenght):
    global _l
    _l = lenght

def get_focal():
    return _f

def get_image_size():
    return _l