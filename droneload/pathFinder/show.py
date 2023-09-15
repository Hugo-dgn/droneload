import numpy as np
import cv2


from droneload.rectFinder.calibration import get_mtx, get_dist

correction_matrice = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])

correction_matrice = np.linalg.inv(correction_matrice)

print(correction_matrice)

def draw_path_cv(frame, path, rvec, tvec):
    
    mtx = get_mtx()
    dist = get_dist()
    
    path = np.array(path, dtype=np.float32).T
    path = path[-path.shape[0]//2:,:]
    imgpts, jac = cv2.projectPoints(path, rvec, tvec, mtx, dist)
    
    # Supposons que vous avez une liste de points appelée 'curve_points'
    imgpts = imgpts.astype(int)
    for i in range(len(imgpts) - 1):
        try:
            pt1 = tuple(map(int, imgpts[i].ravel()))
            pt2 = tuple(map(int, imgpts[i + 1].ravel()))
            cv2.line(frame, pt1, pt2, (255, 0, 0), thickness=2)
        except:
            pass

    
    
def draw_path_plt(ax, path):
    x, y, z = path[0,:], path[1,:], path[2,:]
    ax.scatter([path[0, 0]], [path[1, 0]], [path[2, 0]], c='r', marker='o')
    ax.plot(x, y, z)