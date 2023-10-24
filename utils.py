import numpy as np

import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm


import droneload


target_rect_corners = np.array(
    [[0, -7.5, -5],
     [0, 7.5, -5],
     [0, 7.5, 5],
     [0, -7.5, 5]]
)


def video_rectangle(args):

    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    droneload.rectFinder.calibrate_image_size(height)

    while True:
        ret, frame = cap.read()
        frame = droneload.rectFinder.undistort(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours = droneload.rectFinder.get_contours_sobel(image)
        rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)

        droneload.rectFinder.remove_old_rects(10)

        for rect in rects:
            rect.define_3D(target_rect_corners)
            center = rect.center2D()
            rect.fit(args.fit)
            pos, retval, rvec, tvec = rect.compute()
            droneload.rectFinder.draw_coordinate(frame, center, rvec, tvec)

        droneload.rectFinder.draw_rectangles(frame, rects)
        droneload.rectFinder.draw_main_rectangle(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()


def image_rectangle(args):
    frame = cv2.imread(args.image)
    height, width, _ = frame.shape

    droneload.rectFinder.calibrate_image_size(height)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = droneload.rectFinder.get_contours_sobel(image)
    rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)

    droneload.rectFinder.remove_old_rects(10)

    for rect in rects:
        rect.define_3D(target_rect_corners)
        center = rect.center2D()
        rect.fit(args.fit)
        pos, retval, rvec, tvec = rect.compute()
        droneload.rectFinder.draw_coordinate(frame, center, rvec, tvec)

    droneload.rectFinder.draw_rectangles(frame, rects)
    droneload.rectFinder.draw_main_rectangle(frame)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def init_scene():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-30, 30])

    return ax


def animate_scene(args):

    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    droneload.rectFinder.calibrate_image_size(height)

    ax = init_scene()

    while True:
        ret, frame = cap.read()
        frame = droneload.rectFinder.undistort(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours = droneload.rectFinder.get_contours_sobel(image)
        rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)

        droneload.rectFinder.remove_old_rects(10)

        ax.clear()
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-30, 30])

        for rect in rects:
            rect.define_3D(target_rect_corners)
            center = rect.center2D()
            rect.fit(args.fit)
            pos, retval, rvec, tvec = rect.compute()

            if retval:
                droneload.rectFinder.draw_coordinate(frame, center, rvec, tvec)

                window = droneload.pathFinder.Window(target_rect_corners)

                x0 = pos
                x1 = window.p
                
                n = window.n * np.sign(np.dot(window.n, window.p-x0)) * 2 * np.linalg.norm(x1-x0)
                L = np.linalg.norm(window.p-x0)

                u = droneload.pathFinder.get_path(x0, x1, n, L, args.n_point)

                droneload.pathFinder.draw_path_plt(ax, u)

        droneload.rectFinder.draw_scene(ax)
        droneload.rectFinder.draw_rectangles(frame, rects)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()


def plot_window(window, ax):
    corners = window.corners.copy().T
    corners = np.column_stack(
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3], corners[:, 0]])
    ax.plot3D(corners[0, :], corners[1, :], corners[2, :], 'green')


def rectify_ax_lim(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    delta = 0.5*max([x_max - x_min, y_max - y_min, z_max - z_min])
    x_c = (x_max + x_min)/2
    y_c = (y_max + y_min)/2
    z_c = (z_max + z_min)/2

    ax.set_xlim([x_c - delta, x_c + delta])
    ax.set_ylim([y_c - delta, y_c + delta])
    ax.set_zlim([z_c - delta, z_c + delta])


def plot_path(args):
    corners = target_rect_corners
    window = droneload.pathFinder.Window(corners)

    n = np.array(args.n)

    q, _ = np.linalg.qr(np.outer(n/np.linalg.norm(n), window.n))

    rotated_corners = (q @ corners.T).T + np.array([10, 10, 10])

    window = droneload.pathFinder.Window(rotated_corners)

    x0 = np.array(args.x0)
    x1 = window.p

    L = args.L
    u = droneload.pathFinder.get_path(x0, x1, args.n, L, args.n_point)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    droneload.pathFinder.draw_path_plt(ax, u)

    plot_window(window, ax)
    rectify_ax_lim(ax)

    plt.show()


def image_path(args):
    frame = cv2.imread(args.image)
    height, width, _ = frame.shape

    droneload.rectFinder.calibrate_image_size(height)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = droneload.rectFinder.get_contours_sobel(image)
    rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)

    droneload.rectFinder.draw_rectangles(frame, rects)

    if len(rects) == 1:
        rect = rects[0]
        rect.define_3D(target_rect_corners)
        center = rect.center2D()
        rect.fit(args.fit)
        pos, retval, rvec, tvec = rect.compute()
    else:
        message = "More than 1 rectangle found"
        raise ValueError(message)

    cv2.imshow("frame", frame)
    cv2.waitKey(100)

    if retval:
        
        ax = init_scene()
        ax.clear()
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-30, 30])
        
        droneload.rectFinder.draw_coordinate(frame, center, rvec, tvec)

        window = droneload.pathFinder.Window(target_rect_corners)

        x0 = pos
        x1 = window.p
        
        n = window.n * np.sign(np.dot(window.n, window.p-x0)) * 2 * np.linalg.norm(x1-x0)
        L = np.linalg.norm(window.p-x0)

        u = droneload.pathFinder.get_path(x0, x1, n, L, args.n_point)

        droneload.pathFinder.draw_path_plt(ax, u)
        plot_window(window, ax)
        rectify_ax_lim(ax)
        
        plt.show()
    
    cv2.destroyAllWindows()


def video_circle(args):

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        circles = droneload.circleDetection.detect_circles_and_measure(frame)

        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (255, 0, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5),
                          (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()
