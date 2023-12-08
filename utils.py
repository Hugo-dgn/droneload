import numpy as np
import time

import cv2
import matplotlib.pyplot as plt

import droneload


target_rect_corners = np.array(
    [[0, -7.5, -5],
     [0, 7.5, -5],
     [0, 7.5, 5],
     [0, -7.5, 5]], dtype=np.float32)

def get_rect(image, args):
    contours = droneload.rectFinder.get_contours_canny(image, seuil=args.seuil, kernel_size=args.ksize)
    rminLineLength = 1/args.houghlength
    rmaxLineGap = 1/args.houghgap
    threshold = args.houghthreshold
    lines = droneload.rectFinder.get_lines(contours, rminLineLength, rmaxLineGap, threshold)
    rects = droneload.rectFinder.find_rectangles(lines, tol=args.tol)
    return rects

def contours(args):
    if args.image is None:
        video_contours(args)
    else:
        image_contours(args)

def video_contours(args):
    
        cap = cv2.VideoCapture(0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        droneload.rectFinder.calibrate_image_size(height)
    
        while True:
            ret, frame = cap.read()
            frame = droneload.rectFinder.undistort(frame)
    
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            contours = droneload.rectFinder.get_contours_canny(image, seuil=args.seuil, kernel_size=args.ksize)
    
            rminLineLength = 1/args.houghlength
            rmaxLineGap = 1/args.houghgap
            threshold = args.houghthreshold
            lines = droneload.rectFinder.get_lines(contours, rminLineLength, rmaxLineGap, threshold)
                
            cv_contours, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv_contours_frame = frame.copy()
            cv2.drawContours(cv_contours_frame, cv_contours, -1, (0, 255, 0), 2)
    
            cv2.imshow('Edge', contours)
            cv2.imshow("contours", cv_contours_frame)
            cv2.imshow("lines", lines)
            if cv2.waitKey(100) == ord('q'):
                break
        cv2.destroyAllWindows()

def image_contours(args):
    frame = cv2.imread(args.image)
    frame = droneload.rectFinder.undistort(frame)
    height, width, _ = frame.shape
    droneload.rectFinder.calibrate_image_size(height)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = droneload.rectFinder.get_contours_canny(image, seuil=args.seuil, kernel_size=args.ksize)
    
    rminLineLength = 1/args.houghlength
    rmaxLineGap = 1/args.houghgap
    threshold = args.houghthreshold
    lines = droneload.rectFinder.get_lines(contours, rminLineLength, rmaxLineGap, threshold)
        
    cv_contours, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours_frame = frame.copy()
    cv2.drawContours(cv_contours_frame, cv_contours, -1, (0, 255, 0), 2)

    cv2.imshow('Edge', contours)
    cv2.imshow("contours", cv_contours_frame)
    cv2.imshow("lines", lines)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
        
def rectangle(args):
    if args.image is None:
        video_rectangle(args)
    else:
        image_rectangle(args)

def video_rectangle(args):

    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    droneload.rectFinder.calibrate_image_size(height)

    while True:
        ret, frame = cap.read()
        frame = droneload.rectFinder.undistort(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = get_rect(image, args)

        droneload.rectFinder.remove_old_rects(10)
        droneload.rectFinder.fit(rects, args.fit)
        
        current_rects = droneload.rectFinder.get_current_rects()

        droneload.rectFinder.draw_rectangles(frame, current_rects)
        droneload.rectFinder.draw_main_rectangle(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(50) == ord('q'):
            break
    cv2.destroyAllWindows()


def image_rectangle(args):
    t0 = time.time()
    frame = cv2.imread(args.image)
    height, width, _ = frame.shape

    droneload.rectFinder.calibrate_image_size(height)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = droneload.rectFinder.get_contours_canny(image, seuil=args.seuil, kernel_size=args.ksize)
    rminLineLength = 1/args.houghlength
    rmaxLineGap = 1/args.houghgap
    threshold = args.houghthreshold
    lines = droneload.rectFinder.get_lines(contours, rminLineLength, rmaxLineGap, threshold)
    rects = droneload.rectFinder.find_rectangles(lines, tol=args.tol)

    droneload.rectFinder.remove_old_rects(10)
    

    for rect in rects:
        rect.define_3D(target_rect_corners)
        center = rect.center2D()
        rect.fit(args.fit)
        pos, retval, rvec, tvec = rect.compute()
        droneload.rectFinder.draw_coordinate(frame, center, rvec, tvec)

    droneload.rectFinder.draw_rectangles(frame, rects)

    t1 = time.time()
    print(f"Number of rectangles found: {len(rects)} in {t1-t0} seconds")
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


def simulate(args):

    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    droneload.rectFinder.calibrate_image_size(height)

    ax = init_scene()

    while True:
        ax.clear()
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-30, 30])
        ret, frame = cap.read()
        frame = droneload.rectFinder.undistort(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours = droneload.rectFinder.get_contours_canny(image, seuil=args.seuil, kernel_size=args.ksize)
        rminLineLength = 1/args.houghlength
        rmaxLineGap = 1/args.houghgap
        threshold = args.houghthreshold
        lines = droneload.rectFinder.get_lines(contours, rminLineLength, rmaxLineGap, threshold)
        rects = droneload.rectFinder.find_rectangles(lines, tol=args.tol)

        droneload.rectFinder.remove_old_rects(10)
        droneload.rectFinder.fit(rects, args.fit)
        
        current_rects = droneload.rectFinder.get_current_rects()

        droneload.rectFinder.draw_rectangles(frame, current_rects)
        droneload.rectFinder.draw_main_rectangle(frame)

        rect = droneload.rectFinder.get_main_rect()
        
        if rect is not None:
            rect.define_3D(target_rect_corners)
            pos, retval, rvec, tvec = rect.compute()
            window = droneload.pathFinder.Window(target_rect_corners)

            x0 = pos
            x1 = window.p
            n = window.n * np.sign(np.dot(window.n, window.p-x0)) * 2 * np.linalg.norm(x1-x0)
            L = np.linalg.norm(window.p-x0)
            u = droneload.pathFinder.get_path(x0, x1, n, L, args.n_point)

            droneload.pathFinder.draw_path_plt(ax, u)
            droneload.rectFinder.draw_scene(ax, rects=[rect])

        cv2.imshow('frame', frame)
        if cv2.waitKey(50) == ord('q'):
            break
    cv2.destroyAllWindows()


def video_circle(args):

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        circles = droneload.circleDetection.detect_circles_and_measure(
            frame)

        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (255, 0, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5),
                          (x + 5, y + 5), (0, 128, 255), -1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    cv2.destroyAllWindows()


def video_qr_code(args):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        decoded_objects = droneload.QRCodeReader.read_qr_code(frame)

        for obj in decoded_objects:
            points = obj.polygon if len(obj.polygon) > 0 else obj.rect
            if len(points) == 4:
                cv2.polylines(frame, [np.array(points, dtype=int)], isClosed=True, color=(
                    0, 0, 255), thickness=2)
            print("QR Code Data:", obj.data.decode('utf-8'))

        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
