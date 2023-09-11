import argparse
import numpy as np

import cv2
import matplotlib.pyplot as plt

import droneload

def main():
    parser = argparse.ArgumentParser(description="droneload tasks")
    
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    vrect_parser = subparsers.add_parser("vrect", help="Draw rectangles on video")
    vrect_parser.add_argument("--tol", help="Tolerance for rectangle detection", type=float, default=50)
    vrect_parser.add_argument("--alpha", help="Alpha for normal calculation", type=float, default=1)
    vrect_parser.add_argument("--info", help="Show info about the rectangle", action="store_true")
    vrect_parser.set_defaults(func=video_rectangle)

    imrect_parser = subparsers.add_parser("imrect", help="Draw rectangles on image")
    imrect_parser.add_argument("image", help="Image to draw rectangles on")
    imrect_parser.add_argument("--alpha", help="Alpha for normal calculation", type=float, default=1)
    imrect_parser.add_argument("--tol", help="Tolerance for rectangle detection", type=float, default=50)
    imrect_parser.set_defaults(func=image_rectangle)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

def print_rect_info(rect, args):
    n1, n2 = droneload.rectFinder.find_normal(rect, args.alpha)
    print(f"||n||={np.linalg.norm(n1)}") 

    d = droneload.rectFinder.find_dist(np.linalg.norm(n1), 25e-4)
    print(f"d = {d}\n")


def video_rectangle(args):
    cap = cv2.VideoCapture(0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    droneload.rectFinder.calibrate_image_size(height)
    droneload.rectFinder.calibrate_focal(1.484)

    while True:
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours = droneload.rectFinder.get_contours_sobel(image)
        rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)
        droneload.rectFinder.draw_rectangles(frame, rects)

        if args.info and len(rects) == 1:
            print_rect_info(rects[0], args)

        cv2.imshow('frame', frame)

        if cv2.waitKey(100) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def image_rectangle(args):
    frame  = cv2.imread(args.image)
    height, width, _ = frame.shape

    droneload.rectFinder.calibrate_image_size(height)
    droneload.rectFinder.calibrate_focal(1.484)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours = droneload.rectFinder.get_contours_sobel(image)
    rects = droneload.rectFinder.find_rectangles(contours, tol=args.tol)
    droneload.rectFinder.draw_rectangles(frame, rects)

    if len(rects) == 1:
        print_rect_info(rects[0], args)

    plt.imshow(frame)
    plt.show()

if __name__ == '__main__':
    main()