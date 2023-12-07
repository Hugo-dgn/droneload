import argparse

import droneload
import utils

droneload.rectFinder.calibration("calibration.yml")

tol = 0.2
fit = 0.2
seuil = 50
ksize = 1

hough_length = 10 # increase to detect shotter lines
hough_gap = 30 # decrease to detect more lines
hough_threshold = 10


def main():
    parser = argparse.ArgumentParser(description="droneload tasks")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    contours_parser = subparsers.add_parser(
        "contours", help="Find contours in image")
    contours_parser.add_argument("--canny", help="Use canny filter", action="store_true")
    contours_parser.add_argument(
        "--seuil", help="Seuil for canny/sobel filter", type=int, default=seuil)
    contours_parser.add_argument(
        "--ksize", help="Kernel size for canny/sobel filter", type=int, default=ksize)
    contours_parser.add_argument(
        "--houghlength", help="Inverse of the ratio for min lenght line detection", type=float, default=hough_length)
    contours_parser.add_argument(
        "--houghgap", help="Inverse of the ratio for max gap line detection", type=float, default=hough_gap)
    contours_parser.add_argument(
        "--houghthreshold", help="Threshold for line detection", type=int, default=hough_threshold)
    contours_parser.add_argument(
        "--image", help="Image to draw rectangles on", type=str, default=None)
    contours_parser.set_defaults(func=utils.contours)

    rect_parser = subparsers.add_parser(
        "rect", help="Draw rectangles on video")
    rect_parser.add_argument(
        "--tol", help="Tolerance for rectangle detection", type=float, default=tol)
    rect_parser.add_argument(
        "--info", help="Show info about the rectangle", action="store_true")
    rect_parser.add_argument(
        "--fit", help="Threshold for rectangle fit", type=float, default=fit)
    rect_parser.add_argument(
        "--seuil", help="Seuil for canny/sobel filter", type=int, default=seuil)
    rect_parser.add_argument(
        "--ksize", help="Kernel size for canny/sobel filter", type=int, default=ksize)
    rect_parser.set_defaults(func=utils.rectangle)
    rect_parser.add_argument(
                             "--image", help="Image to draw rectangles on", type=str, default=None)

    path_parser = subparsers.add_parser(
        "path", help="Find path from parameters")
    path_parser.add_argument(
        "--x0", help="Initial x position", type=float, nargs=3, default=[0, 0, 0])
    path_parser.add_argument(
        "--n", help="Normal vector of the window", type=float, nargs=3, default=[0, 10, 0])
    path_parser.add_argument(
        "--L", help="After window lenght", type=float, default=3)
    path_parser.add_argument(
        "--n_point", help="Number of point to calculate the path", type=int, default=100)
    path_parser.set_defaults(func=utils.plot_path)

    path = subparsers.add_parser("path", help="Animate path")
    path.add_argument(
        "--fps", help="Frames per second", type=int, default=30)
    path.add_argument(
        "--tol", help="Tolerance for rectangle detection", type=float, default=tol)
    path.add_argument(
        "--fit", help="Threshold for rectangle fit", type=float, default=fit)
    path.add_argument(
        "--n_point", help="Number of point to calculate the path", type=int, default=100)
    path.add_argument(
        "--image", help="Image to draw rectangles on", type=str, default=None)
    path.set_defaults(func=utils.path)

    circle_parser = subparsers.add_parser(
        "circles", help="Find cirles in image")
    circle_parser.set_defaults(func=utils.video_circle)

    qr_parser = subparsers.add_parser(
        "qrcode", help="Read QR codes in image")
    qr_parser.set_defaults(func=utils.video_qr_code)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
