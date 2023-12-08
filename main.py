import argparse

import droneload
import utils

droneload.rectFinder.calibration("calibration.yml")

tol = 0.2
fit = 0.5
seuil = 30
ksize = 3

hough_length = 10 # increase to detect shotter lines
hough_gap = 100 # decrease to detect more lines
hough_threshold = 100


def main():
    parser = argparse.ArgumentParser(description="droneload tasks")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    contours_parser = subparsers.add_parser(
        "contours", help="Find contours in image")
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
    rect_parser.add_argument(
        "--houghlength", help="Inverse of the ratio for min lenght line detection", type=float, default=hough_length)
    rect_parser.add_argument(
        "--houghgap", help="Inverse of the ratio for max gap line detection", type=float, default=hough_gap)
    rect_parser.add_argument(
        "--houghthreshold", help="Threshold for line detection", type=int, default=hough_threshold)
    rect_parser.add_argument(
                             "--image", help="Image to draw rectangles on", type=str, default=None)
    rect_parser.set_defaults(func=utils.rectangle)

    sim_parser = subparsers.add_parser("sim", help="Animate path")
    sim_parser.add_argument(
        "--fps", help="Frames per second", type=int, default=30)
    sim_parser.add_argument(
        "--tol", help="Tolerance for rectangle detection", type=float, default=tol)
    sim_parser.add_argument(
        "--fit", help="Threshold for rectangle fit", type=float, default=fit)
    sim_parser.add_argument(
        "--n_point", help="Number of point to calculate the path", type=int, default=100)
    sim_parser.add_argument(
        "--seuil", help="Seuil for canny/sobel filter", type=int, default=seuil)
    sim_parser.add_argument(
        "--ksize", help="Kernel size for canny/sobel filter", type=int, default=ksize)
    sim_parser.add_argument(
        "--houghlength", help="Inverse of the ratio for min lenght line detection", type=float, default=hough_length)
    sim_parser.add_argument(
        "--houghgap", help="Inverse of the ratio for max gap line detection", type=float, default=hough_gap)
    sim_parser.add_argument(
        "--houghthreshold", help="Threshold for line detection", type=int, default=hough_threshold)
    sim_parser.set_defaults(func=utils.simulate)

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
