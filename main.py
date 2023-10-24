import argparse

import droneload
import utils

droneload.rectFinder.calibration("calibration.yml")

tol = 2
fit = 0.2


def main():
    parser = argparse.ArgumentParser(description="droneload tasks")

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    vrect_parser = subparsers.add_parser(
        "vrect", help="Draw rectangles on video")
    vrect_parser.add_argument(
        "--tol", help="Tolerance for rectangle detection", type=float, default=tol)
    vrect_parser.add_argument(
        "--info", help="Show info about the rectangle", action="store_true")
    vrect_parser.add_argument(
        "--fit", help="Threshold for rectangle fit", type=float, default=fit)
    vrect_parser.set_defaults(func=utils.video_rectangle)

    imrect_parser = subparsers.add_parser(
        "imrect", help="Draw rectangles on image")
    imrect_parser.add_argument("image", help="Image to draw rectangles on")
    imrect_parser.add_argument(
        "--fit", help="Threshold for rectangle fit", type=float, default=fit)
    imrect_parser.add_argument(
        "--tol", help="Tolerance for rectangle detection", type=float, default=tol)
    imrect_parser.set_defaults(func=utils.image_rectangle)

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

    impath_parser = subparsers.add_parser(
        "impath", help="Find path from image")
    impath_parser.add_argument("image", help="Image to find path on")
    impath_parser.add_argument(
        "--fit", help="Threshold for rectangle fit", type=float, default=fit)
    impath_parser.add_argument(
        "--tol", help="Tolerance for rectangle detection", type=float, default=tol)
    impath_parser.add_argument(
        "--T", help="Time to travel", type=float, default=10)
    impath_parser.add_argument(
        "--x0", help="Initial x position", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument(
        "--v0", help="Initial velocity", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument(
        "--a0", help="Initial acceleration", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument(
        "--a1", help="Final acceleration", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument(
        "--norme_v1", help="Final velocity norm", type=float, default=1)
    impath_parser.add_argument(
        "--n", help="Normal vector of the window", type=float, nargs=3, default=[0, 1, 0])
    impath_parser.add_argument(
        "--n_point", help="Number of point to calculate the path", type=int, default=100)
    impath_parser.set_defaults(func=utils.image_path)

    vpath = subparsers.add_parser("vpath", help="Animate path")
    vpath.add_argument(
        "--fps", help="Frames per second", type=int, default=30)
    vpath.add_argument(
        "--tol", help="Tolerance for rectangle detection", type=float, default=tol)
    vpath.add_argument(
        "--fit", help="Threshold for rectangle fit", type=float, default=fit)
    vpath.add_argument(
        "--n_point", help="Number of point to calculate the path", type=int, default=100)
    vpath.set_defaults(func=utils.animate_scene)

    circle_parser = subparsers.add_parser(
        "circles", help="Find cirles in image")
    circle_parser.set_defaults(func=utils.video_circle)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
