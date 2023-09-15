import argparse

import droneload
import utils

droneload.rectFinder.calibration("calibration.yml")

def main():
    parser = argparse.ArgumentParser(description="droneload tasks")
    
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    vrect_parser = subparsers.add_parser("vrect", help="Draw rectangles on video")
    vrect_parser.add_argument("--tol", help="Tolerance for rectangle detection", type=float, default=5)
    vrect_parser.add_argument("--info", help="Show info about the rectangle", action="store_true")
    vrect_parser.add_argument("--fit", help="Threshold for rectangle fit", type=float, default=0.8)
    vrect_parser.set_defaults(func=utils.video_rectangle)

    imrect_parser = subparsers.add_parser("imrect", help="Draw rectangles on image")
    imrect_parser.add_argument("image", help="Image to draw rectangles on")
    imrect_parser.add_argument("--tol", help="Tolerance for rectangle detection", type=float, default=5)
    imrect_parser.set_defaults(func=utils.image_rectangle)
    
    path_parser = subparsers.add_parser("path", help="Find path from parameters")
    path_parser.add_argument("--T", help="Time to travel", type=float, default=10)
    path_parser.add_argument("--x0", help="Initial x position", type=float, nargs=3, default=[0, 0, 0])
    path_parser.add_argument("--v0", help="Initial velocity", type=float, nargs=3, default=[0, 0, 0])
    path_parser.add_argument("--a0", help="Initial acceleration", type=float, nargs=3, default=[0, 0, 0])
    path_parser.add_argument("--a1", help="Final acceleration", type=float, nargs=3, default=[0, 0, 0])
    path_parser.add_argument("--norme_v1", help="Final velocity norm", type=float, default=1)
    path_parser.add_argument("--n", help="Normal vector of the window", type=float, nargs=3, default=[0, 1, 0])
    path_parser.add_argument("--n_point", help="Number of point to calculate the path", type=int, default=100)
    path_parser.set_defaults(func=utils.plot_path)
    
    impath_parser = subparsers.add_parser("impath", help="Find path from image")
    impath_parser.add_argument("image", help="Image to find path on")
    impath_parser.add_argument("--tol", help="Tolerance for rectangle detection", type=float, default=50)
    impath_parser.add_argument("--T", help="Time to travel", type=float, default=10)
    impath_parser.add_argument("--x0", help="Initial x position", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument("--v0", help="Initial velocity", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument("--a0", help="Initial acceleration", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument("--a1", help="Final acceleration", type=float, nargs=3, default=[0, 0, 0])
    impath_parser.add_argument("--norme_v1", help="Final velocity norm", type=float, default=1)
    impath_parser.add_argument("--n", help="Normal vector of the window", type=float, nargs=3, default=[0, 1, 0])
    impath_parser.add_argument("--n_point", help="Number of point to calculate the path", type=int, default=100)
    impath_parser.set_defaults(func=utils.image_path)
    
    r3D_parser = subparsers.add_parser("r3D", help="Animate path")
    r3D_parser.add_argument("--fps", help="Frames per second", type=int, default=30)
    r3D_parser.add_argument("--tol", help="Tolerance for rectangle detection", type=float, default=5)
    r3D_parser.add_argument("--fit", help="Threshold for rectangle fit", type=float, default=0.8)
    r3D_parser.add_argument("--l", help="Lenght of a side of the displayed cube", type=float, default=30)
    r3D_parser.set_defaults(func=utils.animate_scene)
    
    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()