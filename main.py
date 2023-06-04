import argparse
import window
import numpy as np
from locals import *
import video

def main():
    parser = argparse.ArgumentParser(
                    prog = '',
                    description = '')
    parser.add_argument('task', type=str,
                        choices=["window",
                                 "video"])
    
    parser.add_argument('--loss', default=False, action="store_true")
    parser.add_argument('--all', default=False, action="store_true")
    parser.add_argument('--total', default=False, action="store_true")
    parser.add_argument('--path', default=False, action="store_true")
    parser.add_argument('--isos', default=False, action="store_true")

    parser.add_argument('--contours', default=False, action="store_true")
    parser.add_argument('--rect', default=False, action="store_true")
    parser.add_argument('--arrows', default=False, action="store_true")
    
    parser.add_argument('vars', nargs='*')
    args = parser.parse_args()

    if args.task == "window":
        if args.loss:
            if args.total:
                A = window.analyse(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, v_win, norme_v1, conv_L_m, conv_T_s,  n_point, n_angle_phi, n_angle_theta)
                window.plot_analyse_total(A, n_angle_phi, n_angle_theta)
            else:
                Loss, L_loss_max_v, L_max_a, L_id_v_dist, L_loss_lenght = window.analyse_direction_constante(L_min, L_max, T_min, T_max, n_T, n_L, x0, v0, a0, a1, direction_win, v_win, norme_v1, conv_L_m, conv_T_s,  n_point, args.all)
                window.plot_analyse_direction_constante(Loss, L_loss_lenght, L_loss_max_v, L_id_v_dist, L_max_a, L_min, L_max, T_min, T_max, n_L, n_T)
        elif args.isos:
            M = window.analyse_all_cube(lenght_side, T, v0, a0, norme_v1*v_win/np.linalg.norm(v_win), a1, n_point_world_matrix, n_point_path, conv_L_m, conv_T_s)
            window.plot_iso_s(M, threshold, lenght_side, v_win, scale, theta)
        elif args.path:
            window.plot_path(L, T, x0, v0, a0, a1, norme_v1, direction_win, v_win,scale, theta, n_point)
    elif args.task == "video":
        if args.contours:
            video.video_contours()
        if args.rect:
            video.video_rectangle(args.arrows)

if __name__ == "__main__":
    main()